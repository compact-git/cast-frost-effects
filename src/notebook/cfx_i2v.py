# Cast Frost 素材メーカー:🎞 画像から動画を作る
# Wan2.1 I2V 14B 480P(Apache-2.0)+ LightX2V の I2V LoRA(4ステップ)で画像を動かし、
# そのあと背景を抜いて(切り抜く / 黒い背景から抜く)、透明な WebM にする
import os, time, random
import numpy as np
import cfx_common as C
import cfx_media as MD

T2V_REPO = 'Wan-AI/Wan2.1-T2V-14B'               # 文字を読む部品(T5)・VAE は動画のセルと同じものを使う
I2V_REPO = 'Wan-AI/Wan2.1-I2V-14B-480P'           # Apache-2.0:画像を読む部品(CLIP)・公式の本体
DIT_REPO = 'Comfy-Org/Wan_2.1_ComfyUI_repackaged'  # Apache-2.0:本体を 1 ファイルにまとめたもの
LORA_CANDS = [('lightx2v/Wan2.1-Distill-Loras', ('i2v', 'rank64')),
              ('lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v', ('loras/', 'rank64'))]
CLIP_FILE = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
MATTES = [('物・キャラ(背景を切り抜く)', 'cut'), ('光・炎・エフェクト(黒い背景から抜く)', 'luma'),
          ('抜かない(背景つきのまま)', 'none')]
LENGTHS = [('2秒', 2), ('3秒', 3), ('4秒', 4), ('5秒(1回で作れる長さ)', 5),
           ('10秒(5秒ずつ 2回作ってつなぐ)', 10), ('15秒(3回)', 15), ('20秒(4回)', 20),
           ('25秒(5回)', 25), ('30秒(6回)', 30)]
SEG_FRAMES = 81           # 1回で作るコマ数(16fps で約5秒)


def chain_plan(sec, loop):
    """長さから、何回作ってつなぐか・1回のコマ数・最後に残すコマ数を決める。
    2回目からは「前の最後のコマ」を元の画像にするので、1コマ目は重なる(落とす)= 1回ごとに 80 コマ増える"""
    import cfx_video as V
    import cfx_wan as WW
    if sec <= 5:
        nf, _ = V.gen_frames_for(sec, loop, 'repeat')
        return 1, nf, None
    if loop == 'pingpong':
        want = int(round(sec / 2.0 * WW.FPS)) + 1
    else:
        # なめらかなループは終わりの 8 コマを始まりに重ねるので 0.5 秒短くなる(そのために 1回ぶん余計に作らない)
        want = int(round(sec * WW.FPS))
    if want <= SEG_FRAMES:
        return 1, WW.frames_for_seconds(want / WW.FPS), want
    segs = 1 + int(np.ceil((want - SEG_FRAMES) / (SEG_FRAMES - 1)))
    return segs, SEG_FRAMES, want


def match_colors(frames, ref):
    """つなぎ目で色がずれないように、次の区切りの色を、前の区切りの最後のコマに合わせる
    (1コマ目と ref の色の平均・ばらつきの差を、区切り全体に同じだけかける。かけすぎないよう制限する)"""
    if not frames:
        return frames
    a0 = frames[0].astype(np.float32).reshape(-1, 3)
    r = ref.astype(np.float32).reshape(-1, 3)
    gain = np.clip((r.std(0) + 1e-3) / (a0.std(0) + 1e-3), 0.9, 1.1)
    off = np.clip(r.mean(0) - a0.mean(0) * gain, -12.0, 12.0)
    if np.all(np.abs(gain - 1) < 0.005) and np.all(np.abs(off) < 0.5):
        return frames
    return [np.clip(f.astype(np.float32) * gain + off, 0, 255).astype(np.uint8) for f in frames]


def gen_chain(gen, rgb, prompts, segs, nf, want, seed, steps, progress=None, log=print):
    """5秒ずつ順に作って、前の最後のコマから次を作り、1本につなぐ(RGB のコマのリストを返す)
    prompts:区切りごとの英語(足りない時は最後のものをくり返す)"""
    from PIL import Image
    out, used, size, cur = [], None, None, rgb
    for j in range(segs):
        if segs > 1:
            log(f'{j + 1} / {segs} 回目(つなぐ区切り)を作っています…')
        pr = prompts[min(j, len(prompts) - 1)]
        cb = (lambda d, t, j=j: progress(j * t + d, segs * t)) if progress else None
        fr, sd, size = gen.generate(cur, pr, nf, (used + j) if used is not None else seed, steps, progress=cb)
        if used is None:
            used = sd
        fr = [np.asarray(f) for f in fr]
        if j == 0:
            out = fr
        else:
            fr = match_colors(fr, out[-1])
            out += fr[1:]          # 1コマ目は前の最後のコマとほぼ同じなので落とす
        cur = Image.fromarray(out[-1], 'RGB')
        if want and len(out) >= want:
            break
    if want:
        out = out[:want]
    return out, used, size
BG = {'cut': (128, 128, 128), 'luma': (0, 0, 0), 'none': (255, 255, 255)}
HINT = {'cut': ' The background is plain flat gray. The camera is static.',
        'luma': ' The background is pure black. The camera is static.', 'none': ''}


class I2VMaker:
    def __init__(self, ui, plan):
        self.ui, self.plan = ui, plan
        self.gen = None
        self.rife = None
        self.cutter = None

    def unload(self):
        self.gen = None
        self.rife = None
        if self.cutter is not None:
            self.cutter.unload()
        self.cutter = None
        C.free_memory()

    def get_cutter(self):
        if self.cutter is None:
            import cfx_matte
            self.cutter = cfx_matte.Cutter(self.ui.log)
        return self.cutter

    def load(self):
        if self.gen is not None:
            return
        import torch
        import cfx_video as V
        log = self.ui.log
        repo = C.git_clone(V.WAN_ALPHA_GIT, V.WAN_ALPHA_COMMIT, C.ROOT + '/Wan-Alpha', log)
        import cfx_wan as WW
        M = WW.import_wan(repo)
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.clip import CLIPModel
        base = os.path.join(C.LOCAL_MODELS, 'wan2.1-t2v-14b')
        files = C.repo_files(T2V_REPO)
        for f in ['models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth'] + [x for x in files if x.startswith('google/umt5-xxl/')]:
            C.fetch_file(T2V_REPO, f, 'wan2.1-t2v-14b', log)
        ifiles = C.repo_files(I2V_REPO)
        idir = os.path.join(C.LOCAL_MODELS, 'wan2.1-i2v-14b-480p')
        for f in [CLIP_FILE] + [x for x in ifiles if x.startswith('xlm-roberta-large/')]:
            C.fetch_file(I2V_REPO, f, 'wan2.1-i2v-14b-480p', log)
        dfiles = C.repo_files(DIT_REPO)
        cands = [f for f in dfiles if f.startswith('split_files/diffusion_models/') and 'i2v_480p_14B' in f
                 and f.endswith(('_bf16.safetensors', '_fp16.safetensors'))]
        cands.sort(key=lambda f: (0 if 'bf16' in f else 1, f))
        lora_path = None
        for rp, need in LORA_CANDS:
            try:
                lf = [f for f in C.repo_files(rp) if f.endswith('.safetensors') and all(n in f.lower() for n in need)]
            except Exception:
                lf = []
            for f in lf:
                p = C.fetch_file(rp, f, 'lightx2v-i2v', log)
                if WW.lora_matches(M, p, WW.I2V_CFG) > 0:
                    lora_path = p
                    break
            if lora_path:
                break
        if not lora_path:
            raise RuntimeError('画像から動画を作る LoRA が見つかりませんでした(配布元が変わった可能性があります)。'
                               '少し時間をおいてから、もう一度押してください。')
        C.download_url('https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl',
                       C.LOCAL_MODELS + '/rife/flownet_v4.25.pkl', log)

        log('文字を読む部品(T5)を読み込んでいます…')
        t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cpu'),
                            checkpoint_path=os.path.join(base, 'models_t5_umt5-xxl-enc-bf16.pth'),
                            tokenizer_path=os.path.join(base, 'google/umt5-xxl'))
        if self.plan['t5_gpu']:
            t5.model.to('cuda')
        log('画像を読む部品(CLIP)を読み込んでいます…')
        clip = CLIPModel(dtype=torch.float16, device=torch.device('cpu'),
                         checkpoint_path=os.path.join(idir, CLIP_FILE),
                         tokenizer_path=os.path.join(idir, 'xlm-roberta-large'))
        clip.model.to('cuda')
        log('VAE を読み込んでいます…')
        vae = WW.PlainVAE(os.path.join(base, 'Wan2.1_VAE.pth'), 'cuda')
        prog = lambda d, t: self.ui.progress(d, t)
        dit = None
        if cands:
            log(f'本体(14B・約33GB)をダウンロードしています。初回は 5〜10 分ほどかかります…')
            dit_path = C.fetch_file(DIT_REPO, cands[0], 'wan2.1-i2v-dit', log)
            log(f'本体(14B)を GPU に読み込んでいます({self.plan["label"]})…')
            try:
                dit = WW.build_dit(M, dit_path, None, lora_path, mode=self.plan['mode'], device='cuda', log=log,
                                   progress=prog, cfg=WW.I2V_CFG, stream=self.plan['stream'])
            except WW.BaseKeyError as e:
                log(f'(参考){e}')
                try:
                    os.remove(dit_path)
                except Exception:
                    pass
        if dit is None:
            log('公式の本体ファイル(約65GB)をダウンロードして使います。10分ほどかかります…')
            shards = [C.fetch_file(I2V_REPO, f, 'wan2.1-i2v-14b-480p', log) for f in ifiles
                      if f.startswith('diffusion_pytorch_model') and f.endswith('.safetensors')]
            dit = WW.build_dit(M, shards, None, lora_path, mode=self.plan['mode'], device='cuda', log=log,
                               progress=prog, cfg=WW.I2V_CFG, stream=self.plan['stream'])
        self.gen = WW.I2VWan(dit, t5, clip, vae, device='cuda', t5_on_gpu=self.plan['t5_gpu'], log=log)
        import cfx_rife
        self.rife = cfx_rife.Interpolator(C.LOCAL_MODELS + '/rife/flownet_v4.25.pkl', 'cuda')
        self.ui.hide_bar()
        log('画像から動画を作るモデルを読み込みました')


# -------------------------------------------------------------- 画面
def start():
    import cfx_video as V
    ui = C.UI('画像から動画を作る', 'i2v', '1枚の画像を動かして、背景が透明な動画にします')
    try:
        g = C.gpu_info()
    except Exception:
        g = None
    ui.set_header(g)
    if g is None:
        ui.say('⚠ GPU がつながっていません。メニューの <b>ランタイム</b> → <b>ランタイムのタイプを変更</b> で、'
               '<b>A100</b>(おすすめ)を選んでから、このセルをもう一度押してください。', 'err')
        return
    ram_t, _ = C.ram_gb()
    plan = V.plan_gpu(g, ram_t)
    if plan is None:
        ui.say(f'⚠ この GPU({C.esc(g["name"])}・{g["total_gb"]:.0f}GB)では、メモリが足りません。'
               'メニューの <b>ランタイム</b> → <b>ランタイムのタイプを変更</b> で <b>A100</b> を選んでください。', 'err')
        return
    note = f'<br><small>動画の設定:{plan["label"]}</small>'
    if plan['exp']:
        note += ('<div class="cfx-warn">A100 ではないので「試験的」に動かします。A100 より時間がかかります'
                 '(5秒の動画 1本で 10分以上かかることがあります)。</div>')
    if C.disk_free_gb() < 70:
        note += '<div class="cfx-warn">Colab のディスクの空きが少なめです。このセルのモデルは約50GBあります。</div>'
    ui.set_header(g, note)
    ui.extra_diag = [f'動画の設定: {plan}']
    maker = I2VMaker(ui, plan)
    try:
        C.switch_to('i2v', ui.log)
        if C.gpu_info()['free_gb'] < g['total_gb'] * 0.8:
            ui.say('⚠ GPU のメモリが前の作業で使われたままです。メニューの <b>ランタイム</b> → <b>セッションを再起動する</b> を押してから、'
                   'このセルを押し直してください。', 'err')
            return
        C.mount_drive(ui.log)
        C.ensure_out_dir()
        ui.set_header(g, note)
        C.fix_conflicts(ui.log)
        C.pip_install(V.PIP + ['timm', 'kornia'], 'i2v', ui.log)
        C.fix_conflicts(ui.log)
        maker.load()
        C.set_active('i2v', maker.unload)
        C.translator(ui.log)
        C.STATE['cache'] = True
    except Exception as e:
        ui.error(e)
        return
    _build_form(ui, maker, plan)


def _build_form(ui, maker, plan):
    import ipywidgets as W
    import cfx_encode as E
    import cfx_matte
    import cfx_video as V
    import cfx_wan as WW
    from PIL import Image
    L = lambda w='100%': W.Layout(width=w)
    st = {'description_width': '120px'}
    sec1 = W.HTML(C.section_html('① 動かす画像を選ぶ', 'i2v', '背景が透明な PNG がいちばんきれいにできます'))
    picker = MD.Picker(ui, ('image',), on_change=lambda p, k: None)
    ui.picker = picker
    sec2 = W.HTML(C.section_html('② どう動かすかを書く', 'i2v'))
    prompt = W.Textarea(placeholder='例:女の子が笑顔で手を振る/炎がゆらゆら燃える(日本語でOK)', description='動き',
                        layout=W.Layout(width='100%', height='70px'), style=st)
    matte = W.Dropdown(options=MATTES, value='cut', description='背景の抜き方', style=st, layout=L('480px'))
    matte_note = W.HTML('<small style="color:#4d6680">物・キャラ:灰色の背景で動かしてから切り抜きます/'
                        '光・炎:黒い背景で動かしてから、明るさで抜きます</small>')
    sec3 = W.HTML(C.section_html('③ 長さと仕上げ', 'i2v'))
    count = W.IntSlider(value=1, min=1, max=4, description='本数', style=st, layout=L('420px'))
    seed = W.IntText(value=-1, description='シード', style=st, layout=L('260px'))
    length = W.Dropdown(options=LENGTHS, value=5, description='長さ', style=st, layout=L('480px'))
    length_note = W.HTML('')

    def _len_note(ch=None):
        n = chain_plan(length.value, loop.value)[0]
        length_note.value = ('' if n <= 1 else C.CSS + (
            f'<div class="cfx cfx-meta">5秒ずつ <b>{n} 回</b>作って、前の最後のコマから続けて 1本につなぎます。'
            f'作る時間も約 {n} 倍かかります(A100 で 1回 3〜5分ほど。L4 ではもっとかかります)。'
            'くわしい設定の「区切りごとの動き」で、区切りごとに動きを変えられます。</div>'))
    loop = W.Dropdown(options=V.LOOPS, value='none', description='ループ', style=st, layout=L('420px'))
    length.observe(_len_note, names='value')
    loop.observe(_len_note, names='value')
    _len_note()
    smooth = W.Checkbox(value=True, description='なめらかにする(コマを補って 16fps → 32fps)', indent=False)
    thr = W.FloatSlider(value=0.06, min=0.0, max=0.3, step=0.01, description='黒を消す強さ', style=st,
                        layout=L('460px'))
    flicker = W.Checkbox(value=True, description='ちらつきを抑える', indent=False)
    steps = W.IntSlider(value=4, min=4, max=8, description='ステップ数', style=st, layout=L('420px'))
    en = W.Textarea(placeholder='ここに英語を書くと、翻訳を使わずにこの英語で作ります', description='英語で直接書く',
                    style=st, layout=W.Layout(width='100%', height='60px'))
    save = W.Checkbox(value=C.drive_ok(), description='できたものをドライブにも保存する', indent=False)
    parts = W.Textarea(placeholder='5秒より長い時だけ使います。1行に1つ、区切りの順に動きを書きます(日本語でOK)。'
                                   '行が足りない時は、最後の行(なければ ② の動き)をくり返します。\n'
                                   '例:\n手を振る\nくるっと回る\nおじぎをする',
                       description='区切りごとの動き', style=st, layout=W.Layout(width='100%', height='96px'))
    adv = C.Fold(W, '⚙ 高度な設定(ふつうは触らなくて大丈夫)', W.VBox([thr, flicker, steps, en, parts, save])).widget
    go = ui.big_button('✨ 動かす')
    ui.form.children = (sec1, picker.widget, sec2, prompt, matte, matte_note, sec3, count, W.HBox([seed]),
                        length, length_note, loop, smooth, adv, go)

    def run(ss, n, first_seed):
        go.disabled = True
        try:
            src = ss['src']
            pe = ss['en']
            if not pe:
                if C.has_japanese(ss['prompt']):
                    ui.log('日本語を英語に直しています…')
                pe = C.translator(ui.log).translate(ss['prompt'], 'video')
            if not pe:
                ui.say('⚠ どう動かすかを書いてください。', 'warn')
                return
            full = pe.rstrip(' .') + '.' + HINT[ss['matte']]
            img = MD.read_image(src)
            rgb = Image.fromarray(cfx_matte.over(np.asarray(img), BG[ss['matte']]), 'RGB')
            segs, nf, want = chain_plan(ss['sec'], ss['loop'])
            prompts = [full]
            if segs > 1 and ss.get('parts_en') is None and ss.get('parts', '').strip():
                lines = [x.strip() for x in ss['parts'].splitlines() if x.strip()]
                if any(C.has_japanese(x) for x in lines):
                    ui.log('区切りごとの動きを英語に直しています…')
                tr = C.translator(ui.log)
                ss['parts_en'] = [x if not C.has_japanese(x) else tr.translate(x, 'video') for x in lines]
            if segs > 1 and ss.get('parts_en'):
                prompts = [x.rstrip(' .') + '.' + HINT[ss['matte']] for x in ss['parts_en'] if x]
            for i in range(n):
                t0 = time.time()
                sd = first_seed + i if first_seed >= 0 else random.randint(0, 2 ** 31 - 1)
                ui.progress(0, ss['steps'], f'{i + 1} / {n} 本目を作っています…'
                            + (f'(5秒ずつ {segs} 回作ってつなぎます)' if segs > 1 else ''))
                C.free_memory()
                fr, used, (w, h) = gen_chain(maker.gen, rgb, prompts, segs, nf, want, sd, ss['steps'],
                                             progress=lambda d, t: ui.progress(d, t), log=ui.log)
                sd = used
                if ss['matte'] == 'luma':
                    ui.log('黒い背景を透明にしています…')
                    frames = [cfx_matte.luma_key(f, ss['thr']) for f in fr]
                elif ss['matte'] == 'cut':
                    ui.log('背景を1コマずつ切り抜いています…')
                    rgba = [np.dstack([f, np.full(f.shape[:2], 255, np.uint8)]) for f in fr]
                    frames = maker.get_cutter().cut_frames(rgba, smooth=ss['flicker'],
                                                           progress=lambda d, t: ui.progress(d, t))
                else:
                    frames = [np.dstack([f, np.full(f.shape[:2], 255, np.uint8)]) for f in fr]
                frames = V.make_loop(frames, ss['loop'])
                fps = WW.FPS
                if ss['smooth']:
                    ui.log('コマを補って、なめらかにしています…')
                    frames = maker.rife.double(frames, loop=ss['loop'] != 'none',
                                               progress=lambda d, t: ui.progress(d, t))
                    fps = WW.FPS * 2
                name = C.new_name('video', i + 1, 'webm')
                path = os.path.join(C.LOCAL_OUT, name)
                ui.log('WebM に書き出しています(30MB 以内に収めます)…')
                info = E.encode_fit(frames, fps, path, log=ui.log)
                prev = path + '.preview.webm'
                E.encode_preview(frames, fps, prev)
                pr = E.probe(path)
                saved = C.save_to_drive(path) if ss['save'] else None
                s2 = dict(ss, en=pe)
                lines = [f'<b>{C.esc(name)}</b>(WebM・{info["width"]}×{info["height"]}・{info["fps"]}fps・'
                         f'{info["seconds"]:.1f}秒・{info["bytes"] / 1e6:.1f}MB)',
                         f'🎞 {C.esc(os.path.basename(src))} から・シード {sd}・{time.time() - t0:.0f}秒・'
                         f'透明:{"あり" if pr["alpha"] else "なし"}'
                         + (f'・5秒ずつ {segs} 回作ってつないでいます' if segs > 1 else ''),
                         '英語:' + C.esc(pe)]
                if segs > 1 and len(prompts) > 1:
                    lines.append('区切りごとの動き:' + ' → '.join(C.esc(x.split('.')[0]) for x in prompts))
                lines += ['⚠ ' + C.esc(c) for c in info['changes']]
                lines.append('ドライブに保存しました' if saved else 'ドライブには保存していません')
                ui.add_card('video', path, prev, lines, lambda s2=s2: run(s2, 1, -1))
            ui.hide_bar()
            ui.say(f'✅ できました({n}本)。⬇ ダウンロード、または Cast Frost の「＋ 素材を追加」で入れてください。')
        except Exception as e:
            ui.error(e)
        finally:
            go.disabled = False

    def on_go(b):
        if not picker.path:
            ui.say('⚠ ① で動かす画像を選んでください。', 'warn')
            return
        if not prompt.value.strip() and not en.value.strip():
            ui.say('⚠ ② でどう動かすかを書いてください。', 'warn')
            return
        ss = dict(src=picker.path, prompt=prompt.value, en=en.value.strip(), matte=matte.value, sec=length.value,
                  loop=loop.value, smooth=smooth.value, thr=thr.value, flicker=flicker.value, steps=steps.value,
                  save=save.value, parts=parts.value)
        run(ss, count.value, seed.value)

    go.on_click(on_go)
    ui.say('✅ 準備ができました。① 画像を選ぶ → ② 動きを書く → ✨ 動かす、の順に進めてください。')
