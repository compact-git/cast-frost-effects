# Cast Frost 素材メーカー:🎬 動画を作る(背景が透明な WebM)
# Wan-Alpha v2.0(Wan2.1-T2V-14B をもとに、透明な部分を持った動画を直接作る)+ LightX2V(4ステップで速く作る)
# できたコマは、ループにする → RIFE でコマを補う → 30MB 以内の WebM(VP9・yuva420p)に書き出す
import os, sys, math, random, time
import numpy as np
import cfx_common as C

WAN_ALPHA_GIT = 'https://github.com/WeChatCV/Wan-Alpha.git'
WAN_ALPHA_COMMIT = '9e5d0533da2dea390bbb0e50c59adfebe5613323'
PIP = ['diffusers==0.37.1', 'transformers==4.57.6', 'accelerate>=1.6', 'safetensors', 'huggingface_hub>=0.34',
       'easydict', 'ftfy', 'einops', 'imageio', 'imageio-ffmpeg', 'opencv-python-headless', 'psutil']

# 使うファイル(どれも商用で使えるライセンス。くわしくは使い方サイトの「モデルとライセンス」)
WAN_BASE_REPO = 'Wan-AI/Wan2.1-T2V-14B'                    # Apache-2.0:文字を読む部品(T5)・VAE・トークナイザー
WAN_DIT_REPO = 'Comfy-Org/Wan_2.1_ComfyUI_repackaged'       # Apache-2.0:本体(14B)を 1 ファイルにまとめたもの
WAN_DIT_FILES = ['split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors',
                 'split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors']
ALPHA_REPO = 'htdong/Wan-Alpha-v2.0'                        # Apache-2.0:透明にするための DoRA
ALPHA_VAE = ('htdong/Wan-Alpha-VAE', 'Wan-Alpha-v2-VAE.bin')  # Apache-2.0:色と透明度を取り出す VAE
LIGHTX2V_REPO = 'lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v'  # Apache-2.0:4ステップで作る LoRA
RIFE_FILE = C.LOCAL_MODELS + '/rife/flownet_v4.25.pkl'

SIZES = [('横長 832×480', (832, 480)), ('縦長 480×832', (480, 832)), ('正方形 624×624', (624, 624))]
LENGTHS = [('2秒', 2), ('3秒', 3), ('4秒', 4), ('5秒(1回で作れるいちばん長い長さ)', 5),
           ('8秒', 8), ('10秒', 10), ('15秒', 15), ('20秒', 20)]
LOOPS = [('なめらかにつなぐ(おすすめ)', 'smooth'), ('行って戻る(往復)', 'pingpong'), ('ループにしない', 'none')]
LONGS = [('同じ動きをくり返す(速い)', 'repeat'), ('続けて何本か作ってつなぐ(時間がかかる)', 'chain')]
FADE = 8   # つなぎ目を重ねるコマ数(16fps で 0.5 秒)


def plan_gpu(g, ram_total):
    """GPU を見て、設定を決める"""
    v = g['total_gb']
    if v >= 70:
        return dict(mode='bf16', t5_gpu=True, offload_decode=False, label='最高画質(bf16)', exp=False)
    if v >= 35:
        return dict(mode='fp8', t5_gpu=False, offload_decode=False, label='標準(fp8)', exp=False)
    if v >= 20:
        return dict(mode='fp8', t5_gpu=False, offload_decode=True, label='試験的(fp8・入れ替えながら動かす)', exp=True)
    return None


class VideoMaker:
    def __init__(self, ui, plan):
        self.ui = ui
        self.plan = plan
        self.wan = None
        self.rife = None

    def unload(self):
        self.wan = None
        self.rife = None
        C.free_memory()

    def load(self):
        if self.wan is not None:
            return
        import torch
        log = self.ui.log
        repo = C.git_clone(WAN_ALPHA_GIT, WAN_ALPHA_COMMIT, C.ROOT + '/Wan-Alpha', log)
        import cfx_wan as WW
        M = WW.import_wan(repo)
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae import WanVAE
        base = os.path.join(C.LOCAL_MODELS, 'wan2.1-t2v-14b')
        files = C.repo_files(WAN_BASE_REPO)
        need = ['models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth'] + [f for f in files if f.startswith('google/umt5-xxl/')]
        for i, f in enumerate(need):
            C.fetch_file(WAN_BASE_REPO, f, 'wan2.1-t2v-14b', log)
        dfiles = C.repo_files(WAN_DIT_REPO)
        dit_name = next((f for f in WAN_DIT_FILES if f in dfiles), None)
        if dit_name is None:
            raise RuntimeError('本体(14B)のファイルが見つかりませんでした。作者に知らせてください。')
        log('本体(14B・約29GB)をダウンロードしています。初回は 5〜10 分ほどかかります…')
        dit_path = C.fetch_file(WAN_DIT_REPO, dit_name, 'wan2.1-dit', log)
        afiles = C.repo_files(ALPHA_REPO)
        dora_name = next((f for f in afiles if f.endswith('.safetensors')), None)
        if dora_name is None:
            raise RuntimeError('Wan-Alpha のファイルが見つかりませんでした。作者に知らせてください。')
        dora_path = C.fetch_file(ALPHA_REPO, dora_name, 'wan-alpha-v2', log)
        mask_name = 'gauss_mask' if 'gauss_mask' in afiles else None
        mask_path = C.fetch_file(ALPHA_REPO, mask_name, 'wan-alpha-v2', log) if mask_name else os.path.join(repo, 'gauss_mask')
        vae_lora = C.fetch_file(ALPHA_VAE[0], ALPHA_VAE[1], 'wan-alpha-vae', log)
        lfiles = C.repo_files(LIGHTX2V_REPO)
        cands = [f for f in lfiles if f.startswith('loras/') and f.endswith('.safetensors')]
        cands.sort(key=lambda f: (0 if 'rank64' in f else 1, f))
        if not cands:
            raise RuntimeError('LightX2V の LoRA が見つかりませんでした。作者に知らせてください。')
        lora_path = C.fetch_file(LIGHTX2V_REPO, cands[0], 'lightx2v', log)
        C.download_url('https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl', RIFE_FILE, log)

        log('文字を読む部品(T5)を読み込んでいます…')
        t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cpu'),
                            checkpoint_path=os.path.join(base, 'models_t5_umt5-xxl-enc-bf16.pth'),
                            tokenizer_path=os.path.join(base, 'google/umt5-xxl'))
        if self.plan['t5_gpu']:
            t5.model.to('cuda')
        log('色と透明度を取り出す部品(VAE)を読み込んでいます…')
        vae = WanVAE(vae_pth=os.path.join(base, 'Wan2.1_VAE.pth'), vae_lora_path=vae_lora, device='cuda')
        log(f'本体(14B)を GPU に読み込んでいます({self.plan["label"]})…')
        prog = lambda d, t: self.ui.progress(d, t)
        try:
            dit = WW.build_dit(M, dit_path, dora_path, lora_path, mode=self.plan['mode'], device='cuda', log=log,
                               progress=prog)
        except WW.BaseKeyError as e:
            # まとめたファイルの形が合わない時は、公式の分割ファイル(約57GB)を使う
            log(f'(参考){e}')
            log('公式の本体ファイル(約57GB)に切り替えて、ダウンロードし直します。10分ほどかかります…')
            try:
                os.remove(dit_path)
            except Exception:
                pass
            shards = [C.fetch_file(WAN_BASE_REPO, f, 'wan2.1-t2v-14b', log) for f in files
                      if f.startswith('diffusion_pytorch_model') and f.endswith('.safetensors')]
            dit = WW.build_dit(M, shards, dora_path, lora_path, mode=self.plan['mode'], device='cuda', log=log,
                               progress=prog)
        mask = torch.load(mask_path, map_location='cpu', weights_only=False)
        mask = torch.as_tensor(mask).float()
        self.wan = WW.AlphaWan(M, dit, t5, vae, mask, device='cuda', t5_on_gpu=self.plan['t5_gpu'],
                               offload_for_decode=self.plan['offload_decode'], log=log)
        import cfx_rife
        self.rife = cfx_rife.Interpolator(RIFE_FILE, 'cuda')
        self.ui.hide_bar()
        log('動画のモデルを読み込みました')


# -------------------------------------------------------------- コマの加工(ループ・つなぐ)
def _blend(a, b, w):
    """RGBA の 2 枚を、アルファを考えて混ぜる(w=0 で a、w=1 で b)"""
    fa, fb = a.astype(np.float32) / 255.0, b.astype(np.float32) / 255.0
    pa = fa[..., :3] * fa[..., 3:]
    pb = fb[..., :3] * fb[..., 3:]
    al = fa[..., 3:] * (1 - w) + fb[..., 3:] * w
    pm = pa * (1 - w) + pb * w
    rgb = np.where(al > 1e-4, pm / np.maximum(al, 1e-4), 0)
    return (np.concatenate([rgb, al], -1) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def make_loop(frames, how, k=FADE):
    """ループの形にする。smooth:終わりを始まりに重ねる / pingpong:行って戻る / none:そのまま"""
    n = len(frames)
    if how == 'pingpong':
        return frames + frames[-2:0:-1]
    if how == 'smooth' and n > 2 * k + 2:
        out = []
        for j in range(n - k):
            if j < k:
                w = (j + 1) / (k + 1)
                out.append(_blend(frames[n - k + j], frames[j], w))
            else:
                out.append(frames[j])
        return out
    return list(frames)


def join_clips(clips, k=FADE):
    """何本かのコマを、つなぎ目を重ねながら 1 本にする"""
    out = list(clips[0])
    for c in clips[1:]:
        if len(out) > k and len(c) > k:
            tail = out[-k:]
            out = out[:-k]
            for j in range(k):
                out.append(_blend(tail[j], c[j], (j + 1) / (k + 1)))
            out += c[k:]
        else:
            out += c
    return out


def gen_frames_for(sec, loop, long_mode):
    """長さとループのしかたから、1回で作るコマ数と回数を決める"""
    import cfx_wan as WW
    if sec <= 5:
        if loop == 'pingpong':
            return WW.frames_for_seconds(sec / 2.0 + 1.0 / WW.FPS), 1
        extra = FADE / WW.FPS if loop == 'smooth' else 0
        return WW.frames_for_seconds(min(5.0, sec + extra)), 1
    if long_mode == 'chain' or loop == 'none':
        per = (WW.MAX_FRAMES - FADE) / WW.FPS
        return WW.MAX_FRAMES, int(math.ceil(sec / per))
    return WW.MAX_FRAMES, 1


# -------------------------------------------------------------- 画面
def start():
    import ipywidgets as W
    ui = C.UI('🎬 動画を作る(背景が透明な WebM)')
    try:
        import torch
        g = C.gpu_info()
    except Exception:
        g = None
    ui.set_header(g)
    if g is None:
        ui.say('⚠ GPU がつながっていません。メニューの <b>ランタイム</b> → <b>ランタイムのタイプを変更</b> で、'
               '<b>A100</b>(おすすめ)か <b>L4</b> を選んでから、このセルをもう一度押してください。', 'err')
        return
    ram_t, ram_a = C.ram_gb()
    plan = plan_gpu(g, ram_t)
    if plan is None:
        ui.say(f'⚠ この GPU({C.esc(g["name"])}・{g["total_gb"]:.0f}GB)では、動画を作るにはメモリが足りません。'
               'メニューの <b>ランタイム</b> → <b>ランタイムのタイプを変更</b> で <b>A100</b>(おすすめ)か <b>L4</b> を選んでください。'
               '静止画は今の GPU でも作れます(「🖼 静止画を作る」のセル)。', 'err')
        return
    note = ''
    if plan['exp']:
        note += ('<div class="cfx-warn">L4 では「試験的」に動かします。A100 より時間がかかり、'
                 'メモリが足りずに止まることがあります。止まった時は A100 に変えてください。</div>')
    if ram_t and ram_t < (45 if plan['exp'] else 35):
        note += ('<div class="cfx-warn">メモリ(RAM)が少なめです({:.0f}GB)。止まる時は、「ランタイムのタイプを変更」で'
                 '<b>ハイメモリ</b>をオンにしてください。</div>'.format(ram_t))
    if C.disk_free_gb() < 60:
        note += '<div class="cfx-warn">Colab のディスクの空きが少なめです。動画のモデルは約45GBあります。</div>'
    ui.set_header(g, f'<br><small>動画の設定:{plan["label"]}</small>' + note)
    ui.extra_diag = [f'動画の設定: {plan}']
    maker = VideoMaker(ui, plan)
    try:
        C.switch_to('video', ui.log)
        free = C.gpu_info()['free_gb']
        if free < g['total_gb'] * 0.8:
            ui.say('⚠ GPU のメモリが前の作業で使われたままです。メニューの <b>ランタイム</b> → <b>セッションを再起動する</b> を押してから、'
                   'このセルを押し直してください。', 'err')
            return
        C.mount_drive(ui.log)
        C.ensure_out_dir()
        ui.set_header(g, f'<br><small>動画の設定:{plan["label"]}</small>' + note)
        C.pip_install(PIP, 'video', ui.log)
        maker.load()
        C.set_active('video', maker.unload)
        C.translator(ui.log)
        C.STATE['cache'] = True   # 翻訳のモデル(約4GB)だけはドライブにしまう
    except Exception as e:
        ui.error(e)
        return
    _build_form(ui, maker, plan)


def _build_form(ui, maker, plan):
    import ipywidgets as W
    import cfx_encode as E
    import cfx_wan as WW
    L = lambda w='100%': W.Layout(width=w)
    st = {'description_width': '120px'}
    prompt = W.Textarea(placeholder='例:金色のきらきらした粒が、下からふわっと舞い上がる(日本語でも英語でも書けます)',
                        description='プロンプト', layout=W.Layout(width='100%', height='80px'), style=st)
    count = W.IntSlider(value=1, min=1, max=4, description='本数', style=st, layout=L('420px'))
    seed = W.IntText(value=-1, description='シード', style=st, layout=L('260px'))
    seed_note = W.HTML('<small style="color:#4d6680">-1 だと毎回おまかせ。同じ数字なら同じ動きになります</small>')
    size = W.Dropdown(options=SIZES, value=SIZES[0][1], description='大きさ', style=st, layout=L('420px'))
    length = W.Dropdown(options=LENGTHS, value=5, description='長さ', style=st, layout=L('520px'))
    loop = W.Dropdown(options=LOOPS, value='smooth', description='ループ', style=st, layout=L('420px'))
    smooth = W.Checkbox(value=True, description='なめらかにする(コマを補って 16fps → 32fps)', indent=False)
    # 高度な設定
    long_mode = W.Dropdown(options=LONGS, value='repeat', description='5秒より長い時', style=st, layout=L('520px'))
    steps = W.IntSlider(value=4, min=4, max=8, description='ステップ数', style=st, layout=L('420px'))
    en = W.Textarea(placeholder='ここに英語を書くと、翻訳を使わずにこの英語で作ります', description='英語で直接書く',
                    style=st, layout=W.Layout(width='100%', height='60px'))
    save = W.Checkbox(value=C.drive_ok(), description='できたものをドライブにも保存する', indent=False)
    adv = W.Accordion(children=[W.VBox([
        W.HTML('<small>ふつうは触らなくて大丈夫です。ステップ数を増やすと少しきれいになることがありますが、時間がかかります。</small>'),
        long_mode, steps, en, save])])
    adv.set_title(0, '⚙ 高度な設定(ふつうは触らなくて大丈夫)')
    adv.selected_index = None
    est = W.HTML()
    make_btn = W.Button(description='✨ 作る', button_style='success', layout=W.Layout(width='200px', height='42px'))

    def update_est(*a):
        nf, clips = gen_frames_for(length.value, loop.value, long_mode.value)
        per = {'bf16': 1.2, 'fp8': 1.5}.get(plan['mode'], 1.5) * (nf / 81.0) * (steps.value / 4.0)
        if plan['exp']:
            per *= 3.0
        mins = per * clips * count.value + 0.3 * count.value
        est.value = (f'<small style="color:#4d6680">1回で作るコマ:{nf}(約{nf / WW.FPS:.1f}秒)×{clips}回'
                     f'・かかる時間の目安:約{max(1, round(mins))}分</small>')
    for w in (length, loop, long_mode, steps, count):
        w.observe(update_est, names='value')
    update_est()
    ui.form.children = (prompt, count, W.HBox([seed, seed_note]), size, length, loop, smooth, adv, est, make_btn)

    def run(settings, n, first_seed):
        make_btn.disabled = True
        try:
            pe = settings['en']
            if not pe:
                if C.has_japanese(settings['prompt']):
                    ui.log('日本語を英語に直しています…')
                pe = C.translator(ui.log).translate(settings['prompt'], 'video')
            if not pe:
                ui.say('⚠ プロンプトを書いてください。', 'warn')
                return
            full = pe if 'transparent background' in pe.lower() else 'This video has a transparent background. ' + pe
            w, h = settings['size']
            nf, clips = gen_frames_for(settings['sec'], settings['loop'], settings['long'])
            for i in range(n):
                t0 = time.time()
                base_seed = first_seed + i if first_seed >= 0 else random.randint(0, 2 ** 31 - 1)
                parts = []
                for c in range(clips):
                    lab = f'{i + 1} / {n} 本目' + (f'({c + 1} / {clips} 回目)' if clips > 1 else '') + 'を作っています…'
                    ui.progress(0, settings['steps'], lab)
                    fr, used = maker.wan.generate(full, w, h, nf, base_seed + c, settings['steps'],
                                                  progress=lambda d, t: ui.progress(d, t))
                    parts.append(fr)
                ui.log('ループの形にしています…')
                frames = join_clips(parts) if len(parts) > 1 else parts[0]
                frames = make_loop(frames, settings['loop'])
                fps = WW.FPS
                if settings['smooth']:
                    ui.log('コマを補って、なめらかにしています…')
                    frames = maker.rife.double(frames, loop=settings['loop'] != 'none',
                                               progress=lambda d, t: ui.progress(d, t))
                    fps = WW.FPS * 2
                cyc = len(frames) / float(fps)
                if settings['sec'] > 5 and settings['long'] == 'repeat' and settings['loop'] != 'none':
                    reps = max(1, int(round(settings['sec'] / cyc)))
                    frames = frames * reps
                name = C.new_name('video', i + 1, 'webm')
                path = os.path.join(C.LOCAL_OUT, name)
                ui.log('WebM に書き出しています(30MB 以内に収めます)…')
                info = E.encode_fit(frames, fps, path, log=ui.log)
                prev = path + '.preview.webm'
                E.encode_preview(frames, fps, prev)
                pr = E.probe(path)
                saved = C.save_to_drive(path) if settings['save'] else None
                ss = dict(settings, en=pe)
                lines = [f'<b>{C.esc(name)}</b>(WebM・{info["width"]}×{info["height"]}・{info["fps"]}fps・'
                         f'{info["seconds"]:.1f}秒・{info["bytes"] / 1e6:.1f}MB)',
                         f'シード {base_seed}・{time.time() - t0:.0f}秒・透明:{"あり" if pr["alpha"] else "なし(要確認)"}',
                         '英語:' + C.esc(pe)]
                lines += ['⚠ ' + C.esc(c) for c in info['changes']]
                lines.append('ドライブに保存しました' if saved else 'ドライブには保存していません')
                ui.add_card('video', path, prev, lines, lambda ss=ss: run(ss, 1, -1))
            ui.hide_bar()
            ui.say(f'✅ できました({n}本)。⬇ ダウンロード、または Cast Frost の「＋ 素材を追加」で入れてください。')
        except Exception as e:
            ui.error(e)
        finally:
            make_btn.disabled = False

    def on_make(b):
        if not prompt.value.strip() and not en.value.strip():
            ui.say('⚠ プロンプトを書いてください。', 'warn')
            return
        settings = dict(prompt=prompt.value, en=en.value.strip(), size=size.value, sec=length.value, loop=loop.value,
                        smooth=smooth.value, long=long_mode.value, steps=steps.value, save=save.value)
        run(settings, count.value, seed.value)

    make_btn.on_click(on_make)
    ui.say('✅ 準備ができました。プロンプトを書いて「✨ 作る」を押してください。')
