# Cast Frost 素材メーカー:🛠 素材を加工する
# 持っている画像・動画を、Cast Frost で使える透明な素材に仕上げる
#  🎬 動画の仕上げ / 🌙 黒い背景を透明に / ✂ 背景を切り抜く / 🎨 画像を元に描き直す
import os, time, random
import numpy as np
import cfx_common as C
import cfx_media as MD

FUNCS = [
    ('finish', '🎬 動画の仕上げ', '#0f9d74', ('video',),
     'ループの形・なめらかさ・大きさを整えて、30MB 以内の WebM にします'),
    ('luma', '🌙 黒い背景を透明に', '#455a64', ('image', 'video'),
     '黒い背景の光・炎・きらきらを、明るさから透明にします'),
    ('cut', '✂ 背景を切り抜く', '#e8457a', ('image', 'video'),
     'キャラ・物・ロゴの背景を、AI で切り抜きます'),
    ('redraw', '🎨 画像を元に描き直す', '#f0a500', ('image',),
     '画像の形を生かして、プロンプトで色や雰囲気を変えます(透明な PNG)'),
]
SIZES = [('そのまま', 0), ('長い辺 1280', 1280), ('長い辺 960', 960), ('長い辺 640', 640), ('長い辺 480', 480)]
LOOPS = [('ループにしない(そのまま)', 'none'), ('なめらかにつなぐ', 'smooth'), ('行って戻る(往復)', 'pingpong')]
RIFE_URL = 'https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl'


class Tools:
    def __init__(self, ui, gpu):
        self.ui = ui
        self.gpu = gpu
        self.cutter = None
        self.rife = None
        self.maker = None

    def unload(self):
        if self.cutter is not None:
            self.cutter.unload()
        if self.maker is not None:
            self.maker.unload()
        self.cutter = self.rife = self.maker = None
        C.free_memory()

    def get_cutter(self):
        if self.cutter is None:
            import cfx_matte
            self.cutter = cfx_matte.Cutter(self.ui.log)
        return self.cutter

    def get_rife(self):
        if self.rife is None:
            import cfx_rife
            path = C.LOCAL_MODELS + '/rife/flownet_v4.25.pkl'
            C.download_url(RIFE_URL, path, self.ui.log)
            self.rife = cfx_rife.Interpolator(path, 'cuda' if self.gpu else 'cpu')
        return self.rife

    def get_maker(self, key, cache=True):
        if self.maker is None:
            import cfx_image
            self.maker = cfx_image.ImageMaker(self.ui)
        self.maker.load(key, cache=cache)
        return self.maker


# -------------------------------------------------------------- 画面
def start():
    import ipywidgets as W
    ui = C.UI('素材を加工する', 'tools', '持っている画像・動画を、Cast Frost で使える透明な素材に仕上げます')
    try:
        g = C.gpu_info()
    except Exception:
        g = None
    ui.set_header(g)
    note = ''
    if g is None:
        note = ('<div class="cfx-warn">GPU がつながっていません。「動画の仕上げ」と「黒い背景を透明に」は使えますが、'
                '「背景を切り抜く」「画像を元に描き直す」には GPU が要ります(ランタイム → ランタイムのタイプを変更)。</div>')
    tools = Tools(ui, g)
    try:
        C.switch_to('tools', ui.log)
        C.mount_drive(ui.log)
        C.ensure_out_dir()
        os.makedirs(MD.IN_DIR, exist_ok=True) if C.drive_ok() else None
        ui.set_header(g, note)
        import cfx_image
        C.fix_conflicts(ui.log)
        C.pip_install(cfx_image.PIP, 'image', ui.log)
        C.fix_conflicts(ui.log)
        C.set_active('tools', tools.unload)
        C.translator(ui.log)
    except Exception as e:
        ui.error(e)
        return
    ui.hide_bar()
    _build_form(ui, tools, g)


def _build_form(ui, tools, g):
    import ipywidgets as W
    import cfx_encode as E
    import cfx_image as IM
    L = lambda w='100%': W.Layout(width=w)
    st = {'description_width': '130px'}
    state = {'func': None}

    # ① 素材を選ぶ
    sec1 = W.HTML(C.section_html('① 素材を選ぶ', 'tools'))
    picker = MD.Picker(ui, ('image', 'video'), on_change=lambda p, k: update())
    ui.picker = picker

    # ② やることを選ぶ(色つきのカード)
    sec2 = W.HTML(C.section_html('② やることを選ぶ', 'tools'))
    cards = {}
    card_boxes = []
    for key, title, color, kinds, desc in FUNCS:
        b = W.Button(description=title, layout=W.Layout(width='100%', height='48px'))
        b.style.button_color = color
        b.style.font_weight = 'bold'
        try:
            b.style.text_color = 'white'
        except Exception:
            pass
        b.on_click(lambda _b, k=key: choose(k))
        d = W.HTML(C.CSS + f'<div class="cfx cfx-meta" style="padding:2px 4px 0">{desc}<br>'
                   f'<small>使える素材:{"・".join("画像" if x == "image" else "動画" for x in kinds)}</small></div>')
        cards[key] = b
        card_boxes.append(W.VBox([b, d], layout=W.Layout(border='1px solid #dfe8f2', padding='6px', margin='2px')))
    grid = W.GridBox(card_boxes, layout=W.Layout(grid_template_columns='repeat(2, minmax(240px, 1fr))',
                                                  grid_gap='6px', width='100%', max_width='760px'))

    # ③ 設定
    sec3 = W.HTML(C.section_html('③ 設定', 'tools', 'やることを選ぶと、ここに設定が出ます'))
    size = W.Dropdown(options=SIZES, value=0, description='大きさ', style=st, layout=L('360px'))
    t_start = W.FloatText(value=0, description='開始(秒)', style=st, layout=L('260px'))
    t_end = W.FloatText(value=0, description='終わり(秒)', style=st, layout=L('260px'))
    t_note = W.HTML('<small style="color:#4d6680">終わりが 0 なら最後まで。長い動画は、はじめの30秒までを使います</small>')
    loop = W.Dropdown(options=LOOPS, value='none', description='ループ', style=st, layout=L('400px'))
    smooth = W.Checkbox(value=False, description='なめらかにする(コマを補って2倍に。30fps 以下の動画だけ)', indent=False)
    thr = W.FloatSlider(value=0.06, min=0.0, max=0.3, step=0.01, description='黒を消す強さ', style=st, layout=L('460px'))
    gain = W.FloatSlider(value=1.0, min=1.0, max=3.0, step=0.1, description='光を濃くする', style=st, layout=L('460px'))
    luma_note = W.HTML('<small style="color:#4d6680">黒を消す強さ:大きいほど、暗いところまで透明になります。'
                       '光を濃くする:光がうすく見える時に上げます</small>')
    flicker = W.Checkbox(value=True, description='ちらつきを抑える(動画)', indent=False)
    prompt = W.Textarea(placeholder='例:青い炎のような色にする、水彩画風にする(日本語でOK)', description='プロンプト',
                        layout=W.Layout(width='100%', height='70px'), style=st)
    strength = W.FloatSlider(value=0.55, min=0.2, max=0.9, step=0.05, description='変える強さ', style=st,
                             layout=L('460px'))
    str_note = W.HTML('<small style="color:#4d6680">小さいほど元の画像に近く、大きいほどプロンプトに近くなります</small>')
    count = W.IntSlider(value=1, min=1, max=4, description='枚数', style=st, layout=L('420px'))
    seed = W.IntText(value=-1, description='シード', style=st, layout=L('260px'))
    precut = W.Checkbox(value=True, description='先に背景を切り抜く(透明がない画像の時)', indent=False)
    model = W.Dropdown(options=[(v['label'], k) for k, v in IM.MODELS.items()], value='base', description='モデル',
                       style=st, layout=L('420px'))
    steps = W.IntSlider(value=30, min=10, max=60, description='ステップ数', style=st, layout=L('420px'))
    cfg = W.FloatSlider(value=6.0, min=1.0, max=12.0, step=0.5, description='CFG', style=st, layout=L('420px'))
    en = W.Textarea(placeholder='ここに英語を書くと、翻訳を使わずにこの英語で描き直します', description='英語で直接書く',
                    style=st, layout=W.Layout(width='100%', height='60px'))
    adv = W.Accordion(children=[W.VBox([model, steps, cfg, en])])
    adv.set_title(0, '⚙ 高度な設定(ふつうは触らなくて大丈夫)')
    adv.selected_index = None
    save = W.Checkbox(value=C.drive_ok(), description='できたものをドライブにも保存する', indent=False)

    def on_model(c):
        mm = IM.MODELS[c['new']]
        steps.value, cfg.value = mm['steps'], mm['cfg']
    model.observe(on_model, names='value')

    panel = W.VBox()
    go = ui.big_button('✨ 加工する')
    go.disabled = True
    msg = W.HTML()

    def panel_for(func, kind):
        rows = []
        if func in ('finish', 'luma', 'cut') and kind == 'video':
            rows += [W.HBox([t_start, t_end]), t_note]
        if func == 'luma':
            rows += [thr, gain, luma_note]
        if func == 'cut' and kind == 'video':
            rows += [flicker]
        if func == 'redraw':
            rows += [prompt, strength, str_note, count, W.HBox([seed]), precut, adv]
        if func in ('finish', 'luma', 'cut'):
            rows += [size]
        if kind == 'video' and func in ('finish', 'luma', 'cut'):
            rows += [loop, smooth]
        rows += [save]
        return rows

    def update():
        f = state['func']
        for key, title, color, kinds, desc in FUNCS:
            cards[key].description = ('✔ ' if key == f else '') + title
            cards[key].layout.border = '3px solid #16304a' if key == f else ''
        k = picker.kind
        if not f:
            panel.children = ()
            go.disabled = True
            msg.value = C.msg_html('warn', '② のカードから、やることを選んでください。')
            return
        kinds = next(x[3] for x in FUNCS if x[0] == f)
        if not k:
            panel.children = tuple(panel_for(f, 'video' if kinds == ('video',) else 'image'))
            go.disabled = True
            msg.value = C.msg_html('warn', '① で素材を選んでください。')
            return
        if k not in kinds:
            panel.children = ()
            go.disabled = True
            what = '画像' if kinds == ('image',) else '動画'
            msg.value = C.msg_html('err', f'⚠ この機能は{what}だけに使えます。① で{what}を選び直してください。')
            return
        if f in ('cut', 'redraw') and not g:
            panel.children = ()
            go.disabled = True
            msg.value = C.msg_html('err', '⚠ この機能には GPU が要ります(ランタイム → ランタイムのタイプを変更)。')
            return
        panel.children = tuple(panel_for(f, k))
        if k == 'image' and f == 'redraw':
            try:
                precut.value = not MD.has_alpha(MD.read_image(picker.path))
            except Exception:
                pass
        go.disabled = False
        msg.value = ''

    def choose(key):
        state['func'] = key
        update()

    ui.form.children = (sec1, picker.widget, sec2, grid, sec3, msg, panel, go)
    update()

    # ------------------------------------------------------ 加工
    def finish_video(frames, fps, info, lines, t0, label):
        import cfx_video as V
        frames = V.make_loop(frames, loop.value) if loop.value != 'none' else frames
        if smooth.value:
            if fps <= 32.5:
                ui.log('コマを補って、なめらかにしています…')
                frames = tools.get_rife().double(frames, loop=loop.value != 'none',
                                                 progress=lambda d, t: ui.progress(d, t))
                fps = fps * 2
            else:
                lines.append('⚠ 30fps を超える動画なので、なめらかにする処理はしませんでした')
        name = C.new_name('video', 1, 'webm')
        path = os.path.join(C.LOCAL_OUT, name)
        ui.log('WebM に書き出しています(30MB 以内に収めます)…')
        fps_out = round(fps, 3)
        inf = E.encode_fit(frames, fps_out, path, log=ui.log)
        prev = path + '.preview.webm'
        E.encode_preview(frames, fps_out, prev)
        pr = E.probe(path)
        saved = C.save_to_drive(path) if save.value else None
        head = [f'<b>{C.esc(name)}</b>(WebM・{inf["width"]}×{inf["height"]}・{inf["fps"]}fps・'
                f'{inf["seconds"]:.1f}秒・{inf["bytes"] / 1e6:.1f}MB)',
                f'{label}・{time.time() - t0:.0f}秒・透明:{"あり" if pr["alpha"] else "なし"}']
        out = head + lines + ['⚠ ' + C.esc(c) for c in inf['changes']]
        out.append('ドライブに保存しました' if saved else 'ドライブには保存していません')
        ui.add_card('video', path, prev, out, None)

    def save_png(img, lines, t0, label, redo=None):
        name = C.new_name('image', 1, 'png')
        path = os.path.join(C.LOCAL_OUT, name)
        img.save(path)
        prev = path + '.preview.png'
        p2 = img.copy()
        p2.thumbnail((320, 320))
        p2.save(prev)
        saved = C.save_to_drive(path) if save.value else None
        out = [f'<b>{C.esc(name)}</b>(PNG・{img.width}×{img.height}・{os.path.getsize(path) / 1e6:.1f}MB)',
               f'{label}・{time.time() - t0:.0f}秒'] + lines
        out.append('ドライブに保存しました' if saved else 'ドライブには保存していません')
        ui.add_card('image', path, prev, out, redo)

    def run_redraw(src, ss, n, first_seed):
        mk = tools.get_maker(ss['model'])
        pe = ss['en']
        if not pe:
            if C.has_japanese(ss['prompt']):
                ui.log('日本語を英語に直しています…')
            pe = C.translator(ui.log).translate(ss['prompt'], IM.MODELS[ss['model']]['tr'])
        img = MD.read_image(src)
        if ss['precut']:
            ui.log('先に背景を切り抜いています…')
            img = mk.cut(img)
        img = MD.fit_sdxl(img)
        for i in range(n):
            t0 = time.time()
            sd = first_seed + i if first_seed >= 0 else random.randint(0, 2 ** 31 - 1)
            ui.progress(0, 1, f'{i + 1} / {n} 枚目を描き直しています…')
            out = mk.redraw(img, pe, ss['model'], sd, ss['steps'], ss['cfg'], ss['strength'])
            s2 = dict(ss, en=pe)
            save_png(out, [f'シード {sd}・変える強さ {ss["strength"]:.2f}', '英語:' + C.esc(pe)], t0,
                     '🎨 描き直し', redo=lambda s2=s2: guarded(lambda: run_redraw(src, s2, 1, -1)))

    def guarded(fn):
        go.disabled = True
        try:
            fn()
            ui.hide_bar()
            ui.say('✅ できました。⬇ ダウンロード、または Cast Frost の「＋ 素材を追加」で入れてください。')
        except Exception as e:
            ui.error(e)
        finally:
            go.disabled = False

    def run():
        f, k, src = state['func'], picker.kind, picker.path
        t0 = time.time()
        if f == 'redraw':
            if not prompt.value.strip() and not en.value.strip():
                ui.say('⚠ プロンプトを書いてください。', 'warn')
                return
            ss = dict(prompt=prompt.value, en=en.value.strip(), model=model.value, steps=steps.value, cfg=cfg.value,
                      strength=strength.value, precut=precut.value)
            run_redraw(src, ss, count.value, seed.value)
            return
        if k == 'image':
            img = MD.read_image(src)
            if size.value:
                s = size.value / float(max(img.size))
                img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
            if f == 'luma':
                import cfx_matte
                ui.log('黒い背景を透明にしています…')
                from PIL import Image
                out = Image.fromarray(cfx_matte.luma_key(np.asarray(img), thr.value, gain.value), 'RGBA')
                save_png(out, [f'黒を消す強さ {thr.value:.2f}・光を濃くする {gain.value:.1f}'], t0, '🌙 黒い背景を透明に')
            elif f == 'cut':
                ui.log('背景を切り抜いています…')
                out = tools.get_cutter().cut_image(img)
                save_png(out, [], t0, '✂ 背景を切り抜く')
            return
        ui.log('動画を読み込んでいます…')
        frames, fps, info = MD.read_video(src, t_start.value, t_end.value, log=ui.log)
        lines = ['⚠ ' + C.esc(x) for x in info['notes']]
        frames = MD.resize_frames(frames, size.value)
        label = dict(FUNCS_T)[f]
        if f == 'luma':
            import cfx_matte
            ui.log('黒い背景を透明にしています…')
            n = len(frames)
            out = []
            for i, fr in enumerate(frames):
                out.append(cfx_matte.luma_key(fr, thr.value, gain.value))
                if i % 10 == 0:
                    ui.progress(i + 1, n)
            frames = out
        elif f == 'cut':
            ui.log('背景を1コマずつ切り抜いています…')
            frames = tools.get_cutter().cut_frames(frames, smooth=flicker.value,
                                                   progress=lambda d, t: ui.progress(d, t))
        finish_video(frames, fps, info, lines, t0, label)

    go.on_click(lambda b: guarded(run))
    ui.say('✅ 準備ができました。① 素材を選ぶ → ② やることを選ぶ → ✨ 加工する、の順に進めてください。')


FUNCS_T = [(k, t) for k, t, _, _, _ in FUNCS]
