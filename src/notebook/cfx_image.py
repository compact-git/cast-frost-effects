# Cast Frost 素材メーカー:🖼 静止画を作る(背景が透明な PNG)
# SDXL + LayerDiffuse(最初から透明な部分を持った画像を作る)
# 予備:ふつうに作ってから BiRefNet で背景を切り抜く
import os, sys, random, time
import cfx_common as C

LAYERDIFFUSE_GIT = 'https://github.com/rootonchair/diffuser_layerdiffuse.git'
LAYERDIFFUSE_COMMIT = 'b4150c1ee8e48c2f9dbf1d1083b19a0c5aa83d9d'
PIP = ['diffusers==0.37.1', 'transformers==4.57.6', 'accelerate>=1.6', 'peft>=0.17', 'safetensors',
       'huggingface_hub>=0.34', 'einops', 'opencv-python-headless']

MODELS = {
    'base': dict(label='ふつう(SDXL 1.0)', repo='stabilityai/stable-diffusion-xl-base-1.0', name='sdxl-base-1.0',
                 steps=30, cfg=6.0, sampler='euler', tr='image',
                 suffix=', high quality, detailed',
                 neg='lowres, blurry, jpeg artifacts, text, watermark, signature, frame, border, worst quality, low quality'),
    'anime': dict(label='アニメ調(Animagine XL 4.0)', repo='cagliostrolab/animagine-xl-4.0', name='animagine-xl-4.0',
                  steps=28, cfg=5.0, sampler='euler_a', tr='anime',
                  suffix=', masterpiece, high score, great score, absurdres',
                  neg='lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, '
                      'worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry'),
    'real': dict(label='実写調(RealVisXL V5.0)', repo='SG161222/RealVisXL_V5.0', name='realvisxl-v5.0',
                 steps=30, cfg=5.0, sampler='dpmpp_2m_karras', tr='image',
                 suffix=', photorealistic, high detail',
                 neg='(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), blurry, text, watermark'),
}
SIZES = [
    ('正方形(アイコン・スタンプ)1024×1024', (1024, 1024)),
    ('横長(バナー・テロップ枠)1344×768', (1344, 768)),
    ('縦長(立ち絵・キャラ)832×1216', (832, 1216)),
    ('横(4:3)1152×896', (1152, 896)),
    ('細長い横(帯)1536×640', (1536, 640)),
]
METHODS = [('透明で作る(おすすめ)', 'layer'), ('ふつうに作って背景を切り抜く', 'cutout')]
LD_LORA = ('rootonchair/diffuser_layerdiffuse', 'diffuser_layer_xl_transparent_attn.safetensors')
LD_DECODER = ('LayerDiffusion/layerdiffusion-v1', 'vae_transparent_decoder.safetensors')
LD_ENCODER = 'vae_transparent_encoder.safetensors'
VAE_FIX = 'madebyollin/sdxl-vae-fp16-fix'


class ImageMaker:
    def __init__(self, ui):
        self.ui = ui
        self.pipe = None
        self.model_key = None
        self.cutter = None
        self.cut_tf = None

    # ---------------------------------------------------------- 読み込み
    def unload(self):
        self.pipe = None
        if self.cutter is not None:
            self.cutter.unload()
        self.cutter = None
        self._tenc = None
        self.model_key = None
        C.free_memory()

    def _choose_files(self, repo):
        files = C.repo_files(repo)
        comps = ['unet', 'text_encoder', 'text_encoder_2']
        fp16 = all(any(f.startswith(c + '/') and f.endswith('.fp16.safetensors') for f in files) for c in comps)
        allow = ['model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*'] + [c + '/config.json' for c in comps]
        if fp16:
            allow += [c + '/*.fp16.safetensors' for c in comps]
        else:
            for c in comps:
                allow += [f for f in files if f.startswith(c + '/') and f.endswith('.safetensors') and '.fp16.' not in f]
        return allow, ('fp16' if fp16 else None)

    def load(self, key, cache=True):
        if self.pipe is not None and self.model_key == key:
            return
        import torch
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        log = self.ui.log
        self.pipe = None
        C.free_memory()
        m = MODELS[key]
        ld = C.git_clone(LAYERDIFFUSE_GIT, LAYERDIFFUSE_COMMIT, C.ROOT + '/diffuser_layerdiffuse', log)
        if ld not in sys.path:
            sys.path.insert(0, ld)
        from layer_diffuse.models import TransparentVAEDecoder
        from diffusers import StableDiffusionXLPipeline
        dt = torch.float16
        allow, variant = self._choose_files(m['repo'])
        mdir = C.fetch_repo(m['repo'], allow, m['name'], cache=cache, log=log)
        vdir = C.fetch_repo(VAE_FIX, ['config.json', 'diffusion_pytorch_model.safetensors'], 'sdxl-vae-fp16-fix',
                            cache=cache, log=log)
        ldir = C.fetch_repo(LD_LORA[0], [LD_LORA[1], 'README.md'], 'layerdiffuse-lora', cache=cache, log=log)
        ddir = C.fetch_repo(LD_DECODER[0], [LD_DECODER[1]], 'layerdiffuse-decoder', cache=cache, log=log)
        log('モデルを GPU に読み込んでいます…')
        vae = TransparentVAEDecoder.from_pretrained(vdir, torch_dtype=dt)
        vae.config.force_upcast = False
        vae.set_transparent_decoder(load_file(os.path.join(ddir, LD_DECODER[1])))
        kw = dict(vae=vae, torch_dtype=dt, use_safetensors=True, add_watermarker=False)
        if variant:
            kw['variant'] = variant
        pipe = StableDiffusionXLPipeline.from_pretrained(mdir, **kw)
        pipe.load_lora_weights(ldir, weight_name=LD_LORA[1], adapter_name='layerdiffuse')
        self._set_sampler(pipe, m['sampler'])
        pipe.to('cuda')
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.model_key = pipe, key
        log(f'読み込みました:{m["label"]}')

    def _set_sampler(self, pipe, name):
        from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
        cfg = pipe.scheduler.config
        if name == 'euler_a':
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
        elif name == 'dpmpp_2m_karras':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True)
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(cfg)

    def cut(self, img):
        """BiRefNet で背景を切り抜き、透明度を作る"""
        if self.cutter is None:
            import cfx_matte
            self.cutter = cfx_matte.Cutter(self.ui.log)
        return self.cutter.cut_image(img)

    def _encoder(self):
        if getattr(self, '_tenc', None) is None:
            from safetensors.torch import load_file
            from layer_diffuse.models import TransparentVAEEncoder
            d = C.fetch_repo(LD_DECODER[0], [LD_ENCODER], 'layerdiffuse-encoder', log=self.ui.log)
            self._tenc = TransparentVAEEncoder(load_file(os.path.join(d, LD_ENCODER)))
        return self._tenc

    def redraw(self, img, prompt_en, key, seed, steps, cfg, strength, neg_extra=''):
        """透明つきの画像を元に、プロンプトで描き直して、透明な PNG にする(LayerDiffuse の image to image)"""
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        m = MODELS[key]
        pos = prompt_en.strip().rstrip(',') + m['suffix']
        neg = m['neg'] + ((', ' + neg_extra.strip()) if neg_extra.strip() else '')
        enc = self._encoder()
        i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe, vae=self.pipe.vae)  # 透明用の VAE は型が違うので、明示して渡す
        i2i.set_progress_bar_config(disable=True)
        latents = enc.encode(img.convert('RGBA'), self.pipe)
        g = torch.Generator('cuda').manual_seed(seed)
        n = max(1, int(round(steps * strength)))
        out = i2i(prompt=pos, negative_prompt=neg, image=latents, strength=strength, num_inference_steps=steps,
                  guidance_scale=cfg, generator=g,
                  callback_on_step_end=self._cb(n), callback_on_step_end_tensor_inputs=['latents']).images[0]
        return out.convert('RGBA')

    # ---------------------------------------------------------- 作る
    def make(self, prompt_en, key, size, seed, steps, cfg, neg_extra, method):
        import torch
        m = MODELS[key]
        w, h = size
        pos = prompt_en.strip().rstrip(',') + m['suffix']
        neg = m['neg'] + ((', ' + neg_extra.strip()) if neg_extra.strip() else '')
        g = torch.Generator('cuda').manual_seed(seed)
        pipe = self.pipe
        if method == 'cutout':
            pipe.disable_lora()
            pipe.vae.mod_number = None
        try:
            img = pipe(prompt=pos, negative_prompt=neg, width=w, height=h, num_inference_steps=steps,
                       guidance_scale=cfg, generator=g,
                       callback_on_step_end=self._cb(steps), callback_on_step_end_tensor_inputs=['latents']).images[0]
        finally:
            if method == 'cutout':
                pipe.enable_lora()
                pipe.vae.mod_number = 1
        if method == 'cutout':
            img = self.cut(img)
        return img.convert('RGBA')

    def _cb(self, steps):
        def cb(pipe, i, t, kw):
            self.ui.progress(i + 1, steps + 2)
            return kw
        return cb


# -------------------------------------------------------------- 画面
def start():
    import ipywidgets as W
    ui = C.UI('静止画を作る(背景が透明な PNG)', 'image', '文字から、背景が透明な画像を作ります')
    try:
        g = C.gpu_info() if _torch_ok() else None
    except Exception:
        g = None
    ui.set_header(g)
    if g is None:
        ui.say('⚠ GPU がつながっていません。メニューの <b>ランタイム</b> → <b>ランタイムのタイプを変更</b> で、'
               '<b>A100 / L4 / T4</b> のどれかを選んでから、このセルをもう一度押してください。', 'err')
        return
    maker = ImageMaker(ui)
    try:
        C.switch_to('image', ui.log)
        C.mount_drive(ui.log)
        C.ensure_out_dir()
        ui.set_header(g)
        C.fix_conflicts(ui.log)
        C.pip_install(PIP, 'image', ui.log)
        C.fix_conflicts(ui.log)
        maker.load('base')
        C.set_active('image', maker.unload)
        C.translator(ui.log)
    except Exception as e:
        ui.error(e)
        if 'out of memory' in str(e).lower():
            ui.say('⚠ GPU のメモリが足りません。メニューの <b>ランタイム</b> → <b>セッションを再起動する</b> を押してから、'
                   'このセルを押し直してください。', 'err')
        return
    ui.hide_bar()
    _build_form(ui, maker, g)


def _torch_ok():
    import torch
    return torch.cuda.is_available()


def _build_form(ui, maker, g):
    import ipywidgets as W
    L = lambda w='100%': W.Layout(width=w)
    st = {'description_width': '120px'}
    prompt = W.Textarea(placeholder='例:青い炎がゆらめくエフェクト、きらきら光る(日本語でも英語でも書けます)',
                        description='プロンプト', layout=W.Layout(width='100%', height='80px'), style=st)
    count = W.IntSlider(value=2, min=1, max=8, description='枚数', style=st, layout=L('420px'))
    seed = W.IntText(value=-1, description='シード', style=st, layout=L('260px'))
    seed_note = W.HTML('<small style="color:#4d6680">-1 だと毎回おまかせ。同じ数字なら同じ絵になります</small>')
    size = W.Dropdown(options=SIZES, value=SIZES[0][1], description='大きさ', style=st, layout=L('520px'))
    # 高度な設定
    model = W.Dropdown(options=[(v['label'], k) for k, v in MODELS.items()], value='base', description='モデル',
                       style=st, layout=L('420px'))
    method = W.Dropdown(options=METHODS, value='layer', description='作り方', style=st, layout=L('420px'))
    steps = W.IntSlider(value=MODELS['base']['steps'], min=10, max=60, description='ステップ数', style=st, layout=L('420px'))
    cfg = W.FloatSlider(value=MODELS['base']['cfg'], min=1.0, max=12.0, step=0.5, description='CFG', style=st,
                        layout=L('420px'))
    neg = W.Text(placeholder='入れたくないもの(英語)例:hands, text', description='入れたくないもの', style=st, layout=L())
    en = W.Textarea(placeholder='ここに英語を書くと、翻訳を使わずにこの英語で作ります', description='英語で直接書く',
                    style=st, layout=W.Layout(width='100%', height='60px'))
    save = W.Checkbox(value=C.drive_ok(), description='できたものをドライブにも保存する', indent=False)
    cache = W.Checkbox(value=True, description='モデルをドライブにしまう(次から速く読み込めます・1つ約7GB)', indent=False)

    def on_model(c):
        mm = MODELS[c['new']]
        steps.value, cfg.value = mm['steps'], mm['cfg']
    model.observe(on_model, names='value')
    adv = W.Accordion(children=[W.VBox([
        W.HTML('<small>ふつうは触らなくて大丈夫です。モデルを変えると、最初の1回だけダウンロード(約7GB)に時間がかかります。</small>'),
        model, method, steps, cfg, neg, en, save, cache])])
    adv.set_title(0, '⚙ 高度な設定(ふつうは触らなくて大丈夫)')
    adv.selected_index = None
    make_btn = ui.big_button('✨ 作る')
    ui.form.children = (prompt, W.HBox([count]), W.HBox([seed, seed_note]), size, adv, make_btn)

    def run(settings, n, first_seed):
        make_btn.disabled = True
        C.STATE['cache'] = settings['cache']
        try:
            maker.load(settings['model'], cache=settings['cache'])
            pe = settings['en']
            if not pe:
                kind = MODELS[settings['model']]['tr']
                if C.has_japanese(settings['prompt']):
                    ui.log('日本語を英語に直しています…')
                pe = C.translator(ui.log).translate(settings['prompt'], kind)
            if not pe:
                ui.say('⚠ プロンプトを書いてください。', 'warn')
                return
            for i in range(n):
                sd = first_seed + i if first_seed >= 0 else random.randint(0, 2 ** 31 - 1)
                ui.progress(0, settings['steps'] + 2, f'{i + 1} / {n} 枚目を作っています…')
                t0 = time.time()
                img = maker.make(pe, settings['model'], settings['size'], sd, settings['steps'], settings['cfg'],
                                 settings['neg'], settings['method'])
                name = C.new_name('image', i + 1, 'png')
                path = os.path.join(C.LOCAL_OUT, name)
                img.save(path)
                prev = path + '.preview.png'
                p2 = img.copy()
                p2.thumbnail((320, 320))
                p2.save(prev)
                saved = C.save_to_drive(path) if settings['save'] else None
                ss = dict(settings, en=pe)
                lines = [f'<b>{C.esc(name)}</b>(PNG・{img.width}×{img.height}・{os.path.getsize(path) / 1e6:.1f}MB)',
                         f'シード {sd}・{C.esc(MODELS[settings["model"]]["label"])}・{time.time() - t0:.0f}秒',
                         '英語:' + C.esc(pe),
                         ('ドライブに保存しました' if saved else 'ドライブには保存していません')]
                ui.add_card('image', path, prev, lines, lambda ss=ss: run(ss, 1, -1))
            ui.hide_bar()
            ui.say(f'✅ できました({n}枚)。⬇ ダウンロード、または Cast Frost の「＋ 素材を追加」で入れてください。')
        except Exception as e:
            ui.error(e)
        finally:
            make_btn.disabled = False

    def on_make(b):
        if not prompt.value.strip() and not en.value.strip():
            ui.say('⚠ プロンプトを書いてください。', 'warn')
            return
        settings = dict(prompt=prompt.value, en=en.value.strip(), model=model.value, method=method.value,
                        size=size.value, steps=steps.value, cfg=cfg.value, neg=neg.value, save=save.value,
                        cache=cache.value)
        run(settings, count.value, seed.value)

    make_btn.on_click(on_make)
    ui.say('✅ 準備ができました。プロンプトを書いて「✨ 作る」を押してください。')
