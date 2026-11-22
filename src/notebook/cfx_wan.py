# Cast Frost 素材メーカー:Wan-Alpha(透明な部分を持った動画を作るモデル)を、GPU 1枚で動かすための部品
# - Wan-Alpha(WeChatCV/Wan-Alpha, MIT License)の推論コードを読み込み、少しだけ差し替えて使う
#   ・flash-attn がなくても動くよう、注意(attention)を PyTorch の SDPA に置き換える
#   ・14B の本体を 1 層ずつ読みながら、Wan-Alpha の DoRA と LightX2V の LoRA を足し込む(CPU のメモリを使いすぎない)
#   ・GPU のメモリが少ない時は、重みを fp8 でしまって、計算の時だけ bf16 に戻す
import gc, math, os, random, sys, logging
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

WAN_CFG = dict(model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=5120, ffn_dim=13824,
               freq_dim=256, text_dim=4096, out_dim=16, num_heads=40, num_layers=40, window_size=(-1, -1),
               qk_norm=True, cross_attn_norm=True, eps=1e-6)
FPS = 16                     # Wan が作る動画の fps
MAX_FRAMES = 81              # 一度に作れるコマ数(約5秒)。4n+1 にする決まりがある
SHIFT = 5.0
ALPHA_SHIFT_MEAN = 0.05      # Wan-Alpha v2.0 の決まった値
STRIP_PREFIXES = ('model.diffusion_model.', 'diffusion_model.', 'model.')


def frames_for_seconds(sec):
    """秒数から、4n+1 のコマ数を決める(最大 81)"""
    n = int(math.ceil(sec * FPS))
    n = int(math.ceil((n - 1) / 4.0)) * 4 + 1
    return max(5, min(MAX_FRAMES, n))


# ---------------------------------------------------------------- 注意(attention)の置き換え
def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                   causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """flash_attention と同じ形で受けて、PyTorch の scaled_dot_product_attention で計算する"""
    out_dtype = q.dtype
    half = (torch.float16, torch.bfloat16)
    cd = q.dtype if q.dtype in half else dtype
    if q_scale is not None:
        q = q * q_scale
    q = q.transpose(1, 2).to(cd)
    k = k.transpose(1, 2).to(cd)
    v = v.transpose(1, 2).to(cd)
    mask = None
    lk = k.size(2)
    if k_lens is not None and int(k_lens.min()) < lk:
        ar = torch.arange(lk, device=k.device)[None, :]
        mask = (ar < k_lens.to(k.device)[:, None])[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=causal,
                                         scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def import_wan(repo_dir):
    """Wan-Alpha のコードを読み込み、注意の計算を差し替える"""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import wan.modules.attention as A
    import wan.modules.model_dora_lightx2v as M
    A.flash_attention = sdpa_attention
    M.flash_attention = sdpa_attention
    try:
        import wan.modules.clip as CL
        CL.flash_attention = sdpa_attention
    except Exception:
        pass
    return M


# ---------------------------------------------------------------- fp8 でしまう Linear
class FP8Linear(nn.Module):
    """重みを float8(e4m3)でしまい、使う時だけ元の精度に戻す Linear"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer('weight', torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer('scale', torch.ones((), dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @torch.no_grad()
    def set_weight(self, w):
        w = w.float()
        s = (w.abs().max() / 448.0).clamp(min=1e-12)
        self.weight = (w / s).to(torch.float8_e4m3fn)
        self.scale = s.to(torch.float32)

    def forward(self, x):
        w = self.weight.to(torch.bfloat16) * self.scale.to(torch.bfloat16)
        b = None if self.bias is None else self.bias.to(torch.bfloat16)
        if torch.is_autocast_enabled(x.device.type):
            return F.linear(x, w, b)
        return F.linear(x.to(torch.bfloat16), w, b).to(x.dtype)


def _swap_linears_to_fp8(model):
    for blk in model.blocks:
        for name, mod in list(blk.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    with torch.device('meta'):
                        new = FP8Linear(child.in_features, child.out_features, child.bias is not None)
                    setattr(mod, cname, new)


# ---------------------------------------------------------------- DoRA / LoRA の読み込み(Wan-Alpha と同じ読み方)
def _strip(k):
    for p in STRIP_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def dora_map(sd):
    """Wan-Alpha の load_dora と同じ名前の解き方"""
    out = {}
    for key in sd:
        if '.lora_B.' not in key:
            continue
        ks = key.split('.')
        i = ks.index('lora_B')
        if len(ks) > i + 2:
            ks.pop(i + 1)
        ks.pop(i)
        if ks[0] == 'diffusion_model':
            ks.pop(0)
        target = '.'.join(ks)
        mag = key.replace('.lora_B.', '.lora_magnitude_vector.').replace('lora_B', 'lora_magnitude_vector')
        out[target] = (key, key.replace('.lora_B.', '.lora_A.'), mag)
    return out


def lora_map(sd):
    """Wan-Alpha の WanLoraWrapper と同じ名前の解き方(lora_A/B・lora_down/up・diff)"""
    pairs, diffs = {}, {}
    for prefix in ('', 'diffusion_model.'):
        for key in sd:
            if not key.startswith(prefix):
                continue
            for a, b in (('lora_A.weight', 'lora_B.weight'), ('lora_down.weight', 'lora_up.weight')):
                if key.endswith(a):
                    base = key[len(prefix):].replace(a, 'weight')
                    pk = key.replace(a, b)
                    if pk in sd:
                        pairs[base] = (key, pk)
            for suf, tgt in (('diff', 'weight'), ('diff_b', 'bias'), ('diff_m', 'modulation')):
                if key.endswith(suf):
                    diffs[key[len(prefix):].replace(suf, tgt)] = key
    return pairs, diffs


def lora_matches(M, lora_path, cfg=None):
    """LoRA のファイルが、本体のどれだけの重みに当てはまるかを数える(読み込む前に確かめる)"""
    from safetensors import safe_open
    cfg = dict(cfg or WAN_CFG)
    with torch.device('meta'):
        model = M.WanModel_dora_lightx2v(**cfg)
    names = set(model.state_dict().keys())
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        keys = list(f.keys())
    pairs, diffs = lora_map({k: None for k in keys})
    return sum(1 for k in list(pairs) + list(diffs) if k in names)


def _load_sd(path):
    from safetensors.torch import load_file
    if path.endswith('.safetensors'):
        return load_file(path, device='cpu')
    return torch.load(path, map_location='cpu', weights_only=False)


class BaseKeyError(RuntimeError):
    pass


@torch.no_grad()
def build_dit(M, base_path, dora_path=None, lora_path=None, mode='fp8', device='cuda', cfg=None, log=print,
              progress=None, stream=False):
    """14B の本体を 1 つずつ読み、DoRA と LoRA を足してから GPU に置く。
    mode: 'fp8'(重みを fp8 でしまう)/ 'bf16'
    stream=True:層(blocks)の重みは CPU のメモリに置き、使う時だけ GPU に運ぶ(GPU のメモリが少ない時)"""
    from safetensors import safe_open
    cfg = dict(cfg or WAN_CFG)
    device = torch.device(device)
    with torch.device('meta'):
        model = M.WanModel_dora_lightx2v(**cfg)
    names = list(model.state_dict().keys())
    if mode == 'fp8':
        _swap_linears_to_fp8(model)
    dsd = _load_sd(dora_path) if dora_path else {}
    dmap = dora_map(dsd)
    lsd = _load_sd(lora_path) if lora_path else {}
    lpairs, ldiffs = lora_map(lsd)
    n_dora = n_lora = 0
    lora_alpha_warn = any(k.endswith('.alpha') for k in lsd)

    paths = base_path if isinstance(base_path, (list, tuple)) else [base_path]
    handles = [safe_open(p, framework='pt', device='cpu') for p in paths]
    base_keys = {}
    for h in handles:
        for k in h.keys():
            base_keys[_strip(k)] = (h, k)
    missing = [n for n in names if n not in base_keys]
    if missing:
        raise BaseKeyError(f'本体のファイルに足りない重みがあります({len(missing)}個):{missing[:3]}')

    for i, name in enumerate(names):
        h, k = base_keys[name]
        w = h.get_tensor(k).to(device=device, dtype=torch.float32)
        if name in dmap:
            kb, ka, km = dmap[name]
            B = dsd[kb].to(device, torch.float32)
            A = dsd[ka].to(device, torch.float32)
            mag = dsd[km].to(device, torch.float32)
            if B.dim() == 4:
                delta = torch.mm(B.squeeze(3).squeeze(2), A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                delta = torch.mm(B, A)
            wd = w + delta
            if wd.dim() == 4:
                nrm = torch.linalg.norm(wd, dim=tuple(range(1, wd.dim())), keepdim=True)
            else:
                nrm = torch.linalg.norm(wd, dim=1, keepdim=True)
            nrm = nrm.clamp(min=1e-8)
            w = (mag.view_as(nrm) / nrm) * wd
            n_dora += 1
        if name in lpairs:
            ka, kb = lpairs[name]
            A = lsd[ka].to(device, torch.float32)
            B = lsd[kb].to(device, torch.float32)
            if w.shape == (B.shape[0], A.shape[1]):
                w = w + torch.matmul(B, A)
                n_lora += 1
        elif name in ldiffs:
            d = lsd[ldiffs[name]].to(device, torch.float32)
            if d.shape == w.shape:
                w = w + d
                n_lora += 1
        mod_name, attr = name.rsplit('.', 1)
        mod = model.get_submodule(mod_name)
        to_cpu = stream and mod_name.startswith('blocks.')
        if isinstance(mod, FP8Linear):
            if attr == 'weight':
                mod.set_weight(w)
                if to_cpu:
                    mod.weight = _pin(mod.weight)
                    mod.scale = _pin(mod.scale)
            else:
                mod.bias = _pin(w.to(torch.float32)) if to_cpu else w.to(torch.float32)
        else:
            big_linear = isinstance(mod, nn.Linear) and attr == 'weight' and mod_name.startswith('blocks.')
            dt = torch.bfloat16 if big_linear else torch.float32
            t = w.to(dt)
            if to_cpu:
                t = _pin(t)
            mod._parameters[attr] = nn.Parameter(t, requires_grad=False)
        del w
        if progress and (i % 40 == 0 or i == len(names) - 1):
            progress(i + 1, len(names))
    del handles, base_keys
    if dmap and n_dora == 0:
        raise RuntimeError('Wan-Alpha の DoRA が 1 つも当てはまりませんでした(ファイルの形が想定と違います)')
    if lpairs and n_lora == 0:
        raise RuntimeError('LightX2V の LoRA が 1 つも当てはまりませんでした(ファイルの形が想定と違います)')
    if lora_alpha_warn:
        log('(参考)LightX2V の LoRA に alpha の値が入っていましたが、Wan-Alpha の作者のやり方に合わせて使っていません')
    d = cfg['dim'] // cfg['num_heads']
    model.freqs = torch.cat([M.rope_params(1024, d - 4 * (d // 6)), M.rope_params(1024, 2 * (d // 6)),
                             M.rope_params(1024, 2 * (d // 6))], dim=1)
    for n_, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_meta:
            raise RuntimeError(f'読み込めていない重みがあります:{n_}')
    model.eval().requires_grad_(False)
    if stream:
        enable_block_streaming(model, device)
    log(f'本体を読み込みました(DoRA {n_dora}か所・LightX2V {n_lora}か所・{mode}' + ('・1層ずつ運ぶ' if stream else '') + ')')
    return model


def _pin(t):
    t = t.detach().to('cpu')
    try:
        return t.pin_memory()
    except Exception:
        return t


def enable_block_streaming(model, device):
    """各層(block)の重みを CPU に置いたままにし、その層を計算する時だけ GPU に運ぶ"""
    device = torch.device(device)
    model._cfx_stream = True
    for blk in model.blocks:
        st = []
        for mod in blk.modules():
            for n, p in mod._parameters.items():
                if p is not None:
                    st.append((mod, 'p', n, p.data))
            for n, b in mod._buffers.items():
                if b is not None:
                    st.append((mod, 'b', n, b))

        def pre(m, args, kwargs, st=st):
            for mod, k, n, t in st:
                g = t.to(device, non_blocking=True)
                if k == 'p':
                    mod._parameters[n].data = g
                else:
                    mod._buffers[n] = g

        def post(m, args, kwargs, out, st=st):
            for mod, k, n, t in st:
                if k == 'p':
                    mod._parameters[n].data = t
                else:
                    mod._buffers[n] = t

        blk.register_forward_pre_hook(pre, with_kwargs=True)
        blk.register_forward_hook(post, with_kwargs=True)


def move_model(model, dev):
    """本体を動かす。1層ずつ運ぶ設定の時は、層(blocks)以外だけを動かす"""
    if getattr(model, '_cfx_stream', False):
        for name, child in model.named_children():
            if name != 'blocks':
                child.to(dev)
    else:
        model.to(dev)


# ---------------------------------------------------------------- 生成
def resample_mask(mask, t, h, w):
    """Wan-Alpha v2.0 のガウスマスク(21×H×W)を、作る長さと大きさに合わせる"""
    m = mask[None, None].float()  # 1,1,T,H,W
    m = F.interpolate(m, size=(t, h, w), mode='trilinear', align_corners=False)
    return m[0, 0]


class AlphaWan:
    """Wan-Alpha で、透明な部分を持った動画のコマを作る"""

    def __init__(self, M, dit, t5, vae, gauss_mask, device='cuda', t5_on_gpu=True, offload_for_decode=False,
                 log=print):
        self.M, self.dit, self.t5, self.vae = M, dit, t5, vae
        self.gauss_mask = gauss_mask
        self.device = torch.device(device)
        self.t5_on_gpu = t5_on_gpu
        self.offload_for_decode = offload_for_decode
        self.log = log
        self._ctx_cache = {}

    def _dit_to(self, dev):
        move_model(self.dit, dev)
        if dev == 'cpu' or (isinstance(dev, torch.device) and dev.type == 'cpu'):
            gc.collect()
            torch.cuda.empty_cache()

    def _free_gb(self):
        return torch.cuda.mem_get_info()[0] / 1024 ** 3

    @torch.no_grad()
    def encode(self, prompt):
        if prompt in self._ctx_cache:
            return self._ctx_cache[prompt]
        if self.t5_on_gpu:
            ctx = self.t5([prompt], self.device)
        else:
            # 文字を読む部品(T5・約11GB)は、使う時だけ GPU に載せる。入らない時は本体をいったん CPU に逃がす
            swapped = self._free_gb() < 13.0
            if swapped:
                self._dit_to('cpu')
            self.t5.model.to(self.device)
            try:
                ctx = self.t5([prompt], self.device)
                ctx = [c.clone() for c in ctx]
            finally:
                self.t5.model.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
                if swapped:
                    self._dit_to(self.device)
        ctx = [c.to(self.device) for c in ctx]
        if len(self._ctx_cache) > 16:
            self._ctx_cache.clear()
        self._ctx_cache[prompt] = ctx
        return ctx

    @torch.no_grad()
    def generate(self, prompt, width, height, frame_num, seed, steps=4, progress=None):
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        vae_stride, patch = (4, 8, 8), (1, 2, 2)
        z_dim = self.vae.model_fgr.z_dim
        shape = (z_dim, (frame_num - 1) // vae_stride[0] + 1, height // vae_stride[1], width // vae_stride[2])
        seq_len = math.ceil((shape[2] * shape[3]) / (patch[1] * patch[2]) * shape[1])
        seed = seed if seed >= 0 else random.randint(0, 2 ** 31 - 1)
        context = self.encode(prompt)
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        noise = torch.randn(*shape, dtype=torch.float32, device=self.device, generator=g)
        # Wan-Alpha v2.0:透明にしたい外側のノイズを少しずらして、透明さを安定させる
        gm = resample_mask(self.gauss_mask.to(self.device), shape[1], shape[2], shape[3])
        noise = noise + (1 - gm) * torch.tensor(ALPHA_SHIFT_MEAN, dtype=torch.bfloat16, device=self.device)
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(steps, device=self.device, shift=SHIFT)
        latents = [noise]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i, t in enumerate(sched.timesteps):
                ts = torch.stack([t])
                pred = self.dit(latents, t=ts, context=context, seq_len=seq_len)[0]
                x0 = sched.step(pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=g)[0]
                latents = [x0.squeeze(0)]
                if progress:
                    progress(i + 1, steps)
        del noise, pred
        torch.cuda.empty_cache()
        off = self.offload_for_decode or self._free_gb() < 12.0   # 色と透明度を取り出す時に足りなければ、本体を逃がす
        if off:
            self._dit_to('cpu')
        try:
            fgr, pha = self.vae.decode(latents)
        finally:
            if off:
                self._dit_to(self.device)
        frames = to_rgba(fgr[0], pha[0])
        del latents, fgr, pha
        gc.collect()
        torch.cuda.empty_cache()
        return frames, seed


def to_rgba(fgr, pha):
    """VAE の出力(色:3×F×H×W、透明度:3×F×H×W、どちらも -1〜1)を、RGBA のコマに直す"""
    import numpy as np
    rgb = ((fgr.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)   # 3,F,H,W
    a = ((pha.clamp(-1, 1) + 1) * 127.5).mean(0, keepdim=True).round().to(torch.uint8)
    x = torch.cat([rgb, a], 0).permute(1, 2, 3, 0).cpu().numpy()       # F,H,W,4
    return [np.ascontiguousarray(f) for f in x]


# ---------------------------------------------------------------- 画像から動画(Wan2.1 I2V 14B 480P)
I2V_CFG = dict(WAN_CFG, model_type='i2v', in_dim=36)
VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
I2V_SHIFT = 5.0


class PlainVAE:
    """ふつうの Wan2.1 の VAE(色だけ。透明度は出さない)"""

    def __init__(self, vae_pth, device='cuda'):
        from wan.modules.vae import _video_vae
        self.device = torch.device(device)
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=16).eval().requires_grad_(False).to(self.device)
        self.scale = [torch.tensor(VAE_MEAN, device=self.device), 1.0 / torch.tensor(VAE_STD, device=self.device)]
        self.z_dim = 16

    @torch.no_grad()
    def encode(self, video):
        return self.model.encode(video.unsqueeze(0), self.scale).float().squeeze(0)

    @torch.no_grad()
    def decode(self, z):
        return self.model.decode(z.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0)


def _autocast(device):
    from contextlib import nullcontext
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def i2v_size(w, h, max_area=832 * 480):
    """画像の縦横の比から、作る動画の大きさ(16 の倍数)を決める"""
    ar = h / float(w)
    lat_h = int(round((max_area * ar) ** 0.5 // 8 // 2 * 2))
    lat_w = int(round((max_area / ar) ** 0.5 // 8 // 2 * 2))
    return lat_w * 8, lat_h * 8


class I2VWan:
    """Wan2.1 I2V 14B(480P)+ LightX2V で、1枚の画像から動画のコマを作る"""

    def __init__(self, dit, t5, clip, vae, device='cuda', t5_on_gpu=False, log=print):
        self.dit, self.t5, self.clip, self.vae = dit, t5, clip, vae
        self.device = torch.device(device)
        self.t5_on_gpu = t5_on_gpu
        self.log = log
        self._ctx_cache = {}

    def _free_gb(self):
        if self.device.type != 'cuda':
            return 999.0
        return torch.cuda.mem_get_info()[0] / 1024 ** 3

    def _dit_to(self, dev):
        move_model(self.dit, dev)
        if str(dev) == 'cpu' and self.device.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, prompt):
        if prompt in self._ctx_cache:
            return self._ctx_cache[prompt]
        if self.t5_on_gpu or self.device.type != 'cuda':
            ctx = self.t5([prompt], self.device)
        else:
            swapped = self._free_gb() < 13.0
            if swapped:
                self._dit_to('cpu')
            self.t5.model.to(self.device)
            try:
                ctx = [c.clone() for c in self.t5([prompt], self.device)]
            finally:
                self.t5.model.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
                if swapped:
                    self._dit_to(self.device)
        ctx = [c.to(self.device) for c in ctx]
        if len(self._ctx_cache) > 16:
            self._ctx_cache.clear()
        self._ctx_cache[prompt] = ctx
        return ctx

    @torch.no_grad()
    def generate(self, image, prompt, frame_num, seed, steps=4, max_area=832 * 480, progress=None):
        """image:PIL の RGB 画像。戻り値:(RGB のコマ list、使ったシード、(幅, 高さ))"""
        import numpy as np
        import torch.nn.functional as F
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        dev = self.device
        img = torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float().div(255.0)
        img = img.sub(0.5).div(0.5).to(dev)
        w, h = i2v_size(img.shape[2], img.shape[1], max_area)
        lat_h, lat_w = h // 8, w // 8
        T = (frame_num - 1) // 4 + 1
        seq_len = T * lat_h * lat_w // 4
        seed = seed if seed >= 0 else random.randint(0, 2 ** 31 - 1)
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        noise = torch.randn(16, T, lat_h, lat_w, dtype=torch.float32, generator=g, device=dev)
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=dev)
        msk[:, 1:] = 0
        msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        context = self.encode(prompt)
        self.clip.model.to(dev)
        with _autocast(dev):
            clip_context = self.clip.visual([img[:, None, :, :].clone()])
        if self.device.type == 'cuda' and self._free_gb() < 20:
            self.clip.model.to('cpu')
        first = F.interpolate(img[None].cpu(), size=(h, w), mode='bicubic', align_corners=False).transpose(0, 1)
        vid = torch.cat([first, torch.zeros(3, frame_num - 1, h, w)], dim=1).to(dev)
        y = self.vae.encode(vid)
        del vid
        y = torch.cat([msk, y])
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(steps, device=dev, shift=I2V_SHIFT)
        latent = noise
        with _autocast(dev):
            for i, t in enumerate(sched.timesteps):
                pred = self.dit([latent], t=torch.stack([t]), context=[context[0]], clip_fea=clip_context,
                                seq_len=seq_len, y=[y])[0]
                latent = sched.step(pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False,
                                    generator=g)[0].squeeze(0)
                if progress:
                    progress(i + 1, steps)
        del noise, y
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        off = self.device.type == 'cuda' and self._free_gb() < 12.0
        if off:
            self._dit_to('cpu')
        try:
            vid = self.vae.decode(latent)
        finally:
            if off:
                self._dit_to(self.device)
        rgb = ((vid.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        frames = [np.ascontiguousarray(f) for f in rgb]
        del vid, latent
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return frames, seed, (w, h)
