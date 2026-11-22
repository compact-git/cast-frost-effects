# Cast Frost 素材メーカー:透明にする部品
# - BiRefNet(MIT)で、物・キャラの背景を切り抜く
# - 黒い背景の光・炎などを、明るさから透明度を作って抜く
import numpy as np
import cfx_common as C

BIREFNET = 'ZhengPeng7/BiRefNet'


class Cutter:
    """BiRefNet で背景を切り抜き、透明度を作る"""

    def __init__(self, log=print):
        self.log = log
        self.model = None

    def load(self):
        if self.model is not None:
            return
        C.pip_install(['timm', 'kornia'], 'birefnet', self.log)
        from transformers import AutoModelForImageSegmentation
        d = C.fetch_repo(BIREFNET, ['*.json', '*.py', '*.safetensors', 'LICENSE', 'README.md'], 'birefnet', log=self.log)
        self.log('切り抜きのモデルを読み込んでいます…')
        self.model = AutoModelForImageSegmentation.from_pretrained(d, trust_remote_code=True).eval().to('cuda').half()

    def unload(self):
        self.model = None
        C.free_memory()

    def alpha(self, rgb):
        """RGB(H×W×3, uint8)から、透明度(H×W, 0〜1 の float32)を作る"""
        import torch
        import torch.nn.functional as F
        self.load()
        h, w = rgb.shape[:2]
        t = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).to('cuda').permute(2, 0, 1)[None].float() / 255.0
        t = F.interpolate(t, size=(1024, 1024), mode='bicubic', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
        t = ((t - mean) / std).half()
        with torch.no_grad():
            pred = self.model(t)[-1].sigmoid().float()
        pred = F.interpolate(pred, size=(h, w), mode='bicubic', align_corners=False)[0, 0]
        return pred.clamp(0, 1).cpu().numpy()

    def cut_image(self, img):
        """PIL の画像の背景を切り抜いて、RGBA にする"""
        from PIL import Image
        rgb = np.asarray(img.convert('RGB'))
        a = self.alpha(rgb)
        return Image.fromarray(np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)]), 'RGBA')

    def cut_frames(self, frames, smooth=True, progress=None):
        """動画のコマ(RGBA か RGB、uint8)の背景を、1コマずつ切り抜く。smooth=True で前後のコマとならして、ちらつきを抑える"""
        alphas = []
        n = len(frames)
        for i, f in enumerate(frames):
            alphas.append(self.alpha(f[..., :3]))
            if progress:
                progress(i + 1, n)
        if smooth and n >= 3:
            sm = []
            for i in range(n):
                a0 = alphas[max(0, i - 1)]
                a2 = alphas[min(n - 1, i + 1)]
                sm.append(0.25 * a0 + 0.5 * alphas[i] + 0.25 * a2)
            alphas = sm
        out = []
        for f, a in zip(frames, alphas):
            if f.shape[2] == 4:
                a = a * (f[..., 3].astype(np.float32) / 255.0)
            out.append(np.dstack([f[..., :3], (a * 255 + 0.5).clip(0, 255).astype(np.uint8)]))
        return out


def luma_key(rgb, threshold=0.06, gain=1.0):
    """黒い背景の光・炎などを透明にする。いちばん明るい色の値を透明度にし、色はその分だけ明るく戻す。
    threshold:これより暗いところは完全に透明にする(0〜0.3)/ gain:透明度を強める(1 = そのまま)"""
    x = rgb[..., :3].astype(np.float32) / 255.0
    m = x.max(axis=2)
    a = np.clip((m - threshold) / max(1e-3, 1.0 - threshold) * gain, 0, 1)
    c = np.where(m[..., None] > 1e-3, x / np.maximum(m[..., None], 1e-3), 0)
    c = c * np.clip(m / np.maximum(a, 1e-3), 0, 1)[..., None]
    c = np.where(a[..., None] > 0, c, 0)
    if rgb.shape[2] == 4:
        a = a * (rgb[..., 3].astype(np.float32) / 255.0)
    return np.dstack([(c * 255 + 0.5).clip(0, 255).astype(np.uint8), (a * 255 + 0.5).clip(0, 255).astype(np.uint8)])


def over(rgba, color=(0, 0, 0)):
    """RGBA を、決めた色の上に重ねて RGB にする"""
    x = rgba.astype(np.float32)
    a = x[..., 3:4] / 255.0 if rgba.shape[2] == 4 else np.ones(x.shape[:2] + (1,), np.float32)
    bg = np.array(color, np.float32).reshape(1, 1, 3)
    return (x[..., :3] * a + bg * (1 - a) + 0.5).clip(0, 255).astype(np.uint8)
