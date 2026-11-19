# Cast Frost 素材メーカー:コマを補ってなめらかにする(RIFE v4.25)
# ネットワークの形は HolyWu/vs-rife(MIT License, Copyright (c) 2021 HolyWu)の IFNet_HDv3_v4_25 をもとにしています。
# 元の RIFE / Practical-RIFE は hzwer さんの作品です(MIT License, Copyright (c) 2021 hzwer)。
# 透明な動画に使えるよう、流れ(flow)と混ぜ方(mask)も返すように変え、透明度(アルファ)も同じ流れで動かします。
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_URL = 'https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl'


def warp(x, flow, div, grid):
    dt = x.dtype
    x = x.float()
    flow = flow.float()
    flow = torch.cat([flow[:, 0:1] / div[0], flow[:, 1:2] / div[1]], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode='bilinear', padding_mode='border', align_corners=True).to(dt)


def conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, inp, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(inp, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear')
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode='bilinear') / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode='bilinear')
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()
        self.scale_list = [16, 8, 4, 2, 1]

    def forward(self, img0, img1, timestep, div, grid, f0, f1):
        img0, img1 = img0.clamp(0, 1), img1.clamp(0, 1)
        w0, w1 = img0, img1
        flow = mask = feat = None
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2], div, grid)
                wf1 = warp(f1, flow[:, 2:4], div, grid)
                fd, mask, feat = blocks[i](torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow,
                                           scale=self.scale_list[i])
                flow = flow + fd
            w0 = warp(img0, flow[:, :2], div, grid)
            w1 = warp(img1, flow[:, 2:4], div, grid)
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m), flow, m


class Interpolator:
    """透明つきのコマ(H×W×4, uint8)の間に、コマを足す"""

    def __init__(self, weight_path, device='cuda'):
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        sd = torch.load(weight_path, map_location='cpu', mmap=True, weights_only=False)
        sd = {k.replace('module.', ''): v for k, v in sd.items() if 'module.' in k}
        net = IFNet()
        missing, unexpected = net.load_state_dict(sd, strict=False)
        unexpected = [u for u in unexpected if not u.startswith(('teacher.', 'caltime.'))]
        if missing or unexpected:
            raise RuntimeError(f'RIFE の重みが合いません: missing={missing[:5]} unexpected={unexpected[:5]}')
        self.net = net.eval().to(self.device, self.dtype)
        self._grid_key = None

    def _prep(self, h, w):
        m = 64
        ph, pw = math.ceil(h / m) * m, math.ceil(w / m) * m
        if self._grid_key != (ph, pw):
            d = self.device
            self.div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=d)
            hor = torch.linspace(-1.0, 1.0, pw, device=d).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
            ver = torch.linspace(-1.0, 1.0, ph, device=d).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
            self.grid = torch.cat([hor, ver], 1)
            self._grid_key = (ph, pw)
        return ph, pw

    def _to_t(self, f, ph, pw):
        t = torch.from_numpy(np.ascontiguousarray(f)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        rgb, a = t[:, :3], t[:, 3:4]
        pm = rgb * a   # 透明なところの色に引っぱられないよう、アルファを掛けた色で動きを見る
        h, w = f.shape[:2]
        pad = (0, pw - w, 0, ph - h)
        return F.pad(pm, pad).to(self.dtype), F.pad(a, pad).to(self.dtype)

    @torch.no_grad()
    def middle(self, fa, fb, t=0.5):
        h, w = fa.shape[:2]
        ph, pw = self._prep(h, w)
        p0, a0 = self._to_t(fa, ph, pw)
        p1, a1 = self._to_t(fb, ph, pw)
        f0, f1 = self.net.encode(p0), self.net.encode(p1)
        ts = torch.full([1, 1, ph, pw], t, dtype=self.dtype, device=self.device)
        pm, flow, m = self.net(p0, p1, ts, self.div, self.grid, f0, f1)
        wa0 = warp(a0, flow[:, :2], self.div, self.grid)
        wa1 = warp(a1, flow[:, 2:4], self.div, self.grid)
        a = (wa0 * m + wa1 * (1 - m)).clamp(0, 1)
        rgb = (pm.float() / a.float().clamp(min=1e-3)).clamp(0, 1)
        out = torch.cat([rgb, a.float()], 1)[0, :, :h, :w].permute(1, 2, 0)
        out = (out * 255.0 + 0.5).clamp(0, 255).byte().cpu().numpy()
        out[..., :3][out[..., 3] == 0] = 0
        return out

    def double(self, frames, loop=False, progress=None):
        """コマ数を 2 倍にする。loop=True なら、最後と最初の間にもコマを足す"""
        out = []
        n = len(frames)
        for i in range(n):
            out.append(frames[i])
            if i + 1 < n:
                out.append(self.middle(frames[i], frames[i + 1]))
            elif loop:
                out.append(self.middle(frames[i], frames[0]))
            if progress:
                progress(i + 1, n)
        return out
