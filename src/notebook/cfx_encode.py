# Cast Frost 素材メーカー:透明な WebM(VP9 / yuva420p)の書き出し
# - 30MB に収まるビットレートを逆算して 2 パスで書き出す
# - それでも超える時は、解像度 → fps の順に下げて作り直す
import os, subprocess, tempfile, json, shutil, math
import numpy as np

LIMIT_BYTES = 30_000_000        # Cast Frost の「＋ 素材を追加」は 1つ 30MB まで(余裕を見てこの値)
SAFE = 0.95                     # 狙う大きさ(上限の 95%)
BPP_CAP = 0.30                  # これ以上ビットレートを上げても見た目は変わらない、という上限(1画素1コマあたりのビット)
BPP_FLOOR = 0.035               # これを下回ると画質が落ちすぎるので、解像度か fps を下げる
FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE = shutil.which('ffprobe') or 'ffprobe'


def _even(n):
    n = int(round(n))
    return max(2, n - (n % 2))


def _run(cmd, inp=None):
    p = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError('ffmpeg の書き出しに失敗しました:\n' + p.stderr.decode('utf-8', 'replace')[-1500:])
    return p


def _raw(frames):
    return b''.join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames)


def _vf(w, h, out_w, out_h, fps_in, fps_out):
    f = []
    if fps_out != fps_in:
        f.append(f'fps={fps_out}')
    if (out_w, out_h) != (w, h):
        f.append(f'scale={out_w}:{out_h}:flags=lanczos')
    return ['-vf', ','.join(f)] if f else []


def encode_once(frames, fps, out_path, kbps, out_w=None, out_h=None, fps_out=None):
    """RGBA の配列(H×W×4, uint8)を、決めたビットレートで 2 パス書き出しする"""
    h, w = frames[0].shape[:2]
    out_w, out_h = out_w or w, out_h or h
    fps_out = fps_out or fps
    raw = _raw(frames)
    src = ['-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    common = _vf(w, h, out_w, out_h, fps, fps_out) + [
        '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', f'{int(kbps)}k',
        '-deadline', 'good', '-cpu-used', '2', '-row-mt', '1', '-auto-alt-ref', '0',
        '-an']
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, 'pass')
        _run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y'] + src + common +
             ['-pass', '1', '-passlogfile', log, '-f', 'webm', os.devnull], raw)
        _run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y'] + src + common +
             ['-pass', '2', '-passlogfile', log, '-metadata', 'encoder=Cast Frost 素材メーカー', out_path], raw)
    return os.path.getsize(out_path)


def plan(w, h, fps):
    """試す順番:そのまま → 解像度を下げる → fps を下げる"""
    steps = [(1.0, fps), (0.75, fps)]
    for f in (24, 16):
        if f < fps:
            steps.append((0.75, f))
    steps.append((0.5, min(fps, 16)))
    out, seen = [], set()
    for s, f in steps:
        key = (_even(w * s), _even(h * s), f)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def encode_fit(frames, fps, out_path, limit=LIMIT_BYTES, log=print):
    """30MB 以内に収まるまで、条件を下げながら書き出す。下げた内容を返す"""
    h, w = frames[0].shape[:2]
    dur = len(frames) / float(fps)
    budget_kbps = limit * SAFE * 8 / dur / 1000.0
    changes = []
    attempts = plan(w, h, fps)
    last = None
    for i, (ow, oh, of) in enumerate(attempts):
        cap_kbps = BPP_CAP * ow * oh * of / 1000.0
        floor_kbps = BPP_FLOOR * ow * oh * of / 1000.0
        kbps = min(cap_kbps, budget_kbps)
        is_last = i == len(attempts) - 1
        if kbps < floor_kbps and not is_last:
            changes.append(f'{ow}×{oh}・{of}fps では画質が落ちすぎるので、もう一段下げます')
            continue
        size = encode_once(frames, fps, out_path, kbps, ow, oh, of)
        last = dict(width=ow, height=oh, fps=of, kbps=int(kbps), bytes=size, seconds=round(dur, 2))
        if size <= limit:
            break
        changes.append(f'{ow}×{oh}・{of}fps で {size/1e6:.1f}MB になったので、作り直します')
        # 同じ条件でビットレートだけ下げて、もう一度だけ試す
        kbps2 = kbps * limit * 0.9 / size
        if kbps2 >= floor_kbps or is_last:
            size = encode_once(frames, fps, out_path, kbps2, ow, oh, of)
            last.update(kbps=int(kbps2), bytes=size)
            if size <= limit:
                break
    if last is None or last['bytes'] > limit:
        raise RuntimeError('30MB 以内に収められませんでした。長さを短くして作り直してください。')
    if (last['width'], last['height']) != (w, h):
        changes.append(f'大きさを {w}×{h} → {last["width"]}×{last["height"]} に下げました')
    if last['fps'] != fps:
        changes.append(f'なめらかさを {fps}fps → {last["fps"]}fps に下げました')
    last['changes'] = changes
    return last


def encode_preview(frames, fps, out_path, max_side=320):
    """セルの中で見せるための、小さく軽い見本(透明つき)"""
    h, w = frames[0].shape[:2]
    s = min(1.0, max_side / float(max(w, h)))
    ow, oh = _even(w * s), _even(h * s)
    raw = _raw(frames)
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-'] + \
        _vf(w, h, ow, oh, fps, fps) + [
        '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-crf', '46', '-b:v', '0',
        '-deadline', 'realtime', '-cpu-used', '8', '-row-mt', '1', '-auto-alt-ref', '0', '-an', out_path]
    _run(cmd, raw)
    return out_path


def probe(path):
    """書き出した WebM の中身(透明があるか、大きさ、fps、長さ)を調べる"""
    p = subprocess.run([FFPROBE, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = json.loads(p.stdout or b'{}')
    st = (d.get('streams') or [{}])[0]
    tags = {k.lower(): v for k, v in (st.get('tags') or {}).items()}
    fr = st.get('r_frame_rate', '0/1').split('/')
    return dict(codec=st.get('codec_name'), width=st.get('width'), height=st.get('height'),
                fps=round(float(fr[0]) / float(fr[1] or 1), 2), alpha=tags.get('alpha_mode') == '1',
                seconds=float((d.get('format') or {}).get('duration') or 0), bytes=os.path.getsize(path))
