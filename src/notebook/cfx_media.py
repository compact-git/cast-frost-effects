# Cast Frost 素材メーカー:渡された素材(画像・動画)を受け取る・読み込む部品
# - セルの中のアップロードボタン(ドライブを意識しなくてよい)
# - ドライブの「CastFrost素材_入力」と「CastFrost素材」のフォルダから選ぶ(大きいファイル向け)
import os, re, json, subprocess, shutil, time, base64, itertools
import numpy as np
import cfx_common as C

IN_DIR = C.MYDRIVE + '/CastFrost素材_入力'
UPLOAD_DIR = C.ROOT + '/in'
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
VIDEO_EXT = ('.webm', '.mp4', '.mov', '.mkv', '.avi', '.m4v', '.gif')
MAX_SECONDS = 30.0          # 読み込む動画の長さの上限
MAX_SIDE = 1280             # 読み込む動画の長い辺の上限(これより大きい時は縮める)
UPLOAD_WARN_MB = 100        # これより大きいファイルは、ドライブから選ぶのをすすめる


def kind_of(path):
    p = path.lower()
    if p.endswith('.gif'):
        return 'video' if _gif_frames(path) > 1 else 'image'
    if p.endswith(IMAGE_EXT):
        return 'image'
    if p.endswith(VIDEO_EXT):
        return 'video'
    return None


def _gif_frames(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return getattr(im, 'n_frames', 1)
    except Exception:
        return 1


# ---------------------------------------------------------------- 受け取る
def save_upload(value):
    """ipywidgets の FileUpload の値を、ファイルにして保存する(ipywidgets 7 と 8 のどちらの形でも)"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    items = []
    if isinstance(value, dict):
        for name, d in value.items():
            items.append((name, d.get('content')))
    else:
        for d in value or ():
            items.append((d.get('name') or d.get('metadata', {}).get('name'), d.get('content')))
    paths = []
    for name, content in items:
        if not name or content is None:
            continue
        safe = os.path.basename(name).replace('/', '_')
        path = os.path.join(UPLOAD_DIR, f'{int(time.time())}_{safe}')
        with open(path, 'wb') as f:
            f.write(bytes(content))
        paths.append(path)
    return paths


def _upload_value(up):
    """FileUpload の中身を取り出す(value が空でも、ipywidgets 7 の metadata / data から組み立てる)"""
    v = getattr(up, 'value', None)
    if v:
        return v
    md, data = getattr(up, 'metadata', None), getattr(up, 'data', None)
    if md and data:
        return {m.get('name'): {'content': d} for m, d in zip(md, data)}
    return v


def _safe_name(name):
    n = os.path.basename(str(name or 'file')).replace('/', '_').replace('\\', '_')
    return n[-120:] or 'file'


_seq = itertools.count(1)


def drive_inputs(kinds=('image', 'video')):
    """ドライブの入力フォルダと、作った素材のフォルダにあるファイルを、新しい順に並べる"""
    out = []
    if not C.drive_ok():
        return out
    os.makedirs(IN_DIR, exist_ok=True)
    for label, d in (('入力', IN_DIR), ('作った素材', C.OUT_DIR)):
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            p = os.path.join(d, n)
            if not os.path.isfile(p) or n.startswith('.'):
                continue
            k = kind_of(p)
            if k in kinds:
                out.append((os.path.getmtime(p), f'{label}/{n}', p))
    out.sort(reverse=True)
    return [(lab, p) for _, lab, p in out[:200]]


# ---------------------------------------------------------------- 画像
def read_image(path):
    """画像を RGBA で読み込む(透明がない時は、すべて不透明)"""
    from PIL import Image, ImageOps
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert('RGBA')


def has_alpha(img):
    a = np.asarray(img.convert('RGBA'))[..., 3]
    return bool((a < 250).mean() > 0.002)


# ---------------------------------------------------------------- 動画
def probe(path):
    p = subprocess.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_streams', '-show_format', '-of', 'json', path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = json.loads(p.stdout or b'{}')
    st = (d.get('streams') or [{}])[0]
    if not st:
        raise RuntimeError('動画として読み込めませんでした。別の形式(MP4・WebM)で試してください。')
    tags = {k.lower(): v for k, v in (st.get('tags') or {}).items()}

    def fr(s):
        try:
            a, b = s.split('/')
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0
    fps = fr(st.get('avg_frame_rate', '0/1')) or fr(st.get('r_frame_rate', '0/1')) or 30.0
    dur = float(st.get('duration') or (d.get('format') or {}).get('duration') or 0)
    codec = st.get('codec_name')
    alpha = tags.get('alpha_mode') == '1' or (st.get('pix_fmt') or '').startswith(('yuva', 'rgba', 'argb', 'bgra', 'ya'))
    return dict(codec=codec, width=int(st.get('width') or 0), height=int(st.get('height') or 0), fps=fps,
                seconds=dur, alpha=alpha)


def read_video(path, start=0.0, end=0.0, max_side=MAX_SIDE, max_seconds=MAX_SECONDS, log=print):
    """動画を RGBA のコマの並びとして読み込む。透明つきの WebM(VP8/VP9)も、透明のまま読む。
    戻り値:(コマの list, fps, 情報)"""
    info = probe(path)
    fps = min(60.0, info['fps'] or 30.0)
    fps = round(fps, 3)
    w, h = info['width'], info['height']
    s = min(1.0, max_side / float(max(w, h) or 1))
    ow, oh = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    notes = []
    if s < 1.0:
        notes.append(f'大きいので {w}×{h} → {ow}×{oh} に縮めて読み込みました')
    dur = info['seconds'] or 0
    start = max(0.0, float(start or 0))
    end = float(end or 0)
    if end <= 0 or (dur and end > dur):
        end = dur or 0
    if end and end - start > max_seconds:
        notes.append(f'長いので、はじめの {max_seconds:.0f} 秒だけを使います')
        end = start + max_seconds
    pre = []
    if info['codec'] == 'vp9':
        pre = ['-c:v', 'libvpx-vp9']
    elif info['codec'] == 'vp8':
        pre = ['-c:v', 'libvpx']
    cmd = [shutil.which('ffmpeg') or 'ffmpeg', '-v', 'error']
    if start > 0:
        cmd += ['-ss', f'{start:.3f}']
    cmd += pre + ['-i', path]
    if end:
        cmd += ['-t', f'{max(0.05, end - start):.3f}']
    vf = [f'fps={fps}', f'scale={ow}:{oh}:flags=lanczos']
    cmd += ['-vf', ','.join(vf), '-an', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError('動画を読み込めませんでした。別の形式(MP4・WebM)で試してください。\n'
                           + p.stderr.decode('utf-8', 'replace')[-600:])
    n = len(p.stdout) // (ow * oh * 4)
    arr = np.frombuffer(p.stdout[:n * ow * oh * 4], np.uint8).reshape(n, oh, ow, 4)
    frames = [np.array(f) for f in arr]
    info.update(read_frames=n, read_width=ow, read_height=oh, read_fps=fps, notes=notes,
                read_alpha=bool(info['alpha'] and any((f[..., 3] < 250).any() for f in frames[:5])))
    return frames, fps, info


def resize_frames(frames, long_side):
    """コマの長い辺を long_side にそろえる(0 ならそのまま)"""
    if not long_side:
        return frames
    from PIL import Image
    h, w = frames[0].shape[:2]
    s = long_side / float(max(w, h))
    if abs(s - 1.0) < 1e-3:
        return frames
    ow, oh = max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)
    return [np.asarray(Image.fromarray(f, 'RGBA').resize((ow, oh), Image.LANCZOS)) for f in frames]


def fit_sdxl(img, area=1024 * 1024):
    """SDXL で扱いやすい大きさ(64 の倍数・面積は約 1024×1024)にそろえる"""
    from PIL import Image
    w, h = img.size
    s = (area / float(w * h)) ** 0.5
    ow, oh = max(512, int(round(w * s / 64)) * 64), max(512, int(round(h * s / 64)) * 64)
    return img.resize((ow, oh), Image.LANCZOS)


# ---------------------------------------------------------------- 素材を選ぶ画面
UPLOADER_HTML = r"""
<div id="__ID__" style="display:none;margin:2px 0 4px">
<label style="display:inline-block;background:#2f86d8;color:#ffffff;font:bold 14px sans-serif;padding:9px 18px;
border-radius:10px;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.35)">📤 ファイルを選ぶ
<input type="file" accept="__ACCEPT__" style="display:none"></label>
<span class="st" style="margin-left:10px;font:13px sans-serif"></span>
</div>
<script>
(async () => {
  const root = document.getElementById('__ID__');
  if (!root || !(window.google && google.colab && google.colab.kernel)) return;
  const call = async (op, n, d) => {
    const r = await google.colab.kernel.invokeFunction('__CB__', [op, n, d], {});
    const j = r && r.data && r.data['application/json'];
    if (j && j.ok === false) throw new Error(j.err);
  };
  try { await call('ready', '', ''); } catch (e) { return; }
  root.style.display = 'block';
  const inp = root.querySelector('input'), st = root.querySelector('.st'), lab = root.querySelector('label');
  const b64 = (buf) => { let s = ''; const u = new Uint8Array(buf);
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
    return btoa(s); };
  inp.addEventListener('change', async () => {
    const f = inp.files[0]; if (!f) return;
    lab.style.opacity = 0.5; lab.style.pointerEvents = 'none';
    try {
      await call('start', f.name, '');
      const CH = 2 * 1024 * 1024;
      for (let o = 0; o < f.size; o += CH) {
        await call('chunk', '', b64(await f.slice(o, o + CH).arrayBuffer()));
        st.textContent = '送っています… ' + Math.min(100, Math.round((o + CH) / f.size * 100)) + '%';
      }
      await call('end', f.name, '');
      st.textContent = '✅ 送りました:' + f.name;
    } catch (e) {
      st.textContent = '⚠ 送れませんでした(' + e.message + ')';
      try { await call('abort', '', ''); } catch (_) {}
    }
    lab.style.opacity = 1; lab.style.pointerEvents = ''; inp.value = '';
  });
})();
</script>
"""


class Picker:
    """素材を選ぶ画面。① アップロード(おすすめ・ドライブを意識しなくてよい)② ドライブのフォルダから選ぶ"""

    def __init__(self, ui, kinds=('image', 'video'), on_change=None):
        import ipywidgets as W
        self.ui, self.kinds, self.on_change = ui, kinds, on_change
        self.path = None
        self.kind = None
        acc = []
        if 'image' in kinds:
            acc.append('image/*')
        if 'video' in kinds:
            acc += ['video/*', '.webm', '.gif', '.mov']
        what = '画像・動画' if len(kinds) == 2 else ('画像' if kinds == ('image',) else '動画')
        self.accept = ','.join(acc)
        # 予備:ipywidgets の FileUpload(Colab の外や、下のボタンが出ない時だけ見える)
        self.up = W.FileUpload(accept=self.accept, multiple=False, description='📤 ファイルを選ぶ',
                               layout=W.Layout(width='220px'))
        self.up.style.button_color = '#2f86d8'
        self.up.style.font_weight = 'bold'
        self.up.observe(self._uploaded, names=['value', 'data', '_counter'])
        self._fh = None
        self._up_path = None
        self.dd = W.Dropdown(options=[('(ドライブから選ぶ時は、ここを開く)', '')], value='',
                             layout=W.Layout(width='100%', max_width='620px'))
        self.dd.observe(self._picked, names='value')
        self.rf = W.Button(description='🔄', tooltip='ドライブのフォルダを読み直す', layout=W.Layout(width='48px'))
        self.rf.on_click(lambda b: self.refresh())
        self.cur = W.HTML(C.msg_html('warn', f'まだ{what}が選ばれていません。'))
        up_note = W.HTML(C.CSS + f'<div class="cfx cfx-meta">パソコンの{what}を選びます(おすすめ)。'
                         f'{UPLOAD_WARN_MB}MB より大きい動画は、下のドライブから選ぶほうが速く確実です。</div>')
        dr_note = W.HTML(C.CSS + '<div class="cfx cfx-meta">大きいファイルは、Google ドライブの '
                         '<b>マイドライブ / CastFrost素材_入力</b> に入れてから 🔄 を押して選びます'
                         '(作った素材の <b>CastFrost素材</b> からも選べます)。</div>')
        drive_box = W.Accordion(children=[W.VBox([dr_note, W.HBox([self.dd, self.rf])])])
        drive_box.set_title(0, '📁 ドライブから選ぶ(大きいファイル向け)')
        drive_box.selected_index = None
        # Colab では、Colab の仕組みでファイルを送るボタンを使う(FileUpload は値が届かないことがあるため)
        self.js = W.Output()
        self.up_row = W.HBox([self.up])
        self.widget = W.VBox([self.js, self.up_row, up_note, drive_box, self.cur])
        self._colab_uploader()
        self.refresh()

    def refresh(self):
        try:
            items = drive_inputs(self.kinds)
        except Exception:
            items = []
        opts = [('(ドライブから選ぶ時は、ここを開く)', '')] + items
        self.dd.options = opts
        if not items:
            self.dd.options = [('(ドライブのフォルダに、まだファイルがありません)', '')]

    def _uploaded(self, ch):
        try:
            paths = save_upload(_upload_value(self.up))
        except Exception as e:
            self.ui.error(e)
            return
        if paths and paths[0] != self.path:
            self.set(paths[0], 'アップロード')

    # ---- Colab の仕組みで送る(google.colab.kernel.invokeFunction で少しずつ送る)
    def _colab_uploader(self):
        try:
            from google.colab import output as co
        except Exception:
            return
        from IPython.display import HTML
        name = f'cfx_upload_{os.getpid()}_{next(_seq)}'
        co.register_callback(name, self._recv)
        self.js.append_display_data(HTML(UPLOADER_HTML.replace('__CB__', name).replace('__ID__', name)
                                         .replace('__ACCEPT__', self.accept)))

    def _recv(self, op, name='', data=''):
        from IPython.display import JSON
        try:
            if op == 'ready':
                self.up_row.layout.display = 'none'      # Colab のボタンが出たので、予備のボタンは隠す
            elif op == 'start':
                self._close()
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                self._up_path = os.path.join(UPLOAD_DIR, f'{int(time.time())}_{_safe_name(name)}')
                self._fh = open(self._up_path, 'wb')
                self.cur.value = C.msg_html('warn', f'⏳ 受け取っています:{C.esc(_safe_name(name))}')
            elif op == 'chunk':
                self._fh.write(base64.b64decode(data))
            elif op == 'end':
                self._close()
                self.set(self._up_path, 'アップロード')
            elif op == 'abort':
                self._close()
                if self._up_path and os.path.exists(self._up_path):
                    os.remove(self._up_path)
            return JSON({'ok': True})
        except Exception as e:
            self._close()
            return JSON({'ok': False, 'err': str(e)})

    def _close(self):
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def _picked(self, ch):
        if ch['new']:
            self.set(ch['new'], 'ドライブ')

    def set(self, path, src):
        k = kind_of(path)
        if k not in self.kinds:
            self.path = self.kind = None
            self.cur.value = C.msg_html('err', f'⚠ このファイルは使えません:{C.esc(os.path.basename(path))}')
            return
        self.path, self.kind = path, k
        mb = os.path.getsize(path) / 1e6
        label = '🖼 画像' if k == 'image' else '🎬 動画'
        thumb = ''
        detail = ''
        try:
            if k == 'image':
                from PIL import Image
                im = read_image(path)
                detail = f'{im.width}×{im.height}・' + ('透明あり' if has_alpha(im) else '透明なし')
                t = im.copy()
                t.thumbnail((160, 160))
                p2 = os.path.join(UPLOAD_DIR, '_thumb.png')
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                t.save(p2)
                thumb = C.preview_html('image', p2, 'chk').replace('max-width:320px', 'max-width:160px')
            else:
                inf = probe(path)
                detail = (f'{inf["width"]}×{inf["height"]}・{inf["fps"]:.0f}fps・約{inf["seconds"]:.1f}秒・'
                          + ('透明あり' if inf['alpha'] else '透明なし'))
        except Exception as e:
            detail = f'中身を読めませんでした({C.esc(e)})'
        self.cur.value = (C.CSS + '<div class="cfx cfx-ok" style="display:flex;gap:12px;align-items:center">'
                          + (f'<div style="width:160px">{thumb}</div>' if thumb else '')
                          + f'<div>✅ <b>{label}</b>({src}):{C.esc(re.sub(r"^\d{9,}_", "", os.path.basename(path)))}<br>'
                          f'<small>{C.esc(detail)}・{mb:.1f}MB</small></div></div>')
        if self.on_change:
            self.on_change(path, k)
