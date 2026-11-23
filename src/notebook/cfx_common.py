# Cast Frost 素材メーカー:静止画と動画で共通の部品
# (ドライブ、GPU の確認、モデルのダウンロード、日本語→英語の翻訳、画面の部品)
import os, sys, re, gc, io, json, time, base64, shutil, threading, subprocess, traceback, html, platform
from datetime import datetime

VERSION = '0.2.6'
AUTHOR = 'made by compact'
SITE_URL = 'https://rider.tokyo/apps/cast/effects/'
APP_URL = 'https://cast.rider.tokyo/'

ROOT = '/content/castfrost'
LOCAL_OUT = ROOT + '/out'
LOCAL_MODELS = ROOT + '/models'
DRIVE_MOUNT = '/content/drive'
MYDRIVE = DRIVE_MOUNT + '/MyDrive'
OUT_DIR = MYDRIVE + '/CastFrost素材'          # 作ったものだけを置く(1階層)
CACHE_DIR = MYDRIVE + '/CastFrost_models'      # モデルをしまう場所(作ったものとは分ける)
MARK = '.cfx_complete'

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # GPU のメモリを細切れにしない
os.environ.setdefault('DIFFUSERS_VERBOSITY', 'error')       # Flax のお知らせなど、関係ない表示を出さない
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
import warnings as _w
_w.filterwarnings('ignore', message='.*HF_TOKEN.*')
_w.filterwarnings('ignore', message='.*Flax classes are deprecated.*')

STATE = {'active': None, 'unload': None, 'translator': None, 'cache': True}


# ---------------------------------------------------------------- 環境
def in_colab():
    return 'google.colab' in sys.modules or os.path.exists('/content')


def jst_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('Asia/Tokyo'))
    except Exception:
        return datetime.now()


def pip_install(pkgs, marker, log=print):
    """足りない部品を入れる(2回目からは飛ばす)"""
    mk = os.path.join(ROOT, '.pip_' + marker)
    if os.path.exists(mk):
        return
    log('部品を入れています(1〜3分ほど)…')
    cmd = [sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check'] + list(pkgs)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError('部品のインストールに失敗しました。\n' + p.stdout[-2000:])
    os.makedirs(ROOT, exist_ok=True)
    open(mk, 'w').write(VERSION)


def fix_conflicts(log=print):
    """Colab に最初から入っている部品で、じゃまになるものを外す(torchao が古いと LoRA が読み込めない)"""
    try:
        from importlib.metadata import version
        v = version('torchao')
    except Exception:
        return
    try:
        nums = tuple(int(x) for x in re.findall(r'\d+', v)[:2])
    except Exception:
        nums = (0, 0)
    if nums >= (0, 16):
        return
    log(f'じゃまになる部品(torchao {v})を外しています…')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for m in list(sys.modules):
        if m == 'torchao' or m.startswith('torchao.'):
            del sys.modules[m]
    for mod, fn in (('peft.import_utils', 'is_torchao_available'), ('transformers.utils.import_utils', 'is_torchao_available'),
                    ('diffusers.utils.import_utils', 'is_torchao_available')):
        try:
            f = getattr(sys.modules[mod], fn)
            if hasattr(f, 'cache_clear'):
                f.cache_clear()
        except Exception:
            pass


def git_clone(url, commit, dest, log=print):
    if os.path.exists(os.path.join(dest, MARK)):
        return dest
    if os.path.exists(dest):
        shutil.rmtree(dest)
    log('プログラムを取ってきています…')
    subprocess.run(['git', 'clone', '-q', url, dest], check=True)
    subprocess.run(['git', '-C', dest, 'checkout', '-q', commit], check=True)
    open(os.path.join(dest, MARK), 'w').write(commit)
    return dest


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    name = p.name
    kind = 'A100' if 'A100' in name else 'H100' if 'H100' in name else 'L4' if 'L4' in name else \
        'T4' if 'T4' in name else 'G4' if ('RTX PRO 6000' in name or 'Blackwell' in name) else 'other'
    return dict(name=name, kind=kind, total_gb=total / 1024 ** 3, free_gb=free / 1024 ** 3,
                bf16=torch.cuda.is_bf16_supported())


def ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3, psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return 0.0, 0.0


def disk_free_gb(path='/content'):
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except Exception:
        return 0.0


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def gpu_mem_text():
    """GPU のメモリの使い方(困った時の記録用)"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 'GPU なし'
        free, total = torch.cuda.mem_get_info()
        return (f'GPU メモリ:使用中 {torch.cuda.memory_allocated() / 2**30:.1f}GB・確保 {torch.cuda.memory_reserved() / 2**30:.1f}GB'
                f'・空き {free / 2**30:.1f}GB / 全体 {total / 2**30:.1f}GB')
    except Exception as e:
        return f'GPU メモリ:不明({e})'


def switch_to(kind, log=print):
    """静止画 ⇔ 動画を切り替える時、前のモデルを片付ける"""
    if STATE['active'] and STATE['active'] != kind and STATE['unload']:
        log('前に使っていたモデルを片付けています…')
        try:
            STATE['unload']()
        except Exception:
            pass
        STATE['unload'] = None
        STATE['active'] = None
        free_memory()


def set_active(kind, unload):
    STATE['active'] = kind
    STATE['unload'] = unload


# ---------------------------------------------------------------- ドライブ
def mount_drive(log=print):
    if os.path.isdir(MYDRIVE):
        return True
    try:
        from google.colab import drive
        log('Google ドライブにつないでいます(許可の画面が出たら「許可」を押してください)…')
        drive.mount(DRIVE_MOUNT)
        return os.path.isdir(MYDRIVE)
    except Exception as e:
        log('ドライブにつなげませんでした:' + str(e))
        return False


def drive_ok():
    return os.path.isdir(MYDRIVE)


def folder_url(path):
    try:
        fid = os.getxattr(path, 'user.drive.id').decode()
        return 'https://drive.google.com/drive/folders/' + fid
    except Exception:
        return 'https://drive.google.com/drive/my-drive'


def ensure_out_dir():
    os.makedirs(LOCAL_OUT, exist_ok=True)
    if drive_ok():
        os.makedirs(OUT_DIR, exist_ok=True)


def new_name(kind, idx, ext):
    """日付_時刻_種類_番号.ext(同じ名前があれば番号を進める)"""
    ts = jst_now().strftime('%Y%m%d_%H%M%S')
    n = idx
    while True:
        name = f'{ts}_{kind}_{n:02d}.{ext}'
        if not os.path.exists(os.path.join(LOCAL_OUT, name)) and not (drive_ok() and os.path.exists(os.path.join(OUT_DIR, name))):
            return name
        n += 1


def save_to_drive(local_path):
    if not drive_ok():
        return None
    dst = os.path.join(OUT_DIR, os.path.basename(local_path))
    shutil.copy2(local_path, dst)
    return dst


# ---------------------------------------------------------------- モデルのダウンロードとドライブへのしまい込み
_copy_threads = []


def _copy_to_drive(src, dst, log):
    try:
        tmp = dst + '.tmp'
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns('.cache', MARK))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.rename(tmp, dst)
        open(os.path.join(dst, MARK), 'w').write(VERSION)
        log(f'ドライブにしまいました:{os.path.basename(dst)}(次からはここから読み込みます)')
    except Exception as e:
        log(f'ドライブにしまえませんでした({os.path.basename(dst)}):{e}')


def drive_free_gb():
    try:
        return shutil.disk_usage(MYDRIVE).free / 1024 ** 3
    except Exception:
        return 0.0


def _dir_gb(path):
    t = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except Exception:
                pass
    return t / 1024 ** 3


def fetch_repo(repo, allow, name, cache=None, log=print):
    """Hugging Face からモデルを取ってくる。cache=True ならドライブにもしまい、次からはそれを使う"""
    from huggingface_hub import snapshot_download
    if cache is None:
        cache = STATE.get('cache', True)
    local = os.path.join(LOCAL_MODELS, name)
    if os.path.exists(os.path.join(local, MARK)):
        return local
    drv = os.path.join(CACHE_DIR, name)
    if cache and drive_ok() and os.path.exists(os.path.join(drv, MARK)):
        log(f'ドライブにしまったモデルを読み込んでいます:{name}')
        shutil.copytree(drv, local, dirs_exist_ok=True)
        return local
    log(f'モデルをダウンロードしています:{name}')
    snapshot_download(repo_id=repo, allow_patterns=allow, local_dir=local)
    open(os.path.join(local, MARK), 'w').write(VERSION)
    if cache and drive_ok():
        need = _dir_gb(local)
        if drive_free_gb() and drive_free_gb() < need * 1.1 + 1:
            log(f'ドライブの空きが足りないので、{name} はしまいません(約{need:.1f}GB 必要)')
            return local
        os.makedirs(CACHE_DIR, exist_ok=True)
        th = threading.Thread(target=_copy_to_drive, args=(local, drv, log), daemon=True)
        th.start()
        _copy_threads.append(th)
    return local


def fetch_file(repo, filename, name, log=print):
    """1つのファイルだけ取ってくる(動画のモデルはドライブにしまわない)"""
    from huggingface_hub import hf_hub_download
    local = os.path.join(LOCAL_MODELS, name)
    path = os.path.join(local, filename)
    if os.path.exists(path + MARK):
        return path
    log(f'ダウンロードしています:{os.path.basename(filename)}')
    p = hf_hub_download(repo_id=repo, filename=filename, local_dir=local)
    open(p + MARK, 'w').write(VERSION)
    return p


def repo_files(repo):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo)


def download_url(url, path, log=print):
    if os.path.exists(path + MARK):
        return path
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f'ダウンロードしています:{os.path.basename(path)}')
    urllib.request.urlretrieve(url, path)
    open(path + MARK, 'w').write(VERSION)
    return path


def drive_usage_gb():
    total = 0
    if os.path.isdir(CACHE_DIR):
        for r, _, fs in os.walk(CACHE_DIR):
            for f in fs:
                try:
                    total += os.path.getsize(os.path.join(r, f))
                except Exception:
                    pass
    return total / 1024 ** 3


# ---------------------------------------------------------------- 日本語 → 英語
JA = re.compile(r'[぀-ヿ㐀-鿿ｦ-ﾟ　-〿]')

SYS = {
    'image': ('You translate Japanese descriptions into English prompts for an image generation AI. '
              'Write one concise English prompt made of short comma-separated phrases. '
              'Keep every detail the user wrote, do not add new objects, and do not mention any background. '
              'Output only the English prompt.'),
    'anime': ('You translate Japanese descriptions into English tags for an anime image generation AI. '
              'Write lowercase English tags separated by commas, like "1girl, long hair, smile, holding sword". '
              'Keep every detail the user wrote and do not add new objects or any background. '
              'Output only the tags.'),
    'video': ('You translate Japanese descriptions of a short video clip into English for a video generation AI. '
              'Write 1 to 3 short English sentences: the shot type (close-up, medium shot or wide shot), '
              'the subject, how it moves, and the visual style. Do not mention any background. '
              'Output only the English text.'),
}
SHOTS = {
    'image': ('青い炎のエフェクト、きらきら光る', 'blue flame effect, glowing, sparkling'),
    'anime': ('笑顔で手を振る女の子、ピンクの髪', '1girl, pink hair, smile, waving hand'),
    'video': ('金色のきらきらした粒が下から上へ舞い上がる、アニメ風',
              'Close-up shot. Golden sparkling particles rise from the bottom to the top. Anime style.'),
}


def has_japanese(s):
    return bool(JA.search(s or ''))


class Translator:
    """日本語のプロンプトを英語に直す(Qwen3-1.7B、Apache-2.0)"""
    REPO = 'Qwen/Qwen3-1.7B'

    def __init__(self, log=print):
        self.log = log
        self.model = None
        self.tok = None
        self.cache = {}

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        local = fetch_repo(self.REPO, ['*.json', '*.safetensors', '*.txt', 'LICENSE'], 'qwen3-1.7b', log=self.log)
        self.tok = AutoTokenizer.from_pretrained(local)
        dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=dt).eval()   # ふだんは CPU に置く

    def unload(self):
        self.model = None
        self.tok = None
        free_memory()

    def translate(self, text, kind='image'):
        text = (text or '').strip()
        if not has_japanese(text):
            return text
        key = (text, kind)
        if key in self.cache:
            return self.cache[key]
        import torch
        self.load()
        ex_ja, ex_en = SHOTS[kind]
        msgs = [{'role': 'system', 'content': SYS[kind]},
                {'role': 'user', 'content': ex_ja}, {'role': 'assistant', 'content': ex_en},
                {'role': 'user', 'content': text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(dev)
        try:
            ids = self.tok(prompt, return_tensors='pt').to(dev)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=200, do_sample=False, repetition_penalty=1.05)
            res = self.tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            self.model.to('cpu')
            free_memory()
        res = re.sub(r'<think>.*?</think>', '', res, flags=re.S).strip().strip('"').strip()
        res = ' '.join(res.split())
        if not res or has_japanese(res):
            raise RuntimeError('翻訳がうまくいきませんでした。言い方を変えるか、「高度な設定」の「英語で直接書く」を使ってください。')
        self.cache[key] = res
        return res


def translator(log=print):
    if STATE['translator'] is None:
        STATE['translator'] = Translator(log)
    STATE['translator'].log = log
    return STATE['translator']


# ---------------------------------------------------------------- わかりやすいエラー
def friendly_error(e):
    s = f'{type(e).__name__}: {e}'
    low = s.lower()
    if 'out of memory' in low or 'outofmemory' in low or 'cuda error: out of memory' in low:
        return ('GPU のメモリが足りなくなりました。大きさや本数を減らして、もう一度押してください。'
                'それでもだめな時は、メニューの「ランタイム」→「セッションを再起動する」を押してから、このセルを押し直してください。')
    if 'no space left' in low:
        return 'Colab のディスクがいっぱいです。「ランタイム」→「ランタイムを接続解除して削除」をしてから、もう一度始めてください。'
    if 'connection' in low or 'timed out' in low or 'http' in low and 'error' in low:
        return 'ダウンロード中に通信が切れました。少し待ってから、このセルをもう一度押してください。'
    if 'torchao' in low:
        return ('部品の組み合わせの問題で止まりました(torchao)。メニューの「ランタイム」→「セッションを再起動する」を押してから、'
                'このセルを押し直してください。')
    if isinstance(e, RuntimeError) and re.search(r'[぀-ヿ]', str(e)):
        return str(e)
    return f'うまくいきませんでした({type(e).__name__})。使い方サイトの「困った時」を見てください:{SITE_URL}trouble.html'


def diag_text(extra=None):
    lines = [f'Cast Frost 素材メーカー v{VERSION}', f'Python {platform.python_version()}']
    try:
        import torch
        lines.append(f'torch {torch.__version__} / CUDA {torch.version.cuda}')
        g = gpu_info()
        lines.append('GPU: ' + (f"{g['name']} {g['total_gb']:.1f}GB(空き {g['free_gb']:.1f}GB)" if g else 'なし'))
    except Exception as e:
        lines.append(f'torch: {e}')
    for m in ('diffusers', 'transformers', 'accelerate', 'peft', 'huggingface_hub'):
        try:
            lines.append(f'{m} {__import__(m).__version__}')
        except Exception:
            pass
    t, a = ram_gb()
    lines.append(f'メモリ(RAM) {t:.0f}GB(空き {a:.0f}GB)/ ディスクの空き {disk_free_gb():.0f}GB')
    lines.append(f'ドライブ: {"つながっています" if drive_ok() else "つながっていません"}')
    if extra:
        lines += list(extra)
    return '\n'.join(lines)


# ---------------------------------------------------------------- 画面の部品
CSS = """
<style>
.cfx-btn,.cfx-btn button{color:#ffffff !important}
.cfx-fold>.cfx-fold-btn{background:#3c5568 !important;color:#ffffff !important;font-weight:bold;text-align:left}
.cfx-fold-body{border-left:3px solid #3c5568;padding-left:10px;margin:2px 0 6px 6px}
.cfx{font-family:"M PLUS Rounded 1c","Hiragino Maru Gothic ProN","Yu Gothic UI","Meiryo",system-ui,sans-serif;color:#16304a;line-height:1.7}
.cfx-banner{display:flex;gap:14px;align-items:center;color:#fff;border-radius:16px;padding:14px 18px;margin:4px 0 6px;box-shadow:0 3px 10px rgba(0,0,0,.12)}
.cfx-bicon{font-size:34px;line-height:1;background:rgba(255,255,255,.22);border-radius:14px;padding:8px 10px}
.cfx-btitle{font-size:21px;font-weight:800;letter-spacing:.02em}
.cfx-bsub{font-size:13px;opacity:.95}
.cfx-head{background:#f6faff;border:1px solid #c9dbee;border-radius:14px;padding:8px 14px;margin:0 0 8px}
.cfx-sec{border-radius:10px;padding:8px 12px;margin:10px 0 6px;font-size:15px}
.cfx-head small{color:#4d6680}
.cfx-pill{display:inline-block;padding:1px 10px;border-radius:999px;background:#dcebfa;color:#1d5a96;font-size:12px;font-weight:700;margin-right:6px}
.cfx-warn{background:#fff3dc;border:1px solid #f0d49a;color:#7a4600;border-radius:10px;padding:8px 12px;margin:6px 0}
.cfx-ok{background:#ddf5ea;border:1px solid #a6e0c6;color:#0f5e43;border-radius:10px;padding:8px 12px;margin:6px 0}
.cfx-err{background:#ffe8e8;border:1px solid #f2b3b3;color:#8a1f1f;border-radius:10px;padding:8px 12px;margin:6px 0}
.cfx-card{display:flex;gap:14px;align-items:flex-start;border:1px solid #c9dbee;border-radius:14px;padding:10px;background:#fff;margin:6px 0}
.cfx-prev{border-radius:10px;overflow:hidden;border:1px solid #c9dbee;flex:0 0 auto;line-height:0}
.cfx-prev img,.cfx-prev video{display:block;max-width:320px;max-height:320px}
.cfx-chk{background-color:#fff;background-image:linear-gradient(45deg,#c8c8c8 25%,transparent 25%),linear-gradient(-45deg,#c8c8c8 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#c8c8c8 75%),linear-gradient(-45deg,transparent 75%,#c8c8c8 75%);background-size:16px 16px;background-position:0 0,0 8px,8px -8px,-8px 0}
.cfx-black{background:#000}.cfx-white{background:#fff}.cfx-green{background:#00b140}
.cfx-meta{font-size:13px;color:#35536f;word-break:break-all;background:#eef5fc;border:1px solid #d3e3f3;border-radius:8px;padding:4px 10px;margin:2px 0}
.cfx-card .cfx-meta,.cfx-head .cfx-meta{background:none;border:0;padding:0}
.cfx-meta b{color:#16304a}
.cfx a{color:#2f86d8}
</style>
"""


def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def preview_html(kind, path, bg='chk'):
    cls = {'chk': 'cfx-chk', 'black': 'cfx-black', 'white': 'cfx-white', 'green': 'cfx-green'}.get(bg, 'cfx-chk')
    if kind == 'image':
        inner = f'<img src="data:image/png;base64,{b64(path)}">'
    else:
        inner = f'<video src="data:video/webm;base64,{b64(path)}" autoplay loop muted playsinline controls></video>'
    return f'<div class="cfx-prev {cls}">{inner}</div>'


ACCENTS = {
    'image': ('#2f86d8', '#6aaef0', '🖼'),
    'video': ('#7c4dff', '#b39dff', '🎬'),
    'i2v': ('#f0700f', '#ffb066', '🎞'),
    'tools': ('#0f9d74', '#5fd3ad', '🛠'),
}


def header_html(title, gpu, note='', accent='image', sub=''):
    c1, c2, icon = ACCENTS.get(accent, ACCENTS['image'])
    g = f"{gpu['name']}({gpu['total_gb']:.0f}GB)" if gpu else 'なし'
    link = folder_url(OUT_DIR) if drive_ok() and os.path.isdir(OUT_DIR) else None
    save = (f'保存先:<a href="{link}" target="_blank">マイドライブ/CastFrost素材/</a>' if link
            else '保存先:ドライブにつながっていないので、ダウンロードだけできます')
    return (CSS + f'<div class="cfx cfx-banner" style="background:linear-gradient(135deg,{c1},{c2})">'
            f'<div class="cfx-bicon">{icon}</div><div><div class="cfx-btitle">{html.escape(title)}</div>'
            f'<div class="cfx-bsub">{html.escape(sub)}</div></div></div>'
            f'<div class="cfx cfx-head" style="border-color:{c2}">'
            f'<span class="cfx-pill" style="background:{c1};color:#fff">v{VERSION}</span>'
            f'<span class="cfx-pill">GPU:{html.escape(g)}</span>'
            f'<small>{AUTHOR}</small><br><small>{save}</small>{note}</div>')


def section_html(title, accent='image', text=''):
    """機能の見出し(色の帯つき)"""
    c1, c2, _ = ACCENTS.get(accent, ACCENTS['image'])
    return (CSS + f'<div class="cfx cfx-sec" style="border-left:6px solid {c1};background:linear-gradient(90deg,{c2}33,transparent)">'
            f'<b style="color:{c1}">{title}</b>' + (f'<div class="cfx-meta">{text}</div>' if text else '') + '</div>')


STOP_HTML = ('<div class="cfx cfx-meta">🛑 使い終わったら GPU を止めましょう(つないだままだとユニットが減り続けます):'
             'メニューの <b>ランタイム</b> → <b>ランタイムを接続解除して削除</b>。'
             'ノートブックのいちばん下の「🛑 GPU を止める」セルでも止められます。'
             f' 使い方:<a href="{SITE_URL}" target="_blank">{SITE_URL}</a></div>')


def msg_html(kind, text):
    cls = {'ok': 'cfx-ok', 'warn': 'cfx-warn', 'err': 'cfx-err'}.get(kind, 'cfx-ok')
    return CSS + f'<div class="cfx {cls}">{text}</div>'


def esc(s):
    return html.escape(str(s))


class Fold:
    """開く/閉じるの箱(Colab では Accordion が出ないことがあるので、ボタンと箱で作る)"""

    def __init__(self, W, title, child, opened=False):
        self.title = title
        self.btn = W.Button(layout=W.Layout(width='auto', min_width='320px'))
        self.btn.style.button_color = '#3c5568'
        self.btn.add_class('cfx-fold-btn')
        self.btn.add_class('cfx-btn')
        self.body = W.VBox([child])
        self.body.add_class('cfx-fold-body')
        self.btn.on_click(lambda b: self.toggle())
        self.widget = W.VBox([self.btn, self.body])
        self.widget.add_class('cfx-fold')
        self.set(opened)

    def set(self, opened):
        self.opened = bool(opened)
        self.body.layout.display = None if self.opened else 'none'
        self.btn.description = ('▾ ' if self.opened else '▸ ') + self.title

    def toggle(self):
        self.set(not self.opened)


class UI:
    """セルの中に出す画面(ipywidgets)。静止画と動画で使い回す"""

    def __init__(self, title, accent='image', sub=''):
        import ipywidgets as W
        from IPython.display import display
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            pass
        self.W = W
        self.title = title
        self.accent = accent
        self.sub = sub
        self.head = W.HTML()
        self.status = W.HTML()
        self.bar = W.IntProgress(min=0, max=100, value=0, layout=W.Layout(width='100%', visibility='hidden'))
        self.form = W.VBox()
        self.results = W.VBox()
        self.log_out = W.Output(layout=W.Layout(max_height='260px', overflow='auto'))
        self.dl_out = W.Output()
        self.bg = W.ToggleButtons(options=[('チェック', 'chk'), ('黒', 'black'), ('白', 'white'), ('緑', 'green')],
                                  value='chk', description='見本の背景', style={'button_width': '70px'})
        self.bg.observe(lambda c: self.redraw(), names='value')
        self.diag_btn = W.Button(description='📋 診断情報を出す', layout=W.Layout(width='180px'))
        self.diag_box = W.Textarea(layout=W.Layout(width='100%', height='160px', display='none'))
        self.diag_btn.style.button_color = '#3c5568'
        self.diag_btn.add_class('cfx-btn')
        self.diag_btn.on_click(lambda b: self.show_diag())
        self.extra_diag = []
        self.logs = Fold(W, '📄 くわしい記録', W.VBox([self.log_out, self.diag_btn, self.diag_box]))
        logs = self.logs.widget
        self.cards = []
        # お知らせと進み具合は、入力欄(ボタン)のすぐ下に出す(準備中は入力欄が空なので、見出しのすぐ下になる)
        self.root = W.VBox([self.head, self.form, self.status, self.bar, self.bg, self.results,
                            W.HTML(STOP_HTML), logs, self.dl_out])
        display(self.root)

    # 表示
    def set_header(self, gpu, note=''):
        self.head.value = header_html(self.title, gpu, note, self.accent, self.sub)

    def big_button(self, text, color=None):
        """「作る」などの大きいボタン(セルの色)"""
        W = self.W
        b = W.Button(description=text, layout=W.Layout(width='220px', height='46px'))
        b.style.button_color = color or ACCENTS.get(self.accent, ACCENTS['image'])[0]
        b.style.font_weight = 'bold'
        b.add_class('cfx-btn')
        try:
            b.style.text_color = 'white'
        except Exception:
            pass
        return b

    def say(self, text, kind='ok'):
        self.status.value = msg_html(kind, text)

    def log(self, text):
        with self.log_out:
            print(f'[{jst_now():%H:%M:%S}] {text}')
        if not text.startswith('(参考)'):
            self.status.value = msg_html('ok', '⏳ ' + esc(text))

    def progress(self, done, total, label=None):
        self.bar.layout.visibility = 'visible'
        self.bar.max = max(1, total)
        self.bar.value = min(done, total)
        if label:
            self.status.value = msg_html('ok', '⏳ ' + esc(label))

    def hide_bar(self):
        self.bar.layout.visibility = 'hidden'

    def error(self, e):
        try:
            import sys
            print('\n―――― うまくいかなかった時の記録 ――――', file=sys.stderr)
            traceback.print_exc()
            print(f'{type(e).__name__}: {e}', file=sys.stderr)
            print(gpu_mem_text(), file=sys.stderr)
        except Exception:
            pass
        with self.log_out:
            traceback.print_exc()
            print(f'{type(e).__name__}: {e}')
            if 'memory' in str(e).lower() or 'OutOfMemory' in type(e).__name__:
                print(gpu_mem_text())
        self.hide_bar()
        self.logs.set(True)   # くわしい記録を開いておく
        self.say('⚠ ' + esc(friendly_error(e)) +
                 '<br><small>くわしい内容は、この画面のいちばん下の「📄 くわしい記録」と、セルのいちばん下の出力に出ています。</small>', 'err')

    def show_diag(self):
        self.diag_box.value = diag_text(self.extra_diag)
        self.diag_box.layout.display = None

    # 結果のカード
    def add_card(self, kind, path, preview_path, lines, on_redo):
        W = self.W
        card = dict(kind=kind, path=path, preview=preview_path, lines=lines)
        prev = W.HTML()
        info = W.HTML(CSS + '<div class="cfx cfx-meta">' + '<br>'.join(lines) + '</div>')
        dl = W.Button(description='⬇ ダウンロード', button_style='primary', layout=W.Layout(width='160px'))
        redo = W.Button(description='🔁 作り直し', layout=W.Layout(width='160px'),
                        tooltip='同じ設定のまま、シードだけ変えてもう1つ作ります')
        dl.on_click(lambda b: self.download(path))
        btns = [dl]
        if on_redo is not None:
            redo.on_click(lambda b: on_redo())
            btns.append(redo)
        card['prev_w'] = prev
        box = W.HBox([prev, W.VBox([info, W.HBox(btns)])],
                     layout=W.Layout(border='1px solid #c9dbee', padding='8px', margin='6px 0'))
        card['box'] = box
        self.cards.insert(0, card)
        self._render_card(card)
        self.results.children = tuple(c['box'] for c in self.cards)

    def _render_card(self, c):
        c['prev_w'].value = CSS + preview_html(c['kind'], c['preview'], self.bg.value)

    def redraw(self):
        for c in self.cards:
            self._render_card(c)

    def download(self, path):
        try:
            from google.colab import files
            with self.dl_out:
                files.download(path)
        except Exception as e:
            self.say('ダウンロードを始められませんでした:' + esc(e) + '(ドライブの保存先からも取り出せます)', 'warn')
