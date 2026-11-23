# ノートブック(.ipynb)を組み立てる:python3 build_notebook.py → ../out/cast-frost-effects-maker_v{VERSION}.ipynb
# cfx_*.py をまとめて圧縮し、開始のセルに埋め込む(セルを押すと /content/castfrost/code に書き出して読み込む)
import os, json, zlib, base64, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, 'cfx_common.py'), encoding='utf-8').read()
VERSION = re.search(r"VERSION = '([^']+)'", src).group(1)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'out')
GITHUB = 'compact-git/cast-frost-effects'
NB_PATH = 'cast-frost-effects-maker.ipynb'
SITE = 'https://rider.tokyo/apps/cast/effects/'
COLAB = f'https://colab.research.google.com/github/{GITHUB}/blob/main/{NB_PATH}'

MODS = ['cfx_common.py', 'cfx_encode.py', 'cfx_rife.py', 'cfx_wan.py', 'cfx_matte.py', 'cfx_media.py',
        'cfx_image.py', 'cfx_video.py', 'cfx_i2v.py', 'cfx_tools.py']
payload = {m: open(os.path.join(HERE, m), encoding='utf-8').read() for m in MODS}
P = base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode('utf-8'), 9)).decode()


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(True))


def code(text, title):
    return dict(cell_type='code', metadata={'cellView': 'form', 'id': title}, execution_count=None, outputs=[],
                source=text.strip('\n').splitlines(True))


BOOT = '''
_CFX_V = '{v}'
_CFX_P = '{p}'
import os as _o, sys as _s, json as _j, zlib as _z, base64 as _b
_d = '/content/castfrost/code'
_o.makedirs(_d, exist_ok=True)
for _n, _t in _j.loads(_z.decompress(_b.b64decode(_CFX_P)).decode('utf-8')).items():
    open(_o.path.join(_d, _n), 'w', encoding='utf-8').write(_t)
if _d not in _s.path:
    _s.path.insert(0, _d)
'''.format(v=VERSION, p=P)

def _x(t):
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _img(svg, alt):
    """Colab のテキストセルは style を消すので、色つきの見出しは SVG の画像にして埋め込む"""
    return f'<img alt="{_x(alt)}" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode()}">'


_FONT = "font-family=\"'Noto Sans JP','Hiragino Sans','Yu Gothic','Meiryo',sans-serif\""


def _grad(c1, c2):
    return (f'<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{c1}"/>'
            f'<stop offset="1" stop-color="{c2}"/></linearGradient></defs>')


def band(icon, title, sub, c1, c2, tag=None, w=780):
    """色つきの見出し。tag を付けると左に白い札(「分類」など)を出す"""
    h = 78 if tag else 64
    x = 18
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {_FONT}>',
             _grad(c1, c2), f'<rect width="{w}" height="{h}" rx="14" fill="url(#g)"/>']
    if tag:
        parts.append(f'<rect x="14" y="14" width="64" height="{h - 28}" rx="10" fill="#ffffff"/>'
                     f'<text x="46" y="{h / 2 + 7:.0f}" text-anchor="middle" font-size="19" font-weight="bold" '
                     f'fill="{c1}">{_x(tag)}</text>')
        x = 94
    ty = 34 if tag else 30
    parts.append(f'<text x="{x}" y="{ty}" font-size="{24 if tag else 21}" font-weight="bold" fill="#ffffff">'
                 f'{_x(icon)} {_x(title)}</text>')
    parts.append(f'<text x="{x + 2}" y="{ty + 24}" font-size="13.5" fill="#ffffff" fill-opacity="0.95">{_x(sub)}</text>')
    parts.append('</svg>')
    return _img(''.join(parts), f'{icon} {title}')


def btn_icon(kind):
    """Colab のセルの再生ボタン ▶ / 停止ボタン ■ に似せた小さな絵(文と並べて使う)"""
    inner = ('<path d="M9.5 7.2v9.6l7.4-4.8z" fill="#ffffff"/>' if kind == 'play'
             else '<rect x="8" y="8" width="8" height="8" rx="1.2" fill="#ffffff"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
           f'<circle cx="12" cy="12" r="11" fill="#5f6368"/>{inner}</svg>')
    alt = '再生ボタン' if kind == 'play' else '停止ボタン'
    return (f'<img alt="{alt}" width="20" height="20" align="absmiddle" '
            f'src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode()}">')


PLAY, STOP = btn_icon('play'), btn_icon('stop')


def chips(items, w=780):
    """機能の一覧(色つきの札を横に並べた画像)"""
    h, gap = 40, 8
    cw = (w - gap * (len(items) - 1)) / len(items)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {_FONT}>']
    for i, (icon, title, c) in enumerate(items):
        x = i * (cw + gap)
        parts.append(f'<rect x="{x:.1f}" y="0" width="{cw:.1f}" height="{h}" rx="10" fill="{c}"/>'
                     f'<text x="{x + cw / 2:.1f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" '
                     f'fill="#ffffff">{_x(icon)} {_x(title)}</text>')
    parts.append('</svg>')
    return _img(''.join(parts), ' / '.join(t for _, t, _ in items))


cells = [
    md(f"""
{band('🎆', f'Cast Frost 素材メーカー v{VERSION}', 'Cast Frost Web の 🎆 演出・⚡ 発動で使う、背景が透明な素材を作ります。made by compact', '#1d5a96', '#6aaef0')}

**つかい方**:① メニューの **ランタイム** → **ランタイムのタイプを変更** で **A100** を選ぶ → ② 使いたいセルの {PLAY}(再生ボタン)を押す → ③ セルの下に出る画面で作る。<br>
途中でやめたい時は、動いているセルの {STOP}(停止ボタン)を押します。使い終わったら、いちばん下の **🛑 GPU を止める** を押します。くわしくは **[使い方サイト]({SITE})**。

{chips([('🖼', '静止画を作る', '#2f86d8'), ('🎬', '動画を作る', '#7c4dff'), ('🎞', '画像から動画を作る', '#f0700f'), ('🛠', '素材を加工する', '#0f9d74')])}
"""),
    md(band('✍', '文章(命令)から作る', 'プロンプト(どんな素材がほしいかを日本語で書いた文章)だけで、背景が透明な素材を作ります', '#1f6fc2', '#6a3de8', tag='分類')),
    code(f"""
#@title 🖼 静止画を作る(文章 → 透明な PNG)
#@markdown {PLAY} を押すと準備が始まり、このセルの下に入力欄が出ます(1回目は 3〜5 分ほど)。T4・L4 でも作れます。
{BOOT}
import cfx_image
cfx_image.start()
""", 'cfx_image'),
    code(f"""
#@title 🎬 動画を作る(文章 → 透明な WebM)
#@markdown {PLAY} を押すと準備が始まります(1回目は 10〜15 分ほど)。GPU は **A100** がおすすめです。
{BOOT}
import cfx_video
cfx_video.start()
""", 'cfx_video'),
    md(band('🧩', '素材(画像/動画)から作る', '持っている画像・動画を渡して、動かしたり、透明に仕上げたりします', '#e0620a', '#0c8a66', tag='分類')),
    code(f"""
#@title 🎞 画像から動画を作る(画像 → 透明な WebM)
#@markdown 1枚の画像を動かして、背景を抜きます(最長30秒。5秒ずつ作ってつなぎます)。{PLAY} を押すと準備が始まります(1回目は 10〜15 分ほど)。GPU は **A100** がおすすめです。
{BOOT}
import cfx_i2v
cfx_i2v.start()
""", 'cfx_i2v'),
    code(f"""
#@title 🛠 素材を加工する(切り抜く・黒を透明に・仕上げ・描き直す)
#@markdown 持っている画像・動画を、Cast Frost で使える透明な素材に仕上げます。{PLAY} を押すと準備が始まります。
{BOOT}
import cfx_tools
cfx_tools.start()
""", 'cfx_tools'),
    md(band('🛑', 'おわる', '使い終わったら GPU を止めます(つないだままだと、ユニットが減り続けます)', '#c0392b', '#e8705f', tag='終了')),
    code('''
#@title 🛑 GPU を止める(使い終わったら押す)
#@markdown ''' + PLAY + ''' を押すと、ドライブへのしまい込みが終わるのを待ってから、Colab の接続を切って GPU を止めます。作った素材は、ドライブの「CastFrost素材」に残ります。
import time as _t
try:
    import cfx_common as _C
    _left = [th for th in _C._copy_threads if th.is_alive()]
    if _left:
        print('ドライブにモデルをしまっている途中です。終わるまで待っています…')
        for _th in _left:
            _th.join(timeout=1800)
except Exception:
    pass
try:
    from google.colab import drive as _dr
    _dr.flush_and_unmount()
except Exception:
    pass
print('GPU を止めます。おつかれさまでした。')
_t.sleep(1)
from google.colab import runtime as _rt
_rt.unassign()
''', 'cfx_stop'),
    md(band('📖', 'くわしい説明', 'ドライブの使い方・GPU の目安・困った時', '#455a64', '#7f97a3', tag='説明') + '''

### Google ドライブの使い方(消費量)

| しまうもの | 場所 | 大きさの目安 |
|---|---|---|
| 作った素材(PNG / WebM) | マイドライブ / **CastFrost素材** | PNG 1枚 1〜5MB、WebM 1本 1〜30MB |
| 静止画のモデル(SDXL 1.0) | マイドライブ / **CastFrost_models** | 約7GB |
| 透明にする部品・VAE | 同上 | 約1.3GB |
| 日本語→英語の翻訳モデル | 同上 | 約4GB |
| アニメ調・実写調のモデル(使った時だけ) | 同上 | 1つ約7GB |
| 背景の切り抜き(使った時だけ) | 同上 | 約1GB |
| 動画・画像から動画のモデル | **ドライブにはしまいません**(毎回 Colab にダウンロード・約45〜50GB) | 0GB |
| 加工に使う素材(大きいファイルを渡す時だけ) | マイドライブ / **CastFrost素材_入力** | 入れたファイルの分 |

- ふつうに使うと、ドライブを **約12GB** 使います。2回目からは、ドライブにしまったモデルを読み込むので速くなります
- ドライブの空きが少ない時は、静止画の「⚙ 高度な設定」で **モデルをドライブにしまう** を外してください(毎回ダウンロードになります)。空きが足りない時は自動でしまいません
- いらなくなったら、マイドライブの **CastFrost_models** フォルダを消してかまいません(作った素材は消えません)

### GPU ごとの目安

| GPU | 静止画(1枚) | 動画(5秒・1本) | ユニットの目安 |
|---|---|---|---|
| A100 | 約10秒 | 約2〜3分 | 1時間あたり 約5.4 |
| L4 | 約25秒 | 約6〜12分(試験的) | 1時間あたり 約1.7 |
| T4 | 約60秒 | 作れません | 1時間あたり 約1.2 |

画像から動画を作る:A100 で 5秒 1本 約3〜5分。10〜30秒は5秒ずつ作ってつなぐので、回数ぶん(30秒なら約6倍)かかります(L4 は試験的で、さらに時間がかかります)。素材を加工する:切り抜き・描き直しは GPU が要ります(T4 でも可)。

※ 1回目はモデルのダウンロードに、静止画で約3〜5分、動画で約10〜15分かかります。数字は目安で、Colab の混み具合で変わります

''' + f'''
## 困った時

- **「GPU のメモリが足りません」**:本数や大きさを減らす。それでもだめなら **ランタイム** → **セッションを再起動する** → セルを押し直す
- **動画のセルで「この GPU では作れません」**:**ランタイムのタイプを変更** で **A100** を選ぶ
- **止まった・エラーになった**:セルの下に出た画面のいちばん下の「📄 くわしい記録」を開き(うまくいかなかった時は自動で開きます。セルのいちばん下の出力にも同じ内容が出ます)、赤い行の英語(エラーの名前)を、使い方サイトの「困った時」と見くらべてください
- くわしくは **[{SITE}]({SITE})** の「困った時」へ

使っているモデル・部品は、すべて商用で使えるライセンスのものです(一覧は使い方サイトの「モデルとライセンス」)。
'''),
]

nb = dict(nbformat=4, nbformat_minor=0, cells=cells, metadata={
    'colab': {'provenance': [], 'gpuType': 'A100', 'machine_shape': 'hm', 'name': NB_PATH},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'},
    'accelerator': 'GPU'})

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f'cast-frost-effects-maker_v{VERSION}.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(path, os.path.getsize(path), 'bytes')
