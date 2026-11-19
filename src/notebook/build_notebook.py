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

MODS = ['cfx_common.py', 'cfx_encode.py', 'cfx_rife.py', 'cfx_wan.py', 'cfx_image.py', 'cfx_video.py']
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

cells = [
    md(f'''
# 🎆 Cast Frost 素材メーカー v{VERSION}

Cast Frost Web の **🎆 演出** と **⚡ 発動** で使う、**背景が透明な素材**を作るノートブックです。made by compact

- 🖼 **静止画**:背景が透明な PNG
- 🎬 **動画**:背景が透明な WebM(VP9)。Cast Frost の「＋ 素材を追加」に入る **30MB 以内**に自動で収めます

**使い方のくわしい説明:[{SITE}]({SITE})**

<a href="{COLAB}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab で開く"></a>
'''),
    md('''
## はじめに(3ステップ)

1. **GPU を選ぶ**:メニューの **ランタイム** → **ランタイムのタイプを変更** → **A100**(動画はこれがおすすめ)。静止画だけなら **L4** や **T4** でも作れます
2. 下の **🖼 静止画を作る** か **🎬 動画を作る** のセルの **▶(再生ボタン)** を押す
3. しばらく待つと、セルの下に入力欄が出ます。プロンプト(日本語でOK)を書いて **✨ 作る** を押す

- 最初の1回は「Google ドライブへのアクセスを許可」の画面が出ます。**許可**を押してください
- 静止画と動画を切り替える時は、もう片方のセルの ▶ を押します。止まった時は **ランタイム** → **セッションを再起動する** を押してから、押し直してください
- **使い終わったら GPU を止めましょう**(いちばん下の 🛑 のセル)。つないだままだと、コンピューティング ユニットが減り続けます

### Google ドライブの使い方(消費量)

| しまうもの | 場所 | 大きさの目安 |
|---|---|---|
| 作った素材(PNG / WebM) | マイドライブ / **CastFrost素材** | PNG 1枚 1〜5MB、WebM 1本 1〜30MB |
| 静止画のモデル(SDXL 1.0) | マイドライブ / **CastFrost_models** | 約7GB |
| 透明にする部品・VAE | 同上 | 約1.3GB |
| 日本語→英語の翻訳モデル | 同上 | 約4GB |
| アニメ調・実写調のモデル(使った時だけ) | 同上 | 1つ約7GB |
| 背景の切り抜き(使った時だけ) | 同上 | 約1GB |
| 動画のモデル | **ドライブにはしまいません**(毎回 Colab にダウンロード・約45GB) | 0GB |

- ふつうに使うと、ドライブを **約12GB** 使います。2回目からは、ドライブにしまったモデルを読み込むので速くなります
- ドライブの空きが少ない時は、静止画の「⚙ 高度な設定」で **モデルをドライブにしまう** を外してください(毎回ダウンロードになります)。空きが足りない時は自動でしまいません
- いらなくなったら、マイドライブの **CastFrost_models** フォルダを消してかまいません(作った素材は消えません)

### GPU ごとの目安

| GPU | 静止画(1枚) | 動画(5秒・1本) | ユニットの目安 |
|---|---|---|---|
| A100 | 約10秒 | 約2〜3分 | 1時間あたり 約5.4 |
| L4 | 約25秒 | 約6〜12分(試験的) | 1時間あたり 約1.7 |
| T4 | 約60秒 | 作れません | 1時間あたり 約1.2 |

※ 1回目はモデルのダウンロードに、静止画で約3〜5分、動画で約10〜15分かかります。数字は目安で、Colab の混み具合で変わります
'''),
    code(f'''
#@title 🖼 静止画を作る(背景が透明な PNG)
#@markdown ▶ を押すと準備が始まり、このセルの下に入力欄が出ます(1回目は 3〜5 分ほど)。
{BOOT}
import cfx_image
cfx_image.start()
''', 'cfx_image'),
    code(f'''
#@title 🎬 動画を作る(背景が透明な WebM)
#@markdown ▶ を押すと準備が始まり、このセルの下に入力欄が出ます(1回目は 10〜15 分ほど)。GPU は **A100** がおすすめです。
{BOOT}
import cfx_video
cfx_video.start()
''', 'cfx_video'),
    code('''
#@title 🛑 GPU を止める(使い終わったら押す)
#@markdown ドライブへのしまい込みが終わるのを待ってから、Colab の接続を切って GPU を止めます。作った素材は、ドライブの「CastFrost素材」に残ります。
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
    md(f'''
## 困った時

- **「GPU のメモリが足りません」**:本数や大きさを減らす。それでもだめなら **ランタイム** → **セッションを再起動する** → セルを押し直す
- **動画のセルで「この GPU では作れません」**:**ランタイムのタイプを変更** で **A100** を選ぶ
- **止まった・エラーになった**:セルの下に出た画面のいちばん下の「📄 くわしい記録(困った時に開く)」を開き(うまくいかなかった時は自動で開きます)、赤い行の英語(エラーの名前)を、使い方サイトの「困った時」と見くらべてください
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
