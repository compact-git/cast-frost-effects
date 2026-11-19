# 使い方サイトの本文で使う短い書き方(Cast Frost Web ヘルプと同じ書き方)
#   [[tab:fx]]           → Cast Frost の 🎆 演出 タブ(ヘルプのページへのリンク)
#   [[p:g-image|静止画を作る]] → このサイトのページへのリンク
#   [[help:g-fx-auto|自動の演出]] → Cast Frost ヘルプのページへのリンク
#   [[ui:✨ 作る]]       → ノートブックの画面のボタン・項目の名前
#   [[cell:🖼 静止画を作る]] → ノートブックのセル
#   [[key:Ctrl+Enter]]   → キー
import re

HELP = 'https://rider.tokyo/apps/cast/help/'
TABS = {'fx': ('🎆', '演出'), 'btn': ('⚡', '発動'), 'set': ('⚙️', '設定'), 'sub': ('🖥', '表示')}

def tab(k):
    i, n = TABS[k]
    return f'<a class="tabchip" href="{HELP}r-{k}.html" target="_blank" rel="noopener">{i} {n}</a>'

def expand(s):
    s = re.sub(r'\[\[tab:(\w+)\]\]', lambda m: tab(m.group(1)), s)
    s = re.sub(r'\[\[help:([\w-]+)\|([^\]]+)\]\]', lambda m: f'<a href="{HELP}{m.group(1)}.html" target="_blank" rel="noopener">{m.group(2)} ↗</a>', s)
    s = re.sub(r'\[\[p:([\w-]+)\|([^\]]+)\]\]', r'<a href="\1.html">\2</a>', s)
    s = re.sub(r'\[\[p:([\w-]+)#([\w-]+)\|([^\]]+)\]\]', r'<a href="\1.html#\2">\3</a>', s)
    s = re.sub(r'\[\[ui:([^\]]+)\]\]', r'<b class="ui">\1</b>', s)
    s = re.sub(r'\[\[cell:([^\]]+)\]\]', r'<b class="cell">\1</b>', s)
    s = re.sub(r'\[\[key:([^\]]+)\]\]', r'<kbd>\1</kbd>', s)
    return s

def steps(*items):
    return '<ol class="steps">' + ''.join(f'<li>{x}</li>' for x in items) + '</ol>'

def tip(x): return f'<div class="note tip"><b>💡 ヒント</b><div>{x}</div></div>'
def warn(x): return f'<div class="note warn"><b>⚠ 注意</b><div>{x}</div></div>'
def need(x): return f'<div class="note need"><b>🧰 必要なもの</b><div>{x}</div></div>'

def items(rows):
    """画面の項目の表:[(項目名, 説明), ...]"""
    return '<dl class="items">' + ''.join(f'<dt>{a}</dt><dd>{b}</dd>' for a, b in rows) + '</dl>'

def ex(rows):
    """プロンプトの例:[(日本語, 英語), ...]"""
    return '<div class="ex"><div>こう書く</div><div>英語にすると(だいたい)</div>' + ''.join(f'<div>{a}</div><div><code>{b}</code></div>' for a, b in rows) + '</div>'

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
VERSION = re.search(r"VERSION = '([^']+)'", open(_os.path.join(_HERE, 'notebook', 'cfx_common.py'), encoding='utf-8').read()).group(1)
NB_FILE = f'cast-frost-effects-maker_v{VERSION}.ipynb'
GITHUB = 'https://github.com/compact-git/cast-frost-effects'
COLAB = 'https://colab.research.google.com/github/compact-git/cast-frost-effects/blob/main/cast-frost-effects-maker.ipynb'
DL = f'<a class="bigBtn" href="dl/{NB_FILE}" download>⬇ ノートブックをダウンロード(v{VERSION})</a>'
BADGE = f'<a class="badge" href="{COLAB}" target="_blank" rel="noopener"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab で開く"></a>'
