# Cast Frost 素材メーカーの使い方サイトを作る(python3 build.py → ../effects/)
# ノートブック(notebook/ の cfx_*.py)も組み立てて、dl/ に入れる
# 中身は pages_*.py に書く。ここは共通の枠(ヘッダー・目次・検索・前後のリンク)
import os, re, json, html, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'effects')
import re as _re
APP_VER = _re.search(r"VERSION = '([^']+)'", open(os.path.join(HERE, 'notebook', 'cfx_common.py'), encoding='utf-8').read()).group(1)
APP_URL = 'https://cast.rider.tokyo/'
HELP_URL = 'https://rider.tokyo/apps/cast/help/'
NB_FILE = f'cast-frost-effects-maker_v{APP_VER}.ipynb'

def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

common = load('common')
PAGES = []
for mod in ('pages_start', 'pages_guide', 'pages_ref', 'pages_misc'):
    PAGES += load(mod).PAGES

GROUPS = [('top', 'はじめに'), ('guide', '使い方'), ('ref', '画面の項目'), ('misc', '資料')]
BY_ID = {p['id']: p for p in PAGES}

def link(pid, text=None):
    p = BY_ID[pid]; return f'<a href="{p["id"]}.html">{text or p["nav"]}</a>'

def nav_html(cur):
    out = []
    for g, label in GROUPS:
        items = [p for p in PAGES if p['group'] == g]
        out.append(f'<div class="navGroup"><div class="navLabel">{label}</div><ul>')
        for p in items:
            c = ' class="cur" aria-current="page"' if p['id'] == cur else ''
            out.append(f'<li><a href="{p["id"]}.html"{c}>{p["nav"]}</a></li>')
        out.append('</ul></div>')
    return '\n'.join(out)

def toc_html(body):
    hs = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    if len(hs) < 3: return ''
    li = ''.join(f'<li><a href="#{i}">{re.sub(r"<[^>]+>", "", t)}</a></li>' for i, t in hs)
    return f'<nav class="toc" aria-label="このページの目次"><div class="tocLabel">このページの目次</div><ol>{li}</ol></nav>'

def add_ids(body):
    n = [0]
    def rep(m):
        if 'id=' in m.group(1): return m.group(0)
        n[0] += 1; return f'<h2 id="s{n[0]}"{m.group(1)}>'
    return re.sub(r'<h2([^>]*)>', rep, body)

def page_html(p, prev, nxt):
    body = add_ids(common.expand(p['body']))
    rel = ''
    if p.get('related'):
        rel = '<aside class="related"><h2 id="related">関連するページ</h2><ul>' + ''.join(f'<li>{link(r)}<span class="relWhy">{BY_ID[r].get("lead","")}</span></li>' for r in p['related']) + '</ul></aside>'
    grp = dict(GROUPS)[p['group']]
    crumbs = f'<a href="index.html">素材メーカー</a> › {grp}' + (f' › {p["nav"]}' if p['id'] != 'index' else '')
    pn = '<nav class="pn">' + (f'<a class="prev" href="{prev["id"]}.html"><small>前へ</small>{prev["nav"]}</a>' if prev else '<span></span>') + (f'<a class="next" href="{nxt["id"]}.html"><small>次へ</small>{nxt["nav"]}</a>' if nxt else '<span></span>') + '</nav>'
    lead = f'<p class="lead">{p["lead"]}</p>' if p.get('lead') else ''
    return f'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(p["title"])} | Cast Frost 素材メーカー</title>
<meta name="description" content="{html.escape(re.sub(r"<[^>]+>", "", p.get("lead", "Cast Frost 素材メーカーの使い方")))}">
<link rel="icon" href="icon.svg">
<link rel="stylesheet" href="help.css?v={APP_VER}">
</head>
<body>
<a class="skip" href="#main">本文へ</a>
<header class="top">
  <button class="menuBtn" id="menuBtn" aria-label="目次を開く" aria-expanded="false">☰</button>
  <a class="brand" href="index.html"><img src="icon.svg" alt="" width="28" height="28"><span>Cast Frost <b>素材メーカー</b></span></a>
  <div class="search"><input id="q" type="search" placeholder="🔍 使い方を検索(例:30MB、ループ、A100)" autocomplete="off" aria-label="使い方を検索"><div class="results" id="results" hidden></div></div>
  <a class="appLink" href="dl/{NB_FILE}" download>⬇ ノートブック</a><a class="appLink" href="{HELP_URL}" target="_blank" rel="noopener">Cast Frost ヘルプ ↗</a>
</header>
<div class="wrap">
  <nav class="side" id="side" aria-label="使い方の目次">
{nav_html(p["id"])}
    <p class="sideVer">ノートブック v{APP_VER} に対応</p>
  </nav>
  <main id="main">
    <div class="crumbs">{crumbs}</div>
    <h1>{p["title"]}</h1>
    {lead}
    {toc_html(body)}
    <article class="body">
{body}
    </article>
    {rel}
    {pn}
    <footer class="foot">Cast Frost 素材メーカー(ノートブック v{APP_VER} に対応)/ made by compact</footer>
  </main>
</div>
<script src="search-index.js?v={APP_VER}"></script>
<script src="help.js?v={APP_VER}"></script>
</body>
</html>
'''

def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    seq = PAGES
    idx = []
    for i, p in enumerate(seq):
        prev = seq[i - 1] if i > 0 else None
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        h = page_html(p, prev, nxt)
        open(os.path.join(OUT, p['id'] + '.html'), 'w', encoding='utf-8').write(h)
        body = common.expand(p['body'])
        heads = [re.sub(r'<[^>]+>', '', t) for t in re.findall(r'<h[23][^>]*>(.*?)</h[23]>', body)]
        text = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', body)))
        idx.append({'u': p['id'] + '.html', 't': p['title'], 'g': dict(GROUPS)[p['group']], 'h': heads, 'k': p.get('kw', ''), 'x': text[:6000]})
    open(os.path.join(OUT, 'search-index.js'), 'w', encoding='utf-8').write('window.HELP_INDEX=' + json.dumps(idx, ensure_ascii=False) + ';\n')
    for f in ('help.css', 'help.js'):
        shutil.copy(os.path.join(HERE, f), os.path.join(OUT, f))
    shutil.copy(os.path.join(HERE, 'icon.svg'), os.path.join(OUT, 'icon.svg'))
    # ノートブックを組み立てて dl/ に入れる
    import subprocess, sys as _sys
    dl = os.path.join(OUT, 'dl')
    subprocess.run([_sys.executable, os.path.join(HERE, 'notebook', 'build_notebook.py'), dl], check=True)
    if not os.path.exists(os.path.join(dl, NB_FILE)): raise SystemExit('ノートブックができていません')
    # ページ内のリンク切れを調べる
    ids = {p['id'] for p in PAGES}
    bad = []
    for p in PAGES:
        for href in re.findall(r'href="([^"#:]+)\.html', common.expand(p['body'])):
            if href not in ids: bad.append((p['id'], href))
        for r in p.get('related', []):
            if r not in ids: bad.append((p['id'], 'related:' + r))
    print(len(PAGES), 'pages', 'broken:', bad)

if __name__ == '__main__':
    main()
