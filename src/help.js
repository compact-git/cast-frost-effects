/* Cast Frost 素材メーカーの使い方:検索とスマホの目次(Cast Frost Web ヘルプと同じ) */
(() => {
  const side = document.getElementById('side'), mb = document.getElementById('menuBtn');
  if (mb && side) mb.onclick = () => { const o = side.classList.toggle('open'); mb.setAttribute('aria-expanded', o); };
  const q = document.getElementById('q'), box = document.getElementById('results');
  const IDX = window.HELP_INDEX || [];
  if (!q || !box) return;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const norm = s => String(s || '').toLowerCase().normalize('NFKC');
  let sel = -1;
  function run() {
    const raw = q.value.trim(); sel = -1;
    if (!raw) { box.hidden = true; box.innerHTML = ''; return; }
    const words = norm(raw).split(/\s+/).filter(Boolean);
    const hits = [];
    for (const p of IDX) {
      const t = norm(p.t), h = norm(p.h.join(' ')), k = norm(p.k), x = norm(p.x);
      let score = 0, ok = true;
      for (const w of words) {
        const s = (t.includes(w) ? 10 : 0) + (h.includes(w) ? 5 : 0) + (k.includes(w) ? 4 : 0) + (x.includes(w) ? 1 : 0);
        if (!s) { ok = false; break; } score += s;
      }
      if (!ok) continue;
      // 本文の、最初に見つかった所の前後を出す
      const i = x.indexOf(words[0]);
      let snip = i >= 0 ? p.x.slice(Math.max(0, i - 30), i + 70) : '';
      snip = esc(snip);
      for (const w of words) snip = snip.replace(new RegExp(w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), m => `<mark>${m}</mark>`);
      hits.push({ p, score, snip });
    }
    hits.sort((a, b) => b.score - a.score);
    box.innerHTML = hits.length ? hits.slice(0, 12).map(({ p, snip }) => `<a href="${p.u}?q=${encodeURIComponent(raw)}"><b>${esc(p.t)}</b><small>${esc(p.g)}${snip ? ' — …' + snip + '…' : ''}</small></a>`).join('')
      : '<div style="padding:10px;color:var(--sub);font-size:14px">見つかりませんでした。別の言葉で探してみてね。</div>';
    box.hidden = false;
  }
  q.addEventListener('input', run);
  q.addEventListener('focus', run);
  q.addEventListener('keydown', e => {
    const links = [...box.querySelectorAll('a')];
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); if (!links.length) return; sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length; links.forEach((a, i) => a.classList.toggle('sel', i === sel)); }
    else if (e.key === 'Enter') { const a = links[sel >= 0 ? sel : 0]; if (a) location.href = a.href; }
    else if (e.key === 'Escape') { box.hidden = true; q.blur(); }
  });
  document.addEventListener('click', e => { if (!e.target.closest('.search')) box.hidden = true; });
  // 検索から来た時は、言葉の出てくる最初の見出しへ
  const want = new URLSearchParams(location.search).get('q');
  if (want) {
    const w = norm(want.split(/\s+/)[0]);
    const el = [...document.querySelectorAll('.body h2,.body h3,.body dt,.body li,.body p')].find(n => norm(n.textContent).includes(w));
    if (el) { el.scrollIntoView({ block: 'center' }); el.style.transition = 'background .6s'; el.style.background = 'var(--acc-soft)'; setTimeout(() => { el.style.background = ''; }, 2200); }
  }
})();
