/* Shared chart pieces for share cards (docs/posts/README.md). No libraries. */
const NS = 'http://www.w3.org/2000/svg';
function el(tag, a, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in a) e.setAttribute(k, a[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, a = {}) {
  const t = el('text', Object.assign({x, y, fill: '#b4bcc8', 'font-size': 14}, a), parent);
  t.textContent = s;
  return t;
}
const money = v => v >= 1000 ? `$${(v / 1000).toFixed(2)}B` : `$${Math.round(v)}M`;

/* Diverging bars: rows [{label, sold, bought}] in $M; sold grows left, bought right.
   `max` is shared across cards so the ETF cards compare honestly. */
function divergingBars(svg, rows, max, opt = {}) {
  const W = +svg.getAttribute('width'), x0 = opt.x0 || 470, half = Math.min(x0 - 130, W - x0 - 120);
  const k = half / max, H = 34, gap = 22, top = 10;
  el('line', {x1: x0, x2: x0, y1: 2, y2: top + rows.length * (H + gap) - gap + 8, stroke: '#3a4554', 'stroke-width': 1.5}, svg);
  rows.forEach((r, i) => {
    const y = top + i * (H + gap);
    txt(svg, 4, y + H / 2 + 7, r.label, {fill: '#eef1f5', 'font-size': 19, 'font-weight': 700});
    const ws = r.sold * k, wb = r.bought * k;
    if (ws > 0) el('rect', {x: x0 - ws - 1, y, width: ws, height: H, rx: 4, fill: '#e66767'}, svg);
    if (wb > 0) el('rect', {x: x0 + 1, y, width: wb, height: H, rx: 4, fill: '#3987e5'}, svg);
    txt(svg, x0 - ws - 10, y + H / 2 + 6, `${money(r.sold)} sold`, {'text-anchor': 'end', fill: '#eef1f5', 'font-size': 15, 'font-weight': 600});
    txt(svg, x0 + wb + 10, y + H / 2 + 6, `${money(r.bought)} bought`, {fill: '#eef1f5', 'font-size': 15, 'font-weight': 600});
  });
  const yb = top + rows.length * (H + gap) - gap + 24;
  txt(svg, x0 - 8, yb, '← sold to open', {'text-anchor': 'end', fill: '#7d8796', 'font-size': 13});
  txt(svg, x0 + 8, yb, 'bought to open →', {fill: '#7d8796', 'font-size': 13});
}

/* Trade list: rows [{d, k, side:'b'|'s'|'x', chip, n, prem ($M), note, pend}] -> <table>. */
function tradeTable(tbl, rows) {
  const max = Math.max(...rows.map(r => r.prem));
  tbl.innerHTML = rows.map(r => {
    const w = Math.max(4, Math.round(118 * r.prem / max));
    const col = r.side === 's' ? 'var(--sold)' : r.side === 'b' ? 'var(--bought)' : '#7d8796';
    return `<tr><td class="d">${r.d}</td>
      <td><span class="k">${r.k}</span><br><span class="note">${r.note || ''}</span></td>
      <td><span class="chip ${r.side}">${r.chip}</span>${r.pend ? '<span class="chip pend">MON</span>' : ''}</td>
      <td class="n num">${r.n}</td>
      <td class="p"><div class="bar"><i style="width:${w}px;background:${col}"></i><span class="num">${money(r.prem)}</span></div></td></tr>`;
  }).join('');
}

/* Simple vertical-free horizontal bars for a comparison: rows [{label, v, text, color}] */
function compareBars(svg, rows, max, opt = {}) {
  const x0 = opt.x0 || 70, W = +svg.getAttribute('width'), k = (W - x0 - 90) / max, H = 26, gap = 14;
  rows.forEach((r, i) => {
    const y = 6 + i * (H + gap), w = Math.max(3, Math.abs(r.v) * k);
    txt(svg, 0, y + H / 2 + 6, r.label, {fill: '#eef1f5', 'font-size': 16, 'font-weight': 700});
    el('rect', {x: x0, y, width: w, height: H, rx: 4, fill: r.color || '#3987e5'}, svg);
    txt(svg, x0 + w + 8, y + H / 2 + 6, r.text, {fill: '#eef1f5', 'font-size': 15, 'font-weight': 700, class: 'num'});
  });
}
