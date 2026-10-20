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
      <td><span class="chip ${r.side}">${r.chip}</span>${r.pend ? `<span class="chip pend">${r.pend === true ? 'MON' : r.pend}</span>` : ''}${
        r.upd === 'ok' ? '<span class="chip ok">✓ CONFIRMED</span>' : r.upd === 'rev' ? '<span class="chip rev">↻ REVISED</span>' : ''}</td>
      <td class="n num">${r.n}</td>
      <td class="p"><div class="bar"><i style="width:${w}px;background:${col}"></i><span class="num">${money(r.prem)}</span></div></td></tr>`;
  }).join('');
}

/* Paired bars: rows [{label, a, b, ta, tb}] -- two series per row (colors ca / cb, e.g. index vs
   ETF), signed values grow left/right from a zero axis. A legend names the two series. */
function pairBars(svg, rows, max, opt = {}) {
  const W = +svg.getAttribute('width'), lab = opt.labelW || 150, H = opt.H || 20, gap = opt.gap || 6, rowGap = opt.rowGap || 20;
  const neg = rows.some(r => r.a < 0 || r.b < 0), pos = rows.some(r => r.a > 0 || r.b > 0);
  const x0 = neg && !pos ? W - 120 : neg ? lab + (W - lab - 120) / 2 : lab + 8;
  const span = neg && pos ? (W - lab - 120) / 2 : (neg ? x0 - lab - 70 : W - x0 - 120);
  const k = span / max;
  const y1 = rows.length * (2 * H + gap + rowGap) - rowGap + 4;
  el('line', {x1: x0, x2: x0, y1: 0, y2: y1, stroke: '#3a4554', 'stroke-width': 1.5}, svg);
  rows.forEach((r, i) => {
    const y = (opt.top || 0) + i * (2 * H + gap + rowGap);
    txt(svg, 0, y + H + 8, r.label, {fill: '#eef1f5', 'font-size': 16, 'font-weight': 700});
    [[r.a, r.ta, opt.ca || '#9085e9', 0], [r.b, r.tb, opt.cb || '#199e70', H + gap]].forEach(([v, t, col, dy]) => {
      const w = Math.max(2, Math.abs(v) * k), left = v < 0;
      el('rect', {x: left ? x0 - w : x0 + 1, y: y + dy, width: w, height: H, rx: 4, fill: col}, svg);
      // long bars carry their label inside (dark ink), so it never runs into the row label
      const inside = opt.inside !== false && w > 170;
      txt(svg, inside ? (left ? x0 - w + 10 : x0 + w - 10) : (left ? x0 - w - 8 : x0 + w + 9), y + dy + H - 5, t,
          {'text-anchor': inside ? (left ? 'start' : 'end') : (left ? 'end' : 'start'),
           fill: inside ? '#0b0f14' : '#eef1f5', 'font-size': 14, 'font-weight': 700, class: 'num'});
    });
  });
}

/* Intraday lines: series [{label, color, pts: [[minute_of_day, value], ...]}], one shared y axis
   (same unit), zero line, hourly ticks, direct end labels, optional annotations
   opt.ann [{x: minute, y: value, text, dx, dy, anchor}], '|' in text = line break. Two series max. */
function lineChart(svg, series, opt = {}) {
  const W = +svg.getAttribute('width'), H = +svg.getAttribute('height');
  const L = opt.left || 54, R = W - (opt.right || 140), T = 10, B = H - 26;
  const x0 = opt.x0 || 570, x1 = opt.x1 || 975;
  const vals = series.flatMap(s => s.pts.map(p => p[1])).concat(opt.zero === false ? [] : [0]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const padv = (hi - lo) * 0.08 || 0.1; lo -= padv; hi += padv;
  const X = m => L + (m - x0) / (x1 - x0) * (R - L), Y = v => B - (v - lo) / (hi - lo) * (B - T);
  const step = opt.ystep || 0.25;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    el('line', {x1: L, x2: R, y1: Y(v), y2: Y(v), stroke: Math.abs(v) < 1e-9 ? '#56606e' : '#1b222c', 'stroke-width': Math.abs(v) < 1e-9 ? 1.5 : 1}, svg);
    txt(svg, L - 8, Y(v) + 4, opt.yfmt ? opt.yfmt(v) : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2), {'text-anchor': 'end', fill: '#7d8796', 'font-size': 12, class: 'num'});
  }
  const xt = opt.xticks || (() => { const a = []; for (let m = Math.ceil(x0 / 60) * 60; m <= x1; m += 60) a.push(m); return a; })();
  const xf = opt.xfmt || (m => `${Math.floor(m / 60) > 12 ? Math.floor(m / 60) - 12 : Math.floor(m / 60)}:00`);
  for (const m of xt) txt(svg, X(m), B + 18, xf(m), {'text-anchor': 'middle', fill: '#7d8796', 'font-size': 12});
  if (opt.zero === false) {} // (the zero line is drawn by the y-grid when 0 is in range)
  series.forEach((s, i) => {
    const pts = s.pts.filter(p => p[0] >= x0 && p[0] <= x1);
    el('path', {d: pts.map((p, j) => `${j ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' '),
      fill: 'none', stroke: s.color, 'stroke-width': 2.5, 'stroke-linejoin': 'round'}, svg);
    const e = pts[pts.length - 1];
    el('circle', {cx: X(e[0]), cy: Y(e[1]), r: 4.5, fill: s.color, stroke: '#121820', 'stroke-width': 2}, svg);
    txt(svg, X(e[0]) + 10, Y(e[1]) + 5 + (s.dy || 0), opt.endfmt ? opt.endfmt(s, e[1]) : `${s.label} ${e[1] >= 0 ? '+' : '−'}$${Math.abs(e[1]).toFixed(2)}B`,
        {fill: '#eef1f5', 'font-size': 14, 'font-weight': 700});
  });
  (opt.ann || []).forEach(a => {
    const ax = X(a.x), ay = Y(a.y), tx = ax + (a.dx || 0), ty = ay + (a.dy || 0);
    el('line', {x1: ax, y1: ay, x2: tx, y2: ty, stroke: '#7d8796', 'stroke-width': 1}, svg);
    a.text.split('|').forEach((s, j) => txt(svg, tx + (a.anchor === 'end' ? -4 : 4), ty + 4 + j * 16, s,
      {fill: j ? '#b4bcc8' : '#eef1f5', 'font-size': 13, 'font-weight': j ? 400 : 600, 'text-anchor': a.anchor || 'start'}));
  });
}

/* Simple vertical-free horizontal bars for a comparison: rows [{label, v, text, color}] */
function compareBars(svg, rows, max, opt = {}) {
  const x0 = opt.x0 || 70, W = +svg.getAttribute('width'), k = (W - x0 - 90) / max, H = opt.H || 26, gap = opt.gap || 14;
  rows.forEach((r, i) => {
    const y = 6 + i * (H + gap), w = Math.max(3, Math.abs(r.v) * k);
    txt(svg, 0, y + H / 2 + 6, r.label, {fill: '#eef1f5', 'font-size': 16, 'font-weight': 700});
    el('rect', {x: x0, y, width: w, height: H, rx: 4, fill: r.color || '#3987e5'}, svg);
    txt(svg, x0 + w + 8, y + H / 2 + 6, r.text, {fill: '#eef1f5', 'font-size': 15, 'font-weight': 700, class: 'num'});
  });
}
