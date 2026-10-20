"""make_iv_card.py -- the 10/19 IV & skew card from probe_iv.py output:
    python docs/posts/make_iv_card.py [_iv_probe.json]   (default: the repo root's _iv_probe.json)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "_iv_probe.json")))
q = d["SPY"]["smile"]
smile = {k: q[k] for k in ("2026-10-12", "2026-10-16", "2026-10-19")}
rr = {t: {k: round(v * 100, 1) for k, v in d[t]["rr"].items()} for t in ("SPY", "SPX", "QQQ", "NDX", "IWM", "RUT")}

html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=1080">
<title>IV and Skew</title>
<!-- Share card (README.md). Numbers: probe_iv.py --expiry 2026-10-30 --start 2026-09-28 --end 2026-10-19 (UW
     term-structure, historical-risk-reversal-skew, per-contract IV; realized vol from daily closes). -->
<link rel="stylesheet" href="card.css">
<style>
  .riv{display:grid;grid-template-columns:120px repeat(5,1fr);font-size:15px}
  .riv div{padding:7px 8px;border-top:1px solid var(--grid)}
  .riv .hd{border-top:none;color:var(--muted);font-size:13px}
  .riv .t{color:var(--ink);font-weight:800}
  .riv .v{color:var(--ink);font-weight:700}
</style></head><body>
<div class="wrap">
  <div class="kicker">10/30 expiry · implied volatility &amp; skew · Sep 28 – Oct 19, 2026</div>
  <h1>Puts got pricey. Broad fear didn't.</h1>
  <div class="sub">Put skew roughly doubled in all three ETFs and their indexes, but overall volatility only rose in tech.
    Protection costs more than recent moves, yet IV ranks sit low to middling: targeted hedging, not panic.</div>
  <div class="tiles">
    <div class="tile"><div class="v num">~2×</div><div class="l">25Δ put skew, 9/28 → 10/16, all six</div></div>
    <div class="tile"><div class="v num">+6–10 pts</div><div class="l">IV of puts 8% below spot, 10/12 → 10/16</div></div>
    <div class="tile"><div class="v num">+4.7 pts</div><div class="l">QQQ 10/30 ATM IV (19.5% → 24.2%); SPY only +1.2</div></div>
    <div class="tile"><div class="v num">21 · 47 · 22</div><div class="l">IV rank (1y): SPY · QQQ · IWM</div></div>
  </div>
  <div class="card">
    <h2>SPY's 10/30 smile: only the puts got bid</h2>
    <div class="h2s">Implied volatility by strike, % from that day's price · out-of-the-money puts left, calls right</div>
    <svg id="sm" width="946" height="190" role="img" aria-label="SPY 10/30 volatility smile on Oct 12, 16 and 19"></svg>
    <div class="legend"><span><span class="sw" style="background:#7d8796"></span>Oct 12 (before)</span>
      <span><span class="sw" style="background:var(--s1)"></span>Oct 16 (the selloff)</span>
      <span><span class="sw" style="background:var(--s2)"></span>Oct 19 (Monday's bounce)</span></div>
  </div>
  <div class="card">
    <h2>25-delta skew: puts priced over calls, vol points</h2>
    <div class="h2s">10/30 expiry · Sep 28 vs the Oct 16 peak (Monday in brackets) · index skew runs steeper; the ETFs rose more</div>
    <svg id="sk" width="946" height="222" role="img" aria-label="25-delta skew Sep 28 vs Oct 16 by ticker"></svg>
    <div class="legend"><span><span class="sw" style="background:#7d8796"></span>Sep 28</span>
      <span><span class="sw" style="background:var(--s1)"></span>Oct 16 peak</span></div>
  </div>
  <div class="card">
    <h2>RIV: how rich is it?</h2>
    <div class="riv">
      <div class="hd"></div><div class="hd">30-day IV</div><div class="hd">21-day realized</div><div class="hd">fear premium</div><div class="hd">IV rank (1y)</div><div class="hd">index skew − ETF</div>
      <div class="t">SPY</div><div class="v num">14.4%</div><div class="v num">10.3%</div><div class="v num">+4.1</div><div class="v num">21</div><div class="v num">SPX +1.1</div>
      <div class="t">QQQ</div><div class="v num">21.8%</div><div class="v num">18.4%</div><div class="v num">+3.4</div><div class="v num">47</div><div class="v num">NDX +1.2</div>
      <div class="t">IWM</div><div class="v num">19.9%</div><div class="v num">14.5%</div><div class="v num">+5.4</div><div class="v num">22</div><div class="v num">RUT +0.9</div>
    </div>
  </div>
  <div class="read"><b>Read:</b> hedgers paid for the tails, not the middle; only in tech did volatility itself rise (QQQ ATM +8 pts in the 10/12 → 10/16 week).
    Monday's bounce barely cheapened protection (skew −0.6 to −0.9): hedges rolled, not dropped.
    <span class="mon">Index level from the ETF · realized vol from daily closes · RUT's UW IV rank unreliable, left out.</span></div>
</div>
<div class="foot">
  <span>Data: Unusual Whales API · method: <b>probe_iv.py</b> · not financial advice</span>
  <span><span class="repo">github.com/levijon2623/WeBullRG</span> · analysis with Claude (Anthropic)</span>
</div>
<script src="card.js"></script>
<script>
const SM = __SMILE__, RR = __RR__;
lineChart(document.getElementById('sm'), [
  {label: 'Oct 12', color: '#7d8796', pts: SM['2026-10-12'], dy: 16},
  {label: 'Oct 16', color: '#9085e9', pts: SM['2026-10-16'], dy: -14},
  {label: 'Oct 19', color: '#199e70', pts: SM['2026-10-19'], dy: 2}],
  {x0: -9, x1: 6, xticks: [-8, -6, -4, -2, 0, 2, 4, 6], xfmt: m => (m > 0 ? '+' : '') + m + '%',
   yfmt: v => v.toFixed(0) + '%', ystep: 5, zero: false, right: 120,
   endfmt: () => '',   // three close endings: the legend names them
   ann: [{x: -8, y: 27.2, text: '+6.6 pts at −8%,|only +1.2 at +4%', dx: 150, dy: 58}]});
const rows = ['SPY', 'SPX', 'QQQ', 'NDX', 'IWM', 'RUT'].map(t => ({label: t,
  a: RR[t]['2026-09-28'], b: RR[t]['2026-10-16'],
  ta: '+' + RR[t]['2026-09-28'].toFixed(1), tb: '+' + RR[t]['2026-10-16'].toFixed(1) + '  (Mon +' + RR[t]['2026-10-19'].toFixed(1) + ')'}));
pairBars(document.getElementById('sk'), rows, 7.5, {ca: '#7d8796', cb: '#9085e9', labelW: 70, H: 13, gap: 2, rowGap: 7, inside: false, top: 6});
</script>
</body></html>
"""
html = html.replace("__SMILE__", json.dumps(smile)).replace("__RR__", json.dumps(rr))
open(os.path.join(HERE, "2026-10-19-iv-skew.html"), "w", encoding="utf-8").write(html)
print("ok")
