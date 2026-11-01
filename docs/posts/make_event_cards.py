"""make_event_cards.py -- event-week positioning cards from probe_event_window.py output (user, 2026-11-01):
one summary card (S&P, Nasdaq, Russell) + one detail card per family.

    python probe_event_window.py --asof 2026-10-30 --out docs/posts/data/event-week-2026-11-02.json
    python docs/posts/make_event_cards.py docs/posts/data/event-week-2026-11-02.json
    python docs/posts/render_post.py docs/posts/<card>.html          (x4)

Numbers come from the JSON; the WORDS block (headlines, reads) is per week -- rewrite it for a new window.
"""
import datetime as dt
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1], encoding="utf-8"))
ASOF = dt.date.fromisoformat(D["asof"])
EXPS = D["expiries"]
F = D["families"]
NAME = {"SP": ("S&P 500", "SPY + SPX"), "QQQ": ("Nasdaq-100", "QQQ + NDX"), "IWM": ("Russell 2000", "IWM + RUT")}
SLUG = {"SP": "sp500", "QQQ": "nasdaq", "IWM": "russell"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day = lambda iso: dt.date.fromisoformat(iso)
lab = lambda iso: f"{DOW[day(iso).weekday()]} {day(iso).month}/{day(iso).day}"
fmtk = lambda v: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"


def stats(k):
    f = F[k]
    win = [x for x in f["days"] if x["in_window"]]
    typ = f["rv21"] / math.sqrt(252) if f.get("rv21") else None
    last = win[-1]
    week = last["atm_iv"] / 100 * math.sqrt((day(last["e"]) - ASOF).days / 365) * 100 if last["atm_iv"] else None
    c, p = sum(x["calls"] for x in win), sum(x["puts"] for x in win)
    ixp = sum(x["puts"] * x["idx_p"] for x in win) / max(p, 1)
    ixc = sum(x["calls"] * x["idx_c"] for x in win) / max(c, 1)
    flip = next((x["flip"] for x in win if x["flip"]), None)
    b = f["oi_build"]
    d0 = {t: sorted(v)[0] for t, v in b.items()}
    d1 = {t: sorted(v)[-1] for t, v in b.items()}
    add_c = sum(b[t][d1[t]][0] - b[t][d0[t]][0] for t in b)
    add_p = sum(b[t][d1[t]][1] - b[t][d0[t]][1] for t in b)
    peak = max(win, key=lambda x: x["move"] or 0)
    return dict(win=win, typ=typ, week=week, calls=c, puts=p, pc=p / max(c, 1), ixp=ixp, ixc=ixc, flip=flip,
                flip_gap=(flip / f["spot"] - 1) * 100 if flip else None, add_c=add_c, add_p=add_p, peak=peak,
                df=f["delta_flow_B"])


ST = {k: stats(k) for k in F}
EV = {}
for e in D["events"]:
    if e.get("tier") == 1:
        EV.setdefault(e["d"], []).append(e)


def ev_label(e):
    if e["kind"] == "election":
        return "Election (results after the close)"
    return {"NFP": "Payrolls 8:30", "ISM-M": "ISM mfg 10:00", "ISM-S": "ISM services 10:00", "REFUND": "Treasury refunding 8:30",
            "CPI": "CPI 8:30", "PPI": "PPI 8:30", "FOMC": "Fed decision 2:00"}.get(e["short"], e["name"])


DAYS_EV = {e: [ev_label(x) for x in EV.get(e, [])] for e in EXPS}

# ------------------------------------------------------------------ WORDS (this window)
EVENT_DAY = "2026-11-04"           # the session the window is about (first one after the election), highlighted
for _s in ST.values():
    _s["peak"] = next(x for x in _s["win"] if x["e"] == EVENT_DAY)
S, Q, R = ST["SP"], ST["QQQ"], ST["IWM"]
WORDS = {
    "summary": dict(
        kicker=f"Options positioning · event week {lab(EXPS[0])} – {lab(EXPS[-1])} · as of the {lab(D['asof'])} close",
        h1="Hedged into election week: every day is priced like an event day",
        sub=f"Midterms Tuesday night, payrolls Friday, ISM and Treasury refunding in between. After Friday's drop (QQQ −3.4%, "
            f"SPY −1.6%), all three sit below their zero-gamma levels, and S&P and Nasdaq books lean hard to puts.",
        read=("<b>What would change the read:</b> a quiet election lets hedgers sell Wednesday's puts and implied volatility "
              "drop, a setup that often helps a bounce. A surprise result hits markets where dealer hedging adds to moves. "
              "<span class='mon'>Positioning is context, not a call: our own tests found it sizes moves but doesn't predict "
              "direction. Monday's open interest adds Friday's trading.</span>")),
    "SP": dict(
        h1=f"S&P: put walls at 747–749 into Wednesday, and SPX holds ~{S['ixp'] * 100:.0f}% of them",
        sub=f"SPY closed {lab(D['asof'])} at {F['SP']['spot']:.2f}, about {S['flip_gap']:.1f}% under the ~{S['flip']:.0f} zero-gamma "
            f"level. Near the money, puts outnumber calls {S['pc']:.1f}× across the week's expiries.",
        read=(f"<b>Read:</b> institutions hedged in SPX (it sold ${abs(S['df']['SPX']['total']):.1f}B of delta into these expiries "
              f"over five sessions) while SPY traders bought ${S['df']['SPY']['total']:.1f}B: the usual split. "
              "The 747–749 puts expire Tue/Wed; if the election passes quietly, that protection decays fast. "
              "<span class='mon'>Strikes in SPY units (SPX ÷ ~10.03). Side is read from ask/bid; open interest is start of day.</span>")),
    "QQQ": dict(
        h1=f"Nasdaq: puts outnumber calls {Q['pc']:.0f}× near the money after Friday's −3.4%",
        sub=f"QQQ closed at {F['QQQ']['spot']:.2f}, about {Q['flip_gap']:.1f}% under the ~{Q['flip']:.0f} zero-gamma level: "
            f"the deepest of the three. The options price Wednesday at ±{Q['peak']['move']:.1f}%, the biggest day of the week.",
        read=(f"<b>Read:</b> the 713 puts went on with QQQ near 715 and are now in the money; 675–680 is the next layer. "
              f"QQQ (+${Q['df']['QQQ']['total']:.1f}B) and NDX (+${Q['df']['NDX']['total']:.1f}B) both bought a little delta: "
              "some dip buying, no panic. "
              "<span class='mon'>Calls drifted out of the ±4% band as QQQ fell, which lifts the ratio; NDX in QQQ units (÷ ~41).</span>")),
    "IWM": dict(
        h1="Russell: the one call-heavy book, sitting just under its zero-gamma level",
        sub=f"IWM closed at {F['IWM']['spot']:.2f}, {R['flip_gap']:.1f}% under the ~{R['flip']:.1f} flip. Near the money, calls "
            f"outnumber puts ({R['pc']:.2f} puts per call), led by RUT's 11/6 2840–2850 calls.",
        read=(f"<b>Read:</b> Wednesday is priced at ±{R['peak']['move']:.1f}%, {R['peak']['move'] / R['typ']:.1f}× a typical day "
              f"lately (±{R['typ']:.2f}%). Delta flow was tiny in both IWM and RUT: this book was "
              "built earlier, not chased. Open interest can't say whether those RUT calls were bought or written. "
              "<span class='mon'>RUT strikes in IWM units (÷ ~10.04); RUT weeklies trade as RUTW.</span>")),
}

HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=1080">
<title>__TITLE__</title>
<!-- Share card (README.md). Numbers: probe_event_window.py --asof __ASOF__ (UW option-contracts, greek-flow,
     oi-per-expiry; positioning_map.py zero-gamma; events: Fed / BLS calendars + UW economic calendar);
     cards: make_event_cards.py. -->
<link rel="stylesheet" href="card.css">
<style>
  .tbl{width:100%;border-collapse:collapse;font-size:14.5px}
  .tbl th{color:var(--muted);font-weight:600;font-size:12.5px;text-align:right;padding:4px 7px;border-bottom:1px solid var(--line)}
  .tbl td{padding:6px 7px;text-align:right;border-bottom:1px solid var(--grid);color:var(--ink2);white-space:nowrap}
  .tbl th:first-child,.tbl td:first-child{text-align:left}
  .tbl td:first-child{color:var(--ink);font-weight:700}
  .tbl .ev{text-align:left;color:var(--accent);font-size:13px;white-space:normal}
  .tbl th.ev{color:var(--muted)}
</style></head><body>
<div class="wrap">
"""
FOOT = """</div>
<div class="foot">
  <span>Data: Unusual Whales API · method: <b>probe_event_window.py</b> · not financial advice</span>
  <span><span class="repo">github.com/levijon2623/WeBullRG</span> · analysis with Claude (Anthropic)</span>
</div>
<script src="card.js"></script>
"""


def tiles(items):
    return '  <div class="tiles">\n' + "\n".join(
        f'    <div class="tile"><div class="v num">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "\n  </div>\n"


def write(name, html):
    open(os.path.join(HERE, name), "w", encoding="utf-8").write(html)
    print(name)


stem = f"{EXPS[0]}-event-week"

# ------------------------------------------------------------------ summary card
w = WORDS["summary"]
rows = ""
for k in ("SP", "QQQ", "IWM"):
    s, f = ST[k], F[k]
    i0, i1 = f["etf"], f["idx"]
    rows += (f"<tr><td>{NAME[k][0]}</td><td class='num'>{f['spot']:.2f} vs ~{s['flip']:.0f} ({-s['flip_gap']:+.1f}%)</td>"
             f"<td class='num'>{s['pc']:.1f}×</td><td class='num'>{s['ixp'] * 100:.0f}%</td>"
             f"<td class='num'>{f['delta_flow_B'][i0]['total']:+.1f} / {f['delta_flow_B'][i1]['total']:+.1f}</td>"
             f"<td class='num'>+{fmtk(s['add_p'])} / +{fmtk(s['add_c'])}</td></tr>")
moves = {k: [[x["e"], x["move"]] for x in ST[k]["win"]] for k in F}
typ = {k: ST[k]["typ"] for k in F}
html = HEAD.replace("__TITLE__", "Event Week Positioning").replace("__ASOF__", D["asof"]) + f"""  <div class="kicker">{w['kicker']}</div>
  <h1>{w['h1']}</h1>
  <div class="sub">{w['sub']}</div>
""" + tiles([
    (f"±{Q['peak']['move']:.1f}%", f"QQQ's {lab(EVENT_DAY)} priced move (SPY ±{S['peak']['move']:.1f}%, IWM ±{R['peak']['move']:.1f}%)"),
    (f"{S['peak']['move'] / S['typ']:.1f}×", f"SPY's Wednesday vs its typical day lately (±{S['typ']:.2f}%)"),
    (f"{S['pc']:.1f}× · {Q['pc']:.1f}×", "puts per call near the money: S&P · Nasdaq"),
    ("3 of 3", "below their zero-gamma level: dealer hedging adds to moves"),
]) + f"""  <div class="card">
    <h2>What each day is priced to move</h2>
    <div class="h2s">1-σ close-to-close move from each daily expiry's implied volatility · tick = typical day lately (21-day realized)</div>
    <svg id="mv" width="946" height="372" role="img" aria-label="Priced daily moves for SPY, QQQ and IWM, Monday to Friday, with the week's events"></svg>
  </div>
  <div class="card">
    <h2>How each book is positioned</h2>
    <div class="h2s">Near the money = within ±{D['band'] * 100:.0f}% of the close, expiries {lab(EXPS[0])} – {lab(EXPS[-1])}, in ETF-contract units</div>
    <table class="tbl"><tr><th>family</th><th>close vs zero-gamma</th><th>puts per call</th><th>index share of puts</th>
      <th>delta bought, ETF / index ($B, 5 days)</th><th>added this week: puts / calls</th></tr>{rows}</table>
  </div>
  <div class="read">{w['read']}</div>
""" + FOOT + f"""<script>
const MV = {json.dumps(moves)}, TYP = {json.dumps(typ)}, EV = {json.dumps(DAYS_EV)}, EXPS = {json.dumps(EXPS)}, EVD = '{EVENT_DAY}';
(() => {{
  const svg = document.getElementById('mv'), W = 946, L = 70, top = 64, rowH = 100, cw = (W - L) / EXPS.length;
  const mx = Math.max(...Object.values(MV).flat().map(p => p[1])) * 1.08, k = (cw - 34) / mx;
  const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  EXPS.forEach((e, j) => {{
    const x = L + j * cw, d = new Date(e + 'T12:00:00');
    txt(svg, x + 8, 16, `${{DOW[d.getDay()]}} ${{d.getMonth() + 1}}/${{d.getDate()}}`, {{fill: '#eef1f5', 'font-size': 15, 'font-weight': 700}});
    (EV[e] || []).slice(0, 2).forEach((s, i) => txt(svg, x + 8, 34 + i * 15, s, {{fill: '#f0b35a', 'font-size': 12.5}}));
    if (j) el('line', {{x1: x, x2: x, y1: 4, y2: top + 3 * rowH - 6, stroke: '#232c38'}}, svg);
  }});
  [['SP', 'SPY'], ['QQQ', 'QQQ'], ['IWM', 'IWM']].forEach(([key, name], i) => {{
    const y = top + i * rowH;
    txt(svg, 0, y + rowH / 2, name, {{fill: '#eef1f5', 'font-size': 17, 'font-weight': 800}});
    MV[key].forEach(([e, v], j) => {{
      const x = L + j * cw + 8, w = v * k, peak = e === EVD;
      el('rect', {{x, y: y + 26, width: w, height: 26, rx: 4, fill: '#9085e9', 'fill-opacity': peak ? 1 : 0.5}}, svg);
      txt(svg, x, y + 74, `±${{v.toFixed(2)}}%`, {{fill: peak ? '#eef1f5' : '#b4bcc8', 'font-size': 14, 'font-weight': peak ? 800 : 600, class: 'num'}});
      const tx = x + TYP[key] * k;
      el('line', {{x1: tx, x2: tx, y1: y + 20, y2: y + 58, stroke: '#eef1f5', 'stroke-width': 2}}, svg);
    }});
    txt(svg, W - 4, y + 92, `typical day ±${{TYP[key].toFixed(2)}}%`, {{'text-anchor': 'end', fill: '#7d8796', 'font-size': 12.5}});
  }});
}})();
</script>
</body></html>
"""
write(f"{stem}-summary.html", html)

# ------------------------------------------------------------------ detail cards
for k in ("SP", "QQQ", "IWM"):
    f, s, w = F[k], ST[k], WORDS[k]
    spot = f["spot"]
    lo, hi = spot * (1 - D["band"]), spot * (1 + D["band"])
    binw = 2 if spot > 400 else 1
    lad = {}
    for ks, (c, p) in f["strikes"].items():
        kk = int(ks)
        if lo <= kk <= hi:
            b = math.floor(kk / binw) * binw
            lad.setdefault(b, [0, 0])
            lad[b][0] += c
            lad[b][1] += p
    lad = sorted(([b, v[0], v[1]] for b, v in lad.items()), key=lambda r: -r[0])
    trs = ""
    for x in s["win"]:
        ev = ", ".join(DAYS_EV.get(x["e"], [])) or "—"
        tp = x["top_p"][0][0] if x["top_p"] else "—"
        tc = x["top_c"][0][0] if x["top_c"] else "—"
        hot = " style='color:var(--ink);font-weight:800'" if x is s["peak"] else ""
        trs += (f"<tr><td>{lab(x['e'])}</td><td class='ev'>{ev}</td><td class='num'{hot}>±{x['move']:.2f}%</td>"
                f"<td class='num'>{x['atm_iv']:.1f}%</td><td class='num'>{fmtk(x['puts'])} / {fmtk(x['calls'])}</td>"
                f"<td class='num'>{x['pc']:.2f}</td><td class='num'>{tp}</td><td class='num'>{tc}</td></tr>")
    i0, i1 = f["etf"], f["idx"]
    html = HEAD.replace("__TITLE__", f"{NAME[k][0]} Event Week").replace("__ASOF__", D["asof"]) + f"""  <div class="kicker">{NAME[k][0]} · {NAME[k][1]} · event week {lab(EXPS[0])} – {lab(EXPS[-1])} · as of the {lab(D['asof'])} close</div>
  <h1>{w['h1']}</h1>
  <div class="sub">{w['sub']}</div>
""" + tiles([
        (f"{spot:.2f}", f"{i0} close · zero-gamma ~{s['flip']:.0f} ({s['flip_gap']:+.1f}% away)"),
        (f"±{s['peak']['move']:.1f}%", f"{lab(s['peak']['e'])} priced move, vs a typical ±{s['typ']:.2f}% lately"),
        (f"±{s['week']:.1f}%", f"priced 1-σ move through {lab(EXPS[-1])}"),
        (f"{f['delta_flow_B'][i0]['total']:+.1f} / {f['delta_flow_B'][i1]['total']:+.1f}", f"$B delta bought into these expiries, 5 days: {i0} / {i1}"),
    ]) + f"""  <div class="card">
    <h2>Where the open interest sits</h2>
    <div class="h2s">Expiries {lab(EXPS[0])} – {lab(EXPS[-1])} combined · {i0} + {i1} in {i0}-contract units · strikes ±{D['band'] * 100:.0f}% of the close, ${binw} bins</div>
    <svg id="ld" width="946" height="410" role="img" aria-label="Call and put open interest by strike for the week's expiries"></svg>
    <div class="legend"><span><span class="sw" style="background:var(--s1)"></span>calls</span>
      <span><span class="sw" style="background:var(--s2)"></span>puts</span>
      <span><span class="sw" style="background:#eef1f5;height:3px"></span>{lab(D['asof'])} close</span>
      <span><span class="sw" style="background:var(--accent);height:3px"></span>zero-gamma level (dealers long calls, short puts)</span></div>
  </div>
  <div class="card">
    <h2>Day by day</h2>
    <table class="tbl"><tr><th>expiry</th><th class="ev">events that day</th><th>priced move</th><th>ATM IV</th>
      <th>puts / calls near</th><th>puts per call</th><th>top put</th><th>top call</th></tr>{trs}</table>
  </div>
  <div class="read">{w['read']}</div>
""" + FOOT + f"""<script>
const LD = {json.dumps(lad)}, SPOT = {spot}, FLIP = {json.dumps(s['flip'])}, BIN = {binw};
(() => {{
  const svg = document.getElementById('ld'), W = 946, H = 410, mid = W / 2, gap = 40, T = 26, B = H - 4;
  const hiK = LD[0][0] + BIN, loK = LD[LD.length - 1][0];
  const Y = v => T + (hiK - v) / (hiK - loK) * (B - T), bh = Math.max(3, (B - T) / LD.length - 2);
  const mx = Math.max(...LD.flatMap(r => [r[1], r[2]])), k = (mid - gap - 70) / mx;
  const fk = v => v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : (v / 1e3).toFixed(0) + 'k';
  const topC = [...LD].sort((a, b) => b[1] - a[1]).slice(0, 3).map(r => r[0]);
  const topP = [...LD].sort((a, b) => b[2] - a[2]).slice(0, 3).map(r => r[0]);
  // reference lines first, so bars and their labels sit on top
  const line = (v, col, label, dash, right) => {{ if (v == null || v < loK || v > hiK) return;
    el('line', {{x1: 0, x2: W, y1: Y(v), y2: Y(v), stroke: col, 'stroke-width': 2, 'stroke-dasharray': dash}}, svg);
    txt(svg, right ? W - 2 : 2, Y(v) - 5, label, {{'text-anchor': right ? 'end' : 'start', fill: col, 'font-size': 13, 'font-weight': 700}}); }};
  line(SPOT, '#eef1f5', `close ${{SPOT.toFixed(2)}}`, '6 4', false);
  line(FLIP, '#f0b35a', `zero-gamma ~${{FLIP.toFixed(1)}}`, '2 3', true);
  LD.forEach(([b, c, p]) => {{
    const y = Y(b + BIN) + 1;
    if (c) el('rect', {{x: mid - gap / 2 - c * k, y, width: Math.max(1, c * k), height: bh, rx: 2, fill: '#9085e9'}}, svg);
    if (p) el('rect', {{x: mid + gap / 2, y, width: Math.max(1, p * k), height: bh, rx: 2, fill: '#199e70'}}, svg);
    if (Math.round(b) % (BIN * 2) === 0 || topC.includes(b) || topP.includes(b))
      txt(svg, mid, y + bh - 1, String(b), {{'text-anchor': 'middle', fill: '#7d8796', 'font-size': 11.5, class: 'num'}});
    if (topC.includes(b)) txt(svg, mid - gap / 2 - c * k - 6, y + bh - 1, fk(c), {{'text-anchor': 'end', fill: '#eef1f5', 'font-size': 12.5, 'font-weight': 700, class: 'num'}});
    if (topP.includes(b)) txt(svg, mid + gap / 2 + p * k + 6, y + bh - 1, fk(p), {{fill: '#eef1f5', 'font-size': 12.5, 'font-weight': 700, class: 'num'}});
  }});
  txt(svg, mid - gap / 2 - 6, 16, '← calls', {{'text-anchor': 'end', fill: '#7d8796', 'font-size': 12.5}});
  txt(svg, mid + gap / 2 + 6, 16, 'puts →', {{fill: '#7d8796', 'font-size': 12.5}});
}})();
</script>
</body></html>
"""
    write(f"{stem}-{SLUG[k]}.html", html)
