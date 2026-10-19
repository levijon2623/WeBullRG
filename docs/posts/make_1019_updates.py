"""make_1019_updates.py -- the Monday 10/19 update versions of the overview and index cards: copies of
the 10/16 cards with the verdicts from the 10/19 open interest. ETF updates: make_etf_cards.py (UPDATES)."""
D = r"C:\Users\OptiPlex 5080\Desktop\Cleanbot\docs\posts"
UPD = ' <span class="upd">MONDAY UPDATE</span>'


def make(src, dst, pairs):
    s = open(f"{D}\\{src}", encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (src, s.count(a), a[:70])
        s = s.replace(a, b)
    open(f"{D}\\{dst}", "w", encoding="utf-8").write(s)
    print("wrote", dst)


make("2026-10-16-etf-overview.html", "2026-10-19-etf-overview-update.html", [
    ("<title>ETF Positioning Overview</title>", "<title>ETF Positioning Update</title>"),
    ('<div class="kicker">SPY · QQQ · IWM options positioning · Oct 5–16, 2026</div>',
     '<div class="kicker">SPY · QQQ · IWM options positioning · Oct 5–16, judged by Oct 19 open interest' + UPD + '</div>'),
    ("""  <div class="sub">All three ETFs saw steady, financed put buying through a two-week slide. Friday's surge was QQQ's alone,
    and the call side split three ways.</div>""",
     """  <div class="sub">All three ETFs saw steady, financed put buying through a two-week slide. Monday's open interest:
    Friday was a hedge <i>reset</i> in tech, and only SPY added fresh protection.</div>"""),
    ("""<div class="v num">+$89M <span class="w">+ $243M Fri</span></div>""",
     """<div class="v num">+$89M <span class="w">· Fri rolled</span></div>"""),
    ("""<div class="rl">Style</div><div class="w">Put spreads, rolled up as it fell</div><div class="w">Heavy puts into Friday's close</div><div class="w">Hedged both ways</div>""",
     """<div class="rl">Style</div><div class="w">Put spreads; new 10/30 spread kept Fri</div><div class="w">Hedge reset into Friday's close</div><div class="w">Hedged both ways, re-struck lower</div>"""),
    ("""      <li><b>Friday was a tech event.</b> QQQ's $2.42B of puts outran SPY's, a rare flip for the smaller market,
        right after QQQ's −6% two-week slide.</li>""",
     """      <li><b>Friday was a hedge reset.</b> QQQ banked ~$45M on its $715 puts and moved its floor to $700–675, and SPX
        closed its $7,750/$7,650 puts. Only SPY added new 10/30 protection (+87k $750 / +81k $735 kept).</li>"""),
    ("""  <div class="read"><b>Monday:</b> Friday's prints are confirmed or not by the 10/19 open interest (QQQ $700P toward
    ~200k = a new hedge; flat = partly a trapped seller covering).<span class="mon">Detail per ETF in the SPY, QQQ and IWM cards.
    Side is read from ask/bid; "opened" counts are a lower bound.</span></div>""",
     """  <div class="read"><b>Monday's open interest (10/19):</b> SPY ✓ new 750/735 spread kept · QQQ ↻ rolled and monetized
    ($715P −64k; $690/$700P +36k; $675P and Nov $720C sold) · SPX ↻ $7,750/$7,650P mostly closed (−15k) · IWM ↻ spread re-struck lower.
    <span class="mon">Protection for the 10/30 window is still large; it now sits lower. Side is read from ask/bid.</span></div>"""),
])

make("2026-10-16-index-vs-etf.html", "2026-10-19-index-vs-etf-update.html", [
    ("<title>Index vs ETF</title>", "<title>Index vs ETF Update</title>"),
    ('<div class="kicker">Index vs ETF options · Oct 5–16, 2026 · SPX / NDX / RUT vs SPY / QQQ / IWM</div>',
     '<div class="kicker">Index vs ETF options · Oct 5–16, judged by Oct 19 open interest' + UPD + '</div>'),
    ("""was 3× SPY's. The SPY cards show the small end of an S&amp;P hedge.</li>""",
     """was 3× SPY's. Monday's open interest: that SPX activity was mostly closing ($7,750/$7,650P −15k), while
        SPY's new 10/30 spread was kept (+168k).</li>"""),
    ("""package had no NDX twin: NDX 10/30 delta was flat and its biggest 10/30 puts traded in single-digit lots.</li>""",
     """package had no NDX twin. Monday's open interest showed that package was a roll: $715s cashed in, floor moved to $700–675.</li>"""),
])
