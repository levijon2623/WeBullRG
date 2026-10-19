"""
probe_notable.py -- numbers for a "Notable Activity" card (operator, 2026-10-19).

    python probe_notable.py --expiry 2026-10-30 --spots SPY=770,QQQ=724,IWM=272.44 \
        --out docs/posts/notable/2026-10-19.json

For one expiry, today so far, for SPY / QQQ / IWM and SPX / NDX / RUT (all roots):
  * directional delta flow (UW /greek-flow/{expiry}?date=; already in SHARES)
    x spot = dollars, per ticker;
  * put / call premium and the biggest contracts by premium (UW
    /option-contracts?expiry=&option_type=, which honors both filters), with
    ask / bid / multi-leg share and start-of-day open interest.
It prints a summary and writes a SKELETON spec for docs/posts/make_notable.py:
numbers filled, words left as "TODO" -- the headline, notes and read are
written by a person from what the numbers show (e.g. two legs printing in
the same size = one package; a leg that reverses the last session's = a roll).
Spots: pass live ones with --spots (index levels = ETF x index_gex.SEED_RATIO);
without them the last close from UW history is used.
Same caveats as probe_uoa.py: side is inferred, today's OI is start-of-day,
the verdict on any print comes with the NEXT session's open interest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                           # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
import uw_options_data_lake as L                         # noqa: E402
import index_gex as IX                                   # noqa: E402

NY = ZoneInfo("America/New_York")
FAMILY = [("SPY", ["SPY"], "SPY"), ("SPX", ["SPX", "SPXW"], "SPY"), ("QQQ", ["QQQ"], "QQQ"),
          ("NDX", ["NDX", "NDXP"], "QQQ"), ("IWM", ["IWM"], "IWM"), ("RUT", ["RUT", "RUTW"], "IWM")]


def f(v):
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def last_close(c, etf, expiry):
    six = expiry[2:].replace("-", "")
    rows = c.get_json(f"/stock/{etf}/option-contracts", {"expiry": expiry, "option_type": "put"}).get("data", [])
    rows = [r for r in rows if r["option_symbol"][-15:-9] == six]
    if not rows:
        return 0.0
    sym = max(rows, key=lambda r: f(r["open_interest"]))["option_symbol"]
    h = sorted(c.get_json(f"/option-contract/{sym}/historic").get("chains", []), key=lambda r: r["date"])
    closes = [f(r["underlying_close"]) for r in h if f(r["underlying_close"])]
    return closes[-1] if closes else 0.0


def chip(ask, bid):
    return ("b", "BOUGHT") if ask >= 0.6 else ("s", "SOLD") if bid >= 0.6 else ("x", "MIXED")


def money(v):
    return f"{'−' if v < 0 else '+'}${abs(v) / 1e9:.2f}B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry", required=True)
    ap.add_argument("--date", default=dt.datetime.now(NY).date().isoformat())
    ap.add_argument("--spots", default="", help="SPY=770,QQQ=724,IWM=272.44 (live); else last close")
    ap.add_argument("--top", type=int, default=3, help="contracts kept per ticker")
    ap.add_argument("--out")
    a = ap.parse_args()
    c = L.Client(os.environ["UW_API_KEY"])
    spot = {k: float(v) for k, v in (x.split("=") for x in a.spots.split(",") if "=" in x)}
    for etf in ("SPY", "QQQ", "IWM"):
        spot.setdefault(etf, last_close(c, etf, a.expiry))
    six = a.expiry[2:].replace("-", "")
    bars, trades, nxt = [], [], dt.date.fromisoformat(a.date) + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    pend = nxt.strftime("%a").upper()
    total = 0.0
    for name, roots, etf in FAMILY:
        s = spot[etf] * (IX.SEED_RATIO[etf] if name != etf else 1.0)
        g = [r for r in c.get_json(f"/stock/{roots[0]}/greek-flow/{a.expiry}", {"date": a.date}).get("data", [])
             if str(r.get("timestamp", ""))[:10] == a.date]
        dd = sum(f(r["dir_delta_flow"]) for r in g) * s
        total += dd
        rows = {}
        for rt in roots:
            for typ in ("put", "call"):
                try:
                    for r in c.get_json(f"/stock/{rt}/option-contracts", {"expiry": a.expiry, "option_type": typ}).get("data", []):
                        if r["option_symbol"][-15:-9] == six:
                            rows[r["option_symbol"]] = r
                except L.NoDataForDate:
                    pass
        pp = sum(f(r["total_premium"]) for r in rows.values() if r["option_symbol"][-9] == "P")
        cp = sum(f(r["total_premium"]) for r in rows.values() if r["option_symbol"][-9] == "C")
        bars.append(dict(label=name, v=abs(dd) / 1e9, text=money(dd)))
        print(f"{name:4} spot {s:>10,.2f}  delta {money(dd)}  put ${pp / 1e6:,.0f}M  call ${cp / 1e6:,.0f}M  "
              f"(to {g[-1]['timestamp'][11:16] + ' UTC' if g else '-'})")
        for r in sorted(rows.values(), key=lambda r: -f(r["total_premium"]))[:a.top]:
            v = f(r["volume"])
            ask, bid, ml = f(r["ask_volume"]) / max(v, 1), f(r["bid_volume"]) / max(v, 1), f(r["multi_leg_volume"]) / max(v, 1)
            side, lab = chip(ask, bid)
            sym = r["option_symbol"]
            k = f"{dt.datetime.strptime(six, '%y%m%d'):%b-%d} ${int(sym[-8:]) / 1000:,g}{sym[-9]}"
            trades.append(dict(d=name, k=k, side=side, chip=lab, n=f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}",
                               prem=round(f(r["total_premium"]) / 1e6, 1), pend=pend,
                               note=f"TODO · ask {ask:.0%} bid {bid:.0%} mleg {ml:.0%} · OI {f(r['open_interest']):,.0f}"))
            print(f"     {sym:22} vol {v:>8,.0f} ask {ask:4.0%} bid {bid:4.0%} mleg {ml:4.0%} prem ${f(r['total_premium']) / 1e6:6.1f}M "
                  f"OI(start) {f(r['open_interest']):>8,.0f}")
    spec = dict(date=a.date, kicker=f"{a.expiry} options flow · {a.date} · through TODO ET",
                h1="TODO headline", sub="TODO sub",
                tiles=[[f"{spot[e]:,.2f}", f"{e} spot"] for e in ("SPY", "QQQ", "IWM")] + [[money(total), f"net {a.expiry} delta, all six"]],
                bars_title=f"Directional delta in the {a.expiry} expiry, so far",
                bars_sub="$ billions · negative = bearish (puts bought / calls sold) · ETFs and their indexes",
                bars_max=round(max(b["v"] for b in bars) * 1.05 or 1, 3), bars=bars,
                trades=sorted(trades, key=lambda t: -t["prem"])[:8],
                confirm=f"confirmed by {nxt:%A}'s open interest", read="<b>Read:</b> TODO", mon="TODO")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(spec, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("skeleton spec ->", a.out)


if __name__ == "__main__":
    main()
