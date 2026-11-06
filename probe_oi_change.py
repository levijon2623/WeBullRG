"""
probe_oi_change.py -- overnight open-interest change for SPY / QQQ / IWM and
SPX / NDX / RUT (operator, 2026-11-04). The premarket half of the daily flow
report; probe_notable.py is the intraday half.

    python probe_oi_change.py --spots SPY=767.28,QQQ=692.51,IWM=283.33 \
        --watch IWM261105C00290000,SPY261120P00718000 --out <scratch>/oi.json

UW /option-contracts rows carry `open_interest` (start of TODAY) and `prev_oi`
(start of the LAST session), so one pass per expiry and side gives the whole
chain's change -- the verdict on the last session's prints (opened, closed,
rolled). For the biggest changes it also pulls the last session's volume,
ask / bid / multi-leg split and average price from /option-contract/{occ}/historic,
which is what says HOW the change happened (+38k OI with 74% at the bid =
calls written, not bought).

Expiries: every trading day from today to +17 calendar days, plus the next three
monthly OPEX Fridays (third Friday). --expiries overrides.
Rules baked in:
  * the expiry / side filters are verified on the returned OCC symbol (UW
    ignores unknown params and answers 200 with unfiltered rows);
  * SPX chains return both roots, RUT weeklies only come back under RUTW;
  * |delta| >= 0.9 is listed separately as "deep ITM" -- box legs and financing
    (SPX 7000C/8000P-style pairs), never read as direction;
  * notional is per leg: two strikes changing by the same size are one spread
    (SPX 11/11 7520P/7510P +8k each on 11-04 = a 10-point spread, not $12B of puts).
Same caveats as probe_notable.py: side is inferred from ask / bid, and a print
can't be classed open vs close until the next session's open interest.
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
import market_calendar as MC                             # noqa: E402
import positioning_map as P                              # noqa: E402

NY = ZoneInfo("America/New_York")
CHAINS = [("SPY", "SPY"), ("SPX", "SPY"), ("QQQ", "QQQ"), ("NDX", "QQQ"),
          ("IWM", "IWM"), ("RUTW", "IWM"), ("RUT", "IWM")]


def third_friday(y, m):
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def default_expiries(today):
    out = set(MC.trading_days(today, today + dt.timedelta(days=17)))
    y, m, monthlies = today.year, today.month, []
    while len(monthlies) < 3:
        f = third_friday(y, m)
        if f >= today:
            monthlies.append(f)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(out | set(monthlies))


def scan(uw, expiries):
    n, seen, out = P.num, set(), []
    for tk, etf in CHAINS:
        for e in expiries:
            for cp in ("call", "put"):
                rows = uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat(),
                              option_type=cp, limit=500).get("data") or []
                for r in rows:
                    s = r["option_symbol"]
                    o = P.parse_occ(s)
                    if s in seen or not o or o[1] != e or o[2] != cp[0].upper():
                        continue
                    seen.add(s)
                    oi, prev = int(n(r.get("open_interest")) or 0), int(n(r.get("prev_oi")) or 0)
                    out.append(dict(s=s, tk=tk, etf=etf, e=e.isoformat(), cp=o[2], k=o[3],
                                    oi=oi, prev=prev, d=oi - prev, delta=n(r.get("delta")) or 0.0,
                                    notl=abs(oi - prev) * o[3] * 100,
                                    vol=int(n(r.get("volume")) or 0), ask=int(n(r.get("ask_volume")) or 0),
                                    bid=int(n(r.get("bid_volume")) or 0),
                                    ml=int(n(r.get("multi_leg_volume")) or 0),
                                    last=n(r.get("last_price")), hi=n(r.get("high_price"))))
    return out


def last_session(uw, sym, before):
    """The last session's row for one contract: how its OI change was made."""
    h = [r for r in uw.get(f"/option-contract/{sym}/historic").get("chains", [])
         if r["date"] < before]
    if not h:
        return {}
    r, n = max(h, key=lambda r: r["date"]), P.num
    return dict(date=r["date"], vol=int(n(r.get("volume")) or 0),
                ask=int(n(r.get("ask_volume")) or 0), bid=int(n(r.get("bid_volume")) or 0),
                ml=int(n(r.get("multi_leg_volume")) or 0), avg=n(r.get("avg_price")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.datetime.now(NY).date().isoformat())
    ap.add_argument("--expiries", default="", help="comma list; default = see docstring")
    ap.add_argument("--spots", default="", help="only for the header line")
    ap.add_argument("--watch", default="", help="comma list of OCC symbols to report")
    ap.add_argument("--top", type=int, default=12, help="contracts per family")
    ap.add_argument("--out")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date)
    exps = ([dt.date.fromisoformat(x) for x in a.expiries.split(",") if x]
            or default_expiries(today))
    uw = P.UW()
    rows = scan(uw, exps)
    by = {r["s"]: r for r in rows}
    print(f"{len(rows)} contracts, expiries {exps[0]} .. {exps[-1]}  {a.spots}")

    def line(r, ls=None):
        t = (f'{r["tk"]:5s} {r["e"][5:]} {r["k"]:>7g}{r["cp"]}  {r["prev"]:>7,} -> {r["oi"]:>7,} '
             f'{r["d"]:>+8,}  ${r["notl"] / 1e6:>6,.0f}M  d {r["delta"]:+.2f}')
        if ls:
            t += (f'  | {ls["date"][5:]} vol {ls["vol"]:,} ask {ls["ask"]:,} bid {ls["bid"]:,} '
                  f'ml {ls["ml"]:,} avg {ls["avg"] or 0:.2f}')
        return t

    report = dict(date=a.date, expiries=[e.isoformat() for e in exps], families={}, watch=[])
    for fam in ("SPY", "QQQ", "IWM"):
        fr = [r for r in rows if r["etf"] == fam]
        live = [r for r in fr if 0.02 < abs(r["delta"]) < 0.9 and r["e"] != a.date]
        top = sorted(live, key=lambda r: -r["notl"])[:a.top]
        print(f"\n===== {fam} family: biggest overnight OI changes by notional "
              f"(ex today's 0DTE, ex deep ITM / far OTM)")
        for r in top:
            r["last_session"] = last_session(uw, r["s"], a.date)
            print("  " + line(r, r["last_session"]))
        deep = sorted([r for r in fr if abs(r["delta"]) >= 0.9], key=lambda r: -r["notl"])[:4]
        print("  deep ITM (financing / box legs, not direction): "
              + ", ".join(f'{r["tk"]} {r["e"][5:]} {r["k"]:g}{r["cp"]} {r["d"]:+,}' for r in deep))
        report["families"][fam] = dict(top=top, deep=deep)
    if a.watch:
        print("\n===== watch list")
        for s in a.watch.split(","):
            r = by.get(s.strip())
            if r:
                r["last_session"] = last_session(uw, r["s"], a.date)
                report["watch"].append(r)
                print("  " + line(r, r["last_session"]))
            else:
                print(f"  {s}: not in the scanned expiries")
    print(f"\nUW calls {uw.calls}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
