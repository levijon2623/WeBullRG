"""
check_calendar_forward.py -- FORWARD PAPER TEST, LOCKED 2026-11-25 (user): the SPY at-the-money calendar (short the 0DTE,
long the next expiry) on BIG NO-FADE mornings. Never retuned.

WHY. check_premium.py (holdout scored 2026-11-25): P2 failed (+$10.26 [-3.75, +24.52], q 0.33), but SPY was the same size
in both halves (+$10.47 / +$10.26 a spread) and beat ordinary mornings both times; QQQ / IWM disagreed. A lead, so it needs
sessions nobody has seen. If it ever passes, the user would add it to a premium-selling book with the butterflies (their
own forward test), not trade it alone.

RULES (imported, not copied -- check_overnight.py and check_premium.py as committed at 5219404)
  night    check_overnight.night_rows: BIG = z >= 1.37, NO FADE = F < 1/3 (SPY extended-hours candles)
  trade    check_premium.calendar at the whole-dollar strike nearest the 09:35 price, calls; entry 09:35, exit 15:55;
           check_premium.spread_pnl at the user's fills (mid + $0.06 / - $0.08 a contract per leg, $0.10 fees per leg)
SESSIONS from 2026-11-27 (the first session after this lock). Needs, for each session, the SPY extended candles
  (uw_ext_candles.py --backfill) and the option-contracts-1m silver (uw_options_data_lake.py build -> silver-build).
READS    --status any time (counts, no P&L). INTERIM at 50 trades (descriptive, no stopping). FINAL at 100 trades
  (~2 years at about one big no-fade morning a week): PASS = mean P&L per spread > 0 with the bootstrap (2,000 resamples)
  95% CI above 0. Reported, not scored: worst-case fills; the same calendar on ordinary mornings; puts.

    python check_calendar_forward.py --status | --read
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
START = "2026-11-27"
INTERIM, FINAL = 50, 100


def trades(score=True):
    import check_overnight as O
    import check_premium as P
    rows = [r for r in O.night_rows("SPY") if r["d"] >= START]
    ext = O.load("SPY")
    out = []
    for r in rows:
        d, B = P.day_book(("SPY", r["d"]))
        x = ext[r["d"]].filter(pl.col("market_time") == "r")
        e, z = x.filter(pl.col("mod") == P.ENTRY)["close"], x.filter(pl.col("mod") == P.EXIT)["close"]
        if not B or not len(e) or not len(z):
            out.append(dict(d=r["d"], group="unpriced"))
            continue
        B = dict(B, d=r["d"])
        grp = "bnf" if (r["big"] and not r["faded"]) else ("ord" if not r["big"] else "bf")
        row = dict(d=r["d"], group=grp)
        if score:
            K = int(round(float(e[0])))
            row["C"] = P.spread_pnl(B, P.calendar(B, K, "C"), float(z[0]))
            row["Cw"] = P.spread_pnl(B, P.calendar(B, K, "C"), float(z[0]), worst=True)
            row["Cp"] = P.spread_pnl(B, P.calendar(B, K, "P"), float(z[0]))
        out.append(row)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.status:
        T = trades(score=False)
        n = sum(1 for t in T if t["group"] == "bnf")
        print(f"sessions from {START} with data: {len(T)} | big no-fade (trades): {n} | "
              f"unpriced (no option data yet): {sum(1 for t in T if t['group'] == 'unpriced')} | final at {FINAL}")
        return
    if a.read:
        import check_premium as P
        T = trades()
        B = [t for t in T if t["group"] == "bnf" and t.get("C") is not None]
        n = len(B)
        stage = "FINAL" if n >= FINAL else ("interim (descriptive)" if n >= INTERIM else "early (descriptive)")
        m, lo, hi, k, _ = P.mean_ci([t["C"] for t in B])
        print(f"{stage}: {n} trades | calendar ${m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
              + ((" -> " + ("PASS" if lo > 0 else "FAIL")) if n >= FINAL else ""))
        for key, lab in (("Cw", "worst-case fills"), ("Cp", "puts")):
            print(f"  {lab}: ${P.mean_ci([t.get(key) for t in B])[0]:+.2f}")
        O_ = [t["C"] for t in T if t["group"] == "ord" and t.get("C") is not None]
        if O_:
            print(f"  ordinary mornings, same calendar: ${np.mean(O_):+.2f} (n {len(O_)})")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
