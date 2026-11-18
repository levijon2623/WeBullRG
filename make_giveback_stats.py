"""
make_giveback_stats.py -- the user's own give-back record, for the viewer's return-from-sideline panel (user, 2026-11-17:
"my biggest problem is still giving back profits intraday").

Input: a Webull "Orders Records (Options)" CSV export. Every FILLED single-leg option order is matched first-in-first-out
into round trips (phone and app closes included -- the bot's ledger misses those). Output: live/giveback_stats.json
(git-ignored: personal P&L never goes in the public repo), which flow_viewer.html shows when you try to lift the sideline.

What it measures (descriptive, in-sample -- your own history, not a test):
  - per day: realized P&L in trade order; days that were green at some point and finished red;
  - by trade number in the day (1-3, 4-6, 7+) and by entry time: trades, win rate, net $;
  - one rule you can follow live, "stop at the first losing close once the day is green", against what happened;
  - optionally (--lake, PC only): how much of each trade's best bid while held you kept at the exit
    (lake option-contracts-1m minute bids, same-day SPY/QQQ/IWM).

Usage:  python make_giveback_stats.py EXPORT.csv [--lake]     then copy live/giveback_stats.json to the box's live/
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "live", "giveback_stats.json")


def round_trips(path):
    fills = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        s = r.get("Symbol") or ""
        if r.get("Status") not in ("Filled", "Partial Filled") or not s or not r.get("Name") or not r.get("Avg Price"):
            continue                                   # blank Name = a leg of a multi-leg order
        t = dt.datetime.strptime(r["Filled Time"][:19], "%m/%d/%Y %H:%M:%S")
        i = next(j for j in range(len(s)) if s[j].isdigit())
        fills.append(dict(sym=s, root=s[:i], exp=dt.datetime.strptime(s[i:i + 6], "%y%m%d").date(), cp=s[i + 6],
                          K=int(s[i + 7:]) / 1000, side=r["Side"], q=int(float(r["Filled"])), px=float(r["Avg Price"]), t=t))
    fills.sort(key=lambda f: f["t"])
    lots, trips = defaultdict(list), []
    for f in fills:
        if f["side"] == "Buy":
            lots[f["sym"]].append([f["q"], f["px"], f["t"]])
            continue
        q = f["q"]
        while q > 0 and lots[f["sym"]]:
            lot = lots[f["sym"]][0]
            n = min(q, lot[0])
            trips.append(dict(sym=f["sym"], root=f["root"], exp=f["exp"], cp=f["cp"], K=f["K"], q=n, entry=lot[1],
                              exit=f["px"], t_in=lot[2], t_out=f["t"], pnl=(f["px"] - lot[1]) * 100 * n))
            lot[0] -= n
            q -= n
            if lot[0] == 0:
                lots[f["sym"]].pop(0)
    return trips


def bucket(ts):
    n = len(ts)
    return dict(n=n, won=round(100 * sum(t["pnl"] > 0 for t in ts) / n) if n else None, net=round(sum(t["pnl"] for t in ts), 2))


def kept_of_peak(trips):
    """median share of the best bid while held kept at the exit, for trades that got to +20% (lake, PC only)."""
    import polars as pl
    lake = os.path.join(ROOT, "lake", "silver", "option-contracts-1m")
    cache, shares = {}, []
    for tr in trips:
        if tr["t_in"].date() != tr["t_out"].date() or tr["root"] not in ("SPY", "QQQ", "IWM"):
            continue
        d = tr["t_in"].date().isoformat()
        if (d, tr["root"]) not in cache:
            f = os.path.join(lake, f"date={d}", "bars.parquet")
            g = defaultdict(dict)
            if os.path.exists(f):
                x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol") == tr["root"])
                     .select(pl.col("expiry").cast(pl.Utf8), "option_type", pl.col("strike").cast(pl.Float64), "minute_et", "bid_close")
                     .collect())
                for e, ot, k, me, b in x.iter_rows():
                    if b is not None:
                        g[(e, ot[0].upper(), round(k, 3))][me.hour * 60 + me.minute] = b
            cache[(d, tr["root"])] = g
        b = cache[(d, tr["root"])].get((tr["exp"].isoformat(), tr["cp"], round(tr["K"], 3)))
        if not b:
            continue
        m0, m1 = tr["t_in"].hour * 60 + tr["t_in"].minute, tr["t_out"].hour * 60 + tr["t_out"].minute
        vals = [v for m, v in b.items() if m0 < m < m1]
        peak = max(vals + [tr["exit"]])
        if peak >= tr["entry"] * 1.2:
            shares.append((tr["exit"] - tr["entry"]) / (peak - tr["entry"]))
    shares.sort()
    return dict(n=len(shares), median=round(100 * shares[len(shares) // 2]) if shares else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lake", action="store_true")
    a = ap.parse_args()
    trips = round_trips(a.csv)
    days = defaultdict(list)
    for t in trips:
        days[t["t_out"].date()].append(t)
    actual = rule = 0.0
    green_red = 0
    for d, ts in days.items():
        ts.sort(key=lambda t: t["t_in"])
        cum, peak, stopped, banked = 0.0, 0.0, False, None
        for t in ts:
            cum += t["pnl"]
            peak = max(peak, cum)
            if banked is None and cum > 0 and t["pnl"] < 0:
                banked = cum
        actual += cum
        rule += banked if banked is not None else cum
        green_red += peak > 0 and cum < 0
    by_no = defaultdict(list)
    by_time = defaultdict(list)
    for ts in days.values():
        for i, t in enumerate(sorted(ts, key=lambda t: t["t_in"]), 1):
            by_no["1-3" if i <= 3 else "4-6" if i <= 6 else "7+"].append(t)
            m = t["t_in"].hour * 60 + t["t_in"].minute
            by_time["09:30-10:30" if m < 630 else "10:30-11:30" if m < 690 else "11:30-13:30" if m < 810 else "13:30-16:00"].append(t)
    out = dict(built=dt.datetime.now().isoformat(timespec="seconds"), source=os.path.basename(a.csv),
               first=min(days).isoformat(), last=max(days).isoformat(), days=len(days), trades=len(trips),
               green_then_red_days=green_red, actual=round(actual, 2), first_loss_rule=round(rule, 2),
               by_trade_no={k: bucket(v) for k, v in sorted(by_no.items())},
               by_entry_time={k: bucket(v) for k, v in sorted(by_time.items())},
               kept_of_peak=kept_of_peak(trips) if a.lake else None)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
