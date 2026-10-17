# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0.0", "numpy>=1.26.0", "httpx", "python-dotenv"]
# ///
"""
probe_uoa.py
============
UNUSUAL OPTIONS ACTIVITY WITH STICKY OPEN INTEREST -- a quick look at named
tickers / expiries (operator, 2026-10-17: IMSR and BYRN calls, 11/20 + 12/18).

Descriptive, not a backtest. For every contract of the asked expiries/right it
pulls Unusual Whales' per-contract daily history and shows, over the window:
  * the underlying close path;
  * per expiry, per day: volume, ask-side / bid-side share, OI, premium;
  * the biggest contract-days: volume vs its trailing 20-day average, vol/OI,
    ask %, sweeps, multi-leg, average price, and whether it STUCK:
        OInext  = the NEXT row's open_interest
        dOI     = OInext - this row's open_interest
        stick   = dOI / volume  (+100% = every contract opened and was kept)
        OInow   = the latest open_interest (did it stay?)

🚨 UW's per-contract `open_interest` is START OF DAY (OCC's overnight figure).
A day's volume shows up in the NEXT row's OI, so the last day in the window
cannot be judged until the next session's row exists.
🚨 Net OI change nets openers against closers, and the ask/bid split only
hints at who opened (ask = bought, bid = sold/written). A big bid-side day
with a big OI rise is call WRITING, not bullish buying.
🚨 /stock/{t}/option-contracts returns only contracts that traded on the
latest day; the full list is /stock/{t}/option-chains.

Usage:
    python probe_uoa.py IMSR BYRN --exp 2026-11-20 2026-12-18 --right C \
        --start 2026-10-05 --end 2026-10-16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
import uw_options_data_lake as L                     # noqa: E402

COLS = ("date", "volume", "open_interest", "ask_volume", "bid_volume", "mid_volume", "avg_price",
        "sweep_volume", "multi_leg_volume", "underlying_close", "trades", "total_premium",
        "implied_volatility")


def pull(c, tk, exps, right):
    yymmdd = {e.replace("-", "")[2:]: e for e in exps}
    syms = [s for s in c.get_json(f"/stock/{tk}/option-chains")["data"]
            if s[len(tk):len(tk) + 6] in yymmdd and s[len(tk) + 6] == right]
    rows = []
    for s in sorted(syms):
        for r in c.get_json(f"/option-contract/{s}/historic").get("chains", []):
            rows.append(dict(tk=tk, sym=s, exp=yymmdd[s[len(tk):len(tk) + 6]][5:].replace("-", "/"),
                             k=int(s[-8:]) / 1000, **{k: r.get(k) for k in COLS}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--exp", nargs="+", required=True, help="expiries YYYY-MM-DD")
    ap.add_argument("--right", default="C", choices=("C", "P"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    c = L.Client(os.environ["UW_API_KEY"])
    df = pd.DataFrame([r for tk in a.tickers for r in pull(c, tk.upper(), a.exp, a.right)])
    if df.empty:
        print("no contracts found"); return
    for col in ("volume", "open_interest", "ask_volume", "bid_volume", "mid_volume", "sweep_volume",
                "multi_leg_volume", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ("avg_price", "underlying_close", "total_premium", "implied_volatility"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")
    df["oi_next"] = g["open_interest"].shift(-1)      # start of next day = after this day's trading
    df["d_oi"] = df["oi_next"] - df["open_interest"]
    df["avg20"] = g["volume"].transform(lambda v: v.shift(1).rolling(20, min_periods=5).mean())
    df["oi_last"] = g["open_interest"].transform("last")
    w = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    print("latest row per ticker:", df.groupby("tk")["date"].max().to_dict(),
          "(OI = start of day; the last day's volume is judged by the NEXT row)")
    for tk in [t.upper() for t in a.tickers]:
        u = df[df.tk == tk].groupby("date")["underlying_close"].first()
        u = u[(u.index >= a.start) & (u.index <= a.end)]
        print(f"\n==== {tk} close: " + "  ".join(f"{d[5:]} {v:.2f}" for d, v in u.items()))
        for ex in sorted(w[w.tk == tk]["exp"].unique()):
            x = w[(w.tk == tk) & (w.exp == ex)]
            day = x.groupby("date").agg(vol=("volume", "sum"), ask=("ask_volume", "sum"),
                                        bid=("bid_volume", "sum"), oi=("open_interest", "sum"),
                                        prem=("total_premium", "sum"))
            print(f"  {ex} {a.right} by day:")
            for d, r in day.iterrows():
                print(f"    {d}  vol {int(r.vol):>6}  ask {r.ask / r.vol if r.vol else 0:4.0%}  "
                      f"bid {r.bid / r.vol if r.vol else 0:4.0%}  OI {int(r.oi):>7}  prem ${r.prem or 0:>10,.0f}")
        x = w[(w.tk == tk) & (w.volume > 0)].sort_values("volume", ascending=False).head(a.top)
        print(f"  top contract-days:\n    {'date':10} {'contract':9} {'vol':>6} {'avg20':>6} {'OI0':>6} "
              f"{'v/OI':>5} {'ask%':>5} {'sweep':>5} {'mleg':>5} {'avgpx':>6} {'prem$':>8} {'OInext':>7} "
              f"{'dOI':>6} {'stick':>6} {'OInow':>6}")
        for _, r in x.iterrows():
            known = r.oi_next == r.oi_next
            stick = r.d_oi / r.volume if known else np.nan
            print(f"    {r.date} {r.exp} {r.k:>4g}{a.right} {int(r.volume):>6} "
                  f"{(r.avg20 if r.avg20 == r.avg20 else 0):>6.0f} {int(r.open_interest):>6} "
                  f"{r.volume / max(r.open_interest, 1):>5.1f} {r.ask_volume / r.volume:>5.0%} "
                  f"{int(r.sweep_volume):>5} {int(r.multi_leg_volume):>5} {(r.avg_price or 0):>6.2f} "
                  f"{(r.total_premium or 0):>8,.0f} {(int(r.oi_next) if known else ''):>7} "
                  f"{(int(r.d_oi) if known else ''):>6} {(f'{stick:+.0%}' if known else ''):>6} {int(r.oi_last):>6}")


if __name__ == "__main__":
    main()
