# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0"]
# ///
"""
build_implied_move.py
=====================
The at-the-money STRADDLE, from the silver option lake, for the implied-move
study (check_implied_move.py). Nothing here is a test.

Two quotes per session and ticker (SPY, QQQ, IWM, SPXW):

  am  same-day expiry, last quote per contract in 09:33-09:35 ET -- prices the
      move over the REST of the session from the 09:35 bar close.
  pm  next-expiry contracts, last quote in 15:53-15:55 ET of the PRIOR session
      -- prices the overnight gap plus the next session (a pre-open level).
      Stored under the date it was quoted; the check shifts it forward.

Straddle = call mid + put mid at the same strike, from bid_close / ask_close
(never trade prints -- a 0DTE last trade can be minutes stale). It is linearly
interpolated between the two strikes that bracket spot, so a $5 SPX grid and
a $1 SPY grid price the same thing; the nearest strike is kept beside it.

Quote filters: 0 <= bid <= ask, ask > 0, and (ask - bid) / mid <= 0.25 on each
leg. Spot = the underlying_close the lake stamped on the latest bars.

  _implied_move_cache/{date}.parquet   one file per lake date (resumable)

Usage:
  python build_implied_move.py            # every lake date not yet cached
  python build_implied_move.py --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import time

import polars as pl

import implied_move as IM

SILVER = "lake/silver/option-contracts-1m"
CACHE = "_implied_move_cache"
ROOTS = ["SPY", "QQQ", "IWM", "SPXW"]
WINDOWS = {"am": (dt.time(9, 33), dt.time(9, 35)),
           "pm": (dt.time(15, 53), dt.time(15, 55))}
MAX_SPREAD = 0.25


def _quotes(path, day):
    t = pl.col("minute_et").dt.time()
    win = pl.when((t >= WINDOWS["am"][0]) & (t <= WINDOWS["am"][1])).then(pl.lit("am")) \
            .when((t >= WINDOWS["pm"][0]) & (t <= WINDOWS["pm"][1])).then(pl.lit("pm"))
    return (pl.scan_parquet(path)
            .filter(pl.col("underlying_symbol").is_in(ROOTS)
                    & (pl.col("expiry") >= day)
                    & (pl.col("expiry") <= day + dt.timedelta(days=5)))
            .with_columns(win.alias("win"))
            .filter(pl.col("win").is_not_null())
            .select("underlying_symbol", "expiry", "strike", "option_type", "minute_et",
                    "bid_close", "ask_close", "underlying_close", "win")
            .collect())


def _one_day(path, day):
    q = _quotes(path, day)
    out = []
    for (root, win), g in q.group_by(["underlying_symbol", "win"]):
        exps = sorted(e for e in g["expiry"].unique().to_list()
                      if (e == day if win == "am" else e > day))
        if not exps:
            continue
        g = g.filter(pl.col("expiry") == exps[0])
        last = g.sort("minute_et").group_by(["strike", "option_type"]).last()
        m_last = last["minute_et"].max()
        spot = last.filter(pl.col("minute_et") == m_last)["underlying_close"].median()
        last = last.filter((pl.col("bid_close") >= 0) & (pl.col("ask_close") > 0)
                           & (pl.col("ask_close") >= pl.col("bid_close"))) \
                   .with_columns(((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("mid")) \
                   .with_columns(((pl.col("ask_close") - pl.col("bid_close")) / pl.col("mid")).alias("spr")) \
                   .filter(pl.col("spr") <= MAX_SPREAD)
        c = last.filter(pl.col("option_type") == "call").select("strike", pl.col("mid").alias("c"), pl.col("spr").alias("sc"))
        p = last.filter(pl.col("option_type") == "put").select("strike", pl.col("mid").alias("p"), pl.col("spr").alias("sp"))
        both = c.join(p, on="strike").sort("strike")
        if both.is_empty() or spot is None:
            continue
        ks, st = both["strike"].to_list(), (both["c"] + both["p"]).to_list()
        # the live band uses this same function (implied_move.py)
        strad, i_near = IM.interp_straddle(ks, st, spot)
        lo = max((i for i in range(len(ks)) if ks[i] <= spot), default=None)
        hi = min((i for i in range(len(ks)) if ks[i] >= spot), default=None)
        row = both.row(i_near, named=True)
        out.append(dict(date=day, root=root, win=win, expiry=exps[0], minute=m_last,
                        spot=float(spot), strad=float(strad), strad_near=float(st[i_near]),
                        k_near=float(ks[i_near]), c_near=row["c"], p_near=row["p"],
                        spr_near=max(row["sc"], row["sp"]),
                        k_lo=float(ks[lo]) if lo is not None else None,
                        k_hi=float(ks[hi]) if hi is not None else None,
                        n_strikes=len(ks)))
    return pl.DataFrame(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    parts = sorted(glob.glob(f"{SILVER}/date=*/bars.parquet"))
    t0, done = time.time(), 0
    for i, path in enumerate(parts, 1):
        ds = os.path.basename(os.path.dirname(path)).split("=")[1]
        dst = os.path.join(CACHE, f"{ds}.parquet")
        if os.path.exists(dst) and not a.force:
            continue
        df = _one_day(path, dt.date.fromisoformat(ds))
        (df if df is not None else pl.DataFrame({"date": [dt.date.fromisoformat(ds)]})).write_parquet(dst)
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(parts)} {ds}  {el / done:.1f}s/day", flush=True)
    print(f"  done: {done} new days in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
