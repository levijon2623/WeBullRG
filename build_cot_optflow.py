"""
build_cot_optflow.py -- the options-flow side of check_cot.py's H5 (user, 2026-11-14): does the COT read work
better when options flow into the SAME horizon agrees? Daily, from the data lake's 1-minute contract bars
(lake/silver/option-contracts-1m, 2023-10..), per index family:

  S&P     SPY + SPX + SPXW        Nasdaq  QQQ + NDX + NDXP        Russell  IWM + RUT + RUTW

  net_delta_usd = sum over contracts expiring 14..91 calendar days out of
                  (ask volume - bid volume) x single-leg share x delta x 100 x underlying price
  (+ = delta BOUGHT: calls bought / puts sold; single-leg share = 1 - multi-leg / volume, since UW gives the
  multi-leg volume but not its side). Options flow only -- no outcome is read here.

Output: _cot_cache/optflow_daily.parquet (date, family, net_delta_usd, contracts). Resumable per day.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
LAKE = os.path.join(ROOT, "lake", "silver", "option-contracts-1m")
OUT = os.path.join(ROOT, "_cot_cache", "optflow_daily.parquet")
PART = os.path.join(ROOT, "_cot_cache", "optflow_parts")
FAMILY = {"SPY": "SP", "SPX": "SP", "SPXW": "SP", "QQQ": "NQ", "NDX": "NQ", "NDXP": "NQ",
          "IWM": "RTY", "RUT": "RTY", "RUTW": "RTY"}
DTE_LO, DTE_HI = 14, 91


def one_day(path):
    d = os.path.basename(os.path.dirname(path))[5:]
    part = os.path.join(PART, d + ".json")
    if os.path.exists(part):
        return d, "have"
    day = dt.date.fromisoformat(d)
    x = (pl.scan_parquet(path)
         .filter(pl.col("underlying_symbol").is_in(list(FAMILY)))
         .with_columns((pl.col("expiry") - pl.lit(day)).dt.total_days().alias("dte"))
         .filter((pl.col("dte") >= DTE_LO) & (pl.col("dte") <= DTE_HI) & pl.col("delta_close").is_not_null())
         .with_columns(pl.col("underlying_symbol").replace_strict(FAMILY).alias("family"),
                       (1 - pl.col("multi_volume").fill_null(0) / pl.col("volume").clip(lower_bound=1)).clip(0, 1).alias("single"))
         .with_columns(((pl.col("ask_volume") - pl.col("bid_volume")) * pl.col("single") * pl.col("delta_close")
                        * 100 * pl.col("underlying_close")).alias("nd"))
         .group_by("family").agg(pl.col("nd").sum().alias("net_delta_usd"), pl.col("volume").sum().alias("contracts"))
         .collect())
    rows = [dict(date=d, family=f, net_delta_usd=float(n), contracts=int(c)) for f, n, c in x.iter_rows()]
    json.dump(rows, open(part, "w"))
    return d, len(rows)


def main():
    os.makedirs(PART, exist_ok=True)
    files = sorted(glob.glob(os.path.join(LAKE, "date=*", "bars.parquet")))
    with ProcessPoolExecutor(4) as ex:
        for i, (d, msg) in enumerate(ex.map(one_day, files)):
            if i % 100 == 0:
                print(f"  {i}/{len(files)} {d} {msg}", flush=True)
    rows = [r for p in sorted(glob.glob(os.path.join(PART, "*.json"))) for r in json.load(open(p))]
    df = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date()).sort(["family", "date"])
    df.write_parquet(OUT)
    print(f"optflow: {df.height} rows, {df['date'].min()}..{df['date'].max()}", flush=True)


if __name__ == "__main__":
    main()
