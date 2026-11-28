"""
probe_packages.py -- pair a session's big option prints into PACKAGES (same second, same underlying) from the lake tape,
so a spread / roll / fly reads as one trade instead of separate legs (operator, 2026-11-25; method in the daily flow
report: probe_oi_change.py premarket, probe_notable.py intraday, this after the lake's trades-core for the day is built).

    python uw_options_data_lake.py trades-core-build 2026-11-25 2026-11-25     # ~2.5 GB download, ~2 min
    python probe_packages.py --date 2026-11-25 --spots SPY=716.79,QQQ=633.95,IWM=268.76

Side is vs the NBBO at the print (>= ask = ASK, <= bid = BID, else mid) and is inferred, like every side here; a package
whose legs sit at opposite sides is the usual spread signature. Which legs OPENED is only known from the next session's
open interest (probe_oi_change.py).
"""
from __future__ import annotations

import argparse
import os

import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--spots", default="", help="SPY=716.79,QQQ=633.95,IWM=268.76 (header only)")
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    ap.add_argument("--min-size", type=int, default=1500, help="contracts per print")
    ap.add_argument("--top", type=int, default=14, help="packages per ticker")
    a = ap.parse_args()
    spots = dict(kv.split("=") for kv in a.spots.split(",") if "=" in kv)
    path = os.path.join(ROOT, "lake", "silver", "trades-core", f"date={a.date}", "trades.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"no lake tape for {a.date}: run uw_options_data_lake.py trades-core-build {a.date} {a.date}")
    x = pl.read_parquet(path).filter(pl.col("canceled").is_null() | (pl.col("canceled") != "true"))
    x = x.with_columns(
        t=pl.col("executed_at").dt.convert_time_zone("America/New_York"),
        side=pl.when(pl.col("price") >= pl.col("nbbo_ask")).then(pl.lit("ASK"))
        .when(pl.col("price") <= pl.col("nbbo_bid")).then(pl.lit("BID")).otherwise(pl.lit("mid")),
        prem=pl.col("price") * pl.col("size") * 100)
    big = x.filter(pl.col("size") >= a.min_size).with_columns(sec=pl.col("t").dt.truncate("1s"))
    g = (big.group_by("underlying_symbol", "sec")
         .agg(pl.col("prem").sum().alias("P"), pl.len().alias("n"),
              pl.struct("expiry", "option_type", "strike", "size", "price", "side", "underlying_price").alias("legs"))
         .sort("P", descending=True))
    for tk in a.tickers.split(","):
        print(f"\n===== {tk} (spot {spots.get(tk, '?')}): biggest same-second packages, prints >= {a.min_size:,} contracts")
        for r in g.filter(pl.col("underlying_symbol") == tk).head(a.top).iter_rows(named=True):
            legs = sorted(r["legs"], key=lambda l: (str(l["expiry"]), l["option_type"], l["strike"]))
            agg = {}
            for l in legs:
                k = (str(l["expiry"])[5:], l["option_type"][0].upper(), l["strike"], l["side"])
                s = agg.setdefault(k, [0, 0.0])
                s[0] += l["size"]
                s[1] += l["size"] * l["price"]
            desc = "  +  ".join(f"{e} {s:g}{c} {n:,} @{v / n:.2f} {sd}" for (e, c, s, sd), (n, v) in agg.items())
            print(f"  {r['sec']:%H:%M:%S}  ${r['P'] / 1e6:5.1f}M  spot {legs[0]['underlying_price']:.2f} | {desc}")


if __name__ == "__main__":
    main()
