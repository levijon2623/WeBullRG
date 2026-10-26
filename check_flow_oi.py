# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_flow_oi.py
================
TRAIN THE FLOW INTERPRETER ON NEXT-DAY OPEN INTEREST (user, 2026-10-25): how much of a
day's volume, by side, actually turns into overnight open-interest change?

The intraday map's flow mode assumes every contract bought at the ask is a new customer long
(and sold at the bid a new customer short). Tomorrow's start-of-day OI is the truth about how
much of today's volume stayed open, so it can calibrate that.

DATA     lake/silver/option-contracts-1m 2023-10-12 .. 2026-09-18: per contract-day, the day's
         ask / bid / mid / no-side / multi-leg volume (all sessions) and start-of-day OI; target
         dOI = OI(next trading day) - OI(today). Roots SPY, SPXW+SPX (S&P), QQQ, IWM; 1-45 DTE
         (0DTE has no next-day OI -- it expires); strikes within 10% of spot. A contract with no
         trade on the next day has no OI row there and drops out.
MODEL    per bucket = market x call/put x DTE (1, 2-7, 8-45) x moneyness (|k/S-1| <= 1%, 1-3%, > 3%):
         dOI = a.ask + b.bid + m.mid + n.noside + k.multi  (least squares, no intercept).
FIT      2023-10-12 .. 2024-08-19 only (the labelled pre-sample; check_gamma_modes doesn't score it).
SCORED   2024-08-20 .. 2026-09-17 (last day with a next day), halves split 2025-08-21.

PRE-REGISTERED
  STAGE 1 (is the fit real?)  per market, out-of-sample R^2 of dOI (pooled over contract-days)
     for: zero (R^2 by definition <= 0), POOLED (one coefficient on total volume, per market),
     BUCKET (the model). PASS for a market if BUCKET's R^2 > 0 and > POOLED's in both halves.
  STAGE 2 (does it make a better map?)  only for markets that pass stage 1: a 4th mode in
     check_gamma_modes -- "fit": dealer position = the usual convention (calls +, puts -) applied to
     OI + predicted dOI from the flow so far (0DTE uses the 1-DTE bucket's coefficients). It replaces
     single-leg flow as the map's default only if, on S&P, IC(fit) - IC(single) is < 0 with the paired
     day-block CI below 0, the same sign in both halves, and in at least one of QQQ / IWM.
  Reported: the coefficients (share of ask / bid / mid / multi volume that stays open), by bucket.

Usage:  python check_flow_oi.py --build    # per contract-day table -> _flowoi_cache/ (~20 min)
        python check_flow_oi.py            # fit, score, write _flowoi_cache/coef.json for check_gamma_modes

RUN LOG
  2026-10-25  8.58M contract-days built; 5.30M with a next-day OI inside 10% / 1-45 DTE (fit 1.12M, scored 4.18M).
    STAGE 1 out-of-sample R^2 (all / half1 / half2):
      SP    zero -0.020  pooled +0.208 / +0.226 / +0.197   bucket +0.211 / +0.238 / +0.193   FAIL (half 2)
      QQQ   zero -0.034  pooled +0.241 / +0.286 / +0.219   bucket +0.287 / +0.376 / +0.242   PASS
      IWM   zero -0.007  pooled +0.107 / +0.140 / +0.082   bucket +0.243 / +0.365 / +0.150   PASS
    Coefficients: ask volume mostly OPENS (+0.3..+0.7 stays open), bid volume mostly CLOSES (-0.1..-0.3);
    mid / no-side / multi noisy and sign-unstable. Volume explains ~20-29% of overnight dOI.
    STAGE 2 (check_gamma_modes, fit vs single-leg flow, scored):
      SP    fit -0.159  single -0.207   +0.047 [+0.021, +0.072]  halves +0.071 / +0.012   (oi -0.166)
      QQQ   fit -0.157  single -0.190   +0.032 [+0.001, +0.065]  halves +0.012 / +0.012   (oi -0.165)
      IWM   fit -0.186  single -0.159   -0.027 [-0.064, +0.008]  halves +0.029 / -0.077   (oi -0.174)
    VERDICT: FAIL (S&P failed stage 1; the fitted mode is also WORSE than single-leg flow on S&P and QQQ).
    Reading: the fit only says how much OI the flow adds, and the convention then signs it (calls +, puts -),
    so it lands back near OI-only. What sizes moves better is the SIDE of today's flow, which the fit throws
    away. The map keeps single-leg side-signed flow as its default.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
LAKE = "lake/silver/option-contracts-1m"
CACHE = "_flowoi_cache"
MKT_OF = {"SPY": "SP", "SPXW": "SP", "SPX": "SP", "QQQ": "QQQ", "IWM": "IWM"}
FIT_END, DEP_START, SPLIT = dt.date(2024, 8, 19), dt.date(2024, 8, 20), dt.date(2025, 8, 21)
X = ["ask", "bid", "mid", "noside", "multi"]


def dte_b(d):
    return np.where(d <= 1, "1", np.where(d <= 7, "2-7", "8-45"))


def mny_b(m):
    return np.where(m <= 0.01, "atm", np.where(m <= 0.03, "near", "far"))


def build():
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(f"{LAKE}/date=*/bars.parquet"))
    out = []
    for n, f in enumerate(files):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        d = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(MKT_OF)))
             .filter(pl.col("expiry") <= pl.lit(day + dt.timedelta(days=46)))
             .group_by("option_chain_id")
             .agg(pl.col("underlying_symbol").first().alias("root"), pl.col("option_type").first().alias("cp"),
                  pl.col("strike").first().alias("k"), pl.col("expiry").first(),
                  pl.col("open_interest").drop_nulls().first().alias("oi"),
                  pl.col("underlying_close").drop_nulls().median().alias("S"),
                  pl.col("ask_volume").sum().alias("ask"), pl.col("bid_volume").sum().alias("bid"),
                  pl.col("mid_volume").sum().alias("mid"), pl.col("no_side_volume").sum().alias("noside"),
                  pl.col("multi_volume").sum().alias("multi"))
             .with_columns(pl.lit(day).alias("day"))
             .collect())
        out.append(d)
        if n % 50 == 0:
            print(f"  {day} ({n + 1}/{len(files)}) contracts {d.height}", flush=True)
    pl.concat(out).write_parquet(f"{CACHE}/contract_days.parquet")
    print("  built", sum(x.height for x in out), "contract-days")


def load():
    d = pl.read_parquet(f"{CACHE}/contract_days.parquet").to_pandas()
    d["day"] = pd.to_datetime(d["day"]).dt.date
    days = sorted(d["day"].unique())
    nxt = {a: b for a, b in zip(days, days[1:]) if (b - a).days <= 4}
    d["next"] = d["day"].map(nxt)
    t = d[["option_chain_id", "day", "oi"]].rename(columns={"day": "next", "oi": "oi_next"})
    d = d.merge(t, on=["option_chain_id", "next"], how="inner")
    d["dte"] = (pd.to_datetime(d["expiry"]) - pd.to_datetime(d["day"])).dt.days
    d = d[(d["dte"] >= 1) & (d["dte"] <= 45) & d["oi"].notna() & d["oi_next"].notna() & d["S"].notna()]
    d["mkt"] = d["root"].map(MKT_OF)
    d["m"] = np.abs(d["k"] / d["S"] - 1)
    d = d[d["m"] <= 0.10]
    d["doi"] = d["oi_next"] - d["oi"]
    d["vol"] = d[X[:4]].sum(axis=1)
    d["bucket"] = d["mkt"] + "|" + d["cp"] + "|" + dte_b(d["dte"].to_numpy()) + "|" + mny_b(d["m"].to_numpy())
    return d


def ols(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def r2(y, p):
    return 1 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def main():
    d = load()
    fit = d[d["day"] <= FIT_END]
    sc = d[d["day"] >= DEP_START]
    print(f"  contract-days {len(d):,}  fit {len(fit):,} ({fit['day'].min()} .. {fit['day'].max()})  scored {len(sc):,}")
    coef, pooled = {}, {}
    for b, g in fit.groupby("bucket"):
        if len(g) >= 200:
            coef[b] = ols(g[X].to_numpy(float), g["doi"].to_numpy(float)).tolist()
    for m, g in fit.groupby("mkt"):
        pooled[m] = float(ols(g[["vol"]].to_numpy(float), g["doi"].to_numpy(float))[0])

    def predict(x):
        p = np.zeros(len(x))
        for b, g in x.groupby("bucket"):
            if b in coef:
                p[x.index.get_indexer(g.index)] = g[X].to_numpy(float) @ np.array(coef[b])
        return p

    sc = sc.reset_index(drop=True)
    sc["p_bucket"] = predict(sc)
    sc["p_pooled"] = sc["vol"] * sc["mkt"].map(pooled)
    print("\n  ======== STAGE 1: out-of-sample R^2 of next-day dOI ========")
    passed = []
    for m in ("SP", "QQQ", "IWM"):
        s = sc[sc["mkt"] == m]
        line, ok = [], True
        for name, h in (("all", s), ("half1", s[s["day"] < SPLIT]), ("half2", s[s["day"] >= SPLIT])):
            y = h["doi"].to_numpy(float)
            rz, rp, rb = r2(y, np.zeros_like(y)), r2(y, h["p_pooled"].to_numpy()), r2(y, h["p_bucket"].to_numpy())
            line.append(f"{name}: zero {rz:+.3f} pooled {rp:+.3f} bucket {rb:+.3f}")
            if name != "all":
                ok &= rb > 0 and rb > rp
        print(f"     {m:4} n={len(s):,}  " + "   ".join(line) + f"   -> {'PASS' if ok else 'FAIL'}")
        if ok:
            passed.append(m)
    print(f"     stage 1 passes: {passed or 'none'}")

    print("\n  REPORTED: share of each kind of volume that stays open overnight (fit window), 1-DTE and 2-7 buckets")
    print(f"     {'bucket':24} " + " ".join(f"{x:>7}" for x in X) + "    n")
    nb = fit.groupby("bucket").size()
    for b in sorted(coef):
        if b.split("|")[2] in ("1", "2-7"):
            print(f"     {b:24} " + " ".join(f"{c:+7.2f}" for c in coef[b]) + f"  {nb[b]:,}")
    print("     pooled (dOI per contract of volume): " + "  ".join(f"{m} {v:+.3f}" for m, v in pooled.items()))
    json.dump({"coef": coef, "pooled": pooled, "passed": passed, "fit_end": FIT_END.isoformat()},
              open(f"{CACHE}/coef.json", "w"), indent=1)
    print(f"\n  wrote {CACHE}/coef.json")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        main()
