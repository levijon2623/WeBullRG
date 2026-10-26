# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_gamma_modes.py
====================
WHICH POSITIONS SHOULD THE INTRADAY MAP'S GAMMA LENS USE? (user, 2026-10-25)

intraday_map.py offers the lens three ways, and on Friday 10-23 they disagreed
sharply: S&P 0-1DTE traded 12.3M contracts; the per-contract ask-minus-bid net
summed to 970k against 1.77M of start-of-day open interest.
  oi    start-of-day open interest, usual dealer convention (long calls, short puts)
  raw   oi + today's volume signed by side so far (bought at the ask -> dealer shorter)
  single  as raw, multi-leg volume taken out at its measured lean (multi-leg prints
        54% bid / 42% ask vs 48/48 single-leg, check_opra_multileg): net + 0.12 x multi
        -- the formula the live map can use (UW gives each contract's multi-leg volume
        but not its own side split)
The gamma lens is the one with support (move size), so the question is which
version best predicts the SIZE of the next move.

DATA     lake/silver/option-contracts-1m, 0-1 DTE, strikes within 5% of spot.
         S&P = SPY + SPXW (+SPX monthlies), QQQ, IWM. OI = start of day; IV =
         each contract's last IV before t; flow = volume from 09:30 to t.
EVAL     t = 10:00, 10:30, ... 15:30. Gamma at spot G_t = dealer $ delta per 1%
         move (Black-Scholes, r = 0), per mode.
TARGET   log( realized 30-minute range after t / implied 30-minute move ), the
         ETF's 1m high-low over (t, t+30], implied = ATM 0-1DTE IV x sqrt(30 min).
         Negative gamma should go with bigger moves: the predicted IC is NEGATIVE.
NORMALISE  G / trailing-20-session median |G| at the same time of day, per market and
         mode (sizes grew over the years). Spearman IC pooled over days x times.
WINDOW   scored 2024-08-20 .. 2026-09-18, halves split 2025-08-21; pre-sample reported.

PRE-REGISTERED
  1. each mode's IC (S&P, QQQ, IWM) with a day-block bootstrap 95% CI.
  2. A FLOW mode replaces oi as the map's default only if, on S&P, its IC is more
     negative than oi's with the paired day-block CI of (IC_flow - IC_oi) below 0,
     the difference has the same sign in both halves, and in at least one of QQQ /
     IWM. Otherwise oi stays the default and the flow modes are labelled "what if".
  Reported: the same by time of day; the sign split (G > 0 vs G < 0) of the target.

Usage:  python check_gamma_modes.py --build   # per-day features -> _gmodes_cache/ (~15 min)
        python check_gamma_modes.py

RUN LOG
  2026-10-25  23,704 obs, 719 days. Median |ask-bid| summed / OI at the evals: SP 0.31, QQQ 0.32, IWM 0.21.
    scored (507 days)      oi       raw      single   raw-oi [95% CI]            single-oi [95% CI]
      SP               -0.166   -0.204   -0.207   -0.038 [-0.060,-0.016]     -0.041 [-0.062,-0.019]
      QQQ              -0.165   -0.201   -0.190   -0.036 [-0.066,-0.007]     -0.025 [-0.055,+0.005]
      IWM              -0.174   -0.166   -0.159   +0.008 [-0.020,+0.034]     +0.015 [-0.012,+0.042]
    S&P halves raw-oi -0.007 / -0.043, single-oi -0.027 / -0.031.
    VERDICT: PASS for both flow modes -> default = single (the more negative S&P difference; vs raw a tie).
    Reported: flow's gain grows through the day (S&P 15:00: oi -0.215, single -0.302; 10:00 all ~-0.13).
    Range vs implied: S&P single-mode +gamma +0.497 vs -gamma +0.732 log points (~26% wider ranges).
    CAVEATS (seen after the verdict, not part of the rule):
      * the absolute IC is regime-heavy: S&P oi by quarter runs +0.095 (2024Q1) .. -0.411 (2025Q1); the
        labelled pre-sample is ~0 for every mode (SP oi -0.025, raw -0.013, single -0.012).
      * raw < oi in 10/12 S&P quarters but only 4/12 QQQ and 4/12 IWM quarters: the flow gain is an S&P result.
      * within-day (each day's mean removed) scored ICs: SP oi -0.023, raw -0.090, single -0.064; QQQ
        -0.021 / -0.101 / -0.067; IWM -0.005 / +0.028 / +0.046. Pre-sample within-day ~0 or wrong-signed.
    The map's gamma lens is context for move SIZE; nothing here is a direction or entry signal.
  2026-10-25 (2)  rebuilt with a 4th mode, "fit" (check_flow_oi stage 2): the oi / raw / single numbers above
    reproduce exactly. fit vs single: SP +0.047 [+0.021,+0.072], QQQ +0.032 [+0.001,+0.065], IWM -0.027
    [-0.064,+0.008] -> FAIL, single stays the default (details in check_flow_oi.py).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import ndtr
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
LAKE = "lake/silver/option-contracts-1m"
CACHE = "_gmodes_cache"
MARKETS = {"SP": ("SPY", ("SPY", "SPXW", "SPX")), "QQQ": ("QQQ", ("QQQ",)), "IWM": ("IWM", ("IWM",))}
EVALS = list(range(600, 931, 30))
MODES = ("oi", "raw", "single")
DEP_START, SPLIT = dt.date(2024, 8, 20), dt.date(2025, 8, 21)
YEAR_MIN = 365.0 * 1440.0
N_BOOT = 2000
VOLS = ["ask_volume", "bid_volume", "multi_volume", "mid_volume", "no_side_volume"]
VOLS_FIT = ["ask_volume", "bid_volume", "mid_volume", "no_side_volume", "multi_volume"]   # check_flow_oi's X order
try:
    FIT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_flowoi_cache", "coef.json")))["coef"]
except (OSError, ValueError, KeyError):
    FIT = {}
rng = np.random.default_rng(20261025)


def etf_bars(t):
    d = (pl.read_parquet(f"historical/{t}.parquet", columns=["date", "minute_et", "high", "low", "close"])
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959)).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return {day: g.set_index("mod")[["high", "low", "close"]] for day, g in d.groupby("date")}


def gamma(S, K, T, iv):
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sd
    return np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sd)


def build():
    os.makedirs(CACHE, exist_ok=True)
    bars = {m: etf_bars(etf) for m, (etf, _) in MARKETS.items()}
    roots = sorted({r for _, rs in MARKETS.values() for r in rs})
    out = []
    files = sorted(glob.glob(f"{LAKE}/date=*/bars.parquet"))
    for n, f in enumerate(files):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        d = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(roots))
             .filter(pl.col("expiry") <= pl.lit(day + dt.timedelta(days=3)))
             .select("underlying_symbol", "option_chain_id", "option_type", "strike", "expiry", "minute_et",
                     "open_interest", "iv_close", "underlying_close", "ask_volume", "bid_volume", "multi_volume",
                     "mid_volume", "no_side_volume")
             .collect())
        if d.is_empty():
            continue
        d = d.with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                            + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"),
                           (pl.col("expiry").cast(pl.Date) - pl.lit(day)).dt.total_days().alias("dte"))
        d = d.filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959)).to_pandas()
        # 0-1 DTE = today's expiry and the next listed one
        for mkt, (etf, rs) in MARKETS.items():
            b = bars[mkt].get(day)
            if b is None or len(b) < 380:
                continue
            x = d[d["underlying_symbol"].isin(rs)]
            if x.empty:
                continue
            exps = sorted(x["dte"].unique())[:2]
            x = x[x["dte"].isin(exps) & ~((x["underlying_symbol"] == "SPX") & (x["dte"] == 0))]
            meta = x.groupby("option_chain_id").agg(root=("underlying_symbol", "first"), cp=("option_type", "first"),
                                                    k=("strike", "first"), dte=("dte", "first"),
                                                    oi=("open_interest", "first")).dropna()
            meta["k"] = meta["k"].astype(float)
            for t in EVALS:
                if t + 30 > 959:
                    continue
                w = b.loc[(b.index > t) & (b.index <= t + 30)]
                if len(w) < 20:
                    continue
                S_etf = float(b["close"].get(t - 1, np.nan))
                if not np.isfinite(S_etf):
                    continue
                past = x[x["mod"] < t]
                if past.empty:
                    continue
                fl = past.groupby("option_chain_id")[VOLS].sum()
                ivl = past.dropna(subset=["iv_close"]).sort_values("mod").groupby("option_chain_id")["iv_close"].last()
                ul = past.dropna(subset=["underlying_close"]).groupby("underlying_symbol")["underlying_close"].apply(
                    lambda s: float(s.iloc[-20:].median()))
                m = meta.join(fl, how="left").join(ivl.rename("iv"), how="left").fillna({v: 0 for v in VOLS})
                m = m[(m["iv"] > 0.02) & (m["iv"] < 5)]
                m["S"] = m["root"].map(ul)
                m = m.dropna(subset=["S"])
                m = m[np.abs(m["k"] / m["S"] - 1) <= 0.05]
                if m.empty:
                    continue
                close_min = m["dte"].to_numpy() * 1440 + np.where(m["root"] == "SPX", 570, 960)
                T = np.maximum(close_min - t, 1.0) / YEAR_MIN
                g1 = gamma(m["S"].to_numpy(), m["k"].to_numpy(), T, m["iv"].to_numpy()) * m["S"].to_numpy() ** 2 * 0.01 * 100
                conv = np.where(m["cp"] == "call", 1.0, -1.0) * m["oi"].to_numpy()
                net = (m["ask_volume"] - m["bid_volume"]).to_numpy()
                net1 = net + 0.12 * m["multi_volume"].to_numpy()
                # stage 2 of check_flow_oi: convention applied to OI + the fitted next-day dOI of the flow so far
                pred = np.zeros(len(m))
                if FIT:
                    bk = (mkt + "|" + m["cp"] + "|" + np.where(m["dte"] <= 7, np.where(m["dte"] <= 1, "1", "2-7"), "8-45")
                          + "|" + np.where(np.abs(m["k"] / m["S"] - 1) <= 0.01, "atm",
                                           np.where(np.abs(m["k"] / m["S"] - 1) <= 0.03, "near", "far"))).to_numpy()
                    A = m[VOLS_FIT].to_numpy(float)
                    for i, bkt in enumerate(bk):
                        c = FIT.get(bkt)
                        if c:
                            pred[i] = A[i] @ c
                conv_fit = np.where(m["cp"] == "call", 1.0, -1.0) * np.maximum(m["oi"].to_numpy() + pred, 0.0)
                atm = m.iloc[np.argsort(np.abs(m["k"] / m["S"] - 1).to_numpy())[:6]]
                iv_atm = float(atm["iv"].median())
                Tn = max(float(atm["dte"].min()) * 1440 + 960 - t, 30.0)
                rng_ = (w["high"].max() - w["low"].min()) / S_etf
                implied = iv_atm * math.sqrt(30.0 / YEAR_MIN)
                out.append(dict(day=day, mkt=mkt, t=t, g_oi=float(np.sum(conv * g1)), g_raw=float(np.sum((conv - net) * g1)),
                                g_single=float(np.sum((conv - net1) * g1)), g_fit=float(np.sum(conv_fit * g1)), target=math.log(max(rng_, 1e-6) / implied),
                                vol=float(m["ask_volume"].sum() + m["bid_volume"].sum()), oi=float(m["oi"].sum()),
                                net=float(np.abs(net).sum())))
        if n % 50 == 0:
            print(f"  {day} ({n + 1}/{len(files)}) rows {len(out)}", flush=True)
    pd.DataFrame(out).to_parquet(f"{CACHE}/features.parquet")
    print("  built", len(out), "rows")


def spear(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def boot_ic(df, cols):
    """paired day-block bootstrap of Spearman ICs for `cols` vs target."""
    days = np.array(sorted(df["day"].unique()))
    by = {d: g for d, g in df.groupby("day")}
    res = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days))
        s = pd.concat([by[d] for d in pick])
        res.append([spear(s[c].to_numpy(float), s["target"].to_numpy(float)) for c in cols])
    return np.array(res)


def main():
    F = pd.read_parquet(f"{CACHE}/features.parquet")
    F["day"] = pd.to_datetime(F["day"]).dt.date
    for mode in MODES:
        col = f"g_{mode}"
        F = F.sort_values(["mkt", "t", "day"])
        F[f"z_{mode}"] = F.groupby(["mkt", "t"])[col].transform(
            lambda s: s / s.abs().shift(1).rolling(20, min_periods=10).median())
    print(f"  rows {len(F)}  days {F['day'].nunique()}  ({F['day'].min()} .. {F['day'].max()})")
    print("  flow vs open interest at the evals (0-1 DTE, +-5%): median |ask-bid| summed / OI:",
          "  ".join(f"{m} {(F[F.mkt == m]['net'] / F[F.mkt == m]['oi']).median():.2f}" for m in MARKETS))
    S = F[F["day"] >= DEP_START]
    print(f"\n  ======== PRIMARY (scored {DEP_START} ..) : IC of normalised dealer gamma at spot vs log(realized/implied 30-min range) ========")
    verdict = {}
    for mkt in MARKETS:
        s = S[S["mkt"] == mkt].dropna(subset=[f"z_{m}" for m in MODES] + ["target"])
        cols = [f"z_{m}" for m in MODES]
        ic = [spear(s[c].to_numpy(float), s["target"].to_numpy(float)) for c in cols]
        B = boot_ic(s, cols)
        halves = [[spear(h[c].to_numpy(float), h["target"].to_numpy(float)) for c in cols]
                  for h in (s[s["day"] < SPLIT], s[s["day"] >= SPLIT])]
        print(f"\n  {mkt}  n={len(s)} obs, {s['day'].nunique()} days")
        for i, m in enumerate(MODES):
            lo, hi = np.nanquantile(B[:, i], [0.025, 0.975])
            print(f"     {m:7} IC {ic[i]:+.3f}  [{lo:+.3f}, {hi:+.3f}]   halves {halves[0][i]:+.3f} / {halves[1][i]:+.3f}")
        for i, m in enumerate(MODES[1:], 1):
            d = B[:, i] - B[:, 0]
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            dh = [halves[0][i] - halves[0][0], halves[1][i] - halves[1][0]]
            verdict[(mkt, m)] = dict(d=ic[i] - ic[0], lo=lo, hi=hi, halves=dh)
            print(f"     {m:7} - oi  {ic[i] - ic[0]:+.3f}  [{lo:+.3f}, {hi:+.3f}]   halves {dh[0]:+.3f} / {dh[1]:+.3f}")
    print("\n  ======== VERDICT ========")
    winner = "oi"
    for m in MODES[1:]:
        v = verdict[("SP", m)]
        ok = (v["d"] < 0 and v["hi"] < 0 and all(x < 0 for x in v["halves"])
              and any(verdict[(o, m)]["d"] < 0 for o in ("QQQ", "IWM")))
        print(f"     {m}: {'BEATS oi' if ok else 'does not beat oi'} on the pre-registered rule")
        if ok and (winner == "oi" or verdict[("SP", m)]["d"] < verdict[("SP", winner)]["d"]):
            winner = m
    print(f"     default for the map: {winner}")

    print("\n  REPORTED: IC by time of day (scored, S&P)")
    s = S[S["mkt"] == "SP"]
    for t, g in s.groupby("t"):
        print(f"     {t // 60}:{t % 60:02d}  " + "  ".join(f"{m} {spear(g[f'z_{m}'].to_numpy(float), g['target'].to_numpy(float)):+.3f}" for m in MODES))
    print("\n  REPORTED: mean log(realized/implied) by gamma sign at spot (scored)")
    for mkt in MARKETS:
        s = S[S["mkt"] == mkt]
        print(f"     {mkt:4} " + "   ".join(f"{m}: +γ {s.loc[s[f'g_{m}'] > 0, 'target'].mean():+.3f} (n {int((s[f'g_{m}'] > 0).sum())}) / −γ {s.loc[s[f'g_{m}'] <= 0, 'target'].mean():+.3f}" for m in MODES))
    P = F[F["day"] < DEP_START]
    print("\n  PRE-SAMPLE (labelled) ICs: " + "   ".join(
        f"{mkt} " + " ".join(f"{m} {spear(P[P.mkt == mkt][f'z_{m}'].to_numpy(float), P[P.mkt == mkt]['target'].to_numpy(float)):+.3f}" for m in MODES)
        for mkt in MARKETS))

    if "g_fit" not in F:
        return
    # ---- STAGE 2 of check_flow_oi: fitted OI (convention) vs the current default, single-leg flow
    try:
        passed = json.load(open("_flowoi_cache/coef.json"))["passed"]
    except (OSError, ValueError, KeyError):
        passed = []
    F = F.sort_values(["mkt", "t", "day"])
    F["z_fit"] = F.groupby(["mkt", "t"])["g_fit"].transform(lambda s: s / s.abs().shift(1).rolling(20, min_periods=10).median())
    S = F[F["day"] >= DEP_START]
    print(f"\n  ======== STAGE 2 (check_flow_oi): fit vs single; stage 1 passed for {passed or 'none'} ========")
    v2 = {}
    for mkt in MARKETS:
        s = S[S["mkt"] == mkt].dropna(subset=["z_fit", "z_single", "z_oi", "target"])
        cols = ["z_fit", "z_single"]
        ic = [spear(s[c].to_numpy(float), s["target"].to_numpy(float)) for c in cols]
        B = boot_ic(s, cols)
        lo, hi = np.nanquantile(B[:, 0] - B[:, 1], [0.025, 0.975])
        dh = [spear(h["z_fit"].to_numpy(float), h["target"].to_numpy(float)) - spear(h["z_single"].to_numpy(float), h["target"].to_numpy(float))
              for h in (s[s["day"] < SPLIT], s[s["day"] >= SPLIT])]
        v2[mkt] = dict(d=ic[0] - ic[1], hi=hi, halves=dh)
        print(f"     {mkt:4} fit IC {ic[0]:+.3f}  single {ic[1]:+.3f}   fit - single {ic[0] - ic[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"   halves {dh[0]:+.3f} / {dh[1]:+.3f}   (oi {spear(s['z_oi'].to_numpy(float), s['target'].to_numpy(float)):+.3f})")
    v = v2["SP"]
    ok = ("SP" in passed and v["d"] < 0 and v["hi"] < 0 and all(x < 0 for x in v["halves"])
          and any(v2[o]["d"] < 0 for o in ("QQQ", "IWM") if o in passed))
    print(f"     VERDICT: fit {'REPLACES single as the default' if ok else 'does not replace single'}")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        main()
