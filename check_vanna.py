"""
check_vanna.py -- DOES VANNA MOVE THE INDEX? SpotGamma's weekly (2026-11-15) as three tests (user, 2026-11-15).
PRE-REGISTERED 2026-11-15, committed before any of V1-V3 was scored.

THE CLAIM. When implied volatility falls, the puts customers bought lose delta; dealers who sold them unwind the stock
they sold as a hedge, and that buying lifts the index (a "vanna rally", e.g. VIX -23% on 2026-11-09). Once implied
vol sits BELOW realized, the fuel is spent and something else must drive the next move.

WHAT IS ALREADY KNOWN (check_currents, 10-24): the vanna flow for an ASSUMED -1 vol point every morning did not
predict the day's direction (S&P IC -0.06). That test ignored whether IV actually moved. The claim here is
CONDITIONAL: vanna matters in proportion to how far IV moves AND how much dealer vanna there is. Spot-vol
correlation (stocks rise as IV falls, whatever dealers hold) makes "IV fell and stocks rose" true by default, so
only the INTERACTION with dealer positioning, or a forward-looking split, tests vanna itself.

V1  MECHANISM (intraday). 30-minute windows 10:00-15:30 (11 a day). x = -(change in implied vol over the window),
    vol points, of a FIXED contract pair chosen at 09:35 (the call and put at the strike nearest spot, expiry nearest
    21 calendar days within 7-45; sticky strike, so the ATM strike moving along the skew is not counted as an IV
    change); y = the underlying's return over the window (bp). VF = the dealer vanna flow at 09:35 from
    check_currents (start-of-day OI, 0-45 DTE, prior-close IV, dealers long calls / short puts: $ of underlying
    dealers must BUY if IV falls one point), divided by its trailing 60-session mean |VF| (VFn).
    Model per market: y = a + b1 x + b2 VFn + b3 x VFn (OLS). Vanna predicts b3 > 0: the market's response to an
    IV drop is bigger when dealers hold more vanna.
    PASS = S&P b3 > 0 with the day-block bootstrap 95% CI above 0 (2,000), b3 > 0 in both halves (split
    2025-08-21), and b3 > 0 in at least one of Nasdaq / Russell.
V2  THE VANNA-RALLY DAY. Days in each market's top decile of fixed-pair IV DROP 09:35 -> 15:55. Spearman(VFn, the
    day's 09:35 -> 15:55 return). PASS = S&P Spearman > 0 with the day-block 95% CI above 0. Power is low
    (~55 days per market) and stated: an inconclusive V2 is a FAIL.
V3  "THE FUEL RUNS LOW" (daily, 2007-01 .. 2026-11, Cboe VIX + SPY closes). Event: VIX down >= 20% over 3
    sessions (close t / close t-3); episodes de-clustered (the next event only >= 10 sessions after the last).
    SPENT = VIX_t below SPY's 21-session realized vol (annualised, %); LEFT = VIX_t at or above it.
    Target: SPY close t -> close t+5 log return. The claim predicts SPENT < LEFT.
    PASS = mean(SPENT) - mean(LEFT) < 0 with the 95% bootstrap CI (episodes resampled, 5,000) below 0, AND the
    same sign in both halves (2007-2016 / 2017-2026). Reported: t+10, each group against all days' 5-day mean.
SAMPLES  V1 / V2 scored 2024-08-20 .. 2026-11-13; 2023-10-12 .. 2024-08-19 (the spent pre-sample) reported,
    labelled. V3 has no holdout: one pass over the whole history.
MACHINERY  M1 a planted y' = y + 3 x VFn (V1 S&P) must pass; M0 VFn shuffled across days must not.
REPORTED, NOT SCORED  V1 with a dealer side inferred from the day's ask-side opening put flow is NOT run (no
    validated side model: check_flow_oi failed); the convention is the assumption, said here.
SpotGamma's own levels (Risk Pivot etc.) are proprietary and not tested.

DEVIATION (2026-11-15, after run 1, logged): run 1's V1 MACHINERY FAILED -- the planted b3 = 3 was not recovered
([-27, +36]) -- because the single fixed 21-day ATM pair has an IV only in minutes when BOTH legs trade (2025-06-12:
SPY 26 of 390 minutes, QQQ 2, IWM 1), leaving 1,319 windows of noisy last-trade IVs (expected ~18,000). Run 1 was
therefore not scorable (its b3s are in the RUN LOG for the record). Run 2 keeps the sticky-strike idea, the model,
the pass rules and the machinery checks, and only changes how the window's IV change is measured: every contract
2-45 days to expiry within 3% of the 09:35 price that has an IV in the 10 minutes before BOTH window ends gives its
own IV change; the window's change is their MEDIAN (>= 5 contracts). Same for V2's 09:35 -> 15:55.

RUN LOG
  2026-11-15 V3: 72 de-clustered VIX-collapse episodes 2007-08..2026-08. SPENT (VIX < RV21) n=27: SPY next 5
    sessions -161 bp (median -8, up 44%); LEFT n=45: +63 bp (median +87, up 69%); all days +17 bp.
    SPENT - LEFT -224 bp [-431, -29], halves -322 / -122 -> PASS. FRAGILE: without 2008-10-20 (-1,630 bp) the gap
    is -168; without the worst three (+ 2020-03-04, 2022-05-04) -85. "Spent" often = VIX fell while realized vol
    was still crash-sized (2008, 2011, 2020, 2022, 2025-04) and the market fell again. 2026-11-10/11 qualify as a
    SPENT episode (VIX -31% in 3 sessions, 19.5 vs RV21 21.7).
  2026-11-15 V1/V2 run 1 (single fixed pair): MACHINERY FAIL (M1 [-27.3, +35.8]); 1,319 windows (S&P 459 days /
    Nasdaq 202 / Russell 81). Not scorable. Record: S&P b3 -13.1 [-30.7, +33.5], Nasdaq +11.9, Russell +79.4;
    V2 S&P n=18 rho -0.003. -> DEVIATION above, run 2.
  2026-11-15 V1/V2 run 2 (many-contract sticky-strike IV change): 18,137 scored windows / 552 sessions (+5,661 pre).
    MACHINERY M1 FAIL again ([-6.0, +20.5]): not a data bug now but POWER -- the CI half-width is ~13 bp per vol pt
    per VFn unit, so only an amplifier of roughly >= 15 (~40% of the base spot-vol slope) is detectable; the planted
    3 (~8%) is not. M0 also excluded 0 ([-28.2, -2.2]): the day-block CIs are not well calibrated here. -> V1 NOT
    SCORABLE as a pass; descriptive: S&P b3 +1.8 [-9.2, +17.6], Nasdaq -4.2, Russell -8.8; bp of index move per vol
    point of IV DROP, low- vs high-vanna days: S&P +37.7 / +40.8, Nasdaq +36.7 / +35.5, Russell +30.5 / +23.4 --
    the spot-vol link is large and the SAME whatever dealers hold. No sign of a vanna amplifier; a big one (> ~40%)
    looks unlikely on S&P.
    V2 (55 days each): S&P rho -0.124 [-0.431, +0.201] -> FAIL; Nasdaq -0.343 [-0.582, -0.063] (the WRONG sign,
    excludes 0); Russell -0.159. On IV-collapse days, more modelled dealer vanna did NOT mean a bigger rally.
    VERDICT: the mechanism (V1/V2) is not supported -- as in check_currents, signs lean against the dealer
    convention. Only V3's VIX-vs-realized split holds, and it is about crash regimes, not dealer positioning.

Usage:  python check_vanna.py --extend   (adds new lake days to _currents_cache)
        python check_vanna.py --v3 | --v1v2
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, ROOT)
import check_currents as CC                             # noqa: E402

LAKE = os.path.join("lake", "silver", "option-contracts-1m")
CACHE = os.path.join("_vanna_cache")
SPLIT = dt.date(2025, 8, 21)
DEP_START, END = dt.date(2024, 8, 20), dt.date(2026, 11, 13)
WINDOWS = [(m, m + 30) for m in range(600, 930, 30)]          # 10:00-10:30 ... 15:00-15:30
ETF = {"S&P": "SPY", "Nasdaq": "QQQ", "Russell": "IWM"}
N_BOOT, SEED = 2000, 20261115


# ------------------------------------------------------------------ V3
def v3():
    px = json.load(open(os.path.join("_cot_cache", "prices_daily.json")))
    spy = pd.Series({r["date"]: r["close"] for r in px["SPY"]}).sort_index()
    vix = pd.Series({r["date"]: r["close"] for r in px["VIX"]}).sort_index()
    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    df.index = pd.to_datetime(df.index)
    lr = np.log(df["spy"]).diff()
    df["rv21"] = lr.rolling(21).std() * np.sqrt(252) * 100
    df["d3"] = df["vix"] / df["vix"].shift(3) - 1
    df["r5"] = np.log(df["spy"].shift(-5) / df["spy"])
    df["r10"] = np.log(df["spy"].shift(-10) / df["spy"])
    df = df.dropna(subset=["rv21", "d3"])
    ev, last = [], -999
    for i, (d, r) in enumerate(df.iterrows()):
        if r["d3"] <= -0.20 and i - last >= 10:
            last = i
            if np.isfinite(r["r5"]):
                ev.append(dict(date=d.date(), vix=r["vix"], rv=r["rv21"], d3=r["d3"], r5=r["r5"], r10=r["r10"],
                               spent=r["vix"] < r["rv21"]))
    E = pd.DataFrame(ev)
    allr5 = df["r5"].mean()
    rng = np.random.default_rng(SEED)
    def diff(e):
        s, l = e[e.spent], e[~e.spent]
        return (s["r5"].mean() - l["r5"].mean()) if len(s) and len(l) else np.nan
    d0 = diff(E)
    bs = []
    for _ in range(5000):
        b = E.sample(len(E), replace=True, random_state=int(rng.integers(1 << 31)))
        bs.append(diff(b))
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    h = [diff(E[pd.to_datetime(E.date).dt.year <= 2016]), diff(E[pd.to_datetime(E.date).dt.year >= 2017])]
    ok = d0 < 0 and hi < 0 and all(x < 0 for x in h)
    print(f"V3 'fuel runs low': {len(E)} VIX-collapse episodes (VIX -20%+ in 3 sessions, de-clustered), "
          f"{E.date.min()}..{E.date.max()}")
    for lab, g in (("SPENT (VIX < RV21)", E[E.spent]), ("LEFT (VIX >= RV21)", E[~E.spent])):
        print(f"  {lab:20s} n={len(g):3d}  SPY next 5 sessions {g.r5.mean() * 1e4:+6.0f} bp (median {g.r5.median() * 1e4:+.0f}), "
              f"next 10 {g.r10.mean() * 1e4:+6.0f} bp, up {np.mean(g.r5 > 0):.0%}")
    print(f"  all days: next 5 sessions {allr5 * 1e4:+.0f} bp")
    print(f"  SPENT - LEFT (5d): {d0 * 1e4:+.0f} bp [{lo * 1e4:+.0f}, {hi * 1e4:+.0f}]  halves {h[0] * 1e4:+.0f} / {h[1] * 1e4:+.0f}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    E.to_csv(os.path.join(CACHE, "v3_episodes.csv"), index=False)


# ------------------------------------------------------------------ cache extension
def extend():
    P = pl.read_parquet(f"{CC.CACHE}/positions.parquet")
    S = pd.read_parquet(f"{CC.CACHE}/spots.parquet")
    have = set(pd.to_datetime(S["date"]).dt.date)
    days = sorted(glob.glob(f"{CC.LAKE}/date=*/bars.parquet"))
    pos, spots = [P], [S]
    n = 0
    for f in days:
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        if day in have:
            continue
        d = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(CC.ROOTS))
             .filter(pl.col("expiry") <= pl.lit(day + dt.timedelta(days=CC.MAX_DTE + 1)))
             .select("underlying_symbol", "option_chain_id", "option_type", "strike", "expiry", "minute_et",
                     "open_interest", "iv_close", "underlying_close").collect())
        if d.is_empty():
            continue
        d = d.with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                            + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
        q = d.filter(pl.col("mod") <= 959).sort("mod")
        c = (q.group_by("option_chain_id").agg(
            pl.col("underlying_symbol").first().alias("root"), pl.col("option_type").first().alias("typ"),
            pl.col("strike").first().cast(pl.Float64).alias("k"), pl.col("expiry").first().alias("exp"),
            pl.col("open_interest").drop_nulls().first().alias("oi"),
            pl.col("iv_close").drop_nulls().last().alias("iv_eod"))
             .with_columns(pl.lit(day).alias("date")))
        pos.append(c.select(P.columns).cast(dict(P.schema)))
        sp = {"date": day}
        for root in CC.ROOTS:
            r = d.filter(pl.col("underlying_symbol") == root)
            for m in (CC.T0, CC.T1, CC.TC):
                v = r.filter((pl.col("mod") >= m - 1) & (pl.col("mod") <= m + 1))["underlying_close"].drop_nulls()
                sp[f"{root}_{m}"] = float(v.median()) if len(v) else None
        spots.append(pd.DataFrame([sp]))
        n += 1
    if n:
        pl.concat(pos, how="vertical_relaxed").write_parquet(f"{CC.CACHE}/positions.parquet")
        pd.concat(spots, ignore_index=True).to_parquet(f"{CC.CACHE}/spots.parquet")
    print(f"extended _currents_cache by {n} sessions")


# ------------------------------------------------------------------ V1 / V2
def vf_table():
    agg = CC.currents()
    agg = agg.sort_values(["mkt", "date"])
    agg["vfn"] = agg.groupby("mkt")["vanna_am"].transform(
        lambda s: s / s.abs().rolling(60, min_periods=40).mean().shift(1))
    return agg.set_index(["date", "mkt"])["vfn"]


def iv_changes(day, f, spans):
    """{market: {(a, b): median per-contract IV change over [a, b], vol fraction}} -- run 2's sticky-strike measure."""
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(ETF.values())))
         .select("underlying_symbol", "option_chain_id", pl.col("strike").cast(pl.Float64), "expiry", "minute_et",
                 "iv_close", "underlying_close").collect())
    x = x.with_columns(mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
    out = {}
    for mkt, tk in ETF.items():
        z = x.filter(pl.col("underlying_symbol") == tk)
        s0 = z.filter((pl.col("mod") >= 574) & (pl.col("mod") <= 576))["underlying_close"].drop_nulls()
        if not len(s0):
            continue
        s0 = float(s0.median())
        z = z.filter((pl.col("expiry") >= pl.lit(day + dt.timedelta(days=2))) & (pl.col("expiry") <= pl.lit(day + dt.timedelta(days=45)))
                     & ((pl.col("strike") / s0 - 1).abs() <= 0.03) & pl.col("iv_close").is_not_null()
                     & (pl.col("iv_close") > 0.02) & (pl.col("iv_close") < 3)).sort("mod")
        series = {}
        for cid, mods, ivs in z.group_by("option_chain_id").agg(pl.col("mod"), pl.col("iv_close")).iter_rows():
            series[cid] = (np.asarray(mods), np.asarray(ivs, float))
        res = {}
        for a, b in spans:
            ch = []
            for mods, ivs in series.values():
                ia = np.searchsorted(mods, a, side="right") - 1
                ib = np.searchsorted(mods, b, side="right") - 1
                if ia >= 0 and ib >= 0 and mods[ia] >= a - 10 and mods[ib] >= b - 10 and ib > ia:
                    ch.append(ivs[ib] - ivs[ia])
            if len(ch) >= 5:
                res[(a, b)] = float(np.median(ch))
        out[mkt] = res
    return out


def iv_paths(day, f):
    """{market: (iv minute series dict, spot minute dict)} for the fixed 09:35 pair."""
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(ETF.values())))
         .select("underlying_symbol", "option_type", pl.col("strike").cast(pl.Float64), "expiry", "minute_et",
                 "iv_close", "underlying_close").collect())
    x = x.with_columns(mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
    out = {}
    for mkt, tk in ETF.items():
        z = x.filter(pl.col("underlying_symbol") == tk)
        s0 = z.filter((pl.col("mod") >= 574) & (pl.col("mod") <= 576))["underlying_close"].drop_nulls()
        if not len(s0):
            continue
        s0 = float(s0.median())
        exps = sorted({e for e in z["expiry"].unique().to_list() if 7 <= (e - day).days <= 45})
        if not exps:
            continue
        e = min(exps, key=lambda e: abs((e - day).days - 21))
        ks = sorted(set(z.filter(pl.col("expiry") == e)["strike"].to_list()))
        k = min(ks, key=lambda k: abs(k - s0))
        pair = z.filter((pl.col("expiry") == e) & (pl.col("strike") == k) & pl.col("iv_close").is_not_null())
        piv = pair.group_by(["mod", "option_type"]).agg(pl.col("iv_close").last())
        cs = {m: v for m, v in piv.filter(pl.col("option_type").str.to_lowercase().str.starts_with("c"))
              .select("mod", "iv_close").iter_rows()}
        ps = {m: v for m, v in piv.filter(pl.col("option_type").str.to_lowercase().str.starts_with("p"))
              .select("mod", "iv_close").iter_rows()}
        iv = {m: (cs[m] + ps[m]) / 2 for m in cs if m in ps and cs[m] and ps[m]}
        out[mkt] = iv
    return out


def near(d, m, tol=2):
    for k in range(tol + 1):
        for mm in (m - k, m + k):
            if mm in d and d[mm] is not None:
                return d[mm]
    return None


def build_panel():
    vf = vf_table()
    px = {}
    for mkt, tk in ETF.items():
        b = pl.read_parquet(f"historical/{tk}.parquet", columns=["date", "minute_et", "close"]).with_columns(
            mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
        for d, m, c in b.select(pl.col("date").cast(pl.Utf8), "mod", "close").iter_rows():
            px.setdefault((mkt, d), {})[m] = c
    rows = []
    for f in sorted(glob.glob(f"{LAKE}/date=*/bars.parquet")):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        if day > END:
            continue
        spans = WINDOWS + [(575, 955)]
        ivc = iv_changes(day, f, spans)                 # run 2 (see DEVIATION)
        for mkt, chg in ivc.items():
            v = vf.get((day, mkt))
            p = px.get((mkt, day.isoformat()))
            if v is None or not np.isfinite(v) or not p:
                continue
            for a, b in spans:
                p0, p1 = near(p, a), near(p, b)
                if (a, b) not in chg or None in (p0, p1) or p0 <= 0:
                    continue
                rows.append(dict(date=day, mkt=mkt, w=f"{a}-{b}", x=-chg[(a, b)] * 100, y=np.log(p1 / p0) * 1e4, vfn=float(v)))
    D = pd.DataFrame(rows)
    os.makedirs(CACHE, exist_ok=True)
    D.to_pickle(os.path.join(CACHE, "panel2.pkl"))
    return D


def b3(d):
    X = np.column_stack([np.ones(len(d)), d.x, d.vfn, d.x * d.vfn])
    beta, *_ = np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)
    return beta[3]


def boot_b3(d, rng):
    days = d.date.unique()
    idx = {k: v.index.to_numpy() for k, v in d.groupby("date")}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days))
        vals.append(b3(d.loc[np.concatenate([idx[k] for k in pick])]))
    return np.quantile(vals, [0.025, 0.975])


def v1v2():
    rng = np.random.default_rng(SEED)
    p = os.path.join(CACHE, "panel2.pkl")
    D = pd.read_pickle(p) if os.path.exists(p) else build_panel()
    W = D[D.w != "575-955"].reset_index(drop=True)
    DAY = D[D.w == "575-955"].reset_index(drop=True)
    sc = W[(W.date >= DEP_START) & (W.date <= END)].reset_index(drop=True)
    pre = W[W.date < DEP_START].reset_index(drop=True)
    print(f"V1 panel: {len(sc)} scored windows ({sc.date.nunique()} sessions), pre-sample {len(pre)} windows")
    # machinery
    s = sc[sc.mkt == "S&P"].reset_index(drop=True)
    m1 = s.assign(y=s.y + 3 * s.x * s.vfn)
    lo, hi = boot_b3(m1, rng)
    perm = s.copy()
    dmap = dict(zip(s.date.unique(), rng.permutation(s.date.unique())))
    vmap = s.groupby("date").vfn.first()
    perm["vfn"] = perm.date.map(lambda d: vmap[dmap[d]])
    l0, h0 = boot_b3(perm, rng)
    print(f"  MACHINERY M1 planted b3 [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if lo > 0 else 'FAIL'};  "
          f"M0 shuffled VFn [{l0:+.3f}, {h0:+.3f}] -> {'PASS' if l0 <= 0 <= h0 else 'FAIL (excludes 0 by chance)'}")
    res = {}
    for mkt in ETF:
        d = sc[sc.mkt == mkt].reset_index(drop=True)
        est = b3(d); lo, hi = boot_b3(d, rng)
        h = [b3(d[d.date < SPLIT]), b3(d[d.date >= SPLIT])]
        pr = pre[pre.mkt == mkt]
        res[mkt] = (est, lo, hi, h)
        # descriptive: slope of y on x by VFn third
        t = d.groupby("date").vfn.first().quantile([1 / 3, 2 / 3]).to_numpy()
        sl = []
        for lab, sel in (("low", d.vfn <= t[0]), ("high", d.vfn > t[1])):
            g = d[sel]; sl.append(np.polyfit(g.x, g.y, 1)[0])
        print(f"  {mkt:8s} b3 {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  halves {h[0]:+.3f} / {h[1]:+.3f}  | bp per vol point of IV drop: "
              f"low-vanna days {sl[0]:+.1f}, high-vanna days {sl[1]:+.1f}  | pre-sample b3 {b3(pr) if len(pr) > 50 else float('nan'):+.3f}")
    e, lo, hi, h = res["S&P"]
    ok = e > 0 and lo > 0 and all(x > 0 for x in h) and (res["Nasdaq"][0] > 0 or res["Russell"][0] > 0)
    print(f"  V1 -> {'PASS' if ok else 'FAIL'}")
    # V2
    print("\nV2 vanna-rally days (top decile of 09:35->15:55 IV drop): Spearman(VFn, day return)")
    from scipy.stats import spearmanr
    for mkt in ETF:
        d = DAY[(DAY.mkt == mkt) & (DAY.date >= DEP_START) & (DAY.date <= END)]
        top = d[d.x >= d.x.quantile(0.9)].reset_index(drop=True)
        rho = spearmanr(top.vfn, top.y)[0]
        bs = []
        for _ in range(N_BOOT):
            b = top.iloc[rng.integers(0, len(top), len(top))]
            bs.append(spearmanr(b.vfn, b.y)[0] if b.vfn.nunique() > 1 and b.y.nunique() > 1 else np.nan)
        lo, hi = np.nanquantile(bs, [0.025, 0.975])
        print(f"  {mkt:8s} n={len(top)} days, IV drop >= {d.x.quantile(0.9):.2f} pts, mean day {top.y.mean():+.0f} bp; "
              f"rho {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (f" -> {'PASS' if lo > 0 else 'FAIL'}" if mkt == "S&P" else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("extend", "v3", "v1v2"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.extend: extend()
    if a.v3: v3()
    if a.v1v2: v1v2()
    if not any(vars(a).values()):
        sys.exit(__doc__)
