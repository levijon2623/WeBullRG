# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy>=1.11"]
# ///
"""
check_currents.py
=================
DO DEALER HEDGING "CURRENTS" (CHARM, VANNA) PREDICT THE DAY'S DIRECTION?
(user, 2026-10-24, first step of the dealer "vector field" map idea)

The map idea: price moves through a field where dealer GAMMA is the fog
(resistance) and the hedge flows forced by CHARM (time passing) and VANNA (IV
changing) are the wind. Peak gamma as a magnet is already busted
(check_magnet, check_peak_gamma) and gamma only sized moves (check_swing_
factors, did not replicate). The untested, falsifiable part is the WIND: from
the open, do the hedge flows that charm and vanna imply point the way the
index then goes?

POSITIONS  start-of-day open interest (known pre-open), expiries 0-45 DTE,
           strikes within 15% of spot. Roots per market:
             S&P      SPY + SPX + SPXW       (PRIMARY)
             Nasdaq   QQQ                    (NDX's own price is often missing)
             Russell  IWM + RUT + RUTW
           IV = each contract's PRIOR-day closing IV (no same-day look-ahead).
           Spot = the root's own price at 09:35 (and 15:00). AM-settled
           monthlies (SPX, RUT roots) expiring today are dropped.
DEALER SIDE  the usual GEX convention: dealers long the calls, short the puts
           (customers sell calls, buy puts). It is an assumption -- so the
           direction test is TWO-SIDED; a strong NEGATIVE IC means the
           convention is backwards, and is reported as such.
CURRENTS (Black-Scholes deltas, r = 0; $ of underlying dealers must trade)
  charm   time passes from 09:35 to 15:55 at the 09:35 spot:
          flow = -sum(sign x OI x 100 x (delta(T_1555) - delta(T_0935))) x S
  vanna   every IV 1 vol point lower (IV usually bleeds intraday), at 09:35:
          flow = -sum(sign x OI x 100 x (delta(iv - 0.01) - delta(iv))) x S
  both    charm + vanna
  (+ = dealers must BUY the underlying = an upward current)
  The same three at 15:00 (charm to 15:55, the 0DTE crunch) for the late window.
FOG (reported) dealer $ delta per 1% move at spot = sum(sign x OI x 100 x gamma x S^2 x 0.01).
TARGETS  R_day = log(SPY|QQQ|IWM at 15:55 / at 09:35); R_late = 15:00 -> 15:55.
NORMALIZE each current / trailing-60-session mean |current| (sizes grew over
           the years); Spearman IC over days.

PRE-REGISTERED
  6 cells per market (3 currents x 2 windows). Null: returns permuted across
  days, 5,000x; per-cell z; family-wise p95 of max |z| over the 6 cells.
  PASS (S&P) = |z| > family-wise p95, the same sign in both halves (split
  2025-08-21), and the same sign in at least one of Nasdaq / Russell.
  Scored 2024-08-20 onward; 2023-10-12 .. 2024-08-19 reported, labelled.
  Reported, unscored: quintiles (bp), the fog split (+ / - gamma at spot),
  |R_day| vs fog, per-market ICs.
  MACHINERY  M1 a planted current (R_day + noise, IC ~0.12) must pass; M0 an
             independent random current must not.

Usage:  python check_currents.py --build     # once: _currents_cache/ from the lake
        python check_currents.py

RUN LOG
  2026-10-24 build: 9.44M contract-days, 726 sessions (7 roots).
  2026-10-24 run 1: scored sessions S&P 510 / Nasdaq 509 / Russell 506.
    Median $ flows at 09:35: S&P charm +4.4B, vanna(-1 vol) +13.7B, fog +4.5B
    per 1% (fog > 0 on 53% of days); Russell fog > 0 on only 25%.
    M1 at noise 8x (planted IC 0.096, z 2.17) FAILED vs p95 2.58 -- below the
    power floor, not a bug: n ~510 detects |IC| >= ~0.11-0.12 family-wise.
    Re-run with noise 5x (IC 0.15-0.20, z 3.5-4.6): M1 PASS, M0 PASS (logged;
    the plant level was changed after seeing the first run).
    PRIMARY: no cell passes. S&P ICs: charm am -0.029, pm +0.035; vanna am
      -0.060, pm -0.029; both am -0.068, pm -0.021 (|z| <= 1.52 vs 2.54).
      Nasdaq and Russell the same size (|IC| <= 0.076, |z| <= 1.69). Most
      signs NEGATIVE (the "wrong" way for the convention) but inside noise;
      pre-sample signs mixed.
    VERDICT: FAIL. Neither charm nor vanna hedge flow (nor both) predicts the
      day's or the last hour's direction, in any of the three markets, at
      any size this sample can see (|IC| >= ~0.11).
    Reported: the FOG works for SIZE, as known -- S&P |R_day| 39 bp on +gamma
      days vs 68 bp on -gamma days (Nasdaq 55 vs 89, Russell 64 vs 82).
      Curiosity: the lowest 'both' quintile (strongest modelled dealer
      SELLING) had the best days in all three (+18.5 / +20.8 / +21.1 bp),
      opposite the current theory -- consistent with heavy-put / fear days
      drifting up (the skew lead), unscored.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import ndtr
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

LAKE = "lake/silver/option-contracts-1m"
CACHE = "_currents_cache"
MARKETS = {"S&P": ("SPY", ("SPY", "SPX", "SPXW")), "Nasdaq": ("QQQ", ("QQQ",)),
           "Russell": ("IWM", ("IWM", "RUT", "RUTW"))}
ROOTS = sorted({r for _, rs in MARKETS.values() for r in rs})
AM_SETTLED = {"SPX", "RUT"}
MAX_DTE, BAND = 45, 0.15
T0, T1, TC = 575, 900, 955                     # 09:35, 15:00, 15:55
DEP_START, SPLIT = dt.date(2024, 8, 20), dt.date(2025, 8, 21)
N_PERM, SEED = 5000, 20261024
YEAR_MIN = 365.0 * 1440.0


# ------------------------------------------------------------------ build
def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(glob.glob(f"{LAKE}/date=*/bars.parquet"))
    pos, spots = [], []
    for i, f in enumerate(days):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        d = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(ROOTS))
             .filter(pl.col("expiry") <= pl.lit(day + dt.timedelta(days=MAX_DTE + 1)))
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
        pos.append(c)
        sp = {"date": day}
        for root in ROOTS:
            r = d.filter(pl.col("underlying_symbol") == root)
            for m in (T0, T1, TC):
                v = r.filter((pl.col("mod") >= m - 1) & (pl.col("mod") <= m + 1))["underlying_close"].drop_nulls()
                sp[f"{root}_{m}"] = float(v.median()) if len(v) else None
        spots.append(sp)
        if i % 100 == 0:
            print(f"  {day} ({i + 1}/{len(days)})", flush=True)
    pl.concat(pos, how="vertical_relaxed").write_parquet(f"{CACHE}/positions.parquet")
    pd.DataFrame(spots).to_parquet(f"{CACHE}/spots.parquet")
    print("  built", sum(p.height for p in pos), "contract-days,", len(spots), "sessions")


# ------------------------------------------------------------------ greeks
def bs_delta(S, K, T, iv, call):
    T = np.maximum(T, 1e-9)
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sd
    nd = ndtr(d1)
    return np.where(call, nd, nd - 1.0)


def bs_gamma(S, K, T, iv):
    T = np.maximum(T, 1e-9)
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sd
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * sd)


def ul_px(t):
    d = (pl.read_parquet(f"historical/{t}.parquet", columns=["date", "minute_et", "open", "close"])
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter(pl.col("mod").is_in([T0, T1, TC - 1])).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    o = d.pivot_table(index="date", columns="mod", values="open")
    c = d.pivot_table(index="date", columns="mod", values="close")
    out = pd.DataFrame({"p0": o.get(T0), "p1": o.get(T1), "pc": c.get(TC - 1)})   # price at 09:35, 15:00, 15:55
    return out.dropna()


def currents():
    P = pl.read_parquet(f"{CACHE}/positions.parquet").to_pandas()
    S = pd.read_parquet(f"{CACHE}/spots.parquet").set_index("date")
    P["date"] = pd.to_datetime(P["date"]).dt.date
    P["exp"] = pd.to_datetime(P["exp"]).dt.date
    days = sorted(P["date"].unique())
    prev = dict(zip(days[1:], days[:-1]))
    iv_prev = P[["option_chain_id", "date", "iv_eod"]].assign(date=lambda x: x["date"].map(dict(zip(days[:-1], days[1:]))))
    P = P.merge(iv_prev.rename(columns={"iv_eod": "iv0"}), on=["option_chain_id", "date"], how="left")
    P = P[(P["oi"] > 0) & (P["iv0"] > 0.02) & (P["iv0"] < 3)]
    rows = []
    for (day, root), g in P.groupby(["date", "root"]):
        if day not in prev:
            continue
        mkt = next(m for m, (_, rs) in MARKETS.items() if root in rs)
        g = g[~((g["exp"] == day) & (root in AM_SETTLED))]
        dte_min = np.array([(e - day).days * 1440 + (960 - 0) for e in g["exp"]], float)   # to 16:00 on expiry
        k, oi, iv = g["k"].to_numpy(float), g["oi"].to_numpy(float), g["iv0"].to_numpy(float)
        call = (g["typ"] == "call").to_numpy()
        sgn = np.where(call, 1.0, -1.0)
        rec = {"date": day, "mkt": mkt, "root": root}
        for lab, t_now in (("am", T0), ("pm", T1)):
            s = S.at[day, f"{root}_{t_now}"] if day in S.index else None
            if s is None or not np.isfinite(s):
                continue
            keep = (np.abs(k / s - 1) <= BAND) & (dte_min - t_now > 0)
            if not keep.any():
                continue
            kk, oo, vv, cc, ss = k[keep], oi[keep], iv[keep], call[keep], sgn[keep]
            T_now = (dte_min[keep] - t_now) / YEAR_MIN
            T_end = np.maximum(dte_min[keep] - TC, 5) / YEAR_MIN
            d_now = bs_delta(s, kk, T_now, vv, cc)
            d_end = bs_delta(s, kk, T_end, vv, cc)
            d_vol = bs_delta(s, kk, T_now, np.maximum(vv - 0.01, 0.01), cc)
            w = ss * oo * 100.0
            rec[f"charm_{lab}"] = -np.sum(w * (d_end - d_now)) * s
            rec[f"vanna_{lab}"] = -np.sum(w * (d_vol - d_now)) * s
            if lab == "am":
                rec["fog"] = np.sum(w * bs_gamma(s, kk, T_now, vv)) * s * s * 0.01
        rows.append(rec)
    C = pd.DataFrame(rows)
    agg = C.groupby(["date", "mkt"]).sum(numeric_only=True, min_count=1).reset_index()
    for lab in ("am", "pm"):
        agg[f"both_{lab}"] = agg[f"charm_{lab}"] + agg[f"vanna_{lab}"]
    return agg


# ------------------------------------------------------------------ stats
def spear(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 40:
        return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


CELLS = [(c, w) for c in ("charm", "vanna", "both") for w in ("am", "pm")]


def panel(agg):
    out = {}
    for mkt, (etf, _) in MARKETS.items():
        a = agg[agg["mkt"] == mkt].set_index("date").sort_index()
        px = ul_px(etf)
        a = a.join(px, how="inner")
        a["R_am"] = np.log(a["pc"] / a["p0"])
        a["R_pm"] = np.log(a["pc"] / a["p1"])
        for c, w in CELLS:
            x = a[f"{c}_{w}"]
            a[f"z_{c}_{w}"] = x / x.abs().shift(1).rolling(60, min_periods=30).mean()
        a["fogz"] = a["fog"] / a["fog"].abs().shift(1).rolling(60, min_periods=30).mean()
        out[mkt] = a
    return out


def score(Pn, rng, title, plant=False):
    res = {}
    print(f"\n  {title}")
    for mkt, a in Pn.items():
        s = a[a.index >= DEP_START].copy()
        if plant:
            s["z_charm_am"] = s["R_am"] + rng.normal(0, s["R_am"].std() * 5.0, len(s))
            s["z_vanna_am"] = rng.normal(size=len(s))
        dates = np.array(s.index)
        real, null = [], np.empty((N_PERM, len(CELLS)))
        X = [s[f"z_{c}_{w}"].to_numpy(float) for c, w in CELLS]
        Y = [s[f"R_{w}"].to_numpy(float) for _, w in CELLS]
        real = np.array([spear(x, y) for x, y in zip(X, Y)])
        for p in range(N_PERM):
            perm = rng.permutation(len(s))
            null[p] = [spear(x, y[perm]) for x, y in zip(X, Y)]
        mu, sd = null.mean(0), null.std(0)
        z = (real - mu) / sd
        p95 = float(np.quantile(np.abs((null - mu) / sd).max(1), 0.95))
        halves = [[spear(x[dates < SPLIT], y[dates < SPLIT]), spear(x[dates >= SPLIT], y[dates >= SPLIT])]
                  for x, y in zip(X, Y)]
        res[mkt] = dict(ic=real, z=z, p95=p95, halves=halves, n=len(s))
    for mkt, r in res.items():
        print(f"     {mkt:8} n={r['n']}  family-wise p95 |z| {r['p95']:.2f}")
        for j, (c, w) in enumerate(CELLS):
            h = r["halves"][j]
            print(f"        {c:6} {w}   IC {r['ic'][j]:+.3f}   z {r['z'][j]:+5.2f}   halves {h[0]:+.3f} / {h[1]:+.3f}")
    hits = []
    sp = res["S&P"]
    for j, cw in enumerate(CELLS):
        ic = sp["ic"][j]
        ok = (abs(sp["z"][j]) > sp["p95"] and all(np.sign(h) == np.sign(ic) for h in sp["halves"][j])
              and any(np.sign(res[m]["ic"][j]) == np.sign(ic) for m in ("Nasdaq", "Russell")))
        if ok:
            hits.append(cw)
    return hits, res


def main():
    rng = np.random.default_rng(SEED)
    agg = currents()
    Pn = panel(agg)
    for mkt, a in Pn.items():
        s = a[a.index >= DEP_START]
        print(f"  {mkt:8} sessions {len(s)}  median $ flow  charm_am {s['charm_am'].median() / 1e9:+.2f}B"
              f"  vanna_am {s['vanna_am'].median() / 1e9:+.2f}B  fog {s['fog'].median() / 1e9:+.2f}B per 1%"
              f"  (fog > 0 on {np.mean(s['fog'] > 0):.0%} of days)")
    h1, _ = score(Pn, rng, "MACHINERY: charm_am replaced by R_day + noise, vanna_am by random", plant=True)
    print(f"  M1 {'PASS' if ('charm', 'am') in h1 else 'FAIL'}   M0 {'PASS' if ('vanna', 'am') not in h1 else 'FAIL'}")
    hits, res = score(Pn, rng, "PRIMARY (scored 2024-08-20 onward)")
    print(f"\n  VERDICT: {('PASS ' + str(hits)) if hits else 'FAIL -- no current predicts direction'}")

    print("\n  REPORTED (unscored)")
    for mkt, a in Pn.items():
        s = a[a.index >= DEP_START].dropna(subset=["z_both_am", "R_am"])
        q = pd.qcut(s["z_both_am"].rank(method="first"), 5, labels=False)
        print(f"     {mkt:8} R_day by 'both' quintile (bp): " + "  ".join(f"{s.loc[q == i, 'R_am'].mean() * 1e4:+5.1f}" for i in range(5)))
        f = s.dropna(subset=["fog"])
        pos, neg = f[f["fog"] > 0], f[f["fog"] <= 0]
        print(f"              |R_day| fog+ {pos['R_am'].abs().mean() * 1e4:5.1f} bp (n {len(pos)})  fog- {neg['R_am'].abs().mean() * 1e4:5.1f} bp (n {len(neg)})"
              f"   IC(both, R) fog+ {spear(pos['z_both_am'].to_numpy(float), pos['R_am'].to_numpy(float)):+.3f}"
              f"  fog- {spear(neg['z_both_am'].to_numpy(float), neg['R_am'].to_numpy(float)):+.3f}")
    print("\n  PRE-SAMPLE (labelled, 2023-10-12 .. 2024-08-19) ICs:")
    for mkt, a in Pn.items():
        s = a[a.index < DEP_START]
        print(f"     {mkt:8} n={len(s)}  " + "  ".join(
            f"{c}_{w} {spear(s[f'z_{c}_{w}'].to_numpy(float), s[f'R_{w}'].to_numpy(float)):+.3f}" for c, w in CELLS))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        main()
