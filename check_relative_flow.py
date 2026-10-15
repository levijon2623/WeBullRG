# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_relative_flow.py
======================
NET PREMIUM FLOW MEASURED LIKE RVOL: AGAINST THE SAME MINUTE OF PAST SESSIONS
(user, 2026-10-15)

"What would cumulative flow look like structured like the 1m RVOL candles --
that minute's flow relative to the last x sessions' same minute? Or 5m, 10m."
Two measures, both against the PRIOR N = 20 sessions (today never included):

  RCUM(T)     today's cumulative net premium from 09:30 to T, ranked against
              the cumulative AT THE SAME T on each of the prior 20 sessions
              (0..1; ties half). 0.95 = more one-way premium by T than 19 of
              the last 20 sessions had by then.
  RFLOW_k(m)  net premium over the k minutes ending at minute m, divided by the
              median |k-minute net premium| at THAT minute over the prior 20
              sessions. Signed. +4 = four times the usual bullish premium for
              that time of day. k = 1, 5, 10.

  net premium = net call premium - net put premium per minute
  (historical/NETPREM{T}.parquet, the quantity the bot accumulates).
  S = the day's 09:35 ATM straddle. SPY, QQQ, IWM. Deployed window
  2024-08-20..2026-08-21 scored, halves split 2025-08-21; the pre-sample
  only warms up the 20-session baselines.

H1  CUMULATIVE. At checkpoints 10:00, 10:30, 11:00, 12:00: days with
    RCUM >= 0.9 (s = +1) or <= 0.1 (s = -1). Outcome s x (16:00 close - price
    at T) / S. Placebo: the same number of observations per checkpoint on
    RANDOM eligible days, carrying the real signs (shuffled) -- so "flow
    days" are compared with random days, drift included. 2,000 draws; the
    MAX over the 4 checkpoints gives a family-wise p95.
H2  INTERVAL. Every minute 09:45..15:20 with |RFLOW_k| >= 3, s = its sign,
    one event per ticker per k per 10 minutes. Outcome s x (close(m+h) -
    close(m)) / S for h = 1, 5, 10, 30 minutes. Placebo: same tickers and
    days, minutes drawn from the real events' own minutes (time of day
    matched), real signs shuffled; family-wise p95 over the 12 (k, h) cells.

PRE-REGISTERED (pooled over the 3 tickers; per cell)
    STAT   mean > family-wise placebo p95, day-block bootstrap 95% CI lower
           > 0, both halves > 0, >= 5 of 6 calendar slices > 0
    TRADE  STAT and mean >= +0.03 S (the scalp floor; an ATM 0DTE moves
           ~0.5 S per 1 S, a round trip costs ~0.01-0.02 S)
    H1 and H2 are scored separately. A pass is a CANDIDATE for stage B (as a
    day filter on the deployed rules), not a strategy.
CONTROLS (reported, unscored) -- does the time-of-day matching add anything?
    H1-SIGN  every day, s = sign of today's cum flow at T (no extremeness).
    H2-RAW   |k-minute flow| >= 3 x the median |k-minute flow| over the prior
             20 sessions at ALL minutes (no time-of-day matching).
MACHINERY
    M1  +0.10 S planted on H1 at 10:30 passes STAT there only.
    M2  +0.05 S planted on H2 (k=5, h=10) passes STAT there only.

RUN LOG
  2026-10-15 run 1: H1 1,756 extreme observations (427-448 per checkpoint,
    ~300 days each); H2 65,834 events (k=1 31,696 / k=5 19,677 / k=10
    14,461), 501 days.
    MACHINERY: M2 PASS (k=5 h=10 only). M1 FAIL -- the planted +0.10 S at
    10:30 did NOT clear the family-wise p95 (+0.105 S): H1 is UNDERPOWERED.
    Close-to-checkpoint moves are ~1 S wide, so with ~430 days per cell the
    smallest detectable effect is ~0.11-0.12 S (from the 10:30 row: CI
    lower -0.109, p95 +0.105). H1's null therefore rules out only LARGE
    effects; a 0.05-0.10 S day-level tilt cannot be excluded.
    H1 FAIL at every checkpoint: 10:00 -0.029, 10:30 -0.001, 11:00 +0.068
    (CI -0.017..+0.149, both halves +, 4/6 slices), 12:00 +0.029 S. The
    11:00 cell is the only consistent one and stays inside the null.
    Sign-only control (no ranking) -0.025..+0.015 S: ranking against the
    same minute did not add anything measurable.
    H2 FAIL in all 12 cells: every mean within +-0.005 S of zero (1m to
    30m), against a 0.03 S floor; best RFLOW_10 -> +30m +0.0048 S (5/6
    slices) under the +0.0056 p95. The un-matched RAW control is the same
    (RAW_1 -> +1m slightly NEGATIVE, -0.001 S, 0/6 slices).
    VERDICT: same-minute relative flow carries no tradeable short-horizon
    direction (H2, well powered). As a day-level filter (H1), no large
    effect; the test cannot see a small one -- 11:00 is a forward-watch
    lead at most.

Usage:  python check_relative_flow.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_implied_breakout_filters as BF             # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
DEP_START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 8, 21)
N_PRIOR = 20
CHECKS = (600, 630, 660, 720)
KS, HS = (1, 5, 10), (1, 5, 10, 30)
RF_THR, GAP, FIRST, LAST = 3.0, 10, 585, 920
EXT_HI, EXT_LO = 0.9, 0.1
FLOOR = 0.03
N_PERM, N_BOOT = 2000, 2000
SEED = 20261015
OPEN = 570


def hm(m):
    return f"{m // 60}:{m % 60:02d}"


def load(tk, am):
    f = pl.read_parquet(f"historical/NETPREM{tk}.parquet").select("minute_et", "net_premium").to_pandas()
    f["minute_et"] = pd.to_datetime(f["minute_et"])
    f["date"] = f["minute_et"].dt.date
    f["mod"] = f["minute_et"].dt.hour * 60 + f["minute_et"].dt.minute
    f = f[(f["mod"] >= OPEN) & (f["mod"] < 960)]
    b = BF.bars_full(tk).to_pandas()
    b["date"] = pd.to_datetime(b["date"]).dt.date
    flows, closes = {}, {}
    for d, g in f.groupby("date"):
        a = np.zeros(390)
        a[(g["mod"] - OPEN).to_numpy()] = pd.to_numeric(g["net_premium"], errors="coerce").fillna(0).to_numpy()
        flows[d] = a
    for d, g in b.groupby("date"):
        c = g.set_index("mod")["close"].reindex(np.arange(OPEN, 960)).ffill().bfill().to_numpy(float)
        if np.isfinite(c).all():
            closes[d] = c
    days = sorted(set(flows) & set(closes))
    S = {d: am[(tk, d)][1] for d in days if (tk, d) in am and am[(tk, d)][1]}
    return days, flows, closes, S


def build(am):
    """-> H1 rows, H2 rows, control rows, and per-ticker arrays for placebos."""
    h1, h1s, h2, h2r, arrays = [], [], [], [], {}
    for tk in TICKERS:
        days, flows, closes, S = load(tk, am)
        F = np.array([flows[d] for d in days])
        C = np.array([closes[d] for d in days])
        cum = np.cumsum(F, axis=1)
        roll = {k: np.concatenate([np.zeros((len(days), 1)), np.cumsum(F, axis=1)], axis=1) for k in KS}
        fk = {k: roll[k][:, k:] - roll[k][:, :-k] for k in KS}        # k-min sum ENDING at bar index i+k-1
        fk = {k: np.concatenate([np.full((len(days), k - 1), np.nan), fk[k]], axis=1) for k in KS}
        arrays[tk] = (days, C, S)
        for i, d in enumerate(days):
            if i < N_PRIOR or d < DEP_START or d > END or d not in S:
                continue
            Sd = S[d]
            prior = slice(i - N_PRIOR, i)
            # ---- H1: cumulative to T = bars 570..T-1, price at T = close of bar T-1
            for T in CHECKS:
                j = T - OPEN - 1
                today, past = cum[i, j], cum[prior, j]
                r = (np.sum(past < today) + 0.5 * np.sum(past == today)) / N_PRIOR
                mv = (C[i, -1] - C[i, j]) / Sd
                if r >= EXT_HI or r <= EXT_LO:
                    s = 1 if r >= EXT_HI else -1
                    h1.append(dict(tk=tk, day=d, T=T, s=s, y=s * mv, mv=mv))
                if today != 0:
                    s = 1 if today > 0 else -1
                    h1s.append(dict(tk=tk, day=d, T=T, s=s, y=s * mv, mv=mv))
            # ---- H2
            for k in KS:
                cur = fk[k][i]
                med = np.nanmedian(np.abs(fk[k][prior]), axis=0)          # same minute, prior sessions
                allmed = np.nanmedian(np.abs(fk[k][prior]))               # all minutes (control)
                for kind, base, out in (("RF", med, h2), ("RAW", allmed, h2r)):
                    last = -999
                    for m in range(FIRST, LAST + 1):
                        j = m - OPEN
                        bm = base[j] if np.ndim(base) else base
                        if not np.isfinite(cur[j]) or not bm or j - last < GAP:
                            continue
                        x = cur[j] / bm
                        if abs(x) < RF_THR:
                            continue
                        last = j
                        s = 1 if x > 0 else -1
                        row = dict(tk=tk, day=d, k=k, m=m, s=s, x=x, di=i)
                        for h in HS:
                            row[f"y{h}"] = s * (C[i, j + h] - C[i, j]) / Sd
                        out.append(row)
    return (pd.DataFrame(h1), pd.DataFrame(h1s), pd.DataFrame(h2), pd.DataFrame(h2r), arrays)


def dboot(days, y, rng):
    ud, inv = np.unique(days, return_inverse=True)
    a = np.bincount(inv, weights=y, minlength=len(ud))
    n = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = [a[j].sum() / n[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def checks(df, col, rng):
    y, days = df[col].to_numpy(float), df["day"].to_numpy()
    lo, hi = dboot(days, y, rng)
    hv = [y[days < SPLIT].mean() if (days < SPLIT).any() else np.nan,
          y[days >= SPLIT].mean() if (days >= SPLIT).any() else np.nan]
    o = np.argsort(days, kind="stable")
    sl = [y[o][ix].mean() for ix in np.array_split(np.arange(len(y)), 6)]
    return y.mean(), lo, hi, hv, sl


def verdict(mean, p95, lo, hv, sl):
    stat = mean > p95 and lo > 0 and all(h > 0 for h in hv) and sum(v > 0 for v in sl) >= 5
    return stat, stat and mean >= FLOOR


# ---------------------------------------------------------------- placebos
def h1_null(df, arrays, rng):
    """Max over checkpoints of the mean outcome on RANDOM eligible days carrying
    the real signs (shuffled)."""
    pools = {}
    for tk, (days, C, S) in arrays.items():
        for i, d in enumerate(days):
            if i >= N_PRIOR and DEP_START <= d <= END and d in S:
                pools.setdefault(tk, []).append((C[i], S[d]))
    mx = np.empty(N_PERM)
    groups = [(T, g) for T, g in df.groupby("T")]
    for p in range(N_PERM):
        best = -9
        for T, g in groups:
            j = T - OPEN - 1
            vals = []
            for tk, gg in g.groupby("tk"):
                pool = pools[tk]
                pick = rng.integers(0, len(pool), len(gg))
                sg = rng.permutation(gg["s"].to_numpy())
                vals += [s * (pool[q][0][-1] - pool[q][0][j]) / pool[q][1] for q, s in zip(pick, sg)]
            best = max(best, float(np.mean(vals)))
        mx[p] = best
    return float(np.quantile(mx, 0.95))


def h2_null(df, arrays, rng, n=N_PERM):
    """Max over (k, h) of the mean outcome at random minutes (from the real
    events' own minutes) on the same days, real signs shuffled."""
    mx = np.empty(n)
    cells = [(k, g) for k, g in df.groupby("k")]
    for p in range(n):
        best = -9
        for k, g in cells:
            outs = {h: [] for h in HS}
            for tk, gg in g.groupby("tk"):
                days, C, S = arrays[tk]
                di = gg["di"].to_numpy()
                mins = rng.permutation(gg["m"].to_numpy()) - OPEN
                sg = rng.permutation(gg["s"].to_numpy())
                Sd = np.array([S[days[i]] for i in di])
                for h in HS:
                    outs[h].append(sg * (C[di, mins + h] - C[di, mins]) / Sd)
            for h in HS:
                best = max(best, float(np.concatenate(outs[h]).mean()))
        mx[p] = best
    return float(np.quantile(mx, 0.95))


# ---------------------------------------------------------------- report
def show(label, df, col, p95, rng, scored=True):
    if df.empty:
        print(f"    {label:22} (none)")
        return False
    mean, lo, hi, hv, sl = checks(df, col, rng)
    stat, trade = verdict(mean, p95, lo, hv, sl)
    tag = (f"STAT {'PASS' if stat else 'fail'}  TRADE {'PASS' if trade else 'fail'}" if scored else "(reported)")
    print(f"    {label:22} n={len(df):>5} ({df['day'].nunique():>3}d)  mean {mean:+.4f} S [{lo:+.4f}, {hi:+.4f}]"
          f"  p95 {p95:+.4f}  halves {hv[0]:+.4f} / {hv[1]:+.4f}  slices>0 {sum(v > 0 for v in sl)}/6  {tag}")
    return stat


def main():
    rng = np.random.default_rng(SEED)
    am = BF.am_straddles()
    h1, h1s, h2, h2r, arrays = build(am)
    print(f"  H1 extreme observations {len(h1)} (sign-only control {len(h1s)});  "
          f"H2 events {len(h2)} (raw control {len(h2r)})")
    print("  H1 by checkpoint: " + "  ".join(f"{hm(T)} {int((h1['T'] == T).sum())}" for T in CHECKS))
    print("  H2 by k: " + "  ".join(f"k={k} {int((h2['k'] == k).sum())}" for k in KS))

    # ---- machinery
    print("\n  ---- MACHINERY ----")
    p1 = h1_null(h1, arrays, rng)
    pl_ = h1.copy()
    pl_.loc[pl_["T"] == 630, "y"] += 0.10
    hits = []
    for T in CHECKS:
        mean, lo, hi, hv, sl = checks(pl_[pl_["T"] == T], "y", rng)
        if verdict(mean, p1, lo, hv, sl)[0]:
            hits.append(T)
    print(f"    M1 plant +0.10 S at 10:30 -> STAT passes at {[hm(T) for T in hits]}  "
          f"{'PASS' if hits == [630] else 'FAIL'}")
    p2 = h2_null(h2, arrays, rng, n=500)
    pl2 = h2.copy()
    pl2.loc[pl2["k"] == 5, "y10"] += 0.05
    hits2 = []
    for k in KS:
        for h in HS:
            g = pl2[pl2["k"] == k]
            mean, lo, hi, hv, sl = checks(g, f"y{h}", rng)
            if verdict(mean, p2, lo, hv, sl)[0]:
                hits2.append((k, h))
    print(f"    M2 plant +0.05 S at k=5 h=10 -> STAT passes at {hits2}  {'PASS' if hits2 == [(5, 10)] else 'FAIL'}")

    # ---- H1
    print(f"\n  ======== H1 -- RCUM >= {EXT_HI} / <= {EXT_LO} at the checkpoint, to the 16:00 close ========")
    print(f"    family-wise placebo p95 = {p1:+.4f} S")
    for T in CHECKS:
        show(f"RCUM at {hm(T)}", h1[h1["T"] == T], "y", p1, rng)
    print("    -- control: sign of today's cum flow, every day (no time-of-day ranking) --")
    for T in CHECKS:
        show(f"SIGN at {hm(T)}", h1s[h1s["T"] == T], "y", p1, rng, scored=False)
    print("    -- RCUM by side (reported) --")
    for T in CHECKS:
        for s, nm in ((1, "high (bullish)"), (-1, "low (bearish)")):
            show(f"{hm(T)} {nm}", h1[(h1["T"] == T) & (h1["s"] == s)], "y", p1, rng, scored=False)

    # ---- H2
    p2 = h2_null(h2, arrays, rng)
    print(f"\n  ======== H2 -- |RFLOW_k| >= {RF_THR:g} (same-minute baseline), next h minutes ========")
    print(f"    family-wise placebo p95 = {p2:+.4f} S")
    for k in KS:
        for h in HS:
            show(f"RFLOW_{k} -> +{h}m", h2[h2["k"] == k], f"y{h}", p2, rng)
    print("    -- control: |k-min flow| >= 3x the all-minute median (no time-of-day matching) --")
    for k in KS:
        for h in HS:
            show(f"RAW_{k} -> +{h}m", h2r[h2r["k"] == k], f"y{h}", p2, rng, scored=False)
    for nm, df in (("h1", h1), ("h1s", h1s), ("h2", h2), ("h2r", h2r)):
        df.to_pickle(os.path.join("_implied_move_cache", f"relflow_{nm}.pkl"))


if __name__ == "__main__":
    main()
