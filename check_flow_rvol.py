# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_flow_rvol.py
==================
FLOW SPIKES THAT DO OR DO NOT LINE UP WITH VOLUME AND PRICE (user, 2026-10-15)

check_relative_flow found net-premium spikes (RFLOW_k >= 3x the same-minute
median) carry no direction at 1-30 minutes. The follow-up: "is there anything
meaningful when they align or misalign with the same minutes' RVOL?" Volume
has no sign, so "aligned" is split two ways and both are tested:

  VOLUME   HEAVY  the underlying's k-minute RVOL >= 2 (yellow/red on the
                  viewer) -- the stock is trading hard with the options flow
           QUIET  RVOL_k < 1 (grey) -- options premium without stock volume
  PRICE    CONFIRM  the underlying moved WITH the flow over the same k minutes
           DIVERGE  it moved AGAINST it (flow bullish, price down) -- the
                    "absorption" reading

  RVOL_k(m) = underlying volume over the k minutes ending at m / the median
  of that same k-minute window at that minute over the prior 20 sessions --
  the viewer's RVOL (live_state, RVOL_DAYS = 20) widened to k minutes.
  RFLOW_k as in check_relative_flow. k = 1, 5, 10.

EVENTS   every minute 09:45..15:20 with |RFLOW_k| >= 3 that falls in one of the
         four classes (HEAVY/QUIET x CONFIRM/DIVERGE; RVOL 1-2 and flat price
         are left out), s = the flow's sign, one event per ticker per k per
         10 minutes. SPY, QQQ, IWM; deployed window 2024-08-20..2026-08-21,
         halves split 2025-08-21.
OUTCOME  s x (close(m+h) - close(m)) / S, h = 1, 5, 10, 30 (toward the flow > 0).
         S = the day's 09:35 ATM straddle.
PLACEBO  per class: same tickers and days, minutes drawn from that class's
         own minutes, signs shuffled; 2,000 draws, family-wise p95 = the max
         over all 48 cells (4 classes x 3 k x 4 h).

PRE-REGISTERED (per cell, pooled tickers)
    STAT   mean > family-wise p95, day-block bootstrap CI lower > 0, both
           halves > 0, >= 5 of 6 slices > 0
    TRADE  STAT and mean >= +0.03 S
    A DIVERGE cell passing NEGATIVE (price keeps going against the flow) is
    tested the same way on -outcome and reported as its own line.
    Reported: |move| (magnitude) per class vs its placebo -- does alignment
    say how MUCH, even if not which way.
MACHINERY  M1: +0.05 S planted on QUIET-DIVERGE k=5 h=10 passes there only.

RUN LOG
  2026-10-15 run 1: 65,171 events on 501 days (k=1/5/10; HEAVY-CONFIRM
    18,647 / HEAVY-DIVERGE 8,331 / QUIET-CONFIRM 24,207 / QUIET-DIVERGE
    13,986). M1 PASS (QUIET-DIVERGE k=5 h=10 only). Family-wise p95
    (either sign) +0.023 S.
    ALL 48 CELLS FAIL, both directions. Largest |mean| 0.012 S
    (HEAVY-DIVERGE k=10 -> 30m, CI -0.020..+0.041).
    The only CIs off zero are at h = 1 minute and are ~0.002 S: after a
    DIVERGE minute price takes back ~0.002 S toward the flow, after a
    HEAVY-CONFIRM minute it gives back ~0.002 S -- the same thing seen from
    both sides: last minute's PRICE move mean-reverts a hair, whatever the
    flow did (bid/ask bounce). ~$0.01 of SPY; nothing to trade.
    MAGNITUDE (reported): HEAVY minutes move 1.08-1.24x more than random
    minutes at the same times (most at 1-10 min, fading by 30), QUIET
    minutes 0.94-1.00x. Stock volume says how BIG the next minutes are; it
    adds nothing on which WAY -- the magnitude-not-direction result again.
    VERDICT: no alignment / misalignment of flow with RVOL or price carries
    direction at 1-30 minutes.

Usage:  python check_flow_rvol.py
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
N_PRIOR, OPEN = 20, 570
KS, HS = (1, 5, 10), (1, 5, 10, 30)
RF_THR, HEAVY, QUIET, GAP, FIRST, LAST = 3.0, 2.0, 1.0, 10, 585, 920
CLASSES = ("HEAVY-CONFIRM", "HEAVY-DIVERGE", "QUIET-CONFIRM", "QUIET-DIVERGE")
FLOOR = 0.03
N_PERM, N_BOOT = 2000, 2000
SEED = 20261016


def ksum(A, k):
    """k-minute trailing sums ending at each bar (NaN for the first k-1)."""
    r = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)
    s = r[:, k:] - r[:, :-k]
    return np.concatenate([np.full((A.shape[0], k - 1), np.nan), s], axis=1)


def load(tk, am):
    f = pl.read_parquet(f"historical/NETPREM{tk}.parquet").select("minute_et", "net_premium").to_pandas()
    f["minute_et"] = pd.to_datetime(f["minute_et"])
    f["date"] = f["minute_et"].dt.date
    f["mod"] = f["minute_et"].dt.hour * 60 + f["minute_et"].dt.minute
    f = f[(f["mod"] >= OPEN) & (f["mod"] < 960)]
    b = BF.bars_full(tk).to_pandas()
    b["date"] = pd.to_datetime(b["date"]).dt.date
    flows, closes, vols = {}, {}, {}
    for d, g in f.groupby("date"):
        a = np.zeros(390)
        a[(g["mod"] - OPEN).to_numpy()] = pd.to_numeric(g["net_premium"], errors="coerce").fillna(0).to_numpy()
        flows[d] = a
    for d, g in b.groupby("date"):
        g = g.set_index("mod")
        c = g["close"].reindex(np.arange(OPEN, 960)).ffill().bfill().to_numpy(float)
        v = g["volume"].reindex(np.arange(OPEN, 960)).fillna(0).to_numpy(float)
        if np.isfinite(c).all():
            closes[d], vols[d] = c, v
    days = sorted(set(flows) & set(closes))
    S = {d: am[(tk, d)][1] for d in days if (tk, d) in am and am[(tk, d)][1]}
    return days, np.array([flows[d] for d in days]), np.array([closes[d] for d in days]), \
        np.array([vols[d] for d in days]), S


def build(am):
    rows, arrays = [], {}
    for tk in TICKERS:
        days, F, C, V, S = load(tk, am)
        arrays[tk] = (days, C, S)
        fk = {k: ksum(F, k) for k in KS}
        vk = {k: ksum(V, k) for k in KS}
        for i, d in enumerate(days):
            if i < N_PRIOR or d < DEP_START or d > END or d not in S:
                continue
            prior = slice(i - N_PRIOR, i)
            for k in KS:
                fmed = np.nanmedian(np.abs(fk[k][prior]), axis=0)
                vmed = np.nanmedian(vk[k][prior], axis=0)
                last = {c: -999 for c in CLASSES}
                for m in range(FIRST, LAST + 1):
                    j = m - OPEN
                    if not fmed[j] or not vmed[j] or not np.isfinite(fk[k][i, j]):
                        continue
                    x = fk[k][i, j] / fmed[j]
                    if abs(x) < RF_THR:
                        continue
                    rv = vk[k][i, j] / vmed[j]
                    s = 1 if x > 0 else -1
                    dp = C[i, j] - C[i, j - k]
                    if rv >= HEAVY:
                        vc = "HEAVY"
                    elif rv < QUIET:
                        vc = "QUIET"
                    else:
                        continue
                    if dp == 0:
                        continue
                    cls = f"{vc}-{'CONFIRM' if np.sign(dp) == s else 'DIVERGE'}"
                    if j - last[cls] < GAP:
                        continue
                    last[cls] = j
                    row = dict(tk=tk, day=d, di=i, k=k, m=m, s=s, cls=cls, x=x, rv=rv)
                    for h in HS:
                        mv = (C[i, j + h] - C[i, j]) / S[d]
                        row[f"y{h}"] = s * mv
                        row[f"a{h}"] = abs(mv)
                    rows.append(row)
    return pd.DataFrame(rows), arrays


def dboot(days, y, rng):
    ud, inv = np.unique(days, return_inverse=True)
    a = np.bincount(inv, weights=y, minlength=len(ud))
    n = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = [a[j].sum() / n[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def stats(df, col, rng, sign=1):
    y, days = sign * df[col].to_numpy(float), df["day"].to_numpy()
    lo, hi = dboot(days, y, rng)
    hv = [y[days < SPLIT].mean(), y[days >= SPLIT].mean()]
    o = np.argsort(days, kind="stable")
    sl = [y[o][ix].mean() for ix in np.array_split(np.arange(len(y)), 6)]
    return y.mean(), lo, hi, hv, sl


def ok(mean, p95, lo, hv, sl):
    st = mean > p95 and lo > 0 and all(h > 0 for h in hv) and sum(v > 0 for v in sl) >= 5
    return st, st and mean >= FLOOR


def placebo(ev, arrays, rng, n=N_PERM, mag=False):
    """-> (family-wise p95 of the signed max, {(cls,k,h): placebo mean |move|})."""
    mx = np.empty(n)
    magsum = {}
    groups = [(key, g) for key, g in ev.groupby(["cls", "k"])]
    for p in range(n):
        best = -9.0
        for (cls, k), g in groups:
            outs = {h: [] for h in HS}
            for tk, gg in g.groupby("tk"):
                days, C, S = arrays[tk]
                di = gg["di"].to_numpy()
                mins = rng.permutation(gg["m"].to_numpy()) - OPEN
                sg = rng.permutation(gg["s"].to_numpy())
                Sd = np.array([S[days[i]] for i in di])
                for h in HS:
                    outs[h].append((C[di, mins + h] - C[di, mins]) / Sd * sg)
            for h in HS:
                o = np.concatenate(outs[h])
                best = max(best, float(o.mean()), float((-o).mean()))
                if mag and p < 200:
                    magsum.setdefault((cls, k, h), []).append(float(np.abs(o).mean()))
        mx[p] = best
    return float(np.quantile(mx, 0.95)), {k: float(np.mean(v)) for k, v in magsum.items()}


def main():
    rng = np.random.default_rng(SEED)
    am = BF.am_straddles()
    ev, arrays = build(am)
    print(f"  events {len(ev)}  ({ev['day'].nunique()} days)")
    print("  by class x k:")
    print(ev.groupby(["cls", "k"]).size().unstack().to_string().replace("\n", "\n    "))

    print("\n  ---- MACHINERY ----")
    p_m1, _ = placebo(ev, arrays, rng, n=400)
    pl_ = ev.copy()
    pl_.loc[(pl_["cls"] == "QUIET-DIVERGE") & (pl_["k"] == 5), "y10"] += 0.05
    hits = []
    for c in CLASSES:
        for k in KS:
            g = pl_[(pl_["cls"] == c) & (pl_["k"] == k)]
            for h in HS:
                mean, lo, hi, hv, sl = stats(g, f"y{h}", rng)
                if ok(mean, p_m1, lo, hv, sl)[0]:
                    hits.append((c, k, h))
    print(f"    M1 plant +0.05 S at QUIET-DIVERGE k=5 h=10 -> passes at {hits}  "
          f"{'PASS' if hits == [('QUIET-DIVERGE', 5, 10)] else 'FAIL'}")

    p95, mag = placebo(ev, arrays, rng, mag=True)
    print(f"\n  family-wise placebo p95 (48 cells, either sign) = {p95:+.4f} S")
    print(f"    {'class':14} {'k':>2} {'h':>3}   {'n':>5}  {'toward flow':>11}  {'95% CI':>19}   halves            slices  STAT  TRADE   |move| vs random")
    for c in CLASSES:
        for k in KS:
            g = ev[(ev["cls"] == c) & (ev["k"] == k)]
            if g.empty:
                continue
            for h in HS:
                mean, lo, hi, hv, sl = stats(g, f"y{h}", rng)
                st, tr = ok(mean, p95, lo, hv, sl)
                mean_n, lo_n, hi_n, hv_n, sl_n = stats(g, f"y{h}", rng, sign=-1)
                st_n, _ = ok(mean_n, p95, lo_n, hv_n, sl_n)
                am_ = g[f"a{h}"].mean()
                pm = mag.get((c, k, h))
                print(f"    {c:14} {k:>2} {h:>3}m  {len(g):>5}  {mean:+11.4f}  [{lo:+.4f}, {hi:+.4f}]   "
                      f"{hv[0]:+.4f} / {hv[1]:+.4f}  {sum(v > 0 for v in sl)}/6   "
                      f"{'PASS' if st else ('AGAINST' if st_n else ' -  ')}  {'PASS' if tr else ' -  '}   "
                      f"{am_:.4f} vs {pm:.4f} ({am_ / pm:.2f}x)" if pm else "")
    ev.to_pickle(os.path.join("_implied_move_cache", "flow_rvol_events.pkl"))


if __name__ == "__main__":
    main()
