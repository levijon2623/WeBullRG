# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0"]
# ///
"""
check_spx_spy_im.py
===================
SPX vs SPY implied-move bands, normalised -- and when they DIFFER, which one
is right? (User, 2026-10-05, watching the live pink band.)

DATA  _implied_move_cache (build_implied_move.py): the 09:35 same-day ATM
      straddle for SPY and SPXW, same quote window (last quote 09:33-09:35),
      same filters, same strike interpolation. p = straddle / spot (each on
      its own spot). Outcome on SPY 1m bars (the band is drawn on SPY):
      RANGE = (max high - min low over 09:36-15:59) / the 09:35 close.

FIRST LOOK (before this file, 2026-10-05): r = pSPX / pSPY has median 0.960,
      IQR 0.939-0.974; |r - 1| > 5% on 39% of 633 days. So SPX's straddle is
      persistently ~4% CHEAPER per unit of spot. Candidate reasons (not
      tested here): a SPY 0DTE holder can exercise on after-hours moves until
      the 17:30 cut-off (American; SPXW is cash-settled at the 16:00 close);
      SPY's $1 strike grid (~0.13%) is coarser than SPX's $5 (~0.065%), and a
      linear interpolation between strikes over-states a convex ATM straddle.
      A persistent offset is NOT a disagreement -- the q ratios calibrate it
      out. The question is the DAY-TO-DAY deviation from it.

DESIGN (pre-registered)
    Rows: days with both, r within [0.7, 1.3] (outside = a broken quote on one
    side; count reported). Window: 2024-08-20 .. 2026-09-18 (the deployed
    slices plus later lake days; the spent pre-sample 2024-02..08 reported
    separately, labelled).
    PRIMARY (forecast encompassing):
        log RANGE = a + b1 log pSPY + b2 log pSPX           (OLS)
        Read the DIFFERENCE b2 - b1, 95% CI by bootstrap over days (2,000):
          CI > 0  -> when they differ, SPX is the more accurate
          CI < 0  -> SPY is the more accurate
          spans 0 -> no detectable difference
        AND the point estimate must have the same sign in both halves
        (2024-08-20..2025-08-20 / 2025-08-21..), else "no difference".
    SECONDARY
        d = log r - median(log r over the prior 60 days) (causal: today's SPX
        richness beyond the usual offset). Spearman(d, log(RANGE / pSPY)):
        > 0 means a relatively rich SPX foretells SPY under-pricing the day.
        And on the top / bottom quartile of d: SPY's RANGE / pSPY median.
    REPORTED: r by year; each source alone (R^2 of log RANGE on log p); the
        65% band containment of each, with q fitted on the other half.

RUN LOG
  2026-10-05 run 1: 630 days with both; 10 broken-quote days dropped. The
    offset narrows by year: r median 0.944 (2024), 0.960 (2025), 0.971 (2026).
    PRIMARY: b2 - b1 = +0.593, CI [-0.519, +1.641] -- spans 0; positive in
    both halves (+0.124, +1.082) but the pre-sample (labelled) is -0.894.
    -> NO DETECTABLE DIFFERENCE. Each alone explains the same share of the
    day's range: R^2 SPY 0.541 / SPX 0.546 (halves 0.643/0.644, 0.400/0.408).
    SECONDARY: Spearman(d, log RANGE/pSPY) +0.085, p 0.058 -- a lean, not a
    finding: an unusually cheap SPX day ranged 1.61 SPY straddles vs 1.69-1.71
    otherwise. Containment of each 65% side band out of half: SPY 64-67%, SPX
    64-67% -- both calibrated. The two carry the same information once the
    steady offset is calibrated out; no reason to draw a second band.

Usage:  python check_spx_spy_im.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_implied_move as CI                         # noqa: E402

START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 9, 18)
R_OK = (0.7, 1.3)
N_BOOT = 2000
SEED = 20261005


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    s = CI.straddles().filter(pl.col("win") == "am")
    w = (s.filter(pl.col("root").is_in(["SPY", "SPXW"]))
          .with_columns((pl.col("strad") / pl.col("spot")).alias("p"))
          .pivot(values="p", index="date", on="root").drop_nulls().sort("date"))
    bars = CI.bars("SPY")
    rows = []
    for day, pspy, pspx in w.select("date", "SPY", "SPXW").iter_rows():
        g = bars.get(day)
        if g is None:
            continue
        a = g.filter(pl.col("mod") == 575)["close"]
        rest = g.filter((pl.col("mod") >= 576) & (pl.col("mod") <= 959))
        if a.is_empty() or rest.height < 300:
            continue
        rng = (rest["high"].max() - rest["low"].min()) / a[0]
        up, dn = (rest["high"].max() - a[0]) / a[0], (a[0] - rest["low"].min()) / a[0]
        rows.append((day, pspy, pspx, rng, up, dn))
    day = np.array([r[0] for r in rows])
    pspy, pspx, rng = (np.array([r[i] for r in rows], float) for i in (1, 2, 3))
    up, dn = (np.array([r[i] for r in rows], float) for i in (4, 5))
    r = pspx / pspy
    bad = (r < R_OK[0]) | (r > R_OK[1])
    print(f"  days with both + SPY bars: {len(rows)}; r outside {R_OK}: {bad.sum()} (dropped)")
    lr = np.log(r)
    # causal deviation from the usual offset
    d = np.full(len(r), np.nan)
    for i in range(len(r)):
        prev = lr[max(0, i - 60):i][~bad[max(0, i - 60):i]]
        if len(prev) >= 20 and not bad[i]:
            d[i] = lr[i] - np.median(prev)
    years = sorted({x.year for x in day})
    print("  r = pSPX/pSPY by year: " + "  ".join(
        f"{y}: median {np.median(r[(~bad) & np.array([x.year == y for x in day])]):.3f}" for y in years))

    def block(mask, label):
        m = mask & ~bad
        y = np.log(rng[m])
        X1, X2 = np.log(pspy[m]), np.log(pspx[m])
        b = ols(y, [X1, X2])
        rng_ = np.random.default_rng(SEED)
        bs = []
        n = m.sum()
        for _ in range(N_BOOT):
            j = rng_.integers(0, n, n)
            bb = ols(y[j], [X1[j], X2[j]])
            bs.append(bb[2] - bb[1])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        r2 = {}
        for nm, X in (("SPY", X1), ("SPX", X2)):
            bb = ols(y, [X])
            res = y - (bb[0] + bb[1] * X)
            r2[nm] = 1 - res.var() / y.var()
        print(f"  {label:34} n {n:4}  b1(SPY) {b[1]:+.3f}  b2(SPX) {b[2]:+.3f}  "
              f"b2-b1 {b[2] - b[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]   R^2 alone: SPY {r2['SPY']:.3f}  SPX {r2['SPX']:.3f}")
        return b[2] - b[1], lo, hi

    dep = np.array([START <= x <= END for x in day])
    h1 = np.array([START <= x < SPLIT for x in day])
    h2 = np.array([SPLIT <= x <= END for x in day])
    pre = np.array([x < START for x in day])
    print("\n  PRIMARY -- log RANGE = a + b1 log pSPY + b2 log pSPX")
    est, lo, hi = block(dep, "2024-08-20..2026-09-18")
    e1, _l, _h = block(h1, "  half 1 (..2025-08-20)")
    e2, _l, _h = block(h2, "  half 2 (2025-08-21..)")
    block(pre, "  pre-sample (SPENT, labelled)")
    same = np.sign(e1) == np.sign(e2) == np.sign(est)
    verdict = ("SPX more accurate" if lo > 0 and same else
               "SPY more accurate" if hi < 0 and same else "NO DETECTABLE DIFFERENCE")
    print(f"  -> {verdict}")

    print("\n  SECONDARY -- today's SPX richness beyond the usual offset (d)")
    m = dep & ~bad & ~np.isnan(d)
    from scipy.stats import spearmanr          # noqa: E402
    rho, p = spearmanr(d[m], np.log(rng[m] / pspy[m]))
    print(f"    Spearman(d, log(RANGE / pSPY)) {rho:+.3f} (p {p:.3f}, n {m.sum()})")
    q1, q3 = np.quantile(d[m], [0.25, 0.75])
    for nm, mm in (("SPX relatively CHEAP (bottom quartile d)", m & (d <= q1)),
                   ("middle half", m & (d > q1) & (d < q3)),
                   ("SPX relatively RICH (top quartile d)", m & (d >= q3))):
        print(f"    {nm:42} n {mm.sum():4}  median RANGE/pSPY {np.median(rng[mm] / pspy[mm]):.3f}  "
              f"RANGE/pSPX {np.median(rng[mm] / pspx[mm]):.3f}")

    print("\n  REPORTED -- 65% band containment, q fitted on the OTHER half")
    for nm, p_ in (("SPY", pspy), ("SPX", pspx)):
        out = []
        for fit, test in ((h1, h2), (h2, h1)):
            f, t = fit & ~bad, test & ~bad
            qu = np.quantile(up[f] / p_[f], 0.65)
            qd = np.quantile(dn[f] / p_[f], 0.65)
            out.append(np.mean((up[t] <= qu * p_[t]) & (dn[t] <= qd * p_[t])))
            out.append((np.mean(up[t] <= qu * p_[t]), np.mean(dn[t] <= qd * p_[t])))
        print(f"    {nm}: both sides inside: half2 {out[0]:.1%}, half1 {out[2]:.1%};  "
              f"up/down side alone: half2 {out[1][0]:.1%}/{out[1][1]:.1%}, half1 {out[3][0]:.1%}/{out[3][1]:.1%}")


if __name__ == "__main__":
    main()
