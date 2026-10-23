# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy>=1.11"]
# ///
"""
check_swing_replicate.py
========================
SWING STAGE A, REPLICATION -- DO THE TWO LEADS HOLD ON TICKERS WE NEVER LOOKED AT?
(user, 2026-10-22, after check_swing_factors found no EDGE but two leads)

UNIVERSE (rule fixed before looking at any of them): the most-traded option
underlyings in the lake (volume summed over 2025-03-10, 2025-09-10,
2026-03-10, 2026-09-10), excluding the 9 tickers already studied, leveraged /
inverse ETFs (TQQQ, SOXL, TSLL), VIX and index roots (SPXW, SPX, XSP), the
duplicate share class GOOG, and anything without historical/{T}.parquet and
GEX{T}.parquet (dropped HYG). First 25 in volume order:
  TSLA AAPL ORCL TLT PLTR AMZN IBIT AMD SLV MU MSTR INTC GOOGL HOOD XLF PCG
  GME USO COIN NFLX TSM XLE WMT MRVL BE
Factors / targets exactly as check_swing_factors (same code, panel()).

WINDOW   the whole lake, 2023-10-12 onward (every ticker here is fresh data);
         halves split at the midpoint 2025-03-17.

PRE-REGISTERED (two hypotheses, signs fixed by the original study)
  H1  skew -> ret10 IC > 0     (steeper 25-delta put skew -> higher 10-session return)
  H2  gex  -> move10 IC < 0    (lower / negative prior-day GEX -> bigger realized
                                move vs the implied one)
  Null: per-cell circular-shift distribution (1,000 shifts of 60+ sessions,
  same shift for every ticker); z of the real group-mean IC.
  PASS = z beyond 1.96 IN THE PREDICTED DIRECTION (one-sided 2.5% each =
  Bonferroni for two), both halves in the predicted direction, >= 60% of
  tickers in the predicted direction; H1 also keeps >= half its IC with
  mom20 + mom60 partialled out.
  Reported, unscored: h 5 / 20, the other target, the other six factors,
  per-ticker ICs, quintiles.

Usage:  python check_swing_replicate.py      (cache: check_swing_factors.build)

RUN LOG
  2026-10-22 run 1: 25 tickers, ~500 sessions each (their 1m bar history,
    not the lake, sets the start: ~2024-09); GME thin (99 days), PCG skew 30%.
    H1 skew -> ret10: IC +0.045, z +0.65, halves +0.092 / +0.054, 16/25 in
      the predicted direction, trend-partial +0.044 -> FAIL (right sign, a
      fifth of the original +0.22, inside the null).
    H2 gex -> move10: IC -0.015, z -0.82, halves +0.009 / -0.017, 12/24 ->
      FAIL.
    No other factor reaches |IC| 0.07 in this universe.
  VERDICT: neither lead replicates. The original skew result lived in
    SPY/QQQ/IWM + mega-cap tech, which share one market path: it was close to
    a single time series (a few fear episodes), not a cross-section of
    independent evidence. Swing stage B not warranted on these factors.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_swing_factors as S                          # noqa: E402

REPL = ("TSLA", "AAPL", "ORCL", "TLT", "PLTR", "AMZN", "IBIT", "AMD", "SLV", "MU", "MSTR", "INTC", "GOOGL",
        "HOOD", "XLF", "PCG", "GME", "USO", "COIN", "NFLX", "TSM", "XLE", "WMT", "MRVL", "BE")
HALF = dt.date(2025, 3, 17)
HYP = (("H1", "skew", "ret", 10, +1), ("H2", "gex", "move", 10, -1))
N_SHIFT, MIN_SHIFT, SEED = 1000, 60, 20261023


def ic_group(F, T, tks, mask=None):
    v = []
    for t in tks:
        y = T[t] if mask is None else np.where(mask, T[t], np.nan)
        v.append(S.spear(F[t], y))
    return np.nanmean(v), v


def main():
    rng = np.random.default_rng(SEED)
    P = {}
    for t in REPL:
        if not os.path.exists(f"{S.CACHE}/eod_{t}.parquet"):
            print(f"  {t}: no cache")
            continue
        try:
            P[t] = S.panel(t)
        except Exception as e:                           # noqa: BLE001
            print(f"  {t}: panel failed {type(e).__name__}: {e}")
    idx = sorted(set().union(*[set(p.index) for p in P.values()]))
    P = {t: p.reindex(idx) for t, p in P.items()}
    tks = list(P)
    dates = np.array(idx)
    n = len(dates)
    for t in tks:
        p = P[t]
        print(f"  {t:5} days {p['S'].notna().sum():4d}  skew {p['skew'].notna().mean():.0%}  gex {p['gex'].notna().mean():.0%}"
              f"  IV30 {p['iv30p'].median():5.1f}  ret10 {p['ret10'].notna().mean():.0%}")

    print(f"\n  ======== PRE-REGISTERED ({len(tks)} tickers, 2023-10-12 .. ) ========")
    verdict = {}
    for name, f, tg, h, sign in HYP:
        F = {t: P[t][f].to_numpy(float) for t in tks}
        T = {t: P[t][f"{tg}{h}"].to_numpy(float) for t in tks}
        real, per = ic_group(F, T, tks)
        null = np.empty(N_SHIFT)
        for i in range(N_SHIFT):
            k = int(rng.integers(MIN_SHIFT, n - MIN_SHIFT))
            null[i] = np.nanmean([S.spear(np.roll(F[t], k), T[t]) for t in tks])
        z = (real - np.nanmean(null)) / np.nanstd(null)
        hv = [ic_group(F, T, tks, dates < HALF)[0], ic_group(F, T, tks, dates >= HALF)[0]]
        same = sum(1 for v in per if not np.isnan(v) and np.sign(v) == sign)
        nv = sum(1 for v in per if not np.isnan(v))
        pic = np.nan
        if tg == "ret":
            pic = np.nanmean([S.partial_ic(F[t], T[t], np.column_stack([P[t]["mom20"].to_numpy(float),
                                                                       P[t]["mom60"].to_numpy(float)])) for t in tks])
        ok = (sign * z > 1.96 and all(sign * x > 0 for x in hv) and same >= 0.6 * nv
              and (tg != "ret" or (sign * pic > 0 and abs(pic) >= 0.5 * abs(real))))
        verdict[name] = ok
        print(f"\n  {name}  {f} -> {tg}{h}, predicted {'+' if sign > 0 else '-'}")
        print(f"     IC {real:+.3f}   z {z:+.2f} (need {'+' if sign > 0 else '-'}1.96)   halves {hv[0]:+.3f} / {hv[1]:+.3f}"
              f"   tickers in predicted direction {same}/{nv}" + ("" if np.isnan(pic) else f"   trend-partial {pic:+.3f}"))
        print("     per ticker: " + "  ".join(f"{t} {v:+.2f}" for t, v in zip(tks, per)))
        print(f"     {name} {'PASS' if ok else 'FAIL'}")

    print("\n  ======== REPORTED (unscored) ========")
    print("  group-mean IC, every factor x target x horizon:")
    for f in S.FACTORS:
        cells = []
        for tg in S.TARGETS:
            for h in S.HS:
                cells.append(f"{tg}{h} {ic_group({t: P[t][f].to_numpy(float) for t in tks}, {t: P[t][f'{tg}{h}'].to_numpy(float) for t in tks}, tks)[0]:+.3f}")
        print(f"     {f:7} " + "  ".join(cells))
    S.quintiles(P, tks, ("skew", "gex"), np.ones(n, bool))
    print(f"\n  VERDICT: H1 {'PASS' if verdict.get('H1') else 'FAIL'}   H2 {'PASS' if verdict.get('H2') else 'FAIL'}")


if __name__ == "__main__":
    main()
