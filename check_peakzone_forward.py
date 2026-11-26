"""
check_peakzone_forward.py -- FORWARD TEST, PRE-REGISTERED 2026-11-24 (user: "go ahead with the forward test on the
neg-gex zones"). LOCKED: never retune; reads are descriptive until the final one.

WHERE IT CAME FROM. check_peak_zone.py (holdout scored 2026-11-24, e2f3ef5) found no peak-gamma effect on its
registered questions, but one UNREGISTERED secondary: arrivals at a NEGATIVE-gamma peak zone in the FIRST 30 MINUTES
went through more often than matched ordinary levels -- holdout +0.120 [+0.069, +0.193], discovery +0.030
[-0.004, +0.094]; pooled over all 552 sessions +0.075 [+0.037, +0.124], 170 matched arrivals in 65 sessions (zone
through rate 8.9%). The same cell for POSITIVE zones pooled -0.005 [-0.031, +0.022]. Found by looking -> it needs data
nobody has looked at.

DEFINITIONS -- exactly check_peak_zone.py as committed at e2f3ef5 (imported, not copied):
  zone     the weekly GEX page's 0DTE map with the index added (SPY + SPXW), rebuilt at 09:31 ET from the lake
           (zone_map): peak |net gamma| strike, joined with its >= 85% runner-up when within $3, else peak +- $1.
           NEGATIVE = the zone's summed net gamma < 0.
  arrival  price enters the zone after >= 5 bp beyond the edge it enters from; t0 in [09:32:00, 10:00:00) ET.
  outcome  THROUGH = the far edge + 5 bp is reached within 15 min before the entry edge - 5 bp.
  matched  same-width windows whose nearest edge is >= $1 from the zone's, same session, arrivals within +-60 min
           (CTRL_BUF 1.0, MATCH_S 3600). Zone arrivals with no control are dropped (counted).
  F1 estimand  mean over negative-zone first-30-min arrivals of (THROUGH - mean THROUGH of its matched controls).

SAMPLE   SPY sessions from 2026-11-25 on (11-24 itself excluded: it is the morning that prompted the question). Needs
         the lake for each session: uw_options_data_lake.py build -> trades-core-build -> silver-build (see the lake
         refresh method). Sessions without a negative zone simply contribute nothing.
READS    INTERIM at 60 sessions (~2027-02-23) and 120 (~2027-05-19): DESCRIPTIVE ONLY -- no stopping, no retuning.
         FINAL at 150 sessions (~2027-07-02).
PASS     F1 > 0 AND the lower bound of its day-block bootstrap 90% CI (= one-sided 5%; 2,000 resamples of sessions)
         > 0. Predicted sign: positive.
POWER    honest: ~0.31 matched arrivals per session -> ~46 by the final read. If the true effect is the pooled +0.075
         (likely overstated -- it was the best-looking cell), power is only ~55%; at the holdout's +0.12, ~90%. A
         FAIL at 150 sessions therefore says "not big", not "zero".
REPORTED, NOT TESTED  the same estimand for POSITIVE zones in the first 30 min (expected ~0) and for negative zones
         after 10:00 (holdout +0.003); raw zone THROUGH / STALL / REVERSE rates; arrivals per session.
AMENDMENT 2026-11-26 -- THE BASELINE IS THE NULL, NOT 0 (user approved; made before any forward session was read --
         the lake had none). check_peakzone_null.py ran F1's exact construction with the zone swapped for a random
         visited window of the same width, on the same 552 historical sessions (1,000 draws): in F1's sessions it
         centres at +0.0571 [-0.0043, +0.1288]; over all sessions +0.0616 [+0.0126, +0.1134]. The first-30-minute
         matching itself favours the "zone" side, so a test against 0 would pass on machinery alone. Against the null,
         the history that prompted this test is: pooled +0.0755 -> +0.018 excess; holdout +0.120 -> +0.063; discovery
         +0.030 -> -0.027 (wrong side). The lead mostly evaporates; the test stays open, re-based:
  PASS (amended)  F1 - NULL_F1 > 0 AND the lower bound of its day-block bootstrap 90% CI minus NULL_F1 > 0, with
         NULL_F1 = +0.0571 frozen. Also printed every read (descriptive): the same null on the forward sessions
         themselves (200 draws), in case the bias drifts.
  POWER (amended)  against an excess of +0.018 (the pooled history), power at ~46 arrivals is near the 10% false-alarm
         floor; only an excess like the holdout's +0.06 has a fair chance (~40%). Expect a FAIL unless the effect is
         real AND large.
TRADING USE  none until a FINAL PASS. Even then it is a direction-agnostic "this level tends to give way early", not a
         side to take.

    python check_peakzone_forward.py            (whatever forward sessions the lake has; prints the read)
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

START = "2026-11-25"
INTERIM = (60, 120)
FINAL = 150
T_END = 10 * 3600
CACHE = os.path.join(ROOT, "_peakzone_cache", "forward")
NULL_F1 = 0.0571          # FROZEN 2026-11-26 (amendment): check_peakzone_null.py, F1's sessions, 1,000 draws


def forward_days():
    have = sorted(os.path.basename(p)[5:15] for p in glob.glob(os.path.join(ROOT, "lake", "silver", "option-contracts-1m", "date=*")))
    tc = {os.path.basename(p)[5:15] for p in glob.glob(os.path.join(ROOT, "lake", "silver", "trades-core", "date=*"))}
    return [d for d in have if d >= START and d in tc]


def cell(S, sign, early):
    import check_peak_zone as Z
    P, nz = {}, 0
    for s in S:
        v = s.get("v", {}).get("SPY+")
        if not v or "rows" not in v or v["zone"]["sign"] != sign:
            continue
        C = [r for r in v["rows"] if Z.ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"])]
        for z in [r for r in v["rows"] if r["z"] and ((r["t0"] < T_END) == early)]:
            nz += 1
            c = [Z.THROUGH(r) for r in C if abs(r["t0"] - z["t0"]) <= Z.MATCH_S]
            if c:
                P.setdefault(s["d"], []).append(Z.THROUGH(z) - sum(c) / len(c))
    return P, nz


def boot90(P, n=2000, seed=7):
    import numpy as np
    import check_peak_zone as Z
    days = sorted(P)
    pt, nobs = Z.est_mean(P, days)
    if not days:
        return pt, float("nan"), float("nan"), nobs, 0
    rng = np.random.default_rng(seed)
    bs = np.array([Z.est_mean(P, [days[i] for i in rng.integers(0, len(days), len(days))])[0] for _ in range(n)])
    bs = bs[bs == bs]
    lo, hi = np.percentile(bs, [5, 95])
    return pt, lo, hi, nobs, len(days)


def main():
    os.environ["PEAKZONE_CACHE"] = CACHE          # read by check_peak_zone at import, in the worker processes too
    import check_peak_zone as Z
    Z.CACHE = CACHE
    days = forward_days()
    print(f"forward sessions in the lake from {START}: {len(days)}"
          + (f" ({days[0]} .. {days[-1]})" if days else ""))
    if not days:
        return
    S = Z.build(days)
    n = len(days)
    stage = "FINAL" if n >= FINAL else ("interim (descriptive)" if n >= INTERIM[0] else "early look (descriptive)")
    for lab, sg, early in (("F1  NEGATIVE zones, first 30 min", -1, True),
                           ("    positive zones, first 30 min", +1, True),
                           ("    negative zones, after 10:00", -1, False)):
        P, nz = cell(S, sg, early)
        pt, lo, hi, nobs, nd = boot90(P)
        verdict = ""
        if lab.startswith("F1"):
            if n >= FINAL:
                verdict = "PASS" if pt - NULL_F1 > 0 and lo - NULL_F1 > 0 else "FAIL"
            verdict = f"| minus the null ({NULL_F1:+.4f}): {pt - NULL_F1:+.3f} [{lo - NULL_F1:+.3f}, {hi - NULL_F1:+.3f}]  " + verdict
        print(f"{lab}: through vs matched {pt:+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]  {nobs} matched of {nz} arrivals, "
              f"{nd} sessions  {verdict}")
    import numpy as np
    from check_peakzone_null import fake_cell
    rng = np.random.default_rng(26)
    a = np.array([m for m, k in (fake_cell(S, rng, -1) for _ in range(200)) if k])
    if len(a):
        print(f"    the null on these forward sessions (descriptive, 200 draws): {a.mean():+.4f} "
              f"[{np.percentile(a, 2.5):+.4f}, {np.percentile(a, 97.5):+.4f}]  (frozen baseline {NULL_F1:+.4f})")
    print(f"stage: {stage} ({n} of {FINAL} sessions)")


if __name__ == "__main__":
    main()
