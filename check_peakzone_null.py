"""
check_peakzone_null.py -- the shuffle NULL for check_peakzone_forward.py's F1 cell, run 2026-11-26 BEFORE any forward
session was read (the user: "do please fix that, very important find").

WHY. F1 asks whether negative-gamma peak-zone arrivals in the first 30 minutes go THROUGH more than matched ordinary
levels, and the forward test tested that difference against 0. But check_peak_zone.py's own discovery null (a random
visited window relabelled as the zone, same matching) centred at +0.022 / +0.025 for its all-day THROUGH questions, not
0: the matching itself favours the zone side a little. F1's cell (first 30 min only) never had its own null. This
script measures it, on the same 552 historical sessions (spent for discovery/holdout, but the null is a property of the
machinery, not of gamma), and check_peakzone_forward.py subtracts the frozen mean.

NULL, exactly F1's construction with the zone swapped:
  sessions  the ones whose REAL zone is negative (the sessions F1 draws from).
  fake zone one window drawn at random from that session's non-zone windows that price arrived at (same width, the
            check_peak_zone null_check rule); its arrivals with t0 < 10:00 are the "zone" arrivals.
  controls  ctrl_ok against the fake zone (nearest edge >= $1 away, never the real zone), arrivals within +-60 min.
  estimand  mean over fake-zone arrivals of (THROUGH - mean THROUGH of matched controls), pooled like F1.
  draws     1,000 (seed 26). Reported: mean, 2.5/97.5 percentiles; also the all-sessions (either sign) version.

    python check_peakzone_null.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
T_END = 10 * 3600


def fake_cell(S, rng, sign):
    import check_peak_zone as Z
    xs = []
    for s in S:
        v = s.get("v", {}).get("SPY+")
        if not v or "rows" not in v or (sign is not None and v["zone"]["sign"] != sign):
            continue
        los = sorted({r["lo"] for r in v["rows"] if not r["z"]})
        if not los:
            continue
        zl = los[rng.integers(0, len(los))]
        C = [r for r in v["rows"] if r["lo"] != zl and Z.ctrl_ok(r, v["w"], zl, zl + v["w"])]
        for z in [r for r in v["rows"] if not r["z"] and r["lo"] == zl and r["t0"] < T_END]:
            c = [Z.THROUGH(r) for r in C if abs(r["t0"] - z["t0"]) <= Z.MATCH_S]
            if c:
                xs.append(Z.THROUGH(z) - sum(c) / len(c))
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def main():
    import numpy as np
    import check_peak_zone as Z
    days = Z.sessions_of("discovery") + Z.sessions_of("holdout")
    S = Z.build(sorted(days))
    print(f"{len(S)} historical sessions")
    # the real F1 cell on the same sessions, for reference (the +0.075 pooled figure in the forward docstring)
    import check_peakzone_forward as F
    P, nz = F.cell(S, -1, True)
    pt, nobs = Z.est_mean(P, sorted(P))
    print(f"real F1 cell, pooled history: {pt:+.4f} ({nobs} matched of {nz} arrivals, {len(P)} sessions)")
    for lab, sign in (("negative-zone sessions (F1's)", -1), ("all sessions, either sign", None)):
        rng = np.random.default_rng(26)
        r = [fake_cell(S, rng, sign) for _ in range(1000)]
        a = np.array([m for m, n in r if n])
        ns = np.array([n for m, n in r if n])
        print(f"NULL {lab}: mean {a.mean():+.4f}  [{np.percentile(a, 2.5):+.4f}, {np.percentile(a, 97.5):+.4f}]  "
              f"median n {int(np.median(ns))} arrivals/draw, {len(a)} draws")


if __name__ == "__main__":
    main()
