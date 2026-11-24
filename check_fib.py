"""
check_fib.py -- FIBONACCI RETRACEMENTS: do they mean anything on SPY / QQQ / IWM intraday? (user, 2026-11-23: "popular in
the Webull community ... a general sort of look".) PRE-REGISTERED 2026-11-23, committed before any of it was computed.

PRIOR (said first): every level we have tested was placebo-like -- whole-dollar strikes (11-06), volume-profile levels
1 day..3 weeks, stacked POCs -- and the PULLBACK screens found the bounce's retracement depth (s_retrace) said nothing
about the next leg (IC +0.024, not significant). The published record mostly finds retracement depths spread smoothly,
with no extra mass at the Fibonacci ratios. The case FOR them is self-fulfilling: if enough traders act at 61.8%, a
small effect could exist. That is what this looks for.

SWINGS   the zigzag used by check_structure (a pivot is confirmed when price reverses by R from the extreme), on the
         session's 1-minute closes with R = 0.10% of price (scale "1m"), and on 5-minute closes with R = 0.25% ("5m").
         RTH bars from historical/{T}.parquet, SPY / QQQ / IWM, each session on its own. A LEG = pivot A -> pivot B; only
         legs of at least 5 R (so even the 23.6% level is past the zigzag's own minimum reversal).
LEVELS   FIB = 0.236, 0.382, 0.500, 0.618, 0.786 of the leg, measured back from B.
         PLACEBO for each Fib ratio f: f - 0.05 and f + 0.05 (its non-Fibonacci neighbours, same leg, same tape).

Q1 CLUSTERING (where do pullbacks END?)  r = |B - C| / |B - A|, C = the next pivot (the counter-move's end).
   MASS(set) = share of legs whose r lies within +-0.0125 of a ratio in the set. NULL = the Fib set SHIFTED by every
   delta in -0.060..+0.060 (step 0.005, |delta| >= 0.020 so it never overlaps itself) -- same spacing, same smooth
   density, not Fibonacci. z = (MASS(Fib) - mean(null)) / sd(null). Per level too (each ratio vs its own shifts).
Q2 REACTION (does price respect a Fib level when it gets there?)  After B is confirmed (only then is the leg known),
   every level not yet reached is followed: at the first minute close at or through it (a TOUCH), Y = 1 if price
   closes back beyond B (the leg resumes) before closing 0.15 of the leg deeper than the level, within 60 minutes;
   else 0. CONTRAST D = mean Y at Fib touches - mean Y at the placebo neighbours' touches. Depth lowers Y smoothly,
   and the neighbours sit symmetrically either side, so a Fibonacci effect shows as D > 0.
SAMPLES  DISCOVERY 2024-08-20 .. 2025-12-31 (halves split 2025-04-30). HOLDOUT 2026-01-02 .. 2026-11-20, scored ONCE
   (--holdout writes _fib_cache/RESULT-holdout.txt and refuses to run again). Pre-sample 2023-10..2024-08 is spent.
PASS (per scale; BH q < 0.10 across the 4 tests = 2 questions x 2 scales)
   Q1  Fib MASS above the 95th percentile of the shifted sets, on the holdout, and the same sign on discovery.
   Q2  D > 0 with the day-block bootstrap (2,000) 95% CI above 0, on the holdout, and the same sign on discovery.
   A pass on Q2 earns a money test (entry at the touch, through the option at the user's fills) as a NEW
   pre-registration. Nothing goes on the viewer from this file.
MACHINERY  M1 a planted series: pullbacks forced to end within +-0.01 of 0.618 on a third of discovery legs must give
   Q1 z > 3; M2 a sample with the Fib set replaced by a shifted set must not stand out (|z| < 2).

Usage:  python check_fib.py --discovery | --holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CACHE = "_fib_cache"
DISC, HOLD, HALF = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-20"), "2025-04-30"
FIB = (0.236, 0.382, 0.500, 0.618, 0.786)
PLACEBO_OFF = 0.05
BAND = 0.0125
SHIFTS = [round(s, 3) for s in np.arange(-0.060, 0.0601, 0.005) if abs(s) >= 0.020 - 1e-9]
SCALES = {"1m": (1, 0.0010), "5m": (5, 0.0025)}
MIN_LEG_R, STOP_DEPTH, HORIZON = 5, 0.15, 60
SEED = 20261124


def zigzag(c, R):
    """pivots [(kind, value, index of the extreme, index where confirmed)] on one session's closes."""
    piv, d = [], 0
    hi = lo = c[0]; hi_i = lo_i = 0
    for i in range(1, len(c)):
        v = c[i]
        if d == 0:
            if v > hi: hi, hi_i = v, i
            if v < lo: lo, lo_i = v, i
            if hi - v >= R: piv.append(("H", hi, hi_i, i)); d, lo, lo_i = -1, v, i
            elif v - lo >= R: piv.append(("L", lo, lo_i, i)); d, hi, hi_i = 1, v, i
        elif d == 1:
            if v > hi: hi, hi_i = v, i
            elif hi - v >= R: piv.append(("H", hi, hi_i, i)); d, lo, lo_i = -1, v, i
        else:
            if v < lo: lo, lo_i = v, i
            elif v - lo >= R: piv.append(("L", lo, lo_i, i)); d, hi, hi_i = 1, v, i
    return piv


def session_rows(d, c, scale):
    """legs (Q1) and touches (Q2) for one session's closes at one scale."""
    step, rf = SCALES[scale]
    R = c[0] * rf
    piv = zigzag(c, R)
    legs, touches = [], []
    ratios = {f: "fib" for f in FIB}
    for f in FIB:
        for s in (-PLACEBO_OFF, PLACEBO_OFF):
            ratios.setdefault(round(f + s, 3), "placebo")
    for i in range(len(piv) - 1):
        A, B = piv[i], piv[i + 1]
        size = abs(B[1] - A[1])
        if size < MIN_LEG_R * R:
            continue
        sgn = 1 if B[0] == "H" else -1                    # +1 up-leg (B a high): a pullback goes DOWN
        if i + 2 < len(piv):
            legs.append({"d": d, "r": abs(B[1] - piv[i + 2][1]) / size})
        conf = B[3]                                       # the leg is only KNOWN from here
        retr_at_conf = sgn * (B[1] - c[conf]) / size
        for f, kind in ratios.items():
            if f <= retr_at_conf:
                continue
            lvl = B[1] - sgn * f * size
            stop = B[1] - sgn * (f + STOP_DEPTH) * size
            t = next((j for j in range(conf + 1, len(c)) if sgn * (c[j] - lvl) <= 0), None)
            if t is None:
                continue
            if sgn * (B[1] - c[t]) / size > f + STOP_DEPTH:   # gapped straight through to the stop: not a touch
                continue
            y = 0
            for j in range(t + 1, min(len(c), t + 1 + HORIZON // step)):
                if sgn * (c[j] - B[1]) >= 0: y = 1; break
                if sgn * (c[j] - stop) <= 0: break
            touches.append({"d": d, "f": f, "kind": kind, "base": min(FIB, key=lambda x: abs(x - f)), "y": y})
    return legs, touches


def load(lo, hi):
    import check_orb as O
    out = {s: ([], []) for s in SCALES}
    for tk in ("SPY", "QQQ", "IWM"):
        bars = O.load_bars(tk)
        for day, (o, h, l, c) in bars.items():
            d = day.isoformat() if hasattr(day, "isoformat") else str(day)
            if not lo <= d <= hi:
                continue
            for s, (step, _rf) in SCALES.items():
                cc = np.asarray(c, float)[step - 1::step] if step > 1 else np.asarray(c, float)
                lg, tc = session_rows(d, cc, s)
                out[s][0].extend(lg); out[s][1].extend(tc)
    return out


def mass(r, ratios):
    r = np.asarray(r)
    return float(np.mean(np.any(np.abs(r[:, None] - np.asarray(ratios)[None, :]) <= BAND, axis=1)))


def q1(legs):
    r = np.array([x["r"] for x in legs])
    m = mass(r, FIB)
    null = np.array([mass(r, [f + s for f in FIB]) for s in SHIFTS])
    per = {f: (mass(r, [f]), np.mean([mass(r, [f + s]) for s in SHIFTS])) for f in FIB}
    return m, null, (m - null.mean()) / null.std(), per


def q2(touches, rng, n=2000):
    D = lambda T: (np.mean([t["y"] for t in T if t["kind"] == "fib"]) - np.mean([t["y"] for t in T if t["kind"] == "placebo"]))
    d0 = D(touches)
    by = defaultdict(list)
    for t in touches:
        by[t["d"]].append(t)
    days = list(by)
    bs = []
    for _ in range(n):
        T = [t for dd in rng.choice(days, len(days)) for t in by[dd]]
        bs.append(D(T))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float(min(1.0, 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))))
    per = {f: (np.mean([t["y"] for t in touches if t["f"] == f]),
               np.mean([t["y"] for t in touches if t["kind"] == "placebo" and t["base"] == f]),
               sum(t["f"] == f for t in touches)) for f in FIB}
    return d0, (lo, hi), p, per


def report(data, title, lines, rng, machinery=False):
    lines.append(f"=== {title}")
    for s, (legs, touches) in data.items():
        m, null, z, per = q1(legs)
        lines.append(f"[{s}] {len(legs)} legs, {len(touches)} touches")
        lines.append(f"  Q1 MASS at Fib ratios {m:.2%} vs shifted sets {null.mean():.2%} (p95 {np.quantile(null, .95):.2%}) z {z:+.2f}")
        lines.append("     per level (share within +-1.25 pts vs its shifts): " +
                     ", ".join(f"{f}: {a:.2%}/{b:.2%}" for f, (a, b) in per.items()))
        d0, ci, p, pr = q2(touches, rng)
        lines.append(f"  Q2 resume-before-stop at Fib touches minus placebo neighbours: {d0:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p {p:.3f}")
        lines.append("     per level (Fib / neighbours / n Fib touches): " +
                     ", ".join(f"{f}: {a:.1%}/{b:.1%}/{n}" for f, (a, b, n) in pr.items()))
    return lines


def machinery(disc, rng, lines):
    legs = list(disc["1m"][0])
    r = np.array([x["r"] for x in legs])
    k = np.arange(len(r)) % 3 == 0
    r2 = r.copy(); r2[k] = 0.618 + rng.uniform(-0.01, 0.01, k.sum())
    null = np.array([mass(r2, [f + s for f in FIB]) for s in SHIFTS])
    z1 = (mass(r2, FIB) - null.mean()) / null.std()
    alt = [f + 0.035 for f in FIB]
    nul2 = np.array([mass(r, [f + s for f in alt]) for s in SHIFTS if abs(s) >= 0.02])
    z2 = (mass(r, alt) - nul2.mean()) / nul2.std()
    ok = z1 > 3 and abs(z2) < 2
    lines.append(f"MACHINERY M1 planted 0.618 z {z1:+.1f} | M2 shifted 'Fib' set z {z2:+.2f} -> {'ok' if ok else 'FAIL'}")
    return ok


def discovery():
    rng = np.random.default_rng(SEED)
    disc = load(*DISC)
    lines = []
    if not machinery(disc, rng, lines):
        print("\n".join(lines)); return
    lines = report(disc, f"DISCOVERY {DISC[0]}..{DISC[1]}", lines, rng)
    for h, (lo, hi) in (("half 1", (DISC[0], HALF)), ("half 2", ("2025-05-01", DISC[1]))):
        part = {s: ([x for x in L if lo <= x["d"] <= hi], [x for x in T if lo <= x["d"] <= hi]) for s, (L, T) in disc.items()}
        lines = report(part, f"discovery {h}", lines, rng)
    print("\n".join(lines))


def holdout():
    import check_breach_ic as B
    out = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(out):
        sys.exit(f"{out} exists -- scored once")
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)
    hold = load(*HOLD)
    lines = report(hold, f"HOLDOUT {HOLD[0]}..{HOLD[1]} (scored once)", [f"check_fib HOLDOUT {dt.date.today()}"], rng)
    ps, tags = [], []
    for s, (legs, touches) in hold.items():
        m, null, z, _ = q1(legs)
        ps.append(float(np.mean(null >= m))); tags.append(f"Q1 {s} (Fib mass above {np.mean(null < m):.0%} of shifts)")
        d0, ci, p, _ = q2(touches, rng)
        ps.append(p if d0 > 0 else 1.0); tags.append(f"Q2 {s} D {d0:+.3f} CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    q = B.bh(ps)
    for t, pp, qq in zip(tags, ps, q):
        lines.append(f"  {t}: p {pp:.3f} q {qq:.3f} -> {'PASS' if qq < 0.10 else 'FAIL'} (needs the same sign on discovery)")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.discovery:
        discovery()
    if a.holdout:
        holdout()
