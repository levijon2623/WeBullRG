# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_implied_move_gate.py
==========================
Does the IMPLIED MOVE band (implied_move.py; calibrated by check_implied_move)
work as an ENTRY GATE for the flow trigger? The user's hypothesis (2026-10-03):
"calls only at or below the day's implied low, puts only at or above the high"
-- buy the trigger only once price is stretched to the edge of the range the
options priced.

POWER CHECK FIRST (counts only, before any scoring, 2026-10-03)
    The DEPLOYED book cannot answer this: its SPY/QQQ/IWM rules are all CALLs,
    and of their 148 sequential trades 0 (SPY), 1 (QQQ) and 21 (IWM, 10 days,
    5 OOS) entered at/below the 65% low. The flow trigger is momentum -- calls
    sat at/above the 65% HIGH more often (0 / 11 / 11). So the test runs on the
    SCREEN, where 79-123 days per ticker x direction carry a stretched p50
    trigger (about half OOS).

POPULATION -- a SCREEN, not the deployed rules
    SPY, QQQ, IWM x CALL, PUT x min_flow_pct 50 / 80 / 90 = 18 cells; hours
    9-14, entry >= 09:35, dte [0, 1], ATM, no regime or AMT gate, book-default
    exit (policy_for -> trail50). Sequential one-position-per-ticker, fill
    "bot" + cushion, 2024-08-20 .. 2026-09-18 (the straddle lake's end).
    Only triggers with a band are used (all arms see the same population).

FEATURE (no lookahead)
    spot = the entry spot build_candidates uses (lake underlying_close at the
    trigger minute). Band = that day's WALK-FORWARD 65% band from
    check_implied_move (quantiles from prior days only): the 09:35 band for
    entries from 09:36, the pre-open (prior 15:55) band for a 09:35 entry.
    STRETCHED  CALL with spot <= low65,  PUT with spot >= high65
    CHASING    CALL with spot >= high65, PUT with spot <= low65

POLICIES (a skip keeps the book flat; the next trigger is eligible at once)
    KEEP-STRETCHED   the user's gate                        -- PRIMARY
    SKIP-STRETCHED   its mirror; scored because METHODOLOGY 7 says score both
    SKIP-CHASING     "don't chase past the band" -- SECONDARY, registered after
                     the power check showed calls sitting at the high (a count
                     of locations, no outcomes seen)

CRITERIA, each policy on its own (fixed before the first scored run)
    G1  pooled OOS per-trade gain vs the unfiltered screen       >= +5pp
    G2  pooled IS gain                                            >= 0
    G3  pooled OOS TOTAL of the filtered book >= unfiltered total
        (lifting the mean by cutting winners is the twins trap)
    G4  G1's gain > p95 of 20 placebo gains. PLACEBO = the band WIDTHS (% of
        anchor) of ANOTHER day of the same ticker, hung on TODAY's anchor:
        keeps how often a trade is "stretched" and destroys only what that
        day's options priced.
    G5  OOS gain > 0 in >= 2/3 of cells with >= 10 kept OOS trades
    G6  G1's sign holds under fill = mid, bot AND worst
    G7  first-trigger-per-day sample (cap=1): OOS gain > 0
    G8  OOS gain > 0 in >= 2 of 3 tickers (pooled over their 6 cells)
    PASS = G1..G8. If KEEP- and SKIP-STRETCHED both pass, FAIL (broken).

REPORTED, NOT SCORED
    day-block bootstrap 90% CI of each pooled OOS gain; kept n / days; win
    rate; the deployed SPY/QQQ/IWM CALL rules under each policy (too thin).

MACHINERY (--selftest; also run before scoring)
    M1  every band used for an entry from 09:36 is the 09:35 band and every
        09:35 entry uses the pre-open band; the walk-forward quantile for day
        d uses only days < d (asserted in check_implied_move.walk)
    M2  a feature planted to equal "trade won" gives a large keep gain; a
        constant feature gives exactly 0 gain (keep-all == unfiltered)
    M3  the placebo derangement maps no day to itself
    M4  meta spot vs the historical 1m close at the trigger minute: median
        |diff| <= 0.03% (the location is measured on the right price)

RUN LOG
  2026-10-03: M1-M4 PASS. 57,608 located candidates, 14.8% stretched, 14.7%
  chasing. Unfiltered screen n=11,415, IS -12.3%, OOS -8.9%.
    KEEP-STRETCHED  FAIL. OOS -14.6% (gain -5.7pp, 90% CI -12.0..+1.0), IS
        -3.3pp, every ticker negative (SPY -7.2 QQQ -3.5 IWM -7.0), every fill,
        first-trigger -5.9pp, win 0.19. Stretched entries are WORSE: the trigger
        is momentum and does not turn into mean reversion at the band.
        But the placebo (another day's widths on today's anchor) loses nearly as
        much (median -4.4pp): the damage is mostly "price already far from the
        09:35 anchor", not what the day's options priced.
    SKIP-STRETCHED  FAIL. +0.9pp, inside its placebo (median +0.8, p95 +1.3).
    SKIP-CHASING    FAIL. -0.4pp.
    G3 passes for all three only because the screen loses money -- fewer
    trades lose less. Deployed rules: SPY CHOP CALL 0 stretched, QQQ 1 of 32,
    IWM 24 of 105 (OOS +13.8% vs +29.8% all) -- the gate mostly switches them off.

Usage:
  python check_implied_move_gate.py --selftest
  python check_implied_move_gate.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                 # noqa: E402
import check_implied_move as CI       # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
CELLS = [(tk, d, p) for tk in TICKERS for d in ("CALL", "PUT") for p in (50, 80, 90)]
AFTER = 9 * 60 + 35
AM_FROM = 9 * 60 + 36
N_PLACEBO = 20
SEED = 20261003
FILLS = ("mid", "bot", "worst")
CACHE = os.path.join("_implied_move_cache", "gate_candidates.pkl")   # git-ignored folder


# ------------------------------------------------------------------ bands
def bands():
    """{(tk, day, fam): (A, up_pct_hat, dn_pct_hat)} at tau .65, walk-forward."""
    st = CI.straddles()
    B = {t: CI.bars(t) for t in TICKERS}
    cal = sorted(set().union(*(set(b) for b in B.values())))
    out = {}
    for fam in ("am", "pm"):
        for tk in TICKERS:
            rows, _ = CI.build_family(fam, st, tk, B[tk], cal)
            for x in CI.walk(rows):
                if x["pred"]:
                    p = x["pred"][0.65]
                    out[(tk, x["day"], fam)] = (x["A"], p["up_hat"], p["dn_hat"])
    return out, B


def level(bd, tk, day, mod, perm=None):
    """(low65, high65) for an entry at `mod`, or None. `perm` = {day: source
    day}: the placebo takes the SOURCE day's widths on TODAY's anchor."""
    fam = "am" if mod >= AM_FROM else "pm"
    own = bd.get((tk, day, fam))
    if own is None:
        return None
    A = own[0]
    up, dn = own[1], own[2]
    if perm is not None:
        src = bd.get((tk, perm[day], fam))
        if src is None:
            return None
        up, dn = src[1], src[2]
    return A * (1 - dn / 100), A * (1 + up / 100)


def derangement(days, rng):
    days = list(days)
    while True:
        p = list(rng.permutation(days))
        if all(a != b for a, b in zip(days, p)):
            return dict(zip(days, p))


# ------------------------------------------------------------------ candidates
def screen_rule(tk, direction, pct):
    return dict(name=f"{tk} {direction} p{pct}", ticker=tk, direction=direction,
                hours=list(range(9, 15)), min_flow_pct=pct, dte=[0, 1],
                target_roe=1.0, rr=1.0)


def candidates(D):
    """{cell: [(date, mod, payload, spot)]}, one build per ticker x direction
    at p50; p80/p90 are subsets by each trigger's own trailing threshold."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from check_config_walkforward import _flow_for
    out = {}
    for tk in TICKERS:
        trigs = D.triggers_for(_flow_for(D, [tk]), tk)
        D.annotate_flow_pct(trigs, 60)
        by_min = {(t["date"], t["dir"], pd.Timestamp(t["ts"]).hour * 60 + pd.Timestamp(t["ts"]).minute): t
                  for t in trigs}
        for direction in ("CALL", "PUT"):
            meta = []
            print(f"  build {tk} {direction} ...", flush=True)
            base = SC.build_candidates(D, screen_rule(tk, direction, 50), trigs=trigs, meta_out=meta)
            for pct in (50, 80, 90):
                keep = []
                for c, m in zip(base, meta):
                    t = by_min.get((c[0], direction, c[1]))
                    if t and t.get("thr") and t["abs_flow"] >= t["thr"][pct]:
                        keep.append((c[0], c[1], c[2], m["spot"]))
                out[(tk, direction, pct)] = keep
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def located(cands, bd, perm=None):
    """Drop triggers without a band; tag each with STRETCHED / CHASING / None."""
    out = {}
    for (tk, direction, pct), rows in cands.items():
        keep = []
        for d, m, payload, spot in rows:
            lv = level(bd, tk, d, m, (perm or {}).get(tk))
            if lv is None:
                continue
            lo, hi = lv
            if direction == "CALL":
                loc = "S" if spot <= lo else "C" if spot >= hi else None
            else:
                loc = "S" if spot >= hi else "C" if spot <= lo else None
            keep.append((d, m, payload, loc))
        out[(tk, direction, pct)] = keep
    return out


POLICIES = {
    "ALL": lambda loc: True,
    "KEEP-STRETCHED": lambda loc: loc == "S",
    "SKIP-STRETCHED": lambda loc: loc != "S",
    "SKIP-CHASING": lambda loc: loc != "C",
}


def run(loc_cands, policy, fill="bot", cap=None):
    pol = SC.policy_for(screen_rule("SPY", "CALL", 50))
    keep = POLICIES[policy] if isinstance(policy, str) else policy
    res = {}
    for cell, rows in loc_cands.items():
        cand = [(d, m, p) for d, m, p, loc in rows if keep(loc)]
        res[cell] = SC.walk(cand, pol, SC.DEFAULT_EOD, fill=fill, cap=cap, after=AFTER)
    return res


def pooled(res, cells=None):
    return [r for k in (cells or CELLS) for r in res[k]]


def mean_of(rows, oos):
    v = [r[1] for r in rows if (r[0] >= SC.SPLIT) == oos]
    return float(np.mean(v)) if v else float("nan")


def total_of(rows, oos=True):
    return float(sum(r[1] for r in rows if (r[0] >= SC.SPLIT) == oos))


def boot_gain(base, filt, n=2000, seed=SEED):
    """Day-block bootstrap 90% CI of OOS mean(filtered) - mean(unfiltered)."""
    rng = np.random.default_rng(seed)
    bd, fd = {}, {}
    for d, p in ((r[0], r[1]) for r in base if r[0] >= SC.SPLIT):
        bd.setdefault(d, []).append(p)
    for d, p in ((r[0], r[1]) for r in filt if r[0] >= SC.SPLIT):
        fd.setdefault(d, []).append(p)
    days = sorted(bd)
    if not days:
        return float("nan"), float("nan")
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days))
        b = [x for i in pick for x in bd[days[i]]]
        f = [x for i in pick for x in fd.get(days[i], [])]
        if b and f:
            out.append(np.mean(f) - np.mean(b))
    return float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))


# ------------------------------------------------------------------ checks
def selftest(D, bd, cands, B):
    ok = True
    # M1: level() must hand a 09:35 entry the PRE-OPEN band and a 09:36 entry
    # the 09:35 band (whose anchor is the 09:35 close -- not known at 09:35)
    bad = n = 0
    for (tk, d, fam) in list(bd)[:4000]:
        if fam != "am" or (tk, d, "pm") not in bd:
            continue
        n += 1
        A, u, dn = bd[(tk, d, "pm")]
        Aa, ua, da = bd[(tk, d, "am")]
        want_pm = (A * (1 - dn / 100), A * (1 + u / 100))
        want_am = (Aa * (1 - da / 100), Aa * (1 + ua / 100))
        bad += level(bd, tk, d, AFTER) != want_pm or level(bd, tk, d, AM_FROM) != want_am
    print(f"  M1 09:35 -> pre-open band, 09:36 -> 09:35 band on {n} ticker-days: "
          f"{'PASS' if bad == 0 and n else 'FAIL'} ({bad} bad)")
    ok &= bad == 0 and n > 0
    # M2: planted and constant features
    lc = located(cands, bd)
    base = pooled(run(lc, "ALL"))
    pol = SC.policy_for(screen_rule("SPY", "CALL", 50))
    planted = {}
    for cell, rows in lc.items():
        won = []
        for d, m, p, loc in rows:
            pnl, _x, _t = SC.simulate(p, pol, SC.DEFAULT_EOD, fill="bot")
            won.append((d, m, p, "S" if pnl > 0 else None))
        planted[cell] = won
    g_plant = mean_of(pooled(run(planted, "KEEP-STRETCHED")), True) - mean_of(base, True)
    const = {cell: [(d, m, p, "S") for d, m, p, _l in rows] for cell, rows in lc.items()}
    g_const = mean_of(pooled(run(const, "KEEP-STRETCHED")), True) - mean_of(base, True)
    m2 = g_plant > 0.20 and abs(g_const) < 1e-12
    print(f"  M2 planted 'won' feature gain {g_plant * 100:+.1f}pp (must be large); constant {g_const * 100:+.4f}pp "
          f"(must be 0): {'PASS' if m2 else 'FAIL'}")
    ok &= m2
    # M3: derangement
    rng = np.random.default_rng(SEED)
    days = sorted({d for (_t, d, _f) in bd})
    p = derangement(days, rng)
    m3 = all(a != b for a, b in p.items())
    print(f"  M3 derangement has no fixed point: {'PASS' if m3 else 'FAIL'}")
    ok &= m3
    # M4: spot agreement with the 1m close
    diffs = []
    for (tk, direction, pct), rows in cands.items():
        if pct != 50:
            continue
        px = {d: dict(zip(g["mod"].to_list(), g["close"].to_list())) for d, g in B[tk].items()}
        for d, m, _p, s in rows[::7]:
            c = (px.get(d) or {}).get(m)
            if c:
                diffs.append(abs(s - c) / c * 100)
    med = float(np.median(diffs)) if diffs else 9.0
    print(f"  M4 entry spot vs 1m close: median |diff| {med:.4f}% on {len(diffs)} (<= 0.03%): "
          f"{'PASS' if med <= 0.03 else 'FAIL'}")
    ok &= med <= 0.03
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import directional_flow_backtester as D
    bd, B = bands()
    last = max(d for (_t, d, _f) in bd)
    cands = candidates(D)
    cands = {k: [r for r in v if r[0] <= last] for k, v in cands.items()}
    print(f"  bands to {last}; candidates per cell: " +
          " ".join(f"{k[0]}{k[1][0]}{k[2]}={len(v)}" for k, v in cands.items()))
    if not selftest(D, bd, cands, B):
        sys.exit("machinery checks failed -- not scoring")
    if a.selftest:
        return

    lc = located(cands, bd)
    nS = sum(1 for rows in lc.values() for r in rows if r[3] == "S")
    nC = sum(1 for rows in lc.values() for r in rows if r[3] == "C")
    nA = sum(len(rows) for rows in lc.values())
    print(f"  located candidates {nA}: stretched {nS} ({nS / nA:.1%}), chasing {nC} ({nC / nA:.1%})")

    rng = np.random.default_rng(SEED)
    perms = []
    for _ in range(N_PLACEBO):
        perms.append({tk: derangement(sorted({d for (t, d, _f) in bd if t == tk}), rng) for tk in TICKERS})

    res = {pol: {f: run(lc, pol, fill=f) for f in FILLS} for pol in POLICIES}
    first = {pol: run(lc, pol, cap=1) for pol in POLICIES}
    base = pooled(res["ALL"]["bot"])
    b_is, b_oos, b_tot = mean_of(base, False), mean_of(base, True), total_of(base)
    print(f"\n  UNFILTERED screen (fill=bot): n={len(base)}  IS {b_is * 100:+.1f}%  OOS {b_oos * 100:+.1f}%  "
          f"OOS total {b_tot:+.2f}")
    verdict = {}
    for pol in ("KEEP-STRETCHED", "SKIP-STRETCHED", "SKIP-CHASING"):
        rows = pooled(res[pol]["bot"])
        g1 = mean_of(rows, True) - b_oos
        g2 = mean_of(rows, False) - b_is
        g3 = total_of(rows) - b_tot
        pg = []
        for perm in perms:
            plc = located(cands, bd, perm)
            pr = pooled(run(plc, pol))
            pb = pooled(run(plc, "ALL"))
            pg.append(mean_of(pr, True) - mean_of(pb, True))
        p95 = float(np.nanquantile(pg, 0.95))
        cell_g = []
        for cell in CELLS:
            kept_oos = [r for r in res[pol]["bot"][cell] if r[0] >= SC.SPLIT]
            if len(kept_oos) >= 10:
                cell_g.append(mean_of(res[pol]["bot"][cell], True) - mean_of(res["ALL"]["bot"][cell], True))
        g5 = sum(x > 0 for x in cell_g)
        fills = {f: mean_of(pooled(res[pol][f]), True) - mean_of(pooled(res["ALL"][f]), True) for f in FILLS}
        g7 = mean_of(pooled(first[pol]), True) - mean_of(pooled(first["ALL"]), True)
        tick = {}
        for tk in TICKERS:
            cs = [c for c in CELLS if c[0] == tk]
            tick[tk] = mean_of(pooled(res[pol]["bot"], cs), True) - mean_of(pooled(res["ALL"]["bot"], cs), True)
        lo, hi = boot_gain(base, rows)
        crit = dict(G1=g1 >= 0.05, G2=g2 >= 0, G3=g3 >= 0, G4=g1 > p95,
                    G5=bool(cell_g) and g5 >= 2 / 3 * len(cell_g),
                    G6=all(np.sign(v) == np.sign(g1) and v != 0 for v in fills.values()),
                    G7=g7 > 0, G8=sum(v > 0 for v in tick.values()) >= 2)
        verdict[pol] = all(crit.values())
        n_oos = sum(1 for r in rows if r[0] >= SC.SPLIT)
        print(f"\n  {pol}  n={len(rows)} (OOS {n_oos}, days {len({r[0] for r in rows if r[0] >= SC.SPLIT})})  "
              f"IS {mean_of(rows, False) * 100:+.1f}%  OOS {mean_of(rows, True) * 100:+.1f}%  "
              f"win {np.mean([r[1] > 0 for r in rows]):.2f}")
        print(f"    G1 OOS gain {g1 * 100:+.1f}pp (90% CI {lo * 100:+.1f}..{hi * 100:+.1f})   G2 IS gain {g2 * 100:+.1f}pp   "
              f"G3 OOS total {total_of(rows):+.2f} vs {b_tot:+.2f}")
        print(f"    G4 placebo p95 {p95 * 100:+.1f}pp (median {np.nanmedian(pg) * 100:+.1f})   "
              f"G5 cells {g5}/{len(cell_g)}   G6 " + " ".join(f"{f} {v * 100:+.1f}" for f, v in fills.items()) +
              f"   G7 first-trigger {g7 * 100:+.1f}pp")
        print("    G8 " + "  ".join(f"{t} {v * 100:+.1f}pp" for t, v in tick.items()))
        print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
              f"   -> {'PASS' if verdict[pol] else 'FAIL'}")
    if verdict["KEEP-STRETCHED"] and verdict["SKIP-STRETCHED"]:
        print("\n  both KEEP- and SKIP-STRETCHED passed -- impossible unless broken: FAIL")

    print("\n  DEPLOYED SPY/QQQ/IWM CALL rules (reported only -- too thin):")
    for rule in SC.research_rules():
        tk = rule["ticker"]
        if tk not in TICKERS:
            continue
        meta = []
        cand = SC.build_candidates(D, rule, meta_out=meta)
        for pol in ("ALL", "KEEP-STRETCHED", "SKIP-STRETCHED", "SKIP-CHASING"):
            keep = POLICIES[pol]
            rows = []
            for c, m in zip(cand, meta):
                if c[0] > last:
                    continue
                lv = level(bd, tk, c[0], c[1])
                if lv is None:
                    continue
                loc = "S" if m["spot"] <= lv[0] else "C" if m["spot"] >= lv[1] else None
                if keep(loc):
                    rows.append(c)
            print(SC.line(f"{rule['name']} {pol}",
                          SC.walk(rows, SC.policy_for(rule), SC.eod_mod(rule), fill="bot"), width=34))


if __name__ == "__main__":
    main()
