"""
check_poc_stack.py
==================
STACKED POCs AND OVERLAPPING VALUE AREAS AS SUPPORT (user, 2026-10-09): "how
much more meaningful, if at all, a POC can be when it overlaps with another
from within the past month -- with 2, 3, 4 ... And what about overlapping VA?
As to measuring meaningful, I reckon we're talking support."

WHY THIS IS NOT A REPEAT. Single prior-day POC/VA (check_amt, 0/48) and
1-3-week COMPOSITE POC/VAH/VAL (check_multiweek_amt) behaved like random
prices. A composite blurs separate auctions together; STACKING asks whether
several separate sessions independently found fair value at the same price.
The claim is a DOSE-RESPONSE: support improves with the number of stacked
POCs (or overlapping value areas). Flat across counts is the null.

LEVELS (only sessions BEFORE day D -- nothing from D itself)
    Each complete session's own profile, exactly check_multiweek_amt's
    construction (integer-cent bins of 0.05% of the median price, 70% VA).
    Look-back: the 20 sessions before D (~a month).
    POC STACK  k = how many of those 20 POCs lie within TAU of a POC (itself
               included, so a lone POC has k = 1). TAU = 0.10% of price
               (user's choice; 0.05% reported). Levels within TAU of each
               other are one level: the higher k is kept (then the newer).
               Buckets 1, 2, 3, 4+.
    VA DEPTH   depth(p) = how many of the 20 value areas contain price p.
               An UPPER EDGE is where depth drops going up; its depth is the
               count just inside. Buckets 1, 2, 3, 4, 5+.
EVENT       the first 1m bar on D whose range contains the level, 09:46-14:59
            (check_multiweek_amt.first_touch). SUPPORT = approached from
            above (POC: s = +1; VA: an upper edge touched from above, i.e.
            price coming down INTO stacked value). From below = resistance,
            reported.
OUTCOMES    HOLD30 (primary): over the 30 minutes from the touch, no 1m close
            more than TAU beyond the level on the far side.
            BOUNCE (reported): +1/4 of the day's 09:35 straddle back on the
            approach side before -1/4 straddle through, within 60 minutes.
            r15 / r30 / r60 (reported): s x (close - level) / level, bp.
CONTROL     placebo levels at +-0.2..2.0% (0.1% steps) of every real level,
            none within 2 bins of ANY prior-session POC/VAH/VAL, first-touched
            by the same rule; each real event matched to up to 5 placebo events
            of the same day, ticker, approach side, session half and approach-
            speed tercile (check_multiweek_amt's design). effect = real -
            mean(matched placebos), per set.

PRE-REGISTERED CRITERIA, each study (POC stack, VA depth), SUPPORT side,
HOLD30, TAU = 0.10%; SPY + QQQ + IWM pooled; ISO-week block bootstrap
(1,000) -- a stacked level persists across days, so touches cluster by week.
    S1 DOSE-RESPONSE  OLS slope of the set effect on the bucket (top bucket
       capped) > 0 with bootstrap p5 > 0
    S2 TOP BUCKET     effect > 0 with bootstrap p5 > 0 (if it has < 50 sets,
       the top two buckets are merged -- decided before any outcome)
    S3 STABLE         top-bucket effect > 0 in both halves (split 2025-08-21)
       and in >= 4 of 6 equal-count calendar slices
    PASS = S1 and S2 and S3. A pass makes stacked levels a candidate input
    (e.g. drawn on the chart / a gate), not a strategy.
MACHINERY   M1 planting +0.30 on the HOLD30 of real top-bucket events must
            pass S1 and S2; M2 every level comes from sessions < D (asserted).
DATA        historical/{SPY,QQQ,IWM}.parquet 1m bars 2023-10-12 .. 2026-09-25,
            the same set check_multiweek_amt used. No API requests.

RUN LOG
  2026-10-09 run 1 ABORTED at M1 by a slices bug (np.array_split on a
    DataFrame). Before it stopped it showed two things: (a) only ~1/3 of
    support touches found matched placebos (583 POC sets of 1,825), so the
    4+ bucket fell under 50 sets and was merged into 3+ (38 sets) by the
    pre-registered rule; (b) a DESIGN FLAW -- M1's table printed REAL and
    PLACEBO hold rates by bucket (1: 0.224 vs 0.274, 2: 0.255 vs 0.251,
    3+: 0.263 vs 0.271), i.e. outcomes were seen before scoring. Fixes for
    run 2, none touching the primary design: the slices bug; M1 now plants
    on the SCORED top bucket (it planted on k>=4 while 3+ was scored, so it
    diluted its own check); matched-fraction diagnostics; and a LOOSE-MATCH
    robustness (ticker, day, side only) -- REPORTED, added after (b).
  2026-10-09 run 2: M1 PASS (planted top bucket +0.293, S1-S3 pass).
    Matched sets: POC 583 of 1,825 support touches (32%), VA 579 of 1,951.
    PRIMARY, both FAIL:
      POC STACK (tau 0.10%), HOLD30 real vs placebo: k=1 0.224 vs 0.274
        (-0.050), k=2 0.255 vs 0.251 (+0.004), 3+ (38 sets) 0.263 vs 0.271
        (-0.007, p5..p95 -0.20..+0.17). Slope +0.03, CI spans 0. S1-S3 fail.
      VA DEPTH (upper edge from above): effects +0.002 / +0.028 / +0.030 /
        -0.038 / +0.021 for depth 1..5+; slope -0.001. Flat. S1-S3 fail.
    No dose-response in either: more stacked POCs or more overlapping value
    areas do not make a level hold better than a random price reached the
    same way. Reported outcomes (bounce, r30, tau 0.05%) agree -- all fail.
    LOOSE MATCH (post hoc): stacked levels hold LESS than random -- POC 3+
      -0.151 (CI -0.27..-0.03), VA 4 / 5+ -0.08 / -0.09, slope -0.03 -- the
      same faint "levels get crossed" lean check_multiweek_amt saw at 1-2w.
    ONE REPORTED CELL PASSES, unscored: POC STACK as RESISTANCE (approached
      from below), 3+ = 24 sets: held 0.500 vs placebo 0.198, +0.302 (p5
      +0.108), both halves and 5/6 slices positive, slope p5 > 0. One cell
      of ~15 reported tables, 24 sets, not pre-registered as a hypothesis --
      a forward-test lead at most, not a finding.

Usage:  python check_poc_stack.py [--counts]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_multiweek_amt as CM                        # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
LOOK = 20
TAUS = (0.0010, 0.0005)
TAU0 = 0.0010
SPLIT = dt.date(2025, 8, 21)
PLACEBO = CM.PLACEBO
K_CTRL, N_BOOT, MIN_TOP = 5, 1000, 50
POC_CAP, VA_CAP = 4, 5
HOLD_H, BOUNCE_H, BOUNCE_S = 30, 60, 0.25
SEED = 20261009
rng = np.random.default_rng(SEED)


def straddle_frac():
    import check_implied_breakout_filters as BF
    return {k: s / spot for k, (spot, s) in BF.am_straddles().items() if spot and s}


def outcomes(b, f, s, L, tau, q):
    """HOLD30, BOUNCE (None without a straddle), r15/r30/r60 in bp."""
    c, h, l = b["gc"], b["gh"], b["gl"]
    seg = c[f:f + HOLD_H + 1]
    hold = float(np.all(s * (seg - L) / L >= -tau))
    bounce = None
    if q:
        up, dn = L * (1 + s * BOUNCE_S * q), L * (1 - s * BOUNCE_S * q)
        bounce = 0.0
        for t in range(f + 1, min(f + 1 + BOUNCE_H, 390)):
            fav = h[t] >= up if s > 0 else l[t] <= up
            adv = l[t] <= dn if s > 0 else h[t] >= dn
            if adv:
                break
            if fav:
                bounce = 1.0
                break
    r = [s * (c[f + hh] - L) / L * 1e4 for hh in (15, 30, 60)]
    return hold, bounce, r


def poc_levels(pocs, tau):
    """[(level, k)] deduplicated: within tau = one level, highest k (then newest) wins."""
    p = np.array(pocs)
    k = np.array([int(np.sum(np.abs(p - x) <= tau * x)) for x in p])
    order = sorted(range(len(p)), key=lambda i: (-k[i], -i))
    out = []
    for i in order:
        if all(abs(p[i] - L) > tau * L for L, _ in out):
            out.append((float(p[i]), int(k[i])))
    return out


def va_edges(vals, vahs, binw):
    """[(price, depth_inside, 'upper'|'lower')] on the bin-edge grid."""
    lo = min(vals) - binw
    hi = max(vahs) + binw
    grid = np.arange(lo, hi + binw / 2, binw / 2)         # half-bin resolution
    depth = np.array([sum(1 for a, z in zip(vals, vahs) if a <= g <= z) for g in grid])
    out = []
    for i in range(len(grid) - 1):
        if depth[i] > depth[i + 1]:
            out.append((float((grid[i] + grid[i + 1]) / 2), int(depth[i]), "upper"))
        elif depth[i + 1] > depth[i]:
            out.append((float((grid[i] + grid[i + 1]) / 2), int(depth[i + 1]), "lower"))
    return out


def build(tk, q_of):
    binw, days, _info = CM.load_days(tk)
    keys = list(days)
    lo = min(float(b["l"].min()) for b in days.values())
    hi = max(float(b["h"].max()) for b in days.values())
    base = int(CM._bin(lo, binw)) - 2
    nb = int(CM._bin(hi, binw)) - base + 3
    lv = [CM.levels_from_hist(CM.day_hist(days[d], binw, base, nb), binw, base) for d in keys]
    real, plc = [], []
    for j, d in enumerate(keys):
        b = days[d]
        if b["kind"] != "full" or j < LOOK:
            continue
        prior = [i for i in range(j - LOOK, j) if lv[i]]
        assert all(i < j for i in prior)                  # M2
        if len(prior) < LOOK - 2:
            continue
        pocs = [lv[i]["poc"] for i in prior]
        vals = [lv[i]["val"] for i in prior]
        vahs = [lv[i]["vah"] for i in prior]
        q = q_of.get((tk, d))
        allreal = np.array(pocs + vals + vahs)
        cands = []
        for tau in TAUS:
            cands += [("POC", tau, L, k, None) for L, k in poc_levels(pocs, tau)]
        cands += [("VA", None, L, k, side) for L, k, side in va_edges(vals, vahs, binw)]
        for study, tau, L, k, side in cands:
            e = CM.first_touch(b, L)
            if e is None:
                continue
            f, s, _r, app = e
            hold, bounce, r = outcomes(b, f, s, L, TAU0 if tau is None else tau, q)
            role = ("support" if s > 0 else "resistance") if study == "POC" else \
                   ("support" if (side == "upper" and s > 0) else
                    "resistance" if (side == "lower" and s < 0) else "inside")
            real.append(dict(tk=tk, day=d, study=study, tau=tau or TAU0, k=k, role=role, f=f, s=s,
                             pm=int(f >= 150), app=app, hold=hold, bounce=bounce,
                             r15=r[0], r30=r[1], r60=r[2]))
        seen = set()
        for L0 in allreal:
            for dd in PLACEBO:
                P = round(L0 * (1 + dd) / binw) * binw
                if P in seen or np.min(np.abs(allreal - P)) <= 2 * binw:
                    continue
                seen.add(P)
                e = CM.first_touch(b, P)
                if e is None:
                    continue
                f, s, _r, app = e
                for tau in TAUS:
                    hold, bounce, r = outcomes(b, f, s, P, tau, q)
                    plc.append(dict(tk=tk, day=d, tau=tau, s=s, pm=int(f >= 150), app=app, hold=hold,
                                    bounce=bounce, r15=r[0], r30=r[1], r60=r[2]))
    return pd.DataFrame(real), pd.DataFrame(plc)


def terciles(real, plc):
    """Approach-speed terciles per ticker, cut on IS real events."""
    for tk in TICKERS:
        x = real.loc[(real["tk"] == tk) & (real["day"] < SPLIT), "app"].to_numpy()
        a, b = np.quantile(x, [1 / 3, 2 / 3])
        for df in (real, plc):
            m = df["tk"] == tk
            df.loc[m, "at"] = np.where(df.loc[m, "app"] < a, 0, np.where(df.loc[m, "app"] > b, 2, 1))
    return real, plc


def matched(real, plc, cols=("hold", "bounce", "r15", "r30", "r60")):
    """Per real event: real value minus mean of up to K_CTRL matched placebos."""
    keys = ["tk", "day", "tau", "s", "pm", "at"]
    g = {k: v for k, v in plc.groupby(keys)}
    rows = []
    for _, e in real.iterrows():
        pool = g.get(tuple(e[k] for k in keys))
        if pool is None or not len(pool):
            continue
        pick = pool.sample(min(K_CTRL, len(pool)), random_state=int(rng.integers(1e9)))
        out = dict(e)
        for c in cols:
            pv = pick[c].dropna()
            out[f"d_{c}"] = (e[c] - pv.mean()) if (e[c] is not None and not pd.isna(e[c]) and len(pv)) else np.nan
            out[f"p_{c}"] = pv.mean() if len(pv) else np.nan
        rows.append(out)
    df = pd.DataFrame(rows)
    df["week"] = [d.isocalendar()[:2] for d in df["day"]]
    return df


def boot(df, stat, n=N_BOOT):
    weeks = sorted(set(df["week"]))
    idx = {w: np.flatnonzero((df["week"] == w).to_numpy()) for w in weeks}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[weeks[i]] for i in rng.integers(0, len(weeks), len(weeks))])
        out.append(stat(df.iloc[pick]))
    return np.array(out)


def slope(df, col="d_hold"):
    x, y = df["kb"].to_numpy(float), df[col].to_numpy(float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(set(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def score(df, cap, label, col="d_hold"):
    df = df[np.isfinite(df[col])].copy()
    df["kb"] = np.minimum(df["k"], cap)
    top = cap
    if (df["kb"] == top).sum() < MIN_TOP:
        top = cap - 1
        df["kb"] = np.minimum(df["kb"], top)
    print(f"\n  {label}  (buckets 1..{top}{'+' if top == cap or True else ''}, outcome {col[2:]})")
    print(f"    {'k':>4} {'sets':>6} {'real':>7} {'placebo':>8} {'effect':>8}   halves (IS / OOS)")
    pcol = "p_" + col[2:]
    for k in sorted(df["kb"].unique()):
        m = df["kb"] == k
        is_, oos = df[m & (df["day"] < SPLIT)][col].mean(), df[m & (df["day"] >= SPLIT)][col].mean()
        print(f"    {str(int(k)) + ('+' if k == top else ''):>4} {m.sum():6} {df.loc[m, col[2:]].mean():7.3f} "
              f"{df.loc[m, pcol].mean():8.3f} {df.loc[m, col].mean():+8.3f}   {is_:+.3f} / {oos:+.3f}")
    sl = slope(df, col)
    bs = boot(df, lambda x: slope(x, col))
    t = df[df["kb"] == top]
    te = t[col].mean()
    bt = boot(t, lambda x: x[col].mean())
    halves = [t[t["day"] < SPLIT][col].mean(), t[t["day"] >= SPLIT][col].mean()]
    o = t.sort_values("day")
    sls = [o.iloc[ix][col].mean() for ix in np.array_split(np.arange(len(o)), 6)]
    crit = dict(S1=bool(sl > 0 and np.nanquantile(bs, 0.05) > 0),
                S2=bool(te > 0 and np.nanquantile(bt, 0.05) > 0),
                S3=bool(all(h > 0 for h in halves) and sum(v > 0 for v in sls) >= 4))
    print(f"    slope per bucket {sl:+.4f} (week-boot p5 {np.nanquantile(bs, .05):+.4f}, p95 {np.nanquantile(bs, .95):+.4f})")
    print(f"    top bucket effect {te:+.4f} (p5 {np.nanquantile(bt, .05):+.4f}, p95 {np.nanquantile(bt, .95):+.4f}); "
          f"halves {halves[0]:+.3f} / {halves[1]:+.3f}; slices " + " ".join(f"{v:+.3f}" for v in sls))
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> {'PASS' if all(crit.values()) else 'FAIL'}")
    return all(crit.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    a = ap.parse_args()
    q_of = straddle_frac()
    R, P = [], []
    for tk in TICKERS:
        r, p = build(tk, q_of)
        print(f"  {tk}: {len(r)} real first touches, {len(p)} placebo", flush=True)
        R.append(r); P.append(p)
    real, plc = terciles(pd.concat(R, ignore_index=True), pd.concat(P, ignore_index=True))

    print("\n  COUNTS -- support-side first touches by bucket (before any outcome)")
    for study, tau, cap in (("POC", TAU0, POC_CAP), ("POC", 0.0005, POC_CAP), ("VA", TAU0, VA_CAP)):
        m = (real["study"] == study) & (real["tau"] == tau) & (real["role"] == "support")
        kb = np.minimum(real.loc[m, "k"], cap)
        print(f"    {study} tau {tau:.2%}: " + "  ".join(f"k={int(k)}{'+' if k == cap else ''}: {int((kb == k).sum())}"
                                                       for k in sorted(kb.unique())))
    if a.counts:
        return

    df = matched(real, plc)
    sup = df[df["role"] == "support"]
    for study, tau in (("POC", TAU0), ("VA", TAU0)):
        m = (real["study"] == study) & (real["tau"] == tau) & (real["role"] == "support")
        n = int(((sup["study"] == study) & (sup["tau"] == tau) & np.isfinite(sup["d_hold"])).sum())
        print(f"  matched sets {study}: {n} of {int(m.sum())} support touches ({n / max(m.sum(), 1):.0%})")

    # ---- M1
    pm = sup[(sup["study"] == "POC") & (sup["tau"] == TAU0)].copy()
    pm = pm[np.isfinite(pm["d_hold"])]
    top = POC_CAP if (np.minimum(pm["k"], POC_CAP) == POC_CAP).sum() >= MIN_TOP else POC_CAP - 1
    pm.loc[pm["k"] >= top, "d_hold"] += 0.30
    print("\n  M1 -- planted +0.30 HOLD30 on top-bucket POC events:")
    ok = score(pm, POC_CAP, "M1 planted")
    if not ok:
        sys.exit("machinery check failed -- not scoring")

    print("\n  ======== PRIMARY (support, HOLD30, tau 0.10%) ========")
    v1 = score(sup[(sup["study"] == "POC") & (sup["tau"] == TAU0)], POC_CAP, "POC STACK")
    v2 = score(sup[sup["study"] == "VA"], VA_CAP, "VA DEPTH (upper edge from above)")
    print(f"\n  VERDICT: POC stack {'PASS' if v1 else 'FAIL'}; VA depth {'PASS' if v2 else 'FAIL'}")

    print("\n  ======== REPORTED ========")
    score(sup[(sup["study"] == "POC") & (sup["tau"] == 0.0005)], POC_CAP, "POC STACK, tau 0.05%")
    for col in ("d_bounce", "d_r30"):
        score(sup[(sup["study"] == "POC") & (sup["tau"] == TAU0)], POC_CAP, "POC STACK", col)
        score(sup[sup["study"] == "VA"], VA_CAP, "VA DEPTH", col)
    # LOOSE MATCH (reported; added AFTER run 1's M1 table showed outcomes --
    # see RUN LOG): placebos matched on ticker, day, tau and approach side only
    rl = real.copy(); pl_ = plc.copy()
    rl["pm"] = 0; rl["at"] = 0; pl_["pm"] = 0; pl_["at"] = 0
    dl = matched(rl, pl_)
    sl_ = dl[dl["role"] == "support"]
    print("\n  ---- LOOSE MATCH (ticker, day, side only) -- reported, post hoc ----")
    score(sl_[(sl_["study"] == "POC") & (sl_["tau"] == TAU0)], POC_CAP, "POC STACK, loose match")
    score(sl_[sl_["study"] == "VA"], VA_CAP, "VA DEPTH, loose match")
    res = df[df["role"] == "resistance"]
    score(res[(res["study"] == "POC") & (res["tau"] == TAU0)], POC_CAP, "POC STACK as RESISTANCE")
    score(res[res["study"] == "VA"], VA_CAP, "VA DEPTH as RESISTANCE (lower edge from below)")


if __name__ == "__main__":
    main()
