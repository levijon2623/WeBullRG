# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.9"]
# ///
"""
check_vol_gauge.py
==================
A CALIBRATED VOLATILITY GAUGE for the viewer: "probability the underlying moves
half a straddle (either way) within the next 60 minutes", shown against the
typical rate for that ticker and time of day. User, 2026-10-03/05: the tail
study (check_ic_tail) found the inputs forecast MAGNITUDE, not direction, and
options price it (check_ic_vol_trade) -- so it is an INDICATOR, not a trade.

check_ic_tail measured it on flow-trigger rows. A gauge is read at EVERY minute,
so this refits it unconditionally, on a 10-minute grid, with inputs the live
bot can actually source.

POPULATION  SPY, QQQ, IWM; every session with a 09:35 ATM straddle S (the
            implied-move cache, same S the live IM band takes at 09:36); grid
            m = 09:40, 09:50, ..., 14:50 (32 per ticker-day; the window m+1..
            m+60 ends by 15:50); >= 55 bars in the window. P (2023-10-12..
            2024-08-19, SPENT pre-sample) training only; slices 1-6 scored.
            Windows on one day overlap -> all inference by calendar-day blocks.

TARGET      BIG = 1 if max(high, m+1..m+60) - close(m) >= 0.5 S
                    or close(m) - min(low, m+1..m+60) >= 0.5 S

INPUTS (all with a live source; ranked as per-ticker trailing percentiles
        over the prior 250 sessions, >= 100 values else 0.5, then - 0.5)
    RVOL      volume(m) / median of minute m over the prior 20 sessions
              (live: live_state _VOLBASE -- the identical definition)
    RVOL15    sum volume(m-14..m) / sum of those minutes' medians
    PATH30    mean |close-to-close| over the last <= 30 minutes x 30 / S (>= 5)
    RANGE     (max - min close since 09:30) / S          (closes: live parity)
    URG       sum |net flow 1m| m-4..m / typical |net flow| (prior 20 days)
    NETGEX    prior-day net GEX                           (engine session_gex)
    VIX       prior VIX close                             (engine session_vix)
    AMP       prior-day amp count 0-3                     (engine session_regime)
    STRADPCT  S / spot at 09:35                           (live IM band)
    GAPABS    |09:30 close - prior session close| / S
  raw, not ranked
    MACRO     CPI / PCE / NFP / FOMC day: +0.5 / -0.5; 0 before 2024-08 (the
              calendar is listed from 2024-08 only -- unknown, not "no")
    CLIM      logit of the CLIMATOLOGY: this ticker's BIG rate at this grid
              slot over the prior 250 sessions (>= 60, else pooled slot rate)

MODEL       L1-penalised logistic regression (sklearn, l1_ratio=1, liblinear).
            C chosen per slice by leave-one-segment-out log-loss inside P +
            earlier slices (segments = P's two halves and each earlier slice)
            over C in logspace(-3, 1, 13). Walk-forward: train on P + slices
            < k, predict slice k, chain the six.

PRE-REGISTERED CRITERIA (OOS = chained slices 1-6)
    G1  ranking: chained Spearman(p, BIG) day-block 95% CI lower > 0 AND above
        the p95 of 20 placebos (non-clock inputs permuted within ticker, CLIM
        kept, C fixed at the real slice's C) -- the inputs beat the clock
    G2  Brier skill vs CLIM, BSS = 1 - Brier(p)/Brier(clim) > 0, day-block
        95% CI lower > 0
    G3  calibration: ECE over 10 equal-count bins of p <= 0.02, and every
        bin's |mean p - observed rate| <= 0.05
    G4  BSS > 0 in >= 5 of 6 slices
    PASS = G1-G4.
    HOW MANY (reported, and the DEPLOY RULE): chained BSS with the model
    capped at 1, 2, 3, 4, 5, 6, 8 and all inputs (the largest C on the grid
    whose fit keeps <= cap). Deploy the smallest cap whose BSS >= 90% of the
    uncapped model's; refit it on everything (P + slices 1-6) for the JSON.

MACHINERY (before scoring)
    M1  causality: every rank and every CLIM uses strictly earlier dates
        (asserted)
    M2  a planted input (= BIG - 0.5) gets the largest |coef| in every slice;
        with every non-clock input constant, |BSS| < 0.01 (CLIM alone)
    M3  coverage per input reported; any < 50% defined is flagged

RECALIBRATION (added for run 2 -- see the RUN LOG): the L1 fit's probability
    is passed through a Platt map, p' = sigmoid(a + b * logit(p)), with a, b
    fitted on the TRAINING rows' out-of-fold predictions (the same
    leave-one-segment-out folds, at the slice's C). Training data only, so
    the OOS calibration check stays a genuine test. Criteria unchanged.

RUN LOG
  2026-10-05 run 1 (no recalibration): 66,631 rows (P 18,740; deployed
    47,891), BIG base rate 40.5%. M2 PASS (planted dominant; clock-only BSS
    -0.003). G1 PASS: chained Spearman +0.401 [+0.380, +0.423], AUC 0.736
    vs clock alone 0.657, placebo p95 +0.266. G2 PASS: BSS +0.081 [+0.066,
    +0.096]. G4 PASS: 6/6 slices (+0.049..+0.128). G3 FAIL: ECE 0.019 (<=
    0.02 ok) but the top decile is UNDER-confident -- pred 73.1% vs observed
    79.4% (gap 0.063 > 0.05); the 9th 62.5 vs 66.2. -> FAIL.
    Inputs: PATH30 dominant (+1.7..+2.0), RVOL15, RVOL, RANGE, URG, MACRO +,
    NETGEX - (positive dealer gamma = smaller moves), CLIM. VIX / AMP / GAPABS
    near zero. Caps: 1 +0.009, 2 +0.040, 3 +0.057, 4 +0.064, 5 +0.066, 6
    +0.072, 8 +0.077, all +0.080; deploy rule -> cap 8.
    The shrinkage under-confidence at the top is what the approved plan's
    recalibration step was for -- the script had left it out. Run 2 adds it
    (above), decided AFTER seeing run 1's G3; logged as such.
  2026-10-05 run 2 (Platt recalibration): Platt b = 0.99..1.06 per slice --
    the training folds show NO under-confidence, so the map changes almost
    nothing. G1 PASS (Spearman +0.401, AUC 0.736), G2 PASS (BSS +0.082
    [+0.067, +0.097]), G4 PASS (6/6). G3 FAIL: ECE 0.015, top decile pred
    73.3% vs observed 79.4% (gap 0.061). -> FAIL. The top-end miss is an
    OOS-only effect; stopped here rather than tune a third time on OOS.
    Caps (recalibrated): 1 -0.001, 2 +0.059, 3 +0.068, 4 +0.073, 5 +0.075,
    6 +0.076, 8 +0.081, all +0.079 -> DEPLOY RULE picks cap 5.
  DIAGNOSTIC, after run 2 (not a criterion, looked at post hoc): the cap-5
    model -- kept CLIM, PATH30, RVOL15 every slice, RVOL 3/6, NETGEX 1/6 --
    BSS +0.075 [+0.061, +0.089], AUC 0.730, ECE 0.010, worst decile gap
    0.026 (top: pred 77.7% / obs 77.6%). Per slice its p >= 65% rows read
    pred 73-78% vs obs 69-81% -- early slices over-, late slices
    under-confident: the top end drifts with the regime.

Usage:
  python check_vol_gauge.py            # the test
  python check_vol_gauge.py --export   # after a PASS: write vol_gauge.json
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_implied_breakout_filters as BF             # noqa: E402
import check_fuzzy_sizing as FZ                         # noqa: E402
import check_ic_screen as IC                            # noqa: E402
import macro_calendar as MC                             # noqa: E402

TICKERS = IC.TICKERS
GRID = list(range(580, 891, 10))                        # 09:40 .. 14:50
BIG_S = 0.5
RANKED = ["RVOL", "RVOL15", "PATH30", "RANGE", "URG", "NETGEX", "VIX", "AMP", "STRADPCT", "GAPABS"]
RAWF = ["MACRO", "CLIM"]
FEATS = RANKED + RAWF
CLOCK = {"CLIM"}
RANK_WIN, RANK_MIN = 250, 100
CLIM_MIN = 60
MACRO_FROM = dt.date(2024, 8, 1)
CS = np.logspace(-3, 1, 13)
CAPS = (1, 2, 3, 4, 5, 6, 8, len(FEATS))
N_PLACEBO = 20
SEED = 20261005
CACHE = os.path.join("_ic_cache", "vol_gauge_rows.pkl")


# ================================================================== rows
def build_rows(D):
    """-> list of dicts: tk, day, m, raw inputs, BIG. Cached (git-ignored)."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    am = BF.am_straddles()
    vix = sorted((dt.date.fromisoformat(k), v) for k, v in json.load(open("_implied_move_vix.json")).items())
    rows = []
    for tk in TICKERS:
        print(f"  {tk}: loading ...", flush=True)
        b = BF.bars_full(tk).to_pandas()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        days = {d: g.set_index("mod") for d, g in b.groupby("date")}
        cal = sorted(days)
        vol = b.pivot_table(index="date", columns="mod", values="volume")
        med = vol.shift(1).rolling(20, min_periods=20).median()
        net, ftyp = FZ.flow_tables(D, tk)
        gx = pd.read_parquet(f"historical/GEX{tk}.parquet")
        gx["date"] = pd.to_datetime(gx["date"]).dt.date
        gser = sorted((d, float(v)) for d, v in zip(gx["date"], gx["net_gex"]) if pd.notna(v))
        amp = D.load_amp_score(SC.HIST, tk)
        for i, day in enumerate(cal):
            if IC.seg(day) is None:
                continue
            st = am.get((tk, day))
            g = days[day]
            if not st or i == 0:
                continue
            spot, S = st
            if not S or S <= 0:
                continue
            prev_close = days[cal[i - 1]]["close"].iloc[-1]
            c = g["close"]
            o = c.get(570)
            gap = abs(o - prev_close) / S if o is not None else None
            vixp = IC.latest_before(vix, day)
            ngx = IC.latest_before(gser, day)
            macro = (0.5 if (MC.is_macro_am_day(day) or MC.is_fomc_day(day)) else -0.5) if day >= MACRO_FROM else 0.0
            typ = ftyp.get(day)
            mrow = med.loc[day] if day in med.index else None
            vrow = vol.loc[day] if day in vol.index else None
            for m in GRID:
                if m not in c.index:
                    continue
                w = g.loc[m + 1:m + 60]
                if len(w) < 55:
                    continue
                c0 = float(c[m])
                big = int(w["high"].max() - c0 >= BIG_S * S or c0 - w["low"].min() >= BIG_S * S)
                F = dict.fromkeys(FEATS)
                if mrow is not None and vrow is not None:
                    md, vv = mrow.get(m), vrow.get(m)
                    if md and np.isfinite(md) and md > 0 and np.isfinite(vv):
                        F["RVOL"] = float(vv / md)
                    ks = [k for k in range(m - 14, m + 1)]
                    mm = np.array([mrow.get(k, np.nan) for k in ks], float)
                    vs = np.array([vrow.get(k, np.nan) for k in ks], float)
                    ok = np.isfinite(mm) & np.isfinite(vs)
                    if ok.sum() >= 10 and mm[ok].sum() > 0:
                        F["RVOL15"] = float(vs[ok].sum() / mm[ok].sum())
                seg = c.loc[max(570, m - 30):m]
                if len(seg) >= 6:
                    F["PATH30"] = float(np.abs(np.diff(seg.to_numpy(float))).mean() * 30 / S)
                sofar = c.loc[570:m]
                if len(sofar):
                    F["RANGE"] = float((sofar.max() - sofar.min()) / S)
                if typ:
                    fl = [net.get((day, k)) for k in range(m - 4, m + 1)]
                    fl = [abs(x) for x in fl if x is not None]
                    if fl:
                        F["URG"] = sum(fl) / typ
                F["NETGEX"] = ngx
                F["VIX"] = vixp
                F["AMP"] = amp.get(day)
                F["STRADPCT"] = S / spot
                F["GAPABS"] = gap
                F["MACRO"] = macro
                rows.append(dict(tk=tk, day=day, m=m, F=F, big=big))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(rows, open(CACHE, "wb"))
    return rows


# ================================================================== memberships + climatology
def rank_inputs(rows):
    """Adds r[f] (ranked - 0.5) for RANKED, CLIM, MACRO. Causal (asserted)."""
    for tk in TICKERS:
        rr = [r for r in rows if r["tk"] == tk]
        days = sorted({r["day"] for r in rr})
        on = {}
        for r in rr:
            on.setdefault(r["day"], []).append(r)
        for f in RANKED:
            vals = {d: [r["F"][f] for r in on[d] if r["F"][f] is not None] for d in days}
            for i, d in enumerate(days):
                win = days[max(0, i - RANK_WIN):i]
                assert all(w < d for w in win)
                ref = sorted(v for w in win for v in vals[w])
                for r in on[d]:
                    v = r["F"][f]
                    r[f] = 0.0 if (v is None or len(ref) < RANK_MIN) else \
                        (bisect.bisect_left(ref, v) + bisect.bisect_right(ref, v)) / 2 / len(ref) - 0.5
        # climatology per slot, trailing 250 sessions
        hist = {m: [] for m in GRID}          # [(day, big)]
        pool = []
        for i, d in enumerate(days):
            lo = days[max(0, i - RANK_WIN)] if i else d
            for r in on[d]:
                h = [b for (dd, b) in hist[r["m"]] if lo <= dd < d]
                assert all(dd < d for (dd, _b) in hist[r["m"]])
                if len(h) >= CLIM_MIN:
                    p = np.mean(h)
                else:
                    ph = [b for (dd, b) in pool if lo <= dd < d]
                    p = np.mean(ph) if len(ph) >= CLIM_MIN else 0.37
                p = min(max(p, 0.02), 0.98)
                r["clim"] = p
                r["CLIM"] = math.log(p / (1 - p))
                r["MACRO"] = r["F"]["MACRO"]
            for r in on[d]:
                hist[r["m"]].append((d, r["big"]))
                pool.append((d, r["big"]))
    return rows


# ================================================================== model
def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=2000).fit(X, y)


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(tr, group, X, y, C):
    """(a, b) of sigmoid(a + b*logit(p)) from out-of-fold training predictions."""
    from sklearn.linear_model import LogisticRegression
    g = group[tr]
    oof = np.empty(len(tr))
    for u in sorted(set(g)):
        a, b = np.where(g != u)[0], np.where(g == u)[0]
        oof[b] = _fit(X[tr[a]], y[tr[a]], C).predict_proba(X[tr[b]])[:, 1]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof)[:, None], y[tr])
    return float(m.intercept_[0]), float(m.coef_[0][0])


def walk(rows, X, y, cap=None, fixed_C=None, recal=True):
    """-> (pred array (nan outside OOS), {slice: C}, {slice: coef}, {slice: (intercept, a, b)})."""
    segs = np.array([IC.seg(r["day"]) for r in rows])
    days = np.array([r["day"] for r in rows])
    pdays = sorted(set(days[segs == -1]))
    pmid = pdays[len(pdays) // 2]
    group = np.where(segs == -1, np.where(days < pmid, -2, -1), segs)
    pred = np.full(len(rows), np.nan)
    Cs, coefs, icpt = {}, {}, {}
    for k in range(6):
        tr = np.where((segs == -1) | ((segs >= 0) & (segs < k)))[0]
        te = np.where(segs == k)[0]
        if len(te) == 0:
            continue
        if fixed_C is not None:
            C = fixed_C[k + 1]
        elif cap is not None:
            C = None
            for c in CS:
                if np.count_nonzero(_fit(X[tr], y[tr], c).coef_) <= cap:
                    C = c
            C = C if C is not None else CS[0]
        else:
            g = group[tr]
            best = None
            for c in CS:
                ll = []
                for u in sorted(set(g)):
                    a, b = tr[g != u], tr[g == u]
                    m = _fit(X[a], y[a], c)
                    ll.append(_logloss(m.predict_proba(X[b])[:, 1], y[b]) * len(b))
                tot = sum(ll)
                if best is None or tot < best[0]:
                    best = (tot, c)
            C = best[1]
        m = _fit(X[tr], y[tr], C)
        raw = m.predict_proba(X[te])[:, 1]
        a, b = platt(tr, group, X, y, C) if recal else (0.0, 1.0)
        pred[te] = 1 / (1 + np.exp(-(a + b * _logit(raw))))
        Cs[k + 1], coefs[k + 1], icpt[k + 1] = C, m.coef_[0].copy(), (float(m.intercept_[0]), a, b)
    return pred, Cs, coefs, icpt


# ================================================================== scoring
def brier_parts(p, q, y, days):
    """BSS = 1 - sum((p-y)^2)/sum((q-y)^2) with a day-block bootstrap CI."""
    a, b = (p - y) ** 2, (q - y) ** 2
    bss = 1 - a.sum() / b.sum()
    ud = sorted(set(days))
    ix = {d: [] for d in ud}
    for i, d in enumerate(days):
        ix[d].append(i)
    A = np.array([a[ix[d]].sum() for d in ud])
    B = np.array([b[ix[d]].sum() for d in ud])
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(1000):
        j = rng.integers(0, len(ud), len(ud))
        bs.append(1 - A[j].sum() / B[j].sum())
    return float(bss), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ece(p, y, nb=10):
    o = np.argsort(p)
    bins = np.array_split(o, nb)
    gaps = [(p[b].mean(), y[b].mean(), len(b)) for b in bins]
    e = sum(abs(a - b) * n for a, b, n in gaps) / len(p)
    return float(e), gaps


def design(rows, feats=FEATS):
    return np.array([[r[f] for f in feats] for r in rows], float)


# ================================================================== main
def main():
    import directional_flow_backtester as D
    rows = build_rows(D)
    rows = sorted(rows, key=lambda r: (r["day"], r["m"], r["tk"]))
    rank_inputs(rows)
    X = design(rows)
    y = np.array([r["big"] for r in rows], float)
    clim = np.array([r["clim"] for r in rows])
    segs = np.array([IC.seg(r["day"]) for r in rows])
    dep = segs >= 0
    print(f"  rows {len(rows)} (P {np.sum(segs == -1)}, deployed {dep.sum()}); "
          f"BIG base rate deployed {y[dep].mean():.1%}, P {y[segs == -1].mean():.1%}")

    # ---- M3 coverage
    print("  M3 coverage (raw defined): " + "  ".join(
        f"{f} {np.mean([r['F'][f] is not None for r in rows]):.0%}" for f in RANKED))
    low = [f for f in RANKED if np.mean([r["F"][f] is not None for r in rows]) < 0.5]
    if low:
        print(f"     FLAG < 50%: {low}")

    # ---- M2
    Xp = X.copy()
    Xp[:, 0] = y - 0.5
    _p, _c, coefs, _i = walk(rows, Xp, y)
    dom = all(int(np.argmax(np.abs(c))) == 0 for c in coefs.values())
    Xc = X.copy()
    Xc[:, [j for j, f in enumerate(FEATS) if f not in CLOCK]] = 0.0
    pc, _c, _cc, _i = walk(rows, Xc, y)
    ok = ~np.isnan(pc)
    b0, _l, _h = brier_parts(pc[ok], clim[ok], y[ok], [rows[i]["day"] for i in np.where(ok)[0]])
    m2 = dom and abs(b0) < 0.01
    print(f"  M2 planted dominant every slice {dom}; clock-only BSS {b0:+.4f} (|.| < 0.01): "
          f"{'PASS' if m2 else 'FAIL'}")
    if not m2:
        sys.exit("machinery checks failed -- not scoring")

    # ---- the model
    pred, Cs, coefs, icpt = walk(rows, X, y)
    ok = ~np.isnan(pred)
    oi = np.where(ok)[0]
    od = [rows[i]["day"] for i in oi]
    p, yy, q = pred[ok], y[ok], clim[ok]
    print("\n  MODEL (walk-forward L1 logistic)")
    print("    C per slice: " + "  ".join(f"{k}:{c:.3g}" for k, c in Cs.items()))
    print("    Platt (a, b) per slice: " + "  ".join(f"{k}:({v[1]:+.3f}, {v[2]:.3f})" for k, v in icpt.items()))
    print(f"    {'input':9} " + " ".join(f"{'s' + str(k):>7}" for k in coefs))
    for j, f in enumerate(FEATS):
        print(f"    {f:9} " + " ".join(f"{coefs[k][j]:+7.3f}" for k in coefs))
    sp, lo, hi, _pv = IC.day_boot_ic(p, yy, od)
    auc_m = auc(p, yy)
    auc_c = auc(q, yy)
    rng = np.random.default_rng(SEED)
    nonclock = [j for j, f in enumerate(FEATS) if f not in CLOCK]
    tk_idx = {tk: np.array([i for i, r in enumerate(rows) if r["tk"] == tk]) for tk in TICKERS}
    pl_ = []
    for _ in range(N_PLACEBO):
        Xq = X.copy()
        for tk, ix in tk_idx.items():
            Xq[np.ix_(ix, nonclock)] = X[np.ix_(rng.permutation(ix), nonclock)]
        pq, _c, _cc, _i = walk(rows, Xq, y, fixed_C=Cs)
        pl_.append(IC.spearman(pq[ok], yy))
    p95 = float(np.quantile(pl_, 0.95))
    bss, blo, bhi = brier_parts(p, q, yy, od)
    e, gaps = ece(p, yy)
    worst = max(abs(a - b) for a, b, _n in gaps)
    per = {}
    for k in range(6):
        ix = np.where(segs[ok] == k)[0]
        per[k + 1] = 1 - ((p[ix] - yy[ix]) ** 2).sum() / ((q[ix] - yy[ix]) ** 2).sum()
    print(f"    chained Spearman {sp:+.4f} [{lo:+.4f}, {hi:+.4f}]  AUC {auc_m:.4f} (clock alone {auc_c:.4f})  "
          f"placebo median {np.median(pl_):+.4f} p95 {p95:+.4f}")
    print(f"    Brier skill vs CLIM {bss:+.4f} [{blo:+.4f}, {bhi:+.4f}]   per slice " +
          " ".join(f"{k}:{v:+.3f}" for k, v in per.items()))
    print(f"    calibration ECE {e:.4f}, worst bin gap {worst:.4f}")
    for a, b, n in gaps:
        print(f"      pred {a:6.1%}  observed {b:6.1%}  n {n}")
    crit = dict(G1=lo > 0 and sp > p95, G2=blo > 0, G3=e <= 0.02 and worst <= 0.05,
                G4=sum(v > 0 for v in per.values()) >= 5)
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> {'PASS' if all(crit.values()) else 'FAIL'}")

    # ---- what it says, by fifth (reported)
    o = np.argsort(p)
    print("    by fifth of the gauge: " + "  ".join(
        f"p {p[b].mean():.0%}/obs {yy[b].mean():.0%}" for b in np.array_split(o, 5)))

    # ---- how many inputs + deploy rule
    print("\n  HOW MANY INPUTS (chained BSS vs CLIM, model capped)")
    res = {}
    for cap in CAPS:
        pc, cc, kc, _i = walk(rows, X, y, cap=cap)
        okc = ~np.isnan(pc)
        b = 1 - ((pc[okc] - y[okc]) ** 2).sum() / ((clim[okc] - y[okc]) ** 2).sum()
        res[cap] = b
        kept = sorted({FEATS[j] for c in kc.values() for j in np.nonzero(c)[0]})
        print(f"    cap {cap:2}: BSS {b:+.4f}   kept (any slice) {', '.join(kept)}")
    pick = min(c for c in CAPS if res[c] >= 0.9 * bss) if bss > 0 else None
    print(f"    DEPLOY RULE: smallest cap with BSS >= 90% of uncapped ({0.9 * bss:+.4f}) -> {pick}")
    return all(crit.values())


EXPORT_CAP = 5           # the deploy rule's pick on run 2
EXPORT_FILE = "vol_gauge.json"
N_Q = 201                # quantile points per ranked input in the export


def export():
    """Refit the cap-EXPORT_CAP model on EVERYTHING (P + slices 1-6) and write
    what live_state needs: inputs, coefficients, intercept, the Platt map
    (out-of-fold over all segments), each ranked input's per-ticker reference
    quantiles (last RANK_WIN sessions) and the per-ticker CLIM table (last
    RANK_WIN sessions per slot). User approved 2026-10-05 as a DISPLAY-ONLY,
    forward-tested gauge after the test FAILED G3 (see RUN LOG)."""
    rows = sorted(build_rows(None), key=lambda r: (r["day"], r["m"], r["tk"]))
    rank_inputs(rows)
    X = design(rows)
    y = np.array([r["big"] for r in rows], float)
    C = None
    for c in CS:
        if np.count_nonzero(_fit(X, y, c).coef_) <= EXPORT_CAP:
            C = c
    m = _fit(X, y, C)
    keep = [j for j in np.nonzero(m.coef_[0])[0]]
    segs = np.array([IC.seg(r["day"]) for r in rows])
    days = np.array([r["day"] for r in rows])
    pdays = sorted(set(days[segs == -1]))
    group = np.where(segs == -1, np.where(days < pdays[len(pdays) // 2], -2, -1), segs)
    a, b = platt(np.arange(len(rows)), group, X, y, C)
    last = sorted(set(days))[-RANK_WIN:]
    lo = last[0]
    ref, clim = {}, {}
    for tk in TICKERS:
        rr = [r for r in rows if r["tk"] == tk and r["day"] >= lo]
        ref[tk] = {}
        for j in keep:
            f = FEATS[j]
            if f in RANKED:
                v = np.array([r["F"][f] for r in rr if r["F"][f] is not None], float)
                ref[tk][f] = [round(float(x), 6) for x in np.quantile(v, np.linspace(0, 1, N_Q))]
        clim[tk] = {str(s): round(float(np.mean([r["big"] for r in rr if r["m"] == s])), 4) for s in GRID}
    out = dict(
        version=dt.date.today().isoformat(), fit_through=str(max(days)), rank_from=str(lo),
        target=f"P(|move| >= {BIG_S} x 09:35 ATM straddle within 60 min)",
        grid=GRID, big_s=BIG_S, C=float(C), intercept=float(m.intercept_[0]),
        coef={FEATS[j]: float(m.coef_[0][j]) for j in keep}, platt=dict(a=a, b=b),
        ranked=[FEATS[j] for j in keep if FEATS[j] in RANKED],
        ref=ref, clim=clim, high=0.65,
        note="check_vol_gauge.py --export; test FAILED G3 (top-decile calibration), "
             "deployed display-only and forward-tested by user decision 2026-10-05")
    json.dump(out, open(EXPORT_FILE, "w"), indent=1)
    print(f"  wrote {EXPORT_FILE}: C {C:.4g}, inputs {list(out['coef'])}, "
          f"coef {', '.join(f'{k} {v:+.3f}' for k, v in out['coef'].items())}, "
          f"intercept {out['intercept']:+.3f}, Platt ({a:+.3f}, {b:.3f}), ranks from {lo}")
    # self-check: the export, applied the way live_state will, reproduces the fit
    p_fit = m.predict_proba(X)[:, 1]
    idx = [i for i, r in enumerate(rows) if r["day"] >= lo]
    p_exp = np.array([gauge_p(out, rows[i]["tk"], rows[i]["m"], rows[i]["F"], raw=True) for i in idx])
    d = np.abs(p_exp - p_fit[idx])
    print(f"  export vs in-memory fit over the last {RANK_WIN} sessions: median |diff| {np.median(d):.4f}, "
          f"p99 {np.quantile(d, 0.99):.4f}  (quantile ranks vs exact trailing ranks)")


def gauge_p(G, tk, m, F, raw=False):
    """The live formula, shared with live_state's copy: -> probability or None.
    F holds the RAW input values; CLIM comes from the table by slot."""
    slots = [s for s in G["grid"] if s <= m]
    if not slots or m > G["grid"][-1] + 9:
        return None
    pc = min(max(G["clim"][tk][str(slots[-1])], 0.02), 0.98)
    z = G["intercept"]
    for f, w in G["coef"].items():
        if f == "CLIM":
            x = math.log(pc / (1 - pc))
        elif f in G["ranked"]:
            v = F.get(f)
            q = G["ref"][tk][f]
            x = 0.0 if v is None else float(np.interp(v, q, np.linspace(0, 1, len(q)))) - 0.5
        else:
            x = F.get(f) or 0.0
        z += w * x
    p = 1 / (1 + math.exp(-z))
    if raw:
        return p
    a, b = G["platt"]["a"], G["platt"]["b"]
    return 1 / (1 + math.exp(-(a + b * math.log(p / (1 - p)))))


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


if __name__ == "__main__":
    export() if "--export" in sys.argv else main()
