# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_ema_stack_signal.py
=========================
STAGE A of the BloodHound-style study (SharkIndicators: a simple trigger plus
a fuzzy-logic confidence layer; user, 2026-10-03). EXIT-FREE: does a 5/8/13
EMA stack on 3-minute bars, and a fuzzy score of seven inputs, predict the
UNDERLYING? No options, fills or exits -- trade management (stage B: trail50,
stack-break, underlying TP/SL brackets, re-entry / scalping) is designed only
if this finds something to manage.

THE TRIGGER
    3-minute bars from the 1m RTH bars, aligned to 09:30 (09:30-09:32,
    09:33-09:35, ...); a bar's close = its last minute's close. EMA 5/8/13 of
    those closes, carried across days. The stack FORMS when e5 > e8 > e13
    becomes true (CALL, s = +1) or e5 < e8 < e13 becomes true (PUT, s = -1)
    at a bar close; a stack that persists does not refire. Trigger minute m =
    the bar's last minute, 09:36..14:59. SPY, QQQ, IWM.

THE OUTCOME -- in units of that day's 09:35 ATM straddle S
    r_H   = s x (close(m + H) - close(m)) / S          H = 15, 30, 60 min
    MFE_H = best 1m high (CALL) / low (PUT) in (m, m+H], same units
    MAE_H = worst, same units
    A trigger whose m + H passes 15:59 has no r_H. PRIMARY: r_30.
    Rough economics (reported only): an ATM option is ~S/2 with delta ~0.5,
    so a move of r straddles is ~r x 100% on the option BEFORE theta and the
    ~10pp entry/exit cost -- breakeven is very roughly r ~ +0.2.

WINDOWS
    P     2023-10-12 .. 2024-08-19  the SPENT pre-sample (PRESAMPLE_PLAN 6a):
          labelled, TRAINING ONLY for the fuzzy weights, never scored
    1..6  METHODOLOGY 8 slices, 2024-08-20 .. 2026-08-21 (HEDGE{T}'s end);
          IS < 2025-08-21 <= OOS
    Inference: day-block bootstrap (whole sessions resampled), because
    forward windows of nearby triggers overlap (METHODOLOGY 2a).

PART S -- THE BARE SIGNAL (deployed slices 1..6)
    S1  mean r_30 > 0, day-block 95% CI lower bound > 0
    S2  mean r_30 > 0 in IS AND in OOS
    S3  mean r_30 > 0 in >= 5 of 6 slices
    S4  mean r_30 > p95 of 20 NAIVE-MOMENTUM baselines: at random minutes of
        the same ticker-days (time-of-day drawn from the triggers' own
        distribution), direction = sign of the last 9 minutes' return.
        Is the stack more than "follow the last 9 minutes"?
    S PASSES = S1..S4.

PART Q -- THE FUZZY LAYER (check_fuzzy_sizing's seven inputs, unchanged:
    urgency, IM stretch, wall ahead, hedge pressure, VWAP distance, RVOL,
    the fixed time-of-day curve; per-ticker trailing-rank memberships)
    For slice k = 1..6: weights = Spearman(membership, r_30) on P + slices
    < k; score = sum w_i (m_i - 0.5); thirds cut on the training scores.
    Chain the six scored slices.
    Q1  chained Spearman(score, r_30) > 0, day-block 95% CI lower bound > 0
    Q2  Q1's Spearman > p95 of 20 placebos (memberships permuted within ticker)
    Q3  top-third minus bottom-third mean r_30 > 0 in >= 4 of 6 slices
    Q PASSES = Q1..Q3.

VERDICT: stage B is worth building if S or Q passes. If only Q passes, the
score finds good triggers inside a trigger that is flat on average -- the
BloodHound premise exactly.

REPORTED, NOT SCORED: r_15 / r_60, MFE / MAE, triggers per day, weights per
slice, top-third mean r_30 against the ~0.2 breakeven, per ticker.

MACHINERY (before scoring)
    M1  3-minute bars rebuilt independently from the 1m bars match on a
        sample of days (close = last minute, bars aligned to 09:30)
    M2  a stack "forms" only on a transition (the previous bar's state
        differs), never on a persisting stack
    M3  the fuzzy weights of a planted input (= r_30 > 0) dominate every
        slice; constant memberships give Spearman 0
    M4  every trigger has S (dropped otherwise, counted)

RUN LOG
  2026-10-03 run 1: M1-M4 PASS. 17,027 formations (7.9 per ticker-day), 16,586
    with r_30; 11,941 in the deployed slices.
    PART S FAIL (S1-S3). r_30 +0.0021 S, 95% CI [-0.0026, +0.0067]; IS -0.0014,
      OOS +0.0055; slices -0.004 -0.001 +0.000 +0.002 +0.001 +0.014. MFE and MAE
      are mirror images at every horizon (30m: +0.233 / -0.233) -- a random
      walk's signature. S4 "passes" only because 9-minute momentum at random
      minutes is slightly NEGATIVE (median -0.0063 S: short-term reversion).
    PART Q FAIL (Q1). Spearman +0.0158, CI [-0.0048, +0.0366]; placebo p95
      +0.0152 (Q2 passes by a hair); top-bottom third positive in 4/6 slices.
      Thirds: bottom -0.0038, middle +0.0044, top +0.0056 S -- the best third
      is ~1/35 of the rough +0.2 S an ATM option needs to cover theta and costs.
      Stable weights again: STRETCH + and VWAP + (with-the-move a touch
      better), TIME - ; everything |rho| <= 0.04.
    No stage B: there is no directional edge to manage, and a mean 15-minute
    favourable excursion of +0.166 S (matched by -0.168 adverse) leaves no
    room for 0DTE option scalps after costs.

Usage:
  python check_ema_stack_signal.py --build     # extend the wall cache into P
  python check_ema_stack_signal.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_implied_breakout_filters as BF             # noqa: E402
import check_fuzzy_sizing as FZ                         # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
P_START = dt.date(2023, 10, 12)
END = BF.END
FIRST, LAST = 9 * 60 + 36, 14 * 60 + 59
HS = (15, 30, 60)
N_PLACEBO = 20
SEED = 20261007


# ================================================================== bars + trigger
def day_bars(tk):
    """{date: DataFrame[mod, close, high, low]} RTH 1m, P_START .. END."""
    d = BF.bars_full(tk).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d[(d["date"] >= P_START) & (d["date"] <= END)]
    return {k: g.set_index("mod")[["close", "high", "low"]] for k, g in d.groupby("date")}


def bars3(B):
    """[(date, last_mod, close)] 3-minute bars in time order (all days)."""
    out = []
    for day in sorted(B):
        g = B[day]
        b = ((g.index.to_numpy() - 570) // 3)
        last = g.groupby(b).tail(1)
        for mod, c in zip(last.index.to_numpy(int), last["close"].to_numpy(float)):
            out.append((day, int(mod), c))
    return out


def triggers(tk, B):
    """[(tk, dir, date, mod)] stack formations, 09:36..14:59."""
    b3 = bars3(B)
    c = pd.Series([x[2] for x in b3])
    e = {n: c.ewm(span=n, adjust=False).mean().to_numpy() for n in (5, 8, 13)}
    st = np.where((e[5] > e[8]) & (e[8] > e[13]), 1, np.where((e[5] < e[8]) & (e[8] < e[13]), -1, 0))
    out = []
    for i in range(1, len(b3)):
        day, mod, _c = b3[i]
        if st[i] != 0 and st[i] != st[i - 1] and FIRST <= mod <= LAST:
            out.append((tk, "CALL" if st[i] > 0 else "PUT", day, mod))
    return out, b3, st


def outcomes(trig, B, S):
    """{key: {r15, r30, r60, mfe30, mae30, ...}} in straddle units."""
    out = {}
    for key in trig:
        tk, d, day, m = key
        s = 1 if d == "CALL" else -1
        g, st = B.get(day), S.get((tk, day))
        if g is None or not st:
            continue
        S_ = st[1]
        c0 = g["close"].get(m)
        if c0 is None:
            continue
        rec = {}
        for H in HS:
            if m + H > 959 or (m + H) not in g.index:
                rec[f"r{H}"] = None
                continue
            seg = g.loc[m + 1:m + H]
            rec[f"r{H}"] = s * (g.at[m + H, "close"] - c0) / S_
            fav = seg["high"].max() if s > 0 else seg["low"].min()
            adv = seg["low"].min() if s > 0 else seg["high"].max()
            rec[f"mfe{H}"] = s * (fav - c0) / S_
            rec[f"mae{H}"] = s * (adv - c0) / S_
        out[key] = rec
    return out


def segment(day):
    if day < SC.DEPLOYED_START:
        return "P"
    return BF.slice_of(day)


def order(day):
    """P -> -1 (always training), slices -> 0..5, outside -> None."""
    s = segment(day)
    return -1 if s == "P" else s


# ================================================================== stats
def day_boot(vals_by_day, fn, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    days = sorted(vals_by_day)
    bs = []
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        bs.append(fn([x for d in pick for x in vals_by_day[d]]))
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def momentum_baseline(trig, B, S, rng):
    """Mean r_30 of 'follow the last 9 minutes' at random minutes of the same
    ticker-days, minutes drawn from the triggers' own time-of-day profile."""
    mins = {}
    for tk, _d, _day, m in trig:
        mins.setdefault(tk, []).append(m)
    vals = []
    for tk, _d, day, _m in trig:
        if segment(day) == "P":
            continue
        g, st = B.get(day), S.get((tk, day))
        if g is None or not st:
            continue
        m = int(rng.choice(mins[tk]))
        if m + 30 > 959 or (m - 9) not in g.index or (m + 30) not in g.index or m not in g.index:
            continue
        s = np.sign(g.at[m, "close"] - g.at[m - 9, "close"])
        if s == 0:
            continue
        vals.append(s * (g.at[m + 30, "close"] - g.at[m, "close"]) / st[1])
    return float(np.mean(vals)) if vals else float("nan")


def spearman(x, y):
    return FZ.spearman(x, y)


def fuzzy_walk(keys, mem, y):
    """Walk-forward over slices 0..5 (P always in training). -> (scored rows
    [(key, score, third, slice)], weights per slice)."""
    rows, weights = [], {}
    for k in range(6):
        train = [q for q in keys if order(q[2]) is not None and order(q[2]) < k]
        test = [q for q in keys if order(q[2]) == k]
        if not train or not test:
            continue
        w = {i: spearman([mem[q][i] for q in train], [y[q] for q in train]) for i in FZ.INPUTS}
        weights[k + 1] = w

        def score(q):
            return sum(w[i] * (mem[q][i] - 0.5) for i in FZ.INPUTS)
        sc = np.array([score(q) for q in train])
        q1, q2 = np.quantile(sc, 1 / 3), np.quantile(sc, 2 / 3)
        for q in test:
            s = score(q)
            rows.append((q, s, 0 if s < q1 else 2 if s > q2 else 1, k + 1))
    return rows, weights


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    am = BF.am_straddles()
    if a.build:
        BF.build_walls(am, start=P_START)
        return
    import directional_flow_backtester as D
    walls = pickle.load(open(BF.WALL_CACHE, "rb"))

    trig, B, st_by = [], {}, {}
    for tk in TICKERS:
        B[tk] = day_bars(tk)
        t, b3, st = triggers(tk, B[tk])
        trig += t
        st_by[tk] = (b3, st)

    # ---- M1: 3-minute bars vs an independent resample
    ok1 = True
    for tk in TICKERS:
        b3 = st_by[tk][0]
        for day in sorted(B[tk])[::97]:
            g = B[tk][day]
            ref = g["close"].groupby((g.index.to_numpy() - 570) // 3).last()
            mine = [c for d_, m_, c in b3 if d_ == day]
            ok1 &= np.allclose(ref.to_numpy(), mine)
    print(f"  M1 3-minute bars match an independent resample: {'PASS' if ok1 else 'FAIL'}")
    # ---- M2: formations only on transitions
    bad = 0
    for tk in TICKERS:
        b3, st = st_by[tk]
        idx = {(d_, m_): i for i, (d_, m_, _c) in enumerate(b3)}
        for _t, d, day, m in (x for x in trig if x[0] == tk):
            i = idx[(day, m)]
            bad += st[i] == st[i - 1] or st[i] != (1 if d == "CALL" else -1)
    print(f"  M2 formations are transitions only: {'PASS' if bad == 0 else 'FAIL'} ({bad} bad of {len(trig)})")

    out = {}
    for tk in TICKERS:
        out.update(outcomes([q for q in trig if q[0] == tk], B[tk], am))
    dropped = len(trig) - len(out)
    keys = [q for q in trig if q in out and out[q]["r30"] is not None]
    print(f"  M4 triggers {len(trig)}, without a straddle or bar {dropped}, with r_30 {len(keys)}")

    # ---- PART S
    dep = [q for q in keys if segment(q[2]) != "P"]
    by_day = {}
    for q in dep:
        by_day.setdefault((q[0], q[2]), []).append(out[q]["r30"])
    r = np.array([out[q]["r30"] for q in dep])
    lo, hi = day_boot(by_day, np.mean)
    is_ = [out[q]["r30"] for q in dep if q[2] < SC.SPLIT]
    oo = [out[q]["r30"] for q in dep if q[2] >= SC.SPLIT]
    sl = {k: [out[q]["r30"] for q in dep if segment(q[2]) == k] for k in range(6)}
    nd = len({(q[0], q[2]) for q in dep})
    print(f"\n  PART S -- the bare 5/8/13 stack (deployed slices; {len(dep)} triggers on {nd} ticker-days, "
          f"{len(dep) / nd:.1f}/day)")
    for H in HS:
        v = [out[q][f"r{H}"] for q in dep if out[q][f"r{H}"] is not None]
        mf = [out[q][f"mfe{H}"] for q in dep if out[q].get(f"mfe{H}") is not None]
        ma = [out[q][f"mae{H}"] for q in dep if out[q].get(f"mae{H}") is not None]
        print(f"    H={H:2}: mean r {np.mean(v):+.4f} S  (median {np.median(v):+.4f})  MFE {np.mean(mf):+.3f}  "
              f"MAE {np.mean(ma):+.3f}  n={len(v)}")
    print(f"    r_30 mean {r.mean():+.4f} S, day-block 95% CI [{lo:+.4f}, {hi:+.4f}];  IS {np.mean(is_):+.4f}  "
          f"OOS {np.mean(oo):+.4f};  slices " + " ".join(f"{np.mean(v):+.3f}" for v in sl.values() if v))
    for tk in TICKERS:
        v = [out[q]["r30"] for q in dep if q[0] == tk]
        print(f"      {tk}: r_30 {np.mean(v):+.4f} S  n={len(v)}")
    rng = np.random.default_rng(SEED)
    base = []
    for _ in range(N_PLACEBO):
        vals = []
        for tk in TICKERS:
            tq = [q for q in keys if q[0] == tk]
            v = momentum_baseline(tq, B[tk], am, rng)
            vals.append((v, len([q for q in tq if segment(q[2]) != "P"])))
        base.append(sum(v * n for v, n in vals) / sum(n for _v, n in vals))
    b95 = float(np.quantile(base, 0.95))
    crit_s = dict(S1=lo > 0, S2=np.mean(is_) > 0 and np.mean(oo) > 0,
                  S3=sum(np.mean(v) > 0 for v in sl.values() if v) >= 5, S4=r.mean() > b95)
    print(f"    naive 9-min momentum at random minutes: median {np.median(base):+.4f} S, p95 {b95:+.4f} S")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit_s.items()) +
          f"   -> PART S {'PASS' if all(crit_s.values()) else 'FAIL'}")

    # ---- PART Q
    print("\n  inputs ...", flush=True)
    raw = {}
    for tk in TICKERS:
        net, ftyp = FZ.flow_tables(D, tk)
        hedge, htyp = FZ.hedge_tables(tk)
        px = FZ.price_tables(tk)
        rv, _hl = BF.series_features(tk)
        ctx = (net, ftyp, hedge, htyp, px, rv, walls, am)
        for q in keys:
            if q[0] == tk:
                raw[q] = FZ.raw_inputs(tk, q[1], q[2], q[3], ctx)
    mem = FZ.memberships(raw)
    cov = {i: np.mean([raw[q][i] is not None for q in keys]) for i in FZ.INPUTS}
    print("  input coverage: " + "  ".join(f"{i[:2]} {v:.0%}" for i, v in cov.items()))
    y = {q: out[q]["r30"] for q in keys}

    planted = {q: dict(mem[q], I1_URGENCY=1.0 if y[q] > 0 else 0.0) for q in keys}
    _rp, wp = fuzzy_walk(keys, planted, y)
    dom = all(max(w, key=lambda i: abs(w[i])) == "I1_URGENCY" for w in wp.values())
    const = {q: dict.fromkeys(FZ.INPUTS, 0.5) for q in keys}
    rc, _wc = fuzzy_walk(keys, const, y)
    zero = abs(spearman([x[1] for x in rc], [y[x[0]] for x in rc])) < 1e-12
    print(f"  M3 planted input dominant every slice: {dom};  constant memberships -> Spearman 0: {zero}  "
          f"-> {'PASS' if dom and zero else 'FAIL'}")
    if not (ok1 and bad == 0 and dom and zero):
        sys.exit("machinery checks failed -- not scoring")

    rows, weights = fuzzy_walk(keys, mem, y)
    rho = spearman([x[1] for x in rows], [y[x[0]] for x in rows])
    pairs = {}
    for q, s, _t, _k in rows:
        pairs.setdefault((q[0], q[2]), []).append((s, y[q]))
    lo_q, hi_q = day_boot(pairs, lambda ps: spearman([p[0] for p in ps], [p[1] for p in ps]))
    print("\n  PART Q -- the fuzzy layer on r_30")
    print("    weights (Spearman with r_30, P + earlier slices):")
    print("    slice " + " ".join(f"{i[:9]:>10}" for i in FZ.INPUTS))
    for k, w in weights.items():
        print(f"    {k:5} " + " ".join(f"{w[i]:+10.3f}" for i in FZ.INPUTS))
    third_d, top = {}, []
    for k in range(1, 7):
        rr = [x for x in rows if x[3] == k]
        if rr:
            t0 = [y[x[0]] for x in rr if x[2] == 0]
            t2 = [y[x[0]] for x in rr if x[2] == 2]
            third_d[k] = (np.mean(t2) - np.mean(t0)) if t0 and t2 else float("nan")
    for t in (0, 1, 2):
        v = [y[x[0]] for x in rows if x[2] == t]
        print(f"    third {['bottom', 'middle', 'top'][t]:6}: n={len(v):5}  mean r_30 {np.mean(v):+.4f} S")
    print(f"    chained Spearman {rho:+.4f}, day-block 95% CI [{lo_q:+.4f}, {hi_q:+.4f}];  top-bottom by slice: " +
          " ".join(f"{k}:{v:+.3f}" for k, v in third_d.items()))
    rng = np.random.default_rng(SEED + 1)
    pr = []
    for _ in range(N_PLACEBO):
        pm = {}
        for tk in TICKERS:
            ks = [q for q in keys if q[0] == tk]
            perm = rng.permutation(len(ks))
            for a_, b_ in zip(ks, perm):
                pm[a_] = mem[ks[b_]]
        rp, _w = fuzzy_walk(keys, pm, y)
        pr.append(spearman([x[1] for x in rp], [y[x[0]] for x in rp]))
    p95 = float(np.quantile(pr, 0.95))
    crit_q = dict(Q1=lo_q > 0, Q2=rho > p95, Q3=sum(v > 0 for v in third_d.values()) >= 4)
    print(f"    placebo Spearman median {np.median(pr):+.4f}, p95 {p95:+.4f}")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit_q.items()) +
          f"   -> PART Q {'PASS' if all(crit_q.values()) else 'FAIL'}")
    print(f"\n  STAGE A VERDICT: {'BUILD STAGE B' if all(crit_s.values()) or all(crit_q.values()) else 'FAIL -- nothing to manage'}")


if __name__ == "__main__":
    main()
