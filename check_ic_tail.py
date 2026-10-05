# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.4"]
# ///
"""
check_ic_tail.py
================
check_ic_screen asked whether any input moves the flow trigger's MEAN 30-minute
outcome: 0 of 32 did, and Lasso kept no input. But the book's edge is the RIGHT
TAIL (METHODOLOGY 7: trades above +50% are 379% of total P&L), and an input can
leave the mean untouched while changing the odds of a big run. Same triggers,
same 32 inputs, same machinery -- a TAIL target. (User, 2026-10-03.)

POPULATION  check_ic_screen.build_dataset: every p50 flow trigger, SPY/QQQ/IWM,
            CALL+PUT, 09:35-14:59; P (spent pre-sample) training-only; slices
            1-6 scored. Triggers with a full 60-minute window (>= 55 bars).

TARGETS (units: that day's 09:35 ATM straddle S; s = trade direction)
    PRIMARY   HIT  = 1 if MFE60 >= +0.5 S: within 60 minutes the underlying
              runs half a straddle the trade's way (~+100% on an ATM option,
              which is worth ~S/2, before theta)
    SECONDARY MFE60, continuous (rank IC)
    REPORTED  ADV  = 1 if MAE60 <= -0.5 S (the adverse tail). An input that
              raises HIT and ADV alike is a VOLATILITY proxy (more movement,
              either way); one that raises HIT alone is DIRECTIONAL. Straddle
              units already price the expected volatility, so a vol proxy can
              still pay a long option -- but it is labelled as such.
              Option outcomes of the trigger's own ATM contract (book exit,
              standalone, deployed window): mean ROE, hit50, loss50 by third.

PART 1 -- SINGLE-FACTOR TAIL SCREEN
    IC = Spearman(membership, HIT) per slice and pooled (deployed); p by
    calendar-day bootstrap (1,000). PASS = Benjamini-Hochberg q < 0.10 over
    the 32 AND the pooled sign in >= 5 of 6 slices. Reported: IC with ADV and
    MFE60, the HIT rate in the input's top vs bottom third (lift), P (labelled).

PART 2 -- LASSO ON THE TAIL, walk-forward (check_ic_screen.lasso_walk: alpha
    by leave-one-segment-out CV inside P + earlier slices; a linear-probability
    model on HIT, used only to RANK)
    T1  chained Spearman(prediction, HIT) > 0, day-block 95% CI lower > 0
    T2  > p95 of 20 placebos (input rows permuted within ticker)
    T3  > 0 in >= 4 of 6 slices
    T4  top-third (cut on training predictions) mean OPTION ROE minus the
        all-trigger mean ROE > 0 (the money, not the proxy)
    PASS = T1-T4. Reported: AUC, inputs kept per slice, coefficients, HIT /
    ADV / hit50 / loss50 by third, and chained IC with the model capped at
    1, 2, 3, 5, 8, 13, all inputs.

MACHINERY: M1 the dataset reproduces check_ic_screen's 58,575 triggers;
M2 a planted input (= HIT) dominates every Lasso slice, constant inputs keep
no coefficient.

RUN LOG
  2026-10-03 run 1: M1 (58,575) M2 PASS. 58,541 triggers with a 60m window,
    44,463 deployed. Base rates HIT 19.4%, ADV 19.6% (both 1.3%).
    PART 1: 16 of 32 pass. Strongest are VOLATILITY proxies -- they raise HIT
      and ADV alike: RVOL +0.095 (lift 1.55, ADV +0.099), URGENCY +0.068,
      TIME +0.051, IVLVL, AMP, VIX +, NETGEX / TERM -. Directional (HIT up,
      ADV down): VANNA +0.044 / -0.044, CUMALIGN +0.025 / -0.021. Directional
      NEGATIVE (aligned = fewer favourable tails, more adverse): DEX, DIR
      (calls), AMTLOC, POCDIST, CHARM -- the tail reverts where the MEAN
      (check_ic_screen) leaned with the move.
    PART 2 PASS (T1-T4). Chained Spearman +0.114, CI [+0.094, +0.132], AUC
      0.583, positive in 6/6 slices (+0.098..+0.135); placebo p95 -0.003.
      Inputs kept grew 5 -> 20 over the slices (RVOL, URGENCY, DIR, SWEEP10,
      TIME from the start). By third: HIT 14.2 / 20.0 / 24.5%, but ADV 17.3 /
      18.7 / 23.2% -- mostly MAGNITUDE, a ~4pp directional residue.
      T4 passes by +0.7pp only: option mean ROE -12.7 / -9.6 / -10.2%, and
      hit50 does NOT rise (11.6 / 11.7 / 10.7%). The score predicts how far
      the underlying moves (in straddle units); the book's ATM option with a
      trail50 exit does not monetise that.
    HOW MANY: caps 1 / 2 / 3 / 5 / 8 / 13 / all -> +0.025 / +0.088 / +0.098 /
      +0.107 / +0.114 / +0.117 / +0.105. Two inputs carry most of it; past
      ~13 it declines.

Usage:
  python check_ic_tail.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_ic_screen as IC                            # noqa: E402

HIT_S = 0.5
N_PLACEBO = 20
SEED = 20261009


def auc(score, y):
    import pandas as pd
    r = pd.Series(score).rank().to_numpy()
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import directional_flow_backtester as D
    keys, raw, _y30, tails = IC.build_dataset(D)
    print(f"  M1 dataset triggers {len(keys)} (check_ic_screen: 58,575): "
          f"{'PASS' if len(keys) == 58575 else 'FAIL'}")
    keys = sorted((k for k in keys if k in tails), key=lambda k: (k[2], k[3], k[0], k[1]))
    mem = IC.memberships({k: raw[k] for k in keys})
    X = np.array([[mem[k][f] - (0.5 if f in IC.SIGNED + IC.UNSIGNED else 0.0) for f in IC.FEATS] for k in keys])
    hit = np.array([1.0 if tails[k]["mfe60"] >= HIT_S else 0.0 for k in keys])
    adv = np.array([1.0 if tails[k]["mae60"] <= -HIT_S else 0.0 for k in keys])
    mfe = np.array([tails[k]["mfe60"] for k in keys])
    dep = np.array([IC.is_dep(k[2]) for k in keys])
    print(f"  triggers with a 60-minute window: {len(keys)} (deployed {dep.sum()});  base rates: "
          f"HIT {hit[dep].mean():.1%}  ADV {adv[dep].mean():.1%}  (both {((hit > 0) & (adv > 0))[dep].mean():.1%})")

    # ---- M2
    Xp = X.copy()
    Xp[:, 0] = hit - 0.5
    _r, coefs, _n = IC.lasso_walk(keys, Xp, hit)
    dom = all(int(np.argmax(np.abs(c))) == 0 for c in coefs.values())
    _r, cc, _n = IC.lasso_walk(keys, np.zeros_like(X), hit)
    zero = all(np.count_nonzero(c) == 0 for c in cc.values())
    print(f"  M2 planted dominant every slice {dom}; constant -> zero coefs {zero}: {'PASS' if dom and zero else 'FAIL'}")
    if not (dom and zero):
        sys.exit("machinery checks failed -- not scoring")

    # ---- PART 1
    di, pi = np.where(dep)[0], np.where(~dep)[0]
    dkeys = [keys[i] for i in di]
    res = []
    for j, f in enumerate(IC.FEATS):
        ic, lo, hi, p = IC.day_boot_ic(X[di, j], hit[di], [k[2] for k in dkeys])
        per = []
        for s in range(6):
            ix = [i for i in di if IC.seg(keys[i][2]) == s]
            per.append(IC.spearman(X[ix, j], hit[ix]))
        same = sum(np.sign(v) == np.sign(ic) for v in per)
        q1, q2 = np.quantile(X[di, j], [1 / 3, 2 / 3])
        top = hit[di][X[di, j] > q2].mean() if (X[di, j] > q2).any() else np.nan
        bot = hit[di][X[di, j] < q1].mean() if (X[di, j] < q1).any() else np.nan
        res.append(dict(f=f, ic=ic, lo=lo, hi=hi, p=p, per=per, same=same,
                        adv=IC.spearman(X[di, j], adv[di]), mfe=IC.spearman(X[di, j], mfe[di]),
                        lift=(top / bot) if bot else np.nan, P=IC.spearman(X[pi, j], hit[pi])))
    rej = IC.bh([r["p"] for r in res])
    print("\n  PART 1 -- SINGLE-FACTOR TAIL SCREEN (HIT = MFE60 >= +0.5 S; deployed slices)")
    print(f"    {'input':9} {'IC hit':>7} {'95% CI':>17} {'p':>6} {'same':>5} {'IC adv':>7} {'IC mfe':>7} "
          f"{'lift':>5} {'P lab.':>7}  read")
    for r, rj in sorted(zip(res, rej), key=lambda z: -abs(z[0]["ic"])):
        ok = rj and r["same"] >= 5
        kind = ("vol" if (r["adv"] > 0) == (r["ic"] > 0) and abs(r["adv"]) > 0.5 * abs(r["ic"]) else "dir")
        print(f"    {r['f']:9} {r['ic']:+7.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:6.3f} {r['same']}/6 "
              f"{r['adv']:+7.4f} {r['mfe']:+7.4f} {r['lift']:5.2f} {r['P']:+7.4f}  {kind}{'  PASS' if ok else ''}")
    print(f"    passing the screen: {sum(1 for r, rj in zip(res, rej) if rj and r['same'] >= 5)} of {len(IC.FEATS)}")

    # ---- PART 2
    print("\n  PART 2 -- LASSO ON THE TAIL, walk-forward")
    rows, coefs, nnz = IC.lasso_walk(keys, X, hit)
    sp, lo, hi = IC.chained_ic(rows, hit, keys)
    per = {s: IC.spearman([r[1] for r in rows if r[3] == s], [hit[r[0]] for r in rows if r[3] == s]) for s in range(1, 7)}
    a = auc([r[1] for r in rows], [hit[r[0]] for r in rows])
    print("    inputs kept per slice: " + "  ".join(f"{s}:{n}" for s, n in nnz.items()))
    for j in [j for j in range(len(IC.FEATS)) if any(c[j] != 0 for c in coefs.values())]:
        print(f"      {IC.FEATS[j]:9} " + " ".join(f"{coefs[s][j]:+.4f}" for s in coefs))
    print(f"    chained Spearman {sp:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  AUC {a:.4f}   per slice " +
          " ".join(f"{s}:{v:+.3f}" for s, v in per.items()))

    # option outcomes (deployed, standalone, book exit)
    import check_implied_move_gate as G
    cands = G.candidates(D)
    pol = SC.policy_for(G.screen_rule("SPY", "CALL", 50))
    roe = {}
    for (tk, d, pct), rr in cands.items():
        if pct == 50:
            for day, m, payload, _s in rr:
                roe[(tk, d, day, m)] = SC.simulate(payload, pol, SC.DEFAULT_EOD, fill="bot")[0]
    allr = [roe[keys[r[0]]] for r in rows if keys[r[0]] in roe]
    for t in (0, 1, 2):
        sel = [r for r in rows if r[2] == t]
        v = np.array([roe[keys[r[0]]] for r in sel if keys[r[0]] in roe])
        print(f"    third {['bottom', 'middle', 'top'][t]:6}: n={len(sel):6}  HIT {np.mean([hit[r[0]] for r in sel]):.1%}  "
              f"ADV {np.mean([adv[r[0]] for r in sel]):.1%}  option: mean ROE {v.mean() * 100:+.1f}%  "
              f"hit50 {np.mean(v >= 0.5):.1%}  loss50 {np.mean(v <= -0.5):.1%}  (n {len(v)})")
    top = np.array([roe[keys[r[0]]] for r in rows if r[2] == 2 and keys[r[0]] in roe])
    t4 = top.mean() - np.mean(allr)
    rng = np.random.default_rng(SEED)
    tk_idx = {tk: np.array([i for i, k in enumerate(keys) if k[0] == tk]) for tk in IC.TICKERS}
    pl_ = []
    for _ in range(N_PLACEBO):
        Xq = X.copy()
        for tk, ix in tk_idx.items():
            Xq[ix] = X[rng.permutation(ix)]
        rq, _c, _n = IC.lasso_walk(keys, Xq, hit)
        pl_.append(IC.spearman([r[1] for r in rq], [hit[r[0]] for r in rq]))
    p95 = float(np.quantile(pl_, 0.95))
    crit = dict(T1=lo > 0, T2=sp > p95, T3=sum(v > 0 for v in per.values()) >= 4, T4=t4 > 0)
    print(f"    top third option ROE minus all-trigger mean: {t4 * 100:+.1f}pp")
    print(f"    placebo chained Spearman median {np.median(pl_):+.4f}, p95 {p95:+.4f}")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> PART 2 {'PASS' if all(crit.values()) else 'FAIL'}")

    print("\n  HOW MANY INPUTS -- chained Spearman with the model capped (reported)")
    for cap in IC.CAPS:
        rq, _c, nq = IC.lasso_walk(keys, X, hit, cap=cap)
        print(f"    cap {cap:2}: {IC.spearman([r[1] for r in rq], [hit[r[0]] for r in rq]):+.4f}   "
              f"kept " + " ".join(str(n) for n in nq.values()))


if __name__ == "__main__":
    main()
