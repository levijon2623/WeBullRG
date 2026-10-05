# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.4"]
# ///
"""
check_ic_direction.py
=====================
Test 2 of the tail follow-up (user, 2026-10-03). check_ic_tail's score is mostly
a MAGNITUDE forecast, and check_ic_vol_trade showed magnitude is priced into the
options. Direction is what a perp or a future pays for -- so isolate it: can the
same 32 inputs forecast WHICH WAY the flow trigger's tail goes?

POPULATION  check_ic_screen.build_dataset (p50 flow triggers, SPY/QQQ/IWM, CALL
            and PUT, 09:35-14:59), with a full 60-minute window. P = the spent
            pre-sample, training only; slices 1-6 scored.

TARGET (units: that day's 09:35 ATM straddle S; s = trade direction)
    DIRT = +1  the move reaches +0.5 S the trade's way within 60 minutes
               WITHOUT reaching -0.5 S
           -1  the reverse
            0  neither, or both
    Ternary, so a pure volatility forecast scores ~0 on it: more movement
    raises +1 and -1 together.

MODEL  check_ic_screen.lasso_walk on DIRT (a linear model used only to rank):
    alpha by leave-one-segment-out CV inside P + earlier slices; thirds cut on
    the training predictions; chained over slices 1-6.

PRE-REGISTERED CRITERIA (fixed 2026-10-03, before the first scored run)
    D1  chained Spearman(prediction, DIRT) > 0, day-block 95% CI lower > 0
    D2  > p95 of 20 placebos (input rows permuted within ticker). Spearman is
        rank-based and tie-safe -- unlike check_ic_vol_trade's third-based V5,
        whose placebo models tied into one third.
    D3  > 0 in >= 4 of 6 slices
    D4  top-third mean r60 > 0, day-block 95% CI lower > 0, where
        r60 = s x (close(m+60) - close(m)) / S -- the underlying's own
        directional return, what a perp / future position earns before costs
    PASS = D1-D4.

REPORTED: the single-factor IC of each input with DIRT (BH q < 0.10, sign in
>= 5/6 slices); r60 in bp by third against a ~2-3bp perp round trip; the
trigger's own ATM option ROE (book exit, standalone) by third; inputs kept and
coefficients per slice.

MACHINERY: a planted continuous input (rank of s x the realised 60-minute
excursion asymmetry, MFE60 + MAE60) must dominate every slice and send the top
third's r60 far above the bottom's; constant inputs keep no coefficient.

RUN LOG
  2026-10-03 run 1: machinery PASS (planted top r60 +0.376 S vs bottom -0.376;
    constant -> no coefs). 58,532 triggers, 44,454 deployed. DIRT +1 18.1%,
    -1 18.4%, 0 63.5%.
    FAIL (D4). The model RANKS tail direction: chained Spearman +0.058, CI
      [+0.029, +0.084], 6/6 slices, placebo p95 -0.0003 -- but it decays
      (slices +0.098 +0.067 +0.086 +0.053 +0.038 +0.026), and it does not move
      the mean: top third r60 +0.0042 S, CI [-0.015, +0.024], = -0.09 bp
      (bottom +0.25 bp). Option ROE by third -12.6 / -9.9 / -10.4%.
    Single-factor (4/32 pass BH + 5/6): VANNA +0.058, DEX -0.053, DIR -0.053
      (calls' favourable tails rarer: downside moves are the fast ones),
      AMTLOC -0.040; near: TREND, POCDIST, CHARM (-), CUMALIGN (+) -- trades
      aligned with trend / value / dealer delta see FEWER favourable tails.
    Arithmetic, not tested: an underlying bracket at +-0.5 S on the top third
      would earn ~0.5 x 0.043 = +0.02 S, ~1 bp at a ~0.45% straddle -- below a
      2-3 bp perp round trip.

Usage:
  python check_ic_direction.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_ic_screen as IC                            # noqa: E402

TAIL_S = 0.5
N_PLACEBO = 20
SEED = 20261011


def boot_mean(vals_by_day, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    days = sorted(vals_by_day)
    bs = [np.mean([x for j in rng.integers(0, len(days), len(days)) for x in vals_by_day[days[j]]])
          for _ in range(n)]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    import directional_flow_backtester as D
    keys, raw, _y30, tails = IC.build_dataset(D)
    keys = sorted((k for k in keys if k in tails), key=lambda k: (k[2], k[3], k[0], k[1]))
    # r60 from the 1m closes
    close, spot_bp = {}, {}
    for tk in IC.TICKERS:
        for day, mod, c in IC.BF.bars_full(tk).select("date", "mod", "close").iter_rows():
            close[(tk, day, int(mod))] = c
    am = IC.BF.am_straddles()
    r60, bp60 = {}, {}
    for k in keys:
        tk, d, day, m = k
        s = 1 if d == "CALL" else -1
        a, b, st = close.get((tk, day, m)), close.get((tk, day, m + 60)), am.get((tk, day))
        if a and b and st:
            r60[k] = s * (b - a) / st[1]
            bp60[k] = s * (b - a) / a * 1e4
    keys = [k for k in keys if k in r60]
    mem = IC.memberships({k: raw[k] for k in keys})
    X = np.array([[mem[k][f] - (0.5 if f in IC.SIGNED + IC.UNSIGNED else 0.0) for f in IC.FEATS] for k in keys])
    fav = np.array([tails[k]["mfe60"] >= TAIL_S for k in keys])
    adv = np.array([tails[k]["mae60"] <= -TAIL_S for k in keys])
    dirt = np.where(fav & ~adv, 1.0, np.where(adv & ~fav, -1.0, 0.0))
    dep = np.array([IC.is_dep(k[2]) for k in keys])
    print(f"  triggers {len(keys)} (deployed {dep.sum()});  DIRT +1 {np.mean(dirt[dep] > 0):.1%}  "
          f"-1 {np.mean(dirt[dep] < 0):.1%}  0 {np.mean(dirt[dep] == 0):.1%}")

    # ---- machinery
    asym = np.array([tails[k]["mfe60"] + tails[k]["mae60"] for k in keys])
    Xp = X.copy()
    Xp[:, 0] = pd.Series(asym).rank(pct=True).to_numpy() - 0.5
    rp, cp, _n = IC.lasso_walk(keys, Xp, dirt)
    dom = all(int(np.argmax(np.abs(c))) == 0 for c in cp.values())
    tp = [r60[keys[r[0]]] for r in rp if r[2] == 2]
    bt = [r60[keys[r[0]]] for r in rp if r[2] == 0]
    _r, cc, _n = IC.lasso_walk(keys, np.zeros_like(X), dirt)
    zero = all(np.count_nonzero(c) == 0 for c in cc.values())
    ok = dom and zero and tp and bt and np.mean(tp) - np.mean(bt) > 0.2
    print(f"  M planted: dominant every slice {dom}; top r60 {np.mean(tp):+.3f} S vs bottom {np.mean(bt):+.3f} S; "
          f"constant -> zero coefs {zero}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("machinery checks failed -- not scoring")

    # ---- reported: single-factor IC with DIRT
    di = np.where(dep)[0]
    res = []
    for j, f in enumerate(IC.FEATS):
        ic, lo, hi, p = IC.day_boot_ic(X[di, j], dirt[di], [keys[i][2] for i in di])
        per = [IC.spearman(X[[i for i in di if IC.seg(keys[i][2]) == s_], j],
                           dirt[[i for i in di if IC.seg(keys[i][2]) == s_]]) for s_ in range(6)]
        res.append((f, ic, lo, hi, p, sum(np.sign(v) == np.sign(ic) for v in per)))
    rej = IC.bh([r[4] for r in res])
    print("\n  SINGLE-FACTOR IC WITH DIRT (reported)")
    for (f, ic, lo, hi, p, same), rj in sorted(zip(res, rej), key=lambda z: -abs(z[0][1]))[:12]:
        print(f"    {f:9} {ic:+.4f} [{lo:+.4f},{hi:+.4f}] p {p:.3f}  same {same}/6  "
              f"{'PASS' if rj and same >= 5 else ''}")
    print(f"    passing: {sum(1 for r, rj in zip(res, rej) if rj and r[5] >= 5)} of {len(IC.FEATS)}")

    # ---- the model
    rows, coefs, nnz = IC.lasso_walk(keys, X, dirt)
    sp, lo, hi = IC.chained_ic(rows, dirt, keys)
    per = {s_: IC.spearman([r[1] for r in rows if r[3] == s_], [dirt[r[0]] for r in rows if r[3] == s_])
           for s_ in range(1, 7)}
    print("\n  LASSO ON DIRT: inputs kept per slice " + "  ".join(f"{s_}:{n}" for s_, n in nnz.items()))
    for j in [j for j in range(len(IC.FEATS)) if any(c[j] != 0 for c in coefs.values())]:
        print(f"      {IC.FEATS[j]:9} " + " ".join(f"{coefs[s_][j]:+.4f}" for s_ in coefs))
    print(f"    chained Spearman {sp:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   per slice " +
          " ".join(f"{s_}:{v:+.3f}" for s_, v in per.items()))
    rng = np.random.default_rng(SEED)
    tk_idx = {tk: np.array([i for i, k in enumerate(keys) if k[0] == tk]) for tk in IC.TICKERS}
    pl_ = []
    for _ in range(N_PLACEBO):
        Xq = X.copy()
        for tk, ix in tk_idx.items():
            Xq[ix] = X[rng.permutation(ix)]
        rq, _c, _n = IC.lasso_walk(keys, Xq, dirt)
        pl_.append(IC.spearman([r[1] for r in rq], [dirt[r[0]] for r in rq]))
    p95 = float(np.quantile(pl_, 0.95))

    import check_implied_move_gate as G
    cands = G.candidates(D)
    pol = SC.policy_for(G.screen_rule("SPY", "CALL", 50))
    roe = {}
    for (tk, d, pct), rr in cands.items():
        if pct == 50:
            for day, m, payload, _s in rr:
                roe[(tk, d, day, m)] = SC.simulate(payload, pol, SC.DEFAULT_EOD, fill="bot")[0]
    top_by_day = defaultdict(list)
    for t in (0, 1, 2):
        sel = [r for r in rows if r[2] == t]
        v = [r60[keys[r[0]]] for r in sel]
        b = [bp60[keys[r[0]]] for r in sel]
        o = [roe[keys[r[0]]] for r in sel if keys[r[0]] in roe]
        print(f"    third {['bottom', 'middle', 'top'][t]:6}: n={len(sel):6}  DIRT mean {np.mean([dirt[r[0]] for r in sel]):+.3f}  "
              f"r60 {np.mean(v):+.4f} S = {np.mean(b):+.2f} bp  option ROE {np.mean(o) * 100:+.1f}% (n {len(o)})")
        if t == 2:
            for r in sel:
                top_by_day[keys[r[0]][2]].append(r60[keys[r[0]]])
    tlo, thi = boot_mean(top_by_day)
    tmean = float(np.mean([x for v in top_by_day.values() for x in v]))
    crit = dict(D1=lo > 0, D2=sp > p95, D3=sum(v > 0 for v in per.values()) >= 4, D4=tlo > 0)
    print(f"    top third r60 {tmean:+.4f} S, 95% CI [{tlo:+.4f}, {thi:+.4f}]  (a perp round trip is ~2-3 bp)")
    print(f"    placebo Spearman median {np.median(pl_):+.4f}, p95 {p95:+.4f}")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> {'PASS' if all(crit.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
