"""
check_contradiction.py -- WHEN THRU AND PULLBACK DISAGREE, WHAT DECIDES WHO IS RIGHT? (user, 2026-11-20)
PRE-REGISTERED 2026-11-20, committed before any input below was scored against the target.

CONTEXT (ad hoc counts on discovery, 2026-11-20, knowable-only): opposite-direction THRU / PULLBACK pairs are common
and largely structural -- a short pullback entry sits on the bounce, which breaches a strike upward. Neither side wins
more than its own base rate (PULL 2R 41.1% after an opposite breach vs 41.7%; THRU next-strike lift after a setup is
already in the frozen score). The question here: is there anything, in either family's inputs, that says WHICH side
wins a contradiction? The answer decides the viewer note: a pass names the drivers; a fail says "normal, neither side
wins more often".

PAIRS (same ticker; both events from the existing caches; nothing re-detected)
  An OPPOSITE PAIR = a THRU breach (_breach_cache) and a PULLBACK entry (_structure_ic_cache) with opposite sides, the
  earlier one at most 120 s before the later one. The LATER event is the decision moment; both are known then.
    type B>P  the breach came first (each PULL entry pairs with its most recent opposite breach in the window)
    type P>B  the setup came first (each breach pairs with its most recent opposite setup in the window)
TARGET  D = +1 the PULLBACK side wins (its Y = 2R first) and the breach does not reach its next strike;
            -1 the THRU side wins (next strike) and the pullback does not reach 2R; 0 otherwise (both lose ~ / both
            win ~2%). Each side keeps its own pre-registered outcome definition.
INPUTS (64) every THRU input (28, prefix T.) and every PULLBACK input (32, prefix P.) as stamped at each event's own
  time (both <= the decision moment), + T.score and P.score (the frozen breach / pullback scores), + gap (seconds
  between the two), + typeBP (1 = breach first). Signs need no alignment: in every pair the two sides are opposite,
  so "+ with the breach" always means "- with the pullback". Ranked within ticker against the DISCOVERY distribution
  (mid-rank ECDF); missing -> 0.5.
SAMPLES DISCOVERY 2024-08-20 .. 2025-12-31 (halves split 2025-04-30). HOLDOUT 2026-01-02 .. 2026-11-13, scored ONCE
  (--holdout writes _contradiction_cache/RESULT-holdout.txt and refuses to run again). PARTLY CLEAN, said now: these
  sessions scored both screens' models before (different question, same events); the counts above were looked at on
  discovery only. Futures inputs exist only from 2026-07-27 -> not in the model; reported on that subset.
PASS RULES
  M1 machinery: a planted input (D + noise, IC ~0.10) must reach IC >= 0.05 with p < 0.01 on discovery.
  P1 IC (discovery): Spearman(input, D), day-block bootstrap p (1,000), BH q < 0.10 across the 64, same sign in both
     halves. Reported.
  P2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on the 64 ranks -> D, fit on discovery. PASS = keeps >= 1 input
     AND holdout Spearman(score, D) > 0 with the day-block (2,000) 95% CI above 0.
  OVERALL PASS = P2 -> the viewer note names the kept drivers and the side the score favours, labelled "in-sample +
  holdout", with a forward read from the live logs. FAIL -> the note says neither side wins more often, with the
  holdout's win rates.
REPORTED, not scored: D's distribution by pair type; on discovery, which inputs make a contradiction HAPPEN (PULL entries
  with vs without an opposite breach in the prior 120 s: Spearman of each P. input with that indicator).

RUN LOG
  2026-11-20 (count typo 66 -> 64 fixed before scoring, e9f503b)
    DISCOVERY 2,313 pairs / 326 sessions: PULL side wins 29.7%, THRU 20.5%, neither/both 49.8%.
      B>P (n 1,677) PULL 39.2% / THRU 19.3%; P>B (n 636) PULL 4.6% / THRU 23.6%. M1 planted +0.111 ok.
      P1: 23 of 64 pass; typeBP +0.253, P.s_speed -0.153, P.k_entry -0.148, P.v_path30 -0.123, T.c_path30 -0.118,
      P.k_high +0.093, gap -0.082, T.score -0.080 ...  LASSO keeps 12 (typeBP +0.601, P.k_entry -0.243, P.s_speed -0.140,
      P.v_path30 -0.097, P.k_high +0.085, T.s_kcall +0.050, T.c_tide +0.029, T.c_gex -0.025, T.c_tr30 +0.021, ...).
      Reported: a PULL entry follows an opposite breach 16.5% of the time, mostly when the entry is near a whole
      dollar (k_entry -0.35) -- the bounce IS the breach.
    HOLDOUT 1,667 pairs / 207 sessions: B>P PULL 39.1% / THRU 21.1%; P>B PULL 1.9% / THRU 29.2%.
      P2 IC +0.303 [+0.260, +0.345] -> PASS. Quintiles PULL/THRU: 2/32, 24/34, 35/26, 43/17, 40/10 %.
    READ (descriptive, after the score, holdout): most of it is ORDER -- the earlier signal has usually failed by the
    time the opposite one fires (a breach against an open short pullback has run through its tight stop). Within
    order the score still ranks (B>P +0.133 [+0.074, +0.185], P>B +0.172 [+0.083, +0.260]) and it ranks the BREACH:
    B>P PULL stays 37-44% in every quintile while THRU goes 37% -> 9%; biggest driver k_entry -- pullback entry 0-2 bp
    from the whole dollar (price back at the strike) THRU 13%, 2-5 bp 25%, 5-10 bp 34%, PULL ~39-40% throughout.
    VIEWER NOTE: breach then pullback against it -> the pullback keeps its usual odds, the breach's odds depend on
    whether price is back at the strike; pullback then breach against it -> the pullback has almost always failed.

Usage:  python check_contradiction.py --discovery | --holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import check_breach_ic as B                        # noqa: E402  (spearman, day_boot, bh)
import check_structure_ic as SI                    # noqa: E402  (INPUTS of the pullback family)

CACHE = "_contradiction_cache"
DISC, HOLD, HALF = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-13"), "2025-04-30"
WIN = 120
T_IN = list(B.INPUTS)
P_IN = list(SI.INPUTS)
INPUTS = [f"T.{k}" for k in T_IN] + [f"P.{k}" for k in P_IN] + ["T.score", "P.score", "gap", "typeBP"]
SEED = 20261120


def load(path, lo, hi):
    return [r for f in sorted(glob.glob(os.path.join(path, "*.json"))) if lo <= os.path.basename(f)[:10] <= hi
            for r in json.load(open(f))]


def pairs(lo, hi):
    import check_breach_forward as BF
    import freeze_pullback as FP
    fb, fp = json.load(open("breach_model_frozen.json")), json.load(open("pullback_model_frozen.json"))
    Bv, Sv = load("_breach_cache/events", lo, hi), load("_structure_ic_cache/events", lo, hi)
    bk, sk = defaultdict(list), defaultdict(list)
    for b in Bv:
        bk[(b["d"], b["t"])].append(b)
    for s in Sv:
        sk[(s["d"], s["t"])].append(s)
    out = []

    def row(b, s, later, typ):
        yT, yP = b["Y"] == 1, s["Y"] == 1
        D = 1 if (yP and not yT) else (-1 if (yT and not yP) else 0)
        r = {"d": later["d"], "t": later["t"], "t_in": later["t_in"], "type": typ, "D": D, "yT": int(yT), "yP": int(yP)}
        r.update({f"T.{k}": b.get(k) for k in T_IN})
        r.update({f"P.{k}": s.get(k) for k in P_IN})
        r["T.score"], r["P.score"] = BF.score(fb, b), FP.score(fp, s)
        r["gap"] = abs(s["t_in"] - b["t_in"]); r["typeBP"] = 1.0 if typ == "B>P" else 0.0
        return r

    for s in Sv:
        prior = [b for b in bk[(s["d"], s["t"])] if b["side"] != s["side"] and 0 < s["t_in"] - b["t_in"] <= WIN]
        if prior:
            out.append(row(max(prior, key=lambda b: b["t_in"]), s, s, "B>P"))
    for b in Bv:
        prior = [s for s in sk[(b["d"], b["t"])] if s["side"] != b["side"] and 0 < b["t_in"] - s["t_in"] <= WIN]
        if prior:
            out.append(row(b, max(prior, key=lambda s: s["t_in"]), b, "P>B"))
    return out, Sv, bk


def ranker(disc):
    cuts = {(tk, k): np.sort(np.array([r[k] for r in disc if r["t"] == tk and r[k] is not None], float))
            for tk in ("SPY", "QQQ", "IWM") for k in INPUTS}

    def X(rows):
        out = np.full((len(rows), len(INPUTS)), 0.5)
        for i, r in enumerate(rows):
            for j, k in enumerate(INPUTS):
                v, c = r[k], cuts[(r["t"], k)]
                if v is not None and c.size:
                    out[i, j] = (np.searchsorted(c, v, "left") + np.searchsorted(c, v, "right")) / (2.0 * c.size)
        return out
    return X


def fit(disc, X):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    Xd = X(disc) - 0.5
    y = np.array([r["D"] for r in disc], float)
    wk = np.array([dt.date.fromisoformat(r["d"]).isocalendar()[0] * 100 + dt.date.fromisoformat(r["d"]).isocalendar()[1] for r in disc])
    cv = list(GroupKFold(n_splits=5).split(Xd, y, wk))
    return LassoCV(cv=cv, random_state=0, alphas=60, max_iter=20000).fit(Xd, y)


def describe(rows, lab):
    D = np.array([r["D"] for r in rows])
    print(f"{lab}: {len(rows)} opposite pairs, {len({r['d'] for r in rows})} sessions | PULL side wins {np.mean(D == 1):.1%}, "
          f"THRU side wins {np.mean(D == -1):.1%}, neither/both {np.mean(D == 0):.1%}")
    for typ in ("B>P", "P>B"):
        d = np.array([r["D"] for r in rows if r["type"] == typ])
        if len(d):
            print(f"    {typ}: n {len(d)}  PULL wins {np.mean(d == 1):.1%}  THRU wins {np.mean(d == -1):.1%}")


def discovery():
    rng = np.random.default_rng(SEED)
    disc, Sv, bk = pairs(*DISC)
    describe(disc, f"DISCOVERY {DISC[0]}..{DISC[1]}")
    X = ranker(disc); Xd = X(disc); y = np.array([r["D"] for r in disc], float)
    # M1 machinery
    planted = y + np.random.default_rng(SEED + 1).normal(0, np.std(y) * 10, len(y))
    ic = B.spearman(planted, y)
    bs = B.day_boot(disc, lambda i: B.spearman(planted[i], y[i]), rng, 1000)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    ok = ic >= 0.05 and p < 0.01
    print(f"  M1 planted IC {ic:+.3f} p {p:.4f} -> {'ok' if ok else 'MACHINERY FAIL'}")
    if not ok:
        return
    h1 = np.array([r["d"] <= HALF for r in disc])
    ps, rows = [], []
    for j, k in enumerate(INPUTS):
        ic = B.spearman(Xd[:, j], y)
        bs = B.day_boot(disc, lambda i: B.spearman(Xd[i, j], y[i]), rng, 1000)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        ps.append(p); rows.append((k, ic, p, B.spearman(Xd[h1, j], y[h1]), B.spearman(Xd[~h1, j], y[~h1]),
                                   np.mean([r[k] is not None for r in disc])))
    qs = B.bh(ps)
    print("  P1 IC vs D (+ = the PULLBACK side wins), sorted by |IC|")
    for (k, ic, p, a, b, cov), q in sorted(zip(rows, qs), key=lambda x: -abs(x[0][1])):
        okk = q < 0.10 and np.sign(a) == np.sign(b) == np.sign(ic)
        print(f"    {k:14s} IC {ic:+.3f}  p {p:.3f}  q {q:.3f}  halves {a:+.3f} / {b:+.3f}  {'PASS' if okk else '    '}  cov {cov:.0%}")
    m = fit(disc, X)
    coef = sorted([(abs(c), k, c) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9], reverse=True)
    print(f"  P2 LASSO keeps {len(coef)} of {len(INPUTS)}: " + ", ".join(f"{k} {c:+.3f}" for _a, k, c in coef))
    sd = m.predict(Xd - 0.5)
    print(f"    in-sample IC {B.spearman(sd, y):+.3f}")
    # reported: what makes a contradiction HAPPEN (PULL entries with vs without a prior opposite breach)
    flag = np.array([any(b["side"] != s["side"] and 0 < s["t_in"] - b["t_in"] <= WIN for b in bk[(s["d"], s["t"])]) for s in Sv], float)
    occ = []
    for k in P_IN:
        v = np.array([np.nan if s.get(k) is None else float(s[k]) for s in Sv]); okm = ~np.isnan(v)
        occ.append((B.spearman(v[okm], flag[okm]), k))
    occ.sort(key=lambda x: -abs(x[0]))
    print(f"  REPORTED: what makes a PULL entry follow an opposite breach ({flag.mean():.1%} of entries): "
          + ", ".join(f"{k} {c:+.2f}" for c, k in occ[:8]))


def holdout():
    out = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(out):
        sys.exit(f"{out} exists -- scored once")
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(SEED)
    disc, _, _ = pairs(*DISC)
    hold, _, _ = pairs(*HOLD)
    X = ranker(disc)
    m = fit(disc, X)
    coef = [(k, float(c)) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9]
    lines = [f"check_contradiction HOLDOUT {dt.date.today()} -- {HOLD[0]}..{HOLD[1]}"]
    D = np.array([r["D"] for r in hold], float)
    lines.append(f"{len(hold)} opposite pairs / {len({r['d'] for r in hold})} sessions: PULL wins {np.mean(D == 1):.1%}, "
                 f"THRU wins {np.mean(D == -1):.1%}, neither/both {np.mean(D == 0):.1%}")
    for typ in ("B>P", "P>B"):
        d = np.array([r["D"] for r in hold if r["type"] == typ])
        lines.append(f"  {typ}: n {len(d)}  PULL wins {np.mean(d == 1):.1%}  THRU wins {np.mean(d == -1):.1%}")
    lines.append(f"LASSO (discovery) keeps {len(coef)}: " + ", ".join(f"{k} {c:+.3f}" for k, c in coef))
    if coef:
        sh = m.predict(X(hold) - 0.5)
        ic = B.spearman(sh, D)
        bs = B.day_boot(hold, lambda i: B.spearman(sh[i], D[i]), rng, 2000)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        ok = ic > 0 and lo > 0
        lines.append(f"P2 holdout IC {ic:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if ok else 'FAIL'}")
        cuts = np.quantile(m.predict(X(disc) - 0.5), [0.2, 0.4, 0.6, 0.8]); qi = np.searchsorted(cuts, sh)
        lines.append("  holdout by score quintile (low = THRU side favoured .. high = PULL side): " +
                     " | ".join(f"Q{i + 1} PULL {np.mean(D[qi == i] == 1):.0%} THRU {np.mean(D[qi == i] == -1):.0%} (n {(qi == i).sum()})" for i in range(5)))
    else:
        lines.append("P2 LASSO kept nothing -> FAIL")
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
