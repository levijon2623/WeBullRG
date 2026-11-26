"""
check_runner_day.py -- RUNNER MORNING OR CHOP MORNING, AND CAN IT BE TOLD BY 09:40? (user, 2026-11-25)
PRE-REGISTERED 2026-11-25, committed before the target below was computed on any session.

WHERE IT COMES FROM. The exit replay of the user's own 311 trades (scratchpad exit_replay.py, 2026-11-25): no fixed exit
beat the user's, but the best exit FLIPPED with the period -- quick exits best 09-25..11-12, holding 90-120 s best in the
trending 11-13..11-25 stretch (about +$5 a trade over the user's exits). The useful thing would be knowing, at the time,
which kind of morning it is. This is a question about the MARKET, so it is tested on market history, not on the user's
trades (which only motivated it).

ONE ROW PER MORNING (SPY primary; QQQ / IWM reported). Sessions 2024-08-20 .. 2026-11-13 (the lake's 5-second path).
TARGET  HP, "holding paid": every 30 s from 09:40:00 to 10:28:00, the CHASE entry of check_scalp_rules (5-minute move
        >= 1.4 bp, no pullback in the last minute, on the 5-s close known at t) -- for each, the move in the trade's
        direction at t + 120 s minus at t + 15 s (bp); HP = the morning's mean (mornings with < 5 entries dropped).
        $ per contract = 0.5 x 100 x spot x HP (the two exits share the entry, so fills cancel to first order).
        Also reported: ER = |P(10:30) - P(09:40)| / the 5-s path length 09:40-10:30 (1 = clean trend, ~0 = chop).
INPUTS  (11, all known by 09:40:00)
  o_z       overnight move size (check_overnight z, vs the 20 prior nights)
  o_F       pre-market fade fraction (check_overnight F)
  o_gap     |09:30 open - prior close| / prior close, bp
  or10      09:30-09:39 high - low (1-minute candles) / its 20-session median
  er10      |P(09:40) - P(09:30)| / the 5-s path length 09:30-09:40
  rvol10    SPY volume 09:30-09:39 / its 20-session median
  pm_rvol   pre-market volume 04:00-09:29 / its 20-session median
  vix       prior close VIX / its 252-day median
  gex       dealer gamma (SPY spot-exposures, gamma per 1% move) at the last minute <= 09:39
  prev_rng  prior session high - low / close, vs its 20-session median
  prev_er   prior session |close - open| / the sum of |1-minute changes|
  ranked against the DISCOVERY distribution (mid-rank ECDF), missing -> 0.5.
PASS RULES (SPY)
  P1 IC (discovery): Spearman(input, HP), bootstrap over mornings, BH across the 11, same sign in both halves. Reported.
  P2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on the discovery ranks -> HP. PASS = keeps >= 1 input AND the
     holdout Spearman(score, HP) > 0 with the bootstrap 95% CI above 0.
  P3 MONEY: holdout mornings in the score's TOP fifth (cuts from discovery): mean HP in $ > 0 with the CI above 0
     (holding 2 minutes beat getting out at 15 s on the mornings the score calls runners).
  OVERALL PASS = P2 and P3 -> a 09:40 "runner / chop" read in the viewer, labelled with the holdout numbers; and a
  forward check on the user's own exits.
SAMPLES DISCOVERY = ODD calendar months, HOLDOUT = EVEN (one morning = one row). PARTLY CLEAN, said now: check_scalp_rules
  used these mornings (09:31-10:05 entries, other outcomes); check_overnight used the night inputs; the exit replay used
  the user's trades from 2026-09-25 (part of this sample period) -- none of them computed HP or ER. --holdout writes
  _runner_cache/RESULT-holdout.txt and refuses to run again.

RUN LOG (2026-11-25; discovery _runner_discovery.txt, holdout scored once _runner_cache/RESULT-holdout-full.txt)
  552 SPY mornings. On average holding 2 min vs 15 s on chase entries is worth ~$0 (SPY +$0.03 / +$0.15 a contract,
  positive on ~50% of mornings; QQQ / IWM the same).
  Discovery: or10 (first-10-minute range vs normal) IC +0.164 passed BH (halves +0.33 / +0.05); prev_er -0.14, o_F +0.12.
  Lasso kept 5 (or10 +0.51, prev_er -0.39, o_F +0.33, gex, pm_rvol); discovery fifths -0.43 / -1.13 / -0.36 / +0.01 / +2.01.
  HOLDOUT: score IC -0.004 [-0.124, +0.112] -> P2 FAIL; top fifth -$0.21 [-1.29, +0.91] -> P3 FAIL. OVERALL FAIL.
  Descriptive, holdout single inputs: or10 +0.026, prev_er +0.031 (sign flipped), o_F +0.016 -- the discovery pattern
  did not repeat. Runner vs chop mornings were not predictable by 09:40 from these inputs.

    python check_runner_day.py --build | --discovery | --holdout
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
CACHE = "_runner_cache"
TKS = ("SPY", "QQQ", "IWM")
OPEN = 9 * 3600 + 30 * 60
W0, W1, EVERY = 9 * 3600 + 40 * 60, 10 * 3600 + 28 * 60, 30
SHORT, LONG = 15, 120
MOVE5, PULL1 = 1.4, 0.7
INPUTS = ["o_z", "o_F", "o_gap", "or10", "er10", "rvol10", "pm_rvol", "vix", "gex", "prev_rng", "prev_er"]
FIRST, LAST = "2024-08-20", "2026-11-13"
SEED = 20261126


def half(d):
    return "discovery" if int(d[5:7]) % 2 == 1 else "holdout"


def day_target(d):
    """{tk: (HP bp, n entries, ER, er10, spot at 09:40)} from the 5-second path; cached."""
    p = os.path.join(CACHE, "days", f"{d}.json")
    if os.path.exists(p):
        return d, json.load(open(p))
    import check_sixpack_lake as L
    try:
        path = L.price_path(d)
    except Exception:                                       # noqa: BLE001
        return d, {}
    out = {}
    for tk in TKS:
        if tk not in path:
            continue
        sc, _hi, _lo, cl = path[tk]
        kt, cl = np.asarray(sc, float) + 5.0, np.asarray(cl, float)

        def P(t):
            j = int(np.searchsorted(kt, t, side="right")) - 1
            return float(cl[j]) if j >= 0 else np.nan
        bp = lambda a, b: (a - b) / b * 1e4                 # noqa: E731
        hp = []
        for t in range(W0, W1 + 1, EVERY):
            p0 = P(t)
            m300 = bp(p0, P(max(t - 300, OPEN + 5)))
            m60 = bp(P(t - 10), P(t - 60))
            if not np.isfinite([m300, m60]).all() or abs(m300) < MOVE5:
                continue
            side = np.sign(m300)
            if side * m60 < -PULL1:
                continue
            hp.append(side * (bp(P(t + LONG), p0) - bp(P(t + SHORT), p0)))
        seg = cl[(kt > W0) & (kt <= 10 * 3600 + 30 * 60)]
        seg0 = cl[(kt > OPEN) & (kt <= W0)]
        er = abs(P(10 * 3600 + 30 * 60) - P(W0)) / np.sum(np.abs(np.diff(seg))) if len(seg) > 2 else None
        er10 = abs(P(W0) - P(OPEN + 5)) / np.sum(np.abs(np.diff(seg0))) if len(seg0) > 2 else None
        out[tk] = dict(hp=float(np.mean(hp)) if len(hp) >= 5 else None, n=len(hp), er=er, er10=er10, spot=P(W0))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    return d, out


def ext(tk):
    import check_overnight as O
    return O.load(tk)


def build():
    import check_overnight as O
    import check_sixpack_lake as L
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "tide", "*.json")))
    days = [d for d in days if FIRST <= d <= LAST]
    with ProcessPoolExecutor(6) as ex:
        tg = dict(ex.map(day_target, days))
    px = json.load(open(os.path.join("_cot_cache", "prices_daily.json")))
    vix = sorted((r["date"], float(r["close"])) for r in px["VIX"])
    vdates = [a for a, _ in vix]
    rows = []
    for tk in TKS:
        E = ext(tk)
        edays = sorted(E)
        nights = {r["d"]: r for r in O.night_rows(tk)}
        per = {}
        for dd in edays:                                    # per-session raw numbers for the 20-session baselines
            x = E[dd]
            r = x.filter(pl.col("market_time") == "r")
            if r.height < 200:
                continue
            f10 = r.filter(pl.col("mod") < 570 + 10)
            pm = x.filter((pl.col("mod") >= 240) & (pl.col("mod") < 570))
            ch = np.abs(np.diff(r["close"].to_numpy()))
            per[dd] = dict(or10=float(f10["high"].max() - f10["low"].min()) if f10.height else None,
                           v10=float(f10["volume"].sum()), pmv=float(pm["volume"].sum()),
                           rng=float(r["high"].max() - r["low"].min()) / float(r["close"][-1]),
                           er=abs(float(r["close"][-1]) - float(r["open"][0])) / ch.sum() if ch.sum() > 0 else None,
                           open=float(r["open"][0]))
        pdays = sorted(per)
        for d in days:
            t = (tg.get(d) or {}).get(tk)
            if not t or t["hp"] is None or d not in per:
                continue
            i = pdays.index(d)
            prev = pdays[max(0, i - 20):i]
            if len(prev) < 15:
                continue
            med = lambda k: float(np.median([per[q][k] for q in prev if per[q][k] is not None]))   # noqa: E731
            n = nights.get(d)
            row = dict(d=d, tk=tk, hp=t["hp"], n=t["n"], er=t["er"], spot=t["spot"],
                       o_z=n["z"] if n else None, o_F=n["F"] if n else None,
                       o_gap=abs(per[d]["open"] - n["C0"]) / n["C0"] * 1e4 if n else None,
                       or10=per[d]["or10"] / med("or10") if per[d]["or10"] and med("or10") else None,
                       er10=t["er10"], rvol10=per[d]["v10"] / med("v10") if med("v10") else None,
                       pm_rvol=per[d]["pmv"] / med("pmv") if med("pmv") else None,
                       prev_rng=per[prev[-1]]["rng"] / float(np.median([per[q]["rng"] for q in prev[:-1]])) if len(prev) > 1 else None,
                       prev_er=per[prev[-1]]["er"])
            j = np.searchsorted(vdates, d) - 1                 # the prior close
            if j >= 252:
                row["vix"] = vix[j][1] / float(np.median([v for _, v in vix[j - 252:j]]))
            g = L.gex_series(d, "SPY", {})
            gm = 579
            while gm >= 560 and gm not in g:
                gm -= 1
            row["gex"] = g.get(gm)
            rows.append(row)
    json.dump(rows, open(os.path.join(CACHE, "mornings.json"), "w"))
    print(f"built {len(rows)} mornings ({sum(r['tk'] == 'SPY' for r in rows)} SPY)")


# ------------------------------------------------------------------ scoring
def ranks(ref_rows, rows):
    X = np.full((len(rows), len(INPUTS)), 0.5)
    for j, k in enumerate(INPUTS):
        ref = np.sort(np.array([r[k] for r in ref_rows if r.get(k) is not None], float))
        for i, r in enumerate(rows):
            v = r.get(k)
            if v is not None and len(ref):
                X[i, j] = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2.0 * len(ref))
    return X


def sp(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def boot(fn, n, k=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(k)])
    v = v[np.isfinite(v)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5), float(min(1, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))


def usd(r, v):
    return 0.5 * 100 * r["spot"] * v / 1e4


def bh(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); q = np.empty(len(ps)); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
    return q


def main():
    import argparse
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return
    if not (a.discovery or a.holdout):
        print(__doc__)
        return
    mark = os.path.join(CACHE, "RESULT-holdout.txt")
    if a.holdout and os.path.exists(mark):
        sys.exit("holdout already scored -- see " + mark)
    rows = json.load(open(os.path.join(CACHE, "mornings.json")))
    L = [f"check_runner_day {'HOLDOUT' if a.holdout else 'DISCOVERY'}"]
    for tk in TKS:
        R = [r for r in rows if r["tk"] == tk]
        for h in ("discovery", "holdout") if a.holdout else ("discovery",):
            H = [r for r in R if half(r["d"]) == h]
            hpu = [usd(r, r["hp"]) for r in H]
            L.append(f"  {tk} {h}: {len(H)} mornings | holding 2 min minus 15 s: mean ${np.mean(hpu):+.2f}/contract, "
                     f"positive on {np.mean(np.array(hpu) > 0):.0%} of mornings | ER median {np.median([r['er'] for r in H if r['er'] is not None]):.2f}")
    R = [r for r in rows if r["tk"] == "SPY"]
    D = [r for r in R if half(r["d"]) == "discovery"]
    Xd, yd = ranks(D, D), np.array([r["hp"] for r in D])
    first, second = [r for r in D if r["d"] < "2025-07-01"], [r for r in D if r["d"] >= "2025-07-01"]
    ics = []
    for j, k in enumerate(INPUTS):
        ic = sp(Xd[:, j], yd)
        _, _, p = boot(lambda idx: sp(Xd[idx, j], yd[idx]), len(D), k=1000)
        h1 = sp(ranks(D, first)[:, j], [r["hp"] for r in first])
        h2 = sp(ranks(D, second)[:, j], [r["hp"] for r in second])
        ics.append((k, ic, p, h1, h2))
    q = bh([x[2] for x in ics])
    L.append("  SPY IC with HP (discovery):")
    for (k, ic, p, h1, h2), qq in sorted(zip(ics, q), key=lambda z: -abs(z[0][1])):
        L.append(f"    {k:9s} {ic:+.3f}  q {qq:.3f}  halves {h1:+.3f} / {h2:+.3f}" + ("  *" if qq < .10 and np.sign(h1) == np.sign(h2) == np.sign(ic) else ""))
        L[-1] += f"   (ER IC {sp(Xd[:, INPUTS.index(k)], [r['er'] if r['er'] is not None else np.nan for r in D]):+.3f})" if all(r['er'] is not None for r in D) else ""
    wk = [f"{dt.date.fromisoformat(r['d']).isocalendar()[0]}-{dt.date.fromisoformat(r['d']).isocalendar()[1]}" for r in D]
    groups = np.unique(wk, return_inverse=True)[1]
    folds = list(GroupKFold(5).split(Xd, yd, groups))
    m = LassoCV(cv=folds, alphas=np.logspace(-3, 1, 40), max_iter=50000).fit(Xd, yd)
    kept = [(k, c) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9]
    L.append(f"  LASSO keeps {len(kept)}: " + ", ".join(f"{k} {c:+.2f}" for k, c in sorted(kept, key=lambda z: -abs(z[1]))))
    sd = m.predict(Xd)
    cuts = np.percentile(sd, [20, 40, 60, 80])
    fifth = lambda s: int(np.searchsorted(cuts, s, side="right"))   # noqa: E731
    if kept:
        qd = [np.mean([usd(r, r["hp"]) for r, s in zip(D, sd) if fifth(s) == b] or [np.nan]) for b in range(5)]
        L.append("  discovery fifths, holding-paid $: " + " / ".join(f"{v:+.2f}" for v in qd))
    if a.holdout:
        Hh = [r for r in R if half(r["d"]) == "holdout"]
        Xh, yh = ranks(D, Hh), np.array([r["hp"] for r in Hh])
        sh = m.predict(Xh)
        if kept:
            icv = sp(sh, yh)
            lo, hi, _ = boot(lambda idx: sp(sh[idx], yh[idx]), len(Hh))
            top = [r for r, s in zip(Hh, sh) if fifth(s) == 4]
            tv = np.array([usd(r, r["hp"]) for r in top])
            tlo, thi, _ = boot(lambda idx: tv[idx].mean(), len(tv))
            qh = [np.mean([usd(r, r["hp"]) for r, s in zip(Hh, sh) if fifth(s) == b] or [np.nan]) for b in range(5)]
            L.append(f"  HOLDOUT {len(Hh)} mornings: score IC {icv:+.3f} [{lo:+.3f}, {hi:+.3f}] -> P2 {'PASS' if lo > 0 else 'fail'}")
            L.append("  holdout fifths, holding-paid $: " + " / ".join(f"{v:+.2f}" for v in qh))
            L.append(f"  P3 top fifth ${tv.mean():+.2f} [{tlo:+.2f}, {thi:+.2f}] n {len(tv)} -> {'PASS' if tlo > 0 else 'fail'}")
            L.append(f"  OVERALL: {'PASS' if (lo > 0 and tlo > 0) else 'FAIL'}")
        else:
            L.append("  OVERALL: FAIL (the Lasso kept no input)")
        open(mark, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
