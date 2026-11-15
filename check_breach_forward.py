"""
check_breach_forward.py -- FORWARD TEST: wider-stop continuation on high-score strike breaches (user, 2026-11-14).
LOCKED 2026-11-14, before a wider-stop P&L was computed on ANY data. EXPLORATORY in origin: check_breach_ic's
holdout (2026-01..09) showed the frozen score lifts the next-strike rate 14% -> 33% across quintiles while the
$0.10 failure rate stays flat ~63%, so the A trade's tight stop sits inside an unchanged shakeout. The wider stop
is chosen AFTER that holdout -- only sessions nobody has looked at can test it.

MODEL      FROZEN: check_breach_ic's discovery Lasso (2024-08-20..2025-12-31, 10 inputs), written once by
           --freeze to breach_model_frozen.json (coefficients, intercept, 1,001-point per-ticker quantile grids of
           each kept input, the discovery 80th-percentile score). Never refit.
EVENTS     check_breach_ic.build_day (same event, inputs, look-ahead rules). Sessions from 2026-11-16 onward.
SELECTION  frozen score >= the frozen discovery 80th percentile (the top quintile).
TRADE (primary, the only scored one)
           buy the ATM K contract in the breach direction at t_in (breach known + 1 s);
           TP   the median path reaches the next strike - $0.05 (K + side x 0.95)
           STOP the median path is back through K by $0.50 (side x (price - K) <= -0.50: halfway to the old strike)
           TIME 15 minutes after entry; exits 1 s after the median that fires them is known.
           Priced on the tape (NBBO on the contract's own prints): buy mid + $0.06, sell mid - $0.08 per contract,
           $0.10 fees round trip (the user's measured fills, 2026-11-14).
REPORTED, not scored: stop $0.30; stop $0.50 with a 30-minute time stop; the same primary trade on ALL breaches
           (baseline) and on the bottom quintile.
PASS       mean P&L per contract > 0 with the day-block bootstrap 95% CI above 0 (2,000 resamples).
READS      --status any time (counts only). --score: early read once >= 60 forward sessions are built
           (~2027-02), FINAL at >= 150 sessions (~2027-06). Never retuned; a FAIL ends it.
DATA       the forward sessions need the lake refreshed first (uw_options_data_lake.py: full tape / trades-core
           and the option-contracts-1m silver), then `check_breach_ic.py --tide` coverage for those dates (this
           script's --build pulls the tide itself) and `--build` here.

Usage:  python check_breach_forward.py --freeze (once) | --build | --status | --score
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import check_breach_ic as B                        # noqa: E402
import check_sixpack_lake as L                     # noqa: E402

FROZEN = os.path.join(ROOT, "breach_model_frozen.json")
FWD = os.path.join(B.CACHE, "forward")
START = "2026-11-16"
EARLY_N, FINAL_N = 60, 150
VARIANTS = {"primary": (0.50, 900), "sl0.30": (0.30, 900), "sl0.50_30m": (0.50, 1800)}


def freeze():
    if os.path.exists(FROZEN):
        sys.exit(f"{FROZEN} exists -- the model is frozen")
    disc = B.load(*B.DISC)
    X, m = B.fit(disc)
    keep = [(k, float(c)) for k, c in zip(B.INPUTS, m.coef_) if abs(c) > 1e-9]
    grids = {}
    for tk in B.TK:
        grids[tk] = {}
        for k, _c in keep:
            v = np.array([r[k] for r in disc if r["t"] == tk and r[k] is not None], float)
            grids[tk][k] = [round(float(x), 8) for x in np.quantile(v, np.linspace(0, 1, 1001))]
    frz = dict(frozen=dt.date.today().isoformat(), source="check_breach_ic.py discovery fit", discovery=list(B.DISC),
               intercept=float(m.intercept_), coef=dict(keep), grids=grids)
    sd = np.array([score(frz, r) for r in disc])
    frz["q80"] = float(np.quantile(sd, 0.8))
    frz["q20"] = float(np.quantile(sd, 0.2))
    # self-check: the frozen file reproduces the in-memory model on the holdout
    hold = B.load(*B.HOLD)
    a = m.predict(X(hold) - 0.5)
    b = np.array([score(frz, r) for r in hold])
    frz["selfcheck_holdout_corr"] = float(np.corrcoef(a, b)[0, 1])
    json.dump(frz, open(FROZEN, "w"), indent=1)
    print(f"froze {len(keep)} inputs, q80 {frz['q80']:+.4f}, self-check corr {frz['selfcheck_holdout_corr']:.5f}")


def score(frz, r):
    z = frz["intercept"]
    for k, c in frz["coef"].items():
        v = r.get(k)
        g = frz["grids"][r["t"]][k]
        x = 0.5 if v is None else float(np.interp(v, g, np.linspace(0, 1, len(g))))
        z += c * (x - 0.5)
    return z


def money_wide(tq, tk, K, side, t_in, pm, sl, hold):
    sc, med = pm
    kt = B._kt(sc)
    end = t_in + hold
    c = tq.get((tk, K, "C" if side > 0 else "P"))
    if c is None:
        return None
    def q(T):
        s, b, a = c
        j = int(np.searchsorted(s, T, side="right")) - 1
        return None if j < 0 or T - s[j] > 30 else (b[j], a[j])
    e = q(t_in)
    if e is None or e[1] <= 0.05:
        return None
    tx = end
    for k in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        T = float(kt[k]) + B.LAT
        if T > end:
            break
        x = side * (med[k] - K)
        if x >= 0.95 or x <= -sl:
            tx = T
            break
    x = q(tx)
    if x is None:
        return None
    return ((x[0] + x[1]) / 2 - B.SELL_C - ((e[0] + e[1]) / 2 + B.BUY_C) - B.FEE_RT) * 100


def build():
    frz = json.load(open(FROZEN))
    B.pull_tide_range(START, dt.date.today().isoformat())
    days = [d for d in L.lake_days(START, "2099-12-31")
            if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))
            or os.path.exists(os.path.join(L.LAKE, "bronze", "full-tape", f"{d}.parquet"))]
    os.makedirs(FWD, exist_ok=True)
    for d in days:
        f = os.path.join(FWD, d + ".json")
        if os.path.exists(f):
            continue
        B.build_day(d)
        ev = json.load(open(os.path.join(B.EV, d + ".json")))
        path, tq = L.price_path(d), B.tape(d)
        out = []
        for r in ev:
            sc, hi, lo, cl = path[r["t"]]
            med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
            r["score"] = score(frz, r)
            for name, (sl, hold) in VARIANTS.items():
                r[name] = money_wide(tq, r["t"], r["K"], r["side"], r["t_in"], (sc, med), sl, hold)
            out.append(r)
        json.dump(out, open(f, "w"))
        print(d, len(out), flush=True)


def load():
    return [r for f in sorted(glob.glob(os.path.join(FWD, "*.json"))) for r in json.load(open(f))]


def status():
    rows = load()
    frz = json.load(open(FROZEN))
    n = len({r["d"] for r in rows})
    top = [r for r in rows if r["score"] >= frz["q80"]]
    print(f"forward sessions built {n} (early read at {EARLY_N}, final at {FINAL_N}); breaches {len(rows)}, top quintile {len(top)}")


def score_fwd():
    rows = load()
    frz = json.load(open(FROZEN))
    n = len({r["d"] for r in rows})
    if n < EARLY_N:
        sys.exit(f"too early: {n} forward sessions, the first read is at {EARLY_N}")
    final = n >= FINAL_N
    rng = np.random.default_rng(B.SEED + 7)
    print(f"{'FINAL' if final else 'EARLY'} READ: {n} sessions from {START}")
    for lab, sel in (("TOP quintile", lambda r: r["score"] >= frz["q80"]), ("ALL breaches", lambda r: True),
                     ("BOTTOM quintile", lambda r: r["score"] <= frz["q20"])):
        for name in VARIANTS:
            rs = [r for r in rows if sel(r) and r[name] is not None]
            if len(rs) < 30:
                continue
            v = np.array([r[name] for r in rs])
            bs = B.day_boot(rs, lambda idx: v[idx].mean(), rng)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            scored = lab.startswith("TOP") and name == "primary"
            verdict = (("PASS" if lo > 0 else "FAIL") if final else ("on track" if lo > 0 else "open")) if scored else "reported"
            print(f"  {lab:15s} {name:11s} n={len(rs):5d} mean ${v.mean():+.2f}/contract [{lo:+.2f}, {hi:+.2f}] -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("freeze", "build", "status", "score"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.freeze: freeze()
    if a.build: build()
    if a.status: status()
    if a.score: score_fwd()
    if not any(vars(a).values()):
        sys.exit(__doc__)
