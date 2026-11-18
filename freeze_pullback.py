"""
freeze_pullback.py -- freeze check_structure_ic's DISCOVERY Lasso for the viewer's PULLBACK pill (user, 2026-11-18).

Writes pullback_model_frozen.json: intercept, coefficients of the kept inputs, per-ticker 1,001-point quantile grids of
each kept input over discovery (scored as a MID-RANK on the grid, like the research ranker), and the discovery
score's quintile cut points; plus what followed in each quintile on the HOLDOUT (2R before the stop within 30 min;
holdout partly clean, scored once 2026-11-17). Display only: the score ranked entries out of sample (IC +0.032) but
the top quintile did not pay through the option. Run once; never refit.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import check_structure_ic as C

OUT = os.path.join(ROOT, "pullback_model_frozen.json")


def rank(g, v):
    g = np.asarray(g)
    return (np.searchsorted(g, v, "left") + np.searchsorted(g, v, "right")) / (2.0 * len(g))


def score(frz, r):
    z = frz["intercept"]
    for k, c in frz["coef"].items():
        v = r.get(k)
        z += c * ((0.5 if v is None else rank(frz["grids"][r["t"]][k], v)) - 0.5)
    return z


if __name__ == "__main__":
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists -- frozen")
    disc, hold = C.load(*C.DISC), C.load(*C.HOLD)
    X, m = C.fit(disc)
    keep = {k: float(c) for k, c in zip(C.INPUTS, m.coef_) if abs(c) > 1e-9}
    grids = {tk: {k: [round(float(x), 10) for x in np.quantile(np.array([r[k] for r in disc if r["t"] == tk and r[k] is not None], float),
                                                                np.linspace(0, 1, 1001))] for k in keep} for tk in ("SPY", "QQQ", "IWM")}
    frz = dict(frozen="2026-11-18", source="check_structure_ic.py discovery fit 2024-08-20..2025-12-31",
               intercept=float(m.intercept_), coef=keep, grids=grids)
    sd = np.array([score(frz, r) for r in disc])
    frz["cuts"] = [float(x) for x in np.quantile(sd, [0.2, 0.4, 0.6, 0.8])]
    sh = np.array([score(frz, r) for r in hold])
    frz["selfcheck_corr"] = float(np.corrcoef(sh, m.predict(X(hold) - 0.5))[0, 1])
    qi = np.searchsorted(frz["cuts"], sh, side="right")
    frz["holdout"] = [dict(n=int((qi == i).sum()), two_r=round(100 * float(np.mean([r["Y"] for r, q in zip(hold, qi) if q == i])), 1))
                      for i in range(5)]
    frz["base"] = dict(two_r=round(100 * float(np.mean([r["Y"] for r in hold])), 1), n=len(hold))
    frz["note"] = "ranks out of sample (holdout IC +0.032, partly clean) but the top quintile did not pay: display only"
    json.dump(frz, open(OUT, "w"), indent=1)
    print("froze", list(keep), "cuts", [round(c, 4) for c in frz["cuts"]], "self-check corr", round(frz["selfcheck_corr"], 4))
    print("holdout by quintile", frz["holdout"], "base", frz["base"])
