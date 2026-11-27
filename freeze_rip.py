"""
freeze_rip.py -- freezes the two quiet-stretch RIP models that PASSED check_rip_screen's holdout (2026-11-26) into
rip_model_frozen.json, for the live RIP pill (strike_cue.py). Run once on the PC; never retuned.

  conf  "A SIZE at confirmation" -- at a quiet stretch's 60th minute: will it end in a rip (>= 3 typical ranges past
        the band within 60 min of its end)? holdout IC +0.102 [+0.030, +0.167]
  brk   "B SIZE at the break" -- at the first minute outside the band: will price carry on >= 3 typical ranges from the
        break price within 60 min? holdout IC +0.169 [+0.105, +0.232]

For each: the Lasso refit exactly as the screen fit it (same discovery events, folds, seed -> the same coefficients;
checked below), the kept inputs only, each input's per-ticker DISCOVERY distribution (the ranks the model was fit on;
stored as up to 512 quantiles), score quintile cut points on discovery, and each quintile's rip rate on the HOLDOUT
(scored once; in-sample now -- the same convention as strike_cue_thru.json). Direction is not frozen: both direction
models FAILED.

    python freeze_rip.py           (writes rip_model_frozen.json and prints the self-check)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "rip_model_frozen.json")
NQ = 512


def score(frz, f, tk):
    """frozen score of one feature dict (raw values; ABS_ inputs are |directional|) for ticker tk."""
    s = 0.0
    for k, c in zip(frz["inputs"], frz["coef"]):
        v = f.get(k)
        ref = frz["ref"][tk][k]
        if v is None or v != v or not ref:
            r = 0.5
        else:
            r = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2 * len(ref))
        s += c * r
    return float(s)


def quintile(frz, s):
    return int(np.searchsorted(frz["cuts"], s, side="right"))          # 0 .. 4


def main():
    import check_rip_screen as R
    plan = (("conf", "RIP", R.SIZE_IN, "A SIZE at confirmation"), ("brk", "RIP_B", R.SIZE_IN, "B SIZE at the break"))
    out = {"frozen": dt.datetime.now().isoformat(timespec="seconds"), "source": "check_rip_screen.py holdout 2026-11-26 (e887a02)",
           "models": {}}
    for mode, tgt, cols, lab in plan:
        F = R.add_abs(R.build(mode))
        Fd, Fh = R.split(F)
        Fd = Fd.filter(pl.col(tgt).cast(pl.Float64).is_finite()); Fh = Fh.filter(pl.col(tgt).cast(pl.Float64).is_finite())
        rank = R.ranker(Fd, cols)
        Xd, yd = rank(Fd), Fd[tgt].cast(pl.Float64).to_numpy()
        las = R.fit_lasso(Xd, yd, np.array([R.iso_week(d) for d in Fd["d"].to_list()]))
        keep = [j for j in range(len(cols)) if las.coef_[j] != 0]
        inputs = [cols[j] for j in keep]
        coef = [float(las.coef_[j]) for j in keep]
        ref = {}
        for tk in R.TKS:
            z = Fd.filter(pl.col("tk") == tk)
            ref[tk] = {}
            for k in inputs:
                a = np.sort(z[k].cast(pl.Float64).drop_nulls().drop_nans().to_numpy())
                if len(a) > NQ:
                    a = np.quantile(a, np.linspace(0, 1, NQ))
                ref[tk][k] = [float(x) for x in a]
        frz = dict(label=lab, target=tgt, inputs=inputs, coef=coef, ref=ref)
        sd = np.array([score(frz, r, r["tk"]) for r in Fd.to_dicts()])
        frz["cuts"] = [float(x) for x in np.quantile(sd, [0.2, 0.4, 0.6, 0.8])]
        hh = Fh.to_dicts()
        sh = np.array([score(frz, r, r["tk"]) for r in hh])
        qh = np.array([quintile(frz, s) for s in sh])
        yh = np.array([r[tgt] for r in hh], float)
        frz["quintiles"] = [dict(rate=round(100 * float(yh[qh == q].mean()), 1) if (qh == q).any() else None, n=int((qh == q).sum()))
                            for q in range(5)]
        frz["base"] = round(100 * float(yh.mean()), 1)
        # self-check: the frozen file's holdout IC vs the screen's (the 512-quantile refs change ranks only slightly)
        from scipy.stats import spearmanr
        sh_screen = rank(Fh) @ las.coef_
        frz["selfcheck"] = dict(holdout_ic_frozen=round(float(spearmanr(sh, yh).statistic), 4),
                                holdout_ic_screen=round(float(spearmanr(sh_screen, yh).statistic), 4),
                                score_corr=round(float(np.corrcoef(sh, sh_screen)[0, 1]), 5))
        out["models"][mode] = frz
        print(f"{lab}: {len(inputs)} inputs | holdout rip rate by score quintile "
              f"{[q['rate'] for q in frz['quintiles']]} (base {frz['base']}%) | self-check {frz['selfcheck']}")
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
