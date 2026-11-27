"""
check_shootout_countdown.py -- PRE-REGISTRATION, written 2026-11-27 before any prediction below was scored (user:
"pre-register the countdown"). Can the SHOOTOUT pill say "today likely runs to ~HH:MM" better than the plain history
it shows now?

WHAT IS ALREADY KNOWN (descriptive, explore_shootout_end.py, odd months, within-ticker ranks): the first-5-minute range
(+0.34) and the ATM 0DTE IV level (-0.27) go with a longer / shorter Shootout; gap, imbalance, spreads, dealer gamma,
macro mornings and 10:00 releases ~0. The even months have not been looked at for any factor.

DATA      _shootout_cache/end_rows.parquet (explore_shootout_end.py; definition A; IV on the daily scale by era): per
          session and ticker the end minute (capped at 11:00), the factors, and the VR path 09:35 .. 11:00.
CHECKPOINTS  m = 09:35, 09:40, 09:45, 09:50, 10:00, 10:15, 10:30. At each, the sessions STILL ON (end > m).
TARGET    remaining minutes = min(end, 11:00) - m.
BASELINE  what the pill shows now: per ticker and checkpoint, the median end of the DISCOVERY sessions still on at m.
MODEL     ordinary least squares on log(1 + remaining), fit on DISCOVERY rows pooled over checkpoints, per ticker, with
          inputs: the within-ticker discovery rank of RNG5 (09:30-09:34 range / T30), of IV0 (ATM 0DTE IV at 09:31), and
          of VR(m) (the tape vs priced at the checkpoint, as the pill computes it), plus a dummy per checkpoint.
          Prediction = exp(fit) - 1, clipped to [0, 11:00 - m]. Three inputs, chosen from the explore read (two) and
          the pill's own live state (one) -- nothing else is tried.
SPLIT     DISCOVERY = odd calendar months (as the explore read), HOLDOUT = even months, scored ONCE (--holdout writes
          _shootout_cache/RESULT-countdown.txt and refuses to run again).
PRIMARY   P1  holdout mean absolute error of the model < of the baseline: the paired difference (baseline |err| - model
              |err|), pooled over checkpoints and tickers, day-block bootstrap (2,000) 95% CI above 0.
          P2  holdout Spearman(predicted, actual remaining) > 0 with the day-block 95% CI above 0.
          PASS = P1 AND P2 -> the pill adds "today likely to ~HH:MM" (the model's line), labelled with its holdout
          error. FAIL -> the pill keeps the plain history.
NULL      on discovery before the holdout: the three inputs shuffled across sessions within ticker (200 draws), model
          refit on the first discovery half and scored on the second -- the mean MAE gain must sit at or below 0.
REPORTED, NOT TESTED  MAE by checkpoint and ticker; the gain from each input alone; how often the prediction lands
          within 5 / 10 minutes.

RESULT (2026-11-27, holdout scored once; _shootout_cache/RESULT-countdown.txt)
  DISCOVERY (fit first half / score second): MAE 26.0 -> 23.6 min, gain +2.48 [+1.81, +3.11], Spearman +0.462; null
    (inputs shuffled) gain -3.19 -- at/below 0 as required.
  HOLDOUT (fit all discovery, 343 even-month sessions, 3,667 checkpoint rows):
    P1 MAE 20.7 -> 16.7 min, gain +3.94 [+2.85, +5.08]  PASS
    P2 Spearman(predicted, actual remaining) +0.499 [+0.452, +0.545]  PASS          OVERALL PASS
    By checkpoint (MAE base / model, within 10 min base / model): 09:35 23.4 / 20.2 (21% / 33%), 09:40 23.0 / 18.4
    (20 / 36%), 09:45 22.9 / 18.4, 09:50 22.4 / 17.3, 10:00 19.9 / 14.7, 10:15 13.2 / 11.2, 10:30 6.5 / 6.6.
  READ: the countdown tells a long Shootout from a short one, mostly in its first 20 minutes; errors stay ~15-20 min
  early -- a lean, not a timer. Per the pre-registration the pill adds the model's line, updated at the checkpoints
  only (as validated), labelled with its holdout error.

    python check_shootout_countdown.py --discovery
    python check_shootout_countdown.py --holdout          (once)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "_shootout_cache", "end_rows.parquet")
RESULT = os.path.join(ROOT, "_shootout_cache", "RESULT-countdown.txt")
CAP, M_END = 660, 575
CHECKS = (575, 580, 585, 590, 600, 615, 630)
INPUTS = ("RNG5", "IV0", "VRm")
SEED = 20261127


def long_rows(R):
    """one row per session-ticker-checkpoint still on at m."""
    out = []
    for r in R.iter_rows(named=True):
        vp = r["vr_path"]
        for m in CHECKS:
            if r["end"] > m:
                k = m - M_END
                out.append(dict(d=r["d"], tk=r["tk"], m=m, RNG5=r["RNG5"], IV0=r["IV0"],
                                VRm=vp[k] if k < len(vp) else np.nan, rem=float(min(r["end"], CAP) - m)))
    return pl.DataFrame(out)


def ranker(Ld):
    ref = {(tk, f): np.sort(Ld.filter(pl.col("tk") == tk)[f].drop_nans().drop_nulls().to_numpy()) for tk in Ld["tk"].unique() for f in INPUTS}

    def rank(L):
        cols = {}
        for f in INPUTS:
            vals = []
            for tk, v in zip(L["tk"].to_list(), L[f].to_list()):
                a = ref.get((tk, f))
                vals.append(0.5 if v is None or v != v or a is None or not len(a) else
                            (np.searchsorted(a, v, "left") + np.searchsorted(a, v, "right")) / (2 * len(a)))
            cols[f"r_{f}"] = vals
        return L.with_columns([pl.Series(k, v) for k, v in cols.items()])
    return rank


def design(L):
    X = [L[f"r_{f}"].to_numpy() for f in INPUTS] + [(L["m"] == m).cast(pl.Float64).to_numpy() for m in CHECKS]
    return np.column_stack(X)


def fit(Ld):
    models = {}
    for tk in Ld["tk"].unique().to_list():
        z = Ld.filter(pl.col("tk") == tk)
        beta, *_ = np.linalg.lstsq(design(z), np.log1p(z["rem"].to_numpy()), rcond=None)
        models[tk] = beta
    return models


def predict(models, L):
    out = np.empty(L.height)
    for i, (tk, m) in enumerate(zip(L["tk"].to_list(), L["m"].to_list())):
        out[i] = np.nan
    for tk, beta in models.items():
        mask = (L["tk"] == tk).to_numpy()
        if mask.any():
            z = L.filter(pl.col("tk") == tk)
            out[mask] = np.clip(np.expm1(design(z) @ beta), 0, CAP - z["m"].to_numpy())
    return out


def baseline(Ld, L):
    med = {(tk, m): float(np.median(Ld.filter((pl.col("tk") == tk) & (pl.col("m") == m))["rem"]))
           for tk in Ld["tk"].unique().to_list() for m in CHECKS}
    return np.array([med.get((tk, m), np.nan) for tk, m in zip(L["tk"].to_list(), L["m"].to_list())])


def boot(days, vals, n=2000, seed=SEED):
    from collections import defaultdict
    pos = defaultdict(list)
    for i, d in enumerate(days):
        pos[d].append(i)
    ud = sorted(pos)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        ix = np.concatenate([pos[ud[k]] for k in rng.integers(0, len(ud), len(ud))])
        bs.append(vals(ix))
    bs = np.array(bs)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.mean(bs <= 0))


def report(Ld, L, label):
    from scipy.stats import spearmanr
    rank = ranker(Ld)
    Ld, L = rank(Ld), rank(L)
    pm, pb = predict(fit(Ld), L), baseline(Ld, L)
    y = L["rem"].to_numpy()
    gain = np.abs(y - pb) - np.abs(y - pm)
    days = L["d"].to_list()
    lo, hi, p = boot(days, lambda ix: float(np.mean(gain[ix])))
    rho = spearmanr(pm, y).statistic
    rlo, rhi, rp = boot(days, lambda ix: float(spearmanr(pm[ix], y[ix]).statistic))
    print(f"=== {label}: {L.height} checkpoint rows, {len(set(days))} sessions")
    print(f"  MAE baseline {np.mean(np.abs(y - pb)):.1f} min | model {np.mean(np.abs(y - pm)):.1f} min | gain {np.mean(gain):+.2f} "
          f"[{lo:+.2f}, {hi:+.2f}] | Spearman(pred, actual) {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}]")
    for m in CHECKS:
        k = (L["m"] == m).to_numpy()
        if k.any():
            print(f"    {m // 60}:{m % 60:02d}  n {k.sum():4d}  MAE base {np.mean(np.abs(y[k] - pb[k])):5.1f}  model {np.mean(np.abs(y[k] - pm[k])):5.1f}  "
                  f"within 10 min: base {np.mean(np.abs(y[k] - pb[k]) <= 10):.0%} model {np.mean(np.abs(y[k] - pm[k]) <= 10):.0%}")
    return dict(gain=float(np.mean(gain)), lo=lo, hi=hi, rho=float(rho), rlo=rlo, rhi=rhi)


def null(Ld, draws=200):
    rng = np.random.default_rng(SEED)
    dd = sorted(Ld["d"].unique().to_list())
    h1 = set(dd[: len(dd) // 2])
    gains = []
    rows = Ld.to_dicts()
    by = {}                                          # (tk, d) -> {m: (RNG5, IV0, VRm)} and the session's first row
    for r in rows:
        e = by.setdefault((r["tk"], r["d"]), {})
        e[r["m"]] = (r["RNG5"], r["IV0"], r["VRm"])
        e.setdefault("first", (r["RNG5"], r["IV0"], r["VRm"]))
    sess_of = {}
    for tk, d in by:
        sess_of.setdefault(tk, []).append(d)
    for _ in range(draws):
        # shuffle each session's inputs onto another session (same ticker); checkpoints / targets stay
        perm = {tk: dict(zip(ss, rng.permutation(ss))) for tk, ss in sess_of.items()}
        sh = []
        for r in rows:
            o = by[(r["tk"], perm[r["tk"]][r["d"]])]
            v = o.get(r["m"], o["first"])
            sh.append(dict(r, RNG5=v[0], IV0=v[1], VRm=v[2]))
        S = pl.DataFrame(sh)
        A, B = S.filter(pl.col("d").is_in(list(h1))), S.filter(~pl.col("d").is_in(list(h1)))
        rank = ranker(A)
        A, B = rank(A), rank(B)
        y = B["rem"].to_numpy()
        gains.append(float(np.mean(np.abs(y - baseline(A, B)) - np.abs(y - predict(fit(A), B)))))
    g = np.array(gains)
    print(f"  NULL (inputs shuffled across sessions, fit first half / score second, {draws}): mean gain {g.mean():+.2f} "
          f"[{np.percentile(g, 2.5):+.2f}, {np.percentile(g, 97.5):+.2f}]  {'OK' if g.mean() <= 0 else 'NOT AT/BELOW 0'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    L = long_rows(pl.read_parquet(SRC))
    odd = L["d"].str.slice(5, 2).cast(pl.Int32) % 2 == 1
    Ld, Lh = L.filter(odd), L.filter(~odd)
    if a.discovery:
        dd = sorted(Ld["d"].unique().to_list())
        h1 = set(dd[: len(dd) // 2])
        report(Ld.filter(pl.col("d").is_in(list(h1))), Ld.filter(~pl.col("d").is_in(list(h1))),
               "DISCOVERY (fit first half, score second half)")
        null(Ld)
    if a.holdout:
        if os.path.exists(RESULT):
            sys.exit(f"holdout already scored: {RESULT}")
        r = report(Ld, Lh, "HOLDOUT (fit all discovery, score the even months once)")
        ok1, ok2 = r["lo"] > 0, r["rlo"] > 0
        line = (f"  P1 MAE gain {r['gain']:+.2f} min [{r['lo']:+.2f}, {r['hi']:+.2f}] {'PASS' if ok1 else 'FAIL'} | "
                f"P2 Spearman {r['rho']:+.3f} [{r['rlo']:+.3f}, {r['rhi']:+.3f}] {'PASS' if ok2 else 'FAIL'} | "
                f"OVERALL {'PASS' if ok1 and ok2 else 'FAIL'}")
        print(line)
        with open(RESULT, "w", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
