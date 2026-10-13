# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trigger_vs_random.py
==========================
DOES THE FLOW TRIGGER'S PATH BEAT A RANDOM ENTRY? (user, 2026-10-13)

check_trigger_paths found 94% of trigger days go green after costs at some
point, median peak +78%, and the underlying ends the day the trade's way 69%
of the time. A 0DTE option is volatile enough that ANY entry may do the
same, so those numbers mean nothing without a placebo. This is that placebo.

REAL     the FIRST trigger per rule-day, full live gate stack
         (sim_core.deployed_candidates), 9 enabled rules -- 192 rule-days.
PLACEBOS (K = 20 per rule-day; same ticker, same right, same rule dte order,
         same contract pick and $0.50 floor, same SELL scoring as
         check_trigger_paths.path_row, to the rule's flatten)
    P1  SAME DAY, random minute   -> does the trigger's TIMING matter?
    P2  RANDOM DAY (any deployed session of that ticker, no regime gate),
        random minute            -> does the whole package (day selection +
                                     timing) matter?
    Placebo minutes are drawn from THAT RULE's own first-trigger minutes,
    so time of day (time left to expiry) is matched.
UNIT     rule-day: d_i = real_i - mean(placebos_i). Day-block bootstrap
         over calendar days (2,000), halves split 2025-08-21.

PRE-REGISTERED (book = all 9 rules pooled; each arm scored on its own)
    C_UL    underlying at the flatten, signed to the trade, bp
    C_50    the option reaches SELL >= +50% at some point (share)
    C_HOLD  option held to the flatten, SELL ROE
    A metric BEATS RANDOM on an arm when mean d > 0, 95% CI lower > 0 and
    both halves > 0. Six tests (3 metrics x 2 arms), no correction -- a lone
    marginal pass is a lead, not a claim.
    Reported, unscored: "ever green" (SELL > 0), median peak, the research
    set (no paper-only rules), per rule.
MACHINERY
    M1  IDENTITY: a placebo at the real minute of the real day reproduces
        the real row exactly (every d = 0) -- the synthetic trigger path picks
        the same contract and scores it the same way.
    M2  PLANT: +30 bp added to the real C_UL must pass on both arms.

RUN LOG
  2026-10-13 run 1: 192 rule-days, 9 rules. Placebos/day P1 16-20, P2 8-18
    (random days without option bars or under the $0.50 floor drop out).
    M1 identity PASS (192/192, max diff 0). M2 plant PASS both arms.
    P1 SAME DAY:  C_UL PASS +15.7 bp [+5.2, +27.8] (halves +26.4 / +5.1);
                  C_50 FAIL +0.061 (OOS -0.020); C_HOLD FAIL +0.030 (OOS -0.121).
    P2 RANDOM DAY: C_UL PASS +36.8 bp [+14.1, +64.4]; C_50 PASS +0.130
                  (49.5% -> 62.5%); C_HOLD PASS +0.235 (-15.6% -> +8.0%);
                  both halves on all three.
    EVER GREEN (reported): real 93.8% vs random 88.0% (same day) / 85.5%
      (random day). ~6-8 pp of the "94% go green" is the trigger; the rest is
      what any 0DTE option does.
    READ: the edge is mostly WHICH DAYS the rules trade (regime gates + a
    trigger firing), and some direction in the minute itself (underlying
    +16 bp vs the same day); the minute does not reliably improve the
    option's outcome out of sample. Caveat: the gates were chosen on this
    same window, so P2 is optimistic; the pre-sample (labelled) is the
    untouched check -- book -17.2% there (check_presample).

Usage:  python check_trigger_vs_random.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import directional_flow_backtester as D                   # noqa: E402
import sim_core                                            # noqa: E402
from check_trigger_paths import path_row, ul_closes       # noqa: E402
from config import RULES                                   # noqa: E402

SPLIT = sim_core.SPLIT
K, N_BOOT, SEED = 20, 2000, 20261013
METRICS = (("ul", "C_UL  underlying at flatten, bp", "{:+.1f}"),
           ("p50", "C_50  reaches +50%", "{:+.3f}"),
           ("hold", "C_HOLD held to flatten, ROE", "{:+.3f}"))
REPORTED = (("green", "ever green", "{:+.3f}"), ("peak", "peak SELL", "{:+.3f}"))


def metrics(row):
    return dict(ul=row.get("ul_last", np.nan), p50=float(row["peak"] >= 0.5),
                hold=row["last"], green=float(row["peak"] > 0), peak=row["peak"])


def synth(r, d, mod):
    ts = pd.Timestamp(d) + pd.Timedelta(minutes=int(mod))
    return dict(date=d, ts=ts.to_datetime64(), hour=int(mod) // 60,
                dir=r["direction"].upper(), abs_flow=0.0)


def run_rule(r, rng, ulc):
    """-> list of dicts, one per real rule-day: real metrics + placebo lists."""
    tk = r["ticker"]
    cand = sim_core.deployed_candidates(D, r)
    if not cand:
        return []
    eod = sim_core.eod_mod(r)
    first, seen = [], set()
    for d, m, payload in cand:                     # cand is sorted by (date, mod)
        if d not in seen:
            seen.add(d)
            first.append((d, m, payload))
    pool = np.array([m for _, m, _ in first])
    days = sorted(d for d in ulc if sim_core.DEPLOYED_START <= d <= max(seen))
    specs = []                                     # (owner, arm, date, mod)
    for i, (d, m, _) in enumerate(first):
        specs.append((i, "ID", d, m))
        for mm in rng.choice(pool, K):
            specs.append((i, "P1", d, int(mm)))
        for dd, mm in zip(rng.choice(len(days), K), rng.choice(pool, K)):
            specs.append((i, "P2", days[dd], int(mm)))
    # one permissive build for every distinct (date, minute): same builder,
    # gates stripped to the instrument choice (right, dte order, offset,
    # $0.50 floor). Regime/amp/amt/macro are day-level and P2 must ignore them.
    rb = {k: r[k] for k in ("name", "ticker", "direction", "dte", "hours", "strike_offset")
          if k in r}
    rb["flow_abs"] = 0.0
    uniq = sorted({(d, m) for _, _, d, m in specs})
    meta = []
    built = sim_core.build_candidates(D, rb, trigs=[synth(r, d, m) for d, m in uniq],
                                      meta_out=meta)
    rows = {}
    for (d, m, payload) in built:
        if (d, m) not in rows:
            pr = path_row(r, d, m, payload, ulc, eod)
            if pr is not None:
                rows[(d, m)] = metrics(pr)
    out = []
    for i, (d, m, payload) in enumerate(first):
        pr = path_row(r, d, m, payload, ulc, eod)
        if pr is None:
            continue
        rec = dict(rule=r["name"], tk=tk, date=d, m=m, paper=r["name"] in sim_core.PAPER_ONLY,
                   real=metrics(pr), ID=[], P1=[], P2=[])
        out.append((i, rec))
    by_i = dict(out)
    for i, arm, d, m in specs:
        if i in by_i and (d, m) in rows:
            by_i[i][arm].append(rows[(d, m)])
    return [rec for _, rec in out]


def diffs(recs, arm, key, plant=0.0):
    rows = []
    for rec in recs:
        ps = [p[key] for p in rec[arm] if p[key] == p[key]]
        x = rec["real"][key]
        if not ps or x != x:
            continue
        rows.append((rec["date"], x + plant, float(np.mean(ps))))
    return pd.DataFrame(rows, columns=["date", "real", "plac"])


def score(df, rng):
    d = (df["real"] - df["plac"]).to_numpy()
    days = df["date"].to_numpy()
    ud, inv = np.unique(days, return_inverse=True)
    a = np.bincount(inv, weights=d, minlength=len(ud))
    k = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = [a[j].sum() / k[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    h = [d[days < SPLIT].mean(), d[days >= SPLIT].mean()]
    ok = d.mean() > 0 and lo > 0 and all(x > 0 for x in h)
    return dict(n=len(d), real=df["real"].mean(), plac=df["plac"].mean(), d=d.mean(),
                lo=lo, hi=hi, h=h, ok=ok)


def line(label, s, fmt, scored=True):
    f = lambda v: fmt.format(v)                   # noqa: E731
    tag = ("PASS" if s["ok"] else "FAIL") if scored else ("  ok" if s["ok"] else "   -")
    return (f"    {label:34} n={s['n']:>4}  real {f(s['real'])}  random {f(s['plac'])}  "
            f"diff {f(s['d'])} [{f(s['lo'])}, {f(s['hi'])}]  halves {f(s['h'][0])} / {f(s['h'][1])}  {tag}")


def main():
    rng = np.random.default_rng(SEED)
    recs, ulc = [], {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        if r["ticker"] not in ulc:
            ulc[r["ticker"]] = ul_closes(r["ticker"])
        got = run_rule(r, rng, ulc[r["ticker"]])
        cov = np.mean([len(x["P1"]) for x in got]) if got else 0
        cov2 = np.mean([len(x["P2"]) for x in got]) if got else 0
        print(f"  {r['name']:24} rule-days {len(got):>3}   placebos/day P1 {cov:4.1f}  P2 {cov2:4.1f} of {K}")
        recs += got
    brng = np.random.default_rng(SEED + 1)

    print("\n  ---- MACHINERY ----")
    worst = max(abs(diffs(recs, "ID", k)["real"].sub(diffs(recs, "ID", k)["plac"])).max()
                for k in ("ul", "p50", "hold", "green", "peak"))
    nid = len(diffs(recs, "ID", "hold"))
    print(f"    M1 identity: {nid} of {len(recs)} rule-days rebuilt, max |real - placebo| = {worst:.2e}"
          f"   -> {'PASS' if worst < 1e-9 and nid == len(recs) else 'FAIL'}")
    m2 = [score(diffs(recs, arm, "ul", plant=30.0), brng)["ok"] for arm in ("P1", "P2")]
    print(f"    M2 plant +30 bp on C_UL: P1 {'PASS' if m2[0] else 'FAIL'}  P2 {'PASS' if m2[1] else 'FAIL'}")

    for arm, title in (("P1", "P1 -- SAME DAY, random minute (timing)"),
                       ("P2", "P2 -- RANDOM DAY, random minute (the whole package)")):
        print(f"\n  ======== {title} ========")
        for key, lab, fmt in METRICS:
            print(line(lab, score(diffs(recs, arm, key), brng), fmt))
        for key, lab, fmt in REPORTED:
            print(line(lab + " (reported)", score(diffs(recs, arm, key), brng), fmt, scored=False))
        res = [x for x in recs if not x["paper"]]
        print("    -- research set (no paper-only rules), reported --")
        for key, lab, fmt in METRICS:
            print(line(lab, score(diffs(res, arm, key), brng), fmt, scored=False))

    print("\n  ======== PER RULE (reported; d = real - random) ========")
    print(f"    {'rule':24} {'n':>3}   {'P1 ul bp':>9} {'P1 +50%':>8} {'P1 hold':>8}   "
          f"{'P2 ul bp':>9} {'P2 +50%':>8} {'P2 hold':>8}")
    for name in dict.fromkeys(x["rule"] for x in recs):
        rr = [x for x in recs if x["rule"] == name]
        cells = []
        for arm in ("P1", "P2"):
            for key, fmt in (("ul", "{:+9.1f}"), ("p50", "{:+8.2f}"), ("hold", "{:+8.2f}")):
                df = diffs(rr, arm, key)
                cells.append(fmt.format((df["real"] - df["plac"]).mean()) if len(df) else "      n/a")
        print(f"    {name:24} {len(rr):>3}   " + " ".join(cells[:3]) + "   " + " ".join(cells[3:]))
    pd.to_pickle(recs, os.path.join("_implied_move_cache", "trigger_vs_random.pkl"))


if __name__ == "__main__":
    main()
