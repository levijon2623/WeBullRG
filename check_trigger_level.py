# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trigger_level.py
======================
THE TRIGGER'S SPOT AS A LEVEL FOR THE REST OF THE DAY (user, 2026-10-16)

"How does price behave on flow_trigger days, after the trigger, relative to
spot at the time of the trigger? How many times does it fall back below, and
break back above? And can it be used as a level for re-entry that day, if
profit was taken earlier when above the level -- or vice versa for puts?"

TRIGGERS  the FIRST trigger per rule-day, full live gate stack
          (sim_core.deployed_candidates), 9 enabled rules. L = the underlying's
          1m close at the trigger minute. Everything is SIGNED to the trade
          (calls: above L = with the trade; puts: below L = with the trade),
          so "above" below means "on the trade's side".
BAND      a cross must clear a dead band so one-tick wiggles don't count:
          b = 0.05 x R, R = median RTH (high - low) / open over the ticker's
          prior 20 sessions (scale-free across SPY .. META). Reported also at
          b = 0 (raw 1m closes) and 0.10 R.

PART A -- DESCRIPTIVE (no score)
    per rule-day, from the trigger to the flatten: the number of times price
    falls back through L (with -> against) and breaks back through (against
    -> with), time on the trade's side, where it closes. Compared with the
    SAME numbers for a random minute's spot on the same day (K = 20, minutes
    drawn from the rule's own first-trigger minutes), because ANY price level
    gets crossed a lot -- the question is whether L is crossed less (held)
    or more than an arbitrary level.

PART B -- RE-ENTRY ON THE RECLAIM (PRE-REGISTERED, scored)
    EVENT    after the trigger price goes above L + b, then falls below L - b,
             then closes back above L + b: re-enter at that 1m close (the first
             such reclaim of the day only), if it is at least 25 min before
             the rule's flatten. Same contract pick as the rule (right, dte
             order, strike offset, $0.50 floor), the bot's entry fill.
             Whether profit was taken on the first leg does not change the
             re-entry's own outcome, so the event is defined on price alone.
    METRICS  C_UL underlying at the flatten, signed, bp; C_50 option reaches
             SELL >= +50%; C_HOLD option held to the flatten, SELL ROE
             (check_trigger_paths.path_row, identical scoring).
    ARMS     (unit = rule-day with an event; d = real - mean(placebos))
      R1 SAME LEVEL, RANDOM TIME: re-enter at a random minute between the
         fall-below and the cutoff -> does WAITING FOR THE RECLAIM add
         anything over just getting back in after the pullback?
      R2 RANDOM LEVEL: the same above / below / reclaim rule applied to a
         random same-day minute's spot (event required) -> is the TRIGGER's
         spot special, or does any reclaimed level do the same?
    PASS     mean d > 0, day-block bootstrap 95% CI lower > 0, both halves
             (split 2025-08-21) > 0. 6 tests (3 x 2), no correction.
    Reported: all reclaims (not just the first), the first-leg trigger's own
    numbers on the same days, and per-rule.
MACHINERY
    M1  IDENTITY: the synthetic builder at the trigger minute reproduces the
        trigger's own path row (every d = 0).
    M2  PLANT: +30 bp on the real C_UL must pass on both arms.

RUN LOG
  2026-10-16 run 1: 192 rule-days. M1 identity PASS (192/192), M2 plant PASS both arms.
    PART A (band 0.05 R): falls back through L mean 0.8 (random level 0.9);
      51% of days never fall back, 30% once, 12% twice. Gets to the trade's
      side 91% (random 88), then falls back 55% (62), then reclaims 66% (74).
      Time on the trade's side 63% vs 56%, ends there 69% vs 62% -- that gap
      is the day's DIRECTION (known, check_trigger_vs_random), not the level:
      crossing counts are the same as an arbitrary same-day level at every
      band (raw closes: 3.8 vs 3.7 falls/day). L is not a level price respects.
    PART B: 95 days fell back, 48 reclaimed >= 25 min before the flatten.
      R1 random time: C_UL +4.2 bp [-8.8, +18.3] FAIL; C_50 +0.057 FAIL;
        C_HOLD -0.019 FAIL. R2 random level: C_UL +8.3 bp [-0.9, +18.2] FAIL
        (both halves +); C_50 +0.081 FAIL; C_HOLD +0.061 FAIL (IS half -).
      Re-entry itself: underlying +49 bp, 75% with the trade, but the option
      held to the flatten -3.5% (median -45%) -- the late entry pays theta.
      Every reclaim (n=81) +12% is IS-only (+27 / OOS -11) and one GLD day.
      Underpowered (n=48: C_HOLD CI is +-20 pp) -- a big edge would show, a
      small one would not.
    VERDICT: no. The trigger's spot is an ordinary level; what persists after
      a pullback is the day's direction, and a random re-entry time after the
      pullback catches it just as well.

Usage:  python check_trigger_level.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import directional_flow_backtester as D                   # noqa: E402
import sim_core                                            # noqa: E402
from check_trigger_paths import path_row, ul_closes       # noqa: E402
from check_trigger_vs_random import synth, diffs, score, line  # noqa: E402
from config import RULES                                   # noqa: E402

SPLIT = sim_core.SPLIT
K, SEED = 20, 20261016
BANDS = (0.0, 0.05, 0.10)          # x R; 0.05 is primary
PRIMARY = 0.05
CUTOFF = 25                        # re-entry no later than flatten - 25 min
METRICS = (("ul", "C_UL  underlying at flatten, bp", "{:+.1f}"),
           ("p50", "C_50  reaches +50%", "{:+.3f}"),
           ("hold", "C_HOLD held to flatten, ROE", "{:+.3f}"))


def day_ranges(tk):
    """{date: R} = median (high - low) / open over the PRIOR 20 sessions."""
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959))
         .group_by("date").agg(pl.col("high").max(), pl.col("low").min(),
                               pl.col("open").sort_by("mod").first())
         .sort("date").to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    r = (d["high"] - d["low"]) / d["open"]
    med = r.shift(1).rolling(20, min_periods=20).median()
    return dict(zip(d["date"], med))


def walk_level(c, m, eod, s, band):
    """Signed path from minute m vs L = c[m]. -> dict of crossing stats + events."""
    i0, i1 = m - 570, min(eod, 960) - 570
    if i1 - i0 < 2:
        return None
    L = c[i0]
    w = s * (c[i0:i1] - L) / L                     # fraction, signed to the trade
    side, falls, breaks = 0, 0, 0
    t_above = t_below = None
    reclaims = []                                  # minutes (mod) of every reclaim
    first_above_seen = False
    for k in range(1, len(w)):
        x = w[k]
        if x > band:
            if side == -1:
                breaks += 1
                if first_above_seen:
                    reclaims.append(m + k)
            if side != 1 and not first_above_seen:
                first_above_seen = True
            side = 1
        elif x < -band:
            if side == 1:
                falls += 1
                if t_below is None:
                    t_below = m + k
            side = -1
    return dict(falls=falls, breaks=breaks, with_frac=float(np.mean(w[1:] > 0)),
                end_with=float(w[-1] > 0), went_above=first_above_seen,
                t_below=t_below, reclaims=reclaims,
                ul_after_fall=(float(w[-1] - w[t_below - m]) * 1e4 if t_below is not None else np.nan))


def metrics(row):
    return dict(ul=row.get("ul_last", np.nan), p50=float(row["peak"] >= 0.5), hold=row["last"])


def run_rule(r, rng, ulc, rng_map):
    tk, eod = r["ticker"], sim_core.eod_mod(r)
    s = 1 if r["direction"].upper() == "CALL" else -1
    cand = sim_core.deployed_candidates(D, r)
    first, seen = [], set()
    for d, m, payload in cand:
        if d not in seen:
            seen.add(d)
            first.append((d, m, payload))
    pool = np.array([m for _, m, _ in first])
    desc, events, specs = [], [], []
    for i, (d, m, payload) in enumerate(first):
        c, R = ulc.get(d), rng_map.get(d)
        if c is None or R is None or R != R:
            continue
        rec = dict(rule=r["name"], tk=tk, date=d, m=m, paper=r["name"] in sim_core.PAPER_ONLY,
                   real={}, rand={b: [] for b in BANDS})
        for b in BANDS:
            rec["real"][b] = walk_level(c, m, eod, s, b * R)
        rand_m = [int(x) for x in rng.choice(pool, K)]
        for mr in rand_m:
            for b in BANDS:
                st = walk_level(c, mr, eod, s, b * R)
                if st is not None:
                    rec["rand"][b].append(st)
        desc.append(rec)
        # ---- part B on the primary band
        st = rec["real"][PRIMARY]
        if st is None:
            continue
        ok = [t for t in st["reclaims"] if t <= eod - CUTOFF]
        ev = dict(rule=r["name"], tk=tk, date=d, m=m, paper=rec["paper"], trig_payload=payload,
                  t_below=st["t_below"], t_re=ok[0] if ok else None, all_re=ok,
                  R1=[], R2=[], ID=[], ALL=[])
        events.append(ev)
        specs.append((ev, "ID", d, m))
        if ev["t_re"] is None:
            continue
        specs.append((ev, "REAL", d, ev["t_re"]))
        for t in ok[1:]:
            specs.append((ev, "ALL", d, t))
        lo_, hi_ = ev["t_below"] + 1, eod - CUTOFF
        if hi_ >= lo_:
            for t in rng.integers(lo_, hi_ + 1, K):
                specs.append((ev, "R1", d, int(t)))
        for mr in rand_m:
            sr = walk_level(c, mr, eod, s, PRIMARY * R)
            if sr is None:
                continue
            ok2 = [t for t in sr["reclaims"] if t <= eod - CUTOFF]
            if ok2:
                specs.append((ev, "R2", d, ok2[0]))
    # one permissive build for every distinct (date, minute)
    rb = {k: r[k] for k in ("name", "ticker", "direction", "dte", "hours", "strike_offset") if k in r}
    rb["flow_abs"] = 0.0
    rb["hours"] = list(range(9, 16))               # re-entries can fall outside the rule's trigger hours
    uniq = sorted({(d, m) for _, _, d, m in specs})
    built = sim_core.build_candidates(D, rb, trigs=[synth(r, d, m) for d, m in uniq])
    rows = {}
    for d, m, payload in built:
        if (d, m) not in rows:
            pr = path_row(r, d, m, payload, ulc, eod)
            if pr is not None:
                rows[(d, m)] = metrics(pr)
    for ev, arm, d, m in specs:
        if (d, m) not in rows:
            continue
        if arm == "REAL":
            ev["real"] = rows[(d, m)]
        else:
            ev[arm].append(rows[(d, m)])
    for ev in events:
        pr = path_row(r, ev["date"], ev["m"], ev["trig_payload"], ulc, eod)
        ev["trig"] = metrics(pr) if pr is not None else None
        ev.pop("trig_payload")
    return desc, events


def part_a(desc):
    print("\n  ======== PART A -- PRICE vs THE TRIGGER'S SPOT (first trigger per rule-day, to the flatten) ========")
    print("    'with' = on the trade's side of L. Random = the same stats for a random same-day minute's spot.")
    for b in BANDS:
        R = [x["real"][b] for x in desc if x["real"][b] is not None]
        P = [st for x in desc for st in x["rand"][b]]
        def blk(X, who):
            f = np.array([x["falls"] for x in X]); br = np.array([x["breaks"] for x in X])
            above = np.array([x["went_above"] for x in X])
            fell = np.array([x["t_below"] is not None for x in X])
            re = np.array([len(x["reclaims"]) > 0 for x in X])
            uaf = np.array([x["ul_after_fall"] for x in X if x["ul_after_fall"] == x["ul_after_fall"]])
            print(f"      {who:7} n={len(X):>5}  falls back through: mean {f.mean():4.1f} med {np.median(f):3.0f}"
                  f" | breaks back: mean {br.mean():4.1f}  | gets to 'with' side {above.mean():4.0%}"
                  f"  then falls back {fell.sum() / max(above.sum(), 1):4.0%}  then reclaims {re.sum() / max(fell.sum(), 1):4.0%}"
                  f" | time 'with' {np.mean([x['with_frac'] for x in X]):4.0%}  ends 'with' {np.mean([x['end_with'] for x in X]):4.0%}"
                  f" | after 1st fall, move to close {np.mean(uaf):+5.1f} bp")
            return f
        print(f"\n    band {b:.2f} R" + ("  (primary)" if b == PRIMARY else ""))
        f = blk(R, "trigger")
        blk(P, "random")
        hist = np.bincount(np.minimum(f, 6), minlength=7) / len(f)
        print("      trigger days by number of falls back through L: " +
              "  ".join(f"{k}{'+' if k == 6 else ''}: {v:3.0%}" for k, v in enumerate(hist)))


def part_b(events):
    E = [e for e in events if e.get("t_re") is not None and "real" in e]
    n_all = len(events)
    n_fell = sum(e["t_below"] is not None for e in events)
    print("\n  ======== PART B -- RE-ENTER ON THE RECLAIM OF L (band 0.05 R, first reclaim, "
          f">= {CUTOFF} min before flatten) ========")
    print(f"    rule-days {n_all}; fell back through L {n_fell}; reclaimed in time with a buildable contract {len(E)}")
    brng = np.random.default_rng(SEED + 1)
    recs = [dict(date=e["date"], real=e["real"], R1=e["R1"], R2=e["R2"], ID=e["ID"],
                 rule=e["rule"], paper=e["paper"]) for e in E]
    idr = [dict(date=e["date"], real=e["trig"], ID=e["ID"]) for e in events if e["trig"] and e["ID"]]
    worst = max((abs(diffs(idr, "ID", k)["real"] - diffs(idr, "ID", k)["plac"]).max()
                 for k in ("ul", "p50", "hold")), default=np.nan)
    print(f"    M1 identity: {len(idr)} of {n_all} trigger rows rebuilt, max |diff| {worst:.2e}"
          f"   -> {'PASS' if worst < 1e-9 and len(idr) == n_all else 'CHECK'}")
    m2 = [score(diffs(recs, a, "ul", plant=30.0), brng)["ok"] for a in ("R1", "R2")]
    print(f"    M2 plant +30 bp on C_UL: R1 {'PASS' if m2[0] else 'FAIL'}  R2 {'PASS' if m2[1] else 'FAIL'}")
    for arm, title in (("R1", "R1 -- same level, RANDOM TIME after the fall-back"),
                       ("R2", "R2 -- RANDOM LEVEL's reclaim, same day")):
        cov = np.mean([len(x[arm]) for x in recs]) if recs else 0
        print(f"\n    {title}   (placebos per day {cov:.1f})")
        for key, lab, fmt in METRICS:
            print(line(lab, score(diffs(recs, arm, key), brng), fmt))
    print("\n    -- reported --")
    re = pd.DataFrame([dict(date=e["date"], **e["real"]) for e in E])
    tr = pd.DataFrame([dict(date=e["date"], **e["trig"]) for e in E if e["trig"]])
    al = pd.DataFrame([dict(date=e["date"], **x) for e in E for x in [e["real"]] + e["ALL"]
                       ] if E else [])
    for lbl, X in (("re-entry (first reclaim)", re), ("the morning trigger, same days", tr),
                   ("every reclaim, not just first", al)):
        if len(X):
            print(f"    {lbl:32} n={len(X):>4}  ul {X['ul'].mean():+6.1f} bp  with {X['ul'].dropna().gt(0).mean():4.0%}"
                  f"  +50% {X['p50'].mean():4.0%}  hold {X['hold'].mean():+6.1%} (med {X['hold'].median():+5.0%})"
                  f"  IS {X[X['date'] < SPLIT]['hold'].mean():+6.1%}  OOS {X[X['date'] >= SPLIT]['hold'].mean():+6.1%}")
    print(f"\n    per rule (reported; real re-entry hold / ul, and d vs R1, R2 on hold):")
    for name in dict.fromkeys(x["rule"] for x in recs):
        rr = [x for x in recs if x["rule"] == name]
        h = np.mean([x["real"]["hold"] for x in rr]); u = np.nanmean([x["real"]["ul"] for x in rr])
        d1, d2 = diffs(rr, "R1", "hold"), diffs(rr, "R2", "hold")
        print(f"      {name:24} n={len(rr):>3}  hold {h:+6.1%}  ul {u:+6.1f} bp   "
              f"d R1 {((d1['real'] - d1['plac']).mean() if len(d1) else np.nan):+6.1%}   "
              f"d R2 {((d2['real'] - d2['plac']).mean() if len(d2) else np.nan):+6.1%}")


def main():
    rng = np.random.default_rng(SEED)
    desc, events, ulc, rmap = [], [], {}, {}
    for r in [x for x in RULES if x.get("enabled", True)]:
        tk = r["ticker"]
        if tk not in ulc:
            ulc[tk], rmap[tk] = ul_closes(tk), day_ranges(tk)
        a, e = run_rule(r, rng, ulc[tk], rmap[tk])
        print(f"  {r['name']:24} rule-days {len(a):>3}   reclaim events "
              f"{sum(1 for x in e if x.get('t_re') is not None):>3}", flush=True)
        desc += a
        events += e
    part_a(desc)
    part_b(events)
    pd.to_pickle(dict(desc=desc, events=events), os.path.join("_implied_move_cache", "trigger_level.pkl"))


if __name__ == "__main__":
    main()
