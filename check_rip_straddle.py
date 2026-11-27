"""
check_rip_straddle.py -- PRE-REGISTRATION, written 2026-11-26 before any straddle P&L below was computed (user: "pre-register
the straddle test"). Is the BREAK pill's size forecast UNDER-PRICED by 0DTE options? A size forecast pays only if
options do not already price it (earlier straddle / vol-gauge tests: priced in), and legging out adds nothing (break
direction is a coin flip at every horizon, scratchpad break_persist.py, discovery only) -- so the test is the plain
straddle.

EVENT     a BREAK out of a quiet stretch (rip_features.scan + features "brk": the first 1-min close outside the band; a
          stretch capped at 15:00 counts when the 15:01 close is outside), scored with the FROZEN B model
          (rip_model_frozen.json, models.brk: inputs, coefficients, per-ticker ranks, quintile cuts). SPY / QQQ / IWM.
          TOP = score quintile 5 by the frozen cuts.
TRADE     buy the 0DTE ATM straddle (call + put at the strike nearest the break minute's close) at the quotes of the minute
          AFTER the break (bar t+1's bid/ask close; a leg without a bar then uses its last bar within the 3 minutes
          before; no quote for a leg -> the event is skipped, counted); sell both legs 60 minutes later (bar t+61, same
          3-minute rule) or at 15:55 if that is earlier. Skipped, counted: entry straddle mid < $0.10, or either leg's
          spread > 15% of its mid.
          Prices at the user's measured fills (cleanbot-real-fills): buy mid + $0.0006 / share, sell mid - $0.0008, fees
          $0.001 / share per leg per side -> R = net P&L / entry cost (return on premium), and $ per 1-lot straddle.
SAMPLES   H1 = the screen's HOLDOUT months (even months, 2024-08-20 .. 2026-11-13): the B score's IC is known there
             (+0.169) -- so H1 asks only whether that size is priced; option P&L was never looked at. PARTLY CLEAN, said.
          H2 = 2023-11-01 .. 2024-08-19, every month: before the cue history, never used by the screen; THRU inputs are
             absent there (no breach events -> ABS_THRU_LAST 0) and the peak zone is missing (rank 0.5), as the frozen
             model treats missing inputs. Independent sessions; the absorption study's bases covered them (counts only).
          D  = the screen's DISCOVERY months: the model was fit there -- reported, never scored.
PRIMARY   S1  H1, TOP breaks: mean R > 0 with the day-block bootstrap 95% CI (2,000 resamples of sessions) above 0.
          S2  H2, TOP breaks: mean R > 0, one-sided day-block p < 0.05.
          PASS = S1 AND S2 -> a forward paper log from the live pill (rip_brk events); nothing traded on it.
REPORTED, NOT TESTED  all breaks; TOP minus the other quintiles; quintiles 4+5; exits at 30 / 120 min and at 15:55;
          R by ticker; the share of TOP breaks that ripped (the screen's RIP_B) next to the straddle's R -- the gap between
          the two is what options already priced.

    python check_rip_straddle.py --discovery        (D only: machinery + descriptive)
    python check_rip_straddle.py --holdout          (H1 + H2, once; writes _ripstraddle_cache/RESULT-holdout.txt)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_ripstraddle_cache")
TKS = ("SPY", "QQQ", "IWM")
BUY, SELL, FEE = 0.0006, 0.0008, 0.001
HOLD, EXITS, LAST_MIN = 60, (30, 60, 120), 15 * 60 + 55
BACK, MIN_MID, MAX_SPR = 3, 0.10, 0.15
H2_LO, H2_HI, LO, HI = "2023-11-01", "2024-08-19", "2024-08-20", "2026-11-13"
SEED = 20261127


def sample_of(d):
    if H2_LO <= d <= H2_HI:
        return "H2"
    if LO <= d <= HI:
        return "D" if int(d[5:7]) % 2 == 1 else "H1"
    return None


def quotes_day(d, tk, strikes):
    """{(K, 'call'|'put'): {minute: (bid, ask)}} for the 0DTE contracts at these strikes."""
    p = os.path.join(LAKE, "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(p):
        return {}
    y = (pl.scan_parquet(p).filter((pl.col("underlying_symbol") == tk) & (pl.col("expiry") == pl.lit(d).str.to_date())
                                   & pl.col("strike").is_in([float(k) for k in strikes]))
         .with_columns(mm=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
         .select("strike", "option_type", "mm", "bid_close", "ask_close").collect())
    out = defaultdict(dict)
    for k, ty, mm, b, a in y.iter_rows():
        if b is not None and a is not None and a >= b > 0:
            out[(int(round(k)), ty)][int(mm)] = (float(b), float(a))
    return out


def q_at(book, m):
    for k in range(m, m - BACK - 1, -1):
        if k in book:
            return book[k]
    return None


def trade(Q, K, t, exit_after):
    """the straddle bought at bar t+1, sold exit_after minutes later (or 15:55); None + reason when skipped."""
    c, p = Q.get((K, "call"), {}), Q.get((K, "put"), {})
    e0, e1 = t + 1, min(t + 1 + exit_after, LAST_MIN)
    if e1 <= e0:
        return None, "no time left"
    qc0, qp0, qc1, qp1 = q_at(c, e0), q_at(p, e0), q_at(c, e1), q_at(p, e1)
    if not (qc0 and qp0 and qc1 and qp1):
        return None, "no quote"
    mc0, mp0 = sum(qc0) / 2, sum(qp0) / 2
    if mc0 + mp0 < MIN_MID:
        return None, "cheap"
    if (qc0[1] - qc0[0]) > MAX_SPR * mc0 or (qp0[1] - qp0[0]) > MAX_SPR * mp0:
        return None, "wide"
    cost = mc0 + mp0 + 2 * BUY + 2 * FEE
    value = sum(qc1) / 2 + sum(qp1) / 2 - 2 * SELL - 2 * FEE
    return dict(R=(value - cost) / cost, usd=100 * (value - cost), cost=100 * cost), None


def events_for(days):
    """break events with frozen B scores and quintiles: the screen's cached features where they exist (D / H1), else built
    with rip_features on a context without cue / peak inputs (H2)."""
    import check_absorption_rip as A
    import check_rip_screen as R
    import rip_cue as RC
    import rip_features as RF
    frz = RC.load_frozen()["models"]["brk"]
    ev = A.build()
    all_days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "ext-1m", "date=*")))
    out = []
    for d in days:
        rows = ev.filter(pl.col("d") == d).to_dicts()
        if not rows:
            continue
        cp = os.path.join(ROOT, "_ripscreen_cache", f"feat-brk-{d}.parquet")
        if os.path.exists(cp):
            F = pl.read_parquet(cp)
            feats = F.to_dicts() if F.height else []
        else:
            i = all_days.index(d)
            a = {tk: R.ext(d, tk) for tk in TKS}
            pa = {tk: R.ext(all_days[i - 1], tk) for tk in TKS} if i else {}
            t30, bvol = {}, {}
            for tk in TKS:
                pr = [x for x in (R.ext(y, tk) for y in all_days[max(0, i - 20): i]) if x is not None and x[-1, 0] >= 840]
                rr = [A.r30(x[:, [0, 2, 3, 4, 5]]) for x in pr]
                t30[tk] = float(np.median([v for v in rr if v])) if rr else None
                cs = []
                for x in pr:
                    cc = np.full(1440, np.nan); cc[x[:, 0].astype(int) % 1440] = np.cumsum(x[:, 5]); cs.append(cc)   # 1440: one H2 day (2024-03-05) has r bars past 16:40
                bvol[tk] = np.array(cs) if cs else None
            bn = defaultdict(list)
            for r in rows:
                bn[r["tk"]].append((r["s"], r["e"]))
            ctx = dict(a=a, pa=pa, t30=t30, base_vol=bvol, expo={tk: R.exposures(d, tk) for tk in TKS},
                       opt={tk: ({}, R.options_day(d).get(tk, ({}, {}))[1]) for tk in TKS}, B=[], S=[], pairs={},
                       hrel=None, peak={}, base_now=bn)
            feats = [f for f in (RF.features(r, "brk", ctx) for r in rows) if f]
        for f in feats:
            for k in RF.DIRECTIONAL:
                v = f.get(k)
                f[f"ABS_{k}"] = abs(float(v)) if v is not None and v == v else np.nan
            z = RC.score(frz, f, f["tk"])
            px = R.ext(d, f["tk"])
            mi = {int(m): j for j, m in enumerate(px[:, 0])}
            out.append(dict(d=d, tk=f["tk"], s=f["s"], t=int(f["t"]), BS=f["BS"], z=z,
                            q=int(np.searchsorted(frz["cuts"], z, side="right")) + 1, RIP_B=f.get("RIP_B"),
                            close=float(px[mi[int(f["t"])], 4]) if int(f["t"]) in mi else None))
    return out


def priced(E):
    rows, skips = [], defaultdict(int)
    by = defaultdict(list)
    for e in E:
        by[(e["d"], e["tk"])].append(e)
    for (d, tk), es in by.items():
        ks = {int(round(e["close"])) for e in es if e["close"]}
        Q = quotes_day(d, tk, ks)
        for e in es:
            if not e["close"]:
                skips["no price"] += 1
                continue
            K = int(round(e["close"]))
            r = dict(e)
            for h in EXITS:
                x, why = trade(Q, K, e["t"], h)
                r[f"R{h}"] = x["R"] if x else None
                if h == HOLD:
                    if x is None:
                        skips[why] += 1
                    else:
                        r.update(usd=x["usd"], cost=x["cost"])
            x, _ = trade(Q, K, e["t"], 10 ** 4)
            r["Reod"] = x["R"] if x else None
            rows.append(r)
    return rows, dict(skips)


def boot(days, vals, n=2000, seed=SEED):
    pos = defaultdict(list)
    for d, v in zip(days, vals):
        pos[d].append(v)
    ud = sorted(pos)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(ud), len(ud))
        xs = [v for k in pick for v in pos[ud[k]]]
        bs.append(np.mean(xs))
    bs = np.array(bs)
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.mean(bs <= 0))


def report(rows, lab):
    ok = [r for r in rows if r.get("R60") is not None]
    print(f"\n=== {lab}: {len(rows)} breaks, {len(ok)} priced ({len({r['d'] for r in ok})} sessions)")
    for name, sel in (("TOP (Q5)", lambda r: r["q"] == 5), ("Q4+Q5", lambda r: r["q"] >= 4), ("Q1-Q4", lambda r: r["q"] <= 4),
                      ("all", lambda r: True)):
        x = [r for r in ok if sel(r)]
        if not x:
            continue
        m, lo, hi, p = boot([r["d"] for r in x], [r["R60"] for r in x], n=1000)
        rip = np.nanmean([r["RIP_B"] for r in x if r["RIP_B"] is not None]) if x else float("nan")
        alt = "  ".join(f"R{h} {np.mean([r[f'R{h}'] for r in x if r.get(f'R{h}') is not None]):+.1%}" for h in (30, 120)) + \
            f"  Reod {np.mean([r['Reod'] for r in x if r.get('Reod') is not None]):+.1%}"
        print(f"  {name:9s} n {len(x):4d} | R60 {m:+.1%} [{lo:+.1%}, {hi:+.1%}] p {p:.3f} | ${np.mean([r['usd'] for r in x]):+.1f}/straddle on "
              f"${np.mean([r['cost'] for r in x]):.0f} | ripped {rip:.0%} | {alt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "ext-1m", "date=*")))
    os.makedirs(CACHE, exist_ok=True)
    if a.discovery:
        rows, sk = priced(events_for([d for d in days if sample_of(d) == "D"]))
        print("skipped:", sk)
        report(rows, "DISCOVERY (in-sample for the model; descriptive)")
    if a.holdout:
        rp = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(rp):
            sys.exit(f"holdout already scored: {rp}")
        lines = []
        for smp, test in (("H1", "S1"), ("H2", "S2")):
            rows, sk = priced(events_for([d for d in days if sample_of(d) == smp]))
            print(f"[{smp}] skipped:", sk)
            report(rows, smp)
            top = [r for r in rows if r["q"] == 5 and r.get("R60") is not None]
            m, lo, hi, p = boot([r["d"] for r in top], [r["R60"] for r in top]) if top else (float("nan"),) * 4
            ok = (m > 0 and lo > 0) if test == "S1" else (m > 0 and p < 0.05)
            lines.append(f"  {test} {smp} TOP breaks: n {len(top)} mean R60 {m:+.2%} [{lo:+.2%}, {hi:+.2%}] p {p:.3f}  {'PASS' if ok else 'FAIL'}")
        print("\n".join(lines))
        with open(rp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
