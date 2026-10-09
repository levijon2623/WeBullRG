# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.9"]
# ///
"""
check_straddle_legout.py
========================
BUY A STRADDLE WHEN THE VOL GAUGE IS HIGH, THEN CUT THE LOSING LEG (user,
2026-10-09: "I think this theory will result in the loss of the theoretical
money involved, but ... one side will be winning while the other losing.
Maybe when one leg is down X%, or cut the put if RSI is climbing and the call
if RSI is falling").

PRIOR. check_ic_vol_trade: a straddle held 60 minutes loses at every forecast
level (the move is priced in). Cutting the loser turns the rest into "own the
side price is already moving toward" -- and check_burst_entry found no follow-
through after a move. So the expectation is a loss; the test is whether
option convexity changes that.

ENTRIES  the volatility gauge's OUT-OF-SAMPLE walk-forward reading
         (check_vol_gauge.walk, the cap-5 model the deploy rule picked, Platt
         map, trained only on earlier slices) on the 10-minute grid 09:40-
         14:50, slices 1-6 (2024-08-20..2026-08-21). HIGH = p >= 0.65 (the
         viewer's VOL HIGH). One position per ticker at a time: the next
         entry is >= 60 minutes after the last. LOW (p <= 0.35) reported.
TRADE    ATM straddle (strike nearest spot; same-day expiry, else the next),
         check_ic_vol_trade's book and quotes. ENTRY at each leg's ASK.
         A leg SOLD at max(0, bid - c x spread), c = 0.5 if that leg is up on
         its cost, 1.5 if down (sim_core's live cushion, per leg). Hold limit
         60 minutes (the gauge's horizon), 15:55 at the latest.
POLICIES (each leg's P&L checked every minute from entry + 1)
    HOLD     both legs to the hold limit (the baseline)
    LEG20/30/40/50   when a leg's bid is down 20 / 30 / 40 / 50% on its own
             cost, sell it that minute; keep the other to the hold limit
    RSI      from entry + 3: RSI14 (1m) >= 60 -> sell the put; <= 40 -> sell
             the call; first trigger only; keep the other
ROE = (proceeds of both legs) / (cost of both legs) - 1.

PRE-REGISTERED (HIGH entries; day-block bootstrap 2,000)
    For each of the 5 legging policies:
    L1  BETTER THAN HOLDING: paired mean(ROE_policy - ROE_HOLD) > 0, with
        Benjamini-Hochberg q < 0.10 across the 5, and > 0 in both halves
        (split 2025-08-21)
    L2  PROFITABLE: mean ROE > 0 with a 95% CI lower bound > 0, both halves
    A policy that passes L1 AND L2 is a candidate for a forward paper test.
REPORTED  mid fills; LOW entries; how often each policy cuts a leg, and how
          often the kept leg finishes a winner; ROE by year.
MACHINERY M1 the 09:35 straddle mid from this builder equals the implied-
          move cache (check_ic_vol_trade's M1) on 50 random ticker-days;
          M2 a policy that sells both legs at entry returns exactly the
          round-trip cost (bid-out / ask-in), never a profit.

RUN LOG
  2026-10-09 run 1: OOS gauge entries HIGH 1,739 / LOW 4,956 (1,735 / 4,945
    priced). M1 PASS (0.64% median vs cache, 51 ticker-days), M2 PASS (an
    instant round trip costs a median 3.5%).
    HIGH, mean ROE (95% CI), vs HOLD:
      HOLD  -3.1% [-4.9, -1.2]
      LEG20 -2.1% [-4.6, +0.2]  +1.0pp (p 0.28)   LEG30 -1.8% [-4.1, +0.6]  +1.3pp (p 0.07)
      LEG40 -2.3% [-4.5, +0.2]  +0.8pp (p 0.14)   LEG50 -2.2% [-4.4, +0.2]  +0.9pp (p 0.045)
      RSI   -1.5% [-3.9, +0.9]  +1.6pp (p 0.14)
    Every policy FAILS L1 (none survives BH) and L2 (none profitable).
    Legging out helps by ~1-1.6pp, consistently signed but not significant;
    the kept leg wins 50-78% of the time (higher the later the cut).
    THE GAUGE MATTERS for straddles: HIGH -3.1% vs LOW -7.1% held (mid fills
    -0.3% vs -2.2%) -- 4pp less bad, i.e. the forecast is only partly priced
    in, and at MID prices a HIGH-gauge straddle is about breakeven. The
    spread + cushion (~3.5% per round trip, M2) is what makes it a loser.
    VERDICT: no candidate. Closest: RSI (-1.5%) and LEG30 (-1.8%, +1.4% at
    mid; 2026 +2.6%) -- fills, not the idea, are the obstacle.

Usage:  python check_straddle_legout.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_ic_vol_trade as VT                         # noqa: E402
import check_implied_breakout_filters as BF             # noqa: E402
import check_vol_gauge as CG                            # noqa: E402

SPLIT = dt.date(2025, 8, 21)
HIGH, LOW = 0.65, 0.35
HOLD_M, EOD = 60, 15 * 60 + 55
LEGS_X = (0.20, 0.30, 0.40, 50 / 100)
RSI_UP, RSI_DN, RSI_FROM = 60.0, 40.0, 3
N_BOOT = 2000
SEED = 20261009
POLICIES = ("HOLD", "LEG20", "LEG30", "LEG40", "LEG50", "RSI")


def gauge_entries():
    rows = sorted(CG.build_rows(None), key=lambda r: (r["day"], r["m"], r["tk"]))
    CG.rank_inputs(rows)
    X = CG.design(rows)
    y = np.array([r["big"] for r in rows], float)
    pred, _c, _k, _i = CG.walk(rows, X, y, cap=5)
    out = {"HIGH": [], "LOW": []}
    last = defaultdict(lambda: {"HIGH": -999, "LOW": -999})
    for r, p in zip(rows, pred):
        if not np.isfinite(p):
            continue
        for lab, ok in (("HIGH", p >= HIGH), ("LOW", p <= LOW)):
            key = (r["tk"], r["day"])
            if ok and r["m"] - last[key][lab] >= HOLD_M:
                last[key][lab] = r["m"]
                out[lab].append((r["tk"], r["day"], r["m"], float(p)))
    return out


def leg_exit(q, cost):
    b, a = q
    c = 0.5 if b >= cost else 1.5
    return max(0.0, b - c * (a - b))


def simulate(book, legs, m, rsi, fill="real"):
    """{policy: roe} for one straddle bought at m."""
    qin = [VT.quote(book, k, m, 3) for k in legs]
    if any(q is None for q in qin):
        return None
    cost = [a for _b, a in qin] if fill == "real" else [(b + a) / 2 for b, a in qin]
    end = min(m + HOLD_M, EOD)
    qend = [VT.quote(book, k, end, 10) for k in legs]
    if any(q is None for q in qend):
        return None
    sell = (lambda q, c0: leg_exit(q, c0)) if fill == "real" else (lambda q, c0: (q[0] + q[1]) / 2)
    hold_val = [sell(qend[i], cost[i]) for i in (0, 1)]
    tot = sum(cost)
    out = {"HOLD": sum(hold_val) / tot - 1}
    for x in LEGS_X:
        cut = None
        for t in range(m + 1, end + 1):
            qs = [VT.quote(book, k, t, 3) for k in legs]
            if any(q is None for q in qs):
                continue
            loss = [qs[i][0] / cost[i] - 1 for i in (0, 1)]
            worst = int(np.argmin(loss))
            if loss[worst] <= -x:
                cut = (worst, sell(qs[worst], cost[worst]))
                break
        if cut is None:
            out[f"LEG{int(x * 100)}"] = out["HOLD"]
            out[f"LEG{int(x * 100)}_cut"] = 0
            out[f"LEG{int(x * 100)}_keptwin"] = np.nan
        else:
            i, v = cut
            kept = hold_val[1 - i]
            out[f"LEG{int(x * 100)}"] = (v + kept) / tot - 1
            out[f"LEG{int(x * 100)}_cut"] = 1
            out[f"LEG{int(x * 100)}_keptwin"] = float(kept > cost[1 - i])
    cut = None
    for t in range(m + RSI_FROM, end + 1):
        r = rsi.get(t)
        if r is None:
            continue
        i = 1 if r >= RSI_UP else 0 if r <= RSI_DN else None      # legs = [call, put]
        if i is None:
            continue
        q = VT.quote(book, legs[i], t, 3)
        if q is None:
            continue
        cut = (i, sell(q, cost[i]))
        break
    if cut is None:
        out["RSI"], out["RSI_cut"], out["RSI_keptwin"] = out["HOLD"], 0, np.nan
    else:
        i, v = cut
        out["RSI"] = (v + hold_val[1 - i]) / tot - 1
        out["RSI_cut"], out["RSI_keptwin"] = 1, float(hold_val[1 - i] > cost[1 - i])
    # M2: both legs sold at entry
    out["M2"] = (sum(leg_exit(qin[i], cost[i]) for i in (0, 1)) / tot - 1) if fill == "real" else 0.0
    return out


def run(entries):
    spots, rsis = {}, {}
    for tk in CG.TICKERS:
        b = BF.bars_full(tk).to_pandas()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        for d, g in b.groupby("date"):
            spots[(tk, d)] = dict(zip(g["mod"].astype(int), g["close"].astype(float)))
        sf, _hl = BF.series_features(tk)
        for (d, m), v in sf.items():
            rsis.setdefault((tk, d), {})[m] = v["rsi"]
    paths = {os.path.basename(os.path.dirname(p)).split("=")[1]: p
             for p in glob.glob(f"{VT.SILVER}/date=*/bars.parquet")}
    by_day = defaultdict(list)
    for lab, es in entries.items():
        for e in es:
            by_day[e[1]].append((lab,) + e)
    rows, m1 = [], []
    for n, day in enumerate(sorted(by_day), 1):
        p = paths.get(day.isoformat())
        if p is None:
            continue
        books = VT.day_book(p, day)
        for lab, tk, _d, m, gp in by_day[day]:
            bk, sp = books.get(tk), (spots.get((tk, day)) or {}).get(m)
            if not bk or not sp:
                continue
            lg = VT.legs_for(bk, day, sp, m)
            if not lg:
                continue
            legs = lg["straddle"]
            res = simulate(bk, legs, m, rsis.get((tk, day), {}), "real")
            mid = simulate(bk, legs, m, rsis.get((tk, day), {}), "mid")
            if res is None or mid is None:
                continue
            rows.append(dict(lab=lab, tk=tk, day=day, m=m, p=gp, **res,
                             **{f"mid_{k}": mid[k] for k in POLICIES}))
        if len(m1) < 50 and n % 7 == 0:
            for tk in CG.TICKERS:
                bk, sp = books.get(tk), (spots.get((tk, day)) or {}).get(575)
                if bk and sp:
                    lg = VT.legs_for(bk, day, sp, 575)
                    if lg:
                        qs = [VT.quote(bk, k, 575, 2) for k in lg["straddle"]]
                        if all(qs):
                            m1.append((tk, day, sum((b + a) / 2 for b, a in qs)))
        if n % 50 == 0:
            print(f"    {day}: {len(rows)} trades", flush=True)
    return pd.DataFrame(rows), m1


def dboot(df, col, rng, n=N_BOOT):
    x = df[col].to_numpy(float)
    ud, inv = np.unique(df["day"].to_numpy(), return_inverse=True)
    a = np.bincount(inv, weights=x, minlength=len(ud))
    k = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = np.array([a[j].sum() / k[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(n))])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return a.sum() / k.sum(), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), min(1.0, p)


def main():
    rng = np.random.default_rng(SEED)
    ent = gauge_entries()
    print(f"  gauge entries (OOS, one per ticker per 60m): HIGH {len(ent['HIGH'])}, LOW {len(ent['LOW'])}")
    df, m1 = run(ent)
    # M1 / M2
    import check_implied_move as CI
    import polars as pl
    st = CI.straddles().filter(pl.col("win") == "am")
    cache = {(r, d): s for r, d, s in st.select("root", "date", "strad").iter_rows()}
    diffs = [abs(v - cache[(tk, d)]) / cache[(tk, d)] for tk, d, v in m1 if (tk, d) in cache]
    m1ok = len(diffs) >= 20 and np.median(diffs) <= 0.01
    m2ok = bool((df["M2"] <= 1e-12).all())
    print(f"  M1 09:35 straddle vs cache: median |diff| {np.median(diffs):.2%} on {len(diffs)}: {'PASS' if m1ok else 'FAIL'};"
          f"  M2 instant round trip never profits (median {df['M2'].median():+.1%}): {'PASS' if m2ok else 'FAIL'}")
    if not (m1ok and m2ok):
        sys.exit("machinery checks failed -- not scoring")

    for lab in ("HIGH", "LOW"):
        d = df[df["lab"] == lab].copy()
        print(f"\n  ===== {lab} gauge entries: {len(d)} straddles ({d['tk'].value_counts().to_dict()}) =====")
        print(f"    {'policy':6} {'mean ROE':>9} {'95% CI':>17} {'halves':>15}  {'vs HOLD':>8} {'95% CI':>17} {'p':>6}"
              f"  {'cut%':>5} {'kept wins':>9}  {'mid':>7}")
        res, ps = {}, []
        for pol in POLICIES:
            m, lo, hi, _p = dboot(d, pol, rng)
            hv = [d.loc[d["day"] < SPLIT, pol].mean(), d.loc[d["day"] >= SPLIT, pol].mean()]
            if pol != "HOLD":
                d[f"{pol}_dh"] = d[pol] - d["HOLD"]
                dm, dlo, dhi, dp = dboot(d, f"{pol}_dh", rng)
                dhv = [d.loc[d["day"] < SPLIT, f"{pol}_dh"].mean(), d.loc[d["day"] >= SPLIT, f"{pol}_dh"].mean()]
                ps.append(dp)
            else:
                dm = dlo = dhi = dp = np.nan
                dhv = [np.nan, np.nan]
            cut = d.get(f"{pol}_cut")
            kw = d.get(f"{pol}_keptwin")
            res[pol] = (m, lo, hv, dm, dlo, dhv)
            print(f"    {pol:6} {m:+9.1%} [{lo:+7.1%},{hi:+7.1%}] {hv[0]:+7.1%}/{hv[1]:+6.1%}  "
                  + (f"{dm:+8.1%} [{dlo:+7.1%},{dhi:+7.1%}] {dp:6.3f}" if pol != "HOLD" else f"{'':8} {'':17} {'':6}")
                  + f"  {(cut.mean() if cut is not None else 0):5.0%} {(np.nanmean(kw) if kw is not None and kw.notna().any() else np.nan):9.0%}"
                  + f"  {d[f'mid_{pol}'].mean():+7.1%}")
        if lab == "HIGH":
            pols = [p for p in POLICIES if p != "HOLD"]
            o = np.argsort(ps)
            k = max((i for i in range(len(ps)) if ps[o[i]] <= 0.10 * (i + 1) / len(ps)), default=-1)
            rej = set(np.array(pols)[o[:k + 1]]) if k >= 0 else set()
            print("\n    VERDICT (HIGH):")
            for pol in pols:
                m, lo, hv, dm, dlo, dhv = res[pol]
                l1 = pol in rej and dm > 0 and all(x > 0 for x in dhv)
                l2 = m > 0 and lo > 0 and all(x > 0 for x in hv)
                print(f"      {pol:6} L1 better-than-hold {'PASS' if l1 else 'FAIL'}   L2 profitable {'PASS' if l2 else 'FAIL'}"
                      f"   -> {'CANDIDATE' if l1 and l2 else 'no'}")
        print("    by year (HOLD / LEG30 / RSI): " + "  ".join(
            f"{y}: {g['HOLD'].mean():+.1%} / {g['LEG30'].mean():+.1%} / {g['RSI'].mean():+.1%}"
            for y, g in d.groupby(pd.to_datetime(d['day']).dt.year)))


if __name__ == "__main__":
    main()
