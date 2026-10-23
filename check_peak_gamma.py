# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_peak_gamma.py
===================
HOW OFTEN DOES PRICE REACH (OR GET CLOSE TO) PEAK GAMMA? (user, 2026-10-22)

A SCREEN, extending check_magnet (2026-09-03: the +gamma node is touched at
exactly the rate of ANY level that far away -- SPY 33 vs 34%, QQQ 37 vs 37%,
IWM 30 vs 31%). New here: heavy vs light volume days, a cleaner regime split,
"close to" (not only touched), and the NEXT day's peak gamma.

PROFILE  check_gamma_walls.profile_at at 09:35 ET, expiries 0-5 DTE (as
         check_magnet), dealer convention calls + / puts -, gamma x OI x S^2.
NODES    POS  = strike with the most positive net gamma (check_magnet's node)
         ABS  = strike with the largest |net gamma| (the heatmap's peak)
         EX0  = POS computed WITHOUT today's 0DTE (tomorrow's-and-later expiries)
         NEXT = tomorrow's 09:35 POS node (descriptive only: does peak gamma
                FOLLOW price? is it nearer today's close than today's open?)
OUTCOME  after 09:35: REACHED (within 5 bp), CLOSE (within 10 / 25 bp).
BASELINE the same ticker's post-09:35 up / down range distribution, built from
         the days IN THE SAME SPLIT: P(any level at that distance and side is
         reached). Heavy days have bigger ranges, so a pooled baseline would
         invent a "magnet" on heavy days -- the split's own baseline removes it.
SPLITS   volume: full-day RVOL (day volume / prior-20 median; known only at the
         close -> descriptive) and first-30-min RVOL (09:30-09:59, known at
         10:00 -> usable), terciles per ticker; regime: sign of the 09:35 net
         profile (+ suppression / - amplification); distance bucket.
SCREEN RULE (what would justify a proper pre-registered test): a cell's edge
  (node rate - its baseline) beyond 2 standard errors, the same sign in both
  halves (split 2025-08-21) and in >= 2 of 3 tickers.
WINDOW   2024-08-20 .. 2026-09-18, SPY / QQQ / IWM.

Usage:  python check_peak_gamma.py

RUN LOG
  2026-10-22 run 1: 512 days per ticker; POS node above spot 90-92% of days;
    net + 58 / 58 / 51%.
    POS (most +gamma): reached 33% vs baseline 37% (-3.1pp, 2se 2.5); within
      10 bp 37 vs 39; within 25 bp 47 vs 48; median closest approach 32 bp.
    ABS (heatmap peak): reached 43 vs 45; 10 bp 47 vs 48; 25 bp 57 vs 58;
      closest approach median 15 bp.
    EX0 (no 0DTE): 21 vs 26 (-4.2pp).
    Every split is AT or BELOW its own baseline, never above: regime + -1.8,
      - -4.4 (raw 46% vs 18% -- the folklore gap -- is all distance / range);
      first-30-min volume light -2.2 / heavy -5.2; full-day light -3.6 /
      heavy -5.3; distance > 1.5% -4.1. Screen flags fire only on the
      NEGATIVE side: a mild "not reached" (-3 to -6pp), not a magnet --
      consistent with walls being weakly respected (check_gamma_walls /
      check_wall_break: respected when reached, not tradeable).
    NEXT: tomorrow's node = today's 40% of days; when both moved, it moved
      WITH today's price 74% of 925 days; nearer today's close than open 55%;
      today's price reached tomorrow's node 25% vs 30% baseline. Peak gamma
      follows price (OI is built where price went), not the reverse.
  VERDICT: no magnet in any volume, regime or next-day cut. Nothing to
    pre-register.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from check_gamma_walls import load_day, profile_at, _minute_px, _silver_dates   # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 9, 18)
ASOF, MAX_DTE = 575, 5
TOL = {"reach": 5e-4, "10bp": 1e-3, "25bp": 2.5e-3}
NODES = ("POS", "ABS", "EX0")


def volumes(t):
    d = (pl.read_parquet(f"historical/{t}.parquet", columns=["date", "minute_et", "volume"])
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959))
         .group_by("date").agg(pl.col("volume").sum().alias("v"),
                               pl.col("volume").filter(pl.col("mod") < 600).sum().alias("v30")).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d.set_index("date").sort_index()
    d["rv"] = d["v"] / d["v"].shift(1).rolling(20, min_periods=15).median()
    d["rv30"] = d["v30"] / d["v30"].shift(1).rolling(20, min_periods=15).median()
    return d


def nodes_of(by):
    pos = {k: v for k, v in by.items() if v > 0}
    out = {}
    if pos:
        out["POS"] = max(pos, key=pos.get)
    if by:
        out["ABS"] = max(by, key=lambda k: abs(by[k]))
    return out


def day_rows(t, days):
    rows, nxt_node = [], {}
    prev = None
    for d in days:
        day = load_day(t, d, MAX_DTE, 0)
        if day is None:
            continue
        pr = profile_at(day, ASOF, False)
        if pr is None:
            continue
        s0, by, _, _ = pr
        if not by:
            continue
        nd = nodes_of(by)
        ex = profile_at(day.filter(pl.col("dte") >= 1), ASOF, False)
        if ex is not None and ex[1]:
            e = nodes_of(ex[1])
            if "POS" in e:
                nd["EX0"] = e["POS"]
        px = _minute_px(day, ASOF, None)
        if px.is_empty():
            continue
        p = px["px"].to_numpy()
        o = _minute_px(day, 570, 575)["px"].to_numpy()
        r = dict(tk=t, d=d, s0=s0, hi=p.max(), lo=p.min(), close=p[-1], open=o[0] if len(o) else s0,
                 net="+" if sum(by.values()) >= 0 else "-", **{f"n_{k}": v for k, v in nd.items()})
        rows.append(r)
        if prev is not None and "POS" in nd:
            nxt_node[prev] = nd["POS"]
        prev = d
    df = pd.DataFrame(rows)
    df["n_NEXT"] = df["d"].map(nxt_node)
    return df


def score_rows(df, node):
    """per-row dist / side / outcomes for one node column."""
    x = df.dropna(subset=[f"n_{node}"]).copy()
    k = x[f"n_{node}"].astype(float)
    x["dist"] = (k - x["s0"]).abs() / x["s0"]
    x["above"] = k > x["s0"]
    x = x[x["dist"] > 1e-4]
    x["up"] = (x["hi"] - x["s0"]) / x["s0"]
    x["dn"] = (x["s0"] - x["lo"]) / x["s0"]
    rng_side = np.where(x["above"], x["up"], x["dn"])
    for lab, tol in TOL.items():
        x[lab] = (rng_side >= x["dist"] - tol).astype(float)
    x["gap"] = np.maximum(0.0, x["dist"] - rng_side)          # closest approach, fraction of spot
    return x


def baseline(sub, lab):
    """mean over rows of P(range on that side >= dist - tol), ranges from the same rows' ticker."""
    tol = TOL[lab]
    out = []
    for t, g in sub.groupby("tk"):
        up, dn = g["up"].to_numpy(), g["dn"].to_numpy()
        for dist, ab in zip(g["dist"], g["above"]):
            arr = up if ab else dn
            out.append(np.mean(arr >= dist - tol))
    return float(np.mean(out)) if out else np.nan


def cell(sub, lab="reach"):
    n = len(sub)
    if n < 20:
        return None
    rate = sub[lab].mean()
    base = baseline(sub, lab)
    se = np.sqrt(max(base * (1 - base), 1e-9) / n)
    halves = []
    for m in (sub["d"] < SPLIT, sub["d"] >= SPLIT):
        s = sub[m]
        halves.append(s[lab].mean() - baseline(s, lab) if len(s) >= 10 else np.nan)
    per = []
    for t in TICKERS:
        s = sub[sub["tk"] == t]
        per.append(s[lab].mean() - baseline(s, lab) if len(s) >= 10 else np.nan)
    return dict(n=n, rate=rate, base=base, edge=rate - base, se=se, halves=halves, per=per)


def line(lab, c):
    if c is None:
        return f"     {lab:30} (thin)"
    same_h = all(np.sign(h) == np.sign(c["edge"]) for h in c["halves"] if not np.isnan(h))
    same_t = sum(1 for p in c["per"] if not np.isnan(p) and np.sign(p) == np.sign(c["edge"]))
    flag = "<< SCREEN" if abs(c["edge"]) > 2 * c["se"] and same_h and same_t >= 2 else ""
    return (f"     {lab:30} n={c['n']:4d}  node {c['rate']:4.0%}  baseline {c['base']:4.0%}  edge {c['edge'] * 100:+5.1f}pp"
            f" (2se {2 * c['se'] * 100:4.1f})  halves {c['halves'][0] * 100:+5.1f}/{c['halves'][1] * 100:+5.1f}"
            f"  tickers {' '.join(f'{p * 100:+.0f}' for p in c['per'])}  {flag}")


def main():
    days = [d for d in _silver_dates() if START <= d <= END]
    frames = []
    for t in TICKERS:
        df = day_rows(t, days)
        v = volumes(t)
        df["rv"] = df["d"].map(v["rv"])
        df["rv30"] = df["d"].map(v["rv30"])
        for c in ("rv", "rv30"):
            q = df[c].quantile([1 / 3, 2 / 3]).to_numpy()
            df[f"{c}_t"] = np.where(df[c] <= q[0], "light", np.where(df[c] >= q[1], "heavy", "mid"))
            df.loc[df[c].isna(), f"{c}_t"] = None
        frames.append(df)
        print(f"  {t}: {len(df)} days, node POS above spot {np.mean(df['n_POS'] > df['s0']):.0%}, "
              f"net + {np.mean(df['net'] == '+'):.0%}")
    D = pd.concat(frames, ignore_index=True)

    for node in NODES:
        X = score_rows(D, node)
        print(f"\n  ======== NODE {node} ========   days {len(X)}   median distance at 09:35 {X['dist'].median() * 100:.2f}%")
        for lab in ("reach", "10bp", "25bp"):
            print(line(f"ALL, {lab}", cell(X, lab)))
        print(f"     closest approach: median {X['gap'].median() * 1e4:.0f} bp (0 = reached)")
        print("   by distance at 09:35 (reach):")
        for lo, hi, lab in ((0, .003, "<0.3%"), (.003, .007, "0.3-0.7%"), (.007, .015, "0.7-1.5%"), (.015, 1, ">1.5%")):
            print(line(f"  {lab}", cell(X[(X["dist"] >= lo) & (X["dist"] < hi)])))
        print("   by regime (09:35 net gamma sign):")
        for s in ("+", "-"):
            print(line(f"  net {s}", cell(X[X["net"] == s])))
        for vc, nm in (("rv_t", "full-day volume (descriptive)"), ("rv30_t", "first-30-min volume (usable)")):
            print(f"   by {nm}:")
            for g in ("light", "mid", "heavy"):
                print(line(f"  {g}", cell(X[X[vc] == g])))
        print("   regime x first-30-min volume:")
        for s in ("+", "-"):
            for g in ("light", "heavy"):
                print(line(f"  net {s}, {g}", cell(X[(X["net"] == s) & (X["rv30_t"] == g)])))

    # NEXT: does tomorrow's peak gamma sit where today's price went?
    N = D.dropna(subset=["n_NEXT", "n_POS"]).copy()
    for c in ("open", "close", "n_POS"):
        N[f"to_{c}"] = (N["n_NEXT"] - N[c]).abs() / N["s0"] * 1e4
    print("\n  ======== NEXT DAY'S PEAK GAMMA (tomorrow's 09:35 POS node), descriptive ========")
    print(f"     days {len(N)}   tomorrow's node == today's node: {np.mean(N['n_NEXT'] == N['n_POS']):.0%}")
    print(f"     median distance from tomorrow's node to: today's OPEN {N['to_open'].median():.0f} bp, "
          f"today's CLOSE {N['to_close'].median():.0f} bp, today's node {N['to_n_POS'].median():.0f} bp")
    print(f"     tomorrow's node nearer today's close than today's open: {np.mean(N['to_close'] < N['to_open']):.0%}")
    mv = np.sign(N["close"] - N["open"])
    sh = np.sign(N["n_NEXT"] - N["n_POS"])
    m = (mv != 0) & (sh != 0)
    print(f"     node moved the same way as today's price (when both moved): {np.mean(mv[m] == sh[m]):.0%} of {m.sum()} days")
    X = score_rows(D.assign(n_NEXT=D["n_NEXT"]), "NEXT")
    print("     did TODAY's price reach tomorrow's node (look-ahead: shows causality direction, not a signal):")
    print(line("  NEXT, reach", cell(X)))


if __name__ == "__main__":
    main()
