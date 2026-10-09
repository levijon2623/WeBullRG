# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_morning_reversion.py
==========================
IS THERE A TIME OF DAY WHEN PRICE GIVES BACK THE MORNING'S MOVE? (user, 2026-10-09)

"If we look at SPY QQQ and IWM, is there a period of time in RTH in which
price reversion from the morning's earlier movement is more common?"

MORNING  M = price(10:30) - open(09:30), s = sign(M). price(T) = close of the
         1m bar ending at T. S = the day's 09:35 ATM straddle (units).
SLOTS    eleven 30-minute slots 10:30-11:00 ... 15:30-16:00.
         rev_w = -s x (price(end) - price(start)) / S   (> 0 = slot moved
         AGAINST the morning). Rate = share of days with rev_w > 0.
SAMPLE   PRIMARY: mornings with |M| >= the ticker's trailing 60-session median
         |M| (>= 20 sessions) -- a morning worth reverting. All mornings
         reported. Deployed window 2024-08-20..2026-08-21 scored, halves
         split 2025-08-21; pre-sample reported, labelled.
NULL     the past two years drifted up, so "down slot after an up morning"
         is not 50/50. Permutation: shuffle the morning signs across days
         within ticker (keeps every slot's real returns, breaks the link to
         the morning) 2,000 times; the MAX over the 11 slots of the mean rev
         gives a family-wise 95th percentile (11 slots looked at, one bar).

PRE-REGISTERED (primary sample, 3 tickers pooled)
    H_TIME   a slot's mean rev_w beats the family-wise permutation p95, both
             halves > 0, >= 5 of 6 slices > 0       (a reversion time exists)
    H_TRADE  that slot's mean rev_w >= +0.03 S (the scalp floor) and its
             day-block bootstrap 95% CI lower > 0  (worth trading)
    Reported, unscored: rate per slot, per ticker, all mornings, morning to
    10:00 / 11:00, and the cumulative give-back curve (median share of the
    morning move retraced by each half hour).
MACHINERY  M1 planting +0.05 S of reversion into 13:00-13:30 must pass H_TIME
           there and only there.

RUN LOG
  2026-10-09 run 1: 1,499 deployed ticker-days, 760 big mornings (median
    |M| 0.44-0.54 S), pre-sample 585. M1 PASS (13:00 only). Family-wise
    permutation p95 +0.029 S.
    H_TIME FAIL: no slot reverts. Best 14:30-15:00 +0.011 S (rate 0.516);
      every rate sits within 0.46-0.52 of a coin flip.
    H_TRADE not reached.
    Shape (reported): rates mostly BELOW 0.5 -- slots lean slightly WITH the
    morning, not against it. Median give-back by the close is NEGATIVE (the
    morning move extends ~14-16%); only 28% of big mornings (34% of all)
    give back over half by 16:00. Same with the morning ending 10:00 or
    11:00, and per ticker.
    Unscored lead: 15:00-15:30 leans CONTINUATION in every cut (rate 0.44-
    0.48, mean -0.013 to -0.024 S; big-morning CI [-0.047, -0.001]) but is
    inside the family-wise band and below the scalp floor -- forward watch
    only (cf. published intraday-momentum findings).
    VERDICT: no time of day when price gives back the morning more often
    than chance. If anything the morning direction persists, weakly.

Usage:  python check_morning_reversion.py
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

import check_implied_breakout_filters as BF             # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
DEP_START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 8, 21)
OPEN = 570
SLOTS = [(a, a + 30) for a in range(630, 960, 30)]       # 10:30 .. 16:00
LOOK, LOOK_MIN = 60, 20
FLOOR = 0.03
N_PERM, N_BOOT = 2000, 2000
SEED = 20261010


def hm(m):
    return f"{m // 60}:{m % 60:02d}"


def load():
    am = BF.am_straddles()
    rows = []
    for tk in TICKERS:
        d = (pl.read_parquet(f"historical/{tk}.parquet")
             .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                            + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
             .filter((pl.col("mod") >= OPEN) & (pl.col("mod") <= 959))
             .select("date", "mod", "open", "close").to_pandas())
        d["date"] = pd.to_datetime(d["date"]).dt.date
        hist = []
        for day, g in d.groupby("date", sort=True):
            g = g.set_index("mod")
            if OPEN not in g.index or day > END:
                continue
            c = g["close"].reindex(np.arange(OPEN, 960)).ffill()
            if c.isna().any() or g.index.max() < 959:
                continue
            st = am.get((tk, day))
            if not st or not st[1]:
                continue
            S = st[1]
            px = lambda T: float(c.loc[T - 1])            # noqa: E731
            o = float(g.loc[OPEN, "open"])
            row = dict(tk=tk, day=day, S=S)
            for end in (600, 630, 660):
                row[f"M{end}"] = (px(end) - o) / S
            row["absM"] = abs(row["M630"])
            prev = hist[-LOOK:]
            row["big"] = (len(prev) >= LOOK_MIN) and row["absM"] >= np.median(prev)
            hist.append(row["absM"])
            for T in range(600, 961, 30):
                row[f"p{T}"] = px(T)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def slot_rev(df, mcol="M630", start=630):
    s = np.sign(df[mcol].to_numpy())
    out = {}
    for a, b in SLOTS:
        if a < start:
            continue
        out[(a, b)] = -s * (df[f"p{b}"] - df[f"p{a}"]).to_numpy() / df["S"].to_numpy()
    return out


def perm_max(df, rng, mcol="M630", start=630):
    """Family-wise null: max over slots of mean rev with morning signs shuffled within ticker."""
    R = {k: v for k, v in slot_rev(df.assign(**{mcol: 1.0}), mcol, start).items()}   # -raw move
    raw = np.vstack([-R[k] for k in R])                  # slots x days, signed slot move / S
    tks = df["tk"].to_numpy()
    s = np.sign(df[mcol].to_numpy())
    mx = np.empty(N_PERM)
    for i in range(N_PERM):
        sp = s.copy()
        for tk in TICKERS:
            ix = np.flatnonzero(tks == tk)
            sp[ix] = rng.permutation(s[ix])
        mx[i] = (-(raw * sp)).mean(axis=1).max()
    return float(np.quantile(mx, 0.95)), list(R)


def dboot(days, x, rng):
    ud, inv = np.unique(days, return_inverse=True)
    a = np.bincount(inv, weights=x, minlength=len(ud))
    k = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = [a[j].sum() / k[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def score(df, rng, title, plant=None):
    rev = slot_rev(df)
    if plant:
        rev[plant] = rev[plant] + 0.05
    p95, _ = perm_max(df, rng)
    days = df["day"].to_numpy()
    o = np.argsort(days, kind="stable")
    print(f"\n  {title}   n = {len(df)} ticker-days   family-wise permutation p95 = {p95:+.4f} S")
    print(f"     slot          rate    mean S   95% CI              halves (IS / OOS)   slices>0  TIME TRADE")
    hits = []
    for k, x in rev.items():
        lo, hi = dboot(days, x, rng)
        hv = [x[days < SPLIT].mean(), x[days >= SPLIT].mean()]
        sl = [x[o][ix].mean() for ix in np.array_split(np.arange(len(x)), 6)]
        t = x.mean() > p95 and all(h > 0 for h in hv) and sum(v > 0 for v in sl) >= 5
        tr = t and x.mean() >= FLOOR and lo > 0
        if t:
            hits.append(k)
        print(f"     {hm(k[0]):>5}-{hm(k[1]):<5}  {np.mean(x > 0):.3f}  {x.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"   {hv[0]:+.4f} / {hv[1]:+.4f}    {sum(v > 0 for v in sl)}/6     "
              f"{'PASS' if t else '  - '}  {'PASS' if tr else '  - '}")
    return hits


def table(df, mcol, start, title):
    rev = slot_rev(df, mcol, start)
    print(f"\n  {title}   n = {len(df)}")
    print("     slot         " + "  ".join(f"{t:>13}" for t in ("ALL rate/mean",) + TICKERS))
    for k, x in rev.items():
        cells = [f"{np.mean(x > 0):.3f} {x.mean():+.3f}"]
        for tk in TICKERS:
            m = (df["tk"] == tk).to_numpy()
            cells.append(f"{np.mean(x[m] > 0):.3f} {x[m].mean():+.3f}")
        print(f"     {hm(k[0]):>5}-{hm(k[1]):<5}  " + "  ".join(f"{c:>13}" for c in cells))


def giveback(df, title):
    print(f"\n  {title}: share of the 09:30-10:30 move given back by time T (median, and % of days > 50%)")
    s = np.sign(df["M630"].to_numpy())
    Mabs = np.abs(df["M630"].to_numpy() * df["S"].to_numpy())
    line = []
    for T in range(660, 961, 30):
        with np.errstate(all="ignore"):
            g = (-s * (df[f"p{T}"] - df["p630"]).to_numpy() / Mabs)[Mabs > 0]
        line.append(f"{hm(T)} {np.median(g):+.2f} ({np.mean(g > 0.5):.0%})")
    for i in range(0, len(line), 4):
        print("     " + "   ".join(line[i:i + 4]))


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    dep = df[(df["day"] >= DEP_START) & (df["day"] <= END)].reset_index(drop=True)
    pre = df[df["day"] < DEP_START].reset_index(drop=True)
    big = dep[dep["big"]].reset_index(drop=True)
    print(f"  deployed ticker-days {len(dep)} (big mornings {len(big)}), pre-sample {len(pre)}")
    print(f"  median |morning| / S: " + "  ".join(
        f"{tk} {dep.loc[dep.tk == tk, 'absM'].median():.3f}" for tk in TICKERS))

    m1 = score(big, rng, "M1 -- planted +0.05 S reversion at 13:00-13:30", plant=(780, 810))
    print(f"    M1 {'PASS' if m1 == [(780, 810)] else 'FAIL'} (hits: {[hm(a) for a, _ in m1]})")

    print("\n  ======== PRIMARY: mornings with |M| >= trailing median ========")
    hits = score(big, rng, "BIG MORNINGS, pooled")
    print(f"\n  VERDICT: H_TIME {'PASS ' + str([hm(a) for a, _ in hits]) if hits else 'FAIL'}")

    print("\n  ======== REPORTED ========")
    table(big, "M630", 630, "BIG mornings by ticker (rate against morning / mean S)")
    table(dep, "M630", 630, "ALL mornings")
    table(dep, "M600", 600, "ALL mornings, morning = 09:30-10:00 (slots from 10:30)")
    table(dep, "M660", 660, "ALL mornings, morning = 09:30-11:00 (slots from 11:00)")
    giveback(big, "BIG mornings give-back")
    giveback(dep, "ALL mornings give-back")
    pb = pre[pre["big"]].reset_index(drop=True)
    if len(pb):
        table(pb, "M630", 630, "PRE-SAMPLE big mornings (extra in-sample, labelled)")


if __name__ == "__main__":
    main()
