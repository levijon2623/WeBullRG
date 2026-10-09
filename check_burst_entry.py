# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_burst_entry.py
====================
AFTER A BURST: GET IN NOW, WAIT FOR THE STALL, OR FADE IT? (user, 2026-10-09)

check_poc_stack's exploratory pause numbers: after price arrives anywhere, the
next 5 minutes move ~40% as far as the 5 before (speed regresses after a
burst). The user's three entries for a scalper:
    IMMEDIATE  front-run the next leg: buy WITH the burst at once
    WAIT       let the deceleration play out, then buy WITH the burst
    REVERSION  buy AGAINST the burst once it stalls ("ripped up -> put")

STAGE A, exit-free, on the underlying (SPY, QQQ, IWM; 1m bars). Exit-free,
REVERSION is exactly minus CONTINUATION at the same entry -- so the mean
answers "which way does price drift after a burst, by entry timing"; the
two sides differ only in their TAILS, which are reported for both.

BURST   minute t in 09:45..14:30: |close(t) - close(t-5)| / S at or above the
        95th percentile of that ticker's 5-minute moves (09:35-15:00) over
        the PRIOR 60 sessions (>= 20 sessions). S = the day's 09:35 ATM
        straddle. s = sign of the burst. One event per ticker per 15 minutes.
ENTRIES at minute e (entry = close(e)):
        IMM    e = t
        D3/D5/D10  e = t + 3 / 5 / 10                       (reported)
        STALL  first e in t+1..t+10 whose 2-minute speed |c(e) - c(e-2)| / 2
               <= 0.4 x the burst's speed |c(t) - c(t-5)| / 5 (the 40% from
               check_poc_stack); none -> no STALL entry (counted)
OUTCOMES (straddle units)
        CONT r_h = s x (close(e + h) - close(e)) / S, h = 5, 15, 30;  REV = -CONT
        TAIL_CONT / TAIL_REV: the move WITH / AGAINST the burst reaches
        0.10 S (a scalp) within 15 minutes of e (high / low)
WINDOW  deployed slices 2024-08-20..2026-08-21 scored; halves split at
        2025-08-21; pre-sample 2023-10..2024-08 reported, labelled.

PRE-REGISTERED (h = 15, day-block bootstrap 2,000, calendar days pooled)
    H_CONT   IMM continuation: mean CONT > +0.03 S, 95% CI lower > 0, both
             halves > 0, >= 5 of 6 slices > 0          (front-running works)
    H_WAIT   STALL continuation beats IMM continuation, paired per event:
             mean(CONT_STALL - CONT_IMM) > 0, CI lower > 0, both halves,
             >= 5 of 6 slices                          (waiting helps)
    H_REV    STALL reversion: mean REV > +0.03 S, CI lower > 0, both halves,
             >= 5 of 6 slices                          (fading works)
    0.03 S = the scalp floor used since check_ma5_delta (an ATM 0DTE moves
    ~0.5 S per 1 S; a round trip costs ~0.01-0.02 S). Each hypothesis is
    scored on its own; a pass is a CANDIDATE for an option simulation
    (stage B), not a strategy.
MACHINERY  M1 planting +0.05 S on CONT_STALL makes H_WAIT pass; M2 RANDOM
           events (same tickers / days / times, random sign) give |mean CONT|
           < 0.01 S at every entry.

RUN LOG
  2026-10-09 run 1: 6,343 deployed bursts (4.4 per ticker-day; median size
    0.33 S in 5 minutes), pre-sample 2,250. STALL found in 98%, median 2
    minutes after the burst. M1 PASS, M2 PASS (12,816 random events).
    H_CONT FAIL: IMM continuation at 15m -0.0007 S [-0.011, +0.009].
    H_REV  FAIL: STALL reversion at 15m -0.0050 S [-0.015, +0.005].
    H_WAIT PASS on its criteria: STALL beats IMM by +0.0070 S [+0.0003,
      +0.0135], both halves, 6/6 slices -- but it is the difference between
      two near-zero drifts and ~0.007 S is ~$0.02 of SPY, a fraction of one
      round-trip cost (0.01-0.02 S). Real, consistent, economically nil.
    Shape (reported): the first 5 minutes after a burst retrace slightly
    (IMM h=5 -0.008 S, CI below 0); entering 3-5 minutes later avoids it
    (D5 h=5 +0.006 S); by 15-30m every entry is ~0. Up and down bursts alike.
    TAILS: a 0.10 S move WITH the burst within 15m happens 70-72% of the
    time, AGAINST it 69-73% -- volatility stays high, direction is a coin
    flip (the magnitude-not-direction result again; options price it,
    check_ic_vol_trade).
    VERDICT: after a big 5-minute burst there is no directional follow-
    through and no reversion worth trading; waiting ~2-5 minutes avoids a
    small immediate pullback but gains < one round trip. No stage B.

Usage:  python check_burst_entry.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_implied_breakout_filters as BF             # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
DEP_START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 8, 21)
FIRST, LAST, REF_LO, REF_HI = 585, 870, 575, 900
PCT, LOOK, LOOK_MIN, GAP = 0.95, 60, 20, 15
HS = (5, 15, 30)
TAIL_S, TAIL_H, FLOOR = 0.10, 15, 0.03
ENTRIES = ("IMM", "D3", "D5", "D10", "STALL")
N_BOOT = 2000
SEED = 20261009


def events(random_ctrl=False):
    am = BF.am_straddles()
    rng = np.random.default_rng(SEED + (1 if random_ctrl else 0))
    out = []
    mods = np.arange(570, 960)
    for tk in TICKERS:
        b = BF.bars_full(tk).to_pandas()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        bd = {d: g.set_index("mod") for d, g in b.groupby("date")}
        days = sorted(bd)
        pool = []
        for i, day in enumerate(days):
            g = bd[day]
            c = g["close"].reindex(mods).ffill().to_numpy(float)
            hi = g["high"].reindex(mods).to_numpy(float)
            lo = g["low"].reindex(mods).to_numpy(float)
            st = am.get((tk, day))
            S = st[1] if st else None
            m5 = np.full(len(c), np.nan)
            m5[5:] = c[5:] - c[:-5]
            win = (mods >= REF_LO) & (mods <= REF_HI)
            today = np.abs(m5[win]) / S if S else np.array([])
            prev = pool[max(0, i - LOOK):i]
            if S and len(prev) >= LOOK_MIN and day <= END:
                thr = np.quantile(np.concatenate(prev), PCT)
                last = -999
                for j in range(len(mods)):
                    m = mods[j]
                    if m < FIRST or m > LAST or not np.isfinite(m5[j]) or j - last < GAP:
                        continue
                    if random_ctrl:
                        if rng.random() > 0.05:
                            continue
                        s = 1 if rng.random() < 0.5 else -1
                    else:
                        if abs(m5[j]) / S < thr:
                            continue
                        s = 1 if m5[j] > 0 else -1
                    last = j
                    speed = abs(m5[j]) / 5
                    ent = {"IMM": j, "D3": j + 3, "D5": j + 5, "D10": j + 10, "STALL": None}
                    for e in range(j + 1, j + 11):
                        if abs(c[e] - c[e - 2]) / 2 <= 0.4 * speed:
                            ent["STALL"] = e
                            break
                    row = dict(tk=tk, day=day, m=int(m), s=s, size=abs(m5[j]) / S)
                    for name, e in ent.items():
                        for h in HS:
                            row[f"{name}_c{h}"] = s * (c[e + h] - c[e]) / S if e is not None else np.nan
                        if e is not None:
                            w = slice(e + 1, e + 1 + TAIL_H)
                            with np.errstate(all="ignore"):
                                up, dn = np.nanmax(hi[w]) - c[e], c[e] - np.nanmin(lo[w])
                            fav, adv = (up, dn) if s > 0 else (dn, up)
                            row[f"{name}_tc"] = float(fav / S >= TAIL_S)
                            row[f"{name}_tr"] = float(adv / S >= TAIL_S)
                        else:
                            row[f"{name}_tc"] = row[f"{name}_tr"] = np.nan
                        row[f"{name}_wait"] = (e - j) if e is not None else np.nan
                    out.append(row)
            pool.append(today[np.isfinite(today)])
    return pd.DataFrame(out)


def dboot(df, col, rng, n=N_BOOT):
    x = df[col].to_numpy(float)
    ok = np.isfinite(x)
    ud, inv = np.unique(df["day"].to_numpy(), return_inverse=True)
    a = np.bincount(inv, weights=np.where(ok, x, 0), minlength=len(ud))
    k = np.bincount(inv, weights=ok.astype(float), minlength=len(ud))
    bs = [a[j].sum() / k[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(n))]
    return a.sum() / k.sum(), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def verdict(df, col, rng, floor):
    m, lo, hi = dboot(df, col, rng)
    halves = [df.loc[df["day"] < SPLIT, col].mean(), df.loc[df["day"] >= SPLIT, col].mean()]
    o = df.sort_values("day")
    sl = [o.iloc[ix][col].mean() for ix in np.array_split(np.arange(len(o)), 6)]
    ok = m > floor and lo > 0 and all(h > 0 for h in halves) and sum(v > 0 for v in sl) >= 5
    return ok, m, lo, hi, halves, sl


def show(name, r):
    ok, m, lo, hi, hv, sl = r
    print(f"    {name:10} mean {m:+.4f} S [{lo:+.4f}, {hi:+.4f}]  halves {hv[0]:+.4f} / {hv[1]:+.4f}  "
          f"slices " + " ".join(f"{v:+.3f}" for v in sl) + f"   -> {'PASS' if ok else 'FAIL'}")


def main():
    rng = np.random.default_rng(SEED)
    df = events()
    dep = df[(df["day"] >= DEP_START) & (df["day"] <= END)].copy()
    pre = df[df["day"] < DEP_START]
    dep["WAITD_c15"] = dep["STALL_c15"] - dep["IMM_c15"]
    dep["IMM_r15"] = -dep["IMM_c15"]
    dep["STALL_r15"] = -dep["STALL_c15"]
    print(f"  bursts: deployed {len(dep)} ({len(dep) / dep.groupby(['tk', 'day']).ngroups:.2f} per ticker-day), "
          f"pre-sample {len(pre)}; median size {dep['size'].median():.3f} S; STALL found in "
          f"{dep['STALL_c15'].notna().mean():.0%} (median wait {dep['STALL_wait'].median():.0f} min)")

    # ---- machinery
    pl = dep.copy()
    pl["WAITD_c15"] = pl["WAITD_c15"] + 0.05
    m1 = verdict(pl[pl["WAITD_c15"].notna()], "WAITD_c15", rng, 0.0)[0]
    rc = events(random_ctrl=True)
    rc = rc[(rc["day"] >= DEP_START) & (rc["day"] <= END)]
    m2 = all(abs(rc[f"{e}_c{h}"].mean()) < 0.01 for e in ENTRIES for h in HS)
    print(f"  M1 planted +0.05 S on the WAIT difference passes: {m1};  M2 random events |mean| < 0.01 S "
          f"everywhere ({len(rc)} events): {m2}  -> {'PASS' if m1 and m2 else 'FAIL'}")
    if not (m1 and m2):
        sys.exit("machinery checks failed -- not scoring")

    print("\n  PRIMARY (h = 15 min, straddle units)")
    show("H_CONT", verdict(dep, "IMM_c15", rng, FLOOR))
    show("H_WAIT", verdict(dep[dep["WAITD_c15"].notna()], "WAITD_c15", rng, 0.0))
    show("H_REV", verdict(dep[dep["STALL_r15"].notna()], "STALL_r15", rng, FLOOR))

    print("\n  REPORTED -- mean CONTINUATION (S) by entry and horizon; REVERSION = minus these")
    print(f"    {'entry':6} {'n':>6} " + " ".join(f"{'h=' + str(h):>16}" for h in HS) +
          f"   {'TAIL with':>9} {'TAIL against':>12}   pre-sample c15")
    for e in ENTRIES:
        sub = dep[dep[f"{e}_c15"].notna()]
        cells = []
        for h in HS:
            m, lo, hi = dboot(sub, f"{e}_c{h}", rng, n=500)
            cells.append(f"{m:+.4f}[{lo:+.3f},{hi:+.3f}]")
        print(f"    {e:6} {len(sub):6} " + " ".join(f"{x:>16}" for x in cells) +
              f"   {sub[f'{e}_tc'].mean():9.1%} {sub[f'{e}_tr'].mean():12.1%}   {pre[f'{e}_c15'].mean():+.4f}")
    print("\n  REPORTED -- by burst direction (IMM / STALL continuation at h = 15)")
    for s, lab in ((1, "UP bursts"), (-1, "DOWN bursts")):
        x = dep[dep["s"] == s]
        print(f"    {lab:11} n {len(x):5}  IMM {x['IMM_c15'].mean():+.4f}  STALL {x['STALL_c15'].mean():+.4f}")


if __name__ == "__main__":
    main()
