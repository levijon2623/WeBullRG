# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_ma5_delta.py
==================
A SCALPING TRIGGER and its DELTA CONTEXT (user, 2026-10-07): the 1m close
crossing its 5-period SMA -- above = CALL, below = PUT -- and what the net
delta of the cross candle and of the two candles before it says about the
cross. Agreement AND divergence: "price crosses above the 5MA, but the delta
of the previous two candles was heavily negative while price refused to drop"
(absorption).

DELTA SOURCE. The user means live VOLUME delta (aggressor buys - sells of the
underlying); there is no history of it for SPY/QQQ/IWM (1m OHLCV only; Webull
equity ticks cover ~50-60% and only going forward), and a bar-based proxy is
circular for a divergence test. So this uses DELTA-WEIGHTED OPTION FLOW,
historical/HEDGE{T}.parquet `hedge_sh_sl` (build_hedge_flow.py): single-leg
aggressor-signed option volume x delta x 100 = the shares dealers must trade;
+ = dealers must BUY. Measured independently of the price bars. Caveat: the
lake's 0DTE greeks are corrupt ~2024-10..2025-12 -- no step is visible in SPY's
hedge/gross ratio across those dates (checked 2026-10-07); ranks are trailing
per ticker, which absorbs level shifts.

TRIGGER   SPY, QQQ, IWM. SMA5 of 1m closes within the session. Cross UP at
          minute m: close(m-1) <= SMA5(m-1) and close(m) > SMA5(m); DOWN
          mirrored. m in 09:35..15:00. Entry = close(m) (known at the bar
          close). s = +1 CALL / -1 PUT. Every cross counts.
DELTA     Dx = hedge(m) (the cross candle); D2 = hedge(m-2) + hedge(m-1).
          Ranked toward the trade: rank of s x Dx among ALL 1-minute values,
          and of s x D2 among ALL 2-minute sums, of that ticker's prior 60
          sessions (09:35-15:00; >= 20 sessions else excluded). Rank 0.9 =
          strongly WITH the cross, 0.1 = strongly AGAINST it.
OUTCOME   r_h = s x (close(m+h) - close(m)) / S, h = 3, 5, 15, 30 minutes, S =
          the day's 09:35 ATM straddle (_implied_move_cache). TAIL = 1 if
          the move toward the trade within m+1..m+30 (high / low) >= 0.25 S.

CELLS (pre-registered; overlapping; each scored AGAINST ALL CROSSES)
    AGREE2      rank(s x D2) >= 0.8   prior two candles strongly with the cross
    ABSORB      rank(s x D2) <= 0.2 AND price HELD over those two candles:
                s x (close(m-1) - close(m-3)) >= 0   -- the user's case
    PUSHED      rank(s x D2) <= 0.2 AND price moved WITH that delta (the
                non-absorbed counterpart; completes the 2x2)
    XAGREE      rank(s x Dx) >= 0.8   the cross candle's own delta agrees
    XDIV        rank(s x Dx) <= 0.2   the cross candle's delta disagrees
    ABSORB_FLIP ABSORB AND XAGREE     absorbed, then the delta turns with it

WINDOWS  deployed slices 2024-08-20..2026-08-21 scored (halves split at
         2025-08-21); the spent pre-sample 2023-10..2024-08 reported, labelled.

CRITERIA (fixed before running)
    BASE  ALL crosses: mean r_h with a day-block 95% CI (is the plain cross
          anything at all?), reported per horizon.
    CELL  for each cell x horizon: d = mean r_h(cell) - mean r_h(ALL), p by a
          day-block bootstrap (2,000; calendar days, tickers pooled). A cell
          PASSES at a horizon if Benjamini-Hochberg q < 0.10 over all 24
          cell x horizon tests AND d has the pooled sign in >= 5 of 6 slices
          AND in both halves. A passing cell is a CANDIDATE for an option
          simulation (stage B), not a strategy.
    SCALE reported against what a scalp must clear: an ATM 0DTE option moves
          ~0.5 S per 1 S of underlying, and a round trip costs ~2-4% of the
          option (~0.01-0.02 S) -- so a cell needs mean r of ~0.03 S or more
          before theta to matter.
    REPORTED  TAIL by cell; n and crosses per ticker-day.

MACHINERY  M1 a PLANTED cell (crosses whose r_5 > 0) must pass at h=5 with
           d >> 0, and a RANDOM cell (a coin flip per cross) must not pass.
           M2 every rank uses strictly earlier sessions (asserted).

RUN LOG
  2026-10-07 run 1: 178,618 crosses (deployed 131,020 = 87 per ticker-day;
    pre-sample 47,598). M1 PASS (planted d +0.100 S at h=5; random nowhere).
    BASE: the plain cross is a coin flip leaning AGAINST itself -- mean r
    -0.0008 / -0.0014 / -0.0013 / -0.0012 S at 3 / 5 / 15 / 30m (CIs just
    below 0); pre-sample -0.0015..-0.0001. TAIL 35.3%.
    CELLS: one passes -- XDIV (cross candle's delta strongly AGAINST the
    cross; 6,373, 5% of crosses) at h=3: d -0.0051 S [-0.0082, -0.0019], 6/6
    slices, both halves; -0.0047 at 5m (q misses the 5/6 rule only at 15/30).
    A NEGATIVE filter and economically nil: ~0.005 S is ~$0.013 of SPY,
    against the ~0.03 S a scalp needs. ABSORB (the user's case; 3,620): d
    -0.002 / -0.002 / +0.005 / +0.001 S, every CI spanning 0 -- no sign of
    absorption in delta-weighted option flow; ABSORB_FLIP (1,268) likewise.
    AGREE2, PUSHED, XAGREE: |d| <= 0.003 S, nothing. Every cell's TAIL runs
    38-41% vs 35% base -- crosses on active flow move more, either way (the
    magnitude effect again), not further toward the trade.
    VERDICT: no tradeable cell. The live VOLUME-delta version stays untested
    (no history) -- forward logging is the only route to it.

Usage:  python check_ma5_delta.py
"""
from __future__ import annotations

import bisect
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
P_START, DEP_START, SPLIT, END = (dt.date(2023, 10, 12), dt.date(2024, 8, 20),
                                  dt.date(2025, 8, 21), dt.date(2026, 8, 21))
FIRST, LAST = 575, 900
HORIZONS = (3, 5, 15, 30)
TAIL_S, TAIL_H = 0.25, 30
RANK_N, RANK_MIN = 60, 20
HI, LO = 0.8, 0.2
CELLS = ("AGREE2", "ABSORB", "PUSHED", "XAGREE", "XDIV", "ABSORB_FLIP")
N_BOOT = 2000
SEED = 20261007


def slice_of(day):
    return BF.slice_of(day)


def events():
    am = BF.am_straddles()
    out = []
    for tk in TICKERS:
        b = BF.bars_full(tk).to_pandas()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        h = pd.read_parquet(f"historical/HEDGE{tk}.parquet", columns=["date", "mod", "hedge_sh_sl"])
        h["date"] = pd.to_datetime(h["date"]).dt.date
        hd = {d: g.set_index("mod")["hedge_sh_sl"] for d, g in h.groupby("date")}
        bd = {d: g.set_index("mod") for d, g in b.groupby("date")}
        days = sorted(set(bd) & set(hd))
        mods = np.arange(570, 960)
        pool1, pool2 = [], []              # per day: arrays of 1m values / 2m sums (FIRST..LAST)
        for i, day in enumerate(days):
            g = bd[day]
            c = g["close"].reindex(mods).to_numpy(float)
            hi_, lo_ = g["high"].reindex(mods).to_numpy(float), g["low"].reindex(mods).to_numpy(float)
            H = hd[day].reindex(mods).to_numpy(float)
            H2 = np.full_like(H, np.nan)
            H2[2:] = H[:-2] + H[1:-1]          # H2[j] = H[j-2] + H[j-1]
            win = (mods >= FIRST) & (mods <= LAST)
            today1, today2 = H[win & np.isfinite(H)], H2[win & np.isfinite(H2)]
            prev = list(range(max(0, i - RANK_N), i))
            st = am.get((tk, day))
            if len(prev) >= RANK_MIN and st and st[1] > 0 and P_START <= day <= END:
                assert all(days[j] < day for j in prev)
                ref1 = np.sort(np.concatenate([pool1[j] for j in prev]))
                ref2 = np.sort(np.concatenate([pool2[j] for j in prev]))
                S = st[1]
                sma = pd.Series(c).rolling(5).mean().to_numpy()
                for j in range(len(mods)):
                    m = mods[j]
                    if m < FIRST or m > LAST or j < 5:
                        continue
                    if not (np.isfinite(c[j]) and np.isfinite(c[j - 1]) and np.isfinite(sma[j]) and np.isfinite(sma[j - 1])):
                        continue
                    up = c[j - 1] <= sma[j - 1] and c[j] > sma[j]
                    dn = c[j - 1] >= sma[j - 1] and c[j] < sma[j]
                    if not (up or dn):
                        continue
                    s = 1 if up else -1
                    if not (np.isfinite(H[j]) and np.isfinite(H2[j]) and np.isfinite(c[j - 3])):
                        continue
                    r = {}
                    for hz in HORIZONS:
                        r[hz] = s * (c[j + hz] - c[j]) / S if j + hz < len(c) and np.isfinite(c[j + hz]) else np.nan
                    w = slice(j + 1, j + 1 + TAIL_H)
                    fav = np.nanmax(hi_[w]) - c[j] if s > 0 else c[j] - np.nanmin(lo_[w])
                    rk = lambda ref, v: (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / 2 / len(ref)
                    out.append(dict(tk=tk, day=day, m=int(m), s=s,
                                    rx=rk(ref1, s * H[j]), r2=rk(ref2, s * H2[j]),
                                    held=s * (c[j - 1] - c[j - 3]) >= 0,
                                    tail=float(fav / S >= TAIL_S), **{f"r{hz}": r[hz] for hz in HORIZONS}))
            pool1.append(today1)
            pool2.append(today2)
        print(f"  {tk}: {sum(1 for e in out if e['tk'] == tk)} crosses", flush=True)
    return pd.DataFrame(out)


def cells(df):
    return {
        "AGREE2": df["r2"] >= HI,
        "ABSORB": (df["r2"] <= LO) & df["held"],
        "PUSHED": (df["r2"] <= LO) & ~df["held"],
        "XAGREE": df["rx"] >= HI,
        "XDIV": df["rx"] <= LO,
        "ABSORB_FLIP": (df["r2"] <= LO) & df["held"] & (df["rx"] >= HI),
    }


def boot_diff(df, mask, col, rng):
    """d = mean(col | mask) - mean(col), day-block bootstrap -> (d, lo, hi, p)."""
    x = df[col].to_numpy(float)
    ok = np.isfinite(x)
    days = df["day"].to_numpy()
    ud, inv = np.unique(days, return_inverse=True)
    m = mask.to_numpy() & ok
    a = np.bincount(inv, weights=np.where(m, x, 0), minlength=len(ud))
    na = np.bincount(inv, weights=m.astype(float), minlength=len(ud))
    t = np.bincount(inv, weights=np.where(ok, x, 0), minlength=len(ud))
    nt = np.bincount(inv, weights=ok.astype(float), minlength=len(ud))
    d = a.sum() / na.sum() - t.sum() / nt.sum()
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, len(ud), len(ud))
        bs.append(a[j].sum() / max(na[j].sum(), 1) - t[j].sum() / nt[j].sum())
    bs = np.array(bs)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return d, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), min(1.0, p)


def boot_mean(df, col, rng):
    x = df[col].to_numpy(float)
    ok = np.isfinite(x)
    ud, inv = np.unique(df["day"].to_numpy(), return_inverse=True)
    a = np.bincount(inv, weights=np.where(ok, x, 0), minlength=len(ud))
    n = np.bincount(inv, weights=ok.astype(float), minlength=len(ud))
    bs = [a[j].sum() / n[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))]
    return a.sum() / n.sum(), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def bh(p, q=0.10):
    p = np.asarray(p)
    o = np.argsort(p)
    k = max((i for i in range(len(p)) if p[o[i]] <= q * (i + 1) / len(p)), default=-1)
    rej = np.zeros(len(p), bool)
    rej[o[:k + 1]] = True
    return rej


def score(df, C, rng, label):
    """-> list of (cell, h, d, lo, hi, p, slices_same, halves_same)."""
    res = []
    for name, mask in C.items():
        for hz in HORIZONS:
            col = f"r{hz}"
            d, lo, hi, p = boot_diff(df, mask, col, rng)
            per = []
            for k in range(6):
                sub = df["slice"] == k
                if (mask & sub).sum() > 10:
                    per.append(df.loc[mask & sub, col].mean() - df.loc[sub, col].mean())
            halves = [df.loc[mask & h_, col].mean() - df.loc[h_, col].mean()
                      for h_ in (df["day"] < SPLIT, df["day"] >= SPLIT)]
            res.append((name, hz, d, lo, hi, p, sum(np.sign(v) == np.sign(d) for v in per),
                        all(np.sign(v) == np.sign(d) for v in halves), int(mask.sum())))
    rej = bh([r[5] for r in res])
    return [r + (bool(rj and r[6] >= 5 and r[7]),) for r, rj in zip(res, rej)]


def main():
    df = events()
    df["slice"] = [slice_of(d) if d >= DEP_START else -1 for d in df["day"]]
    dep = df[(df["day"] >= DEP_START) & (df["day"] <= END)].reset_index(drop=True)
    pre = df[df["day"] < DEP_START].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    nd = dep.groupby(["tk", "day"]).size()
    print(f"  crosses: deployed {len(dep)} ({nd.mean():.1f} per ticker-day), pre-sample {len(pre)}")

    # ---- M1
    Cm = {"PLANTED": dep["r5"] > 0, "RANDOM": pd.Series(rng.random(len(dep)) < 0.3)}
    mres = score(dep, {**cells(dep), **Cm}, rng, "m1")
    pl = [r for r in mres if r[0] == "PLANTED" and r[1] == 5][0]
    rnd = [r for r in mres if r[0] == "RANDOM"]
    m1 = pl[-1] and pl[2] > 0.05 and not any(r[-1] for r in rnd)
    print(f"  M1 planted passes at h=5 (d {pl[2]:+.3f} S) and random passes nowhere: {'PASS' if m1 else 'FAIL'}")
    if not m1:
        sys.exit("machinery check failed -- not scoring")

    print("\n  BASE -- every cross, mean r_h in straddle units (day-block 95% CI)")
    for hz in HORIZONS:
        m, lo, hi = boot_mean(dep, f"r{hz}", rng)
        mp = pre[f"r{hz}"].mean()
        print(f"    h={hz:2}m  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   pre-sample (labelled) {mp:+.4f}")
    print(f"    TAIL (>= {TAIL_S} S toward the trade within {TAIL_H}m): {dep['tail'].mean():.1%}")

    C = cells(dep)
    res = score(dep, C, rng, "cells")
    print("\n  CELLS -- d = mean r_h(cell) - mean r_h(all), straddle units")
    print(f"    {'cell':12} {'n':>6} {'h':>3} {'d':>8} {'95% CI':>19} {'p':>6} {'slices':>6} {'halves':>6}  "
          f"{'cell mean':>9} {'TAIL':>6} {'pre d':>7}")
    Cp = cells(pre)
    for r in res:
        name, hz, d, lo, hi, p, ss, hv, n, ok = r
        cm = dep.loc[C[name], f"r{hz}"].mean()
        tl = dep.loc[C[name], "tail"].mean()
        pdd = pre.loc[Cp[name], f"r{hz}"].mean() - pre[f"r{hz}"].mean()
        print(f"    {name:12} {n:6} {hz:3} {d:+8.4f} [{lo:+.4f},{hi:+.4f}] {p:6.3f} {ss:4}/6 {'yes' if hv else 'no':>6}  "
              f"{cm:+9.4f} {tl:6.1%} {pdd:+7.4f}{'  PASS' if ok else ''}")
    passed = sorted({r[0] for r in res if r[-1]})
    print(f"\n  cells passing at any horizon: {passed or 'none'}")


if __name__ == "__main__":
    main()
