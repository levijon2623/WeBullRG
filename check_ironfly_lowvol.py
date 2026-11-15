"""
check_ironfly_lowvol.py -- SELL AN IRON BUTTERFLY WHEN THE VOL GAUGE IS LOW (user, 2026-11-14; Level 3 = defined-risk
spreads). STAGE 1 of 2, EXPLORATORY.

WHY. check_straddle_legout (10-09) reported, without testing it, that an ATM straddle BOUGHT on a LOW gauge reading
loses ~2.2% of its premium in 60 minutes even at mid prices (4,945 readings); at the user's measured fills the buyer
loses -2.6% [-3.8, -1.3] (re-score 2026-11-14). The seller's side of that is roughly +2%/hour -- three times the
size of anything the leg-out found -- but a naked short straddle is not allowed at Level 3 and carries the tail, so
the question is whether it survives as an IRON BUTTERFLY: sell the ATM call + put, buy a call W above and a put W below.

STAGE 1 (this run) sizes it on the same gauge entries the leg-out used (the gauge's OUT-OF-SAMPLE walk-forward, cap 5,
10-minute grid 09:40-14:50, one entry per ticker per 60 min per label, slices 1-6 = 2024-08-20..2026-08-21). LOW
readings were looked at on 10-09, so this is after the fact and decides only whether a forward test is worth it.
  LABELS   LOW p <= 0.35 (the trade), MID 0.35-0.65 and HIGH >= 0.65 (does the gauge matter?)
  FLY      expiry: same day if quoted, else next (check_ic_vol_trade.legs_for); ATM = strike nearest spot;
           wings W = $1, $2, $3, $5 (all four legs must be quoted)
  HOLD     60 minutes (the gauge's horizon), 15:55 at the latest; reported: held to 15:55
  FILLS    user  = each leg at mid, selling -$0.0008 / buying +$0.0006 per share (their 448 real 0DTE fills), fees
                   $0.045 per contract per order = $0.36 per fly round trip
           cross = sell at the bid, buy at the ask (no fees)  -- a bracket
           mid   = mid, no fees
  METRICS  $ per fly (100 shares), return on max risk (W - credit), win rate, worst 1%, by half (split 2025-08-21)
           and by ticker; day-block bootstrap CIs (2,000).
STAGE 2 RULE (written now, applied to these numbers): the primary width is the one with the highest mean return on
max risk among widths whose LOW mean $ is > 0 in BOTH halves at user fills; if no width qualifies, the idea stops
here. A qualifying width goes to a pre-registered forward test on the live gauge log (live/vol_gauge_log.jsonl,
from 2026-10-06) priced from the refreshed lake.

RUN LOG
  2026-11-14 run 1: entries LOW 4,956 / MID 4,224 / HIGH 1,739; 10,864 priced, all same-day expiry.
    60-MINUTE HOLD (the gauge's horizon), user fills, $ per fly [day-block 95% CI], mean $ / mean risk:
      LOW  W1 -0.36 [-0.66,-0.07]  W2 +0.42 [-0.31,+1.09] 0.5%  W3 +0.70 [-0.47,+1.79] 0.5%  W5 +0.91 [-1.02,+2.76] 0.3%
      MID  all <= +0.33;  HIGH all negative (-0.40..-1.20).
      STAGE-2 RULE -> W2 (both halves > 0: +0.62 / +0.21), but its CI spans 0 and it earns ~0.5% of risk per trade:
      a forward test could not resolve it (the leg-out's power problem again).
    HELD TO 15:55 (reported, not the rule's horizon), user fills:
      LOW  W2 +4.29 [+1.65,+7.15] 5.2% of risk   W3 +6.03 [+1.41,+10.38] 4.0%   W5 +10.39 [+2.98,+17.52] 3.3%
      MID  W2 +2.65 [+0.00,+5.24] 4.2%           W3 +5.02 [+0.44,+9.34] 4.0%    W5 +9.88 [+2.04,+17.72] 3.6%
      HIGH W2 +0.73 [-2.70,+4.13] 1.5%           W3 +1.79 [-3.74,+7.80] 1.7%    W5 +4.51 [-6.78,+14.73] 1.9%
      Both halves positive for LOW/MID, the second half about a third of the first (W3 LOW +9.29 / +2.74).
      Entries at 14:xx earn least (W3 +2.6..2.7 vs +6..9 at 11:00-13:59).
      At CROSS fills (sell bid / buy ask) it is gone (LOW W2 -4.84, W3 -3.06, W5 +1.86 [-5.6,+9.6]).
    READ: the money is in holding short 0DTE premium into the close (the variance premium), not in the gauge's
    60-minute horizon; the gauge mainly says when NOT to (HIGH earns a third of LOW/MID). It lives entirely on
    near-mid fills, measured so far on the user's single-leg orders, never on a 4-leg combo. After the fact; any
    EOD-hold version is a new hypothesis for forward data only.

Usage:  python check_ironfly_lowvol.py
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
import check_straddle_legout as LS                      # noqa: E402
import check_vol_gauge as CG                            # noqa: E402

SPLIT = dt.date(2025, 8, 21)
WINGS = (1.0, 2.0, 3.0, 5.0)
SELL_C, BUY_C, FEE_FLY = 0.0008, 0.0006, 0.0036     # $ per share; fee = 8 contract-orders x $0.045 / 100
HOLD_M, EOD = 60, 15 * 60 + 55
N_BOOT, SEED = 2000, 20261114


def entries():
    rows = sorted(CG.build_rows(None), key=lambda r: (r["day"], r["m"], r["tk"]))
    CG.rank_inputs(rows)
    X = CG.design(rows)
    y = np.array([r["big"] for r in rows], float)
    pred, _c, _k, _i = CG.walk(rows, X, y, cap=5)
    out, last = defaultdict(list), defaultdict(lambda: -999)
    for r, p in zip(rows, pred):
        if not np.isfinite(p):
            continue
        lab = "LOW" if p <= LS.LOW else ("HIGH" if p >= LS.HIGH else "MID")
        key = (r["tk"], r["day"], lab)
        if r["m"] - last[key] >= HOLD_M:
            last[key] = r["m"]
            out[lab].append((r["tk"], r["day"], r["m"], float(p)))
    return out


def fly(book, legs, W, m_in, m_out):
    """{fill: (pnl $/share, max risk $/share)} for a short iron fly, or None."""
    (exp, _c, K) = legs[0]
    keys = [(exp, "call", K), (exp, "put", K), (exp, "call", K + W), (exp, "put", K - W)]
    qi = [VT.quote(book, k, m_in, 3) for k in keys]
    qo = [VT.quote(book, k, m_out, 10) for k in keys]
    if any(q is None for q in qi + qo):
        return None
    out = {}
    for fill in ("user", "cross", "mid"):
        if fill == "user":
            sell = lambda q: (q[0] + q[1]) / 2 - SELL_C
            buy = lambda q: (q[0] + q[1]) / 2 + BUY_C
            fee = FEE_FLY
        elif fill == "cross":
            sell, buy, fee = (lambda q: q[0]), (lambda q: q[1]), 0.0
        else:
            sell = buy = (lambda q: (q[0] + q[1]) / 2)
            fee = 0.0
        credit = sell(qi[0]) + sell(qi[1]) - buy(qi[2]) - buy(qi[3])
        debit = buy(qo[0]) + buy(qo[1]) - max(sell(qo[2]), 0.0) - max(sell(qo[3]), 0.0)
        risk = W - credit
        if credit <= 0 or risk <= 0:
            return None
        out[fill] = (credit - debit - fee, risk, credit)
    return out


def run(ent):
    spots = {}
    for tk in CG.TICKERS:
        b = BF.bars_full(tk).to_pandas()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        for d, g in b.groupby("date"):
            spots[(tk, d)] = dict(zip(g["mod"].astype(int), g["close"].astype(float)))
    paths = {os.path.basename(os.path.dirname(p)).split("=")[1]: p for p in glob.glob(f"{VT.SILVER}/date=*/bars.parquet")}
    by_day = defaultdict(list)
    for lab, es in ent.items():
        for e in es:
            by_day[e[1]].append((lab,) + e)
    rows = []
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
            for W in WINGS:
                for hold, m_out in (("60m", min(m + HOLD_M, EOD)), ("eod", EOD)):
                    r = fly(bk, lg["straddle"], W, m, m_out)
                    if r is None:
                        continue
                    for fill, (pnl, risk, credit) in r.items():
                        rows.append(dict(lab=lab, tk=tk, day=day, m=m, p=gp, W=W, hold=hold, fill=fill,
                                         pnl=pnl * 100, risk=risk * 100, credit=credit * 100, roe=pnl / risk,
                                         dte0=lg["straddle"][0][0] == day))
        if n % 50 == 0:
            print(f"    {day}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


def dboot(d, col, rng):
    x = d[col].to_numpy(float)
    ud, inv = np.unique(d["day"].to_numpy(), return_inverse=True)
    a = np.bincount(inv, weights=x, minlength=len(ud))
    k = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = np.array([a[j].sum() / k[j].sum() for j in (rng.integers(0, len(ud), len(ud)) for _ in range(N_BOOT))])
    return a.sum() / k.sum(), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    rng = np.random.default_rng(SEED)
    ent = entries()
    print("  gauge entries (OOS walk, one per ticker per 60 min per label): "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(ent.items())))
    df = run(ent)
    print(f"  priced: {df[['lab', 'tk', 'day', 'm']].drop_duplicates().shape[0]} entries; same-day expiry "
          f"{df['dte0'].mean():.0%}")
    for hold in ("60m", "eod"):
        print(f"\n  ===== HOLD {hold} =====")
        for fill in ("user", "cross", "mid"):
            print(f"\n  -- fills: {fill} --")
            print(f"    {'label':5} {'W':>3} {'n':>5}  {'$/fly':>7} {'95% CI':>17}  {'ROE risk':>8} {'95% CI':>17}  "
                  f"{'win':>4} {'credit':>7} {'worst1%':>8}  {'halves $':>15}  by ticker $")
            for lab in ("LOW", "MID", "HIGH"):
                for W in WINGS:
                    d = df[(df.lab == lab) & (df.W == W) & (df.hold == hold) & (df.fill == fill)]
                    if len(d) < 30:
                        continue
                    m, lo, hi = dboot(d, "pnl", rng)
                    r, rlo, rhi = dboot(d, "roe", rng)
                    hv = [d.loc[d.day < SPLIT, "pnl"].mean(), d.loc[d.day >= SPLIT, "pnl"].mean()]
                    print(f"    {lab:5} {W:3.0f} {len(d):5d}  {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  {r:+8.2%} [{rlo:+7.2%},{rhi:+7.2%}]"
                          f"  {(d.pnl > 0).mean():4.0%} {d.credit.mean():7.1f} {d.pnl.quantile(0.01):+8.1f}"
                          f"  {hv[0]:+6.2f}/{hv[1]:+6.2f}  "
                          + " ".join(f"{t} {g.pnl.mean():+.2f}" for t, g in d.groupby("tk")))
    # stage-2 rule
    d = df[(df.lab == "LOW") & (df.hold == "60m") & (df.fill == "user")]
    ok = []
    for W in WINGS:
        x = d[d.W == W]
        if len(x) < 30:
            continue
        h = [x.loc[x.day < SPLIT, "pnl"].mean(), x.loc[x.day >= SPLIT, "pnl"].mean()]
        if all(v > 0 for v in h):
            ok.append((x.roe.mean(), W))
    print("\n  STAGE-2 RULE (LOW, 60m, user fills, both halves > 0): "
          + (f"primary width ${max(ok)[1]:.0f} (ROE on risk {max(ok)[0]:+.2%})" if ok else "NO width qualifies -> stop"))
    df.to_pickle(os.path.join("_ic_cache", "ironfly_lowvol.pkl"))


if __name__ == "__main__":
    main()
