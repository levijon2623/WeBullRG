"""
check_shootout_iv.py -- IS IT IV / VEGA? (user, 2026-11-23, after check_shootout.py: the opening minutes pay the right
side of an ATM 0DTE ~1.5x as often within 5 min as midday.) Descriptive.

Two candidate engines for that: GAMMA (the opening simply moves more per minute than the options price in) or VEGA
(implied vol rising in the first minutes). The usual expectation for 0DTE is the opposite of the vega story -- IV is
highest at the bell and bleeds -- which would work AGAINST a buyer.

DATA     lake option-contracts-1m (minute bars: bid / ask close, iv_close, delta_close), same-day expiry, SPY / QQQ / IWM,
         every lake session. iv_close is a DAILY volatility with time in calendar days (verified 2026-11-23: SPY
         2025-03-12 09:30 ATM straddle $6.37 on 562 = ~1.4% to the close = 0.027 x sqrt(6.5/24)). Spot = the 5-s
         underlying price at each minute's close.
AT EACH ENTRY MINUTE m (09:30, 09:31, 09:32, 09:33, 09:35, 09:40, 10:30, 12:00, 14:00), K = nearest $1 strike:
  IV       the ATM IV (mean of the K call and put) at m, and its change by m+5 and m+15, in % of itself.
  PRICED   realized |S(m+h) - S(m)| / the move the IV priced in, S x iv x sqrt(h / 1440 days) x sqrt(2/pi) (the mean
           absolute move of a normal). Above 1 = the tape moved more than the option paid for.
  SPLIT    each leg's mid change m -> m+h, rebuilt with Black-Scholes in the same convention:
             move + time  = BS(S1, iv0, T1) - BS(S0, iv0, T0)     (delta, gamma and theta)
             vega         = BS(S1, iv1, T1) - BS(S1, iv0, T1)     (the IV change alone)
           in % of the entry mid, all legs and the legs that gained >= 20% (the shootout's winners).

    python check_shootout_iv.py

RUN LOG
  2026-11-23, 552 sessions x 3 tickers. Model check: BS at the entry IV reprices the mid within 7.8% (median).
    ATM IV (annualised): 40% at 09:30 -> 34% at 10:30 -> 33% at 12:00 -> 36% at 14:00.
    IV change over 15 min: 09:30 -4.9% (falls in 78% of sessions), 09:40 -3.7%, 10:30 -2.0%, 12:00 -0.3%, 14:00 +0.7%.
    Moved / priced over 5 min: 09:30 1.20, 09:33 1.00, 10:30 0.81, 12:00 0.59, 14:00 0.60 -- the opening tape moves MORE
      than even its (higher) IV pays for; midday moves ~60% of it.
    Gain split, 15 min, MEDIANS (rows with an IV jump > 50% or an entry misprice > 25% dropped; the first table's means
      were dragged by bad IV minutes and did not add up):
      all legs  09:30 total -7.1% = move+time -1.7% + vega -4.2% (rest -1.3%); 12:00 -3.7% = -3.3% + -0.4%;
                14:00 -7.7% = -6.7% + +0.6%.
      winners (>= +20%) 09:30 +45.4% = move +48.5% + vega -2.8%; 12:00 +34.9% = +35.1% + -0.1%; 14:00 +42.7% = +41.1% + +1.6%.
    READ: not vega. IV FALLS after the bell -- the biggest drag on a typical opening leg (-4%) -- and the winners win
    despite it, on the move: the opening simply moves faster than its options price in (gamma), midday slower.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

MINS = {"09:30": 570, "09:31": 571, "09:32": 572, "09:33": 573, "09:35": 575, "09:40": 580, "10:30": 630, "12:00": 720,
        "14:00": 840}
HS = (5, 15)


def bs(S, K, T, v, call):
    from scipy.stats import norm
    if T <= 0 or v <= 0:
        return max(0.0, (S - K) if call else (K - S))
    sd = v * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return S * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - S * norm.cdf(-d1)


def day(d):
    import polars as pl
    import check_sixpack_lake as L
    try:
        path = L.price_path(d)
        x = (pl.scan_parquet(os.path.join("lake", "silver", "option-contracts-1m", f"date={d}", "bars.parquet"))
             .filter(pl.col("underlying_symbol").is_in(["SPY", "QQQ", "IWM"]) & (pl.col("expiry").cast(pl.Utf8) == d))
             .select("underlying_symbol", "option_type", pl.col("strike").cast(pl.Float64), "minute_et", "bid_close", "ask_close",
                     "iv_close").collect())
    except Exception as e:                               # noqa: BLE001
        return d, [], type(e).__name__
    x = x.with_columns(m=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32)))
    book = {}
    for tk, cp, k, m, b, a, iv in x.select("underlying_symbol", "option_type", "strike", "m", "bid_close", "ask_close",
                                           "iv_close").iter_rows():
        if b and a and a > b and iv and abs(k - round(k)) < 1e-6:
            book[(tk, cp[0].upper(), int(round(k)), m)] = ((b + a) / 2, iv)
    out = []
    for tk, (sc, hi, lo, cl) in path.items():
        if tk not in ("SPY", "QQQ", "IWM"):
            continue

        def S(m):
            j = int(np.searchsorted(sc, (m + 1) * 60, side="right")) - 1
            return float(cl[j]) if j >= 0 else None
        for lab, m in MINS.items():
            s0 = S(m)
            if not s0:
                continue
            K = int(round(s0))
            c0, p0 = book.get((tk, "C", K, m)), book.get((tk, "P", K, m))
            if not c0 or not p0:
                continue
            iv0 = (c0[1] + p0[1]) / 2
            rec = {"d": d, "tk": tk, "t": lab, "iv0": iv0}
            for h in HS:
                s1 = S(m + h)
                c1, p1 = book.get((tk, "C", K, m + h)), book.get((tk, "P", K, m + h))
                if not s1 or not c1 or not p1:
                    continue
                iv1 = (c1[1] + p1[1]) / 2
                rec[f"div{h}"] = (iv1 - iv0) / iv0
                rec[f"priced{h}"] = abs(s1 - s0) / (s0 * iv0 * np.sqrt(h / 1440.0) * np.sqrt(2 / np.pi))
                T0, T1 = (960 - m) / 1440.0, (960 - m - h) / 1440.0
                for cp, (o0, o1) in (("C", (c0, c1)), ("P", (p0, p1))):
                    call = cp == "C"
                    a0 = bs(s0, K, T0, o0[1], call)
                    mv = bs(s1, K, T1, o0[1], call) - a0
                    vg = bs(s1, K, T1, o1[1], call) - bs(s1, K, T1, o0[1], call)
                    rec[f"{cp}{h}"] = [(o1[0] - o0[0]) / o0[0], mv / o0[0], vg / o0[0], (a0 - o0[0]) / o0[0]]
            out.append(rec)
    return d, out, None


def main():
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")))
    rows, errs = [], []
    with ProcessPoolExecutor(6) as ex:
        for d, r, e in ex.map(day, days):
            rows += r
            if e:
                errs.append((d, e))
    os.makedirs("_shootout_cache", exist_ok=True)
    json.dump(rows, open(os.path.join("_shootout_cache", "iv_rows.json"), "w"))
    print(f"{len(rows)} entries, {len({r['d'] for r in rows})} sessions, errors {len(errs)}")
    fit = [abs(r[f"{cp}5"][3]) for r in rows for cp in "CP" if f"{cp}5" in r]
    print(f"model check: |BS(entry IV) - entry mid| / mid, median {np.median(fit):.1%} (small = the convention is right)\n")
    print(f"{'entry':6s} {'n':>5s} | {'ATM IV':>7s} {'IV chg 5m':>9s} {'15m':>6s} {'IV fell 15m':>11s} | {'moved/priced 5m':>15s} {'15m':>5s} | "
          f"{'all legs 15m: total':>19s} {'move':>6s} {'vega':>6s} | {'winners (>=+20% @15m): total':>28s} {'move':>6s} {'vega':>6s} {'n':>5s}")
    for lab in MINS:
        R = [r for r in rows if r["t"] == lab and "div15" in r]
        if not R:
            continue
        legs = [r[f"{cp}15"] for r in R for cp in "CP" if f"{cp}15" in r]
        win = [l for l in legs if l[0] >= 0.20]
        med = lambda v: np.median(v) if v else np.nan
        print(f"{lab:6s} {len(R):5d} | {med([r['iv0'] * np.sqrt(365) for r in R]):7.0%} {med([r['div5'] for r in R if 'div5' in r]):+9.1%} "
              f"{med([r['div15'] for r in R]):+6.1%} {np.mean([r['div15'] < 0 for r in R]):11.0%} | "
              f"{np.mean([r['priced5'] for r in R if 'priced5' in r]):15.2f} {np.mean([r['priced15'] for r in R]):5.2f} | "
              f"{np.mean([l[0] for l in legs]):+19.1%} {np.mean([l[1] for l in legs]):+6.1%} {np.mean([l[2] for l in legs]):+6.1%} | "
              f"{np.mean([l[0] for l in win]) if win else np.nan:+28.1%} {np.mean([l[1] for l in win]) if win else np.nan:+6.1%} "
              f"{np.mean([l[2] for l in win]) if win else np.nan:+6.1%} {len(win):5d}")


if __name__ == "__main__":
    main()
