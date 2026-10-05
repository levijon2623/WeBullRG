# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.4"]
# ///
"""
check_ic_vol_trade.py
=====================
check_ic_tail found the 32 inputs forecast HOW FAR the underlying moves within
60 minutes (chained Spearman +0.114), mostly not which way -- and a long ATM
call / put with trail50 did not monetise it (top third -10.2% vs all -10.8%).
The natural instrument for a magnitude forecast is non-directional. So: buy a
STRADDLE, or a STRANGLE (+-1 strike), at the flow trigger when the score is
high. (User, 2026-10-03.)

THE SCORE -- walk-forward Lasso (check_ic_screen.lasso_walk), 32 inputs,
    trained on BIG = 1 if the underlying reaches +-0.5 x that day's 09:35
    straddle within 60 minutes, either way (MFE60 >= 0.5 or MAE60 <= -0.5).
    Training: P (spent pre-sample, labelled) + earlier slices; thirds cut on
    the training predictions. The tail-HIT model's score is reported too.

THE TRADES, at every deployed flow trigger (check_ic_screen.build_dataset;
    p50 crossings, SPY/QQQ/IWM, 09:35-14:59), standalone:
    STRADDLE  ATM call + ATM put (strike nearest spot), same expiry
    STRANGLE  the first listed strike ABOVE spot (call) + the first BELOW (put)
    Expiry: same day if listed, else the next listed (dte 0 then 1).
    Quotes: the silver lake's bid_close / ask_close; entry = the last quote
    per leg in m-3..m, exit = the last quote in [exit-10, exit].
    ENTRY at each leg's ASK. EXIT after 60 minutes at bid - c x spread per
    leg, c = 0.5 when the position is up at the bids, 1.5 when it is down
    (sim_core's live exit cushion, applied to the package).
    ROE = (exit proceeds - entry cost) / entry cost.

PRE-REGISTERED CRITERIA, for each structure on its own (deployed slices)
    V1  top-third mean ROE minus all-trigger mean ROE > 0, day-block 95% CI
        lower bound > 0
    V2  Spearman(score, ROE) > 0, day-block 95% CI lower bound > 0
    V3  top-third mean ROE > 0 -- PROFITABLE after realistic fills
    V4  V1's difference > 0 in >= 4 of 6 slices
    V5  V1's difference > p95 of 20 placebos (input rows permuted within
        ticker, the Lasso refit on each)
    PASS = V1..V5.

REPORTED, NOT SCORED
    exits at 30 minutes and 15:55, mid fills; ROE by third; the tail-HIT
    model's top third; SEQUENTIAL (one position per ticker, only top-third
    triggers taken, next allowed after exit): n, mean, total.

MACHINERY (before scoring)
    M1  this builder's 09:35 straddle mid (quotes in 09:33-09:35, nearest
        strike) agrees with build_implied_move's cached nearest-strike
        straddle: median |diff| <= 1% of the straddle
    M2  a planted input (the rank of the realised 60-minute move size) sends
        the top third's straddle ROE far above the bottom third's
    M3  the strangle costs less than the straddle on >= 95% of triggers

RUN LOG
  2026-10-03 run 1: blocked by M2 before scoring -- the planted input was
    binary (= BIG), its predictions two-valued, and both third-cuts landed on
    those values, so every trade fell in the middle third (nan). Replaced by
    a continuous plant (rank of the realised move size). No outcome seen.
  2026-10-03 run 2: M1 (09:35 straddle = cache, 0.00% median diff on 1,495
    ticker-days), M2 (planted top +14.2% vs bottom -21.3%), M3 (strangle
    cheaper 100%) PASS. The BIG model forecasts magnitude well: chained
    Spearman with BIG +0.161 (RVOL, URGENCY, TIME from slice 1; up to 11
    inputs by slice 6).
    STRADDLE FAIL (V1 V2 V3 V5): thirds -7.4 / -6.1 / -5.4%, all -6.3%;
      top - all +0.94pp, CI [-0.07, +2.00]. Mid fills: top -0.7%.
    STRANGLE FAIL (V3 V5): thirds -9.6 / -8.1 / -6.5%, all -8.2%; top - all
      +1.64pp, CI [+0.30, +3.09], Spearman +0.034 CI [+0.012, +0.062] -- it
      RANKS, but every third loses. Mid fills: top -1.1%.
    Sequential top-third only: straddle -5.9% (n 3,163), strangle -7.5%.
    WHY (post-hoc diagnostic, labelled): the forecast is largely PRICED IN.
      The straddle at the trigger costs 0.565 / 0.590 / 0.660 of the 09:35
      straddle by third (top third +17% dearer) while the realised 60-minute
      move is 0.377 / 0.413 / 0.490 S (+30%): move per unit paid rises only
      0.667 -> 0.742, short of theta + the ~5pp spread/cushion.
    CAVEAT: V5's placebo medians are positive (+0.74 / +0.52pp); placebo
      models shrink to near-constant predictions whose thirds tie, so that
      distribution is unreliable. The verdict does not rest on it -- V3
      (every third loses money) fails on its own.

Usage:
  python check_ic_vol_trade.py
"""
from __future__ import annotations

import bisect
import datetime as dt
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import check_ic_screen as IC                            # noqa: E402

SILVER = "lake/silver/option-contracts-1m"
CACHE = os.path.join("_ic_cache", "vol_trades.pkl")
BIG_S = 0.5
EXITS = {"60m": 60, "30m": 30, "eod": None}
EOD = 15 * 60 + 55
N_PLACEBO = 20
SEED = 20261010


# ================================================================== option books
def day_book(path, day):
    """{root: {(expiry, type, strike): (mods[], bid[], ask[])}} for 0..5 DTE,
    SPY / QQQ / IWM. One lake partition."""
    t = pl.col("minute_et").dt
    df = (pl.scan_parquet(path)
          .filter(pl.col("underlying_symbol").is_in(list(IC.TICKERS))
                  & (pl.col("expiry") >= day) & (pl.col("expiry") <= day + dt.timedelta(days=5)))
          .with_columns((t.hour().cast(pl.Int32) * 60 + t.minute().cast(pl.Int32)).alias("mod"))
          .select("underlying_symbol", "expiry", "option_type", "strike", "mod", "bid_close", "ask_close")
          .sort("mod")
          .collect())
    out = defaultdict(dict)
    for (root, e, typ, k), g in df.group_by(["underlying_symbol", "expiry", "option_type", "strike"]):
        out[root][(e, typ, float(k))] = (g["mod"].to_list(), g["bid_close"].to_list(), g["ask_close"].to_list())
    return out


def quote(book, key, mod, lookback):
    """(bid, ask) of the last bar in [mod - lookback, mod], or None."""
    x = book.get(key)
    if x is None:
        return None
    mods, b, a = x
    i = bisect.bisect_right(mods, mod) - 1
    if i < 0 or mods[i] < mod - lookback:
        return None
    bid, ask = b[i], a[i]
    if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid:
        return None
    return bid, ask


def legs_for(book, day, spot, mod):
    """-> {"straddle": [(key_call), (key_put)], "strangle": [...]} or None.
    Expiry: same day if quoted, else the next listed."""
    exps = sorted({k[0] for k in book})
    exp = next((e for e in exps if e == day), None) or next((e for e in exps if e > day), None)
    if exp is None:
        return None
    strikes = sorted({k[2] for k in book if k[0] == exp
                      and quote(book, k, mod, 3) is not None})
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - spot))
    above = [k for k in strikes if k > spot]
    below = [k for k in strikes if k < spot]
    out = {"straddle": [(exp, "call", atm), (exp, "put", atm)]}
    if above and below:
        out["strangle"] = [(exp, "call", above[0]), (exp, "put", below[-1])]
    return out


def trade(book, legs, mod_in, mod_out, fill):
    """ROE of a package bought at mod_in and sold at mod_out. fill: 'real'
    (ask in, bid - cushion out) or 'mid'."""
    ins = [quote(book, k, mod_in, 3) for k in legs]
    outs = [quote(book, k, mod_out, 10) for k in legs]
    if any(q is None for q in ins + outs):
        return None
    if fill == "mid":
        cost = sum((b + a) / 2 for b, a in ins)
        proc = sum((b + a) / 2 for b, a in outs)
    else:
        cost = sum(a for _b, a in ins)
        at_bid = sum(b for b, _a in outs)
        c = 0.5 if at_bid >= cost else 1.5
        proc = sum(max(0.0, b - c * (a - b)) for b, a in outs)
    if cost <= 0:
        return None
    return proc / cost - 1.0


def build_trades(keys, spots):
    """{key: {structure: {exit+fill: roe}, 'cost': {structure: cost}}} for
    deployed keys; also the 09:35 straddle mid per ticker-day for M1."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    by_day = defaultdict(list)
    for k in keys:
        by_day[k[2]].append(k)
    out, s0935 = {}, {}
    paths = {os.path.basename(os.path.dirname(p)).split("=")[1]: p
             for p in glob.glob(f"{SILVER}/date=*/bars.parquet")}
    for n, day in enumerate(sorted(by_day), 1):
        p = paths.get(day.isoformat())
        if p is None:
            continue
        books = day_book(p, day)
        for tk in IC.TICKERS:
            bk = books.get(tk)
            sp = spots.get((tk, day, 575))
            if bk and sp:
                lg = legs_for(bk, day, sp, 575)
                if lg:
                    qs = [quote(bk, k, 575, 2) for k in lg["straddle"]]
                    if all(qs):
                        s0935[(tk, day)] = sum((b + a) / 2 for b, a in qs)
        for k in by_day[day]:
            tk, _d, _day, m = k
            bk, sp = books.get(tk), spots.get((tk, day, m))
            if not bk or not sp:
                continue
            lg = legs_for(bk, day, sp, m)
            if not lg:
                continue
            rec = {"cost": {}}
            for st, legs in lg.items():
                r = {}
                for ex, h in EXITS.items():
                    mo = EOD if h is None else min(m + h, EOD)
                    for fill in ("real", "mid"):
                        r[f"{ex}_{fill}"] = trade(bk, legs, m, mo, fill)
                rec[st] = r
                ins = [quote(bk, q, m, 3) for q in legs]
                rec["cost"][st] = sum(a for _b, a in ins) if all(ins) else None
            out[k] = rec
        if n % 50 == 0:
            print(f"    trades {day}  ({len(out)})", flush=True)
    res = (out, s0935)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(res, f)
    return res


# ================================================================== scoring
def thirds_stats(rows, roe, keys):
    """rows from lasso_walk -> per third list of roe (deployed, defined)."""
    out = {0: [], 1: [], 2: []}
    for i, _p, t, _s in rows:
        v = roe.get(keys[i])
        if v is not None:
            out[t].append(v)
    return out


def v1_stat(rows, roe, keys):
    vals = [roe.get(keys[r[0]]) for r in rows]
    top = [roe.get(keys[r[0]]) for r in rows if r[2] == 2]
    vals = [v for v in vals if v is not None]
    top = [v for v in top if v is not None]
    return (np.mean(top) - np.mean(vals)) if vals and top else float("nan")


def day_boot(rows, roe, keys, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[keys[r[0]][2]].append(r)
    days = sorted(by)
    bs = []
    for _ in range(n):
        pick = [r for j in rng.integers(0, len(days), len(days)) for r in by[days[j]]]
        bs.append(fn(pick))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def sequential(rows, keys, roe_exit):
    """Top-third triggers only, one position per ticker, next allowed after
    the 60-minute exit. -> list of roe."""
    taken, busy = [], {}
    for i, _p, t, _s in sorted(rows, key=lambda r: (keys[r[0]][2], keys[r[0]][3])):
        if t != 2:
            continue
        tk, _d, day, m = keys[i]
        if busy.get((tk, day), -1) > m:
            continue
        v = roe_exit.get(keys[i])
        if v is None:
            continue
        taken.append(v)
        busy[(tk, day)] = m + 60
    return taken


# ================================================================== main
def main():
    import directional_flow_backtester as D
    keys, raw, _y30, tails = IC.build_dataset(D)
    keys = sorted((k for k in keys if k in tails), key=lambda k: (k[2], k[3], k[0], k[1]))
    mem = IC.memberships({k: raw[k] for k in keys})
    X = np.array([[mem[k][f] - (0.5 if f in IC.SIGNED + IC.UNSIGNED else 0.0) for f in IC.FEATS] for k in keys])
    big = np.array([1.0 if (tails[k]["mfe60"] >= BIG_S or tails[k]["mae60"] <= -BIG_S) else 0.0 for k in keys])
    hit = np.array([1.0 if tails[k]["mfe60"] >= BIG_S else 0.0 for k in keys])
    dep_keys = [k for k in keys if IC.is_dep(k[2])]
    print(f"  triggers {len(keys)} (deployed {len(dep_keys)});  BIG base rate {big.mean():.1%}")

    # spots at the trigger minute and 09:35 (1m closes)
    spots = {}
    for tk in IC.TICKERS:
        b = IC.BF.bars_full(tk)
        for day, mod, c in b.select("date", "mod", "close").iter_rows():
            spots[(tk, day, int(mod))] = c
    print("  building option trades (lake, deployed days) ...", flush=True)
    trades, s0935 = build_trades(dep_keys, spots)

    # ---- M1 / M3
    import check_implied_move as CI
    near = {(r, d): s for r, d, s in CI.straddles().filter(pl.col("win") == "am")
            .select("root", "date", "strad_near").iter_rows()}
    diffs = [abs(v - near[k]) / near[k] for k, v in s0935.items() if near.get(k)]
    m1 = len(diffs) > 100 and np.median(diffs) <= 0.01
    print(f"  M1 09:35 straddle vs cache: median |diff| {np.median(diffs):.2%} on {len(diffs)} ticker-days: "
          f"{'PASS' if m1 else 'FAIL'}")
    cheaper = [t["cost"].get("strangle") < t["cost"].get("straddle") for t in trades.values()
               if t["cost"].get("strangle") and t["cost"].get("straddle")]
    m3 = np.mean(cheaper) >= 0.95
    print(f"  M3 strangle cheaper than straddle: {np.mean(cheaper):.1%} of {len(cheaper)}: {'PASS' if m3 else 'FAIL'}")

    def roe_of(structure, col):
        return {k: (trades[k].get(structure) or {}).get(col) for k in trades}

    # ---- M2 planted
    # a CONTINUOUS plant: a binary one (= BIG) gives two-valued predictions,
    # and with a 37% base rate both third-cuts land on those values, putting
    # every trade in the middle third (found on the first run, before scoring)
    import pandas as pd
    size = np.array([max(tails[k]["mfe60"], -tails[k]["mae60"]) for k in keys])
    Xp = X.copy()
    Xp[:, 0] = pd.Series(size).rank(pct=True).to_numpy() - 0.5
    rp, _c, _n = IC.lasso_walk(keys, Xp, big)
    tp = thirds_stats(rp, roe_of("straddle", "60m_real"), keys)
    m2 = np.mean(tp[2]) - np.mean(tp[0]) > 0.10
    print(f"  M2 planted BIG: straddle top third {np.mean(tp[2]) * 100:+.1f}% vs bottom {np.mean(tp[0]) * 100:+.1f}%: "
          f"{'PASS' if m2 else 'FAIL'}")
    if not (m1 and m2 and m3):
        sys.exit("machinery checks failed -- not scoring")

    rows, coefs, nnz = IC.lasso_walk(keys, X, big)
    rows_hit, _c, _n = IC.lasso_walk(keys, X, hit)
    print("\n  BIG model: inputs kept per slice " + "  ".join(f"{s}:{n}" for s, n in nnz.items()))
    for j in [j for j in range(len(IC.FEATS)) if any(c[j] != 0 for c in coefs.values())]:
        print(f"      {IC.FEATS[j]:9} " + " ".join(f"{coefs[s][j]:+.4f}" for s in coefs))
    print(f"    chained Spearman(score, BIG) {IC.spearman([r[1] for r in rows], [big[r[0]] for r in rows]):+.4f}")

    rng = np.random.default_rng(SEED)
    tk_idx = {tk: np.array([i for i, k in enumerate(keys) if k[0] == tk]) for tk in IC.TICKERS}
    prow = []
    for _ in range(N_PLACEBO):
        Xq = X.copy()
        for tk, ix in tk_idx.items():
            Xq[ix] = X[rng.permutation(ix)]
        prow.append(IC.lasso_walk(keys, Xq, big)[0])

    for structure in ("straddle", "strangle"):
        roe = roe_of(structure, "60m_real")
        print(f"\n  {structure.upper()} -- 60-minute exit, ask in / bid-cushion out")
        th = thirds_stats(rows, roe, keys)
        allv = [v for v in roe.values() if v is not None]
        for t in (0, 1, 2):
            v = np.array(th[t])
            print(f"    third {['bottom', 'middle', 'top'][t]:6}: n={len(v):6}  mean ROE {v.mean() * 100:+6.1f}%  "
                  f"median {np.median(v) * 100:+6.1f}%  win {np.mean(v > 0):.1%}")
        print(f"    all triggers: n={len(allv)}  mean ROE {np.mean(allv) * 100:+.1f}%")
        d1 = v1_stat(rows, roe, keys)
        lo1, hi1 = day_boot(rows, roe, keys, lambda rr: v1_stat(rr, roe, keys))

        def sp(rr):
            ps = [(r[1], roe.get(keys[r[0]])) for r in rr if roe.get(keys[r[0]]) is not None]
            return IC.spearman([a for a, b in ps], [b for a, b in ps])
        s2 = sp(rows)
        lo2, hi2 = day_boot(rows, roe, keys, sp)
        per = {s: v1_stat([r for r in rows if r[3] == s], roe, keys) for s in range(1, 7)}
        pv = [v1_stat(rq, roe, keys) for rq in prow]
        p95 = float(np.nanquantile(pv, 0.95))
        top_mean = np.mean(th[2])
        crit = dict(V1=lo1 > 0, V2=lo2 > 0, V3=top_mean > 0, V4=sum(v > 0 for v in per.values()) >= 4, V5=d1 > p95)
        print(f"    V1 top - all {d1 * 100:+.2f}pp  CI [{lo1 * 100:+.2f}, {hi1 * 100:+.2f}]   V2 Spearman {s2:+.4f} "
              f"CI [{lo2:+.4f}, {hi2:+.4f}]   V3 top mean {top_mean * 100:+.1f}%")
        print("    V4 by slice: " + " ".join(f"{s}:{v * 100:+.1f}" for s, v in per.items()) +
              f"   V5 placebo p95 {p95 * 100:+.2f}pp (median {np.nanmedian(pv) * 100:+.2f})")
        print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
              f"   -> {structure.upper()} {'PASS' if all(crit.values()) else 'FAIL'}")
        for col in ("30m_real", "eod_real", "60m_mid"):
            r2 = roe_of(structure, col)
            t2 = thirds_stats(rows, r2, keys)
            print(f"      [{col}] top {np.mean(t2[2]) * 100:+.1f}%  bottom {np.mean(t2[0]) * 100:+.1f}%  "
                  f"all {np.mean([v for v in r2.values() if v is not None]) * 100:+.1f}%")
        th_h = thirds_stats(rows_hit, roe, keys)
        print(f"      [tail-HIT model's top third] {np.mean(th_h[2]) * 100:+.1f}%")
        seq = sequential(rows, keys, roe)
        print(f"      [sequential, top third only] n={len(seq)}  mean {np.mean(seq) * 100:+.1f}%  "
              f"total {np.sum(seq):+.2f} (premium units)")


if __name__ == "__main__":
    main()
