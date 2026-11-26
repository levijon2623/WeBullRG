"""
check_premium.py -- SELLING PREMIUM AFTER A BIG, UNFADED OVERNIGHT MOVE (user, 2026-11-25).
PRE-REGISTERED 2026-11-25, committed before any option P&L below was computed.

WHERE IT COMES FROM. check_overnight.py (holdout scored 2026-11-25): after a BIG NO-FADE night the prior close was touched
during the session on only 26-29% of days vs 61-77% otherwise, in both halves and on SPY / QQQ / IWM (a descriptive, not a
registered test). The user: interesting for premium selling. The open question is whether the options ALREADY price it.
The user added a calendar: short the 0DTE, long the 1DTE.

NIGHTS  check_overnight.night_rows (frozen definitions, imported): BIG = z >= 1.37, NO FADE = F < 1/3; s = the night's
        direction; C0 = the prior close. ORDINARY = not BIG.
OPTIONS lake/silver/option-contracts-1m (2024-08-20 .. its last day), SPY. A leg's quote at minute m = the last bar of
        that contract at or before m, at most 10 minutes old (30 at the exit for the 0DTE short; a 0DTE leg still
        unquoted then and >= $0.50 out of the money at the 15:55 price is worth 0). A trade with an unquoted leg is not
        priced (counted). ENTRY 09:35 (bar 09:35's quote), EXIT 15:55, nothing in between.
FILLS   the user's measured fills per leg: buy mid + $0.0006, sell mid - $0.0008 per share, fees $0.001 per share per leg
        round trip ($0.06 / $0.08 / $0.10 a contract). Reported also: WORST CASE, every leg crossing the spread.
TRADES  per 1 spread, $ (x100)
  V  VERTICAL toward the prior close (the gap-fill side): gap up (s = +1) -> sell the 0DTE put at K = floor(C0), buy the
     put at K - 1; gap down -> sell the 0DTE call at K = ceil(C0), buy K + 1. Max loss $100 - credit.
  C  CALENDAR at the money: K = the whole-dollar strike nearest the 09:35 price; sell the 0DTE call, buy the call of the
     NEXT listed expiry (the next session's) at K. Exit both at 15:55.
PRIMARY (SPY, BIG NO-FADE nights)  P1 = mean V P&L > 0;  P2 = mean C P&L > 0. PASS each = holdout mean > 0 with the
     bootstrap (2,000 resamples of nights) 95% CI above 0, Benjamini-Hochberg q < 0.10 across the two, discovery mean > 0.
REPORTED, no pass/fail
  calibration  the short strike's 09:35 |delta| (the market's rough odds of finishing beyond it) vs how often price
               finished beyond it at 15:55, and how often it touched it.
  controls     ORDINARY nights: V on the side toward the prior close with the short strike whose 09:35 |delta| is nearest
               the BIG NO-FADE discovery median (same odds, ordinary morning); C at the money. Does a big no-fade morning
               pay more than an ordinary one at the same odds?
  variants     V $2 wide; C with puts; C struck at the prior close (a bet that price goes BACK there -- the opposite of V);
               worst-case fills; QQQ / IWM.
  risk         share of losing days, worst day, the sum of the worst 5 days.
SAMPLES  DISCOVERY = odd calendar months, HOLDOUT = even (as check_overnight). PARTLY CLEAN, said now: the gap-fill /
  touch rates behind the idea were seen on both halves (check_overnight); no option P&L was ever computed. --holdout writes
  _premium_cache/RESULT-holdout.txt and refuses a second run.

    python check_premium.py --discovery
    python check_premium.py --holdout        (once)
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "_premium_cache")
BUY_C, SELL_C, FEE = 0.0006, 0.0008, 0.0010
ENTRY, EXIT = 9 * 60 + 35, 15 * 60 + 55
MAXLAG, MAXLAG_EXIT0 = 10, 30


def day_book(args):
    """{(type 'C'/'P', strike, expiry): [(minute, bid, ask, delta), ...]} for the session's 0DTE and next expiry."""
    tk, d = args
    f = os.path.join(ROOT, "lake", "silver", "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(f):
        return d, None
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol") == tk)
         .with_columns(e=pl.col("expiry").cast(pl.Utf8),
                       m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
         .filter(pl.col("e") >= d).select("option_type", "strike", "e", "m", "bid_close", "ask_close", "delta_close").collect())
    if not x.height:
        return d, None
    exps = sorted(set(x["e"].to_list()))
    if exps[0] != d or len(exps) < 2:
        return d, None
    nxt = exps[1]
    x = x.filter(pl.col("e").is_in([d, nxt]))
    book = {}
    for t, k, e, m, b, a, dl in x.sort("m").iter_rows():
        if abs(k - round(k)) > 1e-6:
            continue
        book.setdefault((t[0].upper(), int(round(k)), e), []).append((m, b, a, dl))
    return d, {"next": nxt, "book": book}


def quote(B, key, m, lag):
    rows = B["book"].get(key)
    if not rows:
        return None
    best = None
    for r in rows:
        if r[0] > m:
            break
        best = r
    if best is None or m - best[0] > lag or not best[1] and not best[2]:
        return None
    b, a = float(best[1] or 0), float(best[2] or 0)
    if a <= 0 or a < b:
        return None
    return b, a, best[3]


def leg_value(q, side, worst):
    """per-share cash for opening a leg: side +1 = buy (pay), -1 = sell (receive)."""
    b, a, _ = q
    mid = (b + a) / 2
    if worst:
        return -a if side > 0 else b
    return -(mid + BUY_C) if side > 0 else (mid - SELL_C)


def close_value(q, side, worst):
    """per-share cash for closing a leg opened with `side`."""
    return leg_value(q, -side, worst)


def spread_pnl(B, legs, spot_exit, worst=False):
    """legs: [(key, side)] -> $ per spread, or None if any leg is unquoted."""
    cash = 0.0
    for key, side in legs:
        q0 = quote(B, key, ENTRY, MAXLAG)
        if q0 is None:
            return None
        lag = MAXLAG_EXIT0 if key[2] == B["d"] else MAXLAG
        q1 = quote(B, key, EXIT, lag)
        if q1 is None and key[2] == B["d"]:
            t, k, _ = key
            otm = (k - spot_exit) if t == "C" else (spot_exit - k)
            q1 = (0.0, 0.01, 0.0) if otm >= 0.50 else None
        if q1 is None:
            return None
        cash += leg_value(q0, side, worst) + close_value(q1, side, worst) - FEE
    return cash * 100.0


def vertical(B, s, C0, width=1, short_k=None):
    d = B["d"]
    if s > 0:
        K = short_k if short_k is not None else int(np.floor(C0))
        return [(("P", K, d), -1), (("P", K - width, d), +1)], ("P", K, d)
    K = short_k if short_k is not None else int(np.ceil(C0))
    return [(("C", K, d), -1), (("C", K + width, d), +1)], ("C", K, d)


def calendar(B, K, right="C"):
    return [((right, K, B["d"]), -1), ((right, K, B["next"]), +1)]


def build(tk="SPY"):
    import check_overnight as O
    rows = O.night_rows(tk)
    ext = O.load(tk)
    days = sorted({r["d"] for r in rows})
    with ProcessPoolExecutor(6) as ex:
        books = dict(ex.map(day_book, [(tk, d) for d in days]))
    out = []
    for r in rows:
        B = books.get(r["d"])
        if not B:
            continue
        B = dict(B, d=r["d"])
        x = ext[r["d"]].filter(pl.col("market_time") == "r")
        e, z = x.filter(pl.col("mod") == ENTRY)["close"], x.filter(pl.col("mod") == EXIT)["close"]
        if not len(e) or not len(z):
            continue
        s0, s1 = float(e[0]), float(z[0])
        hi, lo = float(x.filter(pl.col("mod") >= ENTRY)["high"].max()), float(x.filter(pl.col("mod") >= ENTRY)["low"].min())
        out.append(dict(r=r, B=B, s0=s0, s1=s1, hi=hi, lo=lo))
    return out


def trades(items, ref_delta=None):
    """per night: V, V2, C, Cp, Cc0 P&L (+ worst-case V and C) and calibration fields."""
    res = []
    for it in items:
        r, B, s0, s1 = it["r"], it["B"], it["s0"], it["s1"]
        s, C0 = r["s"], r["C0"]
        big_nf = r["big"] and not r["faded"]
        row = {"d": r["d"], "half": r["half"], "group": "bnf" if big_nf else ("ord" if not r["big"] else "bf")}
        side = 1 if s0 > C0 else -1                       # the gap-fill side is toward C0
        if big_nf:
            legs, sk = vertical(B, side, C0)
        elif ref_delta is not None:
            # ordinary night: the short strike on the side toward C0 whose 09:35 |delta| is nearest ref_delta
            t = "P" if side > 0 else "C"
            cands = []
            for (tt, k, e), _ in B["book"].items():
                if tt != t or e != r["d"]:
                    continue
                if (t == "P" and k >= s0) or (t == "C" and k <= s0):
                    continue
                q = quote(B, (tt, k, e), ENTRY, MAXLAG)
                if q and q[2] is not None:
                    cands.append((abs(abs(q[2]) - ref_delta), k))
            if not cands:
                legs = sk = None
            else:
                k = min(cands)[1]
                legs, sk = vertical(B, side, C0, short_k=k)
        else:
            legs = sk = None
        if legs:
            row["V"] = spread_pnl(B, legs, s1)
            row["Vw"] = spread_pnl(B, legs, s1, worst=True)
            legs2, _ = vertical(B, side, C0, width=2, short_k=sk[1])
            row["V2"] = spread_pnl(B, legs2, s1)
            q = quote(B, sk, ENTRY, MAXLAG)
            row["delta"] = abs(q[2]) if q and q[2] is not None else None
            row["beyond"] = (s1 < sk[1]) if sk[0] == "P" else (s1 > sk[1])
            row["touched"] = (it["lo"] <= sk[1]) if sk[0] == "P" else (it["hi"] >= sk[1])
        Katm = int(round(s0))
        row["C"] = spread_pnl(B, calendar(B, Katm, "C"), s1)
        row["Cw"] = spread_pnl(B, calendar(B, Katm, "C"), s1, worst=True)
        row["Cp"] = spread_pnl(B, calendar(B, Katm, "P"), s1)
        Kc0 = int(round(C0))
        row["Cc0"] = spread_pnl(B, calendar(B, Kc0, "P" if Kc0 < s0 else "C"), s1)
        res.append(row)
    return res


def mean_ci(v, n=2000, seed=3):
    v = np.array([x for x in v if x is not None], float)
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    return v.mean(), lo, hi, len(v), min(1.0, p)


def risk(v):
    v = np.sort(np.array([x for x in v if x is not None], float))
    if not len(v):
        return ""
    return f"lose {np.mean(v < 0):.0%} of days, worst ${v[0]:+.0f}, worst-5 sum ${v[:5].sum():+.0f}"


def report(tk, res, half, primary=False):
    H = [x for x in res if x["half"] == half]
    print(f"=== {tk} {half.upper()}")
    ps = []
    for g, lab in (("bnf", "BIG NO-FADE"), ("ord", "ORDINARY (same-delta vertical)")):
        G = [x for x in H if x["group"] == g]
        print(f"  {lab}: {len(G)} nights")
        for key, name in (("V", "vertical $1, toward the prior close"), ("V2", "vertical $2"), ("Vw", "vertical $1, WORST-CASE fills"),
                          ("C", "calendar ATM calls (short 0DTE / long next)"), ("Cp", "calendar ATM puts"),
                          ("Cw", "calendar ATM calls, WORST-CASE fills"), ("Cc0", "calendar at the prior close")):
            v = [x.get(key) for x in G]
            m, lo, hi, n, p = (*mean_ci(v),) if any(a is not None for a in v) else (np.nan, np.nan, np.nan, 0, 1.0)
            if n == 0:
                continue
            tag = ""
            if primary and g == "bnf" and key in ("V", "C"):
                ps.append((key, m, lo, hi, p))
            print(f"    {name:44s} ${m:+7.2f} [{lo:+7.2f}, {hi:+7.2f}] n {n:3d}  {risk(v)}")
        dl = [x["delta"] for x in G if x.get("delta") is not None]
        if dl:
            print(f"    calibration: short strike 09:35 |delta| {np.mean(dl):.2f} | finished beyond {np.mean([x['beyond'] for x in G if 'beyond' in x]):.0%}"
                  f" | touched {np.mean([x['touched'] for x in G if 'touched' in x]):.0%}")
    if primary and ps:
        pv = np.array([x[4] for x in ps])
        order = np.argsort(pv)
        q = np.empty(len(pv)); prev = 1.0
        for rank, i in reversed(list(enumerate(order, 1))):
            prev = min(prev, pv[i] * len(pv) / rank); q[i] = prev
        for (key, m, lo, hi, p), qq in zip(ps, q):
            print(f"  PRIMARY {'P1 vertical' if key == 'V' else 'P2 calendar'}: ${m:+.2f} [{lo:+.2f}, {hi:+.2f}] q {qq:.3f}"
                  + (("  " + ("PASS" if lo > 0 and qq < 0.10 else "fail")) if half == "holdout" else ""))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if not (a.discovery or a.holdout):
        print(__doc__)
        return
    os.makedirs(CACHE, exist_ok=True)
    mark = os.path.join(CACHE, "RESULT-holdout.txt")
    if a.holdout and os.path.exists(mark):
        sys.exit("holdout already scored -- see " + mark)
    half = "holdout" if a.holdout else "discovery"
    for tk in ("SPY", "QQQ", "IWM"):
        items = build(tk)
        first = trades(items)                             # big no-fade verticals -> the reference delta (DISCOVERY)
        ref = [x["delta"] for x in first if x["group"] == "bnf" and x["half"] == "discovery" and x.get("delta") is not None]
        ref_delta = float(np.median(ref)) if ref else None
        res = trades(items, ref_delta)
        print(f"{tk}: reference short-strike |delta| (big no-fade, discovery median) {ref_delta:.3f}")
        report(tk, res, half, primary=(tk == "SPY"))
    if a.holdout:
        open(mark, "w").write("scored\n")


if __name__ == "__main__":
    main()
