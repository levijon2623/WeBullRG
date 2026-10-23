# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_uoa_events.py
===================
DO STICKY UNUSUAL OPTIONS PRINTS PREDICT THE NEXT FEW DAYS? (user, 2026-10-22)

The morning UOA reads (probe_uoa / probe_notable) flag single contracts by
hand: big volume against existing open interest, one-sided at the ask or the
bid, and still there in the next morning's OI ("sticky"). The user asked
whether stage A of the swing study would have caught the plays found that way
(QQQ 10/30 puts 10/16, IWM 10/21 $275-277 calls 10/19) -- it could not
(horizon, daily aggregation, mid-traded packages, data ends 09-18). This is
the test of that method itself, on every print that fits the definition, not
only the memorable ones.

DATA     _swing_cache/eod_{T}.parquet (check_swing_factors.build): per contract
         and day -- volume, ask / bid / multi-leg volume, premium, start-of-day
         OI, closing bid / ask. Underlying open / close from historical/{T}.
EVENT (fixed before the run), contract c on day d:
         1 <= DTE <= 45, volume >= 1,000 and >= 2 x start-of-day OI,
         premium >= $100k, >= 70% at the ask (BOUGHT) or the bid (SOLD),
         multi-leg < 50% (single-leg; packages reported separately),
         STICKY: OI(d+1) - OI(d) >= 50% of the volume (contract must print on
         d+1 so its OI is seen; unseen = not counted, reported).
         Direction: bought call / sold put = +1, sold call / bought put = -1.
UNIT     ticker-day: direction = sign of the premium-weighted sum of its
         events. Entry = the underlying's OPEN on d+1 (stickiness is known
         pre-open); r_h = log(close of d+h / open of d+1), h = 1 / 2 / 5.
         Score = direction x r_h (bp).
GROUPS   PRIMARY SPY / QQQ / IWM. SECOND: the 6 bot names + the 25
         replication tickers (check_swing_replicate), scored separately.
WINDOW   scored 2024-08-20 onward, halves split 2025-08-21; pre-sample
         2023-10-12 .. 2024-08-19 reported, labelled.

PRE-REGISTERED (per group, 3 horizons)
  NULL   day directions permuted within ticker (keeps timing and the long /
         short mix, so the up-drift is matched), 5,000 times; per-horizon z,
         family-wise p95 of the max z over the 3 horizons.
  EDGE   z > family-wise p95; both halves > 0; >= 2/3 ETFs (>= 60% of the
         second group's tickers with events) > 0.
  Reported, unscored: bought vs sold x calls vs puts; DTE <= 7 vs 8-45;
  premium tiers $1M / $5M; the SAME prints that did NOT stick (does the
  sticky check add anything?); packages (multi-leg >= 50%); following the
  contract itself (bought events: closing ask on d+1 -> closing bid 1 / 3
  sessions later).
  MACHINERY  M1: +25 bp planted in the event direction at h = 1 must pass;
             M0: random directions must not.

Usage:  python check_uoa_events.py

RUN LOG
  2026-10-22 run 1: ETF candidates SPY 12,545 / QQQ 9,181 / IWM 6,568;
    single-leg 7,701 / 5,275 / 3,399; next-day OI seen 98-99%; sticky 67 /
    65 / 81%. 1,468 ETF event-days (scored), 7,612 in the second group.
    M1 PASS, M0 PASS.
    ETFs: h1 -1.4 bp (z -0.54), h2 -0.2 (+0.30), h5 -2.0 (+0.35) vs
      family-wise z 2.08; hit rates 0.48-0.49. EDGE none.
    Second group (31 tickers): h1 -0.1, h2 +4.8 (z 1.01), h5 -12.0 (-1.38).
      EDGE none.
    Reported: bought calls / sold puts positive and bought puts / sold calls
      negative, symmetric -- that is the window's up-drift (the primary's
      permutation null removes it), not information. Stickiness adds
      nothing: ETF h5 sticky -2.0 bp, did-not-stick -7.3, ignored +0.8.
      Packages -10.5 bp. $5M+ prints: ETFs +17.9 bp h5 (n 192, hit 0.48),
      second +35 bp (n 687) -- unscored, inside noise.
      FOLLOWING the bought contract (d+1 closing ask -> bid): ETFs 1 session
      mean -17%, median -32%, win 0.28; 3 sessions mean -10%, median -50%,
      win 0.25. Second group similar (-12% / -10%).
    Pre-sample (labelled) ETFs: -1.4 / +0.6 / -7.1 bp.
  VERDICT: sticky one-sided UOA, as defined, carries no directional
    information for the next 1-5 days in ETFs or 31 liquid names, and copying
    the bought contracts loses about 10-17% on average after the spread. The
    morning UOA reads stay context (who is hedging what), not a signal.
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

import check_swing_factors as S                          # noqa: E402
from check_swing_replicate import REPL                   # noqa: E402

ETFS = ("SPY", "QQQ", "IWM")
SECOND = S.NAMES + REPL
DEP_START, SPLIT = dt.date(2024, 8, 20), dt.date(2025, 8, 21)
HS = (1, 2, 5)
MIN_VOL, VOL_X_OI, MIN_PREM, SIDE, MLEG, STICK = 1000, 2.0, 1e5, 0.70, 0.50, 0.50
N_PERM, SEED = 5000, 20261024


def ohlc(t):
    d = (pl.read_parquet(f"historical/{t}.parquet", columns=["date", "minute_et", "open", "close"])
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959)).sort("minute_et")
         .group_by("date").agg(pl.col("open").first().alias("o"), pl.col("mod").min().alias("m0"),
                               pl.col("close").last().alias("c"), pl.col("mod").max().alias("m1")).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d[(d["m0"] == 570) & (d["m1"] >= 955)].set_index("date").sort_index()
    return d[["o", "c"]]


def events(t):
    """All candidate prints with their stickiness and outcome inputs."""
    e = pl.read_parquet(f"{S.CACHE}/eod_{t}.parquet").to_pandas()
    e["date"] = pd.to_datetime(e["date"]).dt.date
    e["expiry"] = pd.to_datetime(e["expiry"]).dt.date
    e["dte"] = [(x - y).days for x, y in zip(e["expiry"], e["date"])]
    days = sorted(e["date"].unique())
    nxt = dict(zip(days[:-1], days[1:]))
    c = e[(e["dte"].between(1, 45)) & (e["vol"] >= MIN_VOL) & (e["vol"] >= VOL_X_OI * e["oi"].fillna(0).clip(lower=1))
          & (e["prem"] >= MIN_PREM)].copy()
    c["side"] = np.where(c["askv"] >= SIDE * c["vol"], 1, np.where(c["bidv"] >= SIDE * c["vol"], -1, 0))
    c = c[c["side"] != 0]
    c["pkg"] = c["mlegv"] >= MLEG * c["vol"]
    c["dir"] = c["side"] * np.where(c["option_type"] == "call", 1, -1)
    c["d1"] = c["date"].map(nxt)
    look = e.set_index(["option_chain_id", "date"])
    key = list(zip(c["option_chain_id"], c["d1"]))
    nx = look.reindex(key)
    c["oi1"] = nx["oi"].to_numpy()
    c["ask1"] = nx["ask_close"].to_numpy()
    c["dOI"] = c["oi1"] - c["oi"].fillna(0)
    c["seen"] = ~np.isnan(c["oi1"].to_numpy(float))
    c["sticky"] = c["seen"] & (c["dOI"] >= STICK * c["vol"])
    c["unstuck"] = c["seen"] & (c["dOI"] < 0.25 * c["vol"])
    # follow the contract (bought prints): closing ask on d+1 -> closing bid 1 / 3 sessions later
    for k in (1, 3):
        dk = c["d1"].map(lambda d, k=k: _shift(days, d, k))
        bk = look.reindex(list(zip(c["option_chain_id"], dk)))["bid_close"].to_numpy()
        c[f"roe{k}"] = bk / c["ask1"] - 1
    c["tk"] = t
    return c


def _shift(days, d, k):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    try:
        i = days.index(d)
    except ValueError:
        return None
    return days[i + k] if i + k < len(days) else None


def day_table(ev, px):
    """ticker-day rows: direction, premium, and the underlying's returns after the event."""
    if ev.empty:
        return pd.DataFrame()
    w = ev.assign(sp=ev["dir"] * ev["prem"]).groupby("date").agg(sp=("sp", "sum"), n=("dir", "size"),
                                                                 prem=("prem", "sum"))
    w = w[w["sp"] != 0]
    w["dir"] = np.sign(w["sp"])
    idx = list(px.index)
    pos = {d: i for i, d in enumerate(idx)}
    for h in HS:
        r = []
        for d in w.index:
            i = pos.get(d)
            if i is None or i + h >= len(idx):
                r.append(np.nan)
                continue
            r.append(np.log(px["c"].iloc[i + h] / px["o"].iloc[i + 1]))
        w[f"r{h}"] = r
    return w


def score(D, rng, title, need_share, plant=None, randomize=False, verbose=True):
    """D: concat of day tables with column tk. -> (hits, details)."""
    D = D[D.index >= DEP_START].dropna(subset=[f"r{h}" for h in HS]).copy()
    if randomize:
        D["dir"] = rng.choice([-1.0, 1.0], len(D))
    tk = D["tk"].to_numpy()
    s = D["dir"].to_numpy(float)
    R = {h: D[f"r{h}"].to_numpy(float) * 1e4 for h in HS}
    if plant:
        R[1] = R[1] + plant * s
    real = {h: np.mean(s * R[h]) for h in HS}
    groups = [np.flatnonzero(tk == t) for t in np.unique(tk)]
    null = np.empty((N_PERM, len(HS)))
    for p in range(N_PERM):
        sp = s.copy()
        for ix in groups:
            sp[ix] = rng.permutation(s[ix])
        null[p] = [np.mean(sp * R[h]) for h in HS]
    mu, sd = null.mean(0), null.std(0)
    z = (np.array([real[h] for h in HS]) - mu) / sd
    p95 = float(np.quantile(((null - mu) / sd).max(1), 0.95))
    dd = np.array(D.index)
    hits = []
    if verbose:
        print(f"\n  {title}   event-days {len(D)} on {len(np.unique(tk))} tickers   family-wise p95 z {p95:.2f}")
        print("     h   mean signed bp   z      IS / OOS bp        tickers > 0    hit rate   EDGE")
    for j, h in enumerate(HS):
        x = s * R[h]
        hv = [x[dd < SPLIT].mean(), x[dd >= SPLIT].mean()]
        per = [x[tk == t].mean() for t in np.unique(tk)]
        pos = sum(v > 0 for v in per)
        edge = z[j] > p95 and all(v > 0 for v in hv) and pos >= need_share * len(per)
        if edge:
            hits.append(h)
        if verbose:
            print(f"     {h}   {real[h]:+8.1f}       {z[j]:+5.2f}   {hv[0]:+7.1f} / {hv[1]:+7.1f}    {pos}/{len(per)}"
                  f"           {np.mean(x > 0):.2f}      {'PASS' if edge else '  - '}")
    return hits


def report_cut(evs, pxs, label, mask_fn, tks):
    rows = []
    for t in tks:
        ev = evs[t]
        if ev.empty:
            continue
        sub = ev[mask_fn(ev)]
        w = day_table(sub, pxs[t])
        if not w.empty:
            rows.append(w.assign(tk=t))
    if not rows:
        print(f"     {label:44} n=0")
        return
    D = pd.concat(rows)
    D = D[D.index >= DEP_START].dropna(subset=[f"r{h}" for h in HS])
    cells = "  ".join(f"h{h} {np.mean(D['dir'] * D[f'r{h}']) * 1e4:+6.1f}bp ({np.mean(D['dir'] * D[f'r{h}'] > 0):.2f})"
                      for h in HS)
    print(f"     {label:44} n={len(D):5d}  {cells}")


def main():
    rng = np.random.default_rng(SEED)
    evs, pxs = {}, {}
    for t in ETFS + SECOND:
        if not os.path.exists(f"{S.CACHE}/eod_{t}.parquet") or not os.path.exists(f"historical/{t}.parquet"):
            continue
        try:
            evs[t], pxs[t] = events(t), ohlc(t)
        except Exception as e:                           # noqa: BLE001
            print(f"  {t}: {type(e).__name__}: {e}")
    print("  candidate prints (one-sided, big vs OI, DTE 1-45) -> sticky single-leg EVENTS:")
    for t in ETFS:
        ev = evs.get(t, pd.DataFrame())
        if ev.empty:
            continue
        sl = ev[~ev["pkg"]]
        print(f"     {t}: {len(ev):6d} candidates, single-leg {len(sl):6d}, OI seen next day {sl['seen'].mean():.0%},"
              f" sticky {sl['sticky'].sum():5d} ({sl['sticky'].mean():.0%})")

    def primary(ev):
        return ev["sticky"] & ~ev["pkg"]

    def table(tks, fn):
        rows = []
        for t in tks:
            if t in evs and not evs[t].empty:
                w = day_table(evs[t][fn(evs[t])], pxs[t])
                if not w.empty:
                    rows.append(w.assign(tk=t))
        return pd.concat(rows) if rows else pd.DataFrame()

    E = table(ETFS, primary)
    h1 = score(E, rng, "M1 -- +25 bp planted at h=1 (ETFs)", 2 / 3, plant=25.0, verbose=False)
    h0 = score(E, rng, "M0 -- random directions (ETFs)", 2 / 3, randomize=True, verbose=False)
    print(f"\n  M1 {'PASS' if 1 in h1 else 'FAIL'}   M0 {'PASS' if not h0 else 'FAIL'}")

    print("\n  ======== PRIMARY: SPY / QQQ / IWM, sticky single-leg events ========")
    hits = score(E, rng, "ETFs", 2 / 3)
    G = table([t for t in SECOND if t in evs], primary)
    print("\n  ======== SECOND GROUP: 31 names / ETFs (bot names + replication universe) ========")
    hits2 = score(G, rng, "second group", 0.6) if not G.empty else []
    print(f"\n  VERDICT: ETFs EDGE {hits or 'none'}   second group EDGE {hits2 or 'none'}")

    print("\n  ======== REPORTED (unscored; mean of direction x return, bp, hit rate) ========")
    for grp, tks in (("ETFs", ETFS), ("second", [t for t in SECOND if t in evs])):
        print(f"  {grp}:")
        cuts = [
            ("all sticky single-leg (primary)", primary),
            ("  bought calls", lambda e: primary(e) & (e["side"] == 1) & (e["option_type"] == "call")),
            ("  sold calls", lambda e: primary(e) & (e["side"] == -1) & (e["option_type"] == "call")),
            ("  bought puts", lambda e: primary(e) & (e["side"] == 1) & (e["option_type"] == "put")),
            ("  sold puts", lambda e: primary(e) & (e["side"] == -1) & (e["option_type"] == "put")),
            ("  DTE <= 7", lambda e: primary(e) & (e["dte"] <= 7)),
            ("  DTE 8-45", lambda e: primary(e) & (e["dte"] > 7)),
            ("  premium >= $1M", lambda e: primary(e) & (e["prem"] >= 1e6)),
            ("  premium >= $5M", lambda e: primary(e) & (e["prem"] >= 5e6)),
            ("same prints that did NOT stick (dOI < 25%)", lambda e: e["unstuck"] & ~e["pkg"]),
            ("all one-sided big prints, stickiness ignored", lambda e: ~e["pkg"]),
            ("packages (multi-leg >= 50%), sticky", lambda e: e["sticky"] & e["pkg"]),
        ]
        for lab, fn in cuts:
            report_cut(evs, pxs, lab, fn, tks)
        b = pd.concat([evs[t][primary(evs[t]) & (evs[t]["side"] == 1)].assign(tk=t) for t in tks if t in evs])
        b = b[b["date"] >= DEP_START]
        for k in (1, 3):
            v = b[f"roe{k}"].replace([np.inf, -np.inf], np.nan).dropna()
            print(f"     follow the BOUGHT contract, d+1 close ask -> {k} session(s) later bid: n={len(v)}  mean {v.mean():+.1%}"
                  f"  median {v.median():+.1%}  win {np.mean(v > 0):.2f}")
    P = table(ETFS, primary)
    P = P[P.index < DEP_START].dropna(subset=[f"r{h}" for h in HS]) if not P.empty else P
    if len(P):
        print("\n  PRE-SAMPLE (labelled), ETFs primary: " + "  ".join(
            f"h{h} {np.mean(P['dir'] * P[f'r{h}']) * 1e4:+.1f}bp" for h in HS) + f"   n={len(P)}")


if __name__ == "__main__":
    main()
