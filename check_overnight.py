"""
check_overnight.py -- BIG OVERNIGHT MOVES, THE PRE-MARKET FADE, AND THE OPEN (user, 2026-11-25).
PRE-REGISTERED 2026-11-25, committed before any outcome after 09:30 was computed on any night.

THE USER'S OBSERVATION. Big overnight gains or losses get faded in pre-market, sometimes on real volume and sometimes
not; when the fading volume is weak, price later retraces toward the overnight high / low. Recent nights also seemed to
reverse in the opening "shootout" and then rip; maybe the regime matters. Low expectations for calling the open.

WHAT WAS LOOKED AT BEFORE THIS FILE (predictors only, no outcome): the 8 nights 2026-11-13..11-25 as a table with the
user; and on DISCOVERY nights the distribution of the night variables below -- big nights (top third) are mostly NOT
faded (90 no-fade vs 33 faded), and by a typical-morning yardstick almost no big-night fade is "weak" (2 of 33). Hence,
agreed with the user 2026-11-25: the PRIMARY question is on big NO-FADE nights; the fade question is descriptive, with
fade volume against the move's own volume AND against a typical morning. The --nights group COUNTS (predictors only)
were also printed for the holdout before commit: big 103 = no fade 73 / faded 30.

DATA  lake/silver/ext-1m/date=D/{SPY,QQQ,IWM}.parquet (uw_ext_candles.py): UW 1-minute candles, pre-market 04:00-09:29
      ('pr'), session 09:30-15:59 ('r'), after-hours 16:00-19:59 ('po'). NOT 20:00-04:00 (no SPY candles exist then).
      Extremes use 1-minute CLOSES (the first pre-market minutes carry stray prints). Daily VIX: _cot_cache/
      prices_daily.json (Cboe). Dealer gamma: lake/silver/spot-exposures-1m (from 2024-08-20).
NIGHT D (between session P and session D; all known by 09:29:59 on D)
  C0     close of P's last session minute (15:59)
  E      P 16:00-19:59 + D 04:00-09:29, 1-minute closes
  s, X   s = +1 if max(E) - C0 >= C0 - min(E), else -1; X = the extreme close of E in direction s; tX its minute
  M      |X - C0| / C0;   z = M / median(M of the 20 prior nights)  (big relative to recent nights, not a fixed %)
  P0     close of the last pre-market minute <= 09:29
  F      fade = (X - P0) / (X - C0): 0 = still at the extreme, 1 = all the way back to C0, > 1 = through it
  BIG    z >= 1.37, the top third of the DISCOVERY z distribution (predictors only), FROZEN
  NO FADE  F < 1/3;   FADED  F >= 1/3
  VR_move  fade volume per minute / move volume per minute -- move = E minutes up to and including tX, fade = after
           tX to 09:29; per CLOCK minute of the window (20:00-04:00 not counted). WEAK_move = VR_move < 1
  VR_typ   fade volume / the median of the same clock span's volume over the 20 prior sessions. WEAK_typ = VR_typ < 1
  Volume never counts 16:00-16:14: 71% of after-hours volume prints there (closing-auction reports; measured on the
  last 120 sessions before commit, after a first count showed 26 of 33 fades "weak" by move volume -- predictors only).
REGIME (reported splits) gamma sign = SPY gamma_per_one_percent_move_oi at P's last minute; WITH / AGAINST = s vs the
  sign of P's session (C0 - P's 09:30 open); VIX = P's close, above / below its 252-day median.
DIRECTIONAL DELTA (user, 2026-11-25, added before commit; same-day-expiry options from lake/silver/option-contracts-1m,
  sum over contracts of (ask volume - bid volume) x delta_close x 100 = shares of delta bought; 2024-08-20 .. the lake's
  last day only, so these splits cover ~2 of the 3 years):
  DD_prev   P's session and P's last hour, signed by s (+ = flow was already pushing the overnight direction)
  DD_shoot  09:30-09:39 on D, signed by sign(S10) (+ = flow CHASED the shootout); terciles cut on discovery nights
  DD_pre    D's pre-market (07:30-09:25 Cboe session) -- exists only from 2026-10-19: forward / descriptive only
OUTCOMES (session D, 1-minute candles; O = D's 09:30 open)
  S10    the "shootout": close(09:39) - O
  REV    -sign(S10) x (close(10:30) - close(09:39)), in units of the median |close(10:30) - close(09:39)| over the 20
         prior sessions. > 0 = the next 51 minutes went AGAINST the shootout.
  RETURN for FADED nights: did the session trade back to X (high >= X for s = +1, low <= X for s = -1) by 10:30, and
         by the close; and the share of (X - P0) recovered by 10:30 (max excursion toward X).

Q1 PRIMARY (SPY): on BIG NO-FADE nights the opening shootout reverses MORE than on ordinary nights.
   estimand  mean REV on BIG NO-FADE nights minus mean REV on NOT-BIG nights (z < 1.37), one value per night.
   PASS      holdout estimate > 0, bootstrap (2,000 resamples of nights) 95% CI above 0, AND the discovery estimate > 0.
             Scored once on the holdout.
   reported  mean REV and P(REV > 0) in each group; S10 in the overnight direction (sign(S10) = s) by group; Q1 within
             the regime splits; Q1 by DD_shoot tercile and by DD_prev sign (does a CHASED shootout reverse more? --
             the strike studies found aggressive one-sided flow into a level stalls / fails); QQQ / IWM the same.
H2 AMENDMENT 2026-11-25 -- added AFTER the discovery run (_overnight_discovery.txt), BEFORE any holdout night was
   scored; agreed with the user. Discovery showed the Q1 gap in the AGAINST-prior-session split on all three tickers
   (SPY +0.53, QQQ +0.40, IWM +0.61) and the opposite in the WITH split (-0.18 / -0.37 / -0.25) -- the user's own
   observation ("the nights faded the day's rally, then it ripped after the shootout"). Not independent across the
   tickers (they move together).
   estimand  SPY, BIG NO-FADE nights: mean REV on nights AGAINST the prior session minus mean REV on nights WITH it.
   PASS      holdout estimate > 0 with the bootstrap 95% CI above 0, AND the discovery estimate > 0. Scored once, with Q1.
Q2 DESCRIPTIVE (the user's fade idea; SPY, QQQ, IWM): on BIG FADED nights, RETURN by 10:30 and by the close for WEAK vs
   STRONG fades, under both yardsticks (VR_move, VR_typ); also vs all FADED nights regardless of size. Day-block CIs, no
   pass / fail (about 33 big faded nights per half per ticker). ES overnight (lake/silver/futures-overnight-1m, complete
   from 2026-07-27): how often ES's overnight extreme falls in 20:00-04:00, outside SPY's window -- the blind spot.
Q3 DESCRIPTIVE  gap fill (C0 touched during the session) and the first-10-minute direction by night type and regime.

SAMPLES  nights 2023-12-01 .. 2026-11-24 (the first 20 nights of the candle history feed the baselines). DISCOVERY =
  nights in ODD calendar months, HOLDOUT = EVEN months (each night intact: P's after-hours and D's pre-market and session
  belong to D). Discovery: everything reported, code checked. Holdout: --holdout writes _overnight_cache/
  RESULT-holdout.txt and refuses to run again.
NULL CHECK (discovery, before scoring): relabel a random set of nights the size of the BIG NO-FADE group as "big no-fade"
  (200 draws); the Q1 estimand must centre on 0 (within +-0.05) or the comparison is redesigned BEFORE the holdout.

RUN LOG (2026-11-25; discovery _overnight_discovery.txt, holdout scored once _overnight_cache/RESULT-holdout-full.txt)
  Null check +0.036 (OK). REV in units of the 20-session median |10:30 - 09:39| move.
  Q1 PRIMARY  SPY big no-fade minus not big: discovery +0.16 [-0.34, +0.68], HOLDOUT +0.26 [-0.31, +0.90] -> FAIL.
              QQQ +0.02 / +0.29, IWM +0.14 / +0.10: positive in all six reads, never distinguishable from 0. The
              shootout reverses ~48-57% of the time after big no-fade nights vs ~50-53% otherwise -- a coin flip.
  H2          SPY AGAINST minus WITH: discovery +0.33, HOLDOUT -0.07 [-1.25, +1.03] -> FAIL (QQQ -0.29, IWM 0.00):
              the discovery split did not repeat.
  Descriptive  shootout goes WITH the overnight move 31-60% (coin flip everywhere). Gap fill: big no-fade 26-29% vs
              61-77% faded / ordinary nights (both halves, all three). User's fade idea, big faded nights back to the
              extreme by 10:30, weak vs strong by move volume: discovery SPY 0/12 vs 38%, QQQ 0/10 vs 45%; HOLDOUT SPY
              19% vs 21%, QQQ 50% vs 17%, IWM 29% vs 24% -- no consistent direction (n 7-21 a cell); "weak" also means
              the extreme was set the evening before (25-50% vs 0-4%). Shootout 0DTE delta: chased top third REV
              discovery +0.05, holdout +0.41 -- inconsistent. ES extreme set 20:00-04:00 on 18% of 85 nights.

    python check_overnight.py --nights          (the night table, predictors only -- what was looked at before)
    python check_overnight.py --discovery
    python check_overnight.py --holdout         (once)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
EXT = os.path.join(ROOT, "lake", "silver", "ext-1m")
CACHE = os.path.join(ROOT, "_overnight_cache")
Z_BIG = 1.37
F_FADE = 1.0 / 3.0
BASE_N = 20
AH_VOL_FROM = 16 * 60 + 15          # after-hours volume counts from 16:15 (see night_rows)
FIRST_NIGHT, LAST_NIGHT = "2023-12-01", "2026-11-24"


def load(tk):
    out = {}
    for f in sorted(glob.glob(os.path.join(EXT, "date=*", f"{tk}.parquet"))):
        d = os.path.basename(os.path.dirname(f))[5:]
        x = pl.read_parquet(f)
        out[d] = x.with_columns(mod=(pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(pl.Int32)))
    return out


def night_rows(tk):
    """the night table (PREDICTORS only, everything known by 09:29:59) for every consecutive pair of sessions."""
    D = load(tk)
    days = sorted(D)
    raw = []
    for i in range(1, len(days)):
        p, d = days[i - 1], days[i]
        a, b = D[p], D[d]
        r = a.filter(pl.col("market_time") == "r")
        pm = b.filter((pl.col("mod") >= 240) & (pl.col("mod") < 570))
        if r.height < 200 or pm.height < 10:
            continue
        ah = a.filter((pl.col("mod") >= 960) & (pl.col("mod") < 1200))
        C0, O_prev = float(r["close"][-1]), float(r["open"][0])
        E = pl.concat([ah.select("mod", "close", "volume").with_columns(k=pl.lit(0)),
                       pm.select("mod", "close", "volume").with_columns(k=pl.lit(1))])
        cl = E["close"].to_numpy()
        s = 1 if cl.max() - C0 >= C0 - cl.min() else -1
        j = int(np.argmax(cl)) if s > 0 else int(np.argmin(cl))
        X, k, tmod = float(cl[j]), int(E["k"][j]), int(E["mod"][j])
        P0 = float(pm["close"][-1])
        M = abs(X - C0) / C0
        F = (X - P0) / (X - C0) if X != C0 else 0.0
        # clock minutes of the move (E up to tX) and of the fade (after tX to 09:29); 20:00-04:00 not counted, and
        # 16:00-16:14 left out of every VOLUME measure: 71% of after-hours volume prints there (closing-auction
        # reports, measured on the last 120 sessions before this file was committed), not overnight trading
        a0 = AH_VOL_FROM
        mv_min = max(0, tmod - a0 + 1) if k == 0 else ((1200 - a0) + (tmod - 240 + 1))
        fd_min = ((1200 - max(tmod + 1, a0)) + 330) if k == 0 else (570 - tmod - 1)
        vol = np.where((E["k"].to_numpy() == 0) & (E["mod"].to_numpy() < a0), 0.0, E["volume"].to_numpy())
        v_mv, v_fd = float(vol[:j + 1].sum()), float(vol[j + 1:].sum())
        span = (k, tmod)
        raw.append(dict(d=d, p=p, s=s, C0=C0, X=X, P0=P0, M=M, F=F, tk_ext=k, tmod=tmod,
                        vr_move=(v_fd / max(fd_min, 1)) / (v_mv / max(mv_min, 1)) if v_mv > 0 and fd_min > 0 else None,
                        v_fd=v_fd, span=span, prev_dir=1 if C0 >= O_prev else -1, ext=(ah, pm)))
    rows = []
    for i, r in enumerate(raw):
        if i < BASE_N:
            continue
        prev = raw[i - BASE_N:i]
        r["z"] = r["M"] / float(np.median([q["M"] for q in prev]))
        k, tmod = r["span"]
        base = []
        for q in prev:                                   # the same clock span on the 20 prior nights
            ah, pm = q["ext"]
            if k == 0:
                v = float(ah.filter(pl.col("mod") > max(tmod, AH_VOL_FROM - 1))["volume"].sum()) + float(pm["volume"].sum())
            else:
                v = float(pm.filter(pl.col("mod") > tmod)["volume"].sum())
            base.append(v)
        b = float(np.median(base))
        r["vr_typ"] = r["v_fd"] / b if b > 0 else None
        r["big"] = r["z"] >= Z_BIG
        r["faded"] = r["F"] >= F_FADE
        r["half"] = "discovery" if int(r["d"][5:7]) % 2 == 1 else "holdout"
        r["with_prev"] = r["s"] == r["prev_dir"]
        rows.append({k2: v for k2, v in r.items() if k2 not in ("ext", "span")})
    return [r for r in rows if FIRST_NIGHT <= r["d"] <= LAST_NIGHT]


# ================================================================ scoring (written after the pre-registration, 0aa2371)
REV_FROM, REV_TO, SHOOT_END = 9 * 60 + 39, 10 * 60 + 30, 9 * 60 + 39     # close(09:39), close(10:30)


def outcomes(tk, rows):
    """attach the session-D outcomes (after 09:30) to each night row."""
    D = load(tk)
    days = sorted(D)
    idx = {d: i for i, d in enumerate(days)}
    unit = {}
    for i, d in enumerate(days):                           # |close(10:30) - close(09:39)| per session
        r = D[d].filter(pl.col("market_time") == "r")
        a = r.filter(pl.col("mod") == REV_FROM)["close"]
        b = r.filter(pl.col("mod") == REV_TO)["close"]
        unit[d] = abs(float(b[0]) - float(a[0])) if len(a) and len(b) else None
    for row in rows:
        d = row["d"]
        r = D[d].filter(pl.col("market_time") == "r")
        i = idx[d]
        prev = [unit[x] for x in days[max(0, i - BASE_N):i] if unit[x] is not None]
        u = float(np.median(prev)) if len(prev) >= 10 else None
        a, b = r.filter(pl.col("mod") == REV_FROM)["close"], r.filter(pl.col("mod") == REV_TO)["close"]
        if not r.height or not len(a) or not len(b) or not u:
            row["ok"] = False
            continue
        O, c939, c1030 = float(r["open"][0]), float(a[0]), float(b[0])
        S10 = c939 - O
        row.update(ok=True, O=O, S10=S10, shoot_with=(np.sign(S10) == row["s"]) if S10 else None,
                   REV=(-np.sign(S10) * (c1030 - c939) / u) if S10 else 0.0)
        s, X, P0 = row["s"], row["X"], row["P0"]
        e = r.filter(pl.col("mod") <= REV_TO)
        hi_e, lo_e = float(e["high"].max()), float(e["low"].min())
        hi_d, lo_d = float(r["high"].max()), float(r["low"].min())
        row["ret_1030"] = (hi_e >= X) if s > 0 else (lo_e <= X)
        row["ret_close"] = (hi_d >= X) if s > 0 else (lo_d <= X)
        span = (X - P0) * s
        row["recover_1030"] = (min(1.0, ((hi_e if s > 0 else lo_e) - P0) * s / span) if span > 0 else None)
        row["gap_fill"] = lo_d <= row["C0"] <= hi_d
    return rows


def regimes(tk, rows):
    """gamma sign at P's last minute, VIX vs its 252-day median at P's close."""
    px = json.load(open(os.path.join(ROOT, "_cot_cache", "prices_daily.json")))
    vix = sorted((r["date"], float(r["close"])) for r in px["VIX"])
    vd = {d: v for d, v in vix}
    vdates = [d for d, _ in vix]
    for row in rows:
        p = row["p"]
        f = os.path.join(ROOT, "lake", "silver", "spot-exposures-1m", f"date={p}", f"{tk}.parquet")
        row["gamma"] = None
        if os.path.exists(f):
            g = pl.read_parquet(f).sort("minute_et")["gamma_per_one_percent_move_oi"].drop_nulls()
            row["gamma"] = float(g[-1]) if len(g) else None
        row["vix_hi"] = None
        if p in vd:
            j = vdates.index(p)
            hist = [vd[x] for x in vdates[max(0, j - 252):j]]
            if len(hist) >= 200:
                row["vix_hi"] = vd[p] > float(np.median(hist))
    return rows


def dd_day(args):
    """same-day-expiry directional delta per minute for one session: {minute-of-day: shares}; cached."""
    tk, d = args
    p = os.path.join(CACHE, "dd", f"{d}_{tk}.json")
    if os.path.exists(p):
        return (tk, d), {int(k): v for k, v in json.load(open(p)).items()}
    f = os.path.join(ROOT, "lake", "silver", "option-contracts-1m", f"date={d}", "bars.parquet")
    out = {}
    if os.path.exists(f):
        x = (pl.scan_parquet(f).filter((pl.col("underlying_symbol") == tk) & (pl.col("expiry").cast(pl.Utf8) == d))
             .with_columns(m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32),
                           dd=(pl.col("ask_volume").fill_null(0) - pl.col("bid_volume").fill_null(0)).cast(pl.Float64)
                              * pl.col("delta_close").fill_null(0.0) * 100.0)
             .group_by("m").agg(pl.col("dd").sum()).collect())
        out = {int(m): float(v) for m, v in x.iter_rows()}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    return (tk, d), out


def deltas(tk, rows):
    from concurrent.futures import ProcessPoolExecutor
    need = sorted({(tk, r["p"]) for r in rows} | {(tk, r["d"]) for r in rows})
    with ProcessPoolExecutor(6) as ex:
        got = dict(ex.map(dd_day, need))
    for row in rows:
        P, D_ = got.get((tk, row["p"])) or {}, got.get((tk, row["d"])) or {}
        row["dd_prev"] = (sum(v for m, v in P.items() if 570 <= m < 960) * row["s"]) if P else None
        row["dd_prev_last"] = (sum(v for m, v in P.items() if 900 <= m < 960) * row["s"]) if P else None
        S10 = row.get("S10")
        row["dd_shoot"] = (sum(v for m, v in D_.items() if 570 <= m <= SHOOT_END) * np.sign(S10)) if (D_ and S10) else None
    return rows


def build(tk):
    os.makedirs(CACHE, exist_ok=True)
    rows = night_rows(tk)
    rows = [r for r in outcomes(tk, rows) if r.get("ok")]
    rows = regimes(tk, rows)
    rows = deltas(tk, rows)
    return rows


def boot_diff(A, B, n=2000, seed=11):
    A, B = np.asarray(A, float), np.asarray(B, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(A, len(A)).mean() - rng.choice(B, len(B)).mean() for _ in range(n)]
    return A.mean() - B.mean(), *np.percentile(bs, [2.5, 97.5])


def ci_mean(A, n=2000, seed=12):
    A = np.asarray([a for a in A if a is not None], float)
    if not len(A):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(A, len(A)).mean() for _ in range(n)]
    return A.mean(), *np.percentile(bs, [2.5, 97.5]), len(A)


def q1(rows):
    bnf = [r["REV"] for r in rows if r["big"] and not r["faded"]]
    nb = [r["REV"] for r in rows if not r["big"]]
    return bnf, nb


def report(tk, rows, half, primary=False):
    H = [r for r in rows if r["half"] == half]
    print(f"=== {tk} {half.upper()}: {len(H)} nights")
    bnf, nb = q1(H)
    est, lo, hi = boot_diff(bnf, nb)
    tag = ""
    if primary:
        tag = "  PRIMARY: " + ("PASS" if lo > 0 and est > 0 else "fail")
    print(f"  Q1 REV big no-fade {np.mean(bnf):+.3f} (n {len(bnf)}, against the shootout {np.mean(np.array(bnf) > 0):.0%}) | "
          f"not big {np.mean(nb):+.3f} (n {len(nb)}, {np.mean(np.array(nb) > 0):.0%}) | difference {est:+.3f} [{lo:+.3f}, {hi:+.3f}]{tag}")
    groups = (("big no-fade", lambda r: r["big"] and not r["faded"]), ("big faded", lambda r: r["big"] and r["faded"]),
              ("not big", lambda r: not r["big"]))
    for lab, f in groups:
        G = [r for r in H if f(r)]
        if not G:
            continue
        sw = [r["shoot_with"] for r in G if r["shoot_with"] is not None]
        print(f"    {lab:12s} n {len(G):3d} | shootout WITH the overnight move {np.mean(sw):.0%} | gap filled {np.mean([r['gap_fill'] for r in G]):.0%}"
              f" | mean |S10| {np.mean([abs(r['S10']) / r['O'] * 1e4 for r in G]):.0f} bp")
    # regime / delta splits of Q1 (descriptive)
    splits = (("gamma +", lambda r: r["gamma"] is not None and r["gamma"] > 0), ("gamma -", lambda r: r["gamma"] is not None and r["gamma"] <= 0),
              ("night WITH prior session", lambda r: r["with_prev"]), ("night AGAINST prior session", lambda r: not r["with_prev"]),
              ("VIX above median", lambda r: r["vix_hi"] is True), ("VIX below median", lambda r: r["vix_hi"] is False),
              ("prior-session delta WITH the night", lambda r: r["dd_prev"] is not None and r["dd_prev"] > 0),
              ("prior-session delta AGAINST", lambda r: r["dd_prev"] is not None and r["dd_prev"] <= 0))
    for lab, f in splits:
        S = [r for r in H if f(r)]
        a, b = q1(S)
        if len(a) >= 5 and len(b) >= 5:
            e, l, h = boot_diff(a, b, n=1000)
            print(f"    split {lab:34s} Q1 {e:+.3f} [{l:+.3f}, {h:+.3f}]  (big no-fade n {len(a)}, not big n {len(b)})")
    return est, lo, hi


def shoot_terciles(tk, rows):
    disc = [r["dd_shoot"] for r in rows if r["half"] == "discovery" and r["dd_shoot"] is not None]
    if len(disc) < 30:
        return None
    return np.percentile(disc, [33.333, 66.667])


def report_delta(tk, rows, half, cuts):
    if cuts is None:
        return
    H = [r for r in rows if r["half"] == half and r["dd_shoot"] is not None]
    lab = ("shootout SOLD by flow (bottom third)", "middle", "shootout CHASED by flow (top third)")
    for k in range(3):
        f = (lambda r, k=k: (r["dd_shoot"] <= cuts[0]) if k == 0 else ((cuts[0] < r["dd_shoot"] <= cuts[1]) if k == 1 else r["dd_shoot"] > cuts[1]))
        S = [r for r in H if f(r)]
        a = [r["REV"] for r in S if r["big"] and not r["faded"]]
        b = [r["REV"] for r in S if not r["big"]]
        allr = ci_mean([r["REV"] for r in S])
        print(f"    0DTE delta in the shootout, {lab[k]:36s} REV all {allr[0]:+.3f} [{allr[1]:+.3f}, {allr[2]:+.3f}] n {allr[3]}"
              + (f" | big no-fade {np.mean(a):+.3f} (n {len(a)}) vs not big {np.mean(b):+.3f} (n {len(b)})" if a and b else ""))


def report_fade(tk, rows, half):
    H = [r for r in rows if r["half"] == half and r["faded"]]
    for scope, G in (("big faded", [r for r in H if r["big"]]), ("all faded", H)):
        for yard in ("vr_move", "vr_typ"):
            for lab, f in (("weak", lambda r, y=yard: r[y] is not None and r[y] < 1), ("strong", lambda r, y=yard: r[y] is not None and r[y] >= 1)):
                S = [r for r in G if f(r)]
                if not S:
                    continue
                a = ci_mean([float(r["ret_1030"]) for r in S]); c = ci_mean([float(r["ret_close"]) for r in S])
                rc = np.mean([r["recover_1030"] for r in S if r["recover_1030"] is not None])
                print(f"    fade {scope:9s} {yard:7s} {lab:6s} n {len(S):3d} | back to the extreme by 10:30 {a[0]:.0%} [{a[1]:.0%}, {a[2]:.0%}]"
                      f" by the close {c[0]:.0%} [{c[1]:.0%}, {c[2]:.0%}] | recovered by 10:30 {rc:.0%} of the fade")


def null_check(rows, draws=200, seed=5):
    D = [r for r in rows if r["half"] == "discovery"]
    n = sum(1 for r in D if r["big"] and not r["faded"])
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(draws):
        pick = set(rng.choice(len(D), n, replace=False))
        a = [D[i]["REV"] for i in pick]
        b = [D[i]["REV"] for i in range(len(D)) if i not in pick and not D[i]["big"]]
        est.append(np.mean(a) - np.mean(b))
    est = np.array(est)
    print(f"--- null check (discovery, {draws} random 'big no-fade' sets of {n}): mean {est.mean():+.3f} "
          f"[{np.percentile(est, 2.5):+.3f}, {np.percentile(est, 97.5):+.3f}] -> {'OK' if abs(est.mean()) <= 0.05 else 'REDESIGN'}")
    return est.mean()


def es_blind(rows):
    """descriptive: how often ES's overnight extreme (in the night's direction) falls in 20:00-04:00 ET."""
    hits, n = 0, 0
    for r in rows:
        f = os.path.join(ROOT, "lake", "silver", "futures-overnight-1m", f"date={r['d']}", "ES.parquet")
        if not os.path.exists(f) or r["d"] < "2026-07-28":
            continue
        x = pl.read_parquet(f)
        j = int(np.argmax(x["close"].to_numpy())) if r["s"] > 0 else int(np.argmin(x["close"].to_numpy()))
        m = x["minute"][j]
        hm = m.hour * 60 + m.minute
        n += 1
        hits += (hm >= 1200) or (hm < 240)
    if n:
        print(f"--- ES blind spot: on {n} nights with ES data, the ES overnight extreme was set 20:00-04:00 ET "
              f"(no SPY candles) on {hits} ({hits / n:.0%})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", action="store_true")
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--tk", default="SPY")
    a = ap.parse_args()
    if a.discovery or a.holdout:
        half = "holdout" if a.holdout else "discovery"
        mark = os.path.join(CACHE, "RESULT-holdout.txt")
        if a.holdout and os.path.exists(mark):
            sys.exit("holdout already scored -- see " + mark)
        built = {tk: build(tk) for tk in ("SPY", "QQQ", "IWM")}
        if a.discovery:
            null_check(built["SPY"])
        for tk in ("SPY", "QQQ", "IWM"):
            rows = built[tk]
            report(tk, rows, half, primary=(tk == "SPY"))
            report_delta(tk, rows, half, shoot_terciles(tk, rows))
            report_fade(tk, rows, half)
        for tk in ("SPY", "QQQ", "IWM"):
            for h in ("discovery", "holdout") if a.holdout else ("discovery",):
                B = [r for r in built[tk] if r["half"] == h and r["big"] and not r["faded"]]
                ag, wi = [r["REV"] for r in B if not r["with_prev"]], [r["REV"] for r in B if r["with_prev"]]
                e, l, hh = boot_diff(ag, wi)
                tag = ("  H2: " + ("PASS" if (h == "holdout" and l > 0) else "fail")) if (tk == "SPY" and h == "holdout") else ""
                print(f"--- H2 {tk} {h}: big no-fade AGAINST prior session {np.mean(ag):+.3f} (n {len(ag)}) minus WITH "
                      f"{np.mean(wi):+.3f} (n {len(wi)}) = {e:+.3f} [{l:+.3f}, {hh:+.3f}]{tag}")
        es_blind(built["SPY"])
        if a.holdout:
            open(mark, "w").write("scored\n")
        return
    if a.nights:
        rows = night_rows(a.tk)
        disc = [r for r in rows if r["half"] == "discovery"]
        print(f"{a.tk}: {len(rows)} nights ({len(disc)} discovery); z cut {Z_BIG} = top "
              f"{100 * np.mean([r['z'] >= Z_BIG for r in disc]):.0f}% of discovery")
        for half in ("discovery", "holdout"):
            H = [r for r in rows if r["half"] == half]
            B = [r for r in H if r["big"]]
            nf, fd = [r for r in B if not r["faded"]], [r for r in B if r["faded"]]
            print(f"  {half:9s} big {len(B):3d}: no fade {len(nf):3d} | faded {len(fd):3d} "
                  f"(weak by move volume {sum(1 for r in fd if r['vr_move'] is not None and r['vr_move'] < 1)}, "
                  f"by a typical morning {sum(1 for r in fd if r['vr_typ'] is not None and r['vr_typ'] < 1)})")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
