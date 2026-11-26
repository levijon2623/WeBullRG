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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", action="store_true")
    ap.add_argument("--tk", default="SPY")
    a = ap.parse_args()
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
