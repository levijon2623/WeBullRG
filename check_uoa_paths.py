# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_uoa_paths.py
==================
FOLLOWING A BIG BOUGHT OPTION PRINT, JUDGED THE WAY IT WOULD BE TRADED (user, 2026-10-31)

The two earlier UOA studies never scored this. check_uoa_target asked whether the
UNDERLYING touched the strike; check_uoa_events bought the contract at the NEXT
day's closing ask and sold at a fixed close 1 / 3 sessions later. The user's point:
a follower buys when the print shows up and takes a +100-200% pop when it comes --
judge the contract's PRICE PATH, not its value at expiry. And cheap short-dated
options spike often on their own, so the comparison has to be against contracts
of the same kind bought with no print behind them.

DATA     lake/silver/option-contracts-1m (one bar per contract-minute with a trade:
         side volumes, multi-leg volume, closing bid / ask, start-of-day OI,
         underlying). 2023-10-12 .. 2026-09-18.
GROUPS   PRIMARY SPY / QQQ / IWM. SECOND: the 6 bot names + the 25 replication
         tickers (check_uoa_events' groups), scored separately.
WINDOW   scored 2024-08-20 onward, halves split 2025-08-21; pre-sample
         2023-10-12 .. 2024-08-19 reported, labelled. Events whose expiry is past
         the data's end are dropped (counted).

EVENT (fixed before the run): contract c, day d, minute m, 09:31 <= m <= 15:58 ET
         calendar DTE 0..21 (user: "maybe 3 calendar weeks"); out of the money,
         0 < OTM <= 10% (calls K/S - 1, puts 1 - K/S, S = underlying at m);
         the 5 minutes ending at m: volume >= 1,000, premium >= $100k,
         >= 70% at the ask (BOUGHT), multi-leg < 50%, volume >= start-of-day OI.
         One event per contract-day: the first minute that qualifies.
ENTRY    the contract's closing ASK at m (what a follower pays right after the
         burst); >= $0.05 with a live bid.
CONTROLS 3 per event, drawn at random (seeded) from contract-minutes on OTHER
         days with no event on that contract-day: same ticker, same right,
         same calendar DTE, same OTM bin (0.5 pp), same time-of-day bin (30 min),
         same entry-price bin (log2, half-octave). Bought at their closing ask.
         So: the same lottery ticket, the same time of day, no print.
EXITS    from the minute after entry, on the closing BID (conservative):
         TP x: the first minute the bid reaches entry x (1 + x) -> exit at that
           price; never reached -> held to expiry (intrinsic at the expiry
           session's close; ETF/stock options settle on the close).
         x = +50% / +100% / +200%; HOLD = to expiry, no target.
HOUR     (the user's add) on the entry day, [m - 60, m + 60] minutes:
         where the entry sits in that window's mid-price range (0 = the low,
         1 = the high); the bid 60 minutes later vs entry; best and worst bid in
         that hour; the mid's change over the 60 minutes BEFORE the print.
         Reported for events and controls; not scored.

PRE-REGISTERED (per group)
  PRIMARY  TP +100%: per event, R(event) - mean R(its controls).
  PASS     mean > 0 with a day-block bootstrap 95% CI above 0 (5,000 draws,
           clustered on the event date); > 0 in both halves; ETFs: >= 2 of 3
           tickers > 0 (SECOND: >= 60% of tickers with >= 10 events).
  Reported, unscored: TP +50 / +200 / HOLD; calls vs puts; DTE 0 / 1-5 / 6-21;
  max gain before expiry (MFE) and when; share expiring worthless; the hour.
  MACHINERY  M1: +25 pp planted on every event's return must PASS;
             M0: an event swapped for one of its own controls must not.

Usage:  python check_uoa_paths.py --build    # pass 1 events + control pool, pass 2 paths
        python check_uoa_paths.py            # score from the cache
        python check_uoa_paths.py --min-tdte 3 --variant print|open|1h|sd   # runs 2-4
        python check_uoa_paths.py --forward --build --build-sameday --build-nextday   # forward build
        python check_uoa_paths.py --forward --min-tdte 3 --variant sd                 # FORWARD PRIMARY

FORWARD TEST (pre-registered 2026-10-31, BEFORE any day after IS_END was scored; the lake was
extended to 2026-10-30 the same day and those days stay unscored until the run below)
  WHY      nothing passed in-sample (runs 1-4). The forward test asks one question on data the
           design never saw: does a big bought print pick a better contract than its unprinted
           same-day twin, now that UOA flow may have changed?
  DATA     silver days > IS_END (2026-09-21 ..), own cache _uoa_paths_cache/fwd.
  EVENT    as above, >= 3 TRADING days to expiry at entry (user's rule), split guard on.
  ENTRY    PRIMARY right after the print (--variant sd: same-day controls).
           SECONDARY next open once OI confirms (--variant open, other-day controls).
  CONTROL  PRIMARY same-day (run 4: other-day controls carry the day's market move -- bought calls
           "beat" them only because call-buying days were up days; same-day twins erase that).
  EXIT     PRIMARY TP +100% on the bid, else expiry. +50 / +150 / +200 / +300 / HOLD reported.
  PASS     per group (ETFs; SECOND = 31 names), each its own verdict: mean event - control > 0,
           day-block bootstrap 95% CI above 0, ETFs >= 2 of 3 tickers > 0 / SECOND >= 60% of
           tickers with >= 10 events. No halves test (window too short). M1 must PASS, M0 FAIL.
  WHEN     ONE scored run once >= 120 trading days after IS_END are in the lake (~2027-04-15).
           Interim runs are allowed for data checks only -- read counts, not results.
  POWER    in-sample same-day CI half-width ~1.5 pp (ETFs, 2,020 events / 2 yrs) and ~2.4 pp
           (names, 1,165): six months should resolve an edge of ~3 pp (ETFs) / ~5 pp (names).

RUN LOG
  2026-10-31 run 1: 32,932 events (32,442 with controls, 96,286 controls); M1 PASS, M0 FAIL (correct).
    ETFs (15,960 scored, 512 days): TP+100 event - control -0.8%, CI [-3.9%, +2.3%], halves -1.8 / +0.0,
      2/3 tickers > 0 -> FAIL. TP+50 -1.1, TP+200 +0.4, HOLD +0.8. Prints and controls are the same
      ticket: hit +100% 36% / 36%, +200% 21% / 21%, MFE median +51% / +53%, worthless 75% / 75%;
      absolute TP+100 -25% / -24% per trade.
    SECOND (9,062): +3.8%, CI [-2.2%, +11.2%], halves -2.8 / +6.5, 43% of 30 tickers > 0 -> FAIL.
    Exploratory only (not pre-registered, many cuts): SECOND DTE 6-21 TP+100 -1% vs controls -12%;
      ETF calls HOLD +5% vs -11% (puts the other way, -43 vs -31).
    Hour: entries sit at median 0.38 of the +/-60m mid range -- same as controls (0.38): cheap OTM
      options decay through the hour (mid -7% the hour before, bid -11% median the hour after);
      0DTE -30% before / -39% after, best in hour +28%, worst -54%. Prints don't time the low.
    Pre-sample (labelled): ETFs +0.1%, SECOND -2.6%.
  2026-10-31 run 2 (user, after run 1: drop everything with <= 2 TRADING days to expiry; --min-tdte 3,
    same events/controls/bar. The user's reason: a design correction, not a pick from run 1 -- UOA needs
    time to play out, so <= 2 days should never have been in; run 1's ETF 6-21 bucket looked no better
    than the rest. Still run after run 1 on the same data, so it is not a replication):
    ETFs (3,412): TP+100 +0.7%, CI [-4.7%, +6.1%], halves -1.3 / +2.2 -> FAIL. M0 PASSED here (a
      control posing as the event cleared the bar) -- the ETF comparison is noise-level at this n.
      hit +100% 36/36, +200% 20/19; TP+100 -24% / -25% per trade.
    SECOND (5,520): TP+100 +8.9%, CI [-0.6%, +20.2%], halves -2.9 / +15.6, 67% of 30 tickers > 0 -> FAIL
      (CI touches 0, first half negative). All four exits +8..+10 pp; calls HOLD +32% vs +16%; prints
      -4% vs controls -12% per trade at TP+100. Pre-sample -0.9. Lives in the 2025-08..2026-09 half.
    Hour (>= 3 days): entry at median 0.47 of the range (controls 0.43); bid +60m -3% both.
  2026-10-31 CORRECTION (same day): runs 1-2's SECOND numbers were inflated by a STOCK SPLIT. XLE split
    2:1 on 2025-12-05; its old $87-89 puts kept their strikes while the close halved, so five events
    "settled" +90x..+140x (hit rates were identical, the mean was not). drop_splits() now removes any
    entry whose holding spans a split (24 entries, 7 events). M0 also fixed: it took the earliest-dated
    control as the fake event (biased; it wrongly PASSED twice) -- now a random control vs the others.
    Re-scored: run 1 ETFs -0.8 / SECOND -0.4 (was +3.8); run 2 ETFs +0.7 / SECOND +2.0, CI [-1.7, +5.7]
    (was +8.9). All FAIL; M1 PASS, M0 FAIL everywhere. Run 2 + TP150 / TP300 (user): ETFs -0.3 / +2.4,
    SECOND +2.6 / +2.6, all CIs across 0; hit rates print = control at every target (ETFs 36/27/20/13%
    vs 36/26/20/12%, names 41/32/25/16% vs 41/31/24/16%).
  2026-10-31 run 3 (user): enter the NEXT session once OI confirms (dOI >= 50% of the burst), at the
    first quote 09:31-09:45 and 10:30-10:45; controls move to their own next session. 96% of events
    stick by this rule, so the confirmation filters almost nothing. TP+100 event - control:
    open ETFs +1.1 CI [-5.1, +7.6] / names +1.8 [-2.5, +6.0]; 1h ETFs +1.3 [-5.4, +8.1] / names
    +1.0 [-3.4, +5.4]. All FAIL. Entry sits mid-range in the hour (0.54 open, 0.41 at 10:30) for both.
    Every variant: bought CALLS beat their other-day matches (HOLD +29..+32% vs +7..+16%), bought
    PUTS trail theirs -- the check_uoa_events drift signature.
  2026-10-31 run 4: SAME-DAY controls (--build-sameday, --variant sd): same ticker/right/DTE/OTM/time/
    price on the event's own day, no print (14,436 of 32,932 events have one). TP+100: ETFs -0.6 CI
    [-2.1, +0.8], 0/3 tickers > 0; names +1.0 [-1.4, +3.3]. FAIL. The calls edge is the DAY: ETF calls
    HOLD print +15% vs same-day twin +33% (other-day +2%); names +32% vs +30%. ETF prints trail their
    same-day twins at every target (TP+150 / +300 / HOLD -1.7 / -1.7 / -1.3, CIs touching 0 from below)
    -- in line with check_uoa_target's anti-predictive OTM-call UOA. Same-day CIs are 3x tighter.
  VERDICT: a big bought print adds nothing to the contract's path. Taking +100% (or +50 / +200) when
    it comes works exactly as often after a print as on the same contract profile with no print, and
    on average both lose ~20-25% per trade in ETFs after the spread.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from check_swing_factors import NAMES                    # noqa: E402
from check_swing_replicate import REPL                   # noqa: E402

SILVER = "lake/silver/option-contracts-1m"
CACHE = "_uoa_paths_cache"
ETFS = ("SPY", "QQQ", "IWM")
SECOND = tuple(NAMES) + tuple(REPL)
UNIVERSE = list(ETFS + SECOND)
DEP_START, SPLIT = dt.date(2024, 8, 20), dt.date(2025, 8, 21)
MAX_DTE, MAX_OTM = 21, 0.10
MIN_VOL, MIN_PREM, SIDE, MLEG, VOL_X_OI, MIN_ASK = 1000, 1e5, 0.70, 0.50, 1.0, 0.05
M0, M1 = 9 * 60 + 31, 15 * 60 + 58
IS_END = dt.date(2026, 9, 18)          # in-sample ends here; later lake days are the forward test's
TPS = (0.5, 1.0, 2.0)                  # flags the path walk keeps (any TP is bmax >= target anyway)
EXIT_TPS = (0.5, 1.0, 1.5, 2.0, 3.0)   # run 3 adds +150 / +300 (user)
STICK = 0.50                           # next-day entries: dOI >= 50% of the burst's volume
OPEN_WIN, HOUR_WIN = (9 * 60 + 31, 9 * 60 + 45), (10 * 60 + 30, 10 * 60 + 45)
PRIMARY = "tp100"
N_CTRL, SEED, N_BOOT = 3, 20261031, 5000
MNY_BIN, MOD_BIN, PX_BIN = 0.005, 30, 0.5
COLS = ["option_chain_id", "underlying_symbol", "option_type", "strike", "expiry", "minute_et", "volume",
        "premium", "ask_volume", "multi_volume", "bid_close", "ask_close", "underlying_close", "open_interest"]


def days(lo=None, hi=IS_END):
    ds = sorted(dt.date.fromisoformat(p.split("date=")[1][:10]) for p in glob.glob(f"{SILVER}/date=*"))
    return [d for d in ds if (lo is None or d >= lo) and (hi is None or d <= hi)]


def load_day(d, ids=None):
    s = pl.scan_parquet(f"{SILVER}/date={d}/bars.parquet")
    s = s.filter(pl.col("option_chain_id").is_in(ids)) if ids is not None else \
        s.filter(pl.col("underlying_symbol").is_in(UNIVERSE))
    return (s.select(COLS)
            .with_columns(dte=(pl.col("expiry") - pl.lit(d)).dt.total_days(),
                          mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                          + pl.col("minute_et").dt.minute().cast(pl.Int32))
            .filter(pl.col("dte").is_between(0, MAX_DTE))
            .collect())


def bins(df):
    return df.with_columns(mb=(pl.col("otm") / MNY_BIN).floor().cast(pl.Int32),
                           tb=(pl.col("mod") // MOD_BIN).cast(pl.Int32),
                           pb=(pl.col("ask").log(2) / PX_BIN).floor().cast(pl.Int32))


def pass1(D):
    """events, control pool and each ticker's close, one day at a time."""
    ev, pool, cl = [], [], []
    for i, d in enumerate(D):
        df = load_day(d)
        if df.is_empty():
            continue
        df = df.with_columns(
            otm=pl.when(pl.col("option_type") == "call").then(pl.col("strike") / pl.col("underlying_close") - 1)
            .otherwise(1 - pl.col("strike") / pl.col("underlying_close")),
            ask=pl.col("ask_close"), bid=pl.col("bid_close"))
        cl.append(df.filter(pl.col("mod") <= 959).sort("minute_et").group_by("underlying_symbol")
                  .agg(pl.col("underlying_close").last().alias("close")).with_columns(date=pl.lit(d)))
        w = (df.sort("option_chain_id", "minute_et")
             .rolling(index_column="minute_et", period="5m", group_by="option_chain_id")
             .agg(v5=pl.col("volume").sum(), p5=pl.col("premium").sum(), a5=pl.col("ask_volume").sum(),
                  m5=pl.col("multi_volume").sum()))
        x = df.join(w, on=["option_chain_id", "minute_et"], how="left")
        live = ((pl.col("mod").is_between(M0, M1)) & (pl.col("otm") > 0) & (pl.col("ask") >= MIN_ASK)
                & (pl.col("bid") > 0) & (pl.col("ask") >= pl.col("bid")))
        e = (x.filter(live & (pl.col("otm") <= MAX_OTM) & (pl.col("v5") >= MIN_VOL) & (pl.col("p5") >= MIN_PREM)
                      & (pl.col("a5") >= SIDE * pl.col("v5")) & (pl.col("m5") < MLEG * pl.col("v5"))
                      & (pl.col("v5") >= VOL_X_OI * pl.col("open_interest").fill_null(0)))
             .sort("minute_et").group_by("option_chain_id").first())
        hit = set(e["option_chain_id"].to_list())
        if not e.is_empty():
            ev.append(e.select("option_chain_id", "underlying_symbol", "option_type", "strike", "expiry", "dte",
                               "otm", "mod", "ask", "bid", "v5", "p5", "a5", "m5", "open_interest")
                      .with_columns(date=pl.lit(d)))
        pool.append(df.filter(live & (pl.col("otm") <= MAX_OTM + 0.02) & (pl.col("mod") % 5 == 0)
                              & ~pl.col("option_chain_id").is_in(list(hit)))
                    .select("option_chain_id", "underlying_symbol", "option_type", "strike", "expiry", "dte", "otm",
                            "mod", "ask").with_columns(date=pl.lit(d)))
        if i % 25 == 0:
            print(f"  pass 1 {d}  events so far {sum(len(z) for z in ev):,}", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    pl.concat(ev).write_parquet(f"{CACHE}/events.parquet")
    pl.concat(pool).write_parquet(f"{CACHE}/pool.parquet")
    pl.concat(cl).write_parquet(f"{CACHE}/closes.parquet")


def pick_controls(same_day=False, dst="entries"):
    """same_day (run 4): controls from the event's OWN day and time bin -- other contracts of the
    same ticker / right / DTE / OTM / price with no print. Removes the day's market move, which the
    other-day controls cannot: is the print's edge the contract, or the day it printed on?"""
    ev = bins(pl.read_parquet(f"{CACHE}/events.parquet")).with_row_index("ev")
    pool = bins(pl.read_parquet(f"{CACHE}/pool.parquet"))
    key = ["underlying_symbol", "option_type", "dte", "mb", "tb", "pb"]
    j = (ev.select(["ev", "date"] + key).join(pool.rename({"date": "cdate"}), on=key, how="inner")
         .filter((pl.col("cdate") == pl.col("date")) if same_day else (pl.col("cdate") != pl.col("date"))))
    rng = np.random.default_rng(SEED)
    j = (j.with_columns(r=pl.Series(rng.random(len(j))))
         .with_columns(k=pl.col("r").rank("ordinal").over("ev")).filter(pl.col("k") <= N_CTRL))
    print(f"  events {len(ev):,}; with >= 1 control {j['ev'].n_unique():,}; controls {len(j):,}")
    E = ev.select(pl.col("ev").alias("parent"), pl.lit("event").alias("kind"), "option_chain_id",
                  "underlying_symbol", "option_type", "strike", "expiry", "dte", "otm", "date", "mod", "ask")
    C = j.select(pl.col("ev").alias("parent"), pl.lit("control").alias("kind"), "option_chain_id",
                 "underlying_symbol", "option_type", "strike", "expiry", "dte", "otm", pl.col("cdate").alias("date"),
                 "mod", "ask")
    out = pl.concat([E, C]).with_row_index("eid")
    out.write_parquet(f"{CACHE}/{dst}.parquet")
    return out


def nextday(D):
    """the user's run-3 entries: the morning after the print, only once OI confirms it.

    Every print entry (event and control) moves to its NEXT session, bought at the first
    quote 09:31-09:45 ("open") and again at the first quote 10:30-10:45 ("1h"). Controls
    move the same way, so both sides carry the overnight. Events are kept only if they
    STUCK: start-of-day OI on d+1 minus OI on d >= 50% of the burst's volume.
    """
    en = pl.read_parquet(f"{CACHE}/entries.parquet").to_pandas()
    ev = pl.read_parquet(f"{CACHE}/events.parquet").with_row_index("ev").to_pandas()
    en["date"] = pd.to_datetime(en["date"]).dt.date
    Da = np.array(D)
    i = np.searchsorted(Da, en["date"].to_numpy(), side="right")
    en = en[i < len(Da)].copy()
    en["d1"] = Da[i[i < len(Da)]]
    got = []
    for k, (d1, g) in enumerate(en.groupby("d1")):
        rows = load_day(d1, g["option_chain_id"].unique().tolist())
        if rows.is_empty():
            continue
        rows = rows.filter((pl.col("bid_close") > 0) & (pl.col("ask_close") >= pl.col("bid_close"))
                           & (pl.col("ask_close") >= MIN_ASK)).sort("minute_et")
        oi = rows.group_by("option_chain_id").agg(oi1=pl.col("open_interest").first())
        for tag, (lo, hi) in (("open", OPEN_WIN), ("1h", HOUR_WIN)):
            w = (rows.filter(pl.col("mod").is_between(lo, hi)).group_by("option_chain_id").first()
                 .select("option_chain_id", pl.col("mod").alias("m1"), pl.col("ask_close").alias("a1"),
                         pl.col("underlying_close").alias("s1"), "dte").join(oi, on="option_chain_id", how="left")
                 .with_columns(tag=pl.lit(tag), d1=pl.lit(d1)))
            got.append(w)
        if k % 50 == 0:
            print(f"  next-day {d1}", flush=True)
    W = pl.concat(got).to_pandas()
    W["d1"] = pd.to_datetime(W["d1"]).dt.date
    x = en.merge(W, on=["option_chain_id", "d1"], how="inner")
    x["otm"] = np.where(x["option_type"] == "call", x["strike"] / x["s1"] - 1, 1 - x["strike"] / x["s1"])
    e0 = ev.set_index("ev")
    x["oi0"] = x["parent"].map(e0["open_interest"]).fillna(0)
    x["v5"] = x["parent"].map(e0["v5"])
    x["stick"] = (x["oi1"] - x["oi0"]) >= STICK * x["v5"]
    for tag in ("open", "1h"):
        y = x[x["tag"] == tag]
        out = (y.drop(columns=["eid", "date", "mod", "ask", "dte_x"], errors="ignore")
               .rename(columns={"d1": "date", "m1": "mod", "a1": "ask", "dte_y": "dte"}))
        out["stick"] = np.where(out["kind"] == "event", out["stick"], True)
        out = out.reset_index(drop=True)
        out.insert(0, "eid", np.arange(len(out)))
        out[["eid", "parent", "kind", "option_chain_id", "underlying_symbol", "option_type", "strike", "expiry",
             "dte", "otm", "date", "mod", "ask", "stick"]].to_parquet(f"{CACHE}/entries_{tag}.parquet")
        n_ev = (out["kind"] == "event").sum()
        print(f"  {tag}: {len(out):,} entries, {n_ev:,} events, {out.loc[out['kind'] == 'event', 'stick'].mean():.0%} stuck")


def pass2(D, src="entries", dst="paths"):
    """walk every entry's contract from its entry minute to expiry."""
    en = pl.read_parquet(f"{CACHE}/{src}.parquet").to_pandas()
    en["date"] = pd.to_datetime(en["date"]).dt.date
    en["expiry"] = pd.to_datetime(en["expiry"]).dt.date
    Ds = np.array(D)
    en["settle"] = [Ds[Ds <= e][-1] if (Ds <= e).any() else None for e in en["expiry"]]
    last = D[-1]
    en = en[en["expiry"] <= last].copy()
    st = {k: np.full(len(en), np.nan) for k in ("bmax", "bmax_day", "b60", "bmax_h", "bmin_h", "mid0",
                                                    "mid_lo", "mid_hi", "mid_pre")}
    for x in TPS:
        st[f"hit{x}"] = np.zeros(len(en), bool)
    pos = {e: i for i, e in enumerate(en["eid"])}
    for i, d in enumerate(D):
        act = en[(en["date"] <= d) & (en["settle"] >= d)]
        if act.empty:
            continue
        rows = load_day(d, act["option_chain_id"].unique().tolist())
        rows = rows.filter((pl.col("bid_close") > 0) & (pl.col("ask_close") >= pl.col("bid_close"))) \
                   .select("option_chain_id", "mod", pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"))
        if rows.is_empty():
            continue
        a = pl.from_pandas(act[["eid", "option_chain_id", "date", "mod", "ask"]].rename(
            columns={"mod": "m0", "ask": "E"})).with_columns(first=pl.col("date") == pl.lit(d))
        j = rows.join(a, on="option_chain_id", how="inner")
        post = j.filter(~pl.col("first") | (pl.col("mod") > pl.col("m0")))
        agg = post.group_by("eid").agg(bmax=pl.col("bid").max(), E=pl.col("E").first(),
                                       **{f"h{x}": (pl.col("bid") >= pl.col("E") * (1 + x)).any() for x in TPS})
        for r in agg.iter_rows(named=True):
            k = pos[r["eid"]]
            if np.isnan(st["bmax"][k]) or r["bmax"] > st["bmax"][k]:
                st["bmax"][k] = r["bmax"]
                st["bmax_day"][k] = (d - en["date"].iat[k]).days
            for x in TPS:
                if r[f"h{x}"]:
                    st[f"hit{x}"][k] = True
        h = j.filter(pl.col("first") & (pl.col("mod") - pl.col("m0")).abs().le(60)) \
             .with_columns(mid=(pl.col("bid") + pl.col("ask")) / 2)
        if not h.is_empty():
            hh = h.sort("mod").group_by("eid").agg(
                mid_lo=pl.col("mid").min(), mid_hi=pl.col("mid").max(),
                mid0=pl.col("mid").filter(pl.col("mod") == pl.col("m0")).first(),
                mid_pre=pl.col("mid").filter(pl.col("mod") < pl.col("m0")).first(),
                b60=pl.col("bid").filter(pl.col("mod") > pl.col("m0")).last(),
                bmax_h=pl.col("bid").filter(pl.col("mod") > pl.col("m0")).max(),
                bmin_h=pl.col("bid").filter(pl.col("mod") > pl.col("m0")).min())
            for r in hh.iter_rows(named=True):
                k = pos[r["eid"]]
                for c in ("mid_lo", "mid_hi", "mid0", "mid_pre", "b60", "bmax_h", "bmin_h"):
                    st[c][k] = np.nan if r[c] is None else r[c]
        if i % 25 == 0:
            print(f"  pass 2 {d}  active {len(act):,}", flush=True)
    for c, v in st.items():
        en[c] = v
    cl = pl.read_parquet(f"{CACHE}/closes.parquet").to_pandas()
    cl["date"] = pd.to_datetime(cl["date"]).dt.date
    cmap = {(t, d): c for t, d, c in zip(cl["underlying_symbol"], cl["date"], cl["close"])}
    S = np.array([cmap.get((t, d), np.nan) for t, d in zip(en["underlying_symbol"], en["settle"])])
    en["intrinsic"] = np.where(en["option_type"] == "call", np.maximum(S - en["strike"], 0),
                               np.maximum(en["strike"] - S, 0))
    en.to_parquet(f"{CACHE}/{dst}.parquet")
    print(f"  paths: {len(en):,} entries ({(en['kind'] == 'event').sum():,} events)")


# ---------------------------------------------------------------- scoring
def returns(en):
    en = en.copy()
    en["hold"] = en["intrinsic"] / en["ask"] - 1
    for x in EXIT_TPS:                      # a TP fires iff the best bid after entry reached it
        en[f"hit{x}"] = en["bmax"].fillna(0) >= en["ask"] * (1 + x) - 1e-9
        en[f"tp{int(x * 100)}"] = np.where(en[f"hit{x}"], x, en["hold"])
    en["mfe"] = en["bmax"] / en["ask"] - 1
    en["r1h"] = en["b60"] / en["ask"] - 1
    rng = en["mid_hi"] - en["mid_lo"]
    en["pos"] = np.where(rng > 0, (en["mid0"] - en["mid_lo"]) / rng, np.nan)
    en["pre"] = en["mid0"] / en["mid_pre"] - 1
    en["best_h"] = en["bmax_h"] / en["ask"] - 1
    en["worst_h"] = en["bmin_h"] / en["ask"] - 1
    return en


RULES = ("tp50", "tp100", "tp150", "tp200", "tp300", "hold")


def split_days():
    """{ticker: [days]} where the close jumped by a split ratio (1/k within 3%, k = 2..20).

    Run 2's single-name +8.9 pp was partly this: XLE split 2:1 on 2025-12-05, its old
    $87-89 puts kept their strikes in the data while the close halved, and five of them
    "settled" +90x to +140x. Splits found: WMT 2024-02-26, NVDA 06-10, AVGO 07-15,
    MSTR 08-08, NFLX 2025-11-17, XLE 12-05.
    """
    cl = pd.read_parquet(f"{CACHE}/closes.parquet").sort_values(["underlying_symbol", "date"])
    cl["r"] = cl.groupby("underlying_symbol")["close"].pct_change() + 1
    out = {}
    for t, d, r in zip(cl["underlying_symbol"], pd.to_datetime(cl["date"]).dt.date, cl["r"]):
        if r < 0.55 and any(abs(r * k - 1) < 0.03 for k in range(2, 21)):
            out.setdefault(t, []).append(d)
    return out


def drop_splits(en):
    sp = split_days()
    settle = pd.to_datetime(en["settle"]).dt.date
    bad = np.array([any(d < x <= s for x in sp.get(t, ())) for t, d, s in
                    zip(en["underlying_symbol"], en["date"], settle)])
    gone = set(en.loc[bad & (en["kind"] == "event"), "parent"])
    print(f"split guard: {bad.sum():,} entries span a split ({len(gone):,} events, dropped with their controls)")
    return en[~bad & ~en["parent"].isin(gone)]


def paired(en):
    ev = en[en["kind"] == "event"].set_index("parent")
    ct = en[en["kind"] == "control"].groupby("parent")[list(RULES)].mean()
    p = ev.join(ct, rsuffix="_c", how="inner")
    for r in RULES:
        p[f"d_{r}"] = p[r] - p[f"{r}_c"]
    p["date"] = pd.to_datetime(p["date"]).dt.date
    return p


def boot_ci(p, col, rng):
    g = p.groupby("date")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


def verdict(p, tickers, need, rng, col=f"d_{PRIMARY}", halves=True):
    if len(p) < 30:
        return False, "too few"
    lo, hi = boot_ci(p, col, rng)
    h1 = p.loc[p["date"] < SPLIT, col].mean()
    h2 = p.loc[p["date"] >= SPLIT, col].mean()
    by = p.groupby("underlying_symbol")[col].agg(["mean", "count"])
    by = by[by["count"] >= (1 if tickers == ETFS else 10)]
    share = (by["mean"] > 0).mean() if len(by) else 0
    ok = p[col].mean() > 0 and lo > 0 and (not halves or (h1 > 0 and h2 > 0)) and share >= need
    return ok, (f"mean {p[col].mean():+.1%}  CI [{lo:+.1%}, {hi:+.1%}]  halves {h1:+.1%} / {h2:+.1%}  "
                f"tickers > 0: {share:.0%} of {len(by)}")


def describe(en, label):
    for kind in ("event", "control"):
        x = en[en["kind"] == kind]
        if x.empty:
            continue
        print(f"   {label:14} {kind:7} n={len(x):6,}  "
              + "  ".join(f"{r} {x[r].mean():+.0%}" for r in RULES)
              + "  hit " + "/".join(f"{x[f'hit{t}'].mean():.0%}" for t in (1.0, 1.5, 2.0, 3.0)) + " (+100/150/200/300)"
              + f"  MFE med {x['mfe'].median():+.0%} p75 {x['mfe'].quantile(.75):+.0%}"
              + f"  worthless {(x['intrinsic'] <= 0).mean():.0%}")


def hour(en, label):
    for kind in ("event", "control"):
        x = en[en["kind"] == kind]
        if x.empty:
            continue
        print(f"   {label:14} {kind:7} entry pos in +/-60m range med {x['pos'].median():.2f} "
              f"(bottom 20%: {(x['pos'] <= .2).mean():.0%}, top 20%: {(x['pos'] >= .8).mean():.0%})  "
              f"prior 60m mid {x['pre'].median():+.0%}  bid +60m vs entry med {x['r1h'].median():+.0%} "
              f"mean {x['r1h'].mean():+.0%}  best in hour med {x['best_h'].median():+.0%}  "
              f"worst med {x['worst_h'].median():+.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-nextday", action="store_true", help="run 3: next-session entries + their paths")
    ap.add_argument("--build-sameday", action="store_true", help="run 4: same-day controls + paths")
    ap.add_argument("--variant", choices=("print", "open", "1h", "sd"), default="print",
                    help="entry: right after the print / next open once OI confirms / an hour into it")
    ap.add_argument("--min-tdte", type=int, default=0,
                    help="keep events with at least this many TRADING days to expiry at ENTRY (run 2+: 3)")
    ap.add_argument("--forward", action="store_true", help="the pre-registered forward window (days > IS_END)")
    a = ap.parse_args()
    global CACHE
    D = days()
    if a.forward:
        CACHE = f"{CACHE}/fwd"
        D = days(lo=IS_END + dt.timedelta(days=1), hi=None)
        print(f"FORWARD window: {len(D)} days {D[0] if D else '-'} .. {D[-1] if D else '-'}  cache {CACHE}")
    if a.build:
        print(f"pass 1 over {len(D)} days ({D[0]} .. {D[-1]})")
        pass1(D)
        pick_controls()
        pass2(D)
    if a.build_sameday:
        pick_controls(same_day=True, dst="entries_sd")
        pass2(D, "entries_sd", "paths_sd")
    if a.build_nextday:
        nextday(D)
        for tag in ("open", "1h"):
            pass2(D, f"entries_{tag}", f"paths_{tag}")
    en = returns(pd.read_parquet(f"{CACHE}/{'paths' if a.variant == 'print' else 'paths_' + a.variant}.parquet"))
    en["date"] = pd.to_datetime(en["date"]).dt.date
    en = drop_splits(en)
    if a.variant in ("open", "1h"):
        unstuck = set(en.loc[(en["kind"] == "event") & ~en["stick"].astype(bool), "parent"])
        en = en[~en["parent"].isin(unstuck)]
        print(f"entry: {a.variant} of the next session, events whose OI stuck only "
              f"({len(unstuck):,} unstuck events dropped with their controls)")
    en["grp"] = np.where(en["underlying_symbol"].isin(ETFS), "ETFs", "SECOND")
    Da = np.array(D)
    en["tdte"] = np.searchsorted(Da, pd.to_datetime(en["expiry"]).dt.date.to_numpy(), side="right") \
        - np.searchsorted(Da, en["date"].to_numpy(), side="right")
    if a.min_tdte:
        keep = set(en.loc[(en["kind"] == "event") & (en["tdte"] >= a.min_tdte), "parent"])
        en = en[en["parent"].isin(keep)]
        print(f"run with events >= {a.min_tdte} trading days to expiry (controls follow their event)")
    en["dteb"] = pd.cut(en["dte"], [-1, 0, 5, 21], labels=["0DTE", "1-5", "6-21"])
    rng = np.random.default_rng(SEED)
    print(f"entries {len(en):,}  events {(en['kind'] == 'event').sum():,}")
    for g, tks, need in (("ETFs", ETFS, 2 / 3), ("SECOND", SECOND, 0.60)):
        x = en[en["grp"] == g]
        sc = x[x["date"] >= DEP_START]
        pre = x[x["date"] < DEP_START]
        p = paired(sc)
        print(f"\n== {g}  scored events with controls {len(p):,} on {p['date'].nunique()} days")
        ok, txt = verdict(p, tks, need, rng, halves=not a.forward)
        print(f"   PRIMARY TP+100% event - control: {txt}  ->  {'PASS' if ok else 'FAIL'}")
        for r in RULES:
            if r != PRIMARY:
                lo_, hi_ = boot_ci(p, f"d_{r}", rng)
                print(f"   {r:6} event - control mean {p[f'd_{r}'].mean():+.1%}  CI [{lo_:+.1%}, {hi_:+.1%}]  "
                      f"(event {p[r].mean():+.0%} / control {p[f'{r}_c'].mean():+.0%} per trade)")
        pm = p.copy()
        pm[f"d_{PRIMARY}"] += 0.25
        print(f"   M1 (+25 pp planted): {'PASS' if verdict(pm, tks, need, rng, halves=not a.forward)[0] else 'FAIL'}  (must PASS)")
        # one RANDOM control per event plays the event, vs the mean of the OTHER controls
        # (run 1-2 used the first row = the earliest-dated control, a biased pick)
        cc = sc[sc["kind"] == "control"].assign(r=lambda z: rng.random(len(z)))
        cc = cc.sort_values("r")
        fake = cc.groupby("parent").first()[PRIMARY]
        rest = cc[~cc.index.isin(cc.groupby("parent").head(1).index)].groupby("parent")[PRIMARY].mean()
        pz = p.copy()
        pz[f"d_{PRIMARY}"] = fake.reindex(pz.index).to_numpy() - rest.reindex(pz.index).to_numpy()
        print(f"   M0 (control as event): {'PASS' if verdict(pz.dropna(subset=[f'd_{PRIMARY}']), tks, need, rng, halves=not a.forward)[0] else 'FAIL'}  (must FAIL)")
        describe(sc, "all")
        for cp in ("call", "put"):
            describe(sc[sc["option_type"] == cp], cp + "s")
        for b in ("0DTE", "1-5", "6-21"):
            describe(sc[sc["dteb"] == b], f"DTE {b}")
        if g == "ETFs":
            for t in ETFS:
                describe(sc[sc["underlying_symbol"] == t], t)
        print("   -- the hour around the print (entry day) --")
        hour(sc, "all")
        for cp in ("call", "put"):
            hour(sc[sc["option_type"] == cp], cp + "s")
        for b in ("0DTE", "1-5", "6-21"):
            hour(sc[sc["dteb"] == b], f"DTE {b}")
        pp = paired(pre)
        if len(pp):
            print(f"   pre-sample (labelled, extra in-sample): n={len(pp):,}  TP+100 event - control "
                  f"{pp[f'd_{PRIMARY}'].mean():+.1%}")


if __name__ == "__main__":
    main()
