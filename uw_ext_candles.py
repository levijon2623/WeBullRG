"""
uw_ext_candles.py -- SPY 1-minute candles INCLUDING extended hours, into the lake (user, 2026-11-25).

WHY. The overnight / pre-market study (overnight moves, the pre-market fade and its volume, overnight highs / lows,
gaps, the open). The lake's stock minute data is regular-hours only; UW's /stock/{t}/ohlc/1m also returns pre-market
(market_time 'pr', 04:00-09:30 ET) and after-hours ('po', 16:00-20:00 ET) minutes. Probed 2026-11-25: present on every
date tried back to 2023-12-04. NOT covered: 20:00-04:00 ET (the overnight session) -- the ES overnight tape
(uw_futures_tape.py --kind ovn) is the only overnight source, and it only reaches back ~2.5 months.
Minutes with no trade are absent (pre-market is thin: ~1-2M shares vs 40-90M in the session).

    lake/silver/ext-1m/date=D/{T}.parquet   minute (ET), open, high, low, close, volume, market_time ('pr' / 'r' / 'po')

    python uw_ext_candles.py --backfill            every trading day from --since (default 2023-11-01) not yet stored
    python uw_ext_candles.py --day 2026-11-25
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "lake", "silver", "ext-1m")
NY = ZoneInfo("America/New_York")


def fetch(uw, tk, d):
    rows = uw.get(f"/stock/{tk}/ohlc/1m", date=d.isoformat(), limit=2500).get("data") or []
    if not rows:
        return None
    df = pl.DataFrame({"start_time": [r["start_time"] for r in rows],
                       "open": [float(r["open"]) for r in rows], "high": [float(r["high"]) for r in rows],
                       "low": [float(r["low"]) for r in rows], "close": [float(r["close"]) for r in rows],
                       "volume": [float(r.get("volume") or 0) for r in rows],
                       "market_time": [r.get("market_time") for r in rows]})
    df = (df.with_columns(minute=pl.col("start_time").str.to_datetime(time_zone="UTC").dt.convert_time_zone("America/New_York"))
            .drop("start_time").unique("minute").sort("minute"))
    df = df.filter(pl.col("minute").dt.date() == d)          # the requested session's calendar day only
    return df if df.height else None


def do_day(uw, tk, d, force=False):
    p = os.path.join(OUT, f"date={d.isoformat()}", f"{tk}.parquet")
    if os.path.exists(p) and not force:
        return d, "have"
    df = fetch(uw, tk, d)
    if df is None:
        return d, "empty"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.write_parquet(p + ".tmp")
    os.replace(p + ".tmp", p)
    c = df.group_by("market_time").len().sort("market_time").rows()
    return d, f"{df.height} min {dict(c)}"


def trading_days(lo, hi):
    import market_calendar as MC
    d, out = lo, []
    while d <= hi:
        if MC.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--since", default="2023-11-01")
    ap.add_argument("--tickers", default="SPY")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import positioning_map as P
    uw = P.UW()
    tks = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    now = dt.datetime.now(NY)
    last = now.date() if now.time() >= dt.time(20, 5) else now.date() - dt.timedelta(days=1)
    days = [dt.date.fromisoformat(a.day)] if a.day else (trading_days(dt.date.fromisoformat(a.since), last) if a.backfill else [])
    jobs = [(t, d) for t in tks for d in days]
    n_empty = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for (t, _), (d, msg) in zip(jobs, ex.map(lambda j: do_day(uw, j[0], j[1], force=bool(a.day)), jobs)):
            n_empty += msg == "empty"
            if msg not in ("have",):
                print(t, d, msg, flush=True)
    print(f"done: {len(jobs)} day-tickers, {n_empty} empty, {uw.calls} UW calls")
