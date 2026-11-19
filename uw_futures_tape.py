"""
uw_futures_tape.py -- collect UW's CME futures trade tape (aggressor side on every print) before it rolls off (user, 2026-11-19).

WHY NOW. UW keeps the futures tape on a ROLLING window (their note 2026-11-18: ~2.5 months; probed 11-19: ES back to
~2026-07-20, NQ / RTY to early August) -- every day not collected is lost for good (the same lesson as
cleanbot-uw-history-floor). Needs the Advanced tier (403 futures_access_required otherwise).

WHAT. /api/futures/flow?products=P (ES, NQ, RTY; spreads excluded), paged backward with older_than/newer_than over
09:25-16:00 ET of each session. Every print: executed_at, contract (sym), price, size, aggressor side, the NBBO at the
print, block / crossed flags, levels swept.
  lake/bronze/futures-flow/date=D/{P}.parquet   all prints (every contract of the product that traded)
  lake/silver/futures-1s/date=D/{P}.parquet     the FRONT contract (most volume that session): per second
                                                buy / sell / other volume, prints, last price, cumulative delta
Page boundaries: the next cursor is the last print's time + 1 microsecond and prints are de-duplicated by trade_id, so
prints sharing a timestamp across a page break are not lost.

    python uw_futures_tape.py --backfill            every session the tape still serves, missing days only
    python uw_futures_tape.py --refresh             the last few sessions (the daily job; box timer after the close)
    python uw_futures_tape.py --day 2026-11-18      one session
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import polars as pl
import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
LAKE = os.path.join(ROOT, "lake")
BRONZE = os.path.join(LAKE, "bronze", "futures-flow")
SILVER = os.path.join(LAKE, "silver", "futures-1s")
BASE = "https://api.unusualwhales.com/api"
PRODUCTS = ("ES", "NQ", "RTY")
NY = ZoneInfo("America/New_York")
START_HM, END_HM = (9, 25), (16, 0)
EARLIEST = dt.date(2026, 7, 15)


def session():
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {os.getenv('UW_API_KEY', '').strip()}", "Accept": "application/json",
                      "User-Agent": "cleanbot-futures-tape/1.0"})
    return s


def iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def pull(s, product, d):
    """every print of `product` on session d, 09:25-16:00 ET. -> list of dicts (de-duplicated)."""
    lo = dt.datetime(d.year, d.month, d.day, *START_HM, tzinfo=NY)
    hi = dt.datetime(d.year, d.month, d.day, *END_HM, tzinfo=NY)
    cursor, seen, rows = hi, set(), []
    while True:
        params = {"products": product, "limit": 500, "older_than": iso(cursor), "newer_than": iso(lo), "include_spreads": "false"}
        for attempt in range(5):
            try:
                r = s.get(f"{BASE}/futures/flow", params=params, timeout=60)
                if r.status_code == 200:
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1)); continue
                raise SystemExit(f"{product} {d}: HTTP {r.status_code} {r.text[:160]}")
            except requests.RequestException as e:
                time.sleep(2 * (attempt + 1))
                if attempt == 4:
                    raise SystemExit(f"{product} {d}: {type(e).__name__}")
        page = r.json().get("data") or []
        new = [x for x in page if x.get("trade_id") not in seen]
        for x in new:
            seen.add(x.get("trade_id"))
        rows += new
        if len(page) < 500 or not new:
            return rows
        last = min(dt.datetime.fromisoformat(x["executed_at"].replace("Z", "+00:00")) for x in page)
        cursor = last + dt.timedelta(microseconds=1)
        if cursor <= lo:
            return rows


def to_frame(rows):
    if not rows:
        return None
    df = pl.DataFrame({
        "executed_at": [x["executed_at"] for x in rows], "sym": [x.get("sym") for x in rows],
        "price": [x.get("price") for x in rows], "size": [x.get("size") for x in rows], "side": [x.get("side") for x in rows],
        "nbbo_bid": [x.get("nbbo_bid") for x in rows], "nbbo_ask": [x.get("nbbo_ask") for x in rows],
        "is_block": [x.get("is_block") for x in rows], "crossed": [x.get("crossed") for x in rows],
        "event_levels": [x.get("event_levels") for x in rows], "trade_id": [str(x.get("trade_id")) for x in rows]})
    return (df.with_columns(pl.col("executed_at").str.to_datetime(time_zone="UTC"),
                            pl.col("price").cast(pl.Float64), pl.col("size").cast(pl.Int64),
                            pl.col("nbbo_bid").cast(pl.Float64), pl.col("nbbo_ask").cast(pl.Float64))
              .unique("trade_id").sort("executed_at"))


def silver(df):
    """per-second aggressor volume of the session's front contract (most volume)."""
    front = df.group_by("sym").agg(pl.col("size").sum()).sort("size", descending=True)["sym"][0]
    x = df.filter((pl.col("sym") == front) & (pl.col("is_block") != True))
    et = x["executed_at"].dt.convert_time_zone("America/New_York")
    x = x.with_columns(sec=(et.dt.hour().cast(pl.Int32) * 3600 + et.dt.minute().cast(pl.Int32) * 60 + et.dt.second().cast(pl.Int32)))
    g = (x.group_by("sec").agg(buy=pl.col("size").filter(pl.col("side") == "buy").sum(),
                                sell=pl.col("size").filter(pl.col("side") == "sell").sum(),
                                other=pl.col("size").filter(~pl.col("side").is_in(["buy", "sell"])).sum(),
                                prints=pl.len(), last=pl.col("price").last())
         .sort("sec").with_columns(cvd=(pl.col("buy") - pl.col("sell")).cum_sum(), sym=pl.lit(front)))
    return g


def do_day(d, products=PRODUCTS, force=False):
    s = session()
    out = []
    for p in products:
        fb = os.path.join(BRONZE, f"date={d.isoformat()}", f"{p}.parquet")
        if os.path.exists(fb) and not force:
            out.append(f"{p} have"); continue
        t0 = time.time()
        df = to_frame(pull(s, p, d))
        if df is None or df.height == 0:
            out.append(f"{p} empty"); continue
        for path, frame in ((fb, df), (os.path.join(SILVER, f"date={d.isoformat()}", f"{p}.parquet"), silver(df))):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            frame.write_parquet(path + ".tmp")
            os.replace(path + ".tmp", path)
        out.append(f"{p} {df.height:,} prints {time.time() - t0:.0f}s")
    return d, " | ".join(out)


def sessions(lo, hi):
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
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    today = dt.datetime.now(NY).date()
    last = today if dt.datetime.now(NY).time() >= dt.time(16, 10) else today - dt.timedelta(days=1)
    if a.day:
        print(do_day(dt.date.fromisoformat(a.day), force=True))
    days = sessions(EARLIEST, last) if a.backfill else (sessions(last - dt.timedelta(days=7), last) if a.refresh else [])
    if days:
        with ThreadPoolExecutor(a.workers) as ex:
            for d, msg in ex.map(do_day, days):
                print(d, msg, flush=True)
