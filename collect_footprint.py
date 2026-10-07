"""
collect_footprint.py
====================
Collects Webull STOCK FOOTPRINTS (aggressor buy / sell volume and delta per
bar, with per-price detail) for SPY, QQQ, IWM -- the underlying's VOLUME
DELTA, for the MA5-cross absorption test (check_ma5_delta.py, user
2026-10-07). Webull keeps only the latest 1,200 bars per timespan (1m ~3
sessions, 5s ~100 minutes), so history exists only if it is collected.

    --m1   after the close: the day's 1-minute bars (count 1200 -> also
           refills any session a missed run left behind)
    --s5   intraday, every ~90 minutes: the last 1,200 5-second bars
           (100 minutes), merged into the day's file

Output: _footprint_cache/{M1|S5}-YYYY-MM-DD.json  {symbol: {bar time: bar}}
(git-ignored). Bars are merged by time, so overlapping pulls are harmless and
a re-run never loses anything. REST only, from the bot folder (the shared
token); never a streaming connection. Runs from systemd timers on the box
(footprint-m1.timer, footprint-s5.timer); exits quietly on non-trading days.

FIRST LOOK (2026-10-07 probe): 1m totals are 19-26% of the bar volume (a
classified subset of the tape, not all of it); same-minute delta vs price
change Spearman SPY +0.19, QQQ +0.16, IWM +0.05; timestamps align at lag 0.

Usage:  python collect_footprint.py --m1 | --s5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
TICKERS = "SPY,QQQ,IWM"
OUT = "_footprint_cache"


def _et(t):
    return dt.datetime.fromisoformat(t.replace("+0000", "+00:00")).astimezone(NY)


def pull(timespan):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import webull_viewer_data as WV
    r = WV._client().market_data.get_footprint(
        symbols=TICKERS, category="US_STOCK", timespan=timespan, count=1200,
        real_time_required=False, trading_sessions="RTH")
    return r.json() if hasattr(r, "json") else r


def merge(timespan, body):
    """Split the pull by ET session date and merge into the per-day files."""
    by_day = {}
    for x in body or []:
        sym = x.get("symbol")
        for b in x.get("result") or []:
            if b.get("trading_session") not in (None, "RTH"):
                continue
            t = _et(b["time"])
            by_day.setdefault(t.date().isoformat(), {}).setdefault(sym, {})[t.isoformat()] = b
    os.makedirs(OUT, exist_ok=True)
    for day, syms in sorted(by_day.items()):
        p = os.path.join(OUT, f"{timespan}-{day}.json")
        have = {}
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                have = json.load(f)
        added = 0
        for sym, bars in syms.items():
            cur = have.setdefault(sym, {})
            added += sum(1 for k in bars if k not in cur)
            cur.update(bars)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(have, f, separators=(",", ":"))
        os.replace(tmp, p)
        n = {s: len(v) for s, v in have.items()}
        print(f"  {timespan} {day}: +{added} new bars -> {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--m1", action="store_true")
    g.add_argument("--s5", action="store_true")
    a = ap.parse_args()
    from market_calendar import is_trading_day
    today = dt.datetime.now(NY).date()
    ts = "M1" if a.m1 else "S5"
    if not is_trading_day(today):
        print(f"  {today}: not a trading day -- nothing to collect")
        return
    try:
        merge(ts, pull(ts))
    except Exception as e:                      # noqa: BLE001 -- report, never dump headers
        print(f"  {ts} pull failed: {type(e).__name__}: {str(e)[-200:]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
