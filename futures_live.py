"""
futures_live.py -- LIVE CME futures aggressor delta for the flow viewer's TIDE face (user, 2026-11-19). Feeds nothing.

The flow viewer's lower pane on its TIDE face is split in two: UW Market Tide on top, and under it the futures
per-minute aggressor delta (bars) with the running CVD (line) for the product behind the ticker on screen:
SPY -> ES, QQQ -> NQ, IWM -> RTY (any other ticker shows ES, labelled).

HOW. UW REST /api/futures/flow (Advanced tier), the same endpoint uw_futures_tape.py collects nightly. Every ~2.5 s
per product: the prints newer than (last print seen - 10 s), de-duplicated by trade_id -- the overlap catches prints
UW publishes a little late; a full page (500) is followed back with older_than until it meets the cursor, so a
burst is never cut. On a start mid-session the day is backfilled from 09:30 first (a few minutes for ES).
Rules match the silver tape: block prints excluded; side buy / sell = aggressor; the FRONT contract = the one with
the most volume so far today; regular session 09:30-16:00 ET.

    live/futures_cvd.json   {ts, day, products: {ES: {sym, rows: [[stamp, buy, sell, delta, cvd, prints]], lag, last}}}
                            stamps = ET wall clock as UTC, minute start (the chart's convention, live_state._et_stamp);
                            lag = seconds from a print's executed_at to its arrival here (median / p90, last 5 min)

DISPLAY ONLY. Untested as a signal: earlier delta work found futures / stock / options alignment mostly AFTER the move
(cleanbot QQQ sub-minute, 11-07). The pre-registered futures-CVD test at THRU breaches / PULLBACK setups decides.

    python futures_live.py          (service cleanbot-futlive: idles outside 09:25-16:05 ET on trading days)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
import sys
import threading
import time
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
BASE = "https://api.unusualwhales.com/api"
OUT = os.path.join(ROOT, "live", "futures_cvd.json")
NY = ZoneInfo("America/New_York")
PRODUCTS = ("ES", "NQ", "RTY")
ON, OFF = 9 * 60 + 25, 16 * 60 + 5
RTH0, RTH1 = 9 * 60 + 30, 16 * 60
POLL, OVERLAP = 2.5, 10.0


def iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


class Feed:
    """one product's prints for one session -> per-minute, per-contract buy / sell / prints."""

    def __init__(self, product, day):
        self.p, self.day = product, day
        self.min = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))     # stamp -> sym -> [buy, sell, prints]
        self.vol = defaultdict(int)                                         # sym -> volume today (front pick)
        self.seen = {}                                                      # trade_id -> executed epoch (pruned)
        self.cursor = None                                                  # newest executed_at seen
        self.lag = deque(maxlen=4000)                                       # (arrival epoch, lag s)
        self.lock = threading.Lock()

    def add(self, rows, now):
        for x in rows:
            tid = x.get("trade_id")
            if tid in self.seen:
                continue
            try:
                ex = parse(x["executed_at"])
            except (KeyError, ValueError):
                continue
            self.seen[tid] = ex.timestamp()
            if self.cursor is None or ex > self.cursor:
                self.cursor = ex
            et = ex.astimezone(NY)
            mod = et.hour * 60 + et.minute
            if not (RTH0 <= mod < RTH1) or x.get("is_block") is True:
                continue
            size = int(x.get("size") or 0)
            sym = x.get("sym") or "?"
            stamp = int(dt.datetime(et.year, et.month, et.day, et.hour, et.minute, tzinfo=dt.timezone.utc).timestamp())
            with self.lock:
                row = self.min[stamp][sym]
                side = x.get("side")
                if side == "buy":
                    row[0] += size
                elif side == "sell":
                    row[1] += size
                row[2] += 1
                self.vol[sym] += size
                self.lag.append((now, now - ex.timestamp()))
        if len(self.seen) > 50000 and self.cursor is not None:              # keep only ids inside the overlap window
            edge = self.cursor.timestamp() - 5 * OVERLAP
            self.seen = {k: v for k, v in self.seen.items() if v >= edge}

    def frame(self, now):
        with self.lock:
            if not self.vol:
                return None
            front = max(self.vol, key=self.vol.get)
            rows, cvd = [], 0
            for stamp in sorted(self.min):
                b, s, n = self.min[stamp].get(front, [0, 0, 0])
                cvd += b - s
                rows.append([stamp, b, s, b - s, cvd, n])
            recent = [l for (a, l) in self.lag if now - a <= 300]
        lag = {"median": round(st.median(recent), 2), "p90": round(sorted(recent)[int(0.9 * (len(recent) - 1))], 2),
               "n": len(recent)} if recent else None
        return {"sym": front, "rows": rows, "lag": lag, "last": iso(self.cursor) if self.cursor else None}


class Poller:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {os.getenv('UW_API_KEY', '').strip()}",
                               "Accept": "application/json", "User-Agent": "cleanbot-futures-live/1.0"})
        self.calls = 0

    def page(self, product, newer, older=None):
        params = {"products": product, "limit": 500, "newer_than": iso(newer), "include_spreads": "false"}
        if older is not None:
            params["older_than"] = iso(older)
        for attempt in range(3):
            try:
                self.calls += 1
                r = self.s.get(f"{BASE}/futures/flow", params=params, timeout=20)
                if r.status_code == 200:
                    return r.json().get("data") or []
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1 + 2 * attempt); continue
                print(f"futures live {product}: HTTP {r.status_code}", flush=True)
                return []
            except requests.RequestException as e:
                print(f"futures live {product}: {type(e).__name__}", flush=True)
                time.sleep(1 + 2 * attempt)
        return []

    def since(self, product, lo, hi=None):
        """every print newer than lo (and older than hi), whatever order UW returns a page in: a full page is
        followed back with older_than = its oldest print until a short page or the floor."""
        out, older = [], hi
        while True:
            pg = self.page(product, lo, older)
            out += pg
            if len(pg) < 500:
                return out
            oldest = min(parse(x["executed_at"]) for x in pg)
            if older is not None and oldest >= older:
                return out                                  # no progress: stop rather than loop
            older = oldest + dt.timedelta(microseconds=1)
            if older <= lo:
                return out


def write(feeds, day):
    now = time.time()
    doc = {"ts": now, "day": day.isoformat(), "products": {}}
    for p, f in feeds.items():
        fr = f.frame(now)
        if fr:
            doc["products"][p] = fr
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(OUT + ".tmp", OUT)


def run_session(day):
    P = Poller()
    feeds = {p: Feed(p, day) for p in PRODUCTS}
    open_t = dt.datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
    now = dt.datetime.now(NY)
    if now > open_t:                                        # started mid-session: backfill from the open first
        t0 = time.time()
        def back(p):
            feeds[p].add(P.since(p, open_t - dt.timedelta(seconds=1), now), time.time())
        th = [threading.Thread(target=back, args=(p,)) for p in PRODUCTS]
        for t in th: t.start()
        for t in th: t.join()
        # backfilled prints are not live arrivals: their lags would swamp the delay reading
        for f in feeds.values(): f.lag.clear()
        print(f"futures live {day}: backfilled from 09:30 in {time.time() - t0:.0f}s, {P.calls} calls", flush=True)
    while True:
        now = dt.datetime.now(NY)
        if now.date() != day or now.hour * 60 + now.minute >= OFF:
            break
        t0 = time.time()
        for p, f in feeds.items():
            lo = (f.cursor - dt.timedelta(seconds=OVERLAP)) if f.cursor else open_t - dt.timedelta(minutes=5)
            f.add(P.since(p, lo), time.time())
        write(feeds, day)
        time.sleep(max(0.2, POLL - (time.time() - t0)))
    write(feeds, day)
    print(f"futures live {day}: done, {P.calls} calls, " +
          ", ".join(f"{p} {sum(f.vol.values()):,}" for p, f in feeds.items()), flush=True)


def main():
    import market_calendar as MC
    while True:
        now = dt.datetime.now(NY)
        mod = now.hour * 60 + now.minute
        if MC.is_trading_day(now.date()) and ON <= mod < OFF:
            try:
                run_session(now.date())
            except Exception as e:                   # noqa: BLE001 -- display only: never crash-loop the box
                print(f"futures live failed: {type(e).__name__} {str(e)[:160]}", flush=True)
                time.sleep(30)
        time.sleep(20)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":       # one write for a given day (render checks)
        d = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.datetime.now(NY).date()
        P = Poller()
        feeds = {p: Feed(p, d) for p in PRODUCTS}
        lo = dt.datetime(d.year, d.month, d.day, 9, 29, 59, tzinfo=NY)
        hi = min(dt.datetime.now(NY), dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY))
        for p in PRODUCTS:
            feeds[p].add(P.since(p, lo, hi), time.time())
            feeds[p].lag.clear()                            # a backfill has no arrival delay to report
        write(feeds, d)
        print(f"wrote {OUT}: {P.calls} calls")
    else:
        main()
