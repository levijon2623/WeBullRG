"""
uw_ws_greeks.py -- COLLECTOR for UW's greek-exposure websocket channels (user, 2026-11-20). Feeds nothing.

Three channels, found in UW's own examples (github unusual-whales/api-examples), collected from the first session so
the history starts now (none of them has history we hold):
  gex:T               SPY / QQQ / IWM whole-chain exposure, ~every 2 s: gamma / charm / vanna / delta per 1% move,
                      by OI, by intraday volume and "dir" (directional), + price. Streaming twin of REST
                      /stock/T/spot-exposures (= lake spot-exposures-1m; c_gex / g_total in the research).
  gex_strike_expiry:T SPY / QQQ / IWM by strike AND expiry: call / put delta, gamma, charm, vanna by OI and by volume,
                      volume split ask / bid. Streaming twin of REST /spot-exposures/expiry-strike, which live_state
                      pages every 5 min (~15 calls a ticker) for the viewer's GEX heat. KEPT: expiries <= 7 calendar
                      days, strikes within +-5% of the row's price; the latest row per (ticker, expiry, strike) is
                      snapshotted once a minute.
  periscope           SPX / XSP / VIX / NANOS (no ticker suffix; one join gets all) net dealer gamma / charm / vanna by
                      strike and expiry, every 60 s, delivered ~71 s after its timestamp, split into 1,024-row chunks
                      (has_more). UW: derived from positions it may not publish as contract counts -- i.e. not a model
                      convention. KEPT: SPX and VIX; expiries <= 35 days; non-zero rows; SPX strikes within +-10% of
                      10 x SPY (the bot's live price; periscope rows carry no price).
Window 09:25-16:05 ET on trading days. Receive loop does parsing and dict updates only; disk writes happen every
10 minutes from the main thread (UW's examples: slow per-message work makes the socket fall behind).

    lake/bronze/uw-ws-greeks/date=D/gex/part-HHMM.parquet            every gex:T message
    lake/bronze/uw-ws-greeks/date=D/strike_expiry/part-HHMM.parquet  minute snapshots (minute = ET wall clock)
    lake/bronze/uw-ws-greeks/date=D/periscope/part-HHMM.parquet      completed snapshots (complete = all chunks)
    live/ws_greeks/D/stats.json      messages, rows kept, update cadence, delivery lag per channel, gaps, duplicates
    live/ws_greeks/D/samples.jsonl   the first 20 raw messages of each channel (rows truncated to 5)

Reconnects: backoff 5 s doubling to 60 s (+ jitter); periscope / gex de-duplicated by (ticker, timestamp) -- UW
restarts a periscope snapshot after a re-join. Gaps are logged, not refilled (REST twins exist for gex and
strike_expiry; periscope has none we know of).

    python uw_ws_greeks.py                       (service cleanbot-wsgreeks)
    python uw_ws_greeks.py --minutes 3 --test    (a short run now, outside the window, into the scratch dirs given)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                    # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
NY = ZoneInfo("America/New_York")
LAKE = os.path.join(ROOT, "lake", "bronze", "uw-ws-greeks")
LIVE = os.path.join(ROOT, "live", "ws_greeks")
TK = ("SPY", "QQQ", "IWM")
ON, OFF = 9 * 60 + 25, 16 * 60 + 5
SE_DAYS, SE_BAND = 7, 0.05
PS_DAYS, PS_BAND, PS_KEEP = 35, 0.10, ("SPX", "VIX")
FLUSH_EVERY = 600
PS_TIMEOUT = 180

GEX_F = ["gamma_per_one_percent_move_oi", "delta_per_one_percent_move_oi", "charm_per_one_percent_move_oi",
         "vanna_per_one_percent_move_oi", "gamma_per_one_percent_move_vol", "delta_per_one_percent_move_vol",
         "charm_per_one_percent_move_vol", "vanna_per_one_percent_move_vol", "gamma_per_one_percent_move_dir",
         "charm_per_one_percent_move_dir", "vanna_per_one_percent_move_dir", "price"]
SE_F = ["price", "call_delta_oi", "put_delta_oi", "call_gamma_oi", "put_gamma_oi", "call_charm_oi", "put_charm_oi",
        "call_vanna_oi", "put_vanna_oi", "call_delta_vol", "put_delta_vol", "call_gamma_vol", "put_gamma_vol",
        "call_charm_vol", "put_charm_vol", "call_vanna_vol", "put_vanna_vol", "call_gamma_ask_vol", "call_gamma_bid_vol",
        "put_gamma_ask_vol", "put_gamma_bid_vol", "call_charm_ask_vol", "call_charm_bid_vol", "put_charm_ask_vol",
        "put_charm_bid_vol", "call_vanna_ask_vol", "call_vanna_bid_vol", "put_vanna_ask_vol", "put_vanna_bid_vol"]


def num(v):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


def spy_price():
    try:
        s = json.load(open(os.path.join(ROOT, "live", "state.json")))
        t = (s.get("tickers") or {}).get("SPY") or {}
        pl_ = t.get("px_live") or ((t.get("px") or [None])[-1])
        return float(pl_[1]) if pl_ else None
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return None


class Collector:
    def __init__(self, day, lake=LAKE, live=LIVE):
        self.day = day
        self.lake = os.path.join(lake, f"date={day.isoformat()}")
        self.live = os.path.join(live, day.isoformat())
        os.makedirs(self.live, exist_ok=True)
        self.lock = threading.Lock()
        self.gex_rows, self.gex_seen = [], set()
        self.se_latest = {}                              # (tk, expiry, strike) -> (ts_ms, rx, values)
        self.se_rows = []
        self.ps_pending, self.ps_rows, self.ps_seen = {}, [], set()
        self.stats = defaultdict(lambda: defaultdict(float))
        self.lag = defaultdict(list)                     # channel -> rx - timestamp (s), sampled
        self.cadence = defaultdict(list)                 # gex: gaps between a ticker's messages (s)
        self.last_ts = {}
        self.samples = defaultdict(int)
        self.smp = open(os.path.join(self.live, "samples.jsonl"), "a")
        self.gaps = []
        self.last_rx = None
        self.spx_ref = None
        self.parts = 0

    # ---------------------------------------------------------------- receive (light work only)
    def on_message(self, ws, message):
        now = time.time()
        self.last_rx = now
        try:
            p = json.loads(message)
        except ValueError:
            return
        if not (isinstance(p, list) and len(p) >= 2 and isinstance(p[1], dict)):
            return
        ch, d = str(p[0]), p[1]
        base = ch.split(":")[0]
        S = self.stats[base]
        S["messages"] += 1
        if self.samples[base] < 20:
            self.samples[base] += 1
            dd = dict(d)
            if isinstance(dd.get("rows"), list):
                dd["rows"] = dd["rows"][:5]
            self.smp.write(json.dumps({"rx": now, "ch": ch, "data": dd}) + "\n"); self.smp.flush()
        if "timestamp" not in d:
            return                                       # join acks
        ts = int(d["timestamp"])
        if S["messages"] % 50 == 1:
            self.lag[base].append(now - ts / 1000)
        with self.lock:
            if base == "gex":
                k = (d.get("ticker"), ts)
                if k in self.gex_seen:
                    S["duplicates"] += 1; return
                self.gex_seen.add(k)
                prev = self.last_ts.get(k[0])
                if prev:
                    self.cadence[k[0]].append((ts - prev) / 1000)
                self.last_ts[k[0]] = ts
                self.gex_rows.append([k[0], ts, now] + [num(d.get(f)) for f in GEX_F])
            elif base == "gex_strike_expiry":
                px, k_ = num(d.get("price")), num(d.get("strike"))
                e = str(d.get("expiry") or "")[:10]
                try:
                    dte = (date.fromisoformat(e) - self.day).days
                except ValueError:
                    return
                if px is None or k_ is None or not (0 <= dte <= SE_DAYS) or abs(k_ / px - 1) > SE_BAND:
                    S["dropped"] += 1; return
                self.se_latest[(d.get("ticker"), e, k_)] = (ts, now, [num(d.get(f)) for f in SE_F])
                S["kept"] += 1
            elif base == "periscope":
                tk = d.get("ticker")
                if tk not in PS_KEEP:
                    return
                key = (tk, ts)
                if key in self.ps_seen:
                    S["duplicates"] += 1; return
                ent = self.ps_pending.setdefault(key, {"rows": [], "chunks": 0, "first": now, "total": d.get("total_rows")})
                ent["chunks"] += 1
                ref = self.spx_ref
                for r in d.get("rows") or []:
                    g, c, v, k_ = num(r.get("gamma")), num(r.get("charm")), num(r.get("vanna")), num(r.get("strike"))
                    e = str(r.get("expiry") or "")[:10]
                    S["rows_seen"] += 1
                    if not (g or c or v) or k_ is None:
                        continue
                    try:
                        if (date.fromisoformat(e) - self.day).days > PS_DAYS:
                            continue
                    except ValueError:
                        continue
                    if tk == "SPX" and ref and abs(k_ / ref - 1) > PS_BAND:
                        continue
                    ent["rows"].append([tk, ts, e, k_, g, c, v])
                if not d.get("has_more"):
                    self._finish_ps(key, complete=True)

    def _finish_ps(self, key, complete):
        ent = self.ps_pending.pop(key, None)
        if ent is None:
            return
        self.ps_seen.add(key)
        S = self.stats["periscope"]
        S["snapshots"] += 1
        S["incomplete"] += 0 if complete else 1
        rx = time.time()
        self.ps_rows += [r + [rx, complete] for r in ent["rows"]]
        S["rows_kept"] += len(ent["rows"])
        self.lag["periscope_done"].append(rx - key[1] / 1000)

    # ---------------------------------------------------------------- main-thread work
    def minute_snapshot(self, stamp):
        """the latest strike-expiry row of every kept (ticker, expiry, strike), stamped with the ET minute."""
        with self.lock:
            snap = [[tk, e, k, stamp, ts, rx] + vals for (tk, e, k), (ts, rx, vals) in self.se_latest.items()]
            self.se_rows += snap
            for key in [k for k, v in self.ps_pending.items() if time.time() - v["first"] > PS_TIMEOUT]:
                self._finish_ps(key, complete=False)
        self.stats["gex_strike_expiry"]["snapshot_rows_last"] = len(snap)

    def write(self, final=False):
        import polars as pl
        with self.lock:
            g, self.gex_rows = self.gex_rows, []
            s, self.se_rows = self.se_rows, []
            p, self.ps_rows = self.ps_rows, []
        tag = datetime.now(NY).strftime("%H%M%S")
        out = []
        if g:
            out.append(("gex", pl.DataFrame(g, schema=["ticker", "ts_ms", "rx"] + GEX_F, orient="row")))
        if s:
            out.append(("strike_expiry", pl.DataFrame(s, schema=["ticker", "expiry", "strike", "minute", "ts_ms", "rx"] + SE_F, orient="row")))
        if p:
            out.append(("periscope", pl.DataFrame(p, schema=["ticker", "ts_ms", "expiry", "strike", "gamma", "charm", "vanna", "rx", "complete"], orient="row")))
        for name, df in out:
            d = os.path.join(self.lake, name)
            os.makedirs(d, exist_ok=True)
            f = os.path.join(d, f"part-{tag}.parquet")
            df.write_parquet(f + ".tmp", compression="zstd")
            os.replace(f + ".tmp", f)
            self.parts += 1
        self.write_stats()

    def write_stats(self):
        def q(v):
            v = sorted(v)
            return {"n": len(v), "median": round(st.median(v), 3), "p90": round(v[int(0.9 * (len(v) - 1))], 3)} if v else None
        doc = {"updated": time.time(), "channels": {k: dict(v) for k, v in self.stats.items()},
               "lag_s": {k: q(v) for k, v in self.lag.items()}, "gex_cadence_s": {k: q(v) for k, v in self.cadence.items()},
               "strike_expiry_keys": len(self.se_latest), "gaps": self.gaps, "parts_written": self.parts}
        p = os.path.join(self.live, "stats.json")
        json.dump(doc, open(p + ".tmp", "w"), indent=1)
        os.replace(p + ".tmp", p)


def run(day, until, lake=LAKE, live=LIVE):
    import websocket
    c = Collector(day, lake, live)
    key = os.getenv("UW_API_KEY", "").strip()
    state = {"joined": False}

    def on_open(ws):
        for t in TK:
            ws.send(json.dumps({"channel": f"gex:{t}", "msg_type": "join"}))
            ws.send(json.dumps({"channel": f"gex_strike_expiry:{t}", "msg_type": "join"}))
        ws.send(json.dumps({"channel": "periscope", "msg_type": "join"}))
        if state["joined"] and c.last_rx:
            c.gaps.append([round(c.last_rx, 1), round(time.time(), 1)])
        state["joined"] = True
        print(f"ws greeks {day}: joined gex x3, gex_strike_expiry x3, periscope", flush=True)

    def on_error(ws, e):
        print(f"ws greeks error: {type(e).__name__} {str(e)[:120]}", flush=True)

    attempt, last_write, last_min = 0, time.time(), None
    while time.time() < until():
        px = spy_price()
        c.spx_ref = px * 10 if px else c.spx_ref
        ws = websocket.WebSocketApp(f"wss://api.unusualwhales.com/socket?token={key}", on_open=on_open,
                                    on_message=c.on_message, on_error=on_error)
        th = threading.Thread(target=lambda: ws.run_forever(ping_interval=20, ping_timeout=10), daemon=True)
        started = time.time()
        th.start()
        while th.is_alive():
            time.sleep(1)
            now = datetime.now(NY)
            stamp = now.hour * 60 + now.minute
            if stamp != last_min and now.second >= 1:   # once a minute, just after the minute turns
                last_min = stamp
                c.minute_snapshot(stamp)
                px = spy_price()
                if px:
                    c.spx_ref = px * 10
            if time.time() - last_write >= FLUSH_EVERY:
                last_write = time.time()
                c.write()
            if time.time() >= until():
                ws.close()
                break
        attempt = 0 if time.time() - started >= 60 else attempt + 1
        if time.time() < until():
            time.sleep(min(60, 5 * 2 ** attempt) + random.uniform(0, 1))
    c.minute_snapshot(datetime.now(NY).hour * 60 + datetime.now(NY).minute)
    c.write(final=True)
    print(f"ws greeks {day}: done -- " + ", ".join(f"{k} {int(v['messages'])} msgs" for k, v in c.stats.items()), flush=True)
    return c


def main():
    import market_calendar as MC
    while True:
        now = datetime.now(NY)
        mod = now.hour * 60 + now.minute
        if MC.is_trading_day(now.date()) and ON <= mod < OFF:
            d = now.date()
            end = datetime(d.year, d.month, d.day, OFF // 60, OFF % 60, tzinfo=NY).timestamp()
            try:
                run(d, lambda: end)
            except Exception as e:                       # noqa: BLE001 -- a collector must never crash-loop the box
                print(f"ws greeks failed: {type(e).__name__} {str(e)[:160]}", flush=True)
                time.sleep(60)
        time.sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--test", nargs=2, metavar=("LAKE", "LIVE"))
    a = ap.parse_args()
    if a.minutes:
        end = time.time() + a.minutes * 60
        lake, live = (a.test if a.test else (LAKE, LIVE))
        FLUSH_EVERY = max(30, a.minutes * 60 / 3)
        run(datetime.now(NY).date(), lambda: end, lake, live)
    else:
        main()
