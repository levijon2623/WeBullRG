"""
uw_ws_shadow.py -- SHADOW LOGGER for UW's websocket (user, 2026-11-19, after the Advanced upgrade). Feeds nothing.

Before the bot's flow trigger or the strike cue move from REST to the websocket, the two must agree. This connects
to wss://api.unusualwhales.com/socket (the bot does not: UW_WEBSOCKET=false in .env) and records, 09:25-16:05 ET:
  live/ws_shadow/DATE/net_flow.jsonl     every net_flow tick for the bot's rule tickers (receive time, payload)
  live/ws_shadow/DATE/opt_minutes.json   SPY / QQQ / IWM same-day option trades summed per contract per minute
                                         (ask / bid / other volume by UW's side tags, total) -- what strike_cue builds
                                         from REST option-contracts polls
  live/ws_shadow/DATE/latency.json       receive time - executed time, per minute (median / p90)
  live/ws_shadow/DATE/samples.jsonl      the first 30 raw messages of each channel (schema record)
  live/ws_shadow/DATE/latency_tk.json    per ticker per minute [trades, late > 5 s, worst lag s] (added 2026-11-19)
  live/ws_shadow/DATE/late_samples.jsonl up to 200 raw option trades that arrived > 30 s late
  live/ws_shadow/DATE/gaps.json          (2026-11-20) every reconnect gap, the minutes refilled from REST, and the
                                         duplicate counts per channel
  live/ws_shadow/DATE/net_flow_patch.json  REST net-prem-ticks minutes (net call - net put) for the gap minutes; a
                                         consumer replaces its WS minute sums with these
compare_ws_rest.py checks them against REST after the close.

2026-11-20 additions (after reading UW's own websocket examples, github unusual-whales/api-examples):
  DEDUPE   UW can resend data after a re-join. option_trades are de-duplicated by `id`, net_flow ticks by (ticker,
           time) -- over a rolling 15-minute window (memory stays bounded) -- and the duplicates are counted.
  GAP FILL a dropped connection loses everything until the re-join (11-19: 21 s and 18 s, the only real differences
           from REST). After each re-join, once the gap's last minute has closed + 90 s, the gap minutes are refilled
           from REST: net flow from /stock/T/net-prem-ticks, and every same-day SPY / QQQ / IWM contract's ask / bid /
           other volume from /option-contract/OCC/intraday (exact per minute; ~1,000 calls a gap).
  BACKOFF  reconnect after 5 s, doubling to 60 s (+ up to 1 s jitter) while connections keep dying within a minute.

    python uw_ws_shadow.py          (service: idles outside 09:25-16:05 ET on trading days)
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                    # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
NY = ZoneInfo("America/New_York")
OUT = os.path.join(ROOT, "live", "ws_shadow")
OPT_TK = ("SPY", "QQQ", "IWM")
ON, OFF = 9 * 60 + 25, 16 * 60 + 5


def rule_tickers():
    try:
        from config import RULES
        return sorted({r["ticker"] for r in RULES if r.get("enabled", True)} | set(OPT_TK))
    except Exception:
        return list(OPT_TK)


class Shadow:
    def __init__(self, day):
        self.day = day
        self.dir = os.path.join(OUT, day.isoformat())
        os.makedirs(self.dir, exist_ok=True)
        self.lock = threading.Lock()
        self.opt = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))   # minute -> occ -> [ask, bid, other, total]
        self.lat = defaultdict(list)
        # per ticker per minute: [trades, late > 5 s, worst lag] + up to 200 raw late trades (2026-11-19: a subset
        # arrived up to 11 min late 14:48-15:09 while the median stayed 0.09 s -- which ticker / kind was unknown)
        self.late = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
        self.late_n = 0
        self.lsm = open(os.path.join(self.dir, "late_samples.jsonl"), "a")
        self.samples = defaultdict(int)
        self.nf = open(os.path.join(self.dir, "net_flow.jsonl"), "a")
        self.smp = open(os.path.join(self.dir, "samples.jsonl"), "a")
        self.msgs = 0
        self.seen_opt, self.seen_nf = {}, {}                # id / (ticker, time) -> receive time, pruned at 15 min
        self.dups = defaultdict(int)
        self.last_rx = None
        self.gaps = []                                      # [{lost, rejoined, minutes, patched_flow, patched_opt}]
        self.nf_patch = defaultdict(dict)                   # ticker -> minute -> REST net premium

    def on_message(self, ws, message):
        now = time.time()
        self.last_rx = now
        try:
            payload = json.loads(message)
        except ValueError:
            return
        if not (isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict)):
            return
        ch, d = str(payload[0]), payload[1]
        self.msgs += 1
        base = ch.split(":")[0]
        if self.samples[base] < 30:
            self.samples[base] += 1
            self.smp.write(json.dumps({"rx": now, "ch": ch, "data": d}) + "\n"); self.smp.flush()
        if base == "net_flow":
            if "time" in d:
                k = (d.get("ticker"), d.get("time"))
                if k in self.seen_nf:
                    self.dups["net_flow"] += 1
                    return
                self.seen_nf[k] = now
            self.nf.write(json.dumps({"rx": now, "ch": ch, "data": d}) + "\n")
        elif base == "option_trades":
            tid = d.get("id")
            if tid is not None:
                if tid in self.seen_opt:
                    self.dups["option_trades"] += 1
                    return
                self.seen_opt[tid] = now
            occ = d.get("option_symbol") or d.get("option_chain_id") or d.get("symbol")
            exp = str(d.get("expiry") or "")[:10]
            if not occ or exp != self.day.isoformat():
                return
            t = d.get("executed_at") or d.get("timestamp")
            try:
                ex = datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp() if not isinstance(t, (int, float)) else (t / 1000 if t > 1e12 else t)
            except ValueError:
                ex = now
            m = datetime.fromtimestamp(ex, NY)
            mod = m.hour * 60 + m.minute
            tags = d.get("tags") or []
            size = float(d.get("size") or 0)
            k = 0 if "ask_side" in tags else (1 if "bid_side" in tags else 2)
            with self.lock:
                row = self.opt[mod][occ]
                row[k] += size; row[3] += size
                self.lat[mod].append(now - ex)
                lt = self.late[str(d.get("underlying_symbol") or ch.split(":")[-1])][mod]
                lt[0] += 1
                if now - ex > 5:
                    lt[1] += 1
                    lt[2] = max(lt[2], round(now - ex, 1))
                    if now - ex > 30 and self.late_n < 200:
                        self.late_n += 1
                        self.lsm.write(json.dumps({"rx": now, "lag": round(now - ex, 1), "data": d}) + "\n"); self.lsm.flush()

    def prune(self):
        edge = time.time() - 900
        for name in ("seen_opt", "seen_nf"):
            cur = getattr(self, name)
            if len(cur) > 20000:
                setattr(self, name, {k: v for k, v in list(cur.items()) if v >= edge})

    def backfill(self, lost, rejoined):
        """refill the gap minutes from REST once the last one has closed (+ 90 s for REST to catch up)."""
        import positioning_map as P
        from concurrent.futures import ThreadPoolExecutor
        m0 = datetime.fromtimestamp(lost, NY); m1 = datetime.fromtimestamp(rejoined, NY)
        mins = list(range(m0.hour * 60 + m0.minute, m1.hour * 60 + m1.minute + 1))
        wait = (datetime(m1.year, m1.month, m1.day, m1.hour, m1.minute, tzinfo=NY).timestamp() + 60 + 90) - time.time()
        if wait > 0:
            time.sleep(wait)
        g = {"lost": round(lost, 1), "rejoined": round(rejoined, 1), "minutes": mins, "patched_flow": 0, "patched_opt": 0}
        try:
            uw = P.UW()
            for tk in rule_tickers():
                rows = uw.get(f"/stock/{tk}/net-prem-ticks", date=self.day.isoformat()).get("data") or []
                for x in rows:
                    t = datetime.fromisoformat(str(x["tape_time"]).replace("Z", "+00:00")).astimezone(NY)
                    m = t.hour * 60 + t.minute
                    if t.date() == self.day and m in mins:
                        self.nf_patch[tk][m] = (P.num(x.get("net_call_premium")) or 0.0) - (P.num(x.get("net_put_premium")) or 0.0)
                        g["patched_flow"] += 1
            occs = set()
            with self.lock:
                for c in self.opt.values():
                    occs |= set(c)
            for tk in OPT_TK:
                rows = uw.get(f"/stock/{tk}/option-contracts", expiry=self.day.isoformat()).get("data") or []
                occs |= {x["option_symbol"] for x in rows if x.get("option_symbol") and (P.num(x.get("volume")) or 0) > 0}

            def one(occ):
                return occ, uw.get(f"/option-contract/{occ}/intraday", date=self.day.isoformat()).get("data") or []
            with ThreadPoolExecutor(16) as ex:
                got = list(ex.map(one, sorted(occs)))
            with self.lock:
                for m in mins:                              # the gap minutes are REPLACED, not added to
                    for occ in list(self.opt.get(m, {})):
                        self.opt[m].pop(occ, None)
                for occ, rows in got:
                    for r in rows:
                        t = datetime.fromisoformat(str(r["start_time"]).replace("Z", "+00:00")).astimezone(NY)
                        m = t.hour * 60 + t.minute
                        if m not in mins:
                            continue
                        a, b = P.num(r.get("volume_ask_side")) or 0.0, P.num(r.get("volume_bid_side")) or 0.0
                        o = (P.num(r.get("volume_mid_side")) or 0.0) + (P.num(r.get("volume_no_side")) or 0.0)
                        if a + b + o > 0:
                            self.opt[m][occ] = [a, b, o, a + b + o]
                            g["patched_opt"] += 1
            g["calls"] = uw.calls
        except Exception as e:                      # noqa: BLE001 -- a shadow must never die over a backfill
            g["error"] = f"{type(e).__name__} {str(e)[:120]}"
        self.gaps.append(g)
        print(f"ws shadow gap {m0:%H:%M:%S}-{m1:%H:%M:%S}: refilled {g['patched_flow']} flow minutes, "
              f"{g['patched_opt']} contract-minutes{' ' + g['error'] if 'error' in g else ''}", flush=True)

    def flush(self):
        self.prune()
        with self.lock:
            opt = {str(m): {o: v for o, v in c.items()} for m, c in self.opt.items()}
            gaps = {"gaps": list(self.gaps), "duplicates": dict(self.dups)}
            nfp = {tk: {str(m): v for m, v in mm.items()} for tk, mm in self.nf_patch.items()}
        for name, obj in (("gaps.json", gaps), ("net_flow_patch.json", nfp)):
            p = os.path.join(self.dir, name)
            json.dump(obj, open(p + ".tmp", "w"))
            os.replace(p + ".tmp", p)
        with self.lock:
            lat = {str(m): {"n": len(v), "median": round(st.median(v), 3), "p90": round(sorted(v)[int(0.9 * (len(v) - 1))], 3)}
                   for m, v in self.lat.items() if v}
            ltk = {tk: {str(m): list(v) for m, v in mm.items()} for tk, mm in self.late.items()}
        for name, obj in (("opt_minutes.json", opt), ("latency.json", lat), ("latency_tk.json", ltk)):
            p = os.path.join(self.dir, name)
            json.dump(obj, open(p + ".tmp", "w"))
            os.replace(p + ".tmp", p)
        self.nf.flush()


def run_session(day):
    import websocket
    sh = Shadow(day)
    key = os.getenv("UW_API_KEY", "").strip()
    tks = rule_tickers()

    state = {"joined_before": False}

    def on_open(ws):
        for t in tks:
            ws.send(json.dumps({"channel": f"net_flow:{t}", "msg_type": "join"}))
        for t in OPT_TK:
            ws.send(json.dumps({"channel": f"option_trades:{t}", "msg_type": "join"}))
        print(f"ws shadow {day}: joined net_flow x{len(tks)}, option_trades x{len(OPT_TK)}", flush=True)
        # a RE-join: everything since the last message received is a gap -> refill it from REST in the background
        if state["joined_before"] and sh.last_rx is not None:
            th = threading.Thread(target=sh.backfill, args=(sh.last_rx, time.time()), daemon=True)
            th.start()
            state.setdefault("fills", []).append(th)
        state["joined_before"] = True

    def on_error(ws, e):
        print(f"ws shadow error: {type(e).__name__} {str(e)[:120]}", flush=True)

    attempt = 0
    while True:
        now = datetime.now(NY)
        if now.date() != day or now.hour * 60 + now.minute >= OFF:
            break
        ws = websocket.WebSocketApp(f"wss://api.unusualwhales.com/socket?token={key}", on_open=on_open,
                                    on_message=sh.on_message, on_error=on_error)
        th = threading.Thread(target=lambda: ws.run_forever(ping_interval=20, ping_timeout=10), daemon=True)
        started = time.time()
        th.start()
        while th.is_alive():
            time.sleep(15)
            sh.flush()
            now = datetime.now(NY)
            if now.date() != day or now.hour * 60 + now.minute >= OFF:
                ws.close()
                break
        # BACKOFF: 5 s after a connection that lived, doubling to 60 s while they keep dying within a minute
        attempt = 0 if time.time() - started >= 60 else attempt + 1
        time.sleep(min(60, 5 * 2 ** attempt) + random.uniform(0, 1))
    for th in state.get("fills", []):                      # a gap near the close: let its refill land first
        th.join(timeout=300)
    sh.flush()
    print(f"ws shadow {day}: done, {sh.msgs} messages", flush=True)


def main():
    import market_calendar as MC
    while True:
        now = datetime.now(NY)
        mod = now.hour * 60 + now.minute
        if MC.is_trading_day(now.date()) and ON <= mod < OFF:
            try:
                run_session(now.date())
            except Exception as e:                   # noqa: BLE001 -- a shadow must never crash-loop the box
                print(f"ws shadow failed: {type(e).__name__} {str(e)[:160]}", flush=True)
                time.sleep(60)
        time.sleep(30)


if __name__ == "__main__":
    main()
