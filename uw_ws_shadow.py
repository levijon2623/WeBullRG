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
compare_ws_rest.py checks them against REST after the close.

    python uw_ws_shadow.py          (service: idles outside 09:25-16:05 ET on trading days)
"""
from __future__ import annotations

import json
import os
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
        self.samples = defaultdict(int)
        self.nf = open(os.path.join(self.dir, "net_flow.jsonl"), "a")
        self.smp = open(os.path.join(self.dir, "samples.jsonl"), "a")
        self.msgs = 0

    def on_message(self, ws, message):
        now = time.time()
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
            self.nf.write(json.dumps({"rx": now, "ch": ch, "data": d}) + "\n")
        elif base == "option_trades":
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

    def flush(self):
        with self.lock:
            opt = {str(m): {o: v for o, v in c.items()} for m, c in self.opt.items()}
            lat = {str(m): {"n": len(v), "median": round(st.median(v), 3), "p90": round(sorted(v)[int(0.9 * (len(v) - 1))], 3)}
                   for m, v in self.lat.items() if v}
        for name, obj in (("opt_minutes.json", opt), ("latency.json", lat)):
            p = os.path.join(self.dir, name)
            json.dump(obj, open(p + ".tmp", "w"))
            os.replace(p + ".tmp", p)
        self.nf.flush()


def run_session(day):
    import websocket
    sh = Shadow(day)
    key = os.getenv("UW_API_KEY", "").strip()
    tks = rule_tickers()

    def on_open(ws):
        for t in tks:
            ws.send(json.dumps({"channel": f"net_flow:{t}", "msg_type": "join"}))
        for t in OPT_TK:
            ws.send(json.dumps({"channel": f"option_trades:{t}", "msg_type": "join"}))
        print(f"ws shadow {day}: joined net_flow x{len(tks)}, option_trades x{len(OPT_TK)}", flush=True)

    def on_error(ws, e):
        print(f"ws shadow error: {type(e).__name__} {str(e)[:120]}", flush=True)

    while True:
        now = datetime.now(NY)
        if now.date() != day or now.hour * 60 + now.minute >= OFF:
            break
        ws = websocket.WebSocketApp(f"wss://api.unusualwhales.com/socket?token={key}", on_open=on_open,
                                    on_message=sh.on_message, on_error=on_error)
        th = threading.Thread(target=lambda: ws.run_forever(ping_interval=20, ping_timeout=10), daemon=True)
        th.start()
        while th.is_alive():
            time.sleep(15)
            sh.flush()
            now = datetime.now(NY)
            if now.date() != day or now.hour * 60 + now.minute >= OFF:
                ws.close()
                break
        time.sleep(5)
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
