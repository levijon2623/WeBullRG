"""
replay_archiver.py -- keep each session so the flow viewer can REPLAY it (user, 2026-11-25: "make previous days playable,
like the intraday positioning map"). Read-only: reads the live/ files the engine and the cue services already write;
never touches the bot, orders or Webull.

WHY. live/state.json carries the day's minute series (price, VWAP, volume, flow, EMA, sweeps) but they reset at
midnight, and everything "as of now" -- the gamma heat, the implied-move band, the vol gauge, the STRIKE / THRU / PULL /
TIEBREAK pills -- is overwritten every few seconds. Without this, a past day can only be rebuilt in part.

WRITES live/replay/DATE/
  series.json    overwritten every minute 09:25-16:10: per ticker the minute series from state.json (px, vwap, vol,
                 flow, ema, sweep, cum, thr, rule, regime) + market_tide.json rows + futures_cvd.json products
  frames.jsonl   one line per minute: {"m": minute-of-day, "tk": {SPY: {spot, state, regime, vgauge, imove, cue}},
                 "im": implied_move.json, "side": sideline, "day": day-goal}; cue = that ticker's strikes.json object
                 (strikes / touch / thru / pullback / contra), so a replay shows the pills exactly as they stood
  gex.jsonl      every 5 minutes: per ticker the gex object (heat + walls), the heaviest piece, kept apart
Ten minutes after the close (16:10; 13:10 on a half day) the day's files are gzipped (series.json.gz / frames.jsonl.gz / gex.jsonl.gz) and live/replay/index.json lists
the days. Old days are kept (a few MB a day).

    python replay_archiver.py            (service cleanbot-replay)
    python replay_archiver.py --once     (one frame now, for a check)
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import shutil
import sys
import time
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NY = ZoneInfo("America/New_York")
LIVE = os.path.join(ROOT, "live")
OUT = os.path.join(LIVE, "replay")
TKS = ("SPY", "QQQ", "IWM")
START_MOD, END_MOD = 9 * 60 + 25, 16 * 60 + 10
SERIES_KEYS = ("px", "vwap", "vol", "flow", "ema", "sweep", "cum", "thr", "rule", "regime", "rvol_days")


def jload(name, default=None):
    try:
        with open(os.path.join(LIVE, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def frame(day_dir, mod):
    st = jload("state.json", {}) or {}
    tk = st.get("tickers") or {}
    cue = (jload("strikes.json", {}) or {}).get("tickers") or {}
    fr = {"m": mod, "ts": time.time(), "tk": {}, "im": jload("implied_move.json"), "side": jload("sideline.json"),
          "day": st.get("day")}
    for t in TKS:
        x = tk.get(t) or {}
        fr["tk"][t] = {k: x.get(k) for k in ("spot", "state", "regime", "vgauge", "imove", "position")}
        fr["tk"][t]["cue"] = cue.get(t)
    append(os.path.join(day_dir, "frames.jsonl"), fr)
    if mod % 5 == 0:
        append(os.path.join(day_dir, "gex.jsonl"), {"m": mod, "tk": {t: (tk.get(t) or {}).get("gex") for t in TKS}})
    series = {"day": day_dir[-10:], "at": mod, "tk": {t: {k: (tk.get(t) or {}).get(k) for k in SERIES_KEYS} for t in TKS},
              "tide": jload("market_tide.json"), "futures": jload("futures_cvd.json")}
    write_atomic(os.path.join(day_dir, "series.json"), series)


def finish(day_dir):
    for name in ("series.json", "frames.jsonl", "gex.jsonl"):
        p = os.path.join(day_dir, name)
        if os.path.exists(p):
            with open(p, "rb") as a, gzip.open(p + ".gz", "wb") as b:
                shutil.copyfileobj(a, b)
            os.remove(p)
    days = sorted(d for d in os.listdir(OUT) if len(d) == 10 and d[4] == "-")
    write_atomic(os.path.join(OUT, "index.json"), {"days": days, "updated": time.time()})


def main():
    import market_calendar as MC
    once = "--once" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    last_min, finished = None, set()
    while True:
        now = dt.datetime.now(NY)
        d, mod = now.date(), now.hour * 60 + now.minute
        day_dir = os.path.join(OUT, d.isoformat())
        if once:
            os.makedirs(day_dir, exist_ok=True)
            frame(day_dir, mod)
            print("one frame written to", day_dir)
            return
        end_mod = MC.close_mod(d) + 10 if MC.is_trading_day(d) else END_MOD     # half days stop 10 min after their close
        if MC.is_trading_day(d) and START_MOD <= mod < end_mod and mod != last_min:
            os.makedirs(day_dir, exist_ok=True)
            try:
                frame(day_dir, mod)
            except Exception as e:                          # noqa: BLE001 -- an archive must never stop the day
                print(f"frame {mod}: {type(e).__name__}: {e}", flush=True)
            last_min = mod
        elif MC.is_trading_day(d) and mod >= end_mod and d not in finished and os.path.isdir(day_dir):
            finish(day_dir)
            finished.add(d)
            print(f"{d} archived", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
