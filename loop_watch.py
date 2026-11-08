"""
loop_watch.py
=============
Tells on the main loop when it stops moving, from a thread of its own.

WHY (2026-11-06)
    bot_runner's loop does everything in one thread: chart state, viewer orders,
    exit checks, UW and Webull REST. At 10:17 that day a network blip made one pass
    take about 65 s. Nothing logged why, because the flow call swallows its own
    timeouts, and state.json simply stopped, so the viewer looked crashed while real
    orders waited. A thread that does not depend on the loop can say both things:
    the page gets live/heartbeat.json, the journal gets where the loop was stuck.

WHAT IT CANNOT DO
    Read-only on the engine: it never touches engine state and sends nothing. If the
    whole process dies the heartbeat goes quiet, which the viewer also reports.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "live", "heartbeat.json")
STALL_S = 10.0          # no beat for this long = stalled
WRITE_EVERY_S = 2.0
SAMPLE_EVERY_S = 5.0

_state = {"beat": None, "main": None, "thread": None}


def beat():
    """The main loop is alive. Cheap enough to call between tickers."""
    _state["beat"] = time.monotonic()


def _where(ident):
    """'bot_runner.run_brain:2171 > unusual_whales_client.get_live_net_premium:499 > socket.recv_into:707'
    Our own frames plus the innermost one, which is what it is actually waiting on."""
    frame = sys._current_frames().get(ident)
    if frame is None:
        return "unknown"
    stack = traceback.extract_stack(frame)
    keep = [f for f in stack if os.path.dirname(os.path.abspath(f.filename)) == ROOT]
    if stack and (not keep or keep[-1] is not stack[-1]):
        keep.append(stack[-1])
    return " > ".join(f"{os.path.splitext(os.path.basename(f.filename))[0]}.{f.name}:{f.lineno}" for f in keep[-5:])


def _write(path, age, stalled):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"ts": int(time.time()), "beat_age": round(age, 1), "stalled": stalled}, f)
    os.replace(tmp, path)


def _run(path, stall_s, write_every, sample_every):
    seen, stalled_at, next_sample, next_write = Counter(), None, 0.0, 0.0
    while True:
        try:
            now = time.monotonic()
            age = now - _state["beat"]
            if age >= stall_s:
                if stalled_at is None:
                    stalled_at, next_sample = _state["beat"], 0.0
                    where = _where(_state["main"])
                    print(f"  ⏱ LOOP STALL {age:.0f}s and counting: {where}", flush=True)
                    seen[where] += 1
                    next_sample = now + sample_every
                elif now >= next_sample:
                    seen[_where(_state["main"])] += 1
                    next_sample = now + sample_every
            elif stalled_at is not None:
                total = now - stalled_at - age
                top = ", ".join(f"{w} ({n}x)" for w, n in seen.most_common(3))
                print(f"  ⏱ LOOP STALL ended after {total:.0f}s; waiting in: {top}", flush=True)
                seen.clear()
                stalled_at = None
            if now >= next_write:
                _write(path, age, age >= stall_s)
                next_write = now + write_every
        except Exception as e:                  # noqa: BLE001 -- must never die or reach the loop
            try:
                print(f"  ⚠️ loop_watch: {type(e).__name__}: {e}", flush=True)
            except Exception:                   # noqa: BLE001
                pass
        time.sleep(0.25)


def start(path=OUT, stall_s=STALL_S, write_every=WRITE_EVERY_S, sample_every=SAMPLE_EVERY_S):
    """Call from the main loop's thread, just before it starts. Safe to call twice."""
    beat()
    _state["main"] = threading.get_ident()
    if _state["thread"] is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _state["thread"] = threading.Thread(target=_run, args=(path, stall_s, write_every, sample_every),
                                            daemon=True, name="loop_watch")
        _state["thread"].start()
