"""The loop: poll sources -> outbox -> sinks.

Every event that passes the privacy filter is written to the OUTBOX once per
sink (UNIQUE(sink, event_key)), ready at now + delay_seconds, and sent from
there. So a sink that is down retries with backoff without blocking the
others, a restart resumes where it stopped, and nothing is ever posted twice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import time
from zoneinfo import ZoneInfo

from .events import Instrument, Leg, TradeEvent
from .formatting import Message, render
from .registry import SINKS, SOURCES
from .sinks.base import SinkError
from .sources.base import SourceError
from .store import Store

ET = ZoneInfo("America/New_York")


def log(msg):
    print(f"{dt.datetime.now(ET):%m-%d %H:%M:%S} {msg}", flush=True)


def _event_to_json(ev: TradeEvent) -> dict:
    d = dataclasses.asdict(ev)
    d["at"] = ev.at.isoformat()
    return d


def _event_from_json(d: dict) -> TradeEvent:
    d = dict(d)
    d["legs"] = [Leg(Instrument(**l["instrument"]), l["side"], l.get("effect"), l.get("ratio", 1.0))
                 for l in d["legs"]]
    d["at"] = dt.datetime.fromisoformat(d["at"])
    return TradeEvent(**d)


class Broadcaster:
    def __init__(self, cfg, store: Store | None = None, sources=None, sinks=None):
        self.cfg = cfg
        self.store = store or Store(cfg["broadcaster"]["state_file"])
        self.sources = sources if sources is not None else [
            SOURCES[s["type"]](s["name"], s, self.store) for s in cfg["sources"]]
        self.sinks = sinks if sinks is not None else [
            SINKS[k["type"]](k["name"], k) for k in cfg["sinks"]]
        self._started = set()
        self._src_backoff: dict[str, float] = {}

    # ---- sources
    def start_sources(self):
        for s in self.sources:
            if s.name in self._started:
                continue
            try:
                s.start()
                self._started.add(s.name)
                log(f"source {s.name} ({s.type_name}) ready")
            except SourceError as e:
                log(f"source {s.name}: {e}")

    def poll_sources(self) -> list[TradeEvent]:
        out, now = [], time.time()
        for s in self.sources:
            if s.name not in self._started or self._src_backoff.get(s.name, 0) > now:
                continue
            try:
                evs = s.poll()
                self._src_backoff.pop(s.name, None)
                out += evs
            except SourceError as e:
                log(f"source {s.name}: {e} -- retrying in 30s")
                self._src_backoff[s.name] = now + 30
        return out

    # ---- outbox
    def enqueue(self, events):
        pv = self.cfg["privacy"]
        ready = time.time() + float(pv.get("delay_seconds") or 0)
        n = 0
        for ev in events:
            if ev.kind not in pv["post"]:
                continue
            for k in self.sinks:
                n += self.store.enqueue(k.name, ev.key, _event_to_json(ev), ready)
        self.store.commit()
        return n

    def flush(self):
        sinks = {k.name: k for k in self.sinks}
        sent = 0
        for row_id, sink, key, payload, attempts in self.store.due(
                time.time(), int(self.cfg["broadcaster"]["max_attempts"])):
            k = sinks.get(sink)
            if k is None:
                continue
            msg: Message = render(_event_from_json(json.loads(payload)), self.cfg["privacy"])
            try:
                k.send(msg)
                self.store.mark_sent(row_id)
                sent += 1
            except SinkError as e:
                wait = (e.retry_after if e.retry_after is not None
                        else min(300, 5 * 2 ** attempts))
                self.store.mark_failed(row_id, str(e), time.time() + wait)
                log(f"sink {sink}: {e} -- retry in {wait:.0f}s ({key})")
        return sent

    # ---- loop
    def active(self, now=None) -> bool:
        b = self.cfg["broadcaster"]
        t = (now or dt.datetime.now(ET))
        if b.get("weekdays_only", True) and t.weekday() >= 5:
            return False
        try:
            a, z = str(b.get("active_hours", "00:00-23:59")).split("-")
            ah, am = map(int, a.split(":"))
            zh, zm = map(int, z.split(":"))
        except ValueError:
            return True
        m = t.hour * 60 + t.minute
        return ah * 60 + am <= m <= zh * 60 + zm

    def step(self):
        evs = self.poll_sources()
        n = self.enqueue(evs)
        if n:
            log(f"{n} post(s) queued")
        return self.flush()

    def run(self, once=False):
        self.start_sources()
        b = self.cfg["broadcaster"]
        while True:
            if len(self._started) < len(self.sources):
                self.start_sources()
            try:
                self.step()
            except Exception as e:                  # noqa: BLE001 -- keep the loop alive
                log(f"step failed: {type(e).__name__}: {e}")
            if once:
                return
            time.sleep(float(b["poll_seconds"] if self.active() else b["idle_poll_seconds"]))
