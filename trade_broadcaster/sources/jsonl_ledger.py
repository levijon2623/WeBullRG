"""A JSON-lines trade ledger as a source -- e.g. Cleanbot's manual desk
(manual_executions.jsonl), which writes one row per confirmed fill:

    {"kind": "open",  "ts": 1791554477, "trade_id": "...", "ticker": "QQQ",
     "strike": 750, "right": "C", "expiry": "2026-10-09", "qty": 1, "entry": 1.31, ...}
    {"kind": "close", "ts": 1791554640, "trade_id": "...", ..., "exit": 1.10, "roe_pct": -16.03}

Fills only (the ledger has no "placed" moment). The rows carry their own ROE,
so the % on a close comes straight from the ledger. Rows marked "dry": true
are skipped. The file is read from where it was left (byte offset in the
store); the FIRST run starts at the end of the file, so history is never
posted.

    [[sources]]
    type = "jsonl_ledger"
    name = "webull-desk"
    path = "/root/bot_engine/Cleanbot/manual_executions.jsonl"
    label = "@yourhandle"
"""
from __future__ import annotations

import datetime as dt
import json
import os

from ..events import FILL, Instrument, Leg, TradeEvent
from ..registry import register_source
from .base import Source, SourceError


@register_source("jsonl_ledger")
class JsonlLedgerSource(Source):
    def start(self):
        self.path = self.options.get("path")
        if not self.path:
            raise SourceError(f"{self.name}: set `path`")
        k = f"offset:{self.name}"
        if self.store.get_meta(k) is None:
            self.store.set_meta(k, os.path.getsize(self.path) if os.path.exists(self.path) else 0)

    def poll(self):
        k = f"offset:{self.name}"
        off = int(self.store.get_meta(k) or 0)
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size < off:                        # file was rotated / rewritten
            off = 0
        out = []
        with open(self.path, "rb") as f:
            f.seek(off)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        for line in data[:end].decode("utf-8", "replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            e = self._event(r)
            if e is not None:
                out.append(e)
        self.store.set_meta(k, off + end + 1)
        return out

    def _event(self, r):
        if r.get("dry") or r.get("kind") not in ("open", "close"):
            return None
        opening = r["kind"] == "open"
        inst = Instrument("option", str(r.get("ticker")), str(r.get("right") or "")[:1] or None,
                          float(r["strike"]) if r.get("strike") is not None else None, r.get("expiry"))
        px = r.get("entry") if opening else r.get("exit")
        # a position can close in pieces (scale-outs): one close row per fill,
        # so the key carries the row's time and size, not just the trade id
        key = (f"{self.name}:{r.get('trade_id')}:open" if opening else
               f"{self.name}:{r.get('trade_id')}:close:{r.get('ts')}:{r.get('qty')}")
        qty = float(r.get("qty") or 0)
        total = qty + float(r.get("remaining") or 0) if not opening else qty
        return TradeEvent(source=self.name, key=key,
                          kind=FILL, legs=[Leg(inst, "buy" if opening else "sell",
                                               "open" if opening else "close")],
                          quantity=qty, total_quantity=total,
                          price=float(px) if px is not None else None,
                          at=dt.datetime.fromtimestamp(int(r.get("ts") or 0), dt.timezone.utc),
                          pnl_pct=(None if opening else r.get("roe_pct")), label=self.label)

    def describe_recent(self, n=5):
        if not os.path.exists(self.path):
            return [f"{self.path}: not found"]
        with open(self.path, encoding="utf-8", errors="replace") as f:
            rows = f.readlines()[-n:]
        return [ln.strip()[:140] for ln in rows]
