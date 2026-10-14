"""Order snapshots -> trade events. Shared by every snapshot-style source.

A broker's "recent orders" endpoint says what each order looks like NOW. The
tracker compares that with what it saw last time (in the Store) and emits:

    new order, still working             -> placed
    filled quantity went up, not done    -> partial   (the increment)
    order reached FILLED                 -> filled    (the remaining increment)
    seen working, now CANCELLED/REJECTED -> cancelled / rejected

🚨 BASELINE FIRST. The first poll of a source only RECORDS what is already
there -- otherwise starting the broadcaster would post a day (or a month) of
old orders to a public channel. Orders that fill later are still posted.

P&L: single-leg fills update an average-cost position per instrument. A
closing fill reports its % against that cost -- only when the opening fill was
seen by this broadcaster (a position opened before it started has no cost, so
no % is shown rather than a wrong one).
"""
from __future__ import annotations

import datetime as dt

from .events import (CANCEL, CANCELLED, FILL, FILLED, OPEN, PARTIAL, PARTIAL_FILL,
                     PLACED, REJECT, REJECTED, TERMINAL, OrderSnapshot, TradeEvent)


class OrderTracker:
    def __init__(self, store, source: str, label: str | None = None):
        self.store, self.source, self.label = store, source, label

    @property
    def baselined(self) -> bool:
        return bool(self.store.get_meta(f"baseline:{self.source}"))

    def diff(self, snaps: list[OrderSnapshot]) -> list[TradeEvent]:
        out: list[TradeEvent] = []
        first = not self.baselined
        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        for s in sorted(snaps, key=lambda x: x.created_at or x.updated_at or epoch):
            prev = self.store.get_order(self.source, s.order_id)
            if first:
                self.store.put_order(self.source, s.order_id, s.state, s.filled_quantity)
                continue
            out += self._events_for(s, prev)
            self.store.put_order(self.source, s.order_id, s.state, s.filled_quantity)
        if first:
            self.store.set_meta(f"baseline:{self.source}", True)
        self.store.commit()
        return out

    def _ev(self, s, kind, qty, price, suffix):
        return TradeEvent(source=self.source, key=f"{self.source}:{s.order_id}:{suffix}",
                          kind=kind, legs=s.legs, quantity=qty, total_quantity=s.quantity,
                          price=price, order_type=s.order_type,
                          at=s.updated_at or s.created_at or dt.datetime.now(dt.timezone.utc),
                          label=self.label)

    def _events_for(self, s: OrderSnapshot, prev):
        ev = []
        p_state, p_filled = prev if prev else (None, 0.0)
        if prev is None and s.state in (OPEN, PARTIAL):
            ev.append(self._ev(s, PLACED, s.quantity, s.limit_price, "placed"))
        inc = round(float(s.filled_quantity) - float(p_filled or 0), 6)
        if inc > 0:
            done = s.state == FILLED
            e = self._ev(s, FILL if done else PARTIAL_FILL, inc, s.avg_fill_price,
                         f"fill:{s.filled_quantity:g}")
            e.pnl_pct = self._book_fill(s, inc)
            ev.append(e)
        if s.state in (CANCELLED, REJECTED) and p_state not in TERMINAL and prev is not None:
            ev.append(self._ev(s, CANCEL if s.state == CANCELLED else REJECT,
                               s.quantity - s.filled_quantity, s.limit_price, s.state.lower()))
        return ev

    def _book_fill(self, s: OrderSnapshot, qty: float):
        """Average-cost bookkeeping for single-leg orders. -> realized % or None."""
        if len(s.legs) != 1 or s.avg_fill_price is None:
            return None
        leg = s.legs[0]
        inst = leg.instrument.key()
        held, cost = self.store.get_position(self.source, inst)
        opening = leg.effect == "open" if leg.effect else leg.side == "buy"
        px = float(s.avg_fill_price)
        if opening:
            nq = held + qty
            self.store.put_position(self.source, inst, nq,
                                    ((cost or 0) * held + px * qty) / nq)
            return None
        if not held or not cost:
            return None
        self.store.put_position(self.source, inst, max(0.0, held - qty), cost)
        return (px - cost) / cost * 100.0
