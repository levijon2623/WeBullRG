"""The broker-neutral vocabulary every source speaks and every sink hears."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

# normalized order states (a source maps its broker's words onto these)
OPEN, PARTIAL, FILLED, CANCELLED, REJECTED = "OPEN", "PARTIAL", "FILLED", "CANCELLED", "REJECTED"
TERMINAL = (FILLED, CANCELLED, REJECTED)

# event kinds -- also the words used in config `post = [...]`
PLACED, PARTIAL_FILL, FILL, CANCEL, REJECT = "placed", "partial", "filled", "cancelled", "rejected"
ALL_KINDS = (PLACED, PARTIAL_FILL, FILL, CANCEL, REJECT)


@dataclass(frozen=True)
class Instrument:
    kind: str                       # "option" | "stock" | "crypto" | ...
    symbol: str                     # underlying / ticker
    right: str | None = None        # "C" | "P" for options
    strike: float | None = None
    expiry: str | None = None       # "YYYY-MM-DD"

    def key(self) -> str:
        if self.kind == "option":
            return f"{self.symbol}|{self.expiry}|{self.right}|{self.strike:g}"
        return f"{self.kind}|{self.symbol}"


@dataclass
class Leg:
    instrument: Instrument
    side: str                       # "buy" | "sell"
    effect: str | None = None       # "open" | "close" | None (unknown)
    ratio: float = 1.0


@dataclass
class OrderSnapshot:
    """One order as a broker reports it right now. Sources that poll a
    "recent orders" endpoint produce these; OrderTracker diffs them."""
    order_id: str
    state: str                      # one of the normalized states above
    legs: list[Leg]
    quantity: float                 # total ordered (contracts / shares)
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None   # per share / per contract (net for spreads)
    limit_price: float | None = None
    order_type: str = "limit"       # "limit" | "market" | "stop" | ...
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None


@dataclass
class TradeEvent:
    source: str                     # the source's configured name
    key: str                        # unique per (source, order, transition) -- the dedupe key
    kind: str                       # one of ALL_KINDS
    legs: list[Leg]
    quantity: float                 # this event's quantity (a fill's increment, or the order size)
    total_quantity: float
    price: float | None             # fill price for fills, limit for placed
    order_type: str = "limit"
    at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    pnl_pct: float | None = None    # realized on a closing fill, when the open was seen
    label: str | None = None        # optional account / trader label shown in posts
