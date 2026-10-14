"""What a source must provide.

Two shapes, pick whichever fits the broker:

  SnapshotSource -- implement fetch_snapshots() -> list[OrderSnapshot] (the
      broker's recent orders as they look now). The shared OrderTracker turns
      them into events, handles the first-run baseline and the P&L. Most
      broker APIs (Robinhood, Webull, Schwab, Tradier, IBKR...) fit this.

  Source -- implement poll() -> list[TradeEvent] directly, for feeds that are
      already events (a ledger file, a webhook inbox).

Both get `name` (unique per config entry), `options` (that entry's table from
config.toml) and `store` (for any state they need to keep, via get_meta /
set_meta). Raise SourceError for a recoverable failure (network, expired
session): the engine logs it and tries again next poll.
"""
from __future__ import annotations

from ..events import OrderSnapshot, TradeEvent
from ..tracker import OrderTracker


class SourceError(Exception):
    pass


class Source:
    type_name = "?"

    def __init__(self, name: str, options: dict, store):
        self.name, self.options, self.store = name, options, store
        self.label = options.get("label")

    def start(self):
        """Log in / open files. Called once before the first poll."""

    def poll(self) -> list[TradeEvent]:
        raise NotImplementedError

    def describe_recent(self, n: int = 5) -> list[str]:
        """For `check`: a few human-readable lines about what the source sees."""
        return []


class SnapshotSource(Source):
    def __init__(self, name, options, store):
        super().__init__(name, options, store)
        self.tracker = OrderTracker(store, name, self.label)

    def fetch_snapshots(self) -> list[OrderSnapshot]:
        raise NotImplementedError

    def poll(self) -> list[TradeEvent]:
        return self.tracker.diff(self.fetch_snapshots())
