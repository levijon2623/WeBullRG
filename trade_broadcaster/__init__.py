"""
trade_broadcaster
=================
Watch a trading account and post its trades to Telegram / Discord as they are
placed and filled.

Built to be extended. A broker is a SOURCE, a destination is a SINK, and the
core in between knows neither:

    sources/   one module per broker. A source either reports ORDER SNAPSHOTS
               (what a broker's "recent orders" endpoint returns) and lets the
               shared OrderTracker work out placed / partial / filled /
               cancelled, or it emits finished TradeEvents itself (a ledger
               file). Register it with @register_source("name").
    sinks/     one module per destination. Takes a rendered Message, sends it.
               Register it with @register_sink("name").
    core       tracker.py (snapshots -> events, P&L on closes), store.py
               (SQLite: what was seen, what was posted, the outbox that makes
               delays and retries survive a restart), engine.py (the loop),
               formatting.py (an event -> a Message, with the privacy options).

Standalone on purpose: nothing here imports the Cleanbot bot, so the folder can
be copied to another machine and run on its own. See README.md.
"""
__version__ = "0.1.0"
