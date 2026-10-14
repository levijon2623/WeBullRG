"""python -m trade_broadcaster <command> [--config config.toml]

    run          poll and post, forever (Ctrl+C to stop)
    once         one poll + one flush, then exit (cron-friendly / testing)
    login        log the sources in interactively (Robinhood's first code / approval)
    check        log in, show what each source sees -- posts NOTHING
    test-sinks   send one test message to every sink
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from .config import ConfigError, load, load_env
from .engine import Broadcaster, log
from .events import FILL, Instrument, Leg, TradeEvent
from .formatting import render
from .registry import load_plugins
from .sinks.base import SinkError
from .sources.base import SourceError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trade_broadcaster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "once", "login", "check", "test-sinks"])
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):        # emoji on a Windows console
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                          # noqa: BLE001
            pass
    load_plugins()
    try:
        cfg = load(a.config)
    except ConfigError as e:
        print(f"config: {e}")
        return 2
    load_env(cfg["broadcaster"].get("env_file"))
    b = Broadcaster(cfg)

    if a.command in ("login", "check"):
        for s in b.sources:
            try:
                if s.type_name == "robinhood":
                    s.start(interactive=True)
                else:
                    s.start()
                print(f"✓ {s.name} ({s.type_name}) logged in / ready")
                if a.command == "check":
                    for line in s.describe_recent(8):
                        print("   ", line)
            except SourceError as e:
                print(f"✗ {s.name}: {e}")
        if a.command == "check":
            for k in b.sinks:
                try:
                    k.secret("token") if k.type_name == "telegram" else \
                        k.secret("webhook") if k.type_name == "discord" else None
                    print(f"✓ sink {k.name} ({k.type_name}) configured")
                except SinkError as e:
                    print(f"✗ sink {k.name}: {e}")
        return 0

    if a.command == "test-sinks":
        ev = TradeEvent(source="test", key="test", kind=FILL,
                        legs=[Leg(Instrument("option", "SPY", "C", 500.0,
                                             dt.date.today().isoformat()), "buy", "open")],
                        quantity=1, total_quantity=1, price=1.23,
                        label=(cfg["sources"][0].get("label") if cfg["sources"] else None))
        msg = render(ev, cfg["privacy"])
        msg.title = "🧪 TEST — " + msg.title
        for k in b.sinks:
            try:
                k.send(msg)
                print(f"✓ {k.name}: sent")
            except SinkError as e:
                print(f"✗ {k.name}: {e}")
        return 0

    try:
        b.run(once=(a.command == "once"))
    except KeyboardInterrupt:
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
