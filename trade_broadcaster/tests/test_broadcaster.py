"""python -m unittest discover -s trade_broadcaster/tests -t ."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

from trade_broadcaster.config import DEFAULTS, validate
from trade_broadcaster.engine import Broadcaster
from trade_broadcaster.formatting import render
from trade_broadcaster.registry import load_plugins
from trade_broadcaster.sinks.base import Sink, SinkError
from trade_broadcaster.sources.jsonl_ledger import JsonlLedgerSource
from trade_broadcaster.sources.robinhood import RobinhoodSource, parse_option_order
from trade_broadcaster.store import Store

load_plugins()
OPT = "https://api.robinhood.com/options/instruments/abc/"
OPT2 = "https://api.robinhood.com/options/instruments/def/"
INSTR = {OPT: dict(chain_symbol="SPY", strike_price="772.0000", expiration_date="2026-10-14", type="call"),
         OPT2: dict(chain_symbol="SPY", strike_price="775.0000", expiration_date="2026-10-14", type="call")}


def rh_order(oid, state, qty=2, filled=0, premium=None, effect="open", side="buy", price="0.93",
             created="2026-10-14T14:41:00Z", legs=None):
    o = dict(id=oid, state=state, quantity=str(qty), processed_quantity=str(filled), price=price,
             type="limit", created_at=created, updated_at=created, chain_symbol="SPY",
             legs=legs or [dict(option=OPT, side=side, position_effect=effect, ratio_quantity=1)])
    if premium is not None:
        o["processed_premium"] = str(premium)
    return o


class FakeAPI:
    def __init__(self):
        self.options, self.stocks = [], []

    def get(self, url, payload=None):
        return INSTR.get(url)

    def first_page(self, url):
        return list(self.options if "options" in url else self.stocks)


class Collect(Sink):
    type_name = "collect"

    def __init__(self, name="c", fail=0):
        super().__init__(name, {})
        self.got, self.fail = [], fail

    def send(self, message):
        if self.fail:
            self.fail -= 1
            raise SinkError("down", retry_after=0)
        self.got.append(message)


def cfg(**privacy):
    c = {k: dict(v) for k, v in DEFAULTS.items()}
    c["privacy"].update(privacy)
    c["sources"], c["sinks"] = [], []
    return c


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.tmp, "s.sqlite"))
        self.api = FakeAPI()
        self.src = RobinhoodSource("rh", {"options": True, "stocks": False, "lookback_hours": 1e6,
                                          "label": "@me"}, self.store, api=self.api)
        self.sink = Collect()
        self.b = Broadcaster(cfg(), store=self.store, sources=[self.src], sinks=[self.sink])
        self.b._started.add("rh")

    def step(self):
        self.b.step()
        return [m.title for m in self.sink.got]

    def test_parse_option_order(self):
        s = parse_option_order(rh_order("1", "filled", qty=2, filled=2, premium="184"), INSTR.get)
        self.assertEqual((s.state, s.filled_quantity, s.avg_fill_price), ("FILLED", 2.0, 0.92))
        i = s.legs[0].instrument
        self.assertEqual((i.symbol, i.right, i.strike, i.expiry), ("SPY", "C", 772.0, "2026-10-14"))

    def test_baseline_posts_nothing(self):
        self.api.options = [rh_order("old", "filled", filled=2, premium="184"),
                            rh_order("old2", "queued")]
        self.assertEqual(self.step(), [])

    def test_lifecycle_and_pnl(self):
        self.step()                                                   # empty baseline
        self.api.options = [rh_order("a", "queued")]
        self.assertIn("ORDER PLACED", self.step()[-1])
        self.api.options = [rh_order("a", "partially_filled", filled=1, premium="92")]
        self.assertIn("PARTIAL FILL", self.step()[-1])
        self.api.options = [rh_order("a", "filled", filled=2, premium="184")]
        self.assertIn("OPENED", self.step()[-1])
        self.api.options.append(rh_order("b", "filled", qty=2, filled=2, premium="240",
                                         effect="close", side="sell", created="2026-10-14T14:58:00Z"))
        self.step()
        last = self.sink.got[-1]
        self.assertIn("CLOSED", last.title)
        self.assertTrue(any("+30.4%" in x for x in last.lines), last.lines)
        self.assertEqual(len(self.sink.got), 4)
        self.step()                                                   # nothing new -> nothing posted
        self.assertEqual(len(self.sink.got), 4)

    def test_cancel(self):
        self.step()
        self.api.options = [rh_order("c", "queued")]
        self.step()
        self.api.options = [rh_order("c", "cancelled")]
        self.assertIn("CANCELLED", self.step()[-1])

    def test_spread_is_one_post(self):
        self.step()
        legs = [dict(option=OPT, side="buy", position_effect="open"),
                dict(option=OPT2, side="sell", position_effect="open")]
        self.api.options = [rh_order("s", "filled", qty=1, filled=1, premium="120", legs=legs)]
        t = self.step()
        self.assertEqual(len(t), 1)
        self.assertIn("SPREAD", t[0])

    def test_restart_does_not_repost(self):
        self.step()
        self.api.options = [rh_order("a", "filled", filled=2, premium="184")]
        self.step()
        store2 = Store(os.path.join(self.tmp, "s.sqlite"))
        sink2 = Collect()
        src2 = RobinhoodSource("rh", dict(self.src.options), store2, api=self.api)
        b2 = Broadcaster(cfg(), store=store2, sources=[src2], sinks=[sink2])
        b2._started.add("rh")
        b2.step()
        self.assertEqual(sink2.got, [])

    def test_delay_and_retry_and_filter(self):
        sink_bad, sink_ok = Collect("bad", fail=1), Collect("ok")
        b = Broadcaster(cfg(delay_seconds=0.3, post=["filled"]), store=self.store,
                        sources=[self.src], sinks=[sink_bad, sink_ok])
        b._started.add("rh")
        b.step()
        self.api.options = [rh_order("q", "queued")]
        b.step()                                                      # placed: filtered out
        self.api.options = [rh_order("q", "filled", filled=2, premium="184")]
        b.step()
        self.assertEqual((sink_ok.got, sink_bad.got), ([], []))       # delayed
        time.sleep(0.35)
        b.flush()
        self.assertEqual(len(sink_ok.got), 1)                         # ok sink unaffected
        self.assertEqual(len(sink_bad.got), 0)                        # failed once
        b.flush()
        self.assertEqual(len(sink_bad.got), 1)                        # retried
        b.flush()
        self.assertEqual((len(sink_ok.got), len(sink_bad.got)), (1, 1))

    def test_privacy(self):
        self.step()
        self.api.options = [rh_order("p", "filled", filled=2, premium="184")]
        b = self.b
        b.cfg["privacy"].update(show_quantity=False, show_price=False, footer="NFA")
        self.step()
        m = self.sink.got[-1]
        self.assertFalse(any("contract" in x or "$" in x for x in m.lines), m.lines)
        self.assertEqual(m.footer, "NFA")


class LedgerTests(unittest.TestCase):
    def test_ledger_tail(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "l.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps(dict(kind="open", ts=1, trade_id="old", ticker="SPY", strike=770,
                                    right="C", expiry="2026-10-14", qty=1, entry=1.0)) + "\n")
        store = Store(os.path.join(tmp, "s.sqlite"))
        src = JsonlLedgerSource("desk", {"path": p, "label": "@me"}, store)
        src.start()
        self.assertEqual(src.poll(), [])                              # history not posted
        with open(p, "a") as f:
            f.write(json.dumps(dict(kind="close", ts=int(time.time()), trade_id="t1", ticker="SPY",
                                    strike=770, right="C", expiry="2026-10-14", qty=1, exit=1.3,
                                    roe_pct=30.0)) + "\n")
            f.write(json.dumps(dict(kind="open", ts=2, trade_id="d", dry=True, ticker="SPY")) + "\n")
            f.write('{"kind": "open", "partial line')                 # no newline yet: not read
        ev = src.poll()
        self.assertEqual(len(ev), 1)
        m = render(ev[0])
        self.assertIn("SELL TO CLOSE SPY 770C", m.title)
        self.assertTrue(any("+30.0%" in x for x in m.lines))


class ScaleOutLedgerTests(unittest.TestCase):
    def test_two_partial_closes_both_post(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "l.jsonl")
        open(p, "w").close()
        store = Store(os.path.join(tmp, "s.sqlite"))
        src = JsonlLedgerSource("desk", {"path": p}, store)
        src.start()
        row = dict(kind="close", trade_id="t", ticker="SPY", strike=770, right="C",
                   expiry="2026-10-14", exit=1.3, roe_pct=30.0)
        with open(p, "a") as f:
            f.write(json.dumps(dict(row, ts=100, qty=1, remaining=2)) + "\n")
            f.write(json.dumps(dict(row, ts=160, qty=2, remaining=0)) + "\n")
        ev = src.poll()
        self.assertEqual(len({e.key for e in ev}), 2)
        self.assertTrue(any("1 of 3 contracts" in x for x in render(ev[0]).lines))


class ConfigTests(unittest.TestCase):
    def test_unknown_type(self):
        c = cfg()
        c["sources"] = [{"type": "nope", "name": "x"}]
        c["sinks"] = [{"type": "console", "name": "s"}]
        with self.assertRaises(Exception):
            validate(c)


if __name__ == "__main__":
    unittest.main()
