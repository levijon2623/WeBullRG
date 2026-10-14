"""Robinhood, via the UNOFFICIAL robin_stocks library (pip install robin_stocks).

⚠️ Robinhood has no official API for stocks or options. robin_stocks drives the
same private endpoints the app uses; Robinhood can change them without notice,
and automated access may conflict with its customer agreement. This source only
READS orders -- it never places, cancels or changes anything -- and polls one
page of recent orders every few seconds, which is light. Use it on your own
account, at your own discretion.

HOW IT WORKS
  Each poll reads the FIRST PAGE of option orders (and of stock orders, if
  enabled) -- the most recent ~100, newest first -- and hands them to the
  shared OrderTracker as snapshots. Contract details (strike, expiry, call /
  put) live behind each leg's `option` URL and are fetched once and cached.

LOGIN (once, interactively):  python -m trade_broadcaster login
  robin_stocks stores the session in `session_dir`; later runs reuse it.
  With `totp_env` set (the base32 secret from setting up an authenticator app
  for Robinhood), the code is generated automatically and re-logins need no
  human. WITHOUT it, a run that is not attached to a terminal REFUSES to log in
  rather than hang forever waiting for an SMS code nobody will type.

CONFIG
  [[sources]]
  type = "robinhood"
  name = "rh"                       # unique; used in state and dedupe keys
  label = "@yourhandle"             # optional, shown on posts
  username_env = "RH_USERNAME"
  password_env = "RH_PASSWORD"
  totp_env = "RH_TOTP_SECRET"       # optional
  options = true
  stocks = true
  lookback_hours = 48               # ignore orders last updated before this
  session_dir = "state"
"""
from __future__ import annotations

import builtins
import datetime as dt
import os
import sys

from ..events import (CANCELLED, FILLED, OPEN, PARTIAL, REJECTED, Instrument, Leg,
                      OrderSnapshot)
from ..registry import register_source
from .base import SnapshotSource, SourceError

OPTION_ORDERS_URL = "https://api.robinhood.com/options/orders/"
STOCK_ORDERS_URL = "https://api.robinhood.com/orders/"

STATE_MAP = {
    "queued": OPEN, "unconfirmed": OPEN, "confirmed": OPEN, "pending": OPEN,
    "partially_filled": PARTIAL, "filled": FILLED,
    "cancelled": CANCELLED, "canceled": CANCELLED,
    "rejected": REJECTED, "failed": REJECTED, "voided": CANCELLED,
}


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _ts(x):
    if not x:
        return None
    try:
        t = dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def parse_option_order(o: dict, resolve) -> OrderSnapshot | None:
    """One Robinhood option order -> OrderSnapshot. `resolve(url)` returns the
    option instrument dict (chain_symbol, strike_price, expiration_date, type)."""
    state = STATE_MAP.get(str(o.get("state", "")).lower())
    if state is None or not o.get("legs"):
        return None
    legs = []
    for lg in o["legs"]:
        ins = resolve(lg.get("option")) or {}
        sym = ins.get("chain_symbol") or o.get("chain_symbol") or "?"
        typ = str(ins.get("type") or "").lower()
        legs.append(Leg(Instrument("option", sym, "C" if typ == "call" else "P" if typ == "put" else None,
                                   _f(ins.get("strike_price")), ins.get("expiration_date")),
                        side=str(lg.get("side") or "").lower(),
                        effect=str(lg.get("position_effect") or "").lower() or None,
                        ratio=_f(lg.get("ratio_quantity")) or 1.0))
    qty = _f(o.get("quantity")) or 0.0
    filled = _f(o.get("processed_quantity")) or 0.0
    avg = None
    if filled > 0:
        # net premium per contract: processed_premium is dollars for the whole fill
        pp = _f(o.get("processed_premium"))
        if pp:
            avg = round(pp / (filled * 100.0), 4)
        elif len(o["legs"]) == 1:
            ex = o["legs"][0].get("executions") or []
            q = sum(_f(e.get("quantity")) or 0 for e in ex)
            if q:
                avg = round(sum((_f(e.get("price")) or 0) * (_f(e.get("quantity")) or 0) for e in ex) / q, 4)
    return OrderSnapshot(order_id=str(o.get("id")), state=state, legs=legs, quantity=qty,
                         filled_quantity=filled, avg_fill_price=avg, limit_price=_f(o.get("price")),
                         order_type=str(o.get("type") or "limit").lower(),
                         created_at=_ts(o.get("created_at")), updated_at=_ts(o.get("updated_at")))


def parse_stock_order(o: dict, resolve) -> OrderSnapshot | None:
    """One Robinhood stock order -> OrderSnapshot. `resolve(url)` returns the
    instrument dict (symbol)."""
    state = STATE_MAP.get(str(o.get("state", "")).lower())
    if state is None:
        return None
    sym = o.get("symbol") or (resolve(o.get("instrument")) or {}).get("symbol") or "?"
    side = str(o.get("side") or "").lower()
    return OrderSnapshot(order_id=str(o.get("id")), state=state,
                         legs=[Leg(Instrument("stock", sym), side=side,
                                   effect="open" if side == "buy" else "close")],
                         quantity=_f(o.get("quantity")) or 0.0,
                         filled_quantity=_f(o.get("cumulative_quantity")) or 0.0,
                         avg_fill_price=_f(o.get("average_price")),
                         limit_price=_f(o.get("price")),
                         order_type=str(o.get("type") or "market").lower(),
                         created_at=_ts(o.get("created_at")), updated_at=_ts(o.get("updated_at")))


class RobinhoodAPI:
    """Thin wrapper over robin_stocks so the parsing above can be tested with a
    fake. Only GETs."""

    def __init__(self, username, password, totp_secret=None, session_dir="state"):
        self.username, self.password, self.totp = username, password, totp_secret
        self.session_dir = session_dir
        self._rh = None

    def login(self, interactive: bool):
        try:
            import robin_stocks.robinhood as rh
        except ImportError as e:
            raise SourceError("robin_stocks is not installed: pip install robin_stocks") from e
        self._rh = rh
        mfa = None
        if self.totp:
            try:
                import pyotp
            except ImportError as e:
                raise SourceError("totp_env is set but pyotp is not installed: pip install pyotp") from e
            mfa = pyotp.TOTP(self.totp).now()
        os.makedirs(self.session_dir, exist_ok=True)
        real_input = builtins.input
        if not interactive:
            def _no_input(*_a, **_k):
                raise SourceError("Robinhood asked for a code/approval and nobody is at the "
                                  "terminal -- run `python -m trade_broadcaster login` once "
                                  "(or set totp_env)")
            builtins.input = _no_input
        try:
            kw = dict(username=self.username, password=self.password, store_session=True,
                      mfa_code=mfa)
            try:
                r = rh.login(**kw, pickle_path=self.session_dir, pickle_name="robinhood")
            except TypeError:                   # older robin_stocks: no pickle_* args
                r = rh.login(**kw)
        finally:
            builtins.input = real_input
        if not r:
            raise SourceError("Robinhood login failed (no session returned)")
        return True

    def get(self, url, payload=None):
        if not url:
            return None
        try:
            return self._rh.helper.request_get(url, "regular", payload)
        except Exception as e:                  # noqa: BLE001 -- network, parse
            raise SourceError(f"Robinhood GET failed: {type(e).__name__}") from None

    def first_page(self, url):
        d = self.get(url)
        if d is None:
            raise SourceError("Robinhood returned nothing (session expired? rate limit?)")
        return d.get("results") or []


@register_source("robinhood")
class RobinhoodSource(SnapshotSource):
    def __init__(self, name, options, store, api=None):
        super().__init__(name, options, store)
        self._cache: dict = {}
        self.api = api

    def start(self, interactive: bool | None = None):
        if self.api is None:
            env = lambda k: os.getenv(self.options.get(k) or "") or None    # noqa: E731
            user, pw = env("username_env"), env("password_env")
            if not user or not pw:
                raise SourceError(f"{self.name}: set {self.options.get('username_env')} and "
                                  f"{self.options.get('password_env')} in .env")
            self.api = RobinhoodAPI(user, pw, env("totp_env"),
                                    self.options.get("session_dir", "state"))
        if hasattr(self.api, "login"):
            self.api.login(sys.stdin.isatty() if interactive is None else interactive)

    def _resolve(self, url):
        if not url:
            return None
        if url not in self._cache:
            self._cache[url] = self.api.get(url) or {}
        return self._cache[url]

    def fetch_snapshots(self):
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            hours=float(self.options.get("lookback_hours", 48)))
        out = []
        feeds = []
        if self.options.get("options", True):
            feeds.append((OPTION_ORDERS_URL, parse_option_order))
        if self.options.get("stocks", True):
            feeds.append((STOCK_ORDERS_URL, parse_stock_order))
        for url, parse in feeds:
            for o in self.api.first_page(url):
                upd = _ts(o.get("updated_at")) or _ts(o.get("created_at"))
                if upd is not None and upd < cutoff:
                    continue
                s = parse(o, self._resolve)
                if s is not None:
                    out.append(s)
        return out

    def describe_recent(self, n=5):
        snaps = sorted(self.fetch_snapshots(), key=lambda s: s.created_at or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
                       reverse=True)[:n]
        lines = []
        for s in snaps:
            lg = s.legs[0]
            i = lg.instrument
            what = (f"{i.symbol} {i.strike:g}{i.right} {i.expiry}" if i.kind == "option" else i.symbol)
            lines.append(f"{(s.created_at or s.updated_at)} {s.state:9} {lg.side} {lg.effect or ''} "
                         f"{s.quantity:g}x {what}" + (f" +{len(s.legs)-1} legs" if len(s.legs) > 1 else "")
                         + (f" @ {s.avg_fill_price}" if s.avg_fill_price else ""))
        return lines
