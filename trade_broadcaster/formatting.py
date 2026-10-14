"""TradeEvent -> Message. Sinks render the Message their own way (Telegram HTML,
Discord embed, plain text), so wording lives here once.

Privacy options (config [privacy]):
    show_quantity  false -> "2 contracts" is left out (followers see the trade, not the size)
    show_price     false -> no fill / limit price
    show_pnl       false -> no % on closes
    footer         a line under every post, e.g. "Not financial advice."
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from .events import CANCEL, FILL, PARTIAL_FILL, PLACED, REJECT, TradeEvent

ET = ZoneInfo("America/New_York")
COLORS = {"open": 0x2EA043, "close": 0xD29922, "placed": 0x58A6FF, "cancel": 0x6E7681}


@dataclass
class Message:
    title: str
    lines: list[str]
    color: int = 0x58A6FF
    footer: str | None = None

    def text(self) -> str:
        body = "\n".join([self.title, *self.lines])
        return body + (f"\n{self.footer}" if self.footer else "")


def _money(x):
    return "—" if x is None else f"${x:,.2f}" if x >= 1 else f"${x:.2f}"


def _expiry(exp: str | None, today: dt.date) -> str:
    if not exp:
        return ""
    try:
        d = dt.date.fromisoformat(exp)
    except ValueError:
        return exp
    return "0DTE" if d == today else f"{d:%b} {d.day}"      # %-d is not portable (Windows)


def _what(leg, today) -> str:
    i = leg.instrument
    if i.kind == "option":
        k = f"{i.strike:g}" if i.strike is not None else "?"
        return f"{i.symbol} {k}{i.right or ''} {_expiry(i.expiry, today)}".strip()
    return i.symbol


def _action(leg) -> str:
    side = leg.side.upper() if leg.side else "?"
    if leg.instrument.kind == "option" and leg.effect in ("open", "close"):
        return f"{side} TO {leg.effect.upper()}"
    return side


def render(ev: TradeEvent, privacy: dict | None = None) -> Message:
    p = {"show_quantity": True, "show_price": True, "show_pnl": True, "footer": None,
         **(privacy or {})}
    at = ev.at.astimezone(ET) if ev.at.tzinfo else ev.at
    today = at.date()
    leg0 = ev.legs[0]
    multi = len(ev.legs) > 1
    closing = (leg0.effect == "close") or (leg0.effect is None and leg0.side == "sell")
    unit = "contract" if leg0.instrument.kind == "option" else "share"
    qty = f"{ev.quantity:g} {unit}{'s' if ev.quantity != 1 else ''}"
    if ev.kind in (FILL, PARTIAL_FILL) and closing and ev.total_quantity > ev.quantity:
        qty = f"{ev.quantity:g} of {ev.total_quantity:g} {unit}s"        # a scale-out

    head_icon, head = {
        PLACED: ("🕓", "ORDER PLACED"),
        PARTIAL_FILL: ("◐", "PARTIAL FILL"),
        FILL: ("🔴" if closing else "🟢", "CLOSED" if closing else "OPENED"),
        CANCEL: ("⚪", "CANCELLED"),
        REJECT: ("⛔", "REJECTED"),
    }.get(ev.kind, ("•", ev.kind.upper()))
    what = "SPREAD " + " / ".join(f"{_action(l)} {_what(l, today)}" for l in ev.legs) if multi \
        else f"{_action(leg0)} {_what(leg0, today)}"
    title = f"{head_icon} {head} · {what}"

    bits = []
    if p["show_quantity"]:
        bits.append(qty)
    if p["show_price"] and ev.price is not None:
        bits.append(("limit " if ev.kind == PLACED else "@ ") + _money(ev.price)
                    + (" net" if multi else ""))
    if ev.kind == PLACED and ev.order_type and ev.order_type != "limit":
        bits.append(ev.order_type)
    lines = [" · ".join(bits)] if bits else []
    if p["show_pnl"] and ev.kind in (FILL, PARTIAL_FILL) and ev.pnl_pct is not None:
        lines.append(f"{'📈' if ev.pnl_pct >= 0 else '📉'} {ev.pnl_pct:+.1f}% on the position")
    stamp = f"{at:%H:%M:%S} ET" + (f" · {ev.label}" if ev.label else "")
    lines.append(stamp)
    color = (COLORS["placed"] if ev.kind == PLACED else COLORS["cancel"] if ev.kind in (CANCEL, REJECT)
             else COLORS["close"] if closing else COLORS["open"])
    return Message(title=title, lines=lines, color=color, footer=p.get("footer"))
