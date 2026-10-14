"""Telegram: a bot (from @BotFather) posting into a chat or channel.

    [[sinks]]
    type = "telegram"
    name = "tg"
    token_env = "TELEGRAM_BOT_TOKEN"
    chat_id_env = "TELEGRAM_CHAT_ID"    # a channel: add the bot as admin, use @channelname or -100...

The token is the credential and is part of the API URL, so neither the URL nor
requests' exception text is ever logged (see sinks/base.py).
"""
from __future__ import annotations

import html

from ..registry import register_sink
from .base import Sink, SinkError


@register_sink("telegram")
class TelegramSink(Sink):
    def send(self, message):
        import requests
        token, chat = self.secret("token"), self.secret("chat_id")
        text = f"<b>{html.escape(message.title)}</b>\n" + "\n".join(html.escape(x) for x in message.lines)
        if message.footer:
            text += f"\n<i>{html.escape(message.footer)}</i>"
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json=dict(chat_id=chat, text=text, parse_mode="HTML",
                                        disable_web_page_preview=True), timeout=10)
        except Exception as e:                  # noqa: BLE001 -- never str(e): it holds the URL
            raise SinkError(f"telegram: {type(e).__name__}") from None
        if r.status_code == 429:
            try:
                ra = float(r.json().get("parameters", {}).get("retry_after", 5))
            except Exception:                   # noqa: BLE001
                ra = 5.0
            raise SinkError("telegram: rate limited", retry_after=ra)
        if r.status_code != 200:
            try:
                why = str(r.json().get("description", ""))[:120]
            except Exception:                   # noqa: BLE001
                why = ""
            raise SinkError(f"telegram: HTTP {r.status_code} {why}")
