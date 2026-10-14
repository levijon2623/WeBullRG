"""Discord: a channel webhook (Channel settings -> Integrations -> Webhooks).

    [[sinks]]
    type = "discord"
    name = "dc"
    webhook_env = "DISCORD_WEBHOOK_URL"
    username = "Trade Alerts"           # optional display name

The webhook URL IS the credential (anyone holding it can post as it), so it is
never logged, and requests' exception text -- which contains it -- is dropped.
"""
from __future__ import annotations

from ..registry import register_sink
from .base import Sink, SinkError


@register_sink("discord")
class DiscordSink(Sink):
    def send(self, message):
        import requests
        url = self.secret("webhook")
        embed = {"title": message.title[:256], "description": "\n".join(message.lines)[:4000],
                 "color": int(message.color)}
        if message.footer:
            embed["footer"] = {"text": message.footer[:2000]}
        body = {"embeds": [embed], "allowed_mentions": {"parse": []}}
        if self.options.get("username"):
            body["username"] = str(self.options["username"])[:80]
        try:
            r = requests.post(url, json=body, timeout=10)
        except Exception as e:                  # noqa: BLE001 -- never str(e): it holds the URL
            raise SinkError(f"discord: {type(e).__name__}") from None
        if r.status_code == 429:
            try:
                ra = float(r.json().get("retry_after", 2))
            except Exception:                   # noqa: BLE001
                ra = 2.0
            raise SinkError("discord: rate limited", retry_after=ra)
        if r.status_code not in (200, 204):
            raise SinkError(f"discord: HTTP {r.status_code}")
