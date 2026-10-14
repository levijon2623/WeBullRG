"""What a sink must provide: send(Message). Raise SinkError on failure; set
`retry_after` on it when the service said how long to wait (rate limits).

🚨 NEVER put a webhook URL or bot token into an exception or a log line. Both
ARE the credential: anyone holding a Discord webhook URL or a Telegram bot
token can post as you. requests' own exceptions embed the URL, so sinks catch
them and raise SinkError with the exception TYPE only.
"""
from __future__ import annotations

import os


class SinkError(Exception):
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class Sink:
    type_name = "?"

    def __init__(self, name: str, options: dict):
        self.name, self.options = name, options

    def secret(self, key: str) -> str:
        """Read a secret through the env var NAMED in config (`<key>_env`)."""
        var = self.options.get(f"{key}_env")
        val = os.getenv(var or "") if var else None
        if not val:
            raise SinkError(f"{self.name}: env var {var or '(' + key + '_env not set)'} is empty")
        return val

    def send(self, message) -> None:
        raise NotImplementedError
