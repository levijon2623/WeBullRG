"""Print posts to the terminal -- for dry runs and testing a new source.

    [[sinks]]
    type = "console"
    name = "screen"
"""
from __future__ import annotations

from ..registry import register_sink
from .base import Sink


@register_sink("console")
class ConsoleSink(Sink):
    def send(self, message):
        print("┌ " + message.text().replace("\n", "\n│ "), flush=True)
