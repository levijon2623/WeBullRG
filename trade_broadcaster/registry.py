"""Name -> class registries. Adding a broker or a destination is one new module
that decorates its class; config.toml then refers to it by that name."""
from __future__ import annotations

import importlib
import pkgutil

SOURCES: dict[str, type] = {}
SINKS: dict[str, type] = {}


def register_source(name: str):
    def deco(cls):
        SOURCES[name] = cls
        cls.type_name = name
        return cls
    return deco


def register_sink(name: str):
    def deco(cls):
        SINKS[name] = cls
        cls.type_name = name
        return cls
    return deco


def load_plugins():
    """Import every module in sources/ and sinks/ so their decorators run."""
    for pkg in ("trade_broadcaster.sources", "trade_broadcaster.sinks"):
        mod = importlib.import_module(pkg)
        for info in pkgutil.iter_modules(mod.__path__):
            if not info.name.startswith("_") and info.name != "base":
                importlib.import_module(f"{pkg}.{info.name}")
