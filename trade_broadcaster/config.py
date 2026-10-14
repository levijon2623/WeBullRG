"""config.toml loading and validation (secrets stay in .env, referenced by name)."""
from __future__ import annotations

import os
import tomllib

from .events import ALL_KINDS

DEFAULTS = {
    "broadcaster": {"poll_seconds": 4, "idle_poll_seconds": 60, "active_hours": "09:20-16:20",
                    "weekdays_only": True, "state_file": "state/broadcaster.sqlite",
                    "max_attempts": 8, "env_file": ".env"},
    "privacy": {"show_quantity": True, "show_price": True, "show_pnl": True, "footer": None,
                "delay_seconds": 0, "post": ["placed", "filled", "partial", "cancelled"]},
}


class ConfigError(Exception):
    pass


def load(path: str) -> dict:
    if not os.path.exists(path):
        raise ConfigError(f"{path} not found -- copy config.example.toml to {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = {k: {**v, **raw.get(k, {})} for k, v in DEFAULTS.items()}
    cfg["sources"] = raw.get("sources", [])
    cfg["sinks"] = raw.get("sinks", [])
    base = os.path.dirname(os.path.abspath(path))
    b = cfg["broadcaster"]
    for k in ("state_file", "env_file"):
        if b.get(k) and not os.path.isabs(b[k]):
            b[k] = os.path.join(base, b[k])
    for s in cfg["sources"]:
        if s.get("session_dir") and not os.path.isabs(s["session_dir"]):
            s["session_dir"] = os.path.join(base, s["session_dir"])
        elif s.get("type") == "robinhood" and not s.get("session_dir"):
            s["session_dir"] = os.path.join(base, "state")
    validate(cfg)
    return cfg


def validate(cfg):
    from .registry import SINKS, SOURCES
    if not cfg["sources"]:
        raise ConfigError("no [[sources]] configured")
    if not cfg["sinks"]:
        raise ConfigError("no [[sinks]] configured")
    for kind, reg, items in (("source", SOURCES, cfg["sources"]), ("sink", SINKS, cfg["sinks"])):
        names = set()
        for it in items:
            t, n = it.get("type"), it.get("name")
            if t not in reg:
                raise ConfigError(f"{kind} type {t!r} unknown (have: {', '.join(sorted(reg))})")
            if not n:
                raise ConfigError(f"every {kind} needs a unique `name`")
            if n in names:
                raise ConfigError(f"duplicate {kind} name {n!r}")
            names.add(n)
    bad = [k for k in cfg["privacy"]["post"] if k not in ALL_KINDS]
    if bad:
        raise ConfigError(f"privacy.post: unknown {bad} (allowed: {', '.join(ALL_KINDS)})")


def load_env(path: str | None):
    """Minimal .env reader (KEY=VALUE lines) so python-dotenv is optional.
    Existing environment variables win."""
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
