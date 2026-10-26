"""
llm_common.py
=============
One place for the Claude API calls (news_score.py, morning_brief.py). Runs in its own venv
on the box (venv-llm: anthropic + python-dotenv only) so the trading bot's environment is
never touched.

  * key: ANTHROPIC_API_KEY from .env via python-dotenv (explicit path) -- never printed
  * model: Claude Opus 5.5; refusals fall back server-side ("default" fallbacks)
  * output: JSON constrained by a schema (output_config.format), parsed with json.loads
  * every call appends its token usage to live/llm/usage.jsonl, and a daily call cap
    stops runaway loops before the Console spend limit has to
"""
from __future__ import annotations

import datetime as dt
import json
import os
import ssl
from zoneinfo import ZoneInfo

import anthropic
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
NY = ZoneInfo("America/New_York")
MODEL = "claude-opus-5-5"
OUT = os.path.join(ROOT, "live", "llm")
DAILY_CALL_CAP = 400

_client = None


class LLMError(Exception):
    pass


def client():
    global _client
    if _client is None:
        kw = {}
        if ssl.SSLContext.__module__ != "ssl":
            # something (pip_system_certs on the dev PC) already swapped ssl.SSLContext for a truststore
            # class; the SDK's own truststore then wraps it again and recurses -- hand it a plain context
            kw["http_client"] = anthropic.DefaultHttpxClient(verify=ssl.create_default_context())
        _client = anthropic.Anthropic(max_retries=3, timeout=180.0, **kw)
    return _client


def _calls_today():
    p = os.path.join(OUT, "usage.jsonl")
    day = dt.datetime.now(NY).date().isoformat()
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if f'"day":"{day}"' in line.replace(" ", ""))
    except OSError:
        return 0


def _log_usage(task, resp):
    os.makedirs(OUT, exist_ok=True)
    u = resp.usage
    rec = {"day": dt.datetime.now(NY).date().isoformat(), "t": dt.datetime.now(NY).strftime("%H:%M:%S"),
           "task": task, "model": resp.model, "in": u.input_tokens, "out": u.output_tokens,
           "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
           "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0, "stop": resp.stop_reason}
    with open(os.path.join(OUT, "usage.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def call_json(task, system, user, schema, effort="low", max_tokens=16000):
    """one request -> the parsed JSON object. Raises LLMError on refusal / truncation / cap."""
    if _calls_today() >= DAILY_CALL_CAP:
        raise LLMError(f"daily call cap {DAILY_CALL_CAP} reached")
    try:
        resp = client().beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            # no prompt caching: the fixed instructions are below the minimum cacheable size and the
            # rest changes on every call, so a cache write would cost more and never be read
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.RateLimitError as e:
        raise LLMError(f"rate limited ({e.status_code})") from None
    except anthropic.APIStatusError as e:
        raise LLMError(f"API {e.status_code}: {str(e.message)[:160]}") from None
    except anthropic.APIConnectionError:
        raise LLMError("connection error") from None
    _log_usage(task, resp)
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        raise LLMError(f"refused ({cat})")
    if resp.stop_reason == "max_tokens":
        raise LLMError("hit max_tokens")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise LLMError("no text block")
    return json.loads(text), resp.model
