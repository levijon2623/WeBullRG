"""
news_score.py
=============
CLAUDE SCORES EACH RELEVANT HEADLINE AS IT ARRIVES (user, 2026-10-26), a forward log to test
in a few months: does a model's read of a headline say anything the keyword filter can't?
(check_news_reaction.py found keyword categories alone carry no tradeable reaction.)

Reads live/news/DATE.jsonl (news_logger.py), takes the records the keyword filter marked
relevant (non-empty `cats`), and scores them in small batches -- at most 20 headlines per
request, one request per minute at most. Writes live/news/ai/DATE.jsonl, one line per headline:
  key, feed, id, t (published), seen (first seen by the logger), scored_at (all UTC), model, pv (prompt
  version), relevance 0-3, direction -2..+2 (SPY), novelty new/update/rehash,
  horizon minutes/hours/days, tickers, why
The model sees only the headline and its metadata, as of when it was first seen -- never prices.
A headline scored more than 5 minutes after it was seen (catch-up after a restart) is marked
late=true and stays out of the timing test below.

PRE-REGISTERED TEST (run once >= 60 trading days are logged, ~2027-01-26; nothing else is scored)
  Sample: headlines first seen 09:35-15:30 ET on trading days, late=false. Story clusters:
  among relevance >= 2 headlines, keep the first in any 10-minute window.
  H1 SIZE   relevance >= 2 and novelty != rehash -> SPY |log return| over the 30 min after `seen`,
            / the 20-day median |30-min return| starting at the same minute. Compared with all
            other scored headlines. PASS: mean difference > 0, day-block bootstrap 95% CI above 0,
            above the p95 of 200 same-day random-time shifts, same sign in both halves (date midpoint).
  H2 SIGN   relevance >= 2, direction != 0 -> sign(direction) x SPY 30-min log return after
            scored_at (the tradable time). PASS: mean > 0 with the bootstrap CI above 0, hit rate
            > 50% at binomial p < 0.05, same sign in both halves.
  Any other cut (tickers, horizon, categories) is reported as exploratory.

Usage:  python news_score.py            # one pass over today's file (the service calls score_pass)
        python news_score.py --date 2026-10-23
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import llm_common as L

NEWS = os.path.join(L.ROOT, "live", "news")
PV = "2026-10-26a"                    # bump when the prompt or schema changes
BATCH = 20
TICKERS = ("SPY", "QQQ", "IWM", "META", "NVDA", "MSFT", "AVGO", "SMH", "GLD")

SYSTEM = f"""You score market news headlines for a research log kept by an options-flow trader.
For each headline judge its likely effect on US equity index prices (SPY, QQQ, IWM) and on these
tickers: {", ".join(TICKERS)}. You see only the headline and its metadata at the moment it was first
seen -- judge it as a trader reading it then would.

Fields, for every headline you are given:
- relevance: 0 = no bearing on US index prices today; 1 = minor or background; 2 = could move the
  indexes or a listed ticker noticeably within the day; 3 = major and market-moving (an unexpected
  Fed decision, a military escalation involving a major power, a new tariff action, a large data surprise).
- direction: the expected direction for SPY if it matters: -2 strongly down, -1 down, 0 none or
  unclear, 1 up, 2 strongly up. Use 0 whenever relevance is 0.
- novelty: "new" = the first report of new information; "update" = a new detail on a known story;
  "rehash" = recap, opinion, preview, analyst commentary or a repeat.
- horizon: "minutes", "hours" or "days" -- how soon a price effect would show.
- tickers: which of the listed tickers the headline directly concerns; empty if none.
- why: at most 20 words.

Be calibrated: most headlines deserve 0 or 1. A dramatic tone is not substance. A preview of a
scheduled event is a rehash unless it carries new numbers. Return one entry per id, using the ids given."""

SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "relevance": {"type": "integer", "enum": [0, 1, 2, 3]},
            "direction": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
            "novelty": {"type": "string", "enum": ["new", "update", "rehash"]},
            "horizon": {"type": "string", "enum": ["minutes", "hours", "days"]},
            "tickers": {"type": "array", "items": {"type": "string", "enum": list(TICKERS)}},
            "why": {"type": "string"},
        },
        "required": ["id", "relevance", "direction", "novelty", "horizon", "tickers", "why"],
        "additionalProperties": False}}},
    "required": ["scores"],
    "additionalProperties": False,
}


def _read(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


def _utc(s):
    """news_logger time string ("2026-10-26T15:27:09Z") -> aware UTC datetime"""
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _et(s):
    try:
        return _utc(s).astimezone(L.NY).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


def _key(r):
    return f"{r.get('feed')}:{r.get('id')}"


def score_pass(day=None, max_batches=3):
    """score up to max_batches x BATCH of today's unscored relevant headlines; returns (n_scored, error)."""
    day = day or dt.datetime.now(L.NY).date()
    recs = [r for r in _read(os.path.join(NEWS, f"{day}.jsonl")) if r.get("cats") and r.get("title")]
    out_p = os.path.join(NEWS, "ai", f"{day}.jsonl")
    done = {x["key"] for x in _read(out_p)}
    todo, seen_keys = [], set()
    for r in sorted(recs, key=lambda r: r.get("seen") or ""):
        k = _key(r)
        if k not in done and k not in seen_keys:
            todo.append(r); seen_keys.add(k)
    n = 0
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    for b in range(max_batches):
        batch = todo[b * BATCH:(b + 1) * BATCH]
        if not batch:
            break
        ids = {f"h{i}": r for i, r in enumerate(batch)}
        items = [{"id": i, "first_seen_et": _et(r.get("seen")), "source": r.get("src"),
                  "headline": r["title"], "tags": r.get("cats"), "tickers": r.get("tickers") or []}
                 for i, r in ids.items()]
        try:
            obj, model = L.call_json("news_score", SYSTEM, json.dumps(items, ensure_ascii=False), SCHEMA,
                                     effort="low", max_tokens=8000)
        except L.LLMError as e:
            return n, str(e)
        now = dt.datetime.now(dt.timezone.utc)                  # news_logger's times are UTC ("...Z"): stay in UTC
        with open(out_p, "a", encoding="utf-8") as f:
            for s in obj.get("scores", []):
                r = ids.get(s.get("id"))
                if r is None:
                    continue
                seen = r.get("seen") or ""
                try:
                    lag = (now - _utc(seen)).total_seconds()
                except ValueError:
                    lag = None
                rec = {"key": _key(r), "feed": r.get("feed"), "id": r.get("id"), "t": r.get("t"), "seen": seen,
                       "scored_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "late": lag is None or lag > 300,
                       "model": model, "pv": PV, **{k: s[k] for k in ("relevance", "direction", "novelty",
                                                                    "horizon", "tickers", "why")}}
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                n += 1
    return n, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--batches", type=int, default=3)
    a = ap.parse_args()
    n, err = score_pass(dt.date.fromisoformat(a.date) if a.date else None, a.batches)
    print(f"scored {n}" + (f"  error: {err}" if err else ""))
