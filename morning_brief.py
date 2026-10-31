"""
morning_brief.py
================
PRE-OPEN BRIEF (user, 2026-10-26): overnight positioning, premarket, volatility and news for
S&P (SPY + SPX), QQQ and IWM, written by Claude at ~09:05 ET from numbers our own code computed.

DIGEST (code, deterministic): every number in the brief comes from here.
  * positioning_map's pre-open build (live/posmap/archive/DATE/{SP,QQQ,IWM}.json): the biggest
    overnight open-interest changes with side, multi-leg share and premium, packages / rolls,
    per-expiry ATM IV, IVRP (ATM IV - 21-day realized), gamma flip, dealer gamma at spot
  * premarket bars (UW 1-minute, market_time "pr") vs the last close and the implied-move band
    quoted at yesterday's close (live/implied_move.json "pm", calibrated: check_implied_move)
  * relevant headlines since yesterday's close (news_logger + news_score)
  * FLAGS, computed here, are the only things the brief may call anomalies (band breaks, spot
    below the flip, IVRP extremes, freshly built positions, big packages). The archive only starts
    2026-10-23, so there is no history to call a size "unusual" against yet: the brief says biggest.
BRIEF (Claude Opus 5.5): narrative from the digest only, inferences labelled, plus what our own
  tests found (house findings) so the brief never leans on a claim we've tested and dropped.
Output: live/brief/DATE.json (digest + brief) and live/brief/DATE.md.

Usage:  python morning_brief.py                 # today, now
        python morning_brief.py --date 2026-10-26 --digest-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

import llm_common as L
import market_calendar as MC

LIVE = os.path.join(L.ROOT, "live")
MARKETS = {"SP": "SPY", "QQQ": "QQQ", "IWM": "IWM"}
UW_API = "https://api.unusualwhales.com/api"
SIDE_MIN = 0.60                                   # as positioning_map: share of side-known volume
PACKAGE_FLAG = {"SP": 20000, "QQQ": 10000, "IWM": 5000}
NEW_FLAG = {"SP": 50000, "QQQ": 15000, "IWM": 10000}   # contracts: a fresh position this size gets a flag
# the UOA page's universe (uoa_flags.UNIVERSE -- not imported: this venv has no numpy/requests)
UOA_UNIVERSE = ("SPY", "QQQ", "IWM", "META", "NVDA", "MSFT", "AVGO", "SMH", "GLD",
                "TSLA", "AAPL", "ORCL", "TLT", "PLTR", "AMZN", "IBIT", "AMD", "SLV", "MU", "MSTR", "INTC", "GOOGL",
                "HOOD", "XLF", "PCG", "GME", "USO", "COIN", "NFLX", "TSM", "XLE", "WMT", "MRVL", "BE")
UOA_FLAG_MAX = 2                                  # at most this many UOA items join `flags`

HOUSE = [
    "Dealer gamma sizes moves, it doesn't call direction: S&P moved 39 bp on positive-gamma days vs 68 bp on "
    "negative; 30-minute ranges ran ~26% wider with negative gamma at spot (check_currents, check_gamma_modes).",
    "Peak gamma is not a magnet: price reaches it no more often than any level at that distance (check_peak_gamma).",
    "Charm/vanna flows did not predict direction (check_currents).",
    "Unusual options activity whose open interest stuck predicted nothing over 1-5 days; copying bought "
    "contracts lost money (check_uoa_events).",
    "The implied-move band (q x ATM straddle) is calibrated: ~65% / ~90% of sessions stay inside (check_implied_move).",
    "Opening range breakouts: drift only, no edge (check_orb).",
    "Headline categories alone showed no tradeable price reaction (check_news_reaction).",
    "Big one-sided option bursts (the UOA flags) reached +100..+300% exactly as often as unprinted contracts "
    "of the same profile, bought right after the print or the next morning once open interest confirmed, and "
    "slightly less often than unprinted same-day twins (check_uoa_paths, 2023-10..2026-09).",
    "Open interest is net and published start-of-day: side (ask = bought, bid = sold) is an inference, and "
    "one session's volume is only judged by the next session's open interest.",
]

SYSTEM = """You write the pre-open brief for one trader who runs an options-flow trading bot and reads
dealer-positioning maps. You get a JSON digest computed by their own code. Write from the digest only.

Rules:
- Every number you write must appear in the digest (rounding is fine). Never invent levels, flows or news.
- Only items in `flags` may be called anomalies or unusual. Everything else is "biggest", "notable" or plain fact.
- Side (bought/sold) and opening/closing are inferences: say "at the ask" or "likely bought", not "bought".
- Use the house findings: never present gamma, charm, peak gamma or unusual activity as a direction call.
  Gamma speaks to how big moves may be, not which way.
- S&P strikes are in SPY units unless the digest names an SPX strike. Times are ET.
- Plain, compact language. No hype, no advice, no price targets.
- If a part of the digest is missing, say so in one short line rather than guessing; otherwise don't mention it.
- `uoa` lists yesterday's UOA flags (big one-sided bursts on single contracts) and whether this morning's open
  interest confirmed them. Put them in overnight_positioning: the biggest few, who was on which side, and what
  stuck. A flag spanning an earnings report (`er`) is mostly an event bet -- say so. Never suggest following them.
- `analysts` lists overnight rating changes on the tickers we follow: mention upgrades / downgrades / initiations
  in the news section as plain facts.

Shape: it is read in two minutes before the open.
- headline: at most 14 words. summary: at most 3 sentences.
- anomalies: pick the 3 to 6 flags that matter most for today's session (fewer only if `flags` has fewer);
  `flag` copies the flag text, `what` names the contract or level, `why_it_matters` is one sentence.
- sections: fill every one of the five with 2 to 5 bullets. Each bullet is one self-contained sentence; name the
  market (SPY / SPX / QQQ / IWM) when it is about one, and don't prefix market-wide news with a list of tickers.
  No sub-bullets, no bullet that only introduces the next ones.
- SPY, QQQ and IWM are ETFs; premarket prices are ETF prices, not futures."""

SECTIONS = {"premarket": "Premarket", "overnight_positioning": "Overnight positioning", "volatility": "Volatility",
            "news": "News", "watch_today": "Watch today"}
SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "sections": {"type": "object",
                     "properties": {k: {"type": "array", "items": {"type": "string"}} for k in SECTIONS},
                     "required": list(SECTIONS), "additionalProperties": False},
        "anomalies": {"type": "array", "items": {
            "type": "object",
            "properties": {"flag": {"type": "string"}, "what": {"type": "string"}, "why_it_matters": {"type": "string"}},
            "required": ["flag", "what", "why_it_matters"], "additionalProperties": False}},
    },
    "required": ["headline", "summary", "sections", "anomalies"],
    "additionalProperties": False,
}


def _load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jsonl(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


def _uw(path, **params):
    key = os.environ.get("UW_API_KEY", "").strip()
    req = urllib.request.Request(f"{UW_API}{path}?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:                                          # noqa: BLE001 -- never echo the request
        return {}


def _label(c, cols):
    x = dict(zip(cols, c))
    k = x["k"]
    return f"{x['root']} {x['exp'][5:].replace('-', '/')} {k:g}{x['cp']}", x


def built_pre(day):
    """has positioning_map's pre-open build run for every market today?"""
    a = os.path.join(LIVE, "posmap", "archive", day.isoformat())
    return all(os.path.exists(os.path.join(a, f"{k}.json")) for k in MARKETS)


def market_digest(key, day):
    d = _load(os.path.join(LIVE, "posmap", "archive", day.isoformat(), f"{key}.json"))
    if not d:
        return None, [f"{key}: no pre-open positioning build for {day}"]
    flags = []
    out = {"etf": d["etf"], "built_et": d.get("built_et"), "prev_close": d.get("prev_close"),
           "spot_at_build": d.get("spot"), "rv21": d.get("rv21"), "oi_change_from_session": d.get("oi_change_session"),
           "index_ratio": d.get("ratio") if key == "SP" else None}
    exps = []
    for e in d.get("expiries", [])[:6]:
        if e.get("dte", 99) > 10:
            continue
        exps.append({"expiry": e["e"], "dte": e["dte"], "atm_iv": e.get("atm_iv"), "ivrp": e.get("ivrp"),
                     "gamma_flip": e.get("flip"), "dealer_gamma_at_spot": "positive" if (e.get("g_spot") or 0) > 0 else "negative",
                     "oi_change_contracts": e.get("d_oi")})
    out["expiries"] = exps
    if exps:
        e0 = exps[0]
        if e0["dealer_gamma_at_spot"] == "negative":
            flags.append(f"{key}: dealer gamma negative at spot for the {e0['expiry']} expiry (flip {e0['gamma_flip']})")
        if e0.get("ivrp") is not None and (e0["ivrp"] > 5 or e0["ivrp"] < 0):
            flags.append(f"{key}: nearest-expiry IVRP {e0['ivrp']:+.1f} vol pts (outside 0..+5)")
    cols = d["detail_cols"]
    rows = sorted(d.get("detail", {}).values(), key=lambda c: -abs(c[cols.index("d_oi")] or 0))
    top = []
    for i, c in enumerate(rows[:10]):
        lab, x = _label(c, cols)
        side = "at the ask" if x["ask"] >= SIDE_MIN else "at the bid" if x["bid"] >= SIDE_MIN else "mixed"
        dte = (dt.date.fromisoformat(x["exp"]) - day).days
        item = {"contract": lab, "strike_etf_units": x["k_etf"], "dte": dte, "oi_now": x["oi"], "oi_change": x["d_oi"],
                "volume": x["vol"], "side": side, "multi_leg_share": x["mleg"], "premium_usd": x["prem"]}
        top.append(item)
        if i < 3 and x["d_oi"] >= NEW_FLAG[key] and x["d_oi"] >= 0.6 * x["oi"]:
            flags.append(f"{key}: {lab} is mostly a new position (+{x['d_oi']:,} of {x['oi']:,} OI), {side}")
    out["biggest_oi_changes"] = top
    pk = []
    for p in sorted(d.get("packages", []), key=lambda p: -p.get("size", 0))[:5]:
        legs = []
        for s in p["legs"]:
            c = d["detail"].get(s)
            legs.append(_label(c, cols)[0] if c else s)
        pk.append({"kind": p["kind"], "legs": legs, "size": p["size"]})
        if p.get("size", 0) >= PACKAGE_FLAG[key]:
            flags.append(f"{key}: {p['kind'].lower()} of {p['size']:,} contracts ({' / '.join(legs)})")
    out["packages_inferred"] = pk
    return out, flags


def premarket(etf, day, now, prev_close, band):
    bars = (_uw(f"/stock/{etf}/ohlc/1m", date=day.isoformat(), limit=2500) or {}).get("data") or []
    cut = now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")       # bars that closed before the brief
    pr = [b for b in bars if b.get("market_time") == "pr" and str(b.get("start_time") or "")[:16] < cut]
    if not pr or not prev_close:
        return None, []
    pr.sort(key=lambda b: b.get("start_time") or "")
    last = float(pr[-1]["close"])
    hi = max(float(b["high"]) for b in pr); lo = min(float(b["low"]) for b in pr)
    out = {"last": round(last, 2), "as_of_utc": str(pr[-1].get("start_time"))[:16], "change_pct": round((last / prev_close - 1) * 100, 2),
           "high": round(hi, 2), "low": round(lo, 2), "volume": int(sum(float(b.get("volume") or 0) for b in pr))}
    flags = []
    if band and band.get("session") == day.isoformat():
        lv = band["levels"]
        out["implied_band_from_close"] = {"pct_move": band.get("pct"), **lv}
        if last > lv["hi90"] or last < lv["lo90"]:
            flags.append(f"{etf}: premarket {last:.2f} outside the 90% implied band ({lv['lo90']}-{lv['hi90']})")
        elif last > lv["hi65"] or last < lv["lo65"]:
            flags.append(f"{etf}: premarket {last:.2f} outside the 65% implied band ({lv['lo65']}-{lv['hi65']})")
    return out, flags


def news(day, now):
    """relevant headlines first seen between the previous session's 16:00 ET close and `now`.
    news_logger's times are UTC ("...Z"); files are named by the ET date."""
    prev = day - dt.timedelta(days=3 if day.weekday() == 0 else 1)
    since = dt.datetime(prev.year, prev.month, prev.day, 16, 0, tzinfo=L.NY)
    recs = []
    for i in range((day - prev).days + 1):
        recs += _jsonl(os.path.join(LIVE, "news", f"{prev + dt.timedelta(days=i)}.jsonl"))
    ai = {}
    for i in range((day - prev).days + 1):
        ai.update({x["key"]: x for x in _jsonl(os.path.join(LIVE, "news", "ai", f"{prev + dt.timedelta(days=i)}.jsonl"))})
    items = []
    for r in recs:
        try:
            seen = dt.datetime.fromisoformat(r.get("seen") or "")
        except ValueError:
            continue
        seen = seen if seen.tzinfo else seen.replace(tzinfo=dt.timezone.utc)
        if not r.get("cats") or not (since <= seen <= now):
            continue
        s = ai.get(f"{r.get('feed')}:{r.get('id')}")
        items.append({"seen_et": seen.astimezone(L.NY).strftime("%m-%d %H:%M"), "source": r.get("src"),
                      "headline": r.get("title"), "tags": r.get("cats"), "claude_relevance": s and s["relevance"],
                      "claude_direction": s and s["direction"], "novelty": s and s["novelty"]})
    # most relevant first (unscored last), then oldest first; cap the list
    items.sort(key=lambda x: (-(x["claude_relevance"] if x["claude_relevance"] is not None else -1), x["seen_et"]))
    return items[:30], len(items)


def digest(day, now):
    band_all = _load(os.path.join(LIVE, "implied_move.json")) or {}
    g = {"date": day.isoformat(), "made_et": now.strftime("%H:%M"), "markets": {}, "flags": [], "missing": []}
    for key, etf in MARKETS.items():
        m, fl = market_digest(key, day)
        if m is None:
            g["missing"] += fl
            continue
        pm, fl2 = premarket(etf, day, now, m["prev_close"], (band_all.get(etf) or {}).get("pm"))
        m["premarket"] = pm
        if pm is None:
            g["missing"].append(f"{etf}: no premarket bars yet")
        g["markets"][key] = m
        g["flags"] += fl + fl2
    g["news"], g["news_total"] = news(day, now)
    g["uoa"], fl3 = uoa(day)
    g["flags"] += fl3
    g["analysts"] = analysts(day, now)
    g["house_findings"] = HOUSE
    return g


def _money(v):
    return f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}k"


def uoa(day):
    """yesterday's UOA flags (uoa_flags.py) with this morning's OI confirmation. Returns (digest part, flags)."""
    prev = MC.previous_trading_day(day - dt.timedelta(days=1))      # "on or before" -> strictly before
    doc = _load(os.path.join(LIVE, "uoa", f"{prev}.json"))
    if not doc or not doc.get("flags"):
        return {"date": prev.isoformat(), "missing": "no UOA file for the last session"}, []
    fl = [f for f in doc["flags"] if f.get("tdte", 0) >= 3]          # the user's rule: UOA needs time
    by = {}
    for f in fl:
        b = by.setdefault(f["tk"], {"flags": 0, "bought_calls": 0, "bought_puts": 0, "sold_calls": 0, "sold_puts": 0})
        b["flags"] += 1
        b[f"{f['side']}_{'calls' if f['cp'] == 'C' else 'puts'}"] += f["prem"]
    status = {}
    for f in fl:
        status[f.get("status") or "pending"] = status.get(f.get("status") or "pending", 0) + 1

    def row(f):
        return {"when": f["t"], "contract": f"{f['tk']} {f['exp'][5:].replace('-', '/')} {f['k']:g}{f['cp']}",
                "side": f["side"], "contracts": f["vol"], "premium": _money(f["prem"]), "avg_price": f["px"],
                "otm_pct": round(f["otm"] * 100, 1), "trading_days_left": f["tdte"],
                "next_day_oi_change": f.get("doi"), "verdict": f.get("status") or "OI not posted yet",
                **({"earnings": f["er"]} if f.get("er") else {})}
    top = sorted(fl, key=lambda f: -f["prem"])[:10]
    out = {"date": prev.isoformat(), "flags": len(fl), "short_dated_left_out": len(doc["flags"]) - len(fl),
           "bought_premium": _money(sum(f["prem"] for f in fl if f["side"] == "bought")),
           "sold_premium": _money(sum(f["prem"] for f in fl if f["side"] == "sold")),
           "verdicts": status,
           "by_ticker": {t: {k: (_money(v) if k != "flags" and v else v) for k, v in b.items() if v}
                         for t, b in sorted(by.items(), key=lambda kv: -sum(x for k, x in kv[1].items() if k != "flags"))[:8]},
           "largest": [row(f) for f in top],
           "definition": "one contract, ~5 min: >= 1,000 contracts and >= $100k, >= 70% at the ask (bought) or bid "
                         "(sold), mostly single-leg, >= the day's starting OI, 0-10% OTM; >= 3 trading days left"}
    flags = [f"UOA {r['contract']}: {r['contracts']:,} {r['side']} at {r['when']} ({r['premium']}), "
             f"next-day OI {r['next_day_oi_change']:+,} ({r['verdict']})"
             for r in out["largest"] if r["verdict"] in ("opened", "written")][:UOA_FLAG_MAX]
    return out, flags


def analysts(day, now):
    """overnight analyst actions on the UOA universe (UW /screener/analysts, since the last close)."""
    since = dt.datetime.combine(MC.previous_trading_day(day - dt.timedelta(days=1)), dt.time(16, 0), tzinfo=L.NY).astimezone(dt.timezone.utc)
    rows, older = [], None
    for _ in range(4):                                   # 500 a page; earnings season fills a page in ~13 h
        params = {"limit": 500, "newer_than": since.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if older:
            params["older_than"] = older
        page = (_uw("/screener/analysts", **params) or {}).get("data") or []
        rows += page
        if len(page) < 500:
            break
        older = min(x["timestamp"] for x in page)
    keep, other = [], {}
    for x in rows:
        if x.get("ticker") not in UOA_UNIVERSE or x.get("timestamp", "") < since.strftime("%Y-%m-%dT%H:%M:%S"):
            continue
        if x.get("action") in ("upgraded", "downgraded", "initiated"):
            keep.append({"ticker": x["ticker"], "action": x["action"], "firm": x.get("firm"),
                         "rating": x.get("recommendation"), "target": x.get("target")})
        else:
            other[x["ticker"]] = other.get(x["ticker"], 0) + 1
    return {"since_et": since.astimezone(L.NY).strftime("%m-%d %H:%M"), "changes": keep[:15],
            "maintained_or_target_only": other}


def to_markdown(day, brief):
    md = [f"# {brief['headline']}", f"_{day} pre-open brief_", "", brief["summary"], ""]
    if brief["anomalies"]:
        md.append("## Flags")
        md += [f"- **{a['what']}**: {a['why_it_matters']}" for a in brief["anomalies"]]
        md.append("")
    for k, title in SECTIONS.items():
        md.append(f"## {title}")
        md += [f"- {b}" for b in brief["sections"].get(k, [])]
        md.append("")
    return "\n".join(md)


def run(day=None, now=None, digest_only=False):
    now = now or dt.datetime.now(L.NY)
    day = day or now.date()
    g = digest(day, now)
    out_dir = os.path.join(LIVE, "brief")
    os.makedirs(out_dir, exist_ok=True)
    if digest_only:
        return g, None
    brief, model = L.call_json("morning_brief", SYSTEM, json.dumps(g, ensure_ascii=False), SCHEMA,
                               effort="high", max_tokens=16000)
    doc = {"date": day.isoformat(), "made_et": now.strftime("%H:%M"), "model": model, "brief": brief, "digest": g}
    tmp = os.path.join(out_dir, f"{day}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, os.path.join(out_dir, f"{day}.json"))
    with open(os.path.join(out_dir, f"{day}.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(day, brief))
    return g, brief


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--at", help="HH:MM ET to treat as now (news cutoff), with --date")
    ap.add_argument("--digest-only", action="store_true")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date) if a.date else None
    now = None
    if day and a.at:
        h, m = map(int, a.at.split(":"))
        now = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=L.NY)
    g, brief = run(day, now, a.digest_only)
    print(json.dumps(g, ensure_ascii=False, indent=1)[:6000] if a.digest_only else to_markdown(g["date"], brief))
