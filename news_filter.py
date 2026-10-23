"""
news_filter.py
==============
Keyword relevance for headlines (user, 2026-10-23): which of ~1,000 headlines
a day belong in the viewer's news strip, and the `cats` field of the news log.

check_news_reaction (2026-10-23) found raw headlines carry no tradeable signal
and the few that matter (ISM, the Fed chair, Iran / sanctions) price inside
their own minute -- so this is CONTEXT for the operator, not a bot input. The
log keeps every headline, filtered or not, so an AI relevance score can be
tried on the history later (the user chose: keywords now, AI once a key is in
.env -- see the `ai` field news_logger leaves null).

Rules, fixed and deliberately plain:
  UW Tradex / NYT / IBD / UW's own   -> categories below; GEO and POLICY count
                                        only when UW marks the item is_major
  UW press-release wires             -> only when they name a watched ticker AND
                                        hit the company-news words
  Webull (Benzinga) SPY / QQQ / IWM  -> categories below (GEO / POLICY too)
  Webull single-name feeds           -> the ticker or company named in the
                                        title, or the company-news words
"""
from __future__ import annotations

import re

ETFS = ("SPY", "QQQ", "IWM")
NAMES = {"META": "meta", "NVDA": "nvidia", "MSFT": "microsoft", "AVGO": "broadcom",
         "SMH": "semiconductor", "GLD": "gold"}
WATCH = set(ETFS) | set(NAMES)
WIRES = {"Business Wire", "PR NewsWire", "GlobeNewswire"}
NEWSY = {"Tradex", "NYT", "Investors.com", "Unusual Whales"}


def _rx(words):
    return re.compile(r"(?<![\w-])(" + words + r")(?![\w-])", re.I)


CATS = [
    ("FED", _rx(r"fed|fed's|fomc|federal reserve|powell|warsh|waller|jefferson|bowman|goolsbee|kashkari"
                r"|musalem|hammack|bostic|rate cuts?|rate hikes?|dot plot|beige book")),
    ("DATA", _rx(r"cpi|core pce|pce|ppi|nonfarm|non-farm|payrolls|jobless claims|initial claims"
                 r"|unemployment rate|jolts|ism manufacturing|ism services|pmi|gdp|retail sales"
                 r"|consumer confidence|consumer sentiment|durable goods|housing starts|inflation")),
    ("TRADE", _rx(r"tariffs?|trade deal|trade war|trade talks|trade agreement|export controls?"
                  r"|sanctions?|section 232|section 301")),
    ("RATES", _rx(r"treasury|treasuries|10-year|2-year|30-year|yields?|bond auction|bessent"
                  r"|debt ceiling|government shutdown|shutdown")),
    ("OIL", _rx(r"oil|crude|brent|wti|opec\+?|natural gas")),
    ("MKT", _rx(r"moc imbalance|circuit breaker|limit down|limit up|market-wide halt")),
    # first live test (10-23) let through 17% of UW, mostly UK politics and the
    # Ukraine war: GEO keeps the oil / risk-off flashpoints, POLICY only binding acts
    ("GEO", _rx(r"iran|iranian|israel|hormuz|taiwan|north korea|nuclear|invasion")),
    ("POLICY", _rx(r"executive order|supreme court")),
]
MAJOR_ONLY = {"GEO", "POLICY"}
CO_NEWS = _rx(r"earnings|eps|revenue|guidance|outlook|beats?|misses|downgraded?|upgraded?"
              r"|price target|antitrust|probe|recall|layoffs?|buyback|stock split")


# Tradex carries every country's data ("UK CONSTRUCTION PMI", "ITALIAN RETAIL SALES"):
# DATA / RATES / FED only count for the US unless the headline also says US
FOREIGN = _rx(r"uk|u\.k\.|boe|britain|british|euro ?zone|euro area|ecb|german|germany|french|france"
              r"|italian|italy|spanish|spain|japan|japanese|boj|china|chinese|pboc|canada|canadian|boc"
              r"|australia|australian|rba|india|indian|swiss|snb|sweden|swedish|norway|norwegian"
              r"|brazil|mexico|korea|korean|new zealand|rbnz|turkey|turkish|poland|south africa")
US = _rx(r"us|u\.s\.|united states|american|fed|fomc|treasury|treasuries")
DOMESTIC = {"DATA", "RATES", "FED"}


def categories(text: str, major: bool = True) -> list[str]:
    out = []
    foreign = bool(FOREIGN.search(text or "")) and not US.search(text or "")
    for name, rx in CATS:
        if name in MAJOR_ONLY and not major:
            continue
        if foreign and name in DOMESTIC:
            continue
        if rx.search(text or ""):
            out.append(name)
    return out


def relevant(rec: dict) -> list[str]:
    """`rec` is a news_logger record. -> categories (empty = not for the strip)."""
    title = rec.get("title") or ""
    feed, src = rec.get("feed"), rec.get("src")
    tks = [t for t in (rec.get("tickers") or []) if t in WATCH]
    if feed == "UW":
        if src in NEWSY:
            return categories(title, bool(rec.get("major")))
        if src in WIRES and tks and CO_NEWS.search(title):
            return ["CO:" + tks[0]]
        return []
    if feed == "WB":
        sym = rec.get("sym")
        if sym in ETFS:
            return categories(title, True)
        if sym in NAMES:
            low = title.lower()
            if sym.lower() in re.findall(r"[a-z]+", low) or NAMES[sym] in low or CO_NEWS.search(title):
                return ["CO:" + sym]
        return []
    return []
