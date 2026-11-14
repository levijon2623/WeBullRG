"""
cot_pull.py -- CFTC Commitments of Traders for the index futures we trade around (user, 2026-11-13).

Pulls every column of four COT datasets for E-mini S&P 500, Nasdaq-100, Russell 2000 and VIX futures
from the CFTC's public reporting site (publicreporting.cftc.gov, Tyler Data & Insights / Socrata) into
_cot_cache/{dataset}.json, one row per (report date, contract).

  TFF (Traders in Financial Futures) -- dealer / asset manager / leveraged funds / other / non-reportable.
      The right split for equity-index futures. Combined (futures + options) yw9f-hn96, futures only gpe5-46if.
      From 2006-06-13.
  Legacy -- commercial / non-commercial / non-reportable. Combined jun7-fc8e, futures only 6dca-aqww.
      From the 1990s.

TIMING: positions are as of TUESDAY and published FRIDAY 15:30 ET. A test may only use a report from the
Friday it was released, never its Tuesday date (store both; see `available`).

ACCESS: the SODA 2 endpoint (/resource/{id}.json) answers without a token, throttled. With APP_TOKEN (or COT_APP_TOKEN) in
.env (a free app token from publicreporting.cftc.gov/profile/edit/developer_settings; the user puts it there)
every call sends it as X-App-Token for a higher rate limit. The token is never printed.

    python cot_pull.py              # full history (first run)
    python cot_pull.py --refresh    # only reports newer than what is cached (the weekly job)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
OUT = os.path.join(ROOT, "_cot_cache")
BASE = "https://publicreporting.cftc.gov/resource"
DATASETS = {"tff_combined": "yw9f-hn96", "tff_futures": "gpe5-46if",
            "legacy_combined": "jun7-fc8e", "legacy_futures": "6dca-aqww"}
CODES = {"13874A": "E-mini S&P 500", "13874+": "S&P 500 consolidated",
         "209742": "Nasdaq-100 E-mini", "20974+": "Nasdaq-100 consolidated",
         "239742": "Russell 2000 E-mini (CME: 2006-08, 2017-)",
         "23977A": "Russell 2000 mini (ICE, 2008-07..2018-06: the CME gap)",
         "1170E1": "VIX futures",
         # Micros (user, 2026-11-14: retail and smaller funds use them), folded in as E-mini equivalents
         "13874U": "Micro E-mini S&P 500", "209747": "Micro E-mini Nasdaq-100", "239747": "Micro E-mini Russell 2000"}
PAGE = 5000


def session():
    s = requests.Session()
    s.headers["User-Agent"] = "cot_pull/1.0"
    # the CFTC app token, as the user named it in .env (APP_TOKEN, 2026-11-14). Only the APP token is sent:
    # the API key/secret pairs (CTFC_API_KEY / _SECRET, APP_SECRET_TOKEN) are for private data and stay unused.
    tok = (os.getenv("APP_TOKEN") or os.getenv("COT_APP_TOKEN") or "").strip()
    if tok:
        s.headers["X-App-Token"] = tok
    return s, bool(tok)


def fetch(s, ds, since=None):
    """All rows for our contract codes, paged; `since` = only report dates after it (ISO date)."""
    codes = ",".join(f"'{c}'" for c in CODES)
    where = f"cftc_contract_market_code in({codes})"
    if since:
        where += f" AND report_date_as_yyyy_mm_dd > '{since}T00:00:00.000'"
    rows, offset = [], 0
    while True:
        params = {"$where": where, "$order": "report_date_as_yyyy_mm_dd, cftc_contract_market_code",
                  "$limit": PAGE, "$offset": offset}
        for attempt in range(4):
            try:
                r = s.get(f"{BASE}/{ds}.json", params=params, timeout=60)
                if r.status_code == 200:
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise SystemExit(f"{ds}: HTTP {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                time.sleep(5 * (attempt + 1))
                if attempt == 3:
                    raise SystemExit(f"{ds}: {type(e).__name__}")
        page = r.json()
        rows += page
        if len(page) < PAGE:
            return rows
        offset += PAGE


def tidy(rows):
    """Add `report_date` (the Tuesday) and `available` (the Friday 15:30 ET it was published, by
    the usual schedule -- a holiday week can slip a day; the conservative use is the NEXT session)."""
    for r in rows:
        d = dt.date.fromisoformat(str(r["report_date_as_yyyy_mm_dd"])[:10])
        r["report_date"] = d.isoformat()
        r["available"] = (d + dt.timedelta(days=(4 - d.weekday()) % 7 or 7)).isoformat() + "T15:30-ET"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    s, tok = session()
    print(f"COT pull ({'app token' if tok else 'no token: anonymous, throttled'})", flush=True)
    for name, ds in DATASETS.items():
        path = os.path.join(OUT, f"{name}.json")
        have = json.load(open(path)) if (a.refresh and os.path.exists(path)) else []
        since = max((r["report_date"] for r in have), default=None) if a.refresh else None
        t0 = time.time()
        new = tidy(fetch(s, ds, since))
        keep = {(r["report_date"], r["cftc_contract_market_code"]) for r in new}
        rows = [r for r in have if (r["report_date"], r["cftc_contract_market_code"]) not in keep] + new
        rows.sort(key=lambda r: (r["report_date"], r["cftc_contract_market_code"]))
        tmp = path + ".tmp"
        json.dump(rows, open(tmp, "w"))
        os.replace(tmp, path)
        by = {}
        for r in rows:
            by.setdefault(r["cftc_contract_market_code"], []).append(r["report_date"])
        span = " | ".join(f"{c} {len(v)}w {min(v)}..{max(v)}" for c, v in sorted(by.items()))
        print(f"  {name:16s} {ds}: +{len(new)} rows ({time.time() - t0:.0f}s), {len(rows)} total | {span}", flush=True)


if __name__ == "__main__":
    main()
