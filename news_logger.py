# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "python-dotenv"]
# ///
"""
news_logger.py
==============
Forward log of market headlines, with OUR OWN first-seen time (user,
2026-10-23). Both feeds keep little history (UW /news/headlines pages back to
~7 weeks, Webull's per-symbol feed 30 days), so a test of news -- above all an
AI relevance score, see news_filter -- needs a log we own. It also feeds the
viewer's news strip (live/news/latest.json).

    python news_logger.py              # run forever (systemd: cleanbot-news)
    python news_logger.py --once       # one poll of each feed, then exit
    python news_logger.py --no-wb      # UW only (anywhere with UW_API_KEY)
    python news_logger.py --out DIR    # write somewhere else (tests)

FEEDS
  UW   GET /api/news/headlines?limit=100 every 60 s; pages further back only
       while a whole page is new (catch-up after a restart, max 5 pages).
  WB   Webull stock-news list, page 1, every 180 s, for SPY QQQ IWM META NVDA
       MSFT AVGO SMH GLD (Benzinga; carries a sentiment label).

🚨 WEBULL TOKEN: READ-ONLY.
  The news endpoints need x-access-token. This process READS the bot's saved
  token (conf/token.txt) before each Webull poll and NEVER creates, refreshes
  or saves one -- the running bot owns that file (memory:
  cleanbot-webull-news-api). If the token is rejected, Webull polling pauses
  with one log line and UW keeps going. REST only: no streaming connection,
  so the bot's one-connection rule is untouched. Run from the bot's folder.

OUTPUT (live/, gitignored)
  live/news/YYYY-MM-DD.jsonl   every headline once (NY date of first sight):
      feed UW|WB, id, t (published, UTC ISO), seen (our first sight, UTC ISO),
      src, title, tickers, sent (WB label), major (UW), sym (WB feed symbol),
      cats (news_filter keywords), ai (null -- reserved for an AI score)
  live/news/latest.json        the last 300 RELEVANT items (cats non-empty),
      newest first, plus feed status -- what the viewer strip reads.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import logging
import os
import sys
import time
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

import news_filter as NF                                  # noqa: E402

NY = ZoneInfo("America/New_York")
UW_URL = "https://api.unusualwhales.com/api/news/headlines"
UW_EVERY, WB_EVERY, MAX_CATCHUP = 60, 180, 5
WB_SYMS = ("SPY", "QQQ", "IWM", "META", "NVDA", "MSFT", "AVGO", "SMH", "GLD")
KEEP_LATEST, LATEST_HOURS = 300, 48


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_time(s):
    """Any feed timestamp -> UTC ISO 'YYYY-MM-DDTHH:MM:SSZ' (None if unparseable)."""
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return iso(t)
    except ValueError:
        return None


class Log:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.seen = set()
        self.latest = collections.deque(maxlen=KEEP_LATEST)
        self.stat = {"UW": {"ok": None, "err": None, "new_today": 0},
                     "WB": {"ok": None, "err": None, "new_today": 0}}
        self.day = None
        now = dt.datetime.now(NY).date()
        for d in (now - dt.timedelta(days=2), now - dt.timedelta(days=1), now):    # restart: no duplicates
            p = os.path.join(out, f"{d.isoformat()}.jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                self.seen.add((r.get("feed"), r.get("id")))
                if r.get("cats"):
                    self.latest.appendleft(r)

    def add(self, rec):
        k = (rec["feed"], rec["id"])
        if k in self.seen:
            return False
        self.seen.add(k)
        rec["cats"] = NF.relevant(rec)
        rec["ai"] = None
        d = dt.datetime.now(NY).date().isoformat()
        if d != self.day:
            self.day = d
            for s in self.stat.values():
                s["new_today"] = 0
        with open(os.path.join(self.out, f"{d}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.stat[rec["feed"]]["new_today"] += 1
        if rec["cats"]:
            self.latest.appendleft(rec)
        return True

    def write_latest(self):
        cut = iso(utc_now() - dt.timedelta(hours=LATEST_HOURS))
        items = sorted((r for r in self.latest if (r.get("t") or r["seen"]) >= cut),
                       key=lambda r: r.get("t") or r["seen"], reverse=True)
        doc = {"updated": iso(utc_now()), "status": self.stat,
               "items": [{k: r.get(k) for k in ("feed", "t", "seen", "src", "title", "cats", "sent", "major",
                                                "sym", "tickers")} for r in items]}
        tmp = os.path.join(self.out, "latest.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.out, "latest.json"))


# ------------------------------------------------------------------ UW
def poll_uw(L, sess):
    key = os.environ.get("UW_API_KEY", "").strip()
    if not key:
        L.stat["UW"]["err"] = "no UW_API_KEY"
        return
    n_new = 0
    for page in range(MAX_CATCHUP):
        try:
            r = sess.get(UW_URL, params={"limit": 100, "page": page}, timeout=15,
                         headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
            if r.status_code != 200:
                L.stat["UW"]["err"] = f"HTTP {r.status_code}"
                return
            data = r.json().get("data") or []
        except Exception as e:                            # noqa: BLE001
            L.stat["UW"]["err"] = type(e).__name__
            return
        new_here = 0
        now = iso(utc_now())
        for x in data:
            t = norm_time(x.get("created_at"))
            head = x.get("headline") or ""
            rec = {"feed": "UW", "id": hashlib.sha1(f"{t}|{head}".encode()).hexdigest()[:16],
                   "t": t, "seen": now, "src": x.get("source"), "title": head,
                   "tickers": x.get("tickers") or [], "sent": None, "major": bool(x.get("is_major")),
                   "sym": None}
            new_here += L.add(rec)
        n_new += new_here
        if new_here < len(data) or not data:              # caught up with what we already have
            break
    L.stat["UW"]["ok"], L.stat["UW"]["err"] = iso(utc_now()), None
    return n_new


# ------------------------------------------------------------------ Webull
class WB:
    def __init__(self):
        logging.disable(logging.CRITICAL)                 # the SDK logs token fragments
        from webull.core.client import ApiClient
        from webull.core.request import ApiRequest
        from webull.core.http.initializer.token.token_manager import TokenManager
        self.ApiRequest, self.TokenManager = ApiRequest, TokenManager
        self.api = ApiClient(os.environ["WEBULL_APP_KEY"], os.environ["WEBULL_APP_SECRET"], region_id="us")
        self.api.add_endpoint("us", "api.webull.com")
        self.paused_for = None

    def poll(self, L):
        tok = self.TokenManager().load_token_from_local()      # READ ONLY -- never init / refresh / save
        if not tok or not tok.token:
            L.stat["WB"]["err"] = "no saved token (conf/token.txt) -- waiting for the bot"
            return
        if self.paused_for == tok.token:
            return                                              # rejected before; wait for the bot to renew it
        self.api.set_token(tok.token)
        now = iso(utc_now())
        for sym in WB_SYMS:
            req = self.ApiRequest("/market-data/news/stock-news/list", version="v3", method="GET", query_params={})
            req.add_query_param("symbol", sym)
            try:
                js = self.api.get_response(req).json()
            except Exception as e:                              # noqa: BLE001
                msg = str(e)
                if "INVALID_TOKEN" in msg or "401" in msg:
                    self.paused_for = tok.token
                    L.stat["WB"]["err"] = "token rejected -- paused until the bot renews it"
                    print(f"  📰 [NEWS] Webull token rejected; Webull polling paused until conf/token.txt changes", flush=True)
                    return
                L.stat["WB"]["err"] = type(e).__name__
                continue
            for x in js.get("data") or []:
                rec = {"feed": "WB", "id": str(x.get("news_id")), "t": norm_time(x.get("publish_time")),
                       "seen": now, "src": x.get("source"), "title": x.get("title") or "",
                       "tickers": x.get("related_symbols") or [], "sent": x.get("sentiment"),
                       "major": None, "sym": sym}
                L.add(rec)
            time.sleep(0.3)
        self.paused_for = None
        L.stat["WB"]["ok"], L.stat["WB"]["err"] = iso(utc_now()), L.stat["WB"]["err"] if L.stat["WB"]["err"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-wb", action="store_true")
    ap.add_argument("--out", default=os.path.join("live", "news"))
    a = ap.parse_args()
    L = Log(a.out)
    sess = requests.Session()
    wb = None
    if not a.no_wb:
        try:
            wb = WB()
        except Exception as e:                            # noqa: BLE001
            print(f"  📰 [NEWS] Webull disabled: {type(e).__name__}", flush=True)
    print(f"  📰 [NEWS] logger up -> {a.out}  (UW every {UW_EVERY}s"
          + (f", Webull every {WB_EVERY}s for {len(WB_SYMS)} symbols)" if wb else ", Webull off)"), flush=True)
    t_uw = t_wb = 0.0
    hb = time.time()
    while True:
        now = time.time()
        if now - t_uw >= UW_EVERY:
            poll_uw(L, sess)
            t_uw = now
        if wb and now - t_wb >= WB_EVERY:
            wb.poll(L)
            t_wb = now
        L.write_latest()
        if a.once:
            print(f"  📰 [NEWS] once: UW {L.stat['UW']}  WB {L.stat['WB']}  latest {len(L.latest)}", flush=True)
            return
        if now - hb >= 3600:
            hb = now
            print(f"  📰 [NEWS] heartbeat: today UW {L.stat['UW']['new_today']} / WB {L.stat['WB']['new_today']} new;"
                  f" errors UW {L.stat['UW']['err']} WB {L.stat['WB']['err']}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
