"""
check_oi_timing.py
==================
WHEN DOES THE NEW OPEN INTEREST ARRIVE? (operator, 2026-10-18)

The weekly gamma heatmap is snapshotted once, at 09:31 ET. The operator wants a
second, PRE-OPEN snapshot "as soon as the fresh OI is available" -- which is
only worth building once we know when that is, per source:

  * Webull option snapshots (the viewer's data source). NOTE: webull_viewer_data
    stamps every row `d = today`, so the existing freshness check in
    live_state._weekly_update_locked cannot tell new OI from Friday's; this
    probe watches the OI VALUES instead.
  * Unusual Whales per-contract history (the latest row's date and OI).

READ-ONLY. One run = one sample, appended to _oi_timing.jsonl (box-local,
not a research log). Webull REST only -- no streaming connection -- and run
from the bot's folder like every other box check. Meant to be fired every
15 min from 05:00 to 09:45 ET by a one-off systemd timer:

  systemd-run --unit=oi-timing --on-calendar="2026-10-19 05..09:00/15:00" \
      -p WorkingDirectory=/root/bot_engine/Cleanbot \
      /root/bot_engine/venv/bin/python check_oi_timing.py

Each sample records, per contract: open_interest, volume, bid/ask, IV, gamma
(bid/ask/IV going live at 07:30 is Cboe GTH's first morning) and any
date-like fields the snapshot carries; plus UW's latest history row.
`--report` prints when each source's OI changed vs the first sample.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NY = ZoneInfo("America/New_York")
OUT = os.path.join(ROOT, "_oi_timing.jsonl")
WATCH = [   # (ticker, expiry yymmdd, right, strike lo, hi) -- the QQQ legs from 10/16 + ATM bands
    ("QQQ", "261030", "P", 640, 730),
    ("SPY", "261023", "C", None, None),
    ("IWM", "261023", "P", None, None),
]
UW_SYMS = ["QQQ261030P00700000", "QQQ261030P00715000", "QQQ261030P00675000"]


def sample():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import webull_viewer_data as WV
    now = dt.datetime.now(NY)
    rec = dict(at=now.isoformat(timespec="seconds"), webull={}, uw={}, keys=None, errors=[])
    data = WV._client()
    for tk, exp6, right, lo, hi in WATCH:
        try:
            syms = WV._directory(tk)
            if lo is None:
                px = WV.spot(tk) or 0
                lo, hi = px * 0.98, px * 1.02
                rec.setdefault("spot", {})[tk] = px
            want = [s for s in syms if s[-15:-9] == exp6 and s[-9] == right
                    and lo <= int(s[-8:]) / 1000 <= hi]
            for i in range(0, len(want), WV.BATCH):
                for x in WV._call(data.option_market_data.get_option_snapshot,
                                  symbols=",".join(want[i:i + WV.BATCH]), category="US_OPTION") or []:
                    if not isinstance(x, dict) or not x.get("symbol"):
                        continue
                    if rec["keys"] is None:
                        rec["keys"] = sorted(x.keys())
                    dates = {k: x[k] for k in x if any(w in k.lower() for w in ("date", "time", "trade"))}
                    rec["webull"][x["symbol"]] = dict(
                        oi=x.get("open_interest"), vol=x.get("volume"), bid=x.get("bid"),
                        ask=x.get("ask"), iv=x.get("imp_vol"), gamma=x.get("gamma"),
                        dates=dates)
        except Exception as e:                   # noqa: BLE001 -- a probe never raises
            rec["errors"].append(f"webull {tk}: {type(e).__name__}")
    try:
        import requests                          # the box has no research deps (httpx)
        hdr = {"Authorization": f"Bearer {os.environ['UW_API_KEY']}", "Accept": "application/json"}
        for s in UW_SYMS:
            r = requests.get(f"https://api.unusualwhales.com/api/option-contract/{s}/historic",
                             headers=hdr, timeout=15)
            h = r.json().get("chains", []) if r.status_code == 200 else []
            if r.status_code != 200:
                rec["errors"].append(f"uw {s[-10:]}: HTTP {r.status_code}")
            last = max(h, key=lambda r: r["date"]) if h else {}
            rec["uw"][s] = dict(date=last.get("date"), oi=last.get("open_interest"))
    except Exception as e:                       # noqa: BLE001
        rec["errors"].append(f"uw: {type(e).__name__}")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"{rec['at']}  webull {len(rec['webull'])} contracts  uw {rec['uw']}  errors {rec['errors']}")


def report():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    if not rows:
        print("no samples"); return
    base = rows[0]
    print(f"{len(rows)} samples, first {base['at']}")
    for r in rows:
        w0, w = base["webull"], r["webull"]
        common = [s for s in w if s in w0]
        changed = sum(1 for s in common if w[s]["oi"] != w0[s]["oi"])
        live = sum(1 for s in w.values() if s.get("bid") not in (None, "", "0", 0))
        q700 = w.get("QQQ261030P00700000", {})
        print(f"  {r['at'][11:16]}  webull OI changed vs first: {changed:>3}/{len(common):<3}  "
              f"quotes with a bid: {live:>3}  QQQ 700P OI {q700.get('oi')} iv {q700.get('iv')}  "
              f"uw {[(k[-10:], v.get('date'), v.get('oi')) for k, v in r['uw'].items()]}"
              + (f"  ERR {r['errors']}" if r["errors"] else ""))
    print("snapshot keys:", rows[0].get("keys"))


if __name__ == "__main__":
    report() if "--report" in sys.argv else sample()
