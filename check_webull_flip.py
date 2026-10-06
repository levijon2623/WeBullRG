"""
check_webull_flip.py
====================
Can the viewer's GAMMA FLIP be computed from WEBULL data (gamma_flip.py) instead
of read from UW's /gex-levels? It is the last level the Webull-data viewer
lacks (it shows "all-expiry walls and gamma flip: Unusual Whales only").

METHOD under test (gamma_flip.py): dealers long calls / short puts; net GEX
re-priced with Black-Scholes at hypothetical spots +-8% (0.1% steps), each
contract's Webull IV held fixed; flip = the zero crossing nearest spot.

THE CHAIN WINDOW -- fixed before any paired sample
    PRIMARY   expiries 0..7 calendar days (today's included, T floored at 30
              min), strikes within +-5% of spot.
    REPORTED  0..14d +-8%, 0..7d +-8%, 0..14d +-5% (from the same pull).
    Chosen from a 3-point SMOKE TEST on 2026-10-03 (Friday-close Webull
    chain vs UW's 16:00 flip): 0..7d +-5% landed 0.1-0.7 from UW on SPY /
    QQQ / IWM, where wider windows put IWM's flip at 291-303 against UW's
    280.83 (long-dated IWM put OI turns net gamma negative). Picked on those
    3 points, so the paired sessions below are the validation.

SAMPLES (on the box, from the bot folder -- REST only, shared token)
    every 15 min, 09:45-15:45 ET, SPY / QQQ / IWM, sessions 2026-10-05 and
    2026-10-06. Each Webull flip is paired with the bot's own UW poll
    (gex_levels_log.jsonl, READ only) nearest in time, within 5 minutes.
    Records go to _webull_gex_cache/flip-YYYY-MM-DD.jsonl.

PRE-REGISTERED CRITERIA (per ticker, both sessions pooled; fixed 2026-10-03)
    F1  median |ours - UW gamma_flip| / spot <= 0.25%
    F2  ours on the same side of spot as UW's flip in >= 80% of pairs
    F3  >= 30 pairs
    PASS = F1-F3 on all three tickers -> draw the Webull flip in Webull mode,
    labelled as computed. Samples 15 min apart are autocorrelated (OI is
    fixed for the day), so two sessions are ~2 independent looks per ticker:
    a pass means "agrees with UW", not "is right".

RUN LOG
  2026-10-06 report (sessions 10-05 + 10-06, 50 pairs per ticker, 13,227
    Webull calls, 0 failures). PRIMARY 7d5: SPY median |diff| 0.058% of spot,
    same side 100% PASS; IWM 0.183%, 88% PASS; QQQ 0.262% (> 0.25%), same
    side 100% -> FAIL on F1. VERDICT: FAIL (all three required). Other
    windows (reported): 14d8 / 14d5 bring QQQ under 0.25% but blow IWM out
    to 2.4-3.0% (44% same side) -- the long-dated IWM put OI again; 7d8 is
    ~7d5. No window passes all three, so no window is swapped in.

REPORTED: the same for the other windows; |ours - nearest of UW's
nearby_flips|; net GEX at spot sign agreement with UW's flip side; calls and
failures per pass.

Usage (on the box, from the bot folder):
  python check_webull_flip.py --sample            # one pass now
  python check_webull_flip.py --loop              # until 15:45 ET
  python check_webull_flip.py --report 2026-10-05 2026-10-06
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
TICKERS = ("SPY", "QQQ", "IWM")
OUT_DIR = "_webull_gex_cache"
LEVELS_LOG = "gex_levels_log.jsonl"
WINDOWS = {"7d5": (7, 0.05), "14d8": (14, 0.08), "7d8": (7, 0.08), "14d5": (14, 0.05)}
PRIMARY = "7d5"
FIRST, LAST, EVERY = (9, 45), (15, 45), 15 * 60
PAIR_S = 300


def _exp(s):
    return dt.date(2000 + int(s[-15:-13]), int(s[-13:-11]), int(s[-11:-9]))


def sample():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import gamma_flip as GF
    import webull_viewer_data as WV
    now = dt.datetime.now(NY)
    data = WV._client()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"flip-{now.date().isoformat()}.jsonl")
    for tk in TICKERS:
        c0, f0 = WV.STATS["calls"], WV.STATS["fails"]
        spot = WV.spot(tk)
        if not spot:
            continue
        maxd = max(d for d, _b in WINDOWS.values())
        band = max(b for _d, b in WINDOWS.values())
        want = [s for s in WV._directory(tk)
                if 0 <= (_exp(s) - now.date()).days <= maxd and abs(int(s[-8:]) / 1000 / spot - 1) <= band]
        got = {}
        for i in range(0, len(want), WV.BATCH):
            for x in WV._call(data.option_market_data.get_option_snapshot,
                              symbols=",".join(want[i:i + WV.BATCH]), category="US_OPTION") or []:
                if isinstance(x, dict) and x.get("symbol"):
                    got[x["symbol"]] = x
            time.sleep(WV.PAUSE)
        rows = []
        for s in want:
            x = got.get(s) or {}
            T = (dt.datetime.combine(_exp(s), dt.time(16, 0), NY) - now).total_seconds() / (365 * 86400)
            rows.append((s, int(s[-8:]) / 1000, T, WV._num(x.get("imp_vol")), WV._num(x.get("open_interest")),
                         s[-9] == "C"))
        res = {}
        for lab, (d, b) in WINDOWS.items():
            sub = [(k, T, v, o, c) for s, k, T, v, o, c in rows
                   if (_exp(s) - now.date()).days <= d and abs(k / spot - 1) <= b]
            f = GF.flip_of(sub, spot)
            if f:
                f.pop("curve", None)
            res[lab] = f
        rec = dict(ts=int(time.time()), et=now.isoformat(timespec="seconds"), ticker=tk, spot=spot,
                   windows=res, contracts=len(want), quoted=len(got),
                   calls=WV.STATS["calls"] - c0, fails=WV.STATS["fails"] - f0)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        p = res.get(PRIMARY) or {}
        print(f"  {now:%H:%M} {tk} spot {spot}: flip {p.get('flip') and round(p['flip'], 2)} "
              f"({len(got)}/{len(want)} quoted, {rec['calls']} calls, {rec['fails']} fails)", flush=True)


def loop():
    while True:
        now = dt.datetime.now(NY)
        hm = (now.hour, now.minute)
        if hm > LAST:
            return
        if hm >= FIRST:
            t0 = time.time()
            try:
                sample()
            except Exception as e:                  # noqa: BLE001
                print(f"  sample failed: {type(e).__name__}", flush=True)
            time.sleep(max(5, EVERY - (time.time() - t0)))
        else:
            time.sleep(30)


def report(days):
    import numpy as np
    uw = {}
    for line in open(LEVELS_LOG, encoding="utf-8"):
        r = json.loads(line)
        if r.get("ticker") in TICKERS and r["et"][:10] in days and r.get("gamma_flip"):
            uw.setdefault(r["ticker"], []).append(r)
    recs = []
    for d in days:
        p = os.path.join(OUT_DIR, f"flip-{d}.jsonl")
        if os.path.exists(p):
            recs += [json.loads(x) for x in open(p, encoding="utf-8")]
    verdict = True
    for lab in [PRIMARY] + [w for w in WINDOWS if w != PRIMARY]:
        print(f"\n  WINDOW {lab}{'  (PRIMARY, scored)' if lab == PRIMARY else '  (reported)'}")
        for tk in TICKERS:
            pairs = []
            for r in recs:
                if r["ticker"] != tk or not (r["windows"].get(lab) or {}).get("flip"):
                    continue
                cand = [u for u in uw.get(tk, []) if abs(u["logged_at"] - r["ts"]) <= PAIR_S]
                if not cand:
                    continue
                u = min(cand, key=lambda u: abs(u["logged_at"] - r["ts"]))
                ours, theirs, s = r["windows"][lab]["flip"], u["gamma_flip"], r["spot"]
                near = min((abs(ours - x) for x in (u.get("nearby_flips") or [theirs])), default=None)
                pairs.append((abs(ours - theirs) / s * 100, (ours < s) == (theirs < s), near / s * 100))
            if not pairs:
                print(f"    {tk}: no pairs")
                verdict &= lab != PRIMARY
                continue
            a = np.array(pairs)
            f1, f2, f3 = np.median(a[:, 0]), a[:, 1].mean(), len(a)
            ok = f1 <= 0.25 and f2 >= 0.80 and f3 >= 30
            if lab == PRIMARY:
                verdict &= ok
            print(f"    {tk}: pairs {f3:3}  median |diff| {f1:.3f}% of spot (F1 <= 0.25%)  same side {f2:.0%} "
                  f"(F2 >= 80%)  median to nearest UW candidate {np.median(a[:, 2]):.3f}%"
                  + (f"  -> {'PASS' if ok else 'FAIL'}" if lab == PRIMARY else ""))
    n_calls = sum(r["calls"] for r in recs)
    n_fail = sum(r["fails"] for r in recs)
    print(f"\n  Webull calls {n_calls}, failures {n_fail}")
    print(f"  VERDICT ({PRIMARY}): {'PASS' if verdict else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", nargs="+")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.sample:
        sample()
    elif a.loop:
        loop()
    elif a.report:
        report(a.report)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
