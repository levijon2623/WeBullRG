"""
check_ironfly_forward.py -- FORWARD TEST: short 0DTE iron butterfly held to the close when the vol gauge is NOT high
(user, 2026-11-14; Level 3 defined-risk premium selling). LOCKED 2026-11-14, before any session it scores.

ORIGIN (after the fact, so only new sessions can test it): check_ironfly_lowvol (stage 1, 2024-08..2026-08) found
nothing at the gauge's 60-minute horizon, but a fly HELD TO 15:55 paid at the user's fills -- LOW $2 wings +$4.29
[+1.65, +7.15] per fly (5.2% of risk), MID close behind, HIGH about a third, entries after 14:00 weakest, the second
half about a third of the first, and nothing left at cross fills.

ENTRY     the LIVE gauge (live/vol_gauge_log.jsonl, the deployed model, 10-minute readings): per ticker per session,
          the FIRST reading from 11:00 through 13:50 with p < 0.65 (not HIGH); no trade that day if every reading
          in the window is HIGH. SPY, QQQ, IWM. Sessions from 2026-11-16.
FLY       same-day expiry; ATM = the quoted strike nearest the reading's price; SELL the ATM call and put, BUY the
          call W above and the put W below. PRIMARY W = $2. Reported: W = $3.
EXIT      held to 15:55 (closed at the 15:55 quotes; never to expiry).
PRICING   lake option-contracts-1m minute quotes (entry the reading's minute, exit 15:55), each leg at mid, selling
          -$0.0008 / buying +$0.0006 per share (the user's measured single-leg fills), fees $0.045 per contract per
          order ($0.36 per fly round trip). Cross fills (sell bid / buy ask) reported as the bracket.
PASS      mean $ per fly (W = $2, user fills) > 0 with the day-block bootstrap 95% CI above 0 (2,000 resamples).
READS     --status any time; --score early at >= 125 sessions (~2027-05), FINAL at >= 250 sessions (~2027-11).
          POWER, stated up front: one fly per ticker per day, and all three close at the same 15:55, so the
          sessions are the sample. At 250 sessions the CI half-width should be roughly $3-4, enough to confirm a
          true edge near the stage-1 +$4 but not the second half's ~+$2.4. An inconclusive final is a FAIL.
NOT TESTED HERE, and the biggest open risk: whether a real 4-leg combo order fills near mid. If the user places
          real or paper flies by this rule, their actual net fills are the evidence that matters most.
DATA      before --build: copy live/vol_gauge_log.jsonl from the box to _ic_cache/vol_gauge_log.jsonl, and refresh
          the lake's option-contracts-1m for the forward sessions (uw_options_data_lake.py).

Usage:  python check_ironfly_forward.py --build | --status | --score
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import check_ic_vol_trade as VT                         # noqa: E402
import check_ironfly_lowvol as IF                       # noqa: E402

LOG = os.path.join("_ic_cache", "vol_gauge_log.jsonl")
OUT = os.path.join("_ic_cache", "ironfly_forward.json")
START = dt.date(2026, 11, 16)
WIN = (11 * 60, 13 * 60 + 50)
HIGH = 0.65
WINGS = {"primary": 2.0, "w3": 3.0}
EARLY_N, FINAL_N = 125, 250


def entries():
    first = {}
    for line in open(LOG):
        r = json.loads(line)
        d = dt.date.fromisoformat(r["day"])
        if d < START or not (WIN[0] <= r["mod"] <= WIN[1]) or r.get("p") is None or r["p"] >= HIGH:
            continue
        k = (r["tk"], d)
        if k not in first or r["mod"] < first[k]["mod"]:
            first[k] = r
    return sorted(first.values(), key=lambda r: (r["day"], r["tk"]))


def build():
    paths = {os.path.basename(os.path.dirname(p)).split("=")[1]: p for p in glob.glob(f"{VT.SILVER}/date=*/bars.parquet")}
    have = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["day"], r["tk"]) for r in have}
    books = {}
    for e in entries():
        if (e["day"], e["tk"]) in done or e["day"] not in paths:
            continue
        d = dt.date.fromisoformat(e["day"])
        if e["day"] not in books:
            books = {e["day"]: VT.day_book(paths[e["day"]], d)}
        bk = books[e["day"]].get(e["tk"])
        lg = VT.legs_for(bk, d, e["close"], e["mod"]) if bk else None
        rec = dict(day=e["day"], tk=e["tk"], mod=e["mod"], p=e["p"])
        for name, W in WINGS.items():
            r = IF.fly(bk, lg["straddle"], W, e["mod"], IF.EOD) if lg and lg["straddle"][0][0] == d else None
            for fill in ("user", "cross"):
                rec[f"{name}_{fill}"] = None if r is None else round(r[fill][0] * 100, 2)
            rec[f"{name}_risk"] = None if r is None else round(r["user"][1] * 100, 2)
        have.append(rec)
        print(e["day"], e["tk"], rec.get("primary_user"), flush=True)
    json.dump(have, open(OUT, "w"))


def status():
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    n = len({r["day"] for r in rows})
    print(f"forward sessions priced {n} (early read {EARLY_N}, final {FINAL_N}); flies {len(rows)}")


def score():
    rows = [r for r in (json.load(open(OUT)) if os.path.exists(OUT) else []) if r.get("primary_user") is not None]
    n = len({r["day"] for r in rows})
    if n < EARLY_N:
        sys.exit(f"too early: {n} sessions, the first read is at {EARLY_N}")
    final = n >= FINAL_N
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    rng = np.random.default_rng(20261116)
    print(f"{'FINAL' if final else 'EARLY'} READ: {n} sessions from {START}, {len(df)} flies")
    for name in WINGS:
        for fill in ("user", "cross"):
            col = f"{name}_{fill}"
            d = df[df[col].notna()]
            m, lo, hi = IF.dboot(d, col, rng)
            scored = name == "primary" and fill == "user"
            v = (("PASS" if lo > 0 else "FAIL") if final else ("on track" if lo > 0 else "open")) if scored else "reported"
            print(f"  W {WINGS[name]:.0f} {fill:5s} n={len(d):4d} ${m:+.2f}/fly [{lo:+.2f}, {hi:+.2f}]  "
                  f"risk ${d[f'{name}_risk'].mean():.0f}  -> {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "status", "score"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.build: build()
    if a.status: status()
    if a.score: score()
    if not any(vars(a).values()):
        sys.exit(__doc__)
