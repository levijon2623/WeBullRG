"""
check_cot_forward.py -- FORWARD test of two leads from check_cot.py's 2018-26 holdout (user, 2026-11-14). LOCKED
2026-11-14, before any report it scores was released. EXPLORATORY: both directions were read off the holdout after
it was scored, so only sessions the holdout never saw can confirm them.

  F1 VIX reversal: VIX-futures leveraged-fund net (% of OI) in its lowest 10% of 3 years (most short)
       -> max VIX close over the next 20 sessions / VIX at entry - 1 BELOW the all-weeks mean.      predicted -
       (holdout: -9.1% [-18.4, -1.6], 6 episodes -- the folklore said the opposite)
  F2a asset-manager contrarian, uptrend cuts: ES asset-manager 4-week change in its lowest 10% while SPY is up over
       63 sessions -> SPY next-40-session return ABOVE the mean of all uptrend weeks.               predicted +
  F2b asset-manager contrarian, rotation: RTY minus NQ asset-manager percentile, ranked over its trailing 156
       reports; top 10% vs bottom 10% -> (IWM - QQQ) next-40-session return, top minus bottom.       predicted -

Same sample rules, entry (close of the `effective` session), outcomes, block bootstrap by quarter and pass rule as
check_cot.py (imported, not copied): PASS = 95% CI excludes 0 on the predicted side and BH q < 0.10 across the
three. Sample: reports dated 2026-11-10 onward only.

READS: `--status` (counts only, any time) | `--score` from 2028-11-14 (early read, two years), final when each test
has >= 6 signal episodes per side or on 2031-11-14, whichever comes first. Never retuned.

WEEKLY (box timer, after the Friday / holiday-Monday release): cot_pull.py --refresh, build_cot_features.py, then
`--log` appends the new report's signal state to _cot_cache/forward_log.jsonl. Prices are pulled at scoring time:
`--prices` (UW daily bars for SPY/QQQ/IWM, Cboe's VIX history).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys

import polars as pl

import check_cot as C

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_cot_cache")
LOG = os.path.join(CACHE, "forward_log.jsonl")
START = dt.date(2026, 11, 10)
EARLY, FINAL = dt.date(2028, 11, 14), dt.date(2031, 11, 14)
SPEC = {"F1": ("H4", "VIX lev funds most short -> max VIX rise 20 sessions", "vs_all", -1),
        "F2a": ("H2", "ES asset-mgr 4-wk cut in an uptrend -> SPY 40 sessions", "vs_all", +1),
        "F2b": ("H3", "asset-mgr RTY minus NQ top vs bottom -> IWM-QQQ 40 sessions", "top_bottom", -1)}


def forward_obs():
    wk = pl.read_parquet(os.path.join(CACHE, "cot_weekly.parquet"))
    px = C.load_prices()
    obs = C.build_obs(wk, px)
    return {f: [o for o in obs[h] if o[0] >= START] for f, (h, _d, _k, _s) in SPEC.items()}, wk


def status():
    wk = pl.read_parquet(os.path.join(CACHE, "cot_weekly.parquet"))
    n = wk.filter(pl.col("report_date") >= START)["report_date"].n_unique()
    rows = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    print(f"forward reports since {START}: {n} | logged {len(rows)}")
    for f in SPEC:
        on = [r for r in rows if r.get(f) not in (None, 0)]
        print(f"  {f}: signal weeks {len(on)}" + (f", latest {on[-1]['report_date']}" if on else ""))


def log_latest():
    """Append the latest report's signal state (no outcomes), once per report."""
    wk = pl.read_parquet(os.path.join(CACHE, "cot_weekly.parquet"))
    have = {json.loads(l)["report_date"] for l in open(LOG)} if os.path.exists(LOG) else set()
    last = wk["report_date"].max()
    if last < START or str(last) in have:
        print(f"nothing new to log (latest report {last})"); return
    g = lambda c, k: (wk.filter((pl.col("contract") == c) & (pl.col("report_date") == last)).to_dicts() or [{}])[0].get(k)
    vix, cut = g("VIX", "lev_money_pct156_fut"), g("ES", "asset_mgr_chg4_pct_fut")
    rty, nq = g("RTY", "asset_mgr_pct156_fut"), g("NQ", "asset_mgr_pct156_fut")
    rec = {"report_date": str(last), "effective": str(g("ES", "effective")), "release_uncertain": g("ES", "release_uncertain"),
           "vix_lev_pct156": vix, "es_am_chg4_pct": cut, "rty_am_pct156": rty, "nq_am_pct156": nq,
           "F1": 1 if vix is not None and vix <= 0.10 else 0,
           "F2a_cut": 1 if cut is not None and cut <= 0.10 else 0,   # the uptrend condition is applied at scoring
           "logged_at": dt.datetime.now().isoformat(timespec="seconds")}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("logged", rec)


def prices():
    """UW daily bars (SPY/QQQ/IWM, a year per call) + Cboe VIX history -> _cot_cache/prices_daily.json."""
    import requests
    import positioning_map as P
    uw, n = P.UW(), (lambda v: P.num(v))
    out = json.load(open(os.path.join(CACHE, "prices_daily.json"))) if os.path.exists(os.path.join(CACHE, "prices_daily.json")) else {}
    for t in ("SPY", "QQQ", "IWM"):
        rows = {r["date"]: r for r in out.get(t, [])}
        first = max(rows) if rows else "2007-01-01"
        for y in range(int(first[:4]), dt.date.today().year + 1):
            kw = {} if y == dt.date.today().year else {"end_date": f"{y}-12-31"}
            for x in uw.get(f"/stock/{t}/ohlc/1d", limit=2500, **kw).get("data") or []:
                if x.get("market_time") not in (None, "r"):
                    continue
                d = str(x.get("date") or x.get("start_time"))[:10]
                rows[d] = {"date": d, "close": n(x.get("close")), "high": n(x.get("high")), "low": n(x.get("low")), "open": n(x.get("open"))}
        out[t] = [rows[d] for d in sorted(rows)]
    r = requests.get("https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                     headers={"User-Agent": "cot-research/1.0"}, timeout=60)
    if r.status_code == 200:
        out["VIX"] = [{"date": dt.datetime.strptime(x["DATE"], "%m/%d/%Y").date().isoformat(), "close": float(x["CLOSE"]),
                       "high": float(x["HIGH"]), "low": float(x["LOW"]), "open": float(x["OPEN"])}
                      for x in csv.DictReader(io.StringIO(r.text))]
    json.dump(out, open(os.path.join(CACHE, "prices_daily.json"), "w"))
    print("prices:", {k: (len(v), v[-1]["date"]) for k, v in out.items()})


def score():
    today = dt.date.today()
    obs, _ = forward_obs()
    eps = {f: (C.episodes(o), C.episodes(o, -1) if SPEC[f][2] == "top_bottom" else None) for f, o in obs.items()}
    enough = all(a >= 6 and (b is None or b >= 6) for a, b in eps.values())
    if today < EARLY and not enough:
        sys.exit(f"too early: the first read is {EARLY} (episodes so far {eps})")
    final = enough or today >= FINAL
    rows, ps = [], []
    for f, (h, desc, kind, sign) in SPEC.items():
        e = C.effect(obs[f], kind); lo, hi, pv = C.boot(obs[f], kind)
        rows.append((f, desc, sign, e, lo, hi)); ps.append(pv)
    qs = C.bh(ps)
    print(f"{'FINAL' if final else 'EARLY'} READ {today}: reports from {START}")
    for (f, desc, sign, e, lo, hi), pv, qv in zip(rows, ps, qs):
        ok = e is not None and lo is not None and ((lo > 0) if sign > 0 else (hi < 0)) and qv < 0.10
        fm = lambda v: "n/a" if v is None else f"{v * 100:+.2f}%"
        print(f"  {f} {desc}: {fm(e)} [{fm(lo)}, {fm(hi)}] q={qv:.3f} episodes {eps[f]} predicted {'+' if sign > 0 else '-'}"
              f" -> {('PASS' if ok else 'FAIL') if final else ('on track' if ok else 'open')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("status", "log", "prices", "score"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.prices: prices()
    if a.log: log_latest()
    if a.status: status()
    if a.score: score()
    if not any(vars(a).values()):
        sys.exit(__doc__)
