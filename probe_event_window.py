"""
probe_event_window.py
=====================
EVENT-WINDOW POSITIONING for the three ETF / index families (user, 2026-11-01): how S&P (SPY + SPX/SPXW),
Nasdaq (QQQ + NDX/NDXP) and Russell (IWM + RUT/RUTW) are positioned into a week of events (elections, payrolls,
Fed, CPI ...), as of one session's close. One JSON per run feeds docs/posts/make_event_cards.py.

PER FAMILY, PER EXPIRY (the window's daily expiries, plus the next one as a baseline)
  atm_iv     the ETF's ATM implied vol at the close (strikes within 0.4% of spot)
  move       that day's priced 1-sigma close-to-close move: sqrt(iv_i^2 T_i - iv_{i-1}^2 T_{i-1}), calendar time
             (the first expiry's move includes any weekend)
  calls/puts open interest within +-BAND of spot, in ETF-contract units (an index contract = ratio ETF
             contracts); P/C; the index's share; the top strikes
  flip       the cumulative zero-gamma level from that morning's positioning map (live/posmap/{SP,QQQ,IWM}.json,
             usual convention: dealers long calls, short puts) when the local copy is from the as-of date
PER FAMILY  strike ladder (+-5%, summed over the window), directional delta flow per ticker over the last 5
            sessions into the window's expiries (UW greek-flow is in shares -> x spot = $), the window's open
            interest build over those sessions, RV21, the window's events (live/posmap/events.json).

UW FACTS (met 2026-11-01)
  - open interest is START OF DAY: the as-of session's own trading shows up the next morning.
  - SPX: /stock/SPX/option-contracts returns both roots (SPX monthlies + SPXW weeklies); NDX likewise NDX + NDXP.
    RUT does NOT: its weeklies live under ticker RUTW (RUT returned nothing for 11/04, 11/06). greek-flow for the
    weeklies: RUTW. oi-per-expiry is per root -> summed.
  - no index quotes: the index level is backed out of its chain (positioning_map.implied_spot); RUT's weekly greeks
    can be too thin -> fallback ETF x index_gex.SEED_RATIO.
  - far tails (e.g. SPX 6350 / 8800 in a weekly) dominate raw notional and P/C but carry little delta or gamma:
    everything near-the-money is +-BAND.
CONTEXT, NOT A SIGNAL: positioning sizes moves; our tests found it doesn't call direction.

    python probe_event_window.py --asof 2026-10-30 --out docs/posts/data/event-week-2026-11-02.json
    python probe_event_window.py --asof D --expiries 2026-11-02 2026-11-03 ... [--band 0.04]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import positioning_map as P                              # noqa: E402  (loads .env, UW client, helpers)
import market_calendar as MC                             # noqa: E402

FAMILIES = {   # key: (ETF, index label, chain tickers, root prefixes, greek-flow ticker, oi-per-expiry roots, seed ratio)
    "SP":  ("SPY", "SPX", ("SPX",), ("SPX", "SPXW"), "SPX", ("SPX", "SPXW"), 10.04),
    "QQQ": ("QQQ", "NDX", ("NDX",), ("NDX", "NDXP"), "NDX", ("NDX", "NDXP"), 41.1),
    "IWM": ("IWM", "RUT", ("RUT", "RUTW"), ("RUT", "RUTW"), "RUTW", ("RUT", "RUTW"), 10.07),
}
n = P.num


def chain(uw, tickers, roots, e):
    rows = []
    for tk in tickers:
        for t in ("call", "put"):
            rows += [r for r in uw.get(f"/stock/{tk}/option-contracts", expiry=e, option_type=t, limit=500).get("data") or []
                     if r["option_symbol"].startswith(roots)]
    return rows


def family(uw, key, asof, exps, band):
    etf, idx, ctks, roots, gtk, oroots, seed = FAMILIES[key]
    allx = exps + [MC.next_trading_day(dt.date.fromisoformat(exps[-1])).isoformat()]
    ch = {}
    for e in allx:
        ch[(etf, e)] = chain(uw, (etf,), (etf,), e)
        ch[(idx, e)] = chain(uw, ctks, roots, e)
    S, _ = P.implied_spot({r["option_symbol"]: r for e in allx for r in ch[(etf, e)]}, asof)
    I, _ = P.implied_spot({r["option_symbol"]: r for e in allx for r in ch[(idx, e)]}, asof)
    I = I or S * seed
    ratio = I / S
    try:
        pm = json.load(open(os.path.join(ROOT, "live", "posmap", f"{key}.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pm = {}
    flips = {x["e"]: x.get("flip") for x in pm.get("expiries", [])} if pm.get("date") == asof.isoformat() else {}
    F = dict(etf=etf, idx=idx, spot=round(S, 2), idx_spot=round(I, 1), ratio=round(ratio, 3),
             prev_close=pm.get("prev_close"), open=pm.get("spot"), rv21=pm.get("rv21"), days=[], strikes={})
    ladder = defaultdict(lambda: [0.0, 0.0])
    prev_var = 0.0
    for e in allx:
        T = (dt.date.fromisoformat(e) - asof).days / 365.0
        tot, ix, ivs, st = [0.0, 0.0], [0.0, 0.0], [], defaultdict(lambda: [0.0, 0.0])
        for tk in (etf, idx):
            m = 1 if tk == etf else ratio
            for r in ch[(tk, e)]:
                p = P.parse_occ(r["option_symbol"])
                k, j = p[3] / m, 0 if p[2] == "C" else 1
                if tk == etf and abs(k / S - 1) < 0.004 and n(r.get("implied_volatility")):
                    ivs.append(n(r.get("implied_volatility")))
                if abs(k / S - 1) > 0.05:
                    continue
                oi = (n(r.get("open_interest")) or 0) * m
                st[round(k)][j] += oi
                if abs(k / S - 1) <= band:
                    tot[j] += oi
                    ix[j] += oi if tk == idx else 0
        atm = sum(ivs) / len(ivs) if ivs else None
        mv = None
        if atm:
            var = atm * atm * T
            mv = math.sqrt(var - prev_var) * 100 if var > prev_var else None
            prev_var = var
        top = lambda j: [[k, round(v[j])] for k, v in sorted(st.items(), key=lambda kv: -kv[1][j])[:3]] if e in exps else []
        if e in exps:
            for k, v in st.items():
                ladder[k][0] += v[0]
                ladder[k][1] += v[1]
        F["days"].append(dict(e=e, in_window=e in exps, atm_iv=round(atm * 100, 1) if atm else None,
                              move=round(mv, 2) if mv else None, calls=round(tot[0]), puts=round(tot[1]),
                              pc=round(tot[1] / max(tot[0], 1), 2), idx_c=round(ix[0] / max(tot[0], 1), 2),
                              idx_p=round(ix[1] / max(tot[1], 1), 2), flip=flips.get(e), top_c=top(0), top_p=top(1)))
    F["strikes"] = {str(k): [round(v[0]), round(v[1])] for k, v in sorted(ladder.items())}
    sessions = [d.isoformat() for d in MC.trading_days(asof - dt.timedelta(days=10), asof)][-5:]
    F["delta_flow_B"] = {}
    for tk, spot in ((etf, S), (gtk, I)):
        per = []
        for e in exps:
            s = 0.0
            for d in sessions:
                if d <= e:
                    s += sum(n(r.get("dir_delta_flow")) or 0 for r in uw.get(f"/stock/{tk}/greek-flow/{e}", date=d).get("data") or [])
            per.append(round(s * spot / 1e9, 2))
        F["delta_flow_B"][etf if tk == etf else idx] = dict(total=round(sum(per), 2), per_exp=per)
    F["oi_build"] = {}
    for lab, rts, m in ((etf, (etf,), 1), (idx, oroots, ratio)):
        snap = {}
        for d in (sessions[0], sessions[-1]):
            c = p = 0.0
            for t in rts:
                for r in uw.get(f"/stock/{t}/oi-per-expiry", date=d).get("data") or []:
                    if (r.get("expiry") or "") in exps:
                        c += (n(r.get("call_oi")) or 0) * m
                        p += (n(r.get("put_oi")) or 0) * m
            snap[d] = [round(c), round(p)]
        F["oi_build"][lab] = snap
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="the session whose close the snapshot reflects")
    ap.add_argument("--expiries", nargs="+", help="the window's expiries (default: the next 5 trading days)")
    ap.add_argument("--band", type=float, default=0.04)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    asof = dt.date.fromisoformat(a.asof)
    exps = a.expiries or []
    d = asof
    while not a.expiries and len(exps) < 5:
        d = MC.next_trading_day(d)
        exps.append(d.isoformat())
    uw = P.UW()
    try:
        ev = json.load(open(os.path.join(ROOT, "live", "posmap", "events.json"), encoding="utf-8")).get("events", [])
    except (OSError, ValueError):
        ev = json.load(open(os.path.join(ROOT, "market_events.json"), encoding="utf-8")).get("events", [])
    doc = dict(asof=asof.isoformat(), expiries=exps, band=a.band,
               events=[e for e in ev if asof.isoformat() < e["d"] <= exps[-1]], families={})
    for key in FAMILIES:
        doc["families"][key] = family(uw, key, asof, exps, a.band)
        f = doc["families"][key]
        print(f"{key}: {f['etf']} {f['spot']} / {f['idx']} {f['idx_spot']} (x{f['ratio']})  "
              + "  ".join(f"{x['e'][5:]} {x['move']}% P/C {x['pc']}" for x in f["days"]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"-> {a.out}  [{uw.calls} UW calls]")


if __name__ == "__main__":
    main()
