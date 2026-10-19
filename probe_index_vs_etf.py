"""
probe_index_vs_etf.py
=====================
INDEX vs ETF OPTIONS POSITIONING (operator, 2026-10-18)

Institutions hedge mostly in index options (SPX / NDX / RUT: cash-settled,
European, 1256 tax treatment, ~10x the notional of an ETF contract), retail
and ETF holders in SPY / QQQ / IWM. Our own research found SPY and SPX flow to
be different signals (memory: cleanbot-index-flow-blend), so an ETF-only read
of "who is hedging" can miss the bigger book. This puts both on ONE scale.

FAMILIES  SPX (roots SPX + SPXW) vs SPY · NDX (NDX + NDXP) vs QQQ · RUT (RUT + RUTW) vs IWM
WINDOW    trading days --start .. --end (default 2026-10-05 .. 2026-10-16)
MEASURES  (all in DOLLARS so an index contract and an ETF contract count fairly)
  1. DAILY     put / call premium per day; the last day vs the first-week average.
  2. OI        open interest per expiry (UW /oi-per-expiry, start of day, per root,
               summed over roots): change from the first to the last date, for the
               focus expiries and for every expiry alive on both dates -> contracts
               x 100 x spot = notional.
  3. DELTA     directional delta flow per expiry per day (UW /greek-flow/{expiry};
               dir_delta_flow = signed by side, delta x contracts) x 100 x spot =
               dollar delta bought (+) or sold (-). Window total and last day.
  4. LEADERS   the last day's biggest focus-expiry puts per family (premium, side,
               multi-leg share, start-of-day OI) -- e.g. was there an index twin of
               QQQ's 10/30 3:47 PM package?
🚨 BOX SPREADS. SPX's quarterly expiries carry huge box-spread financing (2026-10-16:
SPX Dec-18 7000/8000, ~33k on each of four legs, 100% multi-leg, ~300k OI per leg).
A box is market-neutral, but side-by-leg reads it as tens of billions of
"directional" delta (+$40B that day). Read the focus expiries' delta with that in
mind; judge the quarterlies by their legs, not the total.
Caveats: side is inferred from ask/bid; open interest nets openers vs closers;
UW greek-flow reports one series per family (it already combines the roots), and
its dir_delta_flow is in SHARES (delta x contracts x 100) -- verified 10-18: QQQ
10/30 on 10/16 = -$6.2B, matching the hand-computed -$6.3B of the two packages.
UW carries no close for index underlyings (underlying_close is null, no index
ohlc), so index levels = the ETF close x index_gex.SEED_RATIO (within ~0.5%).

Usage:  python probe_index_vs_etf.py [--start D --end D] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                           # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
import uw_options_data_lake as L                         # noqa: E402

FAMILIES = [("SPX", ["SPX", "SPXW"], "SPY"), ("NDX", ["NDX", "NDXP"], "QQQ"), ("RUT", ["RUT", "RUTW"], "IWM")]
FOCUS = ["2026-10-23", "2026-10-30", "2026-11-20", "2026-12-18"]
SPOT_SYM = {"SPX": "SPXW261030P07650000", "NDX": None, "RUT": None,
            "SPY": "SPY261030P00750000", "QQQ": "QQQ261030P00700000", "IWM": "IWM261120P00258000"}


def f(v):
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def spots(c, tk, start, end):
    """{date: underlying close} from any contract's history on that underlying."""
    sym = SPOT_SYM.get(tk)
    if not sym:
        rows = c.get_json(f"/stock/{tk}/option-contracts", {"expiry": FOCUS[1], "option_type": "put"}).get("data", [])
        rows = [r for r in rows if r["option_symbol"][-15:-9] == FOCUS[1][2:].replace("-", "")]
        sym = max(rows, key=lambda r: f(r["open_interest"]))["option_symbol"] if rows else None
    if not sym:
        return {}
    h = c.get_json(f"/option-contract/{sym}/historic").get("chains", [])
    return {r["date"]: f(r["underlying_close"]) for r in h if start <= r["date"] <= end and f(r["underlying_close"])}


def daily(c, roots, start, end):
    out = {}
    for tk in roots:
        for r in c.get_json(f"/stock/{tk}/options-volume", {"limit": "30"}).get("data", []):
            if start <= r["date"] <= end:
                d = out.setdefault(r["date"], dict(put=0.0, call=0.0, nput=0.0, ncall=0.0))
                d["put"] += f(r["put_premium"]); d["call"] += f(r["call_premium"])
                d["nput"] += f(r["net_put_premium"]); d["ncall"] += f(r["net_call_premium"])
    return dict(sorted(out.items()))


def oi_by_expiry(c, roots, d):
    out = {}
    for tk in roots:
        try:
            rows = c.get_json(f"/stock/{tk}/oi-per-expiry", {"date": d}).get("data", [])
        except L.NoDataForDate:
            rows = []
        for r in rows:
            if r.get("date") != d:
                continue
            e = out.setdefault(r["expiry"], [0.0, 0.0])
            e[0] += f(r["call_oi"]); e[1] += f(r["put_oi"])
    return out


def delta_flow(c, tk, expiry, d):
    try:
        rows = c.get_json(f"/stock/{tk}/greek-flow/{expiry}", {"date": d}).get("data", [])
    except L.NoDataForDate:
        return 0.0, 0.0
    rows = [r for r in rows if str(r.get("timestamp", ""))[:10] == d]
    return sum(f(r["dir_delta_flow"]) for r in rows), sum(f(r["volume"]) for r in rows)


def leaders(c, roots, expiry, n=4):
    rows = []
    for tk in roots:
        try:
            rows += c.get_json(f"/stock/{tk}/option-contracts", {"expiry": expiry, "option_type": "put"}).get("data", [])
        except L.NoDataForDate:
            pass
    six = expiry[2:].replace("-", "")
    rows = [r for r in rows if r["option_symbol"][-15:-9] == six and r["option_symbol"][-9] == "P"]
    rows = list({r["option_symbol"]: r for r in rows}.values())   # SPX's list already includes SPXW
    out = []
    for r in sorted(rows, key=lambda r: -f(r["total_premium"]))[:n]:
        v = f(r["volume"])
        out.append(dict(sym=r["option_symbol"], k=int(r["option_symbol"][-8:]) / 1000, vol=v,
                        ask=f(r["ask_volume"]) / max(v, 1), bid=f(r["bid_volume"]) / max(v, 1),
                        mleg=f(r["multi_leg_volume"]) / max(v, 1), prem=f(r["total_premium"]),
                        oi=f(r["open_interest"]), px=f(r["avg_price"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-05")
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--out", default=os.path.join(ROOT, "_index_vs_etf.json"))
    a = ap.parse_args()
    c = L.Client(os.environ["UW_API_KEY"])
    res = {}
    import index_gex as IX
    for fam, roots, etf in FAMILIES:
        etf_sp = spots(c, etf, a.start, a.end)
        for name, rts in ((fam, roots), (etf, [etf])):
            sp = etf_sp if name == etf else {d: v * IX.SEED_RATIO[etf] for d, v in etf_sp.items()}
            days = sorted(sp)
            if not days:
                print(f"{name}: no spot history"); continue
            dl = daily(c, rts, a.start, a.end)
            first_week = [v["put"] for d, v in dl.items() if d <= days[min(4, len(days) - 1)]]
            o0, o1 = oi_by_expiry(c, rts, days[0]), oi_by_expiry(c, rts, days[-1])
            alive = [e for e in o1 if e in o0 and e > days[-1]]
            s_end = sp[days[-1]]
            oi = {e: dict(dc=o1[e][0] - o0[e][0], dp=o1[e][1] - o0[e][1]) for e in FOCUS if e in o0 and e in o1}
            oi_all = dict(dc=sum(o1[e][0] - o0[e][0] for e in alive), dp=sum(o1[e][1] - o0[e][1] for e in alive))
            gf = roots[0] if name == fam else etf
            dfl = {}
            for e in FOCUS:
                for d in days:
                    dd, vv = delta_flow(c, gf, e, d)
                    dfl.setdefault(e, {})[d] = dd * sp[d]   # dir_delta_flow is already SHARES (x100 included)
            res[name] = dict(family=fam, roots=rts, spot=sp, daily=dl,
                             put_mult=(dl[days[-1]]["put"] / (sum(first_week) / len(first_week))) if first_week and days[-1] in dl else None,
                             oi=oi, oi_all=oi_all, notional_per_ct=100 * s_end, dflow=dfl,
                             leaders=leaders(c, rts, FOCUS[1]))
            r = res[name]
            tot_delta = sum(sum(v.values()) for v in dfl.values())
            last_delta = sum(v.get(days[-1], 0) for v in dfl.values())
            print(f"\n== {name} ({'+'.join(rts)})  spot {sp[days[0]]:,.2f} -> {s_end:,.2f}  "
                  f"({s_end / sp[days[0]] - 1:+.1%})  Fri put prem ${dl.get(days[-1], {}).get('put', 0) / 1e6:,.0f}M "
                  f"= {r['put_mult'] or 0:.1f}x first-week avg")
            print(f"   OI change {days[0]} -> {days[-1]} (all live expiries): puts {oi_all['dp']:+,.0f} "
                  f"(${oi_all['dp'] * 100 * s_end / 1e9:+.2f}B notional)  calls {oi_all['dc']:+,.0f} "
                  f"(${oi_all['dc'] * 100 * s_end / 1e9:+.2f}B)")
            for e, v in oi.items():
                print(f"     {e}: puts {v['dp']:+,.0f} (${v['dp'] * 100 * s_end / 1e9:+.2f}B)  calls {v['dc']:+,.0f} "
                      f"(${v['dc'] * 100 * s_end / 1e9:+.2f}B)   delta flow window ${sum(dfl[e].values()) / 1e9:+.2f}B, "
                      f"last day ${dfl[e].get(days[-1], 0) / 1e9:+.2f}B")
            print(f"   directional delta flow, 4 focus expiries: window ${tot_delta / 1e9:+.2f}B, last day ${last_delta / 1e9:+.2f}B")
            print(f"   {FOCUS[1]} put leaders on {days[-1]}:")
            for x in r["leaders"]:
                print(f"     {x['sym']:22} vol {x['vol']:>8,.0f} ask {x['ask']:4.0%} bid {x['bid']:4.0%} mleg {x['mleg']:4.0%} "
                      f"avg {x['px']:8.2f} prem ${x['prem'] / 1e6:6.1f}M  OI(start) {x['oi']:>8,.0f}")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, default=str)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
