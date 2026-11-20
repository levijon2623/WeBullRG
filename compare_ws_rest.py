"""
compare_ws_rest.py -- does UW's WEBSOCKET give the same numbers as the REST endpoints our rules were validated on?
(user, 2026-11-19, the UW Advanced checklist). Reads uw_ws_shadow.py's logs for one session and pulls the REST side:

  1. FLOW  -- the bot's trigger series. net_flow:T ticks (increments, ~1/s) summed per minute as net call - net put
     premium, vs REST /stock/T/net-prem-ticks?date= (one row per minute; the bot sums net_call_premium -
     net_put_premium from 09:29). Per-minute correlation, end-of-day cum, worst cum gap as % of the day's range,
     and whether the cum-vs-EMA(5) crossings (the trigger's shape) land on the same minutes.
  2. OPTION TOTALS -- what strike_cue reads. option_trades:T same-day trades summed per contract (ask / bid / all, by
     UW's side tags) vs REST /stock/T/option-contracts?expiry=today ask_volume / bid_volume / volume.
  3. OPTION MINUTES -- near-the-money same-day contracts, per minute: WS vs REST /option-contract/OCC/intraday
     volume_ask_side / volume_bid_side.
  4. LATENCY -- receive time - executed time of option trades (latency.json, per minute).
The shadow's own reconnect gaps are listed (journal) -- trades during a gap are missing from the WS side by design.

    python compare_ws_rest.py [--day 2026-11-19]

RUN LOG
  2026-11-19 (first session; shadow reconnected twice, 12:21:49-12:22:10 and 13:58:48-13:59:06):
    FLOW, 9 rule tickers: per-minute corr 0.9987-1.0000; cum-vs-EMA(5) crossings 89/89 SPY, 77/77 QQQ, 66/66 IWM, all
      9 tickers within 1 min (AVGO 66 of 68, MSFT WS 69 vs 67). Every ticker's worst cum gap sits at 12:22 or 13:59 --
      the reconnects (SPY 4.7% of the day's range, others 0.1-2.3%). Outside the gaps the two are the same series.
    OPTION TOTALS, 0DTE contracts >= 100 lots: WS/REST volume 0.990-0.992 (SPY 139, QQQ 177, IWM 42 contracts), ask
      and bid alike; the shortfall is the 16:05-16:15 prints the shadow does not record plus the reconnects.
    OPTION MINUTES, ATM +-2 calls and puts: corr ask 0.979-0.9999, bid 0.9997-0.9998, total 0.994-0.9998; 98.9-99.2%
      of contract-minutes EXACTLY equal.
    LATENCY: median 0.085 s every minute (REST polling = 5 s). BUT 14:48-15:09 a SUBSET of trades arrived late -- the
      minute p90 ramped 338 s -> 657 s (14:56) -> 105 s (15:09) while the median stayed 0.09 s. Which ticker / kind is
      not recorded (latency.json pools them).
    VERDICT: WS = REST in value. Before anything switches: per-ticker late-trade logging, a REST backfill after every
    reconnect, and a few more sessions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                    # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
import positioning_map as P                       # noqa: E402

SH = os.path.join(ROOT, "live", "ws_shadow")
OPT_TK = ("SPY", "QQQ", "IWM")
S0, S1 = 9 * 60 + 29, 16 * 60


def mod_of(t):
    return t.hour * 60 + t.minute


def ema(xs, n=5):
    a, out, e = 2 / (n + 1), [], None
    for x in xs:
        e = x if e is None else a * x + (1 - a) * e
        out.append(e)
    return out


def crossings(cum):
    e = ema(cum)
    s = [np.sign(c - x) for c, x in zip(cum, e)]
    return {i for i in range(1, len(s)) if s[i] != 0 and s[i - 1] != 0 and s[i] != s[i - 1]}


def flow(day, uw):
    ws = {}
    for line in open(os.path.join(SH, day.isoformat(), "net_flow.jsonl")):
        r = json.loads(line)
        d = r["data"]
        if "net_call_prem" not in d or d.get("trading_hours", "rth") != "rth":
            continue
        t = dt.datetime.fromtimestamp(d["time"] / 1000, P.NY)
        m = mod_of(t)
        if t.date() != day or not (S0 <= m < S1):
            continue
        tk = d["ticker"]
        ws.setdefault(tk, {}).setdefault(m, 0.0)
        ws[tk][m] += (P.num(d["net_call_prem"]) or 0.0) - (P.num(d["net_put_prem"]) or 0.0)
    out = {}
    for tk in sorted(ws):
        rows = uw.get(f"/stock/{tk}/net-prem-ticks", date=day.isoformat()).get("data") or []
        rest = {}
        for x in rows:
            t = dt.datetime.fromisoformat(str(x["tape_time"]).replace("Z", "+00:00")).astimezone(P.NY)
            m = mod_of(t)
            if t.date() == day and S0 <= m < S1:
                rest[m] = rest.get(m, 0.0) + (P.num(x.get("net_call_premium")) or 0.0) - (P.num(x.get("net_put_premium")) or 0.0)
        mins = list(range(S0, S1))
        a = np.array([ws[tk].get(m, 0.0) for m in mins])
        b = np.array([rest.get(m, 0.0) for m in mins])
        ca, cb = np.cumsum(a), np.cumsum(b)
        rng = max(cb.max() - cb.min(), 1.0)
        xa, xb = crossings(list(ca)), crossings(list(cb))
        near = sum(1 for i in xb if any(abs(i - j) <= 1 for j in xa))
        out[tk] = dict(minutes_ws=len(ws[tk]), minutes_rest=len(rest),
                       corr_minute=round(float(np.corrcoef(a, b)[0, 1]), 4) if a.std() and b.std() else None,
                       cum_ws=round(float(ca[-1])), cum_rest=round(float(cb[-1])),
                       worst_gap_pct_range=round(float(np.abs(ca - cb).max() / rng * 100), 2),
                       worst_gap_min=int(mins[int(np.abs(ca - cb).argmax())]),
                       crosses_rest=len(xb), crosses_ws=len(xa), crosses_matched_1min=near)
    return out


def option_totals(day, uw, opt):
    ws = {}
    for m, c in opt.items():
        for occ, (ask, bid, other, tot) in c.items():
            w = ws.setdefault(occ, [0.0, 0.0, 0.0])
            w[0] += ask; w[1] += bid; w[2] += tot
    out = {}
    for tk in OPT_TK:
        rows = uw.get(f"/stock/{tk}/option-contracts", expiry=day.isoformat()).get("data") or []
        if len(rows) >= 500:
            rows = sum(((uw.get(f"/stock/{tk}/option-contracts", expiry=day.isoformat(), option_type=ty).get("data") or [])
                        for ty in ("call", "put")), [])
        rest = {x["option_symbol"]: (P.num(x.get("ask_volume")) or 0.0, P.num(x.get("bid_volume")) or 0.0,
                                     P.num(x.get("volume")) or 0.0) for x in rows if x.get("option_symbol")}
        keys = [k for k in rest if rest[k][2] >= 100]
        if not keys:
            out[tk] = {"contracts": 0}
            continue
        tw = sum(ws.get(k, [0, 0, 0])[2] for k in keys); tr = sum(rest[k][2] for k in keys)
        aw = sum(ws.get(k, [0, 0, 0])[0] for k in keys); ar = sum(rest[k][0] for k in keys)
        bw = sum(ws.get(k, [0, 0, 0])[1] for k in keys); br = sum(rest[k][1] for k in keys)
        ratio = [ws.get(k, [0, 0, 0])[2] / rest[k][2] for k in keys]
        sh_w = [ws[k][0] / ws[k][2] for k in keys if k in ws and ws[k][2] > 0]
        sh_r = [rest[k][0] / rest[k][2] for k in keys if k in ws and ws[k][2] > 0]
        out[tk] = dict(contracts=len(keys), vol_ws=int(tw), vol_rest=int(tr), vol_ratio=round(tw / tr, 4),
                       ask_ratio=round(aw / ar, 4) if ar else None, bid_ratio=round(bw / br, 4) if br else None,
                       per_contract_ratio_median=round(st.median(ratio), 4),
                       per_contract_ratio_p10_p90=[round(float(np.quantile(ratio, .1)), 4), round(float(np.quantile(ratio, .9)), 4)],
                       ask_share_corr=round(float(np.corrcoef(sh_w, sh_r)[0, 1]), 4) if len(sh_w) > 2 else None)
    return out


def option_minutes(day, uw, opt):
    """near-the-money same-day SPY / QQQ / IWM contracts, minute by minute."""
    out = {}
    for tk in OPT_TK:
        bars = uw.get(f"/stock/{tk}/ohlc/1m", date=day.isoformat(), limit=500).get("data") or []
        closes = [P.num(b.get("close")) for b in bars if P.num(b.get("close"))]
        if not closes:
            continue
        spot = closes[0] if len(closes) < 2 else st.median(closes)
        ks = [int(round(spot)) + i for i in (-2, -1, 0, 1, 2)]
        pairs = []
        for k in ks:
            for cp in "CP":
                occ = f"{tk}{day.strftime('%y%m%d')}{cp}{k * 1000:08d}"
                rows = uw.get(f"/option-contract/{occ}/intraday", date=day.isoformat()).get("data") or []
                for r in rows:
                    t = dt.datetime.fromisoformat(str(r["start_time"]).replace("Z", "+00:00")).astimezone(P.NY)
                    m = mod_of(t)
                    if not (9 * 60 + 30 <= m < S1):
                        continue
                    w = (opt.get(str(m)) or {}).get(occ, [0, 0, 0, 0])
                    # intraday has no plain `volume`: the total is the four side columns
                    tot = sum((P.num(r.get(f)) or 0.0) for f in ("volume_ask_side", "volume_bid_side", "volume_mid_side", "volume_no_side"))
                    pairs.append((w[0], P.num(r.get("volume_ask_side")) or 0.0, w[1], P.num(r.get("volume_bid_side")) or 0.0,
                                  w[3], tot))
        if not pairs:
            continue
        a = np.array(pairs)
        out[tk] = dict(contract_minutes=len(a), strikes=ks,
                       corr_ask=round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 4),
                       corr_bid=round(float(np.corrcoef(a[:, 2], a[:, 3])[0, 1]), 4),
                       corr_total=round(float(np.corrcoef(a[:, 4], a[:, 5])[0, 1]), 4),
                       exact_total_share=round(float(np.mean(np.abs(a[:, 4] - a[:, 5]) < 0.5)), 4),
                       sum_ws=int(a[:, 4].sum()), sum_rest=int(a[:, 5].sum()))
    return out


def latency(day):
    la = json.load(open(os.path.join(SH, day.isoformat(), "latency.json")))
    rows = [(int(m), v) for m, v in la.items() if 570 <= int(m) < 960]
    if not rows:
        return {}
    med = [v["median"] for _, v in rows]
    p90 = [v["p90"] for _, v in rows]
    worst = sorted(rows, key=lambda x: -x[1]["p90"])[:3]
    return dict(minutes=len(rows), median_of_minute_medians=round(st.median(med), 3),
                median_of_minute_p90=round(st.median(p90), 3), worst_p90=[(f"{m // 60:02d}:{m % 60:02d}", v["p90"]) for m, v in worst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.day) if a.day else dt.datetime.now(P.NY).date()
    uw = P.UW()
    opt = json.load(open(os.path.join(SH, day.isoformat(), "opt_minutes.json")))
    res = {"day": day.isoformat(), "flow": flow(day, uw), "option_totals": option_totals(day, uw, opt),
           "option_minutes": option_minutes(day, uw, opt), "latency": latency(day), "uw_calls": uw.calls}
    json.dump(res, open(os.path.join(SH, day.isoformat(), "compare.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
