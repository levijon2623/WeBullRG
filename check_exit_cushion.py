"""
check_exit_cushion.py -- IS THE BOOK'S ADVERSE-EXIT CUSHION RIGHT? Measured on the tape (user, 2026-11-14).

sim_core prices a trail / stop exit at   min(minute-close bid, level) - spread x m
with m = CUSHION_CAP (SPY 1.50, QQQ 1.50, IWM 1.27) for `botcap`, 1.5 for `bot`. The cushion stands in for the
time between the trigger and the fill, which minute bars cannot see. The live bot fires the moment the BID touches
the level and sends a marketable sell; the user's 220 real 0DTE sells (July-Oct 2026, matched to their own prints)
filled on average $0.08/contract from the mid, i.e. at the bid or better.

For every trail / stop exit the deployed SPY / QQQ / IWM rules take in the sim (full live gates, sequential walk,
2024-08-20 onward), this finds on the tape (lake trades-core: every print carries the NBBO) the first print, from the
start of the sim's exit minute, whose bid is at or below the level, and prices the exit 1 s later:
  tape_bid  = the bid then (a marketable sell that gets no improvement)
  tape_user = mid - $0.0008 (the user's measured average on sells)
and compares both with the sim's price, in ROE points of the entry.

RUN LOG
  2026-11-14: 105 trail/stop exits (SPY 7, QQQ 24, IWM 74), the bid reached the level on the tape in 104.
    Tape bid 1 s after the bid touched the level vs the sim's `botcap` price: mean +4.9pp of entry, median +3.2pp
    (SPY +6.7/+5.8, QQQ +8.9/+3.0, IWM +3.5/+3.0); at the user's measured sell fill +7.8pp. `bot`: +4.4 / +2.9.
    The bid 1 s later sits 4.5pp BELOW the level on average, so slippage is real -- but the sim charges it twice
    (the minute-close bid, then the cushion on top). Small sample; the live cushion only prices a marketable limit
    and is NOT changed by this. Research on SPY/QQQ/IWM trail exits is ~3-5pp per adverse exit too pessimistic.

Only the tape's three tickers (trades-core holds SPY / QQQ / IWM); the wide names keep their cushions untested.
Usage:  python check_exit_cushion.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

import sim_core
from config import ADVERSE_TAGS, CUSHION_CAP, RULES, TRAIL_PCT

LAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lake")
TAPE_TK = ("SPY", "QQQ", "IWM")
LAT, USER_SELL = 1.0, 0.0008
_TAPE = {}


def tape(d):
    if d not in _TAPE:
        src = glob.glob(os.path.join(LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))
        if not src:
            _TAPE[d] = None
            return None
        x = (pl.scan_parquet(src).filter(pl.col("canceled") != True)
             .select("underlying_symbol", "executed_at", "option_type", pl.col("strike").cast(pl.Float64),
                     pl.col("expiry").cast(pl.Utf8), pl.col("nbbo_bid").cast(pl.Float64), pl.col("nbbo_ask").cast(pl.Float64))
             .collect())
        et = x["executed_at"].dt.convert_time_zone("America/New_York")
        x = x.with_columns(sec=et.dt.hour().cast(pl.Float64) * 3600 + et.dt.minute().cast(pl.Float64) * 60
                           + et.dt.second().cast(pl.Float64) + et.dt.microsecond().cast(pl.Float64) / 1e6,
                           cp=pl.col("option_type").str.to_uppercase().str.slice(0, 1))
        x = x.filter((pl.col("nbbo_ask") > pl.col("nbbo_bid")) & (pl.col("nbbo_bid") > 0)).sort("executed_at")
        _TAPE[d] = {k: (g["sec"].to_numpy(), g["nbbo_bid"].to_numpy(), g["nbbo_ask"].to_numpy())
                    for k, g in x.group_by(["underlying_symbol", "expiry", "cp", "strike"])}
    return _TAPE[d]


def parse_cid(cid):
    """'SPY250612C00600000' -> ('SPY', '2025-06-12', 'C', 600.0)"""
    i = next(j for j in range(len(cid)) if cid[j].isdigit())
    r, y = cid[:i], cid[i:i + 6]
    return r, f"20{y[:2]}-{y[2:4]}-{y[4:6]}", cid[i + 6], int(cid[i + 7:]) / 1000.0


def replay_level(path, pol):
    """The level and bar index at which sim_core.simulate's stop / trail fires (same logic, no fills)."""
    e_mid, _ea, cl, _hi, lo, _b, _a, mod = path
    ref, peak = e_mid, e_mid
    hard = ref * (1 - pol["stop"]) if pol.get("stop") else None
    tp = ref * (1 + pol["tp"]) if pol.get("tp") else None
    for i in range(len(cl)):
        lvl = hard
        if pol.get("trail"):
            t = peak * (1 - pol["trail"])
            if lvl is None or t > lvl:
                lvl = t
        if lvl is not None and lo[i] <= lvl:
            return i, lvl
        if tp is not None and cl[i] >= tp:
            return None
        peak = max(peak, cl[i])
    return None


def main():
    import directional_flow_backtester as D
    rows = []
    for r in RULES:
        if not r.get("enabled", True) or r["ticker"] not in TAPE_TK:
            continue
        tk = r["ticker"]
        meta = []
        cand = sim_core.deployed_candidates(D, r, meta_out=meta)
        assert len(meta) == len(cand) and all(m["mod"] == c[1] for m, c in zip(meta, cand)), "meta misaligned"
        pol, em = sim_core.policy_for(r, TRAIL_PCT), sim_core.eod_mod(r)
        for fill, cap in (("botcap", CUSHION_CAP.get(tk)), ("bot", None)):
            picks = []
            res = sim_core.walk(cand, pol, em, fill=fill, cush_cap=cap, with_tags=True, picks_out=picks)
            for (d, pnl, tag), (ci, xm) in zip(res, picks):
                if tag not in ADVERSE_TAGS:
                    continue
                path = cand[ci][2]
                hit = replay_level(path, pol)
                if hit is None:
                    continue
                i, lvl = hit
                if int(path[7][i]) != xm:
                    continue
                entry = min(round(path[0] + 0.01, 2), round(path[1], 2))
                sim_px = (pnl + D.commission_frac(entry) + 1) * entry
                t = tape(str(d))
                if t is None:
                    continue
                root, exp, cp, k = parse_cid(meta[ci]["cid"])
                c = t.get((root, exp, cp, k))
                if c is None:
                    continue
                s, b, a = c
                j0 = int(np.searchsorted(s, xm * 60.0))
                j1 = int(np.searchsorted(s, em * 60.0))
                w = np.nonzero(b[j0:j1] <= lvl + 1e-9)[0]
                if not w.size:
                    rows.append(dict(rule=r["name"], tk=tk, fill=fill, d=d, found=False))
                    continue
                T = s[j0 + w[0]] + LAT
                q = int(np.searchsorted(s, T, side="right")) - 1
                xb, xa = b[q], a[q]
                rows.append(dict(rule=r["name"], tk=tk, fill=fill, d=d, found=True, entry=entry, lvl=lvl, sim=sim_px,
                                 bid=xb, user=(xb + xa) / 2 - USER_SELL, spread=xa - xb, wait=s[j0 + w[0]] - xm * 60.0,
                                 sim_close_bid=float(path[5][i])))
    print(f"adverse (trail/stop) exits on the tape tickers, sim from 2024-08-20, tape to the end of trades-core\n")
    for fill in ("botcap", "bot"):
        print(f"=== sim fill `{fill}`")
        for tk in TAPE_TK + ("ALL",):
            rs = [x for x in rows if x["fill"] == fill and (tk == "ALL" or x["tk"] == tk)]
            ok = [x for x in rs if x["found"]]
            if not ok:
                continue
            f = lambda key: np.array([(x[key] - x["sim"]) / x["entry"] * 100 for x in ok])
            gb, gu = f("bid"), f("user")
            lvl_gap = np.array([(x["bid"] - x["lvl"]) / x["entry"] * 100 for x in ok])
            print(f"  {tk:4s} exits {len(rs):4d} (bid reached the level on the tape: {len(ok)})  "
                  f"tape bid vs sim: mean {gb.mean():+5.1f}pp median {np.median(gb):+5.1f}pp | "
                  f"user-like vs sim: mean {gu.mean():+5.1f}pp | tape bid vs level {lvl_gap.mean():+5.1f}pp | "
                  f"median wait into the minute {np.median([x['wait'] for x in ok]):.0f}s | "
                  f"spread ${np.median([x['spread'] for x in ok]) * 100:.0f}/contract")
    import json
    json.dump([{k: (str(v) if k == "d" else v) for k, v in x.items()} for x in rows],
              open(os.path.join("_book_cache_exit_cushion.json"), "w"), default=float)


if __name__ == "__main__":
    main()
