"""
check_shootout.py -- THE "SHOOTOUT" (user, 2026-11-23): how often does an ATM 0DTE call or put bought in the first minutes
of the session get INTO PROFIT at some point -- the path, not the hold? Descriptive (not a strategy test): the user's
best trades are the day's first few; this asks how much of that the opening tape hands to anyone.

ENTRY    SPY / QQQ / IWM, every session in the lake with a same-day expiry. At T = 09:30:30, 09:31, 09:32, 09:33, 09:35,
         09:40 (the shootout) and 10:30, 12:00, 14:00 (comparison): spot = the underlying's 5-s price at T; K = the
         nearest $1 strike; BOTH the K call and the K put, priced from their OWN prints on the tape (the last NBBO at or
         before T, within 30 s). Cost = mid + $0.0006 a share (the user's fills: mid + $0.06 a contract).
PATH     every print after T: exit value = mid - $0.0008 - $0.0010 (sell at the user's fills, $0.10 fees round trip).
         PROFIT = exit value > cost, i.e. the mid has risen ~ $0.0024 (24c a contract) above the entry mid.
         Reported per horizon H = 5 / 15 / 30 min: share reaching profit, reaching +10% and +20% of cost, the median best
         % gain, the median minutes to the first profit, and the share of sessions where BOTH the call and the put got
         into profit (two-way chop) vs exactly one (a clean leg) vs neither.

    python check_shootout.py            (writes _shootout_cache/rows.json and prints the tables)

RUN LOG
  2026-11-23 run 1 (mid-based, any profit): ~95% of entries clear the profit bar within seconds at EVERY hour, call and
    put both ~92% -- quote jitter, not information. Added the STRICT exit (sell at the bid, held two prints).
  2026-11-23 run 2, STRICT, 552 sessions x 3 tickers (~3,300 entries per time):
    +20% within 5 min   09:30:30 39% · 09:33 32% · 09:40 29% · 10:30 27% · 12:00 25% · 14:00 31%   (SPY 32 / 27 / 22 / 25 / 23 / 32)
    +20% within 15 min  53% · 50% · 51% · 49% · 44% · 49%
    minutes to +20%     2.4 · 3.4 · 4.0 · 4.1 · 4.2 · 3.2
    which side (15 min, +20%): ONE side 73-80% at every hour; BOTH 15% at 09:30 vs 7% at 12:00; NEITHER 9% vs 18%.
    READ: the opening minutes pay the RIGHT side ~1.5x as often within 5 minutes and about twice as fast as midday, and
    midday is the slowest hour on every measure (more sessions where neither side gets anywhere). It is speed, not
    direction: in three sessions of four only one side gets to +20%, so the opening rewards being right quickly and
    punishes being wrong just as quickly. Descriptive; random entries still have no edge (check_orb, breach studies).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CACHE = "_shootout_cache"
TIMES = {"09:30:30": 34230, "09:31": 34260, "09:32": 34320, "09:33": 34380, "09:35": 34500, "09:40": 34800,
         "10:30": 37800, "12:00": 43200, "14:00": 50400}
HORIZONS = (300, 900, 1800)
BUY, SELL, FEE = 0.0006, 0.0008, 0.0010


def day(d):
    import check_breach_ic as B
    import check_sixpack_lake as L
    try:
        path = L.price_path(d)
        tq = B.tape(d)
    except Exception as e:                               # noqa: BLE001
        return d, [], type(e).__name__
    rows = []
    for tk, (sc, hi, lo, cl) in path.items():
        if tk not in ("SPY", "QQQ", "IWM") or len(sc) < 100:
            continue
        for lab, T in TIMES.items():
            j = int(np.searchsorted(sc, T, side="right")) - 1
            if j < 0 or T - sc[j] > 30:
                continue
            spot = float(cl[j]); K = int(round(spot))
            rec = {"d": d, "tk": tk, "t": lab, "spot": spot, "K": K}
            for cp in ("C", "P"):
                c = tq.get((tk, K, cp))
                if c is None:
                    continue
                s, b, a = c
                i0 = int(np.searchsorted(s, T, side="right")) - 1
                if i0 < 0 or T - s[i0] > 30 or a[i0] <= 0.05:
                    continue
                cost = (b[i0] + a[i0]) / 2 + BUY
                out = {"cost": round(float(cost), 4)}
                for H in HORIZONS:
                    k0, k1 = i0 + 1, int(np.searchsorted(s, T + H, side="right"))
                    if k1 <= k0:
                        out[f"best{H}"] = None
                        continue
                    val = (b[k0:k1] + a[k0:k1]) / 2 - SELL - FEE
                    gain = (val - cost) / cost
                    best = float(gain.max())
                    hit = np.nonzero(val > cost)[0]
                    out[f"best{H}"] = round(best, 4)
                    out[f"tprof{H}"] = None if not hit.size else round(float(s[k0 + hit[0]] - T), 1)
                    # STRICT (added after run 1, which showed the mid-based bar is cleared by quote jitter within
                    # seconds at every hour): sell at the BID, and the gain must hold on two consecutive prints
                    vb = (b[k0:k1] - FEE - cost) / cost
                    sus = np.minimum(vb[:-1], vb[1:]) if vb.size > 1 else vb
                    out[f"strict{H}"] = round(float(sus.max()), 4) if sus.size else None
                    for thr in (0.10, 0.20):
                        h2 = np.nonzero(sus >= thr)[0]
                        out[f"t{int(thr * 100)}_{H}"] = None if not h2.size else round(float(s[k0 + h2[0] + 1] - T), 1)
                rec[cp] = out
            if "C" in rec or "P" in rec:
                rows.append(rec)
    return d, rows, None


def main():
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")))
    days += [d for d in ("2026-11-19", "2026-11-20") if d not in days]
    rows, errs = [], []
    with ProcessPoolExecutor(6) as ex:
        for d, r, e in ex.map(day, days):
            rows += r
            if e:
                errs.append((d, e))
    json.dump(rows, open(os.path.join(CACHE, "rows.json"), "w"))
    print(f"{len(rows)} entries over {len({r['d'] for r in rows})} sessions; errors {len(errs)} {errs[:3]}")
    strict_table(rows)


def strict_table(rows):
    """sell at the bid, gain held on two consecutive prints; chop = BOTH the call and the put reached +20% in 15 min."""
    for tk in ("SPY", "QQQ", "IWM", "ALL"):
        R = [r for r in rows if tk == "ALL" or r["tk"] == tk]
        print(f"\n===== STRICT {tk}: exit at the bid, held 2 prints")
        print(f"{'entry':9s} {'n':>5s} | {'+10% 5m':>8s} {'+20% 5m':>8s} {'+20% 15m':>8s} {'+50% 30m':>8s} | {'best15 med':>10s} "
              f"{'min to +20%':>11s} | {'both +20%':>9s} {'one':>5s} {'none':>5s} (15m)")
        for lab in TIMES:
            X = [r[cp] for r in R if r["t"] == lab for cp in ("C", "P") if cp in r and r[cp].get("strict900") is not None]
            if not X:
                continue
            f = lambda H, thr: np.mean([x[f"strict{H}"] >= thr for x in X if x.get(f"strict{H}") is not None])
            t20 = [x["t20_900"] for x in X if x.get("t20_900") is not None]
            pairs = [(r["C"]["strict900"], r["P"]["strict900"]) for r in R if r["t"] == lab and "C" in r and "P" in r
                     and r["C"].get("strict900") is not None and r["P"].get("strict900") is not None]
            both = np.mean([c >= .2 and p >= .2 for c, p in pairs]); one = np.mean([(c >= .2) != (p >= .2) for c, p in pairs])
            print(f"{lab:9s} {len(X):5d} | {f(300, .1):8.0%} {f(300, .2):8.0%} {f(900, .2):8.0%} {f(1800, .5):8.0%} | "
                  f"{np.median([x['strict900'] for x in X]):+10.0%} {np.median(t20) / 60 if t20 else np.nan:9.1f}m | "
                  f"{both:9.0%} {one:5.0%} {1 - both - one:5.0%}")
    for tk in ("SPY", "QQQ", "IWM", "ALL"):
        R = [r for r in rows if tk == "ALL" or r["tk"] == tk]
        print(f"\n===== {tk}   (each cell: calls and puts pooled; 'both' = the call AND the put got into profit that session)")
        print(f"{'entry':9s} {'n':>5s} | {'5 min: profit':>13s} {'+10%':>5s} {'+20%':>5s} | {'15 min: profit':>14s} {'+20%':>5s} {'best med':>8s} "
              f"{'t-to-profit':>11s} | {'30 min: profit':>14s} | {'both':>5s} {'one':>5s} {'none':>5s} (15 min)")
        for lab in TIMES:
            X = [r[cp] for r in R if r["t"] == lab for cp in ("C", "P") if cp in r]
            if not X:
                continue
            def rate(H, thr):
                v = [x[f"best{H}"] for x in X if x.get(f"best{H}") is not None]
                return np.mean([b > thr for b in v]) if v else np.nan
            b15 = [x["best900"] for x in X if x.get("best900") is not None]
            tp = [x["tprof900"] for x in X if x.get("tprof900") is not None]
            pairs = [(r["C"].get("best900"), r["P"].get("best900")) for r in R if r["t"] == lab and "C" in r and "P" in r
                     and r["C"].get("best900") is not None and r["P"].get("best900") is not None]
            both = np.mean([c > 0 and p > 0 for c, p in pairs]) if pairs else np.nan
            one = np.mean([(c > 0) != (p > 0) for c, p in pairs]) if pairs else np.nan
            print(f"{lab:9s} {len(X):5d} | {rate(300, 0):13.0%} {rate(300, .10):5.0%} {rate(300, .20):5.0%} | {rate(900, 0):14.0%} "
                  f"{rate(900, .20):5.0%} {np.median(b15):+8.0%} {np.median(tp) / 60 if tp else np.nan:9.1f}m | {rate(1800, 0):14.0%} | "
                  f"{both:5.0%} {one:5.0%} {1 - both - one:5.0%}")


if __name__ == "__main__":
    main()
