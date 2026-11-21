"""
check_breach_vehicle.py -- PART B: AT A THRU BREACH, WHICH CONTRACT IS THE BETTER TRADE -- K, K+1 OR K+2?
(user, 2026-11-20; Part A, check_breach_next.py, found no information at K+2.) PRE-REGISTERED 2026-11-20, committed
before any P&L below was computed.

QUESTION. The strike two out is cheaper and more leveraged if price carries through, but its spread is wider in % and
it decays faster. Priced honestly at the user's fills, does buying K+1 or K+2 in the breach direction beat buying the
breached strike K -- per dollar of premium -- and does any of the three pay at all?

EVENTS   _breach_cache/events (check_breach_ic): the same breaches, t_in = when the breach is KNOWN (+1 s latency).
CONTRACTS the breach-direction contract (call on an up-break, put on a down-break), same-day expiry, at
          K0 = K, K1 = K + side, K2 = K + 2 x side.
PRICING  check_breach_ic's method, unchanged: each contract's NBBO from its OWN prints on the lake tape (the last print at
         or before the moment, within 30 s; no quote -> that trade is not priced). User fills (cleanbot-real-fills):
         buy mid + $0.06, sell mid - $0.08 per contract, $0.10 fees round trip. Exits judged on the same 5-s median
         path with its knowable time + 1 s.
EXITS    A  take profit when price reaches the next strike (K + side x 0.95), stop when it falls back through K by
            $0.10, else out at t_in + 15 min (the breach study's continuation trade).
         B  take profit two strikes out (K + side x 1.95), the same stop, else out at t_in + 30 min (let it run).
MEASURES R% = P&L / (entry cost x 100): return on the premium paid (the leverage question). $ = P&L per contract.
         Primary comparison on the COMMON subset (all three contracts priced), so a contract that is missing when it
         is illiquid cannot flatter itself. Groups: ALL breaches and the frozen THRU score's TOP fifth (>= q80).
SAMPLES  DISCOVERY 2024-08-20 .. 2025-12-31 (reported). HOLDOUT 2026-01-02 .. 2026-11-13, scored ONCE (--holdout writes
         _breach_vehicle_cache/RESULT-holdout.txt and refuses to run again). PARTLY CLEAN, said now: check_breach_ic
         scored the K contract's exit-A money on 2026-01..09-25 (top fifth +$0.20, FAIL); K1 / K2 and exit B never.
PASS RULES (holdout, day-block bootstrap 2,000, BH q < 0.10 across each family)
  V1 BETTER VEHICLE: paired R%(K1) - R%(K0) and R%(K2) - R%(K0), per exit, per group (8 tests). A pass = mean > 0 with
     the 95% CI above 0 and BH q < 0.10.
  V2 PAYS: mean R% > 0 with the CI above 0 for each contract x exit x group (12 tests), BH q < 0.10.
  Either way the result goes in the THRU pill's tooltip as "which contract, at your fills". Nothing changes in the bot.
Reported: coverage (how often each contract is priced), win rate, mean entry price, $ per contract.

Usage:  python check_breach_vehicle.py --build | --discovery | --holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CACHE = "_breach_vehicle_cache"
DISC, HOLD = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-13")
EXITS = {"A": (0.95, 900), "B": (1.95, 1800)}
CON = {"K0": 0, "K1": 1, "K2": 2}
SEED = 20261122


def price(tq, tk, Kc, cp, side, K, t_in, pm, tp_d, hold):
    """(pnl $, R%, entry mid) for one contract and one exit, or None."""
    import check_breach_ic as B
    sc, med = pm
    end = t_in + hold
    c = tq.get((tk, Kc, cp))
    if c is None:
        return None

    def q(T):
        s, b, a = c
        j = int(np.searchsorted(s, T, side="right")) - 1
        return None if j < 0 or T - s[j] > 30 else (b[j], a[j])
    e = q(t_in)
    if e is None or e[1] <= 0.05:
        return None
    kt = B._kt(sc)
    j0 = int(np.searchsorted(kt, t_in, side="right"))
    tx = end
    for k in range(j0, len(sc)):
        T = float(kt[k]) + B.LAT
        if T > end:
            break
        if side * (med[k] - K) >= tp_d or side * (med[k] - K) <= -B.ZB:
            tx = T
            break
    x = q(tx)
    if x is None:
        return None
    emid = (e[0] + e[1]) / 2
    cost = emid + B.BUY_C
    pnl = ((x[0] + x[1]) / 2 - B.SELL_C - cost - B.FEE_RT) * 100
    return round(pnl, 4), round(pnl / (cost * 100) * 100, 4), round(emid, 4)


def build_day(d):
    import check_breach_ic as B
    import check_sixpack_lake as L
    evs = json.load(open(os.path.join("_breach_cache", "events", f"{d}.json")))
    try:
        path = L.price_path(d)
        tq = B.tape(d)
    except Exception as e:                               # noqa: BLE001
        return d, [], f"{type(e).__name__}"
    per = {}
    for tk, (sc, hi, lo, cl) in path.items():
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        per[tk] = (sc, med)
    rows = []
    for j, e in enumerate(evs):
        tk, K, side, t_in = e["t"], int(e["K"]), int(e["side"]), float(e["t_in"])
        if tk not in per:
            continue
        cp = "C" if side > 0 else "P"
        r = {"d": d, "j": j, "t": tk, "t_in": t_in}
        for cn, off in CON.items():
            for ex, (tp_d, hold) in EXITS.items():
                r[f"{cn}_{ex}"] = price(tq, tk, K + off * side, cp, side, K, t_in, per[tk], tp_d, hold)
        rows.append(r)
    return d, rows, None


def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")))
    days = [d for d in days if DISC[0] <= d <= HOLD[1]]
    out = {"disc": [], "hold": []}
    errs = []
    with ProcessPoolExecutor(6) as ex:
        for d, rows, err in ex.map(build_day, days):
            (out["disc"] if d <= DISC[1] else out["hold"]).extend(rows)
            if err:
                errs.append((d, err))
    for k, rows in out.items():
        json.dump(rows, open(os.path.join(CACHE, f"{k}.json"), "w"))
    print(f"built: discovery {len(out['disc'])} breaches, holdout {len(out['hold'])} (not read); errors {len(errs)} {errs[:5]}")


def enrich(rows, lo, hi):
    import check_breach_forward as BF
    frz = json.load(open("breach_model_frozen.json"))
    ev = {}
    for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")):
        d = os.path.basename(f)[:10]
        if lo <= d <= hi:
            for j, e in enumerate(json.load(open(f))):
                ev[(d, j)] = e
    for r in rows:
        e = ev[(r["d"], r["j"])]
        assert e["t"] == r["t"] and e["t_in"] == r["t_in"], "event cache changed since --build"
        r["top"] = BF.score(frz, e) >= frz["q80"]
    return rows


def boot(rows, vals, rng, n):
    days = sorted({r["d"] for r in rows}); by = {d: [] for d in days}
    for i, r in enumerate(rows):
        by[r["d"]].append(i)
    v = np.asarray(vals, float); out = []
    for _ in range(n):
        idx = np.concatenate([by[d] for d in rng.choice(days, len(days), replace=True)])
        out.append(v[idx].mean())
    out = np.array(out)
    return np.quantile(out, [.025, .975]).tolist(), float(min(1.0, 2 * min(np.mean(out <= 0), np.mean(out >= 0))))


def report(rows, rng, n_boot, lines):
    import check_breach_ic as B
    cover = {c: np.mean([r[f"{c}_A"] is not None for r in rows]) for c in CON}
    lines.append("coverage (priced at entry and exit, exit A): " + ", ".join(f"{c} {v:.0%}" for c, v in cover.items()))
    v1, v2 = [], []
    for grp in ("all", "top"):
        G = [r for r in rows if grp == "all" or r["top"]]
        for ex in EXITS:
            com = [r for r in G if all(r[f"{c}_{ex}"] is not None for c in CON)]
            lines.append(f"[{grp:3s} exit {ex}] common subset n {len(com)} / {len(G)} breaches, {len({r['d'] for r in com})} sessions")
            for c in CON:
                R = [r[f"{c}_{ex}"][1] for r in com]; P = [r[f"{c}_{ex}"][0] for r in com]; E = [r[f"{c}_{ex}"][2] for r in com]
                ci, p = boot(com, R, rng, n_boot)
                v2.append((f"{c} {ex} {grp}", np.mean(R), ci, p))
                lines.append(f"    {c}: entry mid ${np.mean(E):.2f} | R% {np.mean(R):+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                             f"${np.mean(P):+.2f}/contract | win {np.mean(np.array(P) > 0):.0%}")
            for c in ("K1", "K2"):
                Dv = [r[f"{c}_{ex}"][1] - r[f"K0_{ex}"][1] for r in com]
                ci, p = boot(com, Dv, rng, n_boot)
                v1.append((f"{c}-K0 {ex} {grp}", np.mean(Dv), ci, p))
                lines.append(f"    {c} - K0 (R% points): {np.mean(Dv):+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    for name, fam in (("V1 better vehicle (paired R% vs K0)", v1), ("V2 pays (R% > 0)", v2)):
        q = B.bh([x[3] for x in fam])
        passes = [x[0] for x, qq in zip(fam, q) if qq < 0.10 and x[2][0] > 0]
        lines.append(f"{name}: PASS {', '.join(passes) if passes else 'none'}")
    return lines


def discovery():
    rows = enrich(json.load(open(os.path.join(CACHE, "disc.json"))), *DISC)
    lines = [f"DISCOVERY {DISC[0]}..{DISC[1]}: {len(rows)} breaches (reported; the verdict is the holdout's)"]
    print("\n".join(report(rows, np.random.default_rng(SEED), 1000, lines)))


def holdout():
    out = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(out):
        sys.exit(f"{out} exists -- scored once")
    rows = enrich(json.load(open(os.path.join(CACHE, "hold.json"))), *HOLD)
    lines = [f"check_breach_vehicle HOLDOUT {dt.date.today()} -- {HOLD[0]}..{HOLD[1]}: {len(rows)} breaches"]
    lines = report(rows, np.random.default_rng(SEED + 1), 2000, lines)
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.discovery:
        discovery()
    if a.holdout:
        holdout()
