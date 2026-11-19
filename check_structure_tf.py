"""
check_structure_tf.py -- THE PULLBACK SETUP ON 5 / 10 / 15-MINUTE SWINGS, AND TOP-DOWN ALIGNMENT (user, 2026-11-19).
PRE-REGISTERED 2026-11-19, committed before any of it was computed.

WHY. check_structure (1-minute swings) beat random entries on "2R before 1R" but the option money stayed inside zero:
the stop was ~0.027% of price (~$0.20 SPY), so spread + fees + decay ate the edge. On larger swings the stop and target
are several times wider while the per-contract cost is the same.

SETUPS (check_structure's rules, rescaled; SPY / QQQ / IWM; entries 09:35-15:30)
  bars        tf = 5, 10, 15 minutes: the close of each tf-bar of the 5-second median path (known 10 s after the bar ends)
  swings      zigzag on tf-bar closes, threshold R = 0.10% x sqrt(tf) of the day's first price (5m 0.224%, 10m 0.316%,
              15m 0.387%) -- moves scale with the square root of time
  setup       a LOWER LOW under the last low (both structures: after a failed breakout or a plain pullback), then the
              bounce's high tracked on the 5-second path; entry when price comes 0.02% x sqrt(tf) off it while it is
              below the prior high (a lower high); stop 0.0067% x sqrt(tf) above the bounce high. Mirror for longs.
              VOID if the bounce reaches the prior high or no entry within 30 tf-bars.
  outcome     2R before the stop within 30 tf-bars, cut at 15:55.
  controls    RANDOM: 3 same-day entries per setup (same ticker, direction, risk in % of price, same horizon), >= 1 tf-bar
              away. PULLBACK-vs-FAILED-BREAKOUT is reported, not scored.
  money       ATM option in the trade's direction, target 2R (T2, fixed -- no target choice), stop, horizon, tape-priced at
              the user's fills (buy mid + $0.06, sell mid - $0.08, $0.10 fees); all setups.
ALIGNMENT (H4) the existing 1-minute setups (_structure_cache/events): split by the 15-MINUTE swing structure at the entry
              (last two confirmed 15m highs and lows, R = 0.387%): aligned = a short with LH/LL or a long with HH/HL.
              Outcome: 2R before the stop (check_structure's own); money T2 (check_structure's T2).
HYPOTHESES (HOLDOUT, scored once; BH across the four money p-values)
  H1 / H2 / H3   5 / 10 / 15-minute setups: T2 mean P&L per contract > 0, day-block 95% CI above 0, BH q < 0.10
  H4             1-minute setups aligned with the 15-minute structure: T2 mean P&L per contract > 0, CI above 0, BH q <
                 0.10 (also reported: 2R rate aligned vs not aligned)
  Reported for each timeframe: 2R rate of setups vs random vs pullback-only vs failed-breakout-only; stop distance in $.
SAMPLES  DISCOVERY 2024-08-20 .. 2025-12-31 (everything reported; nothing chosen). HOLDOUT 2026-01-02 .. 2026-11-13 --
         PARTLY CLEAN (the 1-minute version of this setup was scored on it 11-17); forward from 2026-11-16 is the real
         confirmation. --holdout writes _structure_tf_cache/RESULT-holdout.txt and refuses to run again.

Usage:  python check_structure_tf.py --build | --discovery | --holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import check_breach_ic as B                        # noqa: E402
import check_sixpack_lake as L                     # noqa: E402
import check_structure as S                        # noqa: E402

CACHE = os.path.join(ROOT, "_structure_tf_cache")
EV = os.path.join(CACHE, "events")
DISC, HOLD = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-13")
TFS = (5, 10, 15)
EOD = 955 * 60
SEED = 20261119


def tf_closes(sc, z, tf):
    """{bar_start_minute: (value, known_at)} closes of tf-minute bars (from 09:30) of the 5-second series z."""
    out = {}
    for s, v in zip(sc, z):
        m = int(s // 60)
        if 570 <= m < 960:
            b = 570 + ((m - 570) // tf) * tf
            out[b] = v
    return {b: (v, (b + tf) * 60 + 10) for b, v in out.items()}


def setups_tf(sc, med, kt, p0, orient, tf):
    k = math.sqrt(tf)
    R, TURN, BUF, WAIT = S.R_FRAC * k * p0, S.TURN_FRAC * k * p0, S.BUF_FRAC * k * p0, 30 * tf * 60
    z = orient * med
    piv = S.zigzag(tf_closes(sc, z, tf), R)
    out = []
    for i in range(2, len(piv)):
        Le, Hd, Lc = piv[i], piv[i - 1], piv[i - 2]
        if Le[0] != "L" or Hd[0] != "H" or not Le[1] < Lc[1]:
            continue
        fb = i >= 4 and piv[i - 3][0] == "H" and piv[i - 4][0] == "L" and Lc[1] > piv[i - 4][1] and Hd[1] > piv[i - 3][1]
        Tc = Le[3]
        if Tc > S.T_HI:
            continue
        k0 = int(np.searchsorted(sc, Le[2] * 60))
        j = int(np.searchsorted(kt, Tc, side="left"))
        if j >= len(sc):
            continue
        M = float(np.max(z[k0:j + 1])) if j >= k0 else float(z[j])
        ent = None
        for kk in range(j, len(sc)):
            if kt[kk] > Tc + WAIT:
                break
            M = max(M, float(z[kk]))
            if M >= Hd[1]:
                break
            if M - z[kk] >= TURN:
                ent = (float(kt[kk]) + B.LAT, float(z[kk]), M)
                break
        if ent is None or not S.T_LO <= ent[0] <= S.T_HI:
            continue
        out.append(dict(fb=fb, t_in=ent[0], z_e=ent[1], z_s=ent[2] + BUF, hold=min(30 * tf * 60, EOD - ent[0])))
    return out


def outcome(sc, med, kt, orient, t_in, z_e, z_s, hold):
    z = orient * med
    risk = z_s - z_e
    if risk <= 0 or hold <= 0:
        return None
    for kk in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        if kt[kk] + B.LAT > t_in + hold:
            break
        if z[kk] >= z_s:
            return dict(hit2=False, risk=risk)
        if z_e - z[kk] >= 2 * risk:
            return dict(hit2=True, risk=risk)
    return dict(hit2=False, risk=risk)


def money_t2(tq, tk, orient, t_in, z_e, z_s, hold, sc, med, kt):
    p_e = orient * z_e
    K = int(round(p_e))
    c = tq.get((tk, K, "P" if orient > 0 else "C"))
    if c is None:
        return None
    def q(T):
        s, b, a = c
        i = int(np.searchsorted(s, T, side="right")) - 1
        return None if i < 0 or T - s[i] > 30 else (b[i], a[i])
    e = q(t_in)
    if e is None or e[1] <= 0.05:
        return None
    z = orient * med
    zt = z_e - 2 * (z_s - z_e)
    tx = t_in + hold
    for kk in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        T = float(kt[kk]) + B.LAT
        if T > t_in + hold:
            break
        if z[kk] >= z_s or z[kk] <= zt:
            tx = T
            break
    x = q(tx)
    if x is None:
        return None
    return ((x[0] + x[1]) / 2 - B.SELL_C - ((e[0] + e[1]) / 2 + B.BUY_C) - B.FEE_RT) * 100


def structure15(sc, med, p0, t):
    """the 15-minute swing structure known at t: 'down' (LH/LL), 'up' (HH/HL), or None."""
    mc = {b: v for b, v in tf_closes(sc, med, 15).items() if v[1] <= t}
    piv = S.zigzag(mc, S.R_FRAC * math.sqrt(15) * p0)
    piv = [p for p in piv if p[3] <= t]
    hs = [p[1] for p in piv if p[0] == "H"][-2:]
    ls = [p[1] for p in piv if p[0] == "L"][-2:]
    if len(hs) < 2 or len(ls) < 2:
        return None
    if hs[1] < hs[0] and ls[1] < ls[0]:
        return "down"
    if hs[1] > hs[0] and ls[1] > ls[0]:
        return "up"
    return None


def build_day(d):
    f = os.path.join(EV, d + ".json")
    if os.path.exists(f):
        return d, "have"
    try:
        path, tq = L.price_path(d), B.tape(d)
    except Exception as e:
        return d, f"skip {type(e).__name__}"
    rng = np.random.default_rng(abs(hash(d)) % (2 ** 32))
    rows = []
    meds = {}
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300 or len(sc) < 200:
            continue
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        kt = B._kt(sc)
        p0 = float(med[0])
        meds[tk] = (sc, med, kt, p0)
        for tf in TFS:
            for orient in (1, -1):
                for s in setups_tf(sc, med, kt, p0, orient, tf):
                    o = outcome(sc, med, kt, orient, s["t_in"], s["z_e"], s["z_s"], s["hold"])
                    if o is None:
                        continue
                    ctl = []
                    for _ in range(3):
                        for _try in range(20):
                            t = float(rng.uniform(S.T_LO, S.T_HI))
                            if abs(t - s["t_in"]) >= tf * 60:
                                break
                        jj = int(np.searchsorted(kt, t, side="right")) - 1
                        if jj < 0:
                            continue
                        ze = float(orient * med[jj])
                        oc = outcome(sc, med, kt, orient, t, ze, ze + o["risk"], min(s["hold"], EOD - t))
                        if oc:
                            ctl.append(oc["hit2"])
                    rows.append(dict(kind=f"tf{tf}", d=d, t=tk, side=-orient, fb=s["fb"], t_in=round(s["t_in"], 1),
                                     hit2=o["hit2"], risk_usd=o["risk"], rand_hit2=float(np.mean(ctl)) if ctl else None,
                                     T2=money_t2(tq, tk, orient, s["t_in"], s["z_e"], s["z_s"], s["hold"], sc, med, kt)))
    # H4: the 1-minute setups, split by the 15-minute structure
    f1 = os.path.join(S.EV, d + ".json")
    if os.path.exists(f1):
        for r in json.load(open(f1)):
            if r["t"] not in meds:
                continue
            sc, med, kt, p0 = meds[r["t"]]
            st = structure15(sc, med, p0, r["t_in"])
            al = None if st is None else ((st == "down" and r["side"] < 0) or (st == "up" and r["side"] > 0))
            rows.append(dict(kind="align", d=d, t=r["t"], side=r["side"], fb=r["struct"] == "fb", t_in=r["t_in"],
                             hit2=r["hit2"], aligned=al, T2=r["T2"]))
    os.makedirs(EV, exist_ok=True)
    json.dump(rows, open(f, "w"))
    return d, len(rows)


def load(lo, hi):
    return [r for f in sorted(glob.glob(os.path.join(EV, "*.json"))) if lo <= os.path.basename(f)[:10] <= hi
            for r in json.load(open(f))]


def report(rows, title, scored=False):
    rng = np.random.default_rng(SEED)
    lines = [f"{title}: {len({r['d'] for r in rows})} sessions"]
    ps, money = [], {}
    def boot_mean(rs, key):
        v = np.array([r[key] for r in rs], float)
        bs = B.day_boot(rs, lambda i: v[i].mean(), rng)
        return v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    for tf in TFS:
        rs = [r for r in rows if r["kind"] == f"tf{tf}"]
        if len(rs) < 30:
            lines.append(f"  {tf}m: n={len(rs)} too few"); ps.append(1.0); continue
        h = lambda x: np.mean([r["hit2"] for r in x]) if x else float("nan")
        rnd = np.nanmean([r["rand_hit2"] for r in rs if r["rand_hit2"] is not None])
        fb = [r for r in rs if r["fb"]]; pb = [r for r in rs if not r["fb"]]
        mv = [r for r in rs if r["T2"] is not None]
        m, lo, hi, p = boot_mean(mv, "T2")
        ps.append(p); money[f"H{TFS.index(tf) + 1} {tf}m"] = (m, lo, hi)
        lines.append(f"  {tf:2d}m: n={len(rs)} ({len(rs) / max(len({r['d'] for r in rs}), 1):.1f}/day) | 2R first {h(rs):.1%} vs random {rnd:.1%}"
                     f" (failed breakout {h(fb):.1%}, pullback {h(pb):.1%}) | median stop ${np.median([r['risk_usd'] for r in rs if r['t'] == 'SPY'] or [np.nan]):.2f} SPY"
                     f" | T2 ${m:+.2f}/contract [{lo:+.2f}, {hi:+.2f}] (n {len(mv)})")
    al = [r for r in rows if r["kind"] == "align"]
    a = [r for r in al if r["aligned"] is True]; n = [r for r in al if r["aligned"] is False]
    if a and n:
        am = [r for r in a if r["T2"] is not None]
        m, lo, hi, p = boot_mean(am, "T2")
        ps.append(p); money["H4 1m aligned w/ 15m"] = (m, lo, hi)
        nm = [r["T2"] for r in n if r["T2"] is not None]
        lines.append(f"  align: 1m setups with the 15m structure n={len(a)} 2R first {np.mean([r['hit2'] for r in a]):.1%} T2 ${m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                     f" | against n={len(n)} {np.mean([r['hit2'] for r in n]):.1%} T2 ${np.mean(nm):+.2f} | no clear 15m structure n={len([r for r in al if r['aligned'] is None])}")
    if scored:
        qs = B.bh(ps)
        for (name, (m, lo, hi)), q in zip(money.items(), qs):
            lines.append(f"  {name}: mean ${m:+.2f} [{lo:+.2f}, {hi:+.2f}] q {q:.3f} -> {'PASS' if m > 0 and lo > 0 and q < 0.10 else 'FAIL'}")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "discovery", "holdout"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.build:
        days = [d for d in L.lake_days(DISC[0], HOLD[1])
                if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))]
        with ProcessPoolExecutor(6) as ex:
            for k, (d, msg) in enumerate(ex.map(build_day, days)):
                if k % 50 == 0:
                    print(k, d, msg, flush=True)
        print("built", len(days))
    if a.discovery:
        print("\n".join(report(load(*DISC), f"DISCOVERY {DISC[0]}..{DISC[1]}")))
    if a.holdout:
        res = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(res):
            sys.exit(f"the holdout was scored already -- see {res}")
        lines = report(load(*HOLD), f"HOLDOUT {HOLD[0]}..{HOLD[1]} scored {dt.datetime.now():%Y-%m-%d %H:%M} (partly clean)", scored=True)
        print("\n".join(lines))
        open(res, "w").write("\n".join(lines) + "\n")
