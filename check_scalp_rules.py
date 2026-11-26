"""
check_scalp_rules.py -- THE USER'S TWO OPENING SCALPS AS PLAIN RULES, THEN AN IC / LASSO SCREEN (user, 2026-11-25).
PRE-REGISTERED 2026-11-25, committed before any P&L below was computed on any historical session.

WHERE IT COMES FROM. The user's last 5 sessions (11-19..11-25, 59 SPY scalps, positive on average, median hold 32 s) were
classified against the ES 1-second path: 34 CHASE entries (the 5-minute move already going the trade's way, no pullback
in the last minute) and 12 FADE entries (against the 5-minute move). On those 5 days the plain rules (delta x move
approximation, about $3 too generous) made CHASE +$3.79 a signal and FADE -$1.70 with a 30 s exit, while the signals
the user actually took did clearly better -- the user's PICKING and EXITS carried the result. The user's view: not
skill until shown; looking for reasonable explanations.

DATA   lake 5-second underlying path (check_sixpack_lake.price_path: option prints' underlying price; median-free closes),
       the option tape for pricing (check_breach_ic.tape), SPY / QQQ / IWM extended 1-minute candles (volume), the
       same-day directional-delta minutes (check_overnight.dd_day cache), Market Tide (_breach_cache/tide), dealer
       gamma (spot-exposures-1m), check_overnight night rows. Sessions 2024-08-20 .. 2026-11-13 (tide + gamma coverage).
CLOCK  signals are checked every 5 s from 09:31:00 to 10:05:00 ET on the 5-s close KNOWN at t (bar end <= t).
       P(t) = that close. The 5-minute lookback is clipped at 09:30:00 (no pre-market path in the lake).
RULES  (thresholds in bp of price; 1.4 / 0.7 / 0.3 bp = the $0.10 / $0.05 / $0.02 used on SPY ~715 in step 1)
  m300 = P(t) - P(max(t - 300, 09:30)),  m60 = P(t - 10) - P(t - 60),  m10 = P(t) - P(t - 10)
  CHASE  |m300| >= 1.4 bp; side = sign(m300); and side x m60 >= -0.7 bp (no pullback in the last minute)
  FADE   |m300| >= 1.4 bp; side = -sign(m300); and side x m10 > 0.3 bp (just ticked back against the 5-minute move)
  one open trade per rule per ticker; after an exit the rule waits 10 s.
TRADE  buy the ATM same-day contract (strike nearest P(t)) in the side's direction at t + 1 s; EXIT A at t + 31 s (the
       user's median hold ~32 s). Priced on the tape: each contract's NBBO from its own last print at or before the
       moment (at most 30 s old), the user's fills: buy mid + $0.06, sell mid - $0.08 a contract, $0.10 fees.
       Unpriced trades are counted, not filled in.
PRIMARY PLAIN RULES (SPY)   R1 CHASE, R2 FADE: mean P&L per trade > 0 with the day-block bootstrap 95% CI above 0.
SCREEN (SPY, one per rule)  inputs (17, all known at t; signed = with the trade is +)
  PRICE   m300, m60, m10 (signed, bp); c_open (signed move since 09:30, bp); c_range (signed position in the range so
          far); c_tod (s since 09:30); q_spread (the ATM contract's spread / mid at entry)
  FLOW    f_dd5, f_dd1 (same-day directional delta, last 5 closed minutes / last closed minute, signed);
          f_vol (STOCK volume pace: last 5 closed minutes / the 15 before, per minute)
  MARKET  c_gex (dealer gamma, last closed minute); c_rvol (stock volume to the last closed minute / the 20-session
          median of the same); c_tide (Market Tide 15-minute change, signed)
  CROSS   x_ret60, x_ret300 (the other two indexes' mean move over 60 s / 5 min, signed, bp)   [the Lasso kept the
          cross-index input in both earlier screens]
  NIGHT   o_gapz (check_overnight z), o_with_gap (1 if the trade's side is the gap's side)
  ranks within ticker against the DISCOVERY distribution (mid-rank ECDF), missing -> 0.5. TARGET = trade P&L ($).
  S1 IC (discovery): Spearman(input, P&L), day-block p, BH across the 17, same sign in both halves. Reported.
  S2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on the discovery ranks. PASS = keeps >= 1 input AND holdout
     Spearman(score, P&L) > 0 with the day-block 95% CI above 0.
  S3 MONEY: the holdout's TOP fifth by score (cuts from discovery): mean P&L > 0 with the day-block CI above 0.
  A rule's screen PASSES only with S2 AND S3.
FAMILY  BH q < 0.10 across the four money tests (R1, R2, S3-CHASE, S3-FADE). Discovery estimate must share the sign.
SAMPLES DISCOVERY = sessions in ODD calendar months, HOLDOUT = EVEN (each session whole). Partly clean, said now: the
  rules came from the user's 5 sessions in 11-2026 (after the sample); the inputs' families were screened on these
  sessions before for other events (strike breaches, pullbacks). --holdout writes _scalp_cache/RESULT-holdout.txt and
  refuses to run again.
REPORTED  exit at t + 61 s; QQQ / IWM plain rules; signals per session; the user's own 5 sessions scored by the frozen
  discovery score once the lake has them (does it rank the user's picks above the other signals?).

    python check_scalp_rules.py --build | --discovery | --holdout
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
CACHE = "_scalp_cache"
TKS = ("SPY", "QQQ", "IWM")
T0, T1, OPEN = 9 * 3600 + 31 * 60, 10 * 3600 + 5 * 60, 9 * 3600 + 30 * 60
STEP, COOL, LAT = 5, 10, 1
HOLD_A, HOLD_B = 30, 60
MOVE5, PULL1, TICK10 = 1.4, 0.7, 0.3          # bp
FIRST, LAST = "2024-08-20", "2026-11-13"
INPUTS = ["m300", "m60", "m10", "c_open", "c_range", "c_tod", "q_spread", "f_dd5", "f_dd1", "f_vol", "c_gex", "c_rvol",
          "c_tide", "x_ret60", "x_ret300", "o_gapz", "o_with_gap"]
SEED = 20261125


def sessions():
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "tide", "*.json")))
    return [d for d in days if FIRST <= d <= LAST]


def known_path(sc, cl):
    """(known time = bar end, close) arrays, bars of 5 s starting at sc."""
    return np.asarray(sc, float) + 5.0, np.asarray(cl, float)


def P(kp, t):
    kt, cl = kp
    j = int(np.searchsorted(kt, t, side="right")) - 1
    return float(cl[j]) if j >= 0 else np.nan


def bp(a, b):
    return (a - b) / b * 1e4


def ext_minutes(tk, d):
    f = os.path.join("lake", "silver", "ext-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(f):
        return None
    x = pl.read_parquet(f).filter(pl.col("market_time") == "r")
    return {m.hour * 60 + m.minute: (c, v, h, l) for m, c, v, h, l in x.select("minute", "close", "volume", "high", "low").iter_rows()}


def build_day(d):
    p = os.path.join(CACHE, "days", f"{d}.json")
    if os.path.exists(p):
        return d, json.load(open(p)), None
    import check_breach_ic as B
    import check_overnight as O
    import check_sixpack_lake as L
    try:
        path = L.price_path(d)
        tq = B.tape(d)
    except Exception as e:                                  # noqa: BLE001
        return d, [], type(e).__name__
    kp = {tk: known_path(*(path[tk][0], path[tk][3])) for tk in TKS if tk in path}
    tide = B.tide_series(d)
    out = []
    for tk in TKS:
        if tk not in kp:
            continue
        gex = L.gex_series(d, tk, {})
        dd = O.dd_day((tk, d))[1]
        mins = ext_minutes(tk, d) or {}
        others = [o for o in TKS if o != tk and o in kp]
        open_px = P(kp[tk], OPEN + 5)
        for rule in ("CHASE", "FADE"):
            busy = 0
            for t in range(T0, T1, STEP):
                if t < busy:
                    continue
                p0 = P(kp[tk], t)
                pa = P(kp[tk], max(t - 300, OPEN + 5))
                m300, m60, m10 = bp(p0, pa), bp(P(kp[tk], t - 10), P(kp[tk], t - 60)), bp(p0, P(kp[tk], t - 10))
                if not np.isfinite([m300, m60, m10]).all() or abs(m300) < MOVE5:
                    continue
                side = int(np.sign(m300)) if rule == "CHASE" else -int(np.sign(m300))
                if (rule == "CHASE" and side * m60 < -PULL1) or (rule == "FADE" and side * m10 <= TICK10):
                    continue
                K = int(round(p0))
                cp = "C" if side > 0 else "P"
                c = tq.get((tk, K, cp))
                row = dict(d=d, tk=tk, rule=rule, t=t, side=side, K=K)
                busy = t + HOLD_A + LAT + COOL

                def q(T, c=c):
                    if c is None:
                        return None
                    s, b, a = c
                    j = int(np.searchsorted(s, T, side="right")) - 1
                    return None if j < 0 or T - s[j] > 30 else (float(b[j]), float(a[j]))
                e = q(t + LAT)
                for lab, hold in (("A", HOLD_A), ("B", HOLD_B)):
                    x = q(t + LAT + hold)
                    row[f"pnl_{lab}"] = (None if (e is None or x is None or e[1] <= 0.05) else
                                         round(((x[0] + x[1]) / 2 - B.SELL_C - (e[0] + e[1]) / 2 - B.BUY_C - B.FEE_RT) * 100, 4))
                # inputs, all known at t (minute data: CLOSED minutes only, m + 1 <= t / 60)
                mc = t // 60 - 1
                row.update(m300=side * m300, m60=side * m60, m10=side * m10, c_open=side * bp(p0, open_px), c_tod=t - OPEN)
                kt, cl = kp[tk]
                seen = cl[(kt <= t) & (kt >= OPEN)]
                hi_, lo_ = (seen.max(), seen.min()) if len(seen) else (np.nan, np.nan)
                row["c_range"] = ((p0 - lo_) / (hi_ - lo_) if side > 0 else (hi_ - p0) / (hi_ - lo_)) if hi_ > lo_ else None
                row["q_spread"] = ((e[1] - e[0]) / ((e[0] + e[1]) / 2)) if e and (e[0] + e[1]) > 0 else None
                row["f_dd5"] = side * sum(dd.get(m, 0.0) for m in range(mc - 4, mc + 1)) if dd else None
                row["f_dd1"] = side * dd.get(mc, 0.0) if dd else None
                v5 = sum(mins.get(m, (0, 0, 0, 0))[1] for m in range(mc - 4, mc + 1))
                v15 = sum(mins.get(m, (0, 0, 0, 0))[1] for m in range(mc - 19, mc - 4))
                row["f_vol"] = (v5 / 5) / (v15 / 15) if v15 > 0 else None    # stock volume pace (option volume per minute not cached)
                gm = mc
                while gm >= 560 and gm not in gex:
                    gm -= 1
                row["c_gex"] = gex.get(gm)
                tm = [m for m in tide if m <= mc]
                row["c_tide"] = (side * (tide[max(tm)] - tide[min(m for m in tm if m >= max(tm) - 15)])) if tm else None
                xr = [(bp(P(kp[o], t), P(kp[o], t - 60)), bp(P(kp[o], t), P(kp[o], max(t - 300, OPEN + 5)))) for o in others]
                row["x_ret60"] = side * float(np.nanmean([a for a, _ in xr])) if xr else None
                row["x_ret300"] = side * float(np.nanmean([b for _, b in xr])) if xr else None
                row["cum_vol_min"] = mc
                out.append(row)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    return d, out, None


def attach_day_level(rows):
    """c_rvol (stock volume to the last closed minute vs the 20-session median) and the night inputs."""
    import check_overnight as O
    by_tk = {}
    for tk in TKS:
        nights = {r["d"]: r for r in O.night_rows(tk)}
        days = sorted(os.path.basename(os.path.dirname(f))[5:] for f in glob.glob(os.path.join("lake", "silver", "ext-1m", "date=*", f"{tk}.parquet")))
        cum = {}
        for d in days:
            m = ext_minutes(tk, d) or {}
            a = np.zeros(390)
            for k, v in m.items():
                if 570 <= k < 960:
                    a[k - 570] = v[1]
            cum[d] = np.cumsum(a)
        by_tk[tk] = (nights, days, cum)
    for r in rows:
        nights, days, cum = by_tk[r["tk"]]
        n = nights.get(r["d"])
        r["o_gapz"] = n["z"] if n else None
        if n:
            gap_side = 1 if n["P0"] > n["C0"] else -1
            r["o_with_gap"] = 1.0 if r["side"] == gap_side else 0.0
        else:
            r["o_with_gap"] = None
        j = r["cum_vol_min"] - 570
        r["c_rvol"] = None
        if r["d"] in cum and 0 <= j < 390:
            i = days.index(r["d"])
            prev = [cum[x][j] for x in days[max(0, i - 20):i]]
            base = float(np.median(prev)) if len(prev) >= 15 else 0
            r["c_rvol"] = cum[r["d"]][j] / base if base > 0 else None
    return rows


def half(d):
    return "discovery" if int(d[5:7]) % 2 == 1 else "holdout"


def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sessions()
    rows, errs = [], []
    with ProcessPoolExecutor(6) as ex:
        for d, r, e in ex.map(build_day, days):
            rows += r
            if e:
                errs.append((d, e))
    rows = attach_day_level(rows)
    json.dump(rows, open(os.path.join(CACHE, "signals.json"), "w"))
    print(f"built {len(rows)} signals over {len(days)} sessions; errors {len(errs)} {errs[:5]}")


# ---------------------------------------------------------------- scoring
def day_boot(rows, vals, n=2000, seed=SEED):
    days = sorted({r["d"] for r in rows})
    by = {d: [] for d in days}
    for i, r in enumerate(rows):
        by[r["d"]].append(i)
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = np.concatenate([by[d] for d in rng.choice(days, len(days))])
        bs.append(np.nanmean(v[idx]))
    bs = np.array(bs)
    return np.nanmean(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(min(1, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def ranks(disc, rows):
    """mid-rank ECDF of each input against the discovery distribution, within ticker; missing -> 0.5"""
    X = np.full((len(rows), len(INPUTS)), 0.5)
    for j, k in enumerate(INPUTS):
        ref = np.sort(np.array([r[k] for r in disc if r.get(k) is not None], float))
        for i, r in enumerate(rows):
            v = r.get(k)
            if v is not None and len(ref):
                X[i, j] = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2.0 * len(ref))
    return X


def bh(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    q = np.empty(len(ps)); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
    return q


def screen(rule, rows, lines, final):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    R = [r for r in rows if r["tk"] == "SPY" and r["rule"] == rule and r.get("pnl_A") is not None]
    disc = [r for r in R if half(r["d"]) == "discovery"]
    hold = [r for r in R if half(r["d"]) == "holdout"]
    Xd, yd = ranks(disc, disc), np.array([r["pnl_A"] for r in disc])
    lines.append(f"  {rule} screen: discovery {len(disc)} signals")
    ics = []
    first = [r for r in disc if r["d"] < "2025-07-01"]
    second = [r for r in disc if r["d"] >= "2025-07-01"]
    for j, k in enumerate(INPUTS):
        ic = spearman(Xd[:, j], yd)
        _, lo, hi, p = day_boot(disc, (Xd[:, j] - Xd[:, j].mean()) * (yd - yd.mean()), n=500)
        h1 = spearman(ranks(disc, first)[:, j], [r["pnl_A"] for r in first]) if first else np.nan
        h2 = spearman(ranks(disc, second)[:, j], [r["pnl_A"] for r in second]) if second else np.nan
        ics.append((k, ic, p, h1, h2))
    q = bh([x[2] for x in ics])
    for (k, ic, p, h1, h2), qq in sorted(zip(ics, q), key=lambda z: -abs(z[0][1])):
        lines.append(f"    IC {k:10s} {ic:+.3f}  q {qq:.3f}  halves {h1:+.3f} / {h2:+.3f}"
                     + ("  *" if qq < 0.10 and np.sign(h1) == np.sign(h2) == np.sign(ic) else ""))
    weeks = [dt.date.fromisoformat(r["d"]).isocalendar()[:2] for r in disc]
    groups = np.unique([f"{a}-{b}" for a, b in weeks], return_inverse=True)[1]
    m = LassoCV(cv=GroupKFold(5), alphas=np.logspace(-4, 0, 40), max_iter=20000).fit(Xd, yd, groups=groups)
    kept = [(k, c) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9]
    lines.append(f"    LASSO keeps {len(kept)}: " + ", ".join(f"{k} {c:+.2f}" for k, c in sorted(kept, key=lambda z: -abs(z[1]))))
    sd = m.predict(Xd)
    cuts = np.percentile(sd, [20, 40, 60, 80])
    qd = [np.mean([r["pnl_A"] for r, s in zip(disc, sd) if (s >= cuts[3] if b == 4 else (s < cuts[0] if b == 0 else cuts[b - 1] <= s < cuts[b]))]) for b in range(5)]
    lines.append("    discovery fifths P&L: " + " / ".join(f"${v:+.2f}" for v in qd))
    res = {"kept": len(kept)}
    if final:
        Xh, yh = ranks(disc, hold), np.array([r["pnl_A"] for r in hold])
        sh = m.predict(Xh)
        icv = spearman(sh, yh)
        _, lo, hi, _p = day_boot(hold, (sh - sh.mean()) * (yh - yh.mean()), n=2000)
        top = [r for r, s in zip(hold, sh) if s >= cuts[3]]
        tm, tlo, thi, tp = day_boot(top, [r["pnl_A"] for r in top])
        res.update(ic=icv, ic_ok=bool(lo > 0 and icv > 0), top=(tm, tlo, thi, tp))
        qh = [np.mean([r["pnl_A"] for r, s in zip(hold, sh) if (s >= cuts[3] if b == 4 else (s < cuts[0] if b == 0 else cuts[b - 1] <= s < cuts[b]))]) for b in range(5)]
        lines.append(f"    HOLDOUT {len(hold)} signals: score IC {icv:+.3f} (covariance CI [{lo:+.3f}, {hi:+.3f}]) -> S2 {'PASS' if res['ic_ok'] and kept else 'fail'}")
        lines.append("    holdout fifths P&L: " + " / ".join(f"${v:+.2f}" for v in qh))
        lines.append(f"    S3 top fifth ${tm:+.2f} [{tlo:+.2f}, {thi:+.2f}] n {len(top)}")
        res["disc_top"] = qd[4]
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return
    if not (a.discovery or a.holdout):
        print(__doc__)
        return
    mark = os.path.join(CACHE, "RESULT-holdout.txt")
    if a.holdout and os.path.exists(mark):
        sys.exit("holdout already scored -- see " + mark)
    rows = json.load(open(os.path.join(CACHE, "signals.json")))
    want = "holdout" if a.holdout else "discovery"
    lines = [f"check_scalp_rules {want.upper()} -- {len(rows)} signals built"]
    fam = []
    for tk in TKS:
        for rule in ("CHASE", "FADE"):
            R = [r for r in rows if r["tk"] == tk and r["rule"] == rule]
            H = [r for r in R if half(r["d"]) == want]
            for ex in ("A", "B"):
                V = [r for r in H if r.get(f"pnl_{ex}") is not None]
                if not V:
                    continue
                mn, lo, hi, p = day_boot(V, [r[f"pnl_{ex}"] for r in V])
                sess = len({r["d"] for r in H})
                lines.append(f"  {tk} {rule:5s} exit {ex} ({30 if ex == 'A' else 60} s): {len(V)} trades ({len(V) / max(sess, 1):.0f}/session, "
                             f"{len(H) - len(V)} unpriced) mean ${mn:+.2f} [{lo:+.2f}, {hi:+.2f}] win {np.mean([r[f'pnl_{ex}'] > 0 for r in V]):.0%}")
                if tk == "SPY" and ex == "A":
                    fam.append((f"R{'1' if rule == 'CHASE' else '2'} {rule}", mn, lo, hi, p))
    for rule in ("CHASE", "FADE"):
        res = screen(rule, rows, lines, final=a.holdout)
        if a.holdout and "top" in res:
            tm, tlo, thi, tp = res["top"]
            fam.append((f"S3 {rule}", tm, tlo, thi, tp))
    if a.holdout:
        q = bh([x[4] for x in fam])
        for (name, mn, lo, hi, p), qq in zip(fam, q):
            lines.append(f"  FAMILY {name}: ${mn:+.2f} [{lo:+.2f}, {hi:+.2f}] q {qq:.3f} -> {'PASS' if lo > 0 and qq < 0.10 else 'fail'}")
        open(mark, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
