"""
check_breach_next.py -- THE NEXT STRIKE OUT: does activity at K+2 (K-2 on a down-break) say whether a breach carries
through? (user, 2026-11-20). PART A of two; PART B (K+2 as the trade vehicle vs K / K+1) is a separate pricing study.
PRE-REGISTERED 2026-11-20, committed before any input below was computed.

WHY. The six-pack (check_sixpack_lake) and the breach screen (check_breach_ic) read K-1 / K / K+1 (and OI at K+1). The
contracts one strike further out in the breach direction -- the pair that BECOMES the first OTM strike if price
carries through -- were never read. If traders position there before the carry, it should show at the breach.

EVENTS  _breach_cache/events (check_breach_ic): confirmed $1-strike breaches, SPY / QQQ / IWM, t_in 09:45-15:30, side
        +1 up / -1 down. TARGET Y (as there): +1 next strike before falling back through K by $0.10, -1 the reverse,
        0 neither in 15 min. CONTROL = the frozen breach score (breach_model_frozen.json).
DATA    lake/silver/option-contracts-1m, same-day (0DTE) contracts only. KNOWABLE: minute m counts only when closed,
        (m + 1) x 60 <= t_in; m_last = the latest such minute.
INPUTS  K2 = K + 2 x side; DIR = the call on an up-break / the put on a down-break; OTH = the other type at K2.
        "signed" = + means with the breach.
  n_surge    K2 call+put volume, last 5 min per minute / (the 15 min before per minute + 1); 0 if < 20 contracts
  n_share    K2 call+put volume, last 5 min / volume of every contract from K - side to K2 (8 contracts), last 5 min
  n_dir      ask - bid volume on K2 DIR, last 5 min (buying the next OTM strike = +)
  n_other    -(ask - bid volume) on K2 OTH, last 5 min (selling the other type = +)
  n_oi       log((start-of-day OI at K2, calls+puts + 1) / (median OI of K-5..K+5 + 1))
  n_iv       15-min IV change of K2 DIR minus that of K+side DIR (the next strike being bid up relative to the first OTM)
  n_spread   (ask - bid) / mid of K2 DIR, last closed minute
  n_delta5   side x sum over K2 call+put of (ask - bid volume) x delta x 100, last 5 min    [user: 0DTE directional delta]
  n_delta15  the same over 15 min                                                          [user]
SAMPLES DISCOVERY 2024-08-20 .. 2025-12-31 (halves split 2025-04-30). HOLDOUT 2026-01-02 .. 2026-11-13, scored ONCE
        (--holdout writes _breach_next_cache/RESULT-holdout.txt and refuses to run again). PARTLY CLEAN, said now: the
        THRU model was scored on that holdout; no K2 input has been computed anywhere.
STATISTICS / PASS
  MACHINERY (discovery; a failure stops the run): M1 a planted input (Y + noise, IC ~0.10) reaches IC >= 0.05 with
     p < 0.01; M2 that input shuffled across all events gives |IC| < 0.03; M3 every input uses minutes with
     (m + 1) x 60 <= t_in.
  DISCOVERY (screen): S1 Spearman(input, Y), day-block bootstrap p (1,000); S2 partial Spearman given the frozen score,
     95% CI; S3 the sign in both halves. A SURVIVOR = BH q < 0.10 across the 9 AND S2's CI excludes 0 with the same
     sign AND the same sign in both halves. Its sign is then fixed.
  HOLDOUT (confirm, once): each survivor's Spearman(input, Y) with the fixed sign, day-block (2,000) 95% CI excluding 0
     on that side, AND the partial beyond the frozen score the same sign; BH across survivors. PASS = >= 1 confirms.
  Reported, not scored: Y by input quintile; Spearman with the screen's own strike inputs (s_surge, s_kcall,
     s_oi_next, f_delta5); M5 / M10 moves.
WHAT A PASS BUYS: the confirmed input(s) shown in the THRU pill's tooltip as context with their holdout rates, and a
  forward read from the live strike cue. The frozen breach score is not refit.

Usage:  python check_breach_next.py --build | --discovery | --holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CACHE = "_breach_next_cache"
LAKE = os.path.join("lake", "silver", "option-contracts-1m")
DISC, HOLD, HALF = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-13"), "2025-04-30"
TK = ("SPY", "QQQ", "IWM")
INPUTS = ["n_surge", "n_share", "n_dir", "n_other", "n_oi", "n_iv", "n_spread", "n_delta5", "n_delta15"]
OWN = ["s_surge", "s_kcall", "s_oi_next", "f_delta5"]
SEED = 20261121


def book_of(d):
    """{(tk, K, cp): {m: (bid, ask, ask_vol, bid_vol, vol, delta, iv, oi)}} for same-day contracts."""
    f = os.path.join(LAKE, f"date={d}", "bars.parquet")
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(TK)) & (pl.col("expiry").cast(pl.Utf8) == d))
         .select("underlying_symbol", "option_type", pl.col("strike").cast(pl.Float64), "minute_et", "bid_close", "ask_close",
                 "ask_volume", "bid_volume", "volume", "delta_close", "iv_close", "open_interest").collect())
    x = x.with_columns(m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32),
                       cp=pl.col("option_type").str.to_uppercase().str.slice(0, 1))
    book = defaultdict(dict)
    for t, k, cp, m, b, a, av, bv, v, de, iv, oi in x.select("underlying_symbol", "strike", "cp", "m", "bid_close", "ask_close",
                                                           "ask_volume", "bid_volume", "volume", "delta_close", "iv_close",
                                                           "open_interest").iter_rows():
        if k is None or abs(k - round(k)) > 1e-6:
            continue
        book[(t, int(round(k)), cp)][m] = (b, a, av or 0, bv or 0, v or 0, de, iv, oi)
    return book


def inputs_for(book, tk, K, side, t_in):
    m_last = int(t_in // 60) - 1                         # (m + 1) x 60 <= t_in
    K2, K1 = K + 2 * side, K + side
    DIR, OTH = ("C", "P") if side > 0 else ("P", "C")
    Z = (None, None, 0, 0, 0, None, None, None)

    def row(k, cp, m):
        return book.get((tk, k, cp), {}).get(m) or Z

    def tot(k, cp, fld, m0, m1):
        return sum(row(k, cp, m)[fld] for m in range(m0, m1 + 1))

    out = {}
    v5 = sum(tot(K2, cp, 4, m_last - 4, m_last) for cp in "CP")
    vb = sum(tot(K2, cp, 4, m_last - 19, m_last - 5) for cp in "CP")
    out["n_surge"] = (v5 / 5) / (vb / 15 + 1) if v5 >= 20 else 0.0
    span = [K - side, K, K1, K2]
    den = sum(tot(k, cp, 4, m_last - 4, m_last) for k in span for cp in "CP")
    out["n_share"] = v5 / den if den > 0 else None
    out["n_dir"] = float(tot(K2, DIR, 2, m_last - 4, m_last) - tot(K2, DIR, 3, m_last - 4, m_last))
    out["n_other"] = float(-(tot(K2, OTH, 2, m_last - 4, m_last) - tot(K2, OTH, 3, m_last - 4, m_last)))

    def oi(k):
        s = 0
        for cp in "CP":
            c = book.get((tk, k, cp), {})
            v = next((c[m][7] for m in sorted(c) if c[m][7]), None)
            s += v or 0
        return s
    ring = [oi(K + j) for j in range(-5, 6)]
    pos = [v for v in ring if v > 0]
    out["n_oi"] = float(np.log((oi(K2) + 1) / (float(np.median(pos)) + 1))) if pos else None

    def iv_at(k, cp, m):
        for mm in range(m, m - 4, -1):                   # the latest IV within 3 minutes, never after m
            r = book.get((tk, k, cp), {}).get(mm)
            if r and r[6]:
                return r[6]
        return None
    a2, b2 = iv_at(K2, DIR, m_last), iv_at(K2, DIR, m_last - 15)
    a1, b1 = iv_at(K1, DIR, m_last), iv_at(K1, DIR, m_last - 15)
    out["n_iv"] = (a2 - b2) - (a1 - b1) if None not in (a2, b2, a1, b1) else None
    q = row(K2, DIR, m_last)
    if not q[0] and not q[1]:
        q = row(K2, DIR, m_last - 1)
    out["n_spread"] = (q[1] - q[0]) / ((q[0] + q[1]) / 2) if q[0] and q[1] and q[1] > q[0] else None

    def dd(m0):
        s = 0.0
        for cp in "CP":
            for m in range(m0, m_last + 1):
                r = row(K2, cp, m)
                if r[5] is not None:
                    s += (r[2] - r[3]) * r[5] * 100
        return side * s
    out["n_delta5"], out["n_delta15"] = dd(m_last - 4), dd(m_last - 14)
    out["_m_last"] = m_last
    return out


def build_day(d):
    evs = json.load(open(os.path.join("_breach_cache", "events", f"{d}.json")))
    try:
        book = book_of(d)
    except Exception as e:                               # noqa: BLE001
        return d, [], f"{type(e).__name__}"
    rows = []
    for j, e in enumerate(evs):
        x = inputs_for(book, e["t"], int(e["K"]), int(e["side"]), float(e["t_in"]))
        rows.append(dict(d=d, j=j, t=e["t"], t_in=e["t_in"], **x))
    return d, rows, None


def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")))
    days = [d for d in days if DISC[0] <= d <= HOLD[1]]
    rows, errs = [], []
    with ProcessPoolExecutor(6) as ex:
        for d, r, err in ex.map(build_day, days):
            rows += r
            if err:
                errs.append((d, err))
    bad = [r for r in rows if (r["_m_last"] + 1) * 60 > r["t_in"]]
    json.dump(dict(rows=rows, errors=errs, m3_violations=len(bad)), open(os.path.join(CACHE, "inputs.json"), "w"))
    print(f"built {len(rows)} breaches over {len(days)} sessions; errors {len(errs)}; M3 violations {len(bad)}")


# ------------------------------------------------------------------ scoring
def load(lo, hi):
    import check_breach_forward as BF
    frz = json.load(open("breach_model_frozen.json"))
    I = json.load(open(os.path.join(CACHE, "inputs.json")))
    ev = {}
    for f in glob.glob(os.path.join("_breach_cache", "events", "*.json")):
        d = os.path.basename(f)[:10]
        if lo <= d <= hi:
            for j, e in enumerate(json.load(open(f))):
                ev[(d, j)] = e
    out = []
    for r in I["rows"]:
        if lo <= r["d"] <= hi:
            e = ev[(r["d"], r["j"])]
            assert e["t"] == r["t"] and e["t_in"] == r["t_in"], "event cache changed since --build"
            out.append(dict(r, Y=float(e["Y"]), z=BF.score(frz, e), M5=e.get("M5"), M10=e.get("M10"), **{f"own_{k}": e.get(k) for k in OWN}))
    return out, I


def stat(rows, k, rng, n):
    import check_futures_screen as FS
    m = np.array([r[k] is not None for r in rows])
    x = np.array([r[k] for r in rows if r[k] is not None], float)
    y = np.array([r["Y"] for r in rows], float)[m]; z = np.array([r["z"] for r in rows], float)[m]
    d = np.array([r["d"] for r in rows])[m]
    ic, pc = FS.spear(x, y), FS.pspear(x, y, z)
    days = np.unique(d); by = {dd: np.where(d == dd)[0] for dd in days}
    bi, bp = [], []
    for _ in range(n):
        idx = np.concatenate([by[dd] for dd in rng.choice(days, len(days), replace=True)])
        bi.append(FS.spear(x[idx], y[idx])); bp.append(FS.pspear(x[idx], y[idx], z[idx]))
    bi, bp = np.array(bi), np.array(bp)
    return dict(n=int(m.sum()), ic=ic, p=float(min(1.0, 2 * min(np.mean(bi <= 0), np.mean(bi >= 0)))),
                ci=np.quantile(bi, [.025, .975]).tolist(), partial=pc, pci=np.quantile(bp, [.025, .975]).tolist(),
                x=x, y=y, d=d)


def discovery():
    import check_futures_screen as FS
    import check_breach_ic as B
    rng = np.random.default_rng(SEED)
    rows, I = load(*DISC)
    print(f"DISCOVERY {DISC[0]}..{DISC[1]}: {len(rows)} breaches, {len({r['d'] for r in rows})} sessions; "
          f"M3 violations (all samples) {I['m3_violations']}; build errors {len(I['errors'])}")
    if I["m3_violations"]:
        print("MACHINERY FAIL (M3)"); return
    y = np.array([r["Y"] for r in rows])
    mr = np.random.default_rng(SEED + 1)
    planted = y + mr.normal(0, np.std(y) * 10, len(y))
    for r, v in zip(rows, planted):
        r["_pl"] = float(v)
    t1 = stat(rows, "_pl", rng, 1000)
    sh = planted[mr.permutation(len(planted))]
    for r, v in zip(rows, sh):
        r["_sh"] = float(v)
    t2 = stat(rows, "_sh", rng, 200)
    ok = t1["ic"] >= 0.05 and t1["p"] < 0.01 and abs(t2["ic"]) < 0.03
    print(f"  M1 planted IC {t1['ic']:+.3f} p {t1['p']:.4f} | M2 shuffled IC {t2['ic']:+.3f} -> {'ok' if ok else 'MACHINERY FAIL'}")
    if not ok:
        return
    res = [(k, stat(rows, k, rng, 1000)) for k in INPUTS]
    q = B.bh([t["p"] for _, t in res])
    h1 = {r["d"]: r["d"] <= HALF for r in rows}
    surv = {}
    print(f"  {'input':10s} {'n':>6s} {'IC':>7s} {'95% CI':>17s} {'q':>6s} {'partial':>8s} {'95% CI':>17s} {'half1':>6s} {'half2':>6s}")
    for (k, t), qq in zip(res, q):
        a = np.array([h1[dd] for dd in t["d"]])
        ha, hb = FS.spear(t["x"][a], t["y"][a]), FS.spear(t["x"][~a], t["y"][~a])
        ok = qq < 0.10 and (t["pci"][0] > 0 or t["pci"][1] < 0) and np.sign(t["partial"]) == np.sign(t["ic"]) == np.sign(ha) == np.sign(hb)
        if ok:
            surv[k] = int(np.sign(t["ic"]))
        print(f"  {k:10s} {t['n']:6d} {t['ic']:+7.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] {qq:6.3f} {t['partial']:+8.3f} "
              f"[{t['pci'][0]:+.3f}, {t['pci'][1]:+.3f}] {ha:+6.3f} {hb:+6.3f}  {'SURVIVOR' if ok else ''}")
    print("  Y by quintile (low -> high) and Spearman with the screen's strike inputs:")
    for k in INPUTS:
        x = np.array([r[k] if r[k] is not None else np.nan for r in rows], float); ok_ = ~np.isnan(x)
        cuts = np.quantile(x[ok_], [.2, .4, .6, .8]); qi = np.searchsorted(cuts, x[ok_], side="right")
        cor = []
        for o in OWN:
            v = np.array([np.nan if r[f"own_{o}"] is None else float(r[f"own_{o}"]) for r in rows], float); b = ok_ & ~np.isnan(v)
            cor.append(f"{o} {FS.spear(x[b], v[b]):+.2f}")
        print(f"    {k:10s} " + " ".join(f"{np.mean(y[ok_][qi == j] == 1):.1%}" for j in range(5)) + "  | " + ", ".join(cor))
    json.dump(surv, open(os.path.join(CACHE, "survivors.json"), "w"))
    print(f"  SURVIVORS (sign fixed for the holdout): {surv or 'none'}")


def holdout():
    import check_breach_ic as B
    out = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(out):
        sys.exit(f"{out} exists -- scored once")
    surv = json.load(open(os.path.join(CACHE, "survivors.json")))
    rng = np.random.default_rng(SEED + 2)
    rows, _ = load(*HOLD)
    lines = [f"check_breach_next HOLDOUT {dt.date.today()} -- {len(rows)} breaches, {len({r['d'] for r in rows})} sessions",
             f"survivors from discovery: {surv or 'none'}"]
    res = []
    for k, sg in surv.items():
        t = stat(rows, k, rng, 2000)
        ok = (t["ci"][0] > 0 if sg > 0 else t["ci"][1] < 0) and np.sign(t["partial"]) == sg
        res.append((k, t, ok))
        lines.append(f"  {k:10s} IC {t['ic']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] partial {t['partial']:+.3f} "
                     f"[{t['pci'][0]:+.3f}, {t['pci'][1]:+.3f}] -> {'CONFIRMS' if ok else 'does not confirm'}")
    if res:
        q = B.bh([t["p"] for _, t, _ in res])
        lines.append("  BH q across survivors: " + ", ".join(f"{k} {qq:.3f}" for (k, _, _), qq in zip(res, q)))
    passed = [k for (k, _, ok), qq in zip(res, B.bh([t["p"] for _, t, _ in res]) if res else []) if ok and qq < 0.10]
    lines.append(f"PASS: {', '.join(passed) if passed else 'none'}")
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
