"""
check_futures_screen.py -- FUTURES ORDER FLOW AT THRU BREACHES AND PULLBACK ENTRIES: an IC screen (user, 2026-11-19).
PRE-REGISTERED 2026-11-19, committed before any input below was computed against any target.

QUESTION. UW Advanced gives the CME futures tape with the aggressor side on every print (uw_futures_tape.py; complete
sessions from 2026-07-27). Does futures order flow at the moment of a strike breach (check_breach_ic) or a pullback
entry (check_structure_ic) say anything about how it turns out -- on its own, and beyond the frozen scores already
being forward-tested? Prior: options/stock/futures order flow lined up mostly AFTER moves (QQQ sub-minute, 11-07), and
the user's eye on 11-19 saw ES CVD track SPY's legs down but end the day positive on a -1.5% day. A screen only: a
pass earns a frozen forward read, never a deployment.

EVENTS (unchanged, from the existing caches; nothing re-detected)
  THRU   _breach_cache/events: confirmed $1-strike breaches, SPY / QQQ / IWM, t_in 09:45-15:30, side +1 up / -1 down.
         TARGET Y (as pre-registered there): +1 next strike before falling back through K by $0.10, -1 the reverse,
         0 neither in 15 min. CONTROL = the frozen breach score (breach_model_frozen.json, check_breach_forward.score).
  PULL   _structure_ic_cache/events: lower-low -> lower-high entries (and mirror), both structures, 09:35-15:30,
         side -1 short / +1 long. TARGET Y = 1 if 2R before the stop within 30 min, else 0. CONTROL = the frozen
         pullback score (pullback_model_frozen.json, freeze_pullback.score).
  SAMPLE every session in BOTH the event caches and the futures lake: 2026-07-27 .. 2026-11-13 (71 sessions; the lake
         has no option data 08-24..08-31, 09-16, 09-17). HALVES = the first 36 / the last 35 sessions.
  These sessions were part of the breach / structure HOLDOUTS (other inputs, already scored); no futures input has
  been computed on them. Their option-side results are known; the futures side is clean.

FUTURES (SPY -> ES, QQQ -> NQ, IWM -> RTY; the session's front contract = most volume; block prints excluded, as in
lake/silver/futures-1s). KNOWABLE = futures seconds s with s + 1 <= t_in - LAT_F, LAT_F = 3 s (the live feed's
~0.3 s arrival + the 2.5 s poll). "signed" = multiplied by side, so + always means "with the event".
  u_imb60    signed (buy - sell) / (buy + sell), aggressor volume, last 60 s
  u_imb300   the same, last 5 min
  u_imb900   the same, last 15 min
  u_day      the same, session so far (from 09:30)
  u_surge    futures volume last 60 s / mean volume per 60 s over the 30 min before that (unsigned)
  u_ret60    signed futures price change over the last 60 s, bp (last traded price)
  u_big60    signed (big buy - big sell) / (big buy + big sell + 1), last 60 s; big = a print >= the 99th percentile of
             the PREVIOUS session's print sizes for that product (from lake/bronze/futures-flow), 0 if none
  7 inputs x 2 event families = 14 tests.

STATISTICS (pooled across the three tickers; an input missing for an event -> that event left out of that test)
  S1 IC       Spearman(input, Y). p from a day-block bootstrap (2,000 resamples of sessions, two-sided).
  S2 BEYOND   partial Spearman(input, Y | frozen score), same bootstrap. Reported with a 95% CI.
  S3 HALVES   the S1 sign in each half.
  Also reported, not scored: Spearman(input, frozen score) -- is it new information -- and Y rate by input quintile.
MACHINERY (before the verdict; a failure stops the run): M1 a planted input (Y + noise, IC ~0.10) must reach IC >=
  0.05 with bootstrap p < 0.01 on both families; M2 the same input shuffled within session must give |IC| < 0.03;
  M3 a knowability probe -- the latest futures second used for any event is <= t_in - LAT_F - 1.
PASS (per test): S1 BH q < 0.10 across the 14 AND S2 95% CI excludes 0 with the same sign AND the same sign in both
  halves. Expected power: ~3,300 / ~1,600 events but only 71 day-blocks -- an IC near 0.03 (what the option-side
  inputs showed) will usually NOT be detectable here; a miss is "not big", not "nothing".
WHAT A PASS BUYS: the passing input(s) are frozen (definition, sign) for a forward read on new sessions (lake refreshes
  after 2026-11-13) at 60 sessions (~2027-02-18) and 125 sessions (~2027-05-21). Nothing on the viewer or the bot.

Usage:  python check_futures_screen.py --build   (inputs per event -> _futscreen_cache/inputs.json; no targets read)
        python check_futures_screen.py --score   (once: writes _futscreen_cache/RESULT.txt and refuses to run again)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CACHE = "_futscreen_cache"
SILVER = os.path.join("lake", "silver", "futures-1s")
BRONZE = os.path.join("lake", "bronze", "futures-flow")
FUT = {"SPY": "ES", "QQQ": "NQ", "IWM": "RTY"}
LO, HI = "2026-07-27", "2026-11-13"
LAT_F = 3
INPUTS = ["u_imb60", "u_imb300", "u_imb900", "u_day", "u_surge", "u_ret60", "u_big60"]
FAM = {"THRU": "_breach_cache/events", "PULL": "_structure_ic_cache/events"}
N_BOOT, SEED = 2000, 20261119
OPEN = 9 * 3600 + 30 * 60


def sessions():
    have = lambda p: {os.path.basename(f)[:10] for f in glob.glob(os.path.join(p, "*.json"))}
    fut = {x.split("=")[1] for x in os.listdir(SILVER)}
    return sorted(d for d in have(FAM["THRU"]) & have(FAM["PULL"]) & fut if LO <= d <= HI)


# ------------------------------------------------------------------ futures per session
def big_threshold(prod, d, days_all):
    prev = [x for x in days_all if x < d]
    for p in reversed(prev[-5:]):
        f = os.path.join(BRONZE, f"date={p}", f"{prod}.parquet")
        if os.path.exists(f):
            b = pl.read_parquet(f, columns=["size", "is_block"]).filter(pl.col("is_block") != True)
            return float(b["size"].quantile(0.99))
    return None


def fut_day(prod, d, days_all):
    s = pl.read_parquet(os.path.join(SILVER, f"date={d}", f"{prod}.parquet")).sort("sec")
    front = s["sym"][0]
    sec = s["sec"].to_numpy().astype(np.int64)
    buy, sell = s["buy"].to_numpy().astype(float), s["sell"].to_numpy().astype(float)
    last = s["last"].to_numpy().astype(float)
    thr = big_threshold(prod, d, days_all)
    bb = pl.read_parquet(os.path.join(BRONZE, f"date={d}", f"{prod}.parquet"),
                         columns=["executed_at", "sym", "size", "side", "is_block"])
    bb = bb.filter((pl.col("sym") == front) & (pl.col("is_block") != True))
    et = bb["executed_at"].dt.convert_time_zone("America/New_York")
    bsec = (et.dt.hour().cast(pl.Int64) * 3600 + et.dt.minute().cast(pl.Int64) * 60 + et.dt.second().cast(pl.Int64)).to_numpy()
    if thr is None:
        big_b = big_s = np.zeros(0); bigsec = np.zeros(0, np.int64)
    else:
        m = (bb["size"].to_numpy() >= thr)
        side = bb["side"].to_numpy()
        bigsec = bsec[m]
        big_b = np.where(side[m] == "buy", bb["size"].to_numpy()[m], 0).astype(float)
        big_s = np.where(side[m] == "sell", bb["size"].to_numpy()[m], 0).astype(float)
    return dict(sec=sec, cb=np.cumsum(buy), cs=np.cumsum(sell), last=last, thr=thr,
                bigsec=bigsec, cbb=np.cumsum(big_b), cbs=np.cumsum(big_s))


def window(F, a, b, kind="flow"):
    """sums over futures seconds a <= s < b (cumsum trick); returns (buy, sell)."""
    sec = F["sec"] if kind == "flow" else F["bigsec"]
    cb, cs = (F["cb"], F["cs"]) if kind == "flow" else (F["cbb"], F["cbs"])
    i0, i1 = np.searchsorted(sec, a, "left"), np.searchsorted(sec, b, "left")
    if i1 == 0:
        return 0.0, 0.0
    B = cb[i1 - 1] - (cb[i0 - 1] if i0 > 0 else 0.0)
    S = cs[i1 - 1] - (cs[i0 - 1] if i0 > 0 else 0.0)
    return float(B), float(S)


def price_at(F, t):
    i = np.searchsorted(F["sec"], t, "left") - 1
    return float(F["last"][i]) if i >= 0 else None


def inputs_for(F, t_in, side):
    end = int(np.floor(t_in)) - LAT_F                  # seconds s < end, i.e. s + 1 <= t_in - LAT_F
    out, probe = {}, None
    i = np.searchsorted(F["sec"], end, "left") - 1
    if i < 0:
        return {k: None for k in INPUTS}, None
    probe = int(F["sec"][i])

    def imb(a):
        b_, s_ = window(F, a, end)
        return side * (b_ - s_) / (b_ + s_) if b_ + s_ > 0 else None

    out["u_imb60"], out["u_imb300"], out["u_imb900"] = imb(end - 60), imb(end - 300), imb(end - 900)
    out["u_day"] = imb(OPEN)
    v60 = sum(window(F, end - 60, end))
    v30 = sum(window(F, end - 60 - 1800, end - 60)) / 30.0
    out["u_surge"] = v60 / v30 if v30 > 0 else None
    p1, p0 = price_at(F, end), price_at(F, end - 60)
    out["u_ret60"] = side * (p1 / p0 - 1) * 1e4 if p1 and p0 else None
    if F["thr"] is None:
        out["u_big60"] = None
    else:
        b_, s_ = window(F, end - 60, end, "big")
        out["u_big60"] = side * (b_ - s_) / (b_ + s_ + 1)
    return out, probe


def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sessions()
    days_all = sorted(x.split("=")[1] for x in os.listdir(SILVER))
    rows, probe_gap = [], []
    for d in days:
        F = {p: fut_day(p, d, days_all) for p in set(FUT.values())}
        for fam, path in FAM.items():
            for j, r in enumerate(json.load(open(os.path.join(path, f"{d}.json")))):
                x, probe = inputs_for(F[FUT[r["t"]]], float(r["t_in"]), int(r["side"]))
                if probe is not None:
                    probe_gap.append(float(r["t_in"]) - probe)
                # targets are NOT read here: only the event's identity (file position) and timing
                rows.append(dict(fam=fam, d=d, j=j, t=r["t"], t_in=r["t_in"], side=r["side"], **x))
        print(d, len(rows), flush=True)
    json.dump(dict(days=days, rows=rows, min_probe_gap=min(probe_gap)), open(os.path.join(CACHE, "inputs.json"), "w"))
    print(f"built {len(rows)} events over {len(days)} sessions; smallest t_in - last futures second used = {min(probe_gap):.1f} s")


# ------------------------------------------------------------------ scoring
def rankv(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[o] = np.arange(len(x))
    # ties -> mean rank
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.bincount(inv, weights=r)
    return (s / cnt)[inv]


def spear(x, y):
    if len(x) < 10:
        return np.nan
    a, b = rankv(x), rankv(y)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pspear(x, y, z):
    a, b, c = rankv(x), rankv(y), rankv(z)
    rxy, rxz, ryz = np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1], np.corrcoef(b, c)[0, 1]
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return float((rxy - rxz * ryz) / den)


def boot(day_idx, stat, rng):
    days = np.unique(day_idx)
    by = {d: np.where(day_idx == d)[0] for d in days}
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days), replace=True)
        idx = np.concatenate([by[d] for d in pick])
        out.append(stat(idx))
    return np.array(out)


def bh(ps):
    ps = np.asarray(ps); n = len(ps); o = np.argsort(ps)
    q = np.empty(n); prev = 1.0
    for k, i in enumerate(o[::-1]):
        rank = n - k
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def test(xs, ys, zs, ds, rng, half_of):
    m = np.array([x is not None for x in xs])
    x, y, z, d = np.array([v for v in xs if v is not None], float), np.asarray(ys, float)[m], np.asarray(zs, float)[m], np.asarray(ds)[m]
    ic = spear(x, y)
    b = boot(d, lambda i: spear(x[i], y[i]), rng)
    p = float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0))))
    pc = pspear(x, y, z)
    bp = boot(d, lambda i: pspear(x[i], y[i], z[i]), rng)
    h = [spear(x[np.array([half_of[v] == k for v in d])], y[np.array([half_of[v] == k for v in d])]) for k in (0, 1)]
    return dict(n=int(m.sum()), ic=ic, p=p, ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                partial=pc, pci=[float(np.quantile(bp, .025)), float(np.quantile(bp, .975))], halves=h,
                with_score=spear(x, z))


def score():
    out = os.path.join(CACHE, "RESULT.txt")
    if os.path.exists(out):
        sys.exit(f"{out} exists -- scored once")
    import check_breach_forward as BF
    import freeze_pullback as FP
    frz = {"THRU": json.load(open("breach_model_frozen.json")), "PULL": json.load(open("pullback_model_frozen.json"))}
    sc = {"THRU": BF.score, "PULL": FP.score}
    I = json.load(open(os.path.join(CACHE, "inputs.json")))
    days = I["days"]
    half_of = {d: (0 if i < 36 else 1) for i, d in enumerate(days)}
    ev = {}
    for fam, path in FAM.items():
        for d in days:
            for j, r in enumerate(json.load(open(os.path.join(path, f"{d}.json")))):
                ev[(fam, d, j)] = r
    rng = np.random.default_rng(SEED)
    lines = [f"check_futures_screen RESULT {dt.date.today()} -- {len(days)} sessions {days[0]}..{days[-1]}",
             f"M3 knowability: smallest t_in - last futures second used = {I['min_probe_gap']:.1f} s (needs >= {LAT_F + 1})"]
    if I["min_probe_gap"] < LAT_F + 1:
        lines.append("MACHINERY FAIL (M3)"); open(out, "w").write("\n".join(lines)); print("\n".join(lines)); return
    data = {}
    for fam in FAM:
        R = [r for r in I["rows"] if r["fam"] == fam]
        E = [ev[(fam, r["d"], r["j"])] for r in R]
        assert all(e["t"] == r["t"] and e["t_in"] == r["t_in"] for e, r in zip(E, R)), "event cache changed since --build"
        Y = [float(e["Y"]) for e in E]
        Z = [sc[fam](frz[fam], e) for e in E]
        D = [r["d"] for r in R]
        data[fam] = (R, Y, Z, D)
        lines.append(f"{fam}: {len(R)} events; Y mean {np.mean(Y):+.3f}")
    # machinery M1 / M2
    mrng = np.random.default_rng(SEED + 1)
    mach_ok = True
    for fam, (R, Y, Z, D) in data.items():
        y = np.array(Y)
        planted = y + mrng.normal(0, np.std(y) * 10.0, len(y))          # corr ~ 1 / sqrt(1 + 100) ~ 0.10
        t1 = test(list(planted), Y, Z, D, rng, half_of)
        sh = planted.copy()
        Dn = np.array(D)
        for d in set(D):
            i = np.where(Dn == d)[0]; sh[i] = sh[mrng.permutation(i)]
        t2 = test(list(sh), Y, Z, D, rng, half_of)
        ok = t1["ic"] >= 0.05 and t1["p"] < 0.01 and abs(t2["ic"]) < 0.03
        mach_ok &= ok
        lines.append(f"  {fam} M1 planted IC {t1['ic']:+.3f} p {t1['p']:.4f} | M2 shuffled-in-day IC {t2['ic']:+.3f} -> {'ok' if ok else 'FAIL'}")
    if not mach_ok:
        lines.append("MACHINERY FAIL (M1/M2): not scored")
        open(out, "w").write("\n".join(lines)); print("\n".join(lines)); return
    res = []
    for fam, (R, Y, Z, D) in data.items():
        for k in INPUTS:
            res.append((fam, k, test([r[k] for r in R], Y, Z, D, rng, half_of)))
    q = bh([t["p"] for _, _, t in res])
    lines.append("")
    lines.append(f"{'fam':5s} {'input':9s} {'n':>5s} {'IC':>7s} {'95% CI':>17s} {'p':>6s} {'q':>6s} {'partial':>8s} {'95% CI':>17s} "
                 f"{'half1':>6s} {'half2':>6s} {'vs score':>8s}  verdict")
    passes = []
    for (fam, k, t), qq in zip(res, q):
        ok = (qq < 0.10 and (t["pci"][0] > 0 or t["pci"][1] < 0) and np.sign(t["partial"]) == np.sign(t["ic"])
              and np.sign(t["halves"][0]) == np.sign(t["ic"]) == np.sign(t["halves"][1]))
        if ok:
            passes.append(f"{fam}:{k}")
        lines.append(f"{fam:5s} {k:9s} {t['n']:5d} {t['ic']:+7.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] {t['p']:6.3f} {qq:6.3f} "
                     f"{t['partial']:+8.3f} [{t['pci'][0]:+.3f}, {t['pci'][1]:+.3f}] {t['halves'][0]:+6.3f} {t['halves'][1]:+6.3f} "
                     f"{t['with_score']:+8.3f}  {'PASS' if ok else '-'}")
    # descriptive: Y by input quintile
    lines.append("")
    lines.append("Y by input quintile (descriptive):")
    for fam, (R, Y, Z, D) in data.items():
        for k in INPUTS:
            xs = np.array([r[k] if r[k] is not None else np.nan for r in R], float); y = np.array(Y)
            m = ~np.isnan(xs)
            if m.sum() < 50:
                continue
            cuts = np.quantile(xs[m], [.2, .4, .6, .8]); qi = np.searchsorted(cuts, xs[m], side="right")
            lines.append(f"  {fam} {k:9s} " + " ".join(f"{np.mean(y[m][qi == j]):+.3f}" for j in range(5)))
    lines.append("")
    lines.append(f"PASSES: {', '.join(passes) if passes else 'none'}")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.score:
        score()
