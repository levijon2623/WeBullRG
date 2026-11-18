"""
check_structure_ic.py -- IC SCREEN + LASSO ON PULLBACK ENTRIES (lower low -> lower high), user 2026-11-17.
PRE-REGISTERED 2026-11-17, committed before any input below was computed against any target.

CONTEXT. check_structure (holdout scored once, 11-17): selling the lower high after a lower low with a tight stop
reaches 2R before 1R ~41-43% of the time vs ~36-37% for random same-day entries, with or without a failed breakout
first; in option money that is +$0.5-1.3 a contract, inside zero. The question now: which of these entries work.

EVENTS   check_structure's setups, BOTH structures (failed breakout and plain pullback), both directions, SPY / QQQ /
         IWM, entries 09:35-15:30 (same zigzag, entry, stop and look-ahead rules). side = -1 short, +1 long.
TARGET   Y = 1 if price reaches entry + side x 2R before the stop within 30 minutes (R = |stop - entry|), else 0.
MONEY    check_structure's T2 (2R target, chosen on its discovery): the ATM option in the trade's direction, tape-priced
         at the user's fills (buy mid + $0.06, sell mid - $0.08, $0.10 fees).
INPUTS (stamped at the entry; option minute bars only when closed; "signed" = with the trade is +)
  THRU       t_score   frozen breach score (breach_model_frozen.json) of the latest same-direction strike breach on this
                       ticker in the 15 minutes before entry (none -> missing)                                    [user]
             t_fell    1 if that breach fell back through its strike by $0.10 before the entry
  GEX        g_flip    signed distance from the gamma flip (0-7 day chain, IVs at 10:00), bp
             g_wall    1 if the strike nearest the entry is the put wall (short) / call wall (long)
             g_total   dealer gamma (spot-exposures, gamma per 1% move), last minute
             g_mag     the largest single-strike net gamma within 1% of price / all strike |gamma| within 5%   [user]
             g_magdist signed distance to that strike, bp                                                          [user]
             g_strike  net strike gamma at the strike nearest the entry                                           [user]
  STRUCTURE  s_fb      1 failed breakout before (higher low + higher high), 0 plain pullback
             s_break   the failed breakout's size, (Hd - Hb) / price, bp (missing for pullbacks)
             s_under   how far the lower low undercut the last low, in R_zigzag
             s_retrace how much of the drop the bounce retraced, (M - Le) / (Hd - Le)
             s_speed   the drop into the lower low, % of price per minute
             s_risk    the stop distance, % of price
             s_swings  confirmed pivots so far today
  STRIKES    k_high    |bounce high - nearest whole dollar|, bp (did the bounce stall AT a strike)
             k_entry   |entry - nearest whole dollar|, bp
  VOL        v_path30  the last 30 minutes' absolute path / the 09:35 straddle
             v_amused  |price - open| / the 09:35 straddle
             v_rvol    cumulative volume / the prior 20 sessions' mean at the same minute
  FLOW       f_delta5  0DTE directional delta bought, 5 min, signed
             f_delta15 the same over 15 minutes
             f_prem15  0DTE net premium change, 15 min, signed
             f_tide    UW Market Tide 15-minute change, signed
             f_atmimb  the ATM contract in the trade's direction, (ask - bid) / (ask + bid + 1), last 3 min
             f_surge   six-pack surge at the strike nearest the entry
  CROSS      x_struct  other tickers' same-direction setup entries in the 15 minutes before
             x_ret15   the other two indices' 15-minute return, signed, bp
  CONTEXT    c_tod     minutes since 09:30
             c_vwap    signed distance from VWAP, bp
             c_range   signed position in the day's range so far
             c_tr30    signed 30-minute trend, bp
SAMPLES  DISCOVERY 2024-08-20 .. 2025-12-31. HOLDOUT 2026-01-02 .. 2026-11-13, scored once (--holdout writes
         _structure_ic_cache/RESULT-holdout.txt). PARTLY CLEAN, said now: check_structure scored this same holdout's
         overall hit rates and T2 money on 11-17 (none of these inputs). The real confirmation is forward data from
         2026-11-16. v_rvol / c_vwap are missing 2026-11-02 .. 11-13 (the volume cache ends 10-30): rank 0.5 there.
PREPROCESSING  ranks within ticker against the discovery distribution (mid-rank ECDF); missing -> 0.5.
PASS RULES
  P1 IC (discovery): Spearman(input, Y), day-block bootstrap p, BH q < 0.10 across the inputs, same sign both halves
     (split 2025-04-30). Reported.
  P2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on all ranks -> Y, fit on discovery. PASS = keeps >= 1 input AND
     holdout Spearman(score, Y) > 0 with the day-block 95% CI above 0.
  P3 MONEY: the holdout's TOP score quintile (cut points from discovery) traded at T2: mean P&L > 0, CI above 0.
  OVERALL PASS = P2 and P3 -> a forward paper log. Nothing deployed on it.

RUN LOG
  2026-11-17. DISCOVERY 10,140 entries / 336 sessions, 2R-first 41.7%.
    P1 IC: 1 of 32 passes -- g_total -0.035 (more dealer gamma -> fewer 2R runs). Near misses: x_ret15 +0.026, s_risk
       -0.024, v_rvol +0.024, s_swings +0.023, g_strike -0.021, g_mag -0.020 (a big gamma wall near price -> worse).
       THRU context: t_score +0.007, t_fell +0.003 -- nothing. Tide -0.001, VWAP -0.001, 30-min trend +0.006.
    P2 LASSO keeps 9: s_risk -0.039, g_total -0.036, x_ret15 +0.022, g_mag -0.017, v_rvol +0.009, s_swings +0.008,
       s_under +0.006, s_speed +0.003, f_delta5 +0.001. Discovery quintiles 2R-first 37.0 -> 46.5%, T2 -$0.62 -> +$1.16.
    HOLDOUT (6,059 / 211 sessions; partly clean): IC +0.032 [+0.003, +0.060] -> P2 PASS (barely).
       The score shifted up in 2026 (more negative gamma): 2,323 of 6,059 land in the discovery top quintile.
       Quintiles 2R-first 38.7 / 41.7 / 40.6 / 38.8 / 44.3%; T2 +0.49 / +0.27 / -0.80 / -2.07 / +0.52.
       P3 top quintile T2 +$0.52 [-0.47, +1.49] -> FAIL. OVERALL FAIL.
    READ: as with the breaches, the inputs rank which entries run (negative gamma, small stops, other indices
    moving the same way, no big gamma wall at price), but not by enough to pay through the option.

Usage:  python check_structure_ic.py --build | --discovery | --holdout
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
import check_breach_forward as BF                  # noqa: E402
import check_breach_ic as B                        # noqa: E402
import check_sixpack_lake as L                     # noqa: E402
import check_structure as S                        # noqa: E402

CACHE = os.path.join(ROOT, "_structure_ic_cache")
EV = os.path.join(CACHE, "events")
DISC, HOLD, HALF = ("2024-08-20", "2025-12-31"), ("2026-01-02", "2026-11-13"), "2025-04-30"
INPUTS = ["t_score", "t_fell", "g_flip", "g_wall", "g_total", "g_mag", "g_magdist", "g_strike",
          "s_fb", "s_break", "s_under", "s_retrace", "s_speed", "s_risk", "s_swings", "k_high", "k_entry",
          "v_path30", "v_amused", "v_rvol", "f_delta5", "f_delta15", "f_prem15", "f_tide", "f_atmimb", "f_surge",
          "x_struct", "x_ret15", "c_tod", "c_vwap", "c_range", "c_tr30"]
SEED = 20261118


def setups_ext(sc, med, kt, R, p0, orient):
    """check_structure.setups with the pivots kept (same rules)."""
    z = orient * med
    piv = S.zigzag(S.minute_closes(sc, z), R)
    out = []
    for i in range(len(piv)):
        if piv[i][0] != "L" or i < 2:
            continue
        Le, Hd, Lc = piv[i], piv[i - 1], piv[i - 2]
        if Hd[0] != "H" or Lc[0] != "L" or not Le[1] < Lc[1]:
            continue
        fb = i >= 4 and piv[i - 3][0] == "H" and piv[i - 4][0] == "L" and Lc[1] > piv[i - 4][1] and Hd[1] > piv[i - 3][1]
        Tc = Le[3]
        if not S.T_LO - 900 <= Tc <= S.T_HI:
            continue
        k0 = int(np.searchsorted(sc, Le[2] * 60))
        j = int(np.searchsorted(kt, Tc, side="left"))
        if j >= len(sc):
            continue
        M = float(np.max(z[k0:j + 1])) if j >= k0 else float(z[j])
        ent = None
        for k in range(j, len(sc)):
            if kt[k] > Tc + S.WAIT_S:
                break
            M = max(M, float(z[k]))
            if M >= Hd[1]:
                break
            if M - z[k] >= S.TURN_FRAC * p0:
                ent = (float(kt[k]) + B.LAT, float(z[k]), M)
                break
        if ent is None or not S.T_LO <= ent[0] <= S.T_HI:
            continue
        out.append(dict(fb=fb, t_in=ent[0], z_e=ent[1], M=ent[2], z_s=ent[2] + S.BUF_FRAC * p0, Hd=Hd[1], Le=Le[1],
                        Lc=Lc[1], Hb=piv[i - 3][1] if fb else None, speed=(Hd[1] - Le[1]) / p0 / max(Le[2] - Hd[2], 1),
                        swings=i + 1, orient=orient))
    return out


def strike_gex(chain, spot):
    """{strike: net dealer gamma $ per 1% at spot} from 0-7 day chain rows (strike, T, iv, oi, is_call)."""
    import gamma_flip as GF
    if not chain:
        return {}
    K, T, iv, oi, sg = GF.chain_arrays(chain)
    sd = iv * np.sqrt(T)
    d1 = (np.log(spot / K) + 0.5 * sd * sd) / sd
    g = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (spot * sd) * oi * 100 * spot * spot * 0.01 * sg
    out = {}
    for k, v in zip(K, g):
        out[float(k)] = out.get(float(k), 0.0) + float(v)
    return out


def build_day(d):
    f = os.path.join(EV, d + ".json")
    if os.path.exists(f):
        return d, "have"
    try:
        path, tq, tide = L.price_path(d), B.tape(d), B.tide_series(d)
        book, chains = B.minute_book(d)
    except Exception as e:
        return d, f"skip {type(e).__name__}"
    gexapi = json.load(open(os.path.join(ROOT, "_sixpack_cache", "gex_api.json")))
    vol = json.load(open(os.path.join(ROOT, "_sixpack_cache", "volume.json")))
    prior = [x for x in L.lake_days("2024-06-01", d) if x < d][-20:]
    frz = json.load(open(BF.FROZEN))
    bfile = os.path.join(B.EV, d + ".json")
    breaches = json.load(open(bfile)) if os.path.exists(bfile) else []
    meds, ups = {}, []
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300 or len(sc) < 200:
            continue
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        kt = B._kt(sc)
        meds[tk] = (sc, med, kt)
        p0 = float(med[0])
        for orient in (1, -1):
            for s in setups_ext(sc, med, kt, S.R_FRAC * p0, p0, orient):
                ups.append((tk, s))
    rows = []
    for tk, s in ups:
        sc, med, kt = meds[tk]
        p0 = float(med[0]); orient = s["orient"]; side = -orient
        o = S.outcome(sc, med, kt, orient, s["t_in"], s["z_e"], s["z_s"])
        if o is None:
            continue
        p_e = orient * s["z_e"]; K = int(round(p_e)); t_in = s["t_in"]
        try:
            F = B.features(d, tk, t_in, 0, K, side, None, (sc, med), book, chains.get(tk) or [], gexapi, vol, prior, tide, [])
        except Exception:
            continue
        r = dict(d=d, t=tk, side=side, t_in=round(t_in, 1), Y=int(o["hit2"]), mfe_r=o["mfe_r"],
                 T2=S.money(tq, tk, orient, t_in, s["z_e"], s["z_s"], sc, med, kt, "T2"))
        # THRU context
        bs = [b for b in breaches if b["t"] == tk and b["side"] == side and t_in - 900 <= b["t_in"] < t_in]
        r["t_score"] = r["t_fell"] = None
        if bs:
            b = max(bs, key=lambda b: b["t_in"])
            r["t_score"] = BF.score(frz, b)
            j0 = int(np.searchsorted(kt, b["t_in"], side="right")); j1 = int(np.searchsorted(kt, t_in, side="right"))
            r["t_fell"] = int(bool(np.any(side * (med[j0:j1] - b["K"]) <= -B.ZB))) if j1 > j0 else 0
        # GEX
        r["g_flip"], r["g_total"], r["g_wall"] = F["s_flip"], F["c_gex"], F["s_wall"]
        sg = strike_gex(chains.get(tk) or [], p_e)
        near = {k: v for k, v in sg.items() if abs(k / p_e - 1) <= 0.01}
        wide = sum(abs(v) for k, v in sg.items() if abs(k / p_e - 1) <= 0.05)
        if near and wide > 0:
            km = max(near, key=lambda k: abs(near[k]))
            r["g_mag"] = abs(near[km]) / wide
            r["g_magdist"] = side * (km - p_e) / p_e * 1e4
        else:
            r["g_mag"] = r["g_magdist"] = None
        r["g_strike"] = sg.get(float(K))
        # structure
        R = S.R_FRAC * p0
        r["s_fb"] = int(s["fb"])
        r["s_break"] = (s["Hd"] - s["Hb"]) / p0 * 1e4 if s["Hb"] is not None else None
        r["s_under"] = (s["Lc"] - s["Le"]) / R
        r["s_retrace"] = (s["M"] - s["Le"]) / (s["Hd"] - s["Le"]) if s["Hd"] > s["Le"] else None
        r["s_speed"], r["s_risk"], r["s_swings"] = s["speed"], o["risk"] / p0, s["swings"]
        hiP = orient * s["M"]
        r["k_high"] = abs(hiP - round(hiP)) / p0 * 1e4
        r["k_entry"] = abs(p_e - round(p_e)) / p0 * 1e4
        # vol / flow / context from check_breach_ic's feature function (signed with the trade)
        r.update(v_path30=F["c_path30"], v_amused=F["c_am_used"], v_rvol=F["c_rvol"], f_delta5=F["f_delta5"],
                 f_delta15=F["f_delta15"], f_prem15=F["f_prem15"], f_tide=F["c_tide"], f_atmimb=F["q_imb"],
                 f_surge=F["s_surge"], c_tod=F["c_tod"], c_vwap=F["c_vwap"], c_range=F["c_range"], c_tr30=F["c_tr30"])
        # cross-index
        r["x_struct"] = sum(1 for tk2, s2 in ups if tk2 != tk and -s2["orient"] == side and t_in - 900 <= s2["t_in"] < t_in)
        xr = []
        for tk2, (sc2, med2, kt2) in meds.items():
            if tk2 == tk:
                continue
            a, b = B._known(sc2, med2, t_in - 900), B._known(sc2, med2, t_in)
            xr.append(side * (b / a - 1) * 1e4)
        r["x_ret15"] = float(np.mean(xr)) if xr else None
        rows.append(r)
    os.makedirs(EV, exist_ok=True)
    json.dump(rows, open(f, "w"))
    return d, len(rows)


def load(lo, hi):
    return [r for f in sorted(glob.glob(os.path.join(EV, "*.json"))) if lo <= os.path.basename(f)[:10] <= hi
            for r in json.load(open(f))]


def ranker(disc):
    cuts = {(tk, k): np.sort(np.array([r[k] for r in disc if r["t"] == tk and r[k] is not None], float))
            for tk in ("SPY", "QQQ", "IWM") for k in INPUTS}
    def X(rows):
        out = np.full((len(rows), len(INPUTS)), 0.5)
        for i, r in enumerate(rows):
            for j, k in enumerate(INPUTS):
                v, c = r[k], cuts[(r["t"], k)]
                if v is not None and c.size:
                    out[i, j] = (np.searchsorted(c, v, "left") + np.searchsorted(c, v, "right")) / (2.0 * c.size)
        return out
    return X


def fit(disc):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    X = ranker(disc)
    Xd = X(disc) - 0.5
    y = np.array([r["Y"] for r in disc], float)
    wk = np.array([dt.date.fromisoformat(r["d"]).isocalendar()[0] * 100 + dt.date.fromisoformat(r["d"]).isocalendar()[1] for r in disc])
    cv = list(GroupKFold(n_splits=5).split(Xd, y, wk))
    return X, LassoCV(cv=cv, random_state=0, alphas=60, max_iter=20000).fit(Xd, y)


def discovery():
    rng = np.random.default_rng(SEED)
    disc = load(*DISC)
    X = ranker(disc); Xd = X(disc); y = np.array([r["Y"] for r in disc], float)
    h1 = np.array([r["d"] <= HALF for r in disc])
    print(f"DISCOVERY {DISC[0]}..{DISC[1]}: {len(disc)} entries, {len({r['d'] for r in disc})} sessions, 2R-first {y.mean():.1%}")
    ps, rows = [], []
    for j, k in enumerate(INPUTS):
        ic = B.spearman(Xd[:, j], y)
        bs = B.day_boot(disc, lambda idx: B.spearman(Xd[idx, j], y[idx]), rng, 500)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        a, b = B.spearman(Xd[h1, j], y[h1]), B.spearman(Xd[~h1, j], y[~h1])
        cov = np.mean([r[k] is not None for r in disc])
        ps.append(p); rows.append((k, ic, p, a, b, cov))
    qs = B.bh(ps)
    print("  P1 IC vs Y (2R first)")
    for (k, ic, p, a, b, cov), q in zip(rows, qs):
        ok = q < 0.10 and np.sign(a) == np.sign(b) == np.sign(ic)
        print(f"    {k:10s} IC {ic:+.3f}  p {p:.3f}  q {q:.3f}  halves {a:+.3f} / {b:+.3f}  {'PASS' if ok else '    '}  coverage {cov:.0%}")
    X, m = fit(disc)
    coef = sorted([(abs(c), k, c) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9], reverse=True)
    print(f"  P2 LASSO keeps {len(coef)} of {len(INPUTS)}: " + ", ".join(f"{k} {c:+.3f}" for _a, k, c in coef))
    sd = m.predict(X(disc) - 0.5)
    print(f"    in-sample IC {B.spearman(sd, y):+.3f}")
    q = np.quantile(sd, [0.2, 0.4, 0.6, 0.8]); qi = np.searchsorted(q, sd)
    for i in range(5):
        rs = [r for r, k in zip(disc, qi) if k == i]
        v = [r["T2"] for r in rs if r["T2"] is not None]
        print(f"    discovery Q{i + 1}: 2R-first {np.mean([r['Y'] for r in rs]):.1%}  T2 ${np.mean(v):+.2f} (n {len(v)})")


def holdout():
    res = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(res):
        sys.exit(f"the holdout was scored already -- see {res}")
    rng = np.random.default_rng(SEED + 1)
    disc, hold = load(*DISC), load(*HOLD)
    X, m = fit(disc)
    kept = [k for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9]
    sh = m.predict(X(hold) - 0.5); y = np.array([r["Y"] for r in hold], float)
    ic = B.spearman(sh, y)
    bs = B.day_boot(hold, lambda idx: B.spearman(sh[idx], y[idx]), rng)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p2 = bool(kept) and lo > 0
    lines = [f"HOLDOUT {HOLD[0]}..{HOLD[1]} scored {dt.datetime.now():%Y-%m-%d %H:%M}: {len(hold)} entries, "
             f"{len({r['d'] for r in hold})} sessions (PARTLY CLEAN: overall rates seen 11-17)",
             f"  P2 LASSO keeps {len(kept)} ({', '.join(kept) or '-'}): holdout IC {ic:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if p2 else 'FAIL'}"]
    sd = m.predict(X(disc) - 0.5)
    q = np.quantile(sd, [0.2, 0.4, 0.6, 0.8]); qi = np.searchsorted(q, sh)
    for i in range(5):
        rs = [r for r, k in zip(hold, qi) if k == i]
        v = [r["T2"] for r in rs if r["T2"] is not None]
        lines.append(f"    holdout Q{i + 1}: n {len(rs)} 2R-first {np.mean([r['Y'] for r in rs]):.1%}  T2 ${np.mean(v):+.2f}")
    top = [r for r, k in zip(hold, qi) if k == 4 and r["T2"] is not None]
    v = np.array([r["T2"] for r in top])
    bm = B.day_boot(top, lambda idx: v[idx].mean(), rng)
    l3, h3 = np.quantile(bm, [0.025, 0.975])
    p3 = v.mean() > 0 and l3 > 0
    lines.append(f"  P3 TOP quintile at T2: n {len(v)} mean ${v.mean():+.2f}/contract [{l3:+.2f}, {h3:+.2f}] -> {'PASS' if p3 else 'FAIL'}")
    lines.append(f"  OVERALL: {'PASS -> forward paper log' if p2 and p3 else 'FAIL'}")
    print("\n".join(lines))
    open(res, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "discovery", "holdout"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.build:
        days = [d for d in L.lake_days(DISC[0], HOLD[1])
                if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))]
        missing = [d for d in days if not os.path.exists(os.path.join(B.EV, d + ".json"))]
        with ProcessPoolExecutor(6) as ex:                      # THRU context needs the breach events for every day
            for d, msg in ex.map(B.build_day, missing):
                pass
        with ProcessPoolExecutor(6) as ex:
            for k, (d, msg) in enumerate(ex.map(build_day, days)):
                if k % 50 == 0:
                    print(k, d, msg, flush=True)
        print("built", len(days))
    if a.discovery: discovery()
    if a.holdout: holdout()
