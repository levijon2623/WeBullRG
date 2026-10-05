# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_fuzzy_sizing.py
=====================
Can a FUZZY CONFIDENCE SCORE built from seven entry-time inputs SIZE the flow
trigger's trades better than flat sizing? (User's input list and design,
2026-10-03.) Sizing, not gating: every trade is still taken, so the trade
list is fixed and the only question is whether higher-scored trades earn
more, per dollar, out of sample.

POPULATION
    The SPY / QQQ / IWM flow-trigger SCREEN of check_implied_move_gate (same
    cached candidates): CALL + PUT x min_flow_pct 50/80/90 = 18 cells, hours
    9-14, entry >= 09:35, ATM, dte [0, 1], trail50, sequential per cell, fill
    "bot" + cushion. 2024-08-20 .. 2026-08-21 (HEDGE{T}'s end).

THE SEVEN INPUTS, at the trigger minute m, s = +1 CALL / -1 PUT
    I1 URGENCY   s x net_flow_1m(m) / typical |net_flow_1m| (mean of the prior
                 20 sessions' daily medians) -- the crossing minute's push
    I2 STRETCH   s x (close(m) - close(09:35)) / the 09:35 ATM straddle
    I3 WALL      distance to the nearest 0-1DTE wall (call or put) AHEAD in
                 the trade's direction, in straddles, capped at 3 (none ahead
                 = 3); walls as in check_implied_breakout_filters (lake OI x
                 BS gamma at the straddle-implied vol, 09:35)
    I4 HEDGE     s x sum of hedge_sh_sl over m-4..m / typical |hedge_sh_sl|
                 per minute (prior 20 sessions). CAVEAT: 0DTE lake deltas are
                 corrupt ~2024-10..2025-12 (METHODOLOGY 7)
    I5 VWAP      s x (close(m) - session VWAP(m)) / the 09:35 straddle
    I6 RVOL      volume(m) / median volume of minute m over the prior 20
                 sessions
    I7 TIME      FIXED curve (the user's prior, not fitted): 1.0 09:35-11:00,
                 ramp to 0 by 11:30, 0 until 14:00, ramp to 1.0 by 14:30

FUZZIFICATION -- membership functions, calibrated PER TICKER
    I1-I6 -> percentile rank (0..1) of the value among that ticker's
    candidate values from the prior 250 sessions (dates strictly before the
    day; >= 100 values, else 0.5). I7 is its own membership. A missing input
    = 0.5 (neutral).

SCORE AND SIZE -- walk-forward, nothing hand-set
    For slice k = 2..6, on the trades of slices 1..k-1 (pooled tickers):
      w_i   = Spearman rank correlation of membership i with trade ROE
      score = sum_i w_i x (membership_i - 0.5)
      cut   = the 1/3 and 2/3 quantiles of the training scores
    Slice-k trades are sized 0.5x below the first cut, 1.5x above the second,
    1.0x between. Chain slices 2..6.

THE STATISTIC -- per dollar, so the comparison holds capital fixed
    G = sum(size x roe) / sum(size)  -  mean(roe)        (chained trades)
    The bot sizes every entry to the same premium budget, so ROE is P&L per
    dollar; G > 0 means the score moved money toward better trades.

PRE-REGISTERED CRITERIA (fixed 2026-10-03, before the first scored run)
    Z1  chained G > 0
    Z2  G > p95 of 20 placebos: memberships PERMUTED across trades within
        each ticker (same distributions, no link to the outcome), the whole
        walk-forward rerun on each
    Z3  G > 0 in >= 3 of the 5 chained slices
    Z4  G > 0 under fill = mid, bot AND worst
    PASS = Z1..Z4.

REPORTED, NOT SCORED
    weights per slice (stability), G per ticker and over the OOS half
    (slices 4-6), mean ROE / loss50 / hit50 by score tercile (the two-tails
    trap: a score that predicts big losses usually predicts big wins), input
    coverage. (The deployed SPY/QQQ/IWM CALL rules, 148 trades, are not
    reported: too thin to say anything, and a second input pipeline.)

MACHINERY (before scoring)
    M1  ranks use earlier dates only (asserted while building), and I2 = 0
        at a 09:35 entry
    M2  a planted input (= "trade won") gets the dominant weight and a large G
    M3  constant memberships give sizes all 1.0 and G = 0 exactly
    M4  every input defined for >= 80% of trades

RUN LOG
  2026-10-03 run 1: M1-M4 PASS (planted input dominant every slice, G +14.8pp;
    constant -> G 0 exactly). 11,415 trades, 9,646 scored (slices 2-6).
    VERDICT FAIL (Z2): chained G +0.23pp per trade vs placebo median -0.02 /
    p95 +0.27. Z1, Z3 (4/5 slices), Z4 (mid +0.49, worst +0.33) pass -- the
    sign is real-looking but the size is within noise.
    Weights are tiny (|Spearman| <= 0.04; METHODOLOGY's skip break-even is
    ~0.10) though some signs hold in every slice: URGENCY +, STRETCH +, VWAP
    + (with-the-move entries a touch better -- the same direction as
    check_implied_move_gate), HEDGE - (aligned dealer pressure slightly WORSE;
    the corrupt 0DTE deltas may be in that), WALL -. By size: 0.5x -10.4%,
    1.0x -10.8%, 1.5x -9.1% -- the score barely ranks trades, and loss50 /
    hit50 do not move. The inputs carry almost no information about a flow
    trigger's outcome on this screen.

Usage:
  python check_fuzzy_sizing.py
"""
from __future__ import annotations

import bisect
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                         # noqa: E402
import check_implied_move_gate as G                          # noqa: E402
import check_implied_breakout_filters as BF                  # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
END = BF.END
INPUTS = ("I1_URGENCY", "I2_STRETCH", "I3_WALL", "I4_HEDGE", "I5_VWAP", "I6_RVOL", "I7_TIME")
RANKED = INPUTS[:6]
FILLS = ("bot", "mid", "worst")
RANK_WIN, RANK_MIN, TYP_N = 250, 100, 20
N_PLACEBO = 20
SEED = 20261006
WALL_CAP = 3.0


def time_curve(m):
    if 575 <= m <= 660:
        return 1.0
    if 660 < m < 690:
        return (690 - m) / 30
    if 690 <= m <= 840:
        return 0.0
    if 840 < m < 870:
        return (m - 840) / 30
    return 1.0


# ================================================================== raw inputs
def flow_tables(D, tk):
    """{(date, mod): net_flow_1m} and {date: typical |net| (prior 20 sessions)}."""
    from check_config_walkforward import _flow_for
    f = _flow_for(D, [tk])
    f = f[f["underlying_symbol"] == tk].copy()
    ts = pd.to_datetime(f["minute_et"])
    f["date"] = ts.dt.date
    f["mod"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
    net = dict(zip(zip(f["date"], f["mod"]), f["net_flow_1m"].astype(float)))
    med = f[(f["mod"] >= 570) & (f["mod"] <= 959)].groupby("date")["net_flow_1m"].apply(
        lambda x: float(np.median(np.abs(x))))
    typ = med.shift(1).rolling(TYP_N, min_periods=TYP_N).mean()
    return net, {d: v for d, v in typ.items() if np.isfinite(v) and v > 0}


def hedge_tables(tk):
    h = BF.hedge_map(tk)
    days = sorted(h)
    meds = [float(np.median(np.abs(list(h[d].values())))) if h[d] else np.nan for d in days]
    typ = pd.Series(meds, index=days).shift(1).rolling(TYP_N, min_periods=TYP_N).mean()
    return h, {d: v for d, v in typ.items() if np.isfinite(v) and v > 0}


def price_tables(tk):
    """{(date, mod): (close, vwap)} from the 1m bars, VWAP anchored at 09:30."""
    d = BF.bars_full(tk).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["tp"] = (d["high"] + d["low"] + d["close"]) / 3
    d["pv"] = d["tp"] * d["volume"]
    d["cpv"] = d.groupby("date")["pv"].cumsum()
    d["cv"] = d.groupby("date")["volume"].cumsum()
    vw = np.where(d["cv"] > 0, d["cpv"] / d["cv"].where(d["cv"] > 0, 1), d["close"])
    return {(a, int(b)): (c, v) for a, b, c, v in zip(d["date"], d["mod"], d["close"], vw)}


def raw_inputs(tk, direction, day, m, ctx):
    net, ftyp, hedge, htyp, px, rv, walls, am = ctx
    s = 1 if direction == "CALL" else -1
    out = dict.fromkeys(INPUTS)
    v = net.get((day, m))
    if v is not None and ftyp.get(day):
        out["I1_URGENCY"] = s * v / ftyp[day]
    p = px.get((day, m))
    a = px.get((day, 575))
    st = am.get((tk, day))
    S = st[1] if st else None
    if p and a and S:
        out["I2_STRETCH"] = s * (p[0] - a[0]) / S
        out["I5_VWAP"] = s * (p[0] - p[1]) / S
    w = walls.get((tk, day))
    if w and p and S:
        ahead = [s * (K - p[0]) for K in w if s * (K - p[0]) >= 0]
        out["I3_WALL"] = min(min(ahead) / S, WALL_CAP) if ahead else WALL_CAP
    hd = hedge.get(day)
    if hd is not None and htyp.get(day):
        out["I4_HEDGE"] = s * sum(hd.get(k, 0.0) for k in range(m - 4, m + 1)) / htyp[day]
    r = (rv.get((day, m)) or {}).get("rvol")
    if r is not None:
        out["I6_RVOL"] = r
    out["I7_TIME"] = time_curve(m)
    return out


# ================================================================== membership
def memberships(raw):
    """raw: {(tk, dir, date, mod): {input: value}} -> same keys, values 0..1.
    Per ticker, the percentile rank among that ticker's values from the prior
    RANK_WIN sessions -- dates STRICTLY before the day (asserted)."""
    out = {k: {} for k in raw}
    for tk in TICKERS:
        keys = [k for k in raw if k[0] == tk]
        days = sorted({k[2] for k in keys})
        on_day = {}
        for k in keys:
            on_day.setdefault(k[2], []).append(k)
        for inp in RANKED:
            vals = {d: [raw[k][inp] for k in on_day[d] if raw[k][inp] is not None] for d in days}
            for i, d in enumerate(days):
                window = days[max(0, i - RANK_WIN):i]
                assert all(w < d for w in window)
                ref = sorted(v for w in window for v in vals[w])
                for k in on_day[d]:
                    v = raw[k][inp]
                    if v is None or len(ref) < RANK_MIN:
                        out[k][inp] = 0.5
                    else:
                        lo_i, hi_i = bisect.bisect_left(ref, v), bisect.bisect_right(ref, v)
                        out[k][inp] = (lo_i + hi_i) / 2 / len(ref)
        for k in keys:
            out[k]["I7_TIME"] = raw[k]["I7_TIME"]
    return out


# ================================================================== trades
def trades(cands, fill):
    """[(date, roe, key, slice)] for every sequential trade of the 18 cells."""
    pol = SC.policy_for(G.screen_rule("SPY", "CALL", 50))
    out = []
    for (tk, direction, pct), rows in cands.items():
        cand = [(d, m, p) for d, m, p, _s in rows if d <= END]
        picks = []
        res = SC.walk(cand, pol, SC.DEFAULT_EOD, fill=fill, after=G.AFTER, picks_out=picks)
        for (ci, _x), (d, roe) in zip(picks, res):
            out.append((d, roe, (tk, direction, d, cand[ci][1]), BF.slice_of(d)))
    return out


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def walk_forward(tr, mem):
    """-> (G chained, G per slice, chained rows [(date, roe, size, key, slice)], weights per slice)."""
    rows, weights, per = [], {}, {}
    for k in range(1, 6):                   # slice index k is scored; 0..k-1 train
        train = [t for t in tr if t[3] is not None and t[3] < k]
        test = [t for t in tr if t[3] == k]
        if not train or not test:
            continue
        y = [t[1] for t in train]
        w = {i: spearman([mem[t[2]][i] for t in train], y) for i in INPUTS}
        weights[k + 1] = w

        def score(t):
            return sum(w[i] * (mem[t[2]][i] - 0.5) for i in INPUTS)
        sc = np.array([score(t) for t in train])
        q1, q2 = np.quantile(sc, 1 / 3), np.quantile(sc, 2 / 3)
        for t in test:
            s = score(t)
            size = 0.5 if s < q1 else 1.5 if s > q2 else 1.0
            rows.append((t[0], t[1], size, t[2], k + 1))
        tk_rows = [r for r in rows if r[4] == k + 1]
        per[k + 1] = gain(tk_rows)
    return gain(rows), per, rows, weights


def gain(rows):
    if not rows:
        return float("nan")
    r = np.array([x[1] for x in rows])
    s = np.array([x[2] for x in rows])
    return float((s * r).sum() / s.sum() - r.mean())


# ================================================================== main
def main():
    import directional_flow_backtester as D
    cands = G.candidates(D)
    am = BF.am_straddles()
    walls = pickle.load(open(BF.WALL_CACHE, "rb"))
    keys = {(tk, d, day, m) for (tk, d, _p), rows in cands.items() for day, m, _pl, _s in rows if day <= END}
    raw = {}
    for tk in TICKERS:
        print(f"  inputs {tk} ...", flush=True)
        net, ftyp = flow_tables(D, tk)
        hedge, htyp = hedge_tables(tk)
        px = price_tables(tk)
        rv, _hl = BF.series_features(tk)
        ctx = (net, ftyp, hedge, htyp, px, rv, walls, am)
        for k in keys:
            if k[0] == tk:
                raw[k] = raw_inputs(tk, k[1], k[2], k[3], ctx)
    mem = memberships(raw)

    tr = {f: trades(cands, f) for f in FILLS}
    bot = tr["bot"]
    # ---- machinery
    cov = {i: np.mean([raw[t[2]][i] is not None for t in bot]) for i in INPUTS}
    m4 = all(v >= 0.80 for v in cov.values())
    print("  M4 coverage: " + "  ".join(f"{i[:2]} {v:.0%}" for i, v in cov.items()) + f"  -> {'PASS' if m4 else 'FAIL'}")
    z0 = [raw[t[2]]["I2_STRETCH"] for t in bot if t[2][3] == 575 and raw[t[2]]["I2_STRETCH"] is not None]
    m1 = all(abs(v) < 1e-12 for v in z0)
    print(f"  M1 stretch = 0 at a 09:35 entry ({len(z0)} trades): {'PASS' if m1 else 'FAIL'}  (rank causality asserted)")
    planted = {k: dict(v) for k, v in mem.items()}
    won = {t[2]: t[1] > 0 for t in bot}
    for k in planted:
        planted[k]["I1_URGENCY"] = 1.0 if won.get(k) else 0.0
    gp, _per, _r, wp = walk_forward(bot, planted)
    dom = all(max(w, key=lambda i: abs(w[i])) == "I1_URGENCY" for w in wp.values())
    m2 = dom and gp > 0.10
    print(f"  M2 planted input dominant in every slice: {dom}, G {gp * 100:+.1f}pp: {'PASS' if m2 else 'FAIL'}")
    const = {k: dict.fromkeys(INPUTS, 0.5) for k in mem}
    gc, _p, rc, _w = walk_forward(bot, const)
    m3 = all(r[2] == 1.0 for r in rc) and abs(gc) < 1e-12
    print(f"  M3 constant memberships -> sizes all 1.0, G {gc:+.2e}: {'PASS' if m3 else 'FAIL'}")
    if not (m1 and m2 and m3 and m4):
        sys.exit("machinery checks failed -- not scoring")

    # ---- scoring
    g, per, rows, weights = walk_forward(bot, mem)
    print(f"\n  TRADES {len(bot)} (scored, slices 2-6: {len(rows)})")
    print("  WEIGHTS (Spearman with ROE on the training slices):")
    print("    slice " + " ".join(f"{i[:9]:>10}" for i in INPUTS))
    for k, w in weights.items():
        print(f"    {k:5} " + " ".join(f"{w[i]:+10.3f}" for i in INPUTS))
    print("  G per slice: " + "  ".join(f"{k}: {v * 100:+.2f}pp" for k, v in per.items()))
    pos = sum(v > 0 for v in per.values())
    oos = [r for r in rows if r[0] >= SC.SPLIT]
    print(f"  CHAINED G {g * 100:+.2f}pp per trade (flat mean ROE {np.mean([r[1] for r in rows]) * 100:+.1f}%)   "
          f"OOS half (slices 4-6) G {gain(oos) * 100:+.2f}pp")
    for tk in TICKERS:
        print(f"    {tk}: G {gain([r for r in rows if r[3][0] == tk]) * 100:+.2f}pp")
    print("  BY SIZE (two-tails check):")
    for sz in (0.5, 1.0, 1.5):
        v = np.array([r[1] for r in rows if r[2] == sz])
        if len(v):
            print(f"    {sz:.1f}x  n={len(v):5}  mean ROE {v.mean() * 100:+6.1f}%  loss50 {np.mean(v <= -0.5):.0%}  "
                  f"hit50 {np.mean(v >= 0.5):.0%}")

    rng = np.random.default_rng(SEED)
    pg = []
    for _ in range(N_PLACEBO):
        pm = {}
        for tk in TICKERS:
            ks = [k for k in mem if k[0] == tk]
            perm = rng.permutation(len(ks))
            for a, b in zip(ks, perm):
                pm[a] = mem[ks[b]]
        pg.append(walk_forward(bot, pm)[0])
    p95 = float(np.quantile(pg, 0.95))
    fills = {f: walk_forward(tr[f], mem)[0] for f in FILLS}
    crit = dict(Z1=g > 0, Z2=g > p95, Z3=pos >= 3, Z4=all(v > 0 for v in fills.values()))
    print(f"  placebo G: median {np.median(pg) * 100:+.2f}pp  p95 {p95 * 100:+.2f}pp")
    print("  fills: " + "  ".join(f"{f} {v * 100:+.2f}pp" for f, v in fills.items()))
    print("  " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> {'PASS' if all(crit.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
