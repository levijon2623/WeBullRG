# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_implied_breakout_filters.py
=================================
STAGE 2 of the implied-band trigger (stage 1: check_implied_breakout -- both
the BREAKOUT and the FADE of the first 1m close beyond the 09:35 65% band
FAILED, each about equal to its placebo, ~800 trades per arm). Can a filter
at the break minute find a subset that pays? Eight filters were proposed
(user, 2026-10-03: RSI, EMA stack, delta-weighted candle, GEX walls, sweep
flow, time of day; plus two of mine). Eight filters x two states x two arms
is a SEARCH, so the verdict is a WALK-FORWARD SELECTION, never the best cell.

THE BREAKS (identical to stage 1)
    First 1m close above high65 / below low65 of that day's walk-forward 09:35
    band, 09:36-14:59, SPY/QQQ/IWM. BREAKOUT buys with the break (call up,
    put down); FADE buys against it. ATM, dte [0, 1], trail50, EOD 15:55,
    fill "bot" + cushion; calls and puts walked together per ticker (one
    position per ticker). Window 2024-08-20 .. 2026-08-21 (HEDGE{T} ends
    2026-08-21; METHODOLOGY 8's last slice ends 2026-08-23).

THE FILTERS -- each a yes/no at the break minute m, "aligned" = in the break's
direction, computed from data at or before m's close, thresholds fixed here
    F1 RSI      Wilder RSI(14) of 1m closes (RTH series, carried across days)
                >= 70 on an up-break, <= 30 on a down-break
    F2 EMA      EMA 8 > 21 > 50 of 1m closes (up-break), 8 < 21 < 50 (down)
    F3 HEDGE    sum of hedge_sh_sl (historical/HEDGE{T}, dealer share demand,
                single-leg) over m-4..m has the break's sign. CAVEAT: built on
                the lake's deltas, and 0DTE greeks are corrupt ~2024-10..
                2025-12 (METHODOLOGY 7); for a SIGN the damage is bounded
                (a too-low IV pushes deltas toward 0/1, not across zero).
    F4 WALL     a 0-1DTE GEX wall (call or put) sits within [-0.25 S, +0.5 S]
                of the break price, measured in the break's direction (S = that
                day's 09:35 straddle). Walls from lake OI at 09:35 x Black-
                Scholes gamma at ONE vol implied by the 09:35 straddle (the
                lake's own 0DTE gamma is corrupt -- see F3). BREAKOUT's
                "vacuum" = F4 false; FADE's "trampoline" = F4 true.
    F5 SWEEP    net ask-side single-leg ISO premium (check_big_sweep_lead's
                parents: ATM +-0.5%, same-day / next expiry; calls +, puts -)
                over m-9..m has the break's sign (no sweeps = false).
                BREAKOUT's "convergence" = true; FADE's "divergence" = false.
    F6 MIDDAY   m in 11:30-13:59 (the proposed home of the fade; AM / late
                for breakouts = false)
    F7 RVOL     the break minute's volume >= 2x the median of that minute over
                the prior 20 sessions (mine: a volume-confirmed break)
    F8 TWOSIDED the OPPOSITE band edge was touched (1m high/low) earlier that
                day (mine: a range day, where a break is more likely to fail)
    A missing feature (no sweep file, no HEDGE row) drops that break from
    both states of that filter only.

POLICIES  34 = {BREAKOUT, FADE} x ({no filter} + 8 filters x {true, false}).
    A filter keeps only the breaks in its state; skipped breaks leave the book
    flat. Per-trade P&L is precomputed per candidate and fill; the fast walk
    reproduces sim_core.walk exactly (M2).

THE VERDICT -- walk-forward selection (METHODOLOGY 5), fixed 2026-10-03
    For slice k = 2..6: among the 34 policies with >= 40 trades on >= 20 days
    in slices 1..k-1, select the highest per-trade mean on slices 1..k-1;
    score its slice-k trades. Chain slices 2..6.
    W1  chained per-trade mean > 0
    W2  chained mean > p95 of the SAME selection run on each of 20 placebo
        datasets (breaks of another day's band widths on today's anchor,
        features recomputed at the placebo minutes) -- the search's own luck
        is inside the null
    W3  >= 3 of the 5 chained slices positive
    W4  W1 holds when the whole procedure is rerun with fill = mid AND worst
    PASS = W1..W4.

REPORTED, NOT SCORED
    the 34-policy table (IS/OOS, n, days) -- a SEARCH: its best OOS cell is
    expected to look good by chance; which policy each slice selected (a
    winner that changes every slice is the signature of noise); feature
    coverage.

MACHINERY (before scoring)
    M1  the break set equals check_implied_breakout's (same minutes)
    M2  the fast walk equals sim_core.walk on the unfiltered BREAKOUT arm
    M3  each feature is defined for >= 80% of real breaks (F3: within its
        file's range), and F6 / F8 agree with a direct recomputation
    M4  the selection can move: a planted filter equal to "trade won" is
        selected in every slice where it is ELIGIBLE (>= 40 trades / 20 days
        of training) and gives a large chained mean

RUN LOG
  2026-10-03 run 1: blocked by M4 before any scoring. The planted policy had
    25 training trades in slice 2 (< MIN_TRADES 40), so it was ineligible
    there and FADE F1_RSI=F was picked; from slice 3 on it was picked every
    time (train means +93..+115%), chained +55%. The selection worked; M4 as
    written ignored eligibility. M4 now checks the eligible slices only. The
    selection rules (40 trades / 20 days) are unchanged. No outcome seen.
  2026-10-03 run 2: M1-M4 PASS. VERDICT FAIL on all four.
    Picks changed every slice (FADE F1=T, FADE F7=F, FADE F4=T, BREAKOUT
    F6=F, BREAKOUT F6=F) -- the signature of noise. Chained n=400, mean
    -13.2%, 0/5 slices positive; placebo median -16.4% / p95 -4.2%; mid
    -11.6%, worst -13.3%.
    No policy of 34 is positive in BOTH halves. The best-looking OOS cells
    flip sign: BREAKOUT F7_RVOL=T OOS +8.5% / IS -21.5%; FADE F3_HEDGE=F OOS
    +20.4% / IS -26.0%. Against the proposals: midday is the WORST window for
    fades too (OOS -22.2%); fading into a GEX wall IS +1.7% / OOS -18.4%;
    sweep convergence/divergence changes nothing. Breakouts into a wall are
    worse than into clear air (OOS -15.7% vs -1.1%), but clear air is still
    negative in-sample (-16.4%). F8 two-sided days are 1.9% of breaks.

Usage:
  python check_implied_breakout_filters.py --build     # sweep + wall caches
  python check_implied_breakout_filters.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_implied_breakout as S1                     # noqa: E402
from check_implied_move_gate import derangement, screen_rule   # noqa: E402

TICKERS = S1.TICKERS
END = dt.date(2026, 8, 21)
FILLS = ("bot", "mid", "worst")
N_PLACEBO = 20
SEED = 20261005
SWEEP_CACHE = os.path.join("_implied_move_cache", "sweeps_net.pkl")
WALL_CACHE = os.path.join("_implied_move_cache", "walls_0935.pkl")
FILTERS = ("F1_RSI", "F2_EMA", "F3_HEDGE", "F4_WALL", "F5_SWEEP", "F6_MIDDAY", "F7_RVOL", "F8_TWOSIDED")
MIN_TRADES, MIN_DAYS = 40, 20


# ================================================================== caches
def build_sweeps(start=None):
    """{(tk, date): {mod: net premium}} of ask-side single-leg ISO parents.
    `start` reaches back past DEPLOYED_START (check_ic_screen trains on the
    spent pre-sample)."""
    from check_big_sweep_lead import parents_for_day
    out = pickle.load(open(SWEEP_CACHE, "rb")) if os.path.exists(SWEEP_CACHE) else {}
    paths = sorted(glob.glob(os.path.join("lake", "silver", "trades-core", "date=*", "trades.parquet")))
    done = {d for (_t, d) in out}
    for i, p in enumerate(paths, 1):
        day = dt.date.fromisoformat(os.path.basename(os.path.dirname(p)).split("=")[1])
        if day < (start or SC.DEPLOYED_START) or day > END or day in done:
            continue
        par = parents_for_day(p, day)
        for tk in TICKERS:
            g = par.filter(pl.col("ticker") == tk)
            mm = {}
            for m, side, prem in g.select("m", "side", "prem").iter_rows():
                k = m.hour * 60 + m.minute
                mm[k] = mm.get(k, 0.0) + side * prem
            out[(tk, day)] = mm
        if i % 25 == 0:
            print(f"    sweeps {day}", flush=True)
            pickle.dump(out, open(SWEEP_CACHE, "wb"))
    pickle.dump(out, open(SWEEP_CACHE, "wb"))
    return out


def _bs_gamma(S, K, sig, T):
    if sig <= 0 or T <= 0:
        return 0.0
    sd = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sd)


def build_walls(am, start=None):
    """{(tk, date): (call_wall, put_wall)} at 09:35 -- lake OI, BS gamma at the
    vol the 09:35 straddle implies (one vol for all strikes and both expiries).
    `start` reaches back past DEPLOYED_START (check_ema_stack_signal uses the
    spent pre-sample as labelled training data)."""
    out = pickle.load(open(WALL_CACHE, "rb")) if os.path.exists(WALL_CACHE) else {}
    paths = sorted(glob.glob(os.path.join("lake", "silver", "option-contracts-1m", "date=*", "bars.parquet")))
    T0 = (16 * 60 - (9 * 60 + 35)) / 525600
    for i, p in enumerate(paths, 1):
        day = dt.date.fromisoformat(os.path.basename(os.path.dirname(p)).split("=")[1])
        if day < (start or SC.DEPLOYED_START) or day > END or all((tk, day) in out for tk in TICKERS):
            continue
        lf = (pl.scan_parquet(p)
              .filter(pl.col("underlying_symbol").is_in(list(TICKERS))
                      & (pl.col("expiry") >= day) & (pl.col("expiry") <= day + dt.timedelta(days=5)))
              .group_by("underlying_symbol", "option_type", "strike", "expiry")
              .agg(pl.col("open_interest").max().alias("oi"))
              .collect())
        for tk in TICKERS:
            a = am.get((tk, day))
            if a is None:
                continue
            S, strad = a
            sig = strad / (0.7979 * S * math.sqrt(T0))
            g = lf.filter(pl.col("underlying_symbol") == tk)
            exps = sorted(g["expiry"].unique().to_list())
            use = [e for e in exps if e == day][:1] + [e for e in exps if e > day][:1]
            agg = {}
            for typ, K, e, oi in g.filter(pl.col("expiry").is_in(use)).select(
                    "option_type", "strike", "expiry", "oi").iter_rows():
                if not oi or abs(K / S - 1) > 0.03:
                    continue
                T = T0 + (e - day).days / 365
                gx = oi * _bs_gamma(S, K, sig, T) * 100 * S * S * 0.01
                c = agg.setdefault(K, [0.0, 0.0])
                c[0 if typ == "call" else 1] += gx
            if agg:
                out[(tk, day)] = (max(agg, key=lambda k: agg[k][0]), max(agg, key=lambda k: agg[k][1]))
        if i % 50 == 0:
            print(f"    walls {day}", flush=True)
            pickle.dump(out, open(WALL_CACHE, "wb"))
    pickle.dump(out, open(WALL_CACHE, "wb"))
    return out


def am_straddles():
    st = CI_straddles()
    d = st.filter(pl.col("win") == "am")
    return {(r, day): (spot, s) for r, day, spot, s in d.select("root", "date", "spot", "strad").iter_rows()}


def CI_straddles():
    import check_implied_move as CI
    return CI.straddles()


# ================================================================== features
def bars_full(tk):
    """(sorted keys [(date, mod)], close, high, low, vol) over RTH, all days."""
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959))
         .sort("date", "mod")
         .select("date", "mod", "close", "high", "low", "volume"))
    return d


def series_features(tk):
    """{(date, mod): dict(rsi, e8, e21, e50, rvol)} and per-day high/low maps."""
    d = bars_full(tk).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date        # dict keys are datetime.date
    c = d["close"].to_numpy(float)
    delta = np.diff(c, prepend=c[0])
    up, dn = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    au = pd.Series(up).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    rsi = np.where(ad > 0, 100 - 100 / (1 + au / np.where(ad > 0, ad, 1)), 100.0)
    e = {n: pd.Series(c).ewm(span=n, adjust=False).mean().to_numpy() for n in (8, 21, 50)}
    # RVOL: volume / median of the same minute over the prior 20 sessions
    vol = d.pivot_table(index="date", columns="mod", values="volume")
    med = vol.shift(1).rolling(20, min_periods=20).median()
    rv = (vol / med)
    out = {}
    for i, (day, mod) in enumerate(zip(d["date"], d["mod"])):
        r = rv.at[day, mod] if (day in rv.index and mod in rv.columns) else np.nan
        out[(day, int(mod))] = dict(rsi=rsi[i], e8=e[8][i], e21=e[21][i], e50=e[50][i],
                                    rvol=None if not np.isfinite(r) else float(r))
    hl = {day: dict(zip(g["mod"].astype(int), zip(g["high"], g["low"]))) for day, g in d.groupby("date")}
    return out, hl


def hedge_map(tk):
    h = pl.read_parquet(f"historical/HEDGE{tk}.parquet").select("date", "mod", "hedge_sh_sl")
    out = {}
    for day, mod, v in h.iter_rows():
        out.setdefault(day, {})[int(mod)] = float(v or 0.0)
    return out


def features(tk, day, m, up, band, ctx):
    """dict filter -> True / False / None for a break at (day, m)."""
    sf, hl, hedge, sweeps, walls, am, _tk = ctx
    sgn = 1 if up else -1
    x = sf.get((day, m))
    f = {}
    if x is None:
        f.update(F1_RSI=None, F2_EMA=None, F7_RVOL=None)
    else:
        f["F1_RSI"] = (x["rsi"] >= 70) if up else (x["rsi"] <= 30)
        f["F2_EMA"] = (x["e8"] > x["e21"] > x["e50"]) if up else (x["e8"] < x["e21"] < x["e50"])
        f["F7_RVOL"] = None if x["rvol"] is None else x["rvol"] >= 2.0
    hd = hedge.get(day)
    f["F3_HEDGE"] = None if hd is None else (sgn * sum(hd.get(k, 0.0) for k in range(m - 4, m + 1)) > 0)
    w, a = walls.get((tk, day)), am.get((tk, day))
    px = (hl.get(day) or {}).get(m)
    if w is None or a is None or px is None:
        f["F4_WALL"] = None
    else:
        S = a[1]
        price = _CLOSE[(tk, day, m)]
        f["F4_WALL"] = any(-0.25 * S <= sgn * (K - price) <= 0.5 * S for K in w)
    sw = sweeps.get((tk, day))
    f["F5_SWEEP"] = None if sw is None else (sgn * sum(sw.get(k, 0.0) for k in range(m - 9, m + 1)) > 0)
    f["F6_MIDDAY"] = 11 * 60 + 30 <= m <= 13 * 60 + 59
    lo, hi = band
    day_hl = hl.get(day) or {}
    if up:
        f["F8_TWOSIDED"] = any(day_hl[k][1] <= lo for k in day_hl if S1.FIRST <= k < m)
    else:
        f["F8_TWOSIDED"] = any(day_hl[k][0] >= hi for k in day_hl if S1.FIRST <= k < m)
    return f


_CLOSE = {}                 # (tk, date, mod) -> 1m close


# ================================================================== walks
def precompute(table):
    """{key: {fill: (pnl, exit_mod)}} for every candidate in the table."""
    pol = SC.policy_for(screen_rule("SPY", "CALL", 50))
    out = {}
    for key, c in table.items():
        out[key] = {f: SC.simulate(c[2], pol, SC.DEFAULT_EOD, fill=f)[:2] for f in FILLS}
    return out


def fast_walk(evs, pre, fill):
    """evs: [(tk, date, mod, buy, key)] -> [(date, pnl)] with the sequential guard
    per ticker, exactly as sim_core.walk with no cap / skip / after."""
    out = []
    for tk in TICKERS:
        cur, busy = None, -1
        for _t, day, m, _b, key in sorted((e for e in evs if e[0] == tk), key=lambda e: (e[1], e[2])):
            if day != cur:
                cur, busy = day, -1
            if m < busy:
                continue
            pnl, xm = pre[key][fill]
            out.append((day, pnl))
            busy = xm
    return out


def events(brk, table, feats, arm, filt=None, state=None):
    out = []
    for (tk, d), evs in brk.items():
        buy = d if arm == "BREAKOUT" else ("PUT" if d == "CALL" else "CALL")
        for day, m in evs:
            if day > END:
                continue
            key = (tk, buy, day, m)
            if key not in table:
                continue
            if filt is not None:
                v = feats.get((tk, d, day, m), {}).get(filt)
                if v is None or v != state:
                    continue
            out.append((tk, day, m, buy, key))
    return out


def policies():
    ps = [(arm, None, None) for arm in ("BREAKOUT", "FADE")]
    ps += [(arm, f, s) for arm in ("BREAKOUT", "FADE") for f in FILTERS for s in (True, False)]
    return ps


def slice_of(d):
    from check_config_walkforward import _slice_idx
    return _slice_idx(d)


def walk_forward(results):
    """results: {policy: [(date, pnl)]} -> (chained rows, picks per slice)."""
    chained, picks = [], []
    for k in range(1, 6):
        best, bm = None, -1e9
        for p, rows in results.items():
            tr = [r for r in rows if (slice_of(r[0]) is not None and slice_of(r[0]) < k)]
            if len(tr) < MIN_TRADES or len({r[0] for r in tr}) < MIN_DAYS:
                continue
            mu = float(np.mean([r[1] for r in tr]))
            if mu > bm:
                best, bm = p, mu
        picks.append((k + 1, best, bm))
        if best is not None:
            chained += [(r[0], r[1], k + 1) for r in results[best] if slice_of(r[0]) == k]
    return chained, picks


def run_all(brk, table, feats, pre, fill):
    return {p: fast_walk(events(brk, table, feats, *p), pre, fill) for p in policies()}


def label(p):
    arm, f, s = p
    return arm if f is None else f"{arm} {f}={'T' if s else 'F'}"


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    am = am_straddles()
    if a.build:
        build_walls(am)
        build_sweeps()
        return
    sweeps = pickle.load(open(SWEEP_CACHE, "rb"))
    walls = pickle.load(open(WALL_CACHE, "rb"))
    bd, B = S1.bands()
    real = S1.breaks(bd, B)
    rng = np.random.default_rng(S1.SEED)                 # stage 1's placebo bands, exactly
    perms = [{tk: derangement(sorted(d for (t, d) in bd if t == tk), rng) for tk in TICKERS}
             for _ in range(S1.N_PLACEBO)]
    pbrk = [S1.breaks(bd, B, perm=p) for p in perms]
    with open(S1.CACHE, "rb") as f:
        table = pickle.load(f)["table"]

    # ---- M1: same breaks as stage 1 (reproduced from the same code + seed)
    n_real = sum(len(v) for v in real.values())
    print(f"  M1 breaks reproduced from stage 1: {n_real} (stage 1 logged 936): "
          f"{'PASS' if n_real == 936 else 'FAIL'}")

    # ---- features
    ctx_t = {}
    for tk in TICKERS:
        sf, hl = series_features(tk)
        g = bars_full(tk)
        for day, mod, c in g.select("date", "mod", "close").iter_rows():
            _CLOSE[(tk, day, int(mod))] = c
        ctx_t[tk] = (sf, hl, hedge_map(tk), sweeps, walls, am, tk)

    def feats_for(brk, perm=None):
        out = {}
        for (tk, d), evs in brk.items():
            for day, m in evs:
                A, up_, dn_ = bd[(tk, day)][0.65]
                if perm is not None:
                    src = bd[(tk, perm[tk][day])][0.65]
                    up_, dn_ = src[1], src[2]
                band = (A * (1 - dn_ / 100), A * (1 + up_ / 100))
                out[(tk, d, day, m)] = features(tk, day, m, d == "CALL", band, ctx_t[tk])
        return out

    feats = feats_for(real)
    # ---- M3 coverage and direct rechecks
    ok3 = True
    for f in FILTERS:
        vals = [v[f] for (tk, d, day, m), v in feats.items() if day <= END]
        cov = np.mean([x is not None for x in vals])
        tr = np.mean([x for x in vals if x is not None]) if any(x is not None for x in vals) else float("nan")
        print(f"  M3 {f:12} defined {cov:6.1%}  true {tr:6.1%}")
        ok3 &= cov >= 0.80
    bad = sum(v["F6_MIDDAY"] != (690 <= m <= 839) for (tk, d, day, m), v in feats.items())
    print(f"  M3 F6 direct recheck: {'PASS' if bad == 0 else 'FAIL'};  coverage >= 80% all: {'PASS' if ok3 else 'FAIL'}")
    ok3 &= bad == 0

    # ---- precompute P&L
    print("  precomputing trade outcomes ...", flush=True)
    pre = precompute(table)

    # ---- M2: fast walk == sim_core.walk on unfiltered BREAKOUT
    ev = events(real, table, feats, "BREAKOUT")
    fw = sorted(fast_walk(ev, pre, "bot"))
    sw = sorted(r for v in S1.walk_arm(table, {k: [(d, m) for d, m in v if d <= END] for k, v in real.items()}).values()
                for r in v)
    m2 = len(fw) == len(sw) and all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-12 for a, b in zip(fw, sw))
    print(f"  M2 fast walk == sim_core.walk ({len(fw)} trades): {'PASS' if m2 else 'FAIL'}")

    # ---- M4: planted filter
    res_bot = run_all(real, table, feats, pre, "bot")
    won = {k: dict(v) for k, v in feats.items()}
    for (tk, d, day, m), v in won.items():
        key = (tk, d, day, m)
        won[key]["F1_RSI"] = (key in table and pre[key]["bot"][0] > 0)
    rp = run_all(real, table, won, pre, "bot")
    ch, pk = walk_forward(rp)
    planted = rp[("BREAKOUT", "F1_RSI", True)]

    def eligible(k):          # k = the scored slice number (2..6)
        tr = [r for r in planted if slice_of(r[0]) is not None and slice_of(r[0]) < k - 1]
        return len(tr) >= MIN_TRADES and len({r[0] for r in tr}) >= MIN_DAYS
    elig = [k for k, _p, _m in pk if eligible(k)]
    m4 = (len(elig) >= 3 and all(p == ("BREAKOUT", "F1_RSI", True) for k, p, _m in pk if k in elig)
          and np.mean([r[1] for r in ch]) > 0.5)
    print(f"  M4 planted 'won' filter selected in all {len(elig)} slices where eligible, chained "
          f"{np.mean([r[1] for r in ch]) * 100:+.0f}%: {'PASS' if m4 else 'FAIL'}")
    if not (n_real == 936 and ok3 and m2 and m4):
        sys.exit("machinery checks failed -- not scoring")

    # ---- the 34-policy table (descriptive)
    print("\n  ALL 34 POLICIES (fill=bot) -- a SEARCH; read the walk-forward below, not this table")
    print(f"    {'policy':30} {'n':>5} {'days':>5} {'IS':>8} {'OOS':>8}  win")
    for p in policies():
        rows = res_bot[p]
        if not rows:
            print(f"    {label(p):30} none"); continue
        s = SC.stat(rows)
        print(f"    {label(p):30} {s['n']:5} {s['nd']:5} {s['is_'] * 100:+7.1f}% {s['oos'] * 100:+7.1f}%  {s['win']:.2f}")

    # ---- walk-forward
    ch, pk = walk_forward(res_bot)
    mu = float(np.mean([r[1] for r in ch])) if ch else float("nan")
    per = {k: [r[1] for r in ch if r[2] == k] for k in range(2, 7)}
    print("\n  WALK-FORWARD SELECTION (fill=bot)")
    for k, p, m_ in pk:
        v = per.get(k) or []
        print(f"    slice {k}: picked {label(p) if p else '(none eligible)':30} (train mean {m_ * 100:+.1f}%)  "
              f"-> slice {k}: n={len(v):3}  mean {np.mean(v) * 100 if v else float('nan'):+.1f}%")
    pos = sum(1 for k in per if per[k] and np.mean(per[k]) > 0)
    print(f"    CHAINED: n={len(ch)}  mean {mu * 100:+.1f}%  slices positive {pos}/5")

    print("  placebo selections ...", flush=True)
    pm = []
    for p in perms:
        pb = S1.breaks(bd, B, perm=p)
        pf = feats_for(pb, perm=p)
        c, _ = walk_forward(run_all(pb, table, pf, pre, "bot"))
        pm.append(float(np.mean([r[1] for r in c])) if c else float("nan"))
    p95 = float(np.nanquantile(pm, 0.95))
    fills = {}
    for f in ("mid", "worst"):
        c, _ = walk_forward(run_all(real, table, feats, pre, f))
        fills[f] = float(np.mean([r[1] for r in c])) if c else float("nan")
    crit = dict(W1=mu > 0, W2=mu > p95, W3=pos >= 3, W4=all(v > 0 for v in fills.values()))
    print(f"    placebo chained means: median {np.nanmedian(pm) * 100:+.1f}%  p95 {p95 * 100:+.1f}%")
    print("    fills: " + "  ".join(f"{f} {v * 100:+.1f}%" for f, v in fills.items()))
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> {'PASS' if all(crit.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
