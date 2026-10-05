# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scikit-learn>=1.4"]
# ///
"""
check_ic_screen.py
==================
A SINGLE-FACTOR INFORMATION-COEFFICIENT SCREEN of every input idea this project
has had, then an L1 (LASSO) combination that picks how many to keep. User,
2026-10-03: "trigger quality can't be useless, and good and bad inputs were
being blended without knowing which is which" -- so measure each input on its
own first, then let a penalised model choose, walk-forward.

THE TRIGGER (one with a known conditional edge): the book's own flow trigger,
EMA(5) crossover of cumulative net premium, every crossing that clears its
trailing-60-day p50 gate. SPY, QQQ, IWM; CALL and PUT; hours 9-14, minute
>= 09:35. One row per trigger (not per sequential trade).

TARGETS
    PRIMARY    r_30 = s x (close(m+30) - close(m)) / that day's 09:35 ATM
               straddle (s = +1 CALL, -1 PUT). Exit-free, low variance.
    SECONDARY  option ROE of the trigger's own ATM 0-1DTE contract, book exit
               (trail50, fill "bot" + cushion), simulated standalone, deployed
               window only (check_implied_move_gate's cached candidates).
               Reported, not scored.

WINDOWS  P = 2023-10-12..2024-08-19, the SPENT pre-sample (PRESAMPLE_PLAN 6a):
         labelled, TRAINING ONLY, never scored. Slices 1-6 (METHODOLOGY 8),
         2024-08-20..2026-08-21. Inference: bootstrap over whole CALENDAR days
         (all tickers move together; forward windows overlap, METHODOLOGY 2a).

THE 32 INPUTS at trigger minute m -- "s x" = signed toward the trade
  trigger / flow
    FLOWPCT   log(|cum flow| / its p50 threshold)  -- how far past the gate
    URGENCY   s x net flow(m) / typical |net flow| per minute (prior 20 days)
    CUMALIGN  s x cum flow(m) / the same typical  (check_flow_align's idea)
    SWEEP10   s x net ask-side single-leg ISO premium, m-9..m (big_sweep_lead)
    AGGR5     s x lake net premium (aggressor-signed), m-4..m (aggressor_imb.)
    HEDGE5    s x dealer share demand hedge_sh_sl, m-4..m (hedge_signal)
    HEDGE01   s x the 0-1DTE part, hedge_sh_d01, m-4..m
    MULTI     multi-leg share of option volume, m-4..m (opra_multileg)
  price / technical
    STRETCH   s x (close(m) - close(09:35)) / straddle
    VWAP      s x (close(m) - session VWAP) / straddle            (check_vwap)
    RVOL      volume(m) / median of minute m over the prior 20 sessions
    MOM15     s x (close(m) - close(m-15)) / straddle
    RSI       s x (RSI14 of 1m closes - 50)
    EMASTACK  s x (+1 if EMA 8>21>50 on 1m, -1 if reversed, else 0)
    GAP       s x (09:30 close - prior session close) / straddle
    POCDIST   s x (close(m) - prior session POC) / straddle        (check_amt)
    AMTLOC    s x (+1 09:30 above prior VAH, -1 below VAL, 0 inside)  (amt_open)
  dealer / options
    WALL      0-1DTE wall distance AHEAD in straddles, cap 3 (gamma_walls)
    NETGEX    prior-day net GEX                     (the GEX regime gate)
    DEX       s x prior-day net delta exposure      (check_dex)
    CHARM     s x prior-day net charm               (check_charm_vanna)
    VANNA     s x prior-day net vanna
    IVLVL     prior-day ATM IV, nearest 1-7 DTE      (ivr_termstructure)
    TERM      prior-day IV(25-35 DTE) - IV(1-7 DTE)
  regime / context
    VIX       prior VIX close                       (VIX overlay)
    TREND     s x (SMA20 - SMA50) / SMA50 of daily closes to the prior day
    RV20      20-day realised vol of daily returns, to the prior day
    VOLRATIO  prior-day RTH volume / its trailing 20-day median  (vol regime)
    AMP       amp count 0-3 (NEGATIVE GEX + LOWVOL + CHOP)        (amp gates)
    MACRO     CPI / PCE / NFP / FOMC day (calendar listed from 2024-08 only)
    TIME      the fixed time-of-day curve of check_fuzzy_sizing
    DIR       1 CALL, 0 PUT -- the direction main effect; without it a
              signed input can win by proxying "calls drift up"
  NOT INCLUDED: MBO / CME order-book features (lake/mbo holds 58 scattered
  NQ / RTY days -- cannot cover the window); the gamma flip (logged forward
  from 2026-09-24 only); RR skew (no cached history).

MEMBERSHIP (as in check_fuzzy_sizing): each input -> its percentile rank
among that ticker's trigger values from the prior 250 sessions (strictly
earlier dates), >= 100 values else 0.5; missing = 0.5. TIME and DIR as-is.

PART 1 -- SINGLE-FACTOR IC SCREEN (deployed slices, r_30)
    IC = Spearman(membership, r_30). Per slice, pooled, and P (labelled).
    p = two-sided day-block bootstrap (1,000). An input PASSES THE SCREEN if
    Benjamini-Hochberg q < 0.10 across the 32 AND its IC has the pooled sign
    in >= 5 of 6 slices. (A screen, not a verdict: 32 tests at 5% would hand
    ~1.6 false positives to an uncorrected read.)

PART 2 -- LASSO, walk-forward
    X = membership - 0.5; y = r_30 winsorised at the training 1st/99th pct.
    For slice k: train on P + slices < k; the penalty alpha is chosen by
    leave-one-segment-out CV inside that training set (segments = P's two
    halves and each earlier slice); fit; predict slice k. Chain six slices.
    L1  chained Spearman(prediction, r_30) > 0, day-block 95% CI lower > 0
    L2  > p95 of 20 placebos (input rows permuted within ticker, y kept)
    L3  > 0 in >= 4 of 6 slices
    PASS = L1-L3.
    HOW MANY INPUTS (reported): the number kept at the CV alpha per slice,
    and the chained IC when the model is capped at 1, 2, 3, 5, 8, 13 or all
    inputs (the lasso path's last point within each cap).
    Reported: coefficients per slice; top / bottom third of predictions
    (cut on training predictions) mean r_30, against the ~+0.2 S an ATM
    option needs; the secondary ROE IC of the chained prediction.

MACHINERY (before scoring)
    M1  causality: every rank uses earlier dates (asserted); every prior-day
        input reads a date < the trigger's
    M2  a planted input (= r_30 > 0) gets the largest IC in Part 1 and the
        largest coefficient in every Lasso slice; constant inputs keep zero
        coefficients (prediction constant)
    M3  coverage per input reported; any input < 50% defined is flagged

RUN LOG
  2026-10-03, three aborted starts (no scoring reached): a value-area index
    overrun in profile(); then `seg(day) or -1` dropped slice 1 (index 0 is
    falsy) from the deployed set, and sklearn 1.9 renamed LassoCV's n_alphas.
    All fixed before any result was printed.
  2026-10-03 run: 58,575 triggers (P 14,084 training-only, deployed 44,491).
    M2 PASS (planted IC +0.866, dominant every slice; constant -> no coefs).
    PART 1: 0 of 32 pass the screen. |IC| <= 0.031 for every input. Best:
      DIR +0.031 (calls beat puts, 6/6 slices, P +0.025 -- the window's
      upward drift, p 0.042, not BH-significant), POCDIST +0.022 (5/6, P
      +0.038), GAP +0.022, VWAP +0.022 (5/6), CUMALIGN +0.021, STRETCH +0.021
      -- the same "with the move / above value" family as every test today,
      all CIs spanning 0. VOLRATIO p 0.036 at IC -0.004 (negligible).
    PART 2 FAIL (L1, L3). CV chose ZERO inputs in 5 of 6 slices (STRETCH
      alone in slice 5): the penalised model's own verdict is that nothing
      generalises. Chained IC +0.005, CI [-0.014, +0.025]. L2 passed only
      because a near-constant prediction beat near-constant placebos.
    HOW MANY: forced caps peak at 5 inputs, chained IC +0.012 -- no count
      makes it useful. SECONDARY (option ROE): IC -0.004; top third -14.6%
      vs bottom -12.8%.

Usage:
  python check_ic_screen.py
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                                   # noqa: E402
import check_implied_breakout_filters as BF             # noqa: E402
import check_fuzzy_sizing as FZ                         # noqa: E402
import macro_calendar as MC                             # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
P_START = dt.date(2023, 10, 12)
END = BF.END
SIGNED = ["URGENCY", "CUMALIGN", "SWEEP10", "AGGR5", "HEDGE5", "HEDGE01", "STRETCH", "VWAP", "MOM15",
          "RSI", "EMASTACK", "GAP", "POCDIST", "AMTLOC", "DEX", "CHARM", "VANNA", "TREND"]
UNSIGNED = ["FLOWPCT", "MULTI", "RVOL", "WALL", "NETGEX", "IVLVL", "TERM", "VIX", "RV20", "VOLRATIO",
            "AMP", "MACRO"]
RAW = ["TIME", "DIR"]
FEATS = ["FLOWPCT", "URGENCY", "CUMALIGN", "SWEEP10", "AGGR5", "HEDGE5", "HEDGE01", "MULTI",
         "STRETCH", "VWAP", "RVOL", "MOM15", "RSI", "EMASTACK", "GAP", "POCDIST", "AMTLOC",
         "WALL", "NETGEX", "DEX", "CHARM", "VANNA", "IVLVL", "TERM",
         "VIX", "TREND", "RV20", "VOLRATIO", "AMP", "MACRO", "TIME", "DIR"]
RANK_WIN, RANK_MIN = 250, 100
N_BOOT, N_PLACEBO = 1000, 20
CAPS = (1, 2, 3, 5, 8, 13, len(FEATS))
SEED = 20261008


# ================================================================== helpers
def latest_before(series, day):
    """series: sorted [(date, value)] -> value of the last date < day."""
    i = bisect.bisect_left(series, (day,)) - 1
    return series[i][1] if i >= 0 else None


def seg(day):
    if day < P_START or day > END:
        return None
    if day < SC.DEPLOYED_START:
        return -1
    return BF.slice_of(day)


def is_dep(day):
    """In a deployed slice? Never write `seg(day) or <default>`: slice 1's
    index is 0, which is falsy, and it silently fell out of the first run."""
    s = seg(day)
    return s is not None and s >= 0


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if len(rx) < 3 or rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def profile(g):
    """(poc, vah, val) of one session's 1m bars: 0.05%-of-price bins, 70% VA."""
    px = g["close"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    if len(px) == 0 or v.sum() <= 0:
        return None
    w = px.mean() * 0.0005
    b = np.round(px / w).astype(int)
    vol = pd.Series(v).groupby(b).sum().sort_index()
    ks, vs = vol.index.to_numpy(), vol.to_numpy()
    i = int(vs.argmax())
    lo, hi, tot = i, i, vs[i]
    target = 0.7 * np.nansum(vs)
    while tot < target and (lo > 0 or hi < len(vs) - 1):
        can_up, can_dn = hi < len(vs) - 1, lo > 0
        up = can_up and (not can_dn or vs[hi + 1] >= vs[lo - 1])
        if up:
            hi += 1; tot += vs[hi]
        else:
            lo -= 1; tot += vs[lo]
    return ks[i] * w, ks[hi] * w, ks[lo] * w


# ================================================================== data per ticker
class TickerData:
    def __init__(self, tk, D, am, walls, sweeps, vix, ivs):
        from check_config_walkforward import _flow_for
        self.tk = tk
        bars = BF.bars_full(tk).to_pandas()
        bars["date"] = pd.to_datetime(bars["date"]).dt.date
        self.days = {d: g.set_index("mod") for d, g in bars.groupby("date")}
        self.cal = sorted(self.days)
        self.px = FZ.price_tables(tk)                     # (date, mod) -> (close, vwap)
        self.sf, _hl = BF.series_features(tk)             # rsi, e8/21/50, rvol
        f = _flow_for(D, [tk])
        f = f[f["underlying_symbol"] == tk].copy()
        ts = pd.to_datetime(f["minute_et"])
        f["date"], f["mod"] = ts.dt.date, (ts.dt.hour * 60 + ts.dt.minute).astype(int)
        self.net = dict(zip(zip(f["date"], f["mod"]), f["net_flow_1m"].astype(float)))
        self.cum = dict(zip(zip(f["date"], f["mod"]), f["cum_flow"].astype(float)))
        med = f[(f["mod"] >= 570) & (f["mod"] <= 959)].groupby("date")["net_flow_1m"].apply(
            lambda x: float(np.median(np.abs(x))))
        typ = med.shift(1).rolling(20, min_periods=20).mean()
        self.ftyp = {d: v for d, v in typ.items() if np.isfinite(v) and v > 0}
        self.trigs = D.triggers_for(f, tk)
        D.annotate_flow_pct(self.trigs, 60)
        h = pd.read_parquet(f"historical/HEDGE{tk}.parquet",
                            columns=["date", "mod", "hedge_sh_sl", "hedge_sh_d01", "net_prem_lake", "multi_frac"])
        h["date"] = pd.to_datetime(h["date"]).dt.date
        self.hedge = {d: g.set_index("mod") for d, g in h.groupby("date")}
        self.am, self.walls, self.sweeps = am, walls, sweeps
        # daily
        closes = pd.Series({d: g["close"].iloc[-1] for d, g in self.days.items()}).sort_index()
        vols = pd.Series({d: g["volume"].sum() for d, g in self.days.items()}).sort_index()
        sma20, sma50 = closes.rolling(20).mean(), closes.rolling(50).mean()
        rv20 = closes.pct_change().rolling(20).std() * math.sqrt(252)
        vr = vols / vols.rolling(20).median()
        self.daily = {d: dict(close=closes[d], trend=(sma20[d] - sma50[d]) / sma50[d] if np.isfinite(sma50[d]) else None,
                              rv20=rv20[d] if np.isfinite(rv20[d]) else None,
                              vr=vr[d] if np.isfinite(vr[d]) else None)
                      for d in closes.index}
        self.prof = {d: profile(g) for d, g in self.days.items()}
        gx = pd.read_parquet(f"historical/GEX{tk}.parquet")
        gx["date"] = pd.to_datetime(gx["date"]).dt.date
        gx = gx.sort_values("date")
        self.gex = {c: [(d, float(v)) for d, v in zip(gx["date"], gx[c]) if pd.notna(v)]
                    for c in ("net_gex", "net_dex", "net_charm", "net_vanna")}
        self.vix = vix
        self.iv = ivs.get(tk, {})
        self.amp = D.load_amp_score(SC.HIST, tk)

    def prev_day(self, day):
        i = bisect.bisect_left(self.cal, day) - 1
        return self.cal[i] if i >= 0 else None

    def features(self, d, day, m):
        s = 1 if d == "CALL" else -1
        F = dict.fromkeys(FEATS)
        st = self.am.get((self.tk, day))
        S = st[1] if st else None
        g = self.days.get(day)
        c = g["close"].get(m) if g is not None else None
        pd_ = self.prev_day(day)
        # trigger / flow
        typ = self.ftyp.get(day)
        if typ:
            if (day, m) in self.net:
                F["URGENCY"] = s * self.net[(day, m)] / typ
            if (day, m) in self.cum:
                F["CUMALIGN"] = s * self.cum[(day, m)] / typ
        sw = self.sweeps.get((self.tk, day))
        if sw is not None:
            F["SWEEP10"] = s * sum(sw.get(k, 0.0) for k in range(m - 9, m + 1))
        h = self.hedge.get(day)
        if h is not None:
            w = h.loc[(h.index >= m - 4) & (h.index <= m)]
            F["AGGR5"] = s * float(w["net_prem_lake"].sum())
            F["HEDGE5"] = s * float(w["hedge_sh_sl"].sum())
            F["HEDGE01"] = s * float(w["hedge_sh_d01"].sum())
            F["MULTI"] = float(w["multi_frac"].mean()) if len(w) else None
        # price / technical
        x = self.sf.get((day, m))
        if x is not None:
            F["RVOL"] = x["rvol"]
            F["RSI"] = s * (x["rsi"] - 50)
            F["EMASTACK"] = s * (1 if x["e8"] > x["e21"] > x["e50"] else -1 if x["e8"] < x["e21"] < x["e50"] else 0)
        if c is not None and S:
            a = g["close"].get(575)
            if a is not None:
                F["STRETCH"] = s * (c - a) / S
            p = self.px.get((day, m))
            if p:
                F["VWAP"] = s * (c - p[1]) / S
            c15 = g["close"].get(m - 15)
            if c15 is not None:
                F["MOM15"] = s * (c - c15) / S
            o = g["close"].get(570)
            if pd_ is not None and o is not None:
                F["GAP"] = s * (o - self.daily[pd_]["close"]) / S
            pr = self.prof.get(pd_) if pd_ else None
            if pr:
                F["POCDIST"] = s * (c - pr[0]) / S
                if o is not None:
                    F["AMTLOC"] = s * (1 if o > pr[1] else -1 if o < pr[2] else 0)
            wl = self.walls.get((self.tk, day))
            if wl:
                ahead = [s * (K - c) for K in wl if s * (K - c) >= 0]
                F["WALL"] = min(min(ahead) / S, 3.0) if ahead else 3.0
        # dealer / options (prior day)
        for key, col, sg in (("NETGEX", "net_gex", False), ("DEX", "net_dex", True),
                             ("CHARM", "net_charm", True), ("VANNA", "net_vanna", True)):
            v = latest_before(self.gex[col], day)
            if v is not None:
                F[key] = s * v if sg else v
        iv = self.iv.get(pd_) if pd_ else None
        if iv:
            F["IVLVL"], F["TERM"] = iv
        # regime / context
        v = latest_before(self.vix, day)
        F["VIX"] = v
        if pd_ is not None:
            dd = self.daily[pd_]
            F["TREND"] = s * dd["trend"] if dd["trend"] is not None else None
            F["RV20"], F["VOLRATIO"] = dd["rv20"], dd["vr"]
        F["AMP"] = self.amp.get(day)
        F["MACRO"] = 1.0 if (MC.is_macro_am_day(day) or MC.is_fomc_day(day)) else 0.0
        F["TIME"] = FZ.time_curve(m)
        F["DIR"] = 1.0 if s > 0 else 0.0
        # FLOWPCT is filled by the caller (needs the trigger record)
        return F


def load_ivs():
    """{tk: {date: (iv 1-7 DTE, iv 25-35 minus iv 1-7)}} from _ivs_cache (last bucket of the day)."""
    out = {}
    for tk in TICKERS:
        p = os.path.join("_ivs_cache", f"{tk}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"]).dt.date
        last = d.sort_values("mod15").groupby(["date", "dte"]).tail(1)
        res = {}
        for day, g in last.groupby("date"):
            f = g[(g["dte"] >= 1) & (g["dte"] <= 7)].sort_values("dte")
            b = g[(g["dte"] >= 25) & (g["dte"] <= 35)]
            if len(f):
                fr = float(f["iv"].iloc[0])
                res[day] = (fr, float(b["iv"].median()) - fr if len(b) else None)
        out[tk] = res
    return out


# ================================================================== memberships
def memberships(raw):
    out = {k: {} for k in raw}
    for tk in TICKERS:
        keys = [k for k in raw if k[0] == tk]
        days = sorted({k[2] for k in keys})
        on_day = {}
        for k in keys:
            on_day.setdefault(k[2], []).append(k)
        for f in SIGNED + UNSIGNED:
            vals = {d: [raw[k][f] for k in on_day[d] if raw[k][f] is not None] for d in days}
            for i, d in enumerate(days):
                window = days[max(0, i - RANK_WIN):i]
                assert all(w < d for w in window)
                ref = sorted(v for w in window for v in vals[w])
                for k in on_day[d]:
                    v = raw[k][f]
                    if v is None or len(ref) < RANK_MIN:
                        out[k][f] = 0.5
                    else:
                        out[k][f] = (bisect.bisect_left(ref, v) + bisect.bisect_right(ref, v)) / 2 / len(ref)
        for k in keys:
            for f in RAW:
                out[k][f] = raw[k][f]
    return out


# ================================================================== inference
def day_boot_ic(x, y, days, n=N_BOOT, seed=SEED):
    """IC and two-sided bootstrap p over CALENDAR days; -> (ic, lo95, hi95, p)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ud = sorted(set(days))
    idx = {d: [] for d in ud}
    for i, d in enumerate(days):
        idx[d].append(i)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    ic = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else 0.0
    arr = [np.array(idx[d]) for d in ud]
    bs = []
    for _ in range(n):
        pick = np.concatenate([arr[j] for j in rng.integers(0, len(arr), len(arr))])
        a, b = rx[pick], ry[pick]          # ranks reused: a monotone proxy, fast
        bs.append(np.corrcoef(a, b)[0, 1] if a.std() and b.std() else 0.0)
    bs = np.array(bs)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return ic, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), float(min(1.0, p))


def bh(pvals, q=0.10):
    """Benjamini-Hochberg: -> list of booleans, rejected at FDR q."""
    m = len(pvals)
    order = np.argsort(pvals)
    thr = [q * (i + 1) / m for i in range(m)]
    k = max((i for i in range(m) if pvals[order[i]] <= thr[i]), default=-1)
    rej = [False] * m
    for i in range(k + 1):
        rej[order[i]] = True
    return rej


# ================================================================== lasso
def lasso_walk(keys, X, y, cap=None):
    """-> (chained rows [(i, pred, third, slice)], coef per slice, nnz per slice)."""
    from sklearn.linear_model import Lasso, LassoCV, lars_path
    segs = np.array([seg(k[2]) for k in keys])
    days = np.array([k[2] for k in keys])
    pmid = sorted(set(days[segs == -1]))
    pmid = pmid[len(pmid) // 2] if pmid else None
    group = np.where(segs == -1, np.where(days < pmid, -2, -1), segs) if pmid else segs
    rows, coefs, nnz = [], {}, {}
    for k in range(6):
        tr = np.where((segs == -1) | ((segs >= 0) & (segs < k)))[0]
        te = np.where(segs == k)[0]
        if len(tr) < 200 or len(te) == 0:
            continue
        lo, hi = np.quantile(y[tr], [0.01, 0.99])
        yt = np.clip(y[tr], lo, hi)
        Xt = X[tr]
        if cap is None:
            g = group[tr]
            ug = sorted(set(g))
            cv = [(np.where(g != u)[0], np.where(g == u)[0]) for u in ug]
            m = LassoCV(cv=cv, alphas=40, max_iter=5000).fit(Xt, yt)
            coef, icpt = m.coef_, m.intercept_
        else:
            alphas, _act, path = lars_path(Xt - Xt.mean(0), yt - yt.mean(), method="lasso")
            j = max(i for i in range(path.shape[1]) if np.count_nonzero(path[:, i]) <= cap)
            coef = path[:, j]
            icpt = yt.mean() - Xt.mean(0) @ coef
        coefs[k + 1], nnz[k + 1] = coef, int(np.count_nonzero(coef))
        ptr = Xt @ coef + icpt
        q1, q2 = np.quantile(ptr, [1 / 3, 2 / 3])
        for i in te:
            p = float(X[i] @ coef + icpt)
            rows.append((i, p, 0 if p < q1 else 2 if p > q2 else 1, k + 1))
    return rows, coefs, nnz


def chained_ic(rows, y, keys):
    if not rows:
        return float("nan"), float("nan"), float("nan")
    xs = [r[1] for r in rows]
    ys = [y[r[0]] for r in rows]
    ic, lo, hi, _p = day_boot_ic(xs, ys, [keys[r[0]][2] for r in rows])
    return ic, lo, hi


# ================================================================== dataset
DATASET_CACHE = os.path.join("_ic_cache", "dataset.pkl")


def build_dataset(D):
    """-> (keys, raw, y_r30, tails). ONE builder for check_ic_screen and
    check_ic_tail: same triggers, same inputs. tails[key] = {mfe60, mae60} in
    straddle units (best / worst 1m high-low over m+1..m+60, signed toward the
    trade) when the full 60 minutes exist. Cached in _ic_cache/ (git-ignored)."""
    if os.path.exists(DATASET_CACHE):
        with open(DATASET_CACHE, "rb") as f:
            return pickle.load(f)
    am = BF.am_straddles()
    walls = pickle.load(open(BF.WALL_CACHE, "rb"))
    sweeps = pickle.load(open(BF.SWEEP_CACHE, "rb"))
    vix = sorted((dt.date.fromisoformat(k), v) for k, v in json.load(open("_implied_move_vix.json")).items())
    ivs = load_ivs()
    keys, raw, y, tails = [], {}, {}, {}
    for tk in TICKERS:
        print(f"  {tk}: loading ...", flush=True)
        T = TickerData(tk, D, am, walls, sweeps, vix, ivs)
        for t in T.trigs:
            ts = pd.Timestamp(t["ts"])
            day, m = t["date"], ts.hour * 60 + ts.minute
            if seg(day) is None or not (9 <= t["hour"] <= 14) or m < 575:
                continue
            if not (t.get("thr") and t["abs_flow"] >= t["thr"][50]) or t["thr"][50] <= 0:
                continue
            st = am.get((tk, day))
            g = T.days.get(day)
            if not st or g is None or m + 30 > 959 or m not in g.index or (m + 30) not in g.index:
                continue
            s = 1 if t["dir"] == "CALL" else -1
            key = (tk, t["dir"], day, m)
            if key in raw:
                continue
            y[key] = s * (g.at[m + 30, "close"] - g.at[m, "close"]) / st[1]
            F = T.features(t["dir"], day, m)
            F["FLOWPCT"] = math.log(t["abs_flow"] / t["thr"][50])
            raw[key] = F
            keys.append(key)
            if m + 60 <= 959:
                w = g.loc[m + 1:m + 60]
                if len(w) >= 55:
                    c0 = g.at[m, "close"]
                    fav = w["high"].max() if s > 0 else w["low"].min()
                    adv = w["low"].min() if s > 0 else w["high"].max()
                    tails[key] = dict(mfe60=s * (fav - c0) / st[1], mae60=s * (adv - c0) / st[1])
    out = (keys, raw, y, tails)
    os.makedirs(os.path.dirname(DATASET_CACHE), exist_ok=True)
    with open(DATASET_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


# ================================================================== main
def main():
    import directional_flow_backtester as D
    keys, raw, y, _tails = build_dataset(D)
    keys = list(keys)
    keys.sort(key=lambda k: (k[2], k[3], k[0], k[1]))
    print(f"  triggers {len(keys)}: P {sum(seg(k[2]) == -1 for k in keys)}, deployed "
          f"{sum(is_dep(k[2]) for k in keys)}")
    cov = {f: np.mean([raw[k][f] is not None for k in keys]) for f in FEATS}
    print("  M3 coverage: " + "  ".join(f"{f} {v:.0%}" for f, v in cov.items()))
    flagged = [f for f, v in cov.items() if v < 0.5]
    if flagged:
        print(f"     flagged (< 50% defined): {flagged}")
    mem = memberships(raw)
    X = np.array([[mem[k][f] - (0.5 if f in SIGNED + UNSIGNED else 0.0) for f in FEATS] for k in keys])
    yv = np.array([y[k] for k in keys])
    dep = np.array([is_dep(k[2]) for k in keys])

    # ---- M2 planted
    Xp = X.copy()
    Xp[:, 0] = (yv > 0).astype(float) - 0.5
    rows, coefs, _n = lasso_walk(keys, Xp, yv)
    dom = all(int(np.argmax(np.abs(c))) == 0 for c in coefs.values())
    ic0 = spearman(Xp[dep, 0], yv[dep])
    Xc = np.zeros_like(X)
    rc, cc, _n = lasso_walk(keys, Xc, yv)
    zero = all(np.count_nonzero(c) == 0 for c in cc.values())
    print(f"  M2 planted input dominant every slice {dom} (IC {ic0:+.3f}); constant inputs -> zero coefs {zero}: "
          f"{'PASS' if dom and zero else 'FAIL'}")
    if not (dom and zero):
        sys.exit("machinery checks failed -- not scoring")

    # ---- PART 1
    print("\n  PART 1 -- SINGLE-FACTOR IC SCREEN (target r_30, deployed slices; P labelled, not scored)")
    dk = [k for k, d in zip(keys, dep) if d]
    di = np.where(dep)[0]
    pi = np.where(~dep)[0]
    res = []
    for j, f in enumerate(FEATS):
        ic, lo, hi, p = day_boot_ic(X[di, j], yv[di], [k[2] for k in dk])
        per = [spearman(X[[i for i in di if seg(keys[i][2]) == s], j], yv[[i for i in di if seg(keys[i][2]) == s]])
               for s in range(6)]
        same = sum(np.sign(v) == np.sign(ic) for v in per)
        icp = spearman(X[pi, j], yv[pi])
        res.append((f, ic, lo, hi, p, per, same, icp))
    rej = bh([r[4] for r in res])
    print(f"    {'input':9} {'IC':>7} {'95% CI':>17} {'p':>6}  {'slices 1..6':>41}  same  {'P (lab.)':>8}  screen")
    for (f, ic, lo, hi, p, per, same, icp), r in sorted(zip(res, rej), key=lambda z: -abs(z[0][1])):
        ok = r and same >= 5
        print(f"    {f:9} {ic:+7.4f} [{lo:+.4f},{hi:+.4f}] {p:6.3f}  " + " ".join(f"{v:+.3f}" for v in per) +
              f"  {same}/6   {icp:+.4f}   {'PASS' if ok else ''}")
    print(f"    passing the screen: {sum(1 for (f, ic, lo, hi, p, per, same, icp), r in zip(res, rej) if r and same >= 5)} of {len(FEATS)}")

    # ---- PART 2
    print("\n  PART 2 -- LASSO, walk-forward (alpha by leave-one-segment-out CV)")
    rows, coefs, nnz = lasso_walk(keys, X, yv)
    ic, lo, hi = chained_ic(rows, yv, keys)
    per = {s: spearman([r[1] for r in rows if r[3] == s], [yv[r[0]] for r in rows if r[3] == s]) for s in range(1, 7)}
    print("    inputs kept per slice: " + "  ".join(f"{s}:{n}" for s, n in nnz.items()))
    print("    coefficients (nonzero in any slice):")
    used = [j for j in range(len(FEATS)) if any(c[j] != 0 for c in coefs.values())]
    for j in used:
        print(f"      {FEATS[j]:9} " + " ".join(f"{coefs[s][j]:+.4f}" for s in coefs))
    for t in (0, 1, 2):
        v = [yv[r[0]] for r in rows if r[2] == t]
        print(f"    prediction third {['bottom', 'middle', 'top'][t]:6}: n={len(v):6}  mean r_30 {np.mean(v):+.4f} S")
    print(f"    chained IC {ic:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   per slice " +
          " ".join(f"{s}:{v:+.3f}" for s, v in per.items()))
    rng = np.random.default_rng(SEED)
    pl_ = []
    tk_idx = {tk: np.array([i for i, k in enumerate(keys) if k[0] == tk]) for tk in TICKERS}
    for _ in range(N_PLACEBO):
        Xq = X.copy()
        for tk, ix in tk_idx.items():
            Xq[ix] = X[rng.permutation(ix)]
        rq, _c, _n = lasso_walk(keys, Xq, yv)
        pl_.append(spearman([r[1] for r in rq], [yv[r[0]] for r in rq]))
    p95 = float(np.quantile(pl_, 0.95))
    crit = dict(L1=lo > 0, L2=ic > p95, L3=sum(v > 0 for v in per.values()) >= 4)
    print(f"    placebo chained IC median {np.median(pl_):+.4f}, p95 {p95:+.4f}")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()) +
          f"   -> PART 2 {'PASS' if all(crit.values()) else 'FAIL'}")

    print("\n  HOW MANY INPUTS -- chained IC with the model capped (reported)")
    for cap in CAPS:
        rq, _c, nq = lasso_walk(keys, X, yv, cap=cap)
        print(f"    cap {cap:2}: chained IC {spearman([r[1] for r in rq], [yv[r[0]] for r in rq]):+.4f}   "
              f"kept per slice " + " ".join(str(n) for n in nq.values()))

    # ---- secondary: option ROE IC of the chained prediction (deployed, standalone trades)
    import check_implied_move_gate as G
    cands = G.candidates(D)
    pol = SC.policy_for(G.screen_rule("SPY", "CALL", 50))
    roe = {}
    for (tk, d, pct), rr in cands.items():
        if pct != 50:
            continue
        for day, m, payload, _s in rr:
            roe[(tk, d, day, m)] = SC.simulate(payload, pol, SC.DEFAULT_EOD, fill="bot")[0]
    pairs = [(r[1], roe[keys[r[0]]]) for r in rows if keys[r[0]] in roe]
    if pairs:
        print(f"\n  SECONDARY: chained prediction vs option ROE (standalone, n={len(pairs)}): "
              f"IC {spearman([a for a, b in pairs], [b for a, b in pairs]):+.4f}")
        for t in (0, 2):
            v = [roe[keys[r[0]]] for r in rows if r[2] == t and keys[r[0]] in roe]
            print(f"    {['bottom', '', 'top'][t]} third mean ROE {np.mean(v) * 100:+.1f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
