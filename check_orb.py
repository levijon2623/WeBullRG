# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_orb.py
============
OPENING RANGE BREAKOUT (ORB) -- DOES IT HOLD UP? (user, 2026-10-21)

The user shared two TradingView scripts ("ORB Ultimate+", "Liquidity Orbs")
and asked how to test ORB theory. Plan agreed 10-21: stage 1 (plain ORB on the
underlying) and stage 3 (ORB as a gate on our own flow trigger) first; the
CW reclaim port (stage 2) waits for TradingView fidelity screenshots; options
(stage 4) only if something passes on the underlying.

DATA     SPY / QQQ / IWM 1m RTH bars (historical/{T}.parquet), full sessions
         only. S = that day's 09:35 ATM straddle (units, as the morning study).
         Bar mod m = the minute the bar STARTS; price at T = close of bar T-1.
WINDOW   scored 2024-08-20 .. 2026-09-18 (straddle coverage end), halves split
         2025-08-21, 6 equal-count chronological slices. Pre-sample 2023-10-12
         .. 2024-08-19 reported, labelled (spent holdout, extra in-sample).

STAGE 1 -- plain ORB on the underlying (default run)
  OR(L)  high/low of the first L minutes, L in {5, 15, 30}.
  CELLS (16, fixed before the run)
    BRK L stop   first 1m CLOSE beyond ORH/ORL, signal bar starting before
                 12:00; enter at the NEXT bar's open; stop none / opposite
                 OR side ("opp") / OR midpoint ("mid"); else out at 15:55.
    MOM L stop   direction of the OR itself (close at OR end vs 09:30 open),
                 enter at the open of the first bar after the OR; stop none /
                 opp; 15:55.
    ZA 5 opp     Zarattini & Aziz (2023) QQQ 5-minute ORB as I understand the
                 paper: MOM 5 opp plus a 10R profit target.
  Fills: stops/targets fill at the level, or the bar open if it gapped
  through; stop checked before target in the same bar (conservative).
  r = side x (exit - entry) / S.

  NULLS
    N_DIR   direction: sides permuted across days within ticker (keeps the
            long/short mix, so the past years' up-drift is matched); a day
            given the other side scores the MIRROR trade (same entry bar,
            stop/target the same distance the other way). 2,000 permutations.
    N_LEVEL (BRK cells) a FAKE range of the same width, shifted by a random
            offset that still contains the price at the OR's end, run through
            the identical rule. 200 draws/day. Asks: do the real opening
            extremes matter more than any equal-width band? (The trap that
            killed the trigger-level idea: persistence = day direction.)
    Family-wise: each null gives every cell a z; threshold = 95th percentile
    of the MAX z over cells (16 cells for N_DIR, 9 BRK cells for N_LEVEL).
    "excess" = r minus the day's N_DIR expectation (q x long + (1-q) x short).

  PRE-REGISTERED (pooled 3 tickers, scored window)
    EDGE   z_dir > family-wise p95; BRK cells also z_level > its p95;
           excess > 0 in both halves; >= 4 of 6 slices excess > 0.
    TRADE  EDGE and mean r >= +0.03 S (the scalp floor) and the day-block
           bootstrap 95% CI of r has lower bound > 0.
    Reported, unscored: per ticker, bp, win rate, stop-outs, entry times,
    pre-sample.
  MACHINERY
    M0  bars shuffled within each day (bar shapes kept, order destroyed):
        no cell should pass EDGE (one false pass allowed at ~5%).
    M1  +0.05 S planted into BRK 15 opp's real trades must pass EDGE there.

STAGE 3 -- ORB as a gate on the flow trigger (--gate)
  The deployed research book (sim_core.research_rules, deployed_candidates,
  walk with fill "bot", policy_for(rule, TRAIL_PCT)). At each trigger minute,
  ORB state from bars that CLOSED before it: first 1m close beyond OR(15) was
  WITH the trigger's direction / AGAINST it / NONE yet.
  PRE-REGISTERED
    GATE   on the taken trades, mean ROE(with + none) - mean ROE(against) >
           the 95th pct of the same difference with states permuted within
           rule (2,000), > 0 in both halves; AND the book re-walked without
           AGAINST triggers has total ROE >= the full book's in both halves.
    Reported, unscored: OR 5 / 30, "WITH only", per rule.

Usage:  python check_orb.py            # stage 1 (+ M0, M1)
        python check_orb.py --gate     # stage 3 (builds the book, ~5 min)

RUN LOG
  2026-10-21 run 1 (stage 1): 2,103 ticker-days (SPY 717, QQQ 717, IWM 669),
    scored 1,519, pre-sample 584. Vectorised trades = naive bar loop on 200
    SPY days (BRK 15 opp, ZA 5), 0 mismatches. M1 PASS (BRK 15 opp only).
    M0 PASS (no hits on shuffled bars; BRK 15 opp -0.003 S).
    Family-wise p95: z_dir 2.74, z_level 2.37.
    EDGE: none. TRADE: none.
      Every cell's raw mean is mildly positive (+0.014 .. +0.066 S, -1 .. +6
      bp) -- mostly the window's up-drift; no cell beats the permuted-side
      null family-wise. Best z_dir MOM 30 none 2.35 (+0.066 S, halves
      +0.086 / +0.048), BRK 15 opp 2.19.
      Only near-miss: BRK 15 opp z_level 2.52 > 2.37 (real opening extremes
      beat an equal-width fake band, +0.045 vs +0.012 S) and BRK 5 opp 2.38
      -- but direction fails and OOS excess is +0.015. Lead, not a claim.
      ZA 5 (the published QQQ rule): +0.015 S pooled, -0.1 bp; QQQ alone
      +0.029 S. Doesn't replicate as an edge here.
      All 95% CIs of r include 0; the best means are below one 0DTE round trip.
    Pre-sample (labelled): mostly flat to negative (BRK 5 -5.0 bp, MOM 15
      -3.7 bp; BRK 30 none +6.0 bp the exception).
  2026-10-21 run 1 (stage 3): research book 5 rules, 215 taken trades with
    bars (META 59, NVDA 27, SPY 8, QQQ 29, IWM 92). OR(15) PRIMARY:
    with +17.0% (n 93), none +6.1% (13), AGAINST +19.4% (109).
    (with+none) - against = -3.7 pp vs perm p95 +21.4 pp; halves +12.7 /
    -26.0 pp. Book without AGAINST: IS +643.7% vs +555.8% but OOS +1,019.6%
    vs +3,219.2% (sum of trade ROE). GATE FAIL.
    Reported: OR(5) against +28.6% vs with +4.4% (the trigger often fires
    AGAINST the opening break and those trades are no worse -- the flow rules
    catch reversals); OR(30) +0.3 pp. Gating on ORB would roughly cut the OOS
    book by two-thirds.
  VERDICT: plain ORB has no edge on SPY/QQQ/IWM after the drift control, and
    it is not a filter for the flow trigger. Stage 2 (CW reclaim port) and
    stage 4 (options) not warranted by stage 1; CW port only if the user wants
    the confluence grade tested on its own.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TICKERS = ("SPY", "QQQ", "IWM")
DEP_START, SPLIT, END = dt.date(2024, 8, 20), dt.date(2025, 8, 21), dt.date(2026, 9, 18)
OPEN, NB = 570, 390                 # 09:30, bars per session
KX = 955 - OPEN                     # exit at the price at 15:55 = close of bar 384
KCUT = 720 - OPEN                   # BRK signal bar must start before 12:00
LENS = (5, 15, 30)
CELLS = ([("BRK", L, s) for L in LENS for s in ("none", "opp", "mid")]
         + [("MOM", L, s) for L in LENS for s in ("none", "opp")]
         + [("ZA", 5, "opp")])
BRK_IX = [i for i, c in enumerate(CELLS) if c[0] == "BRK"]
PLANT = CELLS.index(("BRK", 15, "opp"))
FLOOR = 0.03
N_PERM, N_LVL, N_BOOT = 2000, 200, 2000
SEED = 20261021


def cname(c):
    return f"{c[0]} {c[1]:>2} {c[2]}"


def hm(m):
    return f"{m // 60}:{m % 60:02d}"


# ------------------------------------------------------------------ data
def load_bars(tk, full=True):
    """{date: (o, h, l, c)} 390-bar RTH arrays. full=True keeps only complete
    sessions (09:30 bar present, runs to 15:59, <= 5 missing bars, ffilled)."""
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= OPEN) & (pl.col("mod") < OPEN + NB))
         .select("date", "mod", "open", "high", "low", "close").to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    out = {}
    for day, g in d.groupby("date", sort=True):
        g = g.drop_duplicates("mod").set_index("mod").reindex(np.arange(OPEN, OPEN + NB))
        have = g["close"].notna()
        if full and (not have.iloc[0] or not have.iloc[-1] or (~have).sum() > 5):
            continue
        if not have.iloc[0]:
            continue
        c = g["close"].ffill().to_numpy(float)
        o = g["open"].fillna(pd.Series(c, index=g.index).shift(1)).to_numpy(float)
        h = g["high"].to_numpy(float)
        lo = g["low"].to_numpy(float)
        h = np.where(np.isnan(h), c, h)
        lo = np.where(np.isnan(lo), c, lo)
        out[day] = (o, h, lo, c)
    return out


def load():
    import check_implied_breakout_filters as BF
    am = BF.am_straddles()
    days = []
    for tk in TICKERS:
        for day, (o, h, lo, c) in load_bars(tk).items():
            st = am.get((tk, day))
            if day > END or not st or not st[1]:
                continue
            days.append(dict(tk=tk, day=day, S=float(st[1]), o=o, h=h, l=lo, c=c))
    days.sort(key=lambda x: (x["day"], x["tk"]))
    return days


def shuffle_day(dd, rng):
    """M0: permute bar order after the first (shapes relative to the prior close kept)."""
    o, h, lo, c = dd["o"], dd["h"], dd["l"], dd["c"]
    pc = np.concatenate([[o[0]], c[:-1]])
    dO, dC, dH, dL = o - pc, c - pc, h - c, lo - c
    p = np.concatenate([[0], 1 + rng.permutation(NB - 1)])
    dO, dC, dH, dL = dO[p], dC[p], dH[p], dL[p]
    c2 = o[0] + np.cumsum(dC)
    pc2 = np.concatenate([[o[0]], c2[:-1]])
    return dict(dd, o=pc2 + dO, c=c2, h=c2 + dH, l=c2 + dL)


# ------------------------------------------------------------------ trades
class Win:
    """Per-day sliding windows: W[k, t] = x[k + t], NaN past the 15:55 exit."""
    def __init__(self, dd):
        self.dd = dd
        pad = np.full(KX, np.nan)
        sw = np.lib.stride_tricks.sliding_window_view
        self.O = sw(np.concatenate([dd["o"][:KX], pad]), KX)[:KX]
        self.H = sw(np.concatenate([dd["h"][:KX], pad]), KX)[:KX]
        self.L = sw(np.concatenate([dd["l"][:KX], pad]), KX)[:KX]

    def run(self, k, side, stop=None, tp=None):
        """Vectorised trades: enter at open of bar k, side +-1, optional stop /
        target levels; out at 15:55. -> side x (exit - entry), price units."""
        k = np.asarray(k, int)
        side = np.asarray(side, float)
        n, big = len(k), KX + 1
        O, H, L = self.O[k], self.H[k], self.L[k]
        e = O[:, 0]
        up = side[:, None] > 0

        def first(mask):
            return np.where(mask.any(1), mask.argmax(1), big)
        with np.errstate(invalid="ignore"):
            ks = first(np.where(up, L <= stop[:, None], H >= stop[:, None])) if stop is not None else np.full(n, big)
            kt = first(np.where(up, H >= tp[:, None], L <= tp[:, None])) if tp is not None else np.full(n, big)
        x = np.full(n, self.dd["c"][KX - 1])
        rows = np.arange(n)
        s_hit = (ks < big) & (ks <= kt)
        t_hit = (kt < big) & ~s_hit
        if s_hit.any():
            oo = O[rows[s_hit], ks[s_hit]]
            x[s_hit] = np.where(side[s_hit] > 0, np.minimum(stop[s_hit], oo), np.maximum(stop[s_hit], oo))
        if t_hit.any():
            oo = O[rows[t_hit], kt[t_hit]]
            x[t_hit] = np.where(side[t_hit] > 0, np.maximum(tp[t_hit], oo), np.minimum(tp[t_hit], oo))
        return side * (x - e), s_hit


def first_break(c, H, Lo, k0, k1):
    """First bar j in [k0, k1) whose close is beyond (H, Lo) -> (j, side) or (None, 0).
    H / Lo may be arrays (vectorised over fake bands)."""
    seg = c[k0:k1]
    mx, mn = np.maximum.accumulate(seg), np.minimum.accumulate(seg)
    ju = np.searchsorted(mx, H, side="right")
    jd = np.searchsorted(-mn, -np.asarray(Lo), side="right")
    n = len(seg)
    side = np.where((ju < n) & (ju <= jd), 1, np.where(jd < n, -1, 0))
    j = np.where(side > 0, ju, jd) + k0
    return j, side


def day_cells(dd, rng=None, n_lvl=0):
    """-> real r, mirror r, side, entry bar, stop flag per cell (NaN = no trade),
    plus (n_lvl,) fake-band outcomes for each BRK cell."""
    w = Win(dd)
    o, h, lo, c = dd["o"], dd["h"], dd["l"], dd["c"]
    nc = len(CELLS)
    r, a = np.full(nc, np.nan), np.full(nc, np.nan)
    side, kent, stp = np.zeros(nc), np.full(nc, -1), np.zeros(nc, bool)
    lvl = {}
    for L in LENS:
        H, Lo = h[:L].max(), lo[:L].min()
        mid, W = (H + Lo) / 2, H - Lo
        j, s = first_break(c, np.array([H]), np.array([Lo]), L, KCUT)
        j, s = int(j[0]), int(s[0])
        ms = int(np.sign(c[L - 1] - o[0]))
        for ci, (kind, LL, stop) in enumerate(CELLS):
            if LL != L:
                continue
            if kind == "BRK":
                if s == 0:
                    continue
                k, sd = j + 1, s
                lev = {"none": None, "opp": (Lo if sd > 0 else H), "mid": mid}[stop]
            else:
                if ms == 0:
                    continue
                k, sd = L, ms
                lev = None if stop == "none" else (Lo if sd > 0 else H)
            e = o[k]
            st = None if lev is None else np.array([lev, 2 * e - lev])
            tp = None
            if kind == "ZA":
                R = abs(e - lev)
                tp = np.array([e + sd * 10 * R, e - sd * 10 * R])
            out, hit = w.run([k, k], [sd, -sd], st, tp)
            r[ci], a[ci] = out / dd["S"]
            side[ci], kent[ci], stp[ci] = sd, k, hit[0]
        if n_lvl:
            ce = c[L - 1]
            u = rng.uniform((ce - H) / W, (ce - Lo) / W, n_lvl) if W > 0 else np.zeros(n_lvl)
            Hs, Ls = H + u * W, Lo + u * W
            jj, ss = first_break(c, Hs, Ls, L, KCUT)
            ok = ss != 0
            for ci, (kind, LL, stop) in enumerate(CELLS):
                if kind != "BRK" or LL != L:
                    continue
                v = np.full(n_lvl, np.nan)
                if ok.any():
                    kk, sd = jj[ok] + 1, ss[ok]
                    lev = {"none": None, "opp": np.where(sd > 0, Ls[ok], Hs[ok]), "mid": ((Hs + Ls) / 2)[ok]}[stop]
                    v[ok] = w.run(kk, sd, lev)[0] / dd["S"]
                lvl[ci] = v
    return r, a, side, kent, stp, lvl


def build(days, rng, n_lvl):
    R, A, SD, KE, ST = [], [], [], [], []
    LV = np.full((len(days), len(CELLS), max(n_lvl, 1)), np.nan)
    for i, dd in enumerate(days):
        r, a, sd, ke, st, lv = day_cells(dd, rng, n_lvl)
        R.append(r), A.append(a), SD.append(sd), KE.append(ke), ST.append(st)
        for ci, v in lv.items():
            LV[i, ci] = v
    return dict(r=np.array(R), a=np.array(A), side=np.array(SD), k=np.array(KE),
                stop=np.array(ST), lvl=LV,
                tk=np.array([d["tk"] for d in days]), day=np.array([d["day"] for d in days]),
                S=np.array([d["S"] for d in days]), px=np.array([d["o"][0] for d in days]))


# ------------------------------------------------------------------ stats
def fw_z(real, null):
    """real (cells,), null (reps, cells) -> real z per cell, family-wise p95 of max z."""
    mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
    sd = np.where(sd > 0, sd, np.nan)
    zr = (real - mu) / sd
    zn = (null - mu) / sd
    return zr, float(np.nanquantile(np.nanmax(zn, 1), 0.95)), mu


def dir_null(B, rng, n_perm=N_PERM):
    r, a, side, tk = B["r"], B["a"], B["side"], B["tk"]
    nc = r.shape[1]
    long_ = np.where(side > 0, r, a)
    short = np.where(side > 0, a, r)
    null = np.empty((n_perm, nc))
    exp_ = np.full_like(r, np.nan)
    idx = {t: np.flatnonzero(tk == t) for t in TICKERS}
    for t, ix in idx.items():
        for ci in range(nc):
            tr = ix[~np.isnan(r[ix, ci])]
            q = (side[tr, ci] > 0).mean() if len(tr) else 0.5
            exp_[tr, ci] = q * long_[tr, ci] + (1 - q) * short[tr, ci]
    for p in range(n_perm):
        sp = side.copy()
        for t, ix in idx.items():
            for ci in range(nc):
                tr = ix[~np.isnan(r[ix, ci])]
                sp[tr, ci] = rng.permutation(side[tr, ci])
        out = np.where(sp == side, r, a)
        null[p] = np.nanmean(out, 0)
    return null, r - exp_


def dboot(days, x, rng):
    m = ~np.isnan(x)
    days, x = days[m], x[m]
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(ud))
    k = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, len(ud), len(ud))
        bs.append(s[j].sum() / k[j].sum())
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def score(B, rng, title, plant=None, verbose=True, n_perm=N_PERM):
    B = dict(B)
    if plant is not None:
        B["r"] = B["r"].copy()
        B["r"][:, plant] += 0.05
    real = np.nanmean(B["r"], 0)
    dnull, exc = dir_null(B, rng, n_perm)
    zd, pd95, _ = fw_z(real, dnull)
    lv = B["lvl"][:, BRK_IX, :]
    lnull = np.nanmean(lv, 0).T                       # (draws, BRK cells)
    zl_b, pl95, lmu = fw_z(real[BRK_IX], lnull)
    zl = np.full(len(CELLS), np.nan)
    zl[BRK_IX] = zl_b
    lmean = np.full(len(CELLS), np.nan)
    lmean[BRK_IX] = lmu
    day = B["day"]
    o = np.argsort(day, kind="stable")
    hits, trade = [], []
    if verbose:
        print(f"\n  {title}   ticker-days {len(day)}   family-wise p95: z_dir {pd95:.2f}, z_level {pl95:.2f}")
        print("     cell          n    mean S  fake-band  z_dir  z_lvl   excess IS / OOS     sl>0  "
              "95% CI of r           EDGE TRADE")
    for ci, c in enumerate(CELLS):
        x, e = B["r"][:, ci], exc[:, ci]
        m = ~np.isnan(x)
        hv = [np.nanmean(e[m & (day < SPLIT)]), np.nanmean(e[m & (day >= SPLIT)])]
        eo = e[o][~np.isnan(e[o])]
        sl = [s.mean() for s in np.array_split(eo, 6)]
        edge = (zd[ci] > pd95 and (c[0] != "BRK" or zl[ci] > pl95)
                and all(h > 0 for h in hv) and sum(v > 0 for v in sl) >= 4)
        lo, hi = dboot(day, x, rng) if (edge or verbose) else (np.nan, np.nan)
        tr = edge and real[ci] >= FLOOR and lo > 0
        if edge:
            hits.append(ci)
        if tr:
            trade.append(ci)
        if verbose:
            fb = f"{lmean[ci]:+.4f}" if c[0] == "BRK" else "   --  "
            zls = f"{zl[ci]:5.2f}" if c[0] == "BRK" else "  -- "
            print(f"     {cname(c):12} {m.sum():5d}  {real[ci]:+.4f}  {fb}   {zd[ci]:5.2f}  {zls}   "
                  f"{hv[0]:+.4f} / {hv[1]:+.4f}   {sum(v > 0 for v in sl)}/6   [{lo:+.4f}, {hi:+.4f}]   "
                  f"{'PASS' if edge else '  - '} {'PASS' if tr else '  - '}")
    return hits, trade


def reported(B, title):
    r, day, tk = B["r"], B["day"], B["tk"]
    bp = B["r"] * (B["S"] / B["px"])[:, None] * 1e4
    print(f"\n  {title}")
    print("     cell          n/days   win   stopped   mean bp   " + "   ".join(f"{t:>8}" for t in TICKERS)
          + "    median entry")
    nd = len(set(zip(day, tk)))
    for ci, c in enumerate(CELLS):
        m = ~np.isnan(r[:, ci])
        if not m.any():
            continue
        per = [np.nanmean(r[(tk == t), ci]) for t in TICKERS]
        ke = B["k"][m, ci]
        print(f"     {cname(c):12} {m.sum() / nd:6.2f}   {np.mean(r[m, ci] > 0):.2f}   {B['stop'][m, ci].mean():5.2f}"
              f"    {np.nanmean(bp[m, ci]):+6.1f}   " + "   ".join(f"{v:+.4f}" for v in per)
              + f"    {hm(OPEN + int(np.median(ke)))}")


def stage1():
    rng = np.random.default_rng(SEED)
    days = load()
    print(f"  loaded {len(days)} ticker-days "
          + "  ".join(f"{t} {sum(d['tk'] == t for d in days)}" for t in TICKERS))
    B = build(days, rng, N_LVL)
    sc = B["day"] >= DEP_START
    D = {k: (v[sc] if isinstance(v, np.ndarray) else v) for k, v in B.items()}
    P = {k: (v[~sc] if isinstance(v, np.ndarray) else v) for k, v in B.items()}
    print(f"  scored {len(D['day'])} ticker-days ({DEP_START} .. {END}), pre-sample {len(P['day'])}")

    h1, _ = score(D, rng, "M1 -- +0.05 S planted into BRK 15 opp", plant=PLANT, verbose=False)
    print(f"\n  M1 {'PASS' if PLANT in h1 else 'FAIL'}  (EDGE hits: {[cname(CELLS[i]) for i in h1]})")
    sdays = [shuffle_day(d, rng) for d in days if d["day"] >= DEP_START]
    B0 = build(sdays, rng, 100)
    h0, _ = score(B0, rng, "M0 -- shuffled bars", verbose=False, n_perm=500)
    print(f"  M0 {'PASS' if len(h0) <= 1 else 'FAIL'}  (EDGE hits on shuffled bars: "
          f"{[cname(CELLS[i]) for i in h0]}; mean r BRK 15 opp {np.nanmean(B0['r'][:, PLANT]):+.4f})")

    print("\n  ======== PRIMARY: scored window, 3 tickers pooled ========")
    hits, trade = score(D, rng, "ORB cells")
    print(f"\n  VERDICT: EDGE {[cname(CELLS[i]) for i in hits] or 'none'}   "
          f"TRADE {[cname(CELLS[i]) for i in trade] or 'none'}")

    print("\n  ======== REPORTED ========")
    reported(D, "scored window: trades per day, win rate, stop-out share, bp, mean S by ticker")
    if len(P["day"]):
        reported(P, "PRE-SAMPLE (2023-10-12 .. 2024-08-19, extra in-sample, labelled)")
        real = np.nanmean(P["r"], 0)
        lm = np.nanmean(np.nanmean(P["lvl"][:, BRK_IX, :], 0), 1)
        print("     pre-sample mean S vs fake band (BRK): " + "  ".join(
            f"{cname(CELLS[i]).strip()} {real[i]:+.3f}/{lm[j]:+.3f}" for j, i in enumerate(BRK_IX)))


# ------------------------------------------------------------------ stage 3
def orb_state(bars, mod, want, L=15):
    """WITH / AGAINST / NONE at a trigger minute, from bars closed before it."""
    if bars is None:
        return "nodata"
    o, h, lo, c = bars
    kt = mod - OPEN
    if kt <= L:
        return "none"
    H, Lo = h[:L].max(), lo[:L].min()
    j, s = first_break(c, np.array([H]), np.array([Lo]), L, kt)
    s = int(s[0])
    return "none" if s == 0 else ("with" if s == want else "against")


def gate():
    import directional_flow_backtester as D
    import sim_core
    from config import TRAIL_PCT
    rng = np.random.default_rng(SEED)
    rows, sums = [], {}
    for r in sim_core.research_rules():
        tk = r["ticker"]
        bars = load_bars(tk, full=False)
        last = max(bars)
        cand = [x for x in sim_core.deployed_candidates(D, r) if x[0] <= last]
        if not cand:
            continue
        want = 1 if r["direction"].upper() == "CALL" else -1
        pol, eod = sim_core.policy_for(r, TRAIL_PCT), sim_core.eod_mod(r)
        st = {L: [orb_state(bars.get(d), m, want, L) for d, m, _ in cand] for L in LENS}
        picks = []
        full = sim_core.walk(cand, pol, eod, fill="bot", picks_out=picks)
        for (d, pnl), (ci, _) in zip(full, picks):
            rows.append(dict(rule=r["name"], day=d, mod=cand[ci][1], pnl=pnl,
                             **{f"s{L}": st[L][ci] for L in LENS}))
        books = {"full": full}
        for L in LENS:
            books[f"no_against{L}"] = sim_core.walk([c for c, s in zip(cand, st[L]) if s != "against"],
                                                    pol, eod, fill="bot")
            books[f"with_only{L}"] = sim_core.walk([c for c, s in zip(cand, st[L]) if s == "with"],
                                                   pol, eod, fill="bot")
        sums[r["name"]] = {k: (sum(p for d, p in v if d < SPLIT), sum(p for d, p in v if d >= SPLIT), len(v))
                           for k, v in books.items()}
        print(f"  {r['name']:18} candidates {len(cand):5d}  taken {len(full):4d}  (bars to {last})")
    df = pd.DataFrame(rows)
    df = df[df["s15"] != "nodata"].reset_index(drop=True)

    def diff(lab, x):
        a, b = x[lab == "against"], x[lab != "against"]
        return (b.mean() - a.mean()) if len(a) and len(b) else np.nan

    print(f"\n  ======== STAGE 3: OR(15) state at the trigger, taken trades n={len(df)} ========")
    x = df["pnl"].to_numpy()
    for L in LENS:
        lab = df[f"s{L}"].to_numpy()
        d0 = diff(lab, x)
        null = np.empty(N_PERM)
        rules = df["rule"].to_numpy()
        for p in range(N_PERM):
            lp = lab.copy()
            for rn in np.unique(rules):
                ix = np.flatnonzero(rules == rn)
                lp[ix] = rng.permutation(lab[ix])
            null[p] = diff(lp, x)
        p95 = float(np.nanquantile(null, 0.95))
        isd = df["day"].to_numpy() < SPLIT
        hv = [diff(lab[isd], x[isd]), diff(lab[~isd], x[~isd])]
        tot = {k: [sum(v[k][0] for v in sums.values()), sum(v[k][1] for v in sums.values()),
                   sum(v[k][2] for v in sums.values())] for k in ("full", f"no_against{L}", f"with_only{L}")}
        book_ok = (tot[f"no_against{L}"][0] >= tot["full"][0]) and (tot[f"no_against{L}"][1] >= tot["full"][1])
        ok = d0 > p95 and all(h > 0 for h in hv) and book_ok
        tag = "PRIMARY" if L == 15 else "reported"
        print(f"\n  OR({L}) [{tag}]")
        for s in ("with", "none", "against"):
            m = lab == s
            print(f"     {s:8} n={m.sum():4d}  mean ROE {x[m].mean() * 100 if m.any() else np.nan:+6.1f}%  "
                  f"win {np.mean(x[m] > 0) if m.any() else np.nan:.2f}  "
                  f"(IS {x[m & isd].mean() * 100 if (m & isd).any() else np.nan:+.1f}% / "
                  f"OOS {x[m & ~isd].mean() * 100 if (m & ~isd).any() else np.nan:+.1f}%)")
        print(f"     (with+none) - against = {d0 * 100:+.1f} pp   perm p95 {p95 * 100:+.1f} pp   "
              f"halves {hv[0] * 100:+.1f} / {hv[1] * 100:+.1f} pp")
        for k, (i, o, n) in tot.items():
            print(f"     book {k:13} n={n:4d}  total ROE IS {i * 100:+8.1f}%  OOS {o * 100:+8.1f}%")
        print(f"     GATE {'PASS' if ok else 'FAIL'}" + ("" if L == 15 else "  (unscored)"))
    print("\n  per rule, OR(15): n with/none/against, mean ROE")
    for rn, g in df.groupby("rule"):
        cells = []
        for s in ("with", "none", "against"):
            m = g["s15"] == s
            cells.append(f"{s} {m.sum():3d} {g.loc[m, 'pnl'].mean() * 100 if m.any() else np.nan:+6.1f}%")
        print(f"     {rn:18} " + "   ".join(cells))


if __name__ == "__main__":
    if "--gate" in sys.argv:
        gate()
    else:
        stage1()
