# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0", "scipy>=1.11"]
# ///
"""
check_swing_factors.py
======================
SWING BOOK, STAGE A -- DO ANY DAILY FACTORS PREDICT THE NEXT 1-4 WEEKS?
(user, 2026-10-22: "explore the bot taking long positions (+14DTE) based on
factors that we decide, and holding them for periods longer than EOD";
factors named by the user: IV, RIV, term structure, flow, OI change.)

Plan agreed 10-22: A = factor -> forward underlying return / move size (this
file); B = survivors as real 14-45 DTE option trades on lake quotes vs random-
day entries; C = paper swing book in the bot. Bot work only at C.

UNIVERSE   ETFs SPY / QQQ / IWM (PRIMARY, scored) and the bot's other names
           META NVDA MSFT AVGO SMH GLD (scored SEPARATELY, own threshold).
DATA       --build: lake/silver/option-contracts-1m -> one end-of-day row per
           contract with DTE <= 75 (last non-null IV / delta / bid / ask at or
           before 15:59, day volume + ask/bid volume, start-of-day OI; lake OI
           == UW per-contract historic OI, checked 10-22) -> _swing_cache/.
           Spot = the 1m close of the 15:59 bar (historical/{T}.parquet).
           Flow = historical/NETPREM{T}.parquet (minutes >= 09:29).
           GEX = historical/GEX{T}.parquet net_gex.
WINDOW     scored 2024-08-20 onward (factor date), halves split 2025-08-21;
           2023-10-12 .. 2024-08-19 pre-sample reported, labelled.

FACTORS (fixed before the run; all known at the CLOSE of the factor date)
  vol family
    ivrank   IV30 percentile in its trailing 252 sessions (min 120)
    riv      IV30 - RV21 (vol points; RV from daily closes)
    term     IV9 / IV30 (> 1 = inverted / stressed)
    skew     25-delta put IV - call IV, expiries 20-45 DTE (vol points)
  direction family
    dskew    skew minus skew 5 sessions earlier
    flow5    5-session net premium / trailing-60 mean |daily net premium|
    oidd     sticky customer delta: sum over 7-60 DTE contracts of
             |dOI| x side x delta x 100 x spot, side = +1 ask-led / -1 bid-led
             (>= 55% of volume) -- dOI is only known the next morning, so the
             value from day d's trading is dated d+1. / trailing-60 mean |.|
    gex      PRIOR session's net GEX / trailing-120 mean |net GEX|
  controls (reported, not scored): mom20, mom60 (log returns)
  ATM IV at tenor tau: per expiry, mean IV of |delta| 0.35-0.65 contracts
  quoted after 15:00; total variance interpolated in calendar days.
TARGETS    for h = 5 / 10 / 20 sessions from the factor date's close:
    ret  log return
    move |log return| / (ATM IV at tau = 7 / 14 / 28 days x sqrt(h/252))
         -- realized move as a multiple of the implied one (> 1: long options
         beat the price paid for the move, roughly)

PRE-REGISTERED (per group; 8 factors x 2 targets x 3 horizons = 48 cells)
  IC = Spearman rank correlation over time, per ticker, averaged over the
  group's tickers.
  NULL  the factor series circularly SHIFTED in time (same shift for every
        ticker; keeps autocorrelation, overlap and cross-ticker structure),
        1,000 shifts of 60+ sessions; family-wise p95 of max |IC| over cells.
  EDGE  |IC| > family-wise p95; same sign in both halves; same sign in >= 2/3
        ETFs (>= 4/6 names); for the RET target the IC must survive trend:
        partial IC controlling mom20 + mom60 has the same sign and >= half the
        size.
  Reported, unscored: quintile means (bp and move ratio), the pre-sample,
  factor correlations, per-ticker ICs.
  MACHINERY  M1: a planted factor = 10-session forward return + noise (IC
             ~0.10) must pass its ret-10 cell. M0: an AR(1) noise factor with
             phi 0.97 must not pass.

Usage:  python check_swing_factors.py --build     # once, ~5-10 min
        python check_swing_factors.py

RUN LOG
  2026-10-22 build: 9 tickers, 726 lake days, 0.4-2.2M contract-days each.
  2026-10-22 run 1: coverage 94-100% per factor; ETF median IV30 14.6 / 20.2 /
    22.0 vs RV21 12.9 / 18.2 / 19.6. MACHINERY FAILED: family-wise p95 of max
    |IC| = 0.293 -- set by the persistent factors (ivrank, skew: their shifted
    copies throw big chance ICs on overlapping returns); the planted IC 0.115
    could never pass. No verdict taken from run 1.
  2026-10-22 run 2 (scoring CHANGED AFTER SEEING RUN 1 -- logged): each cell
    z-scored against its own shift distribution, family-wise p95 of max |z|
    (as check_orb). M1 PASS (ret5 z 3.39, ret10 z 4.03 vs 3.26), M0 PASS.
    ETFs p95 |z| 3.14: EDGE none.  Names p95 |z| 3.09: EDGE none.
    Leads (|z| 2.0-2.9, not family-wise):
      skew -> ret (steeper put skew, higher forward return): ETFs z 2.64 /
        2.52 / 2.16 (h 5/10/20, IC +0.17..+0.23, both halves +, 3/3, trend-
        partial unchanged); names z 2.78 / 2.12 (6/6, 5/6); pre-sample ETF
        ret10 IC +0.31; without the 2025-03-24..05-30 tariff episode still
        +0.13 (ETFs) / +0.11 (names). Quintiles ETFs h10: Q1 -12 bp .. Q5
        +187 bp. The most consistent thing in the file.
      gex -> move (negative prior-day GEX, bigger realized-vs-implied move):
        ETFs z -2.56 (h10) / -2.26 (h20), both halves, 3/3, pre-sample -0.14,
        ex-episode -0.13 / -0.17; names ~0.
      ivrank -> ret ETFs z 2.08 but names ~0 / negative; term flips sign
        between ETFs (+) and names (-) -- not a lead.
    Move ratio (realized |r| / implied) averages 0.6-0.8 in every quintile of
    every factor (an ATM straddle breaks even near 0.8): no vol factor makes
    long premium cheap on its own.
  VERDICT: no factor clears the pre-registered bar; stage B not warranted as
    designed. Honest next step = replicate the two leads on a FRESH universe
    (lake has the whole market) as two pre-registered hypotheses, or forward.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ETFS = ("SPY", "QQQ", "IWM")
NAMES = ("META", "NVDA", "MSFT", "AVGO", "SMH", "GLD")
ALL = ETFS + NAMES
CACHE = "_swing_cache"
LAKE = "lake/silver/option-contracts-1m"
MAX_DTE = 75
DEP_START, SPLIT = dt.date(2024, 8, 20), dt.date(2025, 8, 21)
HS = (5, 10, 20)
TAU = {5: 7, 10: 14, 20: 28}
FACTORS = ("ivrank", "riv", "term", "skew", "dskew", "flow5", "oidd", "gex")
TARGETS = ("ret", "move")
N_SHIFT, MIN_SHIFT = 1000, 60
SEED = 20261022


# ------------------------------------------------------------------ build
def build():
    os.makedirs(CACHE, exist_ok=True)
    days = sorted(glob.glob(f"{LAKE}/date=*/bars.parquet"))
    out = {t: [] for t in ALL}
    cols = ["underlying_symbol", "option_chain_id", "expiry", "option_type", "strike", "minute_et",
            "iv_close", "delta_close", "bid_close", "ask_close", "open_interest", "volume", "ask_volume", "bid_volume"]
    for i, f in enumerate(days):
        day = dt.date.fromisoformat(f.split("date=")[1][:10])
        d = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(ALL)))
             .filter(pl.col("expiry") <= pl.lit(day + dt.timedelta(days=MAX_DTE)))
             .select(cols).collect())
        if d.is_empty():
            continue
        d = d.with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                            + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
        q = d.filter(pl.col("mod") <= 959).sort("mod")
        last = (q.group_by("option_chain_id").agg(
            pl.col("mod").max().alias("last_mod"),
            *[pl.col(c).drop_nulls().last().alias(c) for c in ("iv_close", "delta_close", "bid_close", "ask_close")]))
        tot = d.group_by("option_chain_id").agg(
            pl.col("underlying_symbol").first(), pl.col("expiry").first(), pl.col("option_type").first(),
            pl.col("strike").first(), pl.col("open_interest").drop_nulls().first().alias("oi"),
            pl.col("volume").sum().alias("vol"), pl.col("ask_volume").sum().alias("askv"),
            pl.col("bid_volume").sum().alias("bidv"))
        g = tot.join(last, on="option_chain_id", how="left").with_columns(pl.lit(day).alias("date"))
        for t in ALL:
            p = g.filter(pl.col("underlying_symbol") == t)
            if p.height:
                out[t].append(p)
        if i % 50 == 0:
            print(f"  {day}  ({i + 1}/{len(days)})", flush=True)
    for t, parts in out.items():
        if parts:
            pl.concat(parts, how="vertical_relaxed").write_parquet(f"{CACHE}/eod_{t}.parquet")
            print(f"  {t}: {sum(p.height for p in parts):,} contract-days")


# ------------------------------------------------------------------ factors
def spot_closes(t):
    d = (pl.read_parquet(f"historical/{t}.parquet", columns=["date", "minute_et", "close"])
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter(pl.col("mod") <= 959).sort("minute_et")
         .group_by("date").agg(pl.col("close").last(), pl.col("mod").max().alias("m")).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d = d[d["m"] >= 955]                                  # full sessions only
    return d.set_index("date")["close"].sort_index()


def iv_at(tau, pts):
    """ATM IV at tau calendar days from [(dte, iv)], total variance interpolation."""
    if not pts:
        return np.nan
    lo = [p for p in pts if p[0] <= tau]
    hi = [p for p in pts if p[0] >= tau]
    if lo and hi:
        (d1, v1), (d2, v2) = max(lo), min(hi)
        if d1 == d2:
            return v1
        w = (tau - d1) / (d2 - d1)
        return float(np.sqrt(((1 - w) * v1 * v1 * d1 + w * v2 * v2 * d2) / tau))
    d0, v0 = min(pts, key=lambda p: abs(p[0] - tau))
    return v0 if abs(d0 - tau) <= max(3, 0.35 * tau) else np.nan


def daily_flow(t):
    d = pl.read_parquet(f"historical/NETPREM{t}.parquet", columns=["date", "minute_et", "net_premium"])
    if d["minute_et"].dtype == pl.Utf8:
        d = d.with_columns(pl.col("minute_et").str.to_datetime())
    d = (d.with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                         + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 569) & (pl.col("mod") <= 960))
         .group_by("date").agg(pl.col("net_premium").sum()).to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.set_index("date")["net_premium"].sort_index()


def scale(x, n):
    """x / trailing-n mean |x| (excluding today), min n//2 observations."""
    m = x.abs().shift(1).rolling(n, min_periods=n // 2).mean()
    return x / m


def panel(t):
    S = spot_closes(t)
    e = pl.read_parquet(f"{CACHE}/eod_{t}.parquet").to_pandas()
    e["date"] = pd.to_datetime(e["date"]).dt.date
    e["expiry"] = pd.to_datetime(e["expiry"]).dt.date
    e["dte"] = [(x - y).days for x, y in zip(e["expiry"], e["date"])]
    q = e[(e["last_mod"] >= 900) & (e["iv_close"] > 0.01) & (e["iv_close"] < 3) & (e["dte"] >= 1)]
    rows = {}
    for day, g in q.groupby("date"):
        a = g[g["delta_close"].abs().between(0.35, 0.65)]
        pts = sorted((int(k), float(v)) for k, v in a.groupby("dte")["iv_close"].mean().items())
        r = {f"iv{k}": iv_at(k, pts) for k in (7, 9, 14, 28, 30)}
        s = g[g["dte"].between(20, 45)]
        p = s[(s["option_type"] == "put") & s["delta_close"].between(-0.30, -0.20)]["iv_close"]
        c = s[(s["option_type"] == "call") & s["delta_close"].between(0.20, 0.30)]["iv_close"]
        r["skew"] = (p.mean() - c.mean()) * 100 if len(p) and len(c) else np.nan
        rows[day] = r
    f = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    f = f.reindex(S.index)
    f["S"] = S
    lr = np.log(S).diff()
    f["rv21"] = lr.rolling(21, min_periods=18).std() * np.sqrt(252) * 100
    f["iv30p"] = f["iv30"] * 100
    f["riv"] = f["iv30p"] - f["rv21"]
    f["term"] = f["iv9"] / f["iv30"]
    f["ivrank"] = f["iv30"].rolling(252, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    f["dskew"] = f["skew"] - f["skew"].shift(5)
    fl = daily_flow(t).reindex(f.index)
    f["flow5"] = fl.rolling(5, min_periods=4).sum() / fl.abs().shift(1).rolling(60, min_periods=30).mean()
    # sticky customer delta from day d's trading, known on d+1
    e2 = e[(e["dte"].between(7, 60)) & (e["vol"] > 0)].copy()
    e2["side"] = np.where(e2["askv"] >= 0.55 * e2["vol"], 1, np.where(e2["bidv"] >= 0.55 * e2["vol"], -1, 0))
    days = list(f.index)
    nxt = dict(zip(days[:-1], days[1:]))
    oi = e[["option_chain_id", "date", "oi"]]
    oi_next = oi.assign(date=oi["date"].map({v: k for k, v in nxt.items()}))   # OI of d+1 keyed on d
    m = e2.merge(oi_next.rename(columns={"oi": "oi_n"}), on=["option_chain_id", "date"], how="inner")
    m["S"] = m["date"].map(S)
    m["dd"] = m["side"] * (m["oi_n"] - m["oi"]).abs() * m["delta_close"] * 100 * m["S"]
    dd = m.groupby("date")["dd"].sum()
    dd.index = [nxt.get(d) for d in dd.index]
    dd = dd[[i is not None for i in dd.index]]
    dd = dd.groupby(level=0).sum().reindex(f.index)
    f["oidd"] = scale(dd, 60)
    try:
        g = pd.read_parquet(f"historical/GEX{t}.parquet")
        g["d"] = pd.to_datetime(g["date"], utc=True).dt.tz_localize(None).dt.date
        gx = g.groupby("d")["net_gex"].last().reindex(f.index).shift(1)
        f["gex"] = gx / gx.abs().shift(1).rolling(120, min_periods=60).mean()
    except Exception:
        f["gex"] = np.nan
    f["mom20"] = np.log(S / S.shift(20))
    f["mom60"] = np.log(S / S.shift(60))
    for h in HS:
        r = np.log(S.shift(-h) / S)
        f[f"ret{h}"] = r
        f[f"move{h}"] = r.abs() / (f[f"iv{TAU[h]}"] * np.sqrt(h / 252))
    f["tk"] = t
    return f


# ------------------------------------------------------------------ stats
def spear(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 40:
        return np.nan
    x, y = rankdata(a[m]), rankdata(b[m])
    return float(np.corrcoef(x, y)[0, 1])


def partial_ic(f, t, ctrl):
    m = ~(np.isnan(f) | np.isnan(t) | np.isnan(ctrl).any(1))
    if m.sum() < 40:
        return np.nan
    X = np.column_stack([np.ones(m.sum())] + [rankdata(c) for c in ctrl[m].T])
    rf, rt = rankdata(f[m]), rankdata(t[m])
    ef = rf - X @ np.linalg.lstsq(X, rf, rcond=None)[0]
    et = rt - X @ np.linalg.lstsq(X, rt, rcond=None)[0]
    return float(np.corrcoef(ef, et)[0, 1])


def cells(factors):
    return [(f, tg, h) for f in factors for tg in TARGETS for h in HS]


def score(P, tickers, factors, rng, title, quint=True):
    """P: {ticker: DataFrame on a common date index}."""
    idx = P[tickers[0]].index
    dates = np.array(idx)
    sc = dates >= DEP_START
    C = cells(factors)
    F = {tk: {f: P[tk][f].to_numpy(float) for f in factors} for tk in tickers}
    T = {tk: {f"{tg}{h}": np.where(sc, P[tk][f"{tg}{h}"].to_numpy(float), np.nan) for tg in TARGETS for h in HS}
         for tk in tickers}
    real = np.array([np.nanmean([spear(F[tk][f], T[tk][f"{tg}{h}"]) for tk in tickers]) for f, tg, h in C])
    n = len(dates)
    null = np.empty((N_SHIFT, len(C)))
    for i in range(N_SHIFT):
        k = int(rng.integers(MIN_SHIFT, n - MIN_SHIFT))
        null[i] = [np.nanmean([spear(np.roll(F[tk][f], k), T[tk][f"{tg}{h}"]) for tk in tickers]) for f, tg, h in C]
    # each cell against ITS OWN shift distribution (a persistent factor's chance |IC| is far wider
    # than a fast one's); family-wise = p95 of the max |z| over cells
    mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
    zr = (real - mu) / sd
    p95 = float(np.quantile(np.nanmax(np.abs((null - mu) / sd), 1), 0.95))
    print(f"\n  {title}   tickers {', '.join(tickers)}   cells {len(C)}   family-wise p95 |z| = {p95:.2f}")
    print("     factor  target  h    IC      z      IS      OOS    tickers same sign   partial(trend)   EDGE")
    hits = []
    for j, (f, tg, h) in enumerate(C):
        ic = real[j]
        halves = []
        for msk in (dates < SPLIT, dates >= SPLIT):
            halves.append(np.nanmean([spear(F[tk][f], np.where(msk, T[tk][f"{tg}{h}"], np.nan)) for tk in tickers]))
        per = [spear(F[tk][f], T[tk][f"{tg}{h}"]) for tk in tickers]
        same = sum(1 for p in per if np.sign(p) == np.sign(ic))
        need = 2 if len(tickers) == 3 else 4
        pic = np.nan
        if tg == "ret":
            pic = np.nanmean([partial_ic(F[tk][f], T[tk][f"{tg}{h}"],
                                         np.column_stack([P[tk]["mom20"].to_numpy(float), P[tk]["mom60"].to_numpy(float)]))
                              for tk in tickers])
        trend_ok = tg != "ret" or (np.sign(pic) == np.sign(ic) and abs(pic) >= 0.5 * abs(ic))
        edge = (abs(zr[j]) > p95 and all(np.sign(x) == np.sign(ic) for x in halves) and same >= need and trend_ok)
        if edge:
            hits.append((f, tg, h))
        print(f"     {f:7} {tg:5} {h:3}  {ic:+.3f}  {zr[j]:+5.2f}  {halves[0]:+.3f}  {halves[1]:+.3f}   {same}/{len(tickers)}"
              f"               {'' if np.isnan(pic) else f'{pic:+.3f}':>7}        {'PASS' if edge else '  - '}")
    if quint:
        quintiles(P, tickers, factors, sc)
    return hits, p95


def quintiles(P, tickers, factors, sc, h=10):
    print(f"\n     quintiles (per-ticker, scored window), h = {h}: mean ret (bp) / mean move ratio, Q1 (low) .. Q5 (high)")
    for f in factors:
        rr, mm = [[] for _ in range(5)], [[] for _ in range(5)]
        for tk in tickers:
            d = P[tk][sc][[f, f"ret{h}", f"move{h}"]].dropna()
            if len(d) < 50:
                continue
            q = pd.qcut(d[f].rank(method="first"), 5, labels=False)
            for k in range(5):
                rr[k] += list(d.loc[q == k, f"ret{h}"] * 1e4)
                mm[k] += list(d.loc[q == k, f"move{h}"])
        print(f"     {f:7} " + "  ".join(f"{np.mean(rr[k]):+6.0f}/{np.mean(mm[k]):.2f}" if rr[k] else "   --   " for k in range(5)))


def main():
    rng = np.random.default_rng(SEED)
    P = {}
    for t in ALL:
        if not os.path.exists(f"{CACHE}/eod_{t}.parquet"):
            print(f"  {t}: no cache (run --build)")
            continue
        P[t] = panel(t)
    idx = sorted(set().union(*[set(p.index) for p in P.values()]))
    P = {t: p.reindex(idx) for t, p in P.items()}
    for t, p in P.items():
        s = p[p.index >= DEP_START]
        print(f"  {t:5} days {s['S'].notna().sum():4d}  coverage: " + "  ".join(
            f"{f} {s[f].notna().mean():.0%}" for f in FACTORS) + f"   median IV30 {s['iv30p'].median():.1f}  RV21 {s['rv21'].median():.1f}")

    etf = [t for t in ETFS if t in P]
    # machinery
    for t in etf:
        r = P[t]["ret10"].to_numpy(float)
        P[t]["M1"] = r + rng.normal(0, np.nanstd(r) * 9.0, len(r))
        a = np.zeros(len(r))
        for i in range(1, len(r)):
            a[i] = 0.97 * a[i - 1] + rng.normal()
        P[t]["M0"] = a
    hits, _ = score(P, etf, ("M1", "M0") + FACTORS, rng, "MACHINERY (planted + AR(1) noise, ETFs)", quint=False)
    print(f"  M1 {'PASS' if ('M1', 'ret', 10) in hits else 'FAIL'}   M0 {'PASS' if not any(h[0] == 'M0' for h in hits) else 'FAIL'}")

    print("\n  ======== PRIMARY: ETFs ========")
    hits, _ = score(P, etf, FACTORS, rng, "ETFs, scored window")
    print(f"\n  VERDICT ETFs: EDGE {hits or 'none'}")
    names = [t for t in NAMES if t in P]
    if names:
        print("\n  ======== SECOND GROUP: the bot's other names (scored separately) ========")
        h2, _ = score(P, names, FACTORS, rng, "names, scored window")
        print(f"\n  VERDICT names: EDGE {h2 or 'none'}")

    print("\n  ======== REPORTED ========")
    s = pd.concat([P[t][P[t].index >= DEP_START][list(FACTORS) + ["mom20"]] for t in etf])
    print("  factor rank correlations (ETFs pooled):")
    print(s.rank().corr().round(2).to_string())
    ex0, ex1 = dt.date(2025, 3, 24), dt.date(2025, 5, 30)
    print(f"\n  ONE-EPISODE CHECK: IC with the tariff crash + rebound ({ex0} .. {ex1}) removed, scored window")
    for grp, tks in (("ETFs", etf), ("names", names)):
        for f in FACTORS:
            v = []
            for tg in TARGETS:
                for h in HS:
                    ics = []
                    for t in tks:
                        d = P[t][(P[t].index >= DEP_START) & ~((P[t].index >= ex0) & (P[t].index <= ex1))]
                        ics.append(spear(d[f].to_numpy(float), d[f"{tg}{h}"].to_numpy(float)))
                    v.append(f"{tg}{h} {np.nanmean(ics):+.3f}")
            print(f"     {grp:5} {f:7} " + "  ".join(v))
    pre = {t: P[t] for t in etf}
    print("\n  PRE-SAMPLE (labelled) IC, h = 10, ETFs:")
    for f in FACTORS:
        v = []
        for tg in TARGETS:
            ics = []
            for t in etf:
                d = pre[t][pre[t].index < DEP_START]
                ics.append(spear(d[f].to_numpy(float), d[f"{tg}10"].to_numpy(float)))
            v.append(f"{tg} {np.nanmean(ics):+.3f}")
        print(f"     {f:7} " + "   ".join(v))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        main()
