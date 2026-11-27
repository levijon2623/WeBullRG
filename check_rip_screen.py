"""
check_rip_screen.py -- IC SCREEN + LASSO ON QUIET-STRETCH -> RIP (user, 2026-11-26: "assemble a good list of inputs for
a screen ... perform the screen with everything we have so far included, and then again later with everything from this
screen as context and the new inputs"). PRE-REGISTERED 2026-11-26, committed before any input below was scored against
either target.

EVENT     check_absorption_rip's BASE, imported unchanged (60+ min of closes within 2.5 typical 30-min ranges, extended
          while inside, ends <= 15:00; outcome over the next 60 min). SAMPLE = sessions 2024-08-20 .. 2026-11-13 that
          have the THRU / PULL cue caches (_breach_cache/events, _structure_ic_cache/events) -- the cues' history.
TARGETS   SIZE  RIP (excursion >= 3 typical ranges within 60 min), 0/1, every base.
          DIR   dir x X (the larger excursion signed by its side, in typical-range units), every base -- which way
                the break goes and how far (all ~1,000+ bases per half, not just the ~4% that rip).
INPUTS    everything stamped at or before the base's last minute e (option / exposure bars only for minutes <= e).
          Units: T30 = that session's typical 30-min range (the event's own), price moves divided by T30 x close.
  DIRECTIONAL (signed, + = up / bullish; DIR uses these as they are, SIZE uses their absolute values)
    SLOPE15   close(e) - close(e-15)
    POS       position of close(e) in the base band, -1 bottom .. +1 top
    EDGE_NET  minutes whose high reached the top 10% of the band minus minutes whose low reached the bottom 10%
    UPVOL     (volume on up-close minutes - on down-close minutes) / base volume (tick-rule delta from candles)
    HOLD      share of base minutes closing above their open minus share closing below (the user's 1-min candle read)
    DOPEN     close(e) - the 09:30 open          DVWAP  close(e) - session VWAP to e
    DPREV     close(e) - prior close             PRE    close(s) - the 09:30 open (the move into the base)
    GAP       09:30 open - prior close           PREVRET  prior session's open -> close
    XMOVE15   the other two tickers' 15-min move to e (mean, each in its own T30)
    CHARM / VANNA  spot-exposures charm / vanna per 1% move (OI), last minute <= e
    DD        0DTE (ask - bid) x delta x 100 over the base  [SPENT: absorption Q4 used the even months]
    NETPREM   0DTE (ask - bid volume) x vwap x 100, calls minus puts, over the base
    CPTILT    0DTE (call volume - put volume) / total over the base
    HREL      check_hedge_release: summed customer put delta sold (fully sided events, package rule) in the base, shares
              [its own H1-H4 holdout is held; using it here SPENDS that holdout for this input -- said now]
    PEAKD     close(e) - the weekly map's 0DTE+index peak-gamma zone mid (check_peak_zone v2 cache, 09:31 map)
    THRU_NET  THRUSTRIKE breaches in the base: up minus down     THRU_LAST  latest one's side x (its frozen score's
              live quintile - 2, i.e. -2 .. +2; strike_cue_thru.json cuts)
    PULL_NET  NEXTLEG setups in the base: long minus short       PULL_LAST  the same with pullback_model_frozen cuts
    TIE_LAST  TIEBREAK: side of the later event of the latest opposite THRU / PULL pair (<= 120 s apart) in the base
  NON-DIRECTIONAL (SIZE only)
    L, VD     base length, last-30 / first-30 volume  [SPENT: absorption Q1 / Q2]
    G         dealer gamma per 1% (OI), last minute <= e  [SPENT as a sign: absorption Q3 PASS -- a known positive]
    TIGHT     base close range          TOD  minute of e          RVOL  volume to e / median same window, prior 20
    XQUIET    other tickers in a base at e (their base >= 60 min old and still inside its band at e)
    IVCHG     0DTE ATM implied vol, e minus s       STRADCHG  0DTE ATM straddle mid, e / s - 1
    THRU_N, PULL_N, TIE_N   counts in the base     PEAKSIGN  the peak zone's gamma sign
    DOW, OPEX (monthly expiry day), MONTHEND (last 2 sessions of the month), FOMC, MACRO_AM (CPI / PCE / NFP)
  Missing -> 0.5 after ranking. Ranks: each input -> its empirical CDF within ticker on the DISCOVERY events.
SPLIT     DISCOVERY = odd calendar months, HOLDOUT = even months (the absorption study's split). PARTLY CLEAN, said now:
          the even months already scored L, VD, G-sign and DD (absorption Q1-Q4) -- in here they are controls; every
          other input has never been scored on these events. Discovery halves for sign checks split at its median date.
PASS RULES (each target separately; BH across the two P2 p-values)
  M1 machinery (discovery): a planted input (target rank + noise, IC ~0.10) must reach IC >= 0.05, p < 0.01.
  M2 null (discovery): target shuffled across events -> the Lasso keeps 0 inputs or the shuffled-fit IC |.| < 0.03.
  P1 IC: Spearman(input, target), day-block bootstrap p (2,000), BH q < 0.10 across the target's inputs, same sign in
     both discovery halves. Reported; not the verdict.
  P2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on all the target's ranks, fit on discovery. PASS = keeps >= 1
     input AND holdout Spearman(score, target) > 0 with the day-block 95% CI above 0, BH q <= 0.10.
  Sanity, not a rule: SIZE should find G (it passed alone).
  No money test here: a DIR pass gets its own pre-registered trade test.
SCREEN 2 (LATER, after an ES trades purchase -- not bought yet): futures aggressor inputs (base delta, delta trend,
  delta at the band edges, delta vs price divergence) ADDED to the frozen screen-1 score, fit on the same discovery;
  PASS = the new score's holdout IC beats screen 1's (paired day-block bootstrap). Partly clean: the holdout will have
  scored screen 1 once.

    python check_rip_screen.py --discovery
    python check_rip_screen.py --holdout          (once)
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
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_ripscreen_cache")
TKS = ("SPY", "QQQ", "IWM")
LO, HI = "2024-08-20", "2026-11-13"
WIN = 120
SEED = 20261126
DIRECTIONAL = ["SLOPE15", "POS", "EDGE_NET", "UPVOL", "HOLD", "DOPEN", "DVWAP", "DPREV", "PRE", "GAP", "PREVRET",
               "XMOVE15", "CHARM", "VANNA", "DD", "NETPREM", "CPTILT", "HREL", "PEAKD", "THRU_NET", "THRU_LAST",
               "PULL_NET", "PULL_LAST", "TIE_LAST"]
NONDIR = ["L", "VD", "G", "TIGHT", "TOD", "RVOL", "XQUIET", "IVCHG", "STRADCHG", "THRU_N", "PULL_N", "TIE_N", "PEAKSIGN",
          "DOW", "OPEX", "MONTHEND", "FOMC", "MACRO_AM"]
SIZE_IN = NONDIR + [f"ABS_{k}" for k in DIRECTIONAL]
DIR_IN = DIRECTIONAL


def mod_of(col):
    return pl.col(col).dt.hour().cast(pl.Int32) * 60 + pl.col(col).dt.minute().cast(pl.Int32)


def ext(d, tk):
    p = os.path.join(LAKE, "ext-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(p):
        return None
    x = pl.read_parquet(p).filter(pl.col("market_time") == "r").with_columns(m=mod_of("minute")).sort("m")
    return x.select("m", "open", "high", "low", "close", "volume").to_numpy() if x.height > 300 else None


def options_day(d):
    """per ticker: minute -> (np_call - np_put, call vol, put vol), and minute -> (ATM straddle mid, ATM iv)."""
    p = os.path.join(LAKE, "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(p):
        return {}
    y = (pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(list(TKS)) & (pl.col("expiry") == pl.lit(d).str.to_date()))
         .with_columns(mm=mod_of("minute_et")).collect())
    out = {}
    for tk in TKS:
        z = y.filter(pl.col("underlying_symbol") == tk)
        if not z.height:
            continue
        fl = (z.with_columns(npv=(pl.col("ask_volume") - pl.col("bid_volume")).cast(pl.Float64) * pl.col("vwap").fill_null(0.0) * 100,
                             call=pl.col("option_type") == "call")
              .group_by("mm").agg(pl.when(pl.col("call")).then(pl.col("npv")).otherwise(-pl.col("npv")).sum().alias("np"),
                                  pl.when(pl.col("call")).then(pl.col("volume")).otherwise(0).sum().alias("cv"),
                                  pl.when(~pl.col("call")).then(pl.col("volume")).otherwise(0).sum().alias("pv")))
        atm = (z.with_columns(K=pl.col("underlying_close").round(0), mid=(pl.col("bid_close") + pl.col("ask_close")) / 2)
               .filter(pl.col("strike") == pl.col("K"))
               .group_by("mm").agg(pl.col("mid").sum().alias("strad"), pl.col("iv_close").mean().alias("iv"), pl.len().alias("n"))
               .filter(pl.col("n") == 2))
        out[tk] = ({a: (b, c, e) for a, b, c, e in fl.iter_rows()}, {a: (b, c) for a, b, c, _ in atm.iter_rows()})
    return out


def exposures(d, tk):
    p = os.path.join(LAKE, "spot-exposures-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(p):
        return {}
    y = pl.read_parquet(p).with_columns(mm=mod_of("minute_et"))
    return {a: (b, c, e) for a, b, c, e in y.select("mm", "gamma_per_one_percent_move_oi", "charm_per_one_percent_move_oi",
                                                    "vanna_per_one_percent_move_oi").iter_rows()}


def last_at(dct, m, back=10):
    for k in range(m, m - back - 1, -1):
        if k in dct and dct[k] is not None:
            return dct[k]
    return None


def cues_day(d, fb, fp):
    import check_breach_forward as BF
    import freeze_pullback as FP
    B = json.load(open(os.path.join(ROOT, "_breach_cache", "events", f"{d}.json")))
    S = json.load(open(os.path.join(ROOT, "_structure_ic_cache", "events", f"{d}.json")))
    # the live pill's QUINTILE of each frozen score (strike_cue_thru.json / pullback_model_frozen.json cuts), centred:
    # -2 .. +2 (raw scores sit mostly below 0, so side x raw score would flip meaning)
    tc = json.load(open(os.path.join(ROOT, "strike_cue_thru.json")))["cuts"]
    pc = fp["cuts"]
    for b in B:
        b["score"] = float(np.searchsorted(tc, BF.score(fb, b), side="right")) - 2.0
    for s in S:
        s["score"] = float(np.searchsorted(pc, FP.score(fp, s), side="right")) - 2.0
    pairs = defaultdict(list)                 # ticker -> [(t of the later event, side of the later event)]
    for s in S:
        if any(b["t"] == s["t"] and b["side"] != s["side"] and 0 < s["t_in"] - b["t_in"] <= WIN for b in B):
            pairs[s["t"]].append((s["t_in"], s["side"]))
    for b in B:
        if any(s["t"] == b["t"] and s["side"] != b["side"] and 0 < b["t_in"] - s["t_in"] <= WIN for s in S):
            pairs[b["t"]].append((b["t_in"], b["side"]))
    return B, S, pairs


def third_friday(y, m):
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def day_features(args):
    d, ev_rows, prior_days, prev_day = args
    cp = os.path.join(CACHE, f"feat-{d}.parquet")
    if os.path.exists(cp):
        return cp
    import check_absorption_rip as A
    import check_hedge_release as HR
    import explore_hedge_release as X
    import macro_calendar as MC
    import market_calendar as CAL
    fb, fp = json.load(open(os.path.join(ROOT, "breach_model_frozen.json"))), json.load(open(os.path.join(ROOT, "pullback_model_frozen.json")))
    a = {tk: ext(d, tk) for tk in TKS}
    pa = {tk: ext(prev_day, tk) for tk in TKS} if prev_day else {}
    t30 = {}
    base_vol = {}
    for tk in TKS:
        pr = [ext(x, tk) for x in prior_days]
        pr = [x for x in pr if x is not None and x[-1, 0] >= 14 * 60]
        r = [A.r30(x[:, [0, 2, 3, 4, 5]]) for x in pr]          # A.r30 wants (m, high, low, close, volume)
        t30[tk] = float(np.median([v for v in r if v])) if r else None
        cums = []
        for x in pr:
            c = np.full(1000, np.nan)
            cv = np.cumsum(x[:, 5])
            c[x[:, 0].astype(int)] = cv
            cums.append(c)
        base_vol[tk] = np.array(cums) if cums else None
    opt = options_day(d)
    expo = {tk: exposures(d, tk) for tk in TKS}
    B, S, pairs = cues_day(d, fb, fp)
    # hedge release (needs the legs cache; built for any month on demand)
    cd = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "option-contracts-1m", "date=*")))
    nxt = next((x for x in cd if x > d), None)
    lp = X.one_day((d, nxt))
    HE = HR.events(pl.read_parquet(lp)) if lp else None
    peak = {}
    zp = os.path.join(ROOT, "_peakzone_cache", "v2", f"{d}.json")
    if os.path.exists(zp):
        zz = json.load(open(zp)).get("v", {})
        for tk in TKS:
            z = (zz.get(f"{tk}+") or {}).get("zone")
            if z and "lo" in z:
                peak[tk] = ((z["lo"] + z["hi"]) / 2, z["sign"])
    base_now = defaultdict(list)
    for r in ev_rows:
        base_now[r["tk"]].append((r["s"], r["e"]))
    day = dt.date.fromisoformat(d)
    tdays = list(CAL.trading_days(day.replace(day=1), (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)))
    tf = third_friday(day.year, day.month)
    opex = day == (tf if CAL.is_trading_day(tf) else tf - dt.timedelta(days=1))
    cal = dict(DOW=day.weekday(), OPEX=float(opex), MONTHEND=float(bool(tdays) and day in tdays[-2:]),
               FOMC=float(bool(MC.is_fomc_day(day))), MACRO_AM=float(bool(MC.is_macro_am_day(day))))
    out = []
    for r in ev_rows:
        tk, s, e = r["tk"], r["s"], r["e"]
        x = a[tk]
        if x is None or t30.get(tk) is None:
            continue
        m = x[:, 0].astype(int)
        idx = {mm: i for i, mm in enumerate(m)}
        ie, is_ = idx.get(e), idx.get(s)
        if ie is None or is_ is None:
            continue
        c, o, h, lo_, v = x[:, 4], x[:, 1], x[:, 2], x[:, 3], x[:, 5]
        unit = r["t30"] * c[ie]
        w = slice(is_, ie + 1)
        top, bot = c[w].max(), c[w].min()
        rng = max(top - bot, 1e-9)
        i15 = idx.get(e - 15, max(0, ie - 15))
        j0 = max(is_, 1)
        dc, vv = c[j0: ie + 1] - c[j0 - 1: ie], v[j0: ie + 1]       # each base minute's close vs the minute before
        vw = (c[: ie + 1] * v[: ie + 1]).sum() / max(v[: ie + 1].sum(), 1)
        pcl = pa.get(tk)[-1, 4] if pa.get(tk) is not None else np.nan
        pret = ((pa[tk][-1, 4] - pa[tk][0, 1]) / (r["t30"] * pa[tk][0, 1])) if pa.get(tk) is not None else np.nan
        f = dict(d=d, tk=tk, s=s, e=e, RIP=float(r["rip"]), YDIR=float(r["dir"] * r["X"]), L=r["L"], VD=r["VD"], G=r["G"],
                 DD=r["DD"],
                 SLOPE15=(c[ie] - c[i15]) / unit, POS=2 * (c[ie] - bot) / rng - 1,
                 EDGE_NET=float((h[w] >= top - 0.1 * rng).sum() - (lo_[w] <= bot + 0.1 * rng).sum()),
                 UPVOL=float((np.sign(dc) * vv).sum() / max(vv.sum(), 1)) if len(dc) else np.nan,
                 HOLD=float((c[w] > o[w]).mean() - (c[w] < o[w]).mean()),
                 DOPEN=(c[ie] - o[0]) / unit, DVWAP=(c[ie] - vw) / unit, DPREV=(c[ie] - pcl) / unit, PRE=(c[is_] - o[0]) / unit,
                 GAP=(o[0] - pcl) / unit, PREVRET=pret, TIGHT=rng / unit, TOD=float(e))
        others = []
        for t2 in TKS:
            if t2 == tk or a[t2] is None or not t30.get(t2):
                continue
            m2 = {mm: i for i, mm in enumerate(a[t2][:, 0].astype(int))}
            j, j15 = m2.get(e), m2.get(e - 15)
            if j is not None and j15 is not None:
                others.append((a[t2][j, 4] - a[t2][j15, 4]) / (t30[t2] * a[t2][j, 4]))
        f["XMOVE15"] = float(np.mean(others)) if others else np.nan
        f["XQUIET"] = float(sum(any(s2 + 59 <= e <= e2 for s2, e2 in base_now[t2]) for t2 in TKS if t2 != tk))
        bv = base_vol.get(tk)
        f["RVOL"] = float(np.cumsum(v[: ie + 1])[-1] / np.nanmedian(bv[:, e])) if bv is not None and np.isfinite(np.nanmedian(bv[:, e])) else np.nan
        ex = last_at(expo[tk], e)
        f["CHARM"], f["VANNA"] = (ex[1], ex[2]) if ex else (np.nan, np.nan)
        fl, atm = opt.get(tk, ({}, {}))
        npv = [fl[k] for k in range(s, e + 1) if k in fl]
        f["NETPREM"] = float(sum(q[0] for q in npv)) if npv else np.nan
        cvs, pvs = sum(q[1] for q in npv), sum(q[2] for q in npv)
        f["CPTILT"] = (cvs - pvs) / (cvs + pvs) if cvs + pvs > 0 else np.nan
        a0, a1 = last_at(atm, s, 5), last_at(atm, e, 5)
        f["IVCHG"] = (a1[1] - a0[1]) if a0 and a1 and a0[1] and a1[1] else np.nan
        f["STRADCHG"] = (a1[0] / a0[0] - 1) if a0 and a1 and a0[0] else np.nan
        if HE is not None and HE.height:
            hz = HE.filter((pl.col("tk") == tk) & (pl.col("mod") >= s) & (pl.col("mod") <= e))
            f["HREL"] = float(hz["D"].sum()) if hz.height else 0.0
        else:
            f["HREL"] = np.nan
        if tk in peak:
            f["PEAKD"], f["PEAKSIGN"] = (c[ie] - peak[tk][0]) / unit, float(peak[tk][1])
        else:
            f["PEAKD"], f["PEAKSIGN"] = np.nan, np.nan
        t0, t1 = s * 60, (e + 1) * 60
        bb = sorted([b for b in B if b["t"] == tk and t0 <= b["t_in"] <= t1], key=lambda b: b["t_in"])
        ss = sorted([q for q in S if q["t"] == tk and t0 <= q["t_in"] <= t1], key=lambda q: q["t_in"])
        pp = sorted([q for q in pairs.get(tk, []) if t0 <= q[0] <= t1])
        f.update(THRU_N=float(len(bb)), THRU_NET=float(sum(b["side"] for b in bb)),
                 THRU_LAST=float(bb[-1]["side"] * bb[-1]["score"]) if bb else 0.0,
                 PULL_N=float(len(ss)), PULL_NET=float(sum(q["side"] for q in ss)),
                 PULL_LAST=float(ss[-1]["side"] * ss[-1]["score"]) if ss else 0.0,
                 TIE_N=float(len(pp)), TIE_LAST=float(pp[-1][1]) if pp else 0.0)
        f.update(cal)
        out.append(f)
    os.makedirs(CACHE, exist_ok=True)
    df = pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()
    df.write_parquet(cp)
    return cp


def build():
    import check_absorption_rip as A
    fp = os.path.join(CACHE, "features.parquet")
    if os.path.exists(fp):
        return pl.read_parquet(fp)
    ev = A.build()
    cue_days = {os.path.basename(p)[:10] for p in glob.glob(os.path.join(ROOT, "_breach_cache", "events", "*.json"))} & \
               {os.path.basename(p)[:10] for p in glob.glob(os.path.join(ROOT, "_structure_ic_cache", "events", "*.json"))}
    ev = ev.filter((pl.col("d") >= LO) & (pl.col("d") <= HI) & pl.col("d").is_in(sorted(cue_days)))
    days_all = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "ext-1m", "date=*")))
    jobs = []
    for (d,), g in sorted(ev.group_by(["d"]), key=lambda t: t[0][0]):
        i = days_all.index(d)
        jobs.append((d, g.to_dicts(), days_all[max(0, i - 20): i], days_all[i - 1] if i else None))
    with ProcessPoolExecutor(6) as ex:
        paths = [p for p in ex.map(day_features, jobs) if p]
    F = pl.concat([pl.read_parquet(p) for p in paths if os.path.getsize(p) > 0], how="diagonal_relaxed")
    F.write_parquet(fp)
    return F


def split(F):
    odd = F["d"].str.slice(5, 2).cast(pl.Int32) % 2 == 1
    return F.filter(odd), F.filter(~odd)


def add_abs(F):
    return F.with_columns([pl.col(k).cast(pl.Float64).abs().alias(f"ABS_{k}") for k in DIRECTIONAL])


def ranker(Fd, cols):
    ref = {}
    for tk in TKS:
        z = Fd.filter(pl.col("tk") == tk)
        ref[tk] = {k: np.sort(z[k].cast(pl.Float64).drop_nulls().drop_nans().to_numpy()) for k in cols}

    def rank(F):
        out = np.full((F.height, len(cols)), 0.5)
        tks = F["tk"].to_list()
        for j, k in enumerate(cols):
            vals = F[k].cast(pl.Float64).to_numpy()
            for i, (tk, v) in enumerate(zip(tks, vals)):
                r = ref[tk][k]
                if v == v and len(r):
                    out[i, j] = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (2 * len(r))
        return out
    return rank


def spearman(x, y):
    from scipy.stats import spearmanr
    v = spearmanr(x, y).statistic
    return float(v) if v == v else 0.0


def day_boot(days, fn, n=2000, seed=SEED):
    ud = sorted(set(days))
    pos = defaultdict(list)
    for i, d in enumerate(days):
        pos[d].append(i)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(ud), len(ud))
        ix = np.concatenate([pos[ud[k]] for k in pick])
        out.append(fn(ix))
    return np.array(out)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    prev = 1.0
    for rank, i in list(enumerate(o))[::-1]:
        prev = min(prev, p[i] * len(p) / (rank + 1))
        q[i] = prev
    return q


def fit_lasso(Xd, yd, weeks):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=5).split(Xd, yd, weeks))
    return LassoCV(cv=folds, random_state=SEED, max_iter=20000).fit(Xd, yd)


def iso_week(d):
    y, w, _ = dt.date.fromisoformat(d).isocalendar()
    return y * 100 + w


def screen(Fd, Fh, target, cols, label):
    rank = ranker(Fd, cols)
    Xd, yd = rank(Fd), Fd[target].cast(pl.Float64).to_numpy()
    dd = Fd["d"].to_list()
    med = sorted(set(dd))[len(set(dd)) // 2]
    h1 = np.array([d < med for d in dd])
    print(f"\n=== {label}: target {target}, {len(cols)} inputs, discovery {Fd.height} bases / {len(set(dd))} sessions")
    # M1 planted
    rng = np.random.default_rng(SEED)
    from scipy.stats import rankdata
    ry = rankdata(yd) / len(yd)
    planted = ry + rng.normal(0, 1.0, len(ry)) * np.std(ry) * 10.0      # corr ~ 1 / sqrt(1 + 10^2) ~ 0.10
    bs = day_boot(dd, lambda ix: spearman(planted[ix], yd[ix]), n=500)
    ic_p = spearman(planted, yd)
    print(f"  M1 planted input: IC {ic_p:+.3f}, p {np.mean(bs <= 0):.3f}  {'OK' if ic_p >= 0.05 and np.mean(bs <= 0) < 0.01 else 'FAIL'}")
    ics, ps, signs = [], [], []
    for j, k in enumerate(cols):
        ic = spearman(Xd[:, j], yd)
        b = day_boot(dd, lambda ix, j=j: spearman(Xd[ix, j], yd[ix]), n=500)
        ps.append(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0))))
        ics.append(ic)
        signs.append((np.sign(spearman(Xd[h1, j], yd[h1])), np.sign(spearman(Xd[~h1, j], yd[~h1]))))
    q = bh(ps)
    order = np.argsort(-np.abs(ics))
    print("  P1 ICs (top 12 by |IC|):")
    for j in order[:12]:
        both = signs[j][0] == signs[j][1] == np.sign(ics[j])
        print(f"    {cols[j]:12s} IC {ics[j]:+.4f}  p {ps[j]:.3f}  q {q[j]:.3f}  halves {'same' if both else 'differ'}"
              f"{'  <- P1' if q[j] < 0.10 and both else ''}")
    weeks = np.array([iso_week(d) for d in dd])
    las = fit_lasso(Xd, yd, weeks)
    kept = [(cols[j], float(las.coef_[j])) for j in np.argsort(-np.abs(las.coef_)) if las.coef_[j] != 0]
    print(f"  P2 Lasso keeps {len(kept)}: " + ", ".join(f"{k} {c:+.3f}" for k, c in kept[:15]))
    # M2 null: shuffled target
    ys = rng.permutation(yd)
    l0 = fit_lasso(Xd, ys, weeks)
    sc0 = Xd @ l0.coef_
    print(f"  M2 shuffled target: Lasso keeps {(l0.coef_ != 0).sum()}, in-sample IC {spearman(sc0, ys) if np.any(l0.coef_) else 0:+.3f}")
    res = dict(target=target, kept=kept, model=las, rank=rank)
    if Fh is not None:
        Xh, yh = rank(Fh), Fh[target].cast(pl.Float64).to_numpy()
        sh = Xh @ las.coef_
        dh = Fh["d"].to_list()
        ic = spearman(sh, yh) if np.any(las.coef_) else 0.0
        b = day_boot(dh, lambda ix: spearman(sh[ix], yh[ix])) if np.any(las.coef_) else np.zeros(10)
        lo, hi = np.percentile(b, [2.5, 97.5])
        res.update(h_ic=ic, h_lo=lo, h_hi=hi, h_p=float(np.mean(b <= 0)))
        print(f"  HOLDOUT ({Fh.height} bases / {len(set(dh))} sessions): Spearman(score, {target}) {ic:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f" p {res['h_p']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    F = add_abs(build())
    Fd, Fh = split(F)
    print(f"features: {F.height} bases, {F['d'].n_unique()} sessions; missing share per input: " +
          ", ".join(f"{k} {F[k].cast(pl.Float64).is_nan().fill_null(True).mean():.0%}" for k in DIRECTIONAL + NONDIR
                    if F[k].cast(pl.Float64).is_nan().fill_null(True).mean() > 0.05))
    if a.discovery:
        screen(Fd, None, "RIP", SIZE_IN, "SIZE")
        screen(Fd, None, "YDIR", DIR_IN, "DIR")
    if a.holdout:
        rp = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(rp):
            sys.exit(f"holdout already scored: {rp}")
        r1 = screen(Fd, Fh, "RIP", SIZE_IN, "SIZE")
        r2 = screen(Fd, Fh, "YDIR", DIR_IN, "DIR")
        qv = bh([r1["h_p"], r2["h_p"]])
        lines = []
        for r, qq in zip((r1, r2), qv):
            ok = bool(r["kept"]) and r["h_ic"] > 0 and r["h_lo"] > 0 and qq <= 0.10
            lines.append(f"  P2 {r['target']}: keeps {len(r['kept'])} ({', '.join(k for k, _ in r['kept'][:10])}) | holdout IC "
                         f"{r['h_ic']:+.4f} [{r['h_lo']:+.4f}, {r['h_hi']:+.4f}] q {qq:.3f}  {'PASS' if ok else 'FAIL'}")
        print("\n".join(lines))
        with open(rp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
