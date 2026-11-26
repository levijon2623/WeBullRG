"""
check_absorption_rip.py -- PRE-REGISTRATION, written 2026-11-26 BEFORE any precursor was looked at (user: "go ahead with
the absorption thresholds as proposed"). The user's squeeze question, index version: a quiet stretch in which buying
is absorbed, then a rip with no news (2026-11-25: SPY 698-700 10:30-11:55, then 699 -> 710 by 12:45 -> 718.4).

WHAT HAS BEEN SEEN (so it cannot be "discovered" here): the 11-25 example in detail (ES net buying every 15-min block
while flat; SPY stock delta two-sided, +0.04 of volume; SPY volume 604k -> 303k per 15 min, then 560k on the break),
and a calibration table of how often a quiet stretch is followed by a big move (SPY: a 45-min stretch within 1.0-2.0
typical ranges is on 87-99% of sessions; one followed by >= 3 typical ranges within 60 min on 6-13%). No precursor has
been computed on any other day. 11-25 is in an odd month, so it sits in DISCOVERY.

DATA   lake/silver/ext-1m (UW 1-min candles, RTH 'r' rows), SPY / QQQ / IWM, 2023-11-01 .. 2026-11-25 (770 sessions);
       half days (RTH ends before 14:00) dropped. 0DTE option bars lake/silver/option-contracts-1m (ask/bid volume,
       delta). Dealer gamma lake/silver/spot-exposures-1m (from 2024-08-20). ES/NQ/RTY aggressor flow
       lake/silver/futures-1s (from 2026-07-15; descriptive only).

DEFINITIONS (thresholds as proposed 2026-11-26, never retuned)
  T30     the typical 30-minute range: median over the PRIOR 20 full sessions of (high - low) / close of the
          non-overlapping 30-min RTH blocks 10:00-15:30. Everything below is in T30 units of that session.
  BASE    from 10:00, the first minute s (s <= 14:00) whose next 60 minutes of closes stay within a band of
          BAND_T30 = 2.5 T30 (max close - min close <= 2.5 * T30 * close at s); extended minute by minute while the
          closes s..e stay within that band and e <= 15:00. The base ends at e; the search resumes at e + 1 (bases
          never overlap). Base length L = e - s + 1 minutes (>= 60).
  OUTCOME the excursion beyond the base in the 60 minutes after e: X = max(max close (e, e+60] - base max close,
          base min close - min close (e, e+60]) / (T30 * close at e). RIP = X >= RIP_T30 = 3. Direction = the side
          of the larger excursion.

PRIMARY QUESTIONS (pooled over SPY / QQQ / IWM base events; predicted sign in brackets)
  Q1 size    VOLUME DRY-UP: VD = volume of the base's last 30 min / its first 30 min. RIP rate of the lowest tercile
             of VD minus the highest tercile [+]. Tercile cuts frozen from the discovery events.
  Q2 size    BASE LENGTH: RIP rate of the longest tercile of L minus the shortest [+]. Cuts frozen from discovery.
  Q3 size    DEALER GAMMA at the base end (gamma_per_one_percent_move_oi): RIP rate when negative minus when
             positive [+]. Sessions from 2024-08-20 only.
  Q4 direction  0DTE OPTIONS DELTA BOUGHT IN THE BASE: DD = sum over the ticker's own 0DTE contracts and the base's
             minutes of (ask_volume - bid_volume) * delta * 100 (share-equivalents; ETF options only, never the index
             blend). Among RIP events with DD != 0: share whose rip goes the way of sign(DD), minus 0.5 [+].
REPORTED, NOT TESTED
  ANATOMY    bases and rips per session, X distribution, rip direction, time of the break, give-back (share of the
             rip's 60-min excursion retraced by e + 120), by ticker and gamma sign.
  ES delta   futures aggressor buy - sell over the base (ES for SPY, NQ for QQQ, RTY for IWM) vs rip direction:
             ~85 sessions, descriptive only.

SPLIT     DISCOVERY = sessions in ODD calendar months, HOLDOUT = EVEN months. Discovery: everything printed, code
          checked, null checks run. Holdout: --holdout scores once, writes _absorb_cache/RESULT-holdout.txt and refuses
          to run again.
STATS     estimand per question as above; 95% CI from a day-block bootstrap (2,000 resamples of sessions, all tickers'
          events of a session move together); one-sided p = share of resamples on the wrong side of 0; BH across
          Q1-Q4.
PASS      holdout estimand on the predicted side, its 95% CI excluding 0, BH q <= 0.10, AND discovery on the same side.
NULL      before the holdout, on discovery: each question with its predictor permuted across events within ticker (200
          draws) must centre on 0 (|mean| < a quarter of the discovery CI half-width); otherwise the estimand is
          re-based on the null mean, documented as a deviation BEFORE the holdout (the peak-zone lesson, 11-26).
TRADING USE  none until a PASS. Q1-Q3 are magnitude forecasts, which earlier tests (straddle, vol gauge) found priced
          in; Q4 is the one that would matter for a trade.

    python check_absorption_rip.py --discovery
    python check_absorption_rip.py --holdout          (once)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_absorb_cache")
TKS = ("SPY", "QQQ", "IWM")
FUT = {"SPY": "ES", "QQQ": "NQ", "IWM": "RTY"}
BAND_T30, RIP_T30 = 2.5, 3.0
BASE_MIN, START_MIN, START_MAX, END_MAX, AFTER = 60, 10 * 60, 14 * 60, 15 * 60, 60
T30_N = 20


def rth(d, tk):
    p = os.path.join(LAKE, "ext-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(p):
        return None
    x = (pl.read_parquet(p).filter(pl.col("market_time") == "r")
         .with_columns(m=(pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(pl.Int32)))
         .sort("m"))
    return x.select("m", "high", "low", "close", "volume").to_numpy() if x.height else None


def r30(a):
    out = []
    for m0 in range(600, 930, 30):
        w = a[(a[:, 0] >= m0) & (a[:, 0] < m0 + 30)]
        if len(w) > 20:
            out.append((w[:, 1].max() - w[:, 2].min()) / w[-1, 3])
    return float(np.median(out)) if out else None


def bases(a, t30):
    m, c, v = a[:, 0], a[:, 3], a[:, 4]
    out, i, n = [], int(np.searchsorted(m, START_MIN)), len(m)
    while i < n and m[i] <= START_MAX:
        j = int(np.searchsorted(m, m[i] + BASE_MIN)) - 1
        if j >= n or m[j] - m[i] < BASE_MIN - 5:
            i += 1
            continue
        band = BAND_T30 * t30 * c[i]
        if c[i:j + 1].max() - c[i:j + 1].min() > band:
            i += 1
            continue
        while j + 1 < n and m[j + 1] <= END_MAX and max(c[i:j + 2].max(), 0) - c[i:j + 2].min() <= band:
            j += 1
        k1 = int(np.searchsorted(m, m[j] + AFTER, side="right"))
        nxt = c[j + 1:k1]
        hi, lo = c[i:j + 1].max(), c[i:j + 1].min()
        if len(nxt) < 30:
            break
        up, dn = nxt.max() - hi, lo - nxt.min()
        unit = t30 * c[j]
        X = max(up, dn) / unit
        d = 1 if up >= dn else -1
        # give-back by e + 120: share of the 60-min excursion retraced
        k2 = int(np.searchsorted(m, m[j] + 120, side="right"))
        peak = (nxt.max() if d > 0 else nxt.min())
        later = c[k2 - 1] if k2 - 1 > j else peak
        exc = (peak - hi) if d > 0 else (lo - peak)
        gb = ((peak - later) / exc if d > 0 else (later - peak) / exc) if exc > 0 else np.nan
        s30 = v[(m >= m[i]) & (m < m[i] + 30)].sum()
        e30 = v[(m > m[j] - 30) & (m <= m[j])].sum()
        out.append(dict(s=int(m[i]), e=int(m[j]), L=int(m[j] - m[i] + 1), X=float(X), rip=X >= RIP_T30, dir=d,
                        VD=float(e30 / s30) if s30 > 0 else np.nan, gb=float(gb)))
        i = j + 1
    return out


def dd_day(d):
    """per ticker, per minute-of-day: 0DTE (ask - bid) * delta * 100 summed over the ticker's own contracts."""
    p = os.path.join(LAKE, "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(p):
        return {}
    y = (pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(list(TKS)) & (pl.col("expiry") == pl.lit(d).str.to_date()))
         .with_columns(mm=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32)),
                       dd=(pl.col("ask_volume") - pl.col("bid_volume")).cast(pl.Float64) * pl.col("delta_close").fill_null(0.0) * 100)
         .group_by("underlying_symbol", "mm").agg(pl.col("dd").sum()).collect())
    out = {}
    for tk, mm, dd in y.iter_rows():
        out.setdefault(tk, {})[int(mm)] = float(dd)
    return out


def gamma_day(d, tk):
    p = os.path.join(LAKE, "spot-exposures-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(p):
        return None
    y = pl.read_parquet(p).select("minute_et", "gamma_per_one_percent_move_oi")
    y = y.with_columns(mm=(pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32)))
    return {int(a): float(b) for a, b in y.select("mm", "gamma_per_one_percent_move_oi").iter_rows() if b is not None}


def fut_day(d, tk):
    p = os.path.join(LAKE, "futures-1s", f"date={d}", f"{FUT[tk]}.parquet")
    if not os.path.exists(p):
        return None
    x = pl.read_parquet(p).with_columns(mm=(pl.col("sec") // 60).cast(pl.Int32))
    return {int(a): float(b) for a, b in x.group_by("mm").agg((pl.col("buy").sum() - pl.col("sell").sum())).iter_rows()}


def build():
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, "events.parquet")
    if os.path.exists(cp):
        return pl.read_parquet(cp).with_columns(pl.col("rip").cast(pl.Boolean))
    days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "ext-1m", "date=*")))
    A = {tk: {} for tk in TKS}
    for d in days:
        for tk in TKS:
            a = rth(d, tk)
            if a is not None and len(a) > 300 and a[-1, 0] >= 14 * 60:
                A[tk][d] = a
    rows = []
    for d_i, d in enumerate(days):
        dd = None
        for tk in TKS:
            prior = [x for x in days[:d_i] if x in A[tk]][-T30_N:]
            if d not in A[tk] or len(prior) < T30_N:
                continue
            t30 = float(np.median([r30(A[tk][x]) for x in prior]))
            ev = bases(A[tk][d], t30)
            if not ev:
                continue
            if dd is None:
                dd = dd_day(d)
            g = gamma_day(d, tk)
            f = fut_day(d, tk)
            for e in ev:
                ddv = dd.get(tk, {})
                e.update(d=d, tk=tk, t30=t30,
                         DD=sum(ddv.get(mm, 0.0) for mm in range(e["s"], e["e"] + 1)) if ddv else None,
                         G=(next((g[mm] for mm in range(e["e"], e["e"] - 6, -1) if mm in g), None) if g else None),
                         FD=sum(f.get(mm, 0.0) for mm in range(e["s"], e["e"] + 1)) if f else None)
                rows.append(e)
        if d_i % 50 == 0:
            print(f"  built {d_i}/{len(days)} sessions, {len(rows)} base events", flush=True)
    ev = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("rip").cast(pl.Boolean))
    ev.write_parquet(cp)
    return ev


def half(ev, which):
    odd = ev["d"].str.slice(5, 2).cast(pl.Int32) % 2 == 1
    return ev.filter(odd if which == "discovery" else ~odd)


def rate(x):
    x = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x)
    return float(np.mean(x.astype(float))) if len(x) else np.nan


def q_est(ev, q, cuts):
    if q == "Q1":
        x = ev.filter(pl.col("VD").is_not_nan())
        return rate(x.filter(pl.col("VD") <= cuts["VD"][0])["rip"]) - rate(x.filter(pl.col("VD") > cuts["VD"][1])["rip"])
    if q == "Q2":
        return rate(ev.filter(pl.col("L") > cuts["L"][1])["rip"]) - rate(ev.filter(pl.col("L") <= cuts["L"][0])["rip"])
    if q == "Q3":
        x = ev.filter(pl.col("G").is_not_null())
        return rate(x.filter(pl.col("G") < 0)["rip"]) - rate(x.filter(pl.col("G") >= 0)["rip"])
    if q == "Q4":
        x = ev.filter(pl.col("rip") & pl.col("DD").is_not_null() & (pl.col("DD") != 0))
        return rate((np.sign(x["DD"].to_numpy()) == x["dir"].to_numpy())) - 0.5
    raise ValueError(q)


QS = (("Q1", "volume dry-up: RIP rate low - high VD tercile"), ("Q2", "base length: RIP rate long - short tercile"),
      ("Q3", "dealer gamma: RIP rate negative - positive"), ("Q4", "0DTE delta in base -> rip direction, hit - 0.5"))


def boot(ev, q, cuts, n=2000, seed=7):
    days = ev["d"].unique().sort().to_list()
    by = ev.partition_by("d", as_dict=True)
    by = {k[0] if isinstance(k, tuple) else k: v for k, v in by.items()}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        bs.append(q_est(pl.concat([by[d] for d in pick]), q, cuts))
    bs = np.array([b for b in bs if b == b])
    return np.percentile(bs, [2.5, 97.5]), float(np.mean(bs <= 0))


def null(ev, q, cuts, draws=200, seed=11):
    col = {"Q1": "VD", "Q2": "L", "Q3": "G", "Q4": "DD"}[q]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        parts = []
        for tk in TKS:
            x = ev.filter(pl.col("tk") == tk)
            parts.append(x.with_columns(pl.Series(col, rng.permutation(x[col].to_numpy()), dtype=x[col].dtype)))
        out.append(q_est(pl.concat(parts), q, cuts))
    a = np.array([o for o in out if o == o])
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def anatomy(ev, label):
    print(f"--- anatomy ({label}): {ev.height} base events, {ev['d'].n_unique()} sessions")
    for tk in TKS:
        x = ev.filter(pl.col("tk") == tk)
        r = x.filter(pl.col("rip"))
        if not x.height:
            continue
        ns = x["d"].n_unique()
        print(f"  {tk}: {x.height / ns:.2f} bases/session, RIP {r.height} ({r.height / x.height:.0%} of bases, "
              f"{r['d'].n_unique() / ns:.0%} of sessions) | X median {x['X'].median():.2f}, rips {r['X'].median() if r.height else float('nan'):.2f} | "
              f"rips up {rate(r['dir'].to_numpy() > 0):.0%} | break at {int(r['e'].median()) // 60 if r.height else 0}:"
              f"{int(r['e'].median()) % 60 if r.height else 0:02d} median | give-back by +120 {r['gb'].median() if r.height else float('nan'):.0%}"
              f" | base L median {x['L'].median():.0f} min")
    g = ev.filter(pl.col("G").is_not_null())
    for lab, c in (("gamma < 0", pl.col("G") < 0), ("gamma >= 0", pl.col("G") >= 0)):
        y = g.filter(c)
        print(f"  {lab}: {y.height} bases, RIP {rate(y['rip']):.1%}")
    f = ev.filter(pl.col("FD").is_not_null() & pl.col("rip"))
    if f.height:
        print(f"  futures delta in base vs rip direction (descriptive): {f.height} rips, agree "
              f"{rate(np.sign(f['FD'].to_numpy()) == f['dir'].to_numpy()):.0%}")


def score(ev_d, ev_h=None):
    cuts = {"VD": tuple(np.nanpercentile(ev_d["VD"].to_numpy(), [100 / 3, 200 / 3])),
            "L": tuple(np.percentile(ev_d["L"].to_numpy(), [100 / 3, 200 / 3]))}
    print(f"frozen cuts (discovery): VD {cuts['VD'][0]:.3f} / {cuts['VD'][1]:.3f}  L {cuts['L'][0]:.0f} / {cuts['L'][1]:.0f} min")
    res = []
    for q, name in QS:
        ed = q_est(ev_d, q, cuts)
        (lo_d, hi_d), _ = boot(ev_d, q, cuts, n=500)
        line = f"  {q} {name:48s} discovery {ed:+.4f} [{lo_d:+.4f}, {hi_d:+.4f}]"
        if ev_h is None:
            nm, nlo, nhi = null(ev_d, q, cuts)
            flag = "OK" if abs(nm) < (hi_d - lo_d) / 8 else "NOT CENTRED -> re-base before the holdout"
            print(line + f" | null {nm:+.4f} [{nlo:+.4f}, {nhi:+.4f}] {flag}")
        else:
            eh = q_est(ev_h, q, cuts)
            (lo, hi), p = boot(ev_h, q, cuts)
            res.append((q, name, ed, eh, lo, hi, p))
    if ev_h is not None:
        ps = np.array([r[-1] for r in res])
        order = np.argsort(ps)
        qv = np.empty(len(ps))
        prev = 1.0
        for rank, i in list(enumerate(order))[::-1]:
            prev = min(prev, ps[i] * len(ps) / (rank + 1))
            qv[i] = prev
        out = []
        for (q, name, ed, eh, lo, hi, p), qq in zip(res, qv):
            ok = eh > 0 and lo > 0 and qq <= 0.10 and ed > 0
            out.append(f"  {q} {name:48s} holdout {eh:+.4f} [{lo:+.4f}, {hi:+.4f}] p {p:.3f} q {qq:.3f} "
                       f"(disc {ed:+.4f})  {'PASS' if ok else 'FAIL'}")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    ev = build()
    ev_d = half(ev, "discovery")
    if a.discovery:
        anatomy(ev_d, "discovery")
        score(ev_d)
    if a.holdout:
        rp = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(rp):
            sys.exit(f"holdout already scored: {rp}")
        ev_h = half(ev, "holdout")
        anatomy(ev_h, "holdout")
        lines = score(ev_d, ev_h)
        print("\n".join(lines))
        with open(rp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
