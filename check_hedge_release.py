"""
check_hedge_release.py -- PRE-REGISTRATION, written 2026-11-26 after the step 1-2 exploration (explore_hedge_release.py,
odd months, counts and OI agreement only -- no price outcome looked at). The user's idea: index put hedges trimmed or
rolled down -> dealers who were short those puts buy their stock hedge back -> quiet buying, a candidate for "absorbed,
then a rip" (11-25: the 44.5k-lot SPY desk re-struck Dec put spreads 15 points lower inside the base).

TIMING (user, 2026-11-26: "we're looking for the effect same day as the trimming flow, and the next day's OI to confirm
the mechanism?"): YES. The OUTCOME is same-day price after the print (60 min primary). Next-morning open interest only
LABELS an event as a trim (sold to close) or a write (sold to open). The label is not known live, so only the tape-only
questions (H1, H2, H4) could ever be traded; H3 is the mechanism check.

DATA  lake/silver/trades-core (every SPY / QQQ / IWM option print with the NBBO), option-contracts-1m (delta per minute,
      start-of-day OI; the next session's OI = this session's end), ext-1m (1-min candles). 2023-10-26 .. 2026-11-25.

EVENTS (explore_hedge_release.one_day legs, then):
  legs       put prints >= 500 contracts 09:30-16:15 ET; |delta| >= 0.9 dropped; grouped per underlying and second;
             seconds with calls set aside. Prints of the same contract in the second are summed: side = sign of the
             size-weighted side (+1 at/above ask = customer bought, -1 at/below bid = sold) if |weighted side| >= 0.8,
             else 0 (unsided).
  PACKAGE RULE (the fix, 2026-11-26): a 2-contract event of equal size (within 10%) with exactly one sided leg ->
             the unsided leg takes the opposite side (a vertical / roll trades as one order; the exploration's biggest
             "releases" were such spreads with the second leg counted as 0).
  event      >= 5,000 contracts in total.
  FULLY SIDED (added before commit, 2026-11-26): an event counts toward LARGE only if every leg has a side after the
             package rule -- a partly sided package (e.g. 4 legs, 2 at the NBBO) would count only some legs and its D
             could be far off (the smoke test's biggest SPY event, 118k contracts, was half sided). Discovery share of
             fully sided events before the fix: SPY 63%, QQQ 42%, IWM 36%.
  D          customer put delta SOLD, in shares: sum over legs of (-side) * |delta| * size * 100 = the shares a delta-
             hedging dealer counterparty would BUY. Unsided legs count 0.
  LARGE      RELEASE: D >= the 90th percentile of positive D on DISCOVERY events, per ticker (frozen). ADD: D <= the
             10th percentile of negative D on discovery, per ticker. Events of one kind within 15 min of an earlier one
             (same ticker) are merged into the first.
  OI label   a large release is a TRIM if every sold leg with a next-day OI has dOI <= -0.5 x size (and at least one
             has), a WRITE if every such leg has dOI >= +0.5 x size; else unlabelled (no next-day OI = expired, or mixed).

OUTCOME   R = the ticker's close 60 min after the event minute / close at the event minute - 1, in bp (event minute
          09:35-15:00). Matched control = the mean R over minutes of the same session and ticker 30-90 min away from the
          event (either side), 09:35-15:00, with no large event of either kind within 15 min. Estimand per event =
          R - control.

PRIMARY QUESTIONS (pooled SPY / QQQ / IWM; predicted sign)
  H1  LARGE RELEASE (tape only): mean (R - control) [+].
  H2  LARGE ADD (tape only): mean (R - control) [-]; scored as minus that, so [+].
  H3  mechanism: H1's estimand on TRIMS minus on WRITES [two-sided; ~0 if dealer re-hedging is the whole story, > 0 if
      the trim itself carries information].
  H4  quiet stretches (check_absorption_rip's BASE, imported unchanged) containing a LARGE RELEASE between their start
      and end: rate of rips UP minus the rate in bases without one [+]. (That study's holdout was for its own Q1-Q4;
      this question was never computed.)
REPORTED, NOT TESTED  R at 30 / 120 min and to the close; H1 by gamma sign; next-session return; counts.

SPLIT     DISCOVERY = odd calendar months (the exploration months), HOLDOUT = even months, scored once (--holdout writes
          _hedge_cache/RESULT-holdout.txt, refuses a second run). Thresholds frozen from discovery.
STATS     95% CI from a day-block bootstrap (2,000 resamples of sessions); one-sided p for H1/H2/H4, two-sided for H3; BH
          across H1-H4.
PASS      holdout on the predicted side (H3: either side), 95% CI excluding 0, BH q <= 0.10, discovery on the same side.
NULL      on discovery before the holdout: every event moved to a random minute of the same session (200 draws) --
          H1 / H2 / H4 must centre on 0 (|mean| < a quarter of the discovery CI half-width), else re-based on the null
          mean as a documented deviation before the holdout.
TRADING USE  none until a PASS on H1 or H2 (H3 is not knowable live).

DISCOVERY (2026-11-26, _hedge_discovery.log; 384 sessions, 7,803 fully sided events; large releases SPY 143 / QQQ 60 /
  IWM 117, large adds 218 / 76 / 128; OI labels on releases: trims 58, writes 116):
    H1 release  -1.05 bp [-6.08, +3.97]  null -0.65 OK        -- wrong side of 0: cannot pass
    H2 add      -0.39 bp [-4.49, +4.33]  null +0.48 OK        -- wrong side of 0: cannot pass
    H3 trims - writes  -8.1 bp [-29.3, +12.4] (two-sided)
    H4 base rip-UP diff  +0.002 [-0.015, +0.022]  null +0.008 NOT CENTRED
  DEVIATION (before any holdout session was built): H4 is re-based on its null mean, +0.0080 -- the holdout estimand
  becomes (rip-UP rate with a release - without) - 0.0080; discovery re-based = -0.006, wrong side, so it cannot pass.
  Descriptive R vs control: 30 min release -0.04 / add +0.02 bp; 120 min +1.88 / +1.17 bp. Nothing in discovery.

    python check_hedge_release.py --discovery
    python check_hedge_release.py --holdout          (once)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_hedge_cache")
TKS = ("SPY", "QQQ", "IWM")
EVENT_MIN, DEEP, CONF, SIDE_W, EQ_TOL, MERGE_MIN = 5000, 0.9, 0.5, 0.8, 0.10, 15
H, CTRL_LO, CTRL_HI, M0, M1 = 60, 30, 90, 9 * 60 + 35, 15 * 60
H4_NULL = 0.0080          # DEVIATION 2026-11-26: H4 re-based on its discovery null mean (see the docstring)


def half_of(d):
    return "discovery" if int(d[5:7]) % 2 == 1 else "holdout"


def legs_for(days):
    import explore_hedge_release as X
    cdays = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "option-contracts-1m", "date=*")))
    nxt = {d: next((x for x in cdays if x > d), None) for d in days}
    with ProcessPoolExecutor(6) as ex:
        paths = [p for p in ex.map(X.one_day, [(d, nxt[d]) for d in days]) if p]
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def events(L):
    L = L.filter((pl.col("delta_close").abs() < DEEP) & pl.col("delta_close").is_not_null())
    calls = L.group_by("d", "underlying_symbol", "sec").agg(pl.col("call_same_sec").any().alias("calls"),
                                                           pl.col("size").sum().alias("tot"))
    L = L.join(calls.filter(pl.col("tot") >= EVENT_MIN).select("d", "underlying_symbol", "sec"),
               on=["d", "underlying_symbol", "sec"], how="semi")
    g = (L.group_by("d", "underlying_symbol", "sec", "cid")
         .agg(pl.col("size").sum(), (pl.col("side").cast(pl.Float64) * pl.col("size")).sum().alias("sw"),
              pl.col("delta_close").mean().alias("delta"), pl.col("mod").first(),
              (pl.col("oi1") - pl.col("oi0")).first().alias("dOI"))
         .with_columns(side=pl.when(pl.col("sw") / pl.col("size") >= SIDE_W).then(1)
                       .when(pl.col("sw") / pl.col("size") <= -SIDE_W).then(-1).otherwise(0)))
    out = []
    for (d, u, sec), x in g.group_by(["d", "underlying_symbol", "sec"]):
        n = int(x["size"].sum())
        if n < EVENT_MIN:
            continue
        sz, sd = x["size"].to_numpy(), x["side"].to_numpy().copy()
        if len(sz) == 2 and abs(sz[0] - sz[1]) <= EQ_TOL * max(sz) and (sd != 0).sum() == 1:
            k = int(np.flatnonzero(sd != 0)[0])
            sd[1 - k] = -sd[k]                                          # PACKAGE RULE
        dl, doi = np.abs(x["delta"].to_numpy()), x["dOI"].to_numpy()
        D = float((-sd * dl * sz * 100).sum())
        sold = sd == -1
        known = sold & ~np.isnan(doi.astype(float))
        trim = bool(known.any() and np.all(doi[known] <= -CONF * sz[known]))
        write = bool(known.any() and np.all(doi[known] >= CONF * sz[known]))
        out.append(dict(d=d, tk=u, sec=sec, mod=int(x["mod"].min()), n=n, D=D if (sd != 0).all() else None,
                        trim=trim, write=write, sided=float((sd != 0).mean())))
    E = pl.DataFrame(out).join(calls, left_on=["d", "tk", "sec"], right_on=["d", "underlying_symbol", "sec"], how="left")
    return E.filter(~pl.col("calls").fill_null(False) & pl.col("D").is_not_null()).drop("calls").sort("d", "tk", "mod")


def cuts_from(Ed):
    c = {}
    for tk in TKS:
        x = Ed.filter(pl.col("tk") == tk)
        c[tk] = (float(np.percentile(x.filter(pl.col("D") > 0)["D"], 90)), float(np.percentile(x.filter(pl.col("D") < 0)["D"], 10)))
    return c


def large(E, cuts):
    rows = []
    for (tk, d), x in E.group_by(["tk", "d"]):
        last = {"rel": -999, "add": -999}
        for r in x.sort("mod").iter_rows(named=True):
            kind = "rel" if r["D"] >= cuts[tk][0] else ("add" if r["D"] <= cuts[tk][1] else None)
            if kind is None:
                continue
            if r["mod"] - last[kind] < MERGE_MIN:
                continue
            last[kind] = r["mod"]
            rows.append(dict(r, kind=kind))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


_PX = {}


def closes(d, tk):
    """close by minute of day as a 1,440-long array (nan where no bar)."""
    k = (d, tk)
    if k not in _PX:
        p = os.path.join(LAKE, "ext-1m", f"date={d}", f"{tk}.parquet")
        if not os.path.exists(p):
            _PX[k] = None
        else:
            x = (pl.read_parquet(p).filter(pl.col("market_time") == "r")
                 .with_columns(m=(pl.col("minute").dt.hour().cast(pl.Int32) * 60 + pl.col("minute").dt.minute().cast(pl.Int32))))
            a = np.full(1440 + 240, np.nan)
            a[x["m"].to_numpy()] = x["close"].to_numpy()
            _PX[k] = a
    return _PX[k]


def outcomes(LG, h=H, shift=None):
    """per large event: R - matched control. shift: {(d,tk,mod) -> new minute} for the null."""
    busy = {}
    for r in LG.iter_rows(named=True):
        busy.setdefault((r["d"], r["tk"]), []).append(shift[(r["d"], r["tk"], r["mod"])] if shift else r["mod"])
    res = []
    for r in LG.iter_rows(named=True):
        m = shift[(r["d"], r["tk"], r["mod"])] if shift else r["mod"]
        if not (M0 <= m <= M1):
            continue
        px = closes(r["d"], r["tk"])
        if px is None:
            continue
        x = (px[m + h] / px[m] - 1) * 1e4
        if x != x:
            continue
        c = np.r_[np.arange(m - CTRL_HI, m - CTRL_LO + 1), np.arange(m + CTRL_LO, m + CTRL_HI + 1)]
        c = c[(c >= M0) & (c <= M1)]
        bz = np.array(busy[(r["d"], r["tk"])])
        c = c[np.all(np.abs(c[:, None] - bz[None, :]) > MERGE_MIN, axis=1)] if len(c) else c
        rc = (px[c + h] / px[c] - 1) * 1e4 if len(c) else np.array([])
        rc = rc[rc == rc]
        if len(rc):
            res.append(dict(d=r["d"], tk=r["tk"], kind=r["kind"], trim=r["trim"], write=r["write"], y=float(x - rc.mean())))
    return pl.DataFrame(res, schema={"d": pl.String, "tk": pl.String, "kind": pl.String, "trim": pl.Boolean,
                                     "write": pl.Boolean, "y": pl.Float64})


def h4_table(LG, sessions):
    import check_absorption_rip as A
    B = A.build().filter(pl.col("d").is_in(sessions))
    rel = LG.filter(pl.col("kind") == "rel")
    S = {}
    for r in rel.iter_rows(named=True):
        S.setdefault((r["d"], r["tk"]), []).append(r["mod"])
    return B.with_columns(has=pl.struct("d", "tk", "s", "e").map_elements(
        lambda b: any(b["s"] <= m <= b["e"] for m in S.get((b["d"], b["tk"]), [])), return_dtype=pl.Boolean),
        up=pl.col("rip") & (pl.col("dir") > 0))


def est(q, O, B4):
    if q == "H1":
        y = O.filter(pl.col("kind") == "rel")["y"]
        return float(y.mean()) if len(y) else np.nan
    if q == "H2":
        y = O.filter(pl.col("kind") == "add")["y"]
        return -float(y.mean()) if len(y) else np.nan
    if q == "H3":
        a = O.filter((pl.col("kind") == "rel") & pl.col("trim"))["y"]
        b = O.filter((pl.col("kind") == "rel") & pl.col("write"))["y"]
        return float(a.mean()) - float(b.mean()) if len(a) and len(b) else np.nan
    if q == "H4":
        a, b = B4.filter(pl.col("has"))["up"], B4.filter(~pl.col("has"))["up"]
        return (float(a.cast(pl.Float64).mean()) - float(b.cast(pl.Float64).mean()) - H4_NULL) if len(a) and len(b) else np.nan
    raise ValueError(q)


QS = (("H1", "large RELEASE -> 60-min return vs control (bp)"), ("H2", "large ADD -> minus 60-min return vs control (bp)"),
      ("H3", "trims minus writes, H1 estimand (bp, two-sided)"), ("H4", "quiet stretch with a release: rip-UP rate diff"))


def boot(q, O, B4, n=2000, seed=7):
    days = sorted(set(O["d"].to_list()) | set(B4["d"].to_list()))
    Od = {k[0] if isinstance(k, tuple) else k: v for k, v in O.partition_by("d", as_dict=True).items()}
    Bd = {k[0] if isinstance(k, tuple) else k: v for k, v in B4.partition_by("d", as_dict=True).items()}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        o = [Od[d] for d in pick if d in Od]
        b = [Bd[d] for d in pick if d in Bd]
        bs.append(est(q, pl.concat(o) if o else O.clear(), pl.concat(b) if b else B4.clear()))
    bs = np.array([b for b in bs if b == b])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p1 = float(np.mean(bs <= 0))
    return lo, hi, (min(1.0, 2 * min(p1, 1 - p1)) if q == "H3" else p1)


def null(LG, sessions, draws=200, seed=11):
    rng = np.random.default_rng(seed)
    out = {q: [] for q in ("H1", "H2", "H4")}
    for _ in range(draws):
        shift = {(r["d"], r["tk"], r["mod"]): int(rng.integers(M0, M1 + 1)) for r in LG.iter_rows(named=True)}
        O = outcomes(LG, shift=shift)
        LG2 = LG.with_columns(pl.Series("mod", [shift[(a, b, c)] for a, b, c in LG.select("d", "tk", "mod").iter_rows()]))
        B4 = h4_table(LG2, sessions)
        for q in out:
            out[q].append(est(q, O, B4))
    return {q: (float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for q, v in out.items()}


def describe(E, LG, O, label):
    n_s = E["d"].n_unique()
    print(f"--- {label}: {n_s} sessions, {E.height:,} events >= {EVENT_MIN:,} put contracts (package rule applied)")
    for tk in TKS:
        x = LG.filter(pl.col("tk") == tk) if LG.height else LG
        r, a = x.filter(pl.col("kind") == "rel"), x.filter(pl.col("kind") == "add")
        print(f"  {tk}: large releases {r.height} ({r.height / n_s:.2f}/session; trims {r['trim'].sum()}, writes "
              f"{r['write'].sum()}), median D {r['D'].median() or 0:,.0f} sh | large adds {a.height}, median D {a['D'].median() or 0:,.0f} sh")
    for h in (30, 120):
        Oh = outcomes(LG, h=h)
        print(f"  R vs control at {h} min (descriptive): release {Oh.filter(pl.col('kind') == 'rel')['y'].mean():+.2f} bp, "
              f"add {Oh.filter(pl.col('kind') == 'add')['y'].mean():+.2f} bp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "trades-core", "date=*")))
    disc = [d for d in days if half_of(d) == "discovery"]
    Ed = events(legs_for(disc))
    cuts = cuts_from(Ed)
    print("frozen cuts (discovery, shares of dealer buy):", {k: (round(v[0]), round(v[1])) for k, v in cuts.items()})
    LGd = large(Ed, cuts)
    Od, B4d = outcomes(LGd), h4_table(LGd, disc)
    if a.discovery:
        describe(Ed, LGd, Od, "DISCOVERY")
        nl = null(LGd, disc)
        for q, name in QS:
            e = est(q, Od, B4d)
            lo, hi, p = boot(q, Od, B4d, n=500)
            extra = ""
            if q in nl:
                m, l2, h2 = nl[q]
                extra = f" | null {m:+.3f} [{l2:+.3f}, {h2:+.3f}] " + ("OK" if abs(m) < (hi - lo) / 8 else "NOT CENTRED")
            print(f"  {q} {name:50s} discovery {e:+.3f} [{lo:+.3f}, {hi:+.3f}]{extra}")
    if a.holdout:
        rp = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(rp):
            sys.exit(f"holdout already scored: {rp}")
        hold = [d for d in days if half_of(d) == "holdout"]
        Eh = events(legs_for(hold))
        LGh = large(Eh, cuts)
        Oh, B4h = outcomes(LGh), h4_table(LGh, hold)
        describe(Eh, LGh, Oh, "HOLDOUT")
        res = []
        for q, name in QS:
            ed, eh = est(q, Od, B4d), est(q, Oh, B4h)
            lo, hi, p = boot(q, Oh, B4h)
            res.append([q, name, ed, eh, lo, hi, p])
        ps = np.array([r[-1] for r in res])
        order = np.argsort(ps)
        qv, prev = np.empty(len(ps)), 1.0
        for rank, i in list(enumerate(order))[::-1]:
            prev = min(prev, ps[i] * len(ps) / (rank + 1))
            qv[i] = prev
        lines = []
        for (q, name, ed, eh, lo, hi, p), qq in zip(res, qv):
            if q == "H3":
                ok = (lo > 0 or hi < 0) and qq <= 0.10 and np.sign(ed) == np.sign(eh)
            else:
                ok = eh > 0 and lo > 0 and qq <= 0.10 and ed > 0
            lines.append(f"  {q} {name:50s} holdout {eh:+.3f} [{lo:+.3f}, {hi:+.3f}] p {p:.3f} q {qq:.3f} (disc {ed:+.3f})  "
                         f"{'PASS' if ok else 'FAIL'}")
        print("\n".join(lines))
        with open(rp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
