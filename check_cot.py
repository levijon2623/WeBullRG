"""
check_cot.py -- PRE-REGISTERED test of CFTC positioning (TFF) on the index ETFs. Hypotheses agreed with the user
2026-11-14 from the common themes in published COT guidance, written here BEFORE any price was joined to the
positioning data. Scored ONCE (--run writes _cot_cache/RESULT-cot.txt and refuses a second run).

INPUTS (built separately, positioning only): _cot_cache/cot_weekly.parquet (build_cot_features.py: TFF, Micros
folded in at 1/10, Russell CME/ICE stitched, 3-yr rolling percentiles), _cot_cache/prices_daily.json (SPY/QQQ/IWM
daily from UW 2007-, VIX from Cboe), _cot_cache/optflow_daily.parquet (build_cot_optflow.py, H5 only).

SAMPLE: one observation per weekly report. Entry = the close of its `effective` session (the first session after
the Friday 15:30 ET release; a day later in federal-holiday weeks) -- conservative. Reports flagged
release_uncertain (federal-shutdown backlogs) and rows without a 3-yr percentile are dropped. Forward returns are
close to close over N trading sessions (4 weeks = 20, 8 weeks = 40). Prices are not dividend-adjusted (about
0.1-0.2% over these horizons, in signal and baseline weeks alike). All features are futures-only (`_fut`).

PERIODS: DISCOVERY 2008-06 .. 2017-12, HOLDOUT 2018-01 .. the last report with a complete horizon.

HYPOTHESES (signal = a contract's own trailing 3-year percentile; "effect" = mean outcome in signal weeks minus
the baseline mean in the same period)
  H1 crowded-short squeeze: ES leveraged-fund net (% of OI) in its lowest 10% (pct156 <= 0.10)
       -> SPY next-20-session return ABOVE the all-weeks mean.                                      predicted +
  H2 real money thinning in an uptrend: ES asset-manager 4-week change in its lowest 10% (chg4_pct <= 0.10)
       while SPY is up over the last 63 sessions -> SPY next-40-session return BELOW the mean of all uptrend weeks.
                                                                                                   predicted -
  H3 rotation: s = RTY asset-manager pct156 - NQ asset-manager pct156, ranked against its own trailing 156
       reports; top 10% vs bottom 10% -> (IWM - QQQ) next-40-session return, top minus bottom.        predicted +
  H4 VIX coiled spring: VIX-futures leveraged-fund net in its lowest 10% (most short)
       -> max VIX close over the next 20 sessions / VIX at entry - 1, ABOVE the all-weeks mean.     predicted +
  PASS (each of H1-H4): the HOLDOUT effect's 95% CI excludes 0 on the predicted side, BH q < 0.10 across H1-H4
       (holdout p-values), and the DISCOVERY point estimate has the predicted sign.
  Confidence intervals: block bootstrap by calendar quarter (2,000 resamples). Episodes (signal runs separated by
  more than 4 weeks) are reported, because a 10% tail clusters into few independent events.

  H5 (DESCRIPTIVE, no pass/fail -- the options tape starts 2023-10, about 3 years): in weeks the lake covers,
       does options flow into the same horizon (net delta bought, 14-91 days to expiry, the week before entry)
       AGREE with each signal? H1 agrees if S&P options bought delta; H2 if they sold it; H3-top if Russell
       bought more than Nasdaq (each scaled by its own mean |flow|), H3-bottom the reverse; H4 if S&P sold
       delta (hedging). Outcome means, agree vs disagree, with counts.

INFORMATION ONLY: every H1-H4 effect excluding roll weeks, and with futures + options combined (`_cmb`); the
latest report's reading on each signal.

    python check_cot.py --run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics as st
import sys

import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_cot_cache")
RESULT = os.path.join(CACHE, "RESULT-cot.txt")
DISC = (dt.date(2008, 6, 1), dt.date(2017, 12, 31))
HOLD = (dt.date(2018, 1, 1), dt.date(2100, 1, 1))
N_BOOT = 2000
OUT = []


def p(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ data
def load_prices():
    px = json.load(open(os.path.join(CACHE, "prices_daily.json")))
    out = {}
    for sym, rows in px.items():
        rows = sorted((dt.date.fromisoformat(r["date"]), float(r["close"])) for r in rows if r.get("close"))
        out[sym] = ([d for d, _ in rows], [c for _, c in rows])
    return out


def at_or_after(dates, d):
    import bisect
    i = bisect.bisect_left(dates, d)
    return i if i < len(dates) else None


def fwd(px, sym, d, n):
    ds, cs = px[sym]
    i = at_or_after(ds, d)
    return None if i is None or i + n >= len(cs) else cs[i + n] / cs[i] - 1


def back(px, sym, d, n):
    ds, cs = px[sym]
    i = at_or_after(ds, d)
    return None if i is None or i - n < 0 else cs[i] / cs[i - n] - 1


def vix_maxrise(px, d, n=20):
    ds, cs = px["VIX"]
    i = at_or_after(ds, d)
    return None if i is None or i + n >= len(cs) else max(cs[i + 1:i + n + 1]) / cs[i] - 1


def weekly(contract, wk):
    w = wk.filter((pl.col("contract") == contract) & ~pl.col("release_uncertain")).sort("report_date")
    return w.to_dicts()


def rolling_rank(vals, i, window=156, min_n=104):
    w = [v for v in vals[max(0, i - window + 1): i + 1] if v is not None]
    if len(w) < min_n or vals[i] is None:
        return None
    return (sum(v < vals[i] for v in w) + 0.5 * (sum(v == vals[i] for v in w) - 1)) / (len(w) - 1)


# ------------------------------------------------------------------ stats
def quarter(d):
    return (d.year, (d.month - 1) // 3)


def effect(obs, kind):
    """obs: [(date, y, flag)] ; kind 'vs_all': mean(y|flag==1) - mean(all) ; 'top_bottom': mean(flag==1) - mean(flag==-1)"""
    if kind == "vs_all":
        s = [y for _, y, f in obs if f == 1]
        return (st.mean(s) - st.mean(y for _, y, _f in obs)) if s and obs else None
    a = [y for _, y, f in obs if f == 1]; b = [y for _, y, f in obs if f == -1]
    return st.mean(a) - st.mean(b) if a and b else None


def boot(obs, kind, seed=7):
    by = {}
    for o in obs:
        by.setdefault(quarter(o[0]), []).append(o)
    qs = sorted(by); random.seed(seed); vals = []
    for _ in range(N_BOOT):
        s = [o for q in random.choices(qs, k=len(qs)) for o in by[q]]
        v = effect(s, kind)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None, None, 1.0
    lo, hi = vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]
    pv = 2 * min(sum(v <= 0 for v in vals), sum(v >= 0 for v in vals)) / len(vals)
    return lo, hi, max(pv, 1 / len(vals))


def episodes(obs, flag=1):
    ds = sorted(d for d, _, f in obs if f == flag)
    n, last = 0, None
    for d in ds:
        if last is None or (d - last).days > 28:
            n += 1
        last = d
    return n


def bh(ps):
    m = len(ps); o = sorted(range(m), key=lambda i: ps[i]); q = [0.0] * m; prev = 1.0
    for rank in range(m, 0, -1):
        i = o[rank - 1]; prev = min(prev, ps[i] * m / rank); q[i] = prev
    return q


# ------------------------------------------------------------------ hypotheses
def build_obs(wk, px, src="fut", drop_roll=False):
    """{H: [(entry_date, y, flag, extra)]} for H1-H4, all periods."""
    out = {"H1": [], "H2": [], "H3": [], "H4": []}
    for r in weekly("ES", wk):
        if drop_roll and r["roll_week"]:
            continue
        e = r["effective"]
        x1 = r.get(f"lev_money_pct156_{src}")
        y20 = fwd(px, "SPY", e, 20)
        if x1 is not None and y20 is not None:
            out["H1"].append((e, y20, 1 if x1 <= 0.10 else 0))
        x2, up, y40 = r.get(f"asset_mgr_chg4_pct_{src}"), back(px, "SPY", e, 63), fwd(px, "SPY", e, 40)
        if x2 is not None and up is not None and up > 0 and y40 is not None:
            out["H2"].append((e, y40, 1 if x2 <= 0.10 else 0))
    rty = {r["report_date"]: r for r in weekly("RTY", wk)}
    nq = {r["report_date"]: r for r in weekly("NQ", wk)}
    dates = sorted(set(rty) & set(nq))
    s = [(rty[d].get(f"asset_mgr_pct156_{src}") - nq[d].get(f"asset_mgr_pct156_{src}"))
         if rty[d].get(f"asset_mgr_pct156_{src}") is not None and nq[d].get(f"asset_mgr_pct156_{src}") is not None else None
         for d in dates]
    for i, d in enumerate(dates):
        r = rty[d]
        if drop_roll and r["roll_week"]:
            continue
        sp = rolling_rank(s, i)
        yi, yq = fwd(px, "IWM", r["effective"], 40), fwd(px, "QQQ", r["effective"], 40)
        if sp is None or yi is None or yq is None:
            continue
        out["H3"].append((r["effective"], yi - yq, 1 if sp >= 0.90 else (-1 if sp <= 0.10 else 0)))
    for r in weekly("VIX", wk):
        if drop_roll and r["roll_week"]:
            continue
        x4, y = r.get(f"lev_money_pct156_{src}"), vix_maxrise(px, r["effective"])
        if x4 is not None and y is not None:
            out["H4"].append((r["effective"], y, 1 if x4 <= 0.10 else 0))
    return out


SPEC = {"H1": ("ES lev funds net in lowest 10% -> SPY 20-session return", "vs_all", +1),
        "H2": ("ES asset-mgr 4-wk change lowest 10%, SPY uptrend -> SPY 40-session return", "vs_all", -1),
        "H3": ("asset-mgr RTY minus NQ top vs bottom 10% -> IWM-QQQ 40-session return", "top_bottom", +1),
        "H4": ("VIX lev funds net in lowest 10% -> max VIX rise over 20 sessions", "vs_all", +1)}


def period(obs, rng):
    return [o for o in obs if rng[0] <= o[0] <= rng[1]]


def score(obs_all, label):
    rows, ps = [], []
    for h, (desc, kind, sign) in SPEC.items():
        d, ho = period(obs_all[h], DISC), period(obs_all[h], HOLD)
        ed, eh = effect(d, kind), effect(ho, kind)
        lo, hi, pv = boot(ho, kind)
        rows.append((h, desc, kind, sign, d, ho, ed, eh, lo, hi)); ps.append(pv)
    qs = bh(ps)
    p(f"\n{label}")
    for (h, desc, kind, sign, d, ho, ed, eh, lo, hi), pv, qv in zip(rows, ps, qs):
        sig_d = sum(f == 1 for _, _, f in d) if kind == "vs_all" else f"{sum(f == 1 for _, _, f in d)}/{sum(f == -1 for _, _, f in d)}"
        sig_h = sum(f == 1 for _, _, f in ho) if kind == "vs_all" else f"{sum(f == 1 for _, _, f in ho)}/{sum(f == -1 for _, _, f in ho)}"
        ok = (eh is not None and lo is not None and ((lo > 0) if sign > 0 else (hi < 0)) and qv < 0.10
              and ed is not None and ed * sign > 0)
        f = lambda v: "  n/a " if v is None else f"{v * 100:+6.2f}%"
        p(f"  {h} {desc}")
        p(f"      discovery {f(ed)} (signal weeks {sig_d}, episodes {episodes(d)}) | holdout {f(eh)} "
          f"[{f(lo)},{f(hi)}] p={pv:.3f} q={qv:.3f} (signal weeks {sig_h}, episodes {episodes(ho)}"
          f"{', bottom ' + str(episodes(ho, -1)) if kind == 'top_bottom' else ''}) | predicted {'+' if sign > 0 else '-'} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return rows


# ------------------------------------------------------------------ H5
def h5(obs_all, px):
    path = os.path.join(CACHE, "optflow_daily.parquet")
    if not os.path.exists(path):
        p("\nH5: optflow_daily.parquet missing -- run build_cot_optflow.py"); return
    of = pl.read_parquet(path)
    fam = {f: dict(zip(g["date"].to_list(), g["net_delta_usd"].to_list())) for f, g in
           ((f, of.filter(pl.col("family") == f)) for f in ("SP", "NQ", "RTY"))}
    scale = {f: st.mean(abs(v) for v in fam[f].values()) or 1.0 for f in fam}
    first, last = of["date"].min(), of["date"].max()
    def week_flow(f, e):
        return sum(v for d, v in fam[f].items() if e - dt.timedelta(days=7) <= d < e)
    p(f"\nH5 (descriptive): options flow, net delta bought 14-91 days to expiry, the week before entry; tape {first}..{last}")
    rules = {"H1": lambda e, fl: week_flow("SP", e) > 0, "H2": lambda e, fl: week_flow("SP", e) < 0,
             "H3": lambda e, fl: ((week_flow("RTY", e) / scale["RTY"] - week_flow("NQ", e) / scale["NQ"]) > 0) == (fl == 1),
             "H4": lambda e, fl: week_flow("SP", e) < 0}
    for h, rule in rules.items():
        cov = [o for o in obs_all[h] if first + dt.timedelta(days=7) <= o[0] <= last + dt.timedelta(days=1)]
        sig = [o for o in cov if o[2] != 0]
        base = st.mean(y for _, y, _ in cov) if cov else None
        ag = [o for o in sig if rule(o[0], o[2])]; dis = [o for o in sig if not rule(o[0], o[2])]
        m = lambda xs: "  n/a " if not xs else f"{st.mean(y for _, y, _ in xs) * 100:+6.2f}%"
        p(f"  {h}: weeks covered {len(cov)} (mean outcome {m(cov)}), signal weeks {len(sig)} -> options AGREE {len(ag)}: {m(ag)} | "
          f"DISAGREE {len(dis)}: {m(dis)}")


# ------------------------------------------------------------------ now
def now(wk):
    p("\nLATEST REPORT (information): where each signal stands")
    for c, keys in (("ES", ("lev_money_pct156_fut", "asset_mgr_chg4_pct_fut", "asset_mgr_pct156_fut")),
                    ("NQ", ("asset_mgr_pct156_fut",)), ("RTY", ("asset_mgr_pct156_fut",)), ("VIX", ("lev_money_pct156_fut",))):
        r = wk.filter(pl.col("contract") == c).sort("report_date").tail(1).to_dicts()[0]
        p(f"  {c:4s} {r['report_date']} (usable from {r['effective']}): " + ", ".join(
            f"{k.replace('_fut', '')} {r[k]:.2f}" if r[k] is not None else f"{k} n/a" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if not a.run:
        sys.exit(__doc__)
    if os.path.exists(RESULT):
        sys.exit(f"already scored once: {RESULT}")
    wk = pl.read_parquet(os.path.join(CACHE, "cot_weekly.parquet"))
    px = load_prices()
    p(f"check_cot.py --run at {dt.datetime.now():%Y-%m-%d %H:%M}")
    obs = build_obs(wk, px)
    score(obs, "PRIMARY (futures-only TFF, all usable weeks)")
    score(build_obs(wk, px, drop_roll=True), "INFORMATION: excluding roll weeks")
    score(build_obs(wk, px, src="cmb"), "INFORMATION: futures + options combined TFF")
    h5(obs, px)
    now(wk)
    open(RESULT, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    p(f"\nwritten {RESULT}")


if __name__ == "__main__":
    main()
