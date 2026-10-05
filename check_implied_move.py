# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "requests", "python-dotenv"]
# ///
"""
check_implied_move.py
=====================
Can the at-the-money straddle, scaled down by its own history, give a CALIBRATED
"implied move high / low" for the session -- the idea behind SpotGamma's 1-day
implied move bands, which are proprietary?

The raw straddle overstates the move (the variance risk premium: sellers are
paid for tail risk), so the band is anchor +/- q x straddle, with q the
tau-quantile of past (realized excursion / straddle) ratios, separately for the
upside and downside. Straddles come from build_implied_move.py (quote mids,
interpolated to spot); price bars from historical/{T}.parquet (RTH).

FAMILIES (each scored on its own)
  am  (primary)  same-day expiry quoted 09:33-09:35; anchor A = 09:35 bar close;
                 excursions over 09:36-15:59 bars.
  pm  (pre-open) next-expiry straddle quoted 15:53-15:55 the prior session;
                 A = that 15:55 bar close; excursions over the next session's
                 RTH (so the overnight gap counts). Expiry must be that session.
  up = max(H - A, 0) / S,  dn = max(A - L, 0) / S,  S = the straddle in $.

MODEL (walk-forward, no look-ahead)
  q_up(d), q_dn(d) = tau-quantiles of up / dn over ALL scored days BEFORE d
  (expanding window, at least 120). Band: A + q_up S, A - q_dn S.
  tau in {0.50, 0.65, 0.80, 0.90}.

CONTROL THAT COULD KILL IT (methodology section 6)
  Same machinery, no options: the denominator is the trailing 20-session mean
  of this family's own realized (up + dn) range in % of A. If the straddle
  band is no sharper than a band built from recent ranges, the options add
  nothing beyond "volatility clusters".

Split: IS < 2025-08-21 <= OOS. Slices: METHODOLOGY section 8 (populated at
>= 20 days). Tickers: SPY, QQQ, IWM.

PRE-REGISTERED CRITERIA (fixed 2026-10-03, before the first run), per family
  M1  straddle + bars on >= 90% of sessions in each ticker's coverage window
  M2  median |lake spot - bar close at the anchor minute| <= 0.03% of price
  C1  OOS coverage per ticker x side: within +/-7pp of tau at 0.65 and
      +/-5pp at 0.90 (~2.5 binomial SEs at n ~ 270)
  C2  pooled coverage at tau 0.65 (3 tickers, both sides) within +/-10pp of
      0.65 in >= 5 of 6 slices
  C3  OOS pinball loss (mean over tau and sides, % of A) lower for the
      straddle band than for the control: pooled day-bootstrap 95% CI of
      (control - straddle) > 0, AND point estimate > 0 in >= 2 of 3 tickers
  VERDICT  PASS = M1 M2 C1 C2 C3.  C1 C2 only = calibrated but no better than
           recent ranges.

REGIME CONDITIONING (separate hypotheses, each its own verdict; scored on
the primary family only)
  R1 VIX   prior-session VIX close: < 15 / 15-20 / >= 20
  R2 EVENT CPI / PCE / NFP (8:30) or FOMC (14:00) day vs not (macro_calendar;
           CPI/PCE/FOMC listed only from 2024-08, so earlier event days count
           as "not" in the learning history -- noted, not corrected)
  R3 GEX   sign of the latest net_gex BEFORE the day (historical/GEX{T})
  Conditional quantile = the same expanding quantile over prior days in the
  same regime cell (>= 60 of them, else the unconditional value).
  PASS if pooled bootstrap 95% CI of (unconditional - conditional) pinball > 0
  AND point estimate > 0 in >= 2 of 3 tickers.

RUN LOG
  2026-10-03 run 1: machinery failure, not a result -- SPY's market_time label
    is null 2025-08..2026-08, so the RTH filter dropped 250 SPY OOS days (n=12)
    and M1 missed it (it measured coverage against SPY's own bars). Fixed: RTH
    by clock, M1 against the all-ticker calendar. Criteria unchanged.
  2026-10-03 run 2:
    AM  PASS. OOS coverage at tau .65: 61.8-68.7% (6 ticker-sides); at .90:
        87.0-90.5%; 6/6 slices. Straddle beats the no-options control on
        pinball by 5.6-7.5% per ticker, pooled CI [+0.0076, +0.0156].
    PM  PASS. Coverage .65: 61.7-64.4%; .90: 87.4-93.1%; 6/6 slices; control
        beaten 4.4-6.9%, pooled CI [+0.0074, +0.0190].
    R1 VIX, R2 EVENT, R3 GEX: all FAIL (pooled CIs straddle 0). R2 is weak
        by construction: the event cell rarely reaches 60 prior days, so it
        mostly fell back to the unconditional band.
    The raw straddle is NOT too wide for highs/lows: A +/- S held the high on
    61-65% of days and the low on 60-63% (both only 22-30%). Its premium shows
    against the CLOSE (|close - A| / S mean 0.92-0.93), but the session's
    extremes reach further than the close and cancel it. The tau-.65 ratio is
    ~1.02-1.10; the median ~0.72-0.79; tau .90 is 1.7-1.8 up, 2.0-2.2 down.

Usage:
  python build_implied_move.py      # once
  python check_implied_move.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os

import numpy as np
import polars as pl

import macro_calendar as MC

TICKERS = ["SPY", "QQQ", "IWM"]
TAUS = [0.50, 0.65, 0.80, 0.90]
SPLIT = dt.date(2025, 8, 21)
SLICE_EDGES = [dt.date(2024, 8, 20), dt.date(2024, 12, 20), dt.date(2025, 4, 21),
               dt.date(2025, 8, 21), dt.date(2025, 12, 21), dt.date(2026, 4, 22),
               dt.date(2026, 8, 23)]
MIN_HIST, MIN_REGIME, CTRL_N = 120, 60, 20
CACHE = "_implied_move_cache"
VIX_CACHE = "_implied_move_vix.json"
N_BOOT = 2000
RNG = np.random.default_rng(20261003)


# ---------------------------------------------------------------- data
def straddles():
    fs = [pl.read_parquet(f) for f in glob.glob(f"{CACHE}/*.parquet")]
    return pl.concat([f for f in fs if f.width > 1], how="diagonal")


def bars(tk):
    # RTH by the CLOCK, not the market_time label: SPY's label is null from
    # 2025-08 to 2026-08 (found on the first run -- it cost SPY 250 OOS days).
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns(pl.col("minute_et").dt.hour().cast(pl.Int32).alias("h"),   # i8 overflows at h*60
                       pl.col("minute_et").dt.minute().cast(pl.Int32).alias("m"))
         .with_columns((pl.col("h") * 60 + pl.col("m")).alias("mod"))
         .filter((pl.col("mod") >= 9 * 60 + 30) & (pl.col("mod") <= 15 * 60 + 59))
         .select("date", "mod", "high", "low", "close"))
    out = {}
    for (day,), g in d.group_by(["date"]):
        out[day] = g.sort("mod")
    return out


def vix_prev():
    if not os.path.exists(VIX_CACHE):
        import requests
        from dotenv import dotenv_values
        key = dotenv_values(".env")["UW_API_KEY"]
        r = requests.get("https://api.unusualwhales.com/api/stock/VIX/volatility/realized",
                         headers={"Authorization": f"Bearer {key}"},
                         params={"timeframe": "5Y"}, timeout=30)
        r.raise_for_status()
        rows = {str(x["date"])[:10]: float(x["price"]) for x in r.json()["data"] if x.get("price")}
        json.dump(rows, open(VIX_CACHE, "w"))
    rows = json.load(open(VIX_CACHE))
    return sorted((dt.date.fromisoformat(k), v) for k, v in rows.items())


def gex_prev(tk):
    g = pl.read_parquet(f"historical/GEX{tk}.parquet").select("date", "net_gex").drop_nulls().sort("date")
    return list(zip(g["date"].to_list(), g["net_gex"].to_list()))


def latest_before(series, day):
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < day:
            lo = mid + 1
        else:
            hi = mid
    return series[lo - 1][1] if lo else None


def build_family(fam, st, tk, B, calendar):
    root = tk
    s = st.filter((pl.col("root") == root) & (pl.col("win") == fam)).sort("date")
    days = sorted(B)
    nxt = {a: b for a, b in zip(days, days[1:])}
    rows, n_cov, gaps = [], 0, []
    for r in s.iter_rows(named=True):
        q_day = r["date"]
        if fam == "am":
            target, a_mod, src = q_day, 9 * 60 + 35, B.get(q_day)
            if src is None:
                continue
            a = src.filter(pl.col("mod") == a_mod)["close"]
            fut = src.filter((pl.col("mod") > a_mod) & (pl.col("mod") <= 15 * 60 + 59))
        else:
            target = nxt.get(q_day)
            if target is None or r["expiry"] != target or q_day not in B:
                continue
            a = B[q_day].filter(pl.col("mod") == 15 * 60 + 55)["close"]
            fut = B[target]
        if a.is_empty() or fut.is_empty():
            continue
        A = float(a[0])
        H, L = float(fut["high"].max()), float(fut["low"].min())
        S = r["strad"]
        if not S or S <= 0:
            continue
        rows.append(dict(day=target, A=A, S=S, H=H, L=L, lake_spot=r["spot"],
                         up=max(H - A, 0.0) / S, dn=max(A - L, 0.0) / S,
                         up_pct=max(H - A, 0.0) / A * 100, dn_pct=max(A - L, 0.0) / A * 100,
                         s_pct=S / A * 100))
    rows.sort(key=lambda x: x["day"])
    # control denominator: trailing CTRL_N mean of (up + dn) % range, prior days only
    for i, x in enumerate(rows):
        prev = rows[max(0, i - CTRL_N):i]
        x["r_pct"] = (sum(p["up_pct"] + p["dn_pct"] for p in prev) / len(prev)) if len(prev) == CTRL_N else None
        x["c_up"] = x["up_pct"] / x["r_pct"] if x["r_pct"] else None
        x["c_dn"] = x["dn_pct"] / x["r_pct"] if x["r_pct"] else None
    # M1 coverage against the TRADING CALENDAR (every ticker's bar dates), not
    # this ticker's own bars -- a gap in the bars must not shrink its denominator
    if rows:
        lo, hi = rows[0]["day"], rows[-1]["day"]
        expected = [d for d in calendar if lo <= d <= hi]
        have = {x["day"] for x in rows}
        n_cov = len(have) / len(expected) if expected else 0.0
    return rows, n_cov


# ---------------------------------------------------------------- model
def pinball(y, q, tau):
    e = y - q
    return max(tau * e, (tau - 1) * e)


def walk(rows, regime=None):
    """Adds per-day predictions for every tau: straddle (unconditional or
    regime-conditional) and control. Prior days only."""
    ups, dns, cus, cds = [], [], [], []
    cell = {}
    for x in rows:
        x["pred"] = {}
        if len(ups) >= MIN_HIST:
            ua, da = np.array(ups), np.array(dns)
            if regime is not None:
                c = cell.get(x.get(regime), ([], []))
                if len(c[0]) >= MIN_REGIME:
                    ua, da = np.array(c[0]), np.array(c[1])
            ca, cd = (np.array(cus), np.array(cds)) if len(cus) >= MIN_HIST else (None, None)
            for t in TAUS:
                qu, qd = float(np.quantile(ua, t)), float(np.quantile(da, t))
                p = dict(qu=qu, qd=qd, up_hat=qu * x["s_pct"], dn_hat=qd * x["s_pct"])
                if ca is not None and x["r_pct"]:
                    p["cu_hat"] = float(np.quantile(ca, t)) * x["r_pct"]
                    p["cd_hat"] = float(np.quantile(cd, t)) * x["r_pct"]
                x["pred"][t] = p
        ups.append(x["up"]); dns.append(x["dn"])
        if x["c_up"] is not None:
            cus.append(x["c_up"]); cds.append(x["c_dn"])
        if regime is not None:
            c = cell.setdefault(x.get(regime), ([], []))
            c[0].append(x["up"]); c[1].append(x["dn"])
    return rows


def loss(x, key_u="up_hat", key_d="dn_hat"):
    ls = []
    for t in TAUS:
        p = x["pred"].get(t)
        if not p or key_u not in p:
            return None
        ls.append(pinball(x["up_pct"], p[key_u], t))
        ls.append(pinball(x["dn_pct"], p[key_d], t))
    return sum(ls) / len(ls)


def boot_ci(by_day):
    """by_day: {day: [diffs across tickers]} -> mean, 95% CI over days."""
    days = sorted(by_day)
    v = np.array([np.mean(by_day[d]) for d in days])
    if len(v) == 0:
        return None, None, None
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def slice_of(d):
    for i, (a, b) in enumerate(zip(SLICE_EDGES, SLICE_EDGES[1:])):
        if a <= d < b:
            return i
    return None


# ---------------------------------------------------------------- report
def run_family(fam, st, B_all, vix, gex):
    print(f"\n{'=' * 78}\nFAMILY {fam.upper()}  "
          f"({'09:35 same-day straddle' if fam == 'am' else 'prior 15:55 next-expiry straddle (pre-open)'})\n{'=' * 78}")
    R, ok = {}, {}
    calendar = sorted(set().union(*(set(B) for B in B_all.values())))
    for tk in TICKERS:
        rows, cov = build_family(fam, st, tk, B_all[tk], calendar)
        for x in rows:
            v = latest_before(vix, x["day"])
            x["vix"] = None if v is None else ("<15" if v < 15 else "15-20" if v < 20 else ">=20")
            x["event"] = (MC.is_macro_am_day(x["day"]) or MC.is_fomc_day(x["day"]))
            g = latest_before(gex[tk], x["day"])
            x["gex"] = None if g is None else ("pos" if g > 0 else "neg")
        R[tk] = walk(rows)
        dev = sorted(abs(x["lake_spot"] - x["A"]) / x["A"] * 100 for x in rows)
        m2 = dev[len(dev) // 2] if dev else 9
        ok[tk] = (cov >= 0.90, m2 <= 0.03)
        print(f"  {tk}: {len(rows)} days {rows[0]['day']}..{rows[-1]['day']}  coverage {cov:.1%}  "
              f"median |lake spot - bar| {m2:.4f}%")
    m1 = all(v[0] for v in ok.values()); m2 = all(v[1] for v in ok.values())
    print(f"  M1 coverage >= 90%: {'PASS' if m1 else 'FAIL'}   M2 anchor agreement: {'PASS' if m2 else 'FAIL'}")

    # descriptive: how much the straddle overstates the move
    print("\n  THE RAW STRADDLE (q = 1): share of days the high / low stayed inside A +/- S, "
          "and the median excursion / straddle")
    for tk in TICKERS:
        for lab, sel in (("IS ", lambda d: d < SPLIT), ("OOS", lambda d: d >= SPLIT)):
            xs = [x for x in R[tk] if sel(x["day"])]
            if not xs:
                continue
            iu = np.mean([x["up"] <= 1 for x in xs]); idn = np.mean([x["dn"] <= 1 for x in xs])
            both = np.mean([x["up"] <= 1 and x["dn"] <= 1 for x in xs])
            print(f"    {tk} {lab} n={len(xs):3}  high inside {iu:.0%}  low inside {idn:.0%}  both {both:.0%}  "
                  f"median up/S {np.median([x['up'] for x in xs]):.2f}  dn/S {np.median([x['dn'] for x in xs]):.2f}  "
                  f"straddle {np.median([x['s_pct'] for x in xs]):.2f}% of A")

    # C1
    print("\n  C1  OOS coverage of the walk-forward band (target = tau)")
    print("      ticker side   " + "  ".join(f"tau {t:.2f}" for t in TAUS) + "     n")
    c1 = True
    for tk in TICKERS:
        xs = [x for x in R[tk] if x["day"] >= SPLIT and x["pred"]]
        for side, key, hat in (("up", "up_pct", "up_hat"), ("dn", "dn_pct", "dn_hat")):
            covs = []
            for t in TAUS:
                cv = np.mean([x[key] <= x["pred"][t][hat] for x in xs])
                covs.append(cv)
                if t == 0.65 and abs(cv - t) > 0.07: c1 = False
                if t == 0.90 and abs(cv - t) > 0.05: c1 = False
            print(f"      {tk:5}  {side}    " + "  ".join(f"{c:8.1%}" for c in covs) + f"   {len(xs)}")
    print(f"      C1: {'PASS' if c1 else 'FAIL'}")

    # C2
    print("\n  C2  pooled coverage at tau 0.65 by slice (3 tickers, both sides)")
    good = 0; populated = 0
    for i in range(6):
        cs, nd = [], set()
        for tk in TICKERS:
            for x in R[tk]:
                if x["pred"] and slice_of(x["day"]) == i:
                    p = x["pred"][0.65]
                    cs += [x["up_pct"] <= p["up_hat"], x["dn_pct"] <= p["dn_hat"]]
                    nd.add(x["day"])
        if len(nd) >= 20:
            populated += 1
            cv = np.mean(cs); g = abs(cv - 0.65) <= 0.10; good += g
            print(f"      slice {i + 1} {SLICE_EDGES[i]}..{SLICE_EDGES[i + 1]}  days {len(nd):3}  coverage {cv:.1%}  {'ok' if g else 'OFF'}")
        else:
            print(f"      slice {i + 1} {SLICE_EDGES[i]}..{SLICE_EDGES[i + 1]}  days {len(nd):3}  (unpopulated)")
    c2 = good >= 5
    print(f"      C2: {good}/6 slices within +/-10pp -> {'PASS' if c2 else 'FAIL'}")

    # C3
    print("\n  C3  OOS pinball loss (% of A; lower is better): straddle band vs no-options control")
    by_day, wins = {}, 0
    for tk in TICKERS:
        ds, ls_s, ls_c, w_s, w_c = [], [], [], [], []
        for x in R[tk]:
            if x["day"] < SPLIT:
                continue
            a, b = loss(x), loss(x, "cu_hat", "cd_hat")
            if a is None or b is None:
                continue
            ls_s.append(a); ls_c.append(b); by_day.setdefault(x["day"], []).append(b - a)
            p, c = x["pred"][0.65], x["pred"][0.65]
            w_s.append(p["up_hat"] + p["dn_hat"]); w_c.append(c["cu_hat"] + c["cd_hat"])
        d = np.mean(ls_c) - np.mean(ls_s); wins += d > 0
        print(f"      {tk}: straddle {np.mean(ls_s):.4f}  control {np.mean(ls_c):.4f}  "
              f"control - straddle {d:+.4f} ({d / np.mean(ls_c):+.1%})   "
              f"tau-0.65 band width: straddle {np.mean(w_s):.2f}%  control {np.mean(w_c):.2f}%  n={len(ls_s)}")
    m, lo, hi = boot_ci(by_day)
    c3 = lo is not None and lo > 0 and wins >= 2
    print(f"      pooled control - straddle {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  tickers better {wins}/3"
          f"  -> C3 {'PASS' if c3 else 'FAIL'}")

    verdict = "PASS" if (m1 and m2 and c1 and c2 and c3) else (
        "CALIBRATED, NOT BETTER THAN RECENT RANGES" if (m1 and m2 and c1 and c2) else "FAIL")
    print(f"\n  FAMILY {fam.upper()} VERDICT: {verdict}")

    # live parameters: quantiles over everything to date
    print("\n  Ratios to use live (all days to date; q x straddle):")
    for tk in TICKERS:
        u = np.array([x["up"] for x in R[tk]]); dn = np.array([x["dn"] for x in R[tk]])
        print(f"    {tk}: " + "  ".join(f"tau {t:.2f} up {np.quantile(u, t):.2f} dn {np.quantile(dn, t):.2f}" for t in TAUS))
    return R


def run_regimes(R0, st, B_all, vix, gex):
    print(f"\n{'=' * 78}\nREGIME CONDITIONING (family AM)\n{'=' * 78}")
    for name, key in (("R1 VIX", "vix"), ("R2 EVENT", "event"), ("R3 GEX", "gex")):
        by_day, wins = {}, 0
        print(f"\n  {name}")
        for tk in TICKERS:
            rows = [dict(x) for x in R0[tk]]
            for x in rows:
                x.pop("pred", None)
            base = {x["day"]: loss(x) for x in R0[tk] if x["day"] >= SPLIT}
            cond = walk(rows, regime=key)
            ds = []
            for x in cond:
                if x["day"] < SPLIT:
                    continue
                a, b = base.get(x["day"]), loss(x)
                if a is None or b is None:
                    continue
                ds.append(a - b); by_day.setdefault(x["day"], []).append(a - b)
            d = float(np.mean(ds)) if ds else 0.0; wins += d > 0
            # descriptive: OOS coverage at 0.65 per regime cell, unconditional band
            cells = {}
            for x in R0[tk]:
                if x["day"] >= SPLIT and x["pred"]:
                    p = x["pred"][0.65]
                    c = cells.setdefault(x.get(key), [0, 0, 0])
                    c[0] += x["up_pct"] <= p["up_hat"]; c[1] += x["dn_pct"] <= p["dn_hat"]; c[2] += 1
            cs = "  ".join(f"{k}: up {v[0] / v[2]:.0%} dn {v[1] / v[2]:.0%} (n {v[2]})"
                           for k, v in sorted(cells.items(), key=lambda kv: str(kv[0])))
            print(f"    {tk}: uncond - cond {d:+.4f}   unconditional tau-0.65 coverage by cell: {cs}")
        m, lo, hi = boot_ci(by_day)
        ok = lo is not None and lo > 0 and wins >= 2
        print(f"    pooled {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  tickers better {wins}/3 -> {'PASS' if ok else 'FAIL'}")


def main():
    st = straddles()
    B_all = {tk: bars(tk) for tk in TICKERS}
    vix = vix_prev()
    gex = {tk: gex_prev(tk) for tk in TICKERS}
    R_am = run_family("am", st, B_all, vix, gex)
    run_family("pm", st, B_all, vix, gex)
    run_regimes(R_am, st, B_all, vix, gex)


if __name__ == "__main__":
    main()
