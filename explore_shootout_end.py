"""
explore_shootout_end.py -- WHEN DOES THE SHOOTOUT END? (user, 2026-11-27: "determine the historical average of the
Shootout Period ... then a dynamic countdown of when the Shootout is ending"; definition A: "the front-end IV crush").
DESCRIPTIVE, step 1. The end-time distribution uses every session (it fits nothing); the factor associations use the
ODD (discovery) months only -- the even months stay untouched for validating a live conditional countdown.

THE SHOOTOUT IS ON while the tape moves faster than the ATM 0DTE options priced in (check_shootout_iv: 1.2x at 09:30,
1.0x by 09:33, 0.6x at midday over 5 minutes):
  VR(m)   realized variance of the 1-minute log returns over minutes m-4 .. m, divided by what the ATM 0DTE IV at m-5
          priced for those 5 minutes: iv^2 x 5 / 1440 (the lake's iv_close is a DAILY vol on calendar days -- the
          check_shootout_iv convention). ATM = the strike nearest the underlying close, IV = mean of the call and put.
  END     the first minute m >= 09:35 from which VR stays < 1 for 5 consecutive minutes. "Never started" = VR < 1 at
          09:35 already; not ended by 11:00 = censored at 11:00. Sensitivity: thresholds 0.8 / 1.2, sustain 3 / 8.
FACTORS (the user's list, 2026-11-27), all known by 09:31-09:35:
  GAP        09:30 open vs prior close, in typical 30-min ranges (T30, prior 20 sessions)
  OPEN_VOL   the 09:30 minute's volume / the median 09:30 volume of the prior 20 sessions (opening cross / arb proxy)
  RNG5       09:30-09:34 range in T30 units (how hot the first 5 minutes ran)
  IMB5       candle tick-rule imbalance 09:30-09:34 (sum sign(close change) x volume / volume) -- no stock-quote history
  IV0        ATM 0DTE IV at 09:31 (level of the front end)          SPR0  ATM call+put spread % of mid at 09:31
  GAM0 / THE0  ATM 0DTE gamma and theta at 09:31 (mean of the legs)  DGEX  dealer gamma per 1% (spot-exposures) at 09:31
  MACRO830   CPI / PCE / NFP morning                                 REL1000  10:00 release by rule (ISM manufacturing =
             1st business day, ISM services = 3rd business day; approximate -- the curated list starts 2026-11)
  DOW        day of week

    python explore_shootout_end.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_shootout_cache", "end_rows.parquet")
TKS = ("SPY", "QQQ", "IWM")
M0, M_END, CAP = 570, 575, 660
FACT = ["GAP", "OPEN_VOL", "RNG5", "IMB5", "IV0", "SPR0", "GAM0", "THE0", "DGEX", "MACRO830", "REL1000", "DOW"]


def mod(col):
    return pl.col(col).dt.hour().cast(pl.Int32) * 60 + pl.col(col).dt.minute().cast(pl.Int32)


def bars(d, tk):
    p = os.path.join(LAKE, "ext-1m", f"date={d}", f"{tk}.parquet")
    if not os.path.exists(p):
        return None
    x = pl.read_parquet(p).filter(pl.col("market_time") == "r").with_columns(m=mod("minute")).sort("m")
    return x.select("m", "open", "high", "low", "close", "volume").to_numpy() if x.height > 300 else None


def r30(a):
    out = []
    for m0 in range(600, 930, 30):
        w = a[(a[:, 0] >= m0) & (a[:, 0] < m0 + 30)]
        if len(w) > 20:
            out.append((w[:, 2].max() - w[:, 3].min()) / w[-1, 4])
    return float(np.median(out)) if out else None


# 🚨 THE LAKE'S iv_close / theta_close UNITS CHANGE BY ERA (found 2026-11-27, SPY 0DTE median iv per day):
#   2023-10-12 .. 2024-09-20 annualised, 2024-09-23 .. 2025-12-31 DAILY (calendar days), 2026-01-02 .. annualised.
# Everything here is put on the DAILY convention (iv / sqrt(365), theta / 365 in the annualised eras).
def annual_era(d):
    return d < "2024-09-23" or d >= "2026-01-02"


def ends(vr, thr=1.0, sustain=5):
    """first minute (>= 09:35) from which vr stays below thr for `sustain` minutes; vr = {minute: value}."""
    for m in range(M_END, CAP + 1):
        if all(vr.get(k) is not None and vr[k] < thr for k in range(m, m + sustain)):
            return m
    return CAP


def biz_day_n(day, n):
    import market_calendar as MC
    tds = list(MC.trading_days(day.replace(day=1), day.replace(day=28)))
    return len(tds) >= n and tds[n - 1] == day


def one(args):
    d, prior = args
    import macro_calendar as MAC
    day = dt.date.fromisoformat(d)
    p = os.path.join(LAKE, "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(p):
        return []
    y = (pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(list(TKS)) & (pl.col("expiry") == pl.lit(d).str.to_date()))
         .with_columns(mm=mod("minute_et")).filter((pl.col("mm") >= M0 - 1) & (pl.col("mm") <= CAP + 10))
         .select("underlying_symbol", "option_type", "strike", "mm", "bid_close", "ask_close", "iv_close", "gamma_close", "theta_close")
         .collect())
    out = []
    for tk in TKS:
        a = bars(d, tk)
        if a is None:
            continue
        pr = [x for x in (bars(q, tk) for q in prior) if x is not None]
        if len(pr) < 10:
            continue
        t30 = float(np.median([v for v in (r30(x) for x in pr) if v]))
        m = a[:, 0].astype(int); c = a[:, 4]; v = a[:, 5]
        idx = {mm: i for i, mm in enumerate(m)}
        z = y.filter(pl.col("underlying_symbol") == tk).drop("underlying_symbol")
        if not z.height or M0 not in idx:
            continue
        legs = {}
        ann = annual_era(d)
        for ty, k, mm, b, s, iv, g, th in z.iter_rows():
            if ann:
                iv = None if iv is None else iv / np.sqrt(365)
                th = None if th is None else th / 365
            legs.setdefault((int(round(k)), ty), {})[int(mm)] = (b, s, iv, g, th)

        def atm(mm):
            i = idx.get(mm)
            if i is None:
                return None
            K = int(round(c[i]))
            for back in range(0, 4):
                cl, pt = legs.get((K, "call"), {}).get(mm - back), legs.get((K, "put"), {}).get(mm - back)
                if cl and pt and cl[2] and pt[2]:
                    return cl, pt
            return None
        iv = {}
        for mm in range(M0, CAP + 6):
            q = atm(mm)
            if q:
                iv[mm] = (q[0][2] + q[1][2]) / 2
        vr, vr1 = {}, {}
        for mm in range(M_END, CAP + 6):
            # PRIMARY (revised 2026-11-27 after the first pass): 5-MINUTE moves, overlapping, over the trailing 15 minutes
            # (as far back as the open allows) -- 1-minute returns carry bid-ask bounce that inflates realized variance
            # (first pass: median 1-min VR still 1.14 at 11:00, IWM "ending" at 10:42). Priced = the IV at the window start.
            k0 = max(M0 + 5, mm - 10)
            r5 = [np.log(c[idx[k]] / c[idx[k - 5]]) for k in range(k0, mm + 1) if k in idx and (k - 5) in idx]
            base = iv.get(k0 - 5) or iv.get(k0 - 4) or iv.get(M0 + 1)
            if len(r5) >= min(3, mm - M_END + 1) and base:      # 09:35 has one 5-min return, 09:37+ three or more
                vr[mm] = float(np.mean(np.square(r5)) / (base ** 2 * 5 / 1440))
            ii = [idx.get(k) for k in range(mm - 5, mm + 1)]
            b1 = iv.get(mm - 5) or iv.get(mm - 6)
            if None not in ii and b1:
                rr = np.diff(np.log(c[ii]))
                vr1[mm] = float(np.sum(rr ** 2) / (b1 ** 2 * 5 / 1440))
        if M_END not in vr:
            continue
        row = dict(d=d, tk=tk, end=ends(vr), started=bool(vr[M_END] >= 1.0),
                   end_08=ends(vr, 0.8), end_12=ends(vr, 1.2), end_s3=ends(vr, 1.0, 3), end_s8=ends(vr, 1.0, 8),
                   vr575=vr[M_END], vr_path=[round(vr.get(k, np.nan), 3) for k in range(M_END, CAP + 1)],
                   end_1m=ends(vr1) if vr1 else CAP)
        pcl = pr[-1][-1, 4]
        o = a[idx[M0], 1]
        f5 = a[(m >= M0) & (m < M0 + 5)]
        dcs = np.diff(c[idx[M0]: idx[M0] + 5]) if idx[M0] + 5 <= len(c) else np.array([])
        vv = v[idx[M0] + 1: idx[M0] + 5]
        ov = [x[x[:, 0] == M0, 5] for x in pr]
        ov = [float(q[0]) for q in ov if len(q)]
        q31 = atm(M0 + 1)
        row.update(GAP=(o - pcl) / (t30 * pcl), OPEN_VOL=float(v[idx[M0]] / np.median(ov)) if ov else np.nan,
                   RNG5=(f5[:, 2].max() - f5[:, 3].min()) / (t30 * o) if len(f5) else np.nan,
                   IMB5=float((np.sign(dcs) * vv[:len(dcs)]).sum() / max(vv[:len(dcs)].sum(), 1)) if len(dcs) else np.nan,
                   IV0=iv.get(M0 + 1, np.nan),
                   SPR0=float(((q31[0][1] - q31[0][0]) + (q31[1][1] - q31[1][0])) / ((q31[0][0] + q31[0][1] + q31[1][0] + q31[1][1]) / 2))
                   if q31 and q31[0][0] and q31[1][0] else np.nan,
                   GAM0=float((q31[0][3] + q31[1][3]) / 2) if q31 and q31[0][3] is not None and q31[1][3] is not None else np.nan,
                   THE0=float((q31[0][4] + q31[1][4]) / 2) if q31 and q31[0][4] is not None and q31[1][4] is not None else np.nan,
                   MACRO830=float(d in MAC.CPI or d in MAC.PCE or MAC.is_nfp_day(day)),
                   REL1000=float(biz_day_n(day, 1) or biz_day_n(day, 3)), DOW=day.weekday())
        ep = os.path.join(LAKE, "spot-exposures-1m", f"date={d}", f"{tk}.parquet")
        row["DGEX"] = np.nan
        if os.path.exists(ep):
            e = pl.read_parquet(ep).with_columns(mm=mod("minute_et")).filter(pl.col("mm") <= M0 + 1).sort("mm")
            if e.height:
                row["DGEX"] = float(e["gamma_per_one_percent_move_oi"][-1])
        out.append(row)
    return out


def main():
    days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "ext-1m", "date=*")))
    if os.path.exists(CACHE):
        R = pl.read_parquet(CACHE)
    else:
        jobs = [(d, days[max(0, i - 20): i]) for i, d in enumerate(days) if i >= 20]
        with ProcessPoolExecutor(6) as ex:
            rows = [r for rs in ex.map(one, jobs) for r in rs]
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        R = pl.DataFrame(rows, infer_schema_length=None)
        R.write_parquet(CACHE)
    hm = lambda x: f"{int(x) // 60}:{int(x) % 60:02d}"          # noqa: E731
    print(f"{R.height} session-tickers, {R['d'].n_unique()} sessions ({R['d'].min()} .. {R['d'].max()})")
    print("\n== WHEN THE SHOOTOUT ENDS (VR < 1 for 5 straight minutes, from 09:35; ALL sessions, descriptive)")
    for tk in TKS:
        x = R.filter(pl.col("tk") == tk)
        e = x["end"].to_numpy()
        st = x["started"].mean()
        q = np.percentile(e, [10, 25, 50, 75, 90])
        print(f"  {tk}: still on at 09:35 {st:.0%} | end median {hm(q[2])}, mean {hm(e.mean())} | middle half {hm(q[1])}-{hm(q[3])} | "
              f"10-90% {hm(q[0])}-{hm(q[4])} | ran past 11:00 {np.mean(e >= CAP):.0%}")
    x = R.filter(pl.col("started"))
    print(f"  only sessions where it was still on at 09:35 ({x.height}): end median {hm(np.median(x['end']))}, "
          f"middle half {hm(np.percentile(x['end'], 25))}-{hm(np.percentile(x['end'], 75))}")
    print("  sensitivity (median end, all): " + "  ".join(f"{k} {hm(np.median(R[k]))}" for k in ("end_08", "end", "end_12", "end_s3", "end_s8", "end_1m")))
    vp = np.array(R["vr_path"].to_list(), float)
    print("  median VR by minute: " + "  ".join(f"{hm(M_END + k)} {np.nanmedian(vp[:, k]):.2f}" for k in (0, 2, 5, 10, 15, 25, 40, 60, 85)))
    Dd = R.filter(pl.col("d").str.slice(5, 2).cast(pl.Int32) % 2 == 1)
    print(f"\n== FACTORS vs END (DISCOVERY = odd months only, {Dd.height} session-tickers; Spearman of WITHIN-TICKER ranks; end median by within-ticker tercile)")
    from scipy.stats import spearmanr
    Dd = pl.concat([z.with_columns([pl.col(f).cast(pl.Float64).rank('average').truediv(z.height).alias(f'r_{f}') for f in FACT + ['end']]) for z in Dd.partition_by('tk')])
    for f in FACT:
        z = Dd.filter(pl.col(f).cast(pl.Float64).is_finite())
        if z.height < 50:
            print(f"  {f:9s} n {z.height} (too few)"); continue
        fv, ev = z[f"r_{f}"].to_numpy(), z["end"].to_numpy()
        rho = spearmanr(fv, z["r_end"].to_numpy()).statistic
        raw = z[f].cast(pl.Float64).to_numpy()
        if len(np.unique(raw)) <= 6:
            cells = "  ".join(f"{u:g}: {hm(np.median(ev[raw == u]))} (n {int((raw == u).sum())})" for u in np.unique(raw))
        else:
            c1, c2 = np.percentile(fv, [33.3, 66.7])
            cells = f"low {hm(np.median(ev[fv <= c1]))}  mid {hm(np.median(ev[(fv > c1) & (fv <= c2)]))}  high {hm(np.median(ev[fv > c2]))}"
        print(f"  {f:9s} rho {rho:+.3f}  n {z.height:4d} | {cells}")


if __name__ == "__main__":
    main()
