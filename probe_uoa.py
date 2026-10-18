# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0.0", "numpy>=1.26.0", "httpx", "python-dotenv"]
# ///
"""
probe_uoa.py
============
UNUSUAL OPTIONS ACTIVITY WITH STICKY OPEN INTEREST (operator, 2026-10-17).
Descriptive, not a backtest. Two modes:

CONTRACTS  named tickers + expiries (first use: IMSR / BYRN calls 11/20, 12/18)
    python probe_uoa.py IMSR BYRN --exp 2026-11-20 2026-12-18 --right C \
        --start 2026-10-05 --end 2026-10-16
  Every contract of those expiries: per-day volume / side / OI, and the biggest
  contract-days with whether they STUCK (next row's OI) and STAYED (latest OI).

SCAN       one whole ticker, no expiries needed (first use: RIG, 2026-10-17)
    python probe_uoa.py RIG --scan --start 2026-10-05 --end 2026-10-16
  1. DAILY   call/put volume, ask share, premium vs the 30-day average, total OI
             (spike days marked *).
  2. OI      the contracts whose OI moved most each day, re-ranked by |dOI|
             (UW ranks by % change, which floods the top with 5-lot contracts).
  3. NET     over the window: the biggest net OI BUILDERS and UNWINDS, with
             volume, side, multi-leg share, premium, stickiness.
  4. ORIGIN  for the standouts: when and how their OI was built (biggest OI
             increases ever, side, multi-leg, price, stock), and for any
             multi-leg build day, the OTHER LEGS of that package.
  5. ALERTS  UW flow alerts in the window, grouped by contract.
  6. PENDING the last day's big prints, which only the next session's OI can judge.

🚨 OI TIMING. UW's per-contract `open_interest` is START OF DAY (OCC's overnight
figure), so day D's volume shows in day D+1's OI. /oi-change?date=D likewise
describes the trading of the session BEFORE D (curr_oi = after it, volume /
prev_* = that session). The last day of any window is unjudgeable until the
next session's OI exists.
🚨 SIDE. Ask-side = bought, bid-side = sold/written -- an inference. Net OI
nets openers against closers. A 100%-multi-leg day is one leg of a package;
read the package, not the leg (RIG 9/16-17: a 40k-lot risk reversal whose
short-put leg looked like "huge put activity" when it was partly bought back).
🚨 /stock/{t}/option-contracts lists only contracts that traded on the latest
day; the full chain is /stock/{t}/option-chains. UW accepts and silently drops
params it doesn't support -- check returned values (curr_date) not row counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
import uw_options_data_lake as L                     # noqa: E402

COLS = ("date", "volume", "open_interest", "ask_volume", "bid_volume", "mid_volume", "avg_price",
        "sweep_volume", "multi_leg_volume", "underlying_close", "trades", "total_premium",
        "implied_volatility")


def fnum(v):
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def occ_label(s):
    """RIG280121P00007000 -> '21Jan28 7P' (any root length)."""
    d = dt.datetime.strptime(s[-15:-9], "%y%m%d")
    return f"{d:%d%b%y} {int(s[-8:]) / 1000:g}{s[-9]}"


def occ_expiry(s):
    return dt.datetime.strptime(s[-15:-9], "%y%m%d").date().isoformat()


def side(ask, bid, vol):
    if vol <= 0:
        return "—"
    return "bought" if ask / vol >= 0.6 else "sold" if bid / vol >= 0.6 else "mixed"


# ============================================================ CONTRACTS mode
def pull(c, tk, exps, right):
    yymmdd = {e.replace("-", "")[2:]: e for e in exps}
    syms = [s for s in c.get_json(f"/stock/{tk}/option-chains")["data"]
            if s[len(tk):len(tk) + 6] in yymmdd and s[len(tk) + 6] == right]
    rows = []
    for s in sorted(syms):
        for r in c.get_json(f"/option-contract/{s}/historic").get("chains", []):
            rows.append(dict(tk=tk, sym=s, exp=yymmdd[s[len(tk):len(tk) + 6]][5:].replace("-", "/"),
                             k=int(s[-8:]) / 1000, **{k: r.get(k) for k in COLS}))
    return rows


def contracts_mode(c, a):
    df = pd.DataFrame([r for tk in a.tickers for r in pull(c, tk.upper(), a.exp, a.right)])
    if df.empty:
        print("no contracts found"); return
    for col in ("volume", "open_interest", "ask_volume", "bid_volume", "mid_volume", "sweep_volume",
                "multi_leg_volume", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ("avg_price", "underlying_close", "total_premium", "implied_volatility"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")
    df["oi_next"] = g["open_interest"].shift(-1)      # start of next day = after this day's trading
    df["d_oi"] = df["oi_next"] - df["open_interest"]
    df["avg20"] = g["volume"].transform(lambda v: v.shift(1).rolling(20, min_periods=5).mean())
    df["oi_last"] = g["open_interest"].transform("last")
    w = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    print("latest row per ticker:", df.groupby("tk")["date"].max().to_dict(),
          "(OI = start of day; the last day's volume is judged by the NEXT row)")
    for tk in [t.upper() for t in a.tickers]:
        u = df[df.tk == tk].groupby("date")["underlying_close"].first()
        u = u[(u.index >= a.start) & (u.index <= a.end)]
        print(f"\n==== {tk} close: " + "  ".join(f"{d[5:]} {v:.2f}" for d, v in u.items()))
        for ex in sorted(w[w.tk == tk]["exp"].unique()):
            x = w[(w.tk == tk) & (w.exp == ex)]
            day = x.groupby("date").agg(vol=("volume", "sum"), ask=("ask_volume", "sum"),
                                        bid=("bid_volume", "sum"), oi=("open_interest", "sum"),
                                        prem=("total_premium", "sum"))
            print(f"  {ex} {a.right} by day:")
            for d, r in day.iterrows():
                print(f"    {d}  vol {int(r.vol):>6}  ask {r.ask / r.vol if r.vol else 0:4.0%}  "
                      f"bid {r.bid / r.vol if r.vol else 0:4.0%}  OI {int(r.oi):>7}  prem ${r.prem or 0:>10,.0f}")
        x = w[(w.tk == tk) & (w.volume > 0)].sort_values("volume", ascending=False).head(a.top)
        print(f"  top contract-days:\n    {'date':10} {'contract':9} {'vol':>6} {'avg20':>6} {'OI0':>6} "
              f"{'v/OI':>5} {'ask%':>5} {'sweep':>5} {'mleg':>5} {'avgpx':>6} {'prem$':>8} {'OInext':>7} "
              f"{'dOI':>6} {'stick':>6} {'OInow':>6}")
        for _, r in x.iterrows():
            known = r.oi_next == r.oi_next
            stick = r.d_oi / r.volume if known else np.nan
            print(f"    {r.date} {r.exp} {r.k:>4g}{a.right} {int(r.volume):>6} "
                  f"{(r.avg20 if r.avg20 == r.avg20 else 0):>6.0f} {int(r.open_interest):>6} "
                  f"{r.volume / max(r.open_interest, 1):>5.1f} {r.ask_volume / r.volume:>5.0%} "
                  f"{int(r.sweep_volume):>5} {int(r.multi_leg_volume):>5} {(r.avg_price or 0):>6.2f} "
                  f"{(r.total_premium or 0):>8,.0f} {(int(r.oi_next) if known else ''):>7} "
                  f"{(int(r.d_oi) if known else ''):>6} {(f'{stick:+.0%}' if known else ''):>6} {int(r.oi_last):>6}")


# ================================================================ SCAN mode
def oi_rows(c, tk, d):
    """/oi-change for date d (= the previous session's trading), as a DataFrame
    with only rows really stamped d (UW drops params it doesn't support)."""
    try:
        rows = c.get_json(f"/stock/{tk}/oi-change", {"date": d, "limit": "500"}).get("data", [])
    except L.NoDataForDate:
        rows = []
    rows = [r for r in rows if r.get("curr_date") == d]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for k in ("curr_oi", "last_oi", "volume", "prev_ask_volume", "prev_bid_volume", "prev_mid_volume",
              "prev_multi_leg_volume", "avg_price", "prev_total_premium"):
        df[k] = df[k].map(fnum)
    df["d_oi"] = df["curr_oi"] - df["last_oi"]
    return df


def leg_line(r, vol_key="volume"):
    v = r[vol_key]
    return (f"{occ_label(r['option_symbol']):12} OI {r['last_oi']:>8,.0f} -> {r['curr_oi']:>8,.0f} "
            f"({r['d_oi']:>+8,.0f})  vol {v:>7,.0f}  ask {r['prev_ask_volume'] / max(v, 1):4.0%} "
            f"bid {r['prev_bid_volume'] / max(v, 1):4.0%}  mleg {r['prev_multi_leg_volume'] / max(v, 1):4.0%}  "
            f"avg {r['avg_price']:5.2f}  prem ${r['prev_total_premium'] / 1e3:>7,.0f}k  "
            f"{side(r['prev_ask_volume'], r['prev_bid_volume'], v)}")


def next_session(d, sessions):
    later = [s for s in sessions if s > d]
    if later:
        return later[0]
    x = dt.date.fromisoformat(d) + dt.timedelta(days=1)
    while x.weekday() >= 5:
        x += dt.timedelta(days=1)
    return x.isoformat()


def scan_mode(c, a):
    tk = a.tickers[0].upper()
    print(f"==== {tk} UOA scan {a.start} .. {a.end}   (OI is start-of-day; dOI = what that day's trading left)")

    # 1. DAILY ----------------------------------------------------------
    vol = sorted(c.get_json(f"/stock/{tk}/options-volume", {"limit": "60"}).get("data", []),
                 key=lambda r: r["date"])
    sessions = [r["date"] for r in vol]
    print("\n  1. DAILY   (* = call or put volume >= 2x its 30-day average)")
    print(f"    {'date':10} {'callvol':>8} {'ask':>4} {'putvol':>7} {'ask':>4} {'call$':>8} {'put$':>8} "
          f"{'avg30 c/p':>13} {'callOI':>9} {'putOI':>9}")
    lo = (dt.date.fromisoformat(a.start) - dt.timedelta(days=14)).isoformat()
    for r in vol:
        if not (lo <= r["date"] <= a.end):
            continue
        cv, pv = fnum(r["call_volume"]), fnum(r["put_volume"])
        ac, ap = fnum(r["avg_30_day_call_volume"]), fnum(r["avg_30_day_put_volume"])
        star = "*" if (ac and cv >= 2 * ac) or (ap and pv >= 2 * ap) else " "
        print(f"   {star}{r['date']} {cv:>8,.0f} {fnum(r['call_volume_ask_side']) / max(cv, 1):4.0%} "
              f"{pv:>7,.0f} {fnum(r['put_volume_ask_side']) / max(pv, 1):4.0%} "
              f"{fnum(r['call_premium']) / 1e3:>7,.0f}k {fnum(r['put_premium']) / 1e3:>7,.0f}k "
              f"{ac:>6,.0f}/{ap:<6,.0f} {fnum(r['call_open_interest']):>9,.0f} {fnum(r['put_open_interest']):>9,.0f}")

    # 2. OI per traded day ----------------------------------------------
    days = [d for d in sessions if a.start <= d <= a.end]
    frames, pending_day = [], None
    print(f"\n  2. OI CHANGE BY TRADING DAY (top {a.per_day} by |dOI|)")
    for d in days:
        nd = next_session(d, sessions)
        df = oi_rows(c, tk, nd)
        if df.empty:
            pending_day = d
            print(f"    {d}: next session's OI ({nd}) not published yet -- PENDING")
            continue
        df["traded"] = d
        frames.append(df)
        print(f"    {d}  ({len(df)} contracts reported)")
        for _, r in df.reindex(df["d_oi"].abs().sort_values(ascending=False).index).head(a.per_day).iterrows():
            print("      " + leg_line(r))
    if not frames:
        print("  no OI-change data"); return
    W = pd.concat(frames, ignore_index=True)

    # 3. NET over the window ----------------------------------------------
    net = (W.groupby("option_symbol")
             .agg(d_oi=("d_oi", "sum"), vol=("volume", "sum"), ask=("prev_ask_volume", "sum"),
                  bid=("prev_bid_volume", "sum"), mleg=("prev_multi_leg_volume", "sum"),
                  prem=("prev_total_premium", "sum"), oi_now=("curr_oi", "last"),
                  days=("volume", lambda v: int((v > 0).sum())))
             .reset_index())
    net["stick"] = net["d_oi"] / net["vol"].clip(lower=1)
    print(f"\n  3. NET OVER THE WINDOW (judged days only)")
    for title, sub in (("BUILDERS (net OI added)", net.sort_values("d_oi", ascending=False).head(a.top)),
                       ("UNWINDS (net OI removed)", net.sort_values("d_oi").head(a.top))):
        print(f"    {title}")
        for _, r in sub.iterrows():
            if (title.startswith("B") and r.d_oi <= 0) or (title.startswith("U") and r.d_oi >= 0):
                continue
            print(f"      {occ_label(r.option_symbol):12} dOI {r.d_oi:>+8,.0f}  OI now {r.oi_now:>8,.0f}  vol {r.vol:>7,.0f} "
                  f"stick {r.stick:+5.0%}  ask {r.ask / max(r.vol, 1):4.0%} bid {r.bid / max(r.vol, 1):4.0%} "
                  f"mleg {r.mleg / max(r.vol, 1):4.0%}  prem ${r.prem / 1e3:>7,.0f}k  "
                  f"{side(r.ask, r.bid, r.vol)}  (traded {int(r.days)}d)")

    # 4. ORIGIN of the standouts + package legs ------------------------------
    stand = list(dict.fromkeys(
        list(net.sort_values("d_oi", ascending=False).head(a.origin)["option_symbol"]) +
        list(net.sort_values("d_oi").head(a.origin)["option_symbol"]) +
        list(net.sort_values("prem", ascending=False).head(a.origin)["option_symbol"])))
    print(f"\n  4. ORIGIN -- how the standouts' OI was built (biggest OI increases ever)")
    seen_pkg, closes = set(), None
    for s in stand:
        try:
            h = pd.DataFrame(c.get_json(f"/option-contract/{s}/historic").get("chains", []))
        except L.NoDataForDate:
            continue
        if h.empty:
            continue
        h = h.sort_values("date").reset_index(drop=True)
        for k in ("volume", "open_interest", "ask_volume", "bid_volume", "multi_leg_volume", "avg_price",
                  "total_premium", "underlying_close"):
            h[k] = h[k].map(fnum)
        if closes is None:
            closes = h.set_index("date")["underlying_close"]
        h["oi_next"] = h["open_interest"].shift(-1)
        h["d_oi"] = h["oi_next"] - h["open_interest"]
        print(f"    {occ_label(s)} ({s}): OI now {h['open_interest'].iloc[-1]:,.0f} (start of {h['date'].iloc[-1]})")
        for _, r in h.dropna(subset=["d_oi"]).sort_values("d_oi", ascending=False).head(3).iterrows():
            if r.d_oi <= 0:
                continue
            ml = r.multi_leg_volume / max(r.volume, 1)
            print(f"      {r.date}  dOI {r.d_oi:>+8,.0f}  vol {r.volume:>7,.0f}  {side(r.ask_volume, r.bid_volume, r.volume):6} "
                  f"mleg {ml:4.0%}  avg {r.avg_price:5.2f}  prem ${r.total_premium / 1e3:>7,.0f}k  stock {r.underlying_close:.2f}")
            if ml >= 0.9 and r.date not in seen_pkg:
                seen_pkg.add(r.date)
                nd = next_session(r.date, list(h["date"]))
                P = oi_rows(c, tk, nd)
                if P.empty:
                    continue
                legs = P[(P["prev_multi_leg_volume"] >= 0.9 * P["volume"]) &
                         (P["d_oi"].abs() >= max(a.package_min, 0.2 * abs(r.d_oi)))]
                if len(legs) > 1:
                    print(f"        PACKAGE traded {r.date} (multi-leg legs, OI after it):")
                    for _, g in legs.reindex(legs["d_oi"].abs().sort_values(ascending=False).index).iterrows():
                        print("          " + leg_line(g))

    # 5. ALERTS ----------------------------------------------------------
    alerts, older = [], None
    for _ in range(6):
        q = {"limit": "200"}
        if older:
            q["older_than"] = older
        try:
            rows = c.get_json(f"/stock/{tk}/flow-alerts", q).get("data", [])
        except L.NoDataForDate:
            break
        rows = [x for x in rows if older is None or x["created_at"] < older]
        if not rows:
            break
        alerts += rows
        older = rows[-1]["created_at"]
        if older[:10] < a.start:
            break
    al = [x for x in alerts if a.start <= x["created_at"][:10] <= a.end]
    print(f"\n  5. FLOW ALERTS {a.start} .. {a.end}: {len(al)}")
    grp = {}
    for x in al:
        key = (x["expiry"], float(x["strike"]), x["type"])
        g = grp.setdefault(key, dict(n=0, prem=0.0, ask=0.0, bid=0.0, sweep=0, days=set()))
        g["n"] += 1; g["prem"] += fnum(x["total_premium"]); g["ask"] += fnum(x["total_ask_side_prem"])
        g["bid"] += fnum(x["total_bid_side_prem"]); g["sweep"] += bool(x.get("has_sweep"))
        g["days"].add(x["created_at"][5:10])
    for (ex, k, ty), g in sorted(grp.items(), key=lambda kv: -kv[1]["prem"])[:a.top]:
        print(f"    {ex} {k:g}{ty[0].upper()}  {g['n']:>3} alerts  ${g['prem'] / 1e3:>7,.0f}k  "
              f"ask ${g['ask'] / 1e3:,.0f}k / bid ${g['bid'] / 1e3:,.0f}k  sweeps {g['sweep']}  {','.join(sorted(g['days']))}")

    # 6. PENDING -----------------------------------------------------------
    if pending_day:
        try:
            last = pd.DataFrame(c.get_json(f"/stock/{tk}/option-contracts").get("data", []))
        except L.NoDataForDate:
            last = pd.DataFrame()
        print(f"\n  6. PENDING -- {pending_day}'s biggest prints, judged by the next session's OI")
        if not last.empty:
            for k in ("volume", "ask_volume", "bid_volume", "open_interest", "total_premium", "multi_leg_volume"):
                last[k] = last[k].map(fnum)
            for _, r in last.sort_values("volume", ascending=False).head(a.top).iterrows():
                v = r.volume
                print(f"    {occ_label(r.option_symbol):12} vol {v:>7,.0f}  OI(start) {r.open_interest:>8,.0f}  "
                      f"ask {r.ask_volume / max(v, 1):4.0%} bid {r.bid_volume / max(v, 1):4.0%} "
                      f"mleg {r.multi_leg_volume / max(v, 1):4.0%}  prem ${r.total_premium / 1e3:>7,.0f}k  "
                      f"{side(r.ask_volume, r.bid_volume, v)}")
    if closes is not None:
        cw = closes[(closes.index >= lo) & (closes.index <= a.end)]
        print(f"\n  {tk} close: " + "  ".join(f"{d[5:]} {v:.2f}" for d, v in cw.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2] if __doc__ else None)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--scan", action="store_true", help="whole-ticker scan (one ticker, no --exp)")
    ap.add_argument("--exp", nargs="+", help="expiries YYYY-MM-DD (contracts mode)")
    ap.add_argument("--right", default="C", choices=("C", "P"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--per-day", type=int, default=6, help="scan: contracts shown per trading day")
    ap.add_argument("--origin", type=int, default=3, help="scan: standouts traced per list")
    ap.add_argument("--package-min", type=float, default=1000, help="scan: min |dOI| for a package leg")
    a = ap.parse_args()
    c = L.Client(os.environ["UW_API_KEY"])
    if a.scan:
        scan_mode(c, a)
    elif a.exp:
        contracts_mode(c, a)
    else:
        ap.error("give --exp (contracts mode) or --scan (whole ticker)")


if __name__ == "__main__":
    main()
