"""
probe_iv.py -- implied volatility, skew and "RIV" for one expiry across the
index / ETF group (operator, 2026-10-19).

    python probe_iv.py --expiry 2026-10-30 --start 2026-09-28 [--snaps 2026-10-05,2026-10-12,2026-10-16,2026-10-19]
                       [--out _iv_probe.json]

For SPY / SPX, QQQ / NDX, IWM / RUT:
  1. ATM IV of the expiry, daily   UW /volatility/term-structure?date=D (the
     expiry's row; verified to honor the date).
  2. SKEW, daily                   UW /historical-risk-reversal-skew?expiry=E&delta=25.
     Verified 10-19: honors the expiry; POSITIVE = puts priced over calls
     (SPY 10/30 on 10/19: RR +0.036, 25d put IV ~17-18% vs 25d call ~12.5%).
  3. SMILE snapshots               per-contract history (UW /option-contract/{occ}/historic,
     daily implied_volatility) on a strike grid: out-of-the-money puts below
     that day's spot, calls above, as moneyness vs IV.
  4. "RIV", three ways
     a. realized vs implied        /volatility/realized: 30-day IV vs 21-day
                                   realized (the fear premium). Index rows carry
                                   no price/RV at UW -> the ETF's RV stands in.
     b. IV rank / percentile       /iv-rank (1y rank) and /interpolated-iv (percentile).
     c. index vs ETF               the expiry's ATM IV and skew, index minus ETF.
Spots: ETF closes from contract history; index = ETF x index_gex.SEED_RATIO.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                           # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
import uw_options_data_lake as L                         # noqa: E402
import index_gex as IX                                   # noqa: E402

GROUP = [("SPY", "SPY", ["SPY"], 5), ("SPX", "SPY", ["SPX", "SPXW"], 50),
         ("QQQ", "QQQ", ["QQQ"], 5), ("NDX", "QQQ", ["NDX", "NDXP"], 250),
         ("IWM", "IWM", ["IWM"], 2), ("RUT", "IWM", ["RUT", "RUTW"], 25)]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get(c, path, q=None):
    try:
        return c.get_json(path, q).get("data", [])
    except L.NoDataForDate:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry", default="2026-10-30")
    ap.add_argument("--start", default="2026-09-28")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--snaps", default="2026-10-05,2026-10-12,2026-10-16,2026-10-19")
    ap.add_argument("--out", default=os.path.join(ROOT, "_iv_probe.json"))
    a = ap.parse_args()
    c = L.Client(os.environ["UW_API_KEY"])
    E, six = a.expiry, a.expiry[2:].replace("-", "")
    snaps = [s for s in a.snaps.split(",") if s]
    closes = {}
    res = {}
    for name, etf, roots, step in GROUP:
        if etf not in closes:   # ETF closes since the expiry was listed (months) -> enough for a 21-day RV
            h = c.get_json(f"/option-contract/{max([r for r in get(c, f'/stock/{etf}/option-contracts', {'expiry': E, 'option_type': 'put'}) if r['option_symbol'][-15:-9] == six], key=lambda r: f(r['open_interest']) or 0)['option_symbol']}/historic").get("chains", [])
            closes[etf] = {r["date"]: f(r["underlying_close"]) for r in h if f(r.get("underlying_close"))}
        ratio = 1.0 if name == etf else IX.SEED_RATIO[etf]
        spot = {d: v * ratio for d, v in closes[etf].items() if a.start <= d <= a.end}
        days = sorted(spot)
        # 1. ATM IV of the expiry, daily
        # the expiry may live on a weekly root only (SPXW / NDXP / RUTW): try each root
        atm = {}
        for d in days:
            for rt in roots:
                row = [r for r in get(c, f"/stock/{rt}/volatility/term-structure", {"date": d}) if r.get("expiry") == E]
                if row and row[0].get("date") == d:
                    atm[d] = f(row[0]["volatility"])
                    break
        # 2. 25-delta risk reversal, daily
        rr = {}
        for rt in roots:
            rr = {r["date"]: f(r["risk_reversal"]) for r in get(c, f"/stock/{rt}/historical-risk-reversal-skew",
                                                              {"expiry": E, "delta": "25"}) if a.start <= r["date"] <= a.end}
            if rr:
                break
        # 3. smile snapshots on a strike grid (OTM side per snapshot day)
        listed = {}
        for rt in roots:
            for typ in ("put", "call"):
                for r in get(c, f"/stock/{rt}/option-contracts", {"expiry": E, "option_type": typ}):
                    s = r["option_symbol"]
                    if s[-15:-9] == six:
                        listed[(s[-9], int(s[-8:]) / 1000)] = s
        lo = min(spot[d] for d in snaps if d in spot) * 0.92
        hi = max(spot[d] for d in snaps if d in spot) * 1.05
        want = set()
        k = (int(lo / step) + 1) * step
        while k <= hi:
            for right in ("P", "C"):
                cands = [kk for (rr_, kk) in listed if rr_ == right]
                if cands:
                    best = min(cands, key=lambda x: abs(x - k))
                    if abs(best - k) <= step / 2:
                        want.add((right, best))
            k += step
        ivs = {}
        for right, kk in sorted(want):
            sym = listed[(right, kk)]
            for r in c.get_json(f"/option-contract/{sym}/historic").get("chains", []):
                if r["date"] in snaps and f(r.get("implied_volatility")):
                    ivs.setdefault(r["date"], []).append((right, kk, f(r["implied_volatility"])))
        smile = {}
        for d in snaps:
            if d not in spot:
                continue
            pts = sorted((kk / spot[d] - 1, iv) for right, kk, iv in ivs.get(d, [])
                         if (right == "P" and kk <= spot[d]) or (right == "C" and kk > spot[d]))
            smile[d] = [[round(m * 100, 2), round(iv * 100, 2)] for m, iv in pts if -9 <= m * 100 <= 6]
        # 4. RIV
        rv = get(c, f"/stock/{etf}/volatility/realized")
        ivr = get(c, f"/stock/{roots[0]}/iv-rank")
        ipv = get(c, f"/stock/{roots[0]}/interpolated-iv")
        vr = get(c, f"/stock/{roots[0]}/volatility/realized")
        # UW aligns realized vol to the NEXT 21 days (recent rows are null), so the
        # trailing 21-day realized vol is computed here from the ETF's daily closes
        import math
        cl = sorted(closes[etf].items())
        rv21 = {}
        for i in range(21, len(cl)):
            rets = [math.log(cl[j][1] / cl[j - 1][1]) for j in range(i - 20, i + 1)]
            m = sum(rets) / len(rets)
            rv21[cl[i][0]] = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1) * 252)
        riv = dict(
            iv30_vs_rv21=[[r["date"], f(r["implied_volatility"]), rv21.get(r["date"])]
                          for r in vr if a.start <= r["date"] <= a.end],
            iv_rank_1y=[[r["date"], f(r["iv_rank_1y"]), f(r["volatility"])] for r in ivr],
            percentile=[[r.get("days"), f(r.get("volatility")), f(r.get("percentile"))] for r in ipv])
        res[name] = dict(etf=etf, roots=roots, spot=spot, atm=atm, rr=rr, smile=smile, riv=riv)
        a0, a1 = (atm[days[0]] if days and days[0] in atm else None), (atm.get(days[-1]) if days else None)
        pk = max(atm.items(), key=lambda kv: kv[1]) if atm else (None, None)
        rk = sorted(rr)
        print(f"\n== {name}  spot {spot[days[0]]:,.2f} -> {spot[days[-1]]:,.2f}")
        print(f"   {E} ATM IV: {days[0]} {100 * (a0 or 0):.1f}%  peak {pk[0]} {100 * (pk[1] or 0):.1f}%  "
              f"{days[-1]} {100 * (a1 or 0):.1f}%")
        if rk:
            rpk = max(rr.items(), key=lambda kv: kv[1])
            print(f"   25d risk reversal (puts over calls, vol pts): {rk[0]} {100 * rr[rk[0]]:+.1f}  peak {rpk[0]} "
                  f"{100 * rpk[1]:+.1f}  {rk[-1]} {100 * rr[rk[-1]]:+.1f}")
        for d, pts in smile.items():
            print(f"   smile {d}: " + "  ".join(f"{m:+.0f}%:{iv:.1f}" for m, iv in pts[::max(1, len(pts) // 8)]))
        last = [x for x in riv["iv30_vs_rv21"] if x[1] is not None]
        if last:
            d, iv, rvv = last[-1]
            rv_last = next((x[2] for x in reversed(last) if x[2] is not None), None)
            print(f"   RIV a. 30d IV {100 * iv:.1f}% vs trailing 21d realized {100 * (rv_last or 0):.1f}% "
                  f"(premium {100 * (iv - (rv_last or 0)):+.1f} pts; RV from the {etf} closes)")
        if riv["iv_rank_1y"]:
            d, rank, vol = sorted(riv["iv_rank_1y"])[-1]
            print(f"   RIV b. IV rank 1y {rank:.0f} (IV {100 * (vol or 0):.1f}%)  percentiles "
                  + " ".join(f"{p[0]}d:{100 * (p[2] or 0):.0f}%" for p in riv["percentile"][:4]))
    for idx, etf in (("SPX", "SPY"), ("NDX", "QQQ"), ("RUT", "IWM")):
        A, B = res.get(idx), res.get(etf)
        if not A or not B:
            continue
        common = sorted(set(A["atm"]) & set(B["atm"]))
        if common:
            d0, d1 = common[0], common[-1]
            print(f"\nRIV c. {idx} vs {etf} {E}: ATM IV diff {d0} {100 * (A['atm'][d0] - B['atm'][d0]):+.1f} -> "
                  f"{d1} {100 * (A['atm'][d1] - B['atm'][d1]):+.1f} pts;  skew diff "
                  + (f"{100 * (A['rr'].get(d1, 0) - B['rr'].get(d1, 0)):+.1f} pts on {d1}" if d1 in A["rr"] and d1 in B["rr"] else "n/a"))
    json.dump(res, open(a.out, "w", encoding="utf-8"), default=str)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
