"""
check_book_fees.py -- RE-PRICE THE BOOK WITH WEBULL'S REAL FEES (user, 2026-11-14).

Every book sim nets directional_flow_backtester.COMMISSION_PCT = 1.5% of premium per round trip (sim_core.simulate).
Its comment says "commission + slippage", but slippage has been modelled on its own since 2026-09-10/16 (entry
min(mid+0.01, ask); exit bid - spread x cushion, CUSHION_CAP / FILL_COST), so the 1.5% counts slippage twice.
Webull charges no option commission, only transaction fees (user, 2026-11-14): 1 contract $0.05, 2 $0.09, 3 $0.14,
4 $0.18 -> ~$0.045 per contract per order, ~$0.10 per contract round trip (taken as $0.10, slightly high).
In premium terms that is 0.10 / (100 x entry price): 0.2% at $0.50, 0.03% at $3.

Same candidates (sim_core.deployed_candidates = full live gate stack), same walk, same slippage model; only the fee
changes: pnl_new = pnl_old + COMMISSION_PCT - 0.001 / entry. Fill models: `bot` (check_book_now's) and `botcap` (the
cushion the live bot charges since 2026-09-18, CUSHION_CAP per ticker). Enabled rules, then the disabled ones.

A re-score of numbers already reported, not new evidence: every gate was chosen on this same data.

RUN LOG
  2026-11-14 (full live gates, from 2024-08-20): the fee fix adds +1.4pp to every rule (median entries $0.83-$2.75).
    BOOK `bot`    +8.6% -> +10.0%/trade (OOS +11.9 -> +13.3);  `botcap` +15.0 -> +16.4 (OOS +21.8 -> +23.2).
    No decision changes: the disabled rules stay negative out of sample (AMZN botcap OOS -7.5 -> -6.1, META midday
    -5.1 -> -3.6, LULU -19.1 -> -17.6), and the paper-only set moves by the same 1.4pp (MSFT OOS -14.2 -> -12.8,
    GLD -12.2 -> -10.8; AVGO +2.8 -> +4.2 and SMH +13.6 -> +15.0 were already positive under botcap).

Usage:  python check_book_fees.py [--disabled]
"""
from __future__ import annotations

import argparse

import numpy as np

import sim_core
from config import CUSHION_CAP, RULES, TRAIL_PCT

SPLIT = sim_core.SPLIT
FEE_RT = 0.10          # $ per contract, round trip


_CAND = {}


def score(D, r, fill):
    if r["name"] not in _CAND:
        _CAND[r["name"]] = sim_core.deployed_candidates(D, r)
    cand = _CAND[r["name"]]
    if not cand:
        return None
    em, pol = sim_core.eod_mod(r), sim_core.policy_for(r, TRAIL_PCT)
    picks = []
    kw = {"cush_cap": CUSHION_CAP.get(r["ticker"])} if fill == "botcap" else {}
    res = sim_core.walk(cand, pol, em, fill=fill, picks_out=picks, **kw)
    out = []
    for (d, p), (ci, _xm) in zip(res, picks):
        e_mid, e_ask = cand[ci][2][0], cand[ci][2][1]
        entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
        # the sims net D.commission_frac (the real fee) since 2026-11-14; `old` re-applies the retired flat 1.5%
        out.append((d, p - D.COMMISSION_PCT + D.commission_frac(entry), p, entry))
    return out


def fmt(rows, k):
    if not rows:
        return "(none)"
    v = np.array([x[k] for x in rows])
    i = [x[k] for x in rows if x[0] < SPLIT]
    o = [x[k] for x in rows if x[0] >= SPLIT]
    f = lambda a: f"{np.mean(a) * 100:+6.1f}%" if a else "   n/a"
    return f"{v.mean() * 100:+6.1f}%  IS {f(i)}  OOS {f(o)}  win {(v > 0).mean():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disabled", action="store_true", help="also score the disabled rules")
    a = ap.parse_args()
    import directional_flow_backtester as D
    sets = [("ENABLED", [r for r in RULES if r.get("enabled", True)])]
    if a.disabled:
        sets.append(("DISABLED", [r for r in RULES if not r.get("enabled", True)]))
    for fill in ("bot", "botcap"):
        print(f"\n{'=' * 118}\n  FILL {fill}   old = 1.5% of premium round trip   new = ${FEE_RT:.2f}/contract round trip\n{'=' * 118}")
        for lab, rules in sets:
            book, names = [], []
            print(f"\n  -- {lab} --")
            for r in rules:
                try:
                    rows = score(D, r, fill)
                except Exception as e:                       # a disabled rule's data may be gone
                    print(f"    {r['name']:28s} skipped: {type(e).__name__}")
                    continue
                if not rows:
                    print(f"    {r['name']:28s} no candidates"); continue
                book += rows
                names += [r["name"]] * len(rows)
                po = "  (paper-only)" if r["name"] in sim_core.PAPER_ONLY else ""
                print(f"    {r['name']:28s} n={len(rows):4d} med entry ${np.median([x[3] for x in rows]):5.2f}"
                      f" | old {fmt(rows, 1)} | new {fmt(rows, 2)}{po}")
            if book and lab == "ENABLED":
                print(f"    {'BOOK':28s} n={len(book):4d}                  | old {fmt(book, 1)} | new {fmt(book, 2)}")
                core = [x for x, nm in zip(book, names) if nm not in sim_core.PAPER_ONLY]
                print(f"    {'BOOK w/o paper-only':28s} n={len(core):4d}                  | old {fmt(core, 1)} | new {fmt(core, 2)}")


if __name__ == "__main__":
    main()
