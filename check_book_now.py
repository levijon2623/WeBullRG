# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_book_now.py
=================
"What is the book worth, exactly as it is configured right now?"

Reads config.RULES as deployed (including the new trailing exit) and scores it
on the fill model the bot actually runs:

  * SEQUENTIAL fills  -- one position per ticker (bot_runner.py:1288). The
    as-screened backtester counts every trigger independently, which inflated
    the book ~10x in trade count and ~2.5x in OOS expectancy.
  * TRAILING exit     -- config.TRAIL_PCT / per-rule trail_pct, peak tracked on
    the bid, no take-profit while trailing.
  * MID and REALISTIC fills -- realistic = enter at ASK, exit at BID, which is
    close to what bot_runner actually does (marketable limit at bid - cushion).

  * FULL LIVE GATE STACK -- sim_core.deployed_candidates (since 2026-10-13):
    flow_zscore, skip_macro_am, dmi_confirm and ema_confirm on top of the
    builder's own gates. Before that this script scored the bare builder.

Everything here nets the real per-contract fee (D.commission_frac) since 2026-11-14; before that a flat 1.5%.
Per-rule lines and the headline use the RESEARCH DEFAULT fill: `botcap` with sim_core.research_cap (the tape-
measured exit cushion on SPY/QQQ/IWM, CUSHION_CAP elsewhere).

RUN LOG
  2026-10-13 full gates vs the bare builder (realistic "bot" fills):
    BARE  n=392  +3.2%/trade  IS +1.7  OOS +4.6  win 0.36  maxLL 20
    FULL  n=328  +8.6%/trade  IS +5.2  OOS +11.9  win 0.39  maxLL 10
    Changed rules: SPY CHOP CALL 11 -> 8 (z-score), QQQ 32 -> 29 / IWM 105 -> 92
    (macro-morning skip; IWM +5.0 -> +10.2), MSFT 50 -> 30 (DMI), GLD 47 -> 22
    (EMA; -26.9 -> -8.1). The rest are identical. Matches check_trigger_paths'
    TAKEN set exactly (328, +8.6%). The gates were each chosen on this same
    data, so this corrects what the book CLAIMS -- it is not new evidence.

  2026-11-14 research default (real per-contract fee + tape-measured SPY/QQQ/IWM exit cushion):
    MID      n=328  +28.2%  OOS +36.5%
    bot      n=328  +10.0%  OOS +13.3%   (legacy cushion, real fee; was +8.6 / +11.9 at the flat 1.5%)
    DEFAULT  n=328  +18.1%  IS +12.4  OOS +23.6  win 0.42  maxLL 8  6/6 slices positive
    The cushion change moves only SPY/QQQ/IWM (IWM +10.2 -> +18.3, QQQ +17.6 -> +26.2); the rest is the fee.
    No rule changes sign out of sample; MSFT and GLD stay negative OOS.

Usage:  python check_book_now.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from check_config_walkforward import _slice_idx
import sim_core

HIST = "historical"
SPLIT = pd.Timestamp("2025-08-21").date()


def _line(lbl, pnls, width=24):
    if not pnls:
        return f"    {lbl:{width}} (none)"
    v = np.array([p for _, p in pnls])
    i = [p for d, p in pnls if d < SPLIT]
    o = [p for d, p in pnls if d >= SPLIT]
    c = m = 0
    for _, p in sorted(pnls):
        c = c + 1 if p <= 0 else 0
        m = max(m, c)
    sl = [[] for _ in range(6)]
    for d, p in pnls:
        k = _slice_idx(d)
        if k is not None:
            sl[k].append(p)
    npop = sum(1 for b in sl if len(b) >= 4)
    slc = " ".join(f"S{j+1}{np.mean(b)*100:+.0f}" if len(b) >= 4 else f"S{j+1}··"
                   for j, b in enumerate(sl))
    return (f"    {lbl:{width}} n={len(v):>4}  {v.mean()*100:>+6.1f}%  "
            f"IS {np.mean(i)*100 if i else float('nan'):>+6.1f}%  "
            f"OOS {np.mean(o)*100 if o else float('nan'):>+6.1f}%  "
            f"win {(v>0).mean():.2f}  maxLL {m:>2}  tot {v.sum():>+7.2f}  "
            f"pop {npop}/6  [{slc}]")


def run(a):
    import directional_flow_backtester as D
    from config import (RULES, TRAIL_PCT, SIZING_TARGET_PREMIUM_PCT,
                        MAX_PORTFOLIO_RISK_PCT)

    rules = [r for r in RULES if r.get("enabled", True)]
    if getattr(a, "rules", None):
        rules = [r for r in rules if r["name"] in a.rules]
    if getattr(a, "drop", None):
        rules = [r for r in rules if r["name"] not in a.drop]
    per_rule, mid_book, real_book, bot_book = {}, [], [], []

    for r in rules:
        # ONE builder, ONE simulator -- see sim_core's header. `rl` now carries
        # the live exit cushion (bid - 1.5x spread on losses, 0.5x on gains),
        # which this script did NOT model before 2026-09-10 and which is worth
        # ~10pp/trade book-wide. `m_` stays mid-fill for the historical contrast.
        # FULL LIVE GATES (2026-10-13): the bare builder skipped flow_zscore,
        # skip_macro_am, dmi_confirm and ema_confirm, so "as configured" was
        # not. --bare reproduces the old numbers for comparison.
        cand = (sim_core.build_candidates(D, r) if getattr(a, "bare", False)
                else sim_core.deployed_candidates(D, r))
        if not cand:
            continue
        em = sim_core.eod_mod(r)
        pol = sim_core.policy_for(r, TRAIL_PCT)
        tp_ = float(r.get("trail_pct", TRAIL_PCT) or 0.0)
        # `bot` mirrors bot_runner exactly: enter min(mid+0.01, ask), exit
        # bid - spread*(0.5|1.5). NOT ask-in -- the bot never pays the ask, and
        # charging it invents ~half a spread per trade (worth +2.7pp book-wide,
        # far more on wide-quote names). `mid` is kept as the frictionless
        # contrast. See check_fill_sensitivity.py for the full band.
        m_ = sim_core.walk(cand, pol, em, fill="mid")
        rl = sim_core.walk(cand, pol, em, fill="bot")
        # RESEARCH DEFAULT since 2026-11-14: `botcap` with the tape-measured cushion (sim_core.research_cap) and the
        # real per-contract fee (D.commission_frac, now netted by every sim). Per-rule lines use it.
        rd = sim_core.walk(cand, pol, em, fill="botcap", cush_cap=sim_core.research_cap(r["ticker"]))
        per_rule[r["name"]] = (m_, rd, tp_)
        mid_book += m_
        real_book += rd
        bot_book.append(rl)

    print("=" * 122)
    print(f"  THE BOOK AS CONFIGURED NOW   TRAIL_PCT={TRAIL_PCT}   sequential fills   split {SPLIT}")
    print("=" * 122)

    print("\n  -- per rule (RESEARCH DEFAULT: mid+0.01 in, tape-measured exit cushion, real per-contract fees) --")
    for rn, (m_, rl, tp_) in sorted(per_rule.items(), key=lambda x: -np.mean(
            [p for d, p in x[1][1] if d >= SPLIT] or [-9])):
        print(_line(f"{rn}", rl))

    print("\n  -- BLENDED BOOK --")
    print(_line("MID fills", mid_book, 24))
    print(_line("legacy `bot` cushion", [x for b in bot_book for x in b], 24))
    print(_line("RESEARCH DEFAULT", real_book, 24))

    v = np.array([p for _, p in real_book])
    o = np.array([p for d, p in real_book if d >= SPLIT])
    days = len({d for d, _ in real_book})
    yrs = (max(d for d, _ in real_book) - min(d for d, _ in real_book)).days / 365.25
    print("\n  -- what that means, roughly --")
    print(f"    {len(v)} trades over {days} trading days / {yrs:.1f} years  "
          f"= {len(v)/max(yrs,1e-9):.0f} trades/yr ({len(v)/max(days,1):.2f} per active day)")
    print("    net expectancy per trade (research default, after $0.10/contract round-trip fees):")
    print(f"       all {v.mean()*100:+.2f}%   IS {np.mean([p for d,p in real_book if d<SPLIT])*100:+.2f}%   "
          f"OOS {o.mean()*100:+.2f}%")
    for eq in (25_000, 50_000):
        prem = eq * SIZING_TARGET_PREMIUM_PCT
        print(f"    at ${eq:,} equity and {SIZING_TARGET_PREMIUM_PCT*100:.0f}% premium/trade "
              f"(~${prem:,.0f} at risk per trade):")
        print(f"       OOS ${prem*o.mean():+,.0f}/trade  ->  ${prem*o.mean()*len(v)/max(yrs,1e-9):+,.0f}/yr "
              f"({prem*o.mean()*len(v)/max(yrs,1e-9)/eq*100:+.1f}% of equity/yr, un-compounded)")
    print(f"\n    caveat: OOS n={len(o)} trades. Sizing is flat-premium so the % of equity "
          f"scales with SIZING_TARGET_PREMIUM_PCT, not with the win rate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=None,
                    help="score ONLY these rules. NOTE: hand-picking rules that already "
                         "tested well is SELECTION ON THE OUTCOME and the resulting book "
                         "number is optimistically biased. See the caveat printed below.")
    ap.add_argument("--drop", nargs="*", default=None, help="exclude these rules")
    ap.add_argument("--bare", action="store_true",
                    help="the pre-2026-10-13 builder: skips flow_zscore / skip_macro_am / "
                         "dmi_confirm / ema_confirm (for comparison only)")
    a = ap.parse_args()
    run(a)
