# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_trigger_paths.py
======================
THE DEPLOYED FLOW TRIGGERS WITH NO EXIT: WHAT DOES THE PATH DO? (user, 2026-10-12)

"Take a look at the flow_triggers as currently deployed, but without an exit.
Just the price path that follows each trigger, and whether or not that path is
ever profitable."

DESCRIPTIVE -- no hypothesis is scored, nothing here changes a rule.

TRIGGERS  every enabled config.RULES entry with its FULL live gate stack:
          regime / amp / hours / dte, min_flow_pct over flow_window_days OR
          flow_zscore, amt_open, skip_macro_am, dmi_confirm (close-only, fail
          open when not warm), ema_confirm -- the same gate code as
          check_config_walkforward's DEPLOYED arm -- then sim_core's ONE
          builder for the contract the bot would buy and the $0.50 floor.
          Deployed window 2024-08-20 .. end of lake; split 2025-08-21.
          PAPER_ONLY rules (sim_core) are included -- they trade live -- and
          marked *.
PATH      the trigger's own contract, minute by minute, until the rule's flatten
          (15:55; META 15:00). Entry = the bot's limit, min(mid + 0.01, ask).
          SELL(i) = what selling at minute i would book: bid - 0.5 x spread
          (the bot's cushion on a profitable exit) less the 1.5% round-trip
          commission, as ROE on the entry.  EVER PROFITABLE = max SELL > 0.
          Underlying: s x (close(i) - close(trigger)) / close, in bp.
TWO POPULATIONS
          ALL    every trigger the rules match (as-screened; overlapping)
          TAKEN  the ones the live bot would actually have held -- one position
                 per ticker, deployed exit (sim_core.walk) -- with their real
                 booked result, so "it was green and still closed red" is visible.

RUN LOG
  2026-10-12 run 1: NaN-past-flatten counted as "not green" at the marks, and
    raw trigger counts clustered (META 612 triggers on 35 days) -- superseded.
  2026-10-12 run 2 (rule-day view added): 2,121 gated candidates on 192
    rule-days (144 calendar days); TAKEN 328.
    FIRST TRIGGER PER RULE-DAY (192): green after costs at some point 94%;
      reaches +10/+25/+50/+100%: 85/74/62/40%; median peak +78% at ~84 min;
      first green ~4 min. Median worst point before flatten -77%. Held to
      the close: mean +8%, median -40% (IS +11 / OOS +5) -- a few big
      winners carry it. Research set (no paper rules) +18%.
    UNDERLYING: coin flip for 15 min (48-52% in the trade's favour), then
      63% at 60 min, 72% at 120 min, 69% at the close; mean +40 bp at close.
      The trigger's direction shows up over hours, not minutes.
    TAKEN (deployed exit, one per ticker): 92% green at some point, booked
      +8.6% (win 39%); 53% were green at some point and still booked a loss
      (IWM 66%, AVGO 70%). Held-to-close on the same trades +25%, but that
      is not a policy comparison -- holding to the close changes which
      triggers are taken; the hold-to-close set IS the first-per-day set
      (+8.0%), level with the deployed exit (+8.6%; research set +18 vs +18).
    VERDICT (descriptive): nearly every trigger is profitable at some point;
      the hard part is the -70% typical drawdown on the way and a peak that
      comes late (~1.5 h) and is only knowable in hindsight. Holding to the
      close and the deployed exit earn about the same.

Usage:  python check_trigger_paths.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import directional_flow_backtester as D                   # noqa: E402
import sim_core                                            # noqa: E402
from check_config_walkforward import _flow_for             # noqa: E402
from check_flow_zscore import annotate_flow_z, _z_matched  # noqa: E402
from check_adx_dmi import _intraday_adx, _intra_asof       # noqa: E402
from amt_profile import amt_open_map, amt_ok               # noqa: E402
from macro_calendar import is_macro_am_day                 # noqa: E402
from config import RULES, TRAIL_PCT                        # noqa: E402

SPLIT = sim_core.SPLIT
LEVELS = (0.0, 0.10, 0.25, 0.50, 1.00)
MARKS = (5, 15, 30, 60, 120)


def gated_trigs(r):
    """The rule's triggers after its FULL deployed gate stack (mirrors
    check_config_walkforward eval_cfg(deployed=True); dex_pct_max unused live)."""
    tk = r["ticker"]
    flow = _flow_for(D, [tk])
    if flow.empty:
        return []
    trigs = D.triggers_for(flow, tk)
    gex, vol, trd = D.load_gex("historical", tk), D.load_volume_regime("historical", tk), \
        D.load_trend_regime("historical", tk)
    _d = set(gex) & set(vol) & set(trd)
    amp = {d: int(gex[d] == "NEGATIVE") + int(vol[d] == "LOWVOL") + int(trd[d] == "CHOP") for d in _d}
    reg_src = {"LOWVOL": vol, "NORMVOL": vol, "HIVOL": vol, "UPTREND": trd, "DOWNTREND": trd, "CHOP": trd}
    zs = r.get("flow_zscore")
    if zs:
        annotate_flow_z(trigs, int(zs.get("window_days") or 60))
        matched = _z_matched(D, r, trigs, gex, vol, trd, amp, reg_src, float(zs["k"]))
    else:
        D.annotate_flow_pct(trigs, int(r.get("flow_window_days") or 60))
        matched = [t for t, _ in D._rule_matched_trigs(r, trigs, gex, vol, trd, amp, reg_src)]
    want_bull = r["direction"].upper() == "CALL"
    amt = amt_open_map(tk) if r.get("amt_open") else {}
    dmi = r.get("dmi_confirm")
    iadx = _intraday_adx(tk, int(dmi.get("tf", 15)), close_only=True) if dmi else {}
    ema = D.load_ema_stack("historical", tk, int(r["ema_confirm"])) if r.get("ema_confirm") else None
    out = []
    for t in matched:
        d, ts = t["date"], t["ts"]
        if r.get("skip_macro_am") and is_macro_am_day(d):
            continue
        if r.get("amt_open") and not amt_ok(r["amt_open"], amt.get(d)):
            continue
        if dmi:
            ia = _intra_asof(iadx.get(d, []), ts)
            if ia is not None:
                agree = (ia[0] > ia[1]) == want_bull
                if agree != (dmi.get("mode") == "agree"):
                    continue
        if ema is not None:
            st = D.ema_state_at(ema, ts)
            if st is not None and st != ("BULL" if want_bull else "BEAR"):
                continue
        out.append(t)
    return out


def ul_closes(tk):
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60
                        + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 959))
         .select("date", "mod", "close").to_pandas())
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return {k: g.set_index("mod")["close"].reindex(np.arange(570, 960)).ffill().to_numpy(float)
            for k, g in d.groupby("date")}


def path_row(r, d, m, payload, ul, eod):
    e_mid, e_ask, cl, hi, lo, bid, ask, mod = payload
    entry = min(round(e_mid + 0.01, 2), round(e_ask, 2))
    keep = mod < eod
    if not keep.any():
        return None
    b, a, mm = bid[keep], ask[keep], mod[keep]
    sp = np.where((a > b) & (b > 0), a - b, 0.0)
    sell = (np.maximum(0.01, np.where(b > 0, b - 0.5 * sp, 0.0)) - entry) / entry - D.COMMISSION_PCT
    sell = np.where(b > 0, sell, np.nan)
    if np.all(np.isnan(sell)):
        return None
    ip = int(np.nanargmax(sell))
    prof = np.flatnonzero(sell > 0)
    row = dict(rule=r["name"], tk=r["ticker"], date=d, m=m, entry=entry,
               peak=float(sell[ip]), t_peak=int(mm[ip] - m),
               t_first=(int(mm[prof[0]] - m) if len(prof) else np.nan),
               mins_green=int(len(prof)), mins=int(len(sell)),
               last=float(sell[~np.isnan(sell)][-1]), trough=float(np.nanmin(sell)))
    for k in MARKS:
        j = np.searchsorted(mm, m + k)
        row[f"o{k}"] = float(sell[j]) if j < len(sell) and not np.isnan(sell[j]) else np.nan
    c = ul.get(d)
    s = 1 if r["direction"].upper() == "CALL" else -1
    if c is not None and 570 <= m < 960:
        i0, i1 = m - 570, min(eod, 960) - 570
        w = s * (c[i0:i1] - c[i0]) / c[i0] * 1e4
        row.update(ul_mfe=float(np.nanmax(w)), ul_mae=float(np.nanmin(w)), ul_last=float(w[-1]))
        for k in MARKS:
            row[f"u{k}"] = float(w[k]) if k < len(w) else np.nan
    return row


def summary(df, title, taken=False):
    if df.empty:
        print(f"    {title:28} (none)")
        return
    lv = "  ".join(f"{df['peak'].gt(x).mean():>4.0%}" for x in LEVELS)
    s = (f"    {title:28} n={len(df):>5} ({df['date'].nunique():>3}d)  ever>0/10/25/50/100%: {lv}   peak med {df['peak'].median():+5.0%}"
         f"  1st green {df['t_first'].median():>3.0f}m  peak at {df['t_peak'].median():>3.0f}m"
         f"  hold-to-close {df['last'].mean():+5.0%} (med {df['last'].median():+4.0%})")
    if taken:
        g2r = (df["peak"].gt(0) & df["booked"].le(0)).mean()
        s += f"\n    {'':28}   booked {df['booked'].mean():+5.1%}  win {df['booked'].gt(0).mean():.0%}" \
             f"   green at some point but booked a loss: {g2r:.0%} of trades"
    print(s)


def main():
    eligible = [r for r in RULES if r.get("enabled", True)]
    rows, taken = [], []
    ulc = {}
    for r in eligible:
        tk = r["ticker"]
        trigs = gated_trigs(r)
        if not trigs:
            print(f"  {r['name']}: no triggers"); continue
        # the builder re-applies hour/regime/amp/amt (idempotent); the flow gate
        # was already applied above (z-score or percentile), so pass it as flow_abs 0
        rb = {k: v for k, v in r.items() if k not in ("min_flow_pct", "flow_zscore")}
        rb["flow_abs"] = 0.0
        meta = []
        cand = sim_core.build_candidates(D, rb, trigs=trigs, meta_out=meta)
        if not cand:
            print(f"  {r['name']}: no candidates"); continue
        if tk not in ulc:
            ulc[tk] = ul_closes(tk)
        eod = sim_core.eod_mod(r)
        picks = []
        booked = sim_core.walk(cand, sim_core.policy_for(r, TRAIL_PCT), eod, fill="bot",
                               picks_out=picks)
        took = {ci: p for (ci, _), (_, p) in zip(picks, booked)}
        for ci, (d, m, payload) in enumerate(cand):
            row = path_row(r, d, m, payload, ulc[tk], eod)
            if row is None:
                continue
            row["paper"] = r["name"] in sim_core.PAPER_ONLY
            rows.append(row)
            if ci in took:
                taken.append(dict(row, booked=took[ci]))
        print(f"  {r['name']:24} gated triggers {len(trigs):>5}  candidates {len(cand):>5}  taken {len(picks):>4}")
    A, T = pd.DataFrame(rows), pd.DataFrame(taken)

    print("\n  SELL = bid - 0.5 spread, net of 1.5% commission, ROE on the bot's entry; to the rule's flatten.")
    print("\n  ======== ALL TRIGGERS (as-screened) ========")
    for name, g in A.groupby("rule", sort=False):
        summary(g, name + (" *" if g["paper"].iloc[0] else ""))
    print()
    summary(A, "BOOK, all rules")
    summary(A[~A["paper"]], "BOOK, research set (no *)")
    summary(A[A["date"] < SPLIT], "  IS  (to 2025-08-20)")
    summary(A[A["date"] >= SPLIT], "  OOS (from 2025-08-21)")

    F = A.sort_values(["rule", "date", "m"]).groupby(["rule", "date"]).head(1)
    print("\n  ======== FIRST TRIGGER PER RULE-DAY (one independent-ish path per day) ========")
    for name, g in F.groupby("rule", sort=False):
        summary(g, name + (" *" if g["paper"].iloc[0] else ""))
    print()
    summary(F, "BOOK, all rules")
    summary(F[~F["paper"]], "BOOK, research set (no *)")
    summary(F[F["date"] < SPLIT], "  IS  (to 2025-08-20)")
    summary(F[F["date"] >= SPLIT], "  OOS (from 2025-08-21)")

    print("\n  ======== TAKEN (one position per ticker, deployed exit) ========")
    for name, g in T.groupby("rule", sort=False):
        summary(g, name + (" *" if g["paper"].iloc[0] else ""), taken=True)
    print()
    summary(T, "BOOK, all rules", taken=True)
    summary(T[~T["paper"]], "BOOK, research set (no *)", taken=True)

    for lbl, X in (("TAKEN trades", T), ("FIRST per rule-day", F)):
        avg_path(X, lbl)
    A.to_pickle(os.path.join("_implied_move_cache", "trigger_paths.pkl"))
    T.to_pickle(os.path.join("_implied_move_cache", "trigger_paths_taken.pkl"))


def avg_path(A, lbl):
    pos = lambda x: (x.dropna() > 0).mean()                 # noqa: E731 -- past-flatten = absent
    print(f"\n  ======== THE AVERAGE PATH ({lbl}, n={len(A)}) ========")
    print("    minutes after trigger:   " + "  ".join(f"{k:>6}" for k in MARKS) + "   close")
    print("    option SELL, mean      : " + "  ".join(f"{A[f'o{k}'].mean():>+6.1%}" for k in MARKS)
          + f"  {A['last'].mean():>+6.1%}")
    print("    option SELL, % > 0     : " + "  ".join(f"{pos(A[f'o{k}']):>6.0%}" for k in MARKS)
          + f"  {pos(A['last']):>6.0%}")
    print("    underlying, mean bp    : " + "  ".join(f"{A[f'u{k}'].mean():>+6.1f}" for k in MARKS)
          + f"  {A['ul_last'].mean():>+6.1f}")
    print("    underlying, % with trade: " + "  ".join(f"{pos(A[f'u{k}']):>5.0%}" for k in MARKS)
          + f"  {pos(A['ul_last']):>6.0%}")
    print("    still open at the mark : " + "  ".join(f"{A[f'o{k}'].notna().mean():>6.0%}" for k in MARKS))
    print(f"    underlying best / worst point before flatten, median: +{A['ul_mfe'].median():.0f} bp / "
          f"{A['ul_mae'].median():.0f} bp")
    print(f"    option worst point before flatten, median: {A['trough'].median():+.0%}")
    print(f"\n    minutes green (SELL > 0) out of minutes held, median: "
          f"{(A['mins_green'] / A['mins']).median():.0%}; never green: {A['peak'].le(0).mean():.0%}")


if __name__ == "__main__":
    main()
