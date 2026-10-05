# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0.0", "numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_implied_breakout.py
=========================
STAGE 1 of the user's idea (2026-10-03): buy a CALL when price BREAKS ABOVE the
day's implied high, a PUT when it breaks below the implied low -- the band
from implied_move.py / check_implied_move. If this makes enough trades, stage
2 adds a momentum filter (RSI / fast EMA stack / delta-weighted candle) -- a
SEARCH, so it will need walk-forward selection (METHODOLOGY 5), not a pick.

WHY BREAKOUT, NOT FADE: check_implied_move_gate showed flow triggers that were
already stretched to the band did WORSE (-5.7pp OOS). The trigger is momentum;
this asks whether the band's own edge is a momentum entry.

TRIGGER (no lookahead)
    Band = that day's WALK-FORWARD 09:35 band at tau .65 (quantiles from prior
    days only; anchor = the 09:35 bar close). From the 09:36 bar to the 14:59
    bar, the FIRST 1m close above high65 -> CALL, the FIRST close below low65
    -> PUT. One break per side per ticker-day (first-break only).
    Price = historical/{T}.parquet 1m closes, RTH by the clock.
CONTRACT / EXIT -- all book defaults, fixed in advance
    ATM, dte [0, 1], $0.50 entry floor (sim_core.build_candidates at the break
    minute), policy_for default (trail50), EOD 15:55, fill "bot" + cushion.
    One position per TICKER: calls and puts are walked together, so a call
    still open blocks that day's put break.
SAMPLE  SPY, QQQ, IWM; 2024-08-20 .. 2026-09-18 (straddle lake end).
        IS < 2025-08-21 <= OOS; METHODOLOGY 8 slices.

CONTROLS
    PLACEBO x20  breakouts of ANOTHER day's band widths (% of anchor, same
                 ticker) hung on TODAY's 09:35 anchor: same machinery, same
                 distance distribution, none of that day's option pricing.

SECOND HYPOTHESIS -- FADE (what the user actually meant; registered
2026-10-03 while the first run was still building, before any output was seen)
    At the SAME first-break minutes, buy the OPPOSITE option: a CALL when price
    breaks BELOW the implied low, a PUT when it breaks ABOVE the high. Scored
    on C1-C6 below exactly like BREAKOUT, its C4 against the FADE of the same
    20 placebo bands. The two arms are mirror images on identical minutes, so
    at most one can be expected to pass; if both pass, the edge is timing or
    volatility rather than direction, and that is the reading.

PRE-REGISTERED CRITERIA (fixed 2026-10-03, before the first scored run)
    C1  pooled IS per-trade mean > 0
    C2  pooled OOS per-trade mean > 0
    C3  >= 5 of 6 slices populated (>= 3 trades) AND positive
    C4  OOS mean > p95 of the 20 placebo OOS means
    C5  OOS mean > 0 in >= 4 of the 6 ticker x direction cells
    C6  OOS mean > 0 under fill = mid, bot AND worst
    PASS = C1..C6. C1-C3, C5, C6 without C4 = "breakouts pay, but the
    implied band is no better than any level that far away".

REPORTED, NOT SCORED
    trade / day counts per cell (the stage-2 power question), win rate, mean
    winner / loser, day-block bootstrap 90% CI of the OOS mean, the same
    trigger at tau .90 (both arms).

MACHINERY (run before scoring)
    M1  every break minute is >= 09:36 and its close is beyond the band, and
        the previous minute's close was not (a first break, not a gap state)
    M2  the union build (one build_candidates per ticker x direction over all
        arms' minutes) equals a direct build for the real arm exactly
    M3  the placebo derangement maps no day to itself

RUN LOG
  2026-10-03 run 1 (old script, BREAKOUT scored, FADE reported): built and
    cached the option table; its output was not read before FADE was
    registered. Run 2 = this script on that cache.
  2026-10-03 run 2: M1 (936 first breaks) M2 M3 PASS.
    BREAKOUT  FAIL on all six. n=834 / 399 days, IS -17.3%, OOS -5.1% (90% CI
        -17.5..+9.8), slices 1/6, win 0.21; placebo median -7.8% / p95 -2.0%;
        mid +0.6 bot -5.1 worst -5.4. tau .90: OOS -8.1%.
    FADE      FAIL on all six. n=802 / 382 days, IS -7.7%, OOS -10.4% (CI
        -17.7..-2.6), slices 0/6, win 0.20; placebo median -9.4% / p95 -6.2%.
        tau .90: OOS -14.4%.
    Neither direction beats its placebo: a break of the implied band carries
    no directional edge either way. Both arms lose about what any 0DTE ATM
    entry costs with this exit. Positive cells (SPY/QQQ breakout puts OOS
    +5.6/+12.7) are negative IS -- noise.

Usage:
  python check_implied_breakout.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import sim_core as SC                  # noqa: E402
import check_implied_move as CI        # noqa: E402
from check_implied_move_gate import derangement, screen_rule   # noqa: E402

TICKERS = ("SPY", "QQQ", "IWM")
DIRS = ("CALL", "PUT")
FIRST, LAST = 9 * 60 + 36, 14 * 60 + 59
N_PLACEBO = 20
SEED = 20261004
FILLS = ("mid", "bot", "worst")
CACHE = os.path.join("_implied_move_cache", "breakout_table.pkl")


def bands():
    """{(tk, day): {tau: (A, up_pct_hat, dn_pct_hat)}} -- the 09:35 band."""
    st = CI.straddles()
    B = {t: CI.bars(t) for t in TICKERS}
    cal = sorted(set().union(*(set(b) for b in B.values())))
    out = {}
    for tk in TICKERS:
        rows, _ = CI.build_family("am", st, tk, B[tk], cal)
        for x in CI.walk(rows):
            if x["pred"]:
                out[(tk, x["day"])] = {t: (x["A"], x["pred"][t]["up_hat"], x["pred"][t]["dn_hat"])
                                       for t in (0.65, 0.90)}
    return out, B


def breaks(bd, B, tau=0.65, perm=None):
    """{(tk, dir): [(date, mod)]} first breaks. `perm` = {tk: {day: src}}."""
    out = {(tk, d): [] for tk in TICKERS for d in DIRS}
    for tk in TICKERS:
        for (t, day), v in bd.items():
            if t != tk or day < SC.DEPLOYED_START:
                continue
            A, up, dn = v[tau]
            if perm is not None:
                src = bd.get((tk, perm[tk][day]))
                if src is None:
                    continue
                up, dn = src[tau][1], src[tau][2]
            hi, lo = A * (1 + up / 100), A * (1 - dn / 100)
            g = B[tk].get(day)
            if g is None:
                continue
            g = g.filter((g["mod"] >= FIRST) & (g["mod"] <= LAST))
            mods, closes = g["mod"].to_list(), g["close"].to_list()
            for d, hit in (("CALL", lambda c: c > hi), ("PUT", lambda c: c < lo)):
                for m, c in zip(mods, closes):
                    if hit(c):
                        out[(tk, d)].append((day, m))
                        break
    return out


def stub(day, mod, direction):
    ts = pd.Timestamp(day) + pd.Timedelta(minutes=mod)
    return dict(date=day, ts=ts, hour=mod // 60, dir=direction, abs_flow=1.0,
                thr={p: 0.0 for p in (50, 60, 65, 70, 75, 80, 85, 90, 95)})


def union_table(D, need):
    """{(tk, dir, date, mod): candidate} from ONE build per ticker x direction."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            have = pickle.load(f)
        if need <= set(have["asked"]):
            return have["table"]
    table, asked = {}, set()
    for tk in TICKERS:
        for direction in DIRS:
            ms = sorted({(d, m) for (t, dr, d, m) in need if t == tk and dr == direction})
            print(f"    build {tk} {direction}: {len(ms)} minutes", flush=True)
            for c in SC.build_candidates(D, screen_rule(tk, direction, 50),
                                         trigs=[stub(d, m, direction) for d, m in ms]):
                table[(tk, direction, c[0], c[1])] = c
            asked |= {(tk, direction, d, m) for d, m in ms}
    with open(CACHE, "wb") as f:
        pickle.dump(dict(table=table, asked=asked), f)
    return table


def walk_arm(table, brk, fill="bot", flip=False):
    """{(tk, dir): [(date, pnl)]} -- calls and puts walked TOGETHER per ticker."""
    pol = SC.policy_for(screen_rule("SPY", "CALL", 50))
    out = {(tk, d): [] for tk in TICKERS for d in DIRS}
    for tk in TICKERS:
        cand = []
        for d in DIRS:
            buy = ("PUT" if d == "CALL" else "CALL") if flip else d
            for day, m in brk[(tk, d)]:
                c = table.get((tk, buy, day, m))
                if c is not None:
                    cand.append((c[0], c[1], c[2], d))
        cand.sort(key=lambda x: (x[0], x[1]))
        picks = []
        rows = SC.walk([(a, b, c) for a, b, c, _ in cand], pol, SC.DEFAULT_EOD,
                       fill=fill, picks_out=picks)
        for (ci, _x), r in zip(picks, rows):
            out[(tk, cand[ci][3])].append(r)
    return out


def pooled(res):
    return [r for v in res.values() for r in v]


def m(rows, oos):
    v = [r[1] for r in rows if (r[0] >= SC.SPLIT) == oos]
    return float(np.mean(v)) if v else float("nan")


def boot(rows, n=2000):
    rng = np.random.default_rng(SEED)
    by = {}
    for d, p in ((r[0], r[1]) for r in rows if r[0] >= SC.SPLIT):
        by.setdefault(d, []).append(p)
    days = sorted(by)
    bs = [np.mean([x for i in rng.choice(len(days), len(days)) for x in by[days[i]]]) for _ in range(n)]
    return float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def main():
    import directional_flow_backtester as D
    bd, B = bands()
    last = max(d for (_t, d) in bd)
    real = breaks(bd, B)
    r90 = breaks(bd, B, tau=0.90)
    rng = np.random.default_rng(SEED)
    perms = [{tk: derangement(sorted(d for (t, d) in bd if t == tk), rng) for tk in TICKERS}
             for _ in range(N_PLACEBO)]
    pbrk = [breaks(bd, B, perm=p) for p in perms]

    # ---- machinery
    bad = 0
    for (tk, d), evs in real.items():
        for day, mm in evs:
            A, up, dn = bd[(tk, day)][0.65]
            hi, lo = A * (1 + up / 100), A * (1 - dn / 100)
            g = B[tk][day]
            px = dict(zip(g["mod"].to_list(), g["close"].to_list()))
            c, prev = px[mm], px.get(mm - 1)
            beyond = c > hi if d == "CALL" else c < lo
            was = (prev > hi if d == "CALL" else prev < lo) if (prev is not None and mm > FIRST) else False
            bad += (mm < FIRST) or (not beyond) or was
    print(f"  M1 first-break minutes: {'PASS' if bad == 0 else 'FAIL'} ({bad} bad of "
          f"{sum(len(v) for v in real.values())})")
    m3 = all(all(a != b for a, b in p[tk].items()) for p in perms for tk in TICKERS)
    print(f"  M3 derangements fixed-point free: {'PASS' if m3 else 'FAIL'}")

    need = set()
    for brk in [real, r90] + pbrk:
        for (tk, d), evs in brk.items():
            for day, mm in evs:
                need.add((tk, d, day, mm))
                need.add((tk, "PUT" if d == "CALL" else "CALL", day, mm))    # FLIP
    table = union_table(D, need)
    # M2: direct build of the real arm for one ticker x direction
    tk0, d0 = "QQQ", "CALL"
    direct = SC.build_candidates(D, screen_rule(tk0, d0, 50),
                                 trigs=[stub(day, mm, d0) for day, mm in real[(tk0, d0)]])
    via = [table[(tk0, d0, day, mm)] for day, mm in real[(tk0, d0)] if (tk0, d0, day, mm) in table]
    same = len(direct) == len(via) and all(a[0] == b[0] and a[1] == b[1] and a[2][0] == b[2][0]
                                          and np.array_equal(a[2][2], b[2][2]) for a, b in zip(direct, via))
    print(f"  M2 union build == direct build ({tk0} {d0}, {len(direct)} candidates): {'PASS' if same else 'FAIL'}")
    if bad or not m3 or not same:
        sys.exit("machinery checks failed -- not scoring")

    # ---- scoring: BREAKOUT and FADE, identical criteria
    verdicts = {}
    for label, flip in (("BREAKOUT", False), ("FADE", True)):
        verdicts[label] = score(label, flip, table, real, r90, pbrk, last)
    if all(v == "PASS" for v in verdicts.values()):
        print("\n  BOTH arms passed on identical minutes: the edge is timing / volatility, not direction")


def score(label, flip, table, real, r90, pbrk, last):
    res = {f: walk_arm(table, real, fill=f, flip=flip) for f in FILLS}
    rows = pooled(res["bot"])
    st = SC.stat(rows)
    lo, hi = boot(rows)
    what = ("CALL on a break ABOVE the high, PUT on a break BELOW the low" if not flip
            else "CALL on a break BELOW the low, PUT on a break ABOVE the high")
    print(f"\n  {label}: {what}  (tau .65, first break, trail50, fill=bot, bands to {last})")
    print(f"    n={st['n']} days={st['nd']}  IS {st['is_'] * 100:+.1f}% (n {st['nis']})  "
          f"OOS {st['oos'] * 100:+.1f}% (n {st['noos']}, 90% CI {lo * 100:+.1f}..{hi * 100:+.1f})  "
          f"win {st['win']:.2f}  mean winner {st['mw'] * 100:+.0f}%  loser {st['ml'] * 100:+.0f}%  "
          f"slices {st['nposs']}/{st['npop']}")
    cell_pos = 0
    for (tk, d), v in res["bot"].items():
        bought = ("PUT" if d == "CALL" else "CALL") if flip else d
        side = "break up  " if d == "CALL" else "break down"
        s = SC.stat(v)
        if s is None:
            print(f"      {tk} {side} -> {bought}: none")
            continue
        cell_pos += bool(s["oos"] > 0)
        print(f"      {tk} {side} -> {bought:4}  n={s['n']:3} days {s['nd']:3}  IS {s['is_'] * 100:+6.1f}% "
              f"(n {s['nis']})  OOS {s['oos'] * 100:+6.1f}% (n {s['noos']})  win {s['win']:.2f}")
    pm = [m(pooled(walk_arm(table, p, flip=flip)), True) for p in pbrk]
    p95 = float(np.nanquantile(pm, 0.95))
    fills = {f: m(pooled(res[f]), True) for f in FILLS}
    crit = dict(C1=st["is_"] > 0, C2=st["oos"] > 0, C3=st["npop"] >= 5 and st["nposs"] >= 5,
                C4=st["oos"] > p95, C5=cell_pos >= 4, C6=all(v > 0 for v in fills.values()))
    print(f"    placebo OOS means: median {np.nanmedian(pm) * 100:+.1f}%  p95 {p95 * 100:+.1f}%")
    print("    fills OOS: " + "  ".join(f"{f} {v * 100:+.1f}%" for f, v in fills.items()))
    s9 = SC.stat(pooled(walk_arm(table, r90, flip=flip)))
    if s9:
        print(f"    tau .90 version: n={s9['n']} days={s9['nd']}  IS {s9['is_'] * 100:+.1f}%  "
              f"OOS {s9['oos'] * 100:+.1f}%  win {s9['win']:.2f}")
    print("    " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in crit.items()))
    core = crit["C1"] and crit["C2"] and crit["C3"] and crit["C5"] and crit["C6"]
    v = "PASS" if all(crit.values()) else "PAYS, BAND NOT SPECIAL" if core else "FAIL"
    print(f"    {label} VERDICT: {v}")
    return v


if __name__ == "__main__":
    main()
