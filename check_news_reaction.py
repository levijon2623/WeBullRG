# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26.0", "pandas>=2.0.0"]
# ///
"""
check_news_reaction.py
======================
DOES PRICE REACT TO HEADLINES, AND CAN A BOT SEE IT IN TIME? (user, 2026-10-23)

A SCREEN before building a news input for the bot. Two feeds, both shallow:
  UW   /news/headlines -- ~1,000 headlines/day, paging stops at page 500
       (2026-09-04 onward). Sources: Tradex (squawk-style wire), Business
       Wire / PR Newswire / GlobeNewswire (press releases), NYT, crypto. Has
       is_major (~36%); its sentiment field is "neutral" on every item.
  WB   Webull stock-news SPY (Benzinga only): 30-day retention (2026-09-23
       onward), ~70/day, with a POSITIVE / NEGATIVE / NEUTRAL label.
Bars: UW /stock/{T}/ohlc/1m regular session (cached in _news_cache/).

EVENT   a headline in RTH 09:35-15:45 ET; bursts collapsed (first per group
        per 3 minutes). m = the minute it was stamped.
MEASURES (SPY; QQQ reported)
  pre_k   log(open[m] / open[m-k])        the k minutes BEFORE the headline
  hit_k   log(close[m+k-1] / open[m])     the headline minute and after
  late_k  log(close[m+k] / close[m])      AFTER the headline minute -- what a
                                          bot reading the feed could still get
  k = 1 / 5 / 15.
BASELINE the same clock minute on every session in the window -> mean |x|.
  ratio = |x| / baseline; 1.0 = an ordinary minute. Day-block bootstrap CI.
  pre >> 1 means headlines FOLLOW moves (written about price); hit / late >> 1
  means price moves after them.
DIRECTION (WB sentiment only): sentiment x return, bp, for pre / hit / late.
SCREEN RULE (what would justify building the bot input): late_5 or late_15
  ratio CI above 1 for some group, or a WB directional late_k CI excluding 0,
  in the sign of the label.

Usage:  python check_news_reaction.py      (needs _news_cache/, built 10-23)

RUN LOG
  2026-10-23 run 1: UW 50,100 headlines (09-04 .. 10-23), 5,207 RTH events
    after de-bursting; SPY / QQQ bars 35 sessions. WB SPY feed 2,071 items
    (09-23 .. 10-23, then nothing: 30-day retention), 818 RTH events.
    UW: every group's ratio 0.93-1.07 with CIs spanning 1 -- Tradex major
      (n 2,140) hit1 1.03, late5 1.03, late15 1.03; press-release wires ~1.0;
      NYT/IBD ~0.95. Same on QQQ.
    WB: all labels mildly elevated 1.1-1.25, BEFORE as much as after (pre15
      1.23-1.26, CIs just above 1): Benzinga writes on busy, moving minutes.
    Direction (Benzinga label x SPY return, n 383): pre15 +0.26, hit1 +0.07,
      late1 +0.13, late5 -0.06, late15 +0.31 bp -- all CIs span 0, hit
      rate 0.48-0.53.
    Reported: Tradex is so dense that 99% of random minutes have a headline
      within 10 min; the 60 biggest SPY minutes all "have" one, mostly
      irrelevant (Kyiv air raid, Darfur); the relevant ones (ISM 10:00, Fed
      chair, "Iran deal reports wrong", Iran sanctions) land in the SAME minute
      as the move or a minute after it.
  VERDICT: the screen rule does not fire. Raw headlines -- UW or Benzinga,
    labelled or not -- carry nothing a bot could still trade 1-15 minutes on;
    the few that matter are priced inside their own minute. The open question
    is RELEVANCE: picking the handful of market-moving items out of ~1,000 a
    day (an AI classifier's job), untestable on 7 weeks without one.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
C = "_news_cache"
KS = (1, 5, 15)
WIRES = {"Business Wire", "PR NewsWire", "GlobeNewswire"}
N_BOOT = 2000
rng = np.random.default_rng(20261023)


def bars(tk):
    b = pd.read_parquet(f"{C}/bars_{tk}.parquet")
    t = pd.to_datetime(b["start_time"], utc=True).dt.tz_convert("America/New_York")
    b = b.assign(day=t.dt.date, mod=t.dt.hour * 60 + t.dt.minute)
    for c in ("open", "close"):
        b[c] = b[c].astype(float)
    b = b.drop_duplicates(["day", "mod"]).sort_values(["day", "mod"])
    O = b.pivot(index="day", columns="mod", values="open").reindex(columns=range(570, 960)).ffill(axis=1)
    Cl = b.pivot(index="day", columns="mod", values="close").reindex(columns=range(570, 960)).ffill(axis=1)
    return O, Cl


def measures(O, Cl, day, m):
    """dict of pre/hit/late for k, or None if out of range."""
    if day not in O.index:
        return None
    o, c = O.loc[day], Cl.loc[day]
    out = {}
    for k in KS:
        if m - k < 570 or m + k > 959:
            return None
        out[f"pre{k}"] = np.log(o[m] / o[m - k])
        out[f"hit{k}"] = np.log(c[m + k - 1] / o[m])
        out[f"late{k}"] = np.log(c[m + k] / c[m])
    return out


def baselines(O, Cl):
    """mean |x| per clock minute over all sessions, for each measure."""
    base = {}
    for k in KS:
        for nm in ("pre", "hit", "late"):
            vals = {}
            for m in range(575, 946):
                if m - k < 570 or m + k > 959:
                    continue
                if nm == "pre":
                    x = np.log(O[m] / O[m - k])
                elif nm == "hit":
                    x = np.log(Cl[m + k - 1] / O[m])
                else:
                    x = np.log(Cl[m + k] / Cl[m])
                vals[m] = float(np.nanmean(np.abs(x)))
            base[f"{nm}{k}"] = vals
    return base


def events(df, tcol, group_fn):
    t = pd.to_datetime(df[tcol], utc=True).dt.tz_convert("America/New_York")
    df = df.assign(day=t.dt.date, mod=t.dt.hour * 60 + t.dt.minute, grp=df.apply(group_fn, axis=1))
    df = df[(df["mod"] >= 575) & (df["mod"] <= 945) & df["grp"].notna()].sort_values(["grp", "day", "mod"])
    keep, last = [], {}
    for i, r in df.iterrows():
        key = (r["grp"], r["day"])
        if key in last and r["mod"] - last[key] < 3:
            continue
        last[key] = r["mod"]
        keep.append(i)
    return df.loc[keep]


def dboot(days, x):
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(ud))
    n = np.bincount(inv, minlength=len(ud)).astype(float)
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, len(ud), len(ud))
        bs.append(s[j].sum() / n[j].sum())
    return np.quantile(bs, [0.025, 0.975])


def ratio_table(ev, O, Cl, base, title):
    rows = []
    for _, e in ev.iterrows():
        mm = measures(O, Cl, e["day"], int(e["mod"]))
        if mm is None:
            continue
        rows.append(dict(grp=e["grp"], day=e["day"], **{k: abs(v) / base[k][int(e["mod"])] for k, v in mm.items()
                                                       if base[k].get(int(e["mod"]))}))
    R = pd.DataFrame(rows)
    print(f"\n  {title}: |move| / ordinary same-minute |move|  (1.00 = nothing special; [95% CI])")
    print("     group                 n     " + "  ".join(f"{c:>18}" for c in ("pre5", "hit1", "late1", "late5", "late15", "pre15")))
    for g, s in R.groupby("grp"):
        cells = []
        for c in ("pre5", "hit1", "late1", "late5", "late15", "pre15"):
            x = s[c].to_numpy()
            lo, hi = dboot(s["day"].to_numpy(), x)
            cells.append(f"{x.mean():.2f} [{lo:.2f},{hi:.2f}]")
        print(f"     {g:20} {len(s):5d}   " + "  ".join(f"{c:>18}" for c in cells))
    return R


def main():
    O, Cl = bars("SPY")
    base = baselines(O, Cl)
    uw = pd.DataFrame(json.load(open(f"{C}/uw_headlines.json")))

    def g_uw(r):
        src = r["source"]
        if src == "Tradex":
            return "UW Tradex major" if r["is_major"] else "UW Tradex other"
        if src in WIRES:
            return "UW press-release wires"
        if src in ("NYT", "Investors.com"):
            return "UW NYT / IBD"
        return None
    E = events(uw, "created_at", g_uw)
    print(f"  UW events (RTH, de-burst): {len(E)} over {E['day'].nunique()} sessions; bars {len(O)} sessions")
    ratio_table(E, O, Cl, base, "SPY around UW headlines")
    Oq, Cq = bars("QQQ")
    ratio_table(E, Oq, Cq, baselines(Oq, Cq), "QQQ around UW headlines")

    wb = pd.DataFrame(json.load(open(f"{C}/wb_news_SPY.json")))
    W = events(wb, "publish_time", lambda r: f"WB {r['sentiment'] or 'NONE'}")
    print(f"\n  WB (Benzinga, SPY feed) events: {len(W)} over {W['day'].nunique()} sessions")
    ratio_table(W, O, Cl, base, "SPY around Benzinga headlines")

    print("\n  DIRECTION (Benzinga label x SPY return, bp; + = moved the label's way) [95% CI]")
    rows = []
    for _, e in W.iterrows():
        s = {"WB POSITIVE": 1, "WB NEGATIVE": -1}.get(e["grp"])
        if s is None:
            continue
        mm = measures(O, Cl, e["day"], int(e["mod"]))
        if mm:
            rows.append(dict(day=e["day"], **{k: s * v * 1e4 for k, v in mm.items()}))
    D = pd.DataFrame(rows)
    print(f"     n = {len(D)} labelled headlines (POSITIVE / NEGATIVE)")
    for c in ("pre15", "pre5", "hit1", "late1", "late5", "late15"):
        lo, hi = dboot(D["day"].to_numpy(), D[c].to_numpy())
        print(f"     {c:7} {D[c].mean():+6.2f} bp  [{lo:+.2f}, {hi:+.2f}]   share > 0 {np.mean(D[c] > 0):.2f}")


if __name__ == "__main__":
    main()
