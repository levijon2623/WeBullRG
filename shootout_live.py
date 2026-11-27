"""
shootout_live.py -- the live SHOOTOUT pill (user, 2026-11-27). DISPLAY ONLY; run from strike_cue.py once a minute.

THE SHOOTOUT IS ON while the tape moves faster than the ATM 0DTE options priced in -- explore_shootout_end.py's
definition A, computed the same way:
  VR(m)   mean of the squared overlapping 5-minute log returns over the trailing 15 minutes (from 09:35; one return at
          09:35, three or more from 09:37) / what the ATM 0DTE IV at the window start priced for 5 minutes,
          iv_daily^2 x 5 / 1440. ATM = the strike nearest the minute's close, IV = mean of call and put (a quote up to
          3 minutes old), DAILY units (live UW implied_volatility is annualised: / sqrt(365)).
  OVER    at the first minute m >= 09:35 from which VR stays < 1 for 5 straight minutes -- known 4 minutes later.
The pill shows the live VR, minutes below 1 so far ("cooling 2/5"), and the HISTORY (shootout_typical.json): the
ticker's median end, and what sessions still on at this point usually did. No factor forecast here -- that waits for
check_shootout_countdown.py.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "shootout_typical.json")
M0, M_END, CAP, SHOW_UNTIL = 570, 575, 660, 660


def load_table():
    try:
        return json.load(open(TABLE))
    except (OSError, ValueError):
        return None


def iv_by_minute(bars, quotes):
    """{minute: ATM IV, daily}. bars = (m, open, high, low, close, volume); quotes = {minute: {(K, 'C'|'P'): (mid, iv_ANNUAL)}}."""
    out = {}
    if bars is None:
        return out
    for mm, cl in zip(bars[:, 0].astype(int), bars[:, 4]):
        K = int(round(cl))
        for back in range(0, 4):
            sn = quotes.get(int(mm) - back) or {}
            c, p = sn.get((K, "C")), sn.get((K, "P"))
            if c and p and c[1] and p[1]:
                out[int(mm)] = (c[1] + p[1]) / 2 / math.sqrt(365)
                break
    return out


def vr_series(bars, ivd, upto):
    """explore_shootout_end's VR for minutes 09:35 .. upto."""
    if bars is None:
        return {}
    m, c = bars[:, 0].astype(int), bars[:, 4]
    idx = {mm: i for i, mm in enumerate(m)}
    vr = {}
    for mm in range(M_END, upto + 1):
        k0 = max(M0 + 5, mm - 10)
        r5 = [math.log(c[idx[k]] / c[idx[k - 5]]) for k in range(k0, mm + 1) if k in idx and (k - 5) in idx]
        base = ivd.get(k0 - 5) or ivd.get(k0 - 4) or ivd.get(M0 + 1)
        if len(r5) >= min(3, mm - M_END + 1) and base:
            vr[mm] = float(np.mean(np.square(r5)) / (base ** 2 * 5 / 1440))
    return vr


def end_of(vr, upto, thr=1.0, sustain=5):
    """the end minute if already known by `upto` (needs sustain minutes below thr), else None."""
    for m_ in range(M_END, upto - sustain + 2):
        if all(vr.get(k) is not None and vr[k] < thr for k in range(m_, m_ + sustain)):
            return m_
    return None


def state(tk, bars, quotes, M, table):
    """the pill's object for ticker tk at the last closed minute M, or None outside 09:30 .. 11:00."""
    if M < M0 or M > SHOW_UNTIL:
        return None
    hm = lambda x: f"{x // 60}:{x % 60:02d}"                     # noqa: E731
    T = (table or {}).get("tickers", {}).get(tk) or {}
    typ = dict(median=hm(T["median"]), q25=hm(T["q25"]), q75=hm(T["q75"])) if T else None
    if M < M_END:
        return dict(state="opening", mins=M - M0 + 1, typical=typ)
    vr = vr_series(bars, iv_by_minute(bars, quotes), M)
    if M not in vr:
        return dict(state="no data", typical=typ)
    end = end_of(vr, M)
    if end is not None:
        return dict(state="over", end=hm(end), end_m=end, vr=round(vr[M], 2), typical=typ)
    cool = 0
    for k in range(M, M_END - 1, -1):
        if vr.get(k) is not None and vr[k] < 1:
            cool += 1
        else:
            break
    cp = str(max(M_END, (M // 5) * 5))
    so = T.get("still_on", {}).get(cp) if T else None
    return dict(state="on", vr=round(vr[M], 2), cool=cool, mins=M - M0 + 1, typical=typ,
                still_on=dict(at=hm(int(cp)), median=hm(so["median"]), q25=hm(so["q25"]), q75=hm(so["q75"]),
                              n=so["n"], past_cap=so["past_cap"]) if so else None)
