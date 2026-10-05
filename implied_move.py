"""
implied_move.py
===============
The viewer's IMPLIED MOVE band: anchor +/- q x the at-the-money straddle, with
q the historical quantile of (realized excursion / straddle). Pure functions
and constants -- live_state fetches the quotes and the anchor price.

Validated by check_implied_move.py (pre-registered, PASSED 2026-10-03 for both
families on SPY / QQQ / IWM): walk-forward OOS coverage within 2-4pp of the
target at tau .65 and .90 in every slice, and 4-7.5% sharper than the same band
built from recent ranges without options. VIX, event-day and GEX-sign
conditioning did NOT improve it, so there is none here.

  am   09:35 same-day straddle; anchor = 09:35 bar close; covers the rest of
       the session.
  pm   15:55 next-expiry straddle the session before; anchor = that 15:55
       close; covers the overnight gap plus the session -- the pre-open band.

The straddle is NOT shrunk the way the variance risk premium suggests: it is
priced to the CLOSE (|close - A| / S averaged 0.92), but highs and lows reach
further than the close, so the 65% multiple lands near 1.0. Only these three
tickers were tested; nothing is drawn for the others.
"""
from __future__ import annotations

# {ticker: {family: {tau: (up, dn)}}} -- quantiles of excursion / straddle over
# every lake session (check_implied_move, 2023-10-12 .. 2026-09-18/21;
# n = 725 / 718 SPY, 725 / 717 QQQ, 677 / 669 IWM). Refresh by re-running the
# check after the lake grows.
RATIOS = {
    "SPY": {"am": {0.65: (1.060, 1.093), 0.90: (1.708, 2.172)},
            "pm": {0.65: (1.087, 0.903), 0.90: (1.794, 2.025)}},
    "QQQ": {"am": {0.65: (1.016, 1.098), 0.90: (1.691, 2.128)},
            "pm": {0.65: (1.063, 0.895), 0.90: (1.805, 2.156)}},
    "IWM": {"am": {0.65: (1.040, 1.082), 0.90: (1.804, 1.981)},
            "pm": {0.65: (1.054, 1.016), 0.90: (1.825, 1.971)}},
}
TICKERS = tuple(RATIOS)
MAX_SPREAD = 0.25          # per leg, (ask - bid) / mid -- build_implied_move's filter


def leg_mid(bid, ask):
    """Quote mid, or None if the quote fails the study's filter."""
    if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    if mid <= 0 or (ask - bid) / mid > MAX_SPREAD:
        return None
    return mid


def interp_straddle(strikes, straddles, spot):
    """Straddle at `spot`: linear between the two strikes that bracket it, the
    nearest strike otherwise. strikes ascending, one straddle per strike.
    -> (straddle, nearest_index) or (None, None)."""
    if not strikes or spot is None:
        return None, None
    i_near = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    lo = max((i for i in range(len(strikes)) if strikes[i] <= spot), default=None)
    hi = min((i for i in range(len(strikes)) if strikes[i] >= spot), default=None)
    if lo is not None and hi is not None and strikes[hi] > strikes[lo]:
        w = (spot - strikes[lo]) / (strikes[hi] - strikes[lo])
        return straddles[lo] * (1 - w) + straddles[hi] * w, i_near
    return straddles[i_near], i_near


def straddle_from_quotes(quotes, spot):
    """{OCC symbol: (bid, ask)} of ONE expiry -> (straddle, detail) or (None, why)."""
    legs = {}
    for sym, (b, a) in quotes.items():
        m = leg_mid(b, a)
        if m is None:
            continue
        legs.setdefault(int(sym[-8:]) / 1000, {})[sym[-9]] = m
    ks = sorted(k for k, v in legs.items() if "C" in v and "P" in v)
    if not ks:
        return None, "no strike with both legs quoted inside the spread filter"
    st = [legs[k]["C"] + legs[k]["P"] for k in ks]
    s, i = interp_straddle(ks, st, spot)
    if s is None:
        return None, "no spot"
    bracketed = ks[0] <= spot <= ks[-1]
    return s, dict(k_near=ks[i], strikes=len(ks), bracketed=bracketed)


def band(tk, fam, anchor, straddle):
    """-> {hi65, lo65, hi90, lo90} or None for an untested ticker."""
    r = (RATIOS.get(tk) or {}).get(fam)
    if not r or not anchor or not straddle:
        return None
    (u65, d65), (u90, d90) = r[0.65], r[0.90]
    return dict(hi65=round(anchor + u65 * straddle, 2), lo65=round(anchor - d65 * straddle, 2),
                hi90=round(anchor + u90 * straddle, 2), lo90=round(anchor - d90 * straddle, 2))
