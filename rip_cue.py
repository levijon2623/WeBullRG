"""
rip_cue.py -- the live RIP pill (user, 2026-11-26: "build the rip-likelihood pill next to THRUSTRIKE"). DISPLAY ONLY.
Run inside strike_cue.py (cleanbot-strikes); writes nothing but the RIP object in live/strikes.json and its log events.

WHAT IT SHOWS, per ticker. A QUIET STRETCH = check_absorption_rip's base (60+ minutes of 1-minute closes inside 2.5
typical 30-minute ranges, from 10:00, start by 14:00, end by 15:00).
  QUIET  from the stretch's 60th minute while it lasts: the frozen "A SIZE at confirmation" score's quintile and its
         holdout rate of ending in a RIP (>= 3 typical ranges past the band within 60 min of the end). Rates by
         quintile 2.3 / 2.3 / 3.6 / 4.4 / 7.7% (all 4.0%), holdout IC +0.102 [+0.030, +0.167].
  BREAK  for 15 minutes after the first close outside the band: the frozen "B SIZE at the break" score's quintile and
         its holdout rate of carrying on >= 3 typical ranges FROM THE BREAK PRICE within 60 min. 0.0 / 4.1 / 2.1 / 8.4 /
         9.3% (all 4.9%), holdout IC +0.169 [+0.105, +0.232].
  SIZE ONLY: both direction models failed the same holdout -- nothing here says which way. The rates are thin (165-280
  stretches a quintile), so the pill words them low / middle / high.

PARITY. The inputs come from rip_features.features (the code that produced the screen's numbers; checked identical to
1e-16) on live data in the same shapes: today's UW 1-minute candles (RTH 'r' rows, closed minutes), the prior sessions'
candles for the typical range / relative volume / prior close (the two rules the screen used, kept apart), dealer gamma
per minute (spot-exposures), the 0DTE ATM straddle mid and IV per minute (NBBO and IV on the 5-second option-contracts
polls instead of the minute bars' closes), the THRU breaches with their live quintile, and the peak-gamma zone built
once at 09:31 the way check_peak_zone.zone_map builds it -- from UW's expiry-strike gamma (the weekly page's source)
instead of the lake (offline the two agreed within $1 on 94% of sessions). Inputs that cannot be built live score as
rank 0.5, as missing inputs did in the screen.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

import rip_features as RF

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "rip_model_frozen.json")
TKS = RF.TKS
BREAK_SHOW_MIN = 15
WORD = {0: "low", 1: "low", 2: "mid", 3: "high", 4: "high"}
NONDIR_CAL = ("DOW", "OPEX", "MONTHEND", "FOMC", "MACRO_AM")


def load_frozen():
    try:
        return json.load(open(FROZEN))
    except (OSError, ValueError):
        return None


def score(frz, f, tk):
    s = 0.0
    for k, c in zip(frz["inputs"], frz["coef"]):
        v, ref = f.get(k), frz["ref"][tk][k]
        r = 0.5 if (v is None or v != v or not ref) else \
            (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2 * len(ref))
        s += c * r
    return float(s)


def r30(bars):
    """check_absorption_rip.r30 on (m, open, high, low, close, volume) bars."""
    out = []
    for m0 in range(600, 930, 30):
        w = bars[(bars[:, 0] >= m0) & (bars[:, 0] < m0 + 30)]
        if len(w) > 20:
            out.append((w[:, 2].max() - w[:, 3].min()) / w[-1, 4])
    return float(np.median(out)) if out else None


def bars_from_uw(rows, upto=None):
    """UW /stock/{t}/ohlc/1m rows -> (m, open, high, low, close, volume), RTH 'r' rows only (= lake ext-1m), minutes
    <= upto."""
    from zoneinfo import ZoneInfo
    ny, out = ZoneInfo("America/New_York"), []
    for r in rows:
        if str(r.get("market_time") or "r") != "r":
            continue
        try:
            t = datetime.fromisoformat(str(r["start_time"]).replace("Z", "+00:00")).astimezone(ny)
            mm = t.hour * 60 + t.minute
            if upto is not None and mm > upto:
                continue
            out.append((mm, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r.get("volume") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort()
    return np.array(out) if out else None


def session_context(uw, tk, d, cache_dir):
    """the prior-session pieces, once a day (cached): event T30 (check_absorption_rip: the last 20 FULL sessions --
    > 300 RTH bars, to 14:00 or later), the screen's T30 / volume curves (the 20 sessions before d, half days out), and
    the previous session's bars (None when it was a half day, as the screen's ext() gave)."""
    import market_calendar as MC
    f = os.path.join(cache_dir, f"ripctx-{d.isoformat()}.json")
    cache = json.load(open(f)) if os.path.exists(f) else {}
    if tk in cache:
        c = cache[tk]
        return dict(t30e=c["t30e"], t30s=c["t30s"], base_vol=np.array(c["bv"], float) if c["bv"] else None,
                    pa=np.array(c["pa"]) if c["pa"] else None)
    days, x = [], d
    while len(days) < 40 and (d - x).days < 75:
        x = MC.previous_trading_day(x - timedelta(days=1))
        days.append(x)
    bars = {}
    for x in days:
        rows = uw.get(f"/stock/{tk}/ohlc/1m", date=x.isoformat(), limit=2500).get("data") or []
        bars[x] = bars_from_uw(rows)
        full = [y for y in days if bars.get(y) is not None and len(bars[y]) > 300 and bars[y][-1, 0] >= 840]
        if len(full) >= 20 and len(bars) >= 20:
            break
    full = [y for y in days if y in bars and bars[y] is not None and len(bars[y]) > 300 and bars[y][-1, 0] >= 840][:20]
    t30e = float(np.median([r30(bars[y]) for y in full])) if full else None
    last20 = [y for y in days[:20] if y in bars and bars[y] is not None and bars[y][-1, 0] >= 840]
    rs = [r30(bars[y]) for y in last20]
    t30s = float(np.median([v for v in rs if v])) if rs else None
    bv = []
    for y in last20:
        cc = np.full(1000, np.nan)
        cc[bars[y][:, 0].astype(int)] = np.cumsum(bars[y][:, 5])
        bv.append(cc)
    prev = days[0]
    pa = bars.get(prev) if bars.get(prev) is not None and len(bars[prev]) > 300 else None
    cache[tk] = dict(t30e=t30e, t30s=t30s, bv=[[None if v != v else float(v) for v in row] for row in bv] if bv else None,
                     pa=pa.tolist() if pa is not None else None)
    os.makedirs(cache_dir, exist_ok=True)
    with open(f + ".tmp", "w") as fh:
        json.dump(cache, fh)
    os.replace(f + ".tmp", f)
    return dict(t30e=t30e, t30s=t30s, base_vol=np.array(bv, float) if bv else None, pa=pa)


def peak_zone_live(uw, tk, d, spot, ratio):
    """check_peak_zone.zone_map's 0DTE + index zone from UW expiry-strike gamma (the weekly page's source)."""
    import index_gex as IX
    BAND, ALT_JOIN = 0.05, 3.0                       # check_peak_zone.BAND / ALT_JOIN (kept here: research module not needed live)
    if not spot or not ratio:
        return None

    def rows_of(sym):
        # expirations[] filters server-side (checked 2026-11-26 by the returned expiries: 219 rows, all the asked
        # expiry); unfiltered pages come far-dated first. Rows of any other expiry are still dropped here.
        out = []
        for page in range(4):
            rows = (uw.get(f"/stock/{sym}/spot-exposures/expiry-strike", limit=500, page=page,
                           **{"expirations[]": d.isoformat()}).get("data") or [])
            for x in rows:
                if (x.get("expiry") or "") == d.isoformat():
                    try:
                        out.append(dict(e=x["expiry"], k=float(x["strike"]), cg=float(x.get("call_gamma_oi") or 0),
                                        pg=float(x.get("put_gamma_oi") or 0)))
                    except (TypeError, ValueError, KeyError):
                        pass
            if len(rows) < 500:
                break
        return out
    etf, idx = rows_of(tk), rows_of(IX.INDEX_OF[tk])
    if not etf:
        return None
    lo_b, hi_b = spot * (1 - BAND), spot * (1 + BAND)
    rows = IX.blend_rows(etf, idx, ratio)[0] if idx else etf
    heat = IX.heat_of(rows, {d.isoformat()}, lo_b, hi_b)
    if not heat:
        return None
    w = IX.walls_with_ties({k: (v[0], v[1]) for k, v in heat.items()}) or {}
    P, A = w.get("peak"), w.get("peak_alt")
    if P is None:
        return None
    lo, hi = (min(P, A), max(P, A)) if A is not None and abs(A - P) <= ALT_JOIN else (P - 1.0, P + 1.0)
    net = sum(sum(v) for k, v in heat.items() if lo - 1e-9 <= k <= hi + 1e-9)
    return ((lo + hi) / 2, 1 if net >= 0 else -1)


class RipTracker:
    """all three tickers; step() once per closed minute with the latest live pieces."""

    def __init__(self, frozen, day, cal):
        self.F = frozen
        self.day, self.cal = day, cal
        self.conf, self.brk, self.res_done = {}, {}, set()     # (tk, s) -> result dicts
        self.state = {tk: None for tk in TKS}

    def step(self, M, ctx, t30e, logf=None):
        """M = the last CLOSED minute. ctx = rip_features context with today's bars up to M."""
        bn = defaultdict(list)
        scans = {}
        for tk in TKS:
            x = ctx["a"].get(tk)
            if x is None or not t30e.get(tk):
                continue
            scans[tk] = RF.scan(x[:, 0], x[:, 4], t30e[tk], M)
            for n_, b in enumerate(scans[tk]):
                # a stretch that ran into the 15:00 cap counts as a BREAK when the 15:01 close is outside its band --
                # the screen's features() did exactly that (4 of 103 break stamps in the replay check)
                if b["state"] == "capped":
                    m_, c_ = x[:, 0], x[:, 4]
                    k = np.flatnonzero(m_ == b["e"] + 1)
                    i0, ie = int(np.searchsorted(m_, b["s"])), int(np.searchsorted(m_, b["e"]))
                    hi, lo = c_[i0:ie + 1].max(), c_[i0:ie + 1].min()
                    if len(k) and (c_[k[0]] > hi or c_[k[0]] < lo):
                        scans[tk][n_] = b = dict(b, state="broken", side=1 if c_[k[0]] > hi else -1, brk=b["e"] + 1)
            for b in scans[tk]:
                bn[tk].append((b["s"], None if b["state"] == "ongoing" else b["e"]))
        ctx = dict(ctx, base_now=bn)
        for tk, bases in scans.items():
            cur = None
            for b in bases:
                key = (tk, b["s"])
                r = dict(tk=tk, s=b["s"], e=b["e"] if b["state"] != "ongoing" else M, t30=t30e[tk])
                if b["s"] + 59 <= M and key not in self.conf:
                    self.conf[key] = self._score("conf", r, ctx, M, logf)
                if b["state"] == "broken" and b["brk"] <= M and key not in self.brk:
                    self.brk[key] = self._score("brk", r, ctx, M, logf, side=b["side"])
                if b["state"] == "ongoing" and b["s"] + 59 <= M:
                    cur = dict(kind="quiet", s=b["s"], since=f"{b['s'] // 60}:{b['s'] % 60:02d}", mins=M - b["s"] + 1,
                               **(self.conf.get(key) or {}))
                elif b["state"] == "broken" and M - b["brk"] < BREAK_SHOW_MIN:
                    cur = dict(kind="break", s=b["s"], e=b["e"], side=b["side"], brk=b["brk"],
                               at=f"{b['brk'] // 60}:{b['brk'] % 60:02d}", age_min=M - b["brk"],
                               conf=self.conf.get(key), **(self.brk.get(key) or {}))
                # the forward record: outcomes once 60 minutes have passed (the screen's own targets)
                if b["state"] == "broken" and M >= b["brk"] + 60 and key not in self.res_done and logf:
                    self.res_done.add(key)
                    fb = RF.features(r, "brk", ctx)
                    xb = ctx["a"][tk]
                    m, c = xb[:, 0], xb[:, 4]
                    i0, ie = int(np.searchsorted(m, b["s"])), int(np.searchsorted(m, b["e"]))
                    hi, lo = c[i0:ie + 1].max(), c[i0:ie + 1].min()
                    nxt = c[(m > b["e"]) & (m <= b["e"] + 60)]
                    X = max(nxt.max() - hi, lo - nxt.min()) / (t30e[tk] * c[ie]) if len(nxt) else None
                    logf({"ev": "rip_res", "tk": tk, "s": b["s"], "e": b["e"], "side": b["side"],
                          "rip_A": None if X is None else bool(X >= 3), "X": None if X is None else round(float(X), 3),
                          "rip_B": None if not fb or fb.get("RIP_B") != fb.get("RIP_B") else bool(fb["RIP_B"]),
                          "cont_B": None if not fb else fb.get("CONT_B")})
            self.state[tk] = cur

    def _score(self, mode, r, ctx, M, logf, side=0):
        f = RF.features(r, mode, ctx)
        if not f:
            return {}
        for k in RF.DIRECTIONAL:
            v = f.get(k)
            f[f"ABS_{k}"] = abs(float(v)) if v is not None and v == v else np.nan
        f.update(self.cal)
        frz = self.F["models"][mode]
        z = score(frz, f, r["tk"])
        q = int(np.searchsorted(frz["cuts"], z, side="right"))
        out = dict(z=round(z, 4), q=q + 1, word=WORD[q], rate=frz["quintiles"][q]["rate"], n=frz["quintiles"][q]["n"],
                   base=frz["base"])
        if logf:
            logf({"ev": f"rip_{mode}", "tk": r["tk"], "s": r["s"], "M": M, "side": side, **out,
                  "inputs": {k: (None if f.get(k) is None or f[k] != f[k] else round(float(f[k]), 5)) for k in frz["inputs"]}})
        return out

    def cue(self, tk):
        return self.state.get(tk)
