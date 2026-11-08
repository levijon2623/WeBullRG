"""
check_orderflow_forward.py
==========================
PRE-REGISTERED forward test of the stock order flow (Webull footprint) and 0DTE
strike-flow leads, written 2026-11-06 BEFORE any session it scores (user + Claude).

    python check_orderflow_forward.py                        # score sessions from 2026-11-06
    python check_orderflow_forward.py --replay-discovery     # code check only, NOT A RESULT

Run after the close, from the bot's folder on the box (the footprint cache and the
gauge log live there). Read-only apart from its own cache; ~80 UW calls per new
session at 40/min, because the live bot shares the key.

WHY FORWARD ONLY
    Every lead below was found by looking through 2026-10-07..11-05, several hundred
    comparisons in one selloff. Those sessions cannot confirm anything, so this script
    refuses them unless asked to replay. Nothing here may be retuned after a look:
    a changed parameter is a new test with a new start date.

THE TOUCH
    A level (or a zone of +-b around it) is armed once price is R away, touched when
    price reaches it, and then either REJECTS (back R from the near edge) or BREAKS
    (R past the far edge). R is in bp of the prior close. 5-second footprint bars.

HYPOTHESES (sign fixed in advance; discovery value in brackets)
    H1  Push-side 0DTE aggression. Whole dollar +-$0.10, R=5. Over the 5 minutes before
        the touch, push = (ask-bid)/(ask+bid) on that strike's same-day contract on the
        side driving in (call from below, put from above); at least 50 sided contracts.
        Reject rate, high third minus low third, using the FROZEN cuts below, is < 0.
        [63.2 -> 57.9, -5.3, CI -8.0,-2.9; holds in each approach-speed third, but
        -1.9 in the first half of the sample and -6.6 in the second]
    H2  Top-3 blended same-day gamma zones. ETF + index gamma from the PRIOR session,
        index x ratio^2, band +-3% of the prior close, zone +-$0.05, R=5. Reject rate
        minus the same strikes shifted .30-.70 is > 0, and mean EV (reject +R, break
        -(R + zone width)) is >= +1 bp, about what an ATM option needs to clear costs.
        [62.2 vs 57.3; EV +0.42 bp, so its EV gate was NOT met in discovery]
    H3  First touch of a whole dollar that session, no buffer, R=3. Reject rate minus
        zones at .30-.70 is < 0. [49.2 vs 53.9]
    H4  First touch of an intraday volume peak (refreshed every 15 min from 10:00, live
        only once it exists), no buffer, R=5. Reject rate minus copies shifted
        +-8/12/20/30 bp is > 0. [51.3 vs 46.2]
    H5a Stock delta share over 5 minutes -> next 5-minute move, Spearman > 0. [+0.027]
    H5b Stock delta share over 15 minutes -> next 15-minute move, Spearman > 0. [+0.020]
        Both on the live gauge's 10-minute grid, in straddle units.
    H6  New session high after 09:45 with CVD below its own high: next 30 minutes,
        non-confirming minus confirmed, paired by ticker-day, is < 0. [+2.9 bp, wrong sign]
    H7  |Stock delta share over 15 minutes| -> largest 60-minute move, given the gauge:
        partial Spearman < 0. [-0.075; may be a thin-tape artifact]
    H8  The bot's paper entries on SPY/QQQ/IWM: 30-minute underlying move in the trade's
        direction, stock delta agreeing minus disagreeing, is > 0. [never measured]

DROPPED, not tested forward: the five-signal footprint checklist (no consistent lift at
any bar size or level), prior-day high/low/POC/nodes, VWAP, developing POC, plain whole
dollars, the same-day gamma fade after the next bar (gone with a buffer).

DECISION RULES
    Early read  2026-12-02 (18 sessions): reported, nothing can pass. Only H1 has the
                power to say much (+-3pp or so). It is ON TRACK if its CI excludes 0 in the
                predicted direction, and DEAD if the point estimate has the wrong sign.
    Final       at least 100 sessions, about 2027-04-01. PASS needs all of: CI excludes 0
                in the predicted direction; Benjamini-Hochberg q < 0.10 across the nine
                tests; same sign in both halves; H2's EV gate. A PASS earns a real test
                with option fills. It is not a trade signal.
    After that  footprint inputs go into check_ic_screen.py / check_ic_tail.py (Lasso);
                too little history before then.

KNOWN LIMITS
    Webull classifies about a fifth to a quarter of each bar's volume. No futures data.
    Multi-leg volume is left in the option sides.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NY = ZoneInfo("America/New_York")

START = "2026-11-06"            # first scored session
DISCOVERY = ("2026-10-07", "2026-11-05")
EARLY_READ = "2026-12-02"
FINAL_MIN_SESSIONS = 100
TK = ("SPY", "QQQ", "IWM")
INDEX_OF = {"SPY": "SPX", "QQQ": "NDX", "IWM": "RUT"}
SEED_RATIO = {"SPY": 10.04, "QQQ": 41.1, "IWM": 10.07}
UW_GAP_S = 1.5                  # 40 calls/min: the live bot shares this key

# ---- frozen 2026-11-06 from the discovery sample; do not recompute
H1_R, H1_ZONE = 5, 0.10
H1_CUTS = {"SPY": (-0.0325, 0.0759), "QQQ": (-0.0297, 0.0817), "IWM": (-0.0696, 0.1284)}
H1_MIN_SIDED = 50
H2_R, H2_ZONE, H2_BAND, H2_EV_GATE = 5, 0.05, 0.03, 1.0
H3_R, H4_R = 3, 5
OFFSETS = (30, 40, 50, 60, 70)
SHIFTS_BP = (-30, -20, -12, -8, 8, 12, 20, 30)
PEAK_SEP_BP, PEAK_TOP, PEAK_START, PEAK_STEP = 15, 3, 600, 15
PROFILE_BIN = {"SPY": 0.10, "QQQ": 0.10, "IWM": 0.05}
RATIO_GUARD = 0.003
BOOT = 2000


def _mod(ts):
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+0000", "+00:00")).astimezone(NY)
    return t.hour * 60 + t.minute


def _sec(ts):
    t = dt.datetime.fromisoformat(ts.replace("+0000", "+00:00")).astimezone(NY)
    return t.hour * 3600 + t.minute * 60 + t.second


def _first(mask):
    if not mask.size:
        return None
    i = int(np.argmax(mask))
    return i if mask[i] else None


def zone_events(hi, lo, zl, zh, R):
    """(side, touch index, resolve index, rejected). side +1 = came up into the zone."""
    out, i, n = [], 0, len(hi)
    while i < n:
        a, b = _first(lo[i:] <= zl - R), _first(hi[i:] >= zh + R)
        if a is None and b is None:
            break
        a = n if a is None else a + i
        b = n if b is None else b + i
        if a == b:
            i = a + 1
            continue
        side = 1 if a < b else -1
        s0 = min(a, b)
        t = _first(hi[s0 + 1:] >= zl) if side == 1 else _first(lo[s0 + 1:] <= zh)
        if t is None:
            break
        t += s0 + 1
        k, m = _first(hi[t:] >= zh + R), _first(lo[t:] <= zl - R)
        if k is None and m is None:
            break
        k = n if k is None else k + t
        m = n if m is None else m + t
        if k == m:
            i = k + 1
            continue
        out.append((side, t, min(k, m), (m < k) if side == 1 else (k < m)))
        i = min(k, m)
    return out


class Store:
    """Footprint files plus UW data, cached under the file names the discovery run used."""

    def __init__(self, footprint_dir, cache_dir):
        self.fp, self.cache = footprint_dir, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._uw = None
        self.files = {}

    def _load(self, name):
        if name not in self.files:
            p = os.path.join(self.cache, name)
            self.files[name] = json.load(open(p)) if os.path.exists(p) else {}
        return self.files[name]

    def _save(self, name):
        json.dump(self.files[name], open(os.path.join(self.cache, name), "w"))

    def uw(self, path, **kw):
        if self._uw is None:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(ROOT, ".env"))
            import positioning_map as P
            self._uw, self._num = P.UW(), P.num
        for attempt in range(4):
            t0 = time.time()
            try:
                rows = self._uw.get(path, **kw).get("data") or []
                time.sleep(max(0, UW_GAP_S - (time.time() - t0)))
                return rows
            except Exception as e:                  # noqa: BLE001 -- retried, then empty
                print(f"  retry {path[:40]} {type(e).__name__}", flush=True)
                time.sleep(20 * (attempt + 1))
        return []

    def n(self, x):
        return (self._num(x) or 0) if self._uw else float(x or 0)

    def m1(self, day):
        d = json.load(open(os.path.join(self.fp, f"M1-{day}.json")))
        return {t: {_mod(r["time"]): r for r in (b if isinstance(b, list) else b.values())
                    if r.get("trading_session", "RTH") == "RTH"} for t, b in d.items()}

    def s5(self, day, t):
        d = json.load(open(os.path.join(self.fp, f"S5-{day}.json")))
        rows = []
        bars = d.get(t) or []
        for x in (bars if isinstance(bars, list) else bars.values()):
            if x.get("trading_session", "RTH") != "RTH":
                continue
            b, s = x.get("buy_detail") or {}, x.get("sell_detail") or {}
            ps = [float(p) for p in list(b) + list(s)]
            if not ps:
                continue
            tot = sum(float(v) for v in b.values()) + sum(float(v) for v in s.values())
            vw = (sum(float(p) * float(v) for p, v in list(b.items()) + list(s.items())) / tot) if tot else ps[0]
            rows.append((_sec(x["time"]), max(ps), min(ps), vw))
        rows.sort()
        return tuple(np.array([r[i] for r in rows]) for i in range(4))

    def flow_px(self, day, t):
        c = self._load("uw_cache.json")
        k = f"{day}|{t}"
        if k not in c:
            ticks, px = {}, {}
            for r in self.uw(f"/stock/{t}/net-prem-ticks", date=day):
                ticks[str(_mod(r["tape_time"]))] = {a: self.n(r.get(a)) for a in ("call_volume", "put_volume", "net_call_premium", "net_put_premium", "net_delta")}
            for r in self.uw(f"/stock/{t}/ohlc/1m", date=day, limit=2500):
                px[str(_mod(r["start_time"]))] = {a: self.n(r[a]) for a in ("open", "high", "low", "close", "volume")}
            c[k] = dict(ticks=ticks, px=px)
            self._save("uw_cache.json")
        return {int(m): v for m, v in c[k]["px"].items()}

    def gamma(self, day, prev, t):
        """(ETF same-day {strike: raw}, index same-day {strike: raw}, ratio or None) from the PRIOR session."""
        g0, ig = self._load("gex0_cache.json"), self._load("index_gamma.json")
        k = f"{day}|{t}"
        mag = lambda r: abs(self.n(r.get("call_gex"))) + abs(self.n(r.get("put_gex")))
        if k not in g0:
            g0[k] = {r["strike"]: mag(r) for r in self.uw(f"/stock/{t}/greek-exposure/strike-expiry", date=prev, expiry=day)
                     if r.get("strike") is not None and r.get("expiry") == day}
            self._save("gex0_cache.json")
        if k not in ig:
            ix = INDEX_OF[t]
            same = {r["strike"]: mag(r) for r in self.uw(f"/stock/{ix}/greek-exposure/strike-expiry", date=prev, expiry=day)
                    if r.get("strike") is not None and r.get("expiry") == day}
            px = self.flow_px(prev, t); rs = []
            for r in self.uw(f"/stock/{ix}/spot-exposures", date=prev):
                ts, p = r.get("time") or r.get("start_time") or "", self.n(r.get("price"))
                if ts and p:
                    m = _mod(ts)
                    if m in px and 570 <= m < 960 and px[m]["close"]:
                        rs.append(p / px[m]["close"])
            rs.sort()
            ig[k] = dict(pv=prev, ratio=rs[len(rs) // 2] if len(rs) >= 10 else None, n=len(rs), same=same)
            self._save("index_gamma.json")
        return g0[k], ig[k]["same"], ig[k]["ratio"]

    def zero_dte(self, day, t, lo, hi):
        """{(strike, 'C'|'P'): {minute: (ask, bid)}} for whole-dollar strikes in the day's range +-$2."""
        z = self._load("zero_dte.json")
        out, new = {}, 0
        for k in range(int(lo) - 2, int(hi) + 3):
            for cp in "CP":
                sym = f"{t}{day[2:4]}{day[5:7]}{day[8:10]}{cp}{k * 1000:08d}"
                if sym not in z:
                    rows = self.uw(f"/option-contract/{sym}/intraday", date=day)
                    z[sym] = dict(d=day, t=t, k=k, cp=cp, m={str(_mod(r["start_time"])): [self.n(r.get(f)) for f in ("volume_ask_side", "volume_bid_side", "volume_mid_side", "volume_no_side", "volume_multi", "premium_ask_side", "premium_bid_side")] for r in rows})
                    new += 1
                out[(k, cp)] = {int(m): (x[0], x[1]) for m, x in z[sym]["m"].items()}
        if new:
            self._save("zero_dte.json")
        return out


def blend_top3(etf, idx, ratio, pc):
    g = defaultdict(float)
    for k, v in etf.items():
        g[float(k)] += v
    for k, v in idx.items():
        m = float(k) / ratio
        low = int(m // 1)
        f = m - low
        g[float(low)] += v * ratio * ratio * (1 - f)
        g[float(low + 1)] += v * ratio * ratio * f
    c = sorted(((k, v) for k, v in g.items() if abs(k / pc - 1) <= H2_BAND and k == int(k) and v > 0), key=lambda x: -x[1])
    return {k for k, _ in c[:3]}


def peaks(fp, w, pc, bd):
    v = defaultdict(float)
    for m, r in fp.items():
        if 570 <= m < bd:
            for key in ("buy_detail", "sell_detail"):
                for p, x in (r.get(key) or {}).items():
                    v[int(float(p) // w)] += float(x)
    if not v:
        return []
    ks = sorted(v)
    sm = {k: v.get(k - 1, 0) + v.get(k, 0) + v.get(k + 1, 0) for k in range(ks[0], ks[-1] + 1)}
    chosen, gap = [], pc * PEAK_SEP_BP / 1e4
    for k in sorted(sm, key=lambda x: -sm[x]):
        p = (k + .5) * w
        if all(abs(p - q) >= gap for q in chosen):
            chosen.append(p)
        if len(chosen) >= PEAK_TOP + 1:
            break
    return chosen[1:]                   # the largest is the developing POC, not a peak


def score_day(st_, day, prev, ratio_med, recs):
    m1, m1p = st_.m1(day), None
    for t in TK:
        sec, hi, lo, vw = st_.s5(day, t)
        if not len(hi) or t not in m1:
            continue
        pxp = st_.flow_px(prev, t)
        rth = [m for m in sorted(pxp) if 570 <= m < 960]
        if not rth:
            continue
        pc = pxp[rth[-1]]["close"]
        n0, n1 = int(lo.min()) - 1, int(hi.max()) + 2
        minutes = sec // 60
        # ---- H1
        tape = st_.zero_dte(day, t, lo.min(), hi.max())
        R = pc * H1_R / 1e4
        for k in range(n0, n1):
            C, P = tape.get((k, "C"), {}), tape.get((k, "P"), {})
            for side, a, b, rej in zone_events(hi, lo, k - H1_ZONE, k + H1_ZONE, R):
                m0 = int(sec[a]) // 60
                if m0 < 590:
                    continue
                sum_ = lambda T: (sum(T[m][0] for m in range(m0 - 5, m0) if m in T), sum(T[m][1] for m in range(m0 - 5, m0) if m in T))
                ca, cb, pa, pb = *sum_(C), *sum_(P)
                if ca + cb + pa + pb < H1_MIN_SIDED:
                    continue
                wa, wb = (ca, cb) if side == 1 else (pa, pb)
                push = (wa - wb) / max(wa + wb, 1)
                lo_c, hi_c = H1_CUTS[t]
                third = "low" if push <= lo_c else "high" if push > hi_c else "mid"
                p_at = lambda mm: float(vw[max(int(np.searchsorted(sec, mm * 60)) - 1, 0)])
                recs["H1"].append(dict(d=day, t=t, third=third, rej=bool(rej), speed=side * (p_at(m0) - p_at(m0 - 5)) / pc * 1e4))
        # ---- H2
        etf, idx, ratio = st_.gamma(day, prev, t)
        ratio = ratio or SEED_RATIO[t]
        if ratio_med.get(t) and abs(ratio / ratio_med[t] - 1) > RATIO_GUARD:
            ratio = ratio_med[t]
        top = blend_top3(etf, idx, ratio, pc)
        R = pc * H2_R / 1e4
        wb_bp = 2 * H2_ZONE / pc * 1e4
        for n in range(n0, n1):
            near = n in top or n + 1 in top
            for off in (0,) + OFFSETS:
                if (off == 0 and n not in top) or (off != 0 and not near):
                    continue
                L = n + off / 100
                for side, a, b, rej in zone_events(hi, lo, L - H2_ZONE, L + H2_ZONE, R):
                    recs["H2"].append(dict(d=day, t=t, real=off == 0, rej=bool(rej), ev=H2_R if rej else -(H2_R + wb_bp)))
        # ---- H3
        R = pc * H3_R / 1e4
        for n in range(n0, n1):
            for off in (0,) + OFFSETS:
                L = n + off / 100
                ev = zone_events(hi, lo, L, L, R)
                if ev:
                    recs["H3"].append(dict(d=day, t=t, real=off == 0, rej=bool(ev[0][3])))
        # ---- H4
        R = pc * H4_R / 1e4
        segs = defaultdict(list)
        w = PROFILE_BIN[t]
        for bd in range(PEAK_START, 960, PEAK_STEP):
            i0, i1 = int(np.searchsorted(sec, bd * 60)), int(np.searchsorted(sec, (bd + PEAK_STEP) * 60))
            if i0 >= i1:
                continue
            for L in peaks(m1[t], w, pc, bd):
                s = segs[round(L, 3)]
                if s and s[-1][1] == i0:
                    s[-1][1] = i1
                else:
                    s.append([i0, i1])
        for L0, spans in segs.items():
            for i0, i1 in spans:
                for sh in (0,) + SHIFTS_BP:
                    L = L0 * (1 + sh / 1e4)
                    ev = zone_events(hi[i0:i1], lo[i0:i1], L, L, R)
                    if ev:
                        recs["H4"].append(dict(d=day, t=t, real=sh == 0, rej=bool(ev[0][3])))
        # ---- H6
        px = st_.flow_px(day, t)
        fp = m1[t]
        top_px = cum = 0.0
        cvd_hi = -1e18
        byd = {"div": [], "conf": []}
        for m in sorted(x for x in fp if 570 <= x < 960 and x in px):
            cum += float(fp[m]["delta"])
            new_high, conf = px[m]["high"] > top_px, cum >= cvd_hi
            cvd_hi, top_px = max(cvd_hi, cum), max(top_px, px[m]["high"])
            if new_high and 585 <= m <= 900 and m + 30 in px:
                byd["conf" if conf else "div"].append((px[m + 30]["close"] / px[m]["close"] - 1) * 1e4)
        if byd["div"] and byd["conf"]:
            recs["H6"].append(dict(d=day, t=t, diff=st.mean(byd["div"]) - st.mean(byd["conf"])))


def score_grid(st_, day_list, gauge_path, ledger_path, recs):
    """H5a/b and H7 on the gauge's own grid; H8 from the paper ledger."""
    days_ok = set(day_list)
    fpc, pxc = {}, {}
    def share(fp, m, k):
        tot = sum(float(fp[x]["total"]) for x in range(m - k, m) if x in fp)
        return sum(float(fp[x]["delta"]) for x in range(m - k, m) if x in fp) / tot if tot else 0.0
    if os.path.exists(gauge_path):
        for line in open(gauge_path):
            g = json.loads(line)
            d, t, m, S = g["day"], g["tk"], g["mod"], g.get("S")
            if d not in days_ok or not S or t not in TK:
                continue
            fp = fpc.setdefault(d, st_.m1(d)).get(t) or {}
            px = pxc.setdefault((d, t), st_.flow_px(d, t))
            if m not in px:
                continue
            c = px[m]["close"]
            r = dict(d=d, t=t, fp5=share(fp, m, 5), fp15=share(fp, m, 15), gauge=g.get("p"))
            r["abs15"] = abs(r["fp15"])
            for k in (5, 15):
                r[f"r{k}"] = (px[m + k]["close"] - c) / S if m + k in px else None
            w = [x for x in range(m, m + 60) if x in px]
            r["move60"] = max(max(abs(px[x]["high"] - c), abs(px[x]["low"] - c)) for x in w) / S if w else None
            recs["grid"].append(r)
    if os.path.exists(ledger_path):
        for line in open(ledger_path):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("action") != "ENTRY_LONG" or x.get("ticker") not in TK:
                continue
            d = str(x.get("timestamp"))[:10]
            if d not in days_ok:
                continue
            t, m = x["ticker"], _mod(x["timestamp"])
            sgn = 1 if str(x.get("contract"))[-9] == "C" else -1
            fp = fpc.setdefault(d, st_.m1(d)).get(t) or {}
            px = pxc.setdefault((d, t), st_.flow_px(d, t))
            f = share(fp, m, 15)
            if f == 0 or m not in px or m + 30 not in px:
                continue
            recs["H8"].append(dict(d=d, t=t, agree=f * sgn > 0, move=(px[m + 30]["close"] / px[m]["close"] - 1) * 1e4 * sgn))


def boot_stat(days, fn, n=BOOT):
    random.seed(11)
    out = []
    for _ in range(n):
        v = fn(random.choices(days, k=len(days)))
        if v is not None and v == v:
            out.append(v)
    if len(out) < n / 2:
        return None, None, None
    out.sort()
    p = 2 * min(sum(x <= 0 for x in out), sum(x >= 0 for x in out)) / len(out)
    return out[int(.025 * len(out))], out[int(.975 * len(out))], p


def rate_diff(a, b, key="rej", scale=100):
    A, B = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for r in a:
        A[r["d"]][0] += r[key]; A[r["d"]][1] += 1
    for r in b:
        B[r["d"]][0] += r[key]; B[r["d"]][1] += 1
    def fn(ds):
        x, y = sum(A[d][0] for d in ds), sum(A[d][1] for d in ds)
        u, v = sum(B[d][0] for d in ds), sum(B[d][1] for d in ds)
        return scale * (x / y - u / v) if y and v else None
    return fn


def rank(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v)); r[order] = np.arange(len(v), dtype=float)
    return r


def spear(rows, a, b, ctrl=None):
    """Within-ticker rank correlation, averaged; optionally partial on ctrl."""
    out = []
    for t in TK:
        s = [r for r in rows if r["t"] == t and r.get(a) is not None and r.get(b) is not None and (ctrl is None or r.get(ctrl) is not None)]
        if len(s) < 10:
            continue
        x, y = rank([r[a] for r in s]), rank([r[b] for r in s])
        if ctrl:
            X = np.column_stack([np.ones(len(s)), rank([r[ctrl] for r in s])])
            x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        if x.std() and y.std():
            out.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(out)) if out else None


def report(recs, days_scored):
    D = sorted(days_scored)
    H = D[len(D) // 2] if D else ""
    res = []
    def add(name, est, fn, want, halves, note=""):
        lo, hi, p = boot_stat(D, fn)
        res.append(dict(name=name, est=est, lo=lo, hi=hi, p=p, want=want, halves=halves, note=note))
    wr = lambda a: 100 * sum(r["rej"] for r in a) / len(a) if a else float("nan")
    r1 = recs["H1"]; hi1, lo1 = [r for r in r1 if r["third"] == "high"], [r for r in r1 if r["third"] == "low"]
    half = lambda a, b, f: (f([r for r in a if r["d"] < H]) - f([r for r in b if r["d"] < H]), f([r for r in a if r["d"] >= H]) - f([r for r in b if r["d"] >= H]))
    add("H1 push-side aggression", wr(hi1) - wr(lo1), rate_diff(hi1, lo1), -1, half(hi1, lo1, wr), f"low {wr(lo1):.1f} (n={len(lo1)}) high {wr(hi1):.1f} (n={len(hi1)})")
    for key, name in (("H2", "H2 blended gamma zones"), ("H3", "H3 first whole-dollar touch"), ("H4", "H4 first intraday-peak touch")):
        a, b = [r for r in recs[key] if r["real"]], [r for r in recs[key] if not r["real"]]
        note = f"real {wr(a):.1f} (n={len(a)}) vs placebo {wr(b):.1f} (n={len(b)})"
        if key == "H2":
            ev = sum(r["ev"] for r in a) / len(a) if a else float("nan")
            note += f" | EV {ev:+.2f} bp, gate >= {H2_EV_GATE:+.1f}: {'met' if ev == ev and ev >= H2_EV_GATE else 'NOT met'}"
        add(name, wr(a) - wr(b), rate_diff(a, b), -1 if key == "H3" else 1, half(a, b, wr), note)
    g = recs["grid"]
    by = defaultdict(list)
    for r in g:
        by[r["d"]].append(r)
    def grid_fn(a, b, c=None):
        def fn(ds):
            s = []
            for d in ds:
                s += by[d]
            return spear(s, a, b, c)
        return fn
    gh = lambda a, b, c=None: (spear([r for r in g if r["d"] < H], a, b, c) or 0, spear([r for r in g if r["d"] >= H], a, b, c) or 0)
    add("H5a delta 5m -> next 5m", spear(g, "fp5", "r5"), grid_fn("fp5", "r5"), 1, gh("fp5", "r5"), f"n={sum(r['r5'] is not None for r in g)} (Spearman)")
    add("H5b delta 15m -> next 15m", spear(g, "fp15", "r15"), grid_fn("fp15", "r15"), 1, gh("fp15", "r15"), "(Spearman)")
    r6 = defaultdict(list)
    for r in recs["H6"]:
        r6[r["d"]].append(r["diff"])
    def fn6(ds):
        v = [x for d in ds for x in r6[d]]
        return sum(v) / len(v) if v else None
    m6 = lambda dd: (lambda v: sum(v) / len(v) if v else 0.0)([x for d in dd for x in r6[d]])
    add("H6 CVD non-confirming highs", m6(D), fn6, -1, (m6([d for d in D if d < H]), m6([d for d in D if d >= H])), f"n={len(recs['H6'])} ticker-days (bp)")
    add("H7 |delta| given gauge -> move", spear(g, "abs15", "move60", "gauge"), grid_fn("abs15", "move60", "gauge"), -1, gh("abs15", "move60", "gauge"), "(partial Spearman)")
    a8, b8 = [r for r in recs["H8"] if r["agree"]], [r for r in recs["H8"] if not r["agree"]]
    mm = lambda a: sum(r["move"] for r in a) / len(a) if a else float("nan")
    add("H8 bot entries, delta agrees", mm(a8) - mm(b8), rate_diff(a8, b8, "move", scale=1) if a8 and b8 else (lambda ds: None), 1, half(a8, b8, mm) if a8 and b8 else (0, 0), f"agree n={len(a8)} {mm(a8):+.1f} bp, disagree n={len(b8)} {mm(b8):+.1f} bp")
    # Benjamini-Hochberg
    ps = sorted((r["p"], i) for i, r in enumerate(res) if r["p"] is not None)
    m_, q = len(ps), 1.0
    for j in range(m_ - 1, -1, -1):
        q = min(q, ps[j][0] * m_ / (j + 1))
        res[ps[j][1]]["q"] = q
    final = len(D) >= FINAL_MIN_SESSIONS
    print(f"\nscored {len(D)} sessions" + (f": {D[0]} .. {D[-1]}" if D else ""))
    for r in res:
        if r["est"] is None or r["lo"] is None:
            print(f"  {r['name']:32s} not enough data  {r['note']}")
            continue
        ok = (r["want"] > 0 and r["lo"] > 0) or (r["want"] < 0 and r["hi"] < 0)
        same = all(x * r["want"] > 0 for x in r["halves"])
        gate = not r["name"].startswith("H2") or r["note"].endswith(": met")
        verdict = ""
        if final:
            verdict = "PASS" if ok and r.get("q", 1) < .10 and same and gate else "FAIL"
        elif r["name"].startswith("H1"):
            verdict = "DEAD" if r["est"] * r["want"] < 0 else "ON TRACK" if ok else "open"
        print(f"  {r['name']:32s} {r['est']:+8.3f} CI [{r['lo']:+.3f},{r['hi']:+.3f}] q {r.get('q', 1):.3f} halves {r['halves'][0]:+.2f}/{r['halves'][1]:+.2f}  {verdict:8s} {r['note']}")
    if r1:
        cs = {t: sorted(r["speed"] for r in r1 if r["t"] == t) for t in TK}
        cs = {t: (v[len(v) // 3], v[2 * len(v) // 3]) if v else (0, 0) for t, v in cs.items()}
        out = []
        for nm, f in (("slow", lambda r: r["speed"] <= cs[r["t"]][0]), ("mid", lambda r: cs[r["t"]][0] < r["speed"] <= cs[r["t"]][1]), ("fast", lambda r: r["speed"] > cs[r["t"]][1])):
            out.append(f"{nm} {wr([r for r in hi1 if f(r)]) - wr([r for r in lo1 if f(r)]):+.1f}")
        print("  H1 within approach-speed thirds (not a separate test): " + " ".join(out))
    print("\n" + ("FINAL: verdicts above stand." if final else f"EARLY READ: nothing can pass before {FINAL_MIN_SESSIONS} sessions. Only H1 has a status."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", default=os.path.join(ROOT, "_footprint_cache"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "_orderflow_cache"))
    ap.add_argument("--gauge", default=os.path.join(ROOT, "live", "vol_gauge_log.jsonl"))
    ap.add_argument("--ledger", default=os.path.join(ROOT, "bot_executions_log.jsonl"))
    ap.add_argument("--replay-discovery", action="store_true", help="code check on the spent sample: NOT A RESULT")
    a = ap.parse_args()
    all_m1 = sorted(os.path.basename(f)[3:13] for f in glob.glob(os.path.join(a.footprint, "M1-*.json")))
    s5 = sorted(os.path.basename(f)[3:13] for f in glob.glob(os.path.join(a.footprint, "S5-*.json")))
    now = dt.datetime.now(NY)
    today = now.date().isoformat()
    if a.replay_discovery:
        print("!!! REPLAY OF THE DISCOVERY SAMPLE: A CODE CHECK, NOT A RESULT !!!")
        days = [d for d in s5 if DISCOVERY[0] <= d <= DISCOVERY[1] and d in all_m1]
    else:
        days = [d for d in s5 if d >= START and d in all_m1 and (d < today or now.hour * 60 + now.minute >= 16 * 60 + 20)]
        assert all(d >= START for d in days)
    print(f"sessions to score: {len(days)}; early read {EARLY_READ}; final at {FINAL_MIN_SESSIONS} sessions")
    st_ = Store(a.footprint, a.cache)
    recs = defaultdict(list)
    ig = st_._load("index_gamma.json")
    ratio_med = {t: st.median(v["ratio"] for k, v in ig.items() if k.endswith(t) and v.get("ratio")) if any(k.endswith(t) and v.get("ratio") for k, v in ig.items()) else SEED_RATIO[t] for t in TK}
    for d in days:
        i = all_m1.index(d)
        if not i:
            continue
        score_day(st_, d, all_m1[i - 1], ratio_med, recs)
        print("  scored", d, flush=True)
    score_grid(st_, days, a.gauge, a.ledger, recs)
    report(recs, days)


if __name__ == "__main__":
    main()
