"""STRIKE CUE -- DISPLAY ONLY. What history says happens when SPY / QQQ / IWM touches the next whole-dollar strike, given the 0DTE flow
in the six contracts around it (strikes K-1, K, K+1, calls and puts) and the gamma / volume regime. Writes live/strikes.json for the
flow viewer's STRIKE pill (cleanbot-strikes service). UW REST only; never touches the bot, Webull or orders.

WHERE THE NUMBERS COME FROM: check_sixpack_lake.py, a pre-registered holdout scored once on 2026-11-07 (2024-08-20..2026-09-25,
512 sessions, 96k touches). Touch = price comes from >= 5 bp beyond the zone K +- 0.10 and enters it; outcome within 10 minutes:
THROUGH (5 bp past), REVERSE (5 bp back) or STALL. What passed:
  - a five-minute volume surge across the six contracts -> the strike rarely STALLS (SPY 9.7 -> 5.0%, QQQ 5.0 -> 2.4, IWM 7.7 -> 5.4);
    strongest where stalls are common: positive gamma (SPY 26 -> 16%) and low relative volume (SPY 18 -> 11%).
  - SPY: push-side ATM bought in the last 2 minutes -> fewer reversals (65.8 -> 62.7%).
  - at the touch: push-side OTM (the strike ahead) bought over the touch minute + the next -> fewer reversals, SPY and IWM
    (about -5 points; QQQ none, so not shown for QQQ).
  The dealer-hedging split did NOT hold: the flow effects are no larger in negative gamma; gamma changes the base rates instead.
  The rates shown are the holdout's own (in-sample now), looked up by the frozen tercile cuts in strike_cue_tables.json.

LIVE MEASUREMENT, matched to the test: same-day contracts' cumulative ask / bid / total volume from /stock/{t}/option-contracts
(one call per ticker, every 15 s on fixed marks), differenced at minute boundaries into the same 1-minute windows; regime = sign of
/spot-exposures gamma_per_one_percent_move_oi (latest minute); relative volume = stock volume 09:30..last closed minute over the median
of the same for the prior 20 sessions; price = the bot's live price (live/state.json, ~5 s), UW stock-state if that goes stale.
Every touch is logged with its cue values to live/strikes/DATE.jsonl (a forward record; the cue itself is never retuned).

    python strike_cue.py                 (service loop)
    python strike_cue.py --build-tables  (once, on the PC: freezes strike_cue_tables.json from _sixpack_cache)
"""
import argparse, glob, json, math, os, sys, time
from collections import deque
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TK = ("SPY", "QQQ", "IWM")
R_BP, BZ, T_MIN = 5, 0.10, 10
POLL_MARKS = (1, 16, 31, 46)            # seconds past each minute
PRICE_S = 2
START_MOD, MIN_TOUCH_MOD, MAX_TOUCH_MOD = 9 * 60 + 10, 9 * 60 + 35, 15 * 60 + 45
TABLES = os.path.join(HERE, "strike_cue_tables.json")
OUT = os.path.join(HERE, "live", "strikes.json")
LOGDIR = os.path.join(HERE, "live", "strikes")
PUSH_ATM_TK = ("SPY",)                  # P4 passed on SPY only
PUSH_OTM_TK = ("SPY", "IWM")            # P7: SPY and IWM carried it, QQQ ~0


# ------------------------------------------------------------------ frozen tables (PC, once)
def build_tables():
    import numpy as np
    import check_sixpack_lake as C
    hold = [os.path.basename(f)[:10] for f in sorted(glob.glob(os.path.join(C.CACHE, "events-holdout", "*.json")))]
    vol, gex = json.load(open(os.path.join(C.CACHE, "volume.json"))), json.load(open(os.path.join(C.CACHE, "gex_api.json")))
    ev = C.load("holdout", hold, vol, gex)
    def terc(vals):
        a = sorted(vals); return [a[len(a) // 3], a[2 * len(a) // 3]]
    def rates(rs):
        n = len(rs)
        return {"through": round(100 * sum(r["res"] == "through" for r in rs) / n, 1), "stall": round(100 * sum(r["res"] == "stall" for r in rs) / n, 1),
                "reverse": round(100 * sum(r["res"] == "reverse" for r in rs) / n, 1), "n": n} if n else None
    T = {"source": "check_sixpack_lake.py holdout 2024-08-20..2026-09-25, scored 2026-11-07", "built": datetime.now().isoformat(timespec="seconds"),
         "R_bp": R_BP, "zone": BZ, "T_min": T_MIN, "tickers": {}}
    for tk in TK:
        q = [r for r in ev if r["t"] == tk and r.get("gex") is not None]
        cut = {"surge": terc([r["pre5|six|surge"] for r in q if r.get("pre5|six|surge") is not None]),
               "rvol": terc([r["rvol"] for r in q if r.get("rvol") is not None]),
               "push_atm2": terc([r["pre2|push ATM|agg"] for r in q if r.get("pre2|push ATM|agg") is not None]),
               "push_otm_at": terc([r["at|push OTM|agg"] for r in q if r.get("at|push OTM|agg") is not None and r["unres2"]])}
        t3 = lambda v, c: None if v is None else (0 if v <= c[0] else (2 if v > c[1] else 1))
        gs = lambda r: "pos" if r["gex"] >= 0 else "neg"
        surge, base = {}, {}
        for r in q:
            su = t3(r.get("pre5|six|surge"), cut["surge"]); rv = t3(r.get("rvol"), cut["rvol"])
            base.setdefault(gs(r), []).append(r)
            if su is None: continue
            surge.setdefault(f"{gs(r)}|*|{su}", []).append(r)
            if rv is not None: surge.setdefault(f"{gs(r)}|{rv}|{su}", []).append(r)
        pa, po = {}, {}
        for r in q:
            v = t3(r.get("pre2|push ATM|agg"), cut["push_atm2"])
            if v is not None: pa.setdefault(f"{gs(r)}|{v}", []).append(r); pa.setdefault(f"*|{v}", []).append(r)
            if r["unres2"]:
                v = t3(r.get("at|push OTM|agg"), cut["push_otm_at"])
                po.setdefault(f"{gs(r)}|open2", []).append(r)
                if v is not None: po.setdefault(f"{gs(r)}|{v}", []).append(r); po.setdefault(f"*|{v}", []).append(r)
        T["tickers"][tk] = {"cuts": cut, "base": {k: rates(v) for k, v in base.items()},
                            "surge": {k: rates(v) for k, v in surge.items()},
                            "push_atm2": {k: rates(v) for k, v in pa.items()}, "push_otm_at": {k: rates(v) for k, v in po.items()}}
    tmp = TABLES + ".tmp"; json.dump(T, open(tmp, "w"), indent=1); os.replace(tmp, TABLES)
    print("wrote", TABLES)
    for tk in TK:
        t = T["tickers"][tk]; print(tk, "cuts", {k: [round(x, 3) for x in v] for k, v in t["cuts"].items()},
                                     "| cells n min", min(v["n"] for v in t["surge"].values()))


# ------------------------------------------------------------------ live
def now_et():
    import positioning_map as P
    return datetime.now(P.NY)


def mod_of(t): return t.hour * 60 + t.minute


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


class Ticker:
    def __init__(self, tk, tables):
        self.tk, self.T = tk, tables["tickers"][tk]
        self.snaps = deque(maxlen=200)       # (as-of epoch, {(K, cp): (ask, bid, vol)}) cumulative
        self.lag = None                      # seconds UW's tape trails the clock (poll time - newest last_tape_time)
        self.price = None; self.price_at = 0.0; self.p0 = None
        self.armed = {}                      # K -> +1 (was below, R away) / -1 (was above)
        self.touch = None                    # active touch dict
        self.recent = deque(maxlen=6)
        self.gex = None; self.gex_at = None
        self.rvol = None; self.vol_base = None
        self.err = None
        self.day0 = self.open_ts = None      # set by run(); lets a touch log its pre-touch cue the moment it happens

    # flow ---------------------------------------------------------------
    def add_snap(self, ts, rows):
        """ts = poll time. The snapshot is stamped with how far UW's tape has got (newest last_tape_time across the day's contracts),
        so a minute counts as complete only once UW's data has passed its end: a lagging feed delays the cue instead of mis-binning it."""
        import positioning_map as P
        newest = None
        for x in rows:
            lt = x.get("last_tape_time")
            if lt:
                try: e = datetime.fromisoformat(str(lt).replace("Z", "+00:00")).timestamp()
                except ValueError: continue
                if e <= ts + 5 and (newest is None or e > newest): newest = e
        asof = min(ts, newest) if newest is not None else ts
        self.lag = round(ts - asof, 1)
        cum = {}
        for x in rows:
            o = P.parse_occ(x.get("option_symbol"))
            if not o: continue
            k = o[3]
            if abs(k - round(k)) > 1e-6: continue
            cum[(int(round(k)), o[2])] = (P.num(x.get("ask_volume")) or 0.0, P.num(x.get("bid_volume")) or 0.0, P.num(x.get("volume")) or 0.0)
        if cum: self.snaps.append((asof, cum))

    def at(self, boundary_ts, open_ts):
        """cumulative snapshot at a minute boundary: the first whose tape has reached it (within 25 s past it); before the open nothing
        has traded."""
        for ts, cum in self.snaps:
            if boundary_ts <= ts < boundary_ts + 25: return cum
        if boundary_ts <= open_ts: return {}
        return None

    def window(self, m_a, m_b, keys, day0, open_ts):
        """{key: (ask, bid, vol)} traded in minutes [m_a, m_b) ET; None if a boundary snapshot is missing."""
        A, B = self.at(day0 + m_a * 60, open_ts), self.at(day0 + m_b * 60, open_ts)
        if A is None or B is None: return None
        out = {}
        for k in keys:
            a, b = A.get(k, (0, 0, 0)), B.get(k, (0, 0, 0))
            out[k] = tuple(max(0.0, y - x) for x, y in zip(a, b))
        return out

    # price / touches ----------------------------------------------------------
    def on_price(self, p, t, logf):
        if p is None: return
        m = mod_of(t)
        if self.p0 is None and m >= 570: self.p0 = p
        self.price, self.price_at = p, time.time()
        if self.p0 is None: return
        Rr = self.p0 * R_BP / 1e4
        if self.touch:
            T = self.touch; K, sd = T["K"], T["side"]
            thr = p >= K + BZ + Rr if sd > 0 else p <= K - BZ - Rr
            rev = p <= K - BZ - Rr if sd > 0 else p >= K + BZ + Rr
            if thr or rev or time.time() - T["t0"] >= T_MIN * 60:
                T["res"] = "through" if thr else ("reverse" if rev else "stall")
                T["res_at"] = t.strftime("%H:%M:%S")
                self.recent.appendleft({k: T[k] for k in ("K", "side", "since", "res", "res_at")})
                if logf: logf({"ev": "resolved", "tk": self.tk, **{k: v for k, v in T.items() if k != "t0"}})
                self.touch = None
        for K in range(int(math.floor(p)) - 3, int(math.ceil(p)) + 4):
            if p <= K - BZ - Rr: self.armed[K] = 1
            elif p >= K + BZ + Rr: self.armed[K] = -1
            elif K - BZ <= p <= K + BZ and K in self.armed and self.touch is None and MIN_TOUCH_MOD <= m <= MAX_TOUCH_MOD:
                sd = self.armed.pop(K)
                self.touch = {"K": K, "side": sd, "since": t.strftime("%H:%M:%S"), "m0": m, "t0": time.time(), "res": None}
                if logf: logf(self._touch_event(K, sd, m))
        # a jump across a zone between samples counts as entering it
        for K in list(self.armed):
            sd = self.armed[K]
            if self.touch is None and MIN_TOUCH_MOD <= m <= MAX_TOUCH_MOD and ((sd > 0 and p > K + BZ) or (sd < 0 and p < K - BZ)):
                self.armed.pop(K)
                self.touch = {"K": K, "side": sd, "since": t.strftime("%H:%M:%S"), "m0": m, "t0": time.time(), "res": None, "jumped": True}
                if logf: logf(dict(self._touch_event(K, sd, m), jumped=True))

    def _touch_event(self, K, sd, m):
        """the forward record: what the pill said at the moment of the touch, logged with the touch itself so a touch that
        resolves within seconds is not missing from it (11-09: the cue used to be logged only once the at-touch read existed)."""
        ev = {"ev": "touch", "tk": self.tk, "K": K, "side": sd, "since": self.touch["since"], "m0": m, "gs": self.gs(), "rvol": self.rvol}
        if self.day0 is not None:
            try: ev["pre"] = self.strike_cue(K, sd, m, self.day0, self.open_ts)
            except Exception as e: ev["pre_err"] = type(e).__name__
        return ev

    # cue -----------------------------------------------------------------------
    def tercile(self, v, name):
        c = self.T["cuts"][name]
        return None if v is None else (0 if v <= c[0] else (2 if v > c[1] else 1))

    def gs(self): return None if self.gex is None else ("pos" if self.gex >= 0 else "neg")

    def strike_cue(self, K, side, m_now, day0, open_ts):
        six = [(K + d, cp) for d in (-1, 0, 1) for cp in ("C", "P")]
        # the latest minute UW's tape fully covers: now, or one minute back when the feed lags
        for m_eff in (m_now, m_now - 1):
            w5 = self.window(m_eff - 5, m_eff, six, day0, open_ts); wb = self.window(m_eff - 20, m_eff - 5, six, day0, open_ts)
            if w5 is not None and wb is not None: break
        out = {"K": K, "side": side, "dist": round(K - self.price, 2) if self.price else None, "armed": self.armed.get(K) == side,
               "as_of": f"{m_eff // 60:02d}:{m_eff % 60:02d}"}
        if w5 is not None and wb is not None:
            v5, vb = sum(x[2] for x in w5.values()), sum(x[2] for x in wb.values())
            su = (v5 / 5) / (vb / 15 + 1) if v5 >= 50 else None
            out["six"] = {"surge": None if su is None else round(su, 2), "vol5": int(v5), "t": self.tercile(su, "surge")}
            g, rv = self.gs(), self.tercile(self.rvol, "rvol")
            if g and out["six"]["t"] is not None:
                key = f"{g}|{rv}|{out['six']['t']}"; cell = self.T["surge"].get(key)
                if cell is None or cell["n"] < 200: key = f"{g}|*|{out['six']['t']}"; cell = self.T["surge"].get(key)
                out["rates"] = dict(cell or {}, key=key)
                out["quiet_vs_surging"] = [self.T["surge"].get(f"{g}|*|0"), self.T["surge"].get(f"{g}|*|2")]
        else:
            out["warming"] = True
        if self.tk in PUSH_ATM_TK:
            k = (K, "C" if side > 0 else "P")
            w2 = self.window(m_eff - 2, m_eff, [k], day0, open_ts)
            if w2 is not None:
                a, b, _ = w2[k]; agg = (a - b) / (a + b) if a + b >= 20 else None
                t = self.tercile(agg, "push_atm2"); g = self.gs()
                out["push_atm2"] = {"agg": None if agg is None else round(agg, 3), "t": t, "contract": f"{K}{k[1]}",
                                    "rates": self.T["push_atm2"].get(f"{g or '*'}|{t}") if t is not None else None,
                                    "low": self.T["push_atm2"].get(f"{g or '*'}|0"), "high": self.T["push_atm2"].get(f"{g or '*'}|2")}
        return out

    def touch_cue(self, m_now, day0, open_ts):
        T = self.touch
        if not T: return None
        out = {k: T[k] for k in ("K", "side", "since", "m0")}
        out["secs"] = int(time.time() - T["t0"])
        if self.tk in PUSH_OTM_TK:
            K, sd = T["K"], T["side"]; k = (K + sd, "C" if sd > 0 else "P")
            out["contract"] = f"{k[0]}{k[1]}"
            g = self.gs() or "*"
            if m_now >= T["m0"] + 2:
                w = self.window(T["m0"], T["m0"] + 2, [k], day0, open_ts)
                if w is not None:
                    a, b, _ = w[k]; agg = (a - b) / (a + b) if a + b >= 20 else None
                    t = self.tercile(agg, "push_otm_at")
                    out["push_otm"] = {"agg": None if agg is None else round(agg, 3), "t": t,
                                       "rates": self.T["push_otm_at"].get(f"{g}|{t}") if t is not None else None,
                                       "open2": self.T["push_otm_at"].get(f"{g}|open2"),
                                       "low": self.T["push_otm_at"].get(f"{g}|0"), "high": self.T["push_otm_at"].get(f"{g}|2")}
                else:
                    out["reading_at"] = f"{(T['m0'] + 2) // 60:02d}:{(T['m0'] + 2) % 60:02d}, waiting on UW's tape"
            else:
                out["reading_at"] = f"{(T['m0'] + 2) // 60:02d}:{(T['m0'] + 2) % 60:02d}"
        else:
            out["no_test"] = "the at-the-touch read held on SPY and IWM, not QQQ"
        return out


def load_vol_base(uw, tk, d):
    """median cumulative stock volume curve (390 minutes) over the prior 20 sessions; cached per day."""
    import market_calendar as MC
    import positioning_map as P
    f = os.path.join(LOGDIR, f"volbase-{d.isoformat()}.json")
    cache = json.load(open(f)) if os.path.exists(f) else {}
    if tk in cache: return cache[tk]
    curves, x = [], d
    while len(curves) < 20:
        x = MC.previous_trading_day(x - timedelta(days=1))
        rows = uw.get(f"/stock/{tk}/ohlc/1m", date=x.isoformat(), limit=2500).get("data") or []
        a = [0.0] * 390
        for r in rows:
            t = datetime.fromisoformat(str(r["start_time"]).replace("Z", "+00:00")).astimezone(P.NY); m = mod_of(t) - 570
            if 0 <= m < 390: a[m] += P.num(r.get("volume")) or 0.0
        c, s = [], 0.0
        for v in a: s += v; c.append(s)
        if s > 0: curves.append(c)
        if (d - x).days > 45: break
    med = [sorted(col)[len(col) // 2] for col in zip(*curves)] if curves else None
    cache[tk] = med; jdump(cache, f)
    return med


TIDE_OUT = os.path.join(HERE, "live", "market_tide.json")


def write_market_tide(uw, d):
    """Market Tide for the flow viewer's lower pane (user, 2026-11-13): UW's market-wide
    running net call premium and net put premium (ask-lift minus bid-hit), 1-minute,
    regular session. Here because this service already polls UW once a minute in its own
    process -- the bot's loop stays out of it. DISPLAY ONLY: in check_market_tide.py the
    15-min tide slope ranked the next 30-60 min of SPY/QQQ at +0.08..+0.10 (IS and OOS,
    weak) and added nothing as a rule filter.
    Stamps are ET wall-clock as UTC, the chart's convention (live_state._et_stamp)."""
    import positioning_map as P
    try:
        rows = uw.get("/market/market-tide", date=d.isoformat(), interval_5m="false").get("data") or []
        out = []
        for r in rows:
            t = datetime.fromisoformat(str(r["timestamp"])).astimezone(P.NY)
            if not (570 <= mod_of(t) <= 960):
                continue
            stamp = int(datetime(t.year, t.month, t.day, t.hour, t.minute, tzinfo=timezone.utc).timestamp())
            out.append([stamp, P.num(r.get("net_call_premium")), P.num(r.get("net_put_premium")), P.num(r.get("net_volume"))])
        jdump({"ts": time.time(), "day": d.isoformat(), "rows": out}, TIDE_OUT)
    except Exception as e:                      # noqa: BLE001 -- context only, never stop the cue
        print(f"  ⚠️ market tide: {type(e).__name__}", flush=True)


def bot_price(tk):
    try:
        s = json.load(open(os.path.join(HERE, "live", "state.json")))
        if time.time() - float(s.get("ts") or 0) > 20: return None
        t = (s.get("tickers") or {}).get(tk) or {}
        pl = t.get("px_live") or ((t.get("px") or [None])[-1])
        return float(pl[1]) if pl else None
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return None


def run():
    import market_calendar as MC
    import positioning_map as P
    tables = json.load(open(TABLES))
    uw = P.UW()
    print(f"strike cue: tables from {tables['source']}", flush=True)
    while True:
        t = now_et(); d = t.date()
        if not MC.is_trading_day(d) or mod_of(t) >= MC.close_mod(d) + 1 or mod_of(t) < START_MOD:
            if os.path.exists(OUT):
                try:
                    o = json.load(open(OUT))
                    if o.get("live"): o["live"] = False; jdump(o, OUT)
                except (OSError, ValueError): pass
            time.sleep(30); continue
        day0 = datetime(d.year, d.month, d.day, tzinfo=P.NY).timestamp(); open_ts = day0 + 570 * 60
        close_m = MC.close_mod(d)
        T = {tk: Ticker(tk, tables) for tk in TK}
        for X in T.values(): X.day0, X.open_ts = day0, open_ts
        os.makedirs(LOGDIR, exist_ok=True)
        logp = os.path.join(LOGDIR, f"{d.isoformat()}.jsonl")
        def logf(o, logp=logp):
            with open(logp, "a") as f: f.write(json.dumps(o, separators=(",", ":")) + "\n")
        for tk in TK:
            try: T[tk].vol_base = load_vol_base(uw, tk, d)
            except Exception as e: T[tk].err = f"volume base: {type(e).__name__}"
        print(f"session {d}: volume bases ready, polling", flush=True)
        last_poll = last_slow = last_write = 0.0; last_mark = None; calls0 = uw.calls; stale_px = {tk: 0.0 for tk in TK}
        while True:
            t = now_et(); m = mod_of(t)
            if t.date() != d or m >= close_m + 1: break
            # price every PRICE_S
            for tk in TK:
                p = bot_price(tk)
                if p is None and time.time() - stale_px[tk] > 10:
                    stale_px[tk] = time.time()
                    p = P.num(((uw.get(f"/stock/{tk}/stock-state").get("data")) or {}).get("close"))
                T[tk].on_price(p, t, logf)
            # flow on fixed marks
            mark = (m, max([s for s in POLL_MARKS if s <= t.second], default=None))
            if mark[1] is not None and mark != last_mark:
                last_mark = mark
                for tk in TK:
                    try:
                        rows = uw.get(f"/stock/{tk}/option-contracts", expiry=d.isoformat()).get("data") or []
                        if len(rows) >= 500:
                            rows = sum(((uw.get(f"/stock/{tk}/option-contracts", expiry=d.isoformat(), option_type=ty).get("data") or [])
                                        for ty in ("call", "put")), [])
                        T[tk].add_snap(time.time(), rows); T[tk].err = None if rows else "option-contracts: empty"
                    except Exception as e:
                        T[tk].err = f"option-contracts: {type(e).__name__}"
                last_poll = time.time()
            # regime once a minute
            if time.time() - last_slow >= 60 and t.second >= 5:
                last_slow = time.time()
                for tk in TK:
                    try:
                        rows = uw.get(f"/stock/{tk}/spot-exposures").get("data") or []
                        if rows:
                            g = P.num(rows[-1].get("gamma_per_one_percent_move_oi"))
                            if g is not None: T[tk].gex, T[tk].gex_at = g, str(rows[-1].get("start_time") or "")[11:16]
                        if T[tk].vol_base and m > 570:
                            rows = uw.get(f"/stock/{tk}/ohlc/1m", date=d.isoformat(), limit=500).get("data") or []
                            cum = 0.0
                            for r in rows:
                                rt = datetime.fromisoformat(str(r["start_time"]).replace("Z", "+00:00")).astimezone(P.NY)
                                if 570 <= mod_of(rt) < m: cum += P.num(r.get("volume")) or 0.0
                            b = T[tk].vol_base[min(m - 570 - 1, 389)]
                            T[tk].rvol = cum / b if b else None
                    except Exception as e:
                        T[tk].err = f"regime: {type(e).__name__}"
                write_market_tide(uw, d)
            # write every 2 s
            if time.time() - last_write >= 2:
                last_write = time.time()
                out = {"ts": time.time(), "at": t.strftime("%H:%M:%S"), "live": True, "day": d.isoformat(), "R_bp": R_BP, "zone": BZ,
                       "T_min": T_MIN, "source": tables["source"], "calls": uw.calls - calls0, "poll_age": round(time.time() - last_poll, 1), "tickers": {}}
                for tk, X in T.items():
                    o = {"spot": X.price, "p0": X.p0, "lag": X.lag, "gex": X.gex, "gex_at": X.gex_at, "gs": X.gs(),
                         "rvol": None if X.rvol is None else round(X.rvol, 2), "rvol_t": X.tercile(X.rvol, "rvol"), "err": X.err,
                         "recent": list(X.recent), "strikes": [], "touch": None}
                    if X.price:
                        up = int(math.floor(X.price + BZ)) + 1; dn = int(math.ceil(X.price - BZ)) - 1
                        o["strikes"] = [X.strike_cue(up, 1, m, day0, open_ts), X.strike_cue(dn, -1, m, day0, open_ts)]
                        o["touch"] = X.touch_cue(m, day0, open_ts)
                        if X.touch and not X.touch.get("logged_at") and o["touch"].get("push_otm"):
                            X.touch["logged_at"] = True       # the at-the-touch read, once it exists (SPY / IWM, touch minute + next)
                            logf({"ev": "at", "tk": tk, "K": X.touch["K"], "side": X.touch["side"], "since": X.touch["since"], "gs": X.gs(),
                                  "rvol": X.rvol, "at": o["touch"]})
                    out["tickers"][tk] = o
                jdump(out, OUT)
            time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-tables", action="store_true")
    a = ap.parse_args()
    if a.build_tables: build_tables()
    else: run()
