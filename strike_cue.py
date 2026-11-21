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
(one call per ticker, every 5 s on fixed marks since the Advanced tier 2026-11-19; was 15 s), differenced at minute boundaries into the same 1-minute windows; regime = sign of
/spot-exposures gamma_per_one_percent_move_oi (latest minute); relative volume = stock volume 09:30..last closed minute over the median
of the same for the prior 20 sessions; price = the bot's live price (live/state.json, ~5 s), UW stock-state if that goes stale.
Every touch is logged with its cue values to live/strikes/DATE.jsonl (a forward record; the cue itself is never retuned).

THRU (user, 2026-11-14): a strike BREACH -- the median of the last three 5-second closes of the sampled price comes from >= 5 bp
beyond K and reaches K + $0.10 (check_breach_ic's event, 09:45-15:30, one per ticker per 5 min) -- is scored with the FROZEN
check_breach_ic Lasso (breach_model_frozen.json, 10 inputs: approach speed, K-call ask-bid, 0DTE delta flow 5/15 min, other indices
breaking, relative volume, gamma, six-pack surge, time of day) and shown for 5 minutes with its quintile's holdout rates from
strike_cue_thru.json (next strike first 14 / 19 / 26 / 28 / 33%). Logged as "breach" / "breach_res" events. Offline replay of
three lake days through this code: 221 of 221 breaches matched, score corr 0.999, same quintile 99-100%. Live differences:
the bot's 2 s price samples instead of option-print prices, UW tape lag, relative volume over a median (not mean) base.
DISPLAY ONLY: the score ranked breaches out of sample (holdout IC +0.047) but failed the money test.

PULLBACK (user, 2026-11-18): check_structure's setup live -- zigzag swings (0.10% of the day's first price) on 1-minute
closes of the same 5-second median path; after a LOWER LOW (under the last low) the bounce's high is tracked and the
setup fires when price comes 0.02% off it while it is still a LOWER HIGH (bullish: the mirror). Scored with the frozen
check_structure_ic Lasso (pullback_model_frozen.json, 9 inputs: stop size, dealer gamma, the gamma wall's share near
price, the other indices' 15-min move, relative volume, swings so far, undercut, drop speed, 0DTE delta) and shown for
5 minutes with its quintile's holdout rate of 2R before the stop (38.8 / 41.7 / 40.5 / 38.8 / 44.3%, base 41.6%).
Between setups the pill shows the live context it scores on: dealer gamma, the biggest gamma wall within 1% of price
(0-7 day chain read once after 10:00, gamma at the live price), today's swing structure (HH/HL, LH/LL...) and the other
two indices' 15-min move. The bot's RULE regime stays the prior close (it gates on that); this is the live view.
Logged as "pullback" / "pullback_res". DISPLAY ONLY: ranks out of sample (IC +0.032, partly clean), did not pay.

    python strike_cue.py                 (service loop)
    python strike_cue.py --build-tables  (once, on the PC: freezes strike_cue_tables.json from _sixpack_cache)
    python strike_cue.py --build-thru    (once, on the PC: strike_cue_thru.json from _breach_cache)
"""
import argparse, glob, json, math, os, sys, time
from collections import deque
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TK = ("SPY", "QQQ", "IWM")
R_BP, BZ, T_MIN = 5, 0.10, 10
POLL_MARKS = tuple(range(1, 60, 5))     # seconds past each minute: every 5 s since the UW Advanced tier (2026-11-19, was 15 s)
PRICE_S = 2
START_MOD, MIN_TOUCH_MOD, MAX_TOUCH_MOD = 9 * 60 + 10, 9 * 60 + 35, 15 * 60 + 45
TABLES = os.path.join(HERE, "strike_cue_tables.json")
OUT = os.path.join(HERE, "live", "strikes.json")
LOGDIR = os.path.join(HERE, "live", "strikes")
PUSH_ATM_TK = ("SPY",)                  # P4 passed on SPY only
PUSH_OTM_TK = ("SPY", "IWM")            # P7: SPY and IWM carried it, QQQ ~0
# THRU: the study's breach window (t_in 09:45-15:30) and how long the pill shows a breach
THRU_MIN_MOD, THRU_MAX_MOD, THRU_SHOW_S = 9 * 60 + 45, 15 * 60 + 30, 300
# CONTRADICTION note (check_contradiction.py, holdout 2026-01..11-13, PASS IC +0.303): (pullback 2R %, breach next-strike %)
# by order and by the pullback entry's distance from the whole dollar (< 2 / 2-5 / >= 5 bp). Discovery agreed.
CONTRA_WIN = 120
CONTRA_RATES = {"B>P": [(39, 13), (39, 25), (39, 34)], "P>B": [(3, 22), (2, 32), (1, 33)]}
FRZ = THRU = None                       # breach_model_frozen.json / strike_cue_thru.json, loaded by run()
PB = None                               # pullback_model_frozen.json, loaded by run()
PB_MODEL = os.path.join(HERE, "pullback_model_frozen.json")
PB_R, PB_TURN, PB_BUF, PB_WAIT, PB_HOLD = 0.0010, 0.0002, 0.0000667, 1800, 1800
PB_LO, PB_HI = 9 * 60 + 35, 15 * 60 + 30
TICKERS = {}                            # tk -> Ticker, for the cross-index input
PB_LABELS = {  # (raises the odds, lowers them)
    "s_risk": ("tight stop", "wide stop"),
    "g_total": ("negative gamma", "positive gamma"),
    "g_mag": ("no big gamma wall at price", "big gamma wall at price"),
    "x_ret15": ("{w} moving with you", "{w} moving against you"),
    "v_rvol": ("heavy volume", "light volume"),
    "s_swings": ("busy tape (many swings)", "quiet tape (few swings)"),
    "s_under": ("deep undercut", "shallow undercut"),
    "s_speed": ("fast drop into the low", "slow drop into the low"),
    "f_delta5": ("0DTE delta with you (5m)", "0DTE delta against you (5m)"),
}


def zigzag(mc, R):
    """[(type, value, minute)] confirmed pivots on {minute: value}, alternating (check_structure.zigzag)."""
    ms = sorted(mc)
    if not ms:
        return []
    piv, d = [], 0
    hi_v = lo_v = mc[ms[0]]
    hi_m = lo_m = ms[0]
    for m in ms[1:]:
        v = mc[m]
        if d == 0:
            if v > hi_v: hi_v, hi_m = v, m
            if v < lo_v: lo_v, lo_m = v, m
            if hi_v - v >= R:
                piv.append(("H", hi_v, hi_m)); d, lo_v, lo_m = -1, v, m
            elif v - lo_v >= R:
                piv.append(("L", lo_v, lo_m)); d, hi_v, hi_m = 1, v, m
        elif d == 1:
            if v > hi_v: hi_v, hi_m = v, m
            elif hi_v - v >= R:
                piv.append(("H", hi_v, hi_m)); d, lo_v, lo_m = -1, v, m
        else:
            if v < lo_v: lo_v, lo_m = v, m
            elif v - lo_v >= R:
                piv.append(("L", lo_v, lo_m)); d, hi_v, hi_m = 1, v, m
    return piv


def pb_rank(g, v):
    import bisect
    return (bisect.bisect_left(g, v) + bisect.bisect_right(g, v)) / (2.0 * len(g))


def chain_rows(uw, tk, d):
    """0-7 calendar-day chain (strike, T years, iv, oi, is_call) for the gamma wall, read once after 10:00."""
    import positioning_map as P
    rows = []
    for k in range(0, 8):
        e = d + timedelta(days=k)
        if e.weekday() >= 5:
            continue
        try:
            data = uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat()).get("data") or []
        except Exception:
            continue
        for x in data:
            o = P.parse_occ(x.get("option_symbol"))
            iv, oi = P.num(x.get("implied_volatility")), P.num(x.get("open_interest"))
            if o and iv and oi and 0.01 < iv < 5:
                rows.append((o[3], max(k, 0) / 365.0 + 6.0 / (24 * 365), iv, oi, o[2] == "C"))
    return rows


def strike_gex(rows, spot):
    import numpy as np
    if not rows:
        return {}
    K = np.array([r[0] for r in rows]); T = np.array([r[1] for r in rows]); iv = np.array([r[2] for r in rows])
    oi = np.array([r[3] for r in rows]); sg = np.where(np.array([r[4] for r in rows]), 1.0, -1.0)
    sd = iv * np.sqrt(T)
    d1 = (np.log(spot / K) + 0.5 * sd * sd) / sd
    g = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (spot * sd) * oi * 100 * spot * spot * 0.01 * sg
    out = {}
    for k, v in zip(K, g):
        out[float(k)] = out.get(float(k), 0.0) + float(v)
    return out
BREACHES = deque(maxlen=200)            # (tk, side, epoch) every confirmed breach, for the cross-index input
LABELS = {  # driver -> (label when it RAISES the odds, label when it LOWERS them); the sign is the contribution's
    "a_speed": ("fast approach", "slow approach"),
    "s_kcall": ("K calls leaning against the break", "\u26a0 K calls chasing the break"),
    "f_delta5": ("0DTE delta leaning against (5m)", "\u26a0 0DTE delta chasing (5m)"),
    "f_delta15": ("0DTE delta leaning against (15m)", "\u26a0 0DTE delta chasing (15m)"),
    "x_breach": ("{w} breaking too", "breaking alone"),
    "c_rvol": ("heavy volume", "light volume"),
    "c_gex": ("negative gamma", "positive gamma (pinning)"),
    "s_surge": ("six-pack surging", "six-pack quiet"),
    "c_tod": ("early in the day", "late in the day"),
}


def _label(k, v, others):
    up, dn = LABELS[k]
    return (up if v > 0 else dn).replace("{w}", "/".join(others) or "others")


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


# ------------------------------------------------------------------ THRU (breach) table (PC, once)
THRU_TABLE = os.path.join(HERE, "strike_cue_thru.json")
MODEL = os.path.join(HERE, "breach_model_frozen.json")


def build_thru():
    """strike_cue_thru.json: quintile cut points of the FROZEN breach score (check_breach_forward / breach_model_frozen.json)
    over check_breach_ic's discovery, and what followed in each quintile on its HOLDOUT (2026-01..09, scored once 11-14)."""
    import numpy as np
    import check_breach_ic as B
    import check_breach_forward as F
    frz = json.load(open(MODEL))
    disc, hold = B.load(*B.DISC), B.load(*B.HOLD)
    sd = np.array([F.score(frz, r) for r in disc])
    cuts = [float(x) for x in np.quantile(sd, [0.2, 0.4, 0.6, 0.8])]
    def rates(rs):
        n = len(rs)
        return {"next": round(100 * sum(r["Y"] == 1 for r in rs) / n, 1), "back": round(100 * sum(r["Y"] == -1 for r in rs) / n, 1),
                "neither": round(100 * sum(r["Y"] == 0 for r in rs) / n, 1), "n": n}
    q = [[] for _ in range(5)]
    for r in hold:
        q[int(np.searchsorted(cuts, F.score(frz, r), side="right"))].append(r)
    T = {"source": "check_breach_ic.py holdout 2026-01-02..2026-09-25 (scored once 2026-11-14); frozen score breach_model_frozen.json",
         "built": datetime.now().isoformat(timespec="seconds"), "cuts": cuts, "quintiles": [rates(x) for x in q], "base": rates(hold),
         "note": "the score ranked breaches out of sample (IC +0.047) but FAILED the money test: display only"}
    json.dump(T, open(THRU_TABLE, "w"), indent=1)
    print("wrote", THRU_TABLE, [x["next"] for x in T["quintiles"]], "base", T["base"])


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
        self.snaps = deque(maxlen=600)       # (as-of epoch, {(K, cp): (ask, bid, vol)}) cumulative; 5-s polls -> ~50 min kept
        self.lag = None                      # seconds UW's tape trails the clock (poll time - newest last_tape_time)
        self.price = None; self.price_at = 0.0; self.p0 = None
        self.armed = {}                      # K -> +1 (was below, R away) / -1 (was above)
        self.touch = None                    # active touch dict
        self.recent = deque(maxlen=6)
        self.gex = None; self.gex_at = None
        self.rvol = None; self.vol_base = None
        self.err = None
        self.day0 = self.open_ts = None      # set by run(); lets a touch log its pre-touch cue the moment it happens
        # THRU (strike BREACH, check_breach_ic / breach_model_frozen.json): 5-second bars of the sampled price, median of the
        # last three closes = the study's centred median, known when its third bar closes
        self.delta = {}                      # (K, cp) -> latest UW delta
        self.bars = deque(maxlen=60); self.cur_b = self.cur_c = None
        self.barm = {}                       # K -> (side, armed at epoch, median then)
        self.thru = None; self.last_breach = 0.0
        # PULLBACK (check_structure live)
        self.mclose = {}; self.medhist = []; self.pb_m = None; self.piv = {1: [], -1: []}; self.pb_done = set()
        self.pb_arm = []; self.pullback = None; self.chain = None; self.chain_tried = False
        self.contra_logged = set()

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
            dl = P.num(x.get("delta"))
            if dl is not None: self.delta[(int(round(k)), o[2])] = dl
        if cum: self.snaps.append((asof, cum, dict(self.delta)))     # deltas as of this snapshot, for the THRU delta flow

    def at(self, boundary_ts, open_ts):
        """cumulative snapshot at a minute boundary: the first whose tape has reached it (within 25 s past it); before the open nothing
        has traded."""
        for ts, cum, *_ in self.snaps:
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

    def delta_flow(self, m_a, m_b, day0, open_ts):
        """sum over minutes [m_a, m_b) and every contract of (ask - bid volume that minute) x its delta at the minute's end x 100
        = check_breach_ic's f_delta (which uses each minute bar's delta_close); None if a boundary snapshot is missing."""
        tot = 0.0
        for mm in range(m_a, m_b):
            A = self.at(day0 + mm * 60, open_ts)
            B = next(((c, d) for ts, c, *d in self.snaps if day0 + (mm + 1) * 60 <= ts < day0 + (mm + 1) * 60 + 25), None)
            if A is None or B is None: return None
            cum, dm = B[0], (B[1][0] if B[1] else self.delta)
            for k, (a, b, _v) in cum.items():
                dl = dm.get(k)
                if dl is None: continue
                a0, b0, _ = A.get(k, (0, 0, 0))
                tot += ((a - a0) - (b - b0)) * dl * 100
        return tot

    # price / touches ----------------------------------------------------------
    def on_price(self, p, t, logf):
        if p is None: return
        m = mod_of(t)
        if self.p0 is None and m >= 570: self.p0 = p
        self.price, self.price_at = p, time.time()
        if self.p0 is None: return
        try: self._bar(p, t, logf)
        except Exception as e: self.err = f"thru: {type(e).__name__}"      # noqa: BLE001 -- never stop the touch cue
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

    # THRU: strike breach ------------------------------------------------------------
    def _bar(self, p, t, logf):
        now = time.time(); b = int(now // 5) * 5
        if self.cur_b is None or b == self.cur_b:
            self.cur_b, self.cur_c = b, p; return
        self.bars.append((self.cur_b, self.cur_c)); self.cur_b, self.cur_c = b, p
        if len(self.bars) < 3: return
        med = sorted(c for _b, c in list(self.bars)[-3:])[1]
        self._breach_step(med, t, now, logf)
        try: self._pull_step(med, self.bars[-2][0], t, now, logf)
        except Exception as e: self.err = f"pullback: {type(e).__name__}"      # noqa: BLE001 -- never stop the cue

    def _breach_step(self, med, t, now, logf):
        Rr, m = self.p0 * R_BP / 1e4, mod_of(t)
        th = self.thru
        if th and th.get("res") is None:                       # resolve the live breach (the study's target, 15 min)
            x = th["side"] * (med - th["K"])
            res = "next" if x >= 0.95 else ("back" if x <= -BZ else ("neither" if now - th["t0"] >= 900 else None))
            if res:
                th["res"], th["res_at"] = res, t.strftime("%H:%M:%S")
                if logf: logf({"ev": "breach_res", "tk": self.tk, "K": th["K"], "side": th["side"], "at": th["at"], "res": res,
                               "res_at": th["res_at"], "q": th.get("q")})
        for K in range(int(math.floor(med)) - 3, int(math.ceil(med)) + 4):
            if med <= K - BZ - Rr: self.barm[K] = (1, now, med)
            elif med >= K + BZ + Rr: self.barm[K] = (-1, now, med)
            a = self.barm.get(K)
            if not a: continue
            if (a[0] > 0 and med >= K + BZ) or (a[0] < 0 and med <= K - BZ):
                self.barm.pop(K)
                BREACHES.append((self.tk, a[0], now))
                if THRU_MIN_MOD <= m <= THRU_MAX_MOD and now - self.last_breach >= 300 and FRZ and THRU:
                    self.last_breach = now
                    self._fire(K, a[0], a[1], a[2], med, t, now, logf)

    def _fire(self, K, side, t_arm, med_arm, p_in, t, now, logf):
        import numpy as np
        m, day0, open_ts = mod_of(t), self.day0, self.open_ts
        x = dict.fromkeys(FRZ["coef"])
        x["a_speed"] = side * (p_in - med_arm) / self.p0 * 1e4 / max((now - t_arm) / 60.0, 0.1)
        six = [(K + d, cp) for d in (-1, 0, 1) for cp in ("C", "P")]
        w5 = wb = None
        for m_eff in (m, m - 1):
            w5 = self.window(m_eff - 5, m_eff, six + [(K, "C")], day0, open_ts)
            wb = self.window(m_eff - 20, m_eff - 5, six, day0, open_ts)
            if w5 is not None and wb is not None: break
        if w5 is not None and wb is not None:
            v5, vb = sum(w5[k][2] for k in six), sum(wb[k][2] for k in six)
            x["s_surge"] = (v5 / 5) / (vb / 15 + 1) if v5 >= 50 else 0.0
            a, b, _v = w5[(K, "C")]; x["s_kcall"] = side * (a - b)
            for name, span in (("f_delta5", 5), ("f_delta15", 15)):
                fl = self.delta_flow(m_eff - span, m_eff, day0, open_ts)
                if fl is not None: x[name] = side * fl
        x["c_rvol"], x["c_gex"] = self.rvol, self.gex
        x["c_tod"] = (now - day0) / 60.0 - 570
        hits = [tk for tk, sd, ts in BREACHES if tk != self.tk and sd == side and now - 60 <= ts <= now]
        others = sorted(set(hits))
        x["x_breach"] = len(hits)                           # the study counts breach EVENTS (one ticker can cross two strikes)
        z, contrib = FRZ["intercept"], []
        for k, c in FRZ["coef"].items():
            g = FRZ["grids"][self.tk][k]
            r = 0.5 if x.get(k) is None else float(np.interp(x[k], g, np.linspace(0, 1, len(g))))
            z += c * (r - 0.5); contrib.append((k, c * (r - 0.5)))
        q = int(np.searchsorted(THRU["cuts"], z, side="right"))
        # an input sitting at zero (no other index breaking, no six-pack surge) gets its rank from the frozen grid's
        # ties, not from anything happening: score it as frozen, but don't name it as a driver (11-16: QQQ said
        # "others breaking too" with nobody breaking)
        idle = {k for k in ("x_breach", "s_surge") if not x.get(k)}
        drv = [[_label(k, v, others), round(v, 4)] for k, v in sorted(contrib, key=lambda kv: -abs(kv[1]))
               if k in LABELS and k not in idle and abs(v) >= 0.004][:3]
        self.thru = {"K": K, "side": side, "at": t.strftime("%H:%M:%S"), "t0": now, "z": round(z, 4), "q": q + 1,
                     "rates": THRU["quintiles"][q], "drivers": drv, "with": others, "res": None,
                     "inputs": {k: (None if v is None else round(float(v), 4)) for k, v in x.items()}}
        if logf: logf({"ev": "breach", "tk": self.tk, **{k: v for k, v in self.thru.items() if k not in ("t0", "rates")}})

    # PULLBACK: lower low -> lower high (and the mirror) --------------------------------------------
    def _pull_step(self, med, cb, t, now, logf):
        """med = the median of the last three 5-s closes, CENTRED on the bar that started at epoch cb (the study's
        med[k]); it is stamped with that bar's minute, so minute closes match check_structure's exactly."""
        if not PB or self.day0 is None:
            return
        m = mod_of(t)
        mm_c = int((cb - self.day0) // 60)
        self.mclose[mm_c] = med                           # that minute's latest centred median = its close so far
        self.medhist.append((cb, mm_c, med))
        p0 = self.p0
        if self.pb_m is None or mm_c > self.pb_m:         # a minute is complete: re-read the swings on CLOSED minutes
            self.pb_m = mm_c
            closed = {mm: v for mm, v in self.mclose.items() if mm < mm_c}
            for o in (1, -1):
                piv = zigzag({mm: o * v for mm, v in closed.items()}, PB_R * p0)
                self.piv[o] = piv
                for i in range(2, len(piv)):
                    Le, Hd, Lc = piv[i], piv[i - 1], piv[i - 2]
                    key = (o, Le[2], i)
                    if key in self.pb_done or Le[0] != "L" or Hd[0] != "H" or not Le[1] < Lc[1]:
                        continue
                    self.pb_done.add(key)
                    if i != len(piv) - 1:
                        continue                          # only the newest confirmed low can still be a live setup
                    fb = i >= 4 and piv[i - 3][0] == "H" and piv[i - 4][0] == "L" and Lc[1] > piv[i - 4][1] and Hd[1] > piv[i - 3][1]
                    M = max(o * v for _cb, mm, v in self.medhist if mm >= Le[2])
                    self.pb_arm.append(dict(o=o, Hd=Hd, Le=Le, Lc=Lc, fb=fb, swings=i + 1, M=M, armed=now))
        keep = []
        for a in self.pb_arm:                             # armed setups: wait for the turn off a LOWER high
            z = a["o"] * med
            a["M"] = max(a["M"], z)
            if a["M"] >= a["Hd"][1] or now - a["armed"] > PB_WAIT:
                continue
            if a["M"] - z >= PB_TURN * p0:
                if PB_LO <= m <= PB_HI:
                    self._pull_fire(a, z, med, t, now, logf)
                continue
            keep.append(a)
        self.pb_arm = keep
        pb = self.pullback
        if pb and pb.get("res") is None:                  # resolve: 2R before the stop, 30 minutes
            z = pb["o"] * med
            res = "2R" if pb["z_e"] - z >= 2 * pb["risk"] else ("stop" if z >= pb["z_s"] else ("time" if now - pb["t0"] >= PB_HOLD else None))
            if res:
                pb["res"], pb["res_at"] = res, t.strftime("%H:%M:%S")
                if logf: logf({"ev": "pullback_res", "tk": self.tk, "side": pb["side"], "at": pb["at"], "res": res,
                               "res_at": pb["res_at"], "q": pb["q"]})

    def med_at(self, T):
        """the median known at epoch T (centre bar + 10 s <= T); the earliest if T is before the session's start."""
        import bisect
        if not self.medhist:
            return None
        i = bisect.bisect_right([h[0] for h in self.medhist], T - 10) - 1
        return self.medhist[max(i, 0)][2]

    def _wall(self, spot):
        """(strike, share) of the biggest single-strike |gamma| within 1% of spot, as a share of all within 5%."""
        if not self.chain or not spot:
            return None, None, None
        sg = strike_gex(self.chain, spot)
        near = {k: v for k, v in sg.items() if abs(k / spot - 1) <= 0.01}
        wide = sum(abs(v) for k, v in sg.items() if abs(k / spot - 1) <= 0.05)
        if not near or wide <= 0:
            return None, None, None
        km = max(near, key=lambda k: abs(near[k]))
        return km, abs(near[km]) / wide, sg.get(float(round(spot)))

    def _pull_fire(self, a, z, med, t, now, logf):
        o, side, p0 = a["o"], -a["o"], self.p0
        z_s = a["M"] + PB_BUF * p0
        risk = z_s - z
        p_e = o * z
        m = mod_of(t)
        x = dict.fromkeys(PB["coef"])
        x["s_risk"] = risk / p0
        x["g_total"] = self.gex
        km, share, _gs = self._wall(p_e)
        x["g_mag"] = share
        xr, names = [], []
        for tk2, X2 in TICKERS.items():
            if tk2 == self.tk:
                continue
            a2, b2 = X2.med_at(now - 900), X2.med_at(now)
            if a2 and b2:
                v = side * (b2 / a2 - 1) * 1e4
                xr.append(v); names.append(tk2)
        x["x_ret15"] = sum(xr) / len(xr) if xr else None
        x["v_rvol"] = self.rvol
        x["s_swings"] = a["swings"]
        x["s_under"] = (a["Lc"][1] - a["Le"][1]) / (PB_R * p0)
        x["s_speed"] = (a["Hd"][1] - a["Le"][1]) / p0 / max(a["Le"][2] - a["Hd"][2], 1)
        for m_eff in (m, m - 1):
            fl = self.delta_flow(m_eff - 5, m_eff, self.day0, self.open_ts)
            if fl is not None:
                x["f_delta5"] = side * fl
                break
        zsc, contrib = PB["intercept"], []
        for k, c in PB["coef"].items():
            r = 0.5 if x.get(k) is None else pb_rank(PB["grids"][self.tk][k], x[k])
            zsc += c * (r - 0.5); contrib.append((k, c * (r - 0.5)))
        import bisect
        q = bisect.bisect_right(PB["cuts"], zsc)
        w = "/".join(n for n, v in zip(names, xr) if v > 0) if (x["x_ret15"] or 0) > 0 else "/".join(n for n, v in zip(names, xr) if v < 0)
        drv = [[PB_LABELS[k][0 if v > 0 else 1].replace("{w}", w or "others"), round(v, 4)]
               for k, v in sorted(contrib, key=lambda kv: -abs(kv[1])) if k in PB_LABELS and abs(v) >= 0.002][:3]
        self.pullback = {"side": side, "o": o, "fb": a["fb"], "at": t.strftime("%H:%M:%S"), "t0": now, "z": round(zsc, 4),
                         "q": q + 1, "rate": PB["holdout"][q], "drivers": drv, "entry": round(p_e, 2),
                         "stop": round(o * z_s, 2), "target": round(p_e + side * 2 * risk, 2), "z_e": z, "z_s": z_s, "risk": risk,
                         "res": None, "inputs": {k: (None if v is None else round(float(v), 6)) for k, v in x.items()}}
        if logf: logf({"ev": "pullback", "tk": self.tk, **{k: v for k, v in self.pullback.items() if k not in ("t0", "rate", "z_e", "z_s", "o")}})

    def pull_cue(self):
        """the PULLBACK pill: the live context, and the latest setup for 5 minutes."""
        out = {"gamma": self.gex}
        km, share, gs = self._wall(self.price)
        out["wall"] = None if km is None else {"K": km, "share": round(share, 3)}
        hs = [v for ty, v, _m in self.piv.get(1, []) if ty == "H"][-2:]
        ls = [v for ty, v, _m in self.piv.get(1, []) if ty == "L"][-2:]
        out["trend"] = (("HH" if hs[1] > hs[0] else "LH") + "/" + ("HL" if ls[1] > ls[0] else "LL")) if len(hs) == 2 and len(ls) == 2 else None
        others, now = {}, time.time()
        for tk2, X2 in TICKERS.items():
            if tk2 == self.tk:
                continue
            a2, b2 = X2.med_at(now - 900), X2.med_at(now)
            if a2 and b2: others[tk2] = round((b2 / a2 - 1) * 1e4, 1)
        out["others"] = others
        pb = self.pullback
        if pb and time.time() - pb["t0"] <= THRU_SHOW_S:
            out["setup"] = dict({k: v for k, v in pb.items() if k not in ("t0", "inputs", "z_e", "z_s", "o", "risk")},
                                age=int(time.time() - pb["t0"]))
        return out

    def thru_cue(self):
        th = self.thru
        if not th or time.time() - th["t0"] > THRU_SHOW_S: return None
        return dict({k: v for k, v in th.items() if k not in ("t0", "inputs")}, age=int(time.time() - th["t0"]))

    def contra_cue(self):
        """THRU and PULLBACK disagreeing (check_contradiction.py, user 2026-11-20): the latest breach and the latest
        setup on this ticker, opposite sides, at most CONTRA_WIN apart, shown while the later one is on screen.
        order B>P = the breach came first; P>B = the setup came first. band = the pullback entry's distance from the
        nearest whole dollar (k_entry, bp of the session price: < 2 / 2-5 / >= 5) -- price back at the strike or
        still holding beyond it. rates = the 2026 holdout's (pullback 2R %, breach next-strike %) for that cell."""
        th, pb = self.thru, self.pullback
        if not th or not pb or th["side"] == pb["side"] or not self.p0:
            return None
        now, later = time.time(), max(th["t0"], pb["t0"])
        gap = abs(th["t0"] - pb["t0"])
        if gap > CONTRA_WIN or now - later > THRU_SHOW_S:
            return None
        order = "B>P" if th["t0"] < pb["t0"] else "P>B"
        ke = abs(pb["entry"] - round(pb["entry"])) / self.p0 * 1e4
        band = 0 if ke < 2 else (1 if ke < 5 else 2)
        pr, tr = CONTRA_RATES[order][band]
        return {"order": order, "K": th["K"], "thru_side": th["side"], "thru_at": th["at"], "pull_side": pb["side"],
                "pull_at": pb["at"], "k_entry_bp": round(ke, 1), "band": band, "gap": int(gap), "age": int(now - later),
                "pull_rate": pr, "thru_rate": tr}

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
    global FRZ, THRU
    try:
        FRZ, THRU = json.load(open(MODEL)), json.load(open(THRU_TABLE))
        global PB
        PB = json.load(open(PB_MODEL))
    except (OSError, ValueError) as e:
        print(f"  THRU off: {type(e).__name__} loading {os.path.basename(MODEL)} / {os.path.basename(THRU_TABLE)}", flush=True)
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
        BREACHES.clear()
        TICKERS.clear(); TICKERS.update(T)
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
                if PB and m >= 600:
                    for tk in TK:
                        if not T[tk].chain_tried:
                            T[tk].chain_tried = True
                            try: T[tk].chain = chain_rows(uw, tk, d)
                            except Exception as e: T[tk].err = f"chain: {type(e).__name__}"
            # write every 2 s
            if time.time() - last_write >= 2:
                last_write = time.time()
                out = {"ts": time.time(), "at": t.strftime("%H:%M:%S"), "live": True, "day": d.isoformat(), "R_bp": R_BP, "zone": BZ,
                       "thru_base": THRU["base"] if THRU else None, "thru_source": THRU["source"] if THRU else None,
                       "pullback_base": PB["base"] if PB else None,
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
                    o["thru"] = X.thru_cue()
                    o["pullback"] = X.pull_cue()
                    o["contra"] = X.contra_cue()
                    if o["contra"]:                  # logged once per pair: the forward read joins breach_res / pullback_res
                        ck = (o["contra"]["thru_at"], o["contra"]["pull_at"])
                        if ck not in X.contra_logged:
                            X.contra_logged.add(ck)
                            logf({"ev": "contra", "tk": tk, **{k: v for k, v in o["contra"].items() if k != "age"}})
                    out["tickers"][tk] = o
                jdump(out, OUT)
            time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-tables", action="store_true")
    ap.add_argument("--build-thru", action="store_true")
    a = ap.parse_args()
    if a.build_tables: build_tables()
    elif a.build_thru: build_thru()
    else: run()
