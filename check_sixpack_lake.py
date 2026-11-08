"""PRE-REGISTERED historical holdout: six 0DTE contracts around a strike touch (SPY, QQQ, IWM). Written 2026-11-07, BEFORE any holdout outcome
was looked at. The ideas come from the 2026-10-07..11-05 exploration (scratchpad fp_6pack*.py); this file tests them once on the data lake.

DATA (lake/, all checked against the discovery inputs on 2026-11-07, no outcomes viewed):
  option flow  silver/option-contracts-1m: per contract-minute ask/bid/mid/no-side volume (matched the UW API pull 100% on 5 overlap days)
  price path   silver/trades-core underlying_price on every option trade -> 5-second high/low/close (matched Webull 5s bars, median 0.0-0.2 bp, no lag)
               (bronze/full-tape for the replay days, which trades-core does not reach)
  GEX regime   silver/spot-exposures-1m gamma_per_one_percent_move_oi (to 2026-08-21), then UW /stock/{t}/spot-exposures (same field)
  stock volume UW /stock/{t}/ohlc/1m volume (for relative volume)

WINDOWS
  HOLDOUT   2024-08-20 .. 2026-09-25 (scored ONCE by --run). 2023-10..2024-08 is excluded (spent 2026-09-12, labelled in-sample only);
            2026-10-07 onward is the discovery sample.
  REPLAY    2026-10-07 .. 2026-10-30 (--replay-discovery): code check only, discovery days, says nothing about validity.

EVENT (frozen)
  Whole-dollar strike K, zone K +- 0.10. Price must first be >= R beyond the zone, then enter it (5-second bars). R = 5 bp of the day's
  first price. Outcome within T = 10 min of the touch: THROUGH (R past the far side first), REVERSE (R back on the near side first), STALL
  (neither). Same-bar ambiguity skipped. Touch minute 09:35..15:45 ET. Early-close days skipped.
  Contracts oriented to the push: price rising into K -> push side = calls, fade side = puts (falling: the reverse). OTM = strike ahead
  (K+1 rising), ATM = K, ITM = strike behind. Same-day expiry only.
  agg   = (ask - bid) / (ask + bid) volume in the window, None if ask + bid < 20.  (+ = that contract bought)
  surge = (window volume per minute) / (volume per minute over m0-20..m0-5 + 1)   (six = all six contracts summed; None if < 50)
  windows: pre2 = the 2 minutes before the touch minute, pre5 = the 5 minutes before, at = the touch minute + the next (only for touches still
  unresolved at the end of the next minute).
  controls: sp5 = approach move over the 5 min before the touch (bp, + = toward K); at-the-touch tests also pos2 (bp past K in the push
  direction 2 min after the touch) and sp2 (move over those 2 min).

PRIMARY HYPOTHESES (partial rank correlation of the measure with the outcome indicator, given the controls; tie-averaged ranks)
  P1 SPY  six surge, pre5   -> STALL    negative       (a surging strike does not stall)
  P2 QQQ  six surge, pre5   -> STALL    negative
  P3 IWM  six surge, pre5   -> STALL    negative
  AMENDED 2026-11-07, before the holdout was scored, with the user's sign-off: P1-P3 were written as pre2. The discovery replay showed the
  six-contract pre2 surge flat for SPY (-0.004) while pre5 held for all three (-0.067 / -0.070 / -0.109), so the primary window is pre5;
  pre2 is reported per ticker as information only.
  P4 SPY  push ATM agg, pre2 -> REVERSE negative       (push side bought -> breaks)
  P5 IWM  fade ATM agg, pre2 -> REVERSE positive       (fade side bought -> turns back)
  P6 ALL  fade ATM agg, pre5 -> REVERSE positive
  P7 ALL  push OTM agg, at  -> REVERSE  negative       (given pos2, sp2, sp5)
  PASS (each): (a) 95% day-clustered bootstrap CI (1000) excludes 0 on the predicted side, (b) BH q < 0.10 across P1-P7,
  (c) the point estimate has the predicted sign in both halves of the holdout (split at the middle date).
  Effect sizes also reported as outcome rates, lowest vs highest third of the measure (cut per ticker).

INFORMATION ONLY (no pass/fail): H1 of check_orderflow_forward.py on this data (ALL push ATM agg pre5 -> REVERSE); per-contract rows; all
  three tickers for every measure. Nothing here retunes or replaces the locked forward test.

REGIME SPLITS (asked by the user 2026-11-07: "if the effects are dealer hedging, they'll differ by environment")
  GEX: sign of the ticker's gamma_per_one_percent_move_oi at the minute before the touch (negative vs positive).
       Prediction (dealer hedging): each primary effect is LARGER in the predicted direction in negative gamma. Test: effect(neg) - effect(pos),
       signed so + = supports, day-bootstrap CI. Also base rates of THROUGH / STALL / REVERSE by regime (dealers short gamma chase -> more
       THROUGH; long gamma lean against -> more REVERSE/STALL).
  VOLUME: relative volume at the touch = stock volume 09:30..touch minute / median of the same over the prior 20 sessions (known in real time),
       top third vs bottom third (cut per ticker); and the day's full volume / prior-20 median (day label, look-ahead, descriptive).
       Two-sided, no prediction.
  Splits are secondary: reported with CIs and BH across the split family; they cannot rescue a failed primary.

USAGE  python check_sixpack_lake.py --build            (fetches + builds the touch table; prints no outcomes)
       python check_sixpack_lake.py --replay-discovery (code check on discovery days)
       python check_sixpack_lake.py --run              (scores the holdout ONCE; writes _sixpack_cache/RESULT-holdout.txt)
"""
import argparse, glob, json, os, random, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAKE = os.path.join(HERE, "lake")
CACHE = os.path.join(HERE, "_sixpack_cache")
TK = ("SPY", "QQQ", "IWM")
HOLD = ("2024-08-20", "2026-09-25")
REPLAY = ("2026-10-07", "2026-10-30")
R_BP, T_MIN, BZ = 5, 10, 0.10
M_LO, M_HI = 575, 945
MIN_SIDED, MIN_SIX = 20, 50
ROLES = ("push OTM", "push ATM", "push ITM", "fade OTM", "fade ATM", "fade ITM")
WINDOWS = {"pre2": (-2, 0), "pre5": (-5, 0), "at": (0, 2)}
N_BOOT = 1000


# ------------------------------------------------------------------ data
def lake_days(lo, hi):
    ds = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "silver", "option-contracts-1m", "date=*")))
    return [d for d in ds if lo <= d <= hi]


def price_path(d):
    """{ticker: (bar_sec, hi, lo, close)} 5-second bars from underlying_price on option trades."""
    import polars as pl
    tc = glob.glob(os.path.join(LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))
    src = tc if tc else [os.path.join(LAKE, "bronze", "full-tape", f"{d}.parquet")]
    x = (pl.scan_parquet(src).filter(pl.col("underlying_symbol").is_in(list(TK)) & (pl.col("canceled") != True))
         .select("underlying_symbol", "executed_at", pl.col("underlying_price").cast(pl.Float64)).collect())
    et = x["executed_at"].dt.convert_time_zone("America/New_York")
    x = x.with_columns(sec=(et.dt.hour().cast(pl.Int64) * 3600 + et.dt.minute().cast(pl.Int64) * 60 + et.dt.second().cast(pl.Int64)))
    x = x.filter((pl.col("sec") >= 34200) & (pl.col("sec") < 57600) & (pl.col("underlying_price") > 0)).sort("executed_at")
    x = x.with_columns(b=(pl.col("sec") // 5) * 5)
    g = x.group_by("underlying_symbol", "b").agg(pl.col("underlying_price").max().alias("hi"), pl.col("underlying_price").min().alias("lo"),
                                               pl.col("underlying_price").last().alias("cl")).sort("b")
    out = {}
    for t in TK:
        z = g.filter(pl.col("underlying_symbol") == t)
        if z.height: out[t] = (z["b"].to_numpy(), z["hi"].to_numpy(), z["lo"].to_numpy(), z["cl"].to_numpy())
    return out


def option_flow(d):
    """{(ticker, strike, 'C'/'P'): {minute: (ask, bid, total)}} same-day expiry, whole-dollar strikes."""
    import polars as pl
    f = os.path.join(LAKE, "silver", "option-contracts-1m", f"date={d}", "bars.parquet")
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(TK)) & (pl.col("expiry").cast(pl.Utf8) == d))
         .select("underlying_symbol", "option_type", "strike", "minute_et", "ask_volume", "bid_volume", "mid_volume", "no_side_volume").collect())
    out = defaultdict(dict)
    for t, ot, k, me, a, b, m, n in x.iter_rows():
        if k is None or abs(k - round(k)) > 1e-6: continue
        a, b, m, n = (v or 0 for v in (a, b, m, n))
        out[(t, int(round(k)), "C" if str(ot).lower().startswith("c") else "P")][me.hour * 60 + me.minute] = (a, b, a + b + m + n)
    return out


def first(mask):
    if not mask.size: return None
    i = int(np.argmax(mask)); return i if mask[i] else None


def touches(sc, hi, lo, K, Rr):
    n = len(hi); out = []; i = 0; zl, zh = K - BZ, K + BZ
    while i < n:
        a, b = first(lo[i:] <= zl - Rr), first(hi[i:] >= zh + Rr)
        if a is None and b is None: break
        a = n if a is None else a + i; b = n if b is None else b + i
        if a == b: i = a + 1; continue
        side = 1 if a < b else -1; s0 = min(a, b)
        t = first(hi[s0 + 1:] >= zl) if side == 1 else first(lo[s0 + 1:] <= zh)
        if t is None: break
        t += s0 + 1
        end = int(np.searchsorted(sc, sc[t] + T_MIN * 60, side="right"))
        thr = first(hi[t:end] >= zh + Rr) if side == 1 else first(lo[t:end] <= zl - Rr)
        rev = first(lo[t:end] <= zl - Rr) if side == 1 else first(hi[t:end] >= zh + Rr)
        if thr is not None and rev is not None and thr == rev: i = t + thr + 1; continue
        if thr is None and rev is None: res, j = "stall", end
        elif rev is None or (thr is not None and thr < rev): res, j = "through", t + thr
        else: res, j = "reverse", t + rev
        out.append((side, t, res, int(sc[min(j, n - 1)])))
        i = max(j, t + 1)
    return out


def build_day(d):
    path, flow = price_path(d), option_flow(d)
    ev = []
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300: continue                      # early close
        p0 = float(cl[0]); Rr = p0 * R_BP / 1e4
        p_at = lambda s_: float(cl[max(int(np.searchsorted(sc, s_)) - 1, 0)])
        for K in range(int(np.floor(lo.min())), int(np.ceil(hi.max())) + 1):
            for side, t, res, rsec in touches(sc, hi, lo, K, Rr):
                tsec = int(sc[t]); m0 = tsec // 60
                if m0 < M_LO or m0 > M_HI: continue
                pc_, fc_ = ("C", "P") if side == 1 else ("P", "C")
                role = {"push OTM": (K + side, pc_), "push ATM": (K, pc_), "push ITM": (K - side, pc_),
                        "fade OTM": (K - side, fc_), "fade ATM": (K, fc_), "fade ITM": (K + side, fc_)}
                if (tk,) + role["push ATM"] not in flow or (tk,) + role["fade ATM"] not in flow: continue
                e = dict(d=d, t=tk, K=K, side=side, m0=m0, res=res, unres2=rsec > (m0 + 2) * 60,
                         sp5=side * (p_at(m0 * 60) - p_at((m0 - 5) * 60)) / p0 * 1e4,
                         sp2=side * (p_at((m0 + 2) * 60) - p_at(m0 * 60)) / p0 * 1e4,
                         pos2=side * (p_at((m0 + 2) * 60) - K) / p0 * 1e4)
                for w, (o0, o1) in WINDOWS.items():
                    six_v = six_b = 0.0
                    for r in ROLES:
                        c = flow.get((tk,) + role[r], {})
                        a = b = v = base = 0.0
                        for m in range(m0 + o0, m0 + o1):
                            x = c.get(m)
                            if x: a += x[0]; b += x[1]; v += x[2]
                        for m in range(m0 - 20, m0 - 5):
                            x = c.get(m)
                            if x: base += x[2]
                        n_ = o1 - o0
                        e[f"{w}|{r}|agg"] = (a - b) / (a + b) if a + b >= MIN_SIDED else None
                        e[f"{w}|{r}|surge"] = (v / n_) / (base / 15 + 1) if v >= MIN_SIDED else None
                        six_v += v; six_b += base
                    e[f"{w}|six|surge"] = (six_v / (o1 - o0)) / (six_b / 15 + 1) if six_v >= MIN_SIX else None
                ev.append(e)
    return d, ev


def fetch_aux(days):
    """stock 1-min volume (prior 20 sessions included) and API GEX for days the lake layer lacks. Cached."""
    sys.path.insert(0, HERE)
    import positioning_map as P
    os.makedirs(CACHE, exist_ok=True)
    fv, fg = os.path.join(CACHE, "volume.json"), os.path.join(CACHE, "gex_api.json")
    vol = json.load(open(fv)) if os.path.exists(fv) else {}
    gex = json.load(open(fg)) if os.path.exists(fg) else {}
    alld = lake_days("2024-07-01", "2026-10-30")
    need_v = [(d, t) for d in alld if d >= "2024-07-22" for t in TK if f"{d}|{t}" not in vol]
    lake_g = {os.path.basename(os.path.dirname(p))[5:] for p in glob.glob(os.path.join(LAKE, "silver", "spot-exposures-1m", "date=*", "SPY.parquet"))}
    need_g = [(d, t) for d in days if d not in lake_g for t in TK if f"{d}|{t}" not in gex]
    uw = P.UW(); num = lambda x: P.num(x) or 0.0
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
    def mod(ts):
        z = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY); return z.hour * 60 + z.minute
    print(f"fetch: volume {len(need_v)}, gex {len(need_g)}", flush=True)
    for i, (d, t) in enumerate(need_v):
        rows = uw.get(f"/stock/{t}/ohlc/1m", date=d, limit=2500).get("data") or []
        vol[f"{d}|{t}"] = {str(mod(r["start_time"])): num(r.get("volume")) for r in rows if 570 <= mod(r["start_time"]) < 960}
        if i % 100 == 99: json.dump(vol, open(fv, "w")); print("  volume", i + 1, flush=True)
    json.dump(vol, open(fv, "w"))
    for d, t in need_g:
        rows = uw.get(f"/stock/{t}/spot-exposures", date=d).get("data") or []
        gex[f"{d}|{t}"] = {str(mod(r["start_time"])): num(r.get("gamma_per_one_percent_move_oi")) for r in rows if r.get("start_time")}
    json.dump(gex, open(fg, "w"))
    return vol, gex


def gex_series(d, t, api):
    import polars as pl
    f = os.path.join(LAKE, "silver", "spot-exposures-1m", f"date={d}", f"{t}.parquet")
    if os.path.exists(f):
        x = pl.read_parquet(f, columns=["minute_et", "gamma_per_one_percent_move_oi"])
        return {me.hour * 60 + me.minute: g for me, g in x.iter_rows() if g is not None}
    return {int(m): g for m, g in (api.get(f"{d}|{t}") or {}).items()}


def build(days, tag):
    out = os.path.join(CACHE, f"events-{tag}"); os.makedirs(out, exist_ok=True)
    todo = [d for d in days if not os.path.exists(os.path.join(out, d + ".json"))]
    print(f"build {tag}: {len(days)} days, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for i, (d, ev) in enumerate(ex.map(build_day, todo)):
            json.dump(ev, open(os.path.join(out, d + ".json"), "w"))
            if i % 25 == 24: print(f"  {i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print(f"build {tag} done ({sum(1 for _ in glob.glob(os.path.join(out, '*.json')))} day files); no outcomes printed", flush=True)


# ------------------------------------------------------------------ scoring
def load(tag, days, vol, gex_api):
    ev = []
    for d in days:
        f = os.path.join(CACHE, f"events-{tag}", d + ".json")
        if os.path.exists(f): ev += json.load(open(f))
    alld = lake_days("2024-07-01", "2026-10-30")
    cum = {}
    for k, v in vol.items():
        a = np.zeros(390)
        for m, x in v.items(): a[int(m) - 570] = x
        cum[k] = np.cumsum(a)
    gx = {}
    for e in ev:
        key = f"{e['d']}|{e['t']}"
        if key not in gx: gx[key] = gex_series(e["d"], e["t"], gex_api)
        g = gx[key]; m = e["m0"] - 1
        while m >= 560 and m not in g: m -= 1
        e["gex"] = g.get(m)
        i = alld.index(e["d"]); prev = [f"{x}|{e['t']}" for x in alld[max(0, i - 20):i]]
        prev = [p for p in prev if p in cum]
        c = cum.get(key)
        if c is not None and len(prev) >= 15:
            j = e["m0"] - 570 - 1
            base = np.median([cum[p][j] for p in prev]); e["rvol"] = c[j] / base if base > 0 else None
            based = np.median([cum[p][-1] for p in prev]); e["rvol_day"] = c[-1] / based if based > 0 else None
        else:
            e["rvol"] = e["rvol_day"] = None
    return ev


def rk(v):
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, float)) / len(v)


def partial(q, key, ctrl, y):
    if len(q) < 40: return float("nan")
    yy = rk([r["res"] == y for r in q]); x = rk([r[key] for r in q])
    if np.std(yy) == 0 or np.std(x) == 0: return float("nan")
    X = np.column_stack([np.ones(len(q))] + [rk([r[c] for r in q]) for c in ctrl])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def boot(q, fn, n=N_BOOT, seed=1):
    by = defaultdict(list)
    for r in q: by[r["d"]].append(r)
    ds = sorted(by); random.seed(seed); o = []
    for _ in range(n):
        s = [r for d in random.choices(ds, k=len(ds)) for r in by[d]]
        v = fn(s)
        if v == v: o.append(v)
    o = np.sort(o)
    return o


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * m / rank); q[i] = prev
    return q


def rate(rs, o): return float(np.mean([r["res"] == o for r in rs]) * 100) if rs else float("nan")


def thirds(q, key):
    cuts = {}
    for t in TK:
        a = sorted(r[key] for r in q if r["t"] == t)
        if len(a) >= 15: cuts[t] = (a[len(a) // 3], a[2 * len(a) // 3])
    lo = [r for r in q if r["t"] in cuts and r[key] <= cuts[r["t"]][0]]
    hi = [r for r in q if r["t"] in cuts and r[key] > cuts[r["t"]][1]]
    return lo, hi


PRIMARY = [("P1", "SPY", "pre5|six|surge", "stall", -1, ["sp5"], False),
           ("P2", "QQQ", "pre5|six|surge", "stall", -1, ["sp5"], False),
           ("P3", "IWM", "pre5|six|surge", "stall", -1, ["sp5"], False),
           ("P4", "SPY", "pre2|push ATM|agg", "reverse", -1, ["sp5"], False),
           ("P5", "IWM", "pre2|fade ATM|agg", "reverse", +1, ["sp5"], False),
           ("P6", "ALL", "pre5|fade ATM|agg", "reverse", +1, ["sp5"], False),
           ("P7", "ALL", "at|push OTM|agg", "reverse", -1, ["pos2", "sp2", "sp5"], True)]


def subset(ev, tk, key, at):
    q = [r for r in ev if (tk == "ALL" or r["t"] == tk) and r.get(key) is not None]
    return [r for r in q if r["unres2"]] if at else q


def report(ev, title, out):
    def p(s=""): print(s); out.append(s)
    days = sorted({r["d"] for r in ev}); mid = days[len(days) // 2]
    p(f"{title}: {len(days)} sessions {days[0]}..{days[-1]}, {len(ev)} touches")
    for t in TK:
        q = [r for r in ev if r["t"] == t]
        p(f"  {t}: n={len(q)} through {rate(q,'through'):.1f} stall {rate(q,'stall'):.1f} reverse {rate(q,'reverse'):.1f}")
    p("\nPRIMARY (partial rank corr with the outcome given the controls; 95% day-bootstrap CI; halves = first/second half of the dates)")
    rows, ps = [], []
    for pid, tk, key, y, sign, ctrl, at in PRIMARY:
        q = subset(ev, tk, key, at)
        v = partial(q, key, ctrl, y)
        b = boot(q, lambda s: partial(s, key, ctrl, y))
        lo_, hi_ = b[int(.025 * len(b))], b[int(.975 * len(b))]
        pv = max(2 * min(np.mean(b <= 0), np.mean(b >= 0)), 1 / len(b))
        h1 = partial([r for r in q if r["d"] < mid], key, ctrl, y); h2 = partial([r for r in q if r["d"] >= mid], key, ctrl, y)
        lo3, hi3 = thirds(q, key)
        rows.append((pid, tk, key, y, sign, v, lo_, hi_, h1, h2, len(q), rate(lo3, y), rate(hi3, y))); ps.append(pv)
    qs = bh(ps)
    for (pid, tk, key, y, sign, v, lo_, hi_, h1, h2, n, rl, rh), qv in zip(rows, qs):
        ok_ci = (hi_ < 0) if sign < 0 else (lo_ > 0)
        ok_h = (h1 * sign > 0) and (h2 * sign > 0)
        verdict = "PASS" if ok_ci and qv < .10 and ok_h else "FAIL"
        p(f"  {pid} {tk:3s} {key:20s} -> {y:7s} predicted {'-' if sign<0 else '+'} | {v:+.3f} [{lo_:+.3f},{hi_:+.3f}] q={qv:.3f} | halves {h1:+.3f} / {h2:+.3f}"
          f" | {y} low third {rl:.1f}% high third {rh:.1f}% | n={n} | {verdict}")
    p("\nINFORMATION ONLY")
    for nm, tk, key, y, ctrl, at in (("H1 (locked test's idea)", "ALL", "pre5|push ATM|agg", "reverse", ["sp5"], False),):
        q = subset(ev, tk, key, at); b = boot(q, lambda s: partial(s, key, ctrl, y))
        p(f"  {nm}: {partial(q,key,ctrl,y):+.3f} [{b[int(.025*len(b))]:+.3f},{b[int(.975*len(b))]:+.3f}] n={len(q)}")
    for tk in TK:
        line = []
        for key, y, ctrl, at in (("pre2|six|surge", "stall", ["sp5"], False), ("pre5|six|surge", "stall", ["sp5"], False),
                                 ("pre2|push ATM|agg", "reverse", ["sp5"], False), ("pre5|push ATM|agg", "reverse", ["sp5"], False),
                                 ("pre2|fade ATM|agg", "reverse", ["sp5"], False), ("pre5|fade ATM|agg", "reverse", ["sp5"], False),
                                 ("at|push OTM|agg", "reverse", ["pos2", "sp2", "sp5"], True)):
            line.append(f"{key}->{y[:3]} {partial(subset(ev, tk, key, at), key, ctrl, y):+.3f}")
        p(f"  {tk}: " + " | ".join(line))
    p("\nREGIME SPLITS (secondary). GEX: + = the effect is larger in the predicted direction in NEGATIVE gamma (dealer-hedging prediction)")
    g_ev = [r for r in ev if r.get("gex") is not None]
    p(f"  touches with a GEX reading {len(g_ev)} of {len(ev)}; negative-gamma share {np.mean([r['gex']<0 for r in g_ev]):.0%}")
    for t in TK + ("ALL",):
        q = [r for r in g_ev if t == "ALL" or r["t"] == t]
        neg, pos = [r for r in q if r["gex"] < 0], [r for r in q if r["gex"] >= 0]
        bd = boot(q, lambda s: rate([r for r in s if r["gex"] < 0], "through") - rate([r for r in s if r["gex"] >= 0], "through"), n=500)
        br = boot(q, lambda s: rate([r for r in s if r["gex"] < 0], "reverse") - rate([r for r in s if r["gex"] >= 0], "reverse"), n=500)
        p(f"  base rates {t}: negative gamma n={len(neg)} thr {rate(neg,'through'):.1f} stall {rate(neg,'stall'):.1f} rev {rate(neg,'reverse'):.1f} | positive n={len(pos)} thr {rate(pos,'through'):.1f} stall {rate(pos,'stall'):.1f} rev {rate(pos,'reverse'):.1f}"
          f" | through neg-pos [{bd[int(.025*len(bd))]:+.1f},{bd[int(.975*len(bd))]:+.1f}] reverse neg-pos [{br[int(.025*len(br))]:+.1f},{br[int(.975*len(br))]:+.1f}]")
    sp_rows, sp_ps = [], []
    for pid, tk, key, y, sign, ctrl, at in PRIMARY:
        q = [r for r in subset(ev, tk, key, at)]
        for lab, A, B in (("GEX neg-pos", lambda r: r.get("gex") is not None and r["gex"] < 0, lambda r: r.get("gex") is not None and r["gex"] >= 0),
                          ("RVOL hi-lo", "rvol", None), ("dayVOL hi-lo", "rvol_day", None)):
            if B is None:
                qq = [r for r in q if r.get(A) is not None]
                cut = {t: np.percentile([r[A] for r in qq if r["t"] == t], [33.3, 66.7]) for t in TK if any(r["t"] == t for r in qq)}
                fa = lambda r, A=A, cut=cut: r[A] > cut[r["t"]][1]; fb = lambda r, A=A, cut=cut: r[A] <= cut[r["t"]][0]
            else:
                qq, fa, fb = q, A, B
            diff = lambda s: sign * (partial([r for r in s if fa(r)], key, ctrl, y) - partial([r for r in s if fb(r)], key, ctrl, y))
            va, vb = partial([r for r in qq if fa(r)], key, ctrl, y), partial([r for r in qq if fb(r)], key, ctrl, y)
            b = boot(qq, diff, n=500, seed=3)
            pv = max(2 * min(np.mean(b <= 0), np.mean(b >= 0)), 1 / max(len(b), 1)) if len(b) else 1.0
            sp_rows.append((pid, tk, key, lab, va, vb, sign * (va - vb), b[int(.025 * len(b))] if len(b) else np.nan, b[int(.975 * len(b))] if len(b) else np.nan,
                            sum(map(fa, qq)), sum(map(fb, qq)))); sp_ps.append(pv)
    sq = bh(sp_ps)
    for (pid, tk, key, lab, va, vb, dv, lo_, hi_, na, nb), qv in zip(sp_rows, sq):
        p(f"  {pid} {tk:3s} {key:20s} {lab:13s}: {va:+.3f} (n={na}) vs {vb:+.3f} (n={nb}) | difference toward the prediction {dv:+.3f} [{lo_:+.3f},{hi_:+.3f}] q={qv:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--replay-discovery", action="store_true"); ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    hold, rep = lake_days(*HOLD), lake_days(*REPLAY)
    hold = [d for d in hold if glob.glob(os.path.join(LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))]
    if a.build:
        fetch_aux(hold + rep)
        build(rep, "replay"); build(hold, "holdout")
    if a.replay_discovery:
        vol, gex = fetch_aux(rep)
        out = []; report(load("replay", rep, vol, gex), "REPLAY (discovery days, code check only)", out)
    if a.run:
        res = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(res):
            sys.exit(f"already scored once: {res}. The holdout is spent; read that file.")
        vol, gex = fetch_aux(hold)
        out = [f"check_sixpack_lake.py --run at {datetime.now():%Y-%m-%d %H:%M}"]
        report(load("holdout", hold, vol, gex), "HOLDOUT", out)
        open(res, "w", encoding="utf-8").write("\n".join(out) + "\n")
        print(f"\nwritten {res}")


if __name__ == "__main__":
    main()
