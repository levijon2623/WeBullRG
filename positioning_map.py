# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "numpy", "python-dotenv"]
# ///
"""
positioning_map.py
==================
Daily POSITIONING MAP (user, 2026-10-24): where options positions sit and how
they changed overnight, by strike x expiry, for S&P (SPY + SPX/SPXW), QQQ and
IWM. It draws the morning read we had been doing by hand (where protection
was placed, rolled or closed, and by what kind of order) as one picture for
positioning_map.html.

CONTEXT, NOT A SIGNAL. Peak gamma does not pull price (check_peak_gamma),
charm / vanna hedge flows do not call direction (check_currents), and sticky
UOA does not predict 1-5 days (check_uoa_events). Gamma sizes moves.

WHAT IT SHOWS, PER MARKET
  cells      strike (in ETF units; SPX strikes / the SPX:SPY ratio) x expiry
             0-45 days: open interest, the OVERNIGHT CHANGE and the volume
             that made it (bought at the ask / sold at the bid / mid,
             multi-leg share, premium), standing dealer gamma, IV - RV.
  flip line  CUMULATIVE zero-gamma level per expiry: the price where dealer
             gamma of everything expiring up to that date turns from + to -.
             Usual convention (dealers long calls, short puts), like the
             weekly GEX page.
  dealer d-gamma  the overnight change signed by HOW it was opened: bought at
             the ask -> dealer short, sold at the bid -> dealer long, mid /
             mixed -> flagged uncertain. Better than the blanket convention for
             hedges, covered calls and spreads.
  IVRP       per expiry ATM IV - 21-day realized vol (vol points), and per
             contract IV - RV: positions placed left (cheap) / right (rich) of 0.
  packages   matched legs (similar size, multi-leg): ROLL (one closed, one
             opened), SPREAD (opened together, opposite sides), PACKAGE -- an
             inference, labelled.
  prints     the biggest prints (time, size, price, side, multi-leg) behind
             the largest overnight changes.

TIMING (box clock is ET; UW open interest turns over ~06:15-06:30 ET)
  pre-open   from 06:35, every 10 min until UW's oi-change carries today's
             date (a holiday never does); spot = pre-market price.
  at open    rebuilt at ~09:32 with the opening price and live IVs.
  intraday   every 5 min (0-14 DTE) / 30 min (15-45 DTE), 09:35-16:05: today's
             volume by contract and side -> {market}_today.json overlay;
             tomorrow's open interest says what stuck. Each pass also runs
             intraday_map.trace_pass (the TRACE-style time x price map) on the
             same pull: {market}_trace.json + replay frames.

OUTPUT (live/, gitignored)
  live/posmap/{SP,QQQ,IWM}.json         latest map
  live/posmap/{SP,QQQ,IWM}_today.json   intraday overlay
  live/posmap/archive/DATE/{..}.json    each morning's map

    python positioning_map.py                       # service loop (cleanbot-posmap)
    python positioning_map.py --build [--date D]    # one build now (D = as-of date)
    python positioning_map.py --today               # one intraday overlay pass
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import requests
from dotenv import load_dotenv
from statistics import NormalDist

# no scipy: the box's bot venv doesn't have it, and installing it could move numpy under the running bot
_erf = np.vectorize(math.erf, otypes=[float])


def ndtr(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

NY = ZoneInfo("America/New_York")
API = "https://api.unusualwhales.com/api"
OUT = os.path.join("live", "posmap")
MAX_DAYS, CELL_BAND, FLIP_BAND, FLIP_STEP = 45, 0.10, 0.10, 0.001
MARKETS = {                         # key: (label, ETF, [(UW ticker, is_index)])
    "SP": ("S&P", "SPY", [("SPY", False), ("SPX", True)]),
    "QQQ": ("QQQ", "QQQ", [("QQQ", False)]),
    "IWM": ("IWM", "IWM", [("IWM", False)]),
}
# UW's SPX chain returns both roots (SPXW weeklies, SPX AM monthlies); its oi-change splits them by ticker
OI_TICKERS = {"SPX": ("SPX", "SPXW")}
AM_SETTLED = {"SPX"}                # SPX root (monthlies) settle at the open; SPXW at the close
SIDE_MIN = 0.60                     # share of side-known volume to call it bought / sold
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
PRE_START, PRE_END, OPEN_BUILD = 6 * 60 + 35, 9 * 60 + 25, 9 * 60 + 32
TODAY_START, TODAY_END = 9 * 60 + 35, 16 * 60 + 5


# ------------------------------------------------------------------ UW
class UW:
    def __init__(self):
        key = os.environ.get("UW_API_KEY", "").strip()
        if not key:
            raise SystemExit("UW_API_KEY not set")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.calls = 0

    def get(self, path, **params):
        for attempt in range(3):
            try:
                self.calls += 1
                r = self.s.get(API + path, params={k: str(v) for k, v in params.items()}, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 + 3 * attempt)
                    continue
                return {}
            except requests.RequestException:
                time.sleep(2 + 3 * attempt)
        return {}


def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def parse_occ(sym):
    m = OCC.match(sym or "")
    if not m:
        return None
    root, ymd, cp, k = m.groups()
    return root, dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])), cp, int(k) / 1000.0


# ------------------------------------------------------------------ prices
def daily_closes(uw, tk):
    rows = uw.get(f"/stock/{tk}/ohlc/1d").get("data") or []
    out = {}
    for x in rows:
        if x.get("market_time") not in (None, "r"):
            continue
        d = str(x.get("date") or x.get("start_time") or "")[:10]
        c = num(x.get("close"))
        if d and c:
            out[d] = c
    return dict(sorted(out.items()))


def spot_now(uw, tk, day, closes):
    """latest 1m price today (pre-market bars count), else the last close before today."""
    rows = uw.get(f"/stock/{tk}/ohlc/1m", date=day.isoformat(), limit=2500).get("data") or []
    best = None
    for x in rows:
        t, c = x.get("start_time") or "", num(x.get("close"))
        if c and (best is None or t > best[0]):
            best = (t, c, x.get("market_time"))
    prev = [c for d, c in closes.items() if d < day.isoformat()]
    if best:
        return best[1], best[2], (prev[-1] if prev else None)
    return (prev[-1] if prev else None), "close", (prev[-1] if prev else None)


def rv21(closes, day):
    c = np.array([v for d, v in closes.items() if d < day.isoformat()][-22:], float)
    if len(c) < 15:
        return None
    return float(np.std(np.diff(np.log(c)), ddof=1) * math.sqrt(252) * 100)


# ------------------------------------------------------------------ greeks
def T_years(exp, root, now):
    """time to expiry in years from `now` (aware ET datetime)."""
    hh, mm = (9, 30) if root in AM_SETTLED else (16, 0)
    end = dt.datetime(exp.year, exp.month, exp.day, hh, mm, tzinfo=NY)
    return max((end - now).total_seconds(), 300.0) / (365.0 * 86400.0)


def gamma(S, K, T, iv):
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sd
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * sd)


def delta_call(S, K, T, iv):
    sd = iv * np.sqrt(T)
    return ndtr((np.log(S / K) + 0.5 * iv * iv * T) / sd)


# ------------------------------------------------------------------ chain
def expiries(uw, tk, day):
    syms = uw.get(f"/stock/{tk}/option-chains").get("data") or []
    out = set()
    for s in syms:
        p = parse_occ(s)
        if p and day <= p[1] <= day + dt.timedelta(days=MAX_DAYS):
            out.add(p[1])
    return sorted(out)


def chain(uw, tk, exps):
    rows = {}
    for e in exps:
        for typ in ("call", "put"):
            for x in uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat(), option_type=typ).get("data") or []:
                p = parse_occ(x.get("option_symbol"))
                if p and p[1] == e:
                    rows[x["option_symbol"]] = x
    return rows


def oi_changes(uw, tks, day, max_pages=80):
    """the change in open interest from the last session's trading: {sym: row}, and the sessions it covers.
    UW lists the biggest INCREASES first and the biggest CLOSES last (SPXW 10-23: ~19.5k rows, the 7750P's
    -16.9k at rank ~23k), and zero-change rows tie in an unstable order -- so read every page to the end."""
    out, dates = {}, set()
    for tk in tks:
        for page in range(max_pages):
            d = uw.get(f"/stock/{tk}/oi-change", date=day.isoformat(), limit=500, page=page).get("data") or []
            if not d:
                break
            for x in d:
                if x.get("curr_date") == day.isoformat() and x["option_symbol"] not in out:
                    out[x["option_symbol"]] = x
                    dates.add(x.get("last_date"))
    return out, sorted(x for x in dates if x)


def implied_spot(rows, day):
    """UW has no index quotes: back the index level out of its own chain -- for near-the-money calls of the
    nearest 1+ DTE expiry, S = K * exp(d1 * iv * sqrt(T) - iv^2 T / 2) with d1 = N^-1(delta) (UW's greeks)."""
    ndtri = NormalDist().inv_cdf
    by = {}
    for s, x in rows.items():
        p = parse_occ(s)
        if not p or p[2] != "C" or p[1] <= day:
            continue
        dl, iv = num(x.get("delta")), num(x.get("implied_volatility"))
        if dl and iv and 0.3 <= dl <= 0.7 and 0.02 < iv < 2:
            by.setdefault(p[1], []).append((p[3], dl, iv))
    for e in sorted(by):
        if len(by[e]) < 3:
            continue
        T = max((e - day).days, 1) / 365.0
        est = [k * math.exp(float(ndtri(dl)) * iv * math.sqrt(T) - 0.5 * iv * iv * T) for k, dl, iv in by[e]]
        return float(np.median(est)), e
    return None, None


# ------------------------------------------------------------------ build
def side_of(ask, bid, mid):
    known = (ask or 0) + (bid or 0) + (mid or 0)
    if known <= 0:
        return 0, 0.0, 0.0
    a, b = (ask or 0) / known, (bid or 0) / known
    return (1 if a >= SIDE_MIN else -1 if b >= SIDE_MIN else 0), round(a, 3), round(b, 3)


def build_market(uw, key, day, now, phase):
    label, etf, parts = MARKETS[key]
    closes = daily_closes(uw, etf)
    S, src, prev_close = spot_now(uw, etf, day, closes)
    if not S:
        return None, "no spot"
    rv = rv21(closes, day)
    contracts, oi_dates, ratio, ratio_src = [], set(), 1.0, "etf"
    for tk, is_index in parts:
        exps = expiries(uw, tk, day)
        rows = chain(uw, tk, exps)
        chg, dates = oi_changes(uw, OI_TICKERS.get(tk, (tk,)), day)
        if not chg:
            return None, f"{tk}: no oi-change for {day} yet"
        oi_dates.update(dates)
        r_tk, s_tk = 1.0, S
        if is_index:
            # the chain's greeks are as of the last session's close -> divide by the ETF's last close
            F, fe = implied_spot(rows, day)
            base = prev_close if phase == "pre" else S       # pre-open: last close's greeks; at open: live
            ok = F and base and 9.0 < F / base < 11.0
            r_tk = F / base if ok else 10.04
            ratio, ratio_src = round(r_tk, 4), (f"chain-implied {fe}" if ok else "default 10.04")
            s_tk = S * r_tk
        for sym, x in rows.items():
            p = parse_occ(sym)
            if not p:
                continue
            root, exp, cp, k = p
            if root in AM_SETTLED and exp == day and now.hour * 60 + now.minute >= 570:
                continue
            c = chg.get(sym)
            oi_now = int(c["curr_oi"]) if c else int(x.get("open_interest") or 0)
            oi_prev = int(c["last_oi"]) if c else oi_now
            iv = num(x.get("implied_volatility"))
            if (oi_now <= 0 and oi_prev <= 0) or not iv or iv <= 0.01 or iv > 4:
                continue
            if abs(k / s_tk - 1) > FLIP_BAND + 0.05:
                continue
            side, ask_sh, bid_sh = (0, 0.0, 0.0)
            vol = mleg = prem = 0.0
            if c:
                side, ask_sh, bid_sh = side_of(num(c.get("prev_ask_volume")), num(c.get("prev_bid_volume")),
                                               num(c.get("prev_mid_volume")))
                vol = num(c.get("volume")) or 0.0
                mleg = (num(c.get("prev_multi_leg_volume")) or 0.0) / vol if vol else 0.0
                prem = num(c.get("prev_total_premium")) or 0.0
            contracts.append(dict(sym=sym, root=root, tk=tk, exp=exp, cp=cp, k=k, ku=k / r_tk, r=r_tk, S=s_tk,
                                  oi=oi_now, oi0=oi_prev, d=oi_now - oi_prev, iv=iv, side=side, ask=ask_sh,
                                  bid=bid_sh, vol=vol, mleg=round(mleg, 3), prem=prem))
    if not contracts:
        return None, "no contracts"

    # ---- greeks at spot
    n = len(contracts)
    K = np.array([c["k"] for c in contracts]); Sx = np.array([c["S"] for c in contracts])
    IV = np.array([c["iv"] for c in contracts])
    T = np.array([T_years(c["exp"], c["root"], now) for c in contracts])
    R = np.array([c["r"] for c in contracts])
    sgn = np.array([1.0 if c["cp"] == "C" else -1.0 for c in contracts])
    OI = np.array([c["oi"] for c in contracts], float); D = np.array([c["d"] for c in contracts], float)
    SIDE = np.array([c["side"] for c in contracts], float)
    g1 = gamma(Sx, K, T, IV) * Sx * Sx * 0.01 * 100.0          # $ dealer delta per 1% per contract
    g_stand = sgn * OI * g1
    # overnight change, dealer-signed by how it was opened (customer bought -> dealer short gamma)
    g_chg = np.where(SIDE != 0, -SIDE * np.abs(D), sgn * D) * g1
    dc = np.abs(delta_call(Sx, K, T, IV) - (sgn < 0))           # |delta|
    for i, c in enumerate(contracts):
        c["g"] = float(g_stand[i]); c["gd"] = float(g_chg[i]); c["dl"] = round(float(dc[i]), 3)
        c["ivrp"] = round(c["iv"] * 100 - rv, 2) if rv else None

    # ---- expiries: ATM IV, IVRP, cumulative flip
    exps = sorted({c["exp"] for c in contracts})
    grid = S * (1 + np.arange(-FLIP_BAND, FLIP_BAND + 1e-9, FLIP_STEP))
    order = np.argsort([exps.index(c["exp"]) for c in contracts], kind="stable")
    cum = np.zeros_like(grid)
    ex_rows = []
    by_e = {}
    for i in order:
        by_e.setdefault(contracts[i]["exp"], []).append(i)
    for e in exps:
        idx = np.array(by_e[e])
        Sg = grid[None, :] * R[idx, None]
        G = sgn[idx, None] * OI[idx, None] * gamma(Sg, K[idx, None], T[idx, None], IV[idx, None]) * Sg * Sg * 0.01 * 100
        cum = cum + G.sum(0)
        flip = None
        s_ = np.sign(cum)
        cross = np.flatnonzero(s_[:-1] * s_[1:] < 0)
        if len(cross):
            j = cross[np.argmin(np.abs(grid[cross] - S))]
            x0, x1, y0, y1 = grid[j], grid[j + 1], cum[j], cum[j + 1]
            flip = round(float(x0 - y0 * (x1 - x0) / (y1 - y0)), 2)
        atm = [contracts[i]["iv"] for i in idx if 0.40 <= dc[i] <= 0.60]
        atm_iv = float(np.mean(atm)) * 100 if atm else None
        ex_rows.append(dict(e=e.isoformat(), dte=(e - day).days, atm_iv=round(atm_iv, 2) if atm_iv else None,
                            ivrp=round(atm_iv - rv, 2) if atm_iv and rv else None, flip=flip,
                            cum_g_spot=float(np.interp(S, grid, cum)),
                            g_spot=float(g_stand[idx].sum()), gd_spot=float(g_chg[idx].sum()),
                            d_oi=int(np.sum(np.abs(D[idx]) * R[idx]))))

    # ---- cells (ETF strike buckets within CELL_BAND)
    cells = {}
    for c in contracts:
        ku = round(c["ku"])
        if abs(ku / S - 1) > CELL_BAND:
            continue
        key2 = (exps.index(c["exp"]), ku, 1 if c["root"] in ("SPX", "SPXW") else 0)
        z = cells.setdefault(key2, dict(oc=0.0, op=0.0, dc=0.0, dp=0.0, g=0.0, gd=0.0, vc=0.0, vp=0.0,
                                        bc=0.0, sc=0.0, bp=0.0, spv=0.0, ml=0.0, pr=0.0, ivw=0.0, ivn=0.0, n=[]))
        r = c["r"]
        if c["cp"] == "C":
            z["oc"] += c["oi"] * r; z["dc"] += c["d"] * r; z["vc"] += c["vol"] * r
            z["bc"] += c["vol"] * c["ask"] * r; z["sc"] += c["vol"] * c["bid"] * r
        else:
            z["op"] += c["oi"] * r; z["dp"] += c["d"] * r; z["vp"] += c["vol"] * r
            z["bp"] += c["vol"] * c["ask"] * r; z["spv"] += c["vol"] * c["bid"] * r
        z["g"] += c["g"]; z["gd"] += c["gd"]; z["ml"] += c["vol"] * c["mleg"] * r; z["pr"] += c["prem"]
        if c["ivrp"] is not None and c["oi"] > 0:
            z["ivw"] += c["ivrp"] * c["oi"] * r; z["ivn"] += c["oi"] * r
        if c["d"] != 0 or c["oi"] > 0:
            z["n"].append(c["sym"])
    cell_rows = []
    for (ei, ku, grp), z in sorted(cells.items()):
        v = z["vc"] + z["vp"]
        cell_rows.append([ei, ku, grp, round(z["oc"]), round(z["op"]), round(z["dc"]), round(z["dp"]),
                          round(z["g"]), round(z["gd"]),
                          round(z["bc"] / z["vc"], 2) if z["vc"] else None, round(z["sc"] / z["vc"], 2) if z["vc"] else None,
                          round(z["bp"] / z["vp"], 2) if z["vp"] else None, round(z["spv"] / z["vp"], 2) if z["vp"] else None,
                          round(z["ml"] / v, 2) if v else None, round(z["pr"]),
                          round(z["ivw"] / z["ivn"], 2) if z["ivn"] else None, round(v)])

    # ---- packages: greedy pairing of the big multi-leg changes. Same underlying contract family only;
    #      ROLL = one closed, one opened, same right; SPREAD / PACKAGE = opened together in one expiry.
    def fam(c):
        return "SPX" if c["root"] in ("SPX", "SPXW") else c["root"]

    def kind_of(a, b):
        if np.sign(a["d"]) != np.sign(b["d"]):
            closed, opened = (a, b) if a["d"] < 0 else (b, a)
            return "ROLL" if a["cp"] == b["cp"] and opened["exp"] >= closed["exp"] else None
        if a["exp"] != b["exp"]:
            return None
        return "SPREAD" if a["side"] and b["side"] and a["side"] != b["side"] else "PACKAGE"

    big = sorted((c for c in contracts if abs(c["d"]) * c["r"] >= 1000), key=lambda c: -abs(c["d"]) * c["r"] * c["S"])[:80]
    used, packages = set(), []
    for i, a in enumerate(big):
        if a["sym"] in used or a["mleg"] < 0.4:
            continue
        best = None
        for b in big[i + 1:]:
            if b["sym"] in used or b["mleg"] < 0.4 or fam(b) != fam(a) or kind_of(a, b) is None:
                continue
            sa, sb = abs(a["d"]) * a["r"], abs(b["d"]) * b["r"]
            if min(sa, sb) / max(sa, sb) < 0.85:
                continue
            if best is None or abs(sa - sb) < best[1]:
                best = (b, abs(sa - sb))
        if best:
            b = best[0]
            used |= {a["sym"], b["sym"]}
            packages.append(dict(kind=kind_of(a, b), legs=[a["sym"], b["sym"]],
                                 size=round(min(abs(a["d"]) * a["r"], abs(b["d"]) * b["r"]))))

    # ---- prints behind the biggest changes (the session that changed the OI)
    prints = {}
    when = sorted(oi_dates)[-1] if oi_dates else None
    if when:
        # the biggest changes PER FAMILY (SPX notional would crowd SPY out of a pooled top list)
        top = []
        for f in sorted({fam(c) for c in contracts}):
            top += sorted((c for c in contracts if fam(c) == f), key=lambda c: -abs(c["d"]) * c["r"] * c["S"])[:8]
        for c in top:
            fl = uw.get(f"/option-contract/{c['sym']}/flow", date=when).get("data") or []
            top = sorted(fl, key=lambda x: -(num(x.get("size")) or 0))[:6]
            prints[c["sym"]] = [[(x.get("executed_at") or "")[:19], int(num(x.get("size")) or 0), num(x.get("price")),
                                 round(num(x.get("premium")) or 0),
                                 ("ask" if "ask_side" in (x.get("tags") or []) else "bid" if "bid_side" in (x.get("tags") or []) else "mid"),
                                 1 if (x.get("upstream_condition_detail") or "").startswith("ml") or (num(x.get("multi_vol")) or 0) > 0 else 0,
                                 num(x.get("underlying_price"))] for x in top]

    detail = {}
    for c in contracts:
        if c["d"] != 0 or c["sym"] in prints:
            detail[c["sym"]] = [c["exp"].isoformat(), c["cp"], c["k"], round(c["ku"], 2), c["oi"], c["d"], c["side"],
                                c["ask"], c["bid"], round(c["vol"]), c["mleg"], round(c["prem"]), c["ivrp"],
                                round(c["gd"]), round(c["g"]), c["root"]]
    pos_c = [[c["sym"], c["root"], c["exp"].isoformat(), c["cp"], c["k"], round(c["r"], 5), c["oi"], round(c["iv"], 4)]
             for c in contracts if (c["exp"] - day).days <= 7 and c["oi"] > 0 and abs(c["ku"] / S - 1) <= 0.08]
    cell_syms = {}
    for (ei, ku, grp), z in cells.items():
        cell_syms.setdefault(f"{ei}|{ku}", []).extend(z["n"])
    doc = dict(market=key, label=label, etf=etf, date=day.isoformat(), phase=phase,
               built_et=now.strftime("%Y-%m-%dT%H:%M:%S"), spot=round(S, 2), spot_src=src, prev_close=prev_close,
               ratio=ratio, ratio_src=ratio_src, rv21=round(rv, 2) if rv else None,
               oi_change_session=when, expiries=ex_rows,
               cell_cols=["ei", "k", "grp", "oi_c", "oi_p", "d_c", "d_p", "g", "gd", "c_ask", "c_bid", "p_ask", "p_bid",
                          "mleg", "prem", "ivrp", "vol"],
               cells=cell_rows, cell_syms=cell_syms,
               detail_cols=["exp", "cp", "k", "k_etf", "oi", "d_oi", "side", "ask", "bid", "vol", "mleg", "prem",
                            "ivrp", "dealer_dg", "dealer_g", "root"],
               detail=detail, packages=packages, prints=prints, uw_calls=uw.calls,
               _pos=dict(date=day.isoformat(), prev_close=prev_close, spot=round(S, 3), ratio=ratio, c=pos_c))
    return doc, None


def write(name, doc, archive_day=None):
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(OUT, name))
    if archive_day:
        a = os.path.join(OUT, "archive", archive_day)
        os.makedirs(a, exist_ok=True)
        with open(os.path.join(a, name), "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))


def built(day, phase):
    """did this phase's build already run today? (its archive holds every market) -- a restart mid-session
    must pick up where the service was, not wait for a pre-open window that has passed"""
    a = os.path.join(OUT, "archive", day.isoformat() + ("" if phase == "pre" else "_open"))
    return all(os.path.exists(os.path.join(a, f"{key}.json")) for key in MARKETS)


def build_all(uw, day, now, phase):
    ok = True
    for key in MARKETS:
        doc, err = build_market(uw, key, day, now, phase)
        if doc is None:
            print(f"  🗺️ [POSMAP] {key} {phase}: {err}", flush=True)
            ok = False
            continue
        write(f"{key}_pos.json", doc.pop("_pos"))
        write(f"{key}.json", doc, day.isoformat() + ("" if phase == "pre" else "_open"))
        e0 = doc["expiries"][0] if doc["expiries"] else {}
        print(f"  🗺️ [POSMAP] {key} {phase}: spot {doc['spot']} ({doc['spot_src']}), {len(doc['cells'])} cells, "
              f"{len(doc['detail'])} changed, {len(doc['packages'])} packages, flip {e0.get('flip')} -> "
              f"{doc['expiries'][-1]['flip'] if doc['expiries'] else None}, OI change from {doc['oi_change_session']}"
              f"{' x' + str(doc['ratio']) if key == 'SP' else ''}  [{uw.calls} UW calls]", flush=True)
    return ok


# ------------------------------------------------------------------ intraday overlay
_last_slow = {}
_said = set()


def today_pass(uw, day, now):
    mod = now.hour * 60 + now.minute
    for key, (label, etf, parts) in MARKETS.items():
        path = os.path.join(OUT, f"{key}_today.json")
        try:
            prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        except ValueError:
            prev = {}
        if prev.get("date") != day.isoformat():
            prev = {"date": day.isoformat(), "rows": {}}
        # UW's chain rows show the LAST session's volume until they roll to today: only trust them once
        # the morning's changed contracts carry today's open interest (curr) rather than yesterday's
        try:
            morning = json.load(open(os.path.join(OUT, f"{key}.json"), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if morning.get("date") != day.isoformat():
            continue
        ref = {s: (v[4], v[4] - v[5]) for s, v in morning.get("detail", {}).items() if v[5]}
        slow = mod - _last_slow.get(key, -999) >= 30
        fresh, got, hit, miss = {}, 0, 0, 0
        for tk, is_index in parts:
            for e in expiries(uw, tk, day):
                if (e - day).days > 14 and not slow:
                    continue
                for typ in ("call", "put"):
                    for x in uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat(), option_type=typ).get("data") or []:
                        s = x["option_symbol"]
                        if s in ref:
                            oi = int(num(x.get("open_interest")) or 0)
                            hit += oi == ref[s][0]
                            miss += oi == ref[s][1]
                        v = num(x.get("volume")) or 0
                        if v:
                            fresh[s] = [int(v), int(num(x.get("ask_volume")) or 0), int(num(x.get("bid_volume")) or 0),
                                        int(num(x.get("multi_leg_volume")) or 0), round(num(x.get("total_premium")) or 0),
                                        num(x.get("implied_volatility"))]
        if hit + miss and hit < miss:
            prev.update(updated_et=now.strftime("%H:%M"), waiting="UW contract rows not rolled to today yet")
            write(f"{key}_today.json", prev)
            continue
        prev["rows"].update(fresh)
        prev.pop("waiting", None)
        if slow:
            _last_slow[key] = mod
        prev.update(updated_et=now.strftime("%H:%M"), n=len(prev["rows"]))
        write(f"{key}_today.json", prev)
        try:
            import intraday_map
            err = intraday_map.trace_pass(uw, key, etf, day, now, prev["rows"])
            if err and (key, err) not in _said:
                _said.add((key, err))
                print(f"  🗺️ [TRACE] {key}: {err}", flush=True)
        except Exception as e:                           # noqa: BLE001 -- never break the overlay
            print(f"  🗺️ [TRACE] {key} error {type(e).__name__}: {str(e)[:160]}", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--phase", default="pre")
    a = ap.parse_args()
    uw = UW()
    now = dt.datetime.now(NY)
    if a.build:
        day = dt.date.fromisoformat(a.date) if a.date else now.date()
        build_all(uw, day, now if not a.date else dt.datetime(day.year, day.month, day.day, 7, 35, tzinfo=NY), a.phase)
        return
    if a.today:
        today_pass(uw, now.date(), now)
        return
    print("  🗺️ [POSMAP] service up (pre-open from 06:35 ET, at-open 09:32, intraday 09:35-16:05)", flush=True)
    done = {}                                    # day -> {"pre", "open"}
    last_today = 0.0
    while True:
        now = dt.datetime.now(NY)
        day, mod = now.date(), now.hour * 60 + now.minute
        if day not in done:
            done[day] = {p for p in ("pre", "open") if built(day, p)}
        st = done[day]
        if now.weekday() < 5:
            try:
                if "pre" not in st and PRE_START <= mod < PRE_END:
                    uw.calls = 0
                    if build_all(uw, day, now, "pre"):
                        st.add("pre")
                    else:
                        time.sleep(600)
                        continue
                if "open" not in st and mod >= OPEN_BUILD and "pre" in st and mod < TODAY_END:
                    uw.calls = 0
                    if build_all(uw, day, now, "open"):
                        st.add("open")
                if "pre" in st and TODAY_START <= mod <= TODAY_END and time.time() - last_today >= 300:
                    today_pass(uw, day, now)
                    last_today = time.time()
            except Exception as e:                       # noqa: BLE001 -- keep the service alive
                print(f"  🗺️ [POSMAP] error {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
