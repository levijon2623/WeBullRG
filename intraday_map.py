"""
intraday_map.py
===============
INTRADAY DEALER MAP, TRACE-style (user, 2026-10-25): time of day x price,
for S&P (SPY + SPX/SPXW, SPY units), QQQ and IWM. Called by positioning_map.py
on every intraday pass (every 5 min, 09:35-16:05 ET) -- it reuses that pass's
UW pull (each contract's volume, ask / bid split and live IV), so it adds one
minute-bar call per market and no new service.

LENSES (each recomputed every pass for the REST of the day; the past columns
are what each earlier pass computed for its own moment, so the map's history
is never rewritten with hindsight)
  gamma           dealer $ delta change per 1% move at (price, time):
                  + damps moves (dealers sell rallies / buy dips), - amplifies.
                  The one lens with support in our tests: S&P moved 39 bp on
                  +gamma days vs 68 bp on -gamma days (check_currents).
  delta pressure  $ dealers must trade if price moved from where it is now to
                  that level at that time: + buy (support), - sell.
  charm           $ dealers must trade over the next 5 minutes from time
                  passing alone, at that price: + buy, - sell. UNPROVEN as a
                  direction call: check_currents found charm flow does not
                  predict the day's or the last hour's direction.
POSITIONS (toggle)
  oi    start-of-day open interest (positioning_map's morning build), usual
        dealer convention: dealers long the calls, short the puts.
  adj   oi + today's volume signed by side: bought at the ask -> dealer
        shorter, sold at the bid -> dealer longer. An ESTIMATE: volume does
        not say whether a trade opened or closed a position.
EXPIRIES (toggle)  0-1 DTE (where the intraday charm and gamma live) / 0-7 DTE.

Grid: price = the last close +-3% in 0.1% steps (fixed for the day so frames
line up), time = 09:30 .. 16:00 in 5-minute columns. Values are quantised to
+-100 of each grid's own scale (stored) to keep the files small.
Output: live/posmap/{SP,QQQ,IWM}_trace.json (latest) and
live/posmap/archive/DATE/{..}_trace.jsonl (every pass, for the replay slider).
CONTEXT, NOT A SIGNAL.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import os
import shutil

import numpy as np

OUT = os.path.join("live", "posmap")
COLS = list(range(570, 961, 5))                  # 09:30 .. 16:00
BAND, STEP = 0.03, 0.001
YEAR_MIN = 365.0 * 1440.0
LENSES = ("g", "dp", "ch")
COMBOS = [f"{lens}|{m}|{s}" for lens in LENSES for m in ("oi", "adj") for s in ("01", "07")]


def ndtr(x):
    """standard normal CDF without scipy (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7), vectorised."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, path)


def trace_pass(uw, key, etf, day, now, rows):
    """one intraday pass for one market. rows: {sym: [vol, ask_vol, bid_vol, mleg, prem, iv]} (today, so far)."""
    pos = _load(os.path.join(OUT, f"{key}_pos.json"))
    if not pos or pos.get("date") != day.isoformat() or not pos.get("c"):
        return "no morning positions for today"
    # ---- price path today (regular session) and spot now
    bars = uw.get(f"/stock/{etf}/ohlc/1m", date=day.isoformat(), limit=2500).get("data") or []
    path = []
    for x in bars:
        if x.get("market_time") != "r":
            continue
        try:
            t = dt.datetime.fromisoformat(str(x["start_time"]).replace("Z", "+00:00")).astimezone(now.tzinfo)
            path.append([t.hour * 60 + t.minute, round(float(x["close"]), 3)])
        except (KeyError, TypeError, ValueError):
            continue
    mod = now.hour * 60 + now.minute
    path = sorted(p for p in path if p[0] < mod)             # bars that have closed by now
    if not path:
        return "no regular-session bars yet"
    S = path[-1][1]
    center = pos.get("prev_close") or path[0][1]
    P = center * (1 + np.arange(-BAND, BAND + 1e-9, STEP))
    Pr = np.append(P, S)                                   # last row = spot now (for delta pressure)
    j0 = next((j for j, c in enumerate(COLS) if c >= mod), len(COLS) - 1)

    # ---- contracts: [sym, root, exp, cp, k, r, oi, iv]
    C = pos["c"]
    K = np.array([c[4] for c in C], float); R = np.array([c[5] for c in C], float)
    OI = np.array([c[6] for c in C], float)
    def live_iv(c):
        r = rows.get(c[0])
        return r[5] if r and len(r) > 5 and r[5] else c[7]
    iv = np.array([live_iv(c) for c in C], float)
    iv = np.where((iv > 0.02) & (iv < 5), iv, np.array([c[7] for c in C], float))
    call = np.array([c[3] == "C" for c in C])
    dte = np.array([(dt.date.fromisoformat(c[2]) - day).days for c in C])
    close_min = np.array([(dt.date.fromisoformat(c[2]) - day).days * 1440 + (570 if c[1] == "SPX" else 960) for c in C], float)
    net = np.array([((rows.get(c[0]) or [0, 0, 0])[1] - (rows.get(c[0]) or [0, 0, 0])[2]) for c in C], float)
    conv = np.where(call, 1.0, -1.0) * OI                   # dealer position, usual convention
    W = {"oi": conv, "adj": conv - net}                     # customers bought at the ask -> dealers shorter
    M = {"01": dte <= 1, "07": dte <= 7}

    def greeks(tau):
        T = np.maximum(close_min - tau, 1.0) / YEAR_MIN
        alive = (close_min - tau) >= 0                    # at the closing minute itself: T floored at 1 min
        Pn = Pr[:, None] * R[None, :]
        sd = iv * np.sqrt(T)
        d1 = (np.log(Pn / K) + 0.5 * iv * iv * T) / sd
        delta = ndtr(d1) - (~call)
        gam = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (Pn * sd)
        return delta * alive, gam * alive

    # per combo, per column: ETF-share-equivalent dealer delta D (rows+spot) and gamma G
    D, G = {}, {}
    for j in range(j0, len(COLS)):
        dl, gm = greeks(COLS[j])
        for m, w in W.items():
            for s, mask in M.items():
                ww = w * mask * 100.0
                D.setdefault((m, s), {})[j] = dl @ (ww * R)                 # shares (ETF-equivalent)
                G.setdefault((m, s), {})[j] = (gm @ (ww * R * R)) * Pr * Pr * 0.01   # $ per 1% at each row
    tail = greeks(COLS[-1] + 5)[0]                                   # for the last column's charm

    out_cols = {}
    for m in W:
        for s in M:
            d, g = D[(m, s)], G[(m, s)]
            for j in range(j0, len(COLS)):
                dn = d[j + 1] if j + 1 < len(COLS) else tail @ (W[m] * M[s] * 100.0 * R)
                out_cols.setdefault(f"g|{m}|{s}", {})[j] = g[j][:-1]
                out_cols.setdefault(f"dp|{m}|{s}", {})[j] = -(d[j][:-1] - d[j][-1]) * P
                # charm ends at 15:55 -> 16:00; the 16:00 column has no 'next 5 minutes'
                out_cols.setdefault(f"ch|{m}|{s}", {})[j] = (-(dn[:-1] - d[j][:-1]) * P if j < len(COLS) - 1
                                                              else np.full(len(P), np.nan))

    # ---- the past stays as it was computed: keep each pass's own first column
    hist_p = os.path.join(OUT, f"{key}_trace_hist.json")
    hist = _load(hist_p) or {}
    if hist.get("date") != day.isoformat() or hist.get("center") != center:
        hist = {"date": day.isoformat(), "center": center, "cols": {}}
    for cmb, cols in out_cols.items():
        hist["cols"].setdefault(cmb, {})[str(j0)] = [round(float(v), 1) for v in cols[j0]]
    _write(hist_p, hist)

    grids = {}
    for cmb in COMBOS:
        full = np.full((len(COLS), len(P)), np.nan)
        for js, v in hist["cols"].get(cmb, {}).items():
            if int(js) < j0:
                full[int(js)] = v
        for j, v in out_cols[cmb].items():
            full[j] = v
        # scale to the 98th percentile and clip: the last minutes' 0DTE spikes would wash out everything else
        fin = np.abs(full[np.isfinite(full)])
        sc = float(np.quantile(fin, 0.98)) if fin.size else 0.0
        q = np.where(np.isfinite(full), np.clip(np.round(full / sc * 100), -100, 100) if sc else 0, -999).astype(int)
        grids[cmb] = {"scale": sc, "q": q.tolist()}

    # ---- strike plot: dealer gamma by ETF strike at spot now, per positions mode x expiries
    strikes = {}
    Sn = S * R
    T0 = np.maximum(close_min - mod, 1.0) / YEAR_MIN
    sd0 = iv * np.sqrt(T0)
    d10 = (np.log(Sn / K) + 0.5 * iv * iv * T0) / sd0
    g0 = np.exp(-0.5 * d10 * d10) / math.sqrt(2 * math.pi) / (Sn * sd0) * ((close_min - mod) > 0)
    ku = np.round(K / R)
    for m, w in W.items():
        for s, mask in M.items():
            v = g0 * w * mask * 100.0 * Sn * Sn * 0.01
            for k, x in zip(ku, v):
                if abs(k / S - 1) <= BAND:
                    strikes.setdefault(int(k), {})[f"{m}|{s}"] = strikes.get(int(k), {}).get(f"{m}|{s}", 0.0) + float(x)
    sp = [[k] + [round(strikes[k].get(f"{m}|{s}", 0.0)) for m in W for s in M] for k in sorted(strikes)]

    doc = {"market": key, "etf": etf, "date": day.isoformat(), "updated_et": now.strftime("%H:%M"), "now_col": j0,
           "spot": S, "center": center, "prices": [round(float(p), 3) for p in P], "cols": COLS,
           "path": path, "grids": grids, "strike_cols": [f"{m}|{s}" for m in W for s in M], "strikes": sp,
           "n_contracts": len(C), "adj_volume": int(np.abs(net).sum())}
    # replay frames: one gzipped file per pass (~5x smaller); the page lists them from `frames`
    a = os.path.join(OUT, "archive", day.isoformat(), "trace")
    os.makedirs(a, exist_ok=True)
    with gzip.open(os.path.join(a, f"{key}_{now.strftime('%H%M')}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    doc["frames"] = sorted(x.split("_")[1][:4] for x in os.listdir(a) if x.startswith(key + "_") and x.endswith(".json.gz"))
    _write(os.path.join(OUT, f"{key}_trace.json"), doc)
    _prune(day)
    return None


def _prune(day, keep_days=10):
    """replay frames older than `keep_days` go (the morning maps in archive/ are kept)."""
    root = os.path.join(OUT, "archive")
    for d in (os.listdir(root) if os.path.isdir(root) else []):
        try:
            old = (day - dt.date.fromisoformat(d)).days > keep_days
        except ValueError:
            continue
        if old and os.path.isdir(os.path.join(root, d, "trace")):
            shutil.rmtree(os.path.join(root, d, "trace"), ignore_errors=True)
