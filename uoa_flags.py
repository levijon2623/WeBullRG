# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "numpy", "python-dotenv"]
# ///
"""
uoa_flags.py
============
UOA FLAGS FOR THE VIEWER (user, 2026-10-31) -- every big one-sided burst on one
contract, with the next morning's open interest and the contract's price since.

CONTEXT, NOT A SIGNAL. check_uoa_paths (runs 1-4, 2026-10-31) bought these
bursts right after the print, at the next open once OI confirmed, and an hour
in, with +50..+300% take-profits: they reached every target exactly as often as
unprinted contracts of the same profile, and trailed their same-day twins.
check_uoa_events: stuck prints predicted nothing over 1-5 days. The page shows
who is positioning where; it is not a list of trades.

FLAG (the check_uoa_paths EVENT, so the page and the forward test agree):
  one contract, one pass interval (~5 min): volume >= 1,000, premium >= $100k,
  >= 70% at the ask (BOUGHT) or at the bid (SOLD), multi-leg < 50%, interval
  volume >= start-of-day OI; out of the money 0-10%; <= 21 calendar days.
  Live = the change between two passes of UW's contract rows. Backfill = the
  lake's minute bars summed into the same 5-minute buckets (--backfill).
UNIVERSE  SPY QQQ IWM + the 31 names of check_uoa_paths (no index roots).
EOD MARKET SCAN (user, 2026-10-31) once after the close, the whole market in ONE call:
  UW /screener/option-contracts (hottest chains) with the DAILY-TOTAL version of the flag --
  day volume >= 1,000 and > start-of-day OI, premium >= $100k, >= 70% one side (min_skew_perc),
  multi-leg < 50%, OTM; then 0-10% OTM and <= 21 days to expiry client-side (the server's
  max_dte counts from TODAY on past dates, and it has no OTM cap). ~150-300 contracts a day.
  Whole-day totals, not a 5-minute burst: a looser, slower cousin of the live flag.
  -> live/uoa/eod/DATE.json, same flag schema (t = "EOD"), same morning confirm.
CONFIRM   each morning from 06:45 ET: /option-contract/{occ}/historic -> the
  next session's start-of-day OI (what the flag day left behind) and the
  contract's price path since (best high, last), for flags of the last 10
  sessions. dOI vs the flagged volume:
    >= 50%  opened (bought) / written (sold) -- "stuck"
    10-50%  partly
    -10..+10%  didn't stick (day trade / churn)
    <= -10%  closing (bought = shorts covering, sold = longs selling out)
  Side and opening/closing are inferences from where trades printed.

UW REST only (~70 calls a pass, ~80 passes a day; ~1,000 history calls a morning).
Never touches the bot, the viewer or Webull.

OUTPUT (live/, gitignored)
  live/uoa/DATE.json     {date, updated_et, source, flags: [...]}
  live/uoa/index.json    dates with a file, newest first
  live/uoa/state.json    the last pass's rows (restart-safe diffing)
  live/uoa/eod/DATE.json + eod/index.json   the EOD market scan

    python uoa_flags.py                              # service loop (cleanbot-uoa)
    python uoa_flags.py --pass                       # one live pass now
    python uoa_flags.py --confirm [--date D]         # one confirm pass (D = as-of morning)
    python uoa_flags.py --backfill 2026-10-19 2026-10-30   # from the local lake (PC only, polars)
    python uoa_flags.py --eod [--date D | --eod-range D1 D2]  # the EOD market scan (D = a past session)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from dotenv import load_dotenv                            # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
import market_calendar as MC                              # noqa: E402
import positioning_map as P                               # noqa: E402  (UW client, OCC parsing)

NY = P.NY
OUT = os.path.join("live", "uoa")
ETFS = ("SPY", "QQQ", "IWM")
NAMES = ("META", "NVDA", "MSFT", "AVGO", "SMH", "GLD",
         "TSLA", "AAPL", "ORCL", "TLT", "PLTR", "AMZN", "IBIT", "AMD", "SLV", "MU", "MSTR", "INTC", "GOOGL",
         "HOOD", "XLF", "PCG", "GME", "USO", "COIN", "NFLX", "TSM", "XLE", "WMT", "MRVL", "BE")
UNIVERSE = ETFS + NAMES
MIN_VOL, MIN_PREM, SIDE, MLEG, VOL_X_OI, MAX_OTM, MAX_DTE = 1000, 1e5, 0.70, 0.50, 1.0, 0.10, 21
START, END = 9 * 60 + 35, 16 * 60 + 5            # pass window (ET minutes)
CONFIRM_START, CONFIRM_END = 6 * 60 + 45, 9 * 60 + 25
LOOKBACK = 10                                    # sessions of flags kept current each morning
EOD_RUN, EOD_LOOKBACK = 16 * 60 + 30, 5          # market scan after the close; ~200/day -> fewer sessions refreshed
EOD = "eod"
CAP = 500                                        # UW's row cap per contract-list call


# ------------------------------------------------------------------ files
def _load(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(name, doc):
    p = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(p + ".tmp", p)


def _index(sub=""):
    d = os.path.join(OUT, sub)
    ds = sorted((f[:10] for f in os.listdir(d) if f[:4].isdigit() and f.endswith(".json")), reverse=True)
    _write(os.path.join(sub, "index.json"), {"dates": ds})


def tdte(day, exp):
    """trading days from day to expiry: Friday -> Monday = 1."""
    return len([d for d in MC.trading_days(day + dt.timedelta(days=1), exp)]) if exp > day else 0


def flag_from(sym, tk, day, t_et, spot, dv, dask, dbid, dmleg, dprem, oi, day_vol, win, src):
    """the check_uoa_paths EVENT on one interval, or None."""
    p = P.parse_occ(sym)
    if not p or not spot or dv < MIN_VOL or dprem < MIN_PREM or dmleg >= MLEG * dv or dv < VOL_X_OI * (oi or 0):
        return None
    _, exp, cp, k = p
    side = "bought" if dask >= SIDE * dv else "sold" if dbid >= SIDE * dv else None
    otm = (k / spot - 1) if cp == "C" else (1 - k / spot)
    dte = (exp - day).days
    if side is None or not (0 < otm <= MAX_OTM) or not (0 <= dte <= MAX_DTE):
        return None
    return {"id": f"{sym}@{t_et}", "t": t_et, "tk": tk, "sym": sym, "cp": cp, "k": k, "exp": exp.isoformat(),
            "dte": dte, "tdte": tdte(day, exp), "spot": round(spot, 2), "otm": round(otm, 4), "side": side,
            "vol": int(dv), "ask": round(dask / dv, 3), "bid": round(dbid / dv, 3), "mleg": round(dmleg / dv, 3),
            "prem": round(dprem), "px": round(dprem / dv / 100, 3), "oi": int(oi or 0),
            "voi": round(dv / oi, 1) if oi else None, "day_vol": int(day_vol), "win": win, "src": src}


# ------------------------------------------------------------------ earnings
_er_cache = {}


def earnings(uw, day):
    """{ticker: [(date, time)]} -- the last and next report (UW stock screener, one call, cached per day).
    A flag whose contract spans a report reads differently: the premium is mostly the event."""
    if day not in _er_cache:
        rows = uw.get("/screener/stocks", ticker=",".join(UNIVERSE), limit=50).get("data") or []
        out = {}
        for x in rows:
            t = x.get("ticker")
            for k in ("last_earnings_date", "next_earnings_date"):
                if t in UNIVERSE and x.get(k):
                    out.setdefault(t, []).append((x[k][:10], x.get("er_time") if k.startswith("next") else None))
        _er_cache.clear()
        _er_cache[day] = out
    return _er_cache[day]


def tag_earnings(f, day, ers):
    """f["er"] = 'YYYY-MM-DD' (+ ' pre' / ' post') when a report falls between the flag day and expiry."""
    for d, t in ers.get(f["tk"], ()):
        if day.isoformat() <= d <= f["exp"]:
            f["er"] = d + ({"premarket": " pre", "postmarket": " post"}.get(t or "", ""))


# ------------------------------------------------------------------ live
_exp_cache = {}


def pull(uw, tk, day):
    """{sym: [vol, ask, bid, mleg, prem, oi]} for today's rows within MAX_DTE, and whether the list was full."""
    rows, full = {}, False

    def take(xs):
        for x in xs:
            s = x.get("option_symbol") or ""
            p = P.parse_occ(s)
            if not p or (p[1] - day).days > MAX_DTE or p[1] < day:
                continue
            tt = (x.get("last_tape_time") or "")[:19]
            try:
                if dt.datetime.fromisoformat(tt).replace(tzinfo=dt.timezone.utc).astimezone(NY).date() != day:
                    continue                      # last session's row, not rolled to today yet
            except ValueError:
                continue
            n = P.num
            rows[s] = [int(n(x.get("volume")) or 0), int(n(x.get("ask_volume")) or 0), int(n(x.get("bid_volume")) or 0),
                       int(n(x.get("multi_leg_volume")) or 0), round(n(x.get("total_premium")) or 0),
                       int(n(x.get("open_interest")) or 0)]

    if tk in ETFS:
        if _exp_cache.get(tk, (None,))[0] != day:
            _exp_cache[tk] = (day, [e for e in P.expiries(uw, tk, day) if (e - day).days <= MAX_DTE])
        for e in _exp_cache[tk][1]:
            xs = uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat()).get("data") or []
            if len(xs) >= CAP:
                xs = [x for typ in ("call", "put")
                      for x in (uw.get(f"/stock/{tk}/option-contracts", expiry=e.isoformat(), option_type=typ).get("data") or [])]
            take(xs)
    else:
        xs = uw.get(f"/stock/{tk}/option-contracts").get("data") or []
        full = len(xs) >= CAP
        take(xs)
    return rows, full


def spot(uw, tk):
    d = uw.get(f"/stock/{tk}/stock-state").get("data") or {}
    return P.num(d.get("close"))


def live_pass(uw, day, now):
    mod = now.hour * 60 + now.minute
    t_et = now.strftime("%H:%M")
    st = _load(os.path.join(OUT, "state.json"), {}) or {}
    fresh_day = st.get("date") != day.isoformat()
    if fresh_day:
        # volume since the open counts as the first interval only if this IS the first pass;
        # a restart later in the day takes a baseline and flags from the next pass
        st = {"date": day.isoformat(), "t": None, "rows": {}, "full": {}, "baseline": mod > START + 6}
    path = os.path.join(OUT, f"{day}.json")
    doc = _load(path) or {"date": day.isoformat(), "source": "live", "flags": []}
    seen = {f["id"] for f in doc["flags"]}
    bursts = {}
    for f in doc["flags"]:
        bursts[f["sym"]] = bursts.get(f["sym"], 0) + 1
    win = None
    if st.get("t"):
        h, m = map(int, st["t"].split(":"))
        win = mod - (h * 60 + m)
    calls0, new, ok = uw.calls, 0, 0
    try:
        ers = earnings(uw, day)
    except Exception:                                      # noqa: BLE001 -- context only
        ers = {}
    for tk in UNIVERSE:
        try:
            rows, full = pull(uw, tk, day)
            s = spot(uw, tk)
        except Exception as e:                             # noqa: BLE001 -- one ticker must not stop the pass
            print(f"  [UOA] {tk} pull error {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if not rows:
            continue
        ok += 1
        prev = st["rows"].get(tk, {})
        prev_full = st["full"].get(tk, False)
        if not st.get("baseline"):
            for sym, r in rows.items():
                q = prev.get(sym)
                approx = q is None and bool(prev) and prev_full   # entered a capped top-500 list: its earlier
                q = q or [0] * 6                                   # volume (< the list's floor) counts as this interval
                dv = r[0] - q[0]
                if dv <= 0:
                    continue
                f = flag_from(sym, tk, day, t_et, s, dv, r[1] - q[1], r[2] - q[2], r[3] - q[3], r[4] - q[4],
                              r[5], r[0], win, "live")
                if f and f["id"] not in seen:
                    if approx:
                        f["approx"] = True
                    tag_earnings(f, day, ers)
                    bursts[sym] = bursts.get(sym, 0) + 1
                    f["burst"] = bursts[sym]
                    doc["flags"].append(f)
                    new += 1
        st["rows"][tk] = rows
        st["full"][tk] = full
    st["t"], st["baseline"] = t_et, False
    _write("state.json", st)
    doc.update(updated_et=t_et, tickers_ok=ok, calls=uw.calls - calls0)
    _write(f"{day}.json", doc)
    _index()
    return new, ok


# ------------------------------------------------------------------ confirm
def verdict(side, doi, vol):
    if vol <= 0:
        return None
    r = doi / vol
    if r >= 0.5:
        return "opened" if side == "bought" else "written"
    if r >= 0.1:
        return "partly"
    if r > -0.1:
        return "didn't stick"
    return "closing"


def confirm_pass(uw, today, sub="", lookback=LOOKBACK):
    """fill next-session OI and the price path for the last `lookback` sessions' flags. Returns (done, pending)."""
    base = os.path.join(OUT, sub)
    if not os.path.isdir(base):
        return 0, 0
    days = sorted(f[:10] for f in os.listdir(base) if f[:4].isdigit() and f.endswith(".json"))
    days = [d for d in days if d < today.isoformat()][-lookback:]
    done = pending = 0
    for dstr in days:
        path = os.path.join(base, f"{dstr}.json")
        doc = _load(path)
        if not doc or not doc.get("flags"):
            continue
        fday = dt.date.fromisoformat(dstr)
        nxt = MC.next_trading_day(fday).isoformat()
        hist = {}
        for f in doc["flags"]:
            if f.get("path_as_of") == today.isoformat() or (f.get("oi_next") is not None and f["exp"] < today.isoformat()):
                continue                          # done this morning, or confirmed and expired
            sym = f["sym"]
            if f["exp"] < nxt:                    # expired before the next session: no OI to judge it by
                f["status"], f["path_as_of"] = "expired", today.isoformat()
                continue
            if sym not in hist:
                rows = uw.get(f"/option-contract/{sym}/historic").get("chains") or []
                hist[sym] = {r.get("date"): r for r in rows}
            h = hist[sym]
            r1 = h.get(nxt)
            if r1 is not None and f.get("oi_next") is None and P.num(r1.get("open_interest")) is not None:
                oi1 = int(P.num(r1.get("open_interest")))
                r0 = h.get(dstr) or {}
                f["oi_next"] = oi1
                f["doi"] = oi1 - int(P.num(r0.get("open_interest")) or f["oi"])
            after = [r for d, r in h.items() if d and d > dstr]
            r0 = h.get(dstr) or {}
            f["close_px"] = P.num(r0.get("last_price"))
            if after:
                f["hi_since"] = max(P.num(r.get("high_price")) or 0 for r in after)
                lows = [P.num(r.get("low_price")) for r in after if P.num(r.get("low_price"))]
                f["lo_since"] = min(lows) if lows else None
                last = max(after, key=lambda r: r["date"])
                f["last_px"], f["last_date"] = P.num(last.get("last_price")), last["date"]
            if f.get("oi_next") is not None:
                f["path_as_of"] = today.isoformat()
                done += 1
            else:
                pending += 1
        # the verdict uses ALL of the day's flagged volume on the contract (several bursts share one dOI)
        tot = {}
        for f in doc["flags"]:
            tot[f["sym"]] = tot.get(f["sym"], 0) + f["vol"]
        for f in doc["flags"]:
            if f.get("doi") is not None and f.get("status") != "expired":
                f["flag_vol_day"] = tot[f["sym"]]
                f["status"] = verdict(f["side"], f["doi"], tot[f["sym"]])
        doc["confirmed_et"] = dt.datetime.now(NY).strftime("%Y-%m-%d %H:%M")
        _write(os.path.join(sub, f"{dstr}.json"), doc)
    return done, pending


# ------------------------------------------------------------------ EOD market scan
EOD_PARAMS = dict(max_dte=MAX_DTE, min_volume=MIN_VOL, min_premium=int(MIN_PREM), vol_greater_oi="true",
                  max_multileg_volume_ratio=str(MLEG), min_skew_perc=str(SIDE), exclude_itm="true",
                  order="premium", order_direction="desc", limit=500)


def eod_scan(uw, day):
    """the whole market's daily-total flags for `day` (a finished session). Returns the flag count."""
    rows = []
    for page in range(8):
        x = uw.get("/screener/option-contracts", date=day.isoformat(), page=page, **EOD_PARAMS).get("data") or []
        rows += x
        if len(x) < 500:
            break
    n, flags = P.num, []
    for r in rows:
        try:
            exp = dt.date.fromisoformat((r.get("expiry") or "")[:10])
        except ValueError:
            continue
        v, s, k = n(r.get("volume")) or 0, n(r.get("stock_price")), n(r.get("strike"))
        if not v or not s or not k:
            continue
        cp = "C" if r.get("option_type") == "call" else "P"
        otm = (k / s - 1) if cp == "C" else (1 - k / s)
        dte = (exp - day).days
        a, b = n(r.get("ask_side_volume")) or 0, n(r.get("bid_side_volume")) or 0
        side = "bought" if a >= SIDE * v else "sold" if b >= SIDE * v else None
        if side is None or not (0 < otm <= MAX_OTM) or not (0 <= dte <= MAX_DTE):
            continue
        oi = int(n(r.get("open_interest")) or 0)
        prem = n(r.get("premium")) or 0
        f = {"id": f"{r['option_symbol']}@EOD", "t": "EOD", "tk": r.get("ticker_symbol"), "sym": r["option_symbol"],
             "cp": cp, "k": k, "exp": exp.isoformat(), "dte": dte, "tdte": tdte(day, exp), "spot": round(s, 2),
             "otm": round(otm, 4), "side": side, "vol": int(v), "ask": round(a / v, 3), "bid": round(b / v, 3),
             "mleg": round((n(r.get("multileg_volume")) or 0) / v, 3), "prem": round(prem),
             "px": round(n(r.get("avg_price")) or prem / v / 100, 3), "oi": oi, "voi": round(v / oi, 1) if oi else None,
             "day_vol": int(v), "win": None, "src": "eod", "issue": r.get("issue_type"), "sector": r.get("sector"),
             "sweep": round((n(r.get("sweep_volume")) or 0) / v, 3), "is_new": bool(r.get("is_new")),
             "days_oi_up": r.get("days_of_oi_increases"), "days_vol_gt_oi": r.get("days_of_vol_greater_than_oi")}
        er = (r.get("next_earnings_date") or "")[:10]
        if er and day.isoformat() <= er <= f["exp"]:
            f["er"] = er + {"premarket": " pre", "afterhours": " post", "postmarket": " post"}.get(r.get("er_time") or "", "")
        flags.append(f)
    _write(os.path.join(EOD, f"{day}.json"), {
        "date": day.isoformat(), "updated_et": dt.datetime.now(NY).strftime("%H:%M"), "flags": flags,
        "source": "EOD market scan (UW hottest chains, whole-day totals)", "screened": len(rows)})
    _index(EOD)
    return len(flags)


# ------------------------------------------------------------------ backfill (PC, local lake)
def backfill(lo, hi):
    import polars as pl
    ers = earnings(P.UW(), hi)                   # last + next report as of now: covers a recent window
    silver = "lake/silver/option-contracts-1m"
    for day in MC.trading_days(lo, hi):
        p = f"{silver}/date={day}/bars.parquet"
        if not os.path.exists(p):
            print(f"  {day}: no lake partition")
            continue
        df = (pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(list(UNIVERSE)))
              .select("option_chain_id", "underlying_symbol", "minute_et", "volume", "ask_volume", "bid_volume",
                      "multi_volume", "premium", "open_interest", "underlying_close").collect()
              .with_columns(mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute()
                            .cast(pl.Int32)))
        # the live passes' 5-minute buckets: (09:30, 09:35] is the 09:35 pass, and so on
        df = df.with_columns(b=((pl.col("mod") - 9 * 60 - 30) // 5 + 1) * 5 + 9 * 60 + 30) \
               .filter(pl.col("b").is_between(START, END))
        g = (df.sort("minute_et").group_by("option_chain_id", "underlying_symbol", "b")
             .agg(pl.col("volume").sum(), pl.col("ask_volume").sum(), pl.col("bid_volume").sum(),
                  pl.col("multi_volume").sum(), pl.col("premium").sum(), pl.col("open_interest").first(),
                  pl.col("underlying_close").last()).sort("b"))
        g = g.with_columns(day_vol=pl.col("volume").cum_sum().over("option_chain_id"))
        flags, bursts = [], {}
        for r in g.filter(pl.col("volume") >= MIN_VOL).iter_rows(named=True):
            t_et = f"{r['b'] // 60:02d}:{r['b'] % 60:02d}"
            f = flag_from(r["option_chain_id"], r["underlying_symbol"], day, t_et, r["underlying_close"], r["volume"],
                          r["ask_volume"], r["bid_volume"], r["multi_volume"], r["premium"], r["open_interest"],
                          r["day_vol"], 5, "lake")
            if f:
                bursts[f["sym"]] = bursts.get(f["sym"], 0) + 1
                f["burst"] = bursts[f["sym"]]
                tag_earnings(f, day, ers)
                flags.append(f)
        flags.sort(key=lambda f: f["t"])
        _write(f"{day}.json", {"date": day.isoformat(), "source": "lake backfill (5-min buckets of minute bars)",
                               "updated_et": "16:05", "flags": flags})
        print(f"  {day}: {len(flags)} flags")
    _index()


# ------------------------------------------------------------------ service
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="one", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--backfill", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--eod", action="store_true")
    ap.add_argument("--eod-range", nargs=2, metavar=("FROM", "TO"))
    a = ap.parse_args()
    if a.backfill:
        backfill(dt.date.fromisoformat(a.backfill[0]), dt.date.fromisoformat(a.backfill[1]))
        return
    uw = P.UW()
    now = dt.datetime.now(NY)
    if a.one:
        print(live_pass(uw, now.date(), now))
        return
    if a.eod or a.eod_range:
        days = (MC.trading_days(*map(dt.date.fromisoformat, a.eod_range)) if a.eod_range
                else [dt.date.fromisoformat(a.date) if a.date else now.date()])
        for d in days:
            print(d, eod_scan(uw, d))
        return
    if a.confirm:
        d = dt.date.fromisoformat(a.date) if a.date else now.date()
        print(confirm_pass(uw, d), confirm_pass(uw, d, EOD, EOD_LOOKBACK))
        return
    print("  🚩 [UOA] service up (confirm 06:45-09:25 ET, passes every 5 min 09:35-16:05)", flush=True)
    last_pass, confirmed, eod_done = 0.0, {}, set()
    while True:
        now = dt.datetime.now(NY)
        day, mod = now.date(), now.hour * 60 + now.minute
        if MC.is_trading_day(day):
            if CONFIRM_START <= mod <= CONFIRM_END and confirmed.get(day) != "done" \
                    and time.time() - confirmed.get((day, "t"), 0) >= 600:
                try:
                    d, pnd = confirm_pass(uw, day)
                    d2, pnd2 = confirm_pass(uw, day, EOD, EOD_LOOKBACK)
                    d, pnd = d + d2, pnd + pnd2
                    confirmed[(day, "t")] = time.time()
                    print(f"  🚩 [UOA] confirm {now:%H:%M}: {d} filled, {pnd} pending", flush=True)
                    if pnd == 0:
                        confirmed[day] = "done"
                except Exception as e:                     # noqa: BLE001
                    print(f"  🚩 [UOA] confirm error {type(e).__name__}: {str(e)[:160]}", flush=True)
            if START <= mod <= END and time.time() - last_pass >= 290:
                last_pass = time.time()
                try:
                    n, ok = live_pass(uw, day, now)
                    if n:
                        print(f"  🚩 [UOA] {now:%H:%M} +{n} flag(s) ({ok} tickers)", flush=True)
                except Exception as e:                     # noqa: BLE001
                    print(f"  🚩 [UOA] pass error {type(e).__name__}: {str(e)[:160]}", flush=True)
            if mod >= EOD_RUN and day not in eod_done:
                eod_done.add(day)
                try:
                    print(f"  🚩 [UOA] EOD market scan: {eod_scan(uw, day)} flag(s)", flush=True)
                except Exception as e:                     # noqa: BLE001
                    print(f"  🚩 [UOA] EOD scan error {type(e).__name__}: {str(e)[:160]}", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
