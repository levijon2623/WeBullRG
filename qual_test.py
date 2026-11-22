"""
qual_test.py -- the user's QUALIFYING TEST tracker (user, 2026-11-21). Personal: every amount lives in git-ignored
live/qual_test/ files, never in this repo.

THE TEST (the user's rules, written down 2026-11-21 in live/qual_test/rules.json before the test ran on):
  score      = account balance + cumulative withdrawals - cumulative deposits   (profit kept, skims included)
  max drawdown  the score must not fall more than drawdown_frac x deposits below its PEAK, positions flat (user,
                2026-11-21). Any limit breach FAILS the test (user, 2026-11-21).
  daily limit   a day's P&L (balance change from the day's start, net of transfers) must not reach -daily_frac x deposits
  pass (end)    score >= pass_frac x deposits AND score >= keep_frac x peak score, no limit breached, and the
                behaviour criteria (rules.json) met
DATA     balance = live/state.json real_cash (the bot reads Webull every few seconds) sampled only while FLAT (no
         bot or manual position); transfers = live/qual_test/cash.json (entered by hand); behaviour = the viewer's
         trade ledger (manual_executions.jsonl) -- trades placed in the Webull app are in the balance but NOT in the
         behaviour metrics.
OUTPUT   live/qual_test/snaps.jsonl (flat balance samples), live/qual_test/status.json (for the viewer).

    python qual_test.py           (service cleanbot-qualtest: samples every 60 s, writes status)
    python qual_test.py --once
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NY = ZoneInfo("America/New_York")
DIR = os.path.join(ROOT, "live", "qual_test")
STATE = os.path.join(ROOT, "live", "state.json")
LEDGER = os.path.join(ROOT, "manual_executions.jsonl")


def jload(p, default):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def dump(name, obj):
    p = os.path.join(DIR, name)
    json.dump(obj, open(p + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(p + ".tmp", p)


def sample():
    """one balance sample if the account is flat and the bot's state is fresh."""
    s = jload(STATE, {})
    cash, ts = s.get("real_cash"), float(s.get("ts") or 0)
    if cash is None or time.time() - ts > 120:
        return None
    flat = not s.get("n_open") and not s.get("manual")
    rec = {"ts": round(time.time(), 1), "cash": float(cash), "flat": bool(flat)}
    with open(os.path.join(DIR, "snaps.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def day_of(ts):
    return datetime.fromtimestamp(ts, NY).date().isoformat()


def behaviour(lo, hi):
    """per day from the viewer ledger: realized path, kept-of-peak, the first-loss-once-green rule."""
    closes, opens = defaultdict(list), defaultdict(list)
    try:
        for line in open(LEDGER, encoding="utf-8"):
            x = json.loads(line)
            if not x.get("ts") or x.get("dry"):
                continue
            d = day_of(x["ts"])
            if not lo <= d <= hi:
                continue
            if x["kind"] == "close":
                closes[d].append(x)
            elif x["kind"] == "open" and not x.get("adopted"):   # an adopted position is not a new entry
                opens[d].append(x)
    except OSError:
        pass
    out = {}
    for d in sorted(set(closes) | set(opens)):
        cs = sorted(closes[d], key=lambda x: x["ts"]); os_ = sorted(opens[d], key=lambda x: x["ts"])
        cum, peak, stop_at = 0.0, 0.0, None
        for c in cs:
            if c.get("pnl") is None:
                continue
            if stop_at is None and cum > 0 and c["pnl"] < 0:
                stop_at = c["ts"]                        # the first losing close once the day was green
            cum += float(c["pnl"]); peak = max(peak, cum)
        broke = None if stop_at is None else any(o["ts"] > stop_at for o in os_)
        out[d] = {"trades": len(cs), "realized": round(cum, 2), "peak": round(peak, 2),
                  "kept": round(cum / peak, 3) if peak > 0 else None,
                  "rule": None if stop_at is None else ("kept" if not broke else "broken")}
    return out


def status():
    R = jload(os.path.join(DIR, "rules.json"), None)
    C = jload(os.path.join(DIR, "cash.json"), {"transfers": []})
    if not R:
        return None
    snaps = []
    try:
        snaps = [json.loads(l) for l in open(os.path.join(DIR, "snaps.jsonl"), encoding="utf-8")]
    except OSError:
        pass
    snaps = [s for s in snaps if s.get("flat")]
    tr = sorted(C.get("transfers", []), key=lambda t: t["date"])
    dep = sum(t["amount"] for t in tr if t["kind"] == "deposit")
    days = sorted({s.get("day") or day_of(s["ts"]) for s in snaps if (s.get("day") or day_of(s["ts"])) >= R["start"]})

    def transfers_through(d):
        w = sum(t["amount"] for t in tr if t["kind"] == "withdrawal" and t["date"] <= d)
        dp = sum(t["amount"] for t in tr if t["kind"] == "deposit" and t["date"] <= d)
        return w, dp

    daily_lim = R["daily_frac"] * dep
    hist, breaches, peak = [], [], None
    prev_eod = R.get("start_balance")
    for d in days:
        S = [s for s in snaps if (s.get("day") or day_of(s["ts"])) == d]
        w, dp = transfers_through(d)
        w_today = sum(t["amount"] for t in tr if t["kind"] == "withdrawal" and t["date"] == d)
        # the BASE deposit is the starting capital already inside start_balance, not money added that day
        d_today = sum(t["amount"] for t in tr if t["kind"] == "deposit" and t["date"] == d and not t.get("base"))
        lo_day = None
        for s in S:                                       # intraday checks on flat samples
            sc = s["cash"] + w - dp
            peak = sc if peak is None else max(peak, sc)
            lim = R["drawdown_frac"] * dep                 # 50% of deposits below the peak score (user)
            if sc < peak - lim:
                breaches.append({"date": d, "kind": "drawdown", "score": round(sc, 2), "peak": round(peak, 2)})
            if prev_eod is not None:
                dpnl = s["cash"] - prev_eod + w_today - d_today
                lo_day = dpnl if lo_day is None else min(lo_day, dpnl)
                if dpnl <= -daily_lim:
                    breaches.append({"date": d, "kind": "daily", "pnl": round(dpnl, 2)})
        eod = S[-1]["cash"]
        day_pnl = None if prev_eod is None else eod - prev_eod + w_today - d_today
        hist.append({"date": d, "eod": round(eod, 2), "score": round(eod + w - dp, 2),
                     "day_pnl": None if day_pnl is None else round(day_pnl, 2), "low": None if lo_day is None else round(lo_day, 2)})
        prev_eod = eod
    seen, uniq = set(), []
    for b in breaches:                                    # one entry per day and kind
        if (b["date"], b["kind"]) not in seen:
            seen.add((b["date"], b["kind"])); uniq.append(b)
    beh_all = behaviour(R["start"], R["end"])
    bstart = R.get("behaviour_start", R["start"])            # behaviour is scored from the day the rules were written
    beh = {d: v for d, v in beh_all.items() if d >= bstart}
    rule_days = [v for v in beh.values() if v["rule"]]
    kept = [v["kept"] for v in beh.values() if v["kept"] is not None]
    reds = [h["day_pnl"] for h in hist if h["day_pnl"] is not None and h["day_pnl"] < 0]
    score = hist[-1]["score"] if hist else None
    import market_calendar as MC
    left, d = 0, date.today()
    from datetime import timedelta
    while d.isoformat() <= R["end"]:
        if MC.is_trading_day(d) and d.isoformat() > (hist[-1]["date"] if hist else ""):
            left += 1
        d += timedelta(days=1)
    target = R["pass_frac"] * dep
    B = R["behaviour"]
    checks = {
        "profit": None if score is None else score >= target,
        "keep_of_peak": None if score is None or not peak else score >= R["keep_frac"] * peak,
        "no_breach": not uniq,
        "rule_kept": None if not rule_days else sum(v["rule"] == "kept" for v in rule_days) / len(rule_days) >= B["rule_kept_frac"],
        "kept_of_peak_median": None if not kept else st.median(kept) >= B["kept_median"],
        "red_day_avg": None if not reds else -st.mean(reds) <= B["red_day_avg_max"],
    }
    doc = {"updated": time.time(), "start": R["start"], "end": R["end"], "deposits": dep,
           "score": score, "peak": None if peak is None else round(peak, 2), "target": target,
           "keep_line": None if peak is None else round(R["keep_frac"] * peak, 2),
           "drawdown_floor": None if peak is None else round(peak - R["drawdown_frac"] * dep, 2),
           "daily_limit": daily_lim, "sessions_done": len(hist), "sessions_left": left,
           "need_per_session": None if score is None or not left else round(max(0.0, target - score) / left, 2),
           "breaches": uniq, "checks": checks, "history": hist[-30:], "behaviour": beh_all, "behaviour_start": bstart,
           "rule_days": len(rule_days), "rule_kept": sum(v["rule"] == "kept" for v in rule_days),
           "kept_median": None if not kept else round(st.median(kept), 3),
           "red_avg": None if not reds else round(st.mean(reds), 2),
           "income_pace": R.get("income_per_session"), "pending": C.get("pending")}
    dump("status.json", doc)
    return doc


def main():
    import market_calendar as MC
    os.makedirs(DIR, exist_ok=True)
    last_status = 0
    while True:
        now = datetime.now(NY)
        mod = now.hour * 60 + now.minute
        if MC.is_trading_day(now.date()) and 9 * 60 + 25 <= mod <= 16 * 60 + 30:
            try:
                sample()
            except Exception as e:                       # noqa: BLE001 -- a tracker never crash-loops
                print(f"qual test sample: {type(e).__name__} {e}", flush=True)
        if time.time() - last_status >= 60:
            last_status = time.time()
            try:
                status()
            except Exception as e:                       # noqa: BLE001
                print(f"qual test status: {type(e).__name__} {e}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    if "--once" in sys.argv:
        os.makedirs(DIR, exist_ok=True)
        print(json.dumps(status(), indent=1)[:3000])
    else:
        main()
