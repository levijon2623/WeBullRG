"""
qual_test.py -- the user's QUALIFYING TEST tracker (user, 2026-11-21). Personal: every amount lives in git-ignored
live/qual_test/ files, never in this repo.

THE TEST (the user's rules, written down 2026-11-21 in live/qual_test/rules.json before the test ran on):
  score      = account balance + cumulative withdrawals - cumulative deposits   (profit kept, skims included)
  max drawdown  the score must not fall more than drawdown_frac x deposits below its PEAK, positions flat (user,
                2026-11-21). A max-drawdown breach FAILS the test; a daily-limit breach makes that day a
                behavioural failure (counted as a broken day in the rule score) (user, 2026-11-21).
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


ACTIVITIES = os.path.join(ROOT, "live", "account_activities.json")


def merge_webull_transfers(C, start):
    """Transfers from Webull's own cash ledger (manual_orders pulls it into live/account_activities.json; 2026-11-27).
    A WITHDRAW / DEPOSIT and the FEES row with the same biz_time are ONE transfer (wire + its fee -- how the user counts
    it; every hand-recorded withdrawal since 11-23 matched that to the cent). A hand-recorded transfer on the same date
    within $0.02 just gains the Webull id; anything new is added with Webull's exact time. Returns True if C changed."""
    rows = (jload(ACTIVITIES, {}) or {}).get("rows", [])
    groups = defaultdict(list)
    for x in rows:
        if x.get("activity_type") in ("WITHDRAW", "DEPOSIT", "FEES"):
            groups[x.get("biz_time")].append(x)
    changed = False
    tr = C.setdefault("transfers", [])
    for bt, xs in groups.items():
        main = [x for x in xs if x.get("activity_type") in ("WITHDRAW", "DEPOSIT")]
        if len(main) != 1:
            continue                                  # a fee on its own (no transfer) is not money moved by the user
        m = main[0]
        kind = "withdrawal" if m["activity_type"] == "WITHDRAW" else "deposit"
        fee = sum(abs(float(x.get("net_amount") or 0)) for x in xs if x.get("activity_type") == "FEES")
        amt = round(abs(float(m.get("net_amount") or 0)) + (fee if kind == "withdrawal" else -fee), 2)
        try:
            t = datetime.fromisoformat(str(bt).replace("Z", "+00:00")).astimezone(NY)
        except ValueError:
            continue
        d = t.date().isoformat()
        if d < start or any(x.get("webull_id") == m.get("id") for x in tr):
            continue
        same = [x for x in tr if x.get("kind") == kind and x.get("date") == d and abs(float(x["amount"]) - amt) <= 0.02
                and not x.get("webull_id")]
        if same:
            same[0]["webull_id"] = m.get("id")
            same[0].setdefault("ts", t.timestamp())
        else:
            tr.append(dict(date=d, kind=kind, amount=amt, ts=t.timestamp(), webull_id=m.get("id"),
                           note=f"from Webull cash activities: {m.get('activity_sub_type')} {m.get('net_amount')}"
                                + (f" + fee {fee:.2f}" if fee else "")))
        changed = True
    if changed:
        tr.sort(key=lambda x: (x["date"], x.get("ts") or 0))
    return changed


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


def stop_rule_for(R, d):
    """the stop rule in force on day d. rules.json behaviour.stop_rule is a dated list, newest last:
      {"from": D, "kind": "first_loss"}                 the first losing close once the day is green -> stop
      {"from": D, "kind": "giveback", "frac": f}        realized P&L falls to (1 - f) x the day's realized peak -> stop
    (user, 2026-11-24: "stop when 25% of peak has been returned to the market"; a loss after any green was too strict
    for the opening minutes. Written after the 11-24 session, so it applies from 11-25.) No list = first_loss."""
    rules = ((R or {}).get("behaviour") or {}).get("stop_rule") or []
    cur = {"kind": "first_loss"}
    for r in sorted(rules, key=lambda r: r["from"]):
        if r["from"] <= d:
            cur = r
    return cur


def stop_rule_label(r):
    if r.get("kind") == "giveback":
        arm = f" (once the peak is above ${float(r['min_peak']):,.0f})" if r.get("min_peak") else ""
        return f"stop after giving back {r['frac'] * 100:.0f}% of the day's peak{arm}"
    return "first loss once green → stop"


def behaviour(lo, hi, R=None):
    """per day from the viewer ledger: realized path, kept-of-peak, and that day's stop rule (stop_rule_for)."""
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
        rule = stop_rule_for(R, d)
        cum, peak, stop_at = 0.0, 0.0, None
        for c in cs:
            if c.get("pnl") is None:
                continue
            if rule.get("kind") == "giveback":
                cum += float(c["pnl"]); peak = max(peak, cum)
                if stop_at is None and peak > max(0.0, float(rule.get("min_peak") or 0)) and cum <= (1 - rule["frac"]) * peak:
                    stop_at = c["ts"]                    # the close that handed back frac of the day's peak
            else:
                if stop_at is None and cum > 0 and c["pnl"] < 0:
                    stop_at = c["ts"]                    # the first losing close once the day was green
                cum += float(c["pnl"]); peak = max(peak, cum)
        broke = None if stop_at is None else any(o["ts"] > stop_at for o in os_)
        out[d] = {"trades": len(cs), "realized": round(cum, 2), "peak": round(peak, 2),
                  "kept": round(cum / peak, 3) if peak > 0 else None,
                  "rule": None if stop_at is None else ("kept" if not broke else "broken"),
                  "stop_rule": rule.get("kind"),
                  # live aid: the realized level at which today's rule says stop (giveback only), and whether the
                  # day's peak has cleared min_peak yet (user, 2026-11-24: a small early peak must not end the day)
                  "stop_line": round((1 - rule["frac"]) * peak, 2) if rule.get("kind") == "giveback" and peak > 0 else None,
                  "armed": (peak > float(rule.get("min_peak") or 0)) if rule.get("kind") == "giveback" else None,
                  "min_peak": rule.get("min_peak") if rule.get("kind") == "giveback" else None}
    return out


def status():
    R = jload(os.path.join(DIR, "rules.json"), None)
    C = jload(os.path.join(DIR, "cash.json"), {"transfers": []})
    if not R:
        return None
    if merge_webull_transfers(C, R["start"]):          # Webull's own ledger: withdrawals / deposits recorded exactly
        dump("cash.json", C)
    snaps = []
    try:
        snaps = [json.loads(l) for l in open(os.path.join(DIR, "snaps.jsonl"), encoding="utf-8")]
    except OSError:
        pass
    snaps = [s for s in snaps if s.get("flat")]
    # VOIDED SAMPLES (cash.json "void": [{"from": ts, "to": ts, "note": why}]): windows whose "flat" balance was wrong --
    # 2026-11-27 09:44-09:46 an ORPHANED contract (a late fill the bot was not tracking) sat in the account while the
    # state said flat, so its cost read as a loss for the day (user: void it). Kept in snaps.jsonl, skipped here, reason kept.
    voids = [(float(v["from"]), float(v["to"])) for v in C.get("void", [])]
    snaps = [s for s in snaps if not any(a <= s["ts"] <= b for a, b in voids)]
    tr = sorted(C.get("transfers", []), key=lambda t: t["date"])
    dep = sum(t["amount"] for t in tr if t["kind"] == "deposit")
    days = sorted({s.get("day") or day_of(s["ts"]) for s in snaps if (s.get("day") or day_of(s["ts"])) >= R["start"]})

    def done_by(t, d, ts=None):
        """has transfer t happened by sample time ts on day d? A transfer with its own "ts" counts from that moment;
        one without counts from the start of its day (a pre-open transfer). 2026-11-23: a withdrawal of the day's
        profit AFTER trading was credited to the whole day, so the morning's samples counted it twice (a false peak)."""
        if t["date"] != d:
            return t["date"] < d
        return ts is None or t.get("ts") is None or t["ts"] <= ts

    def transfers_through(d, ts=None):
        w = sum(t["amount"] for t in tr if t["kind"] == "withdrawal" and done_by(t, d, ts))
        dp = sum(t["amount"] for t in tr if t["kind"] == "deposit" and done_by(t, d, ts))
        return w, dp

    daily_lim = R["daily_frac"] * dep
    _S = R.get("simple") or {}
    lim_of = lambda d: float(_S["daily_stop"]) if _S and d >= _S.get("from", "9999") else daily_lim    # noqa: E731
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
            ws, dps = transfers_through(d, s["ts"])       # only transfers already made at this sample
            sc = s["cash"] + ws - dps
            peak = sc if peak is None else max(peak, sc)
            lim = R["drawdown_frac"] * dep                 # 50% of deposits below the peak score (user)
            if sc < peak - lim:
                breaches.append({"date": d, "kind": "drawdown", "score": round(sc, 2), "peak": round(peak, 2)})
            if prev_eod is not None:
                w_s = sum(t["amount"] for t in tr if t["kind"] == "withdrawal" and t["date"] == d and done_by(t, d, s["ts"]))
                d_s = sum(t["amount"] for t in tr if t["kind"] == "deposit" and t["date"] == d and not t.get("base")
                          and done_by(t, d, s["ts"]))
                dpnl = s["cash"] - prev_eod + w_s - d_s
                lo_day = dpnl if lo_day is None else min(lo_day, dpnl)
                if dpnl <= -lim_of(d):
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
    beh_all = behaviour(R["start"], R["end"], R)
    bstart = R.get("behaviour_start", R["start"])            # behaviour is scored from the day the rules were written
    # a DAILY-limit breach is a behavioural failure for that day (user, 2026-11-21): it counts in the rule score as a
    # failed day, whether or not the first-loss rule was triggered (a day traded only in the app gets an entry too)
    for b in uniq:
        if b["kind"] == "daily":
            v = beh_all.setdefault(b["date"], {"trades": 0, "realized": None, "peak": None, "kept": None, "rule": None})
            v["rule"], v["daily_breach"] = "broken", True
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
        "no_breach": not any(b["kind"] == "drawdown" for b in uniq),     # the MAX drawdown fails the test (user)
        "rule_kept": None if not rule_days else sum(v["rule"] == "kept" for v in rule_days) / len(rule_days) >= B["rule_kept_frac"],
        "kept_of_peak_median": None if not kept else st.median(kept) >= B["kept_median"],
        "red_day_avg": None if not reds else -st.mean(reds) <= B["red_day_avg_max"],
    }
    # SIMPLE RULES (user, 2026-11-27: "Daily Hard Stop at -$100, and -20% of peak once over $100, no other rules beyond
    # the max drawdown -- it's simpler"): rules.json "simple" = {from, daily_stop, giveback_frac, giveback_min_peak}.
    # The day rules are now ENGINE hard stops (manual_orders); the test itself keeps only the profit target and the max
    # drawdown. The retired behaviour numbers stay in the status file for the record, not in the verdict.
    S = R.get("simple")
    if S:
        checks = {k: checks[k] for k in ("profit", "no_breach")}
        daily_lim = float(S.get("daily_stop") or daily_lim)
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
           "income_pace": R.get("income_per_session"), "pending": C.get("pending"), "simple": R.get("simple"),
           "stop_rule": stop_rule_for(R, datetime.now(NY).date().isoformat()),
           "stop_rule_label": stop_rule_label(stop_rule_for(R, datetime.now(NY).date().isoformat()))}
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
