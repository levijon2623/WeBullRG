# /// script
# requires-python = ">=3.11"
# ///
"""
check_manual_shadow.py
======================
DO I TAKE PROFITS TOO FAST? -- the operator's OWN exits vs holding (2026-10-16)

The flow-trigger path study (check_trigger_paths) says holding pays on
average for the BOT's entries. The operator trades chart entries by hand, a
different population, so the question has to be answered with their own
trades. manual_orders shadows every manual exit: from the first close of a
contract until 15:55 it records that contract's bid/ask once a minute
(manual_shadow.jsonl). This script lines each closing fill in
manual_executions.jsonl up against what the same contract was worth later.

FOR EACH CLOSING FILL (a scale-out counts as its own fill)
    actual   (exit - entry) / entry
    held+15 / +30 / +60 / close    (bid at the first shadow minute at least
             that long after the exit, or the last one before 15:55) -
             entry, over entry. The BID, i.e. what a marketable sell would
             roughly have got -- no exit cushion, so mildly optimistic.
    best     the highest bid after the exit (hindsight; reported, never a target)
REPORTED, by exit type -- winners (actual > 0) and losers -- and contract-
weighted: mean held - actual, and the share of exits where holding would
have done better. Descriptive. A few weeks of trades is a SMALL sample, and
it is one trader's choices, so read it as a mirror, not a backtest.

Usage (on the box, where the files are):
    python check_manual_shadow.py [--since YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
MARKS = (15, 30, 60)


def rows(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--ledger", default=os.path.join(ROOT, "manual_executions.jsonl"))
    ap.add_argument("--shadow", default=os.path.join(ROOT, "manual_shadow.jsonl"))
    a = ap.parse_args()
    since = dt.date.fromisoformat(a.since) if a.since else None

    sh = defaultdict(list)                      # (date, oid) -> [(ts, bid)]
    for r in rows(a.shadow):
        if r.get("bid") is not None:
            sh[(r["date"], r["oid"])].append((int(r["ts"]), float(r["bid"])))
    for v in sh.values():
        v.sort()

    fills = []
    for r in rows(a.ledger):
        if r.get("kind") != "close" or r.get("dry"):
            continue
        d = dt.datetime.fromtimestamp(int(r["ts"]), NY).date()
        if since and d < since:
            continue
        entry, ex = float(r.get("entry") or 0), r.get("exit")
        if entry <= 0 or ex is None:
            continue
        path = [(t, b) for t, b in sh.get((d.isoformat(), r.get("option_id")), []) if t >= int(r["ts"])]
        if not path:
            continue
        rec = dict(date=d, tk=r.get("ticker"), oid=r.get("option_id"), qty=float(r.get("qty") or 1),
                   reason=str(r.get("exit_reason") or "")[:24], actual=(float(ex) - entry) / entry)
        for mk in MARKS:
            later = [b for t, b in path if t >= int(r["ts"]) + mk * 60]
            rec[f"h{mk}"] = ((later[0] - entry) / entry) if later else None
        rec["close"] = (path[-1][1] - entry) / entry
        rec["best"] = (max(b for _, b in path) - entry) / entry
        fills.append(rec)

    print(f"  closing fills with a shadow record: {len(fills)}"
          + (f" (since {since})" if since else ""))
    if not fills:
        print("  nothing to compare yet -- the shadow record starts with the first manual exit "
              "after this was deployed.")
        return
    print(f"\n  {'date':10} {'tk':4} {'qty':>3}  {'exit':>7}  " +
          "  ".join(f"{'+' + str(m) + 'm':>7}" for m in MARKS) + f"  {'close':>7}  {'best':>7}  reason")
    for f in fills:
        cell = lambda v: "      —" if v is None else f"{v * 100:+6.0f}%"   # noqa: E731
        print(f"  {f['date']} {f['tk']:4} {f['qty']:>3g}  {cell(f['actual'])}  " +
              "  ".join(cell(f[f'h{m}']) for m in MARKS) + f"  {cell(f['close'])}  {cell(f['best'])}  {f['reason']}")

    for title, grp in (("ALL exits", fills), ("WINNING exits (closed green)", [f for f in fills if f["actual"] > 0]),
                       ("LOSING exits (closed red)", [f for f in fills if f["actual"] <= 0])):
        if not grp:
            continue
        print(f"\n  {title}: {len(grp)} fills, {sum(f['qty'] for f in grp):g} contracts")
        for key, lbl in [(f"h{m}", f"held {m} more min") for m in MARKS] + [("close", "held to 15:55")]:
            g = [f for f in grp if f[key] is not None]
            if not g:
                continue
            w = sum(f["qty"] for f in g)
            diff = sum((f[key] - f["actual"]) * f["qty"] for f in g) / w
            better = sum(f["qty"] for f in g if f[key] > f["actual"]) / w
            print(f"    {lbl:18} vs actual: {diff * 100:+6.1f} pp of entry (contract-weighted)   "
                  f"holding better on {better:.0%} of contracts")
    print("\n  bid-based and cushion-free, so 'held' is mildly optimistic; small sample -- a mirror, not a backtest.")


if __name__ == "__main__":
    main()
