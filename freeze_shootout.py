"""
freeze_shootout.py -- the SHOOTOUT pill's history table (user, 2026-11-27), from explore_shootout_end.py's rows
(_shootout_cache/end_rows.parquet: every session 2023-11-30 .. 2026-11-25, definition A). DESCRIPTIVE ONLY -- elapsed
time and the on/off state, no factors (a factor-based "today runs long" countdown waits for check_shootout_countdown.py's
pre-registered validation). Writes shootout_typical.json:

  per ticker   share still on at 09:35, median end, middle half, 10-90%
  still_on[m]  for checkpoints 09:35 .. 10:55 every 5 min: of the sessions still ON at m, the median end and its middle
               half (what "still going at 09:50" has usually meant), with n. Censored at 11:00 (the study's cap).

    python freeze_shootout.py
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "_shootout_cache", "end_rows.parquet")
OUT = os.path.join(ROOT, "shootout_typical.json")
CAP = 660


def main():
    R = pl.read_parquet(SRC)
    out = {"built": dt.datetime.now().isoformat(timespec="seconds"), "source": "explore_shootout_end.py (definition A)",
           "sessions": int(R["d"].n_unique()), "span": [R["d"].min(), R["d"].max()], "cap": CAP, "tickers": {}}
    for tk in ("SPY", "QQQ", "IWM"):
        x = R.filter(pl.col("tk") == tk)
        e = x["end"].to_numpy()
        q = np.percentile(e, [10, 25, 50, 75, 90])
        t = {"n": int(len(e)), "on_at_935": round(float(x["started"].mean()), 3),
             "median": int(q[2]), "q25": int(q[1]), "q75": int(q[3]), "q10": int(q[0]), "q90": int(q[4]),
             "past_cap": round(float(np.mean(e >= CAP)), 3), "still_on": {}}
        for m in range(575, CAP, 5):
            on = e[e > m]
            if len(on) >= 20:
                t["still_on"][str(m)] = {"n": int(len(on)), "median": int(np.median(on)),
                                         "q25": int(np.percentile(on, 25)), "q75": int(np.percentile(on, 75)),
                                         "past_cap": round(float(np.mean(on >= CAP)), 3)}
        out["tickers"][tk] = t
        print(tk, {k: v for k, v in t.items() if k != "still_on"}, "| still on at 09:50 ->", t["still_on"].get("590"))
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
