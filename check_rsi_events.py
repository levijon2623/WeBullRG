"""
check_rsi_events.py -- DESCRIPTIVE: does RSI say anything at THRU breaches / PULLBACK entries that the screens'
momentum inputs did not? (user, 2026-11-20: "was RSI one of the inputs?" -- it was not, in either screen.)

NOT A PRE-REGISTERED TEST. Run on the DISCOVERY samples only (2024-08-20 .. 2025-12-31), which are in-sample for
both screens; both holdouts are spent. If RSI shows something the existing inputs do not, it becomes a one-input
forward pre-registration on sessions after 2026-11-13. If it is mostly a copy of them, that settles it.

EVENTS   _breach_cache/events (THRU; Y +1 next strike / -1 back through K / 0) and _structure_ic_cache/events (PULL;
         Y = 2R before the stop). side +1 long / up-break, -1 short / down-break.
RSI      Wilder RSI(14) on the underlying's 1-minute closes and on 5-minute closes (historical/{T}.parquet, RTH
         bars run continuously across sessions so 09:45 has a warm-up). KNOWABLE: a bar counts only once closed
         at or before t_in. Signed: side x (RSI - 50), so + = momentum with the trade.
REPORTED Spearman(RSI, Y) pooled with a day-block bootstrap 95% CI (2,000); partial Spearman given the frozen score;
         Spearman with the screen's own momentum inputs; Y by RSI quintile and in the textbook zones (>= 70 with the
         trade = "overbought" for a long / "oversold" for a short, <= 30 against).

    python check_rsi_events.py

RUN LOG
  2026-11-20 (discovery, THRU 15,576 / PULL 10,140 events):
    THRU  1m IC +0.003 [-0.017, +0.022], beyond the frozen score +0.011 [-0.007, +0.029]; 5m +0.004.
          A near-copy of the trend inputs: Spearman with c_tr30 +0.76 (1m), c_range +0.70 / c_vwap +0.71 (5m).
          Quintiles flat (-0.36 .. -0.45). Textbook zones: ">= 70 with the break" -0.40 = the middle (-0.41);
          "<= 30 against" -0.09 on 1m but n 172, and 5m (n 957) shows -0.36 -- not repeated.
    PULL  1m IC +0.017 [-0.005, +0.039], beyond the frozen score -0.002; 5m +0.004. With c_tr30 +0.74, x_ret15 +0.59.
    VERDICT: RSI carries no outcome information at either event and is mostly the 30-min trend restated. No forward
    test proposed.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import check_futures_screen as FS                 # noqa: E402  (spear, pspear, rankv)

LO, HI = "2024-08-20", "2025-12-31"
FAM = {"THRU": ("_breach_cache/events", ["a_speed", "a_accel", "c_tr30", "c_range", "c_vwap"]),
       "PULL": ("_structure_ic_cache/events", ["s_speed", "s_retrace", "c_tr30", "c_range", "x_ret15"])}
N_BOOT, SEED = 2000, 20261120


def wilder(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n + 1, len(c)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def series(tk):
    """continuous RTH 1m and 5m closes with (date, close time in seconds) -> RSI."""
    d = (pl.read_parquet(f"historical/{tk}.parquet")
         .with_columns((pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32)).alias("mod"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") < 960))
         .with_columns(pl.col("date").cast(pl.Utf8))
         .sort(["date", "mod"]).unique(["date", "mod"], keep="first", maintain_order=True)
         .select("date", "mod", "close"))
    c1 = d["close"].to_numpy().astype(float)
    r1 = wilder(c1)
    one = {}
    for (day, m), r in zip(zip(d["date"].to_list(), d["mod"].to_list()), r1):
        one.setdefault(day, []).append(((m + 1) * 60, r))          # closes at the end of minute m
    d5 = (d.with_columns((pl.col("mod") // 5 * 5).alias("b")).group_by(["date", "b"], maintain_order=True)
          .agg(pl.col("close").last()).sort(["date", "b"]))
    r5 = wilder(d5["close"].to_numpy().astype(float))
    five = {}
    for (day, b), r in zip(zip(d5["date"].to_list(), d5["b"].to_list()), r5):
        five.setdefault(day, []).append(((b + 5) * 60, r))
    return one, five


def at(lst, t):
    """latest RSI whose bar closed at or before t."""
    best = None
    for tc, r in lst or []:
        if tc <= t:
            best = r
        else:
            break
    return best


def boot(D, x, y, stat, rng):
    days = np.unique(D)
    by = {d: np.where(D == d)[0] for d in days}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[d] for d in rng.choice(days, len(days), replace=True)])
        vals.append(stat(idx))
    return np.quantile(vals, [0.025, 0.975])


def main():
    import check_breach_forward as BF
    import freeze_pullback as FP
    frz = {"THRU": (json.load(open("breach_model_frozen.json")), BF.score),
           "PULL": (json.load(open("pullback_model_frozen.json")), FP.score)}
    S = {tk: series(tk) for tk in ("SPY", "QQQ", "IWM")}
    rng = np.random.default_rng(SEED)
    for fam, (path, mom) in FAM.items():
        E = []
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            day = os.path.basename(f)[:10]
            if LO <= day <= HI:
                E += json.load(open(f))
        rows = []
        for e in E:
            one, five = S[e["t"]]
            a, b = at(one.get(e["d"]), e["t_in"]), at(five.get(e["d"]), e["t_in"])
            if a is None or b is None or np.isnan(a) or np.isnan(b):
                continue
            rows.append((e, e["side"] * (a - 50), e["side"] * (b - 50)))
        y = np.array([float(e["Y"]) for e, _, _ in rows]); D = np.array([e["d"] for e, _, _ in rows])
        z = np.array([frz[fam][1](frz[fam][0], e) for e, _, _ in rows])
        print(f"\n== {fam}: {len(rows)} of {len(E)} discovery events (Y mean {y.mean():+.3f})")
        for k, lab in ((1, "RSI(14) 1m"), (2, "RSI(14) 5m")):
            x = np.array([r[k] for r in rows])
            ic = FS.spear(x, y)
            ci = boot(D, x, y, lambda i: FS.spear(x[i], y[i]), rng)
            pc = FS.pspear(x, y, z)
            pci = boot(D, x, y, lambda i: FS.pspear(x[i], y[i], z[i]), rng)
            cor = []
            for m in mom:
                v = np.array([np.nan if e.get(m) is None else float(e[m]) for e, _, _ in rows])
                ok = ~np.isnan(v)
                cor.append(f"{m} {FS.spear(x[ok], v[ok]):+.2f}")
            cuts = np.quantile(x, [.2, .4, .6, .8]); qi = np.searchsorted(cuts, x, side="right")
            hi_, lo_ = x >= 20, x <= -20
            print(f"  {lab}: IC {ic:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | beyond frozen score {pc:+.3f} [{pci[0]:+.3f}, {pci[1]:+.3f}] "
                  f"| with score {FS.spear(x, z):+.2f}")
            print(f"     vs screen inputs: {', '.join(cor)}")
            print(f"     Y by quintile (against -> with the trade): " + " ".join(f"{y[qi == j].mean():+.3f}" for j in range(5)))
            print(f"     textbook zones: RSI >= 70 with the trade {y[hi_].mean():+.3f} (n {hi_.sum()}), "
                  f"<= 30 against {y[lo_].mean():+.3f} (n {lo_.sum()}), middle {y[~hi_ & ~lo_].mean():+.3f}")


if __name__ == "__main__":
    main()
