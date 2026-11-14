"""
build_cot_features.py -- CFTC positioning features for E-mini S&P, Nasdaq-100, Russell 2000 and VIX futures
(user, 2026-11-14). Built from cot_pull.py's _cot_cache/tff_*.json. POSITIONING ONLY: no prices or returns
are read here, so building it spends no test. The hypotheses are written and locked separately, before any
outcome is joined.

WHY TFF, AND WHICH GROUPS (checked 2026-11-14, futures-only since 2006): in the E-mini S&P leveraged funds are
net SHORT 96% of weeks (basis trades and hedges, not direction), asset managers net LONG every week, dealers
mostly short (the other side of the asset managers). So every group is kept and nothing assumes a "smart" or
"dumb" side.

TIMING (no look-ahead). A report holds positions as of TUESDAY and is released FRIDAY 15:30 ET, after the
cash close. `effective` = the first trading session it could have been used: the Monday after. In a week with
a US federal holiday (the CFTC is shut) the release is often a business day late, so in those weeks
`effective` moves one more session -- conservative, a day of lag at worst.

FEATURES, per contract x group (dealer, asset_mgr, lev_money, other_rept, nonrept) x source (fut = futures
only, cmb = futures + options combined):
  net        long - short, contracts
  net_oi     net / open interest (comparable across years as open interest grows)
  pct156     rank of net_oi in the trailing 156 reports including this one (0..1; needs >= 104, else null)
  chg4       net_oi - net_oi four reports earlier
  chg4_pct   rank of chg4 in the trailing 156 reports (same rule)

MICROS (Micro E-mini S&P 13874U, Nasdaq 209747, Russell 239747; from 2020-21) are folded into the main contract at
1/10, positions and open interest alike. `roll_week` flags report dates from 10 days before to 3 days after a
quarterly expiry, where the roll distorts open interest and spreads (tests report with and without them).

RUSSELL is stitched: the E-mini traded on CME until 2008, on ICE (23977A, $100 x index) from 2008-07 to
2018-06, and on CME (239742, $50 x index) again from 2017-08. In overlap weeks the venue with the larger
notional open interest (contracts x multiplier) is used; net_oi is a ratio, so it does not mix the units.

SKIPPED, not guessed: report dates inside federal-shutdown publication gaps (release_uncertain) and daily rows
whose latest report is over MAX_AGE_DAYS old. `usable` in the daily file is the filter every test must apply.

OUTPUT (_cot_cache/, gitignored):
  cot_weekly.parquet  one row per report: report_date, effective, contract, and every feature
  cot_daily.parquet   one row per trading day: the latest report EFFECTIVE on that day, forward-filled,
                      plus `report_date` and `report_age_days` so a join can see how stale it is
"""
from __future__ import annotations

import datetime as dt
import json
import os

import polars as pl

import market_calendar as MC

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_cot_cache")
CONTRACTS = {"ES": ["13874A"], "NQ": ["209742"], "RTY": ["239742", "23977A"], "VIX": ["1170E1"]}
MULT = {"239742": 50.0, "23977A": 100.0}       # Russell venues, for the stitch only
MICRO = {"ES": "13874U", "NQ": "209747", "RTY": "239747"}   # folded in at 1/10 (user, 2026-11-14)
GROUPS = {"dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
          "asset_mgr": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
          "lev_money": ("lev_money_positions_long", "lev_money_positions_short"),
          "other_rept": ("other_rept_positions_long", "other_rept_positions_short"),
          "nonrept": ("nonrept_positions_long_all", "nonrept_positions_short_all")}
WINDOW, MIN_WEEKS = 156, 104
MAX_AGE_DAYS = 14          # a daily row older than this (e.g. VIX had no reports 2008-12..2009-06) is not usable
# Federal shutdowns stopped publication; the backlog came out weeks later under the normal Tuesday dates, so the
# Friday rule would date them too early. Report dates in these windows (generous margins) are flagged
# release_uncertain and are not `usable` -- excluded, rather than guessing their real release days.
SHUTDOWN_REPORTS = [(dt.date(2013, 9, 24), dt.date(2013, 11, 5)),     # Oct 1-16, 2013
                    (dt.date(2018, 12, 18), dt.date(2019, 3, 5)),     # Dec 22 2018 - Jan 25 2019; catch-up into March
                    (dt.date(2025, 9, 23), dt.date(2026, 1, 27))]     # Oct 1 - Nov 12, 2025; catch-up after


def federal_holidays(year):
    """US federal holidays (observed), the days the CFTC does not work."""
    d = dt.date
    def nth(month, weekday, n):
        x = d(year, month, 1)
        x += dt.timedelta(days=(weekday - x.weekday()) % 7)
        return x + dt.timedelta(weeks=n - 1)
    def last(month, weekday):
        x = d(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else d(year, 12, 31)
        return x - dt.timedelta(days=(x.weekday() - weekday) % 7)
    def observed(x):
        return x - dt.timedelta(days=1) if x.weekday() == 5 else x + dt.timedelta(days=1) if x.weekday() == 6 else x
    fixed = [d(year, 1, 1), d(year, 7, 4), d(year, 11, 11), d(year, 12, 25)] + ([d(year, 6, 19)] if year >= 2021 else [])
    return {observed(x) for x in fixed} | {nth(1, 0, 3), nth(2, 0, 3), last(5, 0), nth(9, 0, 1), nth(10, 0, 2), nth(11, 3, 4)}


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(y, month, day + 1)


# NYSE closures that no rule produces (2023+ come from market_calendar)
_NYSE_SPECIAL = {dt.date(2007, 1, 2), dt.date(2012, 10, 29), dt.date(2012, 10, 30), dt.date(2018, 12, 5)}


def nyse_holidays(y):
    """NYSE full-day holidays. market_calendar's table starts in 2023; earlier years by rule here."""
    if y >= 2023:
        return set(MC.market_holidays(y))
    d = dt.date
    fed = federal_holidays(y)
    hol = {x for x in fed if x.month not in (10, 11) or x.day > 20}     # drop Columbus and Veterans Day
    hol -= {x for x in fed if x.month == 6 and x.day >= 18 and y < 2022}  # Juneteenth: NYSE from 2022
    hol = {x for x in hol if not (x.month == 12 and x.day == 31)}      # Sat New Year: no Friday close
    hol.add(_easter(y) - dt.timedelta(days=2))                          # Good Friday
    return hol | {x for x in _NYSE_SPECIAL if x.year == y}


def is_td(x):
    return x.weekday() < 5 and x not in nyse_holidays(x.year)


def next_td(x):
    while not is_td(x):
        x += dt.timedelta(days=1)
    return x


def effective_session(report_date):
    """First trading session a Tuesday report could have been used (see TIMING)."""
    friday = report_date + dt.timedelta(days=(4 - report_date.weekday()) % 7)
    monday = report_date - dt.timedelta(days=report_date.weekday())
    hol = federal_holidays(report_date.year) | federal_holidays(friday.year)
    late = any(monday + dt.timedelta(days=i) in hol for i in range(5))
    s = next_td(friday + dt.timedelta(days=1))
    return next_td(s + dt.timedelta(days=1)) if late else s


def rows_for(src, contract):
    """One row per report: the main contract (Russell: the venue with more notional open interest that week)
    with its Micro folded in at 1/10 (a Micro is a tenth of an E-mini), positions and open interest alike."""
    rows = json.load(open(os.path.join(CACHE, f"tff_{src}.json"), encoding="utf-8"))
    codes, micro = CONTRACTS[contract], MICRO.get(contract)
    by_week, micros = {}, {}
    for r in rows:
        c = r["cftc_contract_market_code"]
        if c == micro:
            micros[r["report_date"]] = r
            continue
        if c not in codes:
            continue
        notional = float(r.get("open_interest_all") or 0) * MULT.get(c, 1.0)
        cur = by_week.get(r["report_date"])
        if cur is None or notional > cur[0]:
            by_week[r["report_date"]] = (notional, r)
    out = []
    for k in sorted(by_week):
        r = dict(by_week[k][1])
        m = micros.get(k)
        if m:
            for f in [x for pair in GROUPS.values() for x in pair] + ["open_interest_all"]:
                r[f] = float(r.get(f) or 0) + 0.1 * float(m.get(f) or 0)
            r["with_micro"] = True
        out.append(r)
    return out


def roll_week(d):
    """Report dates from 10 days before to 3 days after a quarterly index expiry (third Friday of Mar/Jun/Sep/Dec):
    the roll moves open interest and spreads between contract months."""
    for y in (d.year - 1, d.year, d.year + 1):
        for mth in (3, 6, 9, 12):
            x = dt.date(y, mth, 15)
            third_fri = x + dt.timedelta(days=(4 - x.weekday()) % 7)
            if -10 <= (d - third_fri).days <= 3:
                return True
    return False


def rolling_rank(vals, i):
    """Rank of vals[i] among the trailing WINDOW values up to and including i (0..1), None if too few."""
    w = [v for v in vals[max(0, i - WINDOW + 1): i + 1] if v is not None]
    if len(w) < MIN_WEEKS or vals[i] is None:
        return None
    below = sum(v < vals[i] for v in w)
    equal = sum(v == vals[i] for v in w)
    return (below + 0.5 * (equal - 1)) / (len(w) - 1)


def weekly():
    out = []
    for contract in CONTRACTS:
        per_src = {}
        for src, tag in (("futures", "fut"), ("combined", "cmb")):
            rs = rows_for(src, contract)
            feats = {}
            for g, (lk, sk) in GROUPS.items():
                net = [float(r.get(lk) or 0) - float(r.get(sk) or 0) for r in rs]
                oi = [float(r.get("open_interest_all") or 0) for r in rs]
                no = [n / o if o > 0 else None for n, o in zip(net, oi)]
                chg = [no[i] - no[i - 4] if i >= 4 and no[i] is not None and no[i - 4] is not None else None for i in range(len(no))]
                for i, r in enumerate(rs):
                    f = feats.setdefault(r["report_date"], {"oi_" + tag: oi[i], "venue_" + tag: r["cftc_contract_market_code"]})
                    f.update({f"{g}_net_{tag}": net[i], f"{g}_net_oi_{tag}": no[i], f"{g}_pct156_{tag}": rolling_rank(no, i),
                              f"{g}_chg4_{tag}": chg[i], f"{g}_chg4_pct_{tag}": rolling_rank(chg, i)})
            per_src[tag] = feats
        for rd in sorted(set(per_src["fut"]) | set(per_src["cmb"])):
            d = dt.date.fromisoformat(rd)
            row = {"contract": contract, "report_date": d, "effective": effective_session(d),
                   "release_uncertain": any(a <= d <= b for a, b in SHUTDOWN_REPORTS),
                   "roll_week": roll_week(d)}
            row.update(per_src["fut"].get(rd, {}))
            row.update(per_src["cmb"].get(rd, {}))
            out.append(row)
    return pl.DataFrame(out, infer_schema_length=None).sort(["contract", "report_date"])


def daily(wk):
    first, last = wk["effective"].min(), dt.date.today() - dt.timedelta(days=1)
    span = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    days = pl.DataFrame({"date": [x for x in span if is_td(x)]}).with_columns(pl.col("date").cast(pl.Date))
    parts = []
    for c in CONTRACTS:
        w = wk.filter(pl.col("contract") == c).sort("effective")
        j = days.join_asof(w, left_on="date", right_on="effective", strategy="backward")
        j = j.with_columns(pl.lit(c).alias("contract"),
                           (pl.col("date") - pl.col("report_date")).dt.total_days().alias("report_age_days"))
        parts.append(j.with_columns(((pl.col("report_age_days") <= MAX_AGE_DAYS)
                                     & ~pl.col("release_uncertain").fill_null(True)).alias("usable")))
    return pl.concat(parts, how="diagonal").filter(pl.col("report_date").is_not_null()).sort(["contract", "date"])


def main():
    wk = weekly()
    dy = daily(wk)
    wk.write_parquet(os.path.join(CACHE, "cot_weekly.parquet"))
    dy.write_parquet(os.path.join(CACHE, "cot_daily.parquet"))
    print(f"weekly {wk.height} rows x {wk.width} cols | daily {dy.height} rows")
    for c in CONTRACTS:
        w = wk.filter(pl.col("contract") == c)
        ok = w.filter(pl.col("lev_money_pct156_fut").is_not_null())
        print(f"  {c:4s} reports {w.height} ({w['report_date'].min()}..{w['report_date'].max()}), "
              f"percentiles from {ok['report_date'].min()} | venues {sorted(set(w['venue_fut'].drop_nulls().to_list()))}")


if __name__ == "__main__":
    main()
