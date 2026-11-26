"""
explore_hedge_release.py -- EXPLORATION, steps 1-2 of the "hedges trimmed -> dealers buy back" idea (user, 2026-11-26).
DESCRIPTIVE, DISCOVERY MONTHS ONLY (odd calendar months): the even months stay untouched for a later pre-registered
holdout. Nothing here is a test.

THE IDEA. A fund that bought index puts from dealers leaves the dealers short puts, hedged short stock. When the fund
sells those puts back (a trim) or rolls them down to lower strikes, the dealers' hedge is too big and they buy stock
back: quiet buying that need not show as aggressive futures flow -- one candidate for "absorbed, then a rip" (11-25:
the 44.5k-lot SPY desk re-struck Dec put spreads 15 points lower at 10:51-10:58, inside the base).

NOTE ON THE MECHANICS. The dealer's re-hedge depends on the TRADE, not on whether it opened or closed: any put the
customer SELLS (to a delta-hedging dealer) makes the dealer buy |delta| x size x 100 shares. Open/close (from the next
morning's open interest) says WHO is acting -- a hedger trimming (sold to close) vs a put writer (sold to open) -- which
is the part of the story this step checks.

STEP 1  EVENTS from the lake tape (trades-core: every SPY / QQQ / IWM option print with the NBBO at that moment).
        Put prints >= LEG_MIN contracts, 09:30-16:15 ET, grouped per underlying and second; an EVENT is a group of
        >= EVENT_MIN contracts in total. Legs with |delta| >= 0.9 (deep ITM: dividend / box / financing trades) dropped,
        and groups containing calls in the same second are flagged (conversions, risk reversals) and set aside.
        Leg side: price >= ask = customer BOUGHT, <= bid = SOLD, else MID (complex orders often print inside the NBBO,
        so a package's legs may be unsided).
STEP 2  OI CONFIRMATION: open interest at the start of the next session minus this session's start (UW per-contract OI
        is start-of-day). A leg CLOSED if dOI <= -CONF x size, OPENED if dOI >= +CONF x size, else unclear (other
        trading in the contract the same day blurs it; the leg's share of the contract's day volume is kept).
        HEDGE RELEASE = an event whose sided legs, signed by the customer, SELL net put delta (dealer buys shares), and
        whose sold legs the OI says were CLOSED (a hedge trimmed, not new puts written).

    python explore_hedge_release.py            (builds _hedge_cache/, prints the descriptive tables)
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
LAKE = os.path.join(ROOT, "lake", "silver")
CACHE = os.path.join(ROOT, "_hedge_cache")
TKS = ("SPY", "QQQ", "IWM")
LEG_MIN, EVENT_MIN, CONF, DEEP = 500, 5000, 0.5, 0.9


def occ(u, e, k, r):
    return f"{u}{e:%y%m%d}{r}{int(round(k * 1000)):08d}"


def contract_day(d):
    p = os.path.join(LAKE, "option-contracts-1m", f"date={d}", "bars.parquet")
    if not os.path.exists(p):
        return None
    return (pl.scan_parquet(p).filter(pl.col("underlying_symbol").is_in(list(TKS)) & (pl.col("option_type") == "put"))
            .select("option_chain_id", "minute_et", "delta_close", "open_interest", "volume", "underlying_close"))


def one_day(args):
    d, nxt = args
    cp = os.path.join(CACHE, f"legs-{d}.parquet")
    if os.path.exists(cp):
        return cp
    tp = os.path.join(LAKE, "trades-core", f"date={d}", "trades.parquet")
    cd, cn = contract_day(d), contract_day(nxt) if nxt else None
    if not os.path.exists(tp) or cd is None:
        return None
    t = (pl.scan_parquet(tp)
         .filter(pl.col("underlying_symbol").is_in(list(TKS)) & (pl.col("size") >= LEG_MIN)
                 & (pl.col("canceled").is_null() | (pl.col("canceled") != "true")))
         .with_columns(et=pl.col("executed_at").dt.convert_time_zone("America/New_York"))
         .with_columns(mod=pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32),
                       sec=pl.col("et").dt.truncate("1s"))
         .filter((pl.col("mod") >= 570) & (pl.col("mod") < 975)).collect())
    if not t.height:
        return None
    calls_at = set(t.filter(pl.col("option_type") == "call").select("underlying_symbol", "sec").iter_rows())
    t = t.filter(pl.col("option_type") == "put")
    t = t.with_columns(
        cid=pl.struct("underlying_symbol", "expiry", "strike").map_elements(
            lambda s: occ(s["underlying_symbol"], s["expiry"], s["strike"], "P"), return_dtype=pl.String),
        side=pl.when(pl.col("price") >= pl.col("nbbo_ask")).then(1).when(pl.col("price") <= pl.col("nbbo_bid")).then(-1)
        .otherwise(0).cast(pl.Int8),
        mleg=pl.col("upstream_condition_detail").fill_null("").str.starts_with("ml"))
    ids = t["cid"].unique().to_list()
    info = (cd.filter(pl.col("option_chain_id").is_in(ids))
            .with_columns(mod=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
            .collect())
    oi0 = info.group_by("option_chain_id").agg(pl.col("open_interest").drop_nulls().first().alias("oi0"),
                                               pl.col("volume").sum().alias("dayvol"))
    if cn is not None:
        oi1 = (cn.filter(pl.col("option_chain_id").is_in(ids)).group_by("option_chain_id")
               .agg(pl.col("open_interest").drop_nulls().first().alias("oi1")).collect())
    else:
        oi1 = pl.DataFrame({"option_chain_id": [], "oi1": []}, schema={"option_chain_id": pl.String, "oi1": pl.Int64})
    dl = info.select("option_chain_id", "mod", "delta_close", "underlying_close").drop_nulls("delta_close")
    t = (t.join(oi0, left_on="cid", right_on="option_chain_id", how="left")
         .join(oi1, left_on="cid", right_on="option_chain_id", how="left")
         .sort("mod").join_asof(dl.sort("mod"), on="mod", by_left="cid", by_right="option_chain_id", strategy="backward"))
    t = t.with_columns(call_same_sec=pl.struct("underlying_symbol", "sec").map_elements(
        lambda s: (s["underlying_symbol"], s["sec"]) in calls_at, return_dtype=pl.Boolean), d=pl.lit(d))
    t = t.select("d", "underlying_symbol", "sec", "mod", "cid", "expiry", "strike", "size", "price", "nbbo_bid", "nbbo_ask",
                 "side", "mleg", "delta_close", "underlying_close", "oi0", "oi1", "dayvol", "call_same_sec")
    os.makedirs(CACHE, exist_ok=True)
    t.write_parquet(cp)
    return cp


def build():
    days = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "trades-core", "date=*")))
    cdays = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(LAKE, "option-contracts-1m", "date=*")))
    nxt = {d: next((x for x in cdays if x > d), None) for d in days}
    disc = [d for d in days if int(d[5:7]) % 2 == 1]          # DISCOVERY ONLY -- never an even month
    with ProcessPoolExecutor(6) as ex:
        paths = [p for p in ex.map(one_day, [(d, nxt[d]) for d in disc]) if p]
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def main():
    L = build()
    assert ((L["d"].str.slice(5, 2).cast(pl.Int32) % 2) == 1).all(), "an even month got in"
    L = L.with_columns(deep=pl.col("delta_close").abs() >= DEEP, dOI=pl.col("oi1") - pl.col("oi0"))
    n_s = L["d"].n_unique()
    print(f"DISCOVERY months: {n_s} sessions, {L.height:,} put legs >= {LEG_MIN} contracts "
          f"({L['delta_close'].is_null().mean():.1%} without a delta match, {L['oi1'].is_null().mean():.1%} without next-day OI)")
    ev = (L.filter(~pl.col("deep") & pl.col("delta_close").is_not_null())
          .group_by("d", "underlying_symbol", "sec")
          .agg(pl.col("size").sum().alias("n"), pl.len().alias("legs"), pl.col("call_same_sec").any().alias("calls"),
               pl.col("mleg").any().alias("mleg"),
               (pl.col("side") != 0).mean().alias("sided"),
               # customer-signed put delta: a SOLD put adds +|delta| to the customer -> the dealer buys that many shares
               (-(pl.col("side").cast(pl.Float64)) * pl.col("delta_close").abs() * pl.col("size") * 100).sum().alias("dealer_buy_sh"),
               (pl.col("delta_close").abs() * pl.col("size") * 100 * pl.col("underlying_close")).sum().alias("delta_usd"),
               # OI-confirmed trimming: sold legs that CLOSED / sold legs with OI
               ((pl.col("side") == -1) & (pl.col("dOI") <= -CONF * pl.col("size"))).sum().alias("sold_closed"),
               ((pl.col("side") == -1) & (pl.col("dOI") >= CONF * pl.col("size"))).sum().alias("sold_opened"),
               ((pl.col("side") == 1) & (pl.col("dOI") <= -CONF * pl.col("size"))).sum().alias("bought_closed"),
               ((pl.col("side") == 1) & (pl.col("dOI") >= CONF * pl.col("size"))).sum().alias("bought_opened"),
               ((pl.col("side") == 0) & (pl.col("dOI") <= -CONF * pl.col("size"))).sum().alias("mid_closed"),
               ((pl.col("side") == 0) & (pl.col("dOI") >= CONF * pl.col("size"))).sum().alias("mid_opened"),
               (pl.col("side") == -1).sum().alias("sold_legs"), (pl.col("side") == 1).sum().alias("bought_legs"),
               (pl.col("side") == 0).sum().alias("mid_legs"), pl.col("mod").first())
          .filter(pl.col("n") >= EVENT_MIN))
    print(f"\nSTEP 1 -- events (>= {EVENT_MIN:,} put contracts in one second, deep-ITM legs dropped): {ev.height:,}; "
          f"{ev.filter(pl.col('calls')).height:,} with calls in the same second set aside")
    E = ev.filter(~pl.col("calls"))
    for tk in TKS:
        x = E.filter(pl.col("underlying_symbol") == tk)
        if not x.height:
            continue
        print(f"  {tk}: {x.height / n_s:.2f} events/session | multi-leg {x['mleg'].mean():.0%} | legs sided "
              f"{x['sided'].mean():.0%} (fully sided events {(x['sided'] == 1).mean():.0%}, none sided {(x['sided'] == 0).mean():.0%})"
              f" | delta notional median ${x['delta_usd'].median() / 1e6:,.0f}M, p90 ${x['delta_usd'].quantile(0.9) / 1e6:,.0f}M")
    tot = E.select([pl.col(c).sum() for c in ("sold_legs", "bought_legs", "mid_legs", "sold_closed", "sold_opened",
                                              "bought_closed", "bought_opened", "mid_closed", "mid_opened")]).row(0, named=True)
    print("\nSTEP 2 -- next-morning OI on the legs of those events (closed = OI fell by >= half the leg, opened = rose):")
    for s in ("sold", "bought", "mid"):
        n = tot[f"{s}_legs"]
        print(f"  {s:6s} legs {n:6,d}: closed {tot[f'{s}_closed'] / max(n, 1):.0%}  opened {tot[f'{s}_opened'] / max(n, 1):.0%}"
              f"  unclear {1 - (tot[f'{s}_closed'] + tot[f'{s}_opened']) / max(n, 1):.0%}")
    lg = L.filter(~pl.col("deep") & pl.col("oi1").is_not_null() & pl.col("dayvol").is_not_null())
    lg = lg.with_columns(share=pl.col("size") / pl.col("dayvol"))
    for lab, c in (("leg >= 50% of its contract's day volume", pl.col("share") >= 0.5), ("leg < 50%", pl.col("share") < 0.5)):
        y = lg.filter(c)
        if y.height:
            cl = (y["dOI"] <= -CONF * y["size"]).mean(); op = (y["dOI"] >= CONF * y["size"]).mean()
            print(f"  {lab}: {y.height:,} legs, OI says closed {cl:.0%} / opened {op:.0%} / unclear {1 - cl - op:.0%}")
    E = E.with_columns(release=(pl.col("dealer_buy_sh") > 0) & (pl.col("sold_closed") > 0) & (pl.col("sold_opened") == 0),
                       add=(pl.col("dealer_buy_sh") < 0) & (pl.col("bought_opened") > 0) & (pl.col("bought_closed") == 0))
    print("\nHEDGE RELEASE events (customer net SOLD put delta, sold legs OI-confirmed CLOSED) vs HEDGE ADDS (net bought, "
          "bought legs OPENED):")
    for tk in TKS:
        x = E.filter(pl.col("underlying_symbol") == tk)
        r, a = x.filter(pl.col("release")), x.filter(pl.col("add"))
        sided = x.filter(pl.col("sided") > 0)
        print(f"  {tk}: releases {r.height} ({r.height / n_s:.2f}/session, {r['d'].n_unique() / n_s:.0%} of sessions), "
              f"median dealer buy {r['dealer_buy_sh'].median() or 0:,.0f} sh | adds {a.height} "
              f"| of {sided.height} events with any sided leg: {r.height / max(sided.height, 1):.0%} release, {a.height / max(sided.height, 1):.0%} add")
    r = E.filter(pl.col("release"))
    if r.height:
        h = (r["mod"] // 60).value_counts().sort("mod")
        print("  releases by hour:", ", ".join(f"{int(a)}h {b}" for a, b in h.iter_rows()))
        print("  biggest releases (dealer buy, shares):")
        for row in r.sort("dealer_buy_sh", descending=True).head(12).iter_rows(named=True):
            print(f"    {row['d']} {row['sec']:%H:%M:%S} {row['underlying_symbol']} {row['n']:,} contracts in {row['legs']} legs, "
                  f"dealer buy ~{row['dealer_buy_sh']:,.0f} sh, delta ${row['delta_usd'] / 1e6:,.0f}M, sided {row['sided']:.0%}")


if __name__ == "__main__":
    main()
