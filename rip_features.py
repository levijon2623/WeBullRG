"""
rip_features.py -- the quiet-stretch -> rip inputs, ONE implementation shared by the research screen
(check_rip_screen.py, scored 2026-11-26) and the live RIP pill (strike_cue.py), so the pill computes exactly what was
tested. Pure functions over arrays and small dicts; no I/O.

    scan(m, c, t30, upto)      the BASE rule of check_absorption_rip.bases on the bars known so far (minute <= upto):
                               confirmed bases with their state (ongoing / broken + side)
    features(r, mode, ctx)     the inputs of one base at its stamp: mode "conf" (the 60th minute) or "brk" (the first
                               minute outside the band)

ctx keys: a {tk: bars (m, open, high, low, close, volume)}, pa {tk: prior-session bars}, t30 {tk: typical 30-min range},
base_vol {tk: array of prior-20 cumulative volume curves by minute}, expo {tk: {minute: (gamma, charm, vanna)}},
opt {tk: (flow {minute: (np, cv, pv, dd)}, atm {minute: (straddle mid, iv)})}, B / S (THRU / PULL events: t, t_in,
side, score), pairs {tk: [(t_in, side)]}, hrel [(tk, minute, D)] or None, peak {tk: (zone mid, sign)},
base_now {tk: [(s, e or None while ongoing)]}.
"""
from __future__ import annotations

import numpy as np

BAND_T30, START_MIN, START_MAX, END_MAX, BASE_MIN = 2.5, 10 * 60, 14 * 60, 15 * 60, 60
TKS = ("SPY", "QQQ", "IWM")
DIRECTIONAL = ["SLOPE15", "POS", "EDGE_NET", "UPVOL", "HOLD", "DOPEN", "DVWAP", "DPREV", "PRE", "GAP", "PREVRET",
               "XMOVE15", "CHARM", "VANNA", "DD", "NETPREM", "CPTILT", "HREL", "PEAKD", "THRU_NET", "THRU_LAST",
               "PULL_NET", "PULL_LAST", "TIE_LAST"]


def last_at(dct, m, back=10):
    for k in range(m, m - back - 1, -1):
        if k in dct and dct[k] is not None:
            return dct[k]
    return None


def scan(m, c, t30, upto):
    """check_absorption_rip.bases' base rule on bars with minute <= upto. Returns [{s, e, state, side, brk}]: state
    'ongoing' (still inside its band at the last bar), 'broken' (brk = the break minute, side +1 / -1) or 'capped' (ran
    to 15:00). A start is only judged once all its 60 minutes are known, so a live scan finds the same bases as the
    whole-day one."""
    m = np.asarray(m)
    c = np.asarray(c)
    keep = m <= upto
    m, c = m[keep], c[keep]
    out, i, n = [], int(np.searchsorted(m, START_MIN)), len(m)
    while i < n and m[i] <= START_MAX:
        if upto < m[i] + BASE_MIN - 1:
            break                                           # this start's 60 minutes are not all known yet
        j = int(np.searchsorted(m, m[i] + BASE_MIN)) - 1
        if j >= n or m[j] - m[i] < BASE_MIN - 5:
            i += 1
            continue
        band = BAND_T30 * t30 * c[i]
        if c[i:j + 1].max() - c[i:j + 1].min() > band:
            i += 1
            continue
        while j + 1 < n and m[j + 1] <= END_MAX and c[i:j + 2].max() - c[i:j + 2].min() <= band:
            j += 1
        hi, lo = c[i:j + 1].max(), c[i:j + 1].min()
        if j + 1 < n and m[j + 1] <= END_MAX:
            k = j + 1
            out.append(dict(s=int(m[i]), e=int(m[j]), state="broken", side=1 if c[k] > hi else -1, brk=int(m[k])))
        elif j + 1 < n:
            out.append(dict(s=int(m[i]), e=int(m[j]), state="capped", side=0, brk=None))
        else:
            out.append(dict(s=int(m[i]), e=int(m[j]), state="ongoing", side=0, brk=None))
            break
        i = j + 1
    return out


def features(r, mode, ctx):
    """inputs of base r = {tk, s, e, t30} at its stamp; None if the stamp minute is missing / the base did not break."""
    tk, s, e = r["tk"], r["s"], r["e"]
    a, pa, t30 = ctx["a"], ctx["pa"], ctx["t30"]
    x = a.get(tk)
    if x is None or t30.get(tk) is None:
        return None
    m = x[:, 0].astype(int)
    idx = {mm: i for i, mm in enumerate(m)}
    ie, is_ = idx.get(e), idx.get(s)
    if ie is None or is_ is None:
        return None
    c, o, h, lo_, v = x[:, 4], x[:, 1], x[:, 2], x[:, 3], x[:, 5]
    btop, bbot = c[is_: ie + 1].max(), c[is_: ie + 1].min()
    if mode == "conf":
        t = s + 59                                   # the base is CONFIRMED here; its end is not known yet
        it = idx.get(t)
        if it is None:
            return None
        iw, bs = it, 0                               # inputs see minutes s .. t
    else:
        t = e + 1                                    # the first minute after the base: the BREAK, known now
        it = idx.get(t)
        if it is None or not (c[it] > btop or c[it] < bbot):
            return None                              # ended on the 15:00 cap or a gap in bars, not a break
        iw, bs = ie, (1 if c[it] > btop else -1)     # band from s .. e; price inputs at the break minute t
    unit = r["t30"] * c[it]
    w = slice(is_, iw + 1)
    top, bot = c[w].max(), c[w].min()
    rng = max(top - bot, 1e-9)
    i15 = idx.get(t - 15, max(0, it - 15))
    j0 = max(is_, 1)
    dc, vv = c[j0: iw + 1] - c[j0 - 1: iw], v[j0: iw + 1]       # each base minute's close vs the minute before
    vw = (c[: it + 1] * v[: it + 1]).sum() / max(v[: it + 1].sum(), 1)
    pcl = pa.get(tk)[-1, 4] if pa.get(tk) is not None else np.nan
    pret = ((pa[tk][-1, 4] - pa[tk][0, 1]) / (r["t30"] * pa[tk][0, 1])) if pa.get(tk) is not None else np.nan
    mw = m[w]
    first30, last30 = v[w][mw < s + 30].sum(), v[w][mw > m[iw] - 30].sum()
    ex = last_at(ctx["expo"].get(tk, {}), t)
    fl, atm = ctx["opt"].get(tk, ({}, {}))
    npv = [fl[k] for k in range(s, m[iw] + 1) if k in fl]
    f = dict(tk=tk, s=s, e=e, t=t, BS=bs,
             L=float(m[iw] - s + 1), VD=float(last30 / first30) if first30 > 0 else np.nan,
             G=ex[0] if ex else np.nan, DD=float(sum(q[3] for q in npv)) if npv else np.nan,
             SLOPE15=(c[it] - c[i15]) / unit, POS=2 * (c[it] - bot) / rng - 1,
             EDGE_NET=float((h[w] >= top - 0.1 * rng).sum() - (lo_[w] <= bot + 0.1 * rng).sum()),
             UPVOL=float((np.sign(dc) * vv).sum() / max(vv.sum(), 1)) if len(dc) else np.nan,
             HOLD=float((c[w] > o[w]).mean() - (c[w] < o[w]).mean()),
             DOPEN=(c[it] - o[0]) / unit, DVWAP=(c[it] - vw) / unit, DPREV=(c[it] - pcl) / unit, PRE=(c[is_] - o[0]) / unit,
             GAP=(o[0] - pcl) / unit, PREVRET=pret, TIGHT=rng / unit, TOD=float(t))
    others = []
    for t2 in TKS:
        if t2 == tk or a.get(t2) is None or not t30.get(t2):
            continue
        m2 = {mm: i for i, mm in enumerate(a[t2][:, 0].astype(int))}
        j, j15 = m2.get(t), m2.get(t - 15)
        if j is not None and j15 is not None:
            others.append((a[t2][j, 4] - a[t2][j15, 4]) / (t30[t2] * a[t2][j, 4]))
    f["XMOVE15"] = float(np.mean(others)) if others else np.nan
    # another ticker is "in a base" at t when its base is >= 60 min old and its closes are still inside the band at t --
    # s2 + 59 <= t <= e2 (a base extends exactly while its closes stay inside; e2 None = still ongoing)
    bn = ctx["base_now"]
    f["XQUIET"] = float(sum(any(s2 + 59 <= t and (e2 is None or t <= e2) for s2, e2 in bn.get(t2, [])) for t2 in TKS if t2 != tk))
    bv = ctx["base_vol"].get(tk)
    f["RVOL"] = float(np.cumsum(v[: it + 1])[-1] / np.nanmedian(bv[:, t])) if bv is not None and np.isfinite(np.nanmedian(bv[:, t])) else np.nan
    f["CHARM"], f["VANNA"] = (ex[1], ex[2]) if ex else (np.nan, np.nan)
    f["NETPREM"] = float(sum(q[0] for q in npv)) if npv else np.nan
    cvs, pvs = sum(q[1] for q in npv), sum(q[2] for q in npv)
    f["CPTILT"] = (cvs - pvs) / (cvs + pvs) if cvs + pvs > 0 else np.nan
    a0, a1 = last_at(atm, s, 5), last_at(atm, t, 5)
    f["IVCHG"] = (a1[1] - a0[1]) if a0 and a1 and a0[1] and a1[1] else np.nan
    f["STRADCHG"] = (a1[0] / a0[0] - 1) if a0 and a1 and a0[0] else np.nan
    hr = ctx.get("hrel")
    f["HREL"] = float(sum(D for t2, mm, D in hr if t2 == tk and s <= mm <= t)) if hr is not None else np.nan
    pk = ctx["peak"].get(tk)
    f["PEAKD"], f["PEAKSIGN"] = ((c[it] - pk[0]) / unit, float(pk[1])) if pk else (np.nan, np.nan)
    t0, t1 = s * 60, (t + 1) * 60
    bb = sorted([b for b in ctx["B"] if b["t"] == tk and t0 <= b["t_in"] <= t1], key=lambda b: b["t_in"])
    ss = sorted([q for q in ctx["S"] if q["t"] == tk and t0 <= q["t_in"] <= t1], key=lambda q: q["t_in"])
    pp = sorted([q for q in ctx["pairs"].get(tk, []) if t0 <= q[0] <= t1])
    f.update(THRU_N=float(len(bb)), THRU_NET=float(sum(b["side"] for b in bb)),
             THRU_LAST=float(bb[-1]["side"] * bb[-1]["score"]) if bb else 0.0,
             PULL_N=float(len(ss)), PULL_NET=float(sum(q["side"] for q in ss)),
             PULL_LAST=float(ss[-1]["side"] * ss[-1]["score"]) if ss else 0.0,
             TIE_N=float(len(pp)), TIE_LAST=float(pp[-1][1]) if pp else 0.0)
    if mode == "conf":
        j60 = int(np.searchsorted(m, t + 60, side="right")) - 1
        f["FWD_A"] = float((c[j60] - c[it]) / unit) if j60 > it else np.nan
    else:
        j60 = int(np.searchsorted(m, t + 60, side="right")) - 1
        fut = c[it + 1: j60 + 1]
        f["RIP_B"] = float(np.max(bs * (fut - c[it])) >= 3 * unit) if len(fut) else np.nan
        f["CONT_B"] = float(bs * (c[j60] - c[it]) / unit) if j60 > it else np.nan
        for k in DIRECTIONAL:                        # "+" = WITH the break for every directional input
            if f.get(k) is not None and f[k] == f[k]:
                f[k] = float(f[k]) * bs
    return f
