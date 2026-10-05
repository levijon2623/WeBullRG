"""
gamma_flip.py
=============
The GAMMA FLIP from our own chain data -- the level where dealers' net gamma
changes sign -- so the Webull-data viewer can show what only UW's
/gex-levels gave it. Pure functions; the caller fetches the chain.

Method (user, 2026-10-03; the SpotGamma / SqueezeMetrics convention):
  * dealers are LONG calls and SHORT puts:
        net GEX(S) = sum over contracts of  sign x OI x gamma(S) x 100 x S^2 x 1%
        sign = +1 call, -1 put  -- UW's units ($ per 1% move)
  * gamma(S) is RECOMPUTED at every hypothetical spot S with Black-Scholes,
    holding each contract's IV fixed (sticky strike). A snapshot's gamma
    column is only valid at the CURRENT spot, so it cannot draw the curve.
  * the flip is where the curve crosses zero. The curve can cross more than
    once (UW returns five `nearby_flips`); ALL crossings are returned and the
    one NEAREST SPOT is "the" flip -- the same choice UW's own gamma_flip
    made in every logged poll inspected (2026-10-02, e.g. SPY 767.56 with spot
    769.24 among 767.56 / 766.33 / 765.46 / 698.52 / 697.55).

Units are the heat's, so the curve's level is comparable with the viewer's
0-1DTE heat. Rates and dividends are ignored (gamma barely moves with them at
these tenors).
"""
from __future__ import annotations

import numpy as np

T_FLOOR_YEARS = 30 / 525600      # 30 minutes: a 0DTE contract's gamma would
                                 # otherwise spike without bound into the close
GRID_HALF = 0.08                 # simulate spot +-8%
GRID_STEP = 0.001                # in 0.1% steps (161 points)


def chain_arrays(rows):
    """[(strike, T_years, iv, oi, is_call)] -> numpy arrays, junk dropped."""
    a = np.array([(k, t, v, o, 1.0 if c else -1.0) for k, t, v, o, c in rows
                  if k and t is not None and v and o and 0.01 < v < 5.0 and o > 0],
                 dtype=float).reshape(-1, 5)
    K, T, iv, oi, sg = a.T
    return K, np.maximum(T, T_FLOOR_YEARS), iv, oi, sg


def gex_curve(K, T, iv, oi, sg, spots):
    """Net GEX at each hypothetical spot -> (len(spots),). Vectorised over
    (spots x contracts)."""
    S = np.asarray(spots, float)[:, None]
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd * sd) / sd
    gamma = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * sd)
    return (sg * oi * gamma).sum(axis=1) * 100 * spots ** 2 * 0.01


def crossings(spots, curve):
    """Every zero crossing, linearly interpolated, ascending."""
    out = []
    for i in range(len(spots) - 1):
        a, b = curve[i], curve[i + 1]
        if a == 0:
            out.append(float(spots[i]))
        elif a * b < 0:
            out.append(float(spots[i] + (spots[i + 1] - spots[i]) * a / (a - b)))
    return out


def flip_of(rows, spot):
    """-> dict(flip, crossings, net_at_spot, call_wall, put_wall, n, curve)
    or None when the chain is empty."""
    if not rows or not spot:
        return None
    K, T, iv, oi, sg = chain_arrays(rows)
    if K.size == 0:
        return None
    spots = spot * (1 + np.arange(-GRID_HALF, GRID_HALF + GRID_STEP / 2, GRID_STEP))
    curve = gex_curve(K, T, iv, oi, sg, spots)
    xs = crossings(spots, curve)
    # walls at the CURRENT spot, all expiries in the chain: call = biggest
    # call gamma strike, put = biggest put gamma strike
    sd = iv * np.sqrt(T)
    d1 = (np.log(spot / K) + 0.5 * sd * sd) / sd
    g = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (spot * sd) * oi * 100 * spot * spot * 0.01
    by = {}
    for k, gg, s in zip(K, g, sg):
        c = by.setdefault(float(k), [0.0, 0.0])
        c[0 if s > 0 else 1] += gg
    call_wall = max(by, key=lambda k: by[k][0]) if by else None
    put_wall = max(by, key=lambda k: by[k][1]) if by else None
    net_now = float(gex_curve(K, T, iv, oi, sg, np.array([spot]))[0])
    return dict(flip=min(xs, key=lambda x: abs(x - spot)) if xs else None,
                crossings=[round(x, 2) for x in xs], net_at_spot=net_now,
                call_wall=call_wall, put_wall=put_wall, n=int(K.size),
                curve=list(zip(np.round(spots, 2).tolist(), np.round(curve).tolist())))
