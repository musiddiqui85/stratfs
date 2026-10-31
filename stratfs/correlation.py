"""The three fitted forms for the annual solar fraction of solar process-water heating with stratified storage.

    explicit  annual balance, return temperature as a fixed function of the collector-to-storage ratio
    implicit  monthly balance with the feedback of the solar fraction on the return temperature
    final     the stratified-storage correlation: the implicit form plus curtailment of a slow charging stream
              and the charging-flow exponent (the form recommended for use)

All inputs are arrays of equal length (or scalars), one design per element; `site` may be one name or one per
design. The groups are Phi_sys, Psi_sys, B_L, R_ch, H_T and kappa_s of the dimensionless model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from stratfs.data import (SCHEDULES, T_BASE, MonthlyClimate, constants, sites, temperature_scale, utilizability)

FORMS = ("explicit", "implicit", "final")
NIGHT = "night_20_06"
ITERATIONS = 12
ACTIVE_FRACTION = {"day_08_18": 10 / 24, "night_20_06": 10 / 24, "continuous_24h": 1.0}


def smin(a, b, n=8.0):
    """Smooth minimum (a^-n + b^-n)^(-1/n) of positive arguments."""
    return (a ** -n + b ** -n) ** (-1.0 / n)


def expected_min(Y, K, sigma):
    """E[min(Y Z, K)] for a lognormal daily multiplier Z of mean 1 and log-standard deviation sigma.
    Floors: Y, K >= 1e-9 and sigma >= 1e-6."""
    Y, K, sigma = np.maximum(Y, 1e-9), np.maximum(K, 1e-9), np.maximum(sigma, 1e-6)
    u = (np.log(K / Y) + 0.5 * sigma ** 2) / sigma
    return Y * norm.cdf(u - sigma) + K * (1.0 - norm.cdf(u))


def phibar(X, KT, ratio, slope):
    """Klein's monthly-average daily utilizability; for X <= 0 the linear continuation 1 - X slope, and for
    c_K < 0 the quadratic is held at its vertex X = -1/(2 c_K)."""
    a = 2.943 - 9.271 * KT + 4.031 * KT ** 2
    b = -4.345 + 8.853 * KT - 3.602 * KT ** 2
    c = -0.170 - 0.306 * KT + 2.936 * KT ** 2
    Xq = np.where(c < 0, np.minimum(X, -0.5 / np.minimum(c, -1e-9)), X)
    pos = np.exp(np.minimum((a + b * ratio) * (Xq + c * Xq ** 2), 0.0))
    return np.where(X > 0, pos, 1.0 - X * slope)


def _phi_site(names, x):
    """phi_site(x) by linear interpolation; for x < 0, 1 - x D with D the daylight fraction of the year."""
    U = utilizability()
    grid = U.columns.to_numpy(float)
    P = U.loc[names].to_numpy()
    xc = np.clip(x, 0.0, grid[-1])
    k = np.clip(np.searchsorted(grid, xc) - 1, 0, len(grid) - 2)
    w = (xc - grid[k]) / (grid[k + 1] - grid[k])
    rows = np.arange(len(x))
    val = (1 - w) * P[rows, k] + w * P[rows, k + 1]
    return np.where(x < 0, 1.0 - x * sites().loc[names].daylight.to_numpy(), val)


def _explicit(g, names, schedule, T_process, p):
    s = sites().loc[names]
    theta_a_w = s.theta_a_w_80.to_numpy() * s.dT_L_80_K.to_numpy() / temperature_scale(names, T_process)
    gamma = g["Phi_sys"] / (g["B_L"] * (1 + g["HT"]))
    theta_r = p["a0"] + p["a1"] * gamma ** p["a2"] / (1 + gamma ** p["a2"])
    Y = (g["Phi_sys"] * _phi_site(names, g["Psi_sys"] * (theta_r - theta_a_w))
         - g["kappa_s"] * g["B_L"] * (p["l1"] + theta_r))
    direct = 0.0 if schedule == NIGHT else p["om"]
    cap = smin(direct + p["kc"] * g["B_L"] * (1 + g["HT"]), np.ones(len(names)))
    return np.clip(expected_min(Y, cap, p["ks"] * s.daily_sd_ln.to_numpy()), 0.0, 1.0), None, None


def _monthly(g, names, schedule, T_process, p, f_L):
    """Monthly balance of the implicit form (no k_d) and the final form (with k_d and n_R)."""
    p = {"rq": 1.0, "cs": 0.0} | p
    m = MonthlyClimate(names, T_process)
    col = lambda k: np.asarray(g[k], float)[:, None]
    Phi, Psi, B, HT, ks = col("Phi_sys"), col("Psi_sys"), col("B_L"), col("HT"), col("kappa_s")
    R, fL = col("R_ch"), np.asarray(f_L, float)[:, None]
    turn = R ** p["rq"] / (fL * B)
    xi = turn / (turn + p["t0"])
    dtheta_ch = Phi * fL / R
    swing = p["A"] * (Phi / B) ** p["m"]
    direct = 0.0 if schedule == NIGHT else p["om"]
    store = p["kc"] * B * (1 + HT)
    cap = smin(direct + store, np.ones_like(store)) * m.L
    sigma = p["ks"] * m.sigma / np.sqrt(1.0 + p["cs"] * store)
    f = np.full(m.S.shape, 0.5)
    for _ in range(ITERATIONS):
        rise = smin(np.maximum(f * (1 - m.tm) + swing, 1e-6), np.maximum(1.0 + HT - m.tm, 1e-6))
        theta_r = m.tm + rise * xi
        Xc = m.xc * Psi * (theta_r - m.ta)
        util = phibar(Xc, m.KT, m.ratio, m.slope)
        if "kd" in p:
            head = np.maximum(1.0 + HT - theta_r, 0.0)
            util = util - phibar(Xc + m.xc * p["kd"] * head / dtheta_ch, m.KT, m.ratio, m.slope)
        Y = Phi * m.S * util - ks * B * (theta_r + p["l1"] - m.ta) / 12.0
        f = 0.5 * f + 0.5 * np.clip(expected_min(Y, cap, sigma) / m.L, 0.0, 1.0)
    annual = (f * m.L).sum(1) / m.L.sum(1)
    return annual, f, m.L


def solar_fraction(Phi_sys, Psi_sys, B_L, R_ch, HT, kappa_s, site, schedule: str = "day_08_18",
                   form: str = "final", T_process: float = T_BASE, f_L=None, monthly: bool = False):
    """Annual solar fraction of each design. With monthly=True also returns the monthly solar fractions f_s,j and
    the monthly demand shares d_L,j (rows: designs, columns: months); the explicit form has no monthly values.

    f_L, the active load fraction, defaults to the schedule's (10/24 for the 10 h loads, 1 for continuous)."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")
    arrays = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float))
                                   for v in (Phi_sys, Psi_sys, B_L, R_ch, HT, kappa_s)))
    g = dict(zip(("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s"), arrays))
    n = len(arrays[0])
    names = list(np.broadcast_to(np.asarray(site, dtype=object), (n,)))
    f_L = np.full(n, ACTIVE_FRACTION[schedule]) if f_L is None else np.broadcast_to(np.asarray(f_L, float), (n,))
    p = constants()["forms"][form][schedule]
    if form == "explicit":
        out = _explicit(g, names, schedule, T_process, p)
    else:
        out = _monthly(g, names, schedule, T_process, p, f_L)
    return out if monthly else out[0]
