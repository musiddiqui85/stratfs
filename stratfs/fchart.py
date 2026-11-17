"""The modified phi-bar,f-chart (Braun, Klein and Pearson, Solar Energy 31 (1983) 597-604) in the groups of the
correlation: the conventional fully mixed preheat-store method the paper compares with (Section 5.4, Table 7).

The method is monthly and iterative. In the normalization of the paper (annual demand 1, theta = (T - T_m,ref) /
dT_L) month j has load share L_j, load capacitance C_j, mean mains and store-ambient temperatures theta_m,j and
theta_a,s,j (aggregates of the normalized hourly forcing, `fchart_inputs.csv`), and

    theta'_min = theta_m,j + f L_j / C_j
    Q_max,j    = Phi_sys s_j phibar(X_c),   X_c from the critical absorbed irradiance Psi_sys (theta'_min - theta_a,j)
    Q_u,j      = Q_max,j - a (exp(b f) - 1)(1 - exp(c X)) exp(d Z) L_j
    f          = (Q_u,j - kappa_s B_L (theta_s - theta_a,s,j) dtau_j) / L_j,   theta_s = theta'_min + g (exp(k f) - 1) exp(h Z)
    X = Phi_sys Psi_sys (100 / dT_L) dtau_j / L_j,  Z = L_j dT_L / (100 C_j),  C_s = B_L (H_abs / 365) / (Phi_sys dT_L) / 1000

with a = 0.015 (C_s/350)^-0.76, g = 0.2136 (C_s/350)^-0.704 / dT_L and b, c, d, h, k of the paper. The store is
fully mixed and has no temperature limit, so the method predicts the mixed-storage solar fraction. The forcing
aggregates are stored for T_p = 80 C, the process temperature of the paper's comparisons.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stratfs.data import T_BASE, _read, monthly_inputs, sites

B, C, D, H, K = 3.85, -0.15, -1.959, -4.002, 4.702
YEAR_S, N_D = 365 * 86400.0, 365.0


@lru_cache(maxsize=1)
def fchart_inputs() -> pd.DataFrame:
    """Monthly forcing aggregates per site and schedule (dtau, L, C, theta_m, theta_a_s) at 80 C."""
    return _read("fchart_inputs.csv").set_index(["site", "schedule"]).sort_index()


def phibar_klein(Xc: float, KT: float, ratio: float) -> float:
    """Klein's monthly-average daily utilizability (1 for X_c <= 0; held at its vertex where the quadratic turns)."""
    if Xc <= 0:
        return 1.0
    a = 2.943 - 9.271 * KT + 4.031 * KT ** 2
    b = -4.345 + 8.853 * KT - 3.602 * KT ** 2
    c = -0.170 - 0.306 * KT + 2.936 * KT ** 2
    if c < 0:
        Xc = min(Xc, -0.5 / c)
    return float(np.exp(min((a + b * ratio) * (Xc + c * Xc ** 2), 0.0)))


def _one(Phi, Psi, B_L, kappa_s, site, schedule):
    f_in = fchart_inputs().loc[(site, schedule)].sort_values("month")
    mo = monthly_inputs()
    mo = mo[mo.site.eq(site)].sort_values("month")
    dT = float(sites().loc[site].dT_L_80_K)
    H_abs = float(mo.H_abs_annual_MJ.iloc[0]) * 1e6
    C_s = B_L * (H_abs / N_D) / (Phi * dT) / 1000.0
    a = 0.015 * (C_s / 350.0) ** -0.76
    g = 0.2136 * (C_s / 350.0) ** -0.704 / dT
    num = den = 0.0
    for sl, m in zip(f_in.itertuples(), mo.itertuples()):
        L, Cm, dtau = sl.L, sl.C, sl.dtau
        if L <= 0:
            continue
        X = Phi * Psi * (100.0 / dT) * dtau / L
        Z = L * dT / (100.0 * Cm)
        corr = a * (1 - np.exp(C * X)) * np.exp(D * Z) * L
        noon = m.r_t_n * m.R_n * m.H * 1e6 / 3600.0

        def residual(f):
            t_min = sl.theta_m + f * L / Cm
            g_crit = Psi * (t_min - m.theta_a_m_80) * H_abs / YEAR_S
            q_max = Phi * m.s_m * phibar_klein((g_crit / m.tau_alpha) / noon, m.KT, m.R_n / m.Rbar)
            q_u = max(q_max - (np.exp(B * f) - 1) * corr, 0.0)
            t_s = t_min + g * (np.exp(K * f) - 1) * np.exp(H * Z)
            return f - (q_u - kappa_s * B_L * (t_s - sl.theta_a_s) * dtau) / L

        if residual(1.0) <= 0:
            f_m = 1.0
        elif residual(0.0) >= 0:
            f_m = 0.0
        else:
            f_m = brentq(residual, 0.0, 1.0, xtol=1e-6)
        num += f_m * L
        den += L
    return num / den


def fchart_solar_fraction(Phi_sys, Psi_sys, B_L, kappa_s, site, schedule: str = "day_08_18",
                          T_process: float = T_BASE) -> np.ndarray:
    """Annual mixed-storage solar fraction by the modified phi-bar,f-chart, one value per design."""
    if abs(T_process - T_BASE) > 1e-9:
        raise ValueError("the f-chart forcing aggregates are stored for T_p = 80 C only")
    arr = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (Phi_sys, Psi_sys, B_L, kappa_s)))
    names = np.broadcast_to(np.asarray(site, dtype=object), arr[0].shape)
    return np.array([_one(p, q, b, k, s, schedule) for p, q, b, k, s in zip(*arr, names)])
