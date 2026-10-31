"""A dimensional plant and its dimensionless groups at a site.

The collector, primary oil loop (two pipe legs) and external heat exchanger reduce to an effective source with
the factor F_sys (Hottel-Whillier heat-removal factor, de Winter heat-exchanger factor, Beckman pipe factors) and
the pipe correction chi_p,o of the loss slope. The defaults are the reference plant of the paper: 650 m2 of
evacuated tubes, a 10 m3 atmospheric storage and a 50 m3/day process load at 80 C.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stratfs.correlation import ACTIVE_FRACTION
from stratfs.data import monthly_inputs, temperature_scale

RHO, CP_W, K_WATER = 1000.0, 4180.0, 0.6      # water density kg/m3, specific heat J/(kg K), conductivity W/(m K)
YEAR_S, N_D = 365 * 86400.0, 365.0
TAU_ALPHA_DATA = 0.75                           # (tau alpha)_n of the absorbed resource in the site tables
LOAD_HOURS = {"day_08_18": 10.0, "night_20_06": 10.0, "continuous_24h": 24.0}


@dataclass(frozen=True)
class Plant:
    collector_area_m2: float = 650.0
    storage_volume_m3: float = 10.0
    daily_load_m3: float = 50.0
    schedule: str = "day_08_18"
    T_process_C: float = 80.0
    charging_rate_w_k: float = 11611.111111111111   # heat-capacity rate of the storage-charging loop
    # collector and primary oil loop
    tau_alpha: float = 0.75
    U_L_w_m2k: float = 1.5
    F_prime: float = 0.9759785328192064
    oil_rate_w_k_m2: float = 6388.888888888889 / 650.0   # oil-loop heat-capacity rate per m2 of aperture
    pipe_w_k_per_leg_m2: float = 7.5 / 650.0                # conductance of each primary pipe leg per m2
    hx_effectiveness: float = 0.75
    # storage
    T_limit_C: float = 99.0
    insulation_w_m2k: float = 0.4
    aspect: float = 2.0                                     # height / diameter of the vertical cylinder

    def with_(self, **kw) -> "Plant":
        return replace(self, **kw)


def primary_loop(area, U_L, F_prime, oil_rate, charging_rate, hx_eff, pipe_leg):
    """F_R, F_sys and chi_p,o of the reduced collector-primary-loop-heat-exchanger source."""
    area, oil_rate, charging_rate, pipe_leg = (np.asarray(v, float) for v in (area, oil_rate, charging_rate, pipe_leg))
    M_C = oil_rate / (area * U_L)
    F_R = M_C * (1.0 - np.exp(-F_prime / M_C))
    mu_hx = np.minimum(oil_rate, charging_rate) / oil_rate
    kx = hx_eff * mu_hx
    gamma = np.exp(-pipe_leg / oil_rate)
    mu_c = area * F_R * U_L / oil_rate
    F_sys = F_R * kx * gamma / (1.0 - gamma * gamma * (1.0 - mu_c) * (1.0 - kx))
    chi = gamma + oil_rate * (1.0 - gamma * gamma) / (area * F_R * U_L * gamma)
    return F_R, F_sys, chi


def storage_geometry(V, aspect):
    D = (4 * np.asarray(V, float) / (np.pi * aspect)) ** (1 / 3)
    H = aspect * D
    return D, H, np.pi * D * H + 0.5 * np.pi * D ** 2


def site_terms(site: str, plant: Plant) -> dict:
    """Normalizing terms of a site for a plant's load: annual demand E_D (J), temperature scale (K), load
    heat-capacity rate (W/K) and the absorbed annual resource (J/m2) of the plant's collector."""
    m = monthly_inputs()
    row = m[m.site.eq(site)].iloc[0]
    dT = float(temperature_scale([site], plant.T_process_C)[0])
    f_L = ACTIVE_FRACTION[plant.schedule]
    C_L = plant.daily_load_m3 * RHO / (LOAD_HOURS[plant.schedule] * 3600.0) * CP_W
    return {"dT_L": dT, "C_L": C_L, "E_D": f_L * YEAR_S * dT * C_L, "f_L": f_L,
            "H_abs": float(row.H_abs_annual_MJ) * 1e6 * plant.tau_alpha / TAU_ALPHA_DATA,
            "T_m_ref": plant.T_process_C - dT}


def groups(plant: Plant, site: str, **sweep) -> pd.DataFrame:
    """Groups of the plant at the site. Keyword arrays (collector_area_m2, storage_volume_m3,
    charging_rate_w_k, ...) override plant fields element-wise, for sweeps and design maps."""
    fields = {k: np.atleast_1d(np.asarray(sweep.get(k, getattr(plant, k)), float))
              for k in ("collector_area_m2", "storage_volume_m3", "charging_rate_w_k", "insulation_w_m2k")}
    A, V, C_ch, U_s = np.broadcast_arrays(*fields.values())
    t = site_terms(site, plant)
    oil = plant.oil_rate_w_k_m2 * A
    F_R, F_sys, chi = primary_loop(A, plant.U_L_w_m2k, plant.F_prime, oil, C_ch, plant.hx_effectiveness,
                                   plant.pipe_w_k_per_leg_m2 * A)
    D, H, A_s = storage_geometry(V, plant.aspect)
    return pd.DataFrame({
        "site": site, "collector_area_m2": A, "storage_volume_m3": V, "charging_rate_w_k": C_ch,
        "F_R": F_R, "F_sys": F_sys, "chi_p_o": chi,
        "Phi_sys": F_sys * A * t["H_abs"] / t["E_D"],
        "Psi_sys": chi * plant.U_L_w_m2k * t["dT_L"] * YEAR_S / t["H_abs"],
        "B_L": N_D * RHO * V * CP_W * t["dT_L"] / t["E_D"],
        "R_ch": C_ch / t["C_L"],
        "HT": (plant.T_limit_C - plant.T_process_C) / t["dT_L"],
        "kappa_s": U_s * A_s * 86400.0 / (RHO * V * CP_W),
        "kappa_z_10": K_WATER * 10 * 86400.0 / (RHO * CP_W * H ** 2),
        "E_D_GJ": t["E_D"] / 1e9, "dT_L_K": t["dT_L"], "UA_s_w_k": U_s * A_s,
    })
