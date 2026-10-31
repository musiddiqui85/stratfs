"""Site data of the 267 climate sites and the constants of the three fitted forms.

Temperatures are normalized with the process temperature T_p and each site's annual mean mains temperature
T_m,ref: theta = (T - T_m,ref) / (T_p - T_m,ref). The tables store the normalized values for T_p = 80 C, the basis of
the fitted database, and are rescaled exactly for any other process temperature.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

SCHEDULES = ("day_08_18", "night_20_06", "continuous_24h")
SCHEDULE_NAMES = {"day_08_18": "Daytime (08:00-18:00)", "night_20_06": "Night-time (20:00-06:00)",
                  "continuous_24h": "Continuous (24 h)"}
GROUPS = ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s")
T_BASE = 80.0   # process temperature of the stored normalized inputs, C


def _read(name: str) -> pd.DataFrame:
    with resources.files("stratfs").joinpath("data", name).open("rb") as f:
        return pd.read_csv(f)


@lru_cache(maxsize=1)
def sites() -> pd.DataFrame:
    """One row per site: location, climate group, training/test split, annual resource and temperatures."""
    return _read("sites.csv").set_index("site", drop=False)


@lru_cache(maxsize=1)
def monthly_inputs() -> pd.DataFrame:
    """Twelve rows per site: monthly climate and the normalized inputs at 80 C."""
    return _read("monthly_inputs.csv").sort_values(["site", "month"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def utilizability() -> pd.DataFrame:
    """phi_site(x) of every site (rows) on the two-part x grid (columns); used by the explicit form only."""
    return _read("utilizability.csv").pivot(index="site", columns="x", values="phi")


@lru_cache(maxsize=1)
def constants() -> dict:
    """Constants of the explicit, implicit and final forms, the variability closure, validity and accuracy ranges."""
    with resources.files("stratfs").joinpath("data", "constants.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def site_names() -> list[str]:
    return sites().index.tolist()


def temperature_scale(site_names, T_process: float = T_BASE) -> np.ndarray:
    """Temperature scale Delta T_L = T_p - T_m,ref (K) of each site, consistent with the stored 80 C basis."""
    s = sites().loc[np.atleast_1d(site_names)]
    return s.dT_L_80_K.to_numpy() + (T_process - T_BASE)


class MonthlyClimate:
    """The twelve monthly inputs of the implicit form and the stratified-storage correlation for a list of sites
    (rows) at a process temperature: resource share s_j, normalized ambient and mains temperatures, clearness
    index, Klein's tilt-factor ratio and continuation slope, the conversion c_j and the raw spread sigma_j."""

    def __init__(self, site_names, T_process: float = T_BASE):
        names = list(np.atleast_1d(site_names))
        m = monthly_inputs()
        unknown = sorted(set(names) - set(m.site))
        if unknown:
            raise KeyError(f"unknown site(s): {unknown}")
        t = m.set_index(["site", "month"])
        piv = lambda c: t[c].unstack("month").loc[names].to_numpy(float)
        scale = (sites().loc[names].dT_L_80_K.to_numpy() / temperature_scale(names, T_process))[:, None]
        self.S = piv("s_m")
        self.ta = piv("theta_a_m_80") * scale
        self.tm = piv("theta_m_m_80") * scale
        self.KT = piv("KT")
        n = piv("N")
        w = n * (1.0 - self.tm)
        self.L = w / w.sum(1, keepdims=True)
        self.ratio = piv("R_n") / piv("Rbar")
        mean_abs = piv("H_abs_annual_MJ") * 1e6 / (8760 * 3600.0)
        noon = piv("tau_alpha") * piv("r_t_n") * piv("R_n") * piv("H") * 1e6 / 3600.0
        self.xc = mean_abs / noon
        self.slope = piv("r_t_n") * piv("R_n") * 2 * piv("omega_s_deg") / 15.0 / piv("Rbar")
        c = constants()["sigma_closure"]
        self.sigma = np.maximum(c["a"] + c["b"] * self.KT, c["floor"])
