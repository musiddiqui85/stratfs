"""stratfs: annual solar fraction of solar process-water heating with stratified storage.

A physics-informed dimensionless correlation, fitted on 49,152 converged simulations at 239 sites and tested on
28 held-out sites, with the site data of all 267 sites. Quick use:

    import stratfs
    stratfs.solar_fraction(0.88, 0.40, 0.20, 2.0, 0.35, 0.022, site="Karachi", schedule="day_08_18")

    plant = stratfs.Plant(collector_area_m2=650, storage_volume_m3=10, daily_load_m3=50, schedule="day_08_18")
    stratfs.predict_plant(plant, "Karachi")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratfs.correlation import FORMS, solar_fraction
from stratfs.data import SCHEDULE_NAMES, SCHEDULES, constants, monthly_inputs, site_names, sites, temperature_scale
from stratfs.plant import Plant, groups
from stratfs.validity import accuracy_range, accuracy_table, check

__version__ = "1.0.0"
__all__ = ["FORMS", "SCHEDULES", "SCHEDULE_NAMES", "Plant", "world_groups", "world_plant", "accuracy_range", "accuracy_table", "check", "constants",
           "groups", "monthly_inputs", "predict_groups", "predict_plant", "site_names", "sites", "solar_fraction",
           "temperature_scale"]


def predict_groups(g: dict | pd.DataFrame, site, schedule: str = "day_08_18", T_process: float = 80.0,
                   forms=("final",)) -> pd.DataFrame:
    """Annual solar fraction of designs given as groups (columns Phi_sys, Psi_sys, B_L, R_ch, HT, kappa_s), with the
    accuracy range, its test RMSE and p95 (pp) and the validity flags."""
    g = pd.DataFrame(g) if not isinstance(g, pd.DataFrame) else g.reset_index(drop=True)
    out = g.copy()
    args = [g[k].to_numpy(float) for k in ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s")]
    for form in forms:
        out[f"f_s_{form}" if len(forms) > 1 else "f_s"] = solar_fraction(*args, site=site, schedule=schedule,
                                                                           form=form, T_process=T_process)
    out["accuracy_range"] = accuracy_range(g.Phi_sys, g.Psi_sys)
    acc = accuracy_table(schedule)
    out["rmse_pp"] = acc.rmse_pp.reindex(out.accuracy_range).to_numpy()
    out["p95_pp"] = acc.p95_pp.reindex(out.accuracy_range).to_numpy()
    sites_ = np.broadcast_to(np.asarray(site, dtype=object), (len(g),))
    flags = check(g, temperature_scale(list(sites_), T_process))
    return pd.concat([out, flags], axis=1)


def predict_plant(plant: Plant, site: str, forms=("final",), **sweep) -> pd.DataFrame:
    """Groups and annual solar fraction of a dimensional plant at a site (keyword arrays sweep plant fields)."""
    g = groups(plant, site, **sweep)
    out = predict_groups(g, site, plant.schedule, plant.T_process_C, forms)
    out["Q_solar_GJ"] = out[[c for c in out if c.startswith("f_s")][0]] * out.E_D_GJ
    return out


def _with_sites(out: pd.DataFrame, names) -> pd.DataFrame:
    s = sites().loc[names, ["site", "group", "split", "latitude", "longitude", "H_kWh_m2", "K_T", "T_a_C"]]
    return pd.concat([s.reset_index(drop=True), out.drop(columns=["site"], errors="ignore").reset_index(drop=True)],
                     axis=1)


def world_plant(plant: Plant, forms=("final",)) -> pd.DataFrame:
    """The same dimensional plant at every site: its groups there and its annual solar fraction."""
    names = site_names()
    g = pd.concat([groups(plant, s) for s in names], ignore_index=True)
    out = predict_groups(g, names, plant.schedule, plant.T_process_C, forms)
    out["Q_solar_GJ"] = out[[c for c in out if c.startswith("f_s")][0]] * out.E_D_GJ
    return _with_sites(out, names)


def world_groups(g: dict, schedule: str = "day_08_18", T_process: float = 80.0, forms=("final",)) -> pd.DataFrame:
    """The same dimensionless specification at every site: its annual solar fraction there."""
    names = site_names()
    G = pd.DataFrame({k: np.full(len(names), float(g[k])) for k in ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s")})
    return _with_sites(predict_groups(G, names, schedule, T_process, forms), names)
