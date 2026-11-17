"""Plant design with the correlation: dimensional sizing and dimensionless specifications (Section 6 of the paper).

Two routes lead from a process to a plant.

Dimensional. The designer fixes the site, the load and the hardware (collector loop, storage insulation and
limit) and searches the plant size. `design_map` evaluates the correlation over collector area x storage volume,
each at the charging flow of highest solar fraction, and `size_for_target` returns the smallest collector area
that reaches a target solar fraction at a given specific storage volume (litres per m2 of collector).

Dimensionless. Because the correlation carries no scale, a plant can be specified by its groups alone and carried
to any site. `realize` converts a specification into the plant that realizes it at a site for a given load:
storage volume from B_L, collector area from Phi_sys (fixed point on F_sys), the collector loss coefficient that
Psi_sys implies there, and the storage limit T_p + H_T dT_L that H_T implies. `world_specification` does this at
every site.

Every evaluation keeps only designs inside the validity domain (all groups in range and dT_L in 45-80 K).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratfs.data import constants, site_names
from stratfs.plant import CP_W, N_D, RHO, YEAR_S, Plant, groups, site_terms

R_CH_GRID = (1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0, 10.0)   # charging-to-load capacity-rate ratios searched
F_SYS_NOMINAL = 0.85      # nominal F_sys that places the area range on the Phi_sys range
CHI_NOMINAL = 1.016       # nominal pipe correction of the loss slope used to state the implied U_L
GROUP_COLUMNS = ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s")
DESIGN_COLUMNS = ["collector_area_m2", "storage_volume_m3", "charging_rate_w_k", "R_ch", "Phi_sys", "Psi_sys", "B_L",
                  "HT", "kappa_s", "kappa_z_10", "F_sys", "f_s", "accuracy_range", "rmse_pp", "p95_pp", "valid"]


def evaluate(plant: Plant, site: str, collector_area_m2, storage_volume_m3, R_ch) -> pd.DataFrame:
    """Groups and annual solar fraction of designs (arrays broadcast together) at the plant's load and hardware;
    the charging loop is given by R_ch, its heat-capacity rate over the load's."""
    from stratfs import predict_plant
    A, V, R = np.broadcast_arrays(*(np.asarray(x, float) for x in (collector_area_m2, storage_volume_m3, R_ch)))
    C_L = site_terms(site, plant)["C_L"]
    out = predict_plant(plant, site, collector_area_m2=A.ravel(), storage_volume_m3=V.ravel(),
                        charging_rate_w_k=R.ravel() * C_L)
    out["R_ch"] = R.ravel()     # the requested ratio, free of the round trip through the charging rate
    return out


def best_charging(d: pd.DataFrame, by) -> pd.DataFrame:
    """Per design (columns `by`), the valid charging flow of highest solar fraction."""
    d = d[d.valid]
    return d.loc[d.groupby(list(by)).f_s.idxmax()].reset_index(drop=True)


def size_range(plant: Plant, site: str, n: int = 40):
    """Collector areas and storage volumes (geometric) spanning Phi_sys and B_L over their fitted ranges, Phi_sys at
    the nominal F_sys of 0.85."""
    v = constants()["validity"]
    t = site_terms(site, plant)
    A = np.geomspace(*(np.array(v["Phi_sys"]) * t["E_D"] / (F_SYS_NOMINAL * t["H_abs"])), n)
    V = np.geomspace(*(np.array(v["B_L"]) * t["E_D"] / (N_D * RHO * CP_W * t["dT_L"])), n)
    return A, V


def kappa_floor_volume(plant: Plant) -> float:
    """Storage volume above which kappa_s falls below its fitted range (a cylinder of the plant's aspect ratio)."""
    floor = constants()["validity"]["kappa_s"][0]
    k = floor * RHO * CP_W / (86400.0 * plant.insulation_w_m2k)
    D = 4 * (plant.aspect + 0.5) / (plant.aspect * k)
    return float(np.pi * plant.aspect * D ** 3 / 4)


def design_map(plant: Plant, site: str, n: int = 60, R_ch=R_CH_GRID, areas=None, volumes=None) -> pd.DataFrame:
    """f_s over an n x n grid of collector area and storage volume, each at its best valid charging flow. By default
    the grid spans the fitted domain, the storage up to the volume at which kappa_s reaches its lower bound."""
    if areas is None or volumes is None:
        A0, V0 = size_range(plant, site, n)
        areas = A0 if areas is None else areas
        volumes = np.geomspace(V0.min(), min(V0.max(), kappa_floor_volume(plant)), n) if volumes is None else volumes
    AA, VV, RR = np.meshgrid(np.asarray(areas, float), np.asarray(volumes, float), np.asarray(R_ch, float),
                             indexing="ij")
    d = best_charging(evaluate(plant, site, AA, VV, RR), ["collector_area_m2", "storage_volume_m3"])
    t = site_terms(site, plant)
    out = d[DESIGN_COLUMNS].copy()
    out["E_D_GJ"] = t["E_D"] / 1e9
    out["yield_kwh_m2"] = out.f_s * t["E_D"] / 3.6e6 / out.collector_area_m2
    return out


def sizing_curve(plant: Plant, site: str, specific_storage_l_m2: float, n: int = 400, R_ch=R_CH_GRID) -> pd.DataFrame:
    """f_s against collector area at a fixed specific storage volume (L per m2 of collector), best valid charging
    flow, over the area range of the fitted Phi_sys domain."""
    A0, _ = size_range(plant, site, 2)
    A = np.geomspace(A0.min(), A0.max(), n)
    AA, RR = np.meshgrid(A, np.asarray(R_ch, float), indexing="ij")
    VV = np.repeat((A * specific_storage_l_m2 / 1000.0)[:, None], len(R_ch), axis=1)
    d = best_charging(evaluate(plant, site, AA, VV, RR), ["collector_area_m2"])
    return d.sort_values("collector_area_m2")[DESIGN_COLUMNS].reset_index(drop=True)


def size_for_target(plant: Plant, site: str, targets=(0.4, 0.6, 0.8), specific_storage_l_m2=(25.0, 50.0, 100.0),
                    n: int = 400, R_ch=R_CH_GRID) -> pd.DataFrame:
    """For each target annual solar fraction and specific storage volume (L/m2), the smallest collector area on the
    sizing curve that reaches the target, with its storage volume, charging flow, groups and accuracy range.
    `reached` is False when no valid design reaches the target (then `f_s_max` is the highest valid one)."""
    rows = []
    for v in np.atleast_1d(specific_storage_l_m2):
        c = sizing_curve(plant, site, float(v), n, R_ch)
        f_max = float(c.f_s.max()) if len(c) else np.nan
        for target in np.atleast_1d(targets):
            hit = c[c.f_s >= target]
            row = {"site": site, "schedule": plant.schedule, "target": float(target), "specific_l_m2": float(v),
                   "reached": not hit.empty, "f_s_max": f_max}
            if not hit.empty:
                row.update(hit.iloc[0].drop("valid").to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def realize(spec: dict, site: str, plant: Plant = Plant(), iterations: int = 60) -> dict:
    """The dimensional plant that realizes a dimensionless specification at a site for the plant's load
    (daily volume, schedule, process temperature) and collector loop.

    Storage volume from B_L; collector area from Phi_sys by fixed point on F_sys(A); the collector loss coefficient
    the specification's Psi_sys implies at the site (stated with the nominal pipe correction 1.016 of the reference
    loop), and the storage limit T_p + H_T dT_L its H_T implies."""
    t = site_terms(site, plant)
    V = float(spec["B_L"]) * t["E_D"] / (N_D * RHO * CP_W * t["dT_L"])
    C_ch = float(spec["R_ch"]) * t["C_L"]
    A = float(spec["Phi_sys"]) * t["E_D"] / (0.8 * t["H_abs"])
    for _ in range(iterations):
        F_sys = float(groups(plant, site, collector_area_m2=A, storage_volume_m3=V, charging_rate_w_k=C_ch).F_sys.iloc[0])
        A_new = float(spec["Phi_sys"]) * t["E_D"] / (F_sys * t["H_abs"])
        done = abs(A_new - A) <= 1e-13 * A
        A = A_new
        if done:
            break
    return {"collector_area_m2": A, "storage_volume_m3": V, "charging_rate_w_k": C_ch,
            "U_L_implied_w_m2k": float(spec["Psi_sys"]) * t["H_abs"] / (CHI_NOMINAL * t["dT_L"] * YEAR_S),
            "T_limit_C": plant.T_process_C + float(spec["HT"]) * t["dT_L"],
            "dT_L_K": t["dT_L"], "E_D_GJ": t["E_D"] / 1e9}


def plant_specification(plant: Plant, site: str) -> dict:
    """The dimensionless specification (groups) of a dimensional plant at a site."""
    g = groups(plant, site).iloc[0]
    return {k: float(g[k]) for k in GROUP_COLUMNS + ("kappa_z_10",)}


def world_specification(spec: dict, plant: Plant = Plant(), forms=("final",), fchart: bool = False) -> pd.DataFrame:
    """One dimensionless specification at every site: its annual solar fraction from the correlation and the
    dimensional plant that realizes it there (`realize`) for the plant's load. The plant's schedule and process
    temperature apply; with fchart=True also the modified phi-bar,f-chart at the same groups (80 C only)."""
    from stratfs import world_groups
    out = world_groups(spec, plant.schedule, plant.T_process_C, forms)
    real = pd.DataFrame([realize(spec, s, plant) for s in out.site])
    out = pd.concat([out, real.drop(columns=["charging_rate_w_k"])], axis=1)
    if fchart:
        from stratfs.fchart import fchart_solar_fraction
        out["f_s_fchart"] = fchart_solar_fraction(spec["Phi_sys"], spec["Psi_sys"], spec["B_L"], spec["kappa_s"],
                                                  out.site.to_numpy(), plant.schedule, plant.T_process_C)
    return out


def world_designs(plant: Plant, spec: dict | None = None, sites=None, fchart: bool = False) -> pd.DataFrame:
    """The plant (dimensional route) or its specification (dimensionless route, when `spec` is given; e.g.
    plant_specification(plant, "Karachi")) at the listed sites (default: all), one row per site."""
    from stratfs import predict_groups
    names = site_names() if sites is None else list(sites)
    if spec is None:
        g = pd.concat([groups(plant, s) for s in names], ignore_index=True)
    else:
        g = pd.DataFrame({k: np.full(len(names), float(spec[k])) for k in GROUP_COLUMNS})
        g.insert(0, "site", names)
    out = predict_groups(g, names, plant.schedule, plant.T_process_C)
    out["site"] = names
    if fchart:
        from stratfs.fchart import fchart_solar_fraction
        out["f_s_fchart"] = fchart_solar_fraction(out.Phi_sys, out.Psi_sys, out.B_L, out.kappa_s, names,
                                                  plant.schedule, plant.T_process_C)
    return out
