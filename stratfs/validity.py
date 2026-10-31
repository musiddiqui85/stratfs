"""Validity domain of the fitted correlation and its accuracy ranges A-E.

The accuracy ranges summarize the errors observed on the 28 held-out test sites (RMSE, bias and 95th percentile
of the absolute error, in percentage points). They indicate the typical error for designs in each range, not a
prediction interval for an individual design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratfs.data import constants


def accuracy_range(Phi_sys, Psi_sys) -> np.ndarray:
    """Accuracy range A-E from Phi_sys and Psi_sys (clipped to the sampled box)."""
    c = constants()
    (P0, P1), (S0, S1) = c["validity"]["Phi_sys"], c["validity"]["Psi_sys"]
    phi = np.clip(np.atleast_1d(np.asarray(Phi_sys, float)), P0, P1)
    psi = np.clip(np.atleast_1d(np.asarray(Psi_sys, float)), S0, S1)
    out = np.full(phi.shape, "", dtype=object)
    for cl in c["accuracy_classes"]:
        (p0, p1), (s0, s1) = cl["Phi_sys"], cl["Psi_sys"]
        inside = ((phi > p0) | (p0 == P0)) & (phi <= p1) & ((psi > s0) | (s0 == S0)) & (psi <= s1)
        out[inside] = cl["label"]
    return out


def accuracy_table(schedule: str) -> pd.DataFrame:
    """Test-site RMSE, bias and p95 (pp) of the final form per accuracy range for a load schedule."""
    return pd.DataFrame(constants()["accuracy"][schedule]).T


def check(groups: dict | pd.DataFrame, dT_L=None) -> pd.DataFrame:
    """Per-design flags: each group inside its sampled range, the temperature lift inside 45-80 K, and `valid`
    when all hold. Outside the domain the correlation extrapolates and its accuracy is not established."""
    c = constants()
    g = pd.DataFrame(groups) if not isinstance(groups, pd.DataFrame) else groups
    out = pd.DataFrame(index=g.index)
    for k in ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s"):
        lo, hi = c["validity"][k]
        out[f"{k}_ok"] = g[k].between(lo * (1 - 1e-9), hi * (1 + 1e-9))
    if dT_L is not None:
        lo, hi = c["process_temperature_lift_K"]
        out["dT_L_ok"] = pd.Series(np.broadcast_to(np.asarray(dT_L, float), len(g)), index=g.index).between(lo, hi)
    out["valid"] = out.all(axis=1)
    return out
