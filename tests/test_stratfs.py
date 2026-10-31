"""The package reproduces the predictions of the code used in the paper, and its conversions are consistent."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stratfs

REF = pd.read_csv(Path(__file__).parent / "data" / "reference_predictions.csv")


@pytest.mark.parametrize("form", stratfs.FORMS)
def test_reproduces_reference_predictions(form):
    for schedule, g in REF.groupby("schedule"):
        f = stratfs.solar_fraction(g.Phi_sys, g.Psi_sys, g.B_L, g.R_ch, g.HT, g.kappa_s, site=g.site.to_numpy(),
                                   schedule=schedule, form=form, f_L=g.f_L_T)
        assert np.abs(f - g[f"f_s_{form}"].to_numpy()).max() < 1e-12


def test_test_error_of_final_form():
    rmse = {s: float(np.sqrt(np.mean((100 * (g.f_s_final - g.f_s_simulated)) ** 2))) for s, g in REF.groupby("schedule")}
    assert rmse == pytest.approx({"day_08_18": 2.20, "night_20_06": 2.21, "continuous_24h": 2.72}, abs=0.01)


def test_reference_plant_karachi():
    r = stratfs.predict_plant(stratfs.Plant(), "Karachi").iloc[0]
    assert r.Phi_sys == pytest.approx(0.8829, abs=1e-4)
    assert r.Psi_sys == pytest.approx(0.4012, abs=1e-4)
    assert r.B_L == pytest.approx(0.2, abs=1e-9)
    assert r.R_ch == pytest.approx(2.0, abs=1e-9)
    assert r.f_s == pytest.approx(0.7887, abs=1e-4)
    assert r.accuracy_range == "D" and bool(r.valid)


def test_monthly_consistent_with_annual():
    f, f_m, d_L = stratfs.solar_fraction(0.883, 0.401, 0.2, 2.0, 0.352, 0.0223, site="Karachi", monthly=True)
    assert f_m.shape == (1, 12) and d_L.sum() == pytest.approx(1.0)
    assert f[0] == pytest.approx(float((f_m * d_L).sum()))


def test_process_temperature_rescaling():
    base = stratfs.solar_fraction(0.883, 0.401, 0.2, 2.0, 0.352, 0.0223, site="Karachi", T_process=80.0)
    other = stratfs.solar_fraction(0.883, 0.401, 0.2, 2.0, 0.352, 0.0223, site="Karachi", T_process=70.0)
    assert np.isfinite(other).all() and abs(other[0] - base[0]) > 1e-6


def test_validity_flags():
    g = pd.DataFrame({"Phi_sys": [0.5, 5.0], "Psi_sys": [0.5, 0.5], "B_L": [0.5, 0.5], "R_ch": [2, 2], "HT": [0.3, 0.3],
                      "kappa_s": [0.02, 0.02]})
    out = stratfs.predict_groups(g, "Seville", "continuous_24h")
    assert out.valid.tolist() == [True, False]
    assert out.accuracy_range.tolist() == ["B", "D"]


def test_world_maps():
    w = stratfs.world_plant(stratfs.Plant())
    assert len(w) == 267 and w.f_s.between(0, 1).all()
    s = stratfs.world_groups(dict(Phi_sys=0.88, Psi_sys=0.40, B_L=0.2, R_ch=2.0, HT=0.35, kappa_s=0.022))
    assert len(s) == 267 and s.f_s.between(0, 1).all()
