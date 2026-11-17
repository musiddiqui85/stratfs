"""The design routes of Section 6 of the paper reproduce its values (paper/expected), and are self-consistent.

Tolerances: the design code of the paper took the temperature scale from the monthly mains table, the package
from the forcing of the fit; the two differ by at most 3e-4 K, which moves f_s by less than 1e-5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stratfs

PAPER = Path(__file__).parents[1] / "paper"
EXPECTED = PAPER / "expected"


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


@pytest.mark.parametrize("schedule", stratfs.SCHEDULES)
def test_sizing_reproduces_paper(schedule):
    """Section 6.1: the 27 designs sized in Karachi for 0.4, 0.6, 0.8 at 25, 50 and 100 L/m2."""
    e = pd.read_csv(EXPECTED / "sizing_designs.csv")
    e = e[e.schedule.eq(schedule)]
    d = stratfs.size_for_target(stratfs.Plant(schedule=schedule), "Karachi", (0.4, 0.6, 0.8), (25.0, 50.0, 100.0))
    m = d.merge(e, on=["schedule", "target", "specific_l_m2"], suffixes=("", "_paper"))
    assert len(m) == 9 and (m.reached == m.reached_paper).all()
    r = m[m.reached_paper]
    np.testing.assert_allclose(r.collector_area_m2, r.A_m2, rtol=1e-6)
    np.testing.assert_allclose(r.R_ch, r.R_ch_paper, rtol=0)
    np.testing.assert_allclose(r.f_s, r.f_s_paper, atol=1e-5)
    assert (r.accuracy_range == r.accuracy_range_paper).all()
    assert (r.f_s >= r.target).all()


def test_sizing_confirmation_numbers():
    """Section 6.1: the sized designs against their converged simulations."""
    sim = pd.read_csv(PAPER / "data" / "sizing_simulation.csv")
    e = pd.read_csv(EXPECTED / "sizing_designs.csv")
    z = e[e.reached].merge(sim, on=["schedule", "target", "specific_l_m2"])
    assert len(z) == 27
    shortfall = 100 * (z.target - z.f_s_sim)
    assert np.ceil(shortfall.abs().max()) == 5 and int((shortfall <= 0).sum()) == 26


@pytest.mark.parametrize("schedule", stratfs.SCHEDULES)
def test_world_plant_reproduces_paper(schedule):
    """Section 6.2: the 650 m2, 10 m3 plant at all 267 sites."""
    e = pd.read_csv(EXPECTED / "world_plant.csv")
    w = stratfs.world_plant(stratfs.Plant(schedule=schedule)).assign(schedule=schedule)
    m = w.merge(e, on=["site", "schedule"], suffixes=("", "_paper"))
    assert len(m) == 267
    for k in ("Phi_sys", "Psi_sys", "B_L", "HT", "kappa_s", "f_s"):
        np.testing.assert_allclose(m[k], m[k + "_paper"], atol=1e-5)
    assert (m.accuracy_range == m.accuracy_range_paper).all() and (m.valid == m.valid_paper).all()


@pytest.mark.parametrize("schedule", stratfs.SCHEDULES)
def test_world_specification_reproduces_paper(schedule):
    """Section 6.3: the plant's groups in Karachi at all 267 sites, and the plant that realizes them there."""
    plant = stratfs.Plant(schedule=schedule)
    spec = stratfs.plant_specification(plant, "Karachi")
    e = pd.read_csv(EXPECTED / "world_specification.csv")
    w = stratfs.world_specification(spec, plant).assign(schedule=schedule)
    m = w.merge(e, on=["site", "schedule"], suffixes=("", "_paper"))
    assert len(m) == 267
    np.testing.assert_allclose(m.f_s, m.f_s_paper, atol=1e-5)
    np.testing.assert_allclose(m.collector_area_m2, m.A_m2, rtol=2e-5)
    np.testing.assert_allclose(m.storage_volume_m3, m.V_m3, rtol=2e-5)
    np.testing.assert_allclose(m.U_L_implied_w_m2k, m.U_L_implied, atol=2e-5)


def test_realize_inverts_groups():
    """Realizing a plant's own specification at its own site returns the plant."""
    plant = stratfs.Plant(collector_area_m2=900, storage_volume_m3=30, daily_load_m3=80, schedule="continuous_24h",
                          charging_rate_w_k=20000.0)
    spec = stratfs.plant_specification(plant, "Seville")
    r = stratfs.realize(spec, "Seville", plant)
    assert r["collector_area_m2"] == pytest.approx(900, rel=1e-9)
    assert r["storage_volume_m3"] == pytest.approx(30, rel=1e-12)
    assert r["charging_rate_w_k"] == pytest.approx(20000.0, rel=1e-12)
    assert r["T_limit_C"] == pytest.approx(plant.T_limit_C, abs=1e-9)


@pytest.mark.parametrize("kind", ["plant", "specification"])
def test_confirmation_tables(kind):
    """Table 7 (the plant) and Table S2.1 (the specification): correlation and f-chart at the 30 confirmation sites
    against the converged simulations."""
    sim = pd.read_csv(PAPER / "data" / f"confirm_{kind}.csv")
    e = pd.read_csv(EXPECTED / f"confirm_{kind}.csv")
    parts = []
    for s, g in sim.groupby("schedule"):
        plant = stratfs.Plant(schedule=s)
        spec = stratfs.plant_specification(plant, "Karachi") if kind == "specification" else None
        d = stratfs.world_designs(plant, spec, sites=g.site.tolist(), fchart=True)
        parts.append(g.reset_index(drop=True).join(d.drop(columns=["site"])))
    c = pd.concat(parts, ignore_index=True)
    m = c.merge(e, on=["site", "schedule"], suffixes=("", "_paper"))
    assert len(m) == 90 and m.site.nunique() == 30
    np.testing.assert_allclose(m.f_s, m.f_s_paper, atol=1e-5)
    np.testing.assert_allclose(m.f_s_fchart, m.f_s_fchart_paper, atol=1e-5)
    pooled = rmse(100 * (c.f_s - c.f_s_sim))
    assert pooled == pytest.approx({"plant": 2.01, "specification": 2.55}[kind], abs=0.005)


def test_design_map_best_flow_is_valid_and_bounded():
    plant = stratfs.Plant(schedule="day_08_18")
    g = stratfs.design_map(plant, "Karachi", n=8)
    assert len(g) > 0 and g.valid.all() and g.f_s.between(0, 1).all()
    assert set(g.R_ch) <= set(stratfs.R_CH_GRID)


def test_fchart_only_at_80C():
    with pytest.raises(ValueError):
        stratfs.fchart_solar_fraction(0.9, 0.4, 0.2, 0.02, "Karachi", "day_08_18", T_process=70.0)
