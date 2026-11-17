"""Reproduce the results of the paper with the package.

    python paper/reproduce.py              # all results, tables and figures into paper/results/
    python paper/reproduce.py --check      # and compare them with the values of the manuscript (paper/expected/)
    python paper/reproduce.py --no-figures

Section 5 (accuracy): the test RMSE of the three forms per load schedule and of the final form per accuracy range,
recomputed from the released simulation database (data/simulation_database.csv.gz).

Section 6 (application to plant design), for the process of the worked example: 50 m3/day of water at 80 C, the
reference collector loop (650 m2 of evacuated tubes at the reference design) and a 99 C storage insulated with
0.4 W/m2 K.

    6.1 dimensional sizing in Karachi: the design maps f_s(A, V) at the best charging flow and the smallest collector
        area for target solar fractions 0.4, 0.6, 0.8 at 25, 50 and 100 L/m2, then the sized designs against their
        converged simulations (paper/data/sizing_simulation.csv)
    6.2 one plant across the world: the 650 m2, 10 m3 plant at all 267 sites, then Table 7, its confirmation at the
        30 confirmation sites against the converged stratified and mixed-storage simulations
        (paper/data/confirm_plant.csv), with the modified phi-bar,f-chart
    6.3 one dimensionless specification across the world: the groups of that plant in Karachi at all 267 sites, the
        plant that realizes them at each site, and their confirmation (paper/data/confirm_specification.csv)

The simulations come from the reference model of the paper (a layered stratified store, N = 100 and 200 layers
and their grid-extrapolated reference, and a fully mixed store) and are released as data; everything else is
computed here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stratfs  # noqa: E402
from stratfs import design as SD  # noqa: E402
from stratfs.fchart import fchart_solar_fraction  # noqa: E402

DATA, EXPECTED, RESULTS = HERE / "data", HERE / "expected", HERE / "results"
DATABASE = HERE.parent / "data" / "simulation_database.csv.gz"
SCH = stratfs.SCHEDULES
NAME = {"day_08_18": "Daytime", "night_20_06": "Night-time", "continuous_24h": "Continuous"}
SIZING_SITE = "Karachi"
TARGETS, SPECIFIC = (0.4, 0.6, 0.8), (25.0, 50.0, 100.0)
REFERENCE_SITE = "Karachi"


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


# ----------------------------------------------------------------------------------------------- Section 5
def section5() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test RMSE per form and schedule, and per accuracy range of the final form, from the simulation database."""
    db = pd.read_csv(DATABASE)
    test = db[db.set.eq("test")]
    rows, ranges = [], []
    for s, g in test.groupby("schedule"):
        args = [g[k] for k in ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s")]
        row = {"schedule": s, "designs": len(g), "sites": g.site.nunique()}
        for form in stratfs.FORMS:
            f = stratfs.solar_fraction(*args, site=g.site.to_numpy(), schedule=s, form=form, f_L=g.f_L)
            row[f"rmse_pp_{form}"] = rmse(100 * (f - g.f_s_reference))
            if form == "final":
                err = 100 * (f - g.f_s_reference.to_numpy())
                cls = stratfs.accuracy_range(g.Phi_sys, g.Psi_sys)
                for c in sorted(set(cls)):
                    e = err[cls == c]
                    ranges.append({"schedule": s, "range": c, "n": len(e), "rmse_pp": rmse(e), "bias_pp": e.mean(),
                                   "p95_pp": float(np.percentile(np.abs(e), 95))})
        row["rmse_pp_mixed_storage_as_predictor"] = rmse(100 * (g.f_s_mixed - g.f_s_reference))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(ranges)


# ----------------------------------------------------------------------------------------------- Section 6.1
def sizing() -> dict:
    grids, designs, timing = [], [], []
    for s in SCH:
        plant = stratfs.Plant(schedule=s)
        t0 = time.perf_counter()
        g = SD.design_map(plant, SIZING_SITE, n=60)
        timing.append({"schedule": s, "evaluations": 60 * 60 * len(SD.R_CH_GRID),
                       "seconds": time.perf_counter() - t0})
        grids.append(g.assign(schedule=s))
        designs.append(SD.size_for_target(plant, SIZING_SITE, TARGETS, SPECIFIC))
    grid, des = pd.concat(grids, ignore_index=True), pd.concat(designs, ignore_index=True)
    sim = pd.read_csv(DATA / "sizing_simulation.csv")
    z = des[des.reached].merge(sim[["schedule", "target", "specific_l_m2", "f_s_N100", "f_s_N200", "f_s_sim",
                                    "f_s_mixed_sim"]], on=["schedule", "target", "specific_l_m2"])
    z["error_pp"] = 100 * (z.f_s - z.f_s_sim)
    z["shortfall_pp"] = 100 * (z.target - z.f_s_sim)
    z["gain_pp"] = 100 * (z.f_s_sim - z.f_s_mixed_sim)
    summary = {"designs sized": len(z), "evaluations per schedule (map)": timing[0]["evaluations"],
               "max |target - simulated|, pp": float(z.shortfall_pp.abs().max()),
               "designs meeting or exceeding the target in simulation": int((z.shortfall_pp <= 0).sum()),
               **{f"RMSE correlation vs simulation, {NAME[s]}, pp": rmse(z[z.schedule.eq(s)].error_pp) for s in SCH},
               "RMSE correlation vs simulation, all, pp": rmse(z.error_pp),
               "max |error|, pp": float(z.error_pp.abs().max()),
               "stratification gain, simulated, min-max, pp": f"{z.gain_pp.min():.1f}-{z.gain_pp.max():.1f}"}
    return {"grid": grid, "designs": des, "confirmation": z, "timing": pd.DataFrame(timing), "summary": summary}


# ----------------------------------------------------------------------------------------------- Sections 6.2, 6.3
def world() -> dict:
    plant_rows, spec_rows = [], []
    for s in SCH:
        plant = stratfs.Plant(schedule=s)
        plant_rows.append(stratfs.world_plant(plant).assign(schedule=s))
        spec = SD.plant_specification(plant, REFERENCE_SITE)
        spec_rows.append(SD.world_specification(spec, plant).assign(schedule=s))
    return {"plant": pd.concat(plant_rows, ignore_index=True), "specification": pd.concat(spec_rows, ignore_index=True)}


def confirmation(kind: str) -> pd.DataFrame:
    """Correlation and f-chart at the 30 confirmation sites against the converged simulations."""
    sim = pd.read_csv(DATA / f"confirm_{kind}.csv")
    parts = []
    for s, g in sim.groupby("schedule"):
        plant = stratfs.Plant(schedule=s)
        spec = SD.plant_specification(plant, REFERENCE_SITE) if kind == "specification" else None
        d = SD.world_designs(plant, spec, sites=g.site.tolist(), fchart=True)
        parts.append(g.reset_index(drop=True).join(d.drop(columns=["site"])))
    c = pd.concat(parts, ignore_index=True)
    c["corr_error_pp"] = 100 * (c.f_s - c.f_s_sim)
    c["fchart_error_pp"] = 100 * (c.f_s_fchart - c.f_s_mixed_sim)
    c["fchart_vs_strat_pp"] = 100 * (c.f_s_fchart - c.f_s_sim)
    c["gain_sim_pp"] = 100 * (c.f_s_sim - c.f_s_mixed_sim)
    c["within_p95"] = c.corr_error_pp.abs() <= c.p95_pp
    return c


def confirmation_table(c: pd.DataFrame) -> pd.DataFrame:
    """Table 7 of the paper (and Table S2.1 for the specification)."""
    rows = []
    for s in SCH:
        q = c[c.schedule.eq(s)]
        rows.append({"Load": NAME[s], "Sites": len(q), "RMSE": round(rmse(q.corr_error_pp), 2),
                     "Bias": round(q.corr_error_pp.mean(), 2), "Max |error|": round(q.corr_error_pp.abs().max(), 2),
                     "f-chart RMSE (mixed)": round(rmse(q.fchart_error_pp), 2),
                     "f-chart bias (strat.)": round(q.fchart_vs_strat_pp.mean(), 1),
                     "Mixed sim. RMSE": round(rmse(100 * (q.f_s_mixed_sim - q.f_s_sim)), 2),
                     "Gain median, pp": round(q.gain_sim_pp.median(), 1),
                     "Gain range, pp": f"{q.gain_sim_pp.min():.1f}-{q.gain_sim_pp.max():.1f}"})
    return pd.DataFrame(rows)


def world_summary(w: pd.DataFrame, label: str) -> dict:
    out = {}
    for s in SCH:
        q = w[w.schedule.eq(s)]
        out[f"{label}: f_s range, {NAME[s]}"] = f"{q.f_s.min():.2f}-{q.f_s.max():.2f}"
    if "collector_area_m2" in w and "U_L_implied_w_m2k" in w:
        out[f"{label}: collector area realizing it, m2"] = (f"{w.collector_area_m2.min():,.0f}-"
                                                          f"{w.collector_area_m2.max():,.0f}")
        out[f"{label}: storage limit T_p + H_T dT_L, C"] = f"{w.T_limit_C.min():.0f}-{w.T_limit_C.max():.0f}"
    return out


# ----------------------------------------------------------------------------------------------- figures
def figures(sz: dict, wd: dict, cplant: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"day_08_18": "#d95f02", "night_20_06": "#1b4f9c", "continuous_24h": "#2a9d4b"}
    # Fig. 7: sizing maps and sizing curves
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for k, s in enumerate(SCH):
        a = ax[0, k]
        w = sz["grid"][sz["grid"].schedule.eq(s)].pivot_table(index="storage_volume_m3", columns="collector_area_m2",
                                                              values="f_s")
        cf = a.contourf(w.columns, w.index, w.to_numpy(), levels=np.arange(0, 1.0001, 0.1), cmap="viridis",
                        vmin=0, vmax=1)
        a.contour(w.columns, w.index, w.to_numpy(), levels=(0.2, 0.4, 0.6, 0.8), colors="w", linewidths=0.6)
        for v in SPECIFIC:
            x = np.array([w.columns.min(), w.columns.max()])
            a.plot(x, x * v / 1000, "w:", lw=0.6)
        d = sz["designs"][sz["designs"].schedule.eq(s) & sz["designs"].reached]
        a.plot(d.collector_area_m2, d.storage_volume_m3, "o", mfc="none", mec="k", ms=4)
        a.set(xscale="log", yscale="log", xlabel="collector area (m²)", ylabel="storage volume (m³)",
              title=f"({'abc'[k]}) {NAME[s]} load", xlim=(w.columns.min(), w.columns.max()),
              ylim=(w.index.min(), w.index.max()))
    fig.colorbar(cf, ax=ax[0, :], label="annual solar fraction $f_s$", shrink=0.9)
    z = sz["confirmation"]
    for k, s in enumerate(SCH):
        a = ax[1, k]
        for v, ls in ((25.0, "--"), (50.0, "-")):
            c = SD.sizing_curve(stratfs.Plant(schedule=s), SIZING_SITE, v, n=240)
            c = c[c.f_s.between(0.15, 0.95)]
            a.plot(c.f_s, c.collector_area_m2, ls, color=color[s], label=f"{v:g} L/m², correlation")
            m = z[z.schedule.eq(s) & z.specific_l_m2.eq(v)]
            a.plot(m.f_s_sim, m.collector_area_m2, "o", color=color[s], mfc=color[s] if v == 50 else "w",
                   label=f"{v:g} L/m², simulated")
        a.set(yscale="log", xlim=(0.15, 0.95), ylim=(55, 2000), xlabel="annual solar fraction $f_s$",
              ylabel="collector area (m²)", title=f"(d{k + 1}) {NAME[s]} load")
        a.legend(fontsize=7, loc="upper left")
    fig.savefig(RESULTS / "fig7_sizing.png", dpi=200)
    plt.close(fig)
    # Fig. 8: the same plant across the world (continuous load), confirmation and stratification gain
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    a = fig.add_subplot(gs[0, :])
    q = wd["plant"][wd["plant"].schedule.eq("continuous_24h")]
    sc = a.scatter(q.longitude, q.latitude, c=q.f_s, cmap="viridis", s=18, edgecolors="0.2", linewidths=0.3)
    fig.colorbar(sc, ax=a, label="$f_s$, continuous load")
    a.set(xlim=(-180, 180), ylim=(-60, 75), xlabel="longitude (°)", ylabel="latitude (°)", aspect="equal",
          title=f"(a) The same 650 m², 10 m³ plant at {len(q)} sites: $f_s$ = {q.f_s.min():.2f}-{q.f_s.max():.2f}")
    b = fig.add_subplot(gs[1, 0])
    for s in SCH:
        qq = cplant[cplant.schedule.eq(s)]
        b.scatter(qq.f_s_sim, qq.f_s, s=14, color=color[s], edgecolors="0.2", linewidths=0.3,
                  label=f"{NAME[s]}: RMSE {rmse(qq.corr_error_pp):.2f} pp")
    x = np.array([0, 1])
    b.plot(x, x, "k-", lw=0.6)
    b.plot(x, x + 0.05, "--", color="0.5", lw=0.5)
    b.plot(x, x - 0.05, "--", color="0.5", lw=0.5)
    b.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$f_s$ simulated, stratified", ylabel="$f_s$ correlation",
          title=f"(b) Confirmation at {cplant.site.nunique()} sites")
    b.legend(fontsize=8, loc="upper left")
    c = fig.add_subplot(gs[1, 1])
    for s in SCH:
        qq = cplant[cplant.schedule.eq(s)]
        c.scatter(qq.f_s_mixed_sim, qq.gain_sim_pp, s=14, color=color[s], edgecolors="0.2", linewidths=0.3,
                  label=f"{NAME[s]}: median {qq.gain_sim_pp.median():.1f} pp")
    c.axhline(0, color="0.5", lw=0.6)
    c.set(xlim=(0, 1), xlabel="$f_s$ simulated, mixed storage", ylabel="$f_{s,strat} - f_{s,mix}$, simulated (pp)",
          title="(c) Simulated stratification gain")
    c.legend(fontsize=8, loc="lower right")
    fig.savefig(RESULTS / "fig8_world.png", dpi=200)
    plt.close(fig)
    # Fig. S2.13: the three load maps of the plant, and the specification maps
    for key, label in (("plant", "the 650 m², 10 m³ plant"), ("specification", "its Karachi specification")):
        fig, ax = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
        for a, s in zip(ax, SCH):
            q = wd[key][wd[key].schedule.eq(s)]
            sc = a.scatter(q.longitude, q.latitude, c=q.f_s, cmap="viridis", s=14, edgecolors="0.2", linewidths=0.3)
            fig.colorbar(sc, ax=a, label="$f_s$")
            a.set(xlim=(-180, 180), ylim=(-60, 75), aspect="equal",
                  title=f"{NAME[s]} load, {label}: $f_s$ = {q.f_s.min():.2f}-{q.f_s.max():.2f}")
        fig.savefig(RESULTS / f"figS_world_{key}.png", dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------- check
def check(sz: dict, wd: dict, cplant: pd.DataFrame, cspec: pd.DataFrame, s5: pd.DataFrame,
          ranges: pd.DataFrame) -> bool:
    """Compare the regenerated values with those of the manuscript (paper/expected). The design code of the paper
    took the temperature scale from the monthly mains table and the package takes it from the forcing of the fit;
    they differ by at most 3e-4 K, which moves f_s by less than 1e-5 and the realized sizes by less than 2e-5."""
    ok = True

    def report(name, err, tol):
        nonlocal ok
        good = bool(err <= tol)
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: max difference {err:.2e} (tolerance {tol:g})")

    print("Comparison with the manuscript values:")
    target = {"day_08_18": 2.30, "night_20_06": 2.25, "continuous_24h": 2.81}
    report("Section 5 test RMSE of the final form (pp)",
           max(abs(r.rmse_pp_final - target[r.schedule]) for r in s5.itertuples()), 0.005)
    acc = stratfs.constants()["accuracy"]      # Table 5 of the paper, as released with the package
    report("Section 5 accuracy ranges A-E: RMSE, bias, p95 (pp) and counts",
           max(max(abs(r.rmse_pp - acc[r.schedule][r.range]["rmse_pp"]), abs(r.bias_pp - acc[r.schedule][r.range]["bias_pp"]),
                   abs(r.p95_pp - acc[r.schedule][r.range]["p95_pp"]), abs(r.n - acc[r.schedule][r.range]["n"]))
               for r in ranges.itertuples()), 1e-6)
    e = pd.read_csv(EXPECTED / "sizing_designs.csv")
    m = sz["designs"].merge(e, on=["schedule", "target", "specific_l_m2"], suffixes=("", "_paper"))
    report("6.1 sized designs: reached", float((m.reached != m.reached_paper).sum()), 0)
    r = m[m.reached_paper]
    report("6.1 sized designs: collector area (relative)", float((r.collector_area_m2 / r.A_m2 - 1).abs().max()), 1e-6)
    report("6.1 sized designs: f_s", float((r.f_s - r.f_s_paper).abs().max()), 1e-5)
    for key, fname, cols in (("plant", "world_plant.csv", ("Phi_sys", "Psi_sys", "B_L", "HT", "f_s")),
                             ("specification", "world_specification.csv", ("f_s",))):
        e = pd.read_csv(EXPECTED / fname)
        m = wd[key].merge(e, on=["site", "schedule"], suffixes=("", "_paper"))
        report(f"6.{2 if key == 'plant' else 3} {key} at {m.site.nunique()} sites x {m.schedule.nunique()} loads: "
               f"{', '.join(cols)}", max(float((m[c] - m[c + '_paper']).abs().max()) for c in cols), 1e-5)
        if key == "specification":
            report("6.3 realization: collector area, storage volume (relative)",
                   max(float((m.collector_area_m2 / m.A_m2 - 1).abs().max()),
                       float((m.storage_volume_m3 / m.V_m3 - 1).abs().max())), 2e-5)
            report("6.3 realization: implied U_L (W/m2 K)", float((m.U_L_implied_w_m2k - m.U_L_implied).abs().max()),
                   2e-5)
    for name, c in (("confirm_plant", cplant), ("confirm_specification", cspec)):
        e = pd.read_csv(EXPECTED / f"{name}.csv")
        m = c.merge(e, on=["site", "schedule"], suffixes=("", "_paper"))
        report(f"{name}: correlation and f-chart at {len(m)} site-load cases",
               max(float((m.f_s - m.f_s_paper).abs().max()), float((m.f_s_fchart - m.f_s_fchart_paper).abs().max())),
               1e-5)
        report(f"{name}: accuracy ranges", float((m.accuracy_range != m.accuracy_range_paper).sum()), 0)
    print("All values reproduced." if ok else "Some values differ from the manuscript.")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare with the values of the manuscript")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    print("Section 5: accuracy on the 28 held-out test sites")
    s5, ranges = section5()
    s5.to_csv(RESULTS / "section5_test_rmse.csv", index=False)
    ranges.to_csv(RESULTS / "section5_accuracy_ranges.csv", index=False)
    print(s5.round(2).to_string(index=False))
    print("\nSection 6.1: sizing for a target solar fraction (Karachi, 50 m3/day at 80 C)")
    sz = sizing()
    sz["grid"].to_csv(RESULTS / "sizing_grid.csv", index=False)
    sz["designs"].to_csv(RESULTS / "sizing_designs.csv", index=False)
    sz["confirmation"].to_csv(RESULTS / "sizing_confirmation.csv", index=False)
    cols = ["schedule", "target", "specific_l_m2", "collector_area_m2", "storage_volume_m3", "R_ch", "accuracy_range",
            "f_s", "f_s_sim", "error_pp", "f_s_mixed_sim"]
    print(sz["confirmation"][cols].round(3).to_string(index=False))
    for k, v in sz["summary"].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nSections 6.2 and 6.3: one plant and one dimensionless specification across the world")
    wd = world()
    wd["plant"].to_csv(RESULTS / "world_plant.csv", index=False)
    wd["specification"].to_csv(RESULTS / "world_specification.csv", index=False)
    for k, v in (world_summary(wd["plant"], "plant") | world_summary(wd["specification"], "specification")).items():
        print(f"  {k}: {v}")
    cplant, cspec = confirmation("plant"), confirmation("specification")
    cplant.to_csv(RESULTS / "confirm_plant.csv", index=False)
    cspec.to_csv(RESULTS / "confirm_specification.csv", index=False)
    t7, ts = confirmation_table(cplant), confirmation_table(cspec)
    t7.to_csv(RESULTS / "table7_confirm_plant.csv", index=False)
    ts.to_csv(RESULTS / "tableS2_1_confirm_specification.csv", index=False)
    print("\nTable 7: the same plant at the 30 confirmation sites (pp)")
    print(t7.to_string(index=False))
    print(f"  pooled RMSE {rmse(cplant.corr_error_pp):.2f} pp; outside the test p95: {int((~cplant.within_p95).sum())} "
          f"of {len(cplant)}")
    print("\nTable S2.1: the same dimensionless specification at the 30 confirmation sites (pp)")
    print(ts.to_string(index=False))
    print(f"  pooled RMSE {rmse(cspec.corr_error_pp):.2f} pp")
    if not a.no_figures:
        figures(sz, wd, cplant)
        print(f"\nfigures written to {RESULTS}")
    print(f"tables written to {RESULTS}\n")
    return 0 if (not a.check or check(sz, wd, cplant, cspec, s5, ranges)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
