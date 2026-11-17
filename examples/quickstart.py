"""Quick start: the reference plant of the paper (Karachi, daytime load) in dimensionless and dimensional form.

    python examples/quickstart.py
"""
import stratfs

# Dimensionless: the six similarity groups at a site
f = stratfs.solar_fraction(0.908, 0.390, 0.20, 2.0, 0.352, 0.0223, site="Karachi", schedule="day_08_18")
print(f"Annual solar fraction from the groups: {f[0]:.4f}")

# Dimensional: 650 m2 of evacuated tubes, a 10 m3 storage and 50 m3/day of process water at 80 C
plant = stratfs.Plant(collector_area_m2=650, storage_volume_m3=10, daily_load_m3=50, schedule="day_08_18")
r = stratfs.predict_plant(plant, "Karachi").iloc[0]
print(f"Groups: Phi_sys {r.Phi_sys:.3f}, Psi_sys {r.Psi_sys:.3f}, B_L {r.B_L:.3f}, R_ch {r.R_ch:.2f}, "
      f"H_T {r.HT:.3f}, kappa_s {r.kappa_s:.4f}")
print(f"Annual solar fraction {r.f_s:.4f}, accuracy range {r.accuracy_range} "
      f"(test RMSE {r.rmse_pp:.2f} pp), inside the fitted domain: {bool(r.valid)}")
print(f"Solar heat {r.Q_solar_GJ:,.0f} GJ of an annual demand of {r.E_D_GJ:,.0f} GJ")

# The three fitted forms
for form in stratfs.FORMS:
    print(f"  {form:8s} f_s = {stratfs.solar_fraction(r.Phi_sys, r.Psi_sys, r.B_L, r.R_ch, r.HT, r.kappa_s, site='Karachi', form=form)[0]:.4f}")

# A collector-area sweep and the same plant at every site
sweep = stratfs.predict_plant(plant, "Karachi", collector_area_m2=[200, 400, 650, 1000, 1500])
print(sweep[["collector_area_m2", "Phi_sys", "f_s", "valid"]].round(3).to_string(index=False))
world = stratfs.world_plant(plant)
print(f"Same plant at {len(world)} sites: f_s from {world.f_s.min():.2f} to {world.f_s.max():.2f}")

# Design (Section 6 of the paper). Dimensional: the smallest collector area for a target solar fraction at
# 25 and 50 litres of storage per m2 of collector, each at its best charging flow
sized = stratfs.size_for_target(plant, "Karachi", targets=(0.4, 0.6, 0.8), specific_storage_l_m2=(25, 50))
print(sized[["target", "specific_l_m2", "collector_area_m2", "storage_volume_m3", "R_ch", "f_s",
             "accuracy_range"]].round(3).to_string(index=False))

# Dimensionless: the plant as a specification (its groups in Karachi), and the plant that realizes it in Seville
spec = stratfs.plant_specification(plant, "Karachi")
r = stratfs.realize(spec, "Seville", plant)
print(f"In Seville the specification needs {r['collector_area_m2']:,.0f} m2, {r['storage_volume_m3']:.1f} m3, "
      f"a collector U_L of {r['U_L_implied_w_m2k']:.2f} W/m2K and a storage limit of {r['T_limit_C']:.0f} C")
spec_world = stratfs.world_specification(spec, plant)
print(f"Same specification at {len(spec_world)} sites: f_s from {spec_world.f_s.min():.2f} to {spec_world.f_s.max():.2f}")
