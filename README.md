# stratfs: stratified-storage solar fraction

Annual solar fraction of solar process-water heating plants with a **stratified storage**, from a
physics-informed dimensionless correlation, with the climate data of 267 sites worldwide and a browser app for
dimensional and dimensionless design.

This repository accompanies the manuscript:

> M. Usama Siddiqui, Syed M. Zubair. **A physics-informed dimensionless correlation for the annual solar fraction
> of solar process-water heating with stratified storage.** Submitted to *Energy Conversion and Management*.

The plant is a collector field with a closed oil loop and an external heat exchanger charging a stratified
water storage, from which a process load draws hot water tempered to the process temperature, with an auxiliary
heater for the remainder. Plants of any size with the same similarity groups and climate behave alike, so the
correlation is fitted once for all plant sizes. It was fitted on 49,152 converged simulations at 239 sites and
tested on 28 held-out sites in seven climate groups.

| Load schedule | Test RMSE (percentage points) |
|---|---:|
| Daytime, 08:00-18:00 | 2.20 |
| Night-time, 20:00-06:00 | 2.21 |
| Continuous, 24 h | 2.72 |

## Install

```bash
git clone https://github.com/musiddiqui85/stratfs.git
cd stratfs
pip install -e .            # the library: numpy, scipy, pandas
pip install -e ".[app]"     # plus the app: streamlit, plotly
```

Python 3.10 or later.

## The app

```bash
streamlit run app.py
```

It opens in the browser. Choose a site, the load schedule and the process temperature, and enter either a
**dimensional plant** (collector area, storage volume, daily load, charging loop, collector and storage
properties) or the **dimensionless groups** directly. It shows:

- **Result:** the annual solar fraction, the accuracy range and its typical error, the monthly solar fractions,
  the groups and whether each lies inside the fitted domain, and optionally the three fitted forms side by side.
- **Parametric sweep:** the solar fraction against one input for the three load schedules, with the part outside
  the fitted domain marked.
- **Design map:** contours of the solar fraction over collector area and storage volume (or Φ_sys and B_L).
- **World map:** the same plant, or the same dimensionless specification, at all 267 sites, downloadable as CSV.
- **Sites:** the site map and the monthly climate of the selected site, with all site data downloadable.

The app can also be hosted free on [Streamlit Community Cloud](https://streamlit.io/cloud) from a public GitHub
repository: point it at `app.py`. It installs `requirements.txt`.

## The library

```python
import stratfs

# Dimensionless: the six groups at a site (Karachi reference plant, daytime load)
stratfs.solar_fraction(0.883, 0.401, 0.20, 2.0, 0.352, 0.0223, site="Karachi", schedule="day_08_18")
# -> array([0.7888])

# Dimensional: a plant at a site, with its groups, accuracy range and validity flags
plant = stratfs.Plant(collector_area_m2=650, storage_volume_m3=10, daily_load_m3=50, schedule="day_08_18")
stratfs.predict_plant(plant, "Karachi")[["Phi_sys", "Psi_sys", "B_L", "f_s", "accuracy_range", "rmse_pp", "valid"]]

# Sweeps: any plant field as an array
stratfs.predict_plant(plant, "Karachi", collector_area_m2=[300, 650, 1300]).f_s

# The same plant at every site
stratfs.world_plant(plant)

# Monthly values, and the explicit and implicit forms
f_s, f_monthly, demand_share = stratfs.solar_fraction(0.883, 0.401, 0.20, 2.0, 0.352, 0.0223, site="Karachi",
                                                      schedule="day_08_18", monthly=True)
stratfs.solar_fraction(0.883, 0.401, 0.20, 2.0, 0.352, 0.0223, site="Karachi", form="implicit")
```

`examples/quickstart.py` runs these.

### Inputs

| Group | Meaning | Fitted range |
|---|---|---|
| Φ_sys | effective collector-system solar-load ratio: annual effective absorbed resource / annual demand | 0.1-3 |
| Ψ_sys | source thermal-loss number: collector loss at the temperature lift / absorbed resource | 0.1-3 |
| B_L | storage-capacity group: daily storage capacity / daily demand | 0.1-3 |
| R_ch | charging-to-load heat-capacity-rate ratio | 1-10 |
| H_T | storage temperature-headroom group, (T_lim - T_p) / ΔT_L | 0.05-3 |
| κ_s | storage-leakage group, UA_s · 1 day / (ρ c_p V) | 0.01-0.2 |

The temperature scale is ΔT_L = T_p - T_m,ref, the process temperature less the site's annual mean mains
temperature; the correlation was fitted for ΔT_L of 45-80 K. The load schedules are a daytime (08:00-18:00), a
night-time (20:00-06:00) and a continuous constant-flow load. `stratfs.Plant` converts a dimensional plant to the
groups, with the reference plant of the paper as its defaults (650 m² of evacuated tubes, (τα)_n 0.75, U_L
1.5 W/m²K; a 10 m³ atmospheric storage limited to 99 °C with 0.4 W/m²K insulation; 50 m³/day at 80 °C).

### Accuracy ranges

The error is set mainly by Φ_sys and Ψ_sys. Five ranges summarize the errors observed on the 28 held-out test
sites; they indicate the typical error for designs in each range, not a prediction interval for one design.

| Range | Φ_sys | Ψ_sys | RMSE daytime / night / continuous (pp) |
|---|---|---|---|
| A | ≤ 0.3 | any | 0.74 / 0.82 / 0.66 |
| B | 0.3-0.6 | ≤ 1 | 1.10 / 1.18 / 1.42 |
| C | 0.3-0.6 | > 1 | 2.02 / 2.21 / 2.05 |
| D | > 0.6 | ≤ 1 | 2.39 / 2.20 / 3.30 |
| E | > 0.6 | > 1 | 3.96 / 4.12 / 4.58 |

Outside the fitted domain the correlation extrapolates and its accuracy is not established; the library flags
every group and the temperature lift (`valid`).

### The three fitted forms

- `final` (default): the **stratified-storage correlation**, a monthly balance with the feedback of the solar
  fraction on the collector return temperature, stratification set by tank turnover, curtailment at the storage
  limit, tank loss and a daily delivery cap under day-to-day variability.
- `implicit`: the same monthly balance without curtailment and with the plain turnover number.
- `explicit`: an annual balance with the return temperature a fixed function of the collector-to-storage ratio.

The constants of all three, per load schedule and at full precision, are in `stratfs/data/constants.json`.

## Data

All in `stratfs/data`:

| File | Content |
|---|---|
| `sites.csv` | 267 sites: location, climate group, training or test site, annual global and absorbed irradiation (kWh/m²), clearness index, ambient and mains temperatures (°C), and the explicit form's site inputs |
| `monthly_inputs.csv` | 12 months per site: days, daily global and diffuse irradiation (MJ/m² per day), clearness index, (τα), ambient and mains temperatures, sunset hour angle, noon-hour share and tilt factors, monthly share of the absorbed resource, and the normalized ambient and mains temperatures at T_p = 80 °C |
| `utilizability.csv` | the site utilizability φ_site(x) on a 101-point grid (explicit form) |
| `constants.json` | constants of the three forms, the variability closure, the validity domain and the accuracy ranges |

The absorbed irradiation is that on a collector tilted at the latitude and facing the equator, with (τα)_n 0.75
and an isotropic sky. Hourly typical meteorological years are from PVGIS (European Commission Joint Research
Centre); mains temperatures follow the NREL relation of Burch and Christensen (2007) with the phase anchored to the
coldest month. Monthly utilizability follows Klein (1978).

`tests/data/reference_predictions.csv` holds the 6,144 held-out test designs with the predictions of the three
forms from the code used in the paper and the simulated (grid-extrapolated) solar fraction; the tests check that
this package reproduces them.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Citation

If you use this package or its data, please cite the manuscript (details in `CITATION.cff`):

> M. Usama Siddiqui, Syed M. Zubair, A physics-informed dimensionless correlation for the annual solar fraction of
> solar process-water heating with stratified storage, submitted to *Energy Conversion and Management*.

The citation will be updated with the volume, pages and DOI once the article is published.

## Licence

Code: MIT licence (`LICENSE`). Data in `stratfs/data` and `tests/data`: Creative Commons Attribution 4.0
(CC BY 4.0); the underlying weather data are from PVGIS, © European Union, reused under the Commission's
reuse policy.
