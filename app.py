"""Streamlit app: annual solar fraction of solar process-water heating with stratified storage.

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import stratfs
from stratfs.data import GROUPS, SCHEDULE_NAMES, SCHEDULES

st.set_page_config(page_title="Stratified-storage solar fraction", page_icon="☀️", layout="wide")

# Colours chosen to read on both a white and a dark page, and on the light land of the maps.
SERIES = {"day_08_18": "#2a78d6", "night_20_06": "#eb6834", "continuous_24h": "#1baf7a"}   # categorical slots 1-3
RANGE_COLORS = {"A": "#74a9cf", "B": "#3690c0", "C": "#0570b0", "D": "#045a8d", "E": "#023858"}   # ordinal A -> E
SEQ = [[0.0, "#9ecae1"], [0.35, "#4292c6"], [0.7, "#2171b5"], [1.0, "#08306b"]]   # solar fraction on maps
CONTOUR = "Blues"
NEUTRAL = "#7f7f7f"
# Symbols: LaTeX for Streamlit text (labels, help, markdown), HTML for Plotly.
TEX = {"Phi_sys": r"$\Phi_\mathrm{sys}$", "Psi_sys": r"$\Psi_\mathrm{sys}$", "B_L": "$B_L$",
       "R_ch": r"$R_\mathrm{ch}$", "HT": "$H_T$", "kappa_s": r"$\kappa_s$"}
HTML = {"Phi_sys": "Φ<sub>sys</sub>", "Psi_sys": "Ψ<sub>sys</sub>", "B_L": "B<sub>L</sub>",
        "R_ch": "R<sub>ch</sub>", "HT": "H<sub>T</sub>", "kappa_s": "κ<sub>s</sub>"}
GROUP_HELP = {
    "Phi_sys": "Effective collector-system solar-load ratio: annual effective absorbed resource over annual demand.",
    "Psi_sys": "Source thermal-loss number: collector loss at the temperature lift over the absorbed resource.",
    "B_L": "Storage-capacity group: daily storage capacity over the daily demand.",
    "R_ch": "Charging-to-load heat-capacity-rate ratio.",
    "HT": r"Storage temperature-headroom group, $(T_\mathrm{lim} - T_p)/\Delta T_L$.",
    "kappa_s": "Storage-leakage group: daily tank loss relative to the heat stored per kelvin."}
FORM_NAME = {"final": "Stratified-storage correlation", "implicit": "Implicit form", "explicit": "Explicit form"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13))
GEO = dict(showcountries=True, countrycolor="#9e9e9e", landcolor="#ecebe7", coastlinecolor="#9e9e9e",
           showocean=False, bgcolor="rgba(0,0,0,0)", showframe=False)
MARKER_EDGE = dict(color="#3d3d3d", width=0.6)


@st.cache_data
def site_table() -> pd.DataFrame:
    return stratfs.sites().reset_index(drop=True)


@st.cache_data
def world_for_plant(plant: stratfs.Plant) -> pd.DataFrame:
    return stratfs.world_plant(plant)


@st.cache_data
def world_for_groups(g: tuple, schedule: str, T_p: float) -> pd.DataFrame:
    return stratfs.world_groups(dict(zip(GROUPS, g)), schedule, T_p)


def label_site(name: str) -> str:
    s = site_table().set_index("site").loc[name]
    return f"{name.replace('_', ' ')} ({s.group.replace('_', ' ')}, {s.split})"


def fmt(x: float) -> str:
    return f"{x:.4g}"


# ----------------------------------------------------------------------------------------------- sidebar inputs
sites = site_table()
st.sidebar.title("Inputs")
mode = st.sidebar.radio("Input mode", ["Dimensional plant", "Dimensionless groups"], horizontal=True)
site = st.sidebar.selectbox("Site", sites.site, index=int(np.flatnonzero(sites.site.eq("Karachi"))[0]),
                            format_func=label_site, help="267 sites; the 28 test sites were never used in fitting.")
schedule = st.sidebar.selectbox("Load schedule", SCHEDULES, format_func=SCHEDULE_NAMES.get)
T_p = st.sidebar.number_input(r"Process temperature $T_p$ (°C)", 40.0, 150.0, 80.0, 1.0,
                              help=r"The correlation was fitted for a temperature lift $T_p - T_{m,\mathrm{ref}}$ "
                                   r"of 45-80 K, where $T_{m,\mathrm{ref}}$ is the annual mean mains temperature.")
compare = st.sidebar.checkbox("Also show the explicit and implicit forms", value=False)
forms = stratfs.FORMS[::-1] if compare else ("final",)

ref = stratfs.Plant(schedule=schedule, T_process_C=T_p)
if mode == "Dimensional plant":
    with st.sidebar.expander("Plant size and load", expanded=True):
        A = st.number_input("Collector aperture area (m²)", 1.0, 1e6, 650.0, 10.0)
        V = st.number_input("Storage volume (m³)", 0.05, 1e5, 10.0, 0.5)
        load = st.number_input("Process water per day (m³/day)", 0.1, 1e5, 50.0, 1.0)
        C_ch = st.number_input("Charging-loop heat-capacity rate (W/K)", 1.0, 1e8, ref.charging_rate_w_k, 100.0,
                               help="Mass flow × specific heat of the storage-charging water loop.")
    with st.sidebar.expander("Collector and primary loop"):
        ta = st.number_input(r"$(\tau\alpha)_n$", 0.3, 1.0, ref.tau_alpha, 0.01)
        UL = st.number_input(r"$U_L$ (W/m²K)", 0.2, 10.0, ref.U_L_w_m2k, 0.1)
        Fp = st.number_input(r"Collector efficiency factor $F'$", 0.5, 1.0, ref.F_prime, 0.01)
        oil = st.number_input("Oil-loop heat-capacity rate per m² of aperture (W/m²K)", 1.0, 200.0,
                              ref.oil_rate_w_k_m2, 0.5)
        pipe = st.number_input("Primary pipe conductance per leg per m² (W/m²K)", 0.0, 1.0, ref.pipe_w_k_per_leg_m2,
                               0.001, format="%.4f")
        hx = st.number_input("Heat-exchanger effectiveness", 0.1, 1.0, ref.hx_effectiveness, 0.01)
    with st.sidebar.expander("Storage"):
        Tlim = st.number_input(r"Storage limit temperature $T_\mathrm{lim}$ (°C)", T_p + 1.0, 200.0,
                               max(99.0, T_p + 1.0), 1.0)
        Uins = st.number_input("Insulation U (W/m²K)", 0.01, 5.0, 0.4, 0.05)
        aspect = st.number_input("Height / diameter", 0.5, 5.0, 2.0, 0.1)
    plant = stratfs.Plant(collector_area_m2=A, storage_volume_m3=V, daily_load_m3=load, schedule=schedule,
                          T_process_C=T_p, charging_rate_w_k=C_ch, tau_alpha=ta, U_L_w_m2k=UL, F_prime=Fp,
                          oil_rate_w_k_m2=oil, pipe_w_k_per_leg_m2=pipe, hx_effectiveness=hx, T_limit_C=Tlim,
                          insulation_w_m2k=Uins, aspect=aspect)
    res = stratfs.predict_plant(plant, site, forms=forms).iloc[0]
    g = {k: float(res[k]) for k in GROUPS}
else:
    defaults = dict(Phi_sys=0.883, Psi_sys=0.401, B_L=0.20, R_ch=2.0, HT=0.352, kappa_s=0.0223)
    g = {}
    with st.sidebar.expander("Similarity groups", expanded=True):
        for k in GROUPS:
            lo, hi = stratfs.constants()["validity"][k]
            g[k] = st.number_input(f"{TEX[k]}  (fitted {lo:g}-{hi:g})", 1e-4, 100.0, defaults[k], 0.01,
                                   format="%.4f", help=GROUP_HELP[k])
    plant = None
    res = stratfs.predict_groups(pd.DataFrame([g]), site, schedule, T_p, forms).iloc[0]

f_col = "f_s" if len(forms) == 1 else "f_s_final"
f_s = float(res[f_col])
monthly = stratfs.solar_fraction(*(g[k] for k in GROUPS), site=site, schedule=schedule, T_process=T_p, monthly=True)
dT = float(stratfs.temperature_scale([site], T_p)[0])

st.title("Annual solar fraction with stratified storage")
st.markdown("Companion app to the manuscript *A physics-informed dimensionless correlation for the annual solar "
            "fraction of solar process-water heating with stratified storage*, M. Usama Siddiqui and Syed M. Zubair, "
            "submitted to **Energy Conversion and Management**.")
st.caption("Solar process-water heating: collector field, closed oil loop, external heat exchanger, stratified "
           "storage and auxiliary heater. A physics-informed dimensionless correlation, fitted on 49,152 converged "
           "simulations at 239 sites and tested on 28 held-out sites.")

tabs = st.tabs(["Result", "Parametric sweep", "Design map", "World map", "Sites", "About"])

# ----------------------------------------------------------------------------------------------- result
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(r"Annual solar fraction $f_s$", f"{f_s:.3f}")
    c2.metric("Accuracy range", res.accuracy_range, help="Observed on the 28 held-out test sites.")
    c3.metric("Typical error (test RMSE)", f"{res.rmse_pp:.1f} pp", help=f"95th percentile {res.p95_pp:.1f} pp")
    if plant is not None:
        c4.metric("Solar heat delivered", f"{res.Q_solar_GJ:,.0f} GJ/yr", help=f"Annual demand {res.E_D_GJ:,.0f} GJ")
    else:
        c4.metric(r"Temperature lift $\Delta T_L$", f"{dT:.1f} K")
    if not bool(res.valid):
        out = [TEX[k] for k in GROUPS if not bool(res[f"{k}_ok"])]
        if not bool(res.get("dT_L_ok", True)):
            out.append(r"the temperature lift $\Delta T_L$")
        st.warning("Outside the fitted domain (" + ", ".join(out) + "): the correlation extrapolates and its "
                   "accuracy is not established.", icon="⚠️")
    left, right = st.columns([3, 2])
    with left:
        f_m, d_L = monthly[1][0], monthly[2][0]
        fig = go.Figure(go.Bar(x=MONTHS, y=f_m, marker_color=SERIES[schedule],
                               hovertemplate="%{x}: f<sub>s,j</sub> = %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Monthly solar fraction", yaxis=dict(title="f<sub>s,j</sub>", range=[0, 1.02]),
                          showlegend=False, height=340, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Bar(x=MONTHS, y=100 * d_L, marker_color=NEUTRAL,
                               hovertemplate="%{x}: %{y:.1f} % of the annual demand<extra></extra>"))
        fig.update_layout(title="Share of the annual demand per month", yaxis_title="% of annual demand",
                          height=260, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with right:
        rows = ["| Group | Value | Fitted range | In range |", "|---|---:|---:|:---:|"]
        for k in GROUPS:
            lo, hi = stratfs.constants()["validity"][k]
            rows.append(f"| {TEX[k]} | {fmt(g[k])} | {lo:g}-{hi:g} | {'✓' if bool(res[f'{k}_ok']) else '✗'} |")
        st.markdown("**Similarity groups**\n\n" + "\n".join(rows))
        if compare:
            st.markdown("**The three fitted forms**\n\n| Form | Annual $f_s$ |\n|---|---:|\n"
                        + "\n".join(f"| {FORM_NAME[f]} | {res[f'f_s_{f}']:.3f} |" for f in forms))
        acc = stratfs.accuracy_table(schedule)
        rows = ["| Range | RMSE (pp) | p95 (pp) | Test designs |", "|:---:|---:|---:|---:|"]
        for r, a in acc.iterrows():
            mark = " **(this design)**" if r == res.accuracy_range else ""
            rows.append(f"| {r}{mark} | {a.rmse_pp:.2f} | {a.p95_pp:.2f} | {int(a.n)} |")
        st.markdown(f"**Accuracy ranges, {SCHEDULE_NAMES[schedule].lower()}** (held-out sites)\n\n" + "\n".join(rows))
        if plant is not None:
            st.caption(rf"$F_\mathrm{{sys}}$ = {res.F_sys:.3f}, $F_R$ = {res.F_R:.3f}, storage loss "
                       rf"$UA_s$ = {res.UA_s_w_k:.1f} W/K, $\Delta T_L$ = {res.dT_L_K:.1f} K.")

# ----------------------------------------------------------------------------------------------- sweep
with tabs[1]:
    if plant is not None:
        choices = {"Collector area (m²)": ("collector_area_m2", plant.collector_area_m2, "Collector area (m²)"),
                   "Storage volume (m³)": ("storage_volume_m3", plant.storage_volume_m3, "Storage volume (m³)"),
                   "Charging-loop rate (W/K)": ("charging_rate_w_k", plant.charging_rate_w_k,
                                                "Charging-loop heat-capacity rate (W/K)"),
                   "Insulation U (W/m²K)": ("insulation_w_m2k", plant.insulation_w_m2k, "Insulation U (W/m²K)")}
    else:
        choices = {TEX[k]: (k, g[k], HTML[k]) for k in GROUPS}
    var = st.radio("Vary", list(choices), horizontal=True)
    key, x0, axis = choices[var]
    b, c = st.columns(2)
    lo = b.number_input("From", value=float(x0) / 4, min_value=1e-6, format="%.4g")
    hi = c.number_input("To", value=float(x0) * 4, min_value=1e-6, format="%.4g")
    x = np.geomspace(min(lo, hi), max(lo, hi), 80)
    fig = go.Figure()
    for s in SCHEDULES:
        if plant is not None:
            out = stratfs.predict_plant(plant.with_(schedule=s), site, **{key: x})
        else:
            G = pd.DataFrame({k: np.full(len(x), g[k]) for k in GROUPS})
            G[key] = x
            out = stratfs.predict_groups(G, site, s, T_p)
        fs, valid = out.f_s.to_numpy(), out.valid.to_numpy()
        fig.add_scatter(x=x, y=np.where(valid, fs, np.nan), mode="lines", name=SCHEDULE_NAMES[s],
                        line=dict(color=SERIES[s], width=2.5), legendgroup=s,
                        hovertemplate=f"{SCHEDULE_NAMES[s]}<br>{axis} = %{{x:.4g}}<br>f<sub>s</sub> = %{{y:.3f}}"
                                      "<extra></extra>")
        fig.add_scatter(x=x, y=np.where(valid, np.nan, fs), mode="lines", showlegend=False, legendgroup=s,
                        line=dict(color=SERIES[s], width=1.5, dash="dot"),
                        hovertemplate="outside the fitted domain<br>%{x:.4g}: %{y:.3f}<extra></extra>")
    fig.add_vline(x=x0, line=dict(color=NEUTRAL, width=1, dash="dash"))
    fig.update_layout(title=f"Annual solar fraction against {axis}, {site.replace('_', ' ')}",
                      xaxis=dict(title=axis, type="log"), yaxis=dict(title="Annual solar fraction", range=[0, 1.02]),
                      legend=dict(orientation="h", y=-0.2), height=480, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Solid: inside the fitted domain. Dotted: outside it (extrapolation). Dashed line: the current design. "
               "Other inputs are held at their current values.")

# ----------------------------------------------------------------------------------------------- design map
with tabs[2]:
    n = 45
    if plant is not None:
        xa = np.geomspace(plant.collector_area_m2 / 5, plant.collector_area_m2 * 5, n)
        yv = np.geomspace(plant.storage_volume_m3 / 5, plant.storage_volume_m3 * 5, n)
        X, Y = np.meshgrid(xa, yv)
        out = stratfs.predict_plant(plant, site, collector_area_m2=X.ravel(), storage_volume_m3=Y.ravel())
        xt, yt, x_now, y_now = "Collector area (m²)", "Storage volume (m³)", plant.collector_area_m2, plant.storage_volume_m3
    else:
        xa, yv = np.geomspace(0.1, 3.0, n), np.geomspace(0.1, 3.0, n)
        X, Y = np.meshgrid(xa, yv)
        G = pd.DataFrame({k: np.full(X.size, g[k]) for k in GROUPS})
        G["Phi_sys"], G["B_L"] = X.ravel(), Y.ravel()
        out = stratfs.predict_groups(G, site, schedule, T_p)
        xt, yt, x_now, y_now = HTML["Phi_sys"], HTML["B_L"], g["Phi_sys"], g["B_L"]
    Z = out.f_s.to_numpy().reshape(X.shape)
    fig = go.Figure(go.Contour(x=xa, y=yv, z=Z, colorscale=CONTOUR, zmin=0, zmax=1,
                               contours=dict(showlabels=True, start=0.1, end=0.9, size=0.1,
                                             labelfont=dict(color="white")),
                               colorbar=dict(title="f<sub>s</sub>"),
                               hovertemplate=f"{xt} %{{x:.4g}}<br>{yt} %{{y:.4g}}<br>f<sub>s</sub> %{{z:.3f}}"
                                             "<extra></extra>"))
    fig.add_scatter(x=[x_now], y=[y_now], mode="markers", name="current design",
                    marker=dict(size=13, color=SERIES["night_20_06"], line=dict(color="white", width=2)))
    fig.update_layout(title=f"Annual solar fraction, {SCHEDULE_NAMES[schedule].lower()}, {site.replace('_', ' ')}",
                      xaxis=dict(title=xt, type="log"), yaxis=dict(title=yt, type="log"), height=560, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    inside = out.valid.to_numpy().reshape(X.shape)
    st.caption(f"{100 * inside.mean():.0f} % of this map lies inside the fitted domain. Other inputs are held at "
               "their current values.")

# ----------------------------------------------------------------------------------------------- world map
with tabs[3]:
    if plant is not None:
        w = world_for_plant(plant)
        what = "the same plant"
    else:
        w = world_for_groups(tuple(g[k] for k in GROUPS), schedule, T_p)
        what = "the same dimensionless specification"
    colour = st.radio("Colour by", ["Annual solar fraction", "Accuracy range"], horizontal=True)
    w = w.assign(name=w.site.str.replace("_", " "), fit=np.where(w.valid, "inside fitted domain", "extrapolated"))
    hover = {"f_s": ":.3f", "accuracy_range": True, "group": True, "fit": True, "latitude": False, "longitude": False}
    if colour == "Annual solar fraction":
        fig = px.scatter_geo(w, lat="latitude", lon="longitude", color="f_s", color_continuous_scale=SEQ,
                             range_color=(0, 1), hover_name="name", hover_data=hover, projection="natural earth",
                             labels={"f_s": "f<sub>s</sub>"})
    else:
        fig = px.scatter_geo(w, lat="latitude", lon="longitude", color="accuracy_range",
                             color_discrete_map=RANGE_COLORS, category_orders={"accuracy_range": list("ABCDE")},
                             hover_name="name", hover_data=hover, projection="natural earth",
                             labels={"accuracy_range": "Accuracy range"})
    fig.update_traces(marker=dict(size=10, line=MARKER_EDGE))
    fig.update_geos(**GEO)
    fig.update_layout(title=f"{what.capitalize()} at all 267 sites, {SCHEDULE_NAMES[schedule].lower()}",
                      height=580, coloraxis_colorbar=dict(orientation="h", y=-0.08, len=0.6, thickness=14),
                      legend=dict(orientation="h", y=-0.05), **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Annual solar fraction from {w.f_s.min():.2f} to {w.f_s.max():.2f}; "
               f"{int(w.valid.sum())} of {len(w)} sites inside the fitted domain.")
    cols = ["site", "group", "split", "latitude", "longitude", "f_s", "accuracy_range", "rmse_pp", "valid"]
    cols += [c for c in ("Phi_sys", "Psi_sys", "B_L", "R_ch", "HT", "kappa_s", "E_D_GJ", "Q_solar_GJ") if c in w]
    st.download_button("Download the map as CSV", w[cols].to_csv(index=False), "solar_fraction_world.csv", "text/csv")

# ----------------------------------------------------------------------------------------------- sites
with tabs[4]:
    sel = sites.assign(name=sites.site.str.replace("_", " "),
                       role=np.where(sites.split.eq("test"), "test (held out)", "training"))
    fig = px.scatter_geo(sel, lat="latitude", lon="longitude", color="role",
                         color_discrete_map={"training": SERIES["day_08_18"], "test (held out)": SERIES["night_20_06"]},
                         hover_name="name", projection="natural earth", labels={"role": "Site"},
                         hover_data={"group": True, "H_kWh_m2": ":.0f", "K_T": ":.2f", "T_a_C": ":.1f",
                                     "latitude": False, "longitude": False, "role": False})
    fig.update_traces(marker=dict(size=9, line=MARKER_EDGE))
    fig.update_geos(**GEO)
    fig.update_layout(title="The 267 sites: 239 training and 28 held-out test sites", height=500, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    m = stratfs.monthly_inputs()
    m = m[m.site.eq(site)]
    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=MONTHS, y=m.H / 3.6, marker_color=SERIES["day_08_18"],
                           hovertemplate="%{x}: %{y:.2f} kWh/m² per day<extra></extra>"))
    fig.update_layout(title=f"Daily global horizontal irradiation, {site.replace('_', ' ')}",
                      yaxis_title="kWh/m² per day", height=320, **LAYOUT)
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=MONTHS, y=m.Ta, name="Ambient", line=dict(color=SERIES["night_20_06"], width=2.5))
    fig.add_scatter(x=MONTHS, y=m.Tmains, name="Mains water", line=dict(color=SERIES["day_08_18"], width=2.5))
    fig.update_layout(title="Monthly mean temperatures", yaxis_title="°C", legend=dict(orientation="h", y=-0.25),
                      height=320, **LAYOUT)
    b.plotly_chart(fig, width="stretch")
    st.download_button("Download all site data (CSV)", stratfs.sites().to_csv(index=False), "sites.csv", "text/csv")
    st.download_button("Download all monthly inputs (CSV)", stratfs.monthly_inputs().to_csv(index=False),
                       "monthly_inputs.csv", "text/csv")

# ----------------------------------------------------------------------------------------------- about
with tabs[5]:
    st.markdown(r"""
**What it predicts.** The annual solar fraction of a solar process-water heating plant with a stratified storage,
for a daytime (08:00-18:00), night-time (20:00-06:00) or continuous load, from six similarity groups and the
monthly climate of the site. Plants of any size with the same groups and climate forcing behave alike.

**Fitted domain.** $\Phi_\mathrm{sys}$, $\Psi_\mathrm{sys}$ and $B_L$ 0.1-3, $R_\mathrm{ch}$ 1-10, $H_T$ 0.05-3,
$\kappa_s$ 0.01-0.2, and a temperature lift $T_p - T_{m,\mathrm{ref}}$ of 45-80 K. Outside it the correlation
extrapolates and its accuracy is not established.

**Accuracy.** Test RMSE 2.20, 2.21 and 2.72 percentage points for the daytime, night-time and continuous loads on
28 held-out sites. The accuracy ranges A-E (set by $\Phi_\mathrm{sys}$ and $\Psi_\mathrm{sys}$) give the typical error
for designs in each range, not a prediction interval for an individual design.

**Forms.** The stratified-storage correlation is the recommended form. The explicit and implicit forms are the
simpler forms of the paper, kept for comparison.

**Manuscript.** M. Usama Siddiqui, Syed M. Zubair. *A physics-informed dimensionless correlation for the annual
solar fraction of solar process-water heating with stratified storage.* Submitted to *Energy Conversion and
Management*. Please cite it if you use this app, the package or its data.

**Assumptions.** Fixed top-inlet and bottom-outlet storage ports, ideal mixing of unstable layers, no charging or
process water pipes, constant-flow loads, climate from typical meteorological years (PVGIS).
""")
