"""
app.py
NADI AI - Streamlit Application
---------------------------------
Run with:  streamlit run app.py

Blue-themed, AI-assistant style UI: pick a station from an interactive map,
view its basic details and the annual data availability plot, and run the
full hydrological analysis to download a technical PDF report. Also lets
the user download the raw discharge and water level (MSL) data used for
the selected station.
"""

import os
import base64
import io
import zipfile

import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

import nadi_data_collec as dc
import nadi_quality as ql
import nadi_statisticaltests as st_tests
import nadi_distfit as dfit
import nadi_plot as pl
import nadi_report as rp
import nadi_chatbot as cb

# ---------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NADI AI",
    page_icon="\U0001F4A7",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_BLUE = "#0B5394"
LIGHT_BLUE = "#3D85C6"
ACCENT_BLUE = "#9FC5E8"
DEEP_BLUE = "#073763"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LOGO_PATH = BASE_DIR / "NADI AI LOGO.jpg"
NAME_FILE = BASE_DIR / "DATA" / "camels_ind_name.csv"
TOPO_FILE = BASE_DIR / "DATA" / "camels_ind_topo.csv"

# ---------------------------------------------------------------------------
# CHATBOT CONFIG (OpenRouter)
# ---------------------------------------------------------------------------
# Prefer st.secrets / environment variable; falls back to the key below only
# if neither is set. For production, set OPENROUTER_API_KEY in
# .streamlit/secrets.toml or as an environment variable instead of leaving
# it hardcoded here.
def _get_openrouter_api_key():
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "XYZ")


OPENROUTER_API_KEY = _get_openrouter_api_key()
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

st.markdown(
    f"""
    <style>
    .main {{ background-color: #F5F9FD; }}
    h1, h2, h3 {{ color: {PRIMARY_BLUE}; }}
    div.stButton > button {{
        background: linear-gradient(90deg, {PRIMARY_BLUE}, {LIGHT_BLUE});
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(90deg, {LIGHT_BLUE}, {ACCENT_BLUE});
        color: white;
    }}
    div.stDownloadButton > button {{
        background: linear-gradient(90deg, {PRIMARY_BLUE}, {LIGHT_BLUE});
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }}
    div.stDownloadButton > button:hover {{
        background: linear-gradient(90deg, {LIGHT_BLUE}, {ACCENT_BLUE});
        color: white;
    }}
    .nadi-header {{
        background: linear-gradient(120deg, {DEEP_BLUE}, {PRIMARY_BLUE} 60%, {LIGHT_BLUE});
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 24px;
    }}
    .nadi-header-text {{
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .nadi-badge {{
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 6px;
    }}
    .nadi-card {{
        background-color: white;
        border: 1px solid {ACCENT_BLUE};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .nadi-selected-banner {{
        background-color: {ACCENT_BLUE};
        color: {DEEP_BLUE};
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        margin: 10px 0 18px 0;
    }}
    .nadi-footer {{
        color: #888888;
        font-size: 0.8rem;
        margin-top: 30px;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CACHED DATA LOADERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_stations():
    return dc.load_station_list()


@st.cache_data(show_spinner="Reading station data...")
def cached_station_data(gauge_id):
    return dc.get_station_data(gauge_id)


@st.cache_data(show_spinner=False)
def load_map_data():
    """Load station names merged with topo (lat/lon + catchment attributes)
    for plotting on the map, exactly like the working reference script."""
    names = pd.read_csv(NAME_FILE)
    topo = pd.read_csv(TOPO_FILE)
    df = pd.merge(names, topo, on="gauge_id", how="inner")
    df = df.dropna(subset=["cwc_lat", "cwc_lon"])
    return df


def get_base64_img(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def _df_to_csv_bytes(df_in):
    """Convert a DataFrame to CSV bytes for st.download_button. Returns empty
    CSV bytes (header-only or fully empty) if df_in is None or empty."""
    if df_in is None or df_in.empty:
        return "".encode("utf-8")
    buf = io.StringIO()
    df_in.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _station_data_zip_bytes(discharge_df, waterlevel_df, safe_name):
    """
    Bundle discharge and water level data into a single zip archive so the
    user gets both files from one download button. If water level data is
    not available for the station, its CSV inside the zip will be empty.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"NADI_AI_Discharge_{safe_name}.csv", _df_to_csv_bytes(discharge_df))
        zf.writestr(f"NADI_AI_WaterLevel_{safe_name}.csv", _df_to_csv_bytes(waterlevel_df))
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

if os.path.isfile(LOGO_PATH):
    try:
        img_base64 = get_base64_img(LOGO_PATH)
        logo_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="width:90px; height:auto; border-radius:8px; border: 1px solid rgba(255,255,255,0.2);">'
    except Exception:
        logo_html = f"<div style='font-size:3.2rem;'>\U0001F4A7</div>"
else:
    logo_html = f"<div style='font-size:3.2rem;'>\U0001F4A7</div>"

st.markdown(
    f"""
    <div class="nadi-header">
        {logo_html}
        <div class="nadi-header-text">
            <h1 style="color:white; margin:0; padding:0; font-size:2.4rem; line-height:1.2; font-weight:700;">NADI AI</h1>
            <p style="color:rgba(255,255,255,0.9); margin:4px 0 0 0; padding:0; font-size:1.1rem;">
                Your AI Assistant for Hydrological Analysis
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOAD STATION LIST (for gauge_id matching) + MAP DATA (for plotting)
# ---------------------------------------------------------------------------

try:
    name_df = load_stations()
except Exception as e:
    st.error(f"Could not load station list. Please check that the DATA folder and "
             f"camels_ind_name.csv exist relative to app.py.\n\nError: {e}")
    st.stop()

try:
    map_df = load_map_data()
except Exception as e:
    st.error(f"Could not load map data. Please check that camels_ind_name.csv and "
             f"camels_ind_topo.csv exist at the configured paths.\n\nError: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# STATION SELECTION VIA MAP
# ---------------------------------------------------------------------------

header_col, locate_col = st.columns([4, 1])
with header_col:
    st.markdown("### \U0001F5FA\uFE0F Select a gauging station")
    st.write(" Zoom in to view individual stations, then click a station to begin analysis")
with locate_col:
    with st.popover("\U0001F4CD Locate Lat/Long"):
        st.write("Enter coordinates to drop a marker on the map.")
        locate_lat = st.number_input("Latitude", value=22.5, format="%.5f", key="locate_lat")
        locate_lon = st.number_input("Longitude", value=79.0, format="%.5f", key="locate_lon")
        go_locate = st.button("Go", key="locate_go")
        if go_locate:
            st.session_state["located_point"] = (locate_lat, locate_lon)

if "selected_station" not in st.session_state:
    st.session_state.selected_station = None

located_point = st.session_state.get("located_point")
india_center = list(located_point) if located_point else [22.5, 79.0]
m = folium.Map(
    location=india_center,
    zoom_start=9 if located_point else 5,
    tiles="OpenStreetMap",
    control_scale=True,
)

if located_point:
    folium.Marker(
        location=list(located_point),
        tooltip=f"Located point: {located_point[0]:.5f}, {located_point[1]:.5f}",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

marker_cluster = MarkerCluster(
    name="Stations",
    overlay=True,
    control=False,
    disableClusteringAtZoom=10,
    spiderfyOnMaxZoom=True,
    showCoverageOnHover=False,
    zoomToBoundsOnClick=True,
)
marker_cluster.add_to(m)

for _, row in map_df.iterrows():
    is_selected = st.session_state.selected_station == row["cwc_site_name"]
    popup_html = f"""
    <div style="width:320px;font-size:13px">
    <h4 style="margin-bottom:5px;">{row['cwc_site_name']}</h4>
    <b>Gauge ID:</b> {row['gauge_id']}<br><br>
    <table style="width:100%;border-collapse:collapse;">
        <tr><td><b>Elevation Mean</b></td><td>{row['elev_mean']}</td></tr>
        <tr><td><b>Elevation Median</b></td><td>{row['elev_median']}</td></tr>
        <tr><td><b>Elevation Min</b></td><td>{row['elev_min']}</td></tr>
        <tr><td><b>Elevation Max</b></td><td>{row['elev_max']}</td></tr>
        <tr><td><b>Slope Mean</b></td><td>{row['slope_mean']}</td></tr>
        <tr><td><b>Slope Median</b></td><td>{row['slope_median']}</td></tr>
        <tr><td><b>Slope Min</b></td><td>{row['slope_min']}</td></tr>
        <tr><td><b>Slope Max</b></td><td>{row['slope_max']}</td></tr>
        <tr><td><b>CWC Area</b></td><td>{row['cwc_area']}</td></tr>
        <tr><td><b>GHI Area</b></td><td>{row['ghi_area']}</td></tr>
        <tr><td><b>Gauge Elevation</b></td><td>{row['gauge_elevation']}</td></tr>
        <tr><td><b>DPSBAR</b></td><td>{row['dpsbar']}</td></tr>
    </table>
    </div>
    """
    folium.Marker(
        location=[row["cwc_lat"], row["cwc_lon"]],
        tooltip=row["cwc_site_name"],
        popup=folium.Popup(popup_html, max_width=350),
        icon=folium.Icon(
            color="red" if is_selected else "blue",
            icon="tint",
            prefix="fa",
        ),
    ).add_to(marker_cluster)

map_output = st_folium(
    m,
    width=None,
    height=600,
    returned_objects=["last_object_clicked_tooltip"],
    key="station_map",
)

clicked_tooltip = map_output.get("last_object_clicked_tooltip") if map_output else None

if clicked_tooltip and clicked_tooltip != st.session_state.selected_station:
    st.session_state.selected_station = clicked_tooltip
    st.rerun()

selected_station = st.session_state.selected_station

if not selected_station:
    st.markdown("### \U0001F44B Welcome")
    st.write(
        "Click a station marker on the map above to select a gauged station and run "
        "detailed hydrological analysis."
    )
    st.stop()

st.markdown(
    f'<div class="nadi-selected-banner">\U0001F4CD Selected station: {selected_station}</div>',
    unsafe_allow_html=True,
)

# resolve gauge_id from selected station name (internal only, not displayed)
matched_row = name_df.loc[name_df["cwc_site_name"] == selected_station]
if matched_row.empty:
    st.error("Selected station could not be matched to a gauge_id. Please try another station.")
    st.stop()

gauge_id = matched_row.iloc[0]["gauge_id"]

# ---------------------------------------------------------------------------
# LOAD STATION DATA
# ---------------------------------------------------------------------------

try:
    station_data = cached_station_data(gauge_id)
except Exception as e:
    st.error(f"Error loading data for this station: {e}")
    st.stop()

meta = station_data["meta"]

st.markdown("---")
st.markdown(f"## \U0001F30A {meta.get('cwc_site_name', 'N/A')}")

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.metric("River Basin", meta.get("river_basin", "N/A"))
with info_col2:
    st.metric("River / Tributary", meta.get("cwc_river", "N/A"))
with info_col3:
    flow_avail = meta.get("flow_availability", None)
    st.metric(f"Flow Availability ({dc.FLOW_RECORD_PERIOD})", f"{flow_avail:.1f}%" if flow_avail is not None else "N/A")

for w in station_data["warnings"]:
    st.warning(w)

sufficient = station_data["sufficient_data"]
usable_years = station_data["usable_years"]

st.markdown(
    f"**Valid years for analysis (>=50% data availability):** {len(usable_years)} year(s) found."
)

if not sufficient:
    st.error(
        "Sufficient data is not available for this station (minimum 10 valid years "
        "required for full statistical analysis). Only a data overview is shown below."
    )

# ---------------------------------------------------------------------------
# DATA OVERVIEW (home page - annual data availability + water level availability)
# ---------------------------------------------------------------------------

st.markdown("### \U0001F4CA Data Overview")

if not station_data["yearly_avail"].empty:
    st.pyplot(pl.plot_yearly_availability(station_data["yearly_avail"]))
else:
    st.write("No streamflow data available for this station.")

wl_avail = station_data.get("waterlevel_yearly_avail")
if wl_avail is not None and not wl_avail.empty:
    st.pyplot(pl.plot_waterlevel_availability(wl_avail))

# ---------------------------------------------------------------------------
# DOWNLOAD STATION DATA (single button - discharge + water level bundled)
# ---------------------------------------------------------------------------


safe_name = "".join(c if c.isalnum() else "_" for c in str(meta.get("cwc_site_name", "station")))
discharge_export = station_data["daily"][["date", "year", "month", "day", "flow"]].copy() \
    if not station_data["daily"].empty else pd.DataFrame(columns=["date", "year", "month", "day", "flow"])
wl_daily = station_data.get("waterlevel_daily")
waterlevel_export = wl_daily[["date", "year", "month", "day", "level"]].copy() \
    if wl_daily is not None and not wl_daily.empty else pd.DataFrame(columns=["date", "year", "month", "day", "level"])

st.download_button(
    label="\U0001F4E5 Download Station Data",
    data=_station_data_zip_bytes(discharge_export, waterlevel_export, safe_name),
    file_name=f"NADI_AI_StationData_{safe_name}.zip",
    mime="application/zip",
)

# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🤖 Run Hydrological Analysis")
st.write(
    "Run the hydrological analysis and download the technical report."
)

if st.button("\U0001F30A Run", type="primary"):
    with st.spinner("Running analysis and generating report... this may take up to a minute."):
        try:
            os.makedirs("generated_reports", exist_ok=True)
            output_path = os.path.join("generated_reports", f"NADI_AI_Report_{safe_name}.pdf")
            rp.generate_report(station_data, output_path)

            with open(output_path, "rb") as f:
                pdf_bytes = f.read()

            # Build the analysis summary once here (same numbers as the PDF)
            # and cache it in session state, keyed by gauge_id, so the
            # chatbot below can answer questions without recomputing anything.
            analysis_summary = rp.build_analysis_summary(station_data)
            st.session_state["nadi_analysis_summary"] = analysis_summary
            st.session_state["nadi_analysis_gauge_id"] = gauge_id
            # A fresh analysis run should start a fresh chat.
            st.session_state["nadi_chat_history"] = []

            st.success("Analysis complete - report generated successfully!")
            st.download_button(
                label="\U00002B07 Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"NADI_AI_Report_{safe_name}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.exception(e)

# ---------------------------------------------------------------------------
# CHATBOT (available once an analysis has been run for this station)
# ---------------------------------------------------------------------------

st.markdown("---")

current_summary = None
if st.session_state.get("nadi_analysis_gauge_id") == gauge_id:
    current_summary = st.session_state.get("nadi_analysis_summary")

cb.render_chatbot_ui(
    current_summary,
    OPENROUTER_API_KEY,
    station_label=meta.get("cwc_site_name", "this station"),
)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="nadi-footer">
        NADI AI - developed by Narala Venkatesh, M.Tech Water Resources Engineering, NIT Warangal.
    </div>
    """,
    unsafe_allow_html=True,
)
