"""
data_collec.py
NADI AI - Data Collection Module
---------------------------------
Reads the CAMELS-IND source files (streamflow + water level), extracts
everything relevant for a single gauge_id, computes data-availability
statistics (annual + monsoon), selects "usable" years (>=50% daily data
available), builds the annual maximum series (AMS), computes basic
descriptive statistics, and (for a defined subset of stations) prepares
stage-discharge data for rating-curve analysis.

All functions are pure (no Streamlit calls) so they can be unit tested and
reused by report.py without any circular imports.
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# CONFIG - EDIT THIS PATH IF YOUR DATA FOLDER IS ELSEWHERE
# ---------------------------------------------------------------------------
# By default this points to your DATA folder at C:\Documents\NADIAI\DATA.
# If you move the project or the data, update DATA_DIR below, e.g.:
#   DATA_DIR = r"C:\Documents\NADIAI\DATA"

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"

NAME_CSV = DATA_DIR / "camels_ind_name.csv"
LAND_CSV = DATA_DIR / "camels_ind_land.csv"
TOPO_CSV = DATA_DIR / "camels_ind_topo.csv"
FLOW_CSV = DATA_DIR / "streamflow_observed.csv"
WATERLEVEL_CSV = DATA_DIR / "waterleveldata.csv"

MIN_YEAR_AVAILABILITY_PCT = 50.0   # a year is "usable" if >=50% of days have data
MIN_YEARS_REQUIRED = 10            # minimum usable years needed to run full analysis
MONSOON_MONTHS = [6, 7, 8, 9]      # JJAS (Indian monsoon convention)

# Streamflow record period (updated from 1980-2020 to 1961-2020 per new CSV)
FLOW_RECORD_PERIOD = "1961-2020"

# ---------------------------------------------------------------------------
# RATING CURVE STATION LIST
# ---------------------------------------------------------------------------
# Only these gauge_ids have reliable, checked stage-discharge relationships
# suitable for rating-curve fitting. For all other stations, the rating
# curve section of the report is left empty (definition only, no fitting).
RATING_CURVE_STATIONS = {
    "3005", "3010", "3013", "3018", "3019", "3020", "3024", "3028", "3029",
    "3030", "3043", "3045", "3048", "3055", "3056", "3057", "3064", "3069",
    "3072", "3073", "3074", "3075", "3081", "3082", "3083", "3085", "3089",
    "3092", "3094", "3099", "3103", "3107", "3109", "3110",
    "4004", "4005", "4007", "4010", "4012", "4017", "4018", "4019", "4021",
    "4022", "4024", "4026", "4028", "4031", "4034", "4038", "4040", "4041",
    "4049", "4051", "4054", "4055", "4061", "4064",
    "5001", "5002", "5004", "5005", "5007", "5008", "5010", "5012", "5013",
    "5014", "5015", "5018", "5022", "5023", "5027", "5029",
    "6001", "6006", "6008",
    "7002", "7005", "7007", "7008", "7009", "7014", "7015",
    "8001", "8003", "8005", "8006", "8009", "8011", "8013", "8016", "8020",
    "8021", "8027", "8029", "8030", "8031", "8034", "8035", "8036",
    "9002", "9006", "9008", "9009", "9011",
    "10008", "10011", "10014",
    "11006", "11007", "11010",
    "12004", "12012", "12015", "12016", "12018", "12020", "12025", "12026",
    "12029", "12035", "12036", "12039", "12043", "12044", "12046",
    "13004", "13006", "13007", "13008", "13010", "13013", "13016", "13017",
    "13025",
    "14002", "14003", "14004", "14008", "14015",
    "15006", "15007", "15008", "15009", "15011", "15013", "15017", "15018",
    "15022", "15023", "15026", "15030", "15031", "15032",
    "16007", "16012", "16014",
    "17009", "17011", "17012", "17019",
}

def station_has_rating_curve(gauge_id):
    """Return True if this gauge_id is in the checked rating-curve station list."""
    return str(gauge_id) in RATING_CURVE_STATIONS

# ---------------------------------------------------------------------------
# LOW LEVEL LOADERS (cached at app.py level via st.cache_data)
# ---------------------------------------------------------------------------

def load_station_list():
    """Return the name/metadata dataframe used to populate the station dropdown."""
    df = pd.read_csv(NAME_CSV)
    df["gauge_id"] = df["gauge_id"].astype(str)
    return df

def load_land_data():
    df = pd.read_csv(LAND_CSV)
    df["gauge_id"] = df["gauge_id"].astype(str)
    return df

def load_topo_data():
    df = pd.read_csv(TOPO_CSV)
    df["gauge_id"] = df["gauge_id"].astype(str)
    return df

def load_flow_data():
    """
    Streamflow file has columns: year, month, day, <gauge_id_1>, <gauge_id_2>, ...
    Column headers for gauge ids may be read as int or str depending on pandas
    version, so we normalize all non year/month/day columns to plain strings.
    Data now spans 1961-2020 (updated source CSV).
    """
    df = pd.read_csv(FLOW_CSV)
    rename_map = {c: str(c) for c in df.columns if c not in ("year", "month", "day")}
    df = df.rename(columns=rename_map)
    return df

def load_waterlevel_data():
    """
    Water level file has the same year/month/day + gauge_id column format as
    the streamflow file, but values are water level (stage), used only for
    rating-curve construction and as a data-availability overview plot.
    Returns an empty-columns DataFrame (year, month, day only) if the file
    is missing, so callers can proceed gracefully with empty results.
    """
    try:
        df = pd.read_csv(WATERLEVEL_CSV)
    except Exception:
        return pd.DataFrame(columns=["year", "month", "day"])
    rename_map = {c: str(c) for c in df.columns if c not in ("year", "month", "day")}
    df = df.rename(columns=rename_map)
    return df

# ---------------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------------------------

def get_station_data(gauge_id):
    """
    Master function - given a gauge_id (str or int), pulls together
    everything needed for the report.

    Returns a dictionary with keys:
        gauge_id, meta, land, topo,
        daily (DataFrame: date, flow),
        yearly_avail (DataFrame per-year % availability, monsoon % availability),
        usable_years (list of ints),
        ams (DataFrame: year, ann_max) -- only for usable years
        monthly_mean_flow (Series indexed by month 1-12)
        annual_mean_flow (DataFrame: year, mean_flow) -- usable years only
        monsoon_mean_flow (DataFrame: year, mean_flow) -- usable years only, JJAS
        basic_stats (dict: mean, max, min, std, cv, skew of AMS + of daily flow)
        sufficient_data (bool) -- True if usable_years >= MIN_YEARS_REQUIRED
        fdc (DataFrame: exceedance_prob, flow) built from usable years' daily data
        warnings (list of str) - messages to surface in UI/report
        waterlevel_daily (DataFrame: date, year, month, day, level) - may be empty
        waterlevel_yearly_avail (DataFrame per-year % water-level availability)
        has_rating_curve (bool) - whether this station is in the checked list
        stage_discharge (DataFrame: date, stage, discharge) - overlapping days,
            only populated when has_rating_curve is True
    """
    gauge_id = str(gauge_id)
    warnings = []

    # ---- metadata ----
    name_df = load_station_list()
    land_df = load_land_data()
    topo_df = load_topo_data()

    meta_row = name_df.loc[name_df["gauge_id"] == gauge_id]
    land_row = land_df.loc[land_df["gauge_id"] == gauge_id]
    topo_row = topo_df.loc[topo_df["gauge_id"] == gauge_id]

    if meta_row.empty:
        raise ValueError(f"gauge_id {gauge_id} not found in {NAME_CSV}")

    meta = meta_row.iloc[0].to_dict()
    land = land_row.iloc[0].to_dict() if not land_row.empty else {}
    topo = topo_row.iloc[0].to_dict() if not topo_row.empty else {}

    if land_row.empty:
        warnings.append("Land-cover attributes not found for this station.")
    if topo_row.empty:
        warnings.append("Topographic attributes not found for this station.")

    # ---- streamflow ----
    flow_df = load_flow_data()
    if gauge_id not in flow_df.columns:
        raise ValueError(f"gauge_id {gauge_id} has no streamflow column in {FLOW_CSV}")

    daily = flow_df[["year", "month", "day", gauge_id]].copy()
    daily = daily.rename(columns={gauge_id: "flow"})

    # build a proper date column (invalid dates like Feb 30 are coerced to NaT and dropped)
    daily["date"] = pd.to_datetime(
        dict(year=daily["year"], month=daily["month"], day=daily["day"]),
        errors="coerce"
    )
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    daily = daily[["date", "year", "month", "day", "flow"]]

    # ---- yearly & monsoon availability ----
    yearly_avail = _compute_yearly_availability(daily, value_col="flow")

    usable_years = yearly_avail.loc[
        yearly_avail["pct_available"] >= MIN_YEAR_AVAILABILITY_PCT, "year"
    ].tolist()
    usable_years = sorted(usable_years)
    sufficient_data = len(usable_years) >= MIN_YEARS_REQUIRED

    if not sufficient_data:
        warnings.append(
            f"Only {len(usable_years)} year(s) meet the >= {MIN_YEAR_AVAILABILITY_PCT:.0f}% "
            f"data-availability threshold (minimum {MIN_YEARS_REQUIRED} required for "
            f"frequency analysis). Only an overview of available data is presented; "
            f"quality checks, trend tests and distribution fitting are skipped."
        )

    daily_usable = daily[daily["year"].isin(usable_years)].copy()

    # ---- annual maximum series (AMS) ----
    if not daily_usable.empty:
        ams = (
            daily_usable.dropna(subset=["flow"])
            .groupby("year")["flow"]
            .max()
            .reset_index()
            .rename(columns={"flow": "ann_max"})
        )
    else:
        ams = pd.DataFrame(columns=["year", "ann_max"])

    # ---- monthly mean flow (climatology across usable years) ----
    if not daily_usable.empty:
        monthly_mean_flow = daily_usable.groupby("month")["flow"].mean()
        monthly_mean_flow = monthly_mean_flow.reindex(range(1, 13))
    else:
        monthly_mean_flow = pd.Series([np.nan] * 12, index=range(1, 13))

    # ---- annual mean flow (usable years only) ----
    if not daily_usable.empty:
        annual_mean_flow = (
            daily_usable.groupby("year")["flow"].mean().reset_index()
            .rename(columns={"flow": "mean_flow"})
        )
    else:
        annual_mean_flow = pd.DataFrame(columns=["year", "mean_flow"])

    # ---- monsoon (JJAS) mean flow (usable years only) ----
    if not daily_usable.empty:
        monsoon_usable = daily_usable[daily_usable["month"].isin(MONSOON_MONTHS)]
        if not monsoon_usable.empty:
            monsoon_mean_flow = (
                monsoon_usable.groupby("year")["flow"].mean().reset_index()
                .rename(columns={"flow": "mean_flow"})
            )
        else:
            monsoon_mean_flow = pd.DataFrame(columns=["year", "mean_flow"])
    else:
        monsoon_mean_flow = pd.DataFrame(columns=["year", "mean_flow"])

    # ---- flow duration curve (FDC) from usable-year daily data ----
    fdc = _compute_fdc(daily_usable["flow"].dropna().values)

    # ---- basic stats ----
    basic_stats = _basic_stats(daily_usable["flow"].dropna().values, ams["ann_max"].values)

    # ---- water level (stage) data ----
    wl_df = load_waterlevel_data()
    if gauge_id in wl_df.columns:
        wl_daily = wl_df[["year", "month", "day", gauge_id]].copy()
        wl_daily = wl_daily.rename(columns={gauge_id: "level"})
        wl_daily["date"] = pd.to_datetime(
            dict(year=wl_daily["year"], month=wl_daily["month"], day=wl_daily["day"]),
            errors="coerce"
        )
        wl_daily = wl_daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        wl_daily = wl_daily[["date", "year", "month", "day", "level"]]
        waterlevel_yearly_avail = _compute_yearly_availability(wl_daily, value_col="level")
    else:
        wl_daily = pd.DataFrame(columns=["date", "year", "month", "day", "level"])
        waterlevel_yearly_avail = pd.DataFrame(columns=["year", "pct_available", "pct_available_monsoon"])

    # ---- rating curve eligibility + stage-discharge pairing ----
    has_rating_curve = station_has_rating_curve(gauge_id)
    stage_discharge = pd.DataFrame(columns=["date", "stage", "discharge"])
    if has_rating_curve and not wl_daily.empty:
        merged = pd.merge(
            wl_daily[["date", "level"]],
            daily[["date", "flow"]],
            on="date", how="inner"
        )
        merged = merged.dropna(subset=["level", "flow"])
        merged = merged.rename(columns={"level": "stage", "flow": "discharge"})
        stage_discharge = merged[["date", "stage", "discharge"]].reset_index(drop=True)

    return {
        "gauge_id": gauge_id,
        "meta": meta,
        "land": land,
        "topo": topo,
        "daily": daily,
        "daily_usable": daily_usable,
        "yearly_avail": yearly_avail,
        "usable_years": usable_years,
        "ams": ams,
        "monthly_mean_flow": monthly_mean_flow,
        "annual_mean_flow": annual_mean_flow,
        "monsoon_mean_flow": monsoon_mean_flow,
        "basic_stats": basic_stats,
        "sufficient_data": sufficient_data,
        "fdc": fdc,
        "warnings": warnings,
        "waterlevel_daily": wl_daily,
        "waterlevel_yearly_avail": waterlevel_yearly_avail,
        "has_rating_curve": has_rating_curve,
        "stage_discharge": stage_discharge,
    }

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _compute_yearly_availability(daily, value_col="flow"):
    """
    For each calendar year present in the record, compute:
      - pct_available: % of days in that year with non-null value_col
      - pct_available_monsoon: % of monsoon days (Jun-Sep) with non-null value_col
    Uses actual calendar length (365/366) as the denominator, not just the
    number of rows present in the file, so genuinely missing rows also count
    against availability.
    """
    if daily.empty:
        return pd.DataFrame(columns=["year", "pct_available", "pct_available_monsoon"])

    years = sorted(daily["year"].unique())
    records = []
    for yr in years:
        yr = int(yr)
        days_in_year = 366 if pd.Timestamp(yr, 12, 31).is_leap_year else 365
        yr_data = daily[daily["year"] == yr]
        n_present = yr_data[value_col].notna().sum()
        pct_available = 100.0 * n_present / days_in_year

        mons_data = yr_data[yr_data["month"].isin(MONSOON_MONTHS)]
        monsoon_days = sum(
            pd.Timestamp(yr, m, 1).days_in_month for m in MONSOON_MONTHS
        )
        n_mons_present = mons_data[value_col].notna().sum()
        pct_available_monsoon = 100.0 * n_mons_present / monsoon_days if monsoon_days else np.nan

        records.append({
            "year": yr,
            "pct_available": round(pct_available, 1),
            "pct_available_monsoon": round(pct_available_monsoon, 1),
        })
    return pd.DataFrame(records)

def _compute_fdc(flow_values):
    """Weibull plotting-position flow duration curve. Returns DataFrame sorted by exceedance %."""
    flow_values = np.asarray(flow_values, dtype=float)
    flow_values = flow_values[~np.isnan(flow_values)]
    if len(flow_values) == 0:
        return pd.DataFrame(columns=["exceedance_prob", "flow"])
    sorted_flow = np.sort(flow_values)[::-1]
    n = len(sorted_flow)
    rank = np.arange(1, n + 1)
    exceedance_prob = 100.0 * rank / (n + 1)  # Weibull plotting position
    return pd.DataFrame({"exceedance_prob": exceedance_prob, "flow": sorted_flow})

FDC_TABLE_PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98]

def fdc_percentile_table(fdc, percentiles=None):
    """
    Given an FDC DataFrame (exceedance_prob, flow), interpolate discharge at
    a standard set of exceedance percentiles (5,10,20,...,95,98% by default)
    for a compact reference table alongside the FDC plots.

    Returns DataFrame with columns: exceedance_prob (%), discharge (m3/s).
    """
    if percentiles is None:
        percentiles = FDC_TABLE_PERCENTILES
    if fdc is None or fdc.empty:
        return pd.DataFrame(columns=["exceedance_prob", "discharge"])

    # fdc is sorted descending in flow / ascending in exceedance_prob
    fdc_sorted = fdc.sort_values("exceedance_prob")
    xp = fdc_sorted["exceedance_prob"].values
    fp = fdc_sorted["flow"].values

    rows = []
    for p in percentiles:
        if p < xp.min() or p > xp.max():
            q = np.nan
        else:
            q = float(np.interp(p, xp, fp))
        rows.append({"exceedance_prob": p, "discharge": q})
    return pd.DataFrame(rows)

def _basic_stats(daily_flow, ams_values):
    """Descriptive statistics for both the daily series and the AMS."""
    def _stats_block(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return dict(n=0, mean=np.nan, max=np.nan, min=np.nan, std=np.nan, cv=np.nan, skew=np.nan)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
        skew = float(pd.Series(arr).skew()) if len(arr) > 2 else np.nan
        return dict(
            n=int(len(arr)),
            mean=mean,
            max=float(np.max(arr)),
            min=float(np.min(arr)),
            std=std,
            cv=(std / mean) if (mean not in (0, np.nan) and not np.isnan(std)) else np.nan,
            skew=skew,
        )

    return {
        "daily": _stats_block(daily_flow),
        "ams": _stats_block(ams_values),
    }

# ---------------------------------------------------------------------------
# Human-readable attribute descriptions used directly in report.py (Station Info section)
# ---------------------------------------------------------------------------
ATTRIBUTE_DESCRIPTIONS = {
    "gauge_id": ("Gauge station identifier (5-digit; first 2 digits are CWC basin code, last 3 digits are station number)", "-", "CWC"),
    "ghi_stn_id": ("Unique station identifier, 10 characters long", "-", "GHI (Goteti, 2023)"),
    "water_frac": ("Water cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "trees_frac": ("Trees cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "flooded_veg_frac": ("Flooded vegetation cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "crops_frac": ("Crop cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "built_area_frac": ("Urban / built-up cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "bare_frac": ("Bare cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "range_frac": ("Range cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "dom_land_cover": ("Dominant land cover type (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "dom_land_cover_frac": ("Dominant land cover fraction (2017-2022)", "-", "ESRI Land Cover (Karra et al., 2021)"),
    "lai_mean": ("Catchment mean leaf area index (2001-2020)", "-", "MODIS MCD15A2H (Myneni et al., 2015)"),
    "lai_min": ("Minimum leaf area index (2001-2020)", "-", "MODIS MCD15A2H (Myneni et al., 2015)"),
    "lai_max": ("Maximum leaf area index (2001-2020)", "-", "MODIS MCD15A2H (Myneni et al., 2015)"),
    "lai_diff": ("Difference between max and min LAI (2001-2020)", "-", "MODIS MCD15A2H (Myneni et al., 2015)"),
    "cwc_lat": ("Latitude of the station", "deg", "CWC"),
    "cwc_lon": ("Longitude of the station", "deg", "CWC"),
    "ghi_lat": ("Latitude of the GHI relocated station", "deg", "GHI (Goteti, 2023)"),
    "ghi_lon": ("Longitude of the GHI relocated station", "deg", "GHI (Goteti, 2023)"),
    "elev_mean": ("Catchment mean elevation", "m", "SRTM DEM 90m"),
    "elev_median": ("Catchment median elevation", "m", "SRTM DEM 90m"),
    "elev_min": ("Catchment minimum elevation", "m", "SRTM DEM 90m"),
    "elev_max": ("Catchment maximum elevation", "m", "SRTM DEM 90m"),
    "slope_mean": ("Catchment mean slope", "%", "SRTM DEM 90m"),
    "slope_median": ("Catchment median slope", "%", "SRTM DEM 90m"),
    "slope_min": ("Catchment minimum slope", "%", "SRTM DEM 90m"),
    "slope_max": ("Catchment maximum slope", "%", "SRTM DEM 90m"),
    "cwc_area": ("Catchment drainage area (CWC)", "km2", "CWC"),
    "ghi_area": ("Catchment drainage area (GHI)", "km2", "GHI (Goteti, 2023)"),
    "gauge_elevation": ("Elevation of the gauging station", "m", "SRTM DEM 90m"),
    "dpsbar": ("Mean basin slope (drainage-path based)", "%", "GHI (Goteti, 2023)"),
    "cwc_site_name": ("Name of the station", "-", "CWC"),
    "river_basin": ("Name of the river basin", "-", "CWC"),
    "cwc_river": ("River / tributary", "-", "CWC"),
    "ghi_group": ("GHI assigned group (G1 or G2)", "-", "GHI (Goteti, 2023)"),
    "flow_availability": (f"Percentage duration for which streamflow data is available ({FLOW_RECORD_PERIOD})", "%", "CWC"),
}
