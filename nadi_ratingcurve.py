"""
ratingcurve.py
NADI AI - Rating Curve Module
-------------------------------
Fits stage-discharge (H-Q) rating curves for the subset of stations with
checked, reliable water-level data (see nadi_data_collec.RATING_CURVE_STATIONS).

Candidate forms:
  - Power-law:     Q = a * (H - h0)^b   (fitted in log-space as Q = a * H^b)
  - Polynomial:     Q = c0 + c1*H + c2*H^2  (2nd order)
  - Exponential:    Q = a * exp(b * H)

The best-fit equation is selected using the highest R^2, and performance
metrics (R^2, NSE, RMSE, MAE) are reported for the top 3 fitted curves.

All functions are pure (no Streamlit calls) so they can be reused directly
by report.py.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# CANDIDATE MODEL FUNCTIONS
# ---------------------------------------------------------------------------

def _power_func(H, a, b):
    return a * np.power(H, b)


def _poly_func(H, c0, c1, c2):
    return c0 + c1 * H + c2 * H ** 2


def _exp_func(H, a, b):
    return a * np.exp(b * H)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def _metrics(obs, pred):
    """R^2, NSE, RMSE, MAE between observed and predicted discharge."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 2:
        return dict(r2=np.nan, nse=np.nan, rmse=np.nan, mae=np.nan)

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    nse = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan  # NSE has same form as R^2 here
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    return dict(r2=float(r2), nse=float(nse), rmse=float(rmse), mae=float(mae))


# ---------------------------------------------------------------------------
# MASTER FITTING FUNCTION
# ---------------------------------------------------------------------------

def fit_rating_curves(stage_discharge):
    """
    stage_discharge: DataFrame with columns 'stage', 'discharge' (overlapping
    days of water level (MSL-converted) and streamflow).

    Returns a DataFrame with one row per candidate equation, columns:
      label, form, params, r2, nse, rmse, mae
    sorted descending by r2 (best fit first). Empty DataFrame if fitting
    is not possible (insufficient data).
    """
    if stage_discharge is None or stage_discharge.empty or len(stage_discharge) < 5:
        return pd.DataFrame(columns=["label", "form", "params", "r2", "nse", "rmse", "mae"])

    df = stage_discharge.dropna(subset=["stage", "discharge"]).copy()
    df = df[(df["discharge"] > 0)]
    if len(df) < 5:
        return pd.DataFrame(columns=["label", "form", "params", "r2", "nse", "rmse", "mae"])

    H = df["stage"].values.astype(float)
    Q = df["discharge"].values.astype(float)

    # shift stage to be strictly positive for power-law / log fitting stability
    h_shift = 0.0
    if H.min() <= 0:
        h_shift = abs(H.min()) + 1.0
    H_fit = H + h_shift

    rows = []

    # ---- Power law: Q = a * H^b ----
    try:
        popt, _ = curve_fit(_power_func, H_fit, Q, p0=[1.0, 1.5], maxfev=10000)
        pred = _power_func(H_fit, *popt)
        m = _metrics(Q, pred)
        rows.append({
            "label": "Power-law",
            "form": "Q = a * (H + shift)^b",
            "params": {"a": popt[0], "b": popt[1], "h_shift": h_shift},
            **m,
        })
    except Exception:
        pass

    # ---- Polynomial (2nd order): Q = c0 + c1*H + c2*H^2 ----
    try:
        coeffs = np.polyfit(H, Q, 2)
        c2, c1, c0 = coeffs
        pred = _poly_func(H, c0, c1, c2)
        m = _metrics(Q, pred)
        rows.append({
            "label": "Polynomial (2nd order)",
            "form": "Q = c0 + c1*H + c2*H^2",
            "params": {"c0": c0, "c1": c1, "c2": c2},
            **m,
        })
    except Exception:
        pass

    # ---- Exponential: Q = a * exp(b*H) ----
    try:
        popt, _ = curve_fit(_exp_func, H, Q, p0=[1.0, 0.1], maxfev=10000)
        pred = _exp_func(H, *popt)
        m = _metrics(Q, pred)
        rows.append({
            "label": "Exponential",
            "form": "Q = a * exp(b*H)",
            "params": {"a": popt[0], "b": popt[1]},
            **m,
        })
    except Exception:
        pass

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    results_df = results_df.sort_values("r2", ascending=False).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    return results_df


def predict_curve(row, H_range):
    """
    Given a fitted-curve row (from fit_rating_curves) and an array of stage
    values H_range, return predicted discharge values for plotting.
    """
    H_range = np.asarray(H_range, dtype=float)
    params = row["params"]
    if row["label"] == "Power-law":
        h_shift = params.get("h_shift", 0.0)
        return _power_func(H_range + h_shift, params["a"], params["b"])
    elif row["label"] == "Polynomial (2nd order)":
        return _poly_func(H_range, params["c0"], params["c1"], params["c2"])
    elif row["label"] == "Exponential":
        return _exp_func(H_range, params["a"], params["b"])
    else:
        return np.full_like(H_range, np.nan)


# ---------------------------------------------------------------------------
# Reference text block reused in report.py
# ---------------------------------------------------------------------------
RATING_CURVE_DEFINITION = (
    "A rating curve is a graph that relates river stage (water level) to "
    "streamflow (discharge), allowing hydrologists to estimate discharge "
    "from stage measurements."
)

RATING_CURVE_METHOD_NOTE = (
    "All days with both water level and discharge observations available "
    "were used to construct the stage-discharge dataset. Water level "
    "observations were converted to Mean Sea Level (MSL) prior to analysis. "
    "Stage (H) was plotted against discharge (Q), and the relationship was "
    "screened for outliers and possible channel changes. Three candidate "
    "equation forms - power-law, polynomial, and exponential - were fitted "
    "to the stage-discharge data, and the best-fit form was selected based "
    "on the highest coefficient of determination (R^2). The fit was further "
    "evaluated using R^2, Nash-Sutcliffe Efficiency (NSE), Root Mean Square "
    "Error (RMSE), and Mean Absolute Error (MAE)."
)
