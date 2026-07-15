"""
quality.py
NADI AI - Data Quality Module
------------------------------
Outlier detection (IQR method + Grubbs-Beck test per USGS Bulletin 17B/17C)
and change-point detection (Pettitt test + CUSUM), all applied to the
ANNUAL MAXIMUM SERIES (AMS) rather than the raw daily series, which is
standard practice in flood-frequency analysis (the daily series is
non-stationary within a year by design - seasonality - so change-point /
outlier tests are only meaningful on the annual extremes).

All functions take a 1-D numpy array / pandas Series of annual maxima and
return plain dictionaries of results so they can be dropped straight into
report tables.
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 1. IQR OUTLIER TEST
# ---------------------------------------------------------------------------

def iqr_outlier_test(values, k=1.5):
    """
    Classic Tukey IQR fence test.
    Values below Q1 - k*IQR or above Q3 + k*IQR are flagged as outliers.
    Returns dict: q1, q3, iqr, lower_fence, upper_fence, low_outliers, high_outliers
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return dict(q1=np.nan, q3=np.nan, iqr=np.nan, lower_fence=np.nan,
                    upper_fence=np.nan, low_outliers=np.array([]), high_outliers=np.array([]),
                    n=len(values))

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    low_outliers = values[values < lower_fence]
    high_outliers = values[values > upper_fence]

    return dict(
        q1=q1, q3=q3, iqr=iqr, lower_fence=lower_fence, upper_fence=upper_fence,
        low_outliers=low_outliers, high_outliers=high_outliers, n=len(values)
    )


# ---------------------------------------------------------------------------
# 2. GRUBBS-BECK TEST (USGS Bulletin 17B / 17C, low-outlier oriented)
# ---------------------------------------------------------------------------
# One-sided Grubbs-Beck critical K values (Bulletin 17B, Table 3-approx) for
# alpha = 0.10 (single-sided), derived from the standard Grubbs approximation.

def _grubbs_critical_value(n, alpha=0.10):
    """
    Generalized Extreme Studentized Deviate critical value approximation
    (two-sided Grubbs test formula, applied one-sided here as per Bulletin 17B
    convention for low/high outlier detection on log-transformed flows).
    """
    if n < 3:
        return np.nan
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(t_dist ** 2)
    denominator = np.sqrt(n) * np.sqrt(n - 2 + t_dist ** 2)
    return numerator / denominator


def grubbs_beck_test(values, alpha=0.10):
    """
    USGS Bulletin 17B/17C style Grubbs-Beck test performed on the natural
    log of the annual maxima (standard practice for flood series).
    Flags both low and high outliers using the one-sided critical K value
    applied to the standardized log-values.

    Returns dict with: log_mean, log_std, K_critical, K_high, K_low,
    high_outliers (original units), low_outliers (original units), n
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]  # log requires positive values
    n = len(values)
    if n < 3:
        return dict(log_mean=np.nan, log_std=np.nan, K_critical=np.nan,
                    high_outliers=np.array([]), low_outliers=np.array([]), n=n)

    log_vals = np.log10(values)
    log_mean = np.mean(log_vals)
    log_std = np.std(log_vals, ddof=1)
    K_critical = _grubbs_critical_value(n, alpha=alpha)

    z = (log_vals - log_mean) / log_std if log_std > 0 else np.zeros_like(log_vals)
    high_mask = z > K_critical
    low_mask = z < -K_critical

    high_outliers = values[high_mask]
    low_outliers = values[low_mask]

    return dict(
        log_mean=log_mean, log_std=log_std, K_critical=K_critical,
        high_outliers=high_outliers, low_outliers=low_outliers, n=n
    )


# ---------------------------------------------------------------------------
# 3. PETTITT CHANGE-POINT TEST
# ---------------------------------------------------------------------------

def pettitt_test(values, years=None):
    """
    Non-parametric Pettitt test for detecting a single change point in the
    mean of a time series.

    Returns dict: U_stat (max |U_t|), K_stat, p_value (approx), change_index,
    change_year (if years provided), significant (bool at alpha=0.05)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if years is not None:
        years = np.asarray(years)

    if n < 4:
        return dict(K_stat=np.nan, p_value=np.nan, change_index=None,
                    change_year=None, significant=False, n=n, U_series=np.array([]))

    # U_t statistic via sign matrix
    U = np.zeros(n)
    for t in range(n):
        s = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                s += np.sign(values[i] - values[j])
        U[t] = s

    K_stat = np.max(np.abs(U))
    change_index = int(np.argmax(np.abs(U)))

    # approximate p-value (Pettitt 1979)
    p_value = 2 * np.exp((-6 * K_stat ** 2) / (n ** 3 + n ** 2))
    p_value = min(p_value, 1.0)

    change_year = int(years[change_index]) if years is not None else None
    significant = p_value < 0.05

    return dict(
        K_stat=K_stat, p_value=p_value, change_index=change_index,
        change_year=change_year, significant=significant, n=n, U_series=U
    )


# ---------------------------------------------------------------------------
# 4. CUSUM CHANGE-POINT ANALYSIS
# ---------------------------------------------------------------------------

def cusum_test(values, years=None):
    """
    Simple CUSUM of departures from the mean. The point of maximum absolute
    cumulative deviation is flagged as the likely change point (visual /
    exploratory test - no formal significance test attached, consistent
    with common hydrological practice of using CUSUM as a qualitative
    cross-check alongside Pettitt).

    Returns dict: cusum (array), change_index, change_year, max_abs_cusum
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return dict(cusum=np.array([]), change_index=None, change_year=None, max_abs_cusum=np.nan)

    mean_val = np.mean(values)
    departures = values - mean_val
    cusum = np.cumsum(departures)
    change_index = int(np.argmax(np.abs(cusum)))
    max_abs_cusum = float(np.max(np.abs(cusum)))
    change_year = int(years[change_index]) if years is not None else None

    return dict(
        cusum=cusum, change_index=change_index, change_year=change_year,
        max_abs_cusum=max_abs_cusum
    )


# ---------------------------------------------------------------------------
# Reference text blocks (definitions/purpose/formulae) reused verbatim in report.py
# ---------------------------------------------------------------------------
IQR_DESCRIPTION = {
    "title": "Inter-Quartile Range (IQR) Outlier Test",
    "definition": (
        "The Inter-Quartile Range (IQR) Outlier Test is a non-parametric "
        "(distribution-free) method used to identify unusually high or low "
        "observations in the annual maximum discharge series. It relies on "
        "the first quartile (Q1) and third quartile (Q3) to define the "
        "spread of the central 50% of the data, making it a simple and "
        "robust technique that does not require any assumption about the "
        "underlying probability distribution."
    ),
    "purpose": (
        "The test computes the Inter-Quartile Range (IQR = Q3 - Q1) and "
        "establishes the lower and upper fences as Q1 - 1.5 x IQR and "
        "Q3 + 1.5 x IQR, respectively. Observations falling outside these "
        "limits are flagged as potential outliers. The analysis uses the "
        "standard Tukey multiplier (k = 1.5), which provides a widely "
        "accepted balance between identifying unusual observations and "
        "retaining genuine extreme flood events. Although the method is "
        "robust and assumption-free, its performance can be influenced by "
        "small sample sizes, where quartile estimates may be less stable."
    ),
    "formula": "Lower fence = Q1 - k x IQR ;  Upper fence = Q3 + k x IQR ;  IQR = Q3 - Q1",
    "standard_values": "k = 1.5 (standard Tukey fence, used here)",
    "assumptions": "No distributional assumption; sensitive to sample size for small n.",
}

GRUBBS_BECK_DESCRIPTION = {
    "title": "Grubbs-Beck Outlier Test (USGS Bulletin 17B/17C)",
    "definition": (
        "The Grubbs-Beck Test is a statistical outlier detection method "
        "recommended in USGS Bulletin 17B/17C for flood frequency analysis. "
        "Unlike the IQR method, it evaluates whether an annual maximum flood "
        "value is statistically inconsistent with the remaining observations. "
        "The test is performed on the log10-transformed annual maximum "
        "discharge series, as flood peaks are commonly assumed to follow an "
        "approximately log-normal distribution. It is primarily used to "
        "identify unusually high or low flood events that may require "
        "special consideration before fitting probability distributions."
    ),
    "purpose": (
        "For each observation, a standardized test statistic (z-score) is "
        "calculated using the mean and standard deviation of the "
        "log10-transformed data. An observation is flagged as a potential "
        "outlier when its absolute test statistic exceeds the critical value "
        "(Kn) corresponding to the sample size. The analysis follows the "
        "USGS-recommended one-sided significance level of alpha = 0.10 (10%). "
        "Since the method assumes an approximately log-normal distribution "
        "of annual maximum floods, its reliability depends on how well this "
        "assumption represents the observed data."
    ),
    "standard_values": (
        "One-sided significance level alpha = 0.10 (10%), as recommended in Bulletin 17B"
    ),
}

PETTITT_DESCRIPTION = {
    "title": "Pettitt Change-Point Test",
    "definition": (
        "The Pettitt Change-Point Test is a non-parametric statistical test "
        "used to identify whether a single significant change has occurred "
        "in a time series. In hydrological analysis, it is applied to the "
        "annual maximum discharge series to determine whether the flood "
        "record remains consistent over time. A detected change point may "
        "be associated with land-use changes, construction of dams or "
        "reservoirs, station relocation, or changes in measurement practices."
    ),
    "purpose": (
        "The test compares observations before and after each possible year "
        "in the record using their relative ranks. The overall test "
        "statistic K represents the largest difference between the two "
        "parts of the series, and the year corresponding to this maximum "
        "value is identified as the most likely change point. If the "
        "p-value is less than 0.05, the change is considered statistically "
        "significant, indicating that the characteristics of the annual "
        "maximum series have changed during the period of record."
    ),
    "standard_values": "Significance level alpha = 0.05",
}

CUSUM_DESCRIPTION = {
    "title": "CUSUM (Cumulative Sum) Change-Point Analysis",
    "definition": (
        "The Cumulative Sum (CUSUM) Test is a graphical method used to "
        "detect changes in a time series by plotting the cumulative "
        "difference between each observation and the overall mean. A clear "
        "change in the slope of the curve suggests that the average "
        "behavior of the series may have changed."
    ),
    "purpose": (
        "Used as a qualitative cross-check alongside the Pettitt test to "
        "corroborate (or question) a detected change point."
    ),
}
