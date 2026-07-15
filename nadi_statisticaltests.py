"""
statisticaltests.py
NADI AI - Trend Analysis Module
---------------------------------
Mann-Kendall trend test + Sen's slope estimator, applied to the annual
maximum series (AMS), the annual mean flow series, and the monsoon
(Jun-Sep) mean flow series -- same test, same plotting, applied to three
different series.

Uses the `pymannkendall` package if available (installed via
requirements.txt); falls back to a self-contained implementation otherwise
so the app never crashes purely due to a missing optional dependency.
"""

import numpy as np
from scipy import stats

try:
    import pymannkendall as mk
    _HAS_PYMK = True
except ImportError:
    _HAS_PYMK = False


def _manual_mann_kendall(x, alpha=0.05):
    """Self-contained Mann-Kendall + Sen's slope (used if pymannkendall is unavailable)."""
    x = np.asarray(x, dtype=float)
    n = len(x)

    # S statistic
    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))

    # variance (no tie correction for simplicity - adequate for annual series)
    unique_vals, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    h = p < alpha

    if z > 0 and h:
        trend = "increasing"
    elif z < 0 and h:
        trend = "decreasing"
    else:
        trend = "no trend"

    tau = s / (0.5 * n * (n - 1))

    # Sen's slope: median of all pairwise slopes
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    slope = np.median(slopes)
    intercept = np.median(x) - slope * np.median(np.arange(n))

    return dict(trend=trend, h=bool(h), p=float(p), z=float(z), Tau=float(tau),
                s=int(s), var_s=float(var_s), slope=float(slope), intercept=float(intercept))


def mann_kendall_test(values, alpha=0.05):
    """
    Run the Mann-Kendall trend test and Sen's slope estimator on any annual
    series (AMS, annual mean flow, or monsoon mean flow).

    Returns dict: trend, h (significant?), p, z, Tau, s, var_s, slope, intercept
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return dict(trend="insufficient data", h=False, p=np.nan, z=np.nan,
                    Tau=np.nan, s=np.nan, var_s=np.nan, slope=np.nan, intercept=np.nan)

    if _HAS_PYMK:
        result = mk.original_test(values, alpha=alpha)
        return dict(
            trend=result.trend, h=bool(result.h), p=float(result.p), z=float(result.z),
            Tau=float(result.Tau), s=float(result.s), var_s=float(result.var_s),
            slope=float(result.slope), intercept=float(result.intercept)
        )
    else:
        return _manual_mann_kendall(values, alpha=alpha)


# ---------------------------------------------------------------------------
# Reference text block reused in report.py
# ---------------------------------------------------------------------------
MANN_KENDALL_DESCRIPTION = {
    "title": "Mann-Kendall Trend Test",
    "definition": (
        "The Mann-Kendall (MK) Test is a non-parametric statistical test used "
        "to identify whether a time series exhibits a monotonic increasing or "
        "decreasing trend over time. Since the test is based on the relative "
        "ranking of observations rather than their actual values, it does not "
        "require the data to follow any specific probability distribution. "
        "The magnitude of the detected trend is estimated using Sen's Slope, "
        "which provides a robust estimate of the average annual rate of change."
    ),
    "purpose": (
        "The Mann-Kendall Test is widely used in hydrology to detect long-term "
        "trends in streamflow and other hydro-meteorological variables. It "
        "helps determine whether observed changes are statistically significant "
        "or are likely due to natural variability."
    ),
    "standard_values": (
        "Significance Level (\u03b1): 0.05 (95% Confidence Level)"
    ),
    "parameter_explanation": (
        "Tau (\u03c4): Indicates the direction and strength of the trend, ranging "
        "from -1 (strong decreasing trend) to +1 (strong increasing trend). "
        "A value close to zero indicates little or no monotonic trend. "
        "P-value: Represents the probability that the observed trend occurred "
        "by chance. A p-value less than 0.05 indicates a statistically "
        "significant trend at the 95% confidence level. "
        "Sen's Slope: Estimates the median rate of change of the variable per "
        "year, where a positive value indicates an increasing trend and a "
        "negative value indicates a decreasing trend."
    ),
}
