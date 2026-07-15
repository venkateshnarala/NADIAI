"""
distfit.py
NADI AI - Distribution Fitting & Frequency Analysis Module
-------------------------------------------------------------
Fits Normal, Lognormal (2P), Gumbel (EV1), GEV, Pearson Type III and
Log-Pearson Type III distributions to the annual maximum series (AMS)
using Method of Moments (MOM), L-Moments and Maximum Likelihood (MLE)
where applicable, runs goodness-of-fit tests (KS, Chi-square, Anderson-
Darling, AIC, RMSE), ranks all fitted distribution/method combinations,
and estimates design quantiles for return periods 2-1000 years for the
top-ranked distributions.

In addition to the original 5 ranking metrics (KS, Chi2, AD, AIC, RMSE),
this module also computes R^2, NSE and MAE (observed plotting-position
quantiles vs. fitted quantiles) so that a compact performance-metrics
table (R^2, NSE, RMSE, MAE) can be shown for the top-5 fitted
distributions in the report, below the observed-vs-fitted overlay plot.

Design choices (documented for transparency / for the report):
 - L-moments are computed using the standard probability-weighted-moments
   (PWM) approach (Hosking, 1990) - no external lmoments package required,
   keeping the dependency list minimal for the professor's VS Code setup.
 - "MOM" for GEV is implemented via approximate L-moment/MOM hybrid
   (pure product-moment GEV fitting is numerically unstable); this is noted
   in the report footnote.
 - Log-Pearson III fits Pearson III to log10(Q).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 500, 1000]

# ---------------------------------------------------------------------------
# L-MOMENTS (Hosking, 1990) via probability weighted moments
# ---------------------------------------------------------------------------

def _l_moments(x):
    """Compute the first four sample L-moments (l1, l2, l3, l4) and L-CV, L-skew, L-kurt."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2:
        return dict(l1=np.nan, l2=np.nan, l3=np.nan, l4=np.nan, t2=np.nan, t3=np.nan, t4=np.nan)

    # probability weighted moments b0, b1, b2, b3
    j = np.arange(n)
    b0 = np.mean(x)
    b1 = np.sum((j / (n - 1)) * x) / n if n > 1 else np.nan
    b2 = np.sum((j * (j - 1)) / ((n - 1) * (n - 2)) * x) / n if n > 2 else np.nan
    b3 = np.sum((j * (j - 1) * (j - 2)) / ((n - 1) * (n - 2) * (n - 3)) * x) / n if n > 3 else np.nan

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0 if n > 2 else np.nan
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0 if n > 3 else np.nan

    t2 = l2 / l1 if l1 != 0 else np.nan       # L-CV
    t3 = l3 / l2 if (l2 not in (0, np.nan) and not np.isnan(l2)) else np.nan  # L-skewness
    t4 = l4 / l2 if (l2 not in (0, np.nan) and not np.isnan(l2)) else np.nan  # L-kurtosis

    return dict(l1=l1, l2=l2, l3=l3, l4=l4, t2=t2, t3=t3, t4=t4)

# ---------------------------------------------------------------------------
# PARAMETER ESTIMATION FOR EACH DISTRIBUTION / METHOD
# ---------------------------------------------------------------------------

def fit_normal(x):
    x = np.asarray(x, dtype=float)
    results = {}
    mean, std = np.mean(x), np.std(x, ddof=1)
    results["MOM"] = {"loc": mean, "scale": std}
    lm = _l_moments(x)
    results["L-Moments"] = {"loc": lm["l1"], "scale": lm["l2"] * np.sqrt(np.pi)}
    loc_mle, scale_mle = stats.norm.fit(x)
    results["MLE"] = {"loc": loc_mle, "scale": scale_mle}
    return results

def fit_lognormal(x):
    """2-parameter lognormal fitted on log10(x); scipy lognorm uses natural log internally."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    logx = np.log(x)
    results = {}
    mean_log, std_log = np.mean(logx), np.std(logx, ddof=1)
    results["MOM"] = {"sigma": std_log, "scale": np.exp(mean_log)}
    lm = _l_moments(logx)
    results["L-Moments"] = {"sigma": lm["l2"] * np.sqrt(np.pi), "scale": np.exp(lm["l1"])}
    shape_mle, loc_mle, scale_mle = stats.lognorm.fit(x, floc=0)
    results["MLE"] = {"sigma": shape_mle, "scale": scale_mle}
    return results

def fit_gumbel(x):
    """Gumbel (EV1, max). scipy.stats.gumbel_r params: loc, scale."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    results = {}
    mean, std = np.mean(x), np.std(x, ddof=1)
    euler_gamma = 0.5772156649
    scale_mom = std * np.sqrt(6) / np.pi
    loc_mom = mean - euler_gamma * scale_mom
    results["MOM"] = {"loc": loc_mom, "scale": scale_mom}
    lm = _l_moments(x)
    scale_lm = lm["l2"] / np.log(2)
    loc_lm = lm["l1"] - euler_gamma * scale_lm
    results["L-Moments"] = {"loc": loc_lm, "scale": scale_lm}
    loc_mle, scale_mle = stats.gumbel_r.fit(x)
    results["MLE"] = {"loc": loc_mle, "scale": scale_mle}
    return results

def fit_gev(x):
    """Generalized Extreme Value. scipy genextreme params: c (shape), loc, scale.
    Note: scipy's shape convention is negated relative to the classic
    hydrological xi convention (kappa = -c)."""
    x = np.asarray(x, dtype=float)
    results = {}
    lm = _l_moments(x)
    t3 = lm["t3"]
    if not np.isnan(t3):
        # Hosking (1990) approximation for GEV shape from L-skewness
        c_approx = 7.8590 * (2 / (3 + t3) - np.log(2) / np.log(3)) + \
                   2.9554 * (2 / (3 + t3) - np.log(2) / np.log(3)) ** 2
        try:
            from scipy.special import gamma as gammafn
            k = -c_approx
            if abs(k) > 1e-6:
                scale_lm = (lm["l2"] * k) / ((1 - 2 ** (-k)) * gammafn(1 + k))
                loc_lm = lm["l1"] - scale_lm * (1 - gammafn(1 + k)) / k
            else:
                scale_lm = lm["l2"] / np.log(2)
                loc_lm = lm["l1"] - 0.5772156649 * scale_lm
                k = 0.0
            results["L-Moments"] = {"shape_k": k, "loc": loc_lm, "scale": scale_lm}
        except Exception:
            pass

    # MOM approximation: use L-moment estimate as MOM starting point (documented in report)
    if "L-Moments" in results:
        results["MOM"] = dict(results["L-Moments"])  # noted as L-moment-based approximation

    try:
        c_mle, loc_mle, scale_mle = stats.genextreme.fit(x)
        results["MLE"] = {"shape_k": -c_mle, "loc": loc_mle, "scale": scale_mle}
    except Exception:
        pass

    return results

def fit_pearson3(x):
    """Pearson Type III. scipy pearson3 params: skew, loc, scale."""
    x = np.asarray(x, dtype=float)
    results = {}
    mean, std, skew = np.mean(x), np.std(x, ddof=1), stats.skew(x, bias=False)
    results["MOM"] = {"skew": skew, "loc": mean, "scale": std}
    lm = _l_moments(x)
    results["L-Moments"] = {"skew": lm["t3"] * 3, "loc": lm["l1"], "scale": lm["l2"] * np.pi ** 0.5}
    try:
        skew_mle, loc_mle, scale_mle = stats.pearson3.fit(x)
        results["MLE"] = {"skew": skew_mle, "loc": loc_mle, "scale": scale_mle}
    except Exception:
        pass
    return results

def fit_log_pearson3(x):
    """Log-Pearson Type III: fit Pearson III to log10(x)."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    logx = np.log10(x)
    results = {}
    mean, std, skew = np.mean(logx), np.std(logx, ddof=1), stats.skew(logx, bias=False)
    results["MOM"] = {"skew": skew, "loc": mean, "scale": std}
    lm = _l_moments(logx)
    results["L-Moments"] = {"skew": lm["t3"] * 3, "loc": lm["l1"], "scale": lm["l2"] * np.pi ** 0.5}
    try:
        skew_mle, loc_mle, scale_mle = stats.pearson3.fit(logx)
        results["MLE"] = {"skew": skew_mle, "loc": loc_mle, "scale": scale_mle}
    except Exception:
        pass
    return results

DIST_FIT_FUNCS = {
    "Normal": fit_normal,
    "Log-Normal (2P)": fit_lognormal,
    "Gumbel (EV1)": fit_gumbel,
    "GEV": fit_gev,
    "Pearson Type III": fit_pearson3,
    "Log-Pearson Type III": fit_log_pearson3,
}

# ---------------------------------------------------------------------------
# SCIPY FROZEN-DISTRIBUTION HELPERS (for CDF/PPF/PDF evaluation & GoF tests)
# ---------------------------------------------------------------------------

def _get_scipy_dist(dist_name, params):
    """Return a frozen scipy.stats distribution object (in ORIGINAL flow units,
    i.e. log-space distributions are wrapped so cdf/ppf take/return real flow)."""
    if dist_name == "Normal":
        return stats.norm(loc=params["loc"], scale=params["scale"])
    elif dist_name == "Log-Normal (2P)":
        return stats.lognorm(s=params["sigma"], loc=0, scale=params["scale"])
    elif dist_name == "Gumbel (EV1)":
        return stats.gumbel_r(loc=params["loc"], scale=params["scale"])
    elif dist_name == "GEV":
        return stats.genextreme(c=-params["shape_k"], loc=params["loc"], scale=params["scale"])
    elif dist_name == "Pearson Type III":
        return stats.pearson3(skew=params["skew"], loc=params["loc"], scale=params["scale"])
    elif dist_name == "Log-Pearson Type III":
        return None  # handled specially (log-space)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

def dist_cdf(dist_name, params, x):
    """CDF evaluated at real-flow value(s) x, for any of the six supported distributions."""
    x = np.asarray(x, dtype=float)
    if dist_name == "Log-Pearson Type III":
        logx = np.log10(np.clip(x, 1e-9, None))
        return stats.pearson3.cdf(logx, skew=params["skew"], loc=params["loc"], scale=params["scale"])
    dist = _get_scipy_dist(dist_name, params)
    return dist.cdf(x)

def dist_ppf(dist_name, params, p):
    """Inverse CDF (quantile function) - p is non-exceedance probability (0-1)."""
    p = np.asarray(p, dtype=float)
    if dist_name == "Log-Pearson Type III":
        log_q = stats.pearson3.ppf(p, skew=params["skew"], loc=params["loc"], scale=params["scale"])
        return 10 ** log_q
    dist = _get_scipy_dist(dist_name, params)
    return dist.ppf(p)

# ---------------------------------------------------------------------------
# GOODNESS-OF-FIT TESTS
# ---------------------------------------------------------------------------

def goodness_of_fit(dist_name, params, data, n_params=2):
    """
    Compute KS, Chi-square, Anderson-Darling, AIC, RMSE for a fitted
    distribution against the observed data. Also computes R^2, NSE and MAE
    (based on observed plotting-position quantiles vs. fitted quantiles) so
    that a compact performance-metrics table can be produced for the report.

    Returns dict: ks_stat, ks_p, chi2_stat, chi2_p, ad_stat, aic, rmse,
                  r2, nse, mae
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # --- KS test ---
    try:
        cdf_func = lambda v: dist_cdf(dist_name, params, v)
        ks_stat, ks_p = stats.kstest(data, cdf_func)
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    # --- Chi-square test (equal-probability binning, k = 1+3.322log10(n)) ---
    try:
        k_bins = max(int(np.ceil(1 + 3.322 * np.log10(n))), 4)
        probs = np.linspace(0, 1, k_bins + 1)[1:-1]
        bin_edges = dist_ppf(dist_name, params, probs)
        bin_edges = np.concatenate(([-np.inf], bin_edges, [np.inf]))
        observed_freq, _ = np.histogram(data, bins=bin_edges)
        expected_freq = np.full(k_bins, n / k_bins)
        chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        dof = max(k_bins - 1 - n_params, 1)
        chi2_p = 1 - stats.chi2.cdf(chi2_stat, dof)
    except Exception:
        chi2_stat, chi2_p = np.nan, np.nan

    # --- Anderson-Darling statistic (generic, via CDF probabilities) ---
    try:
        sorted_data = np.sort(data)
        F = dist_cdf(dist_name, params, sorted_data)
        F = np.clip(F, 1e-10, 1 - 1e-10)
        i = np.arange(1, n + 1)
        ad_stat = -n - np.sum((2 * i - 1) / n * (np.log(F) + np.log(1 - F[::-1])))
    except Exception:
        ad_stat = np.nan

    # --- Log-likelihood based AIC ---
    try:
        if dist_name == "Log-Pearson Type III":
            logx = np.log10(data)
            logpdf = stats.pearson3.logpdf(logx, skew=params["skew"], loc=params["loc"], scale=params["scale"])
            loglik = np.sum(logpdf) - np.sum(np.log(data * np.log(10)))  # Jacobian for log10 transform
        else:
            dist = _get_scipy_dist(dist_name, params)
            loglik = np.sum(dist.logpdf(data))
        aic = 2 * n_params - 2 * loglik
    except Exception:
        aic = np.nan

    # --- RMSE / R^2 / NSE / MAE between empirical quantiles (Weibull plotting
    #     position) and fitted quantiles ---
    rmse, r2, nse, mae = np.nan, np.nan, np.nan, np.nan
    try:
        sorted_data = np.sort(data)
        ranks = np.arange(1, n + 1)
        plotting_pos = ranks / (n + 1)
        fitted_quantiles = dist_ppf(dist_name, params, plotting_pos)
        residuals = sorted_data - fitted_quantiles
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((sorted_data - np.mean(sorted_data)) ** 2)
        if ss_tot != 0:
            r2 = 1 - ss_res / ss_tot
            nse = 1 - ss_res / ss_tot  # NSE has the same form as R^2 here
    except Exception:
        pass

    return dict(ks_stat=ks_stat, ks_p=ks_p, chi2_stat=chi2_stat, chi2_p=chi2_p,
                ad_stat=ad_stat, aic=aic, rmse=rmse, r2=r2, nse=nse, mae=mae)

# ---------------------------------------------------------------------------
# MASTER PIPELINE: fit all distributions/methods, score, rank, quantiles
# ---------------------------------------------------------------------------

def fit_all_distributions(ams_values):
    """
    Fit all 6 distributions x up to 3 methods each to the AMS, run GoF tests,
    and return a tidy results table (one row per distribution-method combo).

    Returns DataFrame with columns:
      distribution, method, params (dict), ks_stat, chi2_stat, ad_stat, aic, rmse,
      r2, nse, mae, composite_rank_score
    """
    ams_values = np.asarray(ams_values, dtype=float)
    ams_values = ams_values[~np.isnan(ams_values)]
    ams_values = ams_values[ams_values > 0]

    rows = []
    for dist_name, fit_func in DIST_FIT_FUNCS.items():
        n_params = 3 if dist_name in ("GEV", "Pearson Type III", "Log-Pearson Type III") else 2
        try:
            method_params = fit_func(ams_values)
        except Exception as e:
            continue

        for method, params in method_params.items():
            if params is None or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in params.values()):
                continue
            try:
                gof = goodness_of_fit(dist_name, params, ams_values, n_params=n_params)
            except Exception:
                continue
            rows.append({
                "distribution": dist_name,
                "method": method,
                "params": params,
                **gof
            })

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    # --- composite rank: rank each metric (lower is better for all of these) and average ranks ---
    for col in ["ks_stat", "chi2_stat", "ad_stat", "aic", "rmse"]:
        results_df[f"rank_{col}"] = results_df[col].rank(method="min", na_option="bottom")

    rank_cols = [f"rank_{c}" for c in ["ks_stat", "chi2_stat", "ad_stat", "aic", "rmse"]]
    results_df["composite_rank_score"] = results_df[rank_cols].mean(axis=1)
    results_df = results_df.sort_values("composite_rank_score").reset_index(drop=True)
    results_df["overall_rank"] = np.arange(1, len(results_df) + 1)

    return results_df

def estimate_quantiles(top_results, return_periods=None):
    """
    For each of the top-ranked distribution/method rows, estimate discharge
    quantiles for the given return periods.

    top_results: DataFrame slice (e.g., top 5 rows of fit_all_distributions output)
    Returns DataFrame with columns: return_period, <label_1>, <label_2>, ...
    where label = "Distribution (Method)"
    """
    if return_periods is None:
        return_periods = RETURN_PERIODS
    return_periods = np.array(return_periods, dtype=float)
    non_exceed_prob = 1 - 1 / return_periods  # P(X <= x) = 1 - 1/T

    quantile_df = pd.DataFrame({"return_period": return_periods.astype(int)})
    for _, row in top_results.iterrows():
        label = f"{row['distribution']} ({row['method']})"
        try:
            q = dist_ppf(row["distribution"], row["params"], non_exceed_prob)
        except Exception:
            q = np.full(len(return_periods), np.nan)
        quantile_df[label] = q
    return quantile_df

def get_plotting_positions(ams_values):
    """Weibull plotting positions for observed AMS, sorted ascending, for overlay plots."""
    ams_values = np.asarray(ams_values, dtype=float)
    ams_values = ams_values[~np.isnan(ams_values)]
    sorted_vals = np.sort(ams_values)
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1)
    plotting_pos = ranks / (n + 1)  # non-exceedance probability
    return sorted_vals, plotting_pos

def build_distribution_curves(top_results, ams_values, n_points=200):
    """
    Build smooth CDF curves (probability vs magnitude) for the top-N distributions,
    for overlay plotting against observed data plotting positions.
    Returns list of dicts: {'label', 'x' (probability), 'y' (magnitude)}
    """
    ams_values = np.asarray(ams_values, dtype=float)
    ams_values = ams_values[~np.isnan(ams_values)]
    p_min, p_max = 1 / (len(ams_values) + 2), 1 - 1 / (len(ams_values) + 2)
    probs = np.linspace(p_min, p_max, n_points)

    curves = []
    for _, row in top_results.iterrows():
        label = f"{row['distribution']} ({row['method']})"
        try:
            y = dist_ppf(row["distribution"], row["params"], probs)
            curves.append({"label": label, "x": probs, "y": y})
        except Exception:
            continue
    return curves

# ---------------------------------------------------------------------------
# GoF reference text blocks reused in report.py
# ---------------------------------------------------------------------------
GOF_DESCRIPTIONS = {
    "KS": {
        "title": "Kolmogorov-Smirnov (KS) Test",
        "definition": "Measures the maximum absolute distance between the empirical and fitted cumulative distribution functions.",
        "purpose": "Tests the overall goodness-of-fit across the entire distribution.",
        "standard_values": "Lower KS statistic indicates a better fit; compared against critical value at alpha = 0.05.",
    },
    "Chi2": {
        "title": "Chi-Square (chi^2) Test",
        "definition": "Compares observed vs. expected frequencies of data falling into equal-probability bins defined by the fitted distribution.",
        "purpose": "Tests goodness-of-fit based on binned frequency counts; useful for checking fit across the full range of data.",
        "standard_values": "Lower chi-square statistic indicates a better fit; compared against critical chi-square at (k - 1 - m) degrees of freedom.",
    },
    "AD": {
        "title": "Anderson-Darling (AD) Test",
        "definition": "A weighted version of the KS statistic that gives more weight to the tails of the distribution.",
        "purpose": "Particularly useful in flood-frequency analysis since accurate tail estimation drives design flood magnitudes.",
        "standard_values": "Lower AD statistic indicates a better fit, especially in the distribution tails.",
    },
    "AIC": {
        "title": "Akaike Information Criterion (AIC)",
        "definition": "A model-selection criterion that balances goodness-of-fit against the number of fitted parameters, penalizing over-fitting.",
        "purpose": "Used to compare non-nested distributions with different numbers of parameters on an equal footing.",
        "standard_values": "Lower AIC indicates a preferred model (accounts for both fit quality and parsimony).",
    },
    "RMSE": {
        "title": "Root Mean Square Error (RMSE)",
        "definition": "The root-mean-square deviation between observed plotting-position quantiles and distribution-predicted quantiles.",
        "purpose": "Directly measures how closely the fitted distribution reproduces the actual observed annual maxima magnitudes.",
        "standard_values": "Lower RMSE (in flow units, e.g. m3/s) indicates a better fit.",
    },
}
