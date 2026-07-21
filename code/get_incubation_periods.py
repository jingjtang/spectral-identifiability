from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import lognorm


# ============================================================
# A) Extracted Table 3 data (Lessler et al., 2009; PMCID: PMC4327893)
# ============================================================

def get_lessler2009_table3_data() -> list[dict]:
    """
    Returns the extracted Table 3 (Lessler et al., 2009; PMCID: PMC4327893)
    with percentiles + dispersion.
    """
    return [
        {
            "disease": "Adenovirus",
            "p5": None, "p5_lo": None, "p5_hi": None,
            "p25": 4.8, "p25_lo": 4.0, "p25_hi": 5.5,
            "p50": 5.6, "p50_lo": 4.8, "p50_hi": 6.3,
            "p75": 6.5, "p75_lo": 5.6, "p75_hi": 7.4,
            "p95": None, "p95_lo": None, "p95_hi": None,
            "dispersion": 1.26, "disp_lo": 1.13, "disp_hi": 1.38,
        },
        {
            "disease": "Human coronavirus",
            "p5": None, "p5_lo": None, "p5_hi": None,
            "p25": 2.9, "p25_lo": 2.5, "p25_hi": 3.3,
            "p50": 3.2, "p50_lo": 2.8, "p50_hi": 3.7,
            "p75": 3.5, "p75_lo": 3.1, "p75_hi": 4.2,
            "p95": None, "p95_lo": None, "p95_hi": None,
            "dispersion": 1.15, "disp_lo": 1.07, "disp_hi": 1.34,
        },
        {
            "disease": "SARS-associated coronavirus",
            "p5": 1.5, "p5_lo": 1.2, "p5_hi": 1.7,
            "p25": 2.7, "p25_lo": 2.3, "p25_hi": 3.0,
            "p50": 4.0, "p50_lo": 3.6, "p50_hi": 4.4,
            "p75": 5.9, "p75_lo": 5.3, "p75_hi": 6.6,
            "p95": 10.6, "p95_lo": 8.9, "p95_hi": 12.2,
            "dispersion": 1.81, "disp_lo": 1.67, "disp_hi": 1.95,
        },
        {
            "disease": "Influenza A",
            "p5": 0.7, "p5_lo": 0.6, "p5_hi": 0.8,
            "p25": 1.1, "p25_lo": 1.0, "p25_hi": 1.2,
            "p50": 1.4, "p50_lo": 1.3, "p50_hi": 1.5,
            "p75": 1.9, "p75_lo": 1.7, "p75_hi": 2.1,
            "p95": 2.8, "p95_lo": 2.5, "p95_hi": 3.2,
            "dispersion": 1.51, "disp_lo": 1.43, "disp_hi": 1.60,
        },
        {
            "disease": "Influenza B",
            "p5": 0.3, "p5_lo": 0.2, "p5_hi": 0.3,
            "p25": 0.4, "p25_lo": 0.4, "p25_hi": 0.5,
            "p50": 0.6, "p50_lo": 0.5, "p50_hi": 0.6,
            "p75": 0.7, "p75_lo": 0.7, "p75_hi": 0.8,
            "p95": 1.1, "p95_lo": 0.9, "p95_hi": 1.3,
            "dispersion": 1.51, "disp_lo": 1.37, "disp_hi": 1.64,
        },
        {
            "disease": "Measles",
            "p5": 8.9, "p5_lo": 8.1, "p5_hi": 9.8,
            "p25": 10.9, "p25_lo": 10.2, "p25_hi": 11.6,
            "p50": 12.5, "p50_lo": 11.8, "p50_hi": 13.3,
            "p75": 14.4, "p75_lo": 13.5, "p75_hi": 15.3,
            "p95": 17.7, "p95_lo": 16.1, "p95_hi": 19.2,
            "dispersion": 1.23, "disp_lo": 1.18, "disp_hi": 1.28,
        },
        {
            "disease": "Parainfluenza",
            "p5": None, "p5_lo": None, "p5_hi": None,
            "p25": 2.1, "p25_lo": 1.6, "p25_hi": 2.6,
            "p50": 2.6, "p50_lo": 2.1, "p50_hi": 3.1,
            "p75": 3.2, "p75_lo": 2.5, "p75_hi": 3.8,
            "p95": None, "p95_lo": None, "p95_hi": None,
            "dispersion": 1.35, "disp_lo": 1.16, "disp_hi": 1.55,
        },
        {
            "disease": "Respiratory syncytial virus (RSV)",
            "p5": 3.1, "p5_lo": 2.5, "p5_hi": 3.8,
            "p25": 3.8, "p25_lo": 3.3, "p25_hi": 4.4,
            "p50": 4.4, "p50_lo": 3.9, "p50_hi": 4.9,
            "p75": 5.1, "p75_lo": 4.5, "p75_hi": 5.7,
            "p95": 6.3, "p95_lo": 5.2, "p95_hi": 7.3,
            "dispersion": 1.24, "disp_lo": 1.13, "disp_hi": 1.35,
        },
        {
            "disease": "Rhinovirus",
            "p5": 0.8, "p5_lo": 0.4, "p5_hi": 1.2,
            "p25": 1.3, "p25_lo": 0.9, "p25_hi": 1.8,
            "p50": 1.9, "p50_lo": 1.4, "p50_hi": 2.4,
            "p75": 2.7, "p75_lo": 2.0, "p75_hi": 3.4,
            "p95": 4.5, "p95_lo": 2.9, "p95_hi": 6.2,
            "dispersion": 1.68, "disp_lo": 1.36, "disp_hi": 2.01,
        },
    ]


# ============================================================
# B) Fitting and discretization
# ============================================================

# def fit_lognormal_from_percentiles(row: pd.Series) -> Tuple[float, float, str]:
#     """
#     Returns (mu, sigma, method) for LogNormal(mu, sigma).
#     - If 5/25/50/75/95 present: quantile-matching over all 5.
#     - Else if 25/50/75 present: quantile-matching over central 3.
#     - Else: use median+dispersion mapping: mu=ln(median), sigma=ln(dispersion)
#     """
#     q_map = [(0.05, "p5"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]
#     qs, xs = [], []
#     for q, key in q_map:
#         val = row.get(key, None)
#         if pd.notna(val) and val is not None:
#             qs.append(q)
#             xs.append(float(val))
#     qs = np.array(qs, dtype=float)
#     xs = np.array(xs, dtype=float)
#
#     if len(xs) == 5:
#         def loss(params):
#             mu, sigma = params
#             model = lognorm.ppf(qs, s=sigma, scale=np.exp(mu))
#             return float(np.sum((model - xs) ** 2))
#
#         x0 = [np.log(float(row["p50"])), 0.5]
#         res = minimize(loss, x0=x0, bounds=[(None, None), (1e-8, None)])
#         return float(res.x[0]), float(res.x[1]), "quantile_match_5"
#
#     central_keys = ["p25", "p50", "p75"]
#     if all(pd.notna(row.get(k, None)) and row.get(k, None) is not None for k in central_keys):
#         qs3 = np.array([0.25, 0.50, 0.75], dtype=float)
#         xs3 = np.array([float(row["p25"]), float(row["p50"]), float(row["p75"])], dtype=float)
#
#         def loss(params):
#             mu, sigma = params
#             model = lognorm.ppf(qs3, s=sigma, scale=np.exp(mu))
#             return float(np.sum((model - xs3) ** 2))
#
#         x0 = [np.log(float(row["p50"])), np.log(float(row["dispersion"]))]
#         res = minimize(loss, x0=x0, bounds=[(None, None), (1e-8, None)])
#         return float(res.x[0]), float(res.x[1]), "quantile_match_3"
#
#     mu = np.log(float(row["p50"]))
#     sigma = np.log(float(row["dispersion"]))
#     return float(mu), float(sigma), "median_dispersion"


def fit_lognormal_from_percentiles(row: pd.Series) -> Tuple[float, float, str]:
    """
    Strictly reconstructs the Lessler et al. Table 3 lognormal distribution.

    Lessler reports the lognormal distribution using:
      median incubation period = p50
      dispersion factor = exp(sigma_log)

    Therefore:
      mu_log = log(p50)
      sigma_log = log(dispersion)
    """
    mu = np.log(float(row["p50"]))
    sigma = np.log(float(row["dispersion"]))
    return float(mu), float(sigma), "median_dispersion"


# ============================================================
# C1) Support-first objects (continuous distributions + reported stats)
# ============================================================

def _to_opt_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return float(x)


@dataclass(frozen=True)
class ReportedQuantiles:
    # point estimates
    p5: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p95: Optional[float]
    # CIs (optional)
    p5_lo: Optional[float];  p5_hi: Optional[float]
    p25_lo: Optional[float]; p25_hi: Optional[float]
    p50_lo: Optional[float]; p50_hi: Optional[float]
    p75_lo: Optional[float]; p75_hi: Optional[float]
    p95_lo: Optional[float]; p95_hi: Optional[float]
    # dispersion + CI
    dispersion: Optional[float]
    disp_lo: Optional[float]
    disp_hi: Optional[float]


@dataclass(frozen=True)
class DelayDistSupport:
    """
    Flexible support package for downstream scripts:
      - continuous SciPy distribution (dist)
      - fitted params (mu, sigma)
      - reported quantiles/CI (reported)
      - provenance (source) and fit_method
    """
    disease: str
    family: str
    mu: float
    sigma: float
    dist: Any
    fit_method: str
    reported: ReportedQuantiles
    source: str


def build_lessler2009_support(
    *,
    source: str = "Lessler et al. 2009 (Table 3, incubation period; PMCID: PMC4327893)",
) -> Tuple[Dict[str, DelayDistSupport], pd.DataFrame]:
    """
    Returns:
      support_dict: disease -> DelayDistSupport (continuous lognormal dist + reported stats)
      params_df: extracted + fitted parameter table (one row per disease), suitable for saving/PDF/export
    """
    params_df = pd.DataFrame(get_lessler2009_table3_data()).copy()

    fits = [fit_lognormal_from_percentiles(r) for _, r in params_df.iterrows()]
    params_df["mu"] = [x[0] for x in fits]
    params_df["sigma"] = [x[1] for x in fits]
    params_df["fit_method"] = [x[2] for x in fits]
    params_df["dist_family"] = "lognormal"

    support_dict: Dict[str, DelayDistSupport] = {}
    for _, r in params_df.iterrows():
        disease = str(r["disease"])
        mu = float(r["mu"])
        sigma = float(r["sigma"])
        dist = lognorm(s=sigma, scale=np.exp(mu))

        reported = ReportedQuantiles(
            p5=_to_opt_float(r.get("p5")),
            p25=_to_opt_float(r.get("p25")),
            p50=_to_opt_float(r.get("p50")),
            p75=_to_opt_float(r.get("p75")),
            p95=_to_opt_float(r.get("p95")),
            p5_lo=_to_opt_float(r.get("p5_lo")),   p5_hi=_to_opt_float(r.get("p5_hi")),
            p25_lo=_to_opt_float(r.get("p25_lo")), p25_hi=_to_opt_float(r.get("p25_hi")),
            p50_lo=_to_opt_float(r.get("p50_lo")), p50_hi=_to_opt_float(r.get("p50_hi")),
            p75_lo=_to_opt_float(r.get("p75_lo")), p75_hi=_to_opt_float(r.get("p75_hi")),
            p95_lo=_to_opt_float(r.get("p95_lo")), p95_hi=_to_opt_float(r.get("p95_hi")),
            dispersion=_to_opt_float(r.get("dispersion")),
            disp_lo=_to_opt_float(r.get("disp_lo")),
            disp_hi=_to_opt_float(r.get("disp_hi")),
        )

        support_dict[disease] = DelayDistSupport(
            disease=disease,
            family="lognormal",
            mu=mu,
            sigma=sigma,
            dist=dist,
            fit_method=str(r["fit_method"]),
            reported=reported,
            source=source,
        )

    return support_dict, params_df


def select_support(
    support_dict: Dict[str, DelayDistSupport],
    diseases: Optional[Iterable[str]] = None,
) -> Dict[str, DelayDistSupport]:
    """
    Convenience: select a subset of diseases (or all if diseases is None).
    Downstream scripts can pass a single disease ["Influenza A"] or many.
    """
    if diseases is None:
        return dict(support_dict)
    diseases_list = list(diseases)
    out: Dict[str, DelayDistSupport] = {}
    for name in diseases_list:
        if name not in support_dict:
            raise KeyError(f"Unknown disease '{name}'. Available: {list(support_dict.keys())}")
        out[name] = support_dict[name]
    return out
