import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages


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


def discretize_lognormal_pmf(mu: float, sigma: float, max_days: int) -> Tuple[np.ndarray, float]:
    """
    Discretize LogNormal(mu, sigma) into integer-day bins:
      pmf[d] = P(d <= T < d+1) for d=0..max_days-1
      tail   = P(T >= max_days)

    Returns (pmf_main, tail). pmf_main length=max_days.
    """
    dist = lognorm(s=sigma, scale=np.exp(mu))
    edges = np.arange(0, max_days + 1, dtype=float)
    cdf_edges = dist.cdf(edges)
    pmf_main = np.diff(cdf_edges)
    tail = float(1.0 - dist.cdf(max_days))

    # Numerical safety + renormalize
    pmf_main = np.clip(pmf_main, 0.0, 1.0)
    tail = max(0.0, min(1.0, tail))
    s = float(pmf_main.sum() + tail)
    if s > 0:
        pmf_main = pmf_main / s
        tail = tail / s
    return pmf_main, tail


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


# ============================================================
# C2) Your original PMF-focused builder (kept for compatibility)
# ============================================================

@dataclass(frozen=True)
class DelayDistribution:
    """
    In-memory representation that downstream scripts can use directly.
    """
    disease: str
    dist_family: str                # "lognormal"
    mu: float
    sigma: float
    fit_method: str
    max_days: int
    pmf_main: np.ndarray            # shape (max_days,), bins [d,d+1)
    tail: float                     # P(T >= max_days)
    scipy_dist: Any                 # scipy.stats distribution object


def build_lessler2009_incubation_distributions(
    *,
    max_days: int = 30,
    include_scipy_dist: bool = True,
) -> Tuple[Dict[str, DelayDistribution], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns:
      dist_dict: disease -> DelayDistribution
      params_df: extracted + fitted parameter table (one row per disease)
      pmf_long_df: tidy PMF (disease, delay_day, prob, bin_left, bin_right, is_tail)
      pmf_wide_df: wide PMF (rows disease, cols bins + tail)
      meta: dict with discretization conventions and provenance
    """
    params_df = pd.DataFrame(get_lessler2009_table3_data()).copy()

    # Fit per disease
    fits = [fit_lognormal_from_percentiles(r) for _, r in params_df.iterrows()]
    params_df["mu"] = [x[0] for x in fits]
    params_df["sigma"] = [x[1] for x in fits]
    params_df["fit_method"] = [x[2] for x in fits]
    params_df["dist_family"] = "lognormal"

    dist_dict: Dict[str, DelayDistribution] = {}
    pmf_long_rows = []
    pmf_wide_rows = []

    for _, r in params_df.iterrows():
        disease = str(r["disease"])
        mu = float(r["mu"])
        sigma = float(r["sigma"])
        pmf_main, tail = discretize_lognormal_pmf(mu, sigma, max_days=max_days)

        # long rows
        for d in range(max_days):
            pmf_long_rows.append({
                "disease": disease,
                "delay_day": int(d),
                "bin_left": float(d),
                "bin_right": float(d + 1),
                "is_tail": False,
                "prob": float(pmf_main[d]),
            })
        pmf_long_rows.append({
            "disease": disease,
            "delay_day": int(max_days),
            "bin_left": float(max_days),
            "bin_right": np.inf,
            "is_tail": True,
            "prob": float(tail),
        })

        pmf_wide_rows.append(np.concatenate([pmf_main, [tail]]))

        scipy_dist = lognorm(s=sigma, scale=np.exp(mu)) if include_scipy_dist else None
        dist_dict[disease] = DelayDistribution(
            disease=disease,
            dist_family="lognormal",
            mu=mu,
            sigma=sigma,
            fit_method=str(r["fit_method"]),
            max_days=max_days,
            pmf_main=pmf_main,
            tail=float(tail),
            scipy_dist=scipy_dist,
        )

    pmf_long_df = pd.DataFrame(pmf_long_rows)
    pmf_wide_df = pd.DataFrame(
        np.vstack(pmf_wide_rows),
        index=params_df["disease"],
        columns=[f"[{d},{d+1})" for d in range(max_days)] + [f">={max_days}"],
    )

    meta = {
        "source": "Lessler et al. 2009 (Table 3, incubation period; PMCID: PMC4327893)",
        "dist_family": "lognormal",
        "paramization": {"mu": "ln(scale)", "sigma": "log-space std"},
        "discretization": {
            "bins": f"[d, d+1) for d=0..{max_days-1}",
            "tail_bin": f"[{max_days}, +inf)",
            "normalization": "pmf_main and tail renormalized to sum to 1 (numerical safety)",
        },
        "max_days": int(max_days),
    }

    return dist_dict, params_df, pmf_long_df, pmf_wide_df, meta


# ============================================================
# D) Optional exporters (keep your save functionality)
# ============================================================

def export_lessler2009_outputs(
    params_df: pd.DataFrame,
    pmf_long_df: pd.DataFrame,
    pmf_wide_df: pd.DataFrame,
    meta: dict,
    *,
    out_prefix: str = "lessler2009_table3_incubation",
) -> dict:
    """
    Saves:
      - extracted+fitted params CSV
      - PMF long CSV
      - PMF wide CSV
      - meta JSON
    Returns dict of file paths.
    """
    params_csv = f"{out_prefix}_params.csv"
    pmf_long_csv = f"{out_prefix}_pmf_long.csv"
    pmf_wide_csv = f"{out_prefix}_pmf_wide.csv"
    meta_json = f"{out_prefix}_meta.json"

    params_df.to_csv(params_csv, index=False)
    pmf_long_df.to_csv(pmf_long_csv, index=False)
    pmf_wide_df.to_csv(pmf_wide_csv, index=True)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "params_csv": params_csv,
        "pmf_long_csv": pmf_long_csv,
        "pmf_wide_csv": pmf_wide_csv,
        "meta_json": meta_json,
    }


def make_lessler2009_inspection_pdf(
    params_df: pd.DataFrame,
    *,
    pdf_path: str = "lessler2009_table3_incubation_inspection.pdf",
    max_days: int = 30,
) -> str:
    """
    Multi-page PDF: per disease shows daily PMF bars + CDF line + annotation.
    """
    with PdfPages(pdf_path) as pdf:
        for _, r in params_df.iterrows():
            disease = r["disease"]
            mu, sigma = float(r["mu"]), float(r["sigma"])
            dist = lognorm(s=sigma, scale=np.exp(mu))

            k = np.arange(0, max_days + 1)
            pmf = dist.cdf(k + 1) - dist.cdf(k)
            cdf = dist.cdf(k + 1)

            fig = plt.figure(figsize=(8.5, 5.5))
            ax1 = plt.gca()
            ax1.bar(k, pmf, width=0.9, align="center")
            ax1.set_xlabel("Delay (days since infection)")
            ax1.set_ylabel("Daily probability mass  P(k ≤ T < k+1)")
            ax1.set_title(f"{disease} — incubation period (Lessler et al., 2009)")
            ax1.set_xlim(-0.5, max_days + 0.5)

            ax2 = ax1.twinx()
            ax2.plot(k, cdf, linewidth=2)
            ax2.set_ylabel("CDF  P(T < k+1)")
            ax2.set_ylim(0, 1.0)

            def fmt(x):
                return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:g}"

            ptxt = (
                f"Reported percentiles (days):\n"
                f"p5={fmt(r.get('p5'))}, p25={fmt(r.get('p25'))}, p50={fmt(r.get('p50'))}, "
                f"p75={fmt(r.get('p75'))}, p95={fmt(r.get('p95'))}\n"
                f"Dispersion={fmt(r.get('dispersion'))}\n"
                f"Fit: {r.get('fit_method')}\n"
                f"μ={mu:.3f}, σ={sigma:.3f}"
            )
            ax1.text(
                0.02, 0.95, ptxt,
                transform=ax1.transAxes,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path