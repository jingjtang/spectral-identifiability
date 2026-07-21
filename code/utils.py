#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure: geometry of the feasible upstream solution space.

This version removes reconstruction-method comparisons and focuses on the
geometry implied by the paper's separability functional:

  A. The same COVID-19 confirmed-case downstream series is fitted by
     cutoff-specific low-frequency upstream classes.
  B. The corresponding upstream feasible bands expand as higher frequencies
     are admitted.
  C. Short-term growth-rate trajectories vary strongly as higher-frequency
     components are admitted.
  D. The average feasible-band width increases as the frequency cutoff is
     relaxed.

Outputs:
  ../figs/feasible_region.png
  ../figs/feasible_region.pdf
  ../figs/band_width.csv
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import json
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import MonthLocator, DateFormatter
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy import sparse

from get_empirical_delays import (
    load_schema,
    load_datasets_cfg,
    load_all_from_catalog,
    build_empirical_delay_support,
    select_empirical_support,
)

# -------------------------
# Shared project paths / data loading
# -------------------------
CODE_DIR = Path(__file__).resolve().parent
PAPER_DIR = CODE_DIR.parent
CONFIG_DIR = PAPER_DIR / "config"
DATA_DIR = CODE_DIR / "data"

DEFAULT_ONSET_REPORT_DELAY_KEY = (
    "zhang2020_cn_covid::symptom_onset_to_report::"
    "date_symptom_onset->date_case_report::lognormal"
)
ONSET_REPORT_COLS = ("date_symptom_onset", "date_case_report")


def _resolve_dataset_paths(datasets_cfg):
    resolved = []
    for cfg in datasets_cfg:
        cfg = dict(cfg)
        source = dict(cfg.get("source", {}))
        if source.get("type") == "path":
            value = Path(source["value"])
            if not value.is_absolute():
                source["value"] = str((CODE_DIR / value).resolve())
                cfg["source"] = source
        elif source.get("type") == "url":
            suffix = Path(urlparse(source["value"]).path).suffix or ".csv"
            cached = DATA_DIR / "delay_distributions" / "remote_sources" / f"{cfg['name']}{suffix}"
            if cached.exists():
                source["type"] = "path"
                source["value"] = str(cached.resolve())
                cfg["source"] = source
        resolved.append(cfg)
    return resolved


def load_delay_catalog(config_dir: Path = CONFIG_DIR):
    schema = load_schema(config_dir / "schema.json")
    datasets_cfg = load_datasets_cfg(config_dir / "datasets")
    return {**schema, "datasets": _resolve_dataset_paths(datasets_cfg)}


def load_delay_datasets(config_dir: Path = CONFIG_DIR):
    return load_all_from_catalog(load_delay_catalog(config_dir))


def load_empirical_delay_support(
    *,
    dist_family: str = "lognormal",
    max_delay_days: int = 120,
    config_dir: Path = CONFIG_DIR,
):
    datasets = load_delay_datasets(config_dir)
    return build_empirical_delay_support(
        datasets,
        dist_family=dist_family,
        max_delay_days=max_delay_days,
    )


def load_empirical_delay_subset(
    *,
    cols_match=ONSET_REPORT_COLS,
    dist_family: str = "lognormal",
    max_delay_days: int = 120,
    config_dir: Path = CONFIG_DIR,
):
    support, summary_df = load_empirical_delay_support(
        dist_family=dist_family,
        max_delay_days=max_delay_days,
        config_dir=config_dir,
    )
    return select_empirical_support(support, cols_match=cols_match), summary_df


def load_empirical_delay_dist(
    delay_key: str = DEFAULT_ONSET_REPORT_DELAY_KEY,
    *,
    cols_match=ONSET_REPORT_COLS,
    dist_family: str = "lognormal",
    max_delay_days: int = 120,
    config_dir: Path = CONFIG_DIR,
):
    subset, _ = load_empirical_delay_subset(
        cols_match=cols_match,
        dist_family=dist_family,
        max_delay_days=max_delay_days,
        config_dir=config_dir,
    )
    if delay_key not in subset:
        raise KeyError(f"{delay_key} not found in empirical support.")
    return subset[delay_key].dist


def load_empirical_delay_pmf(
    delay_key: str = DEFAULT_ONSET_REPORT_DELAY_KEY,
    *,
    tau_max: int = 60,
    fallback_lognormal: bool = False,
):
    try:
        dist = load_empirical_delay_dist(delay_key)
        return pdf_dist_to_daily_pmf(dist, tau_max=tau_max)
    except Exception:
        if not fallback_lognormal:
            raise
        from scipy.stats import lognorm
        return pdf_dist_to_daily_pmf(lognorm(s=0.5, scale=5.0), tau_max=tau_max)


def set_epidata_api_key(config_dir: Path = CONFIG_DIR) -> bool:
    key_path = config_dir / "epidata_api_key.txt"
    if not key_path.exists():
        return False
    os.environ["DELPHI_EPIDATA_KEY"] = key_path.read_text().strip()
    return True


def apply_confirmed_cases_axis(ax, observed_values, *, offset_text_size=8.0, nbins=5):
    observed_values = np.asarray(observed_values, float)
    observed_values = observed_values[np.isfinite(observed_values)]
    if observed_values.size:
        ymax = float(np.nanmax(observed_values))
        ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(offset_text_size)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins))


def load_local_signal_df(
    *,
    data_source: str,
    signals: str,
    geo_type: str,
    geo_values: str,
) -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob(f"covid_{geo_type}_*_{signals}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No local cached signal CSV for {data_source}/{signals} ({geo_type})."
        )

    source_hint = "jhu" if data_source == "jhu-csse" else data_source.replace("-", "_")
    preferred = [p for p in matches if f"_{source_hint}_" in p.name]
    csv_path = preferred[0] if preferred else matches[0]

    df = pd.read_csv(csv_path)
    if "geo_value" in df.columns:
        df = df[df["geo_value"].astype(str).str.lower() == str(geo_values).lower()]
    if df.empty:
        raise ValueError(f"No local rows for {geo_type}={geo_values} in {csv_path}.")

    df["date"] = pd.to_datetime(df["time_value"], errors="raise")
    out = df[["date", "value"]].rename(columns={"value": "y"}).sort_values("date")
    out["data_source"] = data_source
    out["signal"] = signals
    out["geo_type"] = geo_type
    out["geo_value"] = geo_values
    return out


OUT_DIR = Path("../figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data helpers
# ============================================================
def fetch_signal_df(
    epidata,
    *,
    data_source: str,
    signals: str,
    time_values,
    geo_type: str,
    geo_values: str,
) -> pd.DataFrame:
    try:
        df = (
            epidata.pub_covidcast(
                data_source=data_source,
                signals=signals,
                time_type="day",
                time_values=time_values,
                geo_type=geo_type,
                geo_values=geo_values,
            )
            .df()
            .copy()
        )
    except Exception:
        return load_local_signal_df(
            data_source=data_source,
            signals=signals,
            geo_type=geo_type,
            geo_values=geo_values,
        )

    if df.empty or "value" not in df.columns or df["value"].isna().all():
        return load_local_signal_df(
            data_source=data_source,
            signals=signals,
            geo_type=geo_type,
            geo_values=geo_values,
        )

    if "time_value" in df.columns:
        tv = df["time_value"]
        if pd.api.types.is_numeric_dtype(tv):
            df["date"] = pd.to_datetime(tv.astype(int).astype(str), format="%Y%m%d")
        else:
            df["date"] = pd.to_datetime(tv.astype(str))
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Could not find a time column.")

    out = (
        df[["date", "value"]]
        .rename(columns={"value": "y"})
        .sort_values("date")
    )
    return out


def align_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    s = df.set_index("date")["y"].reindex(idx)
    # Note: limit_direction="both" fills missing values at both ends. This is
    # useful for small internal gaps, but if the local cache does not cover the
    # requested window, it effectively extrapolates the edge values and can
    # create boundary artifacts. Keep cached data wider than plotted windows.
    s = s.interpolate(limit_direction="both")
    y = np.maximum(s.values.astype(float), 0.0)
    return pd.DataFrame({"date": idx, "y": y})


def slice_date_range(dates, *arrays, start: str, end: str):
    dates = pd.to_datetime(np.asarray(dates))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    out = [dates[mask]]
    for arr in arrays:
        arr = np.asarray(arr)
        out.append(arr[mask])
    return out


# ============================================================
# Trend filtering via R genlasso
# ============================================================
R_TRENDFILTER_SCRIPT = r"""
args <- commandArgs(trailingOnly=TRUE)
in_json <- args[1]
out_csv <- args[2]

suppressPackageStartupMessages({
  if (!requireNamespace("genlasso", quietly=TRUE)) {
    stop("R package 'genlasso' is required. Install in R: install.packages('genlasso')")
  }
  library(genlasso)
  library(jsonlite)
})

cfg <- jsonlite::fromJSON(in_json)
y <- as.numeric(cfg$y)
ord <- as.integer(cfg$ord)
df_target <- as.numeric(cfg$df_target)

fit <- genlasso::trendfilter(y, ord=ord)

idx <- which.min(abs(fit$df - df_target))
mu_hat <- as.numeric(fit$beta[, idx])
df_used <- as.numeric(fit$df[idx])
lam_used <- as.numeric(fit$lambda[idx])

out <- data.frame(mu_hat=mu_hat, df_used=df_used, lam_used=lam_used)
write.csv(out, out_csv, row.names=FALSE)
"""


def trendfilter_mean_genlasso(
    y: np.ndarray,
    *,
    df_target: float,
    ord: int = 2,
    rscript_bin: str = "Rscript",
):
    y = np.asarray(y, float)

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "cfg.json"
        out_csv = Path(td) / "out.csv"
        r_path = Path(td) / "trendfilter.R"

        cfg = {"y": y.tolist(), "df_target": float(df_target), "ord": int(ord)}
        cfg_path.write_text(json.dumps(cfg))
        r_path.write_text(R_TRENDFILTER_SCRIPT)

        cmd = [rscript_bin, str(r_path), str(cfg_path), str(out_csv)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "R trendfilter call failed.\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
            )

        out = pd.read_csv(out_csv)

    mu_hat = np.maximum(out["mu_hat"].values.astype(float), 1e-8)
    df_used = float(out["df_used"].iloc[0])
    lam_used = float(out["lam_used"].iloc[0])
    return mu_hat, df_used, lam_used


# ============================================================
# Noise fits
# ============================================================
@dataclass
class NoiseFit:
    model: str
    loglik: float
    params: Dict[str, float]
    aic: float
    bic: float


@dataclass
class JointFit:
    df_target: float
    df_used: float
    lam_used: float
    mu_hat: np.ndarray
    best_noise: NoiseFit


def loglik_gaussian(y: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
    resid = y - mu
    sigma2 = max(float(np.mean(resid ** 2)), 1e-12)
    sigma = float(np.sqrt(sigma2))
    n = len(y)
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    return float(ll), sigma


def loglik_poisson(y: np.ndarray, mu: np.ndarray) -> float:
    from scipy.special import gammaln
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))


def loglik_nb_phi(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    from scipy.special import gammaln
    mu = np.maximum(mu, 1e-12)
    phi = max(float(phi), 1e-12)
    return float(
        np.sum(
            gammaln(y + phi)
            - gammaln(phi)
            - gammaln(y + 1.0)
            + phi * (np.log(phi) - np.log(phi + mu))
            + y * (np.log(mu) - np.log(phi + mu))
        )
    )


def fit_noise_models(y: np.ndarray, mu: np.ndarray, *, k_mean_df: float) -> List[NoiseFit]:
    n = len(y)
    fits: List[NoiseFit] = []

    ll_g, sigma = loglik_gaussian(y, mu)
    k = k_mean_df + 1
    fits.append(NoiseFit("gaussian", ll_g, {"sigma": sigma}, -2 * ll_g + 2 * k, -2 * ll_g + np.log(n) * k))

    ll_p = loglik_poisson(y, mu)
    k = k_mean_df
    fits.append(NoiseFit("poisson", ll_p, {}, -2 * ll_p + 2 * k, -2 * ll_p + np.log(n) * k))

    def obj(log_phi):
        return -loglik_nb_phi(y, mu, np.exp(log_phi))

    res = minimize_scalar(obj, bounds=(-8, 12), method="bounded")
    phi_hat = float(np.exp(res.x))
    ll_nb = -float(res.fun)
    k = k_mean_df + 1
    fits.append(NoiseFit("nb", ll_nb, {"phi": phi_hat}, -2 * ll_nb + 2 * k, -2 * ll_nb + np.log(n) * k))

    return fits


def select_best_df_and_noise(
    y: np.ndarray,
    *,
    df_grid: List[float],
    ord: int = 2,
    criterion: str = "AIC",
    rscript_bin: str = "Rscript",
) -> JointFit:
    best = None
    best_score = np.inf

    for df_target in df_grid:
        mu_hat, df_used, lam_used = trendfilter_mean_genlasso(
            y, df_target=df_target, ord=ord, rscript_bin=rscript_bin
        )
        fits = fit_noise_models(y, mu_hat, k_mean_df=df_used)

        if criterion.upper() == "BIC":
            noise_best = min(fits, key=lambda f: f.bic)
            score = noise_best.bic
        else:
            noise_best = min(fits, key=lambda f: f.aic)
            score = noise_best.aic

        if score < best_score:
            best_score = score
            best = JointFit(float(df_target), float(df_used), float(lam_used), mu_hat, noise_best)

    if best is None:
        raise RuntimeError("Failed to select best trendfilter/noise model.")
    return best


def get_variance_scale(mu_hat: np.ndarray, noise_fit: NoiseFit) -> np.ndarray:
    mu_hat = np.asarray(mu_hat, float)
    if noise_fit.model == "gaussian":
        sigma = float(noise_fit.params.get("sigma", np.std(mu_hat)))
        return np.full_like(mu_hat, max(sigma**2, 1e-8))
    if noise_fit.model == "poisson":
        return np.maximum(mu_hat, 1.0)
    if noise_fit.model == "nb":
        phi = float(noise_fit.params.get("phi", 1.0))
        return np.maximum(mu_hat + (mu_hat**2) / max(phi, 1e-8), 1.0)
    raise ValueError(f"Unknown noise model {noise_fit.model}")


def get_effective_noise_scale(mu_hat: np.ndarray, noise_fit: NoiseFit) -> float:
    """Constant Methods normalization for separability bounds.

    Gaussian uses the fitted variance. Poisson and negative-binomial use the
    minimum downstream mean because the KL upper bounds require m_min and
    V(m_min), respectively; using a mean or maximum variance would not match
    the conservative testing-error bound.
    """
    mu_hat = np.asarray(mu_hat, float)
    if noise_fit.model == "gaussian":
        sigma = float(noise_fit.params.get("sigma", np.std(mu_hat)))
        return max(sigma**2, 1e-8)

    m_min = max(float(np.min(mu_hat)), 1e-8)
    if noise_fit.model == "poisson":
        return m_min
    if noise_fit.model == "nb":
        phi = max(float(noise_fit.params.get("phi", 1.0)), 1e-8)
        return m_min + (m_min**2) / phi
    raise ValueError(f"Unknown noise model {noise_fit.model}")


# ============================================================
# Delay / convolution / Fourier geometry
# ============================================================
def normalize_pmf(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, float)
    g = np.clip(g, 0.0, None)
    s = g.sum()
    if s <= 0:
        raise ValueError("Delay PMF must have positive sum.")
    return g / s


def pdf_dist_to_daily_pmf(dist, tau_max: int):
    edges = np.arange(0, tau_max + 2)
    g = dist.cdf(edges[1:]) - dist.cdf(edges[:-1])
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_pmf(g)


def convolve_mean(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return fftconvolve(x, g, mode="full")[: len(x)]


def build_convolution_matrix(g: np.ndarray, T: int) -> sparse.csr_matrix:
    g = np.asarray(g, float)
    kmax = len(g) - 1
    rows, cols, vals = [], [], []
    for t in range(T):
        km = min(kmax, t)
        for k in range(km + 1):
            rows.append(t)
            cols.append(t - k)
            vals.append(g[k])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(T, T)).tocsr()


def real_fourier_basis(T: int, dt: float = 1.0, include_dc: bool = True):
    t = np.arange(T, dtype=float)
    freqs = np.fft.rfftfreq(T, d=dt)
    basis = []

    if include_dc:
        basis.append((0.0, np.ones(T, dtype=float) / np.sqrt(T), "dc"))

    for k, f in enumerate(freqs):
        if k == 0:
            continue
        if (T % 2 == 0) and (k == T // 2):
            basis.append((float(f), np.cos(2 * np.pi * f * t) / np.sqrt(T), f"nyq_{f:.8f}"))
        else:
            basis.append((float(f), np.sqrt(2.0 / T) * np.cos(2 * np.pi * f * t), f"cos_{f:.8f}"))
            basis.append((float(f), np.sqrt(2.0 / T) * np.sin(2 * np.pi * f * t), f"sin_{f:.8f}"))
    return basis


def build_cutoff_design_matrix(T: int, f_cut: float, dt: float = 1.0):
    cols, freqs, names = [], [], []
    for f, psi, name in real_fourier_basis(T=T, dt=dt, include_dc=True):
        if f <= f_cut:
            cols.append(psi)
            freqs.append(float(f))
            names.append(name)
    return np.column_stack(cols), freqs, names


def delay_response_rfft(g: np.ndarray, T: int) -> np.ndarray:
    gpad = np.zeros(T)
    gpad[: len(g)] = g
    return np.fft.rfft(gpad)


def lambda_for_frequency_map(g: np.ndarray, T: int, sigma2_eff: float):
    f_rfft = np.fft.rfftfreq(T, d=1.0)
    G = delay_response_rfft(g, T)
    Gpow = np.abs(G) ** 2
    lam_map = {
        float(f): max(float(gp / max(sigma2_eff, 1e-12)), 1e-12)
        for f, gp in zip(f_rfft, Gpow)
    }
    lam_map[0.0] = 1.0 / max(sigma2_eff, 1e-12)
    return lam_map


def lambdas_for_basis(freqs: List[float], lam_map: Dict[float, float]) -> np.ndarray:
    return np.array([lam_map[float(f)] for f in freqs], dtype=float)


def analytic_band_width_time_domain(
    T: int,
    g: np.ndarray,
    sigma2_eff: float,
    f_cut: float,
    tau: float,
    dt: float = 1.0,
) -> np.ndarray:
    lam_map = lambda_for_frequency_map(g, T, sigma2_eff)
    B2 = np.zeros(T, dtype=float)
    for f, psi, _ in real_fourier_basis(T=T, dt=dt, include_dc=True):
        if f <= f_cut:
            B2 += (psi ** 2) / lam_map[float(f)]
    return np.sqrt(tau * B2)


def compute_cutoff_specific_reference_strict(
    y: np.ndarray,
    mu_hat: np.ndarray,
    g: np.ndarray,
    *,
    f_cut: float,
    ridge_theta: float = 1e-6,
    solver_gauss: str = "OSQP",
):
    import cvxpy as cp

    y = np.asarray(y, float)
    mu_hat = np.asarray(mu_hat, float)
    g = normalize_pmf(g)
    T = len(y)

    Phi, freqs, names = build_cutoff_design_matrix(T=T, f_cut=f_cut, dt=1.0)
    K = build_convolution_matrix(g, T)

    theta = cp.Variable(Phi.shape[1])
    U_expr = Phi @ theta
    m_expr = K @ U_expr

    if Phi.shape[1] >= 2:
        reg = ridge_theta * cp.sum_squares(theta[1:])
    else:
        reg = 0.0

    obj = cp.sum_squares(m_expr - mu_hat) + reg
    constraints = [U_expr >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=solver_gauss, verbose=False)

    if theta.value is None:
        raise RuntimeError(f"Strict cutoff optimizer failed for f_cut={f_cut:.6f}")

    theta_hat = np.asarray(theta.value).reshape(-1)
    U_fc = np.maximum(Phi @ theta_hat, 0.0)
    KU_fc = convolve_mean(U_fc, g)
    return U_fc, KU_fc, theta_hat, freqs, names


def tau_from_alpha(alpha_total_error: float) -> float:
    return 4.0 * (1.0 - alpha_total_error) ** 2


def cutoff_label(fc: float) -> str:
    known = {
        1 / 35: "1/35",
        1 / 21: "1/21",
        1 / 14: "1/14",
        1 / 7: "1/7",
    }
    for val, lab in known.items():
        if abs(fc - val) < 1e-10:
            return lab
    return f"{fc:.3f}"


# ============================================================
# Epidemiological feature summaries
# ============================================================
def smooth_reflect(x, window=7):
    x = np.asarray(x, float)
    if window <= 1:
        return x.copy()
    if np.any(~np.isfinite(x)):
        s = pd.Series(x)
        x = (
            s.interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
    pad = window // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def rolling_growth_rate(x, horizon=7, log_offset=1.0):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x) - horizon):
        out[i] = (np.log(x[i + horizon] + log_offset) - np.log(x[i] + log_offset)) / horizon
    return out


# ============================================================
# Plot
# ============================================================
def plot_feasible_region_v2(
    *,
    dates_plot,
    y_plot,
    mu_plot,
    cutoff_to_ref_plot,
    cutoff_to_refdown_plot,
    cutoff_to_band_plot,
    cutoff_to_growth_plot,
    band_width_df,
    out_png,
    out_pdf,
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    dates_plot = pd.to_datetime(np.asarray(dates_plot))
    cutoffs = [1 / 35, 1 / 14, 1 / 7]
    c_fc = {
        1 / 35: "#4C78A8",
        1 / 14: "#B279A2",
        1 / 7: "#F58518",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    axA, axB, axC, axD = axes.ravel()

    # A. Downstream fits.
    axA.scatter(dates_plot, y_plot, s=5, color="0.65", alpha=0.55, linewidths=0, label="Observed cases")
    axA.plot(dates_plot, mu_plot, color="0.15", lw=1.25, label="Smoothed downstream")
    for fc in cutoffs:
        axA.plot(
            dates_plot,
            cutoff_to_refdown_plot[fc],
            lw=1.2,
            color=c_fc[fc],
            label=rf"$D^{{ref}}_{{f_c={cutoff_label(fc)}}}$",
        )
    axA.set_title("Comparable downstream fits")
    axA.set_ylabel("Incidence")
    axA.grid(True, axis="y")
    axA.xaxis.set_major_locator(MonthLocator(interval=2))
    axA.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    axA.legend(frameon=False, loc="upper left", ncol=1)

    # B. Upstream feasible bands.
    ymax = 0.0
    for fc in cutoffs:
        Ufc = np.asarray(cutoff_to_ref_plot[fc], float)
        Bfc = np.asarray(cutoff_to_band_plot[fc], float)
        lo = np.maximum(Ufc - Bfc, 0.0)
        hi = Ufc + Bfc
        ymax = max(ymax, float(np.nanmax(hi)))
        axB.fill_between(dates_plot, lo, hi, color=c_fc[fc], alpha=0.15)
        axB.plot(dates_plot, Ufc, color=c_fc[fc], lw=1.25, label=rf"$U^{{ref}}_{{f_c={cutoff_label(fc)}}}$")
    axB.set_ylim(0, ymax * 1.05)
    axB.set_title("Feasible upstream bands")
    axB.set_ylabel("Incidence")
    axB.grid(True, axis="y")
    axB.xaxis.set_major_locator(MonthLocator(interval=2))
    axB.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    axB.legend(frameon=False, loc="upper left")

    # C. Time-series growth rates.
    for fc in cutoffs:
        axC.plot(
            dates_plot,
            cutoff_to_growth_plot[fc],
            color=c_fc[fc],
            lw=1.25,
            label=rf"$r^{{ref}}_{{f_c={cutoff_label(fc)}}}$",
        )
    axC.axhline(0.0, color="0.55", lw=0.8, ls="--")
    axC.set_title("Short-term growth rates")
    axC.set_ylabel("7-day log growth rate")
    axC.grid(True, axis="y")
    axC.xaxis.set_major_locator(MonthLocator(interval=2))
    axC.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    axC.legend(frameon=False, loc="upper left")

    # D. Feasible-band width as the cutoff is relaxed.
    band_width_df = band_width_df.sort_values("f_cut")
    x = np.arange(len(band_width_df))
    colors = [c_fc[fc] for fc in band_width_df["f_cut"]]
    axD.bar(
        x,
        band_width_df["mean_relative_width"].values,
        color=colors,
        width=0.62,
    )
    axD.plot(
        x,
        band_width_df["mean_relative_width"].values,
        color="0.25",
        lw=1.0,
        marker="o",
        ms=3.0,
    )
    axD.set_xticks(x)
    axD.set_xticklabels([cutoff_label(fc) for fc in band_width_df["f_cut"]])
    axD.set_xlabel(r"Cutoff frequency $f_c$ (cycles/day)")
    axD.set_ylabel("Mean relative band width")
    axD.set_title("Feasible region expands with cutoff")
    axD.grid(True, axis="y")

    for ax, lab in zip([axA, axB, axC, axD], "ABCD"):
        ax.text(-0.12, 1.04, lab, transform=ax.transAxes,
                fontweight="bold", fontsize=11, va="bottom", ha="left")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_feasible_region_v3_single_us(
    *, payload, out_png, out_pdf, figure_label=None
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    cutoffs = [1 / 35, 1 / 14, 1 / 7]
    c_fc = {
        1 / 35: "#4C78A8",
        1 / 14: "#B279A2",
        1 / 7: "#F58518",
    }

    dates_plot = pd.to_datetime(np.asarray(payload["dates_plot"]))
    y_plot = payload["y_plot"]
    mu_plot = payload["mu_plot"]
    cutoff_to_ref_plot = payload["cutoff_to_ref_plot"]
    cutoff_to_refdown_plot = payload["cutoff_to_refdown_plot"]
    cutoff_to_band_plot = payload["cutoff_to_band_plot"]
    cutoff_to_growth_plot = payload["cutoff_to_growth_plot"]
    feature_df = payload["feature_df"].sort_values("f_cut", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(7.6, 4))
    if figure_label:
        fig.suptitle(figure_label, fontsize=11, fontweight="bold", y=0.985)

    gs = fig.add_gridspec(
        2, 2,
        left=0.08, right=0.98, bottom=0.09,
        top=0.91 if figure_label else 0.94,
        wspace=0.28, hspace=0.38,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])

    gsD = gs[1, 1].subgridspec(
        2, 2,
        height_ratios=[0.12, 0.88],
        width_ratios=[1, 1],
        hspace=0.02,
        wspace=0.42,
    )

    axD_title = fig.add_subplot(gsD[0, :])
    axD_left = fig.add_subplot(gsD[1, 0])
    axD_right = fig.add_subplot(gsD[1, 1])

    axD_title.axis("off")
    axD_title.set_title("Peak summaries", fontsize=9.5, pad=0)

    # A. Feasible upstream bands.
    ymax = 0.0
    for fc in cutoffs:
        Ufc = np.asarray(cutoff_to_ref_plot[fc], float)
        Bfc = np.asarray(cutoff_to_band_plot[fc], float)
        lo = np.maximum(Ufc - Bfc, 0.0)
        hi = Ufc + Bfc
        ymax = max(ymax, float(np.nanmax(hi)))

        axA.fill_between(dates_plot, lo, hi, color=c_fc[fc], alpha=0.15)
        axA.plot(
            dates_plot,
            Ufc,
            color=c_fc[fc],
            lw=1.25,
            label=rf"$U^{{ref}}_{{f_c={cutoff_label(fc)}}}$",
        )

    axA.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
    axA.set_title("Admissible upstream reconstructions")
    axA.set_ylabel("Incidence")
    axA.grid(True, axis="y")
    axA.xaxis.set_major_locator(MonthLocator(interval=2))
    axA.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    axA.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axA.yaxis.get_offset_text().set_size(7.8)
    axA.yaxis.set_major_locator(mticker.MaxNLocator(5))
    axA.legend(frameon=False, loc="upper left")

    # B. Comparable downstream fits.
    axB.scatter(
        dates_plot,
        y_plot,
        s=5,
        color="0.65",
        alpha=0.50,
        linewidths=0,
        label="Observed cases",
    )
    axB.plot(
        dates_plot,
        mu_plot,
        color="0.15",
        lw=1.25,
        label="Smoothed downstream",
    )

    for fc in cutoffs:
        axB.plot(
            dates_plot,
            cutoff_to_refdown_plot[fc],
            lw=1.15,
            color=c_fc[fc],
            label=rf"$D^{{ref}}_{{f_c={cutoff_label(fc)}}}$",
        )

    axB.set_title("Comparable downstream fits")
    axB.set_ylabel("Incidence")
    axB.grid(True, axis="y")
    axB.xaxis.set_major_locator(MonthLocator(interval=2))
    axB.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    apply_confirmed_cases_axis(axB, y_plot, offset_text_size=7.8, nbins=5)
    axB.legend(frameon=False, loc="upper left")

    # C. Seven-day growth rates.
    for fc in cutoffs:
        axC.plot(
            dates_plot,
            cutoff_to_growth_plot[fc],
            color=c_fc[fc],
            lw=1.25,
            label=rf"$r^{{ref}}_{{f_c={cutoff_label(fc)}}}$",
        )

    axC.axhline(0.0, color="0.55", lw=0.8, ls="--")
    axC.set_title("Short-term growth-rate estimates")
    axC.set_ylabel("7-day log growth rate")
    axC.set_xlabel("Reference date")
    axC.grid(True, axis="y")
    axC.xaxis.set_major_locator(MonthLocator(interval=2))
    axC.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    axC.legend(frameon=False, loc="lower left")

    # D. Peak timing and peak magnitude.
    y_pos = np.arange(len(feature_df))
    labels = [cutoff_label(fc) for fc in feature_df["f_cut"]]
    colors = [c_fc[fc] for fc in feature_df["f_cut"]]

    rel_timing = feature_df["relative_peak_timing"].values.astype(float)
    rel_magnitude = feature_df["relative_peak_magnitude"].values.astype(float)

    # Left: relative peak timing.
    axD_left.barh(
        y_pos,
        rel_timing,
        color=colors,
        alpha=0.75,
        height=0.58,
    )
    axD_left.invert_xaxis()
    axD_left.set_xlim(1.0, 0.0)
    axD_left.set_xlabel("Relative peak timing")
    axD_left.set_xticks([1.0, 0.5, 0.0])
    axD_left.set_xticklabels(["1", "0.5", "0"])
    axD_left.set_yticks(y_pos)
    axD_left.set_yticklabels(labels)
    axD_left.yaxis.tick_right()
    axD_left.tick_params(
        axis="y",
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        length=0,
        pad=8,
    )
    axD_left.tick_params(axis="y", left=False, labelleft=False)
    axD_left.spines["left"].set_visible(False)
    axD_left.spines["top"].set_visible(False)
    axD_left.spines["right"].set_visible(True)
    axD_left.grid(True, axis="x")

    for yi, val in zip(y_pos, rel_timing):
        axD_left.text(
            val + 0.04,
            yi,
            f"{val:.2f}",
            va="center",
            ha="right",
            fontsize=7.3,
        )

    # Right: relative peak magnitude.
    axD_right.barh(
        y_pos,
        rel_magnitude,
        color=colors,
        alpha=0.75,
        height=0.58,
    )
    axD_right.set_xlim(0.0, 1.05)
    axD_right.set_xlabel("Relative peak magnitude")
    axD_right.set_xticks([0.0, 0.5, 1.0])
    axD_right.set_xticklabels(["0", "0.5", "1"])
    axD_right.set_yticks(y_pos)
    axD_right.set_yticklabels(labels)
    axD_right.yaxis.tick_left()
    axD_right.tick_params(
        axis="y",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        length=0,
        pad=8,
    )
    axD_right.tick_params(axis="y", left=False, labelleft=False)
    axD_right.spines["left"].set_visible(True)
    axD_right.spines["right"].set_visible(False)
    axD_right.spines["top"].set_visible(False)
    axD_right.grid(True, axis="x")

    for yi, val in zip(y_pos, rel_magnitude):
        axD_right.text(
            val + 0.025,
            yi,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=7.3,
        )

    # Panel labels: place all labels in figure coordinates for consistent alignment.
    def add_panel_label(fig, ax, label, dx=-0.035, dy=0.010):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + dx,
            bbox.y1 + dy,
            label,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    add_panel_label(fig, axA, "A")
    add_panel_label(fig, axB, "B")
    add_panel_label(fig, axC, "C")

    # Use the title axis as the visual top of panel D.
    add_panel_label(fig, axD_title, "D")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def run_pipeline():
    START = "2020-04-01"
    END = "2021-06-30"
    DISPLAY_START = "2020-09-01"
    DISPLAY_END = "2021-03-31"

    DF_GRID = [10, 15, 20, 30, 40, 60, 80, 100, 120]
    RSCRIPT_BIN = "Rscript"
    GEOM_RIDGE_THETA = 1e-6
    ALPHA_TOTAL_ERROR = 0.20
    TAU = tau_from_alpha(ALPHA_TOTAL_ERROR)
    CUTOFFS = [1 / 35, 1 / 14, 1 / 7]
    CUTOFF_CURVE = np.linspace(1 / 60, 1 / 5, 90)

    # Empirical COVID-19 onset-to-report delay.
    delay_key = (
        "zhang2020_cn_covid::symptom_onset_to_report::"
        "date_symptom_onset->date_case_report::lognormal"
    )
    try:
        emp_subset, _ = load_empirical_delay_subset()
        if delay_key not in emp_subset:
            raise KeyError(f"{delay_key} not found in empirical support.")
        dist_emp = emp_subset[delay_key].dist
        g = pdf_dist_to_daily_pmf(dist_emp, tau_max=60)
        print(f"Loaded empirical delay: {delay_key}")
    except Exception as e:
        print("Warning: failed to load empirical delay. Falling back to fixed lognormal.")
        print("Reason:", repr(e))
        from scipy.stats import lognorm
        dist_emp = lognorm(s=0.5, scale=5.0)
        g = pdf_dist_to_daily_pmf(dist_emp, tau_max=60)

    # U.S. national COVID-19 confirmed-case incidence.
    set_epidata_api_key()

    from epidatpy import EpiDataContext, EpiRange
    epidata = EpiDataContext(use_cache=False)
    def build_row_payload(*, row_label: str, geo_type: str, geo_values: str):
        print(f"\n=== Building feasible-region row: {row_label} ({geo_type}={geo_values}) ===")
        df = fetch_signal_df(
            epidata,
            data_source="jhu-csse",
            signals="confirmed_incidence_num",
            time_values=EpiRange(START, END),
            geo_type=geo_type,
            geo_values=geo_values,
        )
        aligned = align_daily(df, start=START, end=END)
        dates = aligned["date"].values
        y = aligned["y"].values.astype(float)

        display_mask = (
            (pd.to_datetime(dates) >= pd.Timestamp(DISPLAY_START))
            & (pd.to_datetime(dates) <= pd.Timestamp(DISPLAY_END))
        )
        joint = select_best_df_and_noise(
            y,
            df_grid=DF_GRID,
            ord=2,
            criterion="AIC",
            rscript_bin=RSCRIPT_BIN,
        )
        mu_hat = joint.mu_hat

        cutoff_to_ref = {}
        cutoff_to_refdown = {}
        cutoff_to_band = {}

        T = len(y)
        sigma2_eff = get_effective_noise_scale(mu_hat[display_mask], joint.best_noise)
        print(
            f"{row_label}: best full-support trendfilter df_target={joint.df_target}, "
            f"df_used≈{joint.df_used:.2f}, noise={joint.best_noise.model}, "
            f"window_sigma2_eff={sigma2_eff:.3g}"
        )
        display_scale = max(float(np.mean(mu_hat[display_mask])), 1e-8)
        T_display = int(np.sum(display_mask))
        band_width_rows = []

        for fc in CUTOFFS:
            U_fc, KU_fc, theta_fc, _, _ = compute_cutoff_specific_reference_strict(
                y=y,
                mu_hat=mu_hat,
                g=g,
                f_cut=fc,
                ridge_theta=GEOM_RIDGE_THETA,
            )
            B_fc_display = analytic_band_width_time_domain(
                T=T_display,
                g=g,
                sigma2_eff=sigma2_eff,
                f_cut=fc,
                tau=TAU,
                dt=1.0,
            )
            B_fc = np.full_like(y, np.nan, dtype=float)
            B_fc[display_mask] = B_fc_display

            cutoff_to_ref[fc] = U_fc
            cutoff_to_refdown[fc] = KU_fc
            cutoff_to_band[fc] = B_fc

            rel_width_t = (2.0 * B_fc) / display_scale
            band_width_rows.append({
                "region": row_label,
                "cutoff": cutoff_label(fc),
                "f_cut": fc,
                "period_days": 1.0 / fc,
                "mean_absolute_width": float(np.mean(2.0 * B_fc[display_mask])),
                "mean_relative_width": float(np.mean(rel_width_t[display_mask])),
                "median_relative_width": float(np.median(rel_width_t[display_mask])),
            })

            print(
                f"{row_label}: cutoff={cutoff_label(fc)} ({fc:.5f} cpd), "
                f"basis_terms={len(theta_fc)}, "
                f"mean_window_band_width={np.mean(2 * B_fc_display):.3g}"
            )

        band_width_curve_rows = []
        for fc in CUTOFF_CURVE:
            B_fc_display = analytic_band_width_time_domain(
                T=T_display,
                g=g,
                sigma2_eff=sigma2_eff,
                f_cut=float(fc),
                tau=TAU,
                dt=1.0,
            )
            B_fc = np.full_like(y, np.nan, dtype=float)
            B_fc[display_mask] = B_fc_display
            rel_width_t = (2.0 * B_fc) / display_scale
            band_width_curve_rows.append({
                "region": row_label,
                "f_cut": float(fc),
                "period_days": float(1.0 / fc),
                "mean_absolute_width": float(np.mean(2.0 * B_fc[display_mask])),
                "mean_relative_width": float(np.mean(rel_width_t[display_mask])),
                "median_relative_width": float(np.median(rel_width_t[display_mask])),
            })

        # Short-term growth rates for cutoff-specific reference trajectories.
        cutoff_to_growth = {}
        for fc in CUTOFFS:
            growth = rolling_growth_rate(cutoff_to_ref[fc], horizon=7, log_offset=1.0)
            cutoff_to_growth[fc] = smooth_reflect(growth, window=5)

        # Peak timing and peak magnitude for cutoff-specific reference trajectories.
        feature_rows = []
        PEAK_SMOOTH_WINDOW = 14

        for fc in CUTOFFS:
            Ufc_display = np.asarray(cutoff_to_ref[fc][display_mask], float)
            dates_display = pd.to_datetime(dates[display_mask])

            Ufc_peak = smooth_reflect(Ufc_display, window=PEAK_SMOOTH_WINDOW)

            peak_idx = int(np.nanargmax(Ufc_peak))
            peak_date = dates_display[peak_idx]

            peak_mag = float(Ufc_display[peak_idx])
            peak_mag_smoothed = float(Ufc_peak[peak_idx])

            feature_rows.append({
                "cutoff": cutoff_label(fc),
                "f_cut": fc,
                "period_days": 1.0 / fc,
                "peak_date": peak_date,
                "peak_day_index": peak_idx,
                "peak_magnitude": peak_mag,
                "peak_magnitude_smoothed": peak_mag_smoothed,
            })

        feature_df = pd.DataFrame(feature_rows)

        # Relative peak timing: position of the peak within the displayed time window.
        # 0 = beginning of displayed window; 1 = end of displayed window.
        feature_df["relative_peak_timing"] = (
                feature_df["peak_day_index"] / max(len(dates_display) - 1, 1)
        )

        # Relative peak magnitude: peak magnitude divided by the maximum peak magnitude
        # across the cutoff-specific reconstructions.
        feature_df["relative_peak_magnitude"] = (
                feature_df["peak_magnitude"] / feature_df["peak_magnitude"].max()
        )

        # Slice to plotting window.
        sliced = slice_date_range(
            dates,
            y,
            mu_hat,
            cutoff_to_ref[1 / 35],
            cutoff_to_ref[1 / 14],
            cutoff_to_ref[1 / 7],
            cutoff_to_refdown[1 / 35],
            cutoff_to_refdown[1 / 14],
            cutoff_to_refdown[1 / 7],
            cutoff_to_band[1 / 35],
            cutoff_to_band[1 / 14],
            cutoff_to_band[1 / 7],
            cutoff_to_growth[1 / 35],
            cutoff_to_growth[1 / 14],
            cutoff_to_growth[1 / 7],
            start=DISPLAY_START,
            end=DISPLAY_END,
        )
        (
            dates_plot,
            y_plot,
            mu_plot,
            u_fc_35_plot,
            u_fc_14_plot,
            u_fc_7_plot,
            d_fc_35_plot,
            d_fc_14_plot,
            d_fc_7_plot,
            B_fc_35_plot,
            B_fc_14_plot,
            B_fc_7_plot,
            g_fc_35_plot,
            g_fc_14_plot,
            g_fc_7_plot,
        ) = sliced

        return {
            "row_label": row_label,
            "dates_plot": dates_plot,
            "y_plot": y_plot,
            "mu_plot": mu_plot,
            "cutoff_to_ref_plot": {
                1 / 35: u_fc_35_plot,
                1 / 14: u_fc_14_plot,
                1 / 7: u_fc_7_plot,
            },
            "cutoff_to_refdown_plot": {
                1 / 35: d_fc_35_plot,
                1 / 14: d_fc_14_plot,
                1 / 7: d_fc_7_plot,
            },
            "cutoff_to_band_plot": {
                1 / 35: B_fc_35_plot,
                1 / 14: B_fc_14_plot,
                1 / 7: B_fc_7_plot,
            },
            "cutoff_to_growth_plot": {
                1 / 35: g_fc_35_plot,
                1 / 14: g_fc_14_plot,
                1 / 7: g_fc_7_plot,
            },
            "band_width_df": pd.DataFrame(band_width_rows),
            "band_width_curve_df": pd.DataFrame(band_width_curve_rows),
            "feature_df": feature_df,
        }

    row_payload = build_row_payload(
        row_label="US",
        geo_type="nation",
        geo_values="us",
    )

    band_width_csv = OUT_DIR / "band_width_v3.csv"
    row_payload["band_width_curve_df"].to_csv(band_width_csv, index=False)

    feature_csv = OUT_DIR / "feature_summary_v3.csv"
    row_payload["feature_df"].to_csv(feature_csv, index=False)

    out_png = OUT_DIR / "feasible_region_v3.png"
    out_pdf = OUT_DIR / "feasible_region_v3.pdf"

    plot_feasible_region_v3_single_us(
        payload=row_payload,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    print("Saved figure to:", out_png)
    print("Saved figure to:", out_pdf)
    print("Saved band-width summary to:", band_width_csv)
    print("Saved feature summary to:", feature_csv)


def make_generation_interval_for_features(mean_days=5.0, sd_days=2.0, max_lag=21):
    from scipy.stats import gamma
    lags = np.arange(1, max_lag + 1)
    shape = (mean_days / sd_days) ** 2
    scale = sd_days**2 / mean_days
    edges = np.arange(0.5, max_lag + 1.5, 1.0)
    cdf = gamma.cdf(edges, a=shape, scale=scale)
    pmf = np.diff(cdf)
    pmf = np.clip(pmf, 0, None)
    pmf = pmf / pmf.sum()
    return lags, pmf


if __name__ == "__main__":
    run_pipeline()
