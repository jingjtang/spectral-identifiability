#!/usr/bin/env python3
# ------------------------------------------------------------
# Full script (UPDATED): real-data pipeline + R trendfilter + RMSE-gated U1 via near-null HF perturbation + J fractional plot
# ------------------------------------------------------------
# Keeps (per your request):
#   1) Real data ingestion via epidatpy
#   2) Real delay distributions via your catalog + pdf->daily PMF
#   3) R genlasso trendfilter to estimate downstream latent mean mu_hat
#
# Replaces (per my suggestion):
#   - Baseline upstream reconstruction: convex inverse using cvxpy
#   - Alternative upstream U1: construct a high-frequency near-null perturbation h* (||K h*|| small),
#     then maximize alpha subject to your RMSE gate:
#       RMSE(K U1 - mu_target) <= gamma * noise_scale(residuals)
#     where U1 is refined by a regularized constrained solve toward a prior (U0 + alpha h*).
#   - J plot: uses cumulative fraction cumJ/J (no delay subplots), plus clipped integrand for readability.
# ------------------------------------------------------------

import os
os.environ["MPLBACKEND"] = "Agg"   # or "Agg"

import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy import sparse

# You already have these in your repo
from singlefreq_settings import (
    SINGLEFREQ_EMPIRICAL_MAX_DELAY_DAYS,
    SINGLEFREQ_INTRO_T_END,
    SINGLEFREQ_INTRO_T_START,
    SINGLEFREQ_PMF_TAU_MAX,
)
from utils import (
    apply_confirmed_cases_axis,
    load_local_signal_df,
    load_empirical_delay_subset,
    pdf_dist_to_daily_pmf,
    set_epidata_api_key,
)


# -------------------------
# Paths / config
# -------------------------
FIG_DIR = Path("../figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_epidata_api_key()


# -------------------------
# Data fetch (epidatpy)
# -------------------------
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
            df["date"] = pd.to_datetime(tv.astype(int).astype(str), format="%Y%m%d", errors="raise")
        else:
            df["date"] = pd.to_datetime(tv.astype(str), errors="raise")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    else:
        raise ValueError("Could not find a time column (expected 'time_value' or 'date').")

    if "value" not in df.columns:
        raise ValueError(f"Expected 'value' column; got {df.columns.tolist()}")

    out = df[["date", "value"]].rename(columns={"value": "y"}).sort_values("date")
    out["data_source"] = data_source
    out["signal"] = signals
    out["geo_type"] = geo_type
    out["geo_value"] = geo_values
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


# -------------------------
# Trendfilter via R genlasso
# -------------------------
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
    y: np.ndarray, *, df_target: float, ord: int = 2, rscript_bin: str = "Rscript"
) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, float)

    with tempfile.TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.json")
        out_csv = os.path.join(td, "out.csv")
        r_path = os.path.join(td, "trendfilter.R")

        cfg = {"y": y.tolist(), "df_target": float(df_target), "ord": int(ord)}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with open(r_path, "w") as f:
            f.write(R_TRENDFILTER_SCRIPT)

        cmd = [rscript_bin, r_path, cfg_path, out_csv]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "R trendfilter call failed.\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
            )

        out = pd.read_csv(out_csv)
        mu_hat = out["mu_hat"].values.astype(float)
        df_used = float(out["df_used"].iloc[0])
        lam_used = float(out["lam_used"].iloc[0])

    mu_hat = np.maximum(mu_hat, 1e-8)
    return mu_hat, df_used, lam_used


# -------------------------
# Noise models: Gaussian / Poisson / NegBin (kept)
# -------------------------
@dataclass
class NoiseFit:
    model: str  # "gaussian" | "poisson" | "nb"
    loglik: float
    params: Dict[str, float]  # sigma or phi
    aic: float
    bic: float


def loglik_gaussian(y: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
    resid = y - mu
    sigma2 = np.mean(resid**2)
    sigma2 = max(sigma2, 1e-12)
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
    k = k_mean_df + 0
    fits.append(NoiseFit("poisson", ll_p, {}, -2 * ll_p + 2 * k, -2 * ll_p + np.log(n) * k))

    def obj(log_phi):
        phi = np.exp(log_phi)
        return -loglik_nb_phi(y, mu, phi)

    res = minimize_scalar(obj, bounds=(-8, 12), method="bounded")
    phi_hat = float(np.exp(res.x))
    ll_nb = -float(res.fun)
    k = k_mean_df + 1
    fits.append(NoiseFit("nb", ll_nb, {"phi": phi_hat}, -2 * ll_nb + 2 * k, -2 * ll_nb + np.log(n) * k))
    return fits


# -------------------------
# Delay operator utilities (kept)
# -------------------------
def normalize_pmf(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, float)
    g = np.clip(g, 0.0, None)
    s = g.sum()
    if s <= 0:
        raise ValueError("Delay PMF must have positive sum.")
    return g / s


def build_convolution_matrix(g: np.ndarray, T: int) -> sparse.csr_matrix:
    g = np.asarray(g, float)
    Kmax = len(g) - 1
    rows, cols, vals = [], [], []
    for t in range(T):
        km = min(Kmax, t)
        for k in range(km + 1):
            rows.append(t)
            cols.append(t - k)
            vals.append(g[k])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(T, T)).tocsr()


def convolve_mean(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return fftconvolve(x, g, mode="full")[: len(x)]


def second_difference_matrix(T: int) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(T - 2):
        rows += [i, i, i]
        cols += [i, i + 1, i + 2]
        vals += [1.0, -2.0, 1.0]
    return sparse.coo_matrix((vals, (rows, cols)), shape=(T - 2, T)).tocsr()


# -------------------------
# New: Baseline upstream reconstruction via convex inverse (cvxpy)
# -------------------------
def reconstruct_upstream_cvxpy(
    y_or_mu: np.ndarray,
    g: np.ndarray,
    *,
    model: str,
    lam_smooth: float = 50.0,
    eps: float = 1e-8,
    solver_gauss: str = "OSQP",
    solver_log: str = "SCS",
) -> np.ndarray:
    """
    If model == "gaussian": minimize ||(Kx - target)||^2 + lam_smooth ||D1 x||_1, x>=0
    If model == "poisson":  minimize sum(m - y log m) + lam_smooth ||D1 x||_1, x>=0, m=Kx>=eps
    Here target = mu_hat (recommended).
    """
    import cvxpy as cp

    target = np.asarray(y_or_mu, float)
    T = len(target)
    g = normalize_pmf(g)
    K = build_convolution_matrix(g, T)
    D1 = second_difference_matrix(T)

    x = cp.Variable(T)
    m = K @ x

    if model == "poisson":
        # assume target is observed y for Poisson; but you can also pass mu_hat and treat as pseudo-y
        y = np.asarray(y_or_mu, float)
        y = np.clip(y, 0.0, None)
        negloglik = cp.sum(m - cp.multiply(y, cp.log(m)))
        obj = negloglik + lam_smooth * cp.norm1(D1 @ x)
        constraints = [x >= 0, m >= eps]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=solver_log, verbose=False)
    else:
        # Least-squares inverse
        obj = cp.sum_squares(m - target) + lam_smooth * cp.norm1(D1 @ x)
        constraints = [x >= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=solver_gauss, verbose=False)

    if x.value is None:
        raise RuntimeError("cvxpy solve failed (x.value is None). Try changing solver or loosening lam_smooth.")
    return np.maximum(np.array(x.value).reshape(-1), 0.0)


# -------------------------
# New: Near-null high-frequency perturbation h* + RMSE-gated U1
# -------------------------
def project_nonneg(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def robust_noise_scale(resid: np.ndarray) -> float:
    resid = np.asarray(resid, float)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    return float(1.4826 * mad + 1e-12)


def sample_targeted_basis_bank(
    T: int,
    *,
    center: int,
    width: float = 28.0,
    target_freqs: Optional[List[float]] = None,
    freq_jitter: float = 0.01,
    n_freq_each: int = 5,
    phase_grid: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """
    Build a deterministic bank of localized oscillatory bases concentrated near
    the target frequencies (default: 1/7 and 1/3 cycles/day).

    Each basis has the form
        h(t) = w(t) * [a1 sin(2π f1 t + φ1) + a2 sin(2π f2 t + φ2)]
    followed by centering and L2 normalization.

    This is MUCH better aligned with your figure goal than random candidates.
    """
    t = np.arange(T, dtype=float)

    if target_freqs is None:
        target_freqs = [1 / 7, 1 / 3]

    if phase_grid is None:
        phase_grid = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    w = np.ones(T, dtype=float)

    bank = []

    def _normalize(h):
        h = np.asarray(h, float)
        nrm = np.linalg.norm(h)
        if nrm < 1e-12:
            return None
        return h / nrm

    # small frequency grids around each target
    freq_grids = []
    for f0 in target_freqs:
        freq_grids.append(np.linspace(f0 - freq_jitter, f0 + freq_jitter, n_freq_each))

    # 1) single-band components
    for fg in freq_grids:
        for f in fg:
            for ph in phase_grid:
                h = w * np.sin(2 * np.pi * f * t + ph)
                h = _normalize(h)
                if h is not None:
                    bank.append(h)

                h = w * np.cos(2 * np.pi * f * t + ph)
                h = _normalize(h)
                if h is not None:
                    bank.append(h)

    # 2) two-band mixtures emphasizing both 1/7 and 1/3
    if len(freq_grids) >= 2:
        fg1, fg2 = freq_grids[0], freq_grids[1]
        mix_weights = [0.5, 1.0, 2.0]

        for f1 in fg1:
            for f2 in fg2:
                for ph1 in phase_grid:
                    for ph2 in phase_grid:
                        for c in mix_weights:
                            h = w * (
                                np.sin(2 * np.pi * f1 * t + ph1)
                                + c * np.sin(2 * np.pi * f2 * t + ph2)
                            )
                            h = _normalize(h)
                            if h is not None:
                                bank.append(h)

                            h = w * (
                                np.cos(2 * np.pi * f1 * t + ph1)
                                + c * np.cos(2 * np.pi * f2 * t + ph2)
                            )
                            h = _normalize(h)
                            if h is not None:
                                bank.append(h)

    return bank


def construct_U1_RMSE_gated_nearnull(
    U0: np.ndarray,
    g: np.ndarray,
    *,
    y: np.ndarray,
    mu_hat: np.ndarray,
    gamma: float = 1.0,
    seed: int = 0,   # kept only for API compatibility
    alpha_hi: float = 2e5,
    n_bisect: int = 40,
    lam_prior: float = 0.0,      # kept only for API compatibility
    lam_smooth_l2: float = 0.0,  # kept only for API compatibility
    target_freqs: Optional[List[float]] = None,
    freq_jitter: float = 0.01,
    basis_width: float = 28.0,
    eta_y: float = 1.05,         # NEW: KU1-to-y fit must remain close to KU0-to-y
    eval_mask: Optional[np.ndarray] = None,
):
    """
    Construct U1 = max(U0 + alpha * h, 0), where h is a localized oscillatory basis.

    Feasibility constraints on the evaluation window:
        RMSE(KU1, KU0) <= gamma * RMSE(KU0, y)
        RMSE(KU1, y)   <= eta_y * RMSE(KU0, y)

    Selection objective:
        maximize visible time-domain separation between U1 and U0 in the peak window,
        while preferring energy in target frequency bands.
    """
    U0 = np.asarray(U0, float)
    y = np.asarray(y, float)
    mu_hat = np.asarray(mu_hat, float)
    g = normalize_pmf(g)

    T = len(U0)
    mu0 = convolve_mean(U0, g)
    if eval_mask is None:
        eval_mask = np.ones(T, dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        if eval_mask.shape != (T,):
            raise ValueError("eval_mask must have the same length as U0.")
        if not np.any(eval_mask):
            raise ValueError("eval_mask must contain at least one True value.")

    baseline_rmse = float(np.sqrt(np.mean((mu0[eval_mask] - y[eval_mask]) ** 2)))
    target_rmse = float(gamma) * baseline_rmse

    # ------------------------------------------------------------------
    # Choose an interior center from mu_hat, not from full-series argmax(U0),
    # to avoid boundary artifacts dominating the perturbation location.
    # ------------------------------------------------------------------
    L = len(g)
    lo = max(L, 30)
    hi = T - max(L, 30)

    if hi <= lo:
        peak_idx = int(np.argmax(mu_hat))
    else:
        peak_idx = lo + int(np.argmax(mu_hat[lo:hi]))

    w0 = max(0, peak_idx - 45)
    w1 = min(T, peak_idx + 45)

    peak_scale = float(np.quantile(U0[w0:w1], 0.90))
    peak_scale = max(peak_scale, 1e-12)

    # ------------------------------------------------------------------
    # Build a deterministic localized oscillatory bank, but WITHOUT
    # global mean-centering (that was causing the fake flat offset).
    # ------------------------------------------------------------------
    t = np.arange(T, dtype=float)
    if target_freqs is None:
        target_freqs = [1 / 7, 1 / 3]

    phase_grid = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    mix_weights = [0.5, 1.0, 2.0]
    envelope = np.ones(T, dtype=float)

    def normalize_no_center(h: np.ndarray) -> Optional[np.ndarray]:
        h = np.asarray(h, float)
        nrm = np.linalg.norm(h)
        if nrm < 1e-12:
            return None
        return h / nrm

    freq_grids = []
    for f0 in target_freqs:
        freq_grids.append(np.linspace(f0 - freq_jitter, f0 + freq_jitter, 5))

    bank: List[np.ndarray] = []

    # ------------------------------------------------------------
    # Force every candidate basis to include BOTH target frequencies.
    # For your current use case, target_freqs = [1/7, 1/3].
    # Each h is a 2-band mixture.
    # ------------------------------------------------------------
    if len(freq_grids) < 2:
        raise ValueError(
            "This version expects at least 2 target frequencies. "
            f"Got {len(freq_grids)} from target_freqs={target_freqs}"
        )

    fg1, fg2 = freq_grids[0], freq_grids[1]

    amp_pairs = [
        (1.0, 1.0),
        (1.5, 1.0),
        (1.0, 1.5),
        (1.2, 0.7),
        (0.7, 1.2),
    ]

    phase_grid_small = [0.0, np.pi / 2, np.pi]

    def add_if_valid(h):
        h = normalize_no_center(h)
        if h is not None:
            bank.append(h)

    for f1 in fg1:
        for f2 in fg2:
            for a1, a2 in amp_pairs:
                for ph1 in phase_grid_small:
                    for ph2 in phase_grid_small:
                        # all-sine mixture
                        h = envelope * (
                                a1 * np.sin(2 * np.pi * f1 * t + ph1)
                                + a2 * np.sin(2 * np.pi * f2 * t + ph2)
                        )
                        add_if_valid(h)

                        # all-cosine mixture
                        h = envelope * (
                                a1 * np.cos(2 * np.pi * f1 * t + ph1)
                                + a2 * np.cos(2 * np.pi * f2 * t + ph2)
                        )
                        add_if_valid(h)

                        # mixed trig version 1
                        h = envelope * (
                                a1 * np.sin(2 * np.pi * f1 * t + ph1)
                                + a2 * np.cos(2 * np.pi * f2 * t + ph2)
                        )
                        add_if_valid(h)

                        # mixed trig version 2
                        h = envelope * (
                                a1 * np.cos(2 * np.pi * f1 * t + ph1)
                                + a2 * np.sin(2 * np.pi * f2 * t + ph2)
                        )
                        add_if_valid(h)

    bank = bank + [-h for h in bank]

    def band_score(d: np.ndarray) -> float:
        d = np.asarray(d, float)
        D = np.fft.rfft(d - np.mean(d))
        f = np.fft.rfftfreq(len(d), d=1.0)
        P = np.abs(D) ** 2

        def band(lo_, hi_):
            m = (f >= lo_) & (f <= hi_)
            return float(P[m].sum())

        p_17 = band(1 / 7 - 0.03, 1 / 7 + 0.03)
        p_13 = band(1 / 3 - 0.04, 1 / 3 + 0.04)

        p_total = float(P.sum())
        return (p_17 + p_13) / (p_total + 1e-12)

    def build_from_alpha(h: np.ndarray, alpha: float):
        U1 = U0 + alpha * h
        U1 = np.maximum(U1, 0.0)

        mu1 = convolve_mean(U1, g)

        rmse_KU1_KU0 = float(np.sqrt(np.mean((mu1[eval_mask] - mu0[eval_mask]) ** 2)))
        rmse_KU1_y = float(np.sqrt(np.mean((mu1[eval_mask] - y[eval_mask]) ** 2)))

        d = U1 - U0

        dabs = np.abs(d)
        vis_amp_q95 = float(np.quantile(dabs, 0.95))
        vis_amp_max = float(np.max(dabs))
        vis_rel = vis_amp_q95 / (np.quantile(U0, 0.90) + 1e-12)

        vis_mean_abs = float(np.mean(dabs))
        vis_rms = float(np.sqrt(np.mean(d ** 2)))


        relL1 = float(np.linalg.norm(d, 1) / (np.linalg.norm(U0, 1) + 1e-12))
        relL2 = float(np.linalg.norm(d, 2) / (np.linalg.norm(U0, 2) + 1e-12))



        return {
            "U1": U1,
            "mu1": mu1,
            "rmse_KU1_KU0": rmse_KU1_KU0,
            "rmse_KU1_y": rmse_KU1_y,
            "relL1": relL1,
            "relL2": relL2,
            "vis_amp_q95": vis_amp_q95,
            "vis_amp_max": vis_amp_max,
            "vis_mean_abs": vis_mean_abs,
            "vis_rms": vis_rms,
            "vis_rel": vis_rel,
            "band_score": band_score(d),
        }

    def feasible(out: dict) -> bool:
        return (
            out["rmse_KU1_KU0"] <= target_rmse
            and out["rmse_KU1_y"] <= eta_y * baseline_rmse
        )

    def eval_alpha_for_h(h: np.ndarray):
        lo_a, hi_a = 0.0, float(alpha_hi)
        out0 = build_from_alpha(h, 0.0)
        best_alpha = 0.0
        best_out = out0

        # alpha_hi is a visualization cap: it prevents the RMSE gate from
        # choosing extremely large, clipped perturbations whose spectra are
        # dominated by clipping artifacts rather than the target frequencies.
        out_hi = build_from_alpha(h, hi_a)
        if feasible(out_hi):
            return hi_a, out_hi

        # Push alpha to the feasibility boundary.
        for _ in range(n_bisect):
            mid = 0.5 * (lo_a + hi_a)
            out_mid = build_from_alpha(h, mid)

            if feasible(out_mid):
                best_alpha = mid
                best_out = out_mid
                lo_a = mid
            else:
                hi_a = mid

        return best_alpha, best_out

    best_global = None
    best_score = -np.inf

    for h in bank:
        alpha_star, out = eval_alpha_for_h(h)
        score = (
                4.0 * out["vis_mean_abs"]
                + 3.0 * out["vis_rms"]
                + 1.5 * out["vis_amp_q95"]
                + 1.0 * np.log1p(out["band_score"])
        )

        if score > best_score:
            best_score = score
            best_global = (alpha_star, h, out)

    alpha_star, h_best, out = best_global

    # diagnostic: how much energy of the selected perturbation falls into each target band?
    d_best = out["U1"] - U0
    D_best = np.fft.rfft(d_best - np.mean(d_best))
    f_best = np.fft.rfftfreq(len(d_best), d=1.0)
    P_best = np.abs(D_best) ** 2

    def _band_energy(fr, halfwidth):
        m = (f_best >= fr - halfwidth) & (f_best <= fr + halfwidth)
        return float(P_best[m].sum())

    e_17 = _band_energy(1 / 7, 0.03)
    e_13 = _band_energy(1 / 3, 0.04)
    e_tot = float(P_best.sum())

    print("Selected perturbation band energies:")
    print(f"  1/7 : {e_17:.4g}   frac={e_17 / (e_tot + 1e-12):.4f}")
    print(f"  1/3 : {e_13:.4g}   frac={e_13 / (e_tot + 1e-12):.4f}")

    metrics = {
        "gamma": float(gamma),
        "eta_y": float(eta_y),
        "baseline_rmse_KU0_y": float(baseline_rmse),
        "target_rmse_KU1_KU0": float(target_rmse),
        "alpha_star": float(alpha_star),
        "rmse_KU1_KU0": float(out["rmse_KU1_KU0"]),
        "rmse_KU1_y": float(out["rmse_KU1_y"]),
        "upstream_rel_L1": float(out["relL1"]),
        "upstream_rel_L2": float(out["relL2"]),
        "visible_amp_peak_window_q95": float(out["vis_amp_q95"]),
        "visible_amp_peak_window_max": float(out["vis_amp_max"]),
        "visible_mean_abs_peak_window": float(out["vis_mean_abs"]),
        "visible_rms_peak_window": float(out["vis_rms"]),
        "visible_rel_peak_window": float(out["vis_rel"]),
        "band_score": float(out["band_score"]),
        "peak_idx": int(peak_idx),
        "peak_window": (int(w0), int(w1)),
        "basis_width": float(basis_width),
        "target_freqs": list(target_freqs),
        "freq_jitter": float(freq_jitter),
        "bank_size": int(len(bank)),
    }

    return out["U1"], out["mu1"], metrics


# -------------------------
# Spectral/J functional utilities (kept, but cleaned + uses cumJ/J)
# -------------------------
def rfft_freqs(T: int, dt: float = 1.0):
    f = np.fft.rfftfreq(T, d=dt)
    df = f[1] - f[0] if len(f) > 1 else 1.0 / (T * dt)
    return f, df


def one_sided_rfft_weights(T: int) -> np.ndarray:
    weights = np.ones(T // 2 + 1, dtype=float)
    if T % 2 == 0:
        if len(weights) > 2:
            weights[1:-1] = 2.0
    else:
        if len(weights) > 1:
            weights[1:] = 2.0
    return weights


def pad_kernel_to_T(g: np.ndarray, T: int):
    g = np.asarray(g, float)
    out = np.zeros(T, dtype=float)
    out[: min(len(g), T)] = g[: min(len(g), T)]
    return out


def attenuation_A2(g: np.ndarray, T: int):
    gT = pad_kernel_to_T(g, T)
    G = np.fft.rfft(gT)
    return (np.abs(G) ** 2), G


def estimate_gaussian_noise_psd(resid: np.ndarray, eps: float = 1e-12):
    resid = np.asarray(resid, float)
    T = len(resid)
    E = np.fft.rfft(resid - np.mean(resid))
    S = (np.abs(E) ** 2) #/ max(T, 1)
    return np.maximum(S, eps)


def poisson_nb_effective_noise_scale(m0: np.ndarray, m1: np.ndarray, noise: NoiseFit):
    m0 = np.asarray(m0, float)
    m1 = np.asarray(m1, float)
    m_min = float(min(np.min(m0), np.min(m1)))
    m_min_safe = max(m_min, 1e-12)

    # The Poisson and negative-binomial KL upper bounds use the minimum
    # candidate downstream mean over the observed window. Using m_max would not
    # justify the conservative testing-error bound in the Methods.
    if noise.model == "poisson":
        return m_min_safe, m_min
    if noise.model == "nb":
        kappa = max(float(noise.params.get("phi", 1.0)), 1e-12)
        return m_min_safe + m_min_safe ** 2 / kappa, m_min
    raise ValueError(f"Expected poisson or nb noise model, got {noise.model}")


def cumulative_from_integrand(f, integrand):
    return np.cumsum(integrand)


def compute_J_and_integrand(
    U0: np.ndarray,
    U1: np.ndarray,
    g: np.ndarray,
    *,
    y: np.ndarray,
    mu_hat: np.ndarray,
    noise: NoiseFit,
    p: float = 1.0,
    eval_mask: Optional[np.ndarray] = None,
):
    U0 = np.asarray(U0, float)
    U1 = np.asarray(U1, float)
    g = normalize_pmf(g)
    y = np.asarray(y, float)
    mu_hat = np.asarray(mu_hat, float)

    T_full = len(U0)
    if eval_mask is None:
        eval_mask = np.ones(T_full, dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        if eval_mask.shape != (T_full,):
            raise ValueError("eval_mask must have the same length as U0.")
        if not np.any(eval_mask):
            raise ValueError("eval_mask must contain at least one True value.")

    U0_eval = U0[eval_mask]
    U1_eval = U1[eval_mask]
    y_eval = y[eval_mask]
    mu_hat_eval = mu_hat[eval_mask]
    T = len(U0_eval)

    f, df = rfft_freqs(T, dt=1.0)
    A2, _ = attenuation_A2(g, T)

    delta_u_fft = np.fft.rfft(U0_eval - U1_eval)
    weights = one_sided_rfft_weights(T)
    one_sided_scale = weights / T
    delta_u_power = one_sided_scale * np.abs(delta_u_fft) ** 2

    if noise.model == "gaussian":
        resid = y_eval - mu_hat_eval
        S = one_sided_scale * estimate_gaussian_noise_psd(resid)
        integrand = (p**2) * A2 * delta_u_power / S
        J_obs = None
    elif noise.model == "poisson":
        # Convolve full trajectories first, then evaluate only the observation
        # window to avoid boundary artifacts from convolving cropped series.
        m1 = p * convolve_mean(U0, g)[eval_mask]
        m2 = p * convolve_mean(U1, g)[eval_mask]
        noise_scale, _ = poisson_nb_effective_noise_scale(m1, m2, noise)
        J_obs = float(np.sum((m2 - m1) ** 2) / max(noise_scale, 1e-12))

        integrand = (
                (p ** 2) * A2 * delta_u_power
                / max(noise_scale, 1e-12)
        )
    elif noise.model == "nb":
        m1 = p * convolve_mean(U0, g)[eval_mask]
        m2 = p * convolve_mean(U1, g)[eval_mask]
        noise_scale, _ = poisson_nb_effective_noise_scale(m1, m2, noise)
        J_obs = float(np.sum((m2 - m1) ** 2) / max(noise_scale, 1e-12))

        integrand = (
                (p ** 2) * A2 * delta_u_power
                / max(noise_scale, 1e-12)
        )
    else:
        raise ValueError(f"Unknown noise model: {noise.model}")

    cumJ = cumulative_from_integrand(f, integrand)
    spectral_sum = float(cumJ[-1]) if len(cumJ) else float(np.sum(integrand) * df)
    Jtot = spectral_sum if J_obs is None else J_obs

    if spectral_sum > 0:
        idx90 = int(np.searchsorted(cumJ, 0.9 * spectral_sum))
        f90 = float(f[min(idx90, len(f) - 1)])
    else:
        f90 = float("nan")

    return {
        "f": f,
        "df": df,
        "A2": A2,
        "delta_u_power": delta_u_power,
        "integrand": integrand,
        "cumJ": cumJ,
        "J": Jtot,
        "spectral_sum": spectral_sum,
        "f0.9": f90,
    }

def plot_intro_figure(
    dates,
    U0,
    U1,
    g,
    y,
    mu_hat,
    noise: NoiseFit,
    *,
    p: float = 1.0,
    fig_name: Optional[str] = None,
    title: str = "",
    f_max: float = 0.5,
    t_start: str = SINGLEFREQ_INTRO_T_START,
    t_end: str = SINGLEFREQ_INTRO_T_END,
    show: bool = True,
):
    """
    Intro figure preserving the plotting conventions and spectral
    definitions used in the previous standalone small figures.

    Row 1:
        A: U0 and U1
        B: delay distribution g
        C: D0, D1, and observed data

    Row 2:
        D: |Delta U_hat(f)|^2
        E: |G_hat(f)|^2
        F: |Delta U_hat(f)|^2 |G_hat(f)|^2 and noise level
    """
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from matplotlib.dates import MonthLocator, DateFormatter

    # ------------------------------------------------------------
    # Preserve the previous small-figure typography
    # ------------------------------------------------------------
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    dates = pd.to_datetime(np.asarray(dates))
    U0 = np.asarray(U0, float)
    U1 = np.asarray(U1, float)
    y = np.asarray(y, float)
    mu_hat = np.asarray(mu_hat, float)
    g = normalize_pmf(g)

    eps = 1e-12

    if not (
        len(dates)
        == len(U0)
        == len(U1)
        == len(y)
        == len(mu_hat)
    ):
        raise ValueError(
            "dates, U0, U1, y, and mu_hat must have equal lengths."
        )

    # ------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------
    t0 = pd.to_datetime(t_start)
    t1 = pd.to_datetime(t_end)
    mt = (dates >= t0) & (dates <= t1)

    if not np.any(mt):
        raise ValueError(
            f"No data fall within {t_start} to {t_end}."
        )

    dates_w = dates[mt]
    U0_w = U0[mt]
    U1_w = U1[mt]
    y_w = y[mt]
    mu_hat_w = mu_hat[mt]

    # Full convolution first, then select the plotting window.
    D0 = p * convolve_mean(U0, g)
    D1 = p * convolve_mean(U1, g)

    D0_w = D0[mt]
    D1_w = D1[mt]

    resid = y - mu_hat
    resid_w = resid[mt]

    # ------------------------------------------------------------
    # Frequency-domain quantities.
    # Note: the spectrum is computed after slicing to the displayed date
    # window. It should be interpreted as a local display-window contrast
    # spectrum, not as a full fitted-support spectrum.
    # One-sided rFFT bins combine positive and negative frequencies.
    # ------------------------------------------------------------
    Tw = len(U0_w)

    f, _ = rfft_freqs(Tw, dt=1.0)
    A2, _ = attenuation_A2(g, Tw)

    delta_u_fft = np.fft.rfft(U0_w - U1_w)
    one_sided_scale = one_sided_rfft_weights(Tw) / Tw
    delta_u_power = one_sided_scale * np.abs(delta_u_fft) ** 2

    mf = (f >= 0.0) & (f <= f_max)

    f_m = f[mf]
    A2_m = A2[mf]
    delta_u_power_m = delta_u_power[mf]

    filtered_delta_power_m = (
        (p ** 2) * delta_u_power_m * A2_m
    )

    # Preserve the previous noise-level construction
    # ------------------------------------------------------------
    if noise.model == "gaussian":
        sigma = float(noise.params["sigma"])
        noise_level = float(2.0 * sigma ** 2)
        noise_label = "One-sided noise level"

    elif noise.model == "poisson":
        m1 = D0_w
        m2 = D1_w
        noise_scale, _ = poisson_nb_effective_noise_scale(m1, m2, noise)
        noise_level = float(2.0 * noise_scale)
        noise_label = r"One-sided noise level"

    elif noise.model == "nb":
        m1 = D0_w
        m2 = D1_w
        noise_scale, _ = poisson_nb_effective_noise_scale(m1, m2, noise)
        noise_level = float(2.0 * noise_scale)
        noise_label = "One-sided noise level"

    else:
        raise ValueError(
            f"Unknown noise model: {noise.model}"
        )

    noise_level = max(float(noise_level), eps)

    # ------------------------------------------------------------
    # Colors from the previous figures
    # ------------------------------------------------------------
    c_u0 = "tab:red"
    c_u1 = "tab:blue"
    c_delay = "tab:green"
    c_delta = "tab:orange"
    c_noise = "0.35"
    c_observed = "k"
    c_filtered = "k"

    # ------------------------------------------------------------
    # Figure geometry
    #
    # The relative column widths preserve the previous design:
    # broad time/frequency panels at left and right, narrow delay
    # panels in the center.
    # ------------------------------------------------------------
    fig = plt.figure(
        figsize=(8.2, 3.8),
        constrained_layout=False,
    )

    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[4.0, 2.0, 4.0],
        left=0.085,
        right=0.99,
        bottom=0.08,
        top=0.90,
        wspace=0.30,
        hspace=0.4,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # Column and row headers for the conceptual grid.
    header_fontsize = 11
    row_header_fontsize = 10
    col_axes = [axA, axB, axC]
    col_titles = ["Upstream", "Delay", "Downstream"]
    for ax, label in zip(col_axes, col_titles):
        bb = ax.get_position()
        fig.text(
            0.5 * (bb.x0 + bb.x1),
            0.945,
            label,
            ha="center",
            va="bottom",
            fontsize=header_fontsize,
            fontweight="bold",
        )

    for row_axes, label in [
        ([axA, axB, axC], "Time Domain"),
        ([axD, axE, axF], "Frequency Domain"),
    ]:
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        fig.text(
            0.018,
            0.5 * (y0 + y1),
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=row_header_fontsize,
            fontweight="bold",
        )

    # ============================================================
    # A: upstream trajectories
    # ============================================================
    axA.plot(
        dates_w,
        U0_w,
        label=r"$U_0(t)$",
        lw=1.0,
        color=c_u0,
        alpha=1.0,
        zorder=3,
    )
    axA.plot(
        dates_w,
        U1_w,
        label=r"$U_1(t)$",
        lw=1.0,
        color=c_u1,
        alpha=1.0,
        zorder=2,
    )

    axA.set_ylabel("Incidence", fontsize=10)
    axA.set_xlabel("Reference date", fontsize=10)
    axA.set_xlim(t0, t1)

    axA.xaxis.set_major_locator(
        MonthLocator(interval=2)
    )
    axA.xaxis.set_major_formatter(
        DateFormatter("%Y%m")
    )

    axA.grid(alpha=0.25)

    axA.legend(
        ncol=1,
        loc="upper left",
        frameon=False,
        handlelength=2.2,
        columnspacing=0.9,
        fontsize=10
    )

    # Scientific notation exactly as in the standalone helper.
    axA.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    axA.yaxis.get_offset_text().set_size(8)
    axA.yaxis.set_major_locator(
        mticker.MaxNLocator(5)
    )

    # ============================================================
    # B: delay distribution
    # ============================================================
    tau = np.arange(len(g))

    axB.plot(
        tau,
        g,
        lw=1.6,
        color=c_delay,
    )

    axB.set_xlabel(r"Delay", fontsize=10)
    axB.set_ylabel("Density", fontsize=10)

    delay_ticks = [
        value
        for value in [0, 7, 14, 28, 49]
        if value <= len(g) - 1
    ]

    axB.set_xticks(delay_ticks)
    axB.set_xticklabels(
        [str(value) for value in delay_ticks]
    )
    axB.set_xlim(0, min(56, len(g) - 1))
    axB.grid(alpha=0.25)

    # ============================================================
    # C: downstream trajectories and observed data
    # ============================================================
    axC.plot(
        dates_w,
        D0_w,
        label=r"$D_0(t)$",
        lw=1.0,
        color=c_u0,
        alpha=0.5,
        zorder=5,
    )
    axC.plot(
        dates_w,
        D1_w,
        label=r"$D_1(t)$",
        lw=1.2,
        color=c_u1,
        alpha=0.5,
        zorder=4,
    )
    axC.plot(
        dates_w,
        y_w,
        label="Observed",
        lw=1.0,
        color=c_observed,
        ls=":",
        alpha=0.9,
        zorder=3,
    )

    axC.set_ylabel("Incidence", fontsize=10)
    axC.set_xlabel("Reference date", fontsize=10)
    axC.set_xlim(t0, t1)

    axC.xaxis.set_major_locator(
        MonthLocator(interval=2)
    )
    axC.xaxis.set_major_formatter(
        DateFormatter("%Y%m")
    )

    axC.grid(alpha=0.25)

    axC.legend(
        ncol=1,
        loc="upper left",
        frameon=False,
        handlelength=2.2,
        columnspacing=0.9,
        fontsize=10
    )

    # Shared incidence axis for panels A and C. The upstream trajectories and
    # downstream/observed trajectories use the same y-range and tick locations
    # so their magnitudes can be compared directly.
    y_all_time = np.concatenate([
        U0_w,
        U1_w,
        D0_w,
        D1_w,
        y_w,
    ])

    y_all_time = y_all_time[
        np.isfinite(y_all_time)
    ]

    if y_all_time.size > 0:
        ymax_t = float(np.nanmax(y_all_time))
        upper = ymax_t * 1.08 if ymax_t > 0 else 1.0
        shared_ticks = mticker.MaxNLocator(5).tick_values(0.0, upper)
        shared_upper = float(np.nanmax(shared_ticks))

        for ax in (axA, axC):
            ax.set_ylim(0.0, shared_upper)
            ax.set_yticks(shared_ticks)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_size(8)

    # ============================================================
    # D: upstream spectral difference
    # ============================================================
    axD.plot(
        f_m,
        delta_u_power_m + eps,
        color=c_delta,
        lw=1.2,
        alpha=1.0,
        zorder=3,
    )

    axD.set_xlim(0, f_max)
    axD.set_yscale("log")
    axD.yaxis.set_minor_locator(
        mticker.NullLocator()
    )
    axD.xaxis.set_minor_locator(
        mticker.NullLocator()
    )

    axD.set_xlabel(
        r"Frequency (day$^{-1}$)", fontsize=10
    )
    axD.set_ylabel(
        "Upstream contrast", fontsize=10
    )
    axD.grid(alpha=0.25)

    # ============================================================
    # E: delay squared magnitude
    # ============================================================
    axE.plot(
        f_m,
        A2_m + eps,
        lw=1.6,
        color=c_delay,
    )

    axE.set_xlim(0, f_max)
    axE.set_yscale("log")
    axE.yaxis.set_minor_locator(
        mticker.NullLocator()
    )
    axE.xaxis.set_minor_locator(
        mticker.NullLocator()
    )
    axE.xaxis.set_major_locator(
        mticker.MultipleLocator(0.1)
    )

    axE.set_xlabel(
        r"Frequency (day$^{-1}$)", fontsize=10
    )
    axE.set_ylabel(
        "Squared magnitude", fontsize=10
    )
    axE.grid(alpha=0.25)

    # ============================================================
    # F: filtered difference and noise level
    # ============================================================
    axF.plot(
        f_m,
        filtered_delta_power_m + eps,
        color=c_filtered,
        lw=1.2,
        alpha=1.0,
        zorder=3,
    )

    axF.plot(
        f_m,
        np.full_like(f_m, noise_level),
        color=c_noise,
        lw=1.0,
        ls="--",
        alpha=1.0,
    )

    axF.set_xlim(0, f_max)
    axF.set_yscale("log")
    axF.yaxis.set_minor_locator(
        mticker.NullLocator()
    )
    axF.xaxis.set_minor_locator(
        mticker.NullLocator()
    )

    axF.set_xlabel(
        r"Frequency (day$^{-1}$)", fontsize=10
    )
    axF.set_ylabel(
        "Filtered contrast", fontsize=10
    )
    axF.grid(alpha=0.25)

    axF.text(
        0.98 * f_max,
        noise_level * 0.98,
        noise_label,
        ha="right",
        va="bottom",
        fontsize=9,
        color=c_noise,
    )

    # ------------------------------------------------------------
    # Preserve the shared D/F y-axis logic from the old small plots
    # ------------------------------------------------------------
    y_DF_shared = np.concatenate([
        delta_u_power_m[
            np.isfinite(delta_u_power_m)
            & (delta_u_power_m > 0)
        ],
        filtered_delta_power_m[
            np.isfinite(filtered_delta_power_m)
            & (filtered_delta_power_m > 0)
        ],
        (
            np.array([noise_level])
            if noise_level > 0
            else np.array([])
        ),
    ])

    if y_DF_shared.size > 0:
        ymin_DF = float(
            np.quantile(y_DF_shared, 0.02)
        )
        ymax_DF = float(
            np.quantile(y_DF_shared, 0.995)
        )

        ylim_DF_shared = (
            ymin_DF,
            ymax_DF * 2.5,
        )

        axD.set_ylim(*ylim_DF_shared)
        axF.set_ylim(*ylim_DF_shared)

    # E keeps its own vertical scale, as before.
    positive_A2 = A2_m[
        np.isfinite(A2_m)
        & (A2_m > 0)
    ]

    if positive_A2.size > 0:
        ymin_E = float(
            np.quantile(positive_A2, 0.02)
        )
        ymax_E = float(
            np.quantile(positive_A2, 0.995)
        )

        axE.set_ylim(
            ymin_E,
            ymax_E * 2.5,
        )

    # ------------------------------------------------------------
    # Target-frequency annotations, preserving old appearance
    # ------------------------------------------------------------
    target_freqs = [
        (1 / 7, "7-day"),
        (1 / 3, "3-day"),
    ]

    for ax in [axD, axF]:
        for f0, label in target_freqs:
            if f0 <= f_max:
                ax.axvline(
                    f0,
                    color="0.6",
                    lw=0.9,
                    ls="--",
                    alpha=0.8,
                    zorder=1,
                )

                if ax.get_ylim()[0] > 0:
                    ax.text(
                        f0 * 0.98,
                        ax.get_ylim()[0] * 1.4,
                        label,
                        rotation=90,
                        ha="right",
                        va="bottom",
                        fontsize=12,
                        color="0.5",
                    )

    # ------------------------------------------------------------
    # Panel letters
    # ------------------------------------------------------------
    for ax, letter in zip(
        [axA, axB, axC, axD, axE, axF],
        list("ABCDEF"),
    ):
        ax.text(
            -0.14,
            1.03,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    if title:
        fig.suptitle(
            title,
            y=0.985,
            fontsize=12,
        )

    if fig_name is not None:
        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    if show:
        plt.show()

    plt.close(fig)

    return {
        "f": f,
        "delta_u_power": delta_u_power,
        "A2": A2,
        "filtered_delta_power": (
            (p ** 2) * delta_u_power * A2
        ),
        "noise_level": noise_level,
    }

def add_panel_letter(fig, ax, letter, dx=0.018, dy=0.012):
    bb = ax.get_position()
    x = bb.x0 - 4 * dx
    y = bb.y1
    fig.text(x, y, letter, fontsize=12, fontweight="bold", ha="left", va="bottom")

@dataclass
class JointFit:
    df_target: float
    df_used: float
    lam_used: float
    mu_hat: np.ndarray
    best_noise: NoiseFit


def select_best_df_and_noise(
    y: np.ndarray,
    *,
    df_grid: List[float],
    ord: int = 2,
    criterion: str = "AIC",
    rscript_bin: str = "Rscript",
) -> JointFit:
    best: Optional[JointFit] = None
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
            best = JointFit(
                df_target=float(df_target),
                df_used=float(df_used),
                lam_used=float(lam_used),
                mu_hat=mu_hat,
                best_noise=noise_best,
            )

    assert best is not None
    return best


def plot_joint_fit(
    dates: pd.DatetimeIndex,
    y: np.ndarray,
    joint: JointFit,
    title: str,
    fig_name: str,
    show: bool = True,
):
    from matplotlib.dates import MonthLocator, DateFormatter

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y, lw=1, label="downstream y(t)")
    ax.plot(dates, joint.mu_hat, lw=2, label=f"trendfilter mean μ̂(t) (df≈{joint.df_used:.1f})")
    ax.set_title(title + f" | best noise={joint.best_noise.model} | AIC={joint.best_noise.aic:.1f}")
    ax.set_xlabel("Reference date")
    ax.set_ylabel("value")
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    outpath = FIG_DIR / fig_name
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_real_signals_pipeline(
    epidata,
    signals: Dict[str, Tuple[str, str, object, str, str]],
    *,
    start: str,
    end: str,
    df_grid: List[float],
    delay_pmf_by_key: Dict[str, np.ndarray],
    ord: int = 2,
    criterion: str = "AIC",
    inverse_model: str = "gaussian",
    lam_smooth_u0: float = 50.0,
    gamma: float = 1.0,
    alpha_hi: float = 2e5,
    lam_prior: float = 0.0,
    lam_smooth_l2: float = 0.0,
    seed: int = 0,
    target_freqs: Optional[List[float]] = None,
    freq_jitter: float = 0.01,
    basis_width: float = 28.0,
    rscript_bin: str = "Rscript",
):
    for sig_key, (src, sig, time_values, geo_type, geo_val) in signals.items():
        print(f"\n=== {sig_key}: {src}/{sig} ({geo_type}={geo_val}) ===")
        df = fetch_signal_df(
            epidata,
            data_source=src,
            signals=sig,
            time_values=time_values,
            geo_type=geo_type,
            geo_values=geo_val,
        )
        aligned = align_daily(df, start=start, end=end)
        dates = aligned["date"]
        y = aligned["y"].values.astype(float)

        joint = select_best_df_and_noise(
            y,
            df_grid=df_grid,
            ord=ord,
            criterion=criterion,
            rscript_bin=rscript_bin,
        )
        dates_ts = pd.to_datetime(dates)
        eval_mask = (
            (dates_ts >= pd.Timestamp(SINGLEFREQ_INTRO_T_START))
            & (dates_ts <= pd.Timestamp(SINGLEFREQ_INTRO_T_END))
        ).to_numpy()
        if not np.any(eval_mask):
            raise ValueError(
                f"No observations in evaluation window "
                f"{SINGLEFREQ_INTRO_T_START} to {SINGLEFREQ_INTRO_T_END}."
            )
        print(
            f"Best full-support df_target={joint.df_target}, df_used≈{joint.df_used:.2f}, "
            f"lam≈{joint.lam_used:.3g}, noise={joint.best_noise.model}, AIC={joint.best_noise.aic:.2f}"
        )

        plot_joint_fit(
            dates,
            y,
            joint,
            title=f"{sig_key}: selected full-support latent mean + noise model",
            fig_name=f"{sig_key}_latent_mean_trendfilter.png",
            show=True,
        )

        if sig_key not in delay_pmf_by_key:
            print(f"  [skip] No delay PMF provided for '{sig_key}'.")
            continue
        g = normalize_pmf(delay_pmf_by_key[sig_key])

        target_for_inverse = joint.mu_hat if inverse_model == "gaussian" else y
        U0 = reconstruct_upstream_cvxpy(
            target_for_inverse, g,
            model=inverse_model,
            lam_smooth=lam_smooth_u0,
        )
        mu0 = convolve_mean(U0, g)

        U1, mu1, metrics = construct_U1_RMSE_gated_nearnull(
            U0, g,
            y=y,
            mu_hat=joint.mu_hat,
            gamma=gamma,
            seed=seed,
            alpha_hi=alpha_hi,
            lam_prior=lam_prior,
            lam_smooth_l2=lam_smooth_l2,
            target_freqs=target_freqs,
            freq_jitter=freq_jitter,
            basis_width=basis_width,
            eta_y=1.05,
            eval_mask=eval_mask,
        )
        print("U1 metrics:", metrics)

        rmse_KU1_KU0 = float(np.sqrt(np.mean((mu1[eval_mask] - mu0[eval_mask]) ** 2)))
        rmse_KU0_y = float(np.sqrt(np.mean((mu0[eval_mask] - y[eval_mask]) ** 2)))
        rmse_KU1_y = float(np.sqrt(np.mean((mu1[eval_mask] - y[eval_mask]) ** 2)))

        print(f"Window RMSE(KU1, KU0) = {rmse_KU1_KU0:.4g}   "
              f"(target <= {metrics['target_rmse_KU1_KU0']:.4g})")
        print(f"Window RMSE(KU0, y)   = {rmse_KU0_y:.4g}")
        print(f"Window RMSE(KU1, y)   = {rmse_KU1_y:.4g}   "
              f"(must be <= {metrics['eta_y']:.3f} * RMSE(KU0,y) = {metrics['eta_y'] * rmse_KU0_y:.4g})")

        intro_stats = plot_intro_figure(
            dates=dates,
            U0=U0,
            U1=U1,
            g=g,
            y=y,
            mu_hat=joint.mu_hat,
            noise=joint.best_noise,
            p=1.0,
            fig_name=str(FIG_DIR / f"{sig_key}_intro_figure.png"),
            title="",
            f_max=0.5,
            t_start=SINGLEFREQ_INTRO_T_START,
            t_end=SINGLEFREQ_INTRO_T_END,
            show=True,
        )


if __name__ == "__main__":
    DF_GRID = [10, 15, 20, 30, 40, 60, 80, 100, 120]

    emp_subset, emp_summary_df = load_empirical_delay_subset(
        dist_family="lognormal",
        max_delay_days=SINGLEFREQ_EMPIRICAL_MAX_DELAY_DAYS,
    )

    delay_key = 'zhang2020_cn_covid::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal'
    dist_emp = emp_subset[delay_key].dist
    g = pdf_dist_to_daily_pmf(dist_emp, tau_max=SINGLEFREQ_PMF_TAU_MAX)

    #
    # test_mean = 10.0
    # test_sd = emp_sd
    #
    # g = lognormal_pmf_from_mean_sd(
    #     mean_days=test_mean,
    #     sd_days=test_sd,
    #     tau_max=SINGLEFREQ_PMF_TAU_MAX,
    # )
    # print("synthetic g shape:", g.shape, "sum:", g.sum(), "test_mean:", test_mean, "test_sd:", test_sd)

    delay_pmf_by_key = {"covid_cases": g}

    from epidatpy import EpiDataContext, EpiRange
    epidata = EpiDataContext(use_cache=False)

    signals = {
        "covid_cases": (
            "jhu-csse",
            "confirmed_incidence_num",
            EpiRange("2020-04-01", "2021-06-30"),
            "nation",
            "us",
        ),
    }

    run_real_signals_pipeline(
        epidata,
        signals,
        start="2020-04-01",
        end="2021-06-30",
        df_grid=DF_GRID,
        delay_pmf_by_key=delay_pmf_by_key,
        ord=2,
        criterion="AIC",
        inverse_model="gaussian",
        lam_smooth_u0=50.0,
        basis_width=8.0,
        gamma=1.0,
        alpha_hi=2e5,
        lam_prior=0.0,
        lam_smooth_l2=0.0,
        seed=0,
        target_freqs=[1 / 7, 1 / 3],
        freq_jitter=0.01,
        rscript_bin="Rscript",
    )
