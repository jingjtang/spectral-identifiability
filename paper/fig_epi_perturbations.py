#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlled synthetic comparison for event-impact identifiability.

Design principle:
    same baseline, same delay kernel, same observation noise, and the same
    epidemiologically interpretable peak proportional change in Rt across
    mass gathering, short intervention, and vaccination rollout scenarios.

Parameterization:
1. The horizontal axis is event duration or rollout duration in days.
2. The vertical axis is the peak proportional change in Rt:
       effect_size = 0.40
   means a maximum 40% increase in Rt for a mass gathering, or a maximum
   40% reduction in Rt for an intervention or vaccination rollout.
3. No event-specific amplitude constants or total-impact calibration are used.
4. Each event class generates its own controlled synthetic observations from
   its reference downstream mean under the same delay and noise model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import lognorm, gamma
from scipy.ndimage import label
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 6.6,
    "axes.titlesize": 7.4,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.linewidth": 0.40,
    "grid.alpha": 0.16,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# Kernels and utilities
# =========================================================
def make_lognormal_delay(mean_days=7.0, sd_days=6.0, max_delay=40):
    sigma2 = np.log(1.0 + (sd_days**2) / (mean_days**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_days) - 0.5 * sigma2

    edges = np.arange(-0.5, max_delay + 1.5, 1.0)
    edges_pos = np.clip(edges, 1e-8, None)
    cdf = lognorm.cdf(edges_pos, s=sigma, scale=np.exp(mu))
    pmf = np.diff(cdf)
    pmf = np.clip(pmf, 0, None)
    return pmf / pmf.sum()


def make_generation_interval(mean_days=5.0, sd_days=2.0, max_lag=21):
    shape = (mean_days / sd_days) ** 2
    scale = sd_days**2 / mean_days

    edges = np.arange(0.5, max_lag + 1.5, 1.0)
    cdf = gamma.cdf(edges, a=shape, scale=scale)
    pmf = np.diff(cdf)
    pmf = np.clip(pmf, 0, None)
    return pmf / pmf.sum()


def causal_convolve(x, g):
    """
    Full causal linear convolution.

    If len(x) = T and len(g) = L, the downstream length is T + L - 1.
    Keeping the complete convolution ensures that

        FFT(D_candidate - D_ref)
        =
        FFT(g) * FFT(U_candidate - U_ref)

    after zero padding to the full convolution length.
    """
    return np.convolve(x, g, mode="full")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smoothed_event_window(t, center, duration, edge=1.0):
    start = center - duration / 2
    end = center + duration / 2
    return sigmoid((t - start) / edge) - sigmoid((t - end) / edge)


def logistic_rollout_coverage(t, start=50, rollout_days=40):
    center = start + 0.5 * rollout_days
    scale = rollout_days / 8.0

    raw = sigmoid((t - center) / scale)
    raw_start = sigmoid((start - center) / scale)
    raw_end = sigmoid((start + rollout_days - center) / scale)

    coverage = (raw - raw_start) / (raw_end - raw_start)
    return np.clip(coverage, 0, 1)


def smooth_baseline_template(t):
    return (
        35
        + 105 * np.exp(-0.5 * ((t - 42) / 16) ** 2)
        + 85 * np.exp(-0.5 * ((t - 88) / 23) ** 2)
    )


def smooth_baseline_family(t, scale=1.0, slope=0.0):
    base = smooth_baseline_template(t)
    x = (t - t.mean()) / (t.max() - t.min())
    return base * scale * np.exp(slope * x)


def infectiousness_from_incidence(U, w, tt):
    val = 0.0
    for k, wk in enumerate(w, start=1):
        if tt - k >= 0:
            val += wk * U[tt - k]
    return val


def infer_baseline_Rt(U, w):
    Rt = np.ones_like(U, dtype=float)
    for tt in range(len(U)):
        lam = infectiousness_from_incidence(U, w, tt)
        if lam > 0:
            Rt[tt] = U[tt] / lam
        elif tt > 0:
            Rt[tt] = Rt[tt - 1]
    return np.clip(Rt, 0.05, 10.0)


def simulate_renewal(U_seed, Rt, w, seed_until=10):
    U = np.asarray(U_seed, dtype=float).copy()
    for tt in range(seed_until, len(U)):
        lam = infectiousness_from_incidence(U, w, tt)
        U[tt] = Rt[tt] * lam
    return np.clip(U, 0, None)


# =========================================================
# Event multipliers
# =========================================================
def multiplier_mass_gathering(
    t,
    effect_size,
    duration,
    center=50,
    edge=1.0,
):
    """
    effect_size is the peak proportional increase in Rt.

    Example:
        effect_size = 0.40
        -> Rt increases by at most 40% during the gathering.
    """
    h = smoothed_event_window(
        t,
        center=center,
        duration=duration,
        edge=edge,
    )

    # Normalize so that max(h) = 1 exactly.
    h_max = np.max(h)
    if h_max > 0:
        h = h / h_max

    return 1.0 + effect_size * h


def multiplier_intervention(
    t,
    effect_size,
    duration,
    center=55,
    edge=1.0,
):
    """
    effect_size is the peak proportional reduction in Rt.

    Example:
        effect_size = 0.40
        -> Rt decreases by at most 40% during the intervention.
    """
    h = smoothed_event_window(
        t,
        center=center,
        duration=duration,
        edge=edge,
    )

    # Normalize so that max(h) = 1 exactly.
    h_max = np.max(h)
    if h_max > 0:
        h = h / h_max

    return 1.0 - effect_size * h


def multiplier_vaccination(
    t,
    effect_size,
    rollout_days,
    start=50,
):
    """
    effect_size is the maximum proportional reduction in Rt after rollout.

    Example:
        effect_size = 0.40
        -> Rt decreases gradually, reaching a maximum 40% reduction.
    """
    coverage = logistic_rollout_coverage(
        t,
        start=start,
        rollout_days=rollout_days,
    )

    return 1.0 - effect_size * coverage


def renewal_event_incidence(U_base, Rt_base, w, multiplier):
    Rt_event = np.clip(Rt_base * multiplier, 0.01, 10.0)
    return simulate_renewal(U_base, Rt_event, w, seed_until=len(w))


# =========================================================
# Metrics
# =========================================================

def downstream_rmse(D0, D1):
    return np.sqrt(np.mean((D0 - D1) ** 2))


def upstream_rmse(U0, U1):
    return np.sqrt(np.mean((U0 - U1) ** 2))


def one_sided_weights(n):
    """
    Weights converting an rFFT sum into the corresponding two-sided
    Fourier sum for a real-valued signal.
    """
    weights = np.ones(n // 2 + 1, dtype=float)

    if n % 2 == 0:
        # DC and Nyquist occur once; all other frequencies have
        # positive- and negative-frequency counterparts.
        if len(weights) > 2:
            weights[1:-1] = 2.0
    else:
        # Only DC occurs once.
        if len(weights) > 1:
            weights[1:] = 2.0

    return weights

def sample_negative_binomial_mean_dispersion(
    mean,
    kappa,
    rng,
):
    """
    Sample from a negative-binomial distribution parameterized by

        E[Y_t]   = mean_t
        Var[Y_t] = mean_t + mean_t^2 / kappa.

    A Gamma--Poisson mixture is used because it directly supports
    noninteger dispersion kappa.

    Parameters
    ----------
    mean : array-like
        Conditional means, required to be nonnegative.
    kappa : float
        Negative-binomial dispersion parameter. Must be positive.
    rng : numpy.random.Generator
        Random-number generator.
    """
    mean = np.asarray(mean, dtype=float)

    if np.any(mean < 0):
        raise ValueError("Negative-binomial means must be nonnegative.")

    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    # Gamma--Poisson representation:
    #
    # lambda_t ~ Gamma(shape=kappa, scale=mean_t/kappa)
    # Y_t      ~ Poisson(lambda_t)
    #
    # This gives:
    # E[Y_t]   = mean_t
    # Var[Y_t] = mean_t + mean_t^2/kappa.
    latent_rate = np.zeros_like(mean)

    positive = mean > 0

    latent_rate[positive] = rng.gamma(
        shape=kappa,
        scale=mean[positive] / kappa,
    )

    return rng.poisson(latent_rate)

def negative_binomial_effective_noise(
    D_candidate,
    D_ref,
    kappa,
):
    """
    Conservative effective noise scale used in the negative-binomial
    separability bound.

    For

        Var(Y_t | m_t) = m_t + m_t^2 / kappa,

    define

        m_max = max_t max(m_candidate(t), m_ref(t))

    and

        N_eff = m_max + m_max^2 / kappa.

    This constant upper variance bound preserves the spectral
    representation used in Methods.
    """
    D_candidate = np.asarray(D_candidate, dtype=float)
    D_ref = np.asarray(D_ref, dtype=float)

    if D_candidate.shape != D_ref.shape:
        raise ValueError(
            "D_candidate and D_ref must have the same shape."
        )

    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    m_max = max(
        float(np.max(D_candidate)),
        float(np.max(D_ref)),
    )

    if m_max <= 0:
        raise ValueError(
            "At least one downstream mean must be positive."
        )

    N_eff = m_max + m_max**2 / kappa

    return N_eff, m_max


def J_identifiability_nb(
    U_candidate,
    U_ref,
    g,
    kappa,
    p=1.0,
    dt=1.0,
):
    """
    Negative-binomial separability functional corresponding to

        Var(Y_t | m_t) = m_t + m_t^2 / kappa.

    For each reference--candidate pair, the effective noise scale is

        N_eff = m_max + m_max^2 / kappa,

    where m_max is the maximum downstream conditional mean across
    the two hypotheses.

    The resulting functional is

        J = integral
            |p G(f)|^2 |Delta U(f)|^2 / N_eff df.

    Under the sufficient testing-error bound used in Methods,
    non-identifiability is guaranteed when

        J <= 4(1 - alpha)^2.
    """
    U_candidate = np.asarray(U_candidate, dtype=float)
    U_ref = np.asarray(U_ref, dtype=float)
    g = np.asarray(g, dtype=float)

    if U_candidate.shape != U_ref.shape:
        raise ValueError(
            "U_candidate and U_ref must have the same shape."
        )

    if p <= 0:
        raise ValueError("p must be positive.")

    delta_u = U_candidate - U_ref

    # Conditional downstream means under the two hypotheses.
    D_candidate = p * causal_convolve(U_candidate, g)
    D_ref = p * causal_convolve(U_ref, g)

    N_eff, m_max = negative_binomial_effective_noise(
        D_candidate=D_candidate,
        D_ref=D_ref,
        kappa=kappa,
    )

    n_full = len(delta_u) + len(g) - 1

    dU = np.fft.rfft(delta_u, n=n_full)
    G = np.fft.rfft(g, n=n_full)
    weights = one_sided_weights(n_full)

    j_bins = (
        dt / n_full
        * weights
        * np.abs(p * G) ** 2
        * np.abs(dU) ** 2
        / N_eff
    )

    return float(np.sum(j_bins)), N_eff, m_max


def spectral_quantities_nb(
    delta_u,
    g,
    N_eff,
    p=1.0,
    dt=1.0,
):
    """
    Frequency-bin decomposition of the negative-binomial separability
    functional using the pair-specific effective noise bound

        N_eff = m_max + m_max^2 / kappa.

    The returned `weighted` array sums exactly to J.
    """
    delta_u = np.asarray(delta_u, dtype=float)
    g = np.asarray(g, dtype=float)

    if N_eff <= 0:
        raise ValueError("N_eff must be positive.")

    n_full = len(delta_u) + len(g) - 1

    dU = np.fft.rfft(delta_u, n=n_full)
    G = np.fft.rfft(g, n=n_full)
    freqs = np.fft.rfftfreq(n_full, d=dt)

    weights = one_sided_weights(n_full)

    raw_power = (
        dt / n_full
        * weights
        * np.abs(dU) ** 2
    )

    weighted = (
        dt / n_full
        * weights
        * np.abs(p * G) ** 2
        * np.abs(dU) ** 2
        / N_eff
    )

    return freqs, raw_power, weighted


def choose_low_separability_high_contrast_example(
    x_grid,
    y_grid,
    U_ref,
    U_candidates,
    D_candidates,
    J_grid,
    x_ref,
    y_ref,
    J_threshold,
    example_J_multiplier=3.0,
    min_param_distance=0.03,
    interior_margin=0.02,
):
    """
    Select an illustrative alternative outside the sufficient
    non-identifiability contour, while retaining relatively low J.

    Requirements
    ------------
    J_threshold < J <= example_J_multiplier * J_threshold

    Objective
    ---------
    Maximize upstream trajectory difference from the reference.

    Important
    ---------
    The selected point is not theoretically guaranteed to be
    non-identifiable because it lies outside J <= J_threshold.
    """
    X, Y = np.meshgrid(x_grid, y_grid)

    x_span = x_grid.max() - x_grid.min()
    y_span = y_grid.max() - y_grid.min()

    param_distance = np.sqrt(
        ((X - x_ref) / x_span) ** 2
        + ((Y - y_ref) / y_span) ** 2
    )

    x_lo = x_grid.min() + interior_margin * x_span
    x_hi = x_grid.max() - interior_margin * x_span
    y_lo = y_grid.min() + interior_margin * y_span
    y_hi = y_grid.max() - interior_margin * y_span

    interior = (
        (X >= x_lo)
        & (X <= x_hi)
        & (Y >= y_lo)
        & (Y <= y_hi)
    )

    J_example_max = example_J_multiplier * J_threshold

    admissible = (
        np.isfinite(J_grid)
        & interior
        & (param_distance >= min_param_distance)
        & (J_grid > J_threshold)
        & (J_grid <= J_example_max)
    )

    ix_ref = int(np.argmin(np.abs(x_grid - x_ref)))
    iy_ref = int(np.argmin(np.abs(y_grid - y_ref)))
    admissible[iy_ref, ix_ref] = False

    score = np.full_like(J_grid, -np.inf, dtype=float)
    upstream_difference = np.full_like(J_grid, np.nan, dtype=float)

    reference_scale = np.sqrt(np.mean(U_ref**2)) + 1e-12

    for iy in range(len(y_grid)):
        for ix in range(len(x_grid)):
            if not admissible[iy, ix]:
                continue

            U = U_candidates[iy][ix]
            D = D_candidates[iy][ix]

            if U is None or D is None:
                continue

            # Relative upstream RMSE across the complete analyzed signal.
            u_rmse = np.sqrt(np.mean((U - U_ref) ** 2))
            relative_u_rmse = u_rmse / reference_scale

            upstream_difference[iy, ix] = relative_u_rmse

            # Upstream contrast is the primary criterion.
            # Parameter distance is only a small tie-breaker.
            score[iy, ix] = (
                relative_u_rmse
                + 0.01 * param_distance[iy, ix]
            )

    if not np.any(np.isfinite(score)):
        raise RuntimeError(
            "No candidate was found outside the theoretical contour "
            "within the specified low-J example range. Increase "
            "example_J_multiplier, expand the parameter grid, or reduce "
            "the minimum parameter-distance requirement."
        )

    iy_alt, ix_alt = np.unravel_index(
        np.nanargmax(score),
        score.shape,
    )

    print(
        f"Selected outside-contour example: "
        f"J={J_grid[iy_alt, ix_alt]:.4f}, "
        f"J/J_threshold="
        f"{J_grid[iy_alt, ix_alt] / J_threshold:.3f}, "
        f"relative upstream RMSE="
        f"{upstream_difference[iy_alt, ix_alt]:.4f}"
    )

    return (
        x_grid[ix_alt],
        y_grid[iy_alt],
        U_candidates[iy_alt][ix_alt],
        D_candidates[iy_alt][ix_alt],
    )

def add_panel_label(ax, label, x=-0.24, y=1.12):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
# =========================================================
# Controlled simulation setup
# =========================================================
PRE_EVENT_DAYS = 30
POST_EVENT_DAYS = 120

T = PRE_EVENT_DAYS + POST_EVENT_DAYS + 1
t = np.arange(T)

EVENT_CENTER = PRE_EVENT_DAYS

g = make_lognormal_delay(mean_days=7.0, sd_days=6.0, max_delay=40)
w = make_generation_interval(mean_days=5.0, sd_days=2.0, max_lag=21)

# Negative-binomial dispersion parameter.

# Var(Y_t | mu_t) = mu_t + mu_t^2 / nb_kappa

nb_kappa = 20.0

# Conversion / ascertainment rate.

p_conversion = 1.0

alpha_target = 0.20
J_threshold = 4 * (1 - alpha_target) ** 2

rng_obs = np.random.default_rng(20260709)

# Peak proportional change in Rt.
# 0.05 = 5%, 0.90 = 90%.
effect_grid = np.linspace(0.05, 0.90, 91)

# Same reference peak Rt effect for all event classes.
REFERENCE_EFFECT = 0.50

effect_grid = np.unique(
    np.sort(
        np.r_[effect_grid, REFERENCE_EFFECT]
    )
)

U_base_ref = smooth_baseline_family(t, scale=1.0, slope=0.0)
Rt_base_ref = infer_baseline_Rt(U_base_ref, w)


# Event reference durations/rollout timescales.
event_specs = [
    {
        "title": "Mass gathering",
        "xlabel": "Duration (days)",
        "x_ref_actual": 3.0,
        "effect_ref": REFERENCE_EFFECT,
        "x_grid": np.unique(
            np.sort(
                np.r_[np.linspace(1.0, 30.0, 91), 3.0]
            )
        ),
        "make_multiplier_actual": (
            lambda x_actual, effect_size:
            multiplier_mass_gathering(
                t,
                effect_size=effect_size,
                duration=x_actual,
                center=EVENT_CENTER,
            )
        ),
        "selection": {
            "example_J_multiplier": 3.0,
            "min_param_distance": 0.03,
            "interior_margin": 0.04,
        },
    },
    {
        "title": "Temporary lockdown",
        "xlabel": "Duration (days)",
        "x_ref_actual": 10.0,
        "effect_ref": 0.5,
        "x_grid": np.unique(
            np.sort(
                np.r_[np.linspace(3.0, 60.0, 91), 10.0]
            )
        ),
        "make_multiplier_actual": (
            lambda x_actual, effect_size:
            multiplier_intervention(
                t,
                effect_size=effect_size,
                duration=x_actual,
                center=EVENT_CENTER,
            )
        ),
        "selection": {
            "example_J_multiplier": 3.0,
            "min_param_distance": 0.03,
            "interior_margin": 0.04,
        },
    },
    {
        "title": "Vaccination rollout",
        "xlabel": "Rollout duration (days)",
        "x_ref_actual": 60.0,
        "effect_ref": REFERENCE_EFFECT,
        "x_grid": np.unique(
            np.sort(
                np.r_[np.linspace(10.0, 120.0, 91), 60.0]
            )
        ),
        "make_multiplier_actual": (
            lambda x_actual, effect_size:
            multiplier_vaccination(
                t,
                effect_size=effect_size,
                rollout_days=x_actual,
                start=EVENT_CENTER,
            )
        ),
        "selection": {
            "example_J_multiplier": 3.0,
            "min_param_distance": 0.03,
            "interior_margin": 0.04,
        },
    },
]
# =========================================================
# Compute event landscapes
# =========================================================
for spec in event_specs:
    x_ref_actual = spec["x_ref_actual"]
    effect_ref = spec["effect_ref"]

    x_grid = spec["x_grid"]
    y_grid = effect_grid

    spec["y_grid"] = y_grid


    def make_candidate(x_actual, effect_size):
        multiplier = spec["make_multiplier_actual"](
            x_actual,
            effect_size,
        )

        U = renewal_event_incidence(
            U_base_ref,
            Rt_base_ref,
            w,
            multiplier,
        )

        D = p_conversion * causal_convolve(U, g)

        return multiplier, U, D

    # Reference event:
    # same peak proportional Rt change across all event classes.
    multiplier_ref, U_ref, D_ref = make_candidate(
        x_ref_actual,
        effect_ref,
    )

    Y_obs = sample_negative_binomial_mean_dispersion(
        mean=D_ref,
        kappa=nb_kappa,
        rng=rng_obs,
    )

    spec["multiplier_ref"] = multiplier_ref
    spec["U_ref"] = U_ref
    spec["D_ref"] = D_ref
    spec["Y_obs"] = Y_obs

    spec["x_ref"] = x_ref_actual
    spec["y_ref"] = effect_ref

    Z = np.zeros((len(y_grid), len(x_grid)))
    D_rmse_grid = np.zeros_like(Z)
    U_rmse_grid = np.zeros_like(Z)

    U_candidates = [
        [None for _ in x_grid]
        for _ in y_grid
    ]
    D_candidates = [
        [None for _ in x_grid]
        for _ in y_grid
    ]

    N_eff_grid = np.zeros_like(Z)
    m_max_grid = np.zeros_like(Z)
    for ix, x_actual in enumerate(x_grid):
        for iy, effect_size in enumerate(y_grid):

            (
                _,
                U_candidate,
                D_candidate,
            ) = make_candidate(
                x_actual,
                effect_size,
            )

            J, N_eff, m_max = J_identifiability_nb(
                U_candidate=U_candidate,
                U_ref=U_ref,
                g=g,
                kappa=nb_kappa,
                p=p_conversion,
                dt=1.0,
            )

            Z[iy, ix] = J

            N_eff_grid[iy, ix] = N_eff
            m_max_grid[iy, ix] = m_max

            D_rmse_grid[iy, ix] = downstream_rmse(
                D_candidate,
                D_ref,
            )

            U_rmse_grid[iy, ix] = upstream_rmse(
                U_candidate,
                U_ref,
            )

            U_candidates[iy][ix] = U_candidate
            D_candidates[iy][ix] = D_candidate

    spec["N_eff"] = N_eff_grid
    spec["m_max"] = m_max_grid

    spec["J"] = Z
    spec["J_plot"] = np.log10(1.0 + Z)
    spec["D_rmse"] = D_rmse_grid
    spec["U_rmse"] = U_rmse_grid
    spec["U_candidates"] = U_candidates
    spec["D_candidates"] = D_candidates

    selection = spec["selection"]

    x_alt, effect_alt, U_alt, D_alt = (
        choose_low_separability_high_contrast_example(
            x_grid=x_grid,
            y_grid=y_grid,
            U_ref=U_ref,
            U_candidates=U_candidates,
            D_candidates=D_candidates,
            J_grid=Z,
            x_ref=spec["x_ref"],
            y_ref=spec["y_ref"],
            J_threshold=J_threshold,
            example_J_multiplier=spec["selection"]["example_J_multiplier"],
            min_param_distance=spec["selection"]["min_param_distance"],
            interior_margin=spec["selection"]["interior_margin"],
        )
    )

    spec["x_alt"] = x_alt
    spec["y_alt"] = effect_alt
    spec["U_alt"] = U_alt
    spec["D_alt"] = D_alt

    spec["multiplier_alt"] = spec["make_multiplier_actual"](
        x_alt,
        effect_alt,
    )

    delta_u_pair = U_alt - U_ref

    J_pair, N_eff_pair, m_max_pair = J_identifiability_nb(
        U_candidate=U_alt,
        U_ref=U_ref,
        g=g,
        kappa=nb_kappa,
        p=p_conversion,
        dt=1.0,
    )

    freqs, raw_power, weighted = spectral_quantities_nb(
        delta_u=delta_u_pair,
        g=g,
        N_eff=N_eff_pair,
        p=p_conversion,
        dt=1.0,
    )

    spec["N_eff_pair"] = N_eff_pair
    spec["m_max_pair"] = m_max_pair
    spec["J_pair"] = J_pair

    spec["freqs"] = freqs
    spec["raw_power"] = raw_power
    spec["weighted"] = weighted

    # ---------------------------------------------------------
    # Consistency checks
    # ---------------------------------------------------------

    # J computed directly from the Methods spectral functional.
    J_methods, N_eff_pair, m_max_pair = J_identifiability_nb(
        U_candidate=U_alt,
        U_ref=U_ref,
        g=g,
        kappa=nb_kappa,
        p=p_conversion,
        dt=1.0,
    )

    J_spectral = np.sum(weighted)

    delta_d_full = (
            p_conversion
            * np.convolve(delta_u_pair, g, mode="full")
    )

    J_downstream = np.sum(
        delta_d_full ** 2 / N_eff_pair
    )

    if not np.isclose(
            J_methods,
            J_spectral,
            rtol=1e-10,
            atol=1e-10,
    ):
        raise RuntimeError(
            f"Spectral-bin mismatch for {spec['title']}: "
            f"J_methods={J_methods:.12f}, "
            f"sum(weighted)={J_spectral:.12f}"
        )

    if not np.isclose(
            J_methods,
            J_downstream,
            rtol=1e-10,
            atol=1e-10,
    ):
        raise RuntimeError(
            f"Fourier/time-domain mismatch for {spec['title']}: "
            f"J_methods={J_methods:.12f}, "
            f"J_downstream={J_downstream:.12f}"
        )

    print(
        f"{spec['title']}: "
        f"m_max={m_max_pair:.3f}, "
        f"N_eff={N_eff_pair:.3f}, "
        f"J={J_methods:.4f}"
    )

# =========================================================
# Shared plot limits
# =========================================================
u_min = min(
    min(np.min(spec["U_ref"]) for spec in event_specs),
    min(np.min(spec["U_alt"]) for spec in event_specs),
)
u_max = max(
    max(np.max(spec["U_ref"]) for spec in event_specs),
    max(np.max(spec["U_alt"]) for spec in event_specs),
)
u_pad = 0.06 * (u_max - u_min)

def nb_sd(mean, kappa):
    mean = np.asarray(mean, dtype=float)
    return np.sqrt(mean + mean**2 / kappa)


d_min = min(
    min(np.min(spec["D_ref"]) for spec in event_specs),
    min(np.min(spec["D_alt"]) for spec in event_specs),
    min(np.min(spec["Y_obs"]) for spec in event_specs),
)

d_max = max(
    max(np.max(spec["D_ref"]) for spec in event_specs),
    max(np.max(spec["D_alt"]) for spec in event_specs),
    max(np.max(spec["Y_obs"]) for spec in event_specs),
)

d_pad = 0.06 * (d_max - d_min)

vmin = 0.0
vmax = max(np.percentile(spec["J_plot"], 97) for spec in event_specs)

# =========================================================
# New layout: one event type per column
# Row 1 = heatmaps
# Row 2 = compatible fraction curves
# Row 3 = 2x2 example panels:
#         Upstream | Event multiplier
#         Downstream | Spectrum
# =========================================================
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# =========================================================
# Shared spectrum y-axis limits and ticks
# =========================================================
spectrum_values = []

for spec in event_specs:
    freqs = spec["freqs"]

    keep = (
        (freqs > 0)
        & (freqs <= 0.5)
    )

    spectrum_values.append(
        spec["raw_power"][keep]
    )
    spectrum_values.append(
        spec["weighted"][keep]
    )

spectrum_values = np.concatenate(spectrum_values)

# Remove zero, negative, NaN, and infinite values.
spectrum_values = spectrum_values[
    np.isfinite(spectrum_values)
    & (spectrum_values > 0)
]

spectrum_ymin = 10 ** np.floor(
    np.log10(np.min(spectrum_values))
)
spectrum_ymax = 10 ** np.ceil(
    np.log10(np.max(spectrum_values))
)

# Integer powers of ten shared by all three panels.
spectrum_log_min = int(
    np.floor(np.log10(spectrum_ymin))
)

spectrum_log_max = int(
    np.ceil(np.log10(spectrum_ymax))
)

spectrum_yticks = 10.0 ** np.arange(
    spectrum_log_min,
    spectrum_log_max + 1,
    3,
)

fig = plt.figure(figsize=(7.5, 4.7), constrained_layout=False)

outer = GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.05, 1.10],
    left=0.07,
    right=0.90,
    bottom=0.13,
    top=0.955,
    wspace=0.22,
    hspace=0.35,
)

c_ref = "tab:red"
c_alt = "tab:blue"
c_obs = "0.35"

heatmap_axes = []
last_im = None

heatmap_labels = ["A", "F", "K"]
upstream_labels = ["B", "G", "L"]
multiplier_labels = ["D", "I", "N"]
downstream_labels = ["C", "H", "M"]
spectrum_labels = ["E", "J", "O"]

for j, spec in enumerate(event_specs):
    # -------------------------
    # Row 1: Heatmap
    # -------------------------
    ax_hm = fig.add_subplot(outer[0, j])
    heatmap_axes.append(ax_hm)

    X, Y = np.meshgrid(spec["x_grid"], spec["y_grid"])

    last_im = ax_hm.contourf(
        X,
        Y,
        spec["J_plot"],
        levels=np.linspace(vmin, vmax, 60),
        cmap="magma",
        extend="max",
    )

    # Auxiliary percentile contours for visual structure.
    # contour_levels = np.percentile(
    #     spec["J_plot"],
    #     [20, 40, 60, 80],
    # )
    # contour_levels = np.unique(contour_levels)
    #
    # if len(contour_levels) >= 2:
    #     ax_hm.contour(
    #         X,
    #         Y,
    #         spec["J_plot"],
    #         levels=contour_levels,
    #         colors="white",
    #         linewidths=0.35,
    #         alpha=0.65,
    #     )

    # Theoretical sufficient non-identifiability threshold:
    # J = 4(1 - alpha)^2.
    J_threshold_plot = np.log10(1.0 + J_threshold)

    if (
            np.nanmin(spec["J_plot"])
            <= J_threshold_plot
            <= np.nanmax(spec["J_plot"])
    ):
        ax_hm.contour(
            X,
            Y,
            spec["J_plot"],
            levels=[J_threshold_plot],
            colors="cyan",
            linewidths=0.95,
            linestyles="-",
            zorder=4,
        )

    # Reference and selected alternative.
    ax_hm.plot(
        spec["x_ref"],
        spec["y_ref"],
        "o",
        color=c_ref,
        ms=4.2,
        mec="white",
        mew=0.45,
        zorder=5,
    )

    ax_hm.plot(
        spec["x_alt"],
        spec["y_alt"],
        "X",
        color=c_alt,
        ms=5.2,
        mec="white",
        mew=0.45,
        zorder=5,
    )

    ax_hm.set_xlabel(spec["xlabel"])
    ax_hm.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    if j == 0:
        ax_hm.set_ylabel(
            r"Peak $R_t$ change"
        )
    else:
        ax_hm.set_ylabel("")

    ax_hm.grid(False)

    box = ax_hm.get_position()


    # -------------------------
    # Row 3: 2x2 example panels
    # -------------------------
    sub = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=outer[1, j],
        wspace=0.58,
        hspace=0.62,
    )

    ax_up = fig.add_subplot(sub[0, 0])
    ax_mult = fig.add_subplot(sub[0, 1])
    ax_down = fig.add_subplot(sub[1, 0])
    ax_spec = fig.add_subplot(sub[1, 1])

    # Upstream: left upper
    ax_up.plot(t, spec["U_ref"], color=c_ref, lw=0.95)
    ax_up.plot(t, spec["U_alt"], color=c_alt, lw=0.90)
    ax_up.set_title("Upstream", pad=2)
    ax_up.set_xlim(t[0], t[-1])
    ax_up.set_ylim(u_min - u_pad, u_max + u_pad)
    ax_up.set_xlabel("Simulation day", labelpad=2)
    ax_up.set_ylabel("Incidence", labelpad=2)
    ax_up.grid(True)

    # Event multiplier: right upper
    ax_mult.plot(t, spec["multiplier_ref"], color=c_ref, lw=0.95)
    ax_mult.plot(t, spec["multiplier_alt"], color=c_alt, lw=0.90)

    ax_mult.axhline(
        1,
        color="0.45",
        lw=0.45,
        alpha=0.7,
    )

    ax_mult.set_title("Event profile", pad=2)

    # Same time range as the upstream trajectory.
    ax_mult.set_xlim(t[0], t[-1])

    ax_mult.set_xlabel("Simulation day", labelpad=2)
    ax_mult.set_ylabel(r"$R_t$ multiplier", labelpad=2)

    ax_mult.grid(True)

    # Downstream: left lower
    t_down = np.arange(T + len(g) - 1)

    ax_down.scatter(
        t_down,
        spec["Y_obs"],
        s=2.4,
        color=c_obs,
        alpha=0.32,
        linewidths=0,
        zorder=1,
    )

    ax_down.plot(
        t_down,
        spec["D_ref"],
        color=c_ref,
        lw=0.95,
        zorder=3,
    )

    ax_down.plot(
        t_down,
        spec["D_alt"],
        color=c_alt,
        lw=0.90,
        zorder=3,
    )

    ax_down.set_xlim(t_down[0], t_down[-1])
    ax_down.set_ylim(d_min - d_pad, d_max + d_pad)

    ax_down.set_title("Downstream", pad=2)
    ax_down.set_xlabel("Simulation day", labelpad=2)
    ax_down.set_ylabel("Incidence", labelpad=2)

    ax_down.grid(True)

    # Spectrum: right lower
    freqs = spec["freqs"]

    keep = (
            (freqs > 0)
            & (freqs <= 0.5)
    )

    f = freqs[keep]

    eps = 1e-16
    raw = np.maximum(spec["raw_power"][keep], eps)
    weighted = np.maximum(spec["weighted"][keep], eps)

    ax_spec.plot(
        f,
        raw,
        color="tab:orange",
        lw=0.90,
    )

    ax_spec.plot(
        f,
        weighted,
        color="black",
        lw=0.90,
    )

    ax_spec.set_yscale("log")
    ax_spec.set_xlim(0, 0.5)
    ax_spec.set_ylim(spectrum_ymin, spectrum_ymax)

    ax_spec.axvline(
        1 / 7,
        color="0.55",
        lw=0.45,
        ls="--",
    )

    ax_spec.axvline(
        1 / 3,
        color="0.55",
        lw=0.45,
        ls="--",
    )

    ax_spec.set_title("Separability", pad=2)
    ax_spec.set_xlabel(r"Frequency (day$^{-1}$)", labelpad=2)
    ax_spec.set_ylabel("Spectral contribution", labelpad=2)

    ax_spec.grid(True)

    ax_spec.set_ylim(
        spectrum_ymin,
        spectrum_ymax,
    )

    ax_spec.set_yticks(
        spectrum_yticks,
    )

    for ax_small in [ax_up, ax_mult, ax_down, ax_spec]:
        ax_small.tick_params(
            labelsize=5.2,
            pad=1.2,
        )
        ax_small.xaxis.label.set_size(5.8)
        ax_small.yaxis.label.set_size(5.8)

    box = ax_up.get_position()

    add_panel_label(ax_hm, heatmap_labels[j], x=-0.22, y=1.08)
    add_panel_label(ax_up, upstream_labels[j], x=-0.34, y=1.12)
    add_panel_label(ax_mult, multiplier_labels[j], x=-0.34, y=1.12)
    add_panel_label(ax_down, downstream_labels[j], x=-0.34, y=1.12)
    add_panel_label(ax_spec, spectrum_labels[j], x=-0.34, y=1.12)


# =========================================================
# Vertical colorbar to the right of the third heatmap
# =========================================================
fig.canvas.draw()

# =========================================================
# Column titles
# =========================================================
for ax_hm, spec in zip(heatmap_axes, event_specs):
    box = ax_hm.get_position()

    header_y = box.y1 + 0.12 * box.height

    fig.text(
        box.x0 + 0.5 * box.width,
        header_y,
        spec["title"],
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="semibold",
    )

right_box = heatmap_axes[2].get_position()

cbar_gap = 0.007
cbar_width = 0.007

cbar_left = right_box.x1 + cbar_gap
cbar_bottom = right_box.y0
cbar_height = right_box.height

cax = fig.add_axes([
    cbar_left,
    cbar_bottom,
    cbar_width,
    cbar_height,
])

cbar = fig.colorbar(
    last_im,
    cax=cax,
    orientation="vertical",
)

cbar.set_ticks(
    np.arange(
        np.ceil(vmin),
        np.floor(vmax) + 1,
        1,
    )
)

cbar.ax.tick_params(
    labelsize=5.6,
    length=1.8,
    pad=1.5,
)

cbar.set_label(
    r"$\log_{10}(1 + J_\mathrm{event})$",
    fontsize=6.2,
    labelpad=7.5,
    rotation=270,
)



# Shared legend
handles = [
    Line2D([0], [0], color=c_ref, lw=1.1),
    Line2D([0], [0], color=c_alt, lw=1.1),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=c_obs, markeredgecolor="none",
           markersize=3.5, alpha=0.45),
    Line2D([0], [0], color="tab:orange", lw=1.0),
    Line2D([0], [0], color="black", lw=1.0),
]
labels = [
    "Reference event",
    "Alternative event",
    "Noisy synthetic observation",
    "Upstream spectral contrast",
    r"Contribution to $J_\mathrm{event}$",
]

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.48, 0.015),
    ncol=5,
    frameon=False,
    handlelength=1.7,
    columnspacing=0.85,
)

os.makedirs("../figs", exist_ok=True)
fig.savefig("../figs/event_controlled_synthetic_column_layout.pdf", bbox_inches="tight")
fig.savefig("../figs/event_controlled_synthetic_column_layout.png", bbox_inches="tight")
print("Saved to ../figs/event_controlled_synthetic_column_layout.pdf/.png")
