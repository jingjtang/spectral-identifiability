#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from pathlib import Path

from get_incubation_periods import build_lessler2009_support
from utils import load_empirical_delay_subset

plt.ioff()
# ============================================================
# USER CONTROLS
# ============================================================
ALPHA = 0.2
P_CONV = 1.0

T_SIGNAL = 240
AMP_SIGNAL = 1.0

MEDIAN_MIN = 0.5
MEDIAN_MAX = 21.0
DISP_MIN = np.exp(0.10)
DISP_MAX = np.exp(1.75)

N_MEDIAN = 220
N_DISP = 220

MAX_DELAY = 160
DT_FINE = 0.05

MISSPEC_GRID = np.linspace(-0.30, 0.30, 61)

DIST_FAMILY = "lognormal"
DROP_KEY = [
    "uscdc_linelist_covid::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
    "xu2020_covid::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
    "hinch2024_uk_mpox::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
]

ONE_POINT_PER_DISEASE = True

OUTDIR = Path("../figs")
OUTFILE = OUTDIR / "sensitivity_analysis.pdf"

# ============================================================
# Style
# ============================================================
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Helpers
# ============================================================
def normalize_pmf(x):
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 0:
        raise ValueError("PMF sum <= 0")
    return x / s


def canonical_delta_u_periodic(T=240, period_days=7.0, amp=1.0, phase=0.0):
    t = np.arange(T, dtype=float)
    return amp * np.sin(2.0 * np.pi * t / period_days + phase)


def pdf_to_discrete_pmf(pdf_func, max_delay=160, dt=0.05):
    t_fine = np.arange(0, max_delay + dt, dt)
    y = np.asarray(pdf_func(t_fine), float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)

    day_edges = np.arange(0, max_delay + 1, 1.0)
    pmf = np.zeros(len(day_edges) - 1, dtype=float)

    for i in range(len(pmf)):
        m = (t_fine >= day_edges[i]) & (t_fine < day_edges[i + 1])
        if np.any(m):
            pmf[i] = np.trapezoid(y[m], t_fine[m])

    return normalize_pmf(pmf)


def lognormal_pdf_factory(mu_log, sigma_log):
    def pdf(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t, dtype=float)
        m = t > 0
        tt = t[m]
        out[m] = (
            1.0 / (tt * sigma_log * np.sqrt(2.0 * np.pi))
        ) * np.exp(
            -((np.log(tt) - mu_log) ** 2) / (2.0 * sigma_log ** 2)
        )
        return out
    return pdf


def lognormal_pmf_from_params(mu_log, sigma_log, max_delay=160, dt=0.05):
    pdf = lognormal_pdf_factory(mu_log, sigma_log)
    return pdf_to_discrete_pmf(pdf, max_delay=max_delay, dt=dt)


def median_dispersion_to_lognormal_params(median_delay, dispersion):
    median_delay = float(median_delay)
    dispersion = float(dispersion)

    if median_delay <= 0 or dispersion <= 1:
        raise ValueError("median delay must be positive and dispersion must be > 1")

    return np.log(median_delay), np.log(dispersion)


def mean_sd_to_siglog(mean_delay, sd_delay):
    mean_delay = float(mean_delay)
    sd_delay = float(sd_delay)
    if mean_delay <= 0 or sd_delay <= 0:
        return np.nan
    return float(np.sqrt(np.log(1.0 + (sd_delay / mean_delay) ** 2)))


def compute_singlefreq_contrast_stat(delta_u, g, contrast_noise_scale, p=1.0, dt=1.0):
    delta_u = np.asarray(delta_u, float)
    g = normalize_pmf(g)

    n = len(delta_u)
    contrast_noise_scale = max(float(contrast_noise_scale), 1e-12)

    G = np.fft.rfft(g, n=n)
    U = np.fft.rfft(delta_u, n=n)

    freqs = np.fft.rfftfreq(n, d=dt)
    df = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    integrand = (
        (np.abs(p * G) ** 2)
        * (np.abs(U) ** 2)
        / (contrast_noise_scale ** 2)
    )

    fft_weight = np.ones_like(integrand)
    if len(fft_weight) > 2:
        fft_weight[1:-1] = 2.0

    return float(np.sum(fft_weight * integrand) * df)


def critical_contrast_noise_scale_for_delay_pmf(g, delta_u, alpha=0.2, p=1.0, dt=1.0):
    J_thresh = 4.0 * (1.0 - alpha) ** 2
    K = compute_singlefreq_contrast_stat(
        delta_u,
        g,
        contrast_noise_scale=1.0,
        p=p,
        dt=dt,
    )

    if K <= 0 or not np.isfinite(K):
        return np.nan

    return float(np.sqrt(K / J_thresh))


def critical_variance_ratio_for_delay_pmf(g, delta_u, alpha=0.2, p=1.0, dt=1.0):
    noise_scale_crit = critical_contrast_noise_scale_for_delay_pmf(
        g=g, delta_u=delta_u, alpha=alpha, p=p, dt=dt
    )

    if not np.isfinite(noise_scale_crit) or noise_scale_crit <= 0:
        return np.nan

    signal_var_up = np.var(delta_u)
    if not np.isfinite(signal_var_up) or signal_var_up <= 0:
        return np.nan

    return float(signal_var_up / (noise_scale_crit ** 2))


def required_variance_ratio(median_delay, dispersion, delta_u,
                            alpha=0.2, p_conv=1.0,
                            max_delay=160, dt_fine=0.05):
    mu_log, sigma_log = median_dispersion_to_lognormal_params(
        median_delay, dispersion
    )
    g = lognormal_pmf_from_params(
        mu_log=mu_log,
        sigma_log=sigma_log,
        max_delay=max_delay,
        dt=dt_fine,
    )
    return critical_variance_ratio_for_delay_pmf(
        g=g,
        delta_u=delta_u,
        alpha=alpha,
        p=p_conv,
        dt=1.0,
    )


def build_critical_variance_ratio_map(median_grid, dispersion_grid, delta_u,
                                      alpha=0.2, p_conv=1.0,
                                      max_delay=160, dt_fine=0.05):
    out = np.full((len(dispersion_grid), len(median_grid)), np.nan, dtype=float)

    for iy, dispersion in enumerate(dispersion_grid):
        for ix, median_delay in enumerate(median_grid):
            try:
                out[iy, ix] = required_variance_ratio(
                    median_delay=median_delay,
                    dispersion=dispersion,
                    delta_u=delta_u,
                    alpha=alpha,
                    p_conv=p_conv,
                    max_delay=max_delay,
                    dt_fine=dt_fine,
                )
            except Exception:
                out[iy, ix] = np.nan

    return out


def disease_from_dataset_name(ds: str) -> str:
    ds = str(ds).lower()
    if "covid" in ds:
        return "COVID-19"
    if "mpox" in ds:
        return "Mpox"
    if "dengue" in ds:
        return "Dengue"
    if "ebola" in ds:
        return "Ebola"
    return "Other"


def extract_median_dispersion_from_dist(dist):
    median_delay = float(dist.median())
    mean_delay = float(dist.mean())
    sd_delay = float(dist.std())
    sigma_log = mean_sd_to_siglog(mean_delay, sd_delay)
    dispersion = float(np.exp(sigma_log))
    return median_delay, dispersion


def log10_safe(x, floor=1e-300):
    x = np.asarray(x, float)
    return np.log10(np.maximum(x, floor))


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.14)


def misspec_curve(marker, delta_u, misspec_grid, perturb="median"):
    m0 = float(marker["median_delay"])
    d0 = float(marker["dispersion"])

    out = []
    for eps in misspec_grid:
        if perturb == "median":
            m = m0 * (1.0 + eps)
            d = d0
        elif perturb == "dispersion":
            m = m0
            d = d0 * (1.0 + eps)
        else:
            raise ValueError("perturb must be 'median' or 'dispersion'")

        if m <= 0 or d <= 1:
            out.append(np.nan)
            continue

        try:
            out.append(required_variance_ratio(
                median_delay=m,
                dispersion=d,
                delta_u=delta_u,
                alpha=ALPHA,
                p_conv=P_CONV,
                max_delay=MAX_DELAY,
                dt_fine=DT_FINE,
            ))
        except Exception:
            out.append(np.nan)

    return np.asarray(out, float)


def misspec_log_relative_curve(marker, delta_u, misspec_grid, perturb="median"):
    y = misspec_curve(marker, delta_u, misspec_grid, perturb=perturb)
    idx0 = np.argmin(np.abs(misspec_grid))
    y0 = y[idx0]

    if not np.isfinite(y0) or y0 <= 0:
        return np.full_like(y, np.nan, dtype=float)

    out = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    out[mask] = np.log10(y[mask] / y0)
    return out


# ============================================================
# Build theoretical maps
# ============================================================
median_grid = np.linspace(MEDIAN_MIN, MEDIAN_MAX, N_MEDIAN)
dispersion_grid = np.linspace(DISP_MIN, DISP_MAX, N_DISP)

delta_u_3 = canonical_delta_u_periodic(T=T_SIGNAL, period_days=3.0, amp=AMP_SIGNAL)
delta_u_7 = canonical_delta_u_periodic(T=T_SIGNAL, period_days=7.0, amp=AMP_SIGNAL)
delta_u_14 = canonical_delta_u_periodic(T=T_SIGNAL, period_days=14.0, amp=AMP_SIGNAL)

crit_var_ratio_3 = build_critical_variance_ratio_map(
    median_grid, dispersion_grid, delta_u_3,
    alpha=ALPHA, p_conv=P_CONV,
    max_delay=MAX_DELAY, dt_fine=DT_FINE
)

crit_var_ratio_7 = build_critical_variance_ratio_map(
    median_grid, dispersion_grid, delta_u_7,
    alpha=ALPHA, p_conv=P_CONV,
    max_delay=MAX_DELAY, dt_fine=DT_FINE
)

crit_var_ratio_14 = build_critical_variance_ratio_map(
    median_grid, dispersion_grid, delta_u_14,
    alpha=ALPHA, p_conv=P_CONV,
    max_delay=MAX_DELAY, dt_fine=DT_FINE
)

log_ratio_3 = log10_safe(crit_var_ratio_3)
log_ratio_7 = log10_safe(crit_var_ratio_7)
log_ratio_14 = log10_safe(crit_var_ratio_14)

all_log_vals = np.concatenate([
    log_ratio_3[np.isfinite(log_ratio_3)],
    log_ratio_7[np.isfinite(log_ratio_7)],
    log_ratio_14[np.isfinite(log_ratio_14)],
])

VMIN = np.floor(np.nanpercentile(all_log_vals, 2))
VMAX = np.ceil(np.nanpercentile(all_log_vals, 98))
VMIN = min(VMIN, 0)
VMAX = max(VMAX, 0)

levels_fill = np.arange(VMIN, VMAX + 0.5, 0.5)
levels_line = np.arange(VMIN, VMAX + 1.0, 1.0)

# ============================================================
# Load empirical overlay points
# ============================================================
inc_support, inc_params_df = build_lessler2009_support()

inc_diseases = [
    "Influenza A",
    "Measles",
    "Respiratory syncytial virus (RSV)",
    "Rhinovirus",
]
inc_label_map = {
    "Influenza A": "Influenza",
    "Measles": "Measles",
    "Respiratory syncytial virus (RSV)": "RSV",
    "Rhinovirus": "Rhinovirus",
}

emp_subset, emp_summary_df = load_empirical_delay_subset(
    dist_family=DIST_FAMILY,
    max_delay_days=MAX_DELAY,
)

MAX_EMP = 12
emp_items = list(emp_subset.items())[:MAX_EMP]
emp_items = [(k, s) for (k, s) in emp_items if k not in DROP_KEY]

emp_markers = []

# Incubation delays from Lessler Table 3
for name in inc_diseases:
    d = inc_support[name]
    label = inc_label_map[name]

    try:
        median_delay = float(d.reported.p50)
        dispersion = float(d.reported.dispersion)
    except Exception as e:
        print("FAILED incubation:", label, e)
        continue

    if np.isfinite(median_delay) and np.isfinite(dispersion):
        emp_markers.append({
            "label": label,
            "group": "Exposure→onset",
            "median_delay": median_delay,
            "dispersion": dispersion,
        })

# Onset-to-report delays
seen = set()
for key, s in emp_items:
    raw = f"{s.dataset}" if hasattr(s, "dataset") else str(key)
    label = disease_from_dataset_name(raw)

    if ONE_POINT_PER_DISEASE and label in seen:
        continue

    dist = s.dist
    try:
        median_delay, dispersion = extract_median_dispersion_from_dist(dist)
    except Exception as e:
        print("FAILED empirical:", label, key, e)
        continue

    if np.isfinite(median_delay) and np.isfinite(dispersion):
        emp_markers.append({
            "label": label,
            "group": "Onset→report",
            "median_delay": median_delay,
            "dispersion": dispersion,
        })
        seen.add(label)

emp_markers = sorted(emp_markers, key=lambda x: (x["group"], x["median_delay"]))

print("\n=== EMPIRICAL DELAY POINTS (median days, dispersion factor) ===")
for d in emp_markers:
    print(
        f"{d['label']:12s}  "
        f"group={d['group']:16s}  "
        f"median={d['median_delay']:.2f}  "
        f"dispersion={d['dispersion']:.2f}"
    )

emp_medians = np.array([d["median_delay"] for d in emp_markers])
emp_disps = np.array([d["dispersion"] for d in emp_markers])

MEDIAN_MIN_PLOT = min(MEDIAN_MIN, np.nanmin(emp_medians) * 0.9)
MEDIAN_MAX_PLOT = max(MEDIAN_MAX, np.nanmax(emp_medians) * 1.1)
DISP_MIN_PLOT = min(DISP_MIN, np.nanmin(emp_disps) * 0.95)
DISP_MAX_PLOT = max(DISP_MAX, np.nanmax(emp_disps) * 1.05)

# ============================================================
# Plot
# ============================================================
# Designing
# directly at final size prevents text from becoming too small after scaling.
fig = plt.figure(figsize=(7.1, 5.55))

gs = fig.add_gridspec(
    3, 4,
    height_ratios=[1.10, 1.0, 1.0],
    width_ratios=[1, 1, 1, 0.25],
    left=0.145,
    right=0.995,
    bottom=0.02,
    top=0.955,
    wspace=0.2,
    hspace=0.25,
)

ax_h3  = fig.add_subplot(gs[0, 0])
ax_h7  = fig.add_subplot(gs[0, 1], sharex=ax_h3, sharey=ax_h3)
ax_h14 = fig.add_subplot(gs[0, 2], sharex=ax_h3, sharey=ax_h3)

# Use the right side as a compact information column.
cbar_gs = gs[0, 3].subgridspec(
    3, 3,
    height_ratios=[0.18, 0.64, 0.18],
    width_ratios=[1, 0.18, 1],
    wspace=0,
    hspace=0,
)
cax = fig.add_subplot(cbar_gs[1, 1])
legend_ax = fig.add_subplot(gs[1:, 3])
legend_ax.axis("off")

ax_m3  = fig.add_subplot(gs[1, 0])
ax_m7  = fig.add_subplot(gs[1, 1], sharex=ax_m3, sharey=ax_m3)
ax_m14 = fig.add_subplot(gs[1, 2], sharex=ax_m3, sharey=ax_m3)

ax_d3  = fig.add_subplot(gs[2, 0], sharex=ax_m3, sharey=ax_m3)
ax_d7  = fig.add_subplot(gs[2, 1], sharex=ax_m3, sharey=ax_m3)
ax_d14 = fig.add_subplot(gs[2, 2], sharex=ax_m3, sharey=ax_m3)

cmap = "viridis"

marker_colors = {
    "Influenza": "#1f77b4",
    "Measles": "#d62728",
    "RSV": "#2ca02c",
    "Rhinovirus": "#9467bd",
    "COVID-19": "#8c564b",
    "Dengue": "#17becf",
    "Ebola": "#e377c2",
    "Mpox": "#ff7f0e",
    "Other": "#4d4d4d",
}

# ============================================================
# Heatmaps
# ============================================================
heat_axes = [ax_h3, ax_h7, ax_h14]
heat_data = [log_ratio_3, log_ratio_7, log_ratio_14]

for ax, Z in zip(heat_axes, heat_data):
    cf = ax.contourf(
        median_grid, dispersion_grid, Z,
        levels=levels_fill,
        cmap=cmap,
        extend="both",
    )

    # Keep contour lines but remove contour labels.
    ax.contour(
        median_grid, dispersion_grid, Z,
        levels=levels_line,
        colors="white",
        linewidths=0.48,
        alpha=0.58,
    )

    for d in emp_markers:
        x0 = d["median_delay"]
        y0 = d["dispersion"]
        c = marker_colors.get(d["label"], "#4d4d4d")

        ax.scatter(
            x0, y0,
            s=34,
            marker="o",
            color=c,
            edgecolor="black",
            linewidth=0.55,
            zorder=10,
            clip_on=False,
        )

    ax.set_xlim(MEDIAN_MIN_PLOT, MEDIAN_MAX_PLOT)
    ax.set_ylim(DISP_MIN_PLOT, DISP_MAX_PLOT)
    ax.set_xlabel("Median delay (days)", labelpad=3, fontsize=8.0)
    ax.xaxis.set_major_locator(mticker.FixedLocator([5, 10, 15, 20]))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_box_aspect(0.80)
    style_ax(ax)

ax_h3.set_ylabel(
    r"Dispersion, $\exp(\sigma_{\log})$",
    labelpad=4,
)
ax_h7.tick_params(labelleft=False)
ax_h14.tick_params(labelleft=False)

# Vertical colorbar aligned with the heatmap row.
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label(
    '$\\log_{10}$(required\ncontrast-to-noise ratio)',
    fontsize=7,
    labelpad=16,
    rotation=270
)
cbar.ax.tick_params(labelsize=7.0, pad=1)

# ============================================================
# Lineplots
# ============================================================
x_pct = MISSPEC_GRID * 100.0

def plot_logrelative_misspec(
    ax, delta_u, perturb, show_ylabel=False, show_xlabel=False
):
    for d in emp_markers:
        y = misspec_log_relative_curve(
            d, delta_u, MISSPEC_GRID, perturb=perturb
        )
        c = marker_colors.get(d["label"], "#4d4d4d")

        ax.plot(
            x_pct, y,
            lw=1.35,
            color=c,
            alpha=0.92,
            label=d["label"],
        )

    ax.axhline(0, color="black", lw=0.80, alpha=0.80)
    ax.axvline(0, color="black", lw=0.70, alpha=0.55)

    ax.set_xlim(-30, 30)
    ax.xaxis.set_major_locator(mticker.FixedLocator([-30, -15, 0, 15, 30]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    if show_xlabel:
        ax.set_xlabel("Misspecification (%)", labelpad=3, fontsize=8.0)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(
            '$\\log_{10}$(required\ncontrast-to-noise ratio)',
            labelpad=4,
            fontsize=8
        )
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.set_box_aspect(0.80)
    ax.tick_params(axis="x", labelsize=7.2)
    style_ax(ax)

# Row 2: median misspecification
plot_logrelative_misspec(ax_m3,  delta_u_3,  "median", True,  True)
plot_logrelative_misspec(ax_m7,  delta_u_7,  "median", False, True)
plot_logrelative_misspec(ax_m14, delta_u_14, "median", False, True)

# Row 3: dispersion misspecification
plot_logrelative_misspec(ax_d3,  delta_u_3,  "dispersion", True,  True)
plot_logrelative_misspec(ax_d7,  delta_u_7,  "dispersion", False, True)
plot_logrelative_misspec(ax_d14, delta_u_14, "dispersion", False, True)

# Preserve panel dimensions/aspect while keeping modest separation by row.
for ax in [ax_h3, ax_h7, ax_h14]:
    ax.set_anchor("S")
for ax in [ax_m3, ax_m7, ax_m14]:
    ax.set_anchor("C")
for ax in [ax_d3, ax_d7, ax_d14]:
    ax.set_anchor("N")

# Match the colorbar to the heatmaps' actual post-aspect height.
fig.canvas.draw()
heat_pos = ax_h14.get_position()
CBAR_SHIFT_LEFT = 0.012
cbar_pos = cax.get_position()
cax.set_position([
    cbar_pos.x0 - CBAR_SHIFT_LEFT,
    heat_pos.y0,
    cbar_pos.width,
    heat_pos.height,
])

# Harmonize y limits across relative lineplots
line_axes = [ax_m3, ax_m7, ax_m14, ax_d3, ax_d7, ax_d14]
all_y = []
for ax in line_axes:
    for line in ax.lines:
        yy = np.asarray(line.get_ydata(), dtype=float)
        yy = yy[np.isfinite(yy)]
        if len(yy):
            all_y.append(yy)

if all_y:
    all_y = np.concatenate(all_y)
    ymax_abs = np.nanpercentile(np.abs(all_y), 99)
    ymax_abs = max(0.25, float(np.ceil(ymax_abs * 10) / 10))
    for ax in line_axes:
        ax.set_ylim(-ymax_abs, ymax_abs)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

# Shared column headers describe perturbation duration across all rows.
for ax, title in zip(
    [ax_h3, ax_h7, ax_h14],
    [
        "3-day\nperturbation",
        "7-day\nperturbation",
        "14-day\nperturbation",
    ],
):
    pos = ax.get_position()
    fig.text(
        0.5 * (pos.x0 + pos.x1),
        pos.y1 + 0.025,
        title,
        ha="center",
        va="bottom",
        multialignment="center",
        fontsize=10.0,
        fontweight="bold",
        linespacing=1.0,
    )

# Bold row headers describe the analysis represented by each row.
row_title_x = 0.04
for ax, title in [
    (ax_h3, "Sensitivity\nsurface"),
    (ax_m3, "Median\nmisspecification"),
    (ax_d3, "Dispersion\nmisspecification"),
]:
    pos = ax.get_position()
    fig.text(
        row_title_x,
        0.5 * (pos.y0 + pos.y1),
        title,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.5,
        fontweight="bold",
        linespacing=1.05,
    )

panel_axes = [
    ax_h3, ax_h7, ax_h14,
    ax_m3, ax_m7, ax_m14,
    ax_d3, ax_d7, ax_d14,
]
for ax, label in zip(panel_axes, "ABCDEFGHI"):
    ax.text(
        -0.10, 1.02, label,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=20,
    )

# ============================================================
# Disease-color legend
# ============================================================
handles, labels = ax_m3.get_legend_handles_labels()
unique = {}
for h, lab in zip(handles, labels):
    if lab not in unique:
        unique[lab] = h

disease_legend = legend_ax.legend(
    unique.values(),
    unique.keys(),
    loc="center left",
    bbox_to_anchor=(0.0, 0.5),
    ncol=1,
    frameon=False,
    fontsize=7.2,
    labelspacing=0.65,
    handlelength=1.25,
    handletextpad=0.35,
    borderaxespad=0.0,
)

OUTDIR.mkdir(parents=True, exist_ok=True)
plt.savefig(
    OUTFILE,
    bbox_inches="tight",
    pad_inches=0.01,
    facecolor="white",
)
PNG_OUTFILE = OUTFILE.with_suffix(".png")
plt.savefig(
    PNG_OUTFILE,
    bbox_inches="tight",
    pad_inches=0.01,
    facecolor="white",
)
plt.close(fig)
print(f"Saved figure to: {OUTFILE}")
print(f"Saved preview to: {PNG_OUTFILE}")
