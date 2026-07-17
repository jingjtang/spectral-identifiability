#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText

from get_incubation_periods import build_lessler2009_support
from utils import load_empirical_delay_subset


# ============================================================
# Helpers
# ============================================================

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



def continuous_pdf_to_power(pdf_func, max_delay, grid_step=0.05, freq_max=0.5):
    """
    Sample a continuous PDF on [0, max_delay], normalize area,
    then compute |G(f)|^2 via rFFT.
    """
    t = np.arange(0, max_delay + grid_step, grid_step)
    y = pdf_func(t)

    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)

    area = np.trapezoid(y, t)
    if area <= 0:
        freq = np.fft.rfftfreq(len(t), d=grid_step)
        power = np.zeros_like(freq)
        mask = freq <= freq_max
        return freq[mask], power[mask]

    y /= area

    G = np.fft.rfft(y)
    power = np.abs(G) ** 2
    freq = np.fft.rfftfreq(len(t), d=grid_step)

    if power[0] > 0:
        power /= power[0] # numerical correction

    power = np.maximum(power, 1e-12)
    mask = freq <= freq_max
    return freq[mask], power[mask]

def normalize_pmf(x):
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 0:
        raise ValueError("PMF sum <= 0")
    return x / s


def pdf_to_discrete_pmf(pdf_func, max_delay=120, dt=0.05):
    """
    Convert a continuous delay PDF into a daily discrete PMF on bins
    [0,1), [1,2), ..., [max_delay-1, max_delay).
    """
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


def dtft_power(g, freq_grid):
    """
    Evaluate |G(f)|^2 for a discrete delay PMF g on an arbitrary frequency grid.
    Frequency f is in cycles/day.
    """
    g = normalize_pmf(g)
    k = np.arange(len(g), dtype=float)

    # shape: (n_freq, n_k)
    expo = np.exp(-2j * np.pi * np.outer(freq_grid, k))
    G = expo @ g
    power = np.abs(G) ** 2
    return power


def required_snr_curve_single_frequency(
    g,
    T_grid,
    alpha=0.2,
    p_conv=1.0,
    eps=1e-15,
):
    """
    Single-frequency identifiability threshold based directly on

        J(f_0) = p^2 |G(f_0)|^2 * SNR(f_0),

    with f_0 = 1 / T. For the testing threshold

        J >= 4 (1 - alpha)^2,

    the minimum required frequency-domain SNR is

        SNR_min(f_0) = 4 (1 - alpha)^2 / (p^2 |G(f_0)|^2).

    Here SNR(f_0) is interpreted consistently with the paper as the
    whole-window Fourier-domain signal-to-noise ratio at frequency f_0.
    """
    T_grid = np.asarray(T_grid, float)
    freq_grid = 1.0 / T_grid

    power = dtft_power(g, freq_grid)
    j_thresh = 4.0 * (1.0 - alpha) ** 2

    snr_min = j_thresh / np.maximum((p_conv ** 2) * power, eps)
    return snr_min, freq_grid, power


def _unique_legend(handles, labels):
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    return list(seen.values()), list(seen.keys())


def add_frequency_reference_lines(ax):
    refs = [
        (1 / 7, "7-day\nweekly"),
        (1 / 3, "3-day"),
    ]

    for f, label in refs:
        ax.axvline(f, color="0.6", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(
            f, 0.02, label,
            rotation=90,
            ha="right", va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="0.35",
        )


def add_power_threshold(ax, level=0.1):
    ax.axhline(level, color="0.5", linestyle=":", linewidth=1.0, zorder=0)
    ax.text(
        0.99, level,
        "10% power",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom",
        fontsize=8,
        color="0.4",
    )


def dist_mean_iqr(dist):
    mean = float(dist.mean())
    q25 = float(dist.ppf(0.25))
    q75 = float(dist.ppf(0.75))
    iqr = q75 - q25
    return mean, iqr


def add_stats_inset(ax, items, *, title="Summary", max_lines=4):
    lines = [title]
    for label, dist in items[:max_lines]:
        m, iqr = dist_mean_iqr(dist)
        short = (label[:22] + "…") if len(label) > 23 else label
        lines.append(f"{short}: {m:.1f} (IQR {iqr:.1f})")

    at = AnchoredText(
        "\n".join(lines),
        loc="upper left",
        prop=dict(size=7),
        frameon=True,
        borderpad=0.25,
        pad=0.25,
    )
    at.patch.set_alpha(0.08)
    at.patch.set_linewidth(0.6)
    ax.add_artist(at)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


def add_panel_labels_outside(fig, axes_list, labels, *, dx=0.030, dy=0.010):
    for lab, ax in zip(labels, axes_list):
        bb = ax.get_position()
        fig.text(
            bb.x0 - dx, bb.y1 + dy, lab,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold",
        )


# ============================================================
# Build supports
# ============================================================

# Column 1: incubation
inc_support, inc_params_df = build_lessler2009_support()

# Keep your current 4 diseases from panel A
inc_diseases = [
    "Influenza A",
    "Measles",
    "Respiratory syncytial virus (RSV)",
    "Rhinovirus",
]
label_to_names = {
    "Influenza A": "Influenza",
    "Measles": "Measles",
    "Respiratory syncytial virus (RSV)": "RSV",
    "Rhinovirus": "Rhinovirus",
}

# Column 2: empirical onset->report
emp_subset, emp_summary_df = load_empirical_delay_subset(
    dist_family="lognormal",
    max_delay_days=120,
)

MAX_EMP = 12
emp_items = list(emp_subset.items())[:MAX_EMP]

DROP_KEY = [
    "uscdc_linelist_covid::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
    "xu2020_covid::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
    "hinch2024_uk_mpox::symptom_onset_to_report::date_symptom_onset->date_case_report::lognormal",
]
emp_items = [(k, s) for (k, s) in emp_items if k not in DROP_KEY]


# ============================================================
# Styling / palettes
# ============================================================

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

left_labels = [label_to_names[name] for name in inc_diseases]

right_labels = []
for key, s in emp_items:
    raw = f"{s.dataset}" if hasattr(s, "dataset") else str(key)
    lab = disease_from_dataset_name(raw)
    if lab not in right_labels:
        right_labels.append(lab)

left_palette = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
]

right_palette = [
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#e377c2",  # pink
]

left_color_map = {lab: col for lab, col in zip(left_labels, left_palette)}
right_color_map = {lab: col for lab, col in zip(right_labels, right_palette)}


# ============================================================
# Global settings
# ============================================================

# Row 1 / Row 2
T_MAX = 60
T_PDF_MAX = 30
F_MAX = 0.5
GRID_STEP = 0.05

# -----------------------------
# Row 3: identifiability threshold settings
# Row 3: single-frequency threshold curve
# Using the paper-level identity
#     J(f_0) = p^2 |G(f_0)|^2 * SNR(f_0)
# with f_0 = 1 / T.
# -----------------------------
ALPHA = 0.2
P_CONV = 1.0
PERIOD_MIN = 2.0
PERIOD_MAX = 30.0
N_PERIOD = 400
MAX_DELAY_IDENT = 120
DT_FINE_IDENT = 0.05
P_LIST = [1.0, 0.2]

period_grid = np.linspace(PERIOD_MIN, PERIOD_MAX, N_PERIOD)
x = np.linspace(1e-6, T_MAX, 2000)


# ============================================================
# Figure layout
# ============================================================

FIG_W, FIG_H = 8.2, 4.6
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(
    2, 3, figure=fig,
    left=0.08, right=0.98, top=0.90,
    bottom=0.12,
    wspace=0.35, hspace=0.42,
)

ax_pdf_inc  = fig.add_subplot(gs[0, 0])
ax_pow_inc  = fig.add_subplot(gs[0, 1])
ax_rmin_inc = fig.add_subplot(gs[0, 2])

ax_pdf_emp  = fig.add_subplot(gs[1, 0])
ax_pow_emp  = fig.add_subplot(gs[1, 1])
ax_rmin_emp = fig.add_subplot(gs[1, 2])


col1_x = 0.5 * (ax_pdf_inc.get_position().x0 + ax_pdf_inc.get_position().x1)
col2_x = 0.5 * (ax_pow_inc.get_position().x0 + ax_pow_inc.get_position().x1)
col3_x = 0.5 * (ax_rmin_inc.get_position().x0 + ax_rmin_inc.get_position().x1)
top_y = max(ax_pdf_inc.get_position().y1, ax_pow_inc.get_position().y1, ax_rmin_inc.get_position().y1)

fig.text(col1_x, top_y + 0.045, "Delay distribution",
         ha="center", va="bottom", fontsize=12, fontweight="bold")
fig.text(col2_x, top_y + 0.045, "Frequency response",
         ha="center", va="bottom", fontsize=12, fontweight="bold")
fig.text(col3_x, top_y + 0.045, "Identifiability thresholds",
         ha="center", va="bottom", fontsize=12, fontweight="bold")

row1_y = 0.5 * (ax_pdf_inc.get_position().y0 + ax_pdf_inc.get_position().y1)
row2_y = 0.5 * (ax_pdf_emp.get_position().y0 + ax_pdf_emp.get_position().y1)
left_x = ax_pdf_inc.get_position().x0 - 0.08

fig.text(left_x, row1_y, "Exposure-to-onset",
         ha="center", va="center", rotation=90,
         fontsize=12, fontweight="bold")

fig.text(left_x, row2_y, "Onset-to-report",
         ha="center", va="center", rotation=90,
         fontsize=12, fontweight="bold")

# ============================================================
# Labels
# ============================================================

ax_pdf_inc.set_xlabel("Delay (days)", labelpad=1)
ax_pdf_emp.set_xlabel("Delay (days)", labelpad=1)
ax_pdf_inc.set_ylabel("Density")
ax_pdf_emp.set_ylabel("Density")

ax_pow_inc.set_xlabel(r"Frequency (day$^{-1}$)")
ax_pow_emp.set_xlabel(r"Frequency (day$^{-1}$)")
ax_pow_inc.set_ylabel("Squared magnitude")
ax_pow_emp.set_ylabel("Squared magnitude")
ax_pow_inc.set_yscale("log")
ax_pow_emp.set_yscale("log")

ax_rmin_inc.set_xlabel(r"Perturbation period $T$ (days)")
ax_rmin_emp.set_xlabel(r"Perturbation period $T$ (days)")
# ax_rmin_inc.set_ylabel("Contrast-to-noise\npower ratio", labelpad=2)
# ax_rmin_emp.set_ylabel("Contrast-to-noise\npower ratio", labelpad=2)
ax_rmin_inc.set_ylabel("Contrast-to-noise Ratio", labelpad=2)
ax_rmin_emp.set_ylabel("Contrast-to-noise Ratio", labelpad=2)
ax_rmin_inc.set_yscale("log")
ax_rmin_emp.set_yscale("log")


# ============================================================
# Row 1 + Row 2: left column
# ============================================================

for name in inc_diseases:
    d = inc_support[name]
    label = label_to_names[name]
    color = left_color_map[label]

    y = np.asarray(d.dist.pdf(x), dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    ax_pdf_inc.plot(x, y, linewidth=1.8, label=label, color=color)

    f, p = continuous_pdf_to_power(
        d.dist.pdf,
        T_MAX,
        grid_step=GRID_STEP,
        freq_max=F_MAX,
    )
    ax_pow_inc.plot(f, p, linewidth=1.8, label=label, color=color)

    g = pdf_to_discrete_pmf(
        d.dist.pdf,
        max_delay=MAX_DELAY_IDENT,
        dt=DT_FINE_IDENT,
    )
    for p_val, ls in zip(P_LIST, ["-", "--"]):
        rmin, _, _ = required_snr_curve_single_frequency(
            g=g,
            T_grid=period_grid,
            alpha=ALPHA,
            p_conv=p_val,
        )

        # ===== DEBUG: check scaling of p =====

        if p_val == 1.0:
            rmin_p1 = rmin.copy()

        if p_val == 0.2:
            ratio = rmin / rmin_p1

            print("\nDEBUG ratio (p=0.2 / p=1.0):")
            print("median ratio:", np.median(ratio))
            print("min ratio:", np.min(ratio))
            print("max ratio:", np.max(ratio))


        alpha_val = 1.0 if p_val == 1.0 else 0.6
        lw_val = 1.8 if p_val == 1.0 else 1.4

        ax_rmin_inc.plot(
            period_grid, rmin,
            linewidth=1.8,
            linestyle=ls,
            color=color,
            alpha=alpha_val,
            label=label if p_val == 1.0 else None,
        )

    for T_target in [14, 21, 28]:
        idx = np.argmin(np.abs(period_grid - T_target))
        print(name)
        print(
            f"T={period_grid[idx]:.1f}, "
            f"p=1: {rmin_p1[idx]:.6g}, "
            f"p=0.2: {rmin[idx]:.6g}, "
            f"ratio: {rmin[idx] / rmin_p1[idx]:.2f}"
        )


# ============================================================
# Row 1 + Row 2 + Row 3: right column
# ============================================================

for key, s in emp_items:
    raw = f"{s.dataset}" if hasattr(s, "dataset") else str(key)
    label = disease_from_dataset_name(raw)
    color = right_color_map[label]

    y = np.asarray(s.dist.pdf(x), dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    ax_pdf_emp.plot(x, y, linewidth=1.8, label=label, color=color)

    f, p = continuous_pdf_to_power(
        s.dist.pdf,
        T_MAX,
        grid_step=GRID_STEP,
        freq_max=F_MAX,
    )
    ax_pow_emp.plot(f, p, linewidth=1.8, label=label, color=color)

    g = pdf_to_discrete_pmf(
        s.dist.pdf,
        max_delay=MAX_DELAY_IDENT,
        dt=DT_FINE_IDENT,
    )
    for p_val, ls in zip(P_LIST, ["-", "--"]):
        rmin, _, _ = required_snr_curve_single_frequency(
            g=g,
            T_grid=period_grid,
            alpha=ALPHA,
            p_conv=p_val,
        )

        alpha_val = 1.0 if p_val == 1.0 else 0.6
        lw_val = 1.8 if p_val == 1.0 else 1.4

        ax_rmin_emp.plot(
            period_grid, rmin,
            linewidth=1.8,
            linestyle=ls,
            color=color,
            alpha=alpha_val,
            label=label if p_val == 1.0 else None,
        )


# ============================================================
# Shared axis ranges
# ============================================================

for ax in (ax_pdf_inc, ax_pdf_emp):
    ax.set_xlim(0, T_PDF_MAX)
    ax.set_xticks([0, 7, 14, 28])

for ax in (ax_pow_inc, ax_pow_emp):
    ax.set_xlim(0, F_MAX)

for ax in (ax_rmin_inc, ax_rmin_emp):
    ax.set_xlim(0, PERIOD_MAX)
    ax.set_xticks([0, 2, 3, 5, 7, 10, 14, 21, 28])

# Shared y within each row
pdf_ymax = max(ax_pdf_inc.get_ylim()[1], ax_pdf_emp.get_ylim()[1])
ax_pdf_inc.set_ylim(0, pdf_ymax)
ax_pdf_emp.set_ylim(0, pdf_ymax)

pow_ymin = min(ax_pow_inc.get_ylim()[0], ax_pow_emp.get_ylim()[0])
pow_ymax = max(ax_pow_inc.get_ylim()[1], ax_pow_emp.get_ylim()[1])
ax_pow_inc.set_ylim(pow_ymin, pow_ymax)
ax_pow_emp.set_ylim(pow_ymin, pow_ymax)

# Better shared y for R_min using actual positive finite values
rmin_vals = []
for ax in (ax_rmin_inc, ax_rmin_emp):
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), float)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            rmin_vals.append(y)

if rmin_vals:
    rmin_all = np.concatenate(rmin_vals)
    rmin_ymin = max(np.nanpercentile(rmin_all, 1), 1e-3)
    rmin_ymax = np.nanpercentile(rmin_all, 99)
    if np.isfinite(rmin_ymin) and np.isfinite(rmin_ymax) and rmin_ymax > rmin_ymin:
        ax_rmin_inc.set_ylim(rmin_ymin, rmin_ymax)
        ax_rmin_emp.set_ylim(rmin_ymin, rmin_ymax)


# ============================================================
# Reference lines / styling
# ============================================================

add_frequency_reference_lines(ax_pow_inc)
add_frequency_reference_lines(ax_pow_emp)
add_power_threshold(ax_pow_inc, level=0.1)
add_power_threshold(ax_pow_emp, level=0.1)

for ax in (ax_pdf_inc, ax_pdf_emp, ax_pow_inc, ax_pow_emp, ax_rmin_inc, ax_rmin_emp):
    _style_axes(ax)
    ax.grid(True, alpha=0.25)


# ============================================================
# Legends
# ============================================================

hL, lL = ax_pdf_inc.get_legend_handles_labels()
hL, lL = _unique_legend(hL, lL)

hR, lR = ax_pdf_emp.get_legend_handles_labels()
hR, lR = _unique_legend(hR, lR)

if ax_pdf_inc.get_legend() is not None:
    ax_pdf_inc.get_legend().remove()
if ax_pdf_emp.get_legend() is not None:
    ax_pdf_emp.get_legend().remove()

ax_pdf_inc.legend(
    hL, lL,
    loc="upper right",
    frameon=False,
    ncol=1,
    handlelength=2.0,
    columnspacing=0.8,
    handletextpad=0.5,
    fontsize=9.5,
    borderaxespad=0.0,
)

ax_pdf_emp.legend(
    hR, lR,
    loc="upper right",
    frameon=False,
    ncol=1,
    handlelength=2.0,
    columnspacing=0.8,
    handletextpad=0.5,
    fontsize=9.5,
    borderaxespad=0.0,
)

from matplotlib.lines import Line2D

style_handles = [
    Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="p = 1.0"),
    Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="p = 0.2"),
]

ax_rmin_inc.legend(
    handles=style_handles,
    loc="upper right",
    frameon=False,
)
# ============================================================
# Panel labels
# ============================================================

add_panel_labels_outside(
    fig,
    [ax_pdf_inc, ax_pow_inc, ax_rmin_inc, ax_pdf_emp, ax_pow_emp, ax_rmin_emp],
    ["A", "B", "C", "D", "E", "F"],
    dx=0.030,
    dy=0.010,
)


# ============================================================
# Save
# ============================================================

outpath = "../figs/empirical_delays_3x2_with_rmin.pdf"
Path("../figs").mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"Saved to: {outpath}")
