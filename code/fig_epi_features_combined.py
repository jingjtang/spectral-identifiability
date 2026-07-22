#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a combined delay-identifiability and epidemiological-feature figure.

This script is intentionally separate from the existing figure scripts. It
reuses the same data, delay kernel, trend-filter fit, cutoff reconstructions,
and local likelihood-weighted bands used by fig_epi_features.py, then lays them
out as a three-row PNAS-style combined figure.
"""

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import utils as base


DEFAULT_OUT_DIR = Path("../figs")
CUTOFFS = [1 / 60, 1 / 14, 1 / 7]
FULL_REFERENCE_KEY = "no_cutoff"
CUTOFF_COLORS = {
    1 / 60: "#4C78A8",
    1 / 14: "#B279A2",
    1 / 7: "#F58518",
}
FULL_REFERENCE_COLOR = "#111111"
FULL_REFERENCE_LW = 0.50
FULL_REFERENCE_ALPHA = 0.72

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the combined US epi-feature / delay figure."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default="2020-04-01")
    parser.add_argument("--end", default="2021-06-30")
    parser.add_argument("--display-start", default="2020-09-01")
    parser.add_argument("--display-end", default="2021-03-31")
    parser.add_argument("--geo-type", default="nation")
    parser.add_argument("--geo-value", default="us")
    parser.add_argument("--label", default="US")
    return parser.parse_args()


def load_delay_pmf():
    try:
        g = base.load_empirical_delay_pmf()
        print(f"Loaded empirical delay: {base.DEFAULT_ONSET_REPORT_DELAY_KEY}")
        return g
    except Exception as exc:
        print("Warning: failed to load empirical delay; using fixed lognormal.")
        print("Reason:", repr(exc))
        return base.load_empirical_delay_pmf(fallback_lognormal=True)


def one_sided_rfft_weights(T):
    weights = np.ones(T // 2 + 1, dtype=float)
    if T % 2 == 0:
        if len(weights) > 2:
            weights[1:-1] = 2.0
    else:
        if len(weights) > 1:
            weights[1:] = 2.0
    return weights


def delay_power(g, T):
    padded = np.zeros(T, dtype=float)
    padded[: min(T, len(g))] = g[: min(T, len(g))]
    return np.abs(np.fft.rfft(padded)) ** 2


def ordered_reference_legend(ax, *, observed_last=False):
    handles, labels = ax.get_legend_handles_labels()
    observed = [(h, lab) for h, lab in zip(handles, labels) if lab == "Observed"]
    no_cutoff = [(h, lab) for h, lab in zip(handles, labels) if "no\\ cutoff" in lab]
    cutoff = [
        (h, lab)
        for h, lab in zip(handles, labels)
        if lab != "Observed" and "no\\ cutoff" not in lab
    ]
    if observed_last:
        ordered = cutoff + no_cutoff + observed
    else:
        ordered = observed + cutoff + no_cutoff
    return [h for h, _ in ordered], [lab for _, lab in ordered]


def contrast_power(x0, x1):
    x0 = np.asarray(x0, float)
    x1 = np.asarray(x1, float)
    T = len(x0)
    scale = one_sided_rfft_weights(T) / T
    return scale * np.abs(np.fft.rfft(x1 - x0)) ** 2


def weakest_boundary_perturbation(g, weight, f_cut, tau):
    T = len(weight)
    Phi, _, _ = base.build_cutoff_design_matrix(T=T, f_cut=f_cut, dt=1.0)
    K = base.build_convolution_matrix(g, T)
    A = K @ Phi
    Aw = A * np.sqrt(np.asarray(weight, float))[:, None]
    info = np.asarray(Aw.T @ Aw, float)
    info = 0.5 * (info + info.T)
    eigvals, eigvecs = np.linalg.eigh(info)
    positive = eigvals > max(float(np.max(eigvals)) * 1e-10, 1e-12)
    if not np.any(positive):
        raise RuntimeError("No positive likelihood-information direction found.")
    idx = np.where(positive)[0][0]
    v = eigvecs[:, idx]
    scale = np.sqrt(tau / max(float(v @ info @ v), 1e-12))
    delta_u = Phi @ (scale * v)
    delta_m = np.asarray(K @ delta_u).reshape(-1)
    return delta_u, delta_m


def build_payload(args):
    g = load_delay_pmf()
    base.set_epidata_api_key()

    from epidatpy import EpiDataContext, EpiRange

    epidata = EpiDataContext(use_cache=False)
    df_grid = [10, 15, 20, 30, 40, 60, 80, 100, 120]
    tau = base.tau_from_alpha(0.20)

    print(f"\n=== Building {args.label} ({args.geo_type}={args.geo_value}) ===")
    df = base.fetch_signal_df(
        epidata,
        data_source="jhu-csse",
        signals="confirmed_incidence_num",
        time_values=EpiRange(args.start, args.end),
        geo_type=args.geo_type,
        geo_values=args.geo_value,
    )
    if df.empty or not df["y"].notna().any():
        raise ValueError(f"No case-incidence data returned for {args.label}")

    aligned = base.align_daily(df, start=args.start, end=args.end)
    dates = aligned["date"].values
    y = aligned["y"].values.astype(float)
    display_mask = (
        (pd.to_datetime(dates) >= pd.Timestamp(args.display_start))
        & (pd.to_datetime(dates) <= pd.Timestamp(args.display_end))
    )
    if not np.any(display_mask):
        raise ValueError("The display interval does not overlap the analysis interval.")

    joint = base.select_best_df_and_noise(
        y,
        df_grid=df_grid,
        ord=2,
        criterion="AIC",
        rscript_bin="Rscript",
    )
    mu_hat = joint.mu_hat
    likelihood_weight = 1.0 / base.get_variance_scale(
        mu_hat[display_mask], joint.best_noise
    )
    print(
        f"{args.label}: df_target={joint.df_target}, df_used={joint.df_used:.2f}, "
        f"noise={joint.best_noise.model}, median local weight={np.median(likelihood_weight):.3g}"
    )

    T_display = int(np.sum(display_mask))
    references, downstream, bands, growth = {}, {}, {}, {}
    edge_upstream, edge_downstream = {}, {}
    feature_rows = []
    dates_display = pd.to_datetime(dates[display_mask])

    full_upstream, full_fitted, _, _, _ = base.compute_cutoff_specific_reference_strict(
        y=y,
        mu_hat=mu_hat,
        g=g,
        f_cut=0.5,
        ridge_theta=1e-6,
    )
    references[FULL_REFERENCE_KEY] = full_upstream
    downstream[FULL_REFERENCE_KEY] = full_fitted
    growth[FULL_REFERENCE_KEY] = base.smooth_reflect(
        base.rolling_growth_rate(full_upstream, horizon=7, log_offset=1.0),
        window=5,
    )
    displayed_full = np.asarray(full_upstream[display_mask], float)
    peak_smoothed_full = base.smooth_reflect(displayed_full, window=14)
    peak_idx_full = int(np.nanargmax(peak_smoothed_full))
    feature_rows.append({
        "cutoff": "No cutoff",
        "f_cut": np.nan,
        "sort_order": 1.0,
        "peak_day_index": peak_idx_full,
        "peak_magnitude": float(displayed_full[peak_idx_full]),
    })

    for fc in CUTOFFS:
        upstream, fitted, theta, _, _ = base.compute_cutoff_specific_reference_strict(
            y=y,
            mu_hat=mu_hat,
            g=g,
            f_cut=fc,
            ridge_theta=1e-6,
        )
        band_display = base.analytic_band_width_likelihood(
            T=T_display,
            g=g,
            weight=likelihood_weight,
            f_cut=fc,
            tau=tau,
        )
        band = np.full_like(y, np.nan, dtype=float)
        band[display_mask] = band_display
        delta_u_edge, delta_m_edge = weakest_boundary_perturbation(
            g=g,
            weight=likelihood_weight,
            f_cut=fc,
            tau=tau,
        )
        delta_u_edge_full = np.full_like(y, np.nan, dtype=float)
        delta_m_edge_full = np.full_like(y, np.nan, dtype=float)
        delta_u_edge_full[display_mask] = delta_u_edge
        delta_m_edge_full[display_mask] = delta_m_edge

        references[fc] = upstream
        downstream[fc] = fitted
        bands[fc] = band
        edge_upstream[fc] = delta_u_edge_full
        edge_downstream[fc] = delta_m_edge_full
        growth[fc] = base.smooth_reflect(
            base.rolling_growth_rate(upstream, horizon=7, log_offset=1.0),
            window=5,
        )

        displayed = np.asarray(upstream[display_mask], float)
        peak_smoothed = base.smooth_reflect(displayed, window=14)
        peak_idx = int(np.nanargmax(peak_smoothed))
        feature_rows.append({
            "cutoff": base.cutoff_label(fc),
            "f_cut": fc,
            "sort_order": fc,
            "peak_day_index": peak_idx,
            "peak_magnitude": float(displayed[peak_idx]),
        })
        print(
            f"{args.label}: cutoff={base.cutoff_label(fc)}, "
            f"basis_terms={len(theta)}, mean band width={np.mean(2 * band_display):.3g}"
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_df["relative_peak_timing"] = (
        feature_df["peak_day_index"] / max(len(dates_display) - 1, 1)
    )
    max_peak = max(float(feature_df["peak_magnitude"].max()), 1e-12)
    feature_df["relative_peak_magnitude"] = feature_df["peak_magnitude"] / max_peak

    arrays = [y, mu_hat]
    for mapping in (references, downstream, growth):
        arrays.append(mapping[FULL_REFERENCE_KEY])
    for mapping in (references, downstream, bands, growth, edge_upstream, edge_downstream):
        arrays.extend(mapping[fc] for fc in CUTOFFS)
    sliced = base.slice_date_range(
        dates, *arrays, start=args.display_start, end=args.display_end
    )
    dates_plot = pd.to_datetime(sliced[0])
    values = iter(sliced[1:])
    y_plot, mu_plot = next(values), next(values)
    sliced_maps = []
    full_refs = {
        "references": next(values),
        "downstream": next(values),
        "growth": next(values),
    }
    for _ in range(6):
        sliced_maps.append({fc: next(values) for fc in CUTOFFS})

    return {
        "label": args.label,
        "g": g,
        "dates_plot": dates_plot,
        "y_plot": y_plot,
        "mu_plot": mu_plot,
        "references": sliced_maps[0],
        "downstream": sliced_maps[1],
        "bands": sliced_maps[2],
        "growth": sliced_maps[3],
        "full_reference": full_refs,
        "edge_upstream": sliced_maps[4],
        "edge_downstream": sliced_maps[5],
        "feature_df": feature_df,
        "likelihood_weight": likelihood_weight,
    }


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.8,
        "axes.titlesize": 8.6,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_combined(payload, out_png, out_pdf):
    setup_style()
    dates = payload["dates_plot"]
    y = payload["y_plot"]
    mu = payload["mu_plot"]
    g = base.normalize_pmf(payload["g"])
    refs = payload["references"]
    downstream = payload["downstream"]
    bands = payload["bands"]
    growth = payload["growth"]
    full_reference = payload["full_reference"]
    feature_df = payload["feature_df"].sort_values("sort_order", ascending=False)
    likelihood_weight = payload["likelihood_weight"]

    T = len(dates)
    freqs = np.fft.rfftfreq(T, d=1.0)
    keep = (freqs > 0) & (freqs <= 0.5)
    A2 = delay_power(g, T)
    variance_reference = 2.0 * float(np.median(1.0 / likelihood_weight))
    eps = 1e-12

    upstream_contrast_power = {
        fc: contrast_power(full_reference["references"], refs[fc]) for fc in CUTOFFS
    }
    downstream_contrast_power = {
        fc: contrast_power(full_reference["downstream"], downstream[fc]) for fc in CUTOFFS
    }
    contrast_values_for_ylim = []
    for fc in CUTOFFS:
        contrast_values_for_ylim.extend(upstream_contrast_power[fc][keep])
        contrast_values_for_ylim.extend(downstream_contrast_power[fc][keep])
    contrast_values_for_ylim.append(variance_reference)
    contrast_values_for_ylim = np.asarray(contrast_values_for_ylim, float)
    contrast_values_for_ylim = contrast_values_for_ylim[
        np.isfinite(contrast_values_for_ylim) & (contrast_values_for_ylim > 0)
    ]
    shared_contrast_ylim = (
        10 ** np.floor(np.log10(float(np.min(contrast_values_for_ylim)))),
        10 ** np.ceil(np.log10(float(np.max(contrast_values_for_ylim)))),
    )

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(
        3,
        3,
        left=0.125,
        right=0.985,
        bottom=0.145,
        top=0.920,
        wspace=0.48,
        hspace=0.62,
        width_ratios=[2.0, 1.0, 2.0],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    axG = fig.add_subplot(gs[2, 0])
    gsH = gs[2, 1:].subgridspec(
        2,
        2,
        height_ratios=[0.16, 0.84],
        width_ratios=[1.0, 1.0],
        hspace=0.02,
        wspace=0.38,
    )
    axH_title = fig.add_subplot(gsH[0, :])
    axH_left = fig.add_subplot(gsH[1, 0])
    axH_right = fig.add_subplot(gsH[1, 1])
    axH_title.axis("off")

    # Row 1: time-domain reconstruction, delay, downstream fit.
    ymax = 0.0
    ymax = max(ymax, float(np.nanmax(full_reference["references"])))
    axA.plot(
        dates,
        full_reference["references"],
        color=FULL_REFERENCE_COLOR,
        lw=FULL_REFERENCE_LW,
        alpha=FULL_REFERENCE_ALPHA,
        label=r"$U^{\rm ref}_{\rm no\ cutoff}$",
        zorder=6,
    )
    for fc in CUTOFFS:
        Ufc = np.asarray(refs[fc], float)
        Bfc = np.asarray(bands[fc], float)
        lo = np.maximum(Ufc - Bfc, 0.0)
        hi = Ufc + Bfc
        ymax = max(ymax, float(np.nanmax(hi)))
        axA.fill_between(dates, lo, hi, color=CUTOFF_COLORS[fc], alpha=0.14)
        axA.plot(
            dates,
            Ufc,
            color=CUTOFF_COLORS[fc],
            lw=1.15,
            label=rf"$U^{{\rm ref}}_{{f_c={base.cutoff_label(fc)}}}$",
            zorder=3,
        )
    axA.set_ylabel("Incidence")
    axA.set_xlabel("Date")
    axA.grid(True, axis="y")

    tau = np.arange(len(g))
    axB.plot(tau, g, color="#3A7D44", lw=1.45)
    axB.set_xlabel("Delay (days)")
    axB.set_ylabel("Density")
    axB.set_xlim(0, min(60, len(g) - 1))
    axB.set_xticks([0, 14, 28, 42, 56])
    axB.grid(True, axis="y")

    axC.scatter(dates, y, s=4.5, color="0.62", alpha=0.45, linewidths=0, label="Observed")
    axC.plot(
        dates,
        full_reference["downstream"],
        color=FULL_REFERENCE_COLOR,
        lw=FULL_REFERENCE_LW,
        alpha=FULL_REFERENCE_ALPHA,
        label=r"$D^{\rm ref}_{\rm no\ cutoff}$",
        zorder=6,
    )
    for fc in CUTOFFS:
        axC.plot(
            dates,
            downstream[fc],
            color=CUTOFF_COLORS[fc],
            lw=1.05,
            label=rf"$D^{{\rm ref}}_{{f_c={base.cutoff_label(fc)}}}$",
            zorder=3,
        )
    axC.set_ylabel("Incidence")
    axC.set_xlabel("Date")
    axC.grid(True, axis="y")

    shared_time_ymax = max(
        ymax,
        float(np.nanmax(y)),
        float(np.nanmax(full_reference["downstream"])),
        max(float(np.nanmax(downstream[fc])) for fc in CUTOFFS),
    )
    shared_time_ylim = (0.0, shared_time_ymax * 1.05 if shared_time_ymax > 0 else 1.0)
    shared_time_ticks = mticker.MaxNLocator(nbins=4).tick_values(*shared_time_ylim)
    shared_time_ticks = shared_time_ticks[
        (shared_time_ticks >= shared_time_ylim[0])
        & (shared_time_ticks <= shared_time_ylim[1])
    ]
    for ax in (axA, axC):
        ax.set_ylim(shared_time_ylim)
        ax.set_yticks(shared_time_ticks)

    # Row 2: spectra.
    for fc in CUTOFFS:
        axD.plot(
            freqs[keep],
            upstream_contrast_power[fc][keep] + eps,
            color=CUTOFF_COLORS[fc],
            lw=1.15,
            label=rf"$f_c={base.cutoff_label(fc)}$",
        )
    axD.set_ylabel("Upstream\ncontrast vs\nno cutoff")
    axD.set_yscale("log")
    axD.set_ylim(shared_contrast_ylim)
    axD.grid(True)

    axE.plot(freqs[keep], A2[keep] + eps, color="#3A7D44", lw=1.45)
    axE.set_ylabel("Squared\nmagnitude")
    axE.set_yscale("log")
    axE.grid(True)

    for fc in CUTOFFS:
        axF.plot(
            freqs[keep],
            downstream_contrast_power[fc][keep] + eps,
            color=CUTOFF_COLORS[fc],
            lw=1.15,
        )
    axF.axhline(
        variance_reference,
        color="0.25",
        lw=0.9,
        ls="--",
    )
    axF.set_ylabel("Downstream\ncontrast vs\nno cutoff")
    axF.set_yscale("log")
    axF.set_ylim(shared_contrast_ylim)
    axF.grid(True)
    axF.text(
        0.98,
        0.78,
        "Observation-variability\nreference",
        transform=axF.transAxes,
        ha="right",
        va="top",
        color="0.25",
        fontsize=6.8,
    )

    for ax in (axD, axE, axF):
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(r"Frequency (day$^{-1}$)")
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        for f_mark, period_label in [(1 / 7, "7-day"), (1 / 3, "3-day")]:
            ax.axvline(f_mark, color="0.58", lw=0.8, ls="--", alpha=0.8)
            ax.text(
                f_mark * 0.985,
                0.04,
                period_label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=6.4,
                color="0.42",
            )
    for ax in (axD, axF):
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))

    # Row 3: feature summaries.
    axG.plot(
        dates,
        full_reference["growth"],
        color=FULL_REFERENCE_COLOR,
        lw=FULL_REFERENCE_LW,
        alpha=FULL_REFERENCE_ALPHA,
        label=r"$r^{\rm ref}_{\rm no\ cutoff}$",
        zorder=6,
    )
    for fc in CUTOFFS:
        axG.plot(
            dates,
            growth[fc],
            color=CUTOFF_COLORS[fc],
            lw=1.15,
            label=rf"$r^{{\rm ref}}_{{f_c={base.cutoff_label(fc)}}}$",
            zorder=3,
        )
    axG.axhline(0.0, color="0.55", lw=0.75, ls="--")
    axG.set_ylabel("7-day log\ngrowth rate")
    axG.set_xlabel("Date")
    axG.grid(True, axis="y")

    y_pos = np.arange(len(feature_df))
    labels = feature_df["cutoff"].tolist()
    colors = [
        FULL_REFERENCE_COLOR if label == "No cutoff" else CUTOFF_COLORS[fc]
        for label, fc in zip(feature_df["cutoff"], feature_df["f_cut"])
    ]

    axH_left.barh(
        y_pos,
        feature_df["relative_peak_timing"].to_numpy(float),
        color=colors,
        alpha=0.78,
        height=0.55,
    )
    axH_left.invert_xaxis()
    axH_left.set_xlim(1.0, 0.0)
    axH_left.set_xlabel("Relative peak timing")
    axH_left.set_xticks([1.0, 0.5, 0.0])
    axH_left.set_xticklabels(["1", "0.5", "0"])
    axH_left.set_yticks(y_pos)
    axH_left.set_yticklabels([])
    axH_left.yaxis.tick_right()
    axH_left.tick_params(
        axis="y",
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        length=0,
    )
    axH_left.spines["left"].set_visible(False)
    axH_left.spines["right"].set_visible(True)
    axH_left.grid(True, axis="x")

    axH_right.barh(
        y_pos,
        feature_df["relative_peak_magnitude"].to_numpy(float),
        color=colors,
        alpha=0.78,
        height=0.55,
    )
    axH_right.set_xlim(0.0, 1.05)
    axH_right.set_xlabel("Relative peak magnitude")
    axH_right.set_xticks([0.0, 0.5, 1.0])
    axH_right.set_xticklabels(["0", "0.5", "1"])
    axH_right.set_yticks(y_pos)
    axH_right.set_yticklabels([])
    axH_right.tick_params(
        axis="y",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        length=0,
    )
    axH_right.spines["right"].set_visible(False)
    axH_right.spines["left"].set_visible(True)
    axH_right.grid(True, axis="x")
    h_center_x = 0.5 * (axH_left.get_position().x1 + axH_right.get_position().x0)
    fig.text(
        h_center_x,
        axH_left.get_position().y1 + 0.015,
        r"$f_c$",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    for ypos, label in zip(y_pos, labels):
        _, y_fig = fig.transFigure.inverted().transform(
            axH_left.transData.transform((0.0, ypos))
        )
        fig.text(
            h_center_x,
            y_fig,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
        )

    for ax in (axA, axC, axG):
        ax.xaxis.set_major_locator(MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%Y%m"))
    for ax in (axA, axC):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(7.0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # Align y-axis labels by column instead of letting tick-label widths move them.
    for ax in (axA, axD):
        ax.yaxis.set_label_coords(-0.170, 0.5)
    axG.yaxis.set_label_coords(-0.225, 0.5)
    for ax in (axB, axE):
        ax.yaxis.set_label_coords(-0.380, 0.5)
    for ax in (axC, axF):
        ax.yaxis.set_label_coords(-0.160, 0.5)

    # Matrix headers.
    col_headers = [
        (axA, "Upstream"),
        (axB, "Delay"),
        (axC, "Downstream"),
    ]
    for ax, label in col_headers:
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            0.970,
            label,
            ha="center",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
        )

    row_headers = [
        ([axA, axB, axC], "Time\ndomain"),
        ([axD, axE, axF], "Frequency\ndomain"),
        ([axG, axH_title], "Epidemiological\nfeatures"),
    ]
    for axes, label in row_headers:
        y0 = min(ax.get_position().y0 for ax in axes)
        y1 = max(ax.get_position().y1 for ax in axes)
        fig.text(
            0.006,
            0.5 * (y0 + y1),
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.6,
            fontweight="bold",
            linespacing=0.9,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def ylabel_left(ax, fallback_ax=None):
        label = ax.yaxis.get_label()
        if label.get_text():
            bbox = label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            return bbox.x0
        fallback = fallback_ax if fallback_ax is not None else ax
        return fallback.get_position().x0 - 0.030

    panel_label_x = {
        axA: ylabel_left(axA),
        axB: ylabel_left(axB),
        axC: ylabel_left(axC),
        axD: ylabel_left(axD),
        axE: ylabel_left(axE),
        axF: ylabel_left(axF),
        axG: ylabel_left(axG),
        axH_title: ylabel_left(axE),
    }
    for ax, label in zip([axA, axB, axC, axD, axE, axF, axG, axH_title], "ABCDEFGH"):
        bbox = ax.get_position()
        label_y = axG.get_position().y1 + 0.022 if ax is axH_title else bbox.y1 + 0.022
        fig.text(
            panel_label_x[ax],
            label_y,
            label,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    legend_handles = [
        Line2D([0], [0], color=CUTOFF_COLORS[fc], lw=1.35, label=rf"$f_c={base.cutoff_label(fc)}$")
        for fc in CUTOFFS
    ]
    legend_handles.extend([
        Line2D(
            [0],
            [0],
            color=FULL_REFERENCE_COLOR,
            lw=FULL_REFERENCE_LW,
            alpha=FULL_REFERENCE_ALPHA,
            label="No cutoff",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.62",
            markeredgecolor="0.62",
            markersize=3.8,
            alpha=0.55,
            label="Observed",
        ),
    ])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.25,
        borderaxespad=0.0,
        fontsize=7.1,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    payload = build_payload(args)
    stem = f"epi_features_combined_{args.geo_value}"
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"
    plot_combined(payload, out_png=out_png, out_pdf=out_pdf)
    print(f"Saved combined figure to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
