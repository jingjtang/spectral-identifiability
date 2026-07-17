#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate epidemiological-feature figures for the US or individual states.

Examples
--------
Generate the US-level figure::

    python fig_epi_features.py

Generate the US figure plus figures for California and New York only::

    python fig_epi_features.py --states ca ny

Generate state figures without the US figure::

    python fig_epi_features.py --no-us
"""

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

DEFAULT_OUT_DIR = Path("../figs")
STATE_CODES = (
    "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms "
    "mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv "
    "wi wy"
).split()


def load_runtime_dependencies():
    """Import scientific dependencies only when figure generation begins."""
    global np, pd, base
    import importlib
    import numpy as np
    import pandas as pd
    import utils as base

    # PyCharm's Python Console keeps imported modules cached between runs.
    # Reload the plotting module so its current on-disk function signature is
    # used (in particular, the optional ``figure_label`` argument).
    base = importlib.reload(base)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate feasible-region epidemiological-feature figures "
                    "for the US and/or US states."
    )
    parser.add_argument(
        "--states",
        nargs="+",
        metavar="STATE",
        help=("Two-letter state abbreviations (for example: ca ny), or 'all'. "
              "Default: no states."),
    )
    parser.add_argument(
        "--no-us",
        action="store_true",
        help="Do not generate the national US figure.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default="2020-04-01")
    parser.add_argument("--end", default="2021-06-30")
    parser.add_argument("--display-start", default="2020-09-01")
    parser.add_argument("--display-end", default="2021-03-31")
    args, unknown = parser.parse_known_args()

    # PyCharm's "Run file in Python Console" leaves its own connection
    # arguments in sys.argv.  Ignore only those known PyDev arguments; keep
    # argparse's normal error handling for misspelled user options.
    pydev_prefixes = ("--mode=", "--host=", "--port=")
    unexpected = [arg for arg in unknown if not arg.startswith(pydev_prefixes)]
    if unexpected:
        parser.error("unrecognized arguments: " + " ".join(unexpected))
    return args


def requested_regions(states, include_us=True):
    """Return (label, geo_type, geo_value, file_slug) tuples."""
    regions = []
    if include_us:
        regions.append(("US", "nation", "us", "us"))

    normalized = [] if states is None else [state.lower() for state in states]
    if "all" in normalized:
        if len(normalized) != 1:
            raise ValueError("Use --states all by itself.")
        normalized = STATE_CODES

    invalid = sorted(set(normalized) - set(STATE_CODES))
    if invalid:
        raise ValueError(
            "Unknown state abbreviation(s): " + ", ".join(invalid)
        )
    for state in dict.fromkeys(normalized):
        regions.append((state.upper(), "state", state, state))
    return regions


def load_delay_pmf():
    try:
        g = base.load_empirical_delay_pmf()
        print(f"Loaded empirical delay: {base.DEFAULT_ONSET_REPORT_DELAY_KEY}")
        return g
    except Exception as exc:
        print("Warning: failed to load empirical delay; using fixed lognormal.")
        print("Reason:", repr(exc))
        return base.load_empirical_delay_pmf(fallback_lognormal=True)


def build_region_payload(
    *, epidata, epi_range, label, geo_type, geo_value, g,
    start, end, display_start, display_end,
):
    df_grid = [10, 15, 20, 30, 40, 60, 80, 100, 120]
    cutoffs = [1 / 35, 1 / 14, 1 / 7]
    cutoff_curve = np.linspace(1 / 60, 1 / 5, 90)
    tau = base.tau_from_alpha(0.20)

    print(f"\n=== Building {label} ({geo_type}={geo_value}) ===")
    df = base.fetch_signal_df(
        epidata,
        data_source="jhu-csse",
        signals="confirmed_incidence_num",
        time_values=epi_range(start, end),
        geo_type=geo_type,
        geo_values=geo_value,
    )
    if df.empty or not df["y"].notna().any():
        raise ValueError(f"No case-incidence data returned for {label}")

    aligned = base.align_daily(df, start=start, end=end)
    dates = aligned["date"].values
    y = aligned["y"].values.astype(float)
    if not np.isfinite(y).all():
        raise ValueError(f"Case-incidence series for {label} contains missing values")

    joint = base.select_best_df_and_noise(
        y, df_grid=df_grid, ord=2, criterion="AIC", rscript_bin="Rscript"
    )
    mu_hat = joint.mu_hat
    variance = base.get_variance_scale(mu_hat, joint.best_noise)
    sigma2_eff = max(float(np.mean(variance)), 1e-8)
    print(
        f"{label}: df_target={joint.df_target}, df_used≈{joint.df_used:.2f}, "
        f"noise={joint.best_noise.model}, sigma2_eff={sigma2_eff:.3g}"
    )

    display_mask = (
        (pd.to_datetime(dates) >= pd.Timestamp(display_start))
        & (pd.to_datetime(dates) <= pd.Timestamp(display_end))
    )
    if not display_mask.any():
        raise ValueError("The display interval does not overlap the analysis interval")
    display_scale = max(float(np.mean(mu_hat[display_mask])), 1e-8)

    references, downstream, bands = {}, {}, {}
    band_rows = []
    for fc in cutoffs:
        upstream, fitted, theta, _, _ = base.compute_cutoff_specific_reference_strict(
            y=y, mu_hat=mu_hat, g=g, f_cut=fc, ridge_theta=1e-6
        )
        band = base.analytic_band_width_time_domain(
            T=len(y), g=g, sigma2_eff=sigma2_eff, f_cut=fc, tau=tau
        )
        references[fc], downstream[fc], bands[fc] = upstream, fitted, band
        relative_width = 2.0 * band / display_scale
        band_rows.append({
            "region": label,
            "geo_type": geo_type,
            "geo_value": geo_value,
            "cutoff": base.cutoff_label(fc),
            "f_cut": fc,
            "period_days": 1.0 / fc,
            "mean_absolute_width": float(np.mean(2.0 * band[display_mask])),
            "mean_relative_width": float(np.mean(relative_width[display_mask])),
            "median_relative_width": float(np.median(relative_width[display_mask])),
        })
        print(
            f"{label}: cutoff={base.cutoff_label(fc)}, "
            f"basis_terms={len(theta)}, mean_band_width={np.mean(2 * band):.3g}"
        )

    curve_rows = []
    for fc in cutoff_curve:
        band = base.analytic_band_width_time_domain(
            T=len(y), g=g, sigma2_eff=sigma2_eff, f_cut=float(fc), tau=tau
        )
        relative_width = 2.0 * band / display_scale
        curve_rows.append({
            "region": label,
            "geo_type": geo_type,
            "geo_value": geo_value,
            "f_cut": float(fc),
            "period_days": float(1.0 / fc),
            "mean_absolute_width": float(np.mean(2.0 * band[display_mask])),
            "mean_relative_width": float(np.mean(relative_width[display_mask])),
            "median_relative_width": float(np.median(relative_width[display_mask])),
        })

    growth = {
        fc: base.smooth_reflect(
            base.rolling_growth_rate(references[fc], horizon=7, log_offset=1.0),
            window=5,
        )
        for fc in cutoffs
    }

    dates_display = pd.to_datetime(dates[display_mask])
    feature_rows = []
    for fc in cutoffs:
        displayed = np.asarray(references[fc][display_mask], float)
        peak_smoothed = base.smooth_reflect(displayed, window=14)
        peak_idx = int(np.nanargmax(peak_smoothed))
        feature_rows.append({
            "region": label,
            "geo_type": geo_type,
            "geo_value": geo_value,
            "cutoff": base.cutoff_label(fc),
            "f_cut": fc,
            "period_days": 1.0 / fc,
            "peak_date": dates_display[peak_idx],
            "peak_day_index": peak_idx,
            "peak_magnitude": float(displayed[peak_idx]),
            "peak_magnitude_smoothed": float(peak_smoothed[peak_idx]),
        })
    feature_df = pd.DataFrame(feature_rows)
    feature_df["relative_peak_timing"] = (
        feature_df["peak_day_index"] / max(len(dates_display) - 1, 1)
    )
    max_peak = feature_df["peak_magnitude"].max()
    feature_df["relative_peak_magnitude"] = (
        feature_df["peak_magnitude"] / max(max_peak, 1e-12)
    )

    arrays = [y, mu_hat]
    for mapping in (references, downstream, bands, growth):
        arrays.extend(mapping[fc] for fc in cutoffs)
    sliced = base.slice_date_range(
        dates, *arrays, start=display_start, end=display_end
    )
    dates_plot = sliced[0]
    values = iter(sliced[1:])
    y_plot, mu_plot = next(values), next(values)
    sliced_maps = []
    for _ in range(4):
        sliced_maps.append({fc: next(values) for fc in cutoffs})

    return {
        "row_label": label,
        "dates_plot": dates_plot,
        "y_plot": y_plot,
        "mu_plot": mu_plot,
        "cutoff_to_ref_plot": sliced_maps[0],
        "cutoff_to_refdown_plot": sliced_maps[1],
        "cutoff_to_band_plot": sliced_maps[2],
        "cutoff_to_growth_plot": sliced_maps[3],
        "band_width_df": pd.DataFrame(band_rows),
        "band_width_curve_df": pd.DataFrame(curve_rows),
        "feature_df": feature_df,
    }


def run_pipeline(args):
    load_runtime_dependencies()
    regions = requested_regions(args.states, include_us=not args.no_us)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    g = load_delay_pmf()

    base.set_epidata_api_key()

    from epidatpy import EpiDataContext, EpiRange
    epidata = EpiDataContext(use_cache=False)
    failures = []
    for label, geo_type, geo_value, slug in regions:
        try:
            payload = build_region_payload(
                epidata=epidata,
                epi_range=EpiRange,
                label=label,
                geo_type=geo_type,
                geo_value=geo_value,
                g=g,
                start=args.start,
                end=args.end,
                display_start=args.display_start,
                display_end=args.display_end,
            )
            stem = f"epi_features_{slug}"
            png = args.out_dir / f"{stem}.png"
            pdf = args.out_dir / f"{stem}.pdf"
            base.plot_feasible_region_v3_single_us(
                payload=payload, out_png=png, out_pdf=pdf,
                figure_label=None if slug == "us" else label,
            )
            print(f"Saved {label} figure to {png} and {pdf}")
        except Exception as exc:
            if len(regions) == 1:
                raise
            failures.append((label, exc))
            print(f"ERROR: skipped {label}: {exc}")

    if failures:
        failed = ", ".join(label for label, _ in failures)
        raise RuntimeError(f"Generation failed for: {failed}")


if __name__ == "__main__":
    run_pipeline(parse_args())
