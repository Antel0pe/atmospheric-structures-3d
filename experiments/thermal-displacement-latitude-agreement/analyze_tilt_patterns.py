from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)
from generate_latitude_score_lines import (  # noqa: E402
    select_sorted_longitudes,
)


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-score-smoothed-sigma20-smoothed-agreement/"
    "tilt-pattern-analysis-stride16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze longitude and pressure tilt in thermal-displacement score profiles."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--longitude-stride", type=int, default=16)
    parser.add_argument("--lon-min", type=float, default=-125.0)
    parser.add_argument("--lon-max", type=float, default=-50.0)
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=89.0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def slug_for_level(level_hpa: float) -> str:
    return f"{level_hpa:g}".replace(".", "p").replace("-", "m") + "hpa"


def slug_for_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def selected_domain(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    longitude_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if longitude_stride < 1:
        raise ValueError("--longitude-stride must be >= 1")
    lon_indices, lon_signed = select_sorted_longitudes(longitudes, lon_min, lon_max)
    lon_indices = lon_indices[::longitude_stride]
    lon_signed = lon_signed[::longitude_stride]
    lat_indices = np.flatnonzero((latitudes >= lat_min) & (latitudes <= lat_max))
    lat_indices = lat_indices[np.argsort(-latitudes[lat_indices])]
    return lat_indices, lon_indices, latitudes[lat_indices], lon_signed


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(np.count_nonzero(finite))}
    xf = x[finite].astype(np.float64)
    yf = y[finite].astype(np.float64)
    design = np.column_stack([xf, np.ones_like(xf)])
    slope, intercept = np.linalg.lstsq(design, yf, rcond=None)[0]
    predicted = slope * xf + intercept
    ss_res = float(np.sum((yf - predicted) ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": int(xf.size)}


def crossing_latitude(
    latitudes_desc: np.ndarray,
    score_profile: np.ndarray,
    threshold: float = 50.0,
) -> float:
    values = score_profile.astype(np.float64)
    lats = latitudes_desc.astype(np.float64)
    signed = values - threshold
    crossing_indices = np.flatnonzero(signed[:-1] * signed[1:] <= 0.0)
    if crossing_indices.size == 0:
        return np.nan
    candidate_lats = []
    candidate_slopes = []
    for index in crossing_indices:
        y0 = values[index]
        y1 = values[index + 1]
        x0 = lats[index]
        x1 = lats[index + 1]
        if y1 == y0:
            lat = 0.5 * (x0 + x1)
        else:
            fraction = (threshold - y0) / (y1 - y0)
            lat = x0 + fraction * (x1 - x0)
        candidate_lats.append(float(lat))
        candidate_slopes.append(float(abs((y1 - y0) / max(abs(x1 - x0), 1e-6))))
    best = int(np.argmax(candidate_slopes))
    return candidate_lats[best]


def max_gradient_latitude(latitudes_desc: np.ndarray, score_profile: np.ndarray) -> float:
    lats = latitudes_desc.astype(np.float64)
    values = score_profile.astype(np.float64)
    gradient = np.gradient(values, lats)
    mask = (lats >= 15.0) & (lats <= 75.0)
    if not np.any(mask):
        return np.nan
    masked_indices = np.flatnonzero(mask)
    best_index = masked_indices[int(np.argmax(np.abs(gradient[mask])))]
    return float(lats[best_index])


def gradient_centroid_latitude(latitudes_desc: np.ndarray, score_profile: np.ndarray) -> float:
    lats = latitudes_desc.astype(np.float64)
    values = score_profile.astype(np.float64)
    gradient = np.abs(np.gradient(values, lats))
    midrange = (values >= 20.0) & (values <= 80.0) & (lats >= 10.0) & (lats <= 80.0)
    weights = gradient * midrange
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.nan
    return float(np.sum(lats * weights) / total)


def profile_shift_from_western_reference(
    latitudes_desc: np.ndarray,
    score_subset: np.ndarray,
    max_shift_deg: float = 30.0,
) -> np.ndarray:
    lat_step = float(abs(np.nanmedian(np.diff(latitudes_desc))))
    max_lag = int(round(max_shift_deg / max(lat_step, 1e-6)))
    ref = score_subset[:, 0].astype(np.float64)
    mid_mask = (latitudes_desc >= 10.0) & (latitudes_desc <= 80.0)
    shifts = np.full(score_subset.shape[1], np.nan, dtype=np.float64)
    shifts[0] = 0.0
    for column in range(1, score_subset.shape[1]):
        profile = score_subset[:, column].astype(np.float64)
        best_lag = 0
        best_corr = -np.inf
        for lag in range(-max_lag, max_lag + 1):
            shifted = np.roll(profile, lag)
            valid = mid_mask.copy()
            if lag > 0:
                valid[:lag] = False
            elif lag < 0:
                valid[lag:] = False
            a = ref[valid]
            b = shifted[valid]
            if a.size < 10 or np.std(a) == 0.0 or np.std(b) == 0.0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        shifts[column] = -best_lag * lat_step
    return shifts


def transition_latitudes_by_method(
    latitudes_desc: np.ndarray,
    score_subset: np.ndarray,
) -> dict[str, np.ndarray]:
    n_lon = score_subset.shape[1]
    methods = {
        "score50_crossing": np.full(n_lon, np.nan, dtype=np.float64),
        "max_gradient": np.full(n_lon, np.nan, dtype=np.float64),
        "gradient_centroid": np.full(n_lon, np.nan, dtype=np.float64),
    }
    for column in range(n_lon):
        profile = score_subset[:, column]
        methods["score50_crossing"][column] = crossing_latitude(latitudes_desc, profile, 50.0)
        methods["max_gradient"][column] = max_gradient_latitude(latitudes_desc, profile)
        methods["gradient_centroid"][column] = gradient_centroid_latitude(latitudes_desc, profile)
    return methods


def plot_method_fits(
    method_name: str,
    method_by_level: dict[float, np.ndarray],
    longitudes: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (level_hpa, values) in zip(axes_flat, method_by_level.items()):
        fit = fit_line(longitudes, values)
        ax.scatter(longitudes, values, s=24, color="#2457c5", alpha=0.85)
        if np.isfinite(fit["slope"]):
            xfit = np.array([np.nanmin(longitudes), np.nanmax(longitudes)])
            ax.plot(xfit, fit["slope"] * xfit + fit["intercept"], color="#d22", linewidth=1.6)
        ax.set_title(
            f"{level_hpa:g} hPa; slope {fit['slope']:.2f} lat/lon, R2 {fit['r2']:.2f}"
        )
        ax.grid(color="#d6d6d6", linewidth=0.6, alpha=0.75)
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimated transition latitude")
    fig.suptitle(f"Within-level longitudinal tilt: {method_name.replace('_', ' ')}")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_profile_shift_fits(
    shifts_by_level: dict[float, np.ndarray],
    longitudes: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (level_hpa, shifts) in zip(axes_flat, shifts_by_level.items()):
        fit = fit_line(longitudes, shifts)
        ax.scatter(longitudes, shifts, s=24, color="#6a2fb8", alpha=0.85)
        if np.isfinite(fit["slope"]):
            xfit = np.array([np.nanmin(longitudes), np.nanmax(longitudes)])
            ax.plot(xfit, fit["slope"] * xfit + fit["intercept"], color="#d22", linewidth=1.6)
        ax.axhline(0.0, color="#1a1a1a", linewidth=0.8)
        ax.set_title(
            f"{level_hpa:g} hPa; shift slope {fit['slope']:.2f} deg/lon, R2 {fit['r2']:.2f}"
        )
        ax.grid(color="#d6d6d6", linewidth=0.6, alpha=0.75)
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude shift vs westernmost profile")
    fig.suptitle("Profile-shift tilt from correlation against westernmost longitude")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_cross_pressure_heatmap(
    method_name: str,
    levels: np.ndarray,
    longitudes: np.ndarray,
    values_by_level: dict[float, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    matrix = np.vstack([values_by_level[float(level)] for level in levels])
    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[float(longitudes[0]), float(longitudes[-1]), -0.5, len(levels) - 0.5],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels([f"{level:g}" for level in levels])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Pressure level hPa")
    ax.set_title(f"Cross-pressure transition latitude: {method_name.replace('_', ' ')}")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Estimated transition latitude")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def pressure_plane_fit(
    levels: np.ndarray,
    longitudes: np.ndarray,
    values_by_level: dict[float, np.ndarray],
) -> dict[str, float]:
    xs = []
    zs = []
    ys = []
    height_proxy = -np.log(levels.astype(np.float64) / 1000.0)
    for level_index, level in enumerate(levels):
        values = values_by_level[float(level)]
        for lon, value in zip(longitudes, values):
            if np.isfinite(value):
                xs.append(float(lon))
                zs.append(float(height_proxy[level_index]))
                ys.append(float(value))
    if len(ys) < 4:
        return {"longitude_slope": np.nan, "height_proxy_slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": len(ys)}
    x = np.asarray(xs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    design = np.column_stack([x, z, np.ones_like(x)])
    lon_slope, height_slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([lon_slope, height_slope, intercept])
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "longitude_slope": float(lon_slope),
        "height_proxy_slope": float(height_slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": int(y.size),
    }


def plot_pressure_waterfall(
    score_by_level: dict[float, np.ndarray],
    levels: np.ndarray,
    latitudes: np.ndarray,
    longitude_label: str,
    longitude_index: int | None,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    cmap = plt.get_cmap("plasma")
    scale = 0.82
    for row_index, level in enumerate(levels):
        field = score_by_level[float(level)]
        if longitude_index is None:
            profile = np.nanmean(field, axis=1)
        else:
            profile = field[:, longitude_index]
        baseline = float(row_index)
        y = baseline + (profile / 100.0) * scale
        ax.plot(latitudes, y, color=cmap(row_index / max(len(levels) - 1, 1)), linewidth=1.8)
        ax.fill_between(latitudes, baseline, y, color=cmap(row_index / max(len(levels) - 1, 1)), alpha=0.14, linewidth=0)
        ax.axhline(baseline, color="#dedede", linewidth=0.55)
    ax.set_xlim(float(np.max(latitudes)), float(np.min(latitudes)))
    ax.set_ylim(-0.2, len(levels) - 1 + scale + 0.25)
    ax.set_yticks(np.arange(len(levels)) + scale / 2.0)
    ax.set_yticklabels([f"{level:g} hPa" for level in levels])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Pressure level, stacked")
    ax.set_title(f"Pressure-level waterfall at {longitude_label}")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    pressure_waterfall_dir = output_dir / "pressure-waterfalls"
    plots_dir.mkdir(exist_ok=True)
    pressure_waterfall_dir.mkdir(exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = np.asarray(parse_requested_levels(args.pressure_levels, level_values), dtype=np.float64)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    lat_indices, lon_indices, selected_lats, selected_lons = selected_domain(
        latitudes=latitudes,
        longitudes=longitudes,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        longitude_stride=args.longitude_stride,
    )

    score_by_level: dict[float, np.ndarray] = {}
    transitions: dict[str, dict[float, np.ndarray]] = {
        "score50_crossing": {},
        "max_gradient": {},
        "gradient_centroid": {},
    }
    shifts_by_level: dict[float, np.ndarray] = {}

    for level_hpa in selected_levels:
        print(f"Processing {level_hpa:g} hPa")
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=float(level_hpa))
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        clim_level = (
            climatology.sel(pressure_level=float(level_hpa))
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        matched_latitude = match_equivalent_latitude(
            raw_level,
            clim_level,
            latitudes,
            "same-hemisphere",
        )
        matched_latitude = np.clip(matched_latitude, -90.0, 90.0).astype(np.float32)
        score_unsmoothed = thermal_displacement_score_points(matched_latitude, latitudes)
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, args.smooth_sigma_cells)
        score_subset = score_smoothed[np.ix_(lat_indices, lon_indices)]
        score_by_level[float(level_hpa)] = score_subset
        method_values = transition_latitudes_by_method(selected_lats, score_subset)
        for method_name, values in method_values.items():
            transitions[method_name][float(level_hpa)] = values
        shifts_by_level[float(level_hpa)] = profile_shift_from_western_reference(
            selected_lats,
            score_subset,
        )

    metric_rows: list[dict[str, float | str]] = []
    for method_name, values_by_level in transitions.items():
        plot_method_fits(
            method_name,
            values_by_level,
            selected_lons,
            plots_dir / f"within_level_fit_{method_name}.png",
            args.dpi,
        )
        plot_cross_pressure_heatmap(
            method_name,
            selected_levels,
            selected_lons,
            values_by_level,
            plots_dir / f"cross_pressure_transition_latitude_{method_name}.png",
            args.dpi,
        )
        plane = pressure_plane_fit(selected_levels, selected_lons, values_by_level)
        metric_rows.append({"scope": "cross_pressure_plane", "method": method_name, **plane})
        for level_hpa, values in values_by_level.items():
            fit = fit_line(selected_lons, values)
            metric_rows.append(
                {
                    "scope": "within_level",
                    "method": method_name,
                    "pressure_level_hpa": level_hpa,
                    "longitude_slope": fit["slope"],
                    "intercept": fit["intercept"],
                    "r2": fit["r2"],
                    "n": fit["n"],
                }
            )

    plot_profile_shift_fits(
        shifts_by_level,
        selected_lons,
        plots_dir / "within_level_fit_profile_shift_correlation.png",
        args.dpi,
    )
    for level_hpa, shifts in shifts_by_level.items():
        fit = fit_line(selected_lons, shifts)
        metric_rows.append(
            {
                "scope": "within_level_profile_shift",
                "method": "profile_shift_correlation",
                "pressure_level_hpa": level_hpa,
                "longitude_slope": fit["slope"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "n": fit["n"],
            }
        )

    representative_indices = sorted(set([0, len(selected_lons) // 4, len(selected_lons) // 2, 3 * len(selected_lons) // 4, len(selected_lons) - 1]))
    plot_pressure_waterfall(
        score_by_level,
        selected_levels,
        selected_lats,
        "longitude mean",
        None,
        pressure_waterfall_dir / "pressure_waterfall_longitude_mean.png",
        args.dpi,
    )
    for index in representative_indices:
        lon = selected_lons[index]
        plot_pressure_waterfall(
            score_by_level,
            selected_levels,
            selected_lats,
            f"{lon:g} longitude",
            index,
            pressure_waterfall_dir / f"pressure_waterfall_lon_{slug_for_number(float(lon))}.png",
            args.dpi,
        )

    with (output_dir / "tilt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in metric_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "longitude_window": [float(args.lon_min), float(args.lon_max)],
        "latitude_window": [float(args.lat_max), float(args.lat_min)],
        "longitude_stride": args.longitude_stride,
        "score_smoothing_sigma_cells": args.smooth_sigma_cells,
        "methods": {
            "score50_crossing": "Interpolated latitude where score crosses 50; if multiple crossings, keep crossing with strongest local slope.",
            "max_gradient": "Latitude of strongest absolute meridional score gradient between 15N and 75N.",
            "gradient_centroid": "Gradient-weighted average latitude where score is between 20 and 80.",
            "profile_shift_correlation": "Latitude shift needed to align each longitude profile against the westernmost profile.",
        },
        "outputs": {
            "plots": "plots/",
            "pressure_waterfalls": "pressure-waterfalls/",
            "metrics_csv": "tilt_metrics.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir.relative_to(Path.cwd()).as_posix()}")


if __name__ == "__main__":
    main()
