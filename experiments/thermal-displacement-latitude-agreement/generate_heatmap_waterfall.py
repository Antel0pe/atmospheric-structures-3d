from __future__ import annotations

import argparse
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
    DEFAULT_BORDER_GEOJSON,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    load_border_segments,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)
from generate_latitude_score_lines import (  # noqa: E402
    longitude_to_signed,
    select_sorted_longitudes,
    slug_for_level,
)


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-score-smoothed-sigma20-smoothed-agreement/"
    "longitude-latitude-heatmap-waterfall-stride16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate longitude-latitude heatmaps and waterfall/ridgeline plots "
            "for same-hemisphere thermal-displacement scores."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--longitude-stride", type=int, default=16)
    parser.add_argument("--contour-step", type=float, default=20.0)
    parser.add_argument("--heatmaps-only", action="store_true")
    parser.add_argument("--lon-min", type=float, default=-125.0)
    parser.add_argument("--lon-max", type=float, default=-50.0)
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


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


def plot_heatmap(
    score_subset: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    contour_step: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score_subset,
        cmap="bwr",
        norm=mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0),
        shading="auto",
        rasterized=True,
    )
    contour_levels = np.arange(contour_step, 100.0, contour_step, dtype=np.float32)
    contours = ax.contour(
        longitudes,
        latitudes,
        score_subset,
        levels=contour_levels,
        colors="#1b1b1b",
        linewidths=[0.75, 0.75, 1.25, 0.75, 0.75],
        alpha=0.82,
    )
    ax.clabel(contours, inline=True, fmt="%g", fontsize=8)
    lon_min = float(np.min(longitudes))
    lon_max = float(np.max(longitudes))
    lat_min = float(np.min(latitudes))
    lat_max = float(np.max(latitudes))
    for segment in border_segments:
        points = [
            (lon, lat)
            for lon, lat in segment
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
        ]
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color="#111111", linewidth=0.45, alpha=0.82)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement score heatmap "
        "with score contours"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_waterfall(
    score_subset: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig_height = max(7.0, 0.34 * len(longitudes) + 2.0)
    fig, ax = plt.subplots(figsize=(12.6, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("turbo")
    normalized_index = np.linspace(0.0, 1.0, len(longitudes))
    scale = 0.82

    for row_index, longitude in enumerate(longitudes):
        baseline = float(row_index)
        y = baseline + (score_subset[:, row_index] / 100.0) * scale
        ax.plot(
            latitudes,
            y,
            color=cmap(normalized_index[row_index]),
            linewidth=1.45,
            alpha=0.95,
        )
        ax.fill_between(
            latitudes,
            baseline,
            y,
            color=cmap(normalized_index[row_index]),
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlim(float(np.max(latitudes)), float(np.min(latitudes)))
    ax.set_ylim(-0.25, len(longitudes) - 1 + scale + 0.25)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude, stacked west to east")
    ax.set_title(
        f"{level_hpa:g} hPa waterfall: profile shape is score vs latitude; "
        "vertical stack is longitude"
    )
    ax.set_yticks(np.arange(len(longitudes)) + scale / 2.0)
    ax.set_yticklabels([f"{lon:g}" for lon in longitudes], fontsize=8)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    for row_index in range(len(longitudes)):
        ax.axhline(row_index, color="#e4e4e4", linewidth=0.45, zorder=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    heatmap_dir = output_dir / "heatmaps"
    waterfall_dir = output_dir / "waterfalls"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    waterfall_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(border_path, np.asarray([-180.0, 180.0], dtype=np.float32))
    lat_indices, lon_indices, selected_lats, selected_lons = selected_domain(
        latitudes=latitudes,
        longitudes=longitudes,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        longitude_stride=args.longitude_stride,
    )

    for level_hpa in selected_levels:
        slug = slug_for_level(level_hpa)
        print(f"Processing {level_hpa:g} hPa")
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        clim_level = (
            climatology.sel(pressure_level=level_hpa)
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

        plot_heatmap(
        score_subset=score_subset,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            level_hpa=level_hpa,
        contour_step=args.contour_step,
        output_path=heatmap_dir / f"heatmap_{slug}.png",
        dpi=args.dpi,
    )
        if not args.heatmaps_only:
            plot_waterfall(
                score_subset=score_subset,
                latitudes=selected_lats,
                longitudes=selected_lons,
                level_hpa=level_hpa,
                output_path=waterfall_dir / f"waterfall_{slug}.png",
                dpi=args.dpi,
            )

    try:
        print(f"Wrote {output_dir.relative_to(Path.cwd()).as_posix()}")
    except ValueError:
        print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
