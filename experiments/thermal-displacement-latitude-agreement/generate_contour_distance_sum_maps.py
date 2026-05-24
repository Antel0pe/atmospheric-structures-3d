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
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_global_contour_paths import (  # noqa: E402
    build_contour_lines,
    draw_borders_clipped,
    sorted_global_domain,
)
from generate_latitude_score_lines import slug_for_level  # noqa: E402
from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_BORDER_GEOJSON,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    load_border_segments,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-contour-two-nearest-distance-sum-step5-250-1000"
)
DEFAULT_LEVELS = "250,500,850,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate global maps colored by the sum of distances from each ERA5 "
            "grid cell to its two closest distinct thermal-displacement score "
            "contour segments."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument(
        "--contour-sample-spacing-degrees",
        type=float,
        default=0.25,
        help="Maximum spacing between sampled points along contour segments.",
    )
    parser.add_argument(
        "--nearest-sample-query-count",
        type=int,
        default=96,
        help="Number of sampled contour points to query before choosing two lines.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=100_000,
        help="Number of ERA5 grid cells per nearest-neighbor query batch.",
    )
    parser.add_argument(
        "--color-vmax-percentile",
        type=float,
        default=99.5,
        help="Percentile used as the white end of the red-white color scale.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def sample_contour_vertices(
    contour_lines,
    max_spacing_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = max(float(max_spacing_degrees), 1.0e-6)
    sampled_points: list[np.ndarray] = []
    sampled_line_ids: list[np.ndarray] = []

    for line_index, contour_line in enumerate(contour_lines):
        vertices = np.asarray(contour_line.vertices, dtype=np.float32)
        if vertices.shape[0] < 2:
            continue

        line_points: list[np.ndarray] = []
        for segment_index, (start, end) in enumerate(zip(vertices[:-1], vertices[1:])):
            distance = float(np.linalg.norm(end - start))
            step_count = max(1, int(np.ceil(distance / spacing)))
            t = np.linspace(
                0.0,
                1.0,
                step_count + 1,
                endpoint=True,
                dtype=np.float32,
            )
            segment_points = start + (end - start) * t[:, np.newaxis]
            if segment_index > 0:
                segment_points = segment_points[1:]
            line_points.append(segment_points)

        points = np.vstack(line_points)
        sampled_points.append(points)
        sampled_line_ids.append(
            np.full(points.shape[0], line_index, dtype=np.int32)
        )

    if not sampled_points:
        raise ValueError("No contour points were sampled.")

    points = np.vstack(sampled_points).astype(np.float32)
    line_ids = np.concatenate(sampled_line_ids).astype(np.int32)
    return points, line_ids


def build_periodic_contour_tree(
    contour_points: np.ndarray,
    line_ids: np.ndarray,
) -> tuple[cKDTree, np.ndarray]:
    shifted_points = [
        contour_points,
        contour_points + np.asarray([360.0, 0.0], dtype=np.float32),
        contour_points - np.asarray([360.0, 0.0], dtype=np.float32),
    ]
    periodic_points = np.vstack(shifted_points).astype(np.float32)
    periodic_line_ids = np.tile(line_ids, 3).astype(np.int32)
    return cKDTree(periodic_points), periodic_line_ids


def choose_two_distinct_line_distances(
    query_distances: np.ndarray,
    query_indices: np.ndarray,
    sample_line_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_ids = sample_line_ids[query_indices]
    first_line_ids = line_ids[:, 0]
    first_distances = query_distances[:, 0].astype(np.float32)
    second_distances = np.full(first_distances.shape, np.nan, dtype=np.float32)
    second_line_ids = np.full(first_line_ids.shape, -1, dtype=np.int32)

    for column_index in range(1, line_ids.shape[1]):
        unresolved = ~np.isfinite(second_distances)
        if not np.any(unresolved):
            break
        different_line = line_ids[:, column_index] != first_line_ids
        take = unresolved & different_line
        second_distances[take] = query_distances[take, column_index].astype(np.float32)
        second_line_ids[take] = line_ids[take, column_index].astype(np.int32)

    return first_distances, second_distances, second_line_ids


def compute_two_nearest_distance_sum(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    contour_lines,
    sample_spacing_degrees: float,
    nearest_query_count: int,
    query_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(contour_lines) < 2:
        raise ValueError("Need at least two contour lines to compute two-line sums.")

    contour_points, contour_line_ids = sample_contour_vertices(
        contour_lines,
        sample_spacing_degrees,
    )
    tree, periodic_line_ids = build_periodic_contour_tree(
        contour_points,
        contour_line_ids,
    )

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    query_points = np.column_stack(
        (
            lon_grid.ravel().astype(np.float32),
            lat_grid.ravel().astype(np.float32),
        )
    )

    sample_count = len(periodic_line_ids)
    query_count = min(max(int(nearest_query_count), 2), sample_count)
    batch_size = max(int(query_batch_size), 1)
    distance_sum = np.full(query_points.shape[0], np.nan, dtype=np.float32)
    unresolved_count = 0

    for start_index in range(0, query_points.shape[0], batch_size):
        stop_index = min(start_index + batch_size, query_points.shape[0])
        batch_points = query_points[start_index:stop_index]
        distances, indices = tree.query(batch_points, k=query_count, workers=-1)
        if query_count == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        first, second, _ = choose_two_distinct_line_distances(
            distances,
            indices,
            periodic_line_ids,
        )

        unresolved = ~np.isfinite(second)
        fallback_count = query_count
        while np.any(unresolved) and fallback_count < sample_count:
            fallback_count = min(
                sample_count,
                max(fallback_count * 4, fallback_count + 1),
                8192,
            )
            fallback_distances, fallback_indices = tree.query(
                batch_points[unresolved],
                k=fallback_count,
                workers=-1,
            )
            if fallback_count == 1:
                fallback_distances = fallback_distances[:, np.newaxis]
                fallback_indices = fallback_indices[:, np.newaxis]
            fallback_first, fallback_second, _ = choose_two_distinct_line_distances(
                fallback_distances,
                fallback_indices,
                periodic_line_ids,
            )
            first[unresolved] = fallback_first
            second[unresolved] = fallback_second
            unresolved = ~np.isfinite(second)
            if fallback_count >= 8192:
                break

        unresolved_count += int(np.count_nonzero(~np.isfinite(second)))
        distance_sum[start_index:stop_index] = first + second

    distance_sum = distance_sum.reshape((len(latitudes), len(longitudes)))
    finite_values = distance_sum[np.isfinite(distance_sum)]
    diagnostics = {
        "contour_line_count": len(contour_lines),
        "sampled_contour_point_count": int(contour_points.shape[0]),
        "periodic_sampled_contour_point_count": int(sample_count),
        "nearest_sample_query_count": int(query_count),
        "grid_cell_count": int(query_points.shape[0]),
        "unresolved_two_line_cell_count": int(unresolved_count),
        "distance_sum_min_degrees": float(np.nanmin(finite_values)),
        "distance_sum_max_degrees": float(np.nanmax(finite_values)),
        "distance_sum_mean_degrees": float(np.nanmean(finite_values)),
        "distance_sum_p50_degrees": float(np.nanpercentile(finite_values, 50.0)),
        "distance_sum_p95_degrees": float(np.nanpercentile(finite_values, 95.0)),
        "distance_sum_p995_degrees": float(np.nanpercentile(finite_values, 99.5)),
    }
    return distance_sum, diagnostics


def red_white_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "small-red-large-white",
        ["#b30000", "#ef3b2c", "#fcae91", "#fff5f0", "#ffffff"],
    )


def plot_distance_sum_map(
    distance_sum: np.ndarray,
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    contour_levels: np.ndarray,
    contour_step: float,
    color_vmax: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        distance_sum,
        cmap=red_white_colormap(),
        norm=mcolors.Normalize(vmin=0.0, vmax=color_vmax),
        shading="auto",
        rasterized=True,
    )
    contours = ax.contour(
        longitudes,
        latitudes,
        score,
        levels=contour_levels,
        colors="#242424",
        linewidths=0.55,
        alpha=0.78,
    )
    ax.clabel(contours, inline=True, fmt="%g", fontsize=5.5)
    draw_borders_clipped(ax, border_segments)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa: sum of distances to two closest distinct "
        f"{contour_step:g}-point score contours"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label(
        "Distance sum in lon/lat degrees; red = small, white = large"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    heatmap_dir = output_dir / "heatmaps"
    array_dir = output_dir / "arrays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    source_latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    source_longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    lat_indices, lon_indices, selected_lats, selected_lons = sorted_global_domain(
        source_latitudes,
        source_longitudes,
    )
    border_segments = load_border_segments(
        border_path,
        np.asarray([-180.0, 180.0], dtype=np.float32),
    )
    contour_levels = np.arange(args.contour_step, 100.0, args.contour_step, dtype=np.float32)

    output_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

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
            source_latitudes,
            "same-hemisphere",
        )
        matched_latitude = np.clip(matched_latitude, -90.0, 90.0).astype(np.float32)
        score_unsmoothed = thermal_displacement_score_points(
            matched_latitude,
            source_latitudes,
        )
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, args.smooth_sigma_cells)
        score_global = score_smoothed[np.ix_(lat_indices, lon_indices)]

        contour_lines = build_contour_lines(
            selected_lons,
            selected_lats,
            score_global,
            contour_levels,
        )
        distance_sum, diagnostics = compute_two_nearest_distance_sum(
            selected_lons,
            selected_lats,
            contour_lines,
            args.contour_sample_spacing_degrees,
            args.nearest_sample_query_count,
            args.query_batch_size,
        )
        finite_values = distance_sum[np.isfinite(distance_sum)]
        color_vmax = float(
            np.nanpercentile(finite_values, args.color_vmax_percentile)
        )

        array_path = array_dir / f"two_nearest_contour_distance_sum_{slug}.npy"
        np.save(array_path, distance_sum.astype(np.float32))

        heatmap_path = heatmap_dir / f"two_nearest_contour_distance_sum_{slug}.png"
        plot_distance_sum_map(
            distance_sum=distance_sum,
            score=score_global,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            level_hpa=level_hpa,
            contour_levels=contour_levels,
            contour_step=args.contour_step,
            color_vmax=color_vmax,
            output_path=heatmap_path,
            dpi=args.dpi,
        )

        row = {
            "pressure_level_hpa": float(level_hpa),
            "map": display_path(heatmap_path),
            "array": display_path(array_path),
            "color_vmax_degrees": color_vmax,
            "color_vmax_percentile": float(args.color_vmax_percentile),
            **diagnostics,
        }
        output_rows.append(row)
        diagnostics_rows.append(row)

    diagnostics_path = output_dir / "distance_sum_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(diagnostics_rows[0].keys()) if diagnostics_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics_rows:
            writer.writerow(row)

    summary = {
        "process": (
            "global thermal-displacement 5-point score contours, colored by "
            "each ERA5 grid cell's summed distance to the two closest distinct "
            "contour segments"
        ),
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "matching_mode": "same-hemisphere",
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cells on score; "
            "longitude wraps and latitude uses nearest edge."
        ),
        "distance_metric": (
            "Euclidean distance in lon/lat degrees to sampled contour paths; "
            "longitude is treated as periodic by duplicating contour samples "
            "at +/-360 degrees."
        ),
        "contour_levels": [float(level) for level in contour_levels],
        "contour_sample_spacing_degrees": float(args.contour_sample_spacing_degrees),
        "nearest_sample_query_count": int(args.nearest_sample_query_count),
        "color_scale": (
            "Linear red-to-white scale. Red is small summed distance; white is "
            f"the per-level {args.color_vmax_percentile:g}th percentile or larger."
        ),
        "diagnostics": display_path(diagnostics_path),
        "outputs": output_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
