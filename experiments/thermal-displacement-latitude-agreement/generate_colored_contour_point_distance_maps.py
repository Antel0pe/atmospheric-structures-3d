from __future__ import annotations

import argparse
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

from generate_contour_line_distance_scatter import (  # noqa: E402
    sample_polyline_by_arclength,
    score_field_for_level,
)
from generate_global_contour_paths import (  # noqa: E402
    ContourLine,
    build_contour_lines,
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
    parse_requested_levels,
    resolve_path,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-contour-point-two-nearest-distance-color-step5-250-1000"
)
DEFAULT_LEVELS = "250,500,850,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render global thermal-displacement score contours on a white map, "
            "coloring each sampled contour point by the summed distance to the "
            "two closest distinct other contour lines."
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
        help="Maximum spacing between colored samples along each contour line.",
    )
    parser.add_argument(
        "--nearest-sample-query-count",
        type=int,
        default=256,
        help="Initial KD-tree sample count used to find distinct neighbor lines.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=100_000,
        help="Number of contour samples per nearest-neighbor query batch.",
    )
    parser.add_argument(
        "--color-vmax-percentile",
        type=float,
        default=99.5,
        help="Percentile used as the black end of the red-black color scale.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def red_black_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "small-red-large-black",
        ["#ff1f1f", "#9b0000", "#2b0000", "#000000"],
    )


def draw_light_borders(
    ax: plt.Axes,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    for segment in border_segments:
        points = [
            (lon, lat)
            for lon, lat in segment
            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
        ]
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color="#b7b7b7", linewidth=0.35, alpha=0.65, zorder=1)


def sample_contour_lines(
    contour_lines: list[ContourLine],
    sample_spacing_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_by_line: list[np.ndarray] = []
    line_ids_by_line: list[np.ndarray] = []
    score_by_line: list[np.ndarray] = []

    for line_index, contour_line in enumerate(contour_lines):
        points = sample_polyline_by_arclength(
            np.asarray(contour_line.vertices, dtype=np.float32),
            sample_spacing_degrees,
        )
        if points.size == 0:
            continue
        points_by_line.append(points)
        line_ids_by_line.append(np.full(points.shape[0], line_index, dtype=np.int32))
        score_by_line.append(
            np.full(points.shape[0], float(contour_line.level), dtype=np.float32)
        )

    if not points_by_line:
        raise ValueError("No contour points were sampled.")

    return (
        np.vstack(points_by_line).astype(np.float32),
        np.concatenate(line_ids_by_line).astype(np.int32),
        np.concatenate(score_by_line).astype(np.float32),
    )


def build_periodic_tree(
    contour_points: np.ndarray,
    line_ids: np.ndarray,
) -> tuple[cKDTree, np.ndarray]:
    longitude_offset = np.asarray([360.0, 0.0], dtype=np.float32)
    periodic_points = np.vstack(
        (
            contour_points,
            contour_points + longitude_offset,
            contour_points - longitude_offset,
        )
    ).astype(np.float32)
    periodic_line_ids = np.tile(line_ids, 3).astype(np.int32)
    return cKDTree(periodic_points), periodic_line_ids


def choose_two_other_line_distances(
    distances: np.ndarray,
    indices: np.ndarray,
    periodic_line_ids: np.ndarray,
    current_line_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_line_ids = periodic_line_ids[indices]
    first_distances = np.full(current_line_ids.shape, np.nan, dtype=np.float32)
    second_distances = np.full(current_line_ids.shape, np.nan, dtype=np.float32)
    first_line_ids = np.full(current_line_ids.shape, -1, dtype=np.int32)

    for column_index in range(neighbor_line_ids.shape[1]):
        candidate_line_ids = neighbor_line_ids[:, column_index]
        is_other_line = candidate_line_ids != current_line_ids

        needs_first = ~np.isfinite(first_distances)
        take_first = needs_first & is_other_line
        first_distances[take_first] = distances[take_first, column_index].astype(
            np.float32
        )
        first_line_ids[take_first] = candidate_line_ids[take_first].astype(np.int32)

        needs_second = np.isfinite(first_distances) & ~np.isfinite(second_distances)
        is_distinct_second = is_other_line & (candidate_line_ids != first_line_ids)
        take_second = needs_second & is_distinct_second
        second_distances[take_second] = distances[take_second, column_index].astype(
            np.float32
        )

        if np.all(np.isfinite(second_distances)):
            break

    return first_distances, second_distances, first_line_ids


def compute_point_distance_sums(
    contour_points: np.ndarray,
    contour_line_ids: np.ndarray,
    nearest_sample_query_count: int,
    query_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(np.unique(contour_line_ids)) < 3:
        raise ValueError("Need at least three contour lines for two other-line neighbors.")

    tree, periodic_line_ids = build_periodic_tree(contour_points, contour_line_ids)
    periodic_sample_count = len(periodic_line_ids)
    query_count = min(max(int(nearest_sample_query_count), 3), periodic_sample_count)
    batch_size = max(int(query_batch_size), 1)
    distance_sums = np.full(contour_points.shape[0], np.nan, dtype=np.float32)
    max_query_count_used = query_count

    for start_index in range(0, contour_points.shape[0], batch_size):
        stop_index = min(start_index + batch_size, contour_points.shape[0])
        batch_points = contour_points[start_index:stop_index]
        batch_line_ids = contour_line_ids[start_index:stop_index]
        active_query_count = query_count

        while True:
            distances, indices = tree.query(
                batch_points,
                k=active_query_count,
                workers=-1,
            )
            if active_query_count == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            first, second, _ = choose_two_other_line_distances(
                distances,
                indices,
                periodic_line_ids,
                batch_line_ids,
            )
            unresolved = ~np.isfinite(second)
            if not np.any(unresolved) or active_query_count >= periodic_sample_count:
                distance_sums[start_index:stop_index] = first + second
                max_query_count_used = max(max_query_count_used, active_query_count)
                break

            active_query_count = min(
                periodic_sample_count,
                max(active_query_count * 4, active_query_count + 1),
            )

    finite_values = distance_sums[np.isfinite(distance_sums)]
    diagnostics = {
        "contour_point_count": int(contour_points.shape[0]),
        "periodic_contour_point_count": int(periodic_sample_count),
        "initial_nearest_sample_query_count": int(query_count),
        "max_nearest_sample_query_count_used": int(max_query_count_used),
        "unresolved_contour_point_count": int(np.count_nonzero(~np.isfinite(distance_sums))),
        "distance_sum_min_degrees": float(np.nanmin(finite_values)),
        "distance_sum_max_degrees": float(np.nanmax(finite_values)),
        "distance_sum_mean_degrees": float(np.nanmean(finite_values)),
        "distance_sum_p50_degrees": float(np.nanpercentile(finite_values, 50.0)),
        "distance_sum_p95_degrees": float(np.nanpercentile(finite_values, 95.0)),
        "distance_sum_p995_degrees": float(np.nanpercentile(finite_values, 99.5)),
    }
    return distance_sums, diagnostics


def plot_colored_contour_points(
    contour_points: np.ndarray,
    distance_sums: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    contour_step: float,
    color_vmax: float,
    output_path: Path,
    dpi: int,
) -> None:
    finite = np.isfinite(distance_sums)
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    draw_light_borders(ax, border_segments)
    scatter = ax.scatter(
        contour_points[finite, 0],
        contour_points[finite, 1],
        c=distance_sums[finite],
        cmap=red_black_colormap(),
        norm=mcolors.Normalize(vmin=0.0, vmax=color_vmax),
        s=0.9,
        linewidths=0.0,
        alpha=0.98,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa: {contour_step:g}-point score contours colored by "
        "two-nearest other-contour distance sum"
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Distance sum in lon/lat degrees; red = small, black = large")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overview(
    level_outputs: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    image_paths = [Path(str(row["map_path"])) for row in level_outputs]
    titles = [f"{float(row['pressure_level_hpa']):g} hPa" for row in level_outputs]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    for ax, image_path, title in zip(axes.ravel(), image_paths, titles):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.ravel()[len(image_paths) :]:
        ax.axis("off")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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

    for level_hpa in selected_levels:
        slug = slug_for_level(level_hpa)
        print(f"Processing {level_hpa:g} hPa")
        score_global = score_field_for_level(
            temperature=temperature,
            climatology=climatology,
            selected_time=selected_time,
            level_hpa=level_hpa,
            source_latitudes=source_latitudes,
            lat_indices=lat_indices,
            lon_indices=lon_indices,
            smooth_sigma_cells=args.smooth_sigma_cells,
        )
        contour_lines = build_contour_lines(
            selected_lons,
            selected_lats,
            score_global,
            contour_levels,
        )
        contour_points, contour_line_ids, _ = sample_contour_lines(
            contour_lines,
            args.contour_sample_spacing_degrees,
        )
        distance_sums, diagnostics = compute_point_distance_sums(
            contour_points=contour_points,
            contour_line_ids=contour_line_ids,
            nearest_sample_query_count=args.nearest_sample_query_count,
            query_batch_size=args.query_batch_size,
        )
        finite_values = distance_sums[np.isfinite(distance_sums)]
        color_vmax = float(
            np.nanpercentile(finite_values, args.color_vmax_percentile)
        )
        plot_path = plot_dir / f"colored_contour_point_distance_sum_{slug}.png"
        plot_colored_contour_points(
            contour_points=contour_points,
            distance_sums=distance_sums,
            border_segments=border_segments,
            level_hpa=level_hpa,
            contour_step=args.contour_step,
            color_vmax=color_vmax,
            output_path=plot_path,
            dpi=args.dpi,
        )
        output_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "map": display_path(plot_path),
                "map_path": str(plot_path),
                "color_vmax_degrees": color_vmax,
                "color_vmax_percentile": float(args.color_vmax_percentile),
                "contour_line_count": int(len(contour_lines)),
                **diagnostics,
            }
        )

    overview_path = plot_dir / "colored_contour_point_distance_sum_overview.png"
    plot_overview(output_rows, overview_path, args.dpi)

    summary_outputs = []
    for row in output_rows:
        row_copy = dict(row)
        row_copy.pop("map_path", None)
        summary_outputs.append(row_copy)

    summary = {
        "process": (
            "Global thermal-displacement score contours on a white background. "
            "Each contour line is sampled every 0.25 lon/lat degrees along "
            "arclength. Each sampled point is colored by the sum of distances "
            "to the two closest distinct other contour lines."
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
        "contour_levels": [float(level) for level in contour_levels],
        "contour_sample_spacing_degrees": float(args.contour_sample_spacing_degrees),
        "distance_metric": (
            "Euclidean distance in lon/lat degrees between sampled contour points; "
            "longitude is treated as periodic by duplicating samples at +/-360 degrees."
        ),
        "current_line_exclusion": (
            "For a sampled point, neighbors on the sampled point's own contour line "
            "are excluded, and the two chosen neighbors must come from two distinct "
            "other contour lines."
        ),
        "color_scale": (
            "Linear red-to-black scale. Red is a small two-nearest distance sum; "
            f"black is the per-level {args.color_vmax_percentile:g}th percentile "
            "or larger. The map background is white and is not data-colored."
        ),
        "overview": display_path(overview_path),
        "outputs": summary_outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
