from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_global_contour_paths import (  # noqa: E402
    ContourLine,
    build_contour_lines,
    closest_point_on_polyline,
    sorted_global_domain,
)
from generate_latitude_score_lines import slug_for_level  # noqa: E402
from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-contour-line-two-nearest-distance-scatter-step5-250-1000"
)
DEFAULT_LEVELS = "250,500,850,1000"


@dataclass(frozen=True)
class LineSampleSet:
    points: np.ndarray
    fallback_used: bool


@dataclass(frozen=True)
class NeighborDistances:
    first_distance: float
    first_line_index: int
    second_distance: float
    second_line_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each separate global thermal-displacement score contour line, "
            "sample the line at 0.25-degree longitude crossings, find the two "
            "nearest different contour lines at each sample, sum those "
            "distances, and plot one scatter dot per separate contour line."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--longitude-sample-step", type=float, default=0.25)
    parser.add_argument(
        "--contour-tree-sample-spacing-degrees",
        type=float,
        default=0.25,
        help=(
            "Maximum spacing for contour samples used to discover candidate "
            "neighbor lines before exact polyline distance refinement."
        ),
    )
    parser.add_argument(
        "--nearest-sample-query-count",
        type=int,
        default=256,
        help="Initial KD-tree sample count used to discover candidate lines.",
    )
    parser.add_argument(
        "--candidate-line-limit",
        type=int,
        default=32,
        help="Maximum candidate neighbor lines to exact-refine per sample point.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def sample_polyline_by_arclength(
    vertices: np.ndarray,
    max_spacing_degrees: float,
) -> np.ndarray:
    spacing = max(float(max_spacing_degrees), 1.0e-6)
    points: list[np.ndarray] = []
    for segment_index, (start, end) in enumerate(zip(vertices[:-1], vertices[1:])):
        distance = float(np.linalg.norm(end - start))
        step_count = max(1, int(math.ceil(distance / spacing)))
        t = np.linspace(0.0, 1.0, step_count + 1, dtype=np.float32)
        segment_points = start + (end - start) * t[:, np.newaxis]
        if segment_index > 0:
            segment_points = segment_points[1:]
        points.append(segment_points)
    if not points:
        return np.asarray(vertices, dtype=np.float32)
    return np.vstack(points).astype(np.float32)


def sample_line_at_longitude_crossings(
    vertices: np.ndarray,
    longitude_step: float,
    longitude_min: float,
    longitude_max: float,
) -> LineSampleSet:
    step = max(float(longitude_step), 1.0e-6)
    epsilon = 1.0e-6
    samples: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    def add_sample(lon: float, lat: float) -> None:
        if lon < longitude_min - epsilon or lon > longitude_max + epsilon:
            return
        key = (int(round(lon / step)), int(round(lat * 1_000_000.0)))
        if key in seen:
            return
        seen.add(key)
        samples.append((float(lon), float(lat)))

    for start, end in zip(vertices[:-1], vertices[1:]):
        lon0 = float(start[0])
        lat0 = float(start[1])
        lon1 = float(end[0])
        lat1 = float(end[1])
        delta_lon = lon1 - lon0

        if abs(delta_lon) <= epsilon:
            grid_lon = round(lon0 / step) * step
            if abs(grid_lon - lon0) <= 1.0e-5:
                add_sample(grid_lon, 0.5 * (lat0 + lat1))
            continue

        low = max(min(lon0, lon1), longitude_min)
        high = min(max(lon0, lon1), longitude_max)
        first_index = int(math.ceil((low - epsilon) / step))
        last_index = int(math.floor((high + epsilon) / step))
        for longitude_index in range(first_index, last_index + 1):
            lon = longitude_index * step
            t = (lon - lon0) / delta_lon
            if t < -epsilon or t > 1.0 + epsilon:
                continue
            lat = lat0 + (lat1 - lat0) * t
            add_sample(lon, lat)

    if samples:
        return LineSampleSet(
            points=np.asarray(samples, dtype=np.float32),
            fallback_used=False,
        )

    fallback_points = sample_polyline_by_arclength(vertices, step)
    return LineSampleSet(points=fallback_points, fallback_used=True)


def sample_all_contours_for_tree(
    contour_lines: list[ContourLine],
    sample_spacing_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_by_line: list[np.ndarray] = []
    line_ids: list[np.ndarray] = []
    for line_index, contour_line in enumerate(contour_lines):
        points = sample_polyline_by_arclength(
            np.asarray(contour_line.vertices, dtype=np.float32),
            sample_spacing_degrees,
        )
        if points.size == 0:
            continue
        points_by_line.append(points)
        line_ids.append(np.full(points.shape[0], line_index, dtype=np.int32))

    if not points_by_line:
        raise ValueError("No contour points were available for neighbor search.")

    return (
        np.vstack(points_by_line).astype(np.float32),
        np.concatenate(line_ids).astype(np.int32),
    )


def build_periodic_tree(
    points: np.ndarray,
    line_ids: np.ndarray,
) -> tuple[cKDTree, np.ndarray]:
    offset = np.asarray([360.0, 0.0], dtype=np.float32)
    periodic_points = np.vstack((points, points + offset, points - offset)).astype(
        np.float32
    )
    periodic_line_ids = np.tile(line_ids, 3).astype(np.int32)
    return cKDTree(periodic_points), periodic_line_ids


def closest_distance_to_line_periodic(
    point: np.ndarray,
    line: ContourLine,
) -> float:
    vertices = np.asarray(line.vertices, dtype=np.float32)
    best_distance = float("inf")
    for longitude_shift in (-360.0, 0.0, 360.0):
        shifted_vertices = vertices + np.asarray(
            [longitude_shift, 0.0],
            dtype=np.float32,
        )
        _, distance = closest_point_on_polyline(point, shifted_vertices)
        if distance < best_distance:
            best_distance = distance
    return best_distance


def candidate_line_ids_for_point(
    point: np.ndarray,
    current_line_index: int,
    tree: cKDTree,
    periodic_line_ids: np.ndarray,
    nearest_sample_query_count: int,
    candidate_line_limit: int,
) -> list[int]:
    sample_count = len(periodic_line_ids)
    query_count = min(max(int(nearest_sample_query_count), 2), sample_count)
    candidate_limit = max(int(candidate_line_limit), 2)

    while True:
        distances, indices = tree.query(point, k=query_count, workers=-1)
        indices = np.atleast_1d(indices)
        candidate_ids: list[int] = []
        seen: set[int] = set()
        for index in indices:
            line_id = int(periodic_line_ids[int(index)])
            if line_id == current_line_index or line_id in seen:
                continue
            seen.add(line_id)
            candidate_ids.append(line_id)
            if len(candidate_ids) >= candidate_limit:
                return candidate_ids
        if len(candidate_ids) >= 2 or query_count >= sample_count:
            return candidate_ids
        query_count = min(sample_count, max(query_count * 4, query_count + 1))


def find_two_closest_other_lines(
    point: np.ndarray,
    current_line_index: int,
    contour_lines: list[ContourLine],
    tree: cKDTree,
    periodic_line_ids: np.ndarray,
    nearest_sample_query_count: int,
    candidate_line_limit: int,
) -> NeighborDistances | None:
    candidate_ids = candidate_line_ids_for_point(
        point=point,
        current_line_index=current_line_index,
        tree=tree,
        periodic_line_ids=periodic_line_ids,
        nearest_sample_query_count=nearest_sample_query_count,
        candidate_line_limit=candidate_line_limit,
    )
    if len(candidate_ids) < 2:
        return None

    distances = [
        (
            closest_distance_to_line_periodic(point, contour_lines[line_id]),
            line_id,
        )
        for line_id in candidate_ids
    ]
    distances.sort(key=lambda item: item[0])
    first_distance, first_line_index = distances[0]
    second_distance, second_line_index = distances[1]
    return NeighborDistances(
        first_distance=float(first_distance),
        first_line_index=int(first_line_index),
        second_distance=float(second_distance),
        second_line_index=int(second_line_index),
    )


def score_field_for_level(
    temperature: xr.DataArray,
    climatology: xr.DataArray,
    selected_time: np.datetime64,
    level_hpa: float,
    source_latitudes: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    smooth_sigma_cells: float,
) -> np.ndarray:
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
    score_smoothed = smooth_wrapped_lon(score_unsmoothed, smooth_sigma_cells)
    return score_smoothed[np.ix_(lat_indices, lon_indices)]


def compute_line_metrics(
    contour_lines: list[ContourLine],
    longitude_step: float,
    longitude_min: float,
    longitude_max: float,
    contour_tree_sample_spacing_degrees: float,
    nearest_sample_query_count: int,
    candidate_line_limit: int,
) -> list[dict[str, object]]:
    if len(contour_lines) < 3:
        raise ValueError("Need at least three contour lines for two non-current neighbors.")

    tree_points, tree_line_ids = sample_all_contours_for_tree(
        contour_lines,
        contour_tree_sample_spacing_degrees,
    )
    tree, periodic_line_ids = build_periodic_tree(tree_points, tree_line_ids)

    rows: list[dict[str, object]] = []
    for line_index, contour_line in enumerate(contour_lines):
        samples = sample_line_at_longitude_crossings(
            np.asarray(contour_line.vertices, dtype=np.float32),
            longitude_step,
            longitude_min,
            longitude_max,
        )
        per_sample_distance_sums: list[float] = []
        first_neighbor_ids: set[int] = set()
        second_neighbor_ids: set[int] = set()
        unresolved_count = 0

        for point in samples.points:
            neighbors = find_two_closest_other_lines(
                point=point,
                current_line_index=line_index,
                contour_lines=contour_lines,
                tree=tree,
                periodic_line_ids=periodic_line_ids,
                nearest_sample_query_count=nearest_sample_query_count,
                candidate_line_limit=candidate_line_limit,
            )
            if neighbors is None:
                unresolved_count += 1
                continue
            first_neighbor_ids.add(neighbors.first_line_index)
            second_neighbor_ids.add(neighbors.second_line_index)
            per_sample_distance_sums.append(
                neighbors.first_distance + neighbors.second_distance
            )

        resolved_count = len(per_sample_distance_sums)
        total_distance = float(np.sum(per_sample_distance_sums, dtype=np.float64))
        mean_distance = (
            float(np.mean(per_sample_distance_sums)) if per_sample_distance_sums else float("nan")
        )
        median_distance = (
            float(np.median(per_sample_distance_sums)) if per_sample_distance_sums else float("nan")
        )

        rows.append(
            {
                "contour_line_index": int(line_index),
                "score_contour": float(contour_line.level),
                "contour_segment": int(contour_line.contour_id[1]),
                "vertex_count": int(len(contour_line.vertices)),
                "sample_count": int(len(samples.points)),
                "resolved_sample_count": int(resolved_count),
                "unresolved_sample_count": int(unresolved_count),
                "used_arclength_fallback": bool(samples.fallback_used),
                "total_two_nearest_distance_degrees": total_distance,
                "mean_two_nearest_distance_degrees": mean_distance,
                "median_two_nearest_distance_degrees": median_distance,
                "unique_first_neighbor_line_count": int(len(first_neighbor_ids)),
                "unique_second_neighbor_line_count": int(len(second_neighbor_ids)),
            }
        )

    return rows


def plot_level_scatter(
    rows: list[dict[str, object]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    scores = np.asarray([row["score_contour"] for row in rows], dtype=np.float32)
    totals = np.asarray(
        [row["total_two_nearest_distance_degrees"] for row in rows],
        dtype=np.float64,
    )
    sample_counts = np.asarray([row["resolved_sample_count"] for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    scatter = ax.scatter(
        scores,
        totals,
        c=sample_counts,
        cmap="viridis",
        s=26,
        alpha=0.82,
        linewidths=0.25,
        edgecolors="#101010",
    )
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks(np.arange(0.0, 101.0, 5.0))
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
    ax.set_xlabel("Thermal-displacement score contour")
    ax.set_ylabel("Total two-nearest-line distance sum (lon/lat degrees)")
    ax.set_title(
        f"{level_hpa:g} hPa: one dot per separate score-contour line"
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Resolved 0.25-degree longitude samples on line")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_combined_scatter(
    rows: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    levels = sorted({float(row["pressure_level_hpa"]) for row in rows})
    colors = {
        level: color
        for level, color in zip(levels, ["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    }

    fig, ax = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    for level in levels:
        level_rows = [row for row in rows if float(row["pressure_level_hpa"]) == level]
        ax.scatter(
            [row["score_contour"] for row in level_rows],
            [row["total_two_nearest_distance_degrees"] for row in level_rows],
            s=18,
            alpha=0.72,
            linewidths=0.2,
            edgecolors="#111111",
            color=colors[level],
            label=f"{level:g} hPa",
        )
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks(np.arange(0.0, 101.0, 5.0))
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
    ax.set_xlabel("Thermal-displacement score contour")
    ax.set_ylabel("Total two-nearest-line distance sum (lon/lat degrees)")
    ax.set_title("Contour-line distance sums by pressure level")
    ax.legend(title="Pressure", frameon=False, ncols=2)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
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
    contour_levels = np.arange(args.contour_step, 100.0, args.contour_step, dtype=np.float32)

    all_rows: list[dict[str, object]] = []
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
        level_rows = compute_line_metrics(
            contour_lines=contour_lines,
            longitude_step=args.longitude_sample_step,
            longitude_min=float(np.min(selected_lons)),
            longitude_max=float(np.max(selected_lons)),
            contour_tree_sample_spacing_degrees=args.contour_tree_sample_spacing_degrees,
            nearest_sample_query_count=args.nearest_sample_query_count,
            candidate_line_limit=args.candidate_line_limit,
        )
        for row in level_rows:
            row["pressure_level_hpa"] = float(level_hpa)
        all_rows.extend(level_rows)

        plot_path = plot_dir / f"contour_line_distance_scatter_{slug}.png"
        plot_level_scatter(level_rows, level_hpa, plot_path, args.dpi)

        finite_totals = np.asarray(
            [
                row["total_two_nearest_distance_degrees"]
                for row in level_rows
                if np.isfinite(row["total_two_nearest_distance_degrees"])
            ],
            dtype=np.float64,
        )
        output_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "plot": display_path(plot_path),
                "contour_line_count": int(len(contour_lines)),
                "sample_count": int(sum(int(row["sample_count"]) for row in level_rows)),
                "resolved_sample_count": int(
                    sum(int(row["resolved_sample_count"]) for row in level_rows)
                ),
                "unresolved_sample_count": int(
                    sum(int(row["unresolved_sample_count"]) for row in level_rows)
                ),
                "arclength_fallback_line_count": int(
                    sum(bool(row["used_arclength_fallback"]) for row in level_rows)
                ),
                "total_distance_sum_min_degrees": float(np.nanmin(finite_totals)),
                "total_distance_sum_max_degrees": float(np.nanmax(finite_totals)),
                "total_distance_sum_mean_degrees": float(np.nanmean(finite_totals)),
                "total_distance_sum_median_degrees": float(np.nanmedian(finite_totals)),
            }
        )

    metrics_path = output_dir / "contour_line_distance_sums.csv"
    write_csv(metrics_path, all_rows)
    combined_plot_path = plot_dir / "contour_line_distance_scatter_all_levels.png"
    plot_combined_scatter(all_rows, combined_plot_path, args.dpi)

    summary = {
        "process": (
            "For every separate thermal-displacement score contour line, sample "
            "0.25-degree longitude crossings along that line. At each sampled "
            "point, find the two closest different contour lines, sum those two "
            "distances, then sum across samples to produce one scatter dot per "
            "separate contour line."
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
        "longitude_sample_step_degrees": float(args.longitude_sample_step),
        "distance_metric": (
            "Euclidean distance in lon/lat degrees. Neighbor discovery uses a "
            "periodic-longitude KD-tree, then distances are exact-refined "
            "against candidate contour polylines with +/-360 longitude shifts."
        ),
        "current_line_exclusion": (
            "The contour line being sampled is excluded, and the two chosen "
            "neighbor points must come from two different other contour lines."
        ),
        "metrics": display_path(metrics_path),
        "combined_plot": display_path(combined_plot_path),
        "outputs": output_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
