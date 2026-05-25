from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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

from generate_colored_contour_point_distance_maps import (  # noqa: E402
    compute_point_distance_sums,
    sample_contour_lines,
)
from generate_contour_line_distance_scatter import score_field_for_level  # noqa: E402
from generate_global_contour_paths import build_contour_lines, sorted_global_domain  # noqa: E402
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
    "global-contour-point-two-nearest-distance-color-step5-250-1000/"
    "vertical-front-stack-candidates"
)
DEFAULT_LEVELS = "250,500,850,1000"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(self, item: tuple[int, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find spatially stable front-like thermal-displacement contour "
            "compression segments across pressure levels."
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
    parser.add_argument("--contour-sample-spacing-degrees", type=float, default=0.25)
    parser.add_argument("--nearest-sample-query-count", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=100_000)
    parser.add_argument("--grid-degrees", type=float, default=5.0)
    parser.add_argument(
        "--smallest-distance-fraction",
        type=float,
        default=0.15,
        help="Per-level tightest distance-sum fraction to treat as compression candidates.",
    )
    parser.add_argument("--score-min", type=float, default=15.0)
    parser.add_argument("--score-max", type=float, default=85.0)
    parser.add_argument("--abs-lat-min", type=float, default=20.0)
    parser.add_argument("--abs-lat-max", type=float, default=75.0)
    parser.add_argument("--min-level-count", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        ax.plot(xs, ys, color="#b8b8b8", linewidth=0.35, alpha=0.65, zorder=1)


def bin_points(
    lon: np.ndarray,
    lat: np.ndarray,
    grid_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_edges = np.arange(-180.0, 180.0 + grid_degrees, grid_degrees, dtype=np.float64)
    lat_edges = np.arange(-90.0, 90.0 + grid_degrees, grid_degrees, dtype=np.float64)
    lon_bins = np.floor((lon + 180.0) / grid_degrees).astype(np.int32)
    lat_bins = np.floor((lat + 90.0) / grid_degrees).astype(np.int32)
    lon_bins = np.clip(lon_bins, 0, len(lon_edges) - 2)
    lat_bins = np.clip(lat_bins, 0, len(lat_edges) - 2)
    return lon_bins, lat_bins, lon_edges, lat_edges


def connected_components_with_lon_wrap(
    active_cells: set[tuple[int, int]],
    lon_bin_count: int,
) -> dict[tuple[int, int], int]:
    union_find = UnionFind()
    for cell in active_cells:
        union_find.add(cell)

    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for lon_bin, lat_bin in active_cells:
        for lon_offset, lat_offset in neighbor_offsets:
            neighbor = ((lon_bin + lon_offset) % lon_bin_count, lat_bin + lat_offset)
            if neighbor in active_cells:
                union_find.union((lon_bin, lat_bin), neighbor)

    roots = sorted({union_find.find(cell) for cell in active_cells})
    root_to_id = {root: index + 1 for index, root in enumerate(roots)}
    return {cell: root_to_id[union_find.find(cell)] for cell in active_cells}


def plot_stack_map(
    grid_level_count: np.ndarray,
    component_by_cell: dict[tuple[int, int], int],
    component_rows: list[dict[str, object]],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    masked = np.ma.masked_where(grid_level_count <= 0, grid_level_count)
    cmap = mcolors.ListedColormap(["#f6c85f", "#e45756", "#6f1d1b", "#101010"])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    draw_light_borders(ax, border_segments)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        alpha=0.82,
        zorder=2,
    )
    for row in component_rows[:25]:
        ax.text(
            float(row["centroid_lon"]),
            float(row["centroid_lat"]),
            str(row["front_stack_id"]),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#111111",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=4,
        )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Vertically coherent contour-compression candidates")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, ticks=[1, 2, 3, 4])
    colorbar.set_label("Number of pressure levels with front-like compression")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_level_candidate_overlay(
    candidate_points_by_level: dict[float, np.ndarray],
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    colors = {
        250.0: "#4c78a8",
        500.0: "#f58518",
        850.0: "#54a24b",
        1000.0: "#e45756",
    }
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    draw_light_borders(ax, border_segments)
    for level, points in sorted(candidate_points_by_level.items()):
        color = colors.get(float(level), "#333333")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=0.45,
            color=color,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
            label=f"{level:g} hPa",
            zorder=2,
        )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Per-level front-like compression candidates")
    ax.legend(title="Pressure", frameon=False, ncols=2, loc="upper center")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    grid_level_sets: dict[tuple[int, int], set[float]] = defaultdict(set)
    grid_stats: dict[tuple[int, int, float], dict[str, list[float]]] = defaultdict(
        lambda: {"distance": [], "score": [], "lon": [], "lat": []}
    )
    candidate_points_by_level: dict[float, np.ndarray] = {}
    level_rows: list[dict[str, object]] = []
    lon_edges = lat_edges = None

    for level_hpa in selected_levels:
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
        contour_points, contour_line_ids, score_values = sample_contour_lines(
            contour_lines,
            args.contour_sample_spacing_degrees,
        )
        distance_sums, diagnostics = compute_point_distance_sums(
            contour_points=contour_points,
            contour_line_ids=contour_line_ids,
            nearest_sample_query_count=args.nearest_sample_query_count,
            query_batch_size=args.query_batch_size,
        )
        finite = np.isfinite(distance_sums)
        q = float(np.nanpercentile(distance_sums[finite], args.smallest_distance_fraction * 100.0))
        lats = contour_points[:, 1]
        abs_lats = np.abs(lats)
        candidate_mask = (
            finite
            & (distance_sums <= q)
            & (score_values >= args.score_min)
            & (score_values <= args.score_max)
            & (abs_lats >= args.abs_lat_min)
            & (abs_lats <= args.abs_lat_max)
        )
        candidate_points = contour_points[candidate_mask]
        candidate_scores = score_values[candidate_mask]
        candidate_distances = distance_sums[candidate_mask]
        candidate_points_by_level[float(level_hpa)] = candidate_points
        lon_bins, lat_bins, level_lon_edges, level_lat_edges = bin_points(
            candidate_points[:, 0],
            candidate_points[:, 1],
            args.grid_degrees,
        )
        lon_edges = level_lon_edges
        lat_edges = level_lat_edges

        for lon_bin, lat_bin, point, score, distance in zip(
            lon_bins,
            lat_bins,
            candidate_points,
            candidate_scores,
            candidate_distances,
        ):
            cell = (int(lon_bin), int(lat_bin))
            grid_level_sets[cell].add(float(level_hpa))
            key = (int(lon_bin), int(lat_bin), float(level_hpa))
            grid_stats[key]["distance"].append(float(distance))
            grid_stats[key]["score"].append(float(score))
            grid_stats[key]["lon"].append(float(point[0]))
            grid_stats[key]["lat"].append(float(point[1]))

        level_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "distance_threshold_degrees": q,
                "candidate_point_count": int(candidate_points.shape[0]),
                "candidate_cell_count": int(len(set(zip(lon_bins, lat_bins)))),
                "candidate_fraction_of_contour_points": float(
                    candidate_points.shape[0] / np.count_nonzero(finite)
                ),
                **diagnostics,
            }
        )

    if lon_edges is None or lat_edges is None:
        raise ValueError("No candidate points were generated.")

    lon_bin_count = len(lon_edges) - 1
    lat_bin_count = len(lat_edges) - 1
    grid_level_count = np.zeros((lon_bin_count, lat_bin_count), dtype=np.int16)
    for (lon_bin, lat_bin), levels in grid_level_sets.items():
        grid_level_count[lon_bin, lat_bin] = len(levels)

    active_cells = {
        cell
        for cell, levels in grid_level_sets.items()
        if len(levels) >= args.min_level_count
    }
    component_ids = connected_components_with_lon_wrap(active_cells, lon_bin_count)
    cells_by_component: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell, component_id in component_ids.items():
        cells_by_component[component_id].append(cell)

    component_rows: list[dict[str, object]] = []
    for component_id, cells in cells_by_component.items():
        levels_present = sorted(
            {
                level
                for cell in cells
                for level in grid_level_sets[cell]
            }
        )
        level_counts = [len(grid_level_sets[cell]) for cell in cells]
        lons = []
        lats = []
        distances = []
        scores = []
        for lon_bin, lat_bin in cells:
            lons.append(0.5 * (lon_edges[lon_bin] + lon_edges[lon_bin + 1]))
            lats.append(0.5 * (lat_edges[lat_bin] + lat_edges[lat_bin + 1]))
            for level in grid_level_sets[(lon_bin, lat_bin)]:
                stats = grid_stats[(lon_bin, lat_bin, level)]
                distances.extend(stats["distance"])
                scores.extend(stats["score"])
        level_token = "|".join(f"{level:g}" for level in levels_present)
        component_rows.append(
            {
                "component_id": int(component_id),
                "front_stack_id": f"F{component_id:03d}",
                "cell_count": int(len(cells)),
                "max_level_count_in_cell": int(max(level_counts)),
                "mean_level_count_per_cell": float(np.mean(level_counts)),
                "all_4_level_cell_count": int(sum(1 for count in level_counts if count == 4)),
                "pressure_levels_present_hpa": level_token,
                "centroid_lon": float(np.mean(lons)),
                "centroid_lat": float(np.mean(lats)),
                "lon_min": float(np.min(lons) - 0.5 * args.grid_degrees),
                "lon_max": float(np.max(lons) + 0.5 * args.grid_degrees),
                "lat_min": float(np.min(lats) - 0.5 * args.grid_degrees),
                "lat_max": float(np.max(lats) + 0.5 * args.grid_degrees),
                "median_distance_sum_degrees": float(np.median(distances)),
                "median_score_contour": float(np.median(scores)),
                "score_p10": float(np.percentile(scores, 10.0)),
                "score_p90": float(np.percentile(scores, 90.0)),
            }
        )

    component_rows.sort(
        key=lambda row: (
            -int(row["max_level_count_in_cell"]),
            -int(row["all_4_level_cell_count"]),
            -int(row["cell_count"]),
            float(row["median_distance_sum_degrees"]),
        )
    )
    component_id_to_ranked_id: dict[int, str] = {}
    for new_index, row in enumerate(component_rows, start=1):
        component_id_to_ranked_id[int(row["component_id"])] = f"F{new_index:03d}"
        row["front_stack_id"] = f"F{new_index:03d}"

    grid_rows: list[dict[str, object]] = []
    for (lon_bin, lat_bin), levels in sorted(grid_level_sets.items()):
        if len(levels) < args.min_level_count:
            continue
        component_id = component_ids[(lon_bin, lat_bin)]
        grid_rows.append(
            {
                "front_stack_id": component_id_to_ranked_id[component_id],
                "lon_bin": int(lon_bin),
                "lat_bin": int(lat_bin),
                "lon_center": float(0.5 * (lon_edges[lon_bin] + lon_edges[lon_bin + 1])),
                "lat_center": float(0.5 * (lat_edges[lat_bin] + lat_edges[lat_bin + 1])),
                "level_count": int(len(levels)),
                "pressure_levels_present_hpa": "|".join(f"{level:g}" for level in sorted(levels)),
            }
        )

    level_csv = output_dir / "front_like_candidate_level_summary.csv"
    component_csv = output_dir / "front_stack_segments.csv"
    grid_csv = output_dir / "front_stack_cells.csv"
    write_csv(level_csv, level_rows)
    component_public_rows = []
    for row in component_rows:
        public_row = dict(row)
        public_row.pop("component_id", None)
        component_public_rows.append(public_row)
    write_csv(component_csv, component_public_rows)
    write_csv(grid_csv, grid_rows)

    stack_map = output_dir / "front_stack_level_count_map.png"
    overlay_map = output_dir / "front_like_candidates_by_level.png"
    plot_stack_map(
        grid_level_count=grid_level_count,
        component_by_cell=component_ids,
        component_rows=component_rows,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        border_segments=border_segments,
        output_path=stack_map,
        dpi=args.dpi,
    )
    plot_level_candidate_overlay(
        candidate_points_by_level=candidate_points_by_level,
        border_segments=border_segments,
        output_path=overlay_map,
        dpi=args.dpi,
    )

    summary = {
        "process": (
            "Stable front-stack candidates from vertically coherent "
            "thermal-displacement contour compression."
        ),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "candidate_rule": {
            "per_level_smallest_distance_fraction": float(args.smallest_distance_fraction),
            "score_range": [float(args.score_min), float(args.score_max)],
            "absolute_latitude_range": [float(args.abs_lat_min), float(args.abs_lat_max)],
            "grid_degrees": float(args.grid_degrees),
            "min_pressure_levels_per_cell": int(args.min_level_count),
        },
        "interpretation": (
            "A front-stack segment is a connected coarse lon/lat region where "
            "tight, non-endmember thermal-displacement contour compression appears "
            "in at least three pressure levels. It is a stable identifier for a "
            "candidate cold/hot air-mass transition zone, not a formal synoptic "
            "front classification."
        ),
        "front_stack_segment_count": int(len(component_rows)),
        "outputs": {
            "front_stack_segments": display_path(component_csv),
            "front_stack_cells": display_path(grid_csv),
            "level_summary": display_path(level_csv),
            "front_stack_level_count_map": display_path(stack_map),
            "front_like_candidates_by_level": display_path(overlay_map),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
