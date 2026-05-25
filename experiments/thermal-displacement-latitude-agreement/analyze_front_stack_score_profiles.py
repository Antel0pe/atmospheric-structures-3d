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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_vertical_front_stack_candidates import bin_points  # noqa: E402
from generate_colored_contour_point_distance_maps import (  # noqa: E402
    compute_point_distance_sums,
    sample_contour_lines,
)
from generate_contour_line_distance_scatter import score_field_for_level  # noqa: E402
from generate_global_contour_paths import build_contour_lines, sorted_global_domain  # noqa: E402
from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    parse_requested_levels,
    resolve_path,
    validate_matching_grid,
)


SOURCE_STACK_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-contour-point-two-nearest-distance-color-step5-250-1000/"
    "vertical-front-stack-candidates"
)
DEFAULT_OUTPUT_DIR = SOURCE_STACK_DIR / "score-profile"
DEFAULT_LEVELS = "250,500,850,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate common Thermal Displacement score profiles for stable "
            "front-stack candidates by hemisphere and pressure level."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--source-stack-dir", type=Path, default=SOURCE_STACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--contour-sample-spacing-degrees", type=float, default=0.25)
    parser.add_argument("--nearest-sample-query-count", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=100_000)
    parser.add_argument("--grid-degrees", type=float, default=5.0)
    parser.add_argument("--smallest-distance-fraction", type=float, default=0.15)
    parser.add_argument("--score-min", type=float, default=15.0)
    parser.add_argument("--score-max", type=float, default=85.0)
    parser.add_argument("--abs-lat-min", type=float, default=20.0)
    parser.add_argument("--abs-lat-max", type=float, default=75.0)
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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = (float(percentile) / 100.0) * float(np.sum(sorted_weights))
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def dominant_discrete_score(values: np.ndarray, weights: np.ndarray) -> float:
    totals: dict[float, float] = defaultdict(float)
    for value, weight in zip(values, weights):
        totals[float(value)] += float(weight)
    return float(max(totals.items(), key=lambda item: (item[1], -abs(item[0] - 50.0)))[0])


def load_stable_cells(source_stack_dir: Path) -> tuple[dict[tuple[int, int], dict[str, object]], dict[str, dict[str, object]]]:
    cells_path = source_stack_dir / "front_stack_cells.csv"
    segments_path = source_stack_dir / "front_stack_segments.csv"
    stable_cells: dict[tuple[int, int], dict[str, object]] = {}
    with cells_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stable_cells[(int(row["lon_bin"]), int(row["lat_bin"]))] = row
    segments = {}
    with segments_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            segments[row["front_stack_id"]] = row
    return stable_cells, segments


def plot_hemisphere_profile(rows: list[dict[str, object]], output_path: Path, dpi: int) -> None:
    pressure_order = [1000.0, 850.0, 500.0, 250.0]
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    for hemisphere, color in (("northern", "#4c78a8"), ("southern", "#e45756")):
        hemi_rows = [
            row
            for row in rows
            if row["hemisphere"] == hemisphere
        ]
        xs = []
        ys = []
        low = []
        high = []
        for pressure in pressure_order:
            match = next(
                (
                    row
                    for row in hemi_rows
                    if float(row["pressure_level_hpa"]) == pressure
                ),
                None,
            )
            if match is None:
                continue
            xs.append(float(match["score_weighted_mean"]))
            ys.append(pressure)
            low.append(float(match["score_weighted_mean"]) - float(match["score_weighted_p25"]))
            high.append(float(match["score_weighted_p75"]) - float(match["score_weighted_mean"]))
        ax.errorbar(
            xs,
            ys,
            xerr=[low, high],
            marker="o",
            linewidth=1.6,
            capsize=4,
            label=hemisphere.title(),
            color=color,
        )
    ax.set_ylim(1040.0, 220.0)
    ax.set_yticks(pressure_order)
    ax.set_xlim(15.0, 85.0)
    ax.set_xlabel("Thermal Displacement score")
    ax.set_ylabel("Pressure level (hPa)")
    ax.set_title("Front-stack score profile by hemisphere")
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_front_stack_profiles(rows: list[dict[str, object]], output_path: Path, dpi: int) -> None:
    pressure_order = [1000.0, 850.0, 500.0, 250.0]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), constrained_layout=True, sharey=True)
    for ax, hemisphere in zip(axes, ("northern", "southern")):
        stack_ids = sorted({row["front_stack_id"] for row in rows if row["hemisphere"] == hemisphere})
        for stack_id in stack_ids:
            stack_rows = [
                row
                for row in rows
                if row["hemisphere"] == hemisphere and row["front_stack_id"] == stack_id
            ]
            xs = []
            ys = []
            for pressure in pressure_order:
                match = next(
                    (
                        row
                        for row in stack_rows
                        if float(row["pressure_level_hpa"]) == pressure
                    ),
                    None,
                )
                if match is None:
                    continue
                xs.append(float(match["score_weighted_mean"]))
                ys.append(pressure)
            if len(xs) >= 2:
                ax.plot(xs, ys, marker="o", linewidth=1.0, alpha=0.55, label=stack_id)
        ax.set_ylim(1040.0, 220.0)
        ax.set_yticks(pressure_order)
        ax.set_xlim(15.0, 85.0)
        ax.set_xlabel("Thermal Displacement score")
        ax.set_title(hemisphere.title())
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
        ax.legend(frameon=False, fontsize=7, ncols=2)
    axes[0].set_ylabel("Pressure level (hPa)")
    fig.suptitle("Individual stable front-stack score profiles")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    source_stack_dir = args.source_stack_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_cells, segments = load_stable_cells(source_stack_dir)
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

    cell_pressure_values: dict[tuple[str, str, float, int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"score": [], "distance": [], "weight": []}
    )

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
        distance_sums, _ = compute_point_distance_sums(
            contour_points=contour_points,
            contour_line_ids=contour_line_ids,
            nearest_sample_query_count=args.nearest_sample_query_count,
            query_batch_size=args.query_batch_size,
        )
        finite = np.isfinite(distance_sums)
        distance_threshold = float(
            np.nanpercentile(distance_sums[finite], args.smallest_distance_fraction * 100.0)
        )
        abs_lats = np.abs(contour_points[:, 1])
        candidate_mask = (
            finite
            & (distance_sums <= distance_threshold)
            & (score_values >= args.score_min)
            & (score_values <= args.score_max)
            & (abs_lats >= args.abs_lat_min)
            & (abs_lats <= args.abs_lat_max)
        )
        candidate_points = contour_points[candidate_mask]
        candidate_scores = score_values[candidate_mask]
        candidate_distances = distance_sums[candidate_mask]
        lon_bins, lat_bins, _, _ = bin_points(
            candidate_points[:, 0],
            candidate_points[:, 1],
            args.grid_degrees,
        )
        for lon_bin, lat_bin, point, score, distance in zip(
            lon_bins,
            lat_bins,
            candidate_points,
            candidate_scores,
            candidate_distances,
        ):
            cell_key = (int(lon_bin), int(lat_bin))
            if cell_key not in stable_cells:
                continue
            stack_id = str(stable_cells[cell_key]["front_stack_id"])
            hemisphere = "northern" if float(point[1]) >= 0.0 else "southern"
            key = (hemisphere, stack_id, float(level_hpa), int(lon_bin), int(lat_bin))
            # Inverse-distance weighting favors the most compressed contour points
            # while preserving one cell-level observation in the final aggregation.
            weight = 1.0 / max(float(distance), 1.0e-6)
            cell_pressure_values[key]["score"].append(float(score))
            cell_pressure_values[key]["distance"].append(float(distance))
            cell_pressure_values[key]["weight"].append(weight)

    cell_rows: list[dict[str, object]] = []
    for (hemisphere, stack_id, pressure, lon_bin, lat_bin), values in cell_pressure_values.items():
        scores = np.asarray(values["score"], dtype=np.float64)
        distances = np.asarray(values["distance"], dtype=np.float64)
        weights = np.asarray(values["weight"], dtype=np.float64)
        if scores.size == 0:
            continue
        stable_cell = stable_cells[(lon_bin, lat_bin)]
        cell_rows.append(
            {
                "hemisphere": hemisphere,
                "front_stack_id": stack_id,
                "pressure_level_hpa": pressure,
                "lon_bin": lon_bin,
                "lat_bin": lat_bin,
                "lon_center": stable_cell["lon_center"],
                "lat_center": stable_cell["lat_center"],
                "score_weighted_mean": float(np.average(scores, weights=weights)),
                "score_weighted_median": weighted_median(scores, weights),
                "dominant_score_contour": dominant_discrete_score(scores, weights),
                "score_unweighted_mean": float(np.mean(scores)),
                "median_distance_sum_degrees": float(np.median(distances)),
                "sample_count": int(scores.size),
            }
        )

    group_rows: list[dict[str, object]] = []
    by_group: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    by_stack_group: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in cell_rows:
        by_group[(str(row["hemisphere"]), float(row["pressure_level_hpa"]))].append(row)
        by_stack_group[
            (str(row["hemisphere"]), str(row["front_stack_id"]), float(row["pressure_level_hpa"]))
        ].append(row)

    for (hemisphere, pressure), rows in sorted(by_group.items()):
        scores = np.asarray([float(row["score_weighted_mean"]) for row in rows], dtype=np.float64)
        # Weight four-level cells slightly higher than three-level cells, but keep
        # the aggregation primarily cell-balanced.
        weights = np.asarray(
            [
                float(stable_cells[(int(row["lon_bin"]), int(row["lat_bin"]))]["level_count"])
                for row in rows
            ],
            dtype=np.float64,
        )
        group_rows.append(
            {
                "hemisphere": hemisphere,
                "pressure_level_hpa": pressure,
                "stable_cell_count": int(len(rows)),
                "score_weighted_mean": float(np.average(scores, weights=weights)),
                "score_weighted_median": weighted_median(scores, weights),
                "score_weighted_p25": weighted_percentile(scores, weights, 25.0),
                "score_weighted_p75": weighted_percentile(scores, weights, 75.0),
                "score_std": float(np.std(scores)),
                "dominant_cell_score_contour": dominant_discrete_score(
                    np.asarray([float(row["dominant_score_contour"]) for row in rows], dtype=np.float64),
                    weights,
                ),
            }
        )

    stack_rows: list[dict[str, object]] = []
    for (hemisphere, stack_id, pressure), rows in sorted(by_stack_group.items()):
        scores = np.asarray([float(row["score_weighted_mean"]) for row in rows], dtype=np.float64)
        weights = np.asarray(
            [
                float(stable_cells[(int(row["lon_bin"]), int(row["lat_bin"]))]["level_count"])
                for row in rows
            ],
            dtype=np.float64,
        )
        stack_rows.append(
            {
                "hemisphere": hemisphere,
                "front_stack_id": stack_id,
                "pressure_level_hpa": pressure,
                "stable_cell_count": int(len(rows)),
                "score_weighted_mean": float(np.average(scores, weights=weights)),
                "score_weighted_median": weighted_median(scores, weights),
                "score_weighted_p25": weighted_percentile(scores, weights, 25.0),
                "score_weighted_p75": weighted_percentile(scores, weights, 75.0),
                "dominant_cell_score_contour": dominant_discrete_score(
                    np.asarray([float(row["dominant_score_contour"]) for row in rows], dtype=np.float64),
                    weights,
                ),
                "segment_centroid_lon": segments[stack_id]["centroid_lon"],
                "segment_centroid_lat": segments[stack_id]["centroid_lat"],
            }
        )

    smoothness_rows: list[dict[str, object]] = []
    stack_profiles: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in stack_rows:
        stack_profiles[(str(row["hemisphere"]), str(row["front_stack_id"]))].append(row)
    for (hemisphere, stack_id), rows in sorted(stack_profiles.items()):
        if len(rows) < 3:
            continue
        sorted_rows = sorted(rows, key=lambda row: float(row["pressure_level_hpa"]))
        scores = np.asarray(
            [float(row["score_weighted_mean"]) for row in sorted_rows],
            dtype=np.float64,
        )
        pressures = [float(row["pressure_level_hpa"]) for row in sorted_rows]
        pressure_score_pairs = ";".join(
            f"{pressure:g}:{score:.1f}"
            for pressure, score in zip(pressures, scores)
        )
        step_sizes = np.abs(np.diff(scores))
        smoothness_rows.append(
            {
                "hemisphere": hemisphere,
                "front_stack_id": stack_id,
                "pressure_level_count": int(len(rows)),
                "all_4_level_cell_count": int(segments[stack_id]["all_4_level_cell_count"]),
                "score_mean": float(np.mean(scores)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "score_range": float(np.max(scores) - np.min(scores)),
                "max_adjacent_pressure_step": float(np.max(step_sizes))
                if step_sizes.size
                else 0.0,
                "pressure_score_pairs": pressure_score_pairs,
            }
        )
    smoothness_rows.sort(
        key=lambda row: (
            str(row["hemisphere"]),
            -int(row["all_4_level_cell_count"]),
            float(row["score_range"]),
            float(row["max_adjacent_pressure_step"]),
        )
    )

    cell_csv = output_dir / "front_stack_cell_score_profiles.csv"
    group_csv = output_dir / "hemisphere_score_profile.csv"
    stack_csv = output_dir / "front_stack_segment_score_profiles.csv"
    smoothness_csv = output_dir / "front_stack_score_smoothness.csv"
    write_csv(cell_csv, cell_rows)
    write_csv(group_csv, group_rows)
    write_csv(stack_csv, stack_rows)
    write_csv(smoothness_csv, smoothness_rows)

    hemisphere_plot = output_dir / "hemisphere_score_profile.png"
    stack_plot = output_dir / "front_stack_segment_score_profiles.png"
    plot_hemisphere_profile(group_rows, hemisphere_plot, args.dpi)
    plot_front_stack_profiles(stack_rows, stack_plot, args.dpi)

    summary = {
        "process": (
            "Cell-balanced, compression-weighted Thermal Displacement score "
            "profiles for stable front-stack candidates."
        ),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "source_stack_dir": display_path(source_stack_dir),
        "score_metric": (
            "Within each stable front-stack cell and pressure level, score "
            "contour samples are weighted by inverse two-nearest distance sum. "
            "Each stable cell then contributes one score estimate to the "
            "hemisphere and segment profile."
        ),
        "interpretation": (
            "A coherent front-stack score should vary smoothly with pressure "
            "because tilted fronts can shift with height, but large jumps would "
            "suggest the score is not a stable boundary identifier."
        ),
        "outputs": {
            "hemisphere_score_profile": display_path(group_csv),
            "front_stack_segment_score_profiles": display_path(stack_csv),
            "front_stack_score_smoothness": display_path(smoothness_csv),
            "front_stack_cell_score_profiles": display_path(cell_csv),
            "hemisphere_score_profile_plot": display_path(hemisphere_plot),
            "front_stack_segment_score_profiles_plot": display_path(stack_plot),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
