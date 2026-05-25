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


DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-contour-point-two-nearest-distance-color-step5-250-1000/"
    "pattern-analysis"
)
DEFAULT_LEVELS = "250,500,850,1000"


LAT_BANDS = (
    ("0-23.5 tropical", 0.0, 23.5),
    ("23.5-35 subtropical", 23.5, 35.0),
    ("35-60 midlatitude", 35.0, 60.0),
    ("60-75 high-latitude", 60.0, 75.0),
    ("75-90 polar", 75.0, 90.1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze patterns in contour-point two-nearest distance sums."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--contour-sample-spacing-degrees", type=float, default=0.25)
    parser.add_argument("--nearest-sample-query-count", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=100_000)
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


def lat_band_name(latitude_abs: float) -> str:
    for name, lower, upper in LAT_BANDS:
        if lower <= latitude_abs < upper:
            return name
    return "outside"


def summarize_values(values: np.ndarray, prefix: str = "") -> dict[str, object]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": float("nan"),
            f"{prefix}median": float("nan"),
            f"{prefix}p10": float("nan"),
            f"{prefix}p90": float("nan"),
        }
    return {
        f"{prefix}count": int(finite.size),
        f"{prefix}mean": float(np.mean(finite)),
        f"{prefix}median": float(np.median(finite)),
        f"{prefix}p10": float(np.percentile(finite, 10.0)),
        f"{prefix}p90": float(np.percentile(finite, 90.0)),
    }


def plot_score_medians(
    score_rows: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    for level in sorted({float(row["pressure_level_hpa"]) for row in score_rows}):
        rows = [
            row
            for row in score_rows
            if float(row["pressure_level_hpa"]) == level
        ]
        rows.sort(key=lambda row: float(row["score_contour"]))
        ax.plot(
            [float(row["score_contour"]) for row in rows],
            [float(row["distance_sum_median_degrees"]) for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{level:g} hPa",
        )
    ax.set_xlabel("Thermal-displacement score contour")
    ax.set_ylabel("Median two-nearest distance sum (lon/lat degrees)")
    ax.set_title("Which score contours are compressed or isolated?")
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
    ax.set_xticks(np.arange(5.0, 100.0, 5.0))
    ax.legend(title="Pressure", frameon=False, ncols=2)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_lat_band_extremes(
    lat_rows: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    levels = sorted({float(row["pressure_level_hpa"]) for row in lat_rows})
    band_names = [band[0] for band in LAT_BANDS]
    colors = ["#2468a2", "#5aa3cf", "#f2c14e", "#e67f51", "#9a3324"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, class_name, title in (
        (axes[0], "smallest_10_percent", "Smallest 10% distance sums"),
        (axes[1], "largest_10_percent", "Largest 10% distance sums"),
    ):
        bottoms = np.zeros(len(levels), dtype=np.float64)
        for band_name, color in zip(band_names, colors):
            values = []
            for level in levels:
                row = next(
                    (
                        row
                        for row in lat_rows
                        if float(row["pressure_level_hpa"]) == level
                        and row["extreme_class"] == class_name
                        and row["latitude_band"] == band_name
                    ),
                    None,
                )
                values.append(0.0 if row is None else float(row["fraction_of_extreme"]))
            ax.bar(
                [f"{level:g}" for level in levels],
                values,
                bottom=bottoms,
                label=band_name,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += np.asarray(values, dtype=np.float64)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction of extreme points")
        ax.set_title(title)
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.55)
    axes[1].set_xlabel("Pressure level (hPa)")
    axes[0].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_extreme_score_heatmap(
    extreme_score_rows: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    levels = sorted({float(row["pressure_level_hpa"]) for row in extreme_score_rows})
    scores = sorted({float(row["score_contour"]) for row in extreme_score_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for ax, class_name, title, cmap in (
        (axes[0], "smallest_10_percent", "Smallest 10% by score contour", "Reds"),
        (axes[1], "largest_10_percent", "Largest 10% by score contour", "Greys"),
    ):
        matrix = np.zeros((len(levels), len(scores)), dtype=np.float64)
        for row in extreme_score_rows:
            if row["extreme_class"] != class_name:
                continue
            level_index = levels.index(float(row["pressure_level_hpa"]))
            score_index = scores.index(float(row["score_contour"]))
            matrix[level_index, score_index] = float(row["fraction_of_extreme"])
        mesh = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0)
        ax.set_yticks(np.arange(len(levels)))
        ax.set_yticklabels([f"{level:g}" for level in levels])
        ax.set_xticks(np.arange(len(scores)))
        ax.set_xticklabels([f"{score:g}" for score in scores], rotation=90)
        ax.set_xlabel("Score contour")
        ax.set_ylabel("Pressure level (hPa)")
        ax.set_title(title)
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        colorbar.set_label("Fraction of extreme points")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_latitude_profiles(
    latitude_rows: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    for level in sorted({float(row["pressure_level_hpa"]) for row in latitude_rows}):
        rows = [
            row
            for row in latitude_rows
            if float(row["pressure_level_hpa"]) == level
        ]
        rows.sort(key=lambda row: float(row["latitude_bin_center"]))
        ax.plot(
            [float(row["latitude_bin_center"]) for row in rows],
            [float(row["distance_sum_median_degrees"]) for row in rows],
            linewidth=1.4,
            label=f"{level:g} hPa",
        )
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Median two-nearest distance sum (lon/lat degrees)")
    ax.set_title("Contour compression by latitude")
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.65)
    ax.legend(title="Pressure", frameon=False, ncols=2)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
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
    contour_levels = np.arange(args.contour_step, 100.0, args.contour_step, dtype=np.float32)

    score_rows: list[dict[str, object]] = []
    lat_band_rows: list[dict[str, object]] = []
    extreme_score_rows: list[dict[str, object]] = []
    latitude_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []

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
        distances = distance_sums[finite].astype(np.float64)
        lats = contour_points[finite, 1].astype(np.float64)
        score_contours = score_values[finite].astype(np.float64)
        q10 = float(np.percentile(distances, 10.0))
        q90 = float(np.percentile(distances, 90.0))
        small_mask = distances <= q10
        large_mask = distances >= q90

        level_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "smallest_10_percent_threshold_degrees": q10,
                "largest_10_percent_threshold_degrees": q90,
                **summarize_values(distances, "distance_sum_"),
                **diagnostics,
            }
        )

        for score_contour in sorted(np.unique(score_contours)):
            mask = score_contours == score_contour
            if not np.any(mask):
                continue
            values = distances[mask]
            score_rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "score_contour": float(score_contour),
                    "point_count": int(np.count_nonzero(mask)),
                    "distance_sum_mean_degrees": float(np.mean(values)),
                    "distance_sum_median_degrees": float(np.median(values)),
                    "smallest_10_percent_fraction": float(np.mean(small_mask[mask])),
                    "largest_10_percent_fraction": float(np.mean(large_mask[mask])),
                    "mean_abs_latitude": float(np.mean(np.abs(lats[mask]))),
                }
            )

        for class_name, class_mask in (
            ("smallest_10_percent", small_mask),
            ("largest_10_percent", large_mask),
        ):
            class_count = int(np.count_nonzero(class_mask))
            for band_name, lower, upper in LAT_BANDS:
                band_mask = (np.abs(lats) >= lower) & (np.abs(lats) < upper)
                count = int(np.count_nonzero(class_mask & band_mask))
                lat_band_rows.append(
                    {
                        "pressure_level_hpa": float(level_hpa),
                        "extreme_class": class_name,
                        "latitude_band": band_name,
                        "point_count": count,
                        "fraction_of_extreme": float(count / class_count) if class_count else 0.0,
                    }
                )

            for score_contour in sorted(np.unique(score_contours)):
                score_mask = score_contours == score_contour
                count = int(np.count_nonzero(class_mask & score_mask))
                extreme_score_rows.append(
                    {
                        "pressure_level_hpa": float(level_hpa),
                        "extreme_class": class_name,
                        "score_contour": float(score_contour),
                        "point_count": count,
                        "fraction_of_extreme": float(count / class_count) if class_count else 0.0,
                    }
                )

        latitude_bins = np.arange(-90.0, 95.0, 5.0)
        for lower, upper in zip(latitude_bins[:-1], latitude_bins[1:]):
            mask = (lats >= lower) & (lats < upper)
            if not np.any(mask):
                continue
            values = distances[mask]
            latitude_rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "latitude_bin_center": float(0.5 * (lower + upper)),
                    "point_count": int(np.count_nonzero(mask)),
                    "distance_sum_mean_degrees": float(np.mean(values)),
                    "distance_sum_median_degrees": float(np.median(values)),
                    "smallest_10_percent_fraction": float(np.mean(small_mask[mask])),
                    "largest_10_percent_fraction": float(np.mean(large_mask[mask])),
                }
            )

    level_csv = output_dir / "level_distance_summary.csv"
    score_csv = output_dir / "score_contour_distance_summary.csv"
    lat_band_csv = output_dir / "extreme_latitude_band_fractions.csv"
    extreme_score_csv = output_dir / "extreme_score_contour_fractions.csv"
    latitude_csv = output_dir / "latitude_profile_summary.csv"
    write_csv(level_csv, level_rows)
    write_csv(score_csv, score_rows)
    write_csv(lat_band_csv, lat_band_rows)
    write_csv(extreme_score_csv, extreme_score_rows)
    write_csv(latitude_csv, latitude_rows)

    score_plot = output_dir / "median_distance_by_score_contour.png"
    lat_band_plot = output_dir / "extreme_distance_latitude_bands.png"
    score_heatmap_plot = output_dir / "extreme_distance_score_contours.png"
    latitude_plot = output_dir / "median_distance_by_latitude.png"
    plot_score_medians(score_rows, score_plot, args.dpi)
    plot_lat_band_extremes(lat_band_rows, lat_band_plot, args.dpi)
    plot_extreme_score_heatmap(extreme_score_rows, score_heatmap_plot, args.dpi)
    plot_latitude_profiles(latitude_rows, latitude_plot, args.dpi)

    smallest_by_level = {}
    largest_by_level = {}
    for level in sorted({float(row["pressure_level_hpa"]) for row in score_rows}):
        rows = [
            row for row in score_rows if float(row["pressure_level_hpa"]) == level
        ]
        smallest_by_level[f"{level:g} hPa"] = sorted(
            rows,
            key=lambda row: float(row["distance_sum_median_degrees"]),
        )[:4]
        largest_by_level[f"{level:g} hPa"] = sorted(
            rows,
            key=lambda row: float(row["distance_sum_median_degrees"]),
            reverse=True,
        )[:4]

    summary = {
        "process": "Pattern analysis of contour-point two-nearest distance sums.",
        "source_experiment": (
            "global-contour-point-two-nearest-distance-color-step5-250-1000"
        ),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "distance_metric": (
            "Euclidean lon/lat-degree distance between sampled contour points; "
            "not kilometers."
        ),
        "smallest_by_median_score_contour": smallest_by_level,
        "largest_by_median_score_contour": largest_by_level,
        "outputs": {
            "level_summary": display_path(level_csv),
            "score_contour_summary": display_path(score_csv),
            "extreme_latitude_band_fractions": display_path(lat_band_csv),
            "extreme_score_contour_fractions": display_path(extreme_score_csv),
            "latitude_profile_summary": display_path(latitude_csv),
            "median_distance_by_score_contour": display_path(score_plot),
            "extreme_distance_latitude_bands": display_path(lat_band_plot),
            "extreme_distance_score_contours": display_path(score_heatmap_plot),
            "median_distance_by_latitude": display_path(latitude_plot),
        },
    }
    summary_path = output_dir / "pattern_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
