from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_thermal_displacement_score_clusters import (  # noqa: E402
    DEFAULT_LEVELS,
    choose_timestamp,
    display_path,
    parse_requested_levels,
    resolve_path,
    slug_for_level,
)
from scripts.thermal_displacement import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    compute_thermal_displacement_level,
)


DEFAULT_EXPERIMENT_DIR = Path("experiments/thermal-displacement-score-clusters-2021-11-08T12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Thermal Displacement scores as one-dimensional scatter strips."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "score-scatter-plots",
    )
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Matplotlib scatter marker area in points squared.",
    )
    return parser.parse_args()


def plot_score_scatter(
    score: np.ndarray,
    level_hpa: float,
    output_path: Path,
    *,
    dpi: int,
    point_size: float,
) -> None:
    score_values = np.asarray(score[np.isfinite(score)], dtype=np.float32).ravel()
    rng = np.random.default_rng(int(round(level_hpa * 10.0)))
    jitter = rng.uniform(-0.44, 0.44, size=score_values.shape[0]).astype(np.float32)
    fig, ax = plt.subplots(figsize=(13, 4.0), constrained_layout=True)
    ax.scatter(
        score_values,
        jitter,
        color="#c9272c",
        s=point_size,
        marker="o",
        linewidths=0,
        alpha=0.35,
        rasterized=True,
    )
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.56, 0.56)
    ax.set_xlabel("Thermal Displacement score; 0 = polar-like, 100 = equator-like")
    ax.set_yticks([])
    ax.set_title(f"{level_hpa:g} hPa Thermal Displacement score 1D scatter")
    ax.set_xticks(np.arange(0.0, 101.0, 10.0))
    ax.grid(axis="x", color="#d4d4d4", linewidth=0.7)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def one_point_bucket_counts(score: np.ndarray) -> np.ndarray:
    score_values = np.asarray(score[np.isfinite(score)], dtype=np.float32).ravel()
    bucket_indices = np.ceil(score_values).astype(np.int16) - 1
    bucket_indices = np.clip(bucket_indices, 0, 99)
    return np.bincount(bucket_indices, minlength=100).astype(np.int64)


def plot_score_bucket_bars(
    counts: np.ndarray,
    level_hpa: float,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    bucket_centers = np.arange(0.5, 100.0, 1.0, dtype=np.float32)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_blue_white_red",
        ["#1f5fbf", "#f8f8f8", "#c9272c"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    colors = cmap(norm(bucket_centers))

    fig, ax = plt.subplots(figsize=(14, 5.0), constrained_layout=True)
    ax.bar(
        bucket_centers,
        counts,
        width=0.94,
        align="center",
        color=colors,
        edgecolor="#2f2f2f",
        linewidth=0.2,
    )
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Thermal Displacement score bucket: 0-1, 1-2, ..., 99-100")
    ax.set_ylabel("Cell count")
    ax.set_title(f"{level_hpa:g} hPa Thermal Displacement score 1-point bucket counts")
    ax.set_xticks(np.arange(0.0, 101.0, 5.0))
    ax.grid(axis="y", color="#d4d4d4", linewidth=0.7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def score_bucket_base_colors() -> np.ndarray:
    bucket_centers = np.arange(0.5, 100.0, 1.0, dtype=np.float32)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_blue_white_red",
        ["#1f5fbf", "#f8f8f8", "#c9272c"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    return np.asarray(cmap(norm(bucket_centers)), dtype=np.float32)


def shade_colors(colors: np.ndarray, factor: float) -> np.ndarray:
    rgb = np.asarray(colors[:, :3], dtype=np.float32)
    if factor < 1.0:
        shaded = rgb * factor
    else:
        shaded = rgb + (1.0 - rgb) * (factor - 1.0)
    alpha = np.ones((rgb.shape[0], 1), dtype=np.float32)
    return np.concatenate((np.clip(shaded, 0.0, 1.0), alpha), axis=1)


def plot_stacked_score_bucket_bars(
    counts_by_level: dict[float, np.ndarray],
    output_path: Path,
    *,
    dpi: int,
) -> list[float]:
    bucket_centers = np.arange(0.5, 100.0, 1.0, dtype=np.float32)
    stack_levels = sorted(counts_by_level.keys(), reverse=True)
    base_colors = score_bucket_base_colors()
    shade_factors = np.linspace(0.72, 1.16, num=max(len(stack_levels), 1))
    level_to_color = {
        level_hpa: shade_colors(base_colors, float(shade_factors[index]))
        for index, level_hpa in enumerate(stack_levels)
    }

    fig, ax = plt.subplots(figsize=(16, 7.0), constrained_layout=True)
    bottoms = np.zeros(100, dtype=np.int64)
    width = 0.94
    for level_hpa in stack_levels:
        counts = np.asarray(counts_by_level[level_hpa], dtype=np.int64)
        ax.bar(
            bucket_centers,
            counts,
            width=width,
            bottom=bottoms,
            align="center",
            color=level_to_color[level_hpa],
            edgecolor="#151515",
            linewidth=0.18,
            label=f"{level_hpa:g} hPa",
        )
        next_bottoms = bottoms + counts
        for x_value, boundary in zip(bucket_centers, next_bottoms):
            ax.hlines(
                int(boundary),
                float(x_value - width / 2.0),
                float(x_value + width / 2.0),
                color="#050505",
                linewidth=0.36,
                zorder=5,
            )
        bottoms = next_bottoms

    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Thermal Displacement score bucket: 0-1, 1-2, ..., 99-100")
    ax.set_ylabel("Stacked cell count")
    ax.set_title("Thermal Displacement score 1-point bucket counts stacked by pressure")
    ax.set_xticks(np.arange(0.0, 101.0, 5.0))
    ax.grid(axis="y", color="#d4d4d4", linewidth=0.7)
    ax.legend(title="Bottom to top", frameon=False, loc="upper right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return [float(level) for level in stack_levels]


def write_bucket_counts_csv(counts: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket_label",
                "lower_bound",
                "lower_inclusive",
                "upper_bound",
                "upper_inclusive",
                "cell_count",
            ],
        )
        writer.writeheader()
        for index, count in enumerate(counts):
            writer.writerow(
                {
                    "bucket_label": f"{index}-{index + 1}",
                    "lower_bound": index,
                    "lower_inclusive": "true" if index == 0 else "false",
                    "upper_bound": index + 1,
                    "upper_inclusive": "true",
                    "cell_count": int(count),
                }
            )


def write_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    dpi: int,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for axis, image_path in zip(axes.flat, image_paths):
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_axis_off()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = (
        (REPO_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = choose_timestamp(temperature, args.timestamp)
    pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, pressure_levels)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)

    scatter_image_paths: list[Path] = []
    bucket_bar_image_paths: list[Path] = []
    bucket_counts_by_level: dict[float, np.ndarray] = {}
    outputs: list[dict[str, object]] = []
    for level_hpa in selected_levels:
        print(f"Processing {level_hpa:g} hPa")
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        climatology_level = (
            climatology.sel(pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        result = compute_thermal_displacement_level(
            raw_level,
            climatology_level,
            latitudes,
            score_smooth_sigma_cells=args.score_smooth_sigma_cells,
            same_hemisphere=True,
        )
        score = result.score_points.astype(np.float32)
        scatter_output_path = output_dir / f"thermal-displacement-score-1d-scatter-{slug_for_level(level_hpa)}.png"
        plot_score_scatter(
            score,
            level_hpa,
            scatter_output_path,
            dpi=args.dpi,
            point_size=args.point_size,
        )
        scatter_image_paths.append(scatter_output_path)

        bucket_counts = one_point_bucket_counts(score)
        bucket_counts_by_level[float(level_hpa)] = bucket_counts
        bucket_bar_output_path = output_dir / f"thermal-displacement-score-bucket-bars-{slug_for_level(level_hpa)}.png"
        bucket_counts_csv_path = output_dir / f"thermal-displacement-score-bucket-counts-{slug_for_level(level_hpa)}.csv"
        plot_score_bucket_bars(
            bucket_counts,
            level_hpa,
            bucket_bar_output_path,
            dpi=args.dpi,
        )
        write_bucket_counts_csv(bucket_counts, bucket_counts_csv_path)
        bucket_bar_image_paths.append(bucket_bar_output_path)

        outputs.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "scatter_plot_png": display_path(scatter_output_path),
                "bucket_bar_plot_png": display_path(bucket_bar_output_path),
                "bucket_counts_csv": display_path(bucket_counts_csv_path),
                "score_min": float(np.nanmin(score)),
                "score_max": float(np.nanmax(score)),
                "score_mean": float(np.nanmean(score)),
                "finite_cell_count": int(np.count_nonzero(np.isfinite(score))),
                "bucket_count_total": int(np.sum(bucket_counts)),
            }
        )

    scatter_contact_sheet_path = output_dir / "thermal-displacement-score-scatter-contact-sheet.png"
    write_contact_sheet(
        scatter_image_paths,
        scatter_contact_sheet_path,
        dpi=args.dpi,
        title="Thermal Displacement score 1D scatter plots",
    )
    bucket_bar_contact_sheet_path = output_dir / "thermal-displacement-score-bucket-bars-contact-sheet.png"
    write_contact_sheet(
        bucket_bar_image_paths,
        bucket_bar_contact_sheet_path,
        dpi=args.dpi,
        title="Thermal Displacement score 1-point bucket counts",
    )
    stacked_bucket_bar_path = output_dir / "thermal-displacement-score-bucket-bars-stacked-levels.png"
    stacked_level_order = plot_stacked_score_bucket_bars(
        bucket_counts_by_level,
        stacked_bucket_bar_path,
        dpi=args.dpi,
    )

    summary = {
        "process": "canonical Thermal Displacement score 1D scatter plots and 1-point bucket bar charts",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "score_method": (
            "Canonical same-longitude same-hemisphere Thermal Displacement: match raw ERA5 "
            "temperature to closest climatology latitude at the same pressure and longitude, "
            "convert matched latitude to 0..100 score points, then smooth the score."
        ),
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "scatter_mark": (
            "one point per finite ERA5 latitude-longitude cell; x is score, "
            "y is deterministic jitter only"
        ),
        "bucket_rule": (
            "100 non-overlapping 1-point buckets: first bucket is 0 <= score <= 1; "
            "each later bucket is previous whole number < score <= next whole number, "
            "ending with 99 < score <= 100."
        ),
        "scatter_color": "single red color for every scatter point",
        "bucket_bar_color_scale": "fixed blue-white-red: 0 blue, 50 white, 100 red",
        "stacked_bucket_bar": display_path(stacked_bucket_bar_path),
        "stacked_bucket_bar_order_bottom_to_top": stacked_level_order,
        "stacked_bucket_bar_style": (
            "each score bucket stacks pressure-level counts bottom-to-top from highest hPa "
            "to lowest hPa, with black divider lines at every level boundary and same-score "
            "colors shaded by pressure level"
        ),
        "scatter_contact_sheet": display_path(scatter_contact_sheet_path),
        "bucket_bar_contact_sheet": display_path(bucket_bar_contact_sheet_path),
        "outputs": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
