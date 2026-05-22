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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts.plot_moist_thermal_displacement import (
    display_path,
    draw_borders,
    load_border_segments,
    resolve_path,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY = Path("data/era5_temperature-climatology_1990-2020_11-08_12.nc")
DEFAULT_OUTPUT_DIR = Path("tmp/moist-thermal-displacement")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_BORDER_GEOJSON_PATH = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot canonical raw-temperature Thermal Displacement maps."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--levels-hpa",
        type=str,
        default=",".join(str(level) for level in DEFAULT_LEVELS_HPA),
    )
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON_PATH)
    return parser.parse_args()


def parse_levels(text: str) -> list[float]:
    levels = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not levels:
        raise ValueError("At least one pressure level is required.")
    return levels


def choose_timestamp(temperature: xr.DataArray, timestamp_text: str) -> np.datetime64:
    valid_times = np.asarray(temperature.coords["valid_time"].values)
    requested = np.datetime64(timestamp_text)
    if requested in valid_times:
        return requested
    nearest_index = int(np.argmin(np.abs(valid_times - requested)))
    return np.datetime64(valid_times[nearest_index])


def plot_score_map(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    selected_center: float,
    title: str,
    output_path: Path,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    finite = score[np.isfinite(score)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not vmin < selected_center < vmax:
        selected_center = 0.5 * (vmin + vmax)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=selected_center, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    image = ax.imshow(
        score,
        extent=(float(longitudes[0]), float(longitudes[-1]), float(latitudes[-1]), float(latitudes[0])),
        origin="upper",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    draw_borders(ax, border_segments)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    colorbar.set_label("Equivalent-latitude score: polar-like 0 -> equator-like 100")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_overview(
    scores_by_level: dict[float, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    levels = [250.0, 500.0, 850.0, 1000.0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160, sharex=True, sharey=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    last_image = None

    for ax, level in zip(axes.ravel(), levels):
        score = scores_by_level[level]
        last_image = ax.imshow(
            score,
            extent=(float(longitudes[0]), float(longitudes[-1]), float(latitudes[-1]), float(latitudes[0])),
            origin="upper",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
            aspect="auto",
        )
        draw_borders(ax, border_segments)
        ax.set_title(f"Raw temperature displacement, {level:g} hPa")

    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude")

    fig.suptitle("Raw Temperature Thermal Displacement Overview, fixed 0/50/100 scale", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.16, hspace=0.28, wspace=0.05)
    if last_image is not None:
        colorbar_axis = fig.add_axes((0.25, 0.06, 0.50, 0.025))
        colorbar = fig.colorbar(last_image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Equivalent-latitude score: polar-like 0 -> equator-like 100")
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    levels_hpa = parse_levels(args.levels_hpa)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)

    rows: list[dict[str, float | int]] = []
    scores_by_level: dict[float, np.ndarray] = {}

    with xr.open_dataset(dataset_path) as dataset, xr.open_dataset(climatology_path) as climatology_dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        climatology = climatology_dataset[CLIMATOLOGY_VARIABLE]
        selected_time = choose_timestamp(temperature, args.timestamp)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(resolve_path(args.border_geojson), longitudes)

        for level in levels_hpa:
            raw_level = np.asarray(
                temperature.sel(valid_time=selected_time, pressure_level=level).values,
                dtype=np.float32,
            )
            climatology_level = np.asarray(
                climatology.sel(pressure_level=level).values,
                dtype=np.float32,
            )
            result = compute_thermal_displacement_level(
                raw_level,
                climatology_level,
                latitudes,
                score_smooth_sigma_cells=args.score_smooth_sigma_cells,
                same_hemisphere=True,
            )
            score = result.score_points
            scores_by_level[level] = score
            finite = score[np.isfinite(score)]
            rows.append(
                {
                    "pressure_hpa": float(level),
                    "selected_white_center": float(result.selected_bucket.center),
                    "selected_bucket_count": int(result.selected_bucket.count),
                    "score_min": float(np.nanmin(finite)),
                    "score_max": float(np.nanmax(finite)),
                    "score_mean": float(np.nanmean(finite)),
                    "raw_temperature_min_k": float(np.nanmin(raw_level)),
                    "raw_temperature_max_k": float(np.nanmax(raw_level)),
                    "raw_temperature_mean_k": float(np.nanmean(raw_level)),
                }
            )
            plot_score_map(
                score=score,
                latitudes=latitudes,
                longitudes=longitudes,
                selected_center=float(result.selected_bucket.center),
                title=f"Raw Temperature Thermal Displacement, {level:g} hPa",
                output_path=output_dir / f"raw_temperature-displacement-{level:g}hpa.png",
                border_segments=border_segments,
            )

    plot_overview(
        scores_by_level=scores_by_level,
        latitudes=latitudes,
        longitudes=longitudes,
        output_path=output_dir / "overview-raw-temperature-displacement.png",
        border_segments=border_segments,
    )

    with (output_dir / "raw_temperature_selected_buckets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "canonical raw-temperature same-longitude same-hemisphere Thermal Displacement",
        "dataset": display_path(args.dataset),
        "climatology": display_path(args.climatology),
        "timestamp": args.timestamp,
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "levels": rows,
    }
    (output_dir / "raw_temperature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_dir={display_path(args.output_dir)}")


if __name__ == "__main__":
    main()
