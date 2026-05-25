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
(CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_ROOT / "fontconfig"))

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
DEFAULT_OUTPUT_DIR = Path("experiments/thermal-displacement-closest-climatology-temperature-delta")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_BORDER_GEOJSON = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the raw-temperature residual against the closest climatology "
            "temperature found by the canonical Thermal Displacement lookup."
        )
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
    parser.add_argument("--lon-min", type=float, default=-180.0)
    parser.add_argument("--lon-max", type=float, default=180.0)
    parser.add_argument("--lat-min", type=float, default=-90.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=170)
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


def resolve_output_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def level_slug(level_hpa: float) -> str:
    return f"{int(round(level_hpa)):04d}hpa"


def subset_domain(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)
    if not np.any(lat_mask):
        raise ValueError("Latitude window did not select any rows.")
    if not np.any(lon_mask):
        raise ValueError("Longitude window did not select any columns.")
    return values[np.ix_(lat_mask, lon_mask)], latitudes[lat_mask], longitudes[lon_mask]


def matched_climatology_temperature(
    climatology_temperature_k: np.ndarray,
    matched_latitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    """Return the climatology temperature at each matched Thermal Displacement row."""

    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    matched_latitudes = np.asarray(matched_latitudes_deg, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)

    ascending_order = np.argsort(latitudes)
    ascending_latitudes = latitudes[ascending_order]
    insertion = np.searchsorted(ascending_latitudes, matched_latitudes)
    lower = np.clip(insertion - 1, 0, ascending_latitudes.size - 1)
    upper = np.clip(insertion, 0, ascending_latitudes.size - 1)
    lower_distance = np.abs(matched_latitudes - ascending_latitudes[lower])
    upper_distance = np.abs(matched_latitudes - ascending_latitudes[upper])
    ascending_rows = np.where(upper_distance < lower_distance, upper, lower)
    source_rows = ascending_order[ascending_rows]

    output = np.full(matched_latitudes.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(matched_latitudes)
    lon_indices = np.broadcast_to(
        np.arange(climatology.shape[1], dtype=np.int64),
        climatology.shape,
    )
    output[finite] = climatology[source_rows[finite], lon_indices[finite]]
    return output


def symmetric_norm(values: np.ndarray, limit: float | None = None) -> mcolors.TwoSlopeNorm:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("Cannot plot an all-NaN field.")
    if limit is None:
        limit = float(np.nanmax(np.abs(finite)))
    limit = max(float(limit), 1.0e-6)
    return mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_delta_map(
    *,
    temperature_delta_k: np.ndarray,
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    color_limit_k: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        temperature_delta_k,
        cmap="RdBu_r",
        norm=symmetric_norm(temperature_delta_k, color_limit_k),
        shading="auto",
        rasterized=True,
    )
    contour = ax.contour(
        longitudes,
        latitudes,
        score,
        levels=np.arange(10.0, 100.0, 10.0),
        colors="black",
        linewidths=0.25,
        alpha=0.45,
    )
    ax.clabel(contour, contour.levels[::2], inline=True, fontsize=6, fmt="%g")
    draw_borders(ax, border_segments)
    ax.set_title(f"{level_hpa:g} hPa raw T minus closest Thermal Displacement climatology T")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.86)
    colorbar.set_label("Raw temperature - matched climatology temperature (K)")
    ax.text(
        0.01,
        0.01,
        "Blue: raw colder than matched climatology. Red: raw hotter. Black contours: Thermal Displacement score.",
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 3},
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overview(
    *,
    deltas_by_level: dict[float, np.ndarray],
    scores_by_level: dict[float, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
    color_limit_k: float,
    dpi: int,
) -> None:
    levels = sorted(deltas_by_level)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    last_mesh = None

    for ax, level_hpa in zip(axes.ravel(), levels):
        delta = deltas_by_level[level_hpa]
        score = scores_by_level[level_hpa]
        last_mesh = ax.pcolormesh(
            longitudes,
            latitudes,
            delta,
            cmap="RdBu_r",
            norm=symmetric_norm(delta, color_limit_k),
            shading="auto",
            rasterized=True,
        )
        ax.contour(
            longitudes,
            latitudes,
            score,
            levels=np.arange(20.0, 100.0, 20.0),
            colors="black",
            linewidths=0.22,
            alpha=0.38,
        )
        draw_borders(ax, border_segments)
        ax.set_title(f"{level_hpa:g} hPa")

    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude")
    for ax in axes.ravel()[len(levels) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Raw temperature departure from the closest climatology temperature found by Thermal Displacement",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.15, hspace=0.22, wspace=0.05)
    if last_mesh is not None:
        colorbar_axis = fig.add_axes((0.25, 0.06, 0.50, 0.025))
        colorbar = fig.colorbar(last_mesh, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Raw temperature - matched climatology temperature (K)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    levels_hpa = parse_levels(args.levels_hpa)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    deltas_by_level: dict[float, np.ndarray] = {}
    scores_by_level: dict[float, np.ndarray] = {}

    with xr.open_dataset(resolve_path(args.dataset)) as dataset, xr.open_dataset(
        resolve_path(args.climatology)
    ) as climatology_dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        climatology = climatology_dataset[CLIMATOLOGY_VARIABLE]
        selected_time = choose_timestamp(temperature, args.timestamp)

        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        _, plot_latitudes, plot_longitudes = subset_domain(
            np.zeros((latitudes.size, longitudes.size), dtype=np.float32),
            latitudes,
            longitudes,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
        )
        border_segments = load_border_segments(resolve_path(args.border_geojson), plot_longitudes)

        for level_hpa in levels_hpa:
            raw_level = np.asarray(
                temperature.sel(valid_time=selected_time, pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            climatology_level = np.asarray(
                climatology.sel(pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            result = compute_thermal_displacement_level(
                raw_level,
                climatology_level,
                latitudes,
                score_smooth_sigma_cells=args.score_smooth_sigma_cells,
                same_hemisphere=True,
            )
            matched_temperature = matched_climatology_temperature(
                climatology_level,
                result.matched_latitudes_deg,
                latitudes,
            )
            temperature_delta = np.asarray(raw_level - matched_temperature, dtype=np.float32)

            score_crop, _, _ = subset_domain(
                result.score_points,
                latitudes,
                longitudes,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )
            delta_crop, _, _ = subset_domain(
                temperature_delta,
                latitudes,
                longitudes,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )
            raw_crop, _, _ = subset_domain(
                raw_level,
                latitudes,
                longitudes,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )
            matched_crop, _, _ = subset_domain(
                matched_temperature,
                latitudes,
                longitudes,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )

            deltas_by_level[level_hpa] = delta_crop
            scores_by_level[level_hpa] = score_crop
            finite_delta = delta_crop[np.isfinite(delta_crop)]
            finite_abs = np.abs(finite_delta)
            color_limit = float(np.nanmax(finite_abs))
            image_path = output_dir / f"raw-minus-matched-climatology-{level_slug(level_hpa)}.png"
            plot_delta_map(
                temperature_delta_k=delta_crop,
                score=score_crop,
                latitudes=plot_latitudes,
                longitudes=plot_longitudes,
                border_segments=border_segments,
                level_hpa=level_hpa,
                output_path=image_path,
                color_limit_k=color_limit,
                dpi=args.dpi,
            )
            rows.append(
                {
                    "pressure_hpa": float(level_hpa),
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "temperature_delta_min_k": float(np.nanmin(finite_delta)),
                    "temperature_delta_max_k": float(np.nanmax(finite_delta)),
                    "temperature_delta_mean_k": float(np.nanmean(finite_delta)),
                    "temperature_delta_abs_mean_k": float(np.nanmean(finite_abs)),
                    "temperature_delta_abs_p95_k": float(np.nanpercentile(finite_abs, 95.0)),
                    "temperature_delta_color_limit_k": color_limit,
                    "raw_temperature_min_k": float(np.nanmin(raw_crop)),
                    "raw_temperature_max_k": float(np.nanmax(raw_crop)),
                    "matched_climatology_temperature_min_k": float(np.nanmin(matched_crop)),
                    "matched_climatology_temperature_max_k": float(np.nanmax(matched_crop)),
                    "thermal_displacement_score_min": float(np.nanmin(score_crop)),
                    "thermal_displacement_score_max": float(np.nanmax(score_crop)),
                    "thermal_displacement_selected_white_center": float(result.selected_bucket.center),
                    "thermal_displacement_selected_bucket_count": int(result.selected_bucket.count),
                }
            )

    global_color_limit = max(float(row["temperature_delta_color_limit_k"]) for row in rows)
    plot_overview(
        deltas_by_level=deltas_by_level,
        scores_by_level=scores_by_level,
        latitudes=plot_latitudes,
        longitudes=plot_longitudes,
        border_segments=border_segments,
        output_path=output_dir / "overview-raw-minus-matched-climatology.png",
        color_limit_k=global_color_limit,
        dpi=args.dpi,
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "canonical same-longitude same-hemisphere Thermal Displacement with raw-minus-matched-climatology temperature residual",
        "dataset": display_path(args.dataset),
        "climatology": display_path(args.climatology),
        "timestamp_requested": args.timestamp,
        "pressure_levels_hpa": [row["pressure_hpa"] for row in rows],
        "domain": {
            "longitude_min": args.lon_min,
            "longitude_max": args.lon_max,
            "latitude_min": args.lat_min,
            "latitude_max": args.lat_max,
        },
        "score_smooth_sigma_cells": args.score_smooth_sigma_cells,
        "temperature_delta_definition": (
            "raw ERA5 temperature minus the climatology temperature at the closest "
            "same-pressure, same-longitude, same-hemisphere matched latitude"
        ),
        "color_scale": "blue-white-red, centered at 0 K; blue is raw colder than matched climatology, red is raw hotter",
        "thermal_displacement_contours": "black contour lines show the smoothed canonical 0-100 Thermal Displacement score",
        "overview_color_limit_k": global_color_limit,
        "levels": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_dir={display_path(args.output_dir)}")
    print(f"levels={len(rows)}")


if __name__ == "__main__":
    main()
