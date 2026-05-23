from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_DIR = Path("tmp/thermal-displacement-raw-temperature-panels")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_BORDER_GEOJSON = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired Thermal Displacement and raw-temperature maps."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--min-level-hpa", type=float, default=250.0)
    parser.add_argument("--max-level-hpa", type=float, default=1000.0)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--lon-min", type=float, default=-180.0)
    parser.add_argument("--lon-max", type=float, default=180.0)
    parser.add_argument("--lat-min", type=float, default=-90.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


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


def level_folder_name(level_hpa: float) -> str:
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


def two_slope_norm(values: np.ndarray, center: float | None = None) -> mcolors.TwoSlopeNorm:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("Cannot plot an all-NaN field.")
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if center is None:
        center = 0.5 * (vmin + vmax)
    center = float(center)
    if not vmin < center < vmax:
        center = 0.5 * (vmin + vmax)
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def plot_panel(
    *,
    score: np.ndarray,
    raw_temperature_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    raw_temperature_c = raw_temperature_k - 273.15
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)

    score_mesh = axes[0].pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap="bwr",
        norm=mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0),
        shading="auto",
        rasterized=True,
    )
    axes[0].set_title(f"{level_hpa:g} hPa thermal displacement")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    draw_borders(axes[0], border_segments)
    score_colorbar = fig.colorbar(score_mesh, ax=axes[0], pad=0.01, shrink=0.84)
    score_colorbar.set_label("Score: 0 polar-like, 100 equator-like")

    raw_mesh = axes[1].pcolormesh(
        longitudes,
        latitudes,
        raw_temperature_c,
        cmap="bwr",
        norm=two_slope_norm(raw_temperature_c),
        shading="auto",
        rasterized=True,
    )
    axes[1].set_title(f"{level_hpa:g} hPa raw temperature")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    draw_borders(axes[1], border_segments)
    raw_colorbar = fig.colorbar(raw_mesh, ax=axes[1], pad=0.01, shrink=0.84)
    raw_colorbar.set_label("Temperature (deg C)")

    for ax in axes:
        ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
        ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))

    fig.suptitle(
        f"{level_hpa:g} hPa: Thermal Displacement first, raw temperature second",
        y=1.02,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with xr.open_dataset(resolve_path(args.dataset)) as dataset, xr.open_dataset(
        resolve_path(args.climatology)
    ) as climatology_dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        climatology = climatology_dataset[CLIMATOLOGY_VARIABLE]
        selected_time = choose_timestamp(temperature, args.timestamp)

        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        pressure_levels = [
            float(level)
            for level in np.asarray(temperature.coords["pressure_level"].values, dtype=np.float32)
            if args.min_level_hpa <= float(level) <= args.max_level_hpa
        ]
        pressure_levels = sorted(pressure_levels, reverse=True)

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

        for level_hpa in pressure_levels:
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
            score_crop, _, _ = subset_domain(
                result.score_points,
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

            image_path = output_dir / f"thermal-displacement-and-raw-temperature-{level_folder_name(level_hpa)}.png"
            plot_panel(
                score=score_crop,
                raw_temperature_k=raw_crop,
                latitudes=plot_latitudes,
                longitudes=plot_longitudes,
                border_segments=border_segments,
                level_hpa=level_hpa,
                output_path=image_path,
                dpi=args.dpi,
            )
            rows.append(
                {
                    "pressure_hpa": level_hpa,
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "thermal_displacement_white": 50.0,
                    "score_min": float(np.nanmin(score_crop)),
                    "score_max": float(np.nanmax(score_crop)),
                    "raw_temperature_min_c": float(np.nanmin(raw_crop - 273.15)),
                    "raw_temperature_max_c": float(np.nanmax(raw_crop - 273.15)),
                }
            )

    summary = {
        "method": "canonical same-longitude same-hemisphere Thermal Displacement paired with raw temperature",
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
        "color_scales": {
            "thermal_displacement": "blue-white-red fixed at 0/50/100",
            "raw_temperature": "blue-white-red per pressure level, centered between cropped min and max",
        },
        "border_geojson": display_path(args.border_geojson),
        "levels": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_dir={display_path(args.output_dir)}")
    print(f"levels={len(rows)}")


if __name__ == "__main__":
    main()
