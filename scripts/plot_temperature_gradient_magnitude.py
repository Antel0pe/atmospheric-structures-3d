from __future__ import annotations

import argparse
import json
import os
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
from scipy.ndimage import gaussian_filter


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("./tmp/temperature-gradient-magnitude")
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
TEMPERATURE_VARIABLE = "t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot smoothed horizontal temperature-gradient magnitude on ERA5 "
            "pressure levels. Gradients are computed with real meter distances, "
            "not raw grid-cell counts."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--levels-hpa",
        type=str,
        default=",".join(str(level) for level in DEFAULT_LEVELS_HPA),
        help="Comma-separated pressure levels in hPa.",
    )
    parser.add_argument(
        "--smooth-sigma-cells",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in native grid cells before derivatives.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.5,
        help=(
            "Upper color limit percentile. Use 100 for strict true-max scaling. "
            "Values above the limit are also yellow."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        return resolved
    repo_relative = (Path.cwd() / path).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(f"File not found: {path}")


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


def smooth_lat_lon(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    sigma = max(float(sigma_cells), 0.0)
    if sigma == 0.0:
        return np.asarray(values, dtype=np.float32)
    return np.asarray(
        gaussian_filter(
            np.asarray(values, dtype=np.float32),
            sigma=(sigma, sigma),
            mode=("nearest", "wrap"),
            truncate=3.0,
        ),
        dtype=np.float32,
    )


def horizontal_temperature_gradient_k_per_100km(
    temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    """Centered finite-difference gradient magnitude using physical distance."""

    values = np.asarray(temperature_k, dtype=np.float64)
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))

    # dT/dy: derivative with respect to latitude radians divided by Earth radius.
    d_t_d_lat_rad = np.gradient(values, lat_rad, axis=0, edge_order=2)
    d_t_dy = d_t_d_lat_rad / EARTH_RADIUS_M

    # dT/dx: centered dateline-wrapped longitude difference, scaled by cos(lat).
    mean_dlon_rad = float(np.mean(np.abs(np.diff(lon_rad))))
    east = np.roll(values, -1, axis=1)
    west = np.roll(values, 1, axis=1)
    d_t_d_lon_rad = (east - west) / (2.0 * mean_dlon_rad)
    cos_lat = np.cos(lat_rad)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_t_dx = d_t_d_lon_rad / (EARTH_RADIUS_M * cos_lat)

    gradient = np.sqrt(np.square(d_t_dx) + np.square(d_t_dy)) * 100_000.0
    gradient[~np.isfinite(gradient)] = np.nan
    return np.asarray(gradient, dtype=np.float32)


def black_to_yellow_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "black_to_yellow",
        ["#000000", "#ffff00"],
    )


def plot_gradient_map(
    gradient_k_per_100km: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
    title: str,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    image = ax.imshow(
        gradient_k_per_100km,
        origin="upper",
        extent=(
            float(np.nanmin(longitudes)),
            float(np.nanmax(longitudes)),
            float(np.nanmin(latitudes)),
            float(np.nanmax(latitudes)),
        ),
        cmap=black_to_yellow_colormap(),
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(longitudes)), float(np.nanmax(longitudes)))
    ax.set_ylim(float(np.nanmin(latitudes)), float(np.nanmax(latitudes)))
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
    colorbar.set_label("Horizontal temperature-gradient magnitude (K / 100 km)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_levels = parse_levels(args.levels_hpa)
    summary: dict[str, object] = {
        "method": {
            "field": "horizontal temperature-gradient magnitude",
            "formula": "sqrt((dT/dx)^2 + (dT/dy)^2)",
            "units": "K / 100 km",
            "distance_basis": "real Earth distances between latitude/longitude grid-cell centers",
            "derivative": "centered finite difference using adjacent native grid cells; longitude wraps at dateline",
            "smoothing": {
                "type": "Gaussian",
                "sigma_native_grid_cells": float(args.smooth_sigma_cells),
            },
            "color_scale": {
                "minimum": 0.0,
                "maximum": (
                    "true finite maximum"
                    if float(args.display_percentile) >= 100.0
                    else f"finite p{float(args.display_percentile):g}; larger values clipped to yellow"
                ),
            },
        },
        "dataset": dataset_path.name,
        "requested_timestamp": args.timestamp,
        "levels": [],
    }

    with xr.open_dataset(dataset_path) as dataset:
        if TEMPERATURE_VARIABLE not in dataset:
            raise KeyError(f"Expected variable {TEMPERATURE_VARIABLE!r}.")

        temperature = dataset[TEMPERATURE_VARIABLE]
        timestamp = choose_timestamp(temperature, args.timestamp)
        selected_time = temperature.sel(valid_time=timestamp)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)

        summary["actual_timestamp"] = np.datetime_as_string(timestamp, unit="m")
        summary["grid"] = {
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_step_degrees": float(abs(latitudes[1] - latitudes[0])),
            "longitude_step_degrees": float(abs(longitudes[1] - longitudes[0])),
            "nominal_latitude_spacing_km": float(
                EARTH_RADIUS_M * np.deg2rad(abs(latitudes[1] - latitudes[0])) / 1000.0
            ),
            "nominal_equator_longitude_spacing_km": float(
                EARTH_RADIUS_M * np.deg2rad(abs(longitudes[1] - longitudes[0])) / 1000.0
            ),
        }

        for requested_level in requested_levels:
            level_slice = selected_time.sel(pressure_level=requested_level, method="nearest")
            actual_level = float(level_slice.coords["pressure_level"].item())
            raw_temperature = np.asarray(level_slice.values, dtype=np.float32)
            smoothed_temperature = smooth_lat_lon(raw_temperature, args.smooth_sigma_cells)
            gradient = horizontal_temperature_gradient_k_per_100km(
                smoothed_temperature,
                latitudes,
                longitudes,
            )

            finite_gradient = gradient[np.isfinite(gradient)]
            if finite_gradient.size == 0:
                raise ValueError(f"No finite gradients for {actual_level:.0f} hPa.")
            if float(args.display_percentile) >= 100.0:
                vmax = float(np.nanmax(finite_gradient))
            else:
                vmax = float(np.nanpercentile(finite_gradient, args.display_percentile))

            image_name = f"temperature-gradient-{int(round(actual_level)):04d}hpa.png"
            image_path = output_dir / image_name
            plot_gradient_map(
                gradient,
                latitudes,
                longitudes,
                image_path,
                (
                    f"{actual_level:.0f} hPa temperature-gradient magnitude, "
                    f"{np.datetime_as_string(timestamp, unit='m')} UTC"
                ),
                vmax=vmax,
            )

            npz_name = f"temperature-gradient-{int(round(actual_level)):04d}hpa.npz"
            np.savez_compressed(
                output_dir / npz_name,
                gradient_k_per_100km=gradient,
                smoothed_temperature_k=smoothed_temperature,
                latitude_deg=latitudes,
                longitude_deg=longitudes,
                pressure_hpa=np.asarray(actual_level, dtype=np.float32),
            )

            summary["levels"].append(
                {
                    "requested_pressure_hpa": float(requested_level),
                    "actual_pressure_hpa": actual_level,
                    "plot": image_name,
                    "data": npz_name,
                    "gradient_min_k_per_100km": float(np.nanmin(finite_gradient)),
                    "gradient_p50_k_per_100km": float(np.nanpercentile(finite_gradient, 50.0)),
                    "gradient_p90_k_per_100km": float(np.nanpercentile(finite_gradient, 90.0)),
                    "gradient_p95_k_per_100km": float(np.nanpercentile(finite_gradient, 95.0)),
                    "gradient_p99_k_per_100km": float(np.nanpercentile(finite_gradient, 99.0)),
                    "gradient_max_k_per_100km": float(np.nanmax(finite_gradient)),
                    "plot_vmax_k_per_100km": vmax,
                }
            )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote temperature-gradient maps to {output_dir}")


if __name__ == "__main__":
    main()
