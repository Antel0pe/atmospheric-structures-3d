from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from scripts.thermal_displacement import (
    choose_rarest_middle_60_bucket,
    match_same_longitude_climatology_latitudes,
    score_points_from_matched_latitudes,
    smooth_score_after_matching,
)


DEFAULT_RAW_DATASET = Path("data/global-air-mass-proxy-bundle_2021-11_p1-to-1000.nc")
DEFAULT_TEMPERATURE_CLIMATOLOGY_STACK = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
DEFAULT_SPECIFIC_HUMIDITY_CLIMATOLOGY_STACK = Path(
    "data/specific-humidity-climatology-for-theta-e-and-theta-w-displacement_1990-to-2020_p250-500-850-1000.nc"
)
DEFAULT_MOIST_CLIMATOLOGY = Path(
    "data/era5_moist-potential-temperature-climatology_1990-2020_11-08_12_p250-500-850-1000.nc"
)
DEFAULT_OUTPUT_DIR = Path("tmp/moist-thermal-displacement")
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_BORDER_GEOJSON_PATH = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)

TEMPERATURE_VARIABLE = "t"
SPECIFIC_HUMIDITY_VARIABLE = "q"
THETA_E_MEAN = "theta_e_climatology_mean"
THETA_E_STD = "theta_e_climatology_std"
THETA_W_MEAN = "theta_w_climatology_mean"
THETA_W_STD = "theta_w_climatology_std"
SAMPLE_COUNT = "sample_count"
EPSILON = 0.622
KAPPA_MOIST_BASE = 0.2854


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Thermal Displacement equivalent-latitude lookup to "
            "Bolton-style theta-e and a theta-w estimate derived by inverting "
            "saturated theta-e at 1000 hPa."
        )
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument(
        "--temperature-climatology-stack",
        type=Path,
        default=DEFAULT_TEMPERATURE_CLIMATOLOGY_STACK,
    )
    parser.add_argument(
        "--specific-humidity-climatology-stack",
        type=Path,
        default=DEFAULT_SPECIFIC_HUMIDITY_CLIMATOLOGY_STACK,
    )
    parser.add_argument("--moist-climatology", type=Path, default=DEFAULT_MOIST_CLIMATOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--levels-hpa",
        type=str,
        default=",".join(str(level) for level in DEFAULT_LEVELS_HPA),
    )
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument(
        "--border-geojson",
        type=Path,
        default=DEFAULT_BORDER_GEOJSON_PATH,
    )
    parser.add_argument(
        "--skip-climatology-rebuild",
        action="store_true",
        help="Reuse --moist-climatology if it already exists.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        return resolved
    repo_relative = (Path.cwd() / path).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(f"File not found: {display_path(path)}")


def parse_levels(text: str) -> list[float]:
    levels = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not levels:
        raise ValueError("At least one pressure level is required.")
    return levels


def target_longitude(lon: float, longitudes: np.ndarray) -> float:
    lon_min = float(np.nanmin(longitudes))
    lon_max = float(np.nanmax(longitudes))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def split_target_longitude_segments(
    points: list[tuple[float, float]],
    longitudes: np.ndarray,
) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for lon, lat in points:
        next_lon = target_longitude(float(lon), longitudes)
        if previous_lon is not None and abs(next_lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((next_lon, float(lat)))
        previous_lon = next_lon

    if len(segment) >= 2:
        yield segment


def load_border_segments(
    geojson_path: Path,
    longitudes: np.ndarray,
) -> list[list[tuple[float, float]]]:
    if not geojson_path.exists():
        return []

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    segments: list[list[tuple[float, float]]] = []

    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue

        for polygon in polygons:
            for ring in polygon:
                points = [(float(point[0]), float(point[1])) for point in ring]
                segments.extend(split_target_longitude_segments(points, longitudes))

    return segments


def draw_borders(
    ax: plt.Axes,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    for segment in border_segments:
        xs = [point[0] for point in segment]
        ys = [point[1] for point in segment]
        ax.plot(xs, ys, color="black", linewidth=0.25, alpha=0.45, zorder=3)


def saturation_vapor_pressure_hpa(temperature_k: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(temperature_k, dtype=np.float32) - 273.15
    return np.asarray(
        6.112 * np.exp((17.67 * temperature_c) / np.maximum(temperature_c + 243.5, 1.0e-6)),
        dtype=np.float32,
    )


def saturation_specific_humidity_kgkg(
    temperature_k: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    pressure = max(float(pressure_hpa), 1.0)
    vapor_pressure = np.minimum(saturation_vapor_pressure_hpa(temperature_k), 0.99 * pressure)
    mixing_ratio = EPSILON * vapor_pressure / np.maximum(pressure - vapor_pressure, 1.0e-6)
    return np.asarray(mixing_ratio / np.maximum(1.0 + mixing_ratio, 1.0e-7), dtype=np.float32)


def equivalent_potential_temperature_k(
    temperature_k: np.ndarray,
    specific_humidity_kgkg: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=np.float32)
    specific_humidity = np.clip(
        np.asarray(specific_humidity_kgkg, dtype=np.float32),
        0.0,
        0.2,
    )
    pressure = max(float(pressure_hpa), 1.0)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mixing_ratio = specific_humidity / np.maximum(1.0 - specific_humidity, 1.0e-7)
        vapor_pressure_hpa = (
            specific_humidity
            * pressure
            / np.maximum(EPSILON + (1.0 - EPSILON) * specific_humidity, 1.0e-7)
        )
        vapor_pressure_hpa = np.maximum(vapor_pressure_hpa, 1.0e-6)
        log_ratio = np.log(vapor_pressure_hpa / 6.112)
        dewpoint_c = (243.5 * log_ratio) / np.maximum(17.67 - log_ratio, 1.0e-7)
        dewpoint_k = dewpoint_c + 273.15
        lifting_condensation_temperature_k = (
            1.0
            / (
                1.0 / np.maximum(dewpoint_k - 56.0, 1.0e-7)
                + np.log(np.maximum(temperature, 1.0) / np.maximum(dewpoint_k, 1.0))
                / 800.0
            )
            + 56.0
        )
        theta = temperature * (1000.0 / pressure) ** (
            KAPPA_MOIST_BASE * (1.0 - 0.28 * mixing_ratio)
        )
        theta_e = theta * np.exp(
            (3.376 / np.maximum(lifting_condensation_temperature_k, 1.0) - 0.00254)
            * mixing_ratio
            * 1000.0
            * (1.0 + 0.81 * mixing_ratio)
        )

    return np.asarray(theta_e, dtype=np.float32)


def wet_bulb_potential_temperature_k(theta_e_k: np.ndarray) -> np.ndarray:
    """Estimate theta-w by inverting saturated theta-e at 1000 hPa.

    This is a diagnostic approximation, but it is internally consistent with the
    Bolton-style theta-e calculation above: find the 1000 hPa saturated parcel
    temperature whose theta-e matches the input theta-e.
    """

    target = np.asarray(theta_e_k, dtype=np.float32)
    low = np.full_like(target, 180.0, dtype=np.float32)
    high = np.full_like(target, 420.0, dtype=np.float32)
    finite = np.isfinite(target)

    for _ in range(32):
        mid = (low + high) * 0.5
        q_sat = saturation_specific_humidity_kgkg(mid, 1000.0)
        mid_theta_e = equivalent_potential_temperature_k(mid, q_sat, 1000.0)
        go_high = mid_theta_e < target
        low = np.where(go_high & finite, mid, low)
        high = np.where((~go_high) & finite, mid, high)

    result = (low + high) * 0.5
    return np.asarray(np.where(finite, result, np.nan), dtype=np.float32)


def choose_existing_time(data_array: xr.DataArray, timestamp_text: str) -> np.datetime64:
    times = np.asarray(data_array.coords["valid_time"].values)
    requested = np.datetime64(timestamp_text)
    if requested in times:
        return requested
    nearest_index = int(np.argmin(np.abs(times - requested)))
    return np.datetime64(times[nearest_index])


def assert_same_grid(first: xr.DataArray, second: xr.DataArray, label: str) -> None:
    for coord_name in ("latitude", "longitude"):
        if not np.array_equal(first.coords[coord_name].values, second.coords[coord_name].values):
            raise ValueError(f"{label} {coord_name} grid does not match.")


def build_moist_climatology(
    temperature_stack_path: Path,
    humidity_stack_path: Path,
    output_path: Path,
    levels_hpa: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(temperature_stack_path) as t_dataset, xr.open_dataset(humidity_stack_path) as q_dataset:
        temperature = t_dataset[TEMPERATURE_VARIABLE]
        humidity = q_dataset[SPECIFIC_HUMIDITY_VARIABLE]
        assert_same_grid(temperature, humidity, "Temperature and specific-humidity climatology stacks")

        time_values = np.asarray(humidity.coords["valid_time"].values)
        theta_e_means: list[np.ndarray] = []
        theta_e_stds: list[np.ndarray] = []
        theta_w_means: list[np.ndarray] = []
        theta_w_stds: list[np.ndarray] = []
        sample_counts: list[np.ndarray] = []

        for level in levels_hpa:
            t_level = temperature.sel(pressure_level=level)
            q_level = humidity.sel(pressure_level=level)

            mean_e: np.ndarray | None = None
            mean_w: np.ndarray | None = None
            m2_e: np.ndarray | None = None
            m2_w: np.ndarray | None = None
            count: np.ndarray | None = None

            for time_index in range(q_level.sizes["valid_time"]):
                t_slice = np.asarray(t_level.isel(valid_time=time_index).values, dtype=np.float32)
                q_slice = np.asarray(q_level.isel(valid_time=time_index).values, dtype=np.float32)
                theta_e = equivalent_potential_temperature_k(t_slice, q_slice, level)
                theta_w = wet_bulb_potential_temperature_k(theta_e)
                valid = np.isfinite(theta_e) & np.isfinite(theta_w)

                if mean_e is None:
                    mean_e = np.zeros_like(theta_e, dtype=np.float64)
                    mean_w = np.zeros_like(theta_w, dtype=np.float64)
                    m2_e = np.zeros_like(theta_e, dtype=np.float64)
                    m2_w = np.zeros_like(theta_w, dtype=np.float64)
                    count = np.zeros(theta_e.shape, dtype=np.int16)

                assert mean_w is not None and m2_e is not None and m2_w is not None and count is not None
                count[valid] += 1

                delta_e = np.where(valid, theta_e - mean_e, 0.0)
                mean_e += np.where(valid, delta_e / np.maximum(count, 1), 0.0)
                m2_e += np.where(valid, delta_e * (theta_e - mean_e), 0.0)

                delta_w = np.where(valid, theta_w - mean_w, 0.0)
                mean_w += np.where(valid, delta_w / np.maximum(count, 1), 0.0)
                m2_w += np.where(valid, delta_w * (theta_w - mean_w), 0.0)

            if mean_e is None or mean_w is None or m2_e is None or m2_w is None or count is None:
                raise ValueError(f"No climatology samples found for {level:g} hPa.")

            divisor = np.maximum(count.astype(np.float32) - 1.0, 1.0)
            theta_e_means.append(mean_e.astype(np.float32))
            theta_w_means.append(mean_w.astype(np.float32))
            theta_e_stds.append(np.sqrt(m2_e / divisor).astype(np.float32))
            theta_w_stds.append(np.sqrt(m2_w / divisor).astype(np.float32))
            sample_counts.append(count.astype(np.int16))

        coords = {
            "pressure_level": np.asarray(levels_hpa, dtype=np.float32),
            "latitude": humidity.coords["latitude"].values,
            "longitude": humidity.coords["longitude"].values,
        }
        climatology = xr.Dataset(
            data_vars={
                THETA_E_MEAN: (("pressure_level", "latitude", "longitude"), np.stack(theta_e_means)),
                THETA_E_STD: (("pressure_level", "latitude", "longitude"), np.stack(theta_e_stds)),
                THETA_W_MEAN: (("pressure_level", "latitude", "longitude"), np.stack(theta_w_means)),
                THETA_W_STD: (("pressure_level", "latitude", "longitude"), np.stack(theta_w_stds)),
                SAMPLE_COUNT: (("pressure_level", "latitude", "longitude"), np.stack(sample_counts)),
            },
            coords=coords,
            attrs={
                "title": "ERA5 moist potential temperature climatology",
                "summary": (
                    "Theta-e and theta-w climatology built from matched ERA5 pressure-level "
                    "temperature and specific humidity samples for Nov 8 12Z, 1990-2020."
                ),
                "source_dataset": "reanalysis-era5-pressure-levels",
                "source_variables": "temperature, specific_humidity",
                "valid_time_samples": ", ".join(str(value) for value in time_values),
                "climatology_method": "compute per-sample theta-e/theta-w, then mean and sample std over valid_time",
                "theta_e_method": "Bolton-style equivalent potential temperature from T, q, and pressure",
                "theta_w_method": "inverted saturated theta-e at 1000 hPa using the same theta-e formula",
            },
        )

        for variable_name in (THETA_E_MEAN, THETA_E_STD, THETA_W_MEAN, THETA_W_STD):
            climatology[variable_name].attrs = {"units": "K"}
        climatology[SAMPLE_COUNT].attrs = {"units": "1"}

        encoding = {
            THETA_E_MEAN: {"zlib": True, "complevel": 4, "dtype": "float32"},
            THETA_E_STD: {"zlib": True, "complevel": 4, "dtype": "float32"},
            THETA_W_MEAN: {"zlib": True, "complevel": 4, "dtype": "float32"},
            THETA_W_STD: {"zlib": True, "complevel": 4, "dtype": "float32"},
            SAMPLE_COUNT: {"zlib": True, "complevel": 4, "dtype": "int16"},
        }
        climatology.to_netcdf(output_path, encoding=encoding)


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
    ax.set_xlim(float(longitudes[0]), float(longitudes[-1]))
    ax.set_ylim(float(latitudes[-1]), float(latitudes[0]))
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    colorbar.set_label("Equivalent-latitude score: polar-like 0 -> equator-like 100")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_overview(
    scores_by_field: dict[str, dict[float, np.ndarray]],
    bucket_by_field: dict[str, dict[float, float]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    fields = [("theta_e", "Theta-e"), ("theta_w", "Theta-w")]
    levels = [250.0, 500.0, 850.0, 1000.0]
    fig, axes = plt.subplots(len(levels), len(fields), figsize=(13, 14), dpi=160, sharex=True, sharey=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    last_image = None

    for row_index, level in enumerate(levels):
        for column_index, (field_key, field_label) in enumerate(fields):
            ax = axes[row_index, column_index]
            score = scores_by_field[field_key][level]
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
            ax.set_title(f"{field_label} displacement, {level:g} hPa")
            if column_index == 0:
                ax.set_ylabel("Latitude")
            if row_index == len(levels) - 1:
                ax.set_xlabel("Longitude")

    fig.suptitle("Moist Thermal Displacement Overview, fixed 0/50/100 scale", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.12, hspace=0.28, wspace=0.05)
    if last_image is not None:
        colorbar_axis = fig.add_axes((0.25, 0.045, 0.50, 0.018))
        colorbar = fig.colorbar(last_image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Equivalent-latitude score: polar-like 0 -> equator-like 100")
    fig.savefig(output_path)
    plt.close(fig)


def compute_raw_fields(
    raw_dataset_path: Path,
    timestamp: str,
    levels_hpa: list[float],
) -> tuple[dict[str, dict[float, np.ndarray]], np.ndarray, np.ndarray]:
    with xr.open_dataset(raw_dataset_path) as dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        humidity = dataset[SPECIFIC_HUMIDITY_VARIABLE]
        assert_same_grid(temperature, humidity, "Raw temperature and specific humidity")
        valid_time = choose_existing_time(temperature, timestamp)
        selected_temperature = temperature.sel(valid_time=valid_time)
        selected_humidity = humidity.sel(valid_time=valid_time)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)

        raw_fields = {"theta_e": {}, "theta_w": {}}
        for level in levels_hpa:
            t_slice = np.asarray(selected_temperature.sel(pressure_level=level).values, dtype=np.float32)
            q_slice = np.asarray(selected_humidity.sel(pressure_level=level).values, dtype=np.float32)
            theta_e = equivalent_potential_temperature_k(t_slice, q_slice, level)
            theta_w = wet_bulb_potential_temperature_k(theta_e)
            raw_fields["theta_e"][level] = theta_e
            raw_fields["theta_w"][level] = theta_w

    return raw_fields, latitudes, longitudes


def compute_displacement_maps(
    raw_fields: dict[str, dict[float, np.ndarray]],
    moist_climatology_path: Path,
    latitudes: np.ndarray,
    levels_hpa: list[float],
    sigma_cells: float,
) -> tuple[dict[str, dict[float, np.ndarray]], dict[str, dict[float, dict[str, float]]]]:
    with xr.open_dataset(moist_climatology_path) as climatology:
        scores_by_field: dict[str, dict[float, np.ndarray]] = {"theta_e": {}, "theta_w": {}}
        summary_by_field: dict[str, dict[float, dict[str, float]]] = {"theta_e": {}, "theta_w": {}}
        variable_lookup = {"theta_e": THETA_E_MEAN, "theta_w": THETA_W_MEAN}

        for field_key, variable_name in variable_lookup.items():
            for level in levels_hpa:
                climatology_slice = np.asarray(
                    climatology[variable_name].sel(pressure_level=level).values,
                    dtype=np.float32,
                )
                raw_slice = raw_fields[field_key][level]
                matched = match_same_longitude_climatology_latitudes(
                    raw_slice,
                    climatology_slice,
                    latitudes,
                    same_hemisphere=True,
                )
                score_unsmoothed = score_points_from_matched_latitudes(matched, latitudes)
                score = smooth_score_after_matching(score_unsmoothed, sigma_cells=sigma_cells)
                _, _, selected = choose_rarest_middle_60_bucket(score)
                finite = score[np.isfinite(score)]
                scores_by_field[field_key][level] = score
                summary_by_field[field_key][level] = {
                    "score_min": float(np.nanmin(finite)),
                    "score_max": float(np.nanmax(finite)),
                    "score_mean": float(np.nanmean(finite)),
                    "selected_white_center": float(selected.center),
                    "selected_bucket_count": int(selected.count),
                    "middle_60_lower": float(selected.middle_60_lower),
                    "middle_60_upper": float(selected.middle_60_upper),
                    "raw_field_min_k": float(np.nanmin(raw_slice)),
                    "raw_field_max_k": float(np.nanmax(raw_slice)),
                    "raw_field_mean_k": float(np.nanmean(raw_slice)),
                }

    return scores_by_field, summary_by_field


def write_summary(
    output_dir: Path,
    summary_by_field: dict[str, dict[float, dict[str, float]]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "method": (
            "For each raw theta-e/theta-w cell, find the same-pressure, same-longitude, "
            "same-hemisphere climatology latitude whose theta-e/theta-w is closest; "
            "convert matched absolute latitude to 0..100 polar-like/equator-like score; "
            "smooth score after matching."
        ),
        "theta_e_method": "Bolton-style equivalent potential temperature from T, q, and pressure.",
        "theta_w_method": "Diagnostic inversion of saturated theta-e at 1000 hPa.",
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "raw_dataset": display_path(args.raw_dataset),
        "temperature_climatology_stack": display_path(args.temperature_climatology_stack),
        "specific_humidity_climatology_stack": display_path(args.specific_humidity_climatology_stack),
        "moist_climatology": display_path(args.moist_climatology),
        "fields": summary_by_field,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "selected_buckets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field",
                "pressure_hpa",
                "selected_white_center",
                "selected_bucket_count",
                "score_min",
                "score_max",
                "score_mean",
                "raw_field_min_k",
                "raw_field_max_k",
                "raw_field_mean_k",
            ],
        )
        writer.writeheader()
        for field_key, by_level in summary_by_field.items():
            for level, values in by_level.items():
                writer.writerow(
                    {
                        "field": field_key,
                        "pressure_hpa": level,
                        **{
                            key: values[key]
                            for key in (
                                "selected_white_center",
                                "selected_bucket_count",
                                "score_min",
                                "score_max",
                                "score_mean",
                                "raw_field_min_k",
                                "raw_field_max_k",
                                "raw_field_mean_k",
                            )
                        },
                    }
                )


def main() -> None:
    args = parse_args()
    levels_hpa = parse_levels(args.levels_hpa)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset_path = resolve_path(args.raw_dataset)
    temperature_stack_path = resolve_path(args.temperature_climatology_stack)
    humidity_stack_path = resolve_path(args.specific_humidity_climatology_stack)
    moist_climatology_path = args.moist_climatology.expanduser().resolve()

    if not args.skip_climatology_rebuild or not moist_climatology_path.exists():
        build_moist_climatology(
            temperature_stack_path=temperature_stack_path,
            humidity_stack_path=humidity_stack_path,
            output_path=moist_climatology_path,
            levels_hpa=levels_hpa,
        )

    raw_fields, latitudes, longitudes = compute_raw_fields(
        raw_dataset_path=raw_dataset_path,
        timestamp=args.timestamp,
        levels_hpa=levels_hpa,
    )
    scores_by_field, summary_by_field = compute_displacement_maps(
        raw_fields=raw_fields,
        moist_climatology_path=moist_climatology_path,
        latitudes=latitudes,
        levels_hpa=levels_hpa,
        sigma_cells=args.score_smooth_sigma_cells,
    )

    border_segments = load_border_segments(resolve_path(args.border_geojson), longitudes)
    field_labels = {"theta_e": "Theta-e", "theta_w": "Theta-w"}

    for field_key, by_level in scores_by_field.items():
        for level, score in by_level.items():
            selected_center = summary_by_field[field_key][level]["selected_white_center"]
            plot_score_map(
                score=score,
                latitudes=latitudes,
                longitudes=longitudes,
                selected_center=selected_center,
                title=(
                    f"{field_labels[field_key]} Thermal-Displacement-Like Score, "
                    f"{level:g} hPa"
                ),
                output_path=output_dir / f"{field_key}-displacement-{level:g}hpa.png",
                border_segments=border_segments,
            )

    bucket_by_field = {
        field_key: {
            level: values["selected_white_center"]
            for level, values in by_level.items()
        }
        for field_key, by_level in summary_by_field.items()
    }
    plot_overview(
        scores_by_field=scores_by_field,
        bucket_by_field=bucket_by_field,
        latitudes=latitudes,
        longitudes=longitudes,
        output_path=output_dir / "overview-theta-e-theta-w-displacement.png",
        border_segments=border_segments,
    )
    write_summary(output_dir, summary_by_field, args)

    print(f"output_dir={display_path(output_dir)}")
    print(f"moist_climatology={display_path(moist_climatology_path)}")


if __name__ == "__main__":
    main()
