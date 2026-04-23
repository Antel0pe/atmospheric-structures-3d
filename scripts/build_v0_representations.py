from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage
import xarray as xr

from scripts.build_air_mass_classification_structures import (
    CLASS_ORDER,
    CLASS_PROXY_LABELS,
    VARIANT_RECIPES,
    apply_bridge_gap,
    build_score_mask,
    build_variant_fields,
    classify_quadrants,
    filter_components,
    filter_surface_attached_components,
    label_wrapped_volume_components,
)
from scripts.build_potential_temperature_structures import compute_dry_potential_temperature


DEFAULT_AIR_MASS_DATASET = Path("data/global-air-mass-proxy-bundle_2021-11_p1-to-1000.nc")
DEFAULT_THETA_CLIMATOLOGY = Path("data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc")
DEFAULT_WIND_DATASET = Path("data/global-850-hpa-wind-uv-for-established-repo-timestamp_2021-11_p850.nc")
DEFAULT_OUTPUT_DIR = Path("tmp/v0-representations/2021-11-08T12-00")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4
DEFAULT_SOURCE_TRACE_HOURS = 24.0
DEFAULT_SOURCE_MIN_SPEED_MS = 5.0
DEFAULT_DISTINCTNESS_SIGMA = 1.5
DEFAULT_DISTINCTNESS_MIN_VOXELS = 20
DEFAULT_DISTINCTNESS_MIN_LEVELS = 2
DEFAULT_TRANSPORT_TOP_PERCENT = 8.0
DEFAULT_FRONT_TOP_PERCENT = 6.0
DEFAULT_FRONT_MIN_CELLS = 24

SURFACE_CLASS_COLORS = {
    "warm_dry": "#d58635",
    "warm_moist": "#df6d56",
    "cold_dry": "#4e88d8",
    "cold_moist": "#58c2b2",
}


@dataclass(frozen=True)
class CoarseFields:
    timestamp: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    specific_humidity: np.ndarray
    theta_k: np.ndarray
    theta_climatology_mean_k: np.ndarray
    theta_climatology_std_k: np.ndarray
    temperature_850_k: np.ndarray
    theta_850_k: np.ndarray
    theta_anomaly_850_k: np.ndarray
    u850_ms: np.ndarray
    v850_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build quick v0 plots and analytics for five atmospheric representation ideas."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_AIR_MASS_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_THETA_CLIMATOLOGY)
    parser.add_argument("--wind-dataset", type=Path, default=DEFAULT_WIND_DATASET)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pressure-min-hpa", type=float, default=DEFAULT_PRESSURE_MIN_HPA)
    parser.add_argument("--pressure-max-hpa", type=float, default=DEFAULT_PRESSURE_MAX_HPA)
    parser.add_argument("--latitude-stride", type=int, default=DEFAULT_LATITUDE_STRIDE)
    parser.add_argument("--longitude-stride", type=int, default=DEFAULT_LONGITUDE_STRIDE)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        tmp_root = Path("/tmp")
        try:
            return f"tmp/{resolved.relative_to(tmp_root).as_posix()}"
        except ValueError:
            return resolved.name or "<external-path>"


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pressure_to_standard_height_km(pressure_hpa: float) -> float:
    safe_pressure = max(float(pressure_hpa), 1.0)
    return 44.33 * (1.0 - (safe_pressure / 1013.25) ** 0.1903)


def wrap_longitude_deg(longitude_deg: np.ndarray) -> np.ndarray:
    return ((np.asarray(longitude_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0


def circular_mean_longitude_deg(longitudes_deg: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)))


def great_circle_distance_km(
    lat_a_deg: float,
    lon_a_deg: float,
    lat_b_deg: float,
    lon_b_deg: float,
) -> float:
    lat_a = math.radians(lat_a_deg)
    lat_b = math.radians(lat_b_deg)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(lon_b_deg - lon_a_deg)
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(max(hav, 0.0))))


def detect_time_index(dataset: xr.Dataset, timestamp: str) -> int:
    values = [
        np.datetime_as_string(value, unit="m").rstrip("Z")
        for value in np.asarray(dataset.coords["valid_time"].values, dtype="datetime64[m]")
    ]
    if timestamp not in values:
        raise ValueError(f"Timestamp {timestamp!r} not available in {dataset.encoding.get('source')}.")
    return values.index(timestamp)


def select_pressure_indices(
    pressure_levels_hpa: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> np.ndarray:
    lower = min(float(pressure_min_hpa), float(pressure_max_hpa))
    upper = max(float(pressure_min_hpa), float(pressure_max_hpa))
    keep = np.flatnonzero(
        (np.asarray(pressure_levels_hpa, dtype=np.float32) >= lower)
        & (np.asarray(pressure_levels_hpa, dtype=np.float32) <= upper)
    )
    if keep.size == 0:
        raise ValueError(f"No pressure levels available within [{lower}, {upper}] hPa.")
    return keep.astype(np.int64)


def align_wind_to_grid(
    wind_dataset: xr.Dataset,
    timestamp: str,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_index = detect_time_index(wind_dataset, timestamp)
    u = (
        wind_dataset["u"]
        .isel(valid_time=time_index, pressure_level=0)
        .sel(latitude=xr.DataArray(latitudes_deg, dims="latitude"))
        .sel(longitude=xr.DataArray(longitudes_deg, dims="longitude"))
    )
    v = (
        wind_dataset["v"]
        .isel(valid_time=time_index, pressure_level=0)
        .sel(latitude=xr.DataArray(latitudes_deg, dims="latitude"))
        .sel(longitude=xr.DataArray(longitudes_deg, dims="longitude"))
    )
    return np.asarray(u.values, dtype=np.float32), np.asarray(v.values, dtype=np.float32)


def load_coarse_fields(args: argparse.Namespace) -> CoarseFields:
    dataset_path = args.dataset.expanduser().resolve()
    climatology_path = args.climatology.expanduser().resolve()
    wind_path = args.wind_dataset.expanduser().resolve()

    with xr.open_dataset(dataset_path) as dataset, xr.open_dataset(
        climatology_path
    ) as climatology, xr.open_dataset(wind_path) as wind_dataset:
        time_index = detect_time_index(dataset, args.timestamp)
        pressure_levels = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        pressure_indices = select_pressure_indices(
            pressure_levels,
            args.pressure_min_hpa,
            args.pressure_max_hpa,
        )
        latitudes = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
        latitudes = latitudes[:: max(int(args.latitude_stride), 1)]
        longitudes = longitudes[:: max(int(args.longitude_stride), 1)]
        pressure_window = pressure_levels[pressure_indices]

        def load_field(name: str) -> np.ndarray:
            values = np.asarray(
                dataset[name]
                .isel(valid_time=time_index, pressure_level=pressure_indices)
                .values[:, :: max(int(args.latitude_stride), 1), :: max(int(args.longitude_stride), 1)],
                dtype=np.float32,
            )
            return values

        temperature = load_field("t")
        relative_humidity = load_field("r")
        specific_humidity = load_field("q")

        climatology_mean = np.asarray(
            climatology["theta_climatology_mean"]
            .isel(pressure_level=pressure_indices)
            .values[:, :: max(int(args.latitude_stride), 1), :: max(int(args.longitude_stride), 1)],
            dtype=np.float32,
        )
        climatology_std = np.asarray(
            climatology["theta_climatology_std"]
            .isel(pressure_level=pressure_indices)
            .values[:, :: max(int(args.latitude_stride), 1), :: max(int(args.longitude_stride), 1)],
            dtype=np.float32,
        )

        theta = compute_dry_potential_temperature(temperature, pressure_window)
        theta_850_index = int(np.argmin(np.abs(pressure_window - 850.0)))
        temperature_850 = np.asarray(temperature[theta_850_index], dtype=np.float32)
        theta_850 = np.asarray(theta[theta_850_index], dtype=np.float32)
        theta_anomaly_850 = np.asarray(theta_850 - climatology_mean[theta_850_index], dtype=np.float32)
        u850, v850 = align_wind_to_grid(
            wind_dataset,
            args.timestamp,
            latitudes,
            longitudes,
        )

    return CoarseFields(
        timestamp=args.timestamp,
        pressure_levels_hpa=pressure_window,
        latitudes_deg=latitudes,
        longitudes_deg=longitudes,
        temperature_k=temperature,
        relative_humidity_pct=relative_humidity,
        specific_humidity=specific_humidity,
        theta_k=theta,
        theta_climatology_mean_k=climatology_mean,
        theta_climatology_std_k=climatology_std,
        temperature_850_k=temperature_850,
        theta_850_k=theta_850,
        theta_anomaly_850_k=theta_anomaly_850,
        u850_ms=u850,
        v850_ms=v850,
    )


def build_component_shape_label(
    pressure_indices: np.ndarray,
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> str:
    if pressure_indices.size < 3:
        return "compact"
    lat_values = latitudes_deg[latitude_indices]
    lon_values = longitudes_deg[longitude_indices]
    lon_cos = np.cos(np.deg2rad(lat_values))
    points = np.column_stack(
        [
            longitude_indices.astype(np.float64) * np.maximum(lon_cos, 0.25),
            latitude_indices.astype(np.float64),
            pressure_indices.astype(np.float64),
        ]
    )
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    if eigenvalues[0] / eigenvalues[1] >= 5.0 and eigenvalues[1] / eigenvalues[2] >= 3.0:
        return "filamentary"
    if eigenvalues[1] / eigenvalues[2] >= 4.0:
        return "sheet-like"
    return "compact"


def summarize_3d_components(
    mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    *,
    class_key: str | None = None,
) -> list[dict[str, Any]]:
    labels, component_count = label_wrapped_volume_components(mask)
    summaries: list[dict[str, Any]] = []
    surface_level_index = int(np.argmax(pressure_levels_hpa))
    total_horizontal_cells = latitudes_deg.size * longitudes_deg.size

    for label_id in range(1, component_count + 1):
        component_mask = labels == label_id
        occupied = np.argwhere(component_mask)
        if occupied.size == 0:
            continue
        pressure_indices = occupied[:, 0]
        latitude_indices = occupied[:, 1]
        longitude_indices = occupied[:, 2]
        footprint_mask = np.any(component_mask, axis=0)
        footprint_cells = int(np.count_nonzero(footprint_mask))
        occupied_levels = np.flatnonzero(np.any(component_mask, axis=(1, 2)))
        component_pressures = pressure_levels_hpa[occupied_levels]
        top_pressure = float(np.max(component_pressures))
        bottom_pressure = float(np.min(component_pressures))
        depth_km = pressure_to_standard_height_km(bottom_pressure) - pressure_to_standard_height_km(
            top_pressure
        )

        level_centroids: list[tuple[float, float]] = []
        for level_index in occupied_levels:
            level_mask = component_mask[level_index]
            level_lat_indices, level_lon_indices = np.nonzero(level_mask)
            if level_lat_indices.size == 0:
                continue
            level_centroids.append(
                (
                    float(np.mean(latitudes_deg[level_lat_indices])),
                    circular_mean_longitude_deg(longitudes_deg[level_lon_indices]),
                )
            )

        surface_like_level = int(np.argmax(component_pressures))
        ref_lat, ref_lon = level_centroids[surface_like_level]
        max_tilt_km = float(
            max(
                great_circle_distance_km(ref_lat, ref_lon, lat, lon)
                for lat, lon in level_centroids
            )
        )

        summary = {
            "component_id": int(label_id),
            "class_key": class_key,
            "class_label": CLASS_PROXY_LABELS.get(class_key, class_key) if class_key else None,
            "voxel_count": int(occupied.shape[0]),
            "footprint_cells": footprint_cells,
            "footprint_fraction": float(footprint_cells / max(total_horizontal_cells, 1)),
            "occupied_level_count": int(occupied_levels.size),
            "pressure_top_hpa": top_pressure,
            "pressure_bottom_hpa": bottom_pressure,
            "depth_km": float(depth_km),
            "surface_attached": bool(np.any(component_mask[surface_level_index])),
            "max_centroid_displacement_km": max_tilt_km,
            "shape_label": build_component_shape_label(
                pressure_indices,
                latitude_indices,
                longitude_indices,
                latitudes_deg,
                longitudes_deg,
            ),
        }
        summaries.append(summary)

    summaries.sort(key=lambda item: item["voxel_count"], reverse=True)
    return summaries


def label_wrapped_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(
        extended,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    root_map = np.arange(component_count + 1, dtype=np.int32)

    def find_root(component_id: int) -> int:
        root = component_id
        while root_map[root] != root:
            root = int(root_map[root])
        while root_map[component_id] != component_id:
            next_component = int(root_map[component_id])
            root_map[component_id] = root
            component_id = next_component
        return root

    seam_pairs = np.column_stack([labels[:, 0].reshape(-1), labels[:, -1].reshape(-1)])
    for first, second in seam_pairs:
        first_id = int(first)
        second_id = int(second)
        if first_id <= 0 or second_id <= 0:
            continue
        first_root = find_root(first_id)
        second_root = find_root(second_id)
        if first_root == second_root:
            continue
        root_map[max(first_root, second_root)] = min(first_root, second_root)

    for component_id in range(1, component_count + 1):
        root_map[component_id] = find_root(component_id)

    unique_roots = np.unique(root_map[1:])
    compact = np.zeros(component_count + 1, dtype=np.int32)
    compact[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    return compact[root_map[labels[:, :-1]]], int(unique_roots.size)


def summarize_2d_components(mask: np.ndarray, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray) -> list[dict[str, Any]]:
    labels, component_count = label_wrapped_2d(mask)
    summaries: list[dict[str, Any]] = []
    total_cells = latitudes_deg.size * longitudes_deg.size

    for label_id in range(1, component_count + 1):
        component_mask = labels == label_id
        coords = np.argwhere(component_mask)
        if coords.size == 0:
            continue
        lat_idx = coords[:, 0]
        lon_idx = coords[:, 1]
        lat_vals = latitudes_deg[lat_idx]
        lon_vals = longitudes_deg[lon_idx]
        points = np.column_stack(
            [
                lon_idx.astype(np.float64) * np.maximum(np.cos(np.deg2rad(lat_vals)), 0.25),
                lat_idx.astype(np.float64),
            ]
        )
        if coords.shape[0] >= 3:
            centered = points - points.mean(axis=0, keepdims=True)
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 1e-6)
            major_axis = eigenvectors[:, order[0]]
            orientation_rad = float(np.arctan2(major_axis[1], major_axis[0]))
        else:
            eigenvalues = np.array([1.0, 1.0], dtype=np.float64)
            orientation_rad = 0.0
        summaries.append(
            {
                "component_id": int(label_id),
                "cell_count": int(coords.shape[0]),
                "coverage_fraction": float(coords.shape[0] / max(total_cells, 1)),
                "mean_latitude_deg": float(np.mean(lat_vals)),
                "mean_longitude_deg": circular_mean_longitude_deg(lon_vals),
                "elongation_ratio": float(eigenvalues[0] / eigenvalues[1]),
                "orientation_deg": float(np.rad2deg(orientation_rad)),
            }
        )

    summaries.sort(key=lambda item: item["cell_count"], reverse=True)
    return summaries


def smooth_2d(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    sigma = max(float(sigma_cells), 0.0)
    if sigma <= 0.0:
        return np.asarray(field, dtype=np.float32)
    return np.asarray(
        ndimage.gaussian_filter(
            np.asarray(field, dtype=np.float32),
            sigma=(sigma, sigma),
            mode=("nearest", "wrap"),
            truncate=2.0,
        ),
        dtype=np.float32,
    )


def compute_horizontal_gradient_km(
    field: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=np.float32)
    lat_step_deg = float(np.mean(np.abs(np.diff(latitudes_deg))))
    lon_step_deg = float(np.mean(np.abs(np.diff(longitudes_deg))))
    dy_km = 111.0 * max(lat_step_deg, 1e-6)
    dfield_dy = np.gradient(field, axis=0) / dy_km

    wrapped = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)
    dfield_dlon = (wrapped[:, 2:] - wrapped[:, :-2]) * 0.5
    cos_lat = np.maximum(np.cos(np.deg2rad(latitudes_deg))[:, None], 0.15)
    dx_km = 111.0 * max(lon_step_deg, 1e-6) * cos_lat
    dfield_dx = dfield_dlon / dx_km
    return np.asarray(dfield_dx, dtype=np.float32), np.asarray(dfield_dy, dtype=np.float32)


def percentile_mask(field: np.ndarray, top_percent: float) -> np.ndarray:
    valid = np.isfinite(field)
    if not valid.any():
        return np.zeros_like(field, dtype=bool)
    threshold = float(np.quantile(field[valid], max(0.0, 1.0 - top_percent / 100.0)))
    return np.asarray(field >= threshold, dtype=bool)


def make_surface_attached_analysis(fields: CoarseFields, output_dir: Path) -> dict[str, Any]:
    recipe = VARIANT_RECIPES["surface-attached-theta-rh-latmean"]
    thermal_z, moisture_z, _ = build_variant_fields(
        recipe,
        temperature_field=fields.temperature_k,
        relative_humidity_field=fields.relative_humidity_pct,
        specific_humidity_field=fields.specific_humidity,
        pressure_levels_hpa=fields.pressure_levels_hpa,
    )
    keep_mask, _ = build_score_mask(
        thermal_z,
        moisture_z,
        keep_top_percent=recipe.keep_top_percent,
        axis_min_abs_zscore=recipe.axis_min_abs_zscore,
    )
    raw_class_masks = classify_quadrants(keep_mask, thermal_z, moisture_z)

    dominant_class = np.full(
        (fields.latitudes_deg.size, fields.longitudes_deg.size),
        fill_value=-1,
        dtype=np.int16,
    )
    max_depth_km = np.zeros_like(dominant_class, dtype=np.float32)
    component_rows: list[dict[str, Any]] = []
    class_totals: dict[str, dict[str, Any]] = {}

    for class_index, class_key in enumerate(CLASS_ORDER):
        bridged = apply_bridge_gap(raw_class_masks[class_key], recipe.bridge_gap_levels)
        filtered, _ = filter_components(
            bridged,
            min_component_voxels=recipe.min_component_voxels,
            min_component_pressure_span_levels=recipe.min_component_pressure_span_levels,
        )
        surface_attached, _ = filter_surface_attached_components(
            filtered,
            fields.pressure_levels_hpa,
        )
        summaries = summarize_3d_components(
            surface_attached,
            fields.pressure_levels_hpa,
            fields.latitudes_deg,
            fields.longitudes_deg,
            class_key=class_key,
        )
        component_rows.extend(summaries)
        class_totals[class_key] = {
            "voxel_count": int(np.count_nonzero(surface_attached)),
            "component_count": int(len(summaries)),
            "largest_component_voxels": int(summaries[0]["voxel_count"] if summaries else 0),
        }

        depth_field = np.zeros_like(max_depth_km, dtype=np.float32)
        for lat_index in range(fields.latitudes_deg.size):
            for lon_index in range(fields.longitudes_deg.size):
                column = np.flatnonzero(surface_attached[:, lat_index, lon_index])
                if column.size == 0:
                    continue
                top_pressure = float(np.max(fields.pressure_levels_hpa[column]))
                bottom_pressure = float(np.min(fields.pressure_levels_hpa[column]))
                depth_km = pressure_to_standard_height_km(bottom_pressure) - pressure_to_standard_height_km(
                    top_pressure
                )
                depth_field[lat_index, lon_index] = float(depth_km)
        replace = depth_field > max_depth_km
        dominant_class[replace] = class_index
        max_depth_km[replace] = depth_field[replace]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    cmap = ListedColormap([SURFACE_CLASS_COLORS[key] for key in CLASS_ORDER])
    masked_dominant = np.ma.masked_where(dominant_class < 0, dominant_class)
    axes[0].imshow(
        masked_dominant,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    axes[0].set_title("Surface-attached dominant proxy class")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[1].imshow(
        np.ma.masked_where(max_depth_km <= 0.0, max_depth_km),
        origin="upper",
        aspect="auto",
        cmap="viridis",
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    axes[1].set_title("Maximum attached depth (km)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.savefig(output_dir / "surface-attached-footprint-and-depth.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for class_key in CLASS_ORDER:
        rows = [row for row in component_rows if row["class_key"] == class_key]
        if not rows:
            continue
        ax.scatter(
            [row["footprint_cells"] for row in rows],
            [row["depth_km"] for row in rows],
            c=[row["max_centroid_displacement_km"] for row in rows],
            cmap="plasma",
            label=class_key.replace("_", " "),
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Horizontal footprint cells")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Surface-attached component size vs depth")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output_dir / "surface-attached-component-scatter.png", dpi=170)
    plt.close(fig)

    component_rows.sort(key=lambda item: item["voxel_count"], reverse=True)
    summary = {
        "decision": "Use 1000 hPa as the near-surface seed proxy and keep only 3D components whose class-connected volume actually touches that seed layer.",
        "global_surface_coverage_fraction": float(np.count_nonzero(dominant_class >= 0) / dominant_class.size),
        "median_attached_depth_km": float(
            np.median(max_depth_km[max_depth_km > 0.0]) if np.any(max_depth_km > 0.0) else 0.0
        ),
        "class_totals": class_totals,
        "top_components": component_rows[:12],
        "artifact_paths": {
            "footprint_map": repo_relative(output_dir / "surface-attached-footprint-and-depth.png"),
            "component_scatter": repo_relative(output_dir / "surface-attached-component-scatter.png"),
        },
    }
    write_json(output_dir / "surface-attached-summary.json", summary)
    return summary


def make_source_map_analysis(fields: CoarseFields, output_dir: Path) -> dict[str, Any]:
    dt_seconds = DEFAULT_SOURCE_TRACE_HOURS * 3600.0
    lat_shift_deg = -(fields.v850_ms * dt_seconds) / 111000.0
    cos_lat = np.maximum(np.cos(np.deg2rad(fields.latitudes_deg))[:, None], 0.15)
    lon_shift_deg = -(fields.u850_ms * dt_seconds) / (111000.0 * cos_lat)
    source_lat = np.clip(fields.latitudes_deg[:, None] + lat_shift_deg, -89.0, 89.0)
    source_lon = wrap_longitude_deg(fields.longitudes_deg[None, :] + lon_shift_deg)
    travel_distance_km = np.sqrt(
        np.square(fields.u850_ms * dt_seconds) + np.square(fields.v850_ms * dt_seconds)
    ) / 1000.0
    speed = np.sqrt(np.square(fields.u850_ms) + np.square(fields.v850_ms))
    active = speed >= DEFAULT_SOURCE_MIN_SPEED_MS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im0 = axes[0].imshow(
        np.ma.masked_where(~active, source_lat - fields.latitudes_deg[:, None]),
        origin="upper",
        aspect="auto",
        cmap="coolwarm",
        vmin=-20,
        vmax=20,
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    axes[0].set_title("24 h source latitude shift (deg)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(
        np.ma.masked_where(~active, travel_distance_km),
        origin="upper",
        aspect="auto",
        cmap="magma",
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    axes[1].set_title("24 h travel distance proxy (km)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.savefig(output_dir / "source-map-latshift-distance.png", dpi=170)
    plt.close(fig)

    summary = {
        "decision": "Use a steady 850 hPa backward-displacement proxy over 24 hours instead of true trajectories because only one global wind analysis is available locally.",
        "active_fraction": float(np.count_nonzero(active) / active.size),
        "median_travel_distance_km": float(np.median(travel_distance_km[active])) if np.any(active) else 0.0,
        "p90_travel_distance_km": float(np.quantile(travel_distance_km[active], 0.9)) if np.any(active) else 0.0,
        "mean_source_latitude_shift_deg": float(
            np.mean((source_lat - fields.latitudes_deg[:, None])[active])
        )
        if np.any(active)
        else 0.0,
        "north_origin_fraction": float(
            np.count_nonzero(active & ((source_lat - fields.latitudes_deg[:, None]) >= 5.0))
            / max(np.count_nonzero(active), 1)
        ),
        "south_origin_fraction": float(
            np.count_nonzero(active & ((source_lat - fields.latitudes_deg[:, None]) <= -5.0))
            / max(np.count_nonzero(active), 1)
        ),
        "artifact_paths": {
            "map": repo_relative(output_dir / "source-map-latshift-distance.png"),
        },
    }
    write_json(output_dir / "source-map-summary.json", summary)
    return summary


def make_distinctness_analysis(fields: CoarseFields, output_dir: Path) -> dict[str, Any]:
    theta_anomaly = np.asarray(fields.theta_k - fields.theta_climatology_mean_k, dtype=np.float32)
    safe_std = np.where(fields.theta_climatology_std_k > 0.25, fields.theta_climatology_std_k, np.nan)
    theta_z = np.asarray(theta_anomaly / safe_std, dtype=np.float32)
    theta_z = np.nan_to_num(theta_z, nan=0.0)
    theta_z = np.asarray(
        ndimage.gaussian_filter(
            theta_z,
            sigma=(0.0, 1.0, 1.0),
            mode=("nearest", "nearest", "wrap"),
            truncate=2.0,
        ),
        dtype=np.float32,
    )
    warm_raw = theta_z >= DEFAULT_DISTINCTNESS_SIGMA
    cold_raw = theta_z <= -DEFAULT_DISTINCTNESS_SIGMA
    warm_mask, _ = filter_components(
        warm_raw,
        min_component_voxels=DEFAULT_DISTINCTNESS_MIN_VOXELS,
        min_component_pressure_span_levels=DEFAULT_DISTINCTNESS_MIN_LEVELS,
    )
    cold_mask, _ = filter_components(
        cold_raw,
        min_component_voxels=DEFAULT_DISTINCTNESS_MIN_VOXELS,
        min_component_pressure_span_levels=DEFAULT_DISTINCTNESS_MIN_LEVELS,
    )

    warm_components = summarize_3d_components(
        warm_mask,
        fields.pressure_levels_hpa,
        fields.latitudes_deg,
        fields.longitudes_deg,
        class_key="warm",
    )
    cold_components = summarize_3d_components(
        cold_mask,
        fields.pressure_levels_hpa,
        fields.latitudes_deg,
        fields.longitudes_deg,
        class_key="cold",
    )

    level_warm_share = np.mean(warm_mask, axis=(1, 2))
    level_cold_share = np.mean(cold_mask, axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im = axes[0].imshow(
        theta_z[int(np.argmin(np.abs(fields.pressure_levels_hpa - 850.0)))],
        origin="upper",
        aspect="auto",
        cmap="coolwarm",
        vmin=-3.0,
        vmax=3.0,
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    axes[0].set_title("850 hPa dry-theta distinctness (sigma)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    axes[1].plot(level_warm_share * 100.0, fields.pressure_levels_hpa, color="#d6604d", label="Warm")
    axes[1].plot(level_cold_share * 100.0, fields.pressure_levels_hpa, color="#4393c3", label="Cold")
    axes[1].invert_yaxis()
    axes[1].set_title("Distinctness occupancy by pressure level")
    axes[1].set_xlabel("Occupied area (%)")
    axes[1].set_ylabel("Pressure (hPa)")
    axes[1].legend()
    fig.savefig(output_dir / "distinctness-map-and-level-share.png", dpi=170)
    plt.close(fig)

    summary = {
        "decision": "Use climatology-relative dry potential temperature with a 1.5 sigma threshold in the 1000-250 hPa window to isolate meaningful warm and cold volumes without letting background stratification dominate.",
        "warm_component_count": int(len(warm_components)),
        "cold_component_count": int(len(cold_components)),
        "warm_voxel_fraction": float(np.count_nonzero(warm_mask) / warm_mask.size),
        "cold_voxel_fraction": float(np.count_nonzero(cold_mask) / cold_mask.size),
        "top_warm_components": warm_components[:8],
        "top_cold_components": cold_components[:8],
        "artifact_paths": {
            "map": repo_relative(output_dir / "distinctness-map-and-level-share.png"),
        },
    }
    write_json(output_dir / "distinctness-summary.json", summary)
    return summary


def make_transport_analysis(fields: CoarseFields, output_dir: Path) -> dict[str, Any]:
    smoothed_anomaly = smooth_2d(fields.theta_anomaly_850_k, 1.2)
    dtheta_dx, dtheta_dy = compute_horizontal_gradient_km(
        smoothed_anomaly,
        fields.latitudes_deg,
        fields.longitudes_deg,
    )
    transport_k_per_day = -(
        fields.u850_ms * 0.001 * dtheta_dx + fields.v850_ms * 0.001 * dtheta_dy
    ) * 86400.0
    speed = np.sqrt(np.square(fields.u850_ms) + np.square(fields.v850_ms))
    mask = percentile_mask(np.abs(transport_k_per_day), DEFAULT_TRANSPORT_TOP_PERCENT)
    mask &= speed >= 7.5
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=np.uint8))
    transport_components = summarize_2d_components(mask, fields.latitudes_deg, fields.longitudes_deg)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    im = ax.imshow(
        transport_k_per_day,
        origin="upper",
        aspect="auto",
        cmap="coolwarm",
        vmin=-12.0,
        vmax=12.0,
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    ax.contour(
        fields.longitudes_deg,
        fields.latitudes_deg,
        mask.astype(np.int8),
        levels=[0.5],
        colors="black",
        linewidths=0.6,
    )
    ax.set_title("850 hPa thermodynamic transport proxy (K day^-1)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output_dir / "transport-tubes-map.png", dpi=170)
    plt.close(fig)

    summary = {
        "decision": "Treat coherent transport tubes as high-magnitude 850 hPa advection of climatology-relative dry-theta anomaly, filtered to strong-flow cells and summarized as 2D pathway components.",
        "occupied_fraction": float(np.count_nonzero(mask) / mask.size),
        "component_count": int(len(transport_components)),
        "warm_transport_fraction": float(
            np.count_nonzero(mask & (transport_k_per_day > 0.0)) / max(np.count_nonzero(mask), 1)
        ),
        "median_abs_transport_k_per_day": float(
            np.median(np.abs(transport_k_per_day[mask])) if np.any(mask) else 0.0
        ),
        "top_components": transport_components[:12],
        "artifact_paths": {
            "map": repo_relative(output_dir / "transport-tubes-map.png"),
        },
    }
    write_json(output_dir / "transport-summary.json", summary)
    return summary


def make_front_analysis(fields: CoarseFields, output_dir: Path) -> dict[str, Any]:
    smoothed_temperature = smooth_2d(fields.temperature_850_k, 2.0)
    zonal_mean = np.mean(smoothed_temperature, axis=1, keepdims=True)
    anomaly_temperature = np.asarray(smoothed_temperature - zonal_mean, dtype=np.float32)
    dtemp_dx, dtemp_dy = compute_horizontal_gradient_km(
        anomaly_temperature,
        fields.latitudes_deg,
        fields.longitudes_deg,
    )
    gradient_mag = np.sqrt(np.square(dtemp_dx) + np.square(dtemp_dy)) * 100.0
    frontal_mask = percentile_mask(gradient_mag, DEFAULT_FRONT_TOP_PERCENT)
    frontal_mask = ndimage.binary_closing(frontal_mask, structure=np.ones((3, 3), dtype=np.uint8))
    labels, component_count = label_wrapped_2d(frontal_mask)
    cleaned = np.zeros_like(frontal_mask, dtype=bool)
    for label_id in range(1, component_count + 1):
        component = labels == label_id
        if int(np.count_nonzero(component)) < DEFAULT_FRONT_MIN_CELLS:
            continue
        cleaned |= component
    frontal_components = summarize_2d_components(cleaned, fields.latitudes_deg, fields.longitudes_deg)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    im = ax.imshow(
        smoothed_temperature,
        origin="upper",
        aspect="auto",
        cmap="coolwarm",
        extent=[
            float(fields.longitudes_deg.min()),
            float(fields.longitudes_deg.max()),
            float(fields.latitudes_deg.min()),
            float(fields.latitudes_deg.max()),
        ],
    )
    ax.contour(
        fields.longitudes_deg,
        fields.latitudes_deg,
        cleaned.astype(np.int8),
        levels=[0.5],
        colors="black",
        linewidths=0.7,
    )
    ax.set_title("Synoptic frontal zones from smoothed 850 hPa temperature")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output_dir / "frontal-zones-map.png", dpi=170)
    plt.close(fig)

    elongations = [component["elongation_ratio"] for component in frontal_components]
    summary = {
        "decision": "Use large-scale 850 hPa temperature structure, subtract the zonal-mean background, and keep only the strongest broad gradient zones after closing and small-component cleanup.",
        "frontal_coverage_fraction": float(np.count_nonzero(cleaned) / cleaned.size),
        "major_segment_count": int(len(frontal_components)),
        "median_segment_elongation": float(np.median(elongations)) if elongations else 0.0,
        "top_segments": frontal_components[:12],
        "artifact_paths": {
            "map": repo_relative(output_dir / "frontal-zones-map.png"),
        },
    }
    write_json(output_dir / "front-summary.json", summary)
    return summary


def build_summary_markdown(
    output_dir: Path,
    surface_summary: dict[str, Any],
    source_summary: dict[str, Any],
    distinctness_summary: dict[str, Any],
    transport_summary: dict[str, Any],
    front_summary: dict[str, Any],
) -> str:
    return f"""# V0 Representation Run

Timestamp: `{DEFAULT_TIMESTAMP}`

## Decisions

- Work on a coarse global `1°` grid (`0.25°` ERA5 strided by 4) so all five ideas can be compared quickly and consistently.
- Limit 3D thermodynamic structure work to `1000-250 hPa` because the current repo notes already show full-column dry-theta anomalies become upper-level dominated.
- Treat `1000 hPa` as the surface seed proxy for v0 because the local bundle does not include a true near-surface thermodynamic stack.
- Treat recent source history and transport as `850 hPa` proxies because only a single global `850 hPa` wind analysis is available locally.

## Outputs

- Surface-attached bodies: `{surface_summary['artifact_paths']['footprint_map']}`
- Source map: `{source_summary['artifact_paths']['map']}`
- Distinctness volumes: `{distinctness_summary['artifact_paths']['map']}`
- Transport tubes: `{transport_summary['artifact_paths']['map']}`
- Frontal zones: `{front_summary['artifact_paths']['map']}`

## Headline Stats

- Surface-attached coverage: `{surface_summary['global_surface_coverage_fraction']:.3f}` of coarse surface cells.
- Median attached depth: `{surface_summary['median_attached_depth_km']:.2f} km`.
- Source-map active fraction: `{source_summary['active_fraction']:.3f}` with median `{source_summary['median_travel_distance_km']:.0f} km` backward travel.
- Distinctness volume counts: `{distinctness_summary['warm_component_count']}` warm and `{distinctness_summary['cold_component_count']}` cold components.
- Transport occupied fraction: `{transport_summary['occupied_fraction']:.3f}` across the coarse grid.
- Major frontal segments: `{front_summary['major_segment_count']}` with coverage `{front_summary['frontal_coverage_fraction']:.3f}`.
"""


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fields = load_coarse_fields(args)

    surface_summary = make_surface_attached_analysis(fields, output_dir)
    source_summary = make_source_map_analysis(fields, output_dir)
    distinctness_summary = make_distinctness_analysis(fields, output_dir)
    transport_summary = make_transport_analysis(fields, output_dir)
    front_summary = make_front_analysis(fields, output_dir)

    overall_summary = {
        "timestamp": fields.timestamp,
        "grid": {
            "pressure_level_count": int(fields.pressure_levels_hpa.size),
            "latitude_count": int(fields.latitudes_deg.size),
            "longitude_count": int(fields.longitudes_deg.size),
            "pressure_window_hpa": [
                float(np.max(fields.pressure_levels_hpa)),
                float(np.min(fields.pressure_levels_hpa)),
            ],
        },
        "surface_attached": surface_summary,
        "source_map": source_summary,
        "distinctness": distinctness_summary,
        "transport": transport_summary,
        "fronts": front_summary,
    }
    write_json(output_dir / "summary.json", overall_summary)
    write_markdown(
        output_dir / "summary.md",
        build_summary_markdown(
            output_dir,
            surface_summary,
            source_summary,
            distinctness_summary,
            transport_summary,
            front_summary,
        ),
    )
    print(f"Wrote v0 representation artifacts to {repo_relative(output_dir)}")


if __name__ == "__main__":
    main()
