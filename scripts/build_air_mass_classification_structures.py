from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_voxel_builder import (  # noqa: E402
    build_exposed_face_mesh_from_mask,
    coordinate_step_degrees,
    timestamp_to_slug,
)


OUTPUT_VERSION = 1
DEFAULT_DATASET_PATH = Path("data/global-air-mass-proxy-bundle_2021-11_p1-to-1000.nc")
DEFAULT_OUTPUT_DIR = Path("public/air-mass-structures")
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 18.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4

THERMAL_TEMPERATURE = "temperature"
THERMAL_THETA = "dry_potential_temperature"
MOISTURE_RH = "relative_humidity"
MOISTURE_Q = "specific_humidity"

CLASS_ORDER = (
    "warm_dry",
    "warm_moist",
    "cold_dry",
    "cold_moist",
)
CLASS_PROXY_LABELS = {
    "warm_dry": "Continental Tropical / Superior Proxy",
    "warm_moist": "Maritime Tropical / Monsoon Proxy",
    "cold_dry": "Continental Polar / Arctic Proxy",
    "cold_moist": "Maritime Polar Proxy",
}
VARIABLE_ALIASES = {
    "temperature": ("t", "temperature"),
    "relative_humidity": ("r", "relative_humidity"),
    "specific_humidity": ("q", "specific_humidity"),
}
VOLUME_COMPONENT_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
REFERENCE_PRESSURE_HPA = 1000.0


@dataclass(frozen=True)
class VariantRecipe:
    name: str
    label: str
    thermal_field: str
    moisture_field: str
    keep_top_percent: float
    axis_min_abs_zscore: float
    smoothing_sigma_cells: float
    bridge_gap_levels: int
    min_component_voxels: int
    min_component_pressure_span_levels: int
    pressure_min_hpa: float
    pressure_max_hpa: float
    surface_attached_only: bool


VARIANT_RECIPES: dict[str, VariantRecipe] = {
    "temperature-rh-latmean": VariantRecipe(
        name="temperature-rh-latmean",
        label="Temperature + RH Anomaly",
        thermal_field=THERMAL_TEMPERATURE,
        moisture_field=MOISTURE_RH,
        keep_top_percent=16.0,
        axis_min_abs_zscore=0.55,
        smoothing_sigma_cells=1.0,
        bridge_gap_levels=1,
        min_component_voxels=24,
        min_component_pressure_span_levels=2,
        pressure_min_hpa=250.0,
        pressure_max_hpa=1000.0,
        surface_attached_only=False,
    ),
    "theta-rh-latmean": VariantRecipe(
        name="theta-rh-latmean",
        label="Theta + RH Anomaly",
        thermal_field=THERMAL_THETA,
        moisture_field=MOISTURE_RH,
        keep_top_percent=14.0,
        axis_min_abs_zscore=0.55,
        smoothing_sigma_cells=1.0,
        bridge_gap_levels=1,
        min_component_voxels=24,
        min_component_pressure_span_levels=2,
        pressure_min_hpa=250.0,
        pressure_max_hpa=1000.0,
        surface_attached_only=False,
    ),
    "theta-q-latmean": VariantRecipe(
        name="theta-q-latmean",
        label="Theta + Specific Humidity",
        thermal_field=THERMAL_THETA,
        moisture_field=MOISTURE_Q,
        keep_top_percent=13.0,
        axis_min_abs_zscore=0.65,
        smoothing_sigma_cells=1.0,
        bridge_gap_levels=2,
        min_component_voxels=28,
        min_component_pressure_span_levels=2,
        pressure_min_hpa=250.0,
        pressure_max_hpa=1000.0,
        surface_attached_only=False,
    ),
    "surface-attached-theta-rh-latmean": VariantRecipe(
        name="surface-attached-theta-rh-latmean",
        label="Surface-Attached Theta + RH",
        thermal_field=THERMAL_THETA,
        moisture_field=MOISTURE_RH,
        keep_top_percent=16.0,
        axis_min_abs_zscore=0.5,
        smoothing_sigma_cells=1.0,
        bridge_gap_levels=1,
        min_component_voxels=20,
        min_component_pressure_span_levels=2,
        pressure_min_hpa=250.0,
        pressure_max_hpa=1000.0,
        surface_attached_only=True,
    ),
}


@dataclass(frozen=True)
class DatasetContents:
    dataset_path: Path
    variable_names: dict[str, str]
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    longitude_order: np.ndarray
    timestamps: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 3D air-mass proxy voxel shells from pressure-level ERA5 temperature, "
            "relative humidity, and specific humidity."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the combined pressure-level NetCDF bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated air-mass assets will be written.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_RECIPES),
        default="temperature-rh-latmean",
        help="Air-mass proxy recipe to build.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default=",".join(DEFAULT_INCLUDE_TIMESTAMPS),
        help="Comma-separated ISO minute timestamps to build.",
    )
    parser.add_argument(
        "--base-radius",
        type=float,
        default=DEFAULT_BASE_RADIUS,
        help="Base world radius of the globe mesh.",
    )
    parser.add_argument(
        "--vertical-span",
        type=float,
        default=DEFAULT_VERTICAL_SPAN,
        help="World units spanning 1000 hPa to 1 hPa.",
    )
    parser.add_argument(
        "--latitude-stride",
        type=int,
        default=DEFAULT_LATITUDE_STRIDE,
        help="Keep every Nth latitude sample when building the shell.",
    )
    parser.add_argument(
        "--longitude-stride",
        type=int,
        default=DEFAULT_LONGITUDE_STRIDE,
        help="Keep every Nth longitude sample when building the shell.",
    )
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if text.endswith("Z"):
        return text[:-1]
    return text


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {format_display_path(resolved)}")
    return resolved


def format_display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        home = Path.home()
        try:
            return f"~/{path.relative_to(home).as_posix()}"
        except ValueError:
            return path.name or "<external-path>"


def normalize_longitudes_with_order(
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.mod(np.asarray(longitudes_deg, dtype=np.float64) + 180.0, 360.0) - 180.0
    order = np.argsort(normalized, kind="stable")
    return normalized[order].astype(np.float32), order.astype(np.int64)


def reorder_longitude_axis(field: np.ndarray, longitude_order: np.ndarray) -> np.ndarray:
    return np.take(np.asarray(field), longitude_order, axis=-1)


def detect_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in dataset.coords:
            return candidate
    for candidate in candidates:
        if candidate in dataset.dims:
            return candidate
    raise KeyError(f"Could not find coordinate among: {', '.join(candidates)}")


def find_variable_name(dataset: xr.Dataset, aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        if alias in dataset.data_vars:
            return alias
    available = ", ".join(sorted(dataset.data_vars))
    raise KeyError(f"Could not find any of {aliases!r}. Available variables: {available}")


def load_dataset_contents(dataset_path: Path) -> DatasetContents:
    dataset = xr.open_dataset(dataset_path)
    try:
        pressure_name = detect_coord_name(dataset, ("pressure_level", "level", "isobaricInhPa"))
        latitude_name = detect_coord_name(dataset, ("latitude", "lat"))
        longitude_name = detect_coord_name(dataset, ("longitude", "lon"))
        time_name = detect_coord_name(dataset, ("valid_time", "time"))
        variable_names = {
            key: find_variable_name(dataset, aliases)
            for key, aliases in VARIABLE_ALIASES.items()
        }
        longitudes_deg, longitude_order = normalize_longitudes_with_order(
            np.asarray(dataset.coords[longitude_name].values, dtype=np.float32)
        )
        return DatasetContents(
            dataset_path=dataset_path,
            variable_names=variable_names,
            pressure_levels_hpa=np.asarray(dataset.coords[pressure_name].values, dtype=np.float32),
            latitudes_deg=np.asarray(dataset.coords[latitude_name].values, dtype=np.float32),
            longitudes_deg=longitudes_deg,
            longitude_order=longitude_order,
            timestamps=[
                timestamp_to_iso_minute(value)
                for value in np.asarray(dataset.coords[time_name].values, dtype="datetime64[m]")
            ],
        )
    finally:
        dataset.close()


def resolve_requested_timestamps(
    all_timestamps: list[str],
    requested_csv: str,
) -> list[str]:
    requested = [part.strip() for part in requested_csv.split(",") if part.strip()]
    if not requested:
        return list(all_timestamps)
    return [timestamp for timestamp in all_timestamps if timestamp in requested]


def load_field_at_timestamp(
    dataset: xr.Dataset,
    *,
    variable_name: str,
    timestamp: str,
) -> np.ndarray:
    time_name = detect_coord_name(dataset, ("valid_time", "time"))
    pressure_name = detect_coord_name(dataset, ("pressure_level", "level", "isobaricInhPa"))
    latitude_name = detect_coord_name(dataset, ("latitude", "lat"))
    longitude_name = detect_coord_name(dataset, ("longitude", "lon"))
    time_values = [
        timestamp_to_iso_minute(value)
        for value in np.asarray(dataset.coords[time_name].values, dtype="datetime64[m]")
    ]
    if timestamp not in time_values:
        raise KeyError(f"Timestamp {timestamp} is not available in {format_display_path(Path(str(dataset.encoding.get('source', 'dataset'))))}.")
    time_index = time_values.index(timestamp)
    data_array = dataset[variable_name].isel({time_name: time_index})
    field = np.asarray(
        data_array.transpose(pressure_name, latitude_name, longitude_name).values,
        dtype=np.float32,
    )
    return field


def compute_dry_potential_temperature(
    temperature_values: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> np.ndarray:
    pressure_scale = np.power(
        REFERENCE_PRESSURE_HPA / np.asarray(pressure_levels_hpa, dtype=np.float32)[:, None, None],
        POTENTIAL_TEMPERATURE_KAPPA,
        dtype=np.float32,
    )
    return np.asarray(np.asarray(temperature_values, dtype=np.float32) * pressure_scale, dtype=np.float32)


def select_pressure_window(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = min(float(pressure_min_hpa), float(pressure_max_hpa))
    upper = max(float(pressure_min_hpa), float(pressure_max_hpa))
    keep = np.asarray((pressure_levels_hpa >= lower) & (pressure_levels_hpa <= upper), dtype=bool)
    if not keep.any():
        raise ValueError(f"No pressure levels fall within [{lower}, {upper}] hPa.")
    return np.asarray(field[keep], dtype=np.float32), np.asarray(pressure_levels_hpa[keep], dtype=np.float32)


def stride_spatial_axes(
    field: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    latitude_stride: int,
    longitude_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_stride = max(int(latitude_stride), 1)
    lon_stride = max(int(longitude_stride), 1)
    return (
        np.asarray(field[:, ::lat_stride, ::lon_stride], dtype=np.float32),
        np.asarray(latitudes_deg[::lat_stride], dtype=np.float32),
        np.asarray(longitudes_deg[::lon_stride], dtype=np.float32),
    )


def build_latitude_mean_anomaly(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latitude_band_mean = np.nanmean(field, axis=2, keepdims=True)
    anomaly = np.asarray(field - latitude_band_mean, dtype=np.float32)
    return anomaly, np.asarray(latitude_band_mean, dtype=np.float32)


def standardize_per_level(anomaly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    level_scale = np.asarray(np.nanstd(anomaly, axis=(1, 2)), dtype=np.float32)
    safe_scale = np.where(level_scale > 1e-6, level_scale, 1.0).astype(np.float32)
    standardized = np.asarray(anomaly / safe_scale[:, None, None], dtype=np.float32)
    return standardized, safe_scale


def smooth_levels(field: np.ndarray, smoothing_sigma_cells: float) -> np.ndarray:
    sigma = max(float(smoothing_sigma_cells), 0.0)
    if sigma <= 0.0:
        return np.asarray(field, dtype=np.float32)
    return np.asarray(
        ndimage.gaussian_filter(
            np.asarray(field, dtype=np.float32),
            sigma=(0.0, sigma, sigma),
            mode=("nearest", "nearest", "wrap"),
            truncate=2.0,
        ),
        dtype=np.float32,
    )


def build_score_mask(
    thermal_z: np.ndarray,
    moisture_z: np.ndarray,
    *,
    keep_top_percent: float,
    axis_min_abs_zscore: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    combined_score = np.asarray(
        np.sqrt(np.abs(thermal_z) * np.abs(moisture_z)),
        dtype=np.float32,
    )
    keep_mask = np.zeros_like(combined_score, dtype=bool)
    thresholds_by_level: list[dict[str, Any]] = []
    keep_fraction = max(min(float(keep_top_percent) / 100.0, 1.0), 0.0)

    for level_index in range(combined_score.shape[0]):
        score_slice = combined_score[level_index]
        valid = np.isfinite(score_slice)
        if not valid.any() or keep_fraction <= 0.0:
            thresholds_by_level.append(
                {
                    "score_threshold": None,
                    "kept_cell_count": 0,
                }
            )
            continue
        level_threshold = float(
            np.quantile(
                score_slice[valid],
                max(0.0, 1.0 - keep_fraction),
            )
        )
        level_keep = np.asarray(score_slice >= level_threshold, dtype=bool)
        level_keep &= np.abs(thermal_z[level_index]) >= float(axis_min_abs_zscore)
        level_keep &= np.abs(moisture_z[level_index]) >= float(axis_min_abs_zscore)
        keep_mask[level_index] = level_keep
        thresholds_by_level.append(
            {
                "score_threshold": level_threshold,
                "kept_cell_count": int(np.count_nonzero(level_keep)),
            }
        )

    return keep_mask, thresholds_by_level


def classify_quadrants(
    keep_mask: np.ndarray,
    thermal_z: np.ndarray,
    moisture_z: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "warm_dry": np.asarray(
            keep_mask & (thermal_z >= 0.0) & (moisture_z < 0.0),
            dtype=bool,
        ),
        "warm_moist": np.asarray(
            keep_mask & (thermal_z >= 0.0) & (moisture_z >= 0.0),
            dtype=bool,
        ),
        "cold_dry": np.asarray(
            keep_mask & (thermal_z < 0.0) & (moisture_z < 0.0),
            dtype=bool,
        ),
        "cold_moist": np.asarray(
            keep_mask & (thermal_z < 0.0) & (moisture_z >= 0.0),
            dtype=bool,
        ),
    }


def apply_bridge_gap(mask: np.ndarray, max_gap_levels: int) -> np.ndarray:
    if max_gap_levels <= 0:
        return np.asarray(mask, dtype=bool)
    bridged = np.asarray(mask, dtype=bool).copy()
    level_count, latitude_count, longitude_count = bridged.shape
    for latitude_index in range(latitude_count):
        for longitude_index in range(longitude_count):
            column = bridged[:, latitude_index, longitude_index]
            active = np.flatnonzero(column)
            if active.size < 2:
                continue
            for left, right in zip(active[:-1], active[1:]):
                gap = int(right - left - 1)
                if 0 < gap <= max_gap_levels:
                    column[left : right + 1] = True
    return bridged


def build_seam_merged_component_info(
    labels: np.ndarray,
    seam_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component_count = int(np.max(labels))
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

    for first, second in seam_pairs:
        first_id = int(first)
        second_id = int(second)
        if first_id <= 0 or second_id <= 0:
            continue
        first_root = find_root(first_id)
        second_root = find_root(second_id)
        if first_root == second_root:
            continue
        high = max(first_root, second_root)
        low = min(first_root, second_root)
        root_map[high] = low

    for component_id in range(1, component_count + 1):
        root_map[component_id] = find_root(component_id)

    return root_map, np.unique(root_map[1:])


def label_wrapped_volume_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    longitude_count = occupied.shape[2]
    extended = np.concatenate([occupied, occupied[..., :1]], axis=2)
    labels, component_count = ndimage.label(extended, structure=VOLUME_COMPONENT_STRUCTURE)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    seam_pairs = np.column_stack([labels[..., 0].reshape(-1), labels[..., -1].reshape(-1)])
    root_map, unique_root_ids = build_seam_merged_component_info(labels, seam_pairs)
    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_root_ids] = np.arange(1, unique_root_ids.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[..., :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_root_ids.size)


def filter_components(
    mask: np.ndarray,
    *,
    min_component_voxels: int,
    min_component_pressure_span_levels: int,
) -> tuple[np.ndarray, dict[str, int]]:
    labels, _ = label_wrapped_volume_components(mask)
    if labels.max() <= 0:
        return np.zeros_like(mask, dtype=bool), {
            "component_count": 0,
            "largest_component_voxel_count": 0,
        }

    filtered = np.zeros_like(mask, dtype=bool)
    kept_sizes: list[int] = []
    component_count = 0
    for label_id in range(1, int(labels.max()) + 1):
        component_mask = labels == label_id
        voxel_count = int(np.count_nonzero(component_mask))
        if voxel_count < max(int(min_component_voxels), 1):
            continue
        occupied_levels = np.flatnonzero(np.any(component_mask, axis=(1, 2)))
        if occupied_levels.size < max(int(min_component_pressure_span_levels), 1):
            continue
        filtered |= component_mask
        kept_sizes.append(voxel_count)
        component_count += 1

    return filtered, {
        "component_count": int(component_count),
        "largest_component_voxel_count": int(max(kept_sizes) if kept_sizes else 0),
    }


def filter_surface_attached_components(
    mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    labels, _ = label_wrapped_volume_components(mask)
    if labels.max() <= 0:
        return np.zeros_like(mask, dtype=bool), {
            "surface_attached_component_count": 0,
            "largest_surface_attached_component_voxel_count": 0,
        }

    surface_level_index = int(np.argmax(np.asarray(pressure_levels_hpa, dtype=np.float32)))
    filtered = np.zeros_like(mask, dtype=bool)
    kept_sizes: list[int] = []
    component_count = 0
    for label_id in range(1, int(labels.max()) + 1):
        component_mask = labels == label_id
        if not bool(np.any(component_mask[surface_level_index])):
            continue
        voxel_count = int(np.count_nonzero(component_mask))
        filtered |= component_mask
        kept_sizes.append(voxel_count)
        component_count += 1

    return filtered, {
        "surface_attached_component_count": int(component_count),
        "largest_surface_attached_component_voxel_count": int(
            max(kept_sizes) if kept_sizes else 0
        ),
    }


def maybe_flip_triangle_winding(indices: np.ndarray) -> np.ndarray:
    normalized = np.asarray(indices, dtype=np.uint32).copy()
    for index in range(0, normalized.size, 3):
        second = int(normalized[index + 1])
        normalized[index + 1] = normalized[index + 2]
        normalized[index + 2] = second
    return normalized


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_variant_dir(base_output_dir: Path, variant_name: str) -> Path:
    return base_output_dir.expanduser().resolve() / "variants" / variant_name


def build_variant_fields(
    recipe: VariantRecipe,
    *,
    temperature_field: np.ndarray,
    relative_humidity_field: np.ndarray,
    specific_humidity_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    thermal_raw = (
        np.asarray(temperature_field, dtype=np.float32)
        if recipe.thermal_field == THERMAL_TEMPERATURE
        else compute_dry_potential_temperature(temperature_field, pressure_levels_hpa)
    )
    moisture_raw = (
        np.asarray(relative_humidity_field, dtype=np.float32)
        if recipe.moisture_field == MOISTURE_RH
        else np.asarray(specific_humidity_field, dtype=np.float32)
    )
    thermal_anomaly, thermal_lat_mean = build_latitude_mean_anomaly(thermal_raw)
    moisture_anomaly, moisture_lat_mean = build_latitude_mean_anomaly(moisture_raw)
    thermal_z, thermal_scale = standardize_per_level(thermal_anomaly)
    moisture_z, moisture_scale = standardize_per_level(moisture_anomaly)
    return (
        smooth_levels(thermal_z, recipe.smoothing_sigma_cells),
        smooth_levels(moisture_z, recipe.smoothing_sigma_cells),
        {
            "thermal_latitude_band_mean": thermal_lat_mean,
            "moisture_latitude_band_mean": moisture_lat_mean,
            "thermal_scale": thermal_scale,
            "moisture_scale": moisture_scale,
        },
    )


def build_timestamp_payload(
    recipe: VariantRecipe,
    *,
    timestamp: str,
    temperature_field: np.ndarray,
    relative_humidity_field: np.ndarray,
    specific_humidity_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    thermal_z, moisture_z, field_metadata = build_variant_fields(
        recipe,
        temperature_field=temperature_field,
        relative_humidity_field=relative_humidity_field,
        specific_humidity_field=specific_humidity_field,
        pressure_levels_hpa=pressure_levels_hpa,
    )
    keep_mask, thresholds_by_level = build_score_mask(
        thermal_z,
        moisture_z,
        keep_top_percent=recipe.keep_top_percent,
        axis_min_abs_zscore=recipe.axis_min_abs_zscore,
    )
    raw_class_masks = classify_quadrants(keep_mask, thermal_z, moisture_z)

    class_masks: dict[str, np.ndarray] = {}
    class_meshes: dict[str, Any] = {}
    class_summaries: dict[str, Any] = {}
    total_voxel_count = 0
    total_component_count = 0
    all_bounds_mask = np.zeros_like(keep_mask, dtype=bool)

    for class_key in CLASS_ORDER:
        bridged = apply_bridge_gap(raw_class_masks[class_key], recipe.bridge_gap_levels)
        filtered, component_summary = filter_components(
            bridged,
            min_component_voxels=recipe.min_component_voxels,
            min_component_pressure_span_levels=recipe.min_component_pressure_span_levels,
        )
        surface_summary = {
            "surface_attached_component_count": int(component_summary["component_count"]),
            "largest_surface_attached_component_voxel_count": int(
                component_summary["largest_component_voxel_count"]
            ),
        }
        if recipe.surface_attached_only:
            filtered, surface_summary = filter_surface_attached_components(
                filtered,
                pressure_levels_hpa,
            )
            component_summary = {
                "component_count": int(surface_summary["surface_attached_component_count"]),
                "largest_component_voxel_count": int(
                    surface_summary["largest_surface_attached_component_voxel_count"]
                ),
            }
        voxel_count = int(np.count_nonzero(filtered))
        mesh = build_exposed_face_mesh_from_mask(
            keep_mask=filtered,
            pressure_levels_hpa=pressure_levels_hpa,
            latitudes_deg=latitudes_deg,
            longitudes_deg=longitudes_deg,
        )
        class_masks[class_key] = filtered
        class_meshes[class_key] = mesh
        class_summaries[class_key] = {
            "label": CLASS_PROXY_LABELS[class_key],
            "voxel_count": voxel_count,
            **component_summary,
            **surface_summary,
        }
        total_voxel_count += voxel_count
        total_component_count += int(component_summary["component_count"])
        all_bounds_mask |= filtered

    occupied_coords = np.argwhere(all_bounds_mask)
    if occupied_coords.size > 0:
        pressure_indices = occupied_coords[:, 0]
        latitude_indices = occupied_coords[:, 1]
        longitude_indices = occupied_coords[:, 2]
        bounds = {
            "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
            "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
            "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
            "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
            "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
            "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        }
    else:
        bounds = {
            "pressure_min_hpa": float(np.min(pressure_levels_hpa)),
            "pressure_max_hpa": float(np.max(pressure_levels_hpa)),
            "latitude_min_deg": float(np.min(latitudes_deg)),
            "latitude_max_deg": float(np.max(latitudes_deg)),
            "longitude_min_deg": float(np.min(longitudes_deg)),
            "longitude_max_deg": float(np.max(longitudes_deg)),
        }

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": timestamp,
        "voxel_count": int(total_voxel_count),
        "component_count": int(total_component_count),
        "class_summaries": class_summaries,
        "pressure_levels_hpa": [float(value) for value in pressure_levels_hpa.tolist()],
        "score_thresholds_by_pressure_level": [
            {
                "pressure_hpa": float(pressure_levels_hpa[level_index]),
                **thresholds_by_level[level_index],
            }
            for level_index in range(pressure_levels_hpa.size)
        ],
        "thermal_axis": {
            "field": recipe.thermal_field,
            "transform": "per-level_latitude-band_anomaly_zscore",
            "scale_by_pressure_level": [
                float(value) for value in field_metadata["thermal_scale"].tolist()
            ],
        },
        "moisture_axis": {
            "field": recipe.moisture_field,
            "transform": "per-level_latitude-band_anomaly_zscore",
            "scale_by_pressure_level": [
                float(value) for value in field_metadata["moisture_scale"].tolist()
            ],
        },
        "selection": {
            "keep_top_percent": float(recipe.keep_top_percent),
            "axis_min_abs_zscore": float(recipe.axis_min_abs_zscore),
            "score_basis": "per-level_geometric_mean_abs_zscore",
            "bridge_gap_levels": int(recipe.bridge_gap_levels),
            "min_component_voxels": int(recipe.min_component_voxels),
            "min_component_pressure_span_levels": int(
                recipe.min_component_pressure_span_levels
            ),
            "surface_attached_only": bool(recipe.surface_attached_only),
            "classes": [
                {
                    "key": class_key,
                    "label": CLASS_PROXY_LABELS[class_key],
                }
                for class_key in CLASS_ORDER
            ],
        },
        "smoothing_sigma_cells": float(recipe.smoothing_sigma_cells),
        **bounds,
    }

    entry = {
        "timestamp": timestamp,
        "metadata": f"{timestamp_to_slug(timestamp)}/metadata.json",
        "voxel_count": int(total_voxel_count),
        "component_count": int(total_component_count),
        "class_counts": {
            class_key: {
                "voxel_count": int(class_summaries[class_key]["voxel_count"]),
                "component_count": int(class_summaries[class_key]["component_count"]),
            }
            for class_key in CLASS_ORDER
        },
    }
    return entry, metadata, class_meshes, class_masks


def write_frame(
    output_dir: Path,
    *,
    timestamp: str,
    metadata: dict[str, Any],
    class_meshes: dict[str, Any],
) -> dict[str, Any]:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    class_summaries = dict(metadata["class_summaries"])
    for class_key in CLASS_ORDER:
        mesh = class_meshes[class_key]
        positions_path = frame_dir / f"{class_key}_positions.bin"
        indices_path = frame_dir / f"{class_key}_indices.bin"
        np.asarray(mesh.positions, dtype="<f4").tofile(positions_path)
        maybe_flip_triangle_winding(np.asarray(mesh.indices, dtype=np.uint32)).astype("<u4").tofile(
            indices_path
        )
        class_summaries[class_key] = {
            **class_summaries[class_key],
            "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
            "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
            "vertex_count": int(mesh.vertex_count),
            "index_count": int(mesh.indices.size),
        }

    metadata_path = frame_dir / "metadata.json"
    write_json(
        metadata_path,
        {
            **metadata,
            "class_summaries": class_summaries,
        },
    )
    return {
        "timestamp": timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "voxel_count": int(metadata["voxel_count"]),
        "component_count": int(metadata["component_count"]),
        "class_counts": {
            class_key: {
                "voxel_count": int(class_summaries[class_key]["voxel_count"]),
                "component_count": int(class_summaries[class_key]["component_count"]),
            }
            for class_key in CLASS_ORDER
        },
    }


def build_manifest(
    *,
    contents: DatasetContents,
    recipe: VariantRecipe,
    entries: list[dict[str, Any]],
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    base_radius: float,
    vertical_span: float,
    latitude_stride: int,
    longitude_stride: int,
) -> dict[str, Any]:
    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variant": recipe.name,
        "variant_label": recipe.label,
        "structure_kind": "air-mass-proxy-shells",
        "geometry_mode": "voxel-faces",
        "variables": contents.variable_names,
        "classification": {
            "thermal_axis_field": recipe.thermal_field,
            "moisture_axis_field": recipe.moisture_field,
            "thermal_transform": "per-level_latitude-band_anomaly_zscore",
            "moisture_transform": "per-level_latitude-band_anomaly_zscore",
            "score_basis": "per-level_geometric_mean_abs_zscore",
            "keep_top_percent": float(recipe.keep_top_percent),
            "axis_min_abs_zscore": float(recipe.axis_min_abs_zscore),
            "bridge_gap_levels": int(recipe.bridge_gap_levels),
            "min_component_voxels": int(recipe.min_component_voxels),
            "min_component_pressure_span_levels": int(recipe.min_component_pressure_span_levels),
            "surface_attached_only": bool(recipe.surface_attached_only),
            "classes": [
                {
                    "key": class_key,
                    "label": CLASS_PROXY_LABELS[class_key],
                }
                for class_key in CLASS_ORDER
            ],
        },
        "sampling": {
            "latitude_stride": int(max(latitude_stride, 1)),
            "longitude_stride": int(max(longitude_stride, 1)),
            "method": "subsample_centers",
        },
        "pressure_window_hpa": {
            "min": float(min(recipe.pressure_min_hpa, recipe.pressure_max_hpa)),
            "max": float(max(recipe.pressure_min_hpa, recipe.pressure_max_hpa)),
            "level_count": int(pressure_levels_hpa.size),
        },
        "globe": {
            "base_radius": float(base_radius),
            "vertical_span": float(vertical_span),
            "reference_pressure_hpa": {
                "min": 1.0,
                "max": 1000.0,
            },
        },
        "grid": {
            "pressure_level_count": int(pressure_levels_hpa.size),
            "latitude_count": int(latitudes_deg.size),
            "longitude_count": int(longitudes_deg.size),
            "latitude_step_degrees": coordinate_step_degrees(latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(longitudes_deg),
        },
        "timestamps": entries,
    }


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    contents = load_dataset_contents(dataset_path)
    recipe = VARIANT_RECIPES[args.variant]
    requested_timestamps = resolve_requested_timestamps(
        contents.timestamps,
        args.include_timestamps,
    )
    if not requested_timestamps:
        raise ValueError("No requested timestamps matched the dataset.")

    output_dir = build_variant_dir(args.output_dir, recipe.name)
    clear_output_dir(output_dir)

    entries: list[dict[str, Any]] = []
    with xr.open_dataset(dataset_path) as dataset:
        for timestamp in requested_timestamps:
            temperature_field = reorder_longitude_axis(
                load_field_at_timestamp(
                    dataset,
                    variable_name=contents.variable_names["temperature"],
                    timestamp=timestamp,
                ),
                contents.longitude_order,
            )
            relative_humidity_field = reorder_longitude_axis(
                load_field_at_timestamp(
                    dataset,
                    variable_name=contents.variable_names["relative_humidity"],
                    timestamp=timestamp,
                ),
                contents.longitude_order,
            )
            specific_humidity_field = reorder_longitude_axis(
                load_field_at_timestamp(
                    dataset,
                    variable_name=contents.variable_names["specific_humidity"],
                    timestamp=timestamp,
                ),
                contents.longitude_order,
            )

            temperature_window, pressure_levels_hpa = select_pressure_window(
                temperature_field,
                contents.pressure_levels_hpa,
                recipe.pressure_min_hpa,
                recipe.pressure_max_hpa,
            )
            relative_humidity_window, _ = select_pressure_window(
                relative_humidity_field,
                contents.pressure_levels_hpa,
                recipe.pressure_min_hpa,
                recipe.pressure_max_hpa,
            )
            specific_humidity_window, _ = select_pressure_window(
                specific_humidity_field,
                contents.pressure_levels_hpa,
                recipe.pressure_min_hpa,
                recipe.pressure_max_hpa,
            )

            temperature_window, latitudes_deg, longitudes_deg = stride_spatial_axes(
                temperature_window,
                contents.latitudes_deg,
                contents.longitudes_deg,
                args.latitude_stride,
                args.longitude_stride,
            )
            relative_humidity_window, _, _ = stride_spatial_axes(
                relative_humidity_window,
                contents.latitudes_deg,
                contents.longitudes_deg,
                args.latitude_stride,
                args.longitude_stride,
            )
            specific_humidity_window, _, _ = stride_spatial_axes(
                specific_humidity_window,
                contents.latitudes_deg,
                contents.longitudes_deg,
                args.latitude_stride,
                args.longitude_stride,
            )

            entry, metadata, class_meshes, _ = build_timestamp_payload(
                recipe,
                timestamp=timestamp,
                temperature_field=temperature_window,
                relative_humidity_field=relative_humidity_window,
                specific_humidity_field=specific_humidity_window,
                pressure_levels_hpa=pressure_levels_hpa,
                latitudes_deg=latitudes_deg,
                longitudes_deg=longitudes_deg,
            )
            entries.append(
                write_frame(
                    output_dir,
                    timestamp=timestamp,
                    metadata=metadata,
                    class_meshes=class_meshes,
                )
            )
            print(
                "Built air-mass proxy shell:",
                timestamp,
                f"variant={recipe.name}",
                f"voxels={entry['voxel_count']}",
                f"components={entry['component_count']}",
            )

    write_json(
        output_dir / "index.json",
        build_manifest(
            contents=contents,
            recipe=recipe,
            entries=entries,
            pressure_levels_hpa=pressure_levels_hpa,
            latitudes_deg=latitudes_deg,
            longitudes_deg=longitudes_deg,
            base_radius=args.base_radius,
            vertical_span=args.vertical_span,
            latitude_stride=args.latitude_stride,
            longitude_stride=args.longitude_stride,
        ),
    )
    print(
        "Built air-mass structures:",
        format_display_path(output_dir),
        f"frames={len(entries)}",
        f"variant={recipe.name}",
    )


if __name__ == "__main__":
    main()
