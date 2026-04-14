from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from scipy import ndimage
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_voxel_builder import (
    build_exposed_face_mesh_from_mask,
    coordinate_step_degrees,
    timestamp_to_slug,
)


TEMPERATURE_VARIABLE = "t"
DERIVED_VARIABLE_NAME = "dry_potential_temperature"
DERIVED_UNITS = "K"
REFERENCE_PRESSURE_HPA = 1000.0
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
VOLUME_COMPONENT_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
OUTPUT_VERSION = 7
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/potential-temperature-structures")
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0
DEFAULT_ABSOLUTE_ANOMALY_PERCENTILE = 50.0
DEFAULT_SMOOTHING_SIGMA_CELLS = 1.0


@dataclass(frozen=True)
class DatasetContents:
    dataset_path: Path
    units: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    longitude_order: np.ndarray
    timestamps: list[str]


@dataclass(frozen=True)
class SignedAnomalyShellMesh:
    positions: np.ndarray
    indices: np.ndarray
    anomaly_values: np.ndarray
    theta_values: np.ndarray
    voxel_count: int


@dataclass(frozen=True)
class SignedAnomalyShellAssetPayload:
    timestamp: str
    warm_positions: np.ndarray
    warm_indices: np.ndarray
    cold_positions: np.ndarray
    cold_indices: np.ndarray
    voxel_count: int
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sign-aware dry-potential-temperature anomaly voxel shells from "
            "ERA5 pressure-level temperature. The builder derives dry potential "
            "temperature, subtracts a per-level latitude-band mean, keeps the top "
            "absolute-anomaly percentile on each pressure level, lightly smooths "
            "that signed anomaly field, and exports warm/cold 3D voxel-face "
            "structures with per-vertex anomaly magnitude and theta values."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 pressure-level temperature NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated potential-temperature assets will be written.",
    )
    parser.add_argument(
        "--absolute-anomaly-percentile",
        type=float,
        default=DEFAULT_ABSOLUTE_ANOMALY_PERCENTILE,
        help=(
            "Keep voxels at or above this per-level percentile of absolute anomaly "
            "before smoothing."
        ),
    )
    parser.add_argument(
        "--smoothing-sigma-cells",
        type=float,
        default=DEFAULT_SMOOTHING_SIGMA_CELLS,
        help=(
            "Horizontal Gaussian smoothing sigma, in strided grid cells, applied "
            "after the per-level absolute-anomaly selection."
        ),
    )
    parser.add_argument(
        "--pressure-min-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MIN_HPA,
        help="Lowest pressure level to keep in the shell.",
    )
    parser.add_argument(
        "--pressure-max-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MAX_HPA,
        help="Highest pressure level to keep in the shell.",
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
        help="Keep every Nth latitude sample when building the voxel shell.",
    )
    parser.add_argument(
        "--longitude-stride",
        type=int,
        default=DEFAULT_LONGITUDE_STRIDE,
        help="Keep every Nth longitude sample when building the voxel shell.",
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
        raise FileNotFoundError(
            f"Dataset file not found: {format_display_path(resolved)}"
        )
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
    return np.take(np.asarray(field, dtype=np.float32), longitude_order, axis=-1)


def load_dataset_contents(dataset_path: Path) -> DatasetContents:
    dataset = xr.open_dataset(dataset_path)
    try:
        variable = dataset[TEMPERATURE_VARIABLE]
        longitudes_deg, longitude_order = normalize_longitudes_with_order(
            np.asarray(variable.coords["longitude"].values, dtype=np.float32)
        )
        return DatasetContents(
            dataset_path=dataset_path,
            units=str(variable.attrs.get("units", "")),
            pressure_levels_hpa=np.asarray(
                variable.coords["pressure_level"].values,
                dtype=np.float32,
            ),
            latitudes_deg=np.asarray(variable.coords["latitude"].values, dtype=np.float32),
            longitudes_deg=longitudes_deg,
            longitude_order=longitude_order,
            timestamps=[
                timestamp_to_iso_minute(value)
                for value in np.asarray(variable.coords["valid_time"].values)
            ],
        )
    finally:
        dataset.close()


def resolve_target_timestamps(
    all_timestamps: list[str],
    include_timestamps_text: str,
) -> list[str]:
    requested = {
        value.strip()
        for value in include_timestamps_text.split(",")
        if value.strip()
    }
    if not requested:
        return list(all_timestamps)
    return [timestamp for timestamp in all_timestamps if timestamp in requested]


def compute_dry_potential_temperature(
    temperature_values: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> np.ndarray:
    temperature = np.asarray(temperature_values, dtype=np.float32)
    pressure_levels = np.asarray(pressure_levels_hpa, dtype=np.float32)
    pressure_scale = np.power(
        REFERENCE_PRESSURE_HPA / pressure_levels[:, None, None],
        POTENTIAL_TEMPERATURE_KAPPA,
        dtype=np.float32,
    )
    return np.asarray(temperature * pressure_scale, dtype=np.float32)


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


def select_pressure_window(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = min(float(pressure_min_hpa), float(pressure_max_hpa))
    upper = max(float(pressure_min_hpa), float(pressure_max_hpa))
    keep = np.asarray(
        (pressure_levels_hpa >= lower) & (pressure_levels_hpa <= upper),
        dtype=bool,
    )
    if not keep.any():
        raise ValueError(f"No pressure levels fall within [{lower}, {upper}] hPa.")
    return np.asarray(field[keep], dtype=np.float32), np.asarray(
        pressure_levels_hpa[keep],
        dtype=np.float32,
    )


def build_latitude_mean_anomaly(theta_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latitude_band_mean = np.nanmean(theta_field, axis=2, keepdims=True)
    anomaly = np.asarray(theta_field - latitude_band_mean, dtype=np.float32)
    return anomaly, np.asarray(latitude_band_mean, dtype=np.float32)


def build_percentile_selected_anomaly(
    anomaly: np.ndarray,
    absolute_anomaly_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    percentile = float(absolute_anomaly_percentile)
    if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise ValueError(
            "absolute_anomaly_percentile must be a finite value between 0 and 100."
        )

    absolute_anomaly = np.asarray(np.abs(anomaly), dtype=np.float32)
    thresholds = np.asarray(
        np.percentile(absolute_anomaly, percentile, axis=(1, 2)),
        dtype=np.float32,
    )
    keep_mask = np.asarray(
        np.isfinite(anomaly) & (absolute_anomaly >= thresholds[:, None, None]),
        dtype=bool,
    )
    selected_anomaly = np.where(keep_mask, anomaly, 0.0).astype(np.float32)
    return selected_anomaly, keep_mask, thresholds


def smooth_selected_sign_anomaly(
    selected_anomaly: np.ndarray,
    smoothing_sigma_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(float(smoothing_sigma_cells), 0.0)
    if sigma <= 0.0:
        anomaly_values = np.asarray(selected_anomaly, dtype=np.float32)
        sign_field = np.sign(anomaly_values).astype(np.int8)
        return anomaly_values, sign_field

    filter_sigma = (0.0, sigma, sigma)
    filter_mode = ("nearest", "nearest", "wrap")
    filter_kwargs = {"sigma": filter_sigma, "mode": filter_mode, "truncate": 2.0}

    positive_presence = ndimage.gaussian_filter(
        np.asarray(selected_anomaly > 0.0, dtype=np.float32),
        **filter_kwargs,
    )
    negative_presence = ndimage.gaussian_filter(
        np.asarray(selected_anomaly < 0.0, dtype=np.float32),
        **filter_kwargs,
    )
    positive_magnitude = ndimage.gaussian_filter(
        np.maximum(selected_anomaly, 0.0).astype(np.float32),
        **filter_kwargs,
    )
    negative_magnitude = ndimage.gaussian_filter(
        np.maximum(-selected_anomaly, 0.0).astype(np.float32),
        **filter_kwargs,
    )

    positive_keep = np.asarray(
        (positive_presence >= 0.5) & (positive_presence > negative_presence),
        dtype=bool,
    )
    negative_keep = np.asarray(
        (negative_presence >= 0.5) & (negative_presence > positive_presence),
        dtype=bool,
    )

    sign_field = np.zeros_like(selected_anomaly, dtype=np.int8)
    sign_field[positive_keep] = 1
    sign_field[negative_keep] = -1

    anomaly_values = np.zeros_like(selected_anomaly, dtype=np.float32)
    anomaly_values[positive_keep] = positive_magnitude[positive_keep]
    anomaly_values[negative_keep] = -negative_magnitude[negative_keep]
    return anomaly_values, sign_field


def build_seam_merged_component_info(
    labels: np.ndarray,
    seam_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component_count = int(labels.max())
    if component_count <= 0:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)

    parent = np.arange(component_count + 1, dtype=np.int32)

    def find(label_id: int) -> int:
        root = label_id
        while parent[root] != root:
            root = int(parent[root])
        while parent[label_id] != label_id:
            next_label = int(parent[label_id])
            parent[label_id] = root
            label_id = next_label
        return root

    def union(a: int, b: int) -> None:
        if a <= 0 or b <= 0:
            return
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    for first_label, duplicate_label in seam_pairs:
        union(int(first_label), int(duplicate_label))

    root_map = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        root_map[label_id] = find(label_id)

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
    if unique_root_ids.size == 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_root_ids] = np.arange(1, unique_root_ids.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[..., :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_root_ids.size)


def summarize_signed_components(sign_field: np.ndarray) -> dict[str, Any]:
    positive_labels, positive_component_count = label_wrapped_volume_components(sign_field > 0)
    negative_labels, negative_component_count = label_wrapped_volume_components(sign_field < 0)

    positive_sizes = (
        np.bincount(positive_labels[positive_labels > 0].ravel())
        if positive_component_count > 0
        else np.zeros(0, dtype=np.int32)
    )
    negative_sizes = (
        np.bincount(negative_labels[negative_labels > 0].ravel())
        if negative_component_count > 0
        else np.zeros(0, dtype=np.int32)
    )

    positive_largest = int(positive_sizes.max()) if positive_sizes.size else 0
    negative_largest = int(negative_sizes.max()) if negative_sizes.size else 0
    return {
        "component_count": int(positive_component_count + negative_component_count),
        "positive_component_count": int(positive_component_count),
        "negative_component_count": int(negative_component_count),
        "largest_component_voxel_count": max(positive_largest, negative_largest),
        "largest_positive_component_voxel_count": positive_largest,
        "largest_negative_component_voxel_count": negative_largest,
    }


def build_asset_payload(
    *,
    timestamp: str,
    theta_field: np.ndarray,
    selected_anomaly: np.ndarray,
    smoothed_anomaly_values: np.ndarray,
    sign_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    absolute_anomaly_percentile: float,
    smoothing_sigma_cells: float,
    anomaly_thresholds_by_level: np.ndarray,
    base_radius: float,
    vertical_span: float,
) -> SignedAnomalyShellAssetPayload:
    warm_mask = np.asarray(sign_field > 0, dtype=bool)
    cold_mask = np.asarray(sign_field < 0, dtype=bool)
    warm_mesh = build_exposed_face_mesh_from_mask(
        keep_mask=warm_mask,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
    )
    cold_mesh = build_exposed_face_mesh_from_mask(
        keep_mask=cold_mask,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
    )
    total_voxel_count = int(np.count_nonzero(sign_field))
    if total_voxel_count <= 0:
        raise ValueError("No voxels survived the potential-temperature anomaly selection.")

    keep_mask = np.asarray(sign_field != 0, dtype=bool)
    occupied_coords = np.argwhere(keep_mask)
    pressure_indices = occupied_coords[:, 0]
    latitude_indices = occupied_coords[:, 1]
    longitude_indices = occupied_coords[:, 2]
    kept_theta = np.asarray(theta_field[keep_mask], dtype=np.float32)
    kept_anomaly = np.asarray(smoothed_anomaly_values[keep_mask], dtype=np.float32)
    component_summary = summarize_signed_components(sign_field)

    metadata = {
        **component_summary,
        "voxel_count": total_voxel_count,
        "positive_voxel_count": int(np.count_nonzero(sign_field > 0)),
        "negative_voxel_count": int(np.count_nonzero(sign_field < 0)),
        "selected_voxel_count_before_smoothing": int(np.count_nonzero(selected_anomaly)),
        "vertex_count": int((warm_mesh.positions.size + cold_mesh.positions.size) // 3),
        "index_count": int(warm_mesh.indices.size + cold_mesh.indices.size),
        "warm_vertex_count": int(warm_mesh.vertex_count),
        "warm_index_count": int(warm_mesh.indices.size),
        "cold_vertex_count": int(cold_mesh.vertex_count),
        "cold_index_count": int(cold_mesh.indices.size),
        "theta_min": float(np.min(kept_theta)),
        "theta_max": float(np.max(kept_theta)),
        "theta_mean": float(np.mean(kept_theta)),
        "anomaly_min": float(np.min(kept_anomaly)),
        "anomaly_max": float(np.max(kept_anomaly)),
        "anomaly_mean": float(np.mean(kept_anomaly)),
        "anomaly_abs_max": float(np.max(np.abs(kept_anomaly))),
        "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
        "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
        "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        "absolute_anomaly_percentile": float(absolute_anomaly_percentile),
        "smoothing_sigma_cells": float(smoothing_sigma_cells),
        "selection": {
            "kept_signs": ["negative", "positive"],
            "thresholds_by_pressure_level": [
                {
                    "pressure_hpa": float(pressure_hpa),
                    "absolute_anomaly_threshold": float(threshold),
                }
                for pressure_hpa, threshold in zip(
                    pressure_levels_hpa.tolist(),
                    anomaly_thresholds_by_level.tolist(),
                    strict=True,
                )
            ],
        },
    }

    return SignedAnomalyShellAssetPayload(
        timestamp=timestamp,
        warm_positions=warm_mesh.positions,
        warm_indices=warm_mesh.indices,
        cold_positions=cold_mesh.positions,
        cold_indices=cold_mesh.indices,
        voxel_count=total_voxel_count,
        metadata=metadata,
    )


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


def write_frame(
    output_dir: Path,
    payload: SignedAnomalyShellAssetPayload,
) -> dict[str, Any]:
    slug = timestamp_to_slug(payload.timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    warm_positions_path = frame_dir / "warm_positions.bin"
    warm_indices_path = frame_dir / "warm_indices.bin"
    cold_positions_path = frame_dir / "cold_positions.bin"
    cold_indices_path = frame_dir / "cold_indices.bin"
    metadata_path = frame_dir / "metadata.json"

    payload.warm_positions.astype("<f4").tofile(warm_positions_path)
    payload.warm_indices.astype("<u4").tofile(warm_indices_path)
    payload.cold_positions.astype("<f4").tofile(cold_positions_path)
    payload.cold_indices.astype("<u4").tofile(cold_indices_path)

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": payload.timestamp,
        **payload.metadata,
        "warm_positions_file": str(warm_positions_path.relative_to(output_dir)).replace("\\", "/"),
        "warm_indices_file": str(warm_indices_path.relative_to(output_dir)).replace("\\", "/"),
        "cold_positions_file": str(cold_positions_path.relative_to(output_dir)).replace("\\", "/"),
        "cold_indices_file": str(cold_indices_path.relative_to(output_dir)).replace("\\", "/"),
    }
    write_json(metadata_path, metadata)

    return {
        "timestamp": payload.timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "warm_positions": str(warm_positions_path.relative_to(output_dir)).replace("\\", "/"),
        "warm_indices": str(warm_indices_path.relative_to(output_dir)).replace("\\", "/"),
        "cold_positions": str(cold_positions_path.relative_to(output_dir)).replace("\\", "/"),
        "cold_indices": str(cold_indices_path.relative_to(output_dir)).replace("\\", "/"),
        "voxel_count": payload.voxel_count,
        "component_count": int(payload.metadata["component_count"]),
        "positive_component_count": int(payload.metadata["positive_component_count"]),
        "negative_component_count": int(payload.metadata["negative_component_count"]),
    }


def build_manifest(
    *,
    contents: DatasetContents,
    entries: list[dict[str, Any]],
    absolute_anomaly_percentile: float,
    smoothing_sigma_cells: float,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
    latitude_stride: int,
    longitude_stride: int,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    base_radius: float,
    vertical_span: float,
) -> dict[str, Any]:
    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": TEMPERATURE_VARIABLE,
        "units": contents.units,
        "derived_variable": {
            "name": DERIVED_VARIABLE_NAME,
            "units": DERIVED_UNITS,
            "reference_pressure_hpa": REFERENCE_PRESSURE_HPA,
            "kappa": POTENTIAL_TEMPERATURE_KAPPA,
        },
        "structure_kind": "potential-temperature-latitude-mean-anomaly-shell",
        "geometry_mode": "voxel-faces",
        "selection": {
            "background": "per-level_latitude-band_mean",
            "threshold_basis": "per-level_absolute-anomaly_percentile",
            "absolute_anomaly_percentile": float(absolute_anomaly_percentile),
            "smoothing_sigma_cells": float(smoothing_sigma_cells),
            "keep_signs": ["negative", "positive"],
            "volume_connectivity": "26-connected-same-sign",
            "wraps_longitude": True,
        },
        "sampling": {
            "latitude_stride": int(max(latitude_stride, 1)),
            "longitude_stride": int(max(longitude_stride, 1)),
            "method": "subsample_centers",
        },
        "pressure_window_hpa": {
            "min": float(min(pressure_min_hpa, pressure_max_hpa)),
            "max": float(max(pressure_min_hpa, pressure_max_hpa)),
            "level_count": int(pressure_levels_hpa.size),
        },
        "globe": {
            "base_radius": base_radius,
            "vertical_span": vertical_span,
            "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
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
    target_timestamps = resolve_target_timestamps(
        contents.timestamps,
        args.include_timestamps,
    )
    if not target_timestamps:
        raise ValueError("No matching timestamps were selected for export.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    raw_dataset = netCDF4.Dataset(dataset_path)
    try:
        variable = raw_dataset.variables[TEMPERATURE_VARIABLE]
        entries: list[dict[str, Any]] = []
        final_latitudes_deg: np.ndarray | None = None
        final_longitudes_deg: np.ndarray | None = None
        final_pressure_levels_hpa: np.ndarray | None = None

        for time_index, timestamp in enumerate(contents.timestamps):
            if timestamp not in target_timestamps:
                continue

            temperature_field = reorder_longitude_axis(
                np.asarray(variable[time_index, :, :, :], dtype=np.float32),
                contents.longitude_order,
            )
            temperature_field, strided_latitudes_deg, strided_longitudes_deg = stride_spatial_axes(
                temperature_field,
                contents.latitudes_deg,
                contents.longitudes_deg,
                latitude_stride=args.latitude_stride,
                longitude_stride=args.longitude_stride,
            )
            pressure_window_field, pressure_window_levels_hpa = select_pressure_window(
                temperature_field,
                contents.pressure_levels_hpa,
                pressure_min_hpa=args.pressure_min_hpa,
                pressure_max_hpa=args.pressure_max_hpa,
            )
            theta_field = compute_dry_potential_temperature(
                pressure_window_field,
                pressure_window_levels_hpa,
            )
            anomaly, _ = build_latitude_mean_anomaly(theta_field)
            selected_anomaly, _, anomaly_thresholds_by_level = build_percentile_selected_anomaly(
                anomaly,
                absolute_anomaly_percentile=args.absolute_anomaly_percentile,
            )
            smoothed_anomaly_values, sign_field = smooth_selected_sign_anomaly(
                selected_anomaly,
                smoothing_sigma_cells=args.smoothing_sigma_cells,
            )
            if not np.any(sign_field):
                print(
                    "Skipped potential temperature frame after smoothing:",
                    timestamp,
                    f"absolute_anomaly_percentile={args.absolute_anomaly_percentile}",
                    f"smoothing_sigma_cells={args.smoothing_sigma_cells}",
                )
                continue

            payload = build_asset_payload(
                timestamp=timestamp,
                theta_field=theta_field,
                selected_anomaly=selected_anomaly,
                smoothed_anomaly_values=smoothed_anomaly_values,
                sign_field=sign_field,
                pressure_levels_hpa=pressure_window_levels_hpa,
                latitudes_deg=strided_latitudes_deg,
                longitudes_deg=strided_longitudes_deg,
                absolute_anomaly_percentile=args.absolute_anomaly_percentile,
                smoothing_sigma_cells=args.smoothing_sigma_cells,
                anomaly_thresholds_by_level=anomaly_thresholds_by_level,
                base_radius=args.base_radius,
                vertical_span=args.vertical_span,
            )
            entries.append(write_frame(output_dir=output_dir, payload=payload))
            final_latitudes_deg = strided_latitudes_deg
            final_longitudes_deg = strided_longitudes_deg
            final_pressure_levels_hpa = pressure_window_levels_hpa

            print(
                "Built potential temperature anomaly shell:",
                timestamp,
                f"voxels={payload.voxel_count}",
                f"components={payload.metadata['component_count']}",
                f"positive_components={payload.metadata['positive_component_count']}",
                f"negative_components={payload.metadata['negative_component_count']}",
                f"anomaly_abs_max={payload.metadata['anomaly_abs_max']:.2f}",
                f"warm_triangles={payload.warm_indices.size // 3}",
                f"cold_triangles={payload.cold_indices.size // 3}",
            )
    finally:
        raw_dataset.close()

    if not entries or final_latitudes_deg is None or final_longitudes_deg is None or final_pressure_levels_hpa is None:
        raise ValueError("No potential-temperature frames were written.")

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        absolute_anomaly_percentile=args.absolute_anomaly_percentile,
        smoothing_sigma_cells=args.smoothing_sigma_cells,
        pressure_min_hpa=args.pressure_min_hpa,
        pressure_max_hpa=args.pressure_max_hpa,
        latitude_stride=args.latitude_stride,
        longitude_stride=args.longitude_stride,
        latitudes_deg=final_latitudes_deg,
        longitudes_deg=final_longitudes_deg,
        pressure_levels_hpa=final_pressure_levels_hpa,
        base_radius=args.base_radius,
        vertical_span=args.vertical_span,
    )
    write_json(output_dir / "index.json", manifest)
    print(
        "Built potential temperature structures:",
        f"{len(entries)} timestamps",
        f"-> {format_display_path(output_dir)}",
    )


if __name__ == "__main__":
    main()
