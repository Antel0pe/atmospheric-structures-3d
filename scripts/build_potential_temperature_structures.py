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
    build_axis_bounds,
    build_radius_lookup_from_pressure_levels,
    coordinate_step_degrees,
    timestamp_to_slug,
)


TEMPERATURE_VARIABLE = "t"
DERIVED_VARIABLE_NAME = "dry_potential_temperature"
DERIVED_UNITS = "K"
REFERENCE_PRESSURE_HPA = 1000.0
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
LEVEL_COMPONENT_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
VOLUME_COMPONENT_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
OUTPUT_VERSION = 6
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/potential-temperature-structures")
DEFAULT_Z_THRESHOLD_SIGMA = 1.0
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4
DEFAULT_MIN_LEVEL_COMPONENT_SIZE = 32
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0


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
class ColdShellMesh:
    positions: np.ndarray
    indices: np.ndarray
    coldness_sigma: np.ndarray
    voxel_count: int


@dataclass(frozen=True)
class ColdShellAssetPayload:
    timestamp: str
    positions: np.ndarray
    indices: np.ndarray
    coldness_sigma: np.ndarray
    voxel_count: int
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cold dry-potential-temperature anomaly voxel shells from ERA5 "
            "pressure-level temperature. The builder derives dry potential "
            "temperature, subtracts a per-level zonal mean background, "
            "standardizes the anomaly by level, keeps only cold anomalies at "
            "or below a sigma threshold, removes small 2D level components, "
            "and exports one 3D voxel-face shell colored by coldness."
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
        "--z-threshold-sigma",
        type=float,
        default=DEFAULT_Z_THRESHOLD_SIGMA,
        help="Keep voxels whose standardized cold anomaly is at or above this sigma value.",
    )
    parser.add_argument(
        "--min-level-component-size",
        type=int,
        default=DEFAULT_MIN_LEVEL_COMPONENT_SIZE,
        help=(
            "Drop 2D connected components smaller than this many strided grid cells "
            "on each pressure level before building the 3D shell."
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
        raise ValueError(
            f"No pressure levels fall within [{lower}, {upper}] hPa."
        )
    return np.asarray(field[keep], dtype=np.float32), np.asarray(pressure_levels_hpa[keep], dtype=np.float32)


def build_standardized_coldness_sigma(
    theta_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zonal_mean = np.nanmean(theta_field, axis=2, keepdims=True)
    anomaly = np.asarray(theta_field - zonal_mean, dtype=np.float32)
    level_std = np.nanstd(anomaly, axis=(1, 2), keepdims=True)
    safe_std = np.where(
        np.isfinite(level_std) & (level_std > 1e-6),
        level_std,
        1.0,
    ).astype(np.float32)
    z_score = np.asarray(anomaly / safe_std, dtype=np.float32)
    coldness_sigma = np.asarray(np.maximum(-z_score, 0.0), dtype=np.float32)
    return z_score, coldness_sigma, safe_std[:, 0, 0]


def build_cold_mask(
    coldness_sigma: np.ndarray,
    z_threshold_sigma: float,
) -> np.ndarray:
    normalized_threshold = float(z_threshold_sigma)
    if not np.isfinite(normalized_threshold) or normalized_threshold <= 0.0:
        raise ValueError(
            f"z_threshold_sigma must be a finite value greater than 0, got {z_threshold_sigma}"
        )
    return np.asarray(
        np.isfinite(coldness_sigma) & (coldness_sigma >= normalized_threshold),
        dtype=bool,
    )


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


def filter_small_wrapped_level_components(
    keep_mask: np.ndarray,
    min_component_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_min_component_size = max(int(min_component_size), 0)
    if normalized_min_component_size <= 0:
        return np.asarray(keep_mask, dtype=bool), {
            "minimum_component_size": 0,
            "connectivity": "8-connected-within-level",
            "wraps_longitude": True,
            "component_count_before_filter": 0,
            "component_count_after_filter": 0,
            "removed_component_count": 0,
            "removed_voxel_count": 0,
        }

    filtered_mask = np.zeros_like(keep_mask, dtype=bool)
    component_count_before = 0
    component_count_after = 0

    for level_index in range(keep_mask.shape[0]):
        layer_mask = np.asarray(keep_mask[level_index], dtype=bool)
        if not layer_mask.any():
            continue

        longitude_count = layer_mask.shape[1]
        extended = np.concatenate([layer_mask, layer_mask[:, :1]], axis=1)
        labels, component_count = ndimage.label(extended, structure=LEVEL_COMPONENT_STRUCTURE)
        if component_count <= 0:
            continue

        seam_pairs = np.column_stack([labels[:, 0].reshape(-1), labels[:, -1].reshape(-1)])
        root_map, unique_root_ids = build_seam_merged_component_info(labels, seam_pairs)
        component_count_before += int(unique_root_ids.size)

        label_ids = np.arange(1, component_count + 1, dtype=np.int32)
        extended_counts = np.asarray(
            ndimage.sum(
                np.ones_like(labels, dtype=np.int32),
                labels=labels,
                index=label_ids,
            ),
            dtype=np.int32,
        )
        root_counts = np.zeros(component_count + 1, dtype=np.int32)
        np.add.at(root_counts, root_map[label_ids], extended_counts)

        seam_duplicate_labels = labels[:, -1][layer_mask[:, 0]]
        seam_duplicate_counts = np.zeros(component_count + 1, dtype=np.int32)
        if seam_duplicate_labels.size:
            np.add.at(seam_duplicate_counts, root_map[seam_duplicate_labels], 1)

        unique_counts = root_counts - seam_duplicate_counts
        kept_root_ids = unique_root_ids[
            unique_counts[unique_root_ids] >= normalized_min_component_size
        ]
        filtered_layer = np.isin(root_map[labels[:, :longitude_count]], kept_root_ids)
        filtered_mask[level_index] = filtered_layer
        component_count_after += int(kept_root_ids.size)

    removed_voxel_count = int(keep_mask.sum() - filtered_mask.sum())
    return filtered_mask, {
        "minimum_component_size": normalized_min_component_size,
        "connectivity": "8-connected-within-level",
        "wraps_longitude": True,
        "component_count_before_filter": int(component_count_before),
        "component_count_after_filter": int(component_count_after),
        "removed_component_count": int(component_count_before - component_count_after),
        "removed_voxel_count": removed_voxel_count,
    }


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


def lat_lon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(-(lon_deg + 270.0))
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.sin(lat)
    z = radius * np.cos(lat) * np.sin(lon)
    return np.array([x, y, z], dtype=np.float32)


def append_colored_quad(
    corners: list[tuple[int, int, int]],
    voxel_coldness_sigma: float,
    *,
    vertex_lookup: dict[tuple[int, int, int], int],
    positions: list[float],
    coldness_values: list[float],
    indices: list[int],
    radius_bounds: np.ndarray,
    latitude_bounds: np.ndarray,
    longitude_bounds: np.ndarray,
) -> None:
    quad_indices: list[int] = []
    for key in corners:
        vertex_index = vertex_lookup.get(key)
        if vertex_index is None:
            radius_index, latitude_index, longitude_index = key
            vertex = lat_lon_to_xyz(
                float(latitude_bounds[latitude_index]),
                float(longitude_bounds[longitude_index]),
                float(radius_bounds[radius_index]),
            )
            vertex_index = len(positions) // 3
            positions.extend(float(value) for value in vertex)
            coldness_values.append(float(voxel_coldness_sigma))
            vertex_lookup[key] = vertex_index
        else:
            coldness_values[vertex_index] = max(
                coldness_values[vertex_index],
                float(voxel_coldness_sigma),
            )
        quad_indices.append(vertex_index)

    indices.extend(
        [
            quad_indices[0],
            quad_indices[1],
            quad_indices[2],
            quad_indices[0],
            quad_indices[2],
            quad_indices[3],
        ]
    )


def build_exposed_face_mesh_with_coldness(
    *,
    keep_mask: np.ndarray,
    coldness_sigma: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    base_radius: float,
    vertical_span: float,
) -> ColdShellMesh:
    occupied = np.asarray(keep_mask, dtype=bool)
    if not occupied.any():
        return ColdShellMesh(
            positions=np.zeros(0, dtype=np.float32),
            indices=np.zeros(0, dtype=np.uint32),
            coldness_sigma=np.zeros(0, dtype=np.float32),
            voxel_count=0,
        )

    radius_values = build_radius_lookup_from_pressure_levels(
        pressure_levels_hpa,
        base_radius=base_radius,
        vertical_span=vertical_span,
    )
    radius_bounds = build_axis_bounds(radius_values)
    latitude_bounds = build_axis_bounds(latitudes_deg)
    longitude_bounds = build_axis_bounds(longitudes_deg)

    level_count, latitude_count, longitude_count = occupied.shape
    occupied_cells = np.argwhere(occupied)
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    positions: list[float] = []
    coldness_values: list[float] = []
    indices: list[int] = []

    for level_index, latitude_index, longitude_index in occupied_cells:
        r0 = int(level_index)
        r1 = r0 + 1
        a0 = int(latitude_index)
        a1 = a0 + 1
        o0 = int(longitude_index)
        o1 = o0 + 1
        voxel_coldness_sigma = float(coldness_sigma[r0, a0, o0])

        west_neighbor = (o0 - 1) % longitude_count
        east_neighbor = (o0 + 1) % longitude_count

        if r0 == 0 or not occupied[r0 - 1, a0, o0]:
            append_colored_quad(
                corners=[(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if r0 == level_count - 1 or not occupied[r0 + 1, a0, o0]:
            append_colored_quad(
                corners=[(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == 0 or not occupied[r0, a0 - 1, o0]:
            append_colored_quad(
                corners=[(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == latitude_count - 1 or not occupied[r0, a0 + 1, o0]:
            append_colored_quad(
                corners=[(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not occupied[r0, a0, west_neighbor]:
            append_colored_quad(
                corners=[(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not occupied[r0, a0, east_neighbor]:
            append_colored_quad(
                corners=[(r0, a0, o1), (r0, a1, o1), (r1, a1, o1), (r1, a0, o1)],
                voxel_coldness_sigma=voxel_coldness_sigma,
                vertex_lookup=vertex_lookup,
                positions=positions,
                coldness_values=coldness_values,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )

    return ColdShellMesh(
        positions=np.asarray(positions, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        coldness_sigma=np.asarray(coldness_values, dtype=np.float32),
        voxel_count=int(occupied_cells.shape[0]),
    )


def build_asset_payload(
    *,
    timestamp: str,
    theta_field: np.ndarray,
    coldness_sigma: np.ndarray,
    keep_mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    z_threshold_sigma: float,
    base_radius: float,
    vertical_span: float,
    selection_metadata: dict[str, Any],
) -> ColdShellAssetPayload:
    mesh = build_exposed_face_mesh_with_coldness(
        keep_mask=keep_mask,
        coldness_sigma=coldness_sigma,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        base_radius=base_radius,
        vertical_span=vertical_span,
    )
    if mesh.voxel_count <= 0:
        raise ValueError("No voxels survived the potential-temperature anomaly selection.")

    occupied_coords = np.argwhere(keep_mask)
    pressure_indices = occupied_coords[:, 0]
    latitude_indices = occupied_coords[:, 1]
    longitude_indices = occupied_coords[:, 2]
    kept_theta = np.asarray(theta_field[keep_mask], dtype=np.float32)
    kept_coldness = np.asarray(coldness_sigma[keep_mask], dtype=np.float32)
    component_labels, component_count = label_wrapped_volume_components(keep_mask)
    component_sizes = (
        np.bincount(component_labels[component_labels > 0].ravel())
        if component_count > 0
        else np.zeros(0, dtype=np.int32)
    )

    metadata = {
        "component_count": int(component_count),
        "largest_component_voxel_count": int(component_sizes.max()) if component_sizes.size else 0,
        "thresholded_voxel_count": int(mesh.voxel_count),
        "vertex_count": int(mesh.positions.size // 3),
        "index_count": int(mesh.indices.size),
        "theta_min": float(np.min(kept_theta)),
        "theta_max": float(np.max(kept_theta)),
        "theta_mean": float(np.mean(kept_theta)),
        "coldness_sigma_min": float(np.min(kept_coldness)),
        "coldness_sigma_max": float(np.max(kept_coldness)),
        "coldness_sigma_mean": float(np.mean(kept_coldness)),
        "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
        "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
        "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        "selection": selection_metadata,
        "z_threshold_sigma": float(z_threshold_sigma),
    }

    return ColdShellAssetPayload(
        timestamp=timestamp,
        positions=mesh.positions,
        indices=mesh.indices,
        coldness_sigma=mesh.coldness_sigma,
        voxel_count=int(mesh.voxel_count),
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
    payload: ColdShellAssetPayload,
) -> dict[str, Any]:
    slug = timestamp_to_slug(payload.timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    positions_path = frame_dir / "positions.bin"
    indices_path = frame_dir / "indices.bin"
    coldness_path = frame_dir / "coldness_sigma.bin"
    metadata_path = frame_dir / "metadata.json"

    payload.positions.astype("<f4").tofile(positions_path)
    payload.indices.astype("<u4").tofile(indices_path)
    payload.coldness_sigma.astype("<f4").tofile(coldness_path)

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": payload.timestamp,
        **payload.metadata,
        "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "coldness_sigma_file": str(coldness_path.relative_to(output_dir)).replace("\\", "/"),
    }
    write_json(metadata_path, metadata)

    return {
        "timestamp": payload.timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "positions": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "coldness_sigma": str(coldness_path.relative_to(output_dir)).replace("\\", "/"),
        "voxel_count": payload.voxel_count,
        "component_count": int(payload.metadata["component_count"]),
    }


def build_manifest(
    *,
    contents: DatasetContents,
    entries: list[dict[str, Any]],
    z_threshold_sigma: float,
    min_level_component_size: int,
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
        "structure_kind": "potential-temperature-cold-zonal-anomaly-shell",
        "geometry_mode": "voxel-faces",
        "selection": {
            "background": "per-level_zonal_mean",
            "standardization": "per-level_stddev",
            "keep_side": "cold_only",
            "z_threshold_sigma": float(z_threshold_sigma),
            "minimum_level_component_size": int(max(min_level_component_size, 0)),
            "level_component_connectivity": "8-connected",
            "volume_connectivity": "26-connected",
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
            _, coldness_sigma, _ = build_standardized_coldness_sigma(theta_field)
            raw_keep_mask = build_cold_mask(
                coldness_sigma,
                z_threshold_sigma=args.z_threshold_sigma,
            )
            keep_mask, level_filter_metadata = filter_small_wrapped_level_components(
                raw_keep_mask,
                min_component_size=args.min_level_component_size,
            )
            if not keep_mask.any():
                print(
                    "Skipped potential temperature frame after postprocess:",
                    timestamp,
                    f"z_threshold_sigma={args.z_threshold_sigma}",
                    f"minimum_level_component_size={max(args.min_level_component_size, 0)}",
                )
                continue

            selection_metadata = {
                "raw_voxel_count": int(raw_keep_mask.sum()),
                "removed_voxel_count": int(raw_keep_mask.sum() - keep_mask.sum()),
                "postprocess": level_filter_metadata,
            }
            payload = build_asset_payload(
                timestamp=timestamp,
                theta_field=theta_field,
                coldness_sigma=coldness_sigma,
                keep_mask=keep_mask,
                pressure_levels_hpa=pressure_window_levels_hpa,
                latitudes_deg=strided_latitudes_deg,
                longitudes_deg=strided_longitudes_deg,
                z_threshold_sigma=args.z_threshold_sigma,
                base_radius=args.base_radius,
                vertical_span=args.vertical_span,
                selection_metadata=selection_metadata,
            )
            entries.append(write_frame(output_dir=output_dir, payload=payload))
            final_latitudes_deg = strided_latitudes_deg
            final_longitudes_deg = strided_longitudes_deg
            final_pressure_levels_hpa = pressure_window_levels_hpa

            print(
                "Built cold potential temperature shell:",
                timestamp,
                f"voxels={payload.voxel_count}",
                f"components={payload.metadata['component_count']}",
                f"removed_level_voxels={level_filter_metadata['removed_voxel_count']}",
                f"coldness_sigma_max={payload.metadata['coldness_sigma_max']:.2f}",
                f"triangles={payload.indices.size // 3}",
            )
    finally:
        raw_dataset.close()

    if not entries or final_latitudes_deg is None or final_longitudes_deg is None or final_pressure_levels_hpa is None:
        raise ValueError("No potential-temperature frames were written.")

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        z_threshold_sigma=args.z_threshold_sigma,
        min_level_component_size=args.min_level_component_size,
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
