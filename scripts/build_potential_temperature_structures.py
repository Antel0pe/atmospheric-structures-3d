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
LABEL_STRUCTURE = ndimage.generate_binary_structure(3, 3)
OUTPUT_VERSION = 5
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/potential-temperature-structures")
DEFAULT_TOP_PERCENT = 20.0
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4


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
class VoxelSurfaceMesh:
    positions: np.ndarray
    indices: np.ndarray
    voxel_count: int


@dataclass(frozen=True)
class ThresholdSidePayload:
    positions: np.ndarray
    indices: np.ndarray
    voxel_count: int
    component_count: int
    touching_component_count: int
    theta_min: float | None
    theta_max: float | None
    theta_mean: float | None
    pressure_min_hpa: float | None
    pressure_max_hpa: float | None


@dataclass(frozen=True)
class PotentialTemperatureAssetPayload:
    timestamp: str
    top_percent: float
    threshold_value: float
    finite_voxel_count: int
    hot_side: ThresholdSidePayload
    cool_side: ThresholdSidePayload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build two opaque dry-potential-temperature voxel structures from ERA5 "
            "pressure-level temperature. The builder derives dry potential "
            "temperature in 3D, finds the global top 20 percent threshold by "
            "default, keeps connected components on the hot and cool sides that "
            "touch that threshold boundary, and exports voxel-face shells for both."
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
        "--top-percent",
        type=float,
        default=DEFAULT_TOP_PERCENT,
        help=(
            "Use the global top N percent of dry potential temperature values as "
            "the hot side of the threshold."
        ),
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default=",".join(DEFAULT_INCLUDE_TIMESTAMPS),
        help=(
            "Comma-separated ISO minute timestamps to build. "
            "Defaults to the comparison frame."
        ),
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
        help="Keep every Nth latitude sample when building the voxel shells.",
    )
    parser.add_argument(
        "--longitude-stride",
        type=int,
        default=DEFAULT_LONGITUDE_STRIDE,
        help="Keep every Nth longitude sample when building the voxel shells.",
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

    if temperature.ndim == 4:
        pressure_scale = np.power(
            REFERENCE_PRESSURE_HPA / pressure_levels[None, :, None, None],
            POTENTIAL_TEMPERATURE_KAPPA,
            dtype=np.float32,
        )
    elif temperature.ndim == 3:
        pressure_scale = np.power(
            REFERENCE_PRESSURE_HPA / pressure_levels[:, None, None],
            POTENTIAL_TEMPERATURE_KAPPA,
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Expected temperature values with shape (time, level, lat, lon) or "
            f"(level, lat, lon), got {temperature.shape}"
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


def build_top_percent_mask(
    field: np.ndarray,
    top_percent: float,
) -> tuple[np.ndarray, float]:
    normalized_top_percent = float(top_percent)
    if not np.isfinite(normalized_top_percent) or not 0.0 < normalized_top_percent <= 100.0:
        raise ValueError(
            f"top_percent must be a finite value in (0, 100], got {top_percent}"
        )

    finite_mask = np.isfinite(field)
    finite_values = np.asarray(field[finite_mask], dtype=np.float32)
    if finite_values.size == 0:
        raise ValueError("Cannot threshold dry potential temperature with no finite values.")

    keep_quantile = 1.0 - normalized_top_percent / 100.0
    threshold_value = float(np.quantile(finite_values, keep_quantile))
    keep_mask = np.asarray(finite_mask & (field >= threshold_value), dtype=bool)
    return keep_mask, threshold_value


def build_seam_merged_component_info(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    seam_pairs = np.column_stack([labels[..., 0].reshape(-1), labels[..., -1].reshape(-1)])
    for first_label, duplicate_label in seam_pairs:
        union(int(first_label), int(duplicate_label))

    root_map = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        root_map[label_id] = find(label_id)

    return root_map, np.unique(root_map[1:])


def label_wrapped_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    longitude_count = occupied.shape[2]
    extended = np.concatenate([occupied, occupied[..., :1]], axis=2)
    labels, component_count = ndimage.label(extended, structure=LABEL_STRUCTURE)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    root_map, unique_root_ids = build_seam_merged_component_info(labels)
    if unique_root_ids.size == 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_root_ids] = np.arange(1, unique_root_ids.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[..., :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_root_ids.size)


def build_gradient_adjacency_mask(mask: np.ndarray, opposite_mask: np.ndarray) -> np.ndarray:
    occupied = np.asarray(mask, dtype=bool)
    opposite = np.asarray(opposite_mask, dtype=bool)
    adjacency = np.zeros_like(occupied, dtype=bool)

    adjacency[:-1, :, :] |= occupied[:-1, :, :] & opposite[1:, :, :]
    adjacency[1:, :, :] |= occupied[1:, :, :] & opposite[:-1, :, :]
    adjacency[:, :-1, :] |= occupied[:, :-1, :] & opposite[:, 1:, :]
    adjacency[:, 1:, :] |= occupied[:, 1:, :] & opposite[:, :-1, :]
    adjacency |= occupied & np.roll(opposite, 1, axis=2)
    adjacency |= occupied & np.roll(opposite, -1, axis=2)
    return adjacency


def keep_components_touching_opposite(
    mask: np.ndarray,
    opposite_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    labels, component_count = label_wrapped_components(mask)
    if component_count <= 0:
        return np.zeros_like(mask, dtype=bool), 0, 0

    touching_voxels = build_gradient_adjacency_mask(mask, opposite_mask)
    touching_component_ids = np.unique(labels[touching_voxels])
    touching_component_ids = touching_component_ids[touching_component_ids > 0]
    if touching_component_ids.size == 0:
        return np.zeros_like(mask, dtype=bool), component_count, 0

    keep_mask = np.isin(labels, touching_component_ids)
    return keep_mask.astype(bool), component_count, int(touching_component_ids.size)


def lat_lon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(-(lon_deg + 270.0))
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.sin(lat)
    z = radius * np.cos(lat) * np.sin(lon)
    return np.array([x, y, z], dtype=np.float32)


def append_quad(
    corners: list[tuple[int, int, int]],
    *,
    vertex_lookup: dict[tuple[int, int, int], int],
    positions: list[float],
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
            vertex_lookup[key] = vertex_index
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


def build_exposed_face_mesh_from_mask(
    *,
    source_mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    base_radius: float,
    vertical_span: float,
    occupancy_mask: np.ndarray | None = None,
) -> VoxelSurfaceMesh:
    emit_mask = np.asarray(source_mask, dtype=bool)
    occupancy = emit_mask if occupancy_mask is None else np.asarray(occupancy_mask, dtype=bool)

    if emit_mask.shape != occupancy.shape:
        raise ValueError(
            f"source_mask shape {emit_mask.shape} must match occupancy_mask shape {occupancy.shape}"
        )

    if not emit_mask.any():
        return VoxelSurfaceMesh(
            positions=np.zeros(0, dtype=np.float32),
            indices=np.zeros(0, dtype=np.uint32),
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

    level_count, latitude_count, longitude_count = emit_mask.shape
    occupied_cells = np.argwhere(emit_mask)
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    positions: list[float] = []
    indices: list[int] = []

    for level_index, latitude_index, longitude_index in occupied_cells:
        r0 = int(level_index)
        r1 = r0 + 1
        a0 = int(latitude_index)
        a1 = a0 + 1
        o0 = int(longitude_index)
        o1 = o0 + 1

        west_neighbor = (o0 - 1) % longitude_count
        east_neighbor = (o0 + 1) % longitude_count

        if r0 == 0 or not occupancy[r0 - 1, a0, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if r0 == level_count - 1 or not occupancy[r0 + 1, a0, o0]:
            append_quad(
                corners=[(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == 0 or not occupancy[r0, a0 - 1, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == latitude_count - 1 or not occupancy[r0, a0 + 1, o0]:
            append_quad(
                corners=[(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not occupancy[r0, a0, west_neighbor]:
            append_quad(
                corners=[(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not occupancy[r0, a0, east_neighbor]:
            append_quad(
                corners=[(r0, a0, o1), (r0, a1, o1), (r1, a1, o1), (r1, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )

    return VoxelSurfaceMesh(
        positions=np.asarray(positions, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        voxel_count=int(occupied_cells.shape[0]),
    )


def build_side_payload(
    *,
    keep_mask: np.ndarray,
    theta_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    base_radius: float,
    vertical_span: float,
    component_count: int,
    touching_component_count: int,
    occupancy_mask: np.ndarray | None = None,
) -> ThresholdSidePayload:
    mesh = build_exposed_face_mesh_from_mask(
        source_mask=keep_mask,
        occupancy_mask=occupancy_mask,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        base_radius=base_radius,
        vertical_span=vertical_span,
    )

    if not keep_mask.any():
        return ThresholdSidePayload(
            positions=mesh.positions,
            indices=mesh.indices,
            voxel_count=0,
            component_count=int(component_count),
            touching_component_count=int(touching_component_count),
            theta_min=None,
            theta_max=None,
            theta_mean=None,
            pressure_min_hpa=None,
            pressure_max_hpa=None,
        )

    occupied_coords = np.argwhere(keep_mask)
    pressure_indices = occupied_coords[:, 0]
    kept_values = np.asarray(theta_field[keep_mask], dtype=np.float32)

    return ThresholdSidePayload(
        positions=mesh.positions,
        indices=mesh.indices,
        voxel_count=int(keep_mask.sum()),
        component_count=int(component_count),
        touching_component_count=int(touching_component_count),
        theta_min=float(np.min(kept_values)),
        theta_max=float(np.max(kept_values)),
        theta_mean=float(np.mean(kept_values)),
        pressure_min_hpa=float(np.min(pressure_levels_hpa[pressure_indices])),
        pressure_max_hpa=float(np.max(pressure_levels_hpa[pressure_indices])),
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


def build_asset_payload(
    *,
    timestamp: str,
    potential_temperature_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    top_percent: float,
    base_radius: float,
    vertical_span: float,
) -> PotentialTemperatureAssetPayload:
    hot_mask, threshold_value = build_top_percent_mask(
        potential_temperature_field,
        top_percent=top_percent,
    )
    finite_mask = np.isfinite(potential_temperature_field)
    cool_mask = np.asarray(finite_mask & ~hot_mask, dtype=bool)

    hot_keep_mask, hot_component_count, hot_touching_component_count = (
        keep_components_touching_opposite(hot_mask, cool_mask)
    )
    cool_keep_mask, cool_component_count, cool_touching_component_count = (
        keep_components_touching_opposite(cool_mask, hot_mask)
    )

    hot_side = build_side_payload(
        keep_mask=hot_keep_mask,
        theta_field=potential_temperature_field,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        base_radius=base_radius,
        vertical_span=vertical_span,
        component_count=hot_component_count,
        touching_component_count=hot_touching_component_count,
        occupancy_mask=hot_keep_mask,
    )
    cool_side = build_side_payload(
        keep_mask=cool_keep_mask,
        theta_field=potential_temperature_field,
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        base_radius=base_radius,
        vertical_span=vertical_span,
        component_count=cool_component_count,
        touching_component_count=cool_touching_component_count,
        occupancy_mask=np.asarray(hot_keep_mask | cool_keep_mask, dtype=bool),
    )

    return PotentialTemperatureAssetPayload(
        timestamp=timestamp,
        top_percent=float(top_percent),
        threshold_value=float(threshold_value),
        finite_voxel_count=int(finite_mask.sum()),
        hot_side=hot_side,
        cool_side=cool_side,
    )


def build_side_metadata(
    output_dir: Path,
    side_slug: str,
    side_payload: ThresholdSidePayload,
    frame_dir: Path,
) -> dict[str, Any]:
    positions_path = frame_dir / f"{side_slug}_positions.bin"
    indices_path = frame_dir / f"{side_slug}_indices.bin"

    side_payload.positions.astype("<f4").tofile(positions_path)
    side_payload.indices.astype("<u4").tofile(indices_path)

    return {
        "voxel_count": side_payload.voxel_count,
        "component_count": side_payload.component_count,
        "touching_component_count": side_payload.touching_component_count,
        "vertex_count": int(side_payload.positions.size // 3),
        "index_count": int(side_payload.indices.size),
        "theta_min": side_payload.theta_min,
        "theta_max": side_payload.theta_max,
        "theta_mean": side_payload.theta_mean,
        "pressure_min_hpa": side_payload.pressure_min_hpa,
        "pressure_max_hpa": side_payload.pressure_max_hpa,
        "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
    }


def write_frame(
    output_dir: Path,
    payload: PotentialTemperatureAssetPayload,
) -> dict[str, Any]:
    slug = timestamp_to_slug(payload.timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = frame_dir / "metadata.json"

    hot_side_metadata = build_side_metadata(output_dir, "hot", payload.hot_side, frame_dir)
    cool_side_metadata = build_side_metadata(output_dir, "cool", payload.cool_side, frame_dir)

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": payload.timestamp,
        "top_percent": payload.top_percent,
        "threshold_value": payload.threshold_value,
        "finite_voxel_count": payload.finite_voxel_count,
        "hot_side": hot_side_metadata,
        "cool_side": cool_side_metadata,
    }
    write_json(metadata_path, metadata)

    return {
        "timestamp": payload.timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "threshold_value": payload.threshold_value,
        "hot_voxel_count": payload.hot_side.voxel_count,
        "cool_voxel_count": payload.cool_side.voxel_count,
        "hot_component_count": payload.hot_side.touching_component_count,
        "cool_component_count": payload.cool_side.touching_component_count,
    }


def build_manifest(
    *,
    contents: DatasetContents,
    entries: list[dict[str, Any]],
    top_percent: float,
    latitude_stride: int,
    longitude_stride: int,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
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
        "structure_kind": "potential-temperature-threshold-shells",
        "geometry_mode": "voxel-faces",
        "threshold": {
            "kind": "global_top_percent",
            "top_percent": float(top_percent),
            "quantile": float(1.0 - top_percent / 100.0),
        },
        "selection": {
            "connectivity": "26-connected",
            "wraps_longitude": True,
            "side_rule": "components_touching_opposite_threshold_side",
            "hot_interface_faces_visible": True,
            "cool_interface_faces_visible": False,
        },
        "sampling": {
            "latitude_stride": int(max(latitude_stride, 1)),
            "longitude_stride": int(max(longitude_stride, 1)),
            "method": "subsample_centers",
        },
        "globe": {
            "base_radius": base_radius,
            "vertical_span": vertical_span,
            "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
        },
        "grid": {
            "pressure_level_count": int(contents.pressure_levels_hpa.size),
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
    strided_latitudes_deg = contents.latitudes_deg[:: max(int(args.latitude_stride), 1)]
    strided_longitudes_deg = contents.longitudes_deg[:: max(int(args.longitude_stride), 1)]
    try:
        variable = raw_dataset.variables[TEMPERATURE_VARIABLE]
        entries: list[dict[str, Any]] = []
        for time_index, timestamp in enumerate(contents.timestamps):
            if timestamp not in target_timestamps:
                continue

            temperature_field = reorder_longitude_axis(
                np.asarray(variable[time_index, :, :, :], dtype=np.float32),
                contents.longitude_order,
            )
            temperature_field, _, _ = stride_spatial_axes(
                temperature_field,
                contents.latitudes_deg,
                contents.longitudes_deg,
                latitude_stride=args.latitude_stride,
                longitude_stride=args.longitude_stride,
            )
            potential_temperature_field = compute_dry_potential_temperature(
                temperature_field,
                contents.pressure_levels_hpa,
            )
            payload = build_asset_payload(
                timestamp=timestamp,
                potential_temperature_field=potential_temperature_field,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=strided_latitudes_deg,
                longitudes_deg=strided_longitudes_deg,
                top_percent=args.top_percent,
                base_radius=args.base_radius,
                vertical_span=args.vertical_span,
            )
            entries.append(write_frame(output_dir=output_dir, payload=payload))
            print(
                "Built potential temperature threshold shells:",
                timestamp,
                f"threshold={payload.threshold_value:.3f}",
                f"hot_voxels={payload.hot_side.voxel_count}",
                f"cool_voxels={payload.cool_side.voxel_count}",
                f"hot_components={payload.hot_side.touching_component_count}",
                f"cool_components={payload.cool_side.touching_component_count}",
                f"hot_triangles={payload.hot_side.indices.size // 3}",
                f"cool_triangles={payload.cool_side.indices.size // 3}",
            )
    finally:
        raw_dataset.close()

    if not entries:
        raise ValueError("No potential-temperature frames were written.")

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        top_percent=args.top_percent,
        latitude_stride=args.latitude_stride,
        longitude_stride=args.longitude_stride,
        latitudes_deg=strided_latitudes_deg,
        longitudes_deg=strided_longitudes_deg,
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
