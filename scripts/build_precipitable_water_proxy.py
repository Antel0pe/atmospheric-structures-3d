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
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.moisture_structures import compute_pressure_thresholds_from_variable
from scripts.build_relative_humidity_voxel_shell import filter_small_wrapped_components
from scripts.simple_voxel_builder import (
    build_exposed_face_mesh_from_mask,
    coordinate_step_degrees,
    timestamp_to_slug,
)


SPECIFIC_HUMIDITY_VARIABLE = "q"
RELATIVE_HUMIDITY_VARIABLE = "r"
OUTPUT_VERSION = 2
DEFAULT_SPECIFIC_HUMIDITY_DATASET_PATH = Path(
    "data/era5_specific-humidity_2021-11_08-12.nc"
)
DEFAULT_RELATIVE_HUMIDITY_DATASET_PATH = Path(
    "data/era5_relative-humidity_2021-11_08-12.nc"
)
DEFAULT_OUTPUT_DIR = Path("public/precipitable-water-proxy")
DEFAULT_SPECIFIC_HUMIDITY_TOP_PERCENT = 40.0
DEFAULT_RELATIVE_HUMIDITY_THRESHOLD_PERCENT = 85.0
DEFAULT_MIN_PRESSURE_HPA = 500.0
DEFAULT_MAX_PRESSURE_HPA = 1000.0
DEFAULT_MINIMUM_ADJACENT_LEVELS = 3
DEFAULT_MIN_COMPONENT_SIZE = 10
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0


@dataclass(frozen=True)
class DatasetContents:
    specific_humidity_dataset_path: Path
    relative_humidity_dataset_path: Path
    specific_humidity_units: str
    relative_humidity_units: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    specific_humidity_longitude_order: np.ndarray
    relative_humidity_longitude_order: np.ndarray
    timestamps: list[str]


@dataclass(frozen=True)
class PrecipitableWaterProxyAssetPayload:
    timestamp: str
    positions: np.ndarray
    indices: np.ndarray
    voxel_count: int
    gate_counts: dict[str, int]
    postprocess_metadata: dict[str, Any]
    component_metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a precipitable-water proxy voxel shell from ERA5 specific humidity "
            "and relative humidity. Kept voxels must live between 500 and 1000 hPa, "
            "sit in the top-q tail for their pressure level, exceed the RH cutoff, "
            "and belong to a run of at least three adjacent passing pressure levels."
        )
    )
    parser.add_argument(
        "--specific-humidity-dataset",
        type=Path,
        default=DEFAULT_SPECIFIC_HUMIDITY_DATASET_PATH,
        help="Path to the ERA5 specific humidity NetCDF file.",
    )
    parser.add_argument(
        "--relative-humidity-dataset",
        type=Path,
        default=DEFAULT_RELATIVE_HUMIDITY_DATASET_PATH,
        help="Path to the ERA5 relative humidity NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated proxy assets will be written.",
    )
    parser.add_argument(
        "--specific-humidity-top-percent",
        type=float,
        default=DEFAULT_SPECIFIC_HUMIDITY_TOP_PERCENT,
        help=(
            "Keep the top N percent of specific humidity values at each pressure level. "
            "The default 40 means: keep values at or above the 60th percentile."
        ),
    )
    parser.add_argument(
        "--relative-humidity-threshold",
        type=float,
        default=DEFAULT_RELATIVE_HUMIDITY_THRESHOLD_PERCENT,
        help="Keep voxels whose relative humidity is at or above this percentage.",
    )
    parser.add_argument(
        "--minimum-adjacent-levels",
        type=int,
        default=DEFAULT_MINIMUM_ADJACENT_LEVELS,
        help=(
            "Minimum number of adjacent passing pressure levels required in one "
            "lat/lon column."
        ),
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help=(
            "Drop connected 3D components smaller than this many voxels after "
            "the q/RH/depth gates are applied."
        ),
    )
    parser.add_argument(
        "--min-pressure-hpa",
        type=float,
        default=DEFAULT_MIN_PRESSURE_HPA,
        help="Lowest-pressure bound for the gate window.",
    )
    parser.add_argument(
        "--max-pressure-hpa",
        type=float,
        default=DEFAULT_MAX_PRESSURE_HPA,
        help="Highest-pressure bound for the gate window.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default="",
        help=(
            "Comma-separated ISO minute timestamps to build. "
            "Defaults to every timestamp shared by both datasets."
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
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if text.endswith("Z"):
        return text[:-1]
    return text


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    return resolved


def normalize_longitudes_with_order(
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.mod(np.asarray(longitudes_deg, dtype=np.float64) + 180.0, 360.0) - 180.0
    order = np.argsort(normalized, kind="stable")
    return normalized[order].astype(np.float32), order.astype(np.int64)


def load_dataset_contents(
    specific_humidity_dataset_path: Path,
    relative_humidity_dataset_path: Path,
) -> DatasetContents:
    q_dataset = xr.open_dataset(specific_humidity_dataset_path)
    r_dataset = xr.open_dataset(relative_humidity_dataset_path)
    try:
        q_variable = q_dataset[SPECIFIC_HUMIDITY_VARIABLE]
        r_variable = r_dataset[RELATIVE_HUMIDITY_VARIABLE]

        q_pressure_levels = np.asarray(
            q_variable.coords["pressure_level"].values,
            dtype=np.float32,
        )
        r_pressure_levels = np.asarray(
            r_variable.coords["pressure_level"].values,
            dtype=np.float32,
        )
        q_latitudes = np.asarray(q_variable.coords["latitude"].values, dtype=np.float32)
        r_latitudes = np.asarray(r_variable.coords["latitude"].values, dtype=np.float32)
        q_longitudes = np.asarray(
            q_variable.coords["longitude"].values,
            dtype=np.float32,
        )
        r_longitudes = np.asarray(
            r_variable.coords["longitude"].values,
            dtype=np.float32,
        )
        normalized_q_longitudes, q_longitude_order = normalize_longitudes_with_order(
            q_longitudes
        )
        normalized_r_longitudes, r_longitude_order = normalize_longitudes_with_order(
            r_longitudes
        )
        q_timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(q_variable.coords["valid_time"].values)
        ]
        r_timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(r_variable.coords["valid_time"].values)
        ]

        if not np.allclose(q_pressure_levels, r_pressure_levels):
            raise ValueError("Specific humidity and relative humidity pressure levels differ.")
        if not np.allclose(q_latitudes, r_latitudes):
            raise ValueError("Specific humidity and relative humidity latitudes differ.")
        if not np.allclose(normalized_q_longitudes, normalized_r_longitudes):
            raise ValueError("Specific humidity and relative humidity longitudes differ.")
        if q_timestamps != r_timestamps:
            raise ValueError("Specific humidity and relative humidity timestamps differ.")

        return DatasetContents(
            specific_humidity_dataset_path=specific_humidity_dataset_path,
            relative_humidity_dataset_path=relative_humidity_dataset_path,
            specific_humidity_units=str(q_variable.attrs.get("units", "")),
            relative_humidity_units=str(r_variable.attrs.get("units", "")),
            pressure_levels_hpa=q_pressure_levels,
            latitudes_deg=q_latitudes,
            longitudes_deg=normalized_q_longitudes,
            specific_humidity_longitude_order=q_longitude_order,
            relative_humidity_longitude_order=r_longitude_order,
            timestamps=q_timestamps,
        )
    finally:
        q_dataset.close()
        r_dataset.close()


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


def specific_humidity_quantile_from_top_percent(top_percent: float) -> float:
    if not 0.0 <= top_percent <= 100.0:
        raise ValueError(
            f"specific_humidity_top_percent must be between 0 and 100, got {top_percent}"
        )
    return 1.0 - top_percent / 100.0


def build_pressure_window_mask(
    pressure_levels_hpa: np.ndarray,
    min_pressure_hpa: float,
    max_pressure_hpa: float,
) -> np.ndarray:
    lower = min(min_pressure_hpa, max_pressure_hpa)
    upper = max(min_pressure_hpa, max_pressure_hpa)
    return np.asarray(
        (pressure_levels_hpa >= lower) & (pressure_levels_hpa <= upper),
        dtype=bool,
    )


def keep_only_runs_with_min_length(
    mask: np.ndarray,
    minimum_adjacent_levels: int,
) -> np.ndarray:
    if minimum_adjacent_levels <= 1:
        return np.asarray(mask, dtype=bool)

    flat_mask = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1)
    forward = np.zeros_like(flat_mask, dtype=np.int16)
    backward = np.zeros_like(flat_mask, dtype=np.int16)

    forward[0] = flat_mask[0]
    for level_index in range(1, flat_mask.shape[0]):
        forward[level_index] = np.where(
            flat_mask[level_index],
            forward[level_index - 1] + 1,
            0,
        )

    backward[-1] = flat_mask[-1]
    for level_index in range(flat_mask.shape[0] - 2, -1, -1):
        backward[level_index] = np.where(
            flat_mask[level_index],
            backward[level_index + 1] + 1,
            0,
        )

    run_lengths = forward + backward - 1
    kept = flat_mask & (run_lengths >= minimum_adjacent_levels)
    return kept.reshape(mask.shape)


def reorder_longitude_axis(field: np.ndarray, longitude_order: np.ndarray) -> np.ndarray:
    return np.take(np.asarray(field, dtype=np.float32), longitude_order, axis=-1)


def build_precipitable_water_proxy_mask(
    specific_humidity_field: np.ndarray,
    relative_humidity_field: np.ndarray,
    specific_humidity_thresholds: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    relative_humidity_threshold: float,
    min_pressure_hpa: float,
    max_pressure_hpa: float,
    minimum_adjacent_levels: int,
) -> tuple[np.ndarray, dict[str, int]]:
    finite_mask = np.asarray(
        np.isfinite(specific_humidity_field) & np.isfinite(relative_humidity_field),
        dtype=bool,
    )
    pressure_window_mask = build_pressure_window_mask(
        pressure_levels_hpa,
        min_pressure_hpa=min_pressure_hpa,
        max_pressure_hpa=max_pressure_hpa,
    )[:, None, None]
    threshold_mask = np.isfinite(specific_humidity_thresholds)[:, None, None]

    pressure_window_support = finite_mask & pressure_window_mask
    specific_humidity_support = (
        pressure_window_support
        & threshold_mask
        & (specific_humidity_field >= specific_humidity_thresholds[:, None, None])
    )
    relative_humidity_support = (
        pressure_window_support & (relative_humidity_field >= relative_humidity_threshold)
    )
    combined_support = specific_humidity_support & relative_humidity_support
    depth_support = keep_only_runs_with_min_length(
        combined_support,
        minimum_adjacent_levels=minimum_adjacent_levels,
    )

    return depth_support, {
        "finite_pressure_window_voxel_count": int(pressure_window_support.sum()),
        "specific_humidity_gate_voxel_count": int(specific_humidity_support.sum()),
        "relative_humidity_gate_voxel_count": int(relative_humidity_support.sum()),
        "combined_gate_voxel_count": int(combined_support.sum()),
        "depth_gate_voxel_count": int(depth_support.sum()),
    }


def maybe_flip_triangle_winding(indices: np.ndarray) -> np.ndarray:
    normalized = np.asarray(indices, dtype=np.uint32).copy()
    if OUTPUT_VERSION < 2:
        return normalized

    for index in range(0, normalized.size, 3):
        second = int(normalized[index + 1])
        normalized[index + 1] = normalized[index + 2]
        normalized[index + 2] = second
    return normalized


def build_asset_payload(
    timestamp: str,
    specific_humidity_field: np.ndarray,
    relative_humidity_field: np.ndarray,
    keep_mask: np.ndarray,
    positions: np.ndarray,
    indices: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    gate_counts: dict[str, int],
    postprocess_metadata: dict[str, Any],
) -> PrecipitableWaterProxyAssetPayload:
    occupied_coords = np.argwhere(keep_mask)
    if occupied_coords.size == 0:
        raise ValueError("No voxels survived the precipitable water proxy gates.")

    pressure_indices = occupied_coords[:, 0]
    latitude_indices = occupied_coords[:, 1]
    longitude_indices = occupied_coords[:, 2]
    specific_humidity_values = specific_humidity_field[keep_mask]
    relative_humidity_values = relative_humidity_field[keep_mask]

    component_metadata = {
        "id": 0,
        "vertex_offset": 0,
        "vertex_count": int(positions.size // 3),
        "index_offset": 0,
        "index_count": int(indices.size),
        "voxel_count": int(keep_mask.sum()),
        "mean_specific_humidity": float(np.mean(specific_humidity_values)),
        "max_specific_humidity": float(np.max(specific_humidity_values)),
        "mean_relative_humidity": float(np.mean(relative_humidity_values)),
        "max_relative_humidity": float(np.max(relative_humidity_values)),
        "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
        "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
        "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        "wraps_longitude_seam": bool(keep_mask[..., 0].any() and keep_mask[..., -1].any()),
    }

    return PrecipitableWaterProxyAssetPayload(
        timestamp=timestamp,
        positions=np.asarray(positions, dtype=np.float32),
        indices=maybe_flip_triangle_winding(indices),
        voxel_count=int(keep_mask.sum()),
        gate_counts=gate_counts,
        postprocess_metadata=postprocess_metadata,
        component_metadata=component_metadata,
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


def build_threshold_entries(
    pressure_levels_hpa: np.ndarray,
    specific_humidity_thresholds: np.ndarray,
    pressure_window_mask: np.ndarray,
    relative_humidity_threshold: float,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for level_index, pressure_hpa in enumerate(pressure_levels_hpa):
        active = bool(pressure_window_mask[level_index])
        entries.append(
            {
                "pressure_hpa": float(pressure_hpa),
                "active_pressure_window": active,
                "specific_humidity_threshold": (
                    float(specific_humidity_thresholds[level_index]) if active else None
                ),
                "relative_humidity_threshold": (
                    float(relative_humidity_threshold) if active else None
                ),
            }
        )
    return entries


def write_frame(
    output_dir: Path,
    payload: PrecipitableWaterProxyAssetPayload,
) -> dict[str, Any]:
    slug = timestamp_to_slug(payload.timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    positions_path = frame_dir / "positions.bin"
    indices_path = frame_dir / "indices.bin"
    metadata_path = frame_dir / "metadata.json"

    payload.positions.astype("<f4").tofile(positions_path)
    payload.indices.astype("<u4").tofile(indices_path)

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": payload.timestamp,
        "component_count": 1,
        "vertex_count": int(payload.positions.size // 3),
        "index_count": int(payload.indices.size),
        "thresholded_voxel_count": payload.voxel_count,
        "gate_counts": payload.gate_counts,
        "postprocess": payload.postprocess_metadata,
        "components": [payload.component_metadata],
        "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
    }
    write_json(metadata_path, metadata)

    return {
        "timestamp": payload.timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "positions": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "component_count": 1,
        "vertex_count": int(payload.positions.size // 3),
        "index_count": int(payload.indices.size),
    }


def build_manifest(
    contents: DatasetContents,
    entries: list[dict[str, Any]],
    specific_humidity_top_percent: float,
    specific_humidity_quantile: float,
    specific_humidity_thresholds: np.ndarray,
    relative_humidity_threshold: float,
    minimum_adjacent_levels: int,
    min_component_size: int,
    min_pressure_hpa: float,
    max_pressure_hpa: float,
    base_radius: float,
    vertical_span: float,
) -> dict[str, Any]:
    pressure_window_mask = build_pressure_window_mask(
        contents.pressure_levels_hpa,
        min_pressure_hpa=min_pressure_hpa,
        max_pressure_hpa=max_pressure_hpa,
    )

    return {
        "version": OUTPUT_VERSION,
        "datasets": {
            "specific_humidity": contents.specific_humidity_dataset_path.name,
            "relative_humidity": contents.relative_humidity_dataset_path.name,
        },
        "variables": {
            "specific_humidity": SPECIFIC_HUMIDITY_VARIABLE,
            "relative_humidity": RELATIVE_HUMIDITY_VARIABLE,
        },
        "units": {
            "specific_humidity": contents.specific_humidity_units,
            "relative_humidity": contents.relative_humidity_units,
        },
        "structure_kind": "precipitable-water-proxy-voxel-shell",
        "geometry_mode": "voxel-faces",
        "postprocess": {
            "minimum_component_size": int(max(min_component_size, 0)),
            "connectivity": "26-connected",
            "wraps_longitude": True,
        },
        "gates": {
            "specific_humidity": {
                "kind": "pressure-relative-quantile",
                "kept_top_percent": float(specific_humidity_top_percent),
                "quantile": float(specific_humidity_quantile),
                "threshold_seed": "midpoint_time_slice",
            },
            "relative_humidity": {
                "minimum_percent": float(relative_humidity_threshold),
            },
            "vertical_depth": {
                "minimum_adjacent_levels": int(minimum_adjacent_levels),
            },
            "pressure_window": {
                "min_hpa": float(min(min_pressure_hpa, max_pressure_hpa)),
                "max_hpa": float(max(min_pressure_hpa, max_pressure_hpa)),
            },
        },
        "globe": {
            "base_radius": base_radius,
            "vertical_span": vertical_span,
            "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
        },
        "grid": {
            "pressure_level_count": int(contents.pressure_levels_hpa.size),
            "latitude_count": int(contents.latitudes_deg.size),
            "longitude_count": int(contents.longitudes_deg.size),
            "latitude_step_degrees": coordinate_step_degrees(contents.latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(contents.longitudes_deg),
        },
        "thresholds": build_threshold_entries(
            contents.pressure_levels_hpa,
            specific_humidity_thresholds=specific_humidity_thresholds,
            pressure_window_mask=pressure_window_mask,
            relative_humidity_threshold=relative_humidity_threshold,
        ),
        "timestamps": entries,
    }


def main() -> None:
    args = parse_args()
    specific_humidity_dataset_path = resolve_dataset_path(args.specific_humidity_dataset)
    relative_humidity_dataset_path = resolve_dataset_path(args.relative_humidity_dataset)
    contents = load_dataset_contents(
        specific_humidity_dataset_path=specific_humidity_dataset_path,
        relative_humidity_dataset_path=relative_humidity_dataset_path,
    )
    target_timestamps = resolve_target_timestamps(
        contents.timestamps,
        args.include_timestamps,
    )
    if not target_timestamps:
        raise ValueError("No matching timestamps were selected for export.")

    specific_humidity_quantile = specific_humidity_quantile_from_top_percent(
        args.specific_humidity_top_percent
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    specific_humidity_raw_dataset = netCDF4.Dataset(specific_humidity_dataset_path)
    relative_humidity_raw_dataset = netCDF4.Dataset(relative_humidity_dataset_path)
    try:
        specific_humidity_variable = specific_humidity_raw_dataset.variables[
            SPECIFIC_HUMIDITY_VARIABLE
        ]
        relative_humidity_variable = relative_humidity_raw_dataset.variables[
            RELATIVE_HUMIDITY_VARIABLE
        ]

        specific_humidity_thresholds = compute_pressure_thresholds_from_variable(
            specific_humidity_variable,
            quantile=specific_humidity_quantile,
        )
        entries: list[dict[str, Any]] = []

        for time_index, timestamp in enumerate(contents.timestamps):
            if timestamp not in target_timestamps:
                continue

            specific_humidity_field = np.asarray(
                specific_humidity_variable[time_index, :, :, :],
                dtype=np.float32,
            )
            relative_humidity_field = np.asarray(
                relative_humidity_variable[time_index, :, :, :],
                dtype=np.float32,
            )
            specific_humidity_field = reorder_longitude_axis(
                specific_humidity_field,
                contents.specific_humidity_longitude_order,
            )
            relative_humidity_field = reorder_longitude_axis(
                relative_humidity_field,
                contents.relative_humidity_longitude_order,
            )
            keep_mask, gate_counts = build_precipitable_water_proxy_mask(
                specific_humidity_field=specific_humidity_field,
                relative_humidity_field=relative_humidity_field,
                specific_humidity_thresholds=specific_humidity_thresholds,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                relative_humidity_threshold=args.relative_humidity_threshold,
                min_pressure_hpa=args.min_pressure_hpa,
                max_pressure_hpa=args.max_pressure_hpa,
                minimum_adjacent_levels=args.minimum_adjacent_levels,
            )
            keep_mask, postprocess_metadata = filter_small_wrapped_components(
                keep_mask,
                min_component_size=args.min_component_size,
            )

            if not keep_mask.any():
                print(
                    "Skipped precipitable water proxy frame:",
                    timestamp,
                    f"combined_voxels={gate_counts['combined_gate_voxel_count']}",
                    f"depth_voxels={gate_counts['depth_gate_voxel_count']}",
                    f"minimum_component_size={max(args.min_component_size, 0)}",
                )
                continue

            mesh = build_exposed_face_mesh_from_mask(
                keep_mask=keep_mask,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
            )
            payload = build_asset_payload(
                timestamp=timestamp,
                specific_humidity_field=specific_humidity_field,
                relative_humidity_field=relative_humidity_field,
                keep_mask=keep_mask,
                positions=mesh.positions,
                indices=mesh.indices,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
                gate_counts=gate_counts,
                postprocess_metadata=postprocess_metadata,
            )
            entries.append(write_frame(output_dir=output_dir, payload=payload))
            print(
                "Built precipitable water proxy frame:",
                timestamp,
                f"voxels={payload.voxel_count}",
                f"combined_voxels={gate_counts['combined_gate_voxel_count']}",
                f"removed_voxels={postprocess_metadata['removed_voxel_count']}",
                f"removed_components={postprocess_metadata['removed_component_count']}",
                f"vertices={payload.positions.size // 3}",
                f"triangles={payload.indices.size // 3}",
            )
    finally:
        specific_humidity_raw_dataset.close()
        relative_humidity_raw_dataset.close()

    if not entries:
        raise ValueError("No precipitable water proxy frames survived the requested gates.")

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        specific_humidity_top_percent=args.specific_humidity_top_percent,
        specific_humidity_quantile=specific_humidity_quantile,
        specific_humidity_thresholds=specific_humidity_thresholds,
        relative_humidity_threshold=args.relative_humidity_threshold,
        minimum_adjacent_levels=args.minimum_adjacent_levels,
        min_component_size=args.min_component_size,
        min_pressure_hpa=args.min_pressure_hpa,
        max_pressure_hpa=args.max_pressure_hpa,
        base_radius=args.base_radius,
        vertical_span=args.vertical_span,
    )
    write_json(output_dir / "index.json", manifest)
    print(
        "Built precipitable water proxy shells:",
        f"{len(entries)} timestamps",
        f"-> {output_dir}",
    )


if __name__ == "__main__":
    main()
