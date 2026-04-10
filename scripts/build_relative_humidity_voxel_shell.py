from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

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


OUTPUT_VERSION = 1
DATASET_VARIABLE = "r"
DEFAULT_DATASET_PATH = Path("data/era5_relative-humidity_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/relative-humidity-shell")
DEFAULT_THRESHOLD_PERCENT = 95.0
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0
DEFAULT_VARIANT = "baseline"
DEFAULT_MIN_COMPONENT_SIZE = 0
LABEL_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class DatasetContents:
    dataset_path: Path
    variable_name: str
    units: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    timestamps: list[str]


@dataclass(frozen=True)
class RelativeHumidityAssetPayload:
    timestamp: str
    positions: np.ndarray
    indices: np.ndarray
    voxel_count: int
    component_metadata: dict
    postprocess_metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple voxel-shell asset from ERA5 relative humidity. "
            "This keeps cells at or above a fixed RH threshold and emits only "
            "their exposed faces."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the source ERA5 relative humidity NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated RH shell assets will be written.",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENT,
        help="Keep cells whose relative humidity is at or above this percentage.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default=",".join(DEFAULT_INCLUDE_TIMESTAMPS),
        help=(
            "Comma-separated ISO minute timestamps to build. "
            "Defaults to the key saved-view frame."
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
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        help=(
            "Output variant name. Use 'baseline' to write to the main RH shell "
            "folder, or any other name to write under variants/<name>."
        ),
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help=(
            "Drop connected components smaller than this many voxels after "
            "thresholding. Uses 26-connected 3D connectivity with longitude wrap."
        ),
    )
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if not text.endswith("Z"):
        return text
    return text[:-1]


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    return resolved


def load_dataset_contents(dataset_path: Path) -> DatasetContents:
    dataset = xr.open_dataset(dataset_path)
    try:
        variable = dataset[DATASET_VARIABLE]
        timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(variable.coords["valid_time"].values)
        ]
        return DatasetContents(
            dataset_path=dataset_path,
            variable_name=DATASET_VARIABLE,
            units=str(variable.attrs.get("units", "")),
            pressure_levels_hpa=np.asarray(
                variable.coords["pressure_level"].values,
                dtype=np.float32,
            ),
            latitudes_deg=np.asarray(variable.coords["latitude"].values, dtype=np.float32),
            longitudes_deg=np.asarray(
                variable.coords["longitude"].values,
                dtype=np.float32,
            ),
            timestamps=timestamps,
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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_output_dir(base_output_dir: Path, variant: str) -> Path:
    output_root = base_output_dir.expanduser().resolve()
    normalized_variant = variant.strip() or DEFAULT_VARIANT
    if normalized_variant == DEFAULT_VARIANT:
        return output_root
    return output_root / "variants" / normalized_variant


def clear_output_dir(output_dir: Path, preserve_child_names: set[str] | None = None) -> None:
    preserve = preserve_child_names or set()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    for child in output_dir.iterdir():
        if child.name in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_seam_merged_component_info(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    component_count = int(labels.max())
    if component_count <= 0:
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)

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

    seam_pairs = np.column_stack(
        [labels[..., 0].reshape(-1), labels[..., -1].reshape(-1)]
    )
    for first_label, duplicate_label in seam_pairs:
        union(int(first_label), int(duplicate_label))

    root_map = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        root_map[label_id] = find(label_id)

    return root_map, np.unique(root_map[1:])


def filter_small_wrapped_components(
    keep_mask: np.ndarray,
    min_component_size: int,
) -> tuple[np.ndarray, dict]:
    normalized_min_component_size = max(int(min_component_size), 0)
    occupied_voxel_count = int(keep_mask.sum())

    if occupied_voxel_count == 0:
        return np.zeros_like(keep_mask, dtype=bool), {
            "minimum_component_size": normalized_min_component_size,
            "connectivity": "26-connected",
            "wraps_longitude": True,
            "component_count_before_filter": 0,
            "component_count_after_filter": 0,
            "removed_component_count": 0,
            "removed_voxel_count": 0,
        }

    longitude_count = keep_mask.shape[2]
    extended = np.concatenate([keep_mask, keep_mask[..., :1]], axis=2)
    labels, component_count = ndimage.label(extended, structure=LABEL_STRUCTURE)
    if component_count == 0:
        return np.zeros_like(keep_mask, dtype=bool), {
            "minimum_component_size": normalized_min_component_size,
            "connectivity": "26-connected",
            "wraps_longitude": True,
            "component_count_before_filter": 0,
            "component_count_after_filter": 0,
            "removed_component_count": 0,
            "removed_voxel_count": 0,
        }

    root_map, unique_root_ids = build_seam_merged_component_info(labels)
    unique_component_count = int(unique_root_ids.size)

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

    seam_duplicate_labels = labels[..., -1][keep_mask[..., 0]]
    seam_duplicate_counts = np.zeros(component_count + 1, dtype=np.int32)
    if seam_duplicate_labels.size:
        np.add.at(seam_duplicate_counts, root_map[seam_duplicate_labels], 1)

    unique_counts = root_counts - seam_duplicate_counts
    kept_root_ids = unique_root_ids[
        unique_counts[unique_root_ids] >= normalized_min_component_size
    ]
    filtered_mask = np.isin(root_map[labels[..., :longitude_count]], kept_root_ids)
    removed_voxel_count = occupied_voxel_count - int(filtered_mask.sum())
    kept_component_count = int(kept_root_ids.size)

    return filtered_mask, {
        "minimum_component_size": normalized_min_component_size,
        "connectivity": "26-connected",
        "wraps_longitude": True,
        "component_count_before_filter": unique_component_count,
        "component_count_after_filter": kept_component_count,
        "removed_component_count": int(unique_component_count - kept_component_count),
        "removed_voxel_count": int(removed_voxel_count),
    }


def build_asset_payload(
    timestamp: str,
    field: np.ndarray,
    keep_mask: np.ndarray,
    positions: np.ndarray,
    indices: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    postprocess_metadata: dict,
) -> RelativeHumidityAssetPayload:
    occupied_coords = np.argwhere(keep_mask)
    if occupied_coords.size == 0:
        raise ValueError("No voxels survived the RH threshold for this timestamp.")

    pressure_indices = occupied_coords[:, 0]
    latitude_indices = occupied_coords[:, 1]
    longitude_indices = occupied_coords[:, 2]
    kept_values = field[keep_mask]

    component_metadata = {
        "id": 0,
        "vertex_offset": 0,
        "vertex_count": int(positions.size // 3),
        "index_offset": 0,
        "index_count": int(indices.size),
        "voxel_count": int(keep_mask.sum()),
        "mean_relative_humidity": float(np.mean(kept_values)),
        "max_relative_humidity": float(np.max(kept_values)),
        "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
        "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
        "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        "wraps_longitude_seam": bool(keep_mask[..., 0].any() and keep_mask[..., -1].any()),
    }

    return RelativeHumidityAssetPayload(
        timestamp=timestamp,
        positions=positions,
        indices=indices,
        voxel_count=int(keep_mask.sum()),
        component_metadata=component_metadata,
        postprocess_metadata=postprocess_metadata,
    )


def write_frame(
    output_dir: Path,
    payload: RelativeHumidityAssetPayload,
) -> dict:
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
    entries: list[dict],
    threshold_percent: float,
    base_radius: float,
    vertical_span: float,
    variant: str,
    min_component_size: int,
) -> dict:
    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "structure_kind": "relative-humidity-voxel-shell",
        "variant": variant,
        "threshold_percent": threshold_percent,
        "geometry_mode": "voxel-faces",
        "postprocess": {
            "minimum_component_size": int(max(min_component_size, 0)),
            "connectivity": "26-connected",
            "wraps_longitude": True,
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
        "thresholds": [
            {
                "pressure_hpa": float(pressure_hpa),
                "threshold": threshold_percent,
            }
            for pressure_hpa in contents.pressure_levels_hpa
        ],
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

    normalized_variant = args.variant.strip() or DEFAULT_VARIANT
    output_dir = resolve_output_dir(args.output_dir, normalized_variant)
    preserve_children = {"variants"} if normalized_variant == DEFAULT_VARIANT else set()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir, preserve_child_names=preserve_children)

    raw_dataset = netCDF4.Dataset(dataset_path)
    try:
        variable = raw_dataset.variables[DATASET_VARIABLE]
        entries: list[dict] = []
        for time_index, timestamp in enumerate(contents.timestamps):
            if timestamp not in target_timestamps:
                continue

            field = np.asarray(variable[time_index, :, :, :], dtype=np.float32)
            keep_mask = np.asarray(
                np.isfinite(field) & (field >= args.threshold_percent),
                dtype=bool,
            )
            keep_mask, postprocess_metadata = filter_small_wrapped_components(
                keep_mask,
                min_component_size=args.min_component_size,
            )
            if not keep_mask.any():
                print(
                    "Skipped RH shell frame after postprocess:",
                    timestamp,
                    f"threshold={args.threshold_percent}",
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
                field=field,
                keep_mask=keep_mask,
                positions=mesh.positions,
                indices=mesh.indices,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
                postprocess_metadata=postprocess_metadata,
            )
            entries.append(write_frame(output_dir=output_dir, payload=payload))
            print(
                "Built RH shell frame:",
                timestamp,
                f"voxels={payload.voxel_count}",
                f"removed_voxels={postprocess_metadata['removed_voxel_count']}",
                f"removed_components={postprocess_metadata['removed_component_count']}",
                f"vertices={payload.positions.size // 3}",
                f"triangles={payload.indices.size // 3}",
            )
    finally:
        raw_dataset.close()

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        threshold_percent=args.threshold_percent,
        base_radius=args.base_radius,
        vertical_span=args.vertical_span,
        variant=normalized_variant,
        min_component_size=args.min_component_size,
    )
    write_json(output_dir / "index.json", manifest)
    print(
        "Built relative humidity voxel shells:",
        f"{len(entries)} timestamps",
        f"-> {output_dir}",
    )


if __name__ == "__main__":
    main()
