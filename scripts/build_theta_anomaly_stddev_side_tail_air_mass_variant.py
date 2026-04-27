from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_theta_anomaly_bucket_air_mass_variants import (
    BUCKET_COLORS,
    DEFAULT_BASE_RADIUS,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_LATITUDE_STRIDE,
    DEFAULT_LONGITUDE_STRIDE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STATS_DIR,
    DEFAULT_TEMPERATURE_PATH,
    DEFAULT_TIMESTAMP,
    DEFAULT_VERTICAL_SPAN,
    OUTPUT_VERSION,
    PRESSURE_MAX_HPA,
    PRESSURE_MIN_HPA,
    RETAINED_BUCKETS,
    build_bounds,
    class_entries,
    class_label,
    component_summary,
    compute_theta_anomaly,
    label_wrapped_components,
    maybe_flip_triangle_winding,
    repo_path,
    six_neighbor_structure,
    standard_deviation_bucket_indices,
    stride_mask,
)
from scripts.simple_voxel_builder import (
    NaiveVoxelMesh,
    append_quad,
    coordinate_step_degrees,
    build_axis_bounds,
    build_radius_lookup_from_pressure_levels,
    timestamp_to_slug,
)


DEFAULT_VARIANT_NAME = "theta-anomaly-stddev-side-6neighbor-min100k"
DEFAULT_VARIANT_LABEL = "Theta Std Dev Side Tails >=100k"
DEFAULT_NO_TOP_VARIANT_NAME = "theta-anomaly-stddev-side-6neighbor-min100k-open-top"
DEFAULT_NO_TOP_VARIANT_LABEL = "Theta Std Dev Side Tails >=100k Open Top"
DEFAULT_MIN_COMPONENT_VOXELS = 100_000
SIDE_GROUPS = (
    ("cold_tail_buckets_0_1_2", (0, 1, 2)),
    ("warm_tail_buckets_7_8_9", (7, 8, 9)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export theta-climatology anomaly stddev tail buckets as a frontend "
            "air-mass variant. Components are labeled on the full-resolution grid "
            "by cold or warm stddev side with 6-neighbor connectivity, then the "
            "surviving bucket cells are subsampled to the viewer asset grid."
        )
    )
    parser.add_argument("--temperature", type=Path, default=DEFAULT_TEMPERATURE_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument("--variant-label", type=str, default=None)
    parser.add_argument("--min-component-voxels", type=int, default=DEFAULT_MIN_COMPONENT_VOXELS)
    parser.add_argument(
        "--omit-top-faces",
        action="store_true",
        help=(
            "Do not export exposed radial top faces. Bottom faces and all side "
            "faces are still exported."
        ),
    )
    parser.add_argument("--latitude-stride", type=int, default=DEFAULT_LATITUDE_STRIDE)
    parser.add_argument("--longitude-stride", type=int, default=DEFAULT_LONGITUDE_STRIDE)
    parser.add_argument("--base-radius", type=float, default=DEFAULT_BASE_RADIUS)
    parser.add_argument("--vertical-span", type=float, default=DEFAULT_VERTICAL_SPAN)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_exposed_face_mesh_from_mask_with_face_options(
    *,
    keep_mask: np.ndarray,
    exposure_mask: np.ndarray,
    top_face_omit_mask: np.ndarray | None,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> NaiveVoxelMesh:
    if keep_mask.shape != exposure_mask.shape:
        raise ValueError("keep_mask and exposure_mask must have the same shape.")
    if top_face_omit_mask is not None and top_face_omit_mask.shape != keep_mask.shape:
        raise ValueError("top_face_omit_mask must match keep_mask shape.")

    radius_values = build_radius_lookup_from_pressure_levels(pressure_levels_hpa)
    radius_bounds = build_axis_bounds(radius_values)
    latitude_bounds = build_axis_bounds(latitudes_deg)
    longitude_bounds = build_axis_bounds(longitudes_deg)

    level_count, latitude_count, longitude_count = keep_mask.shape
    occupied_cells = np.argwhere(keep_mask)
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

        if r0 == 0 or not exposure_mask[r0 - 1, a0, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        omit_this_top_face = bool(top_face_omit_mask is not None and top_face_omit_mask[r0, a0, o0])
        if (
            not omit_this_top_face
            and (r0 == level_count - 1 or not exposure_mask[r0 + 1, a0, o0])
        ):
            append_quad(
                corners=[(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == 0 or not exposure_mask[r0, a0 - 1, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == latitude_count - 1 or not exposure_mask[r0, a0 + 1, o0]:
            append_quad(
                corners=[(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not exposure_mask[r0, a0, west_neighbor]:
            append_quad(
                corners=[(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not exposure_mask[r0, a0, east_neighbor]:
            append_quad(
                corners=[(r0, a0, o1), (r0, a1, o1), (r1, a1, o1), (r1, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )

    positions_array = np.asarray(positions, dtype=np.float32)
    indices_array = np.asarray(indices, dtype=np.uint32)
    return NaiveVoxelMesh(
        positions=positions_array,
        indices=indices_array,
        cube_count=int(occupied_cells.shape[0]),
        vertex_count=int(positions_array.size // 3),
        triangle_count=int(indices_array.size // 3),
    )


def build_highest_column_cell_mask(mask: np.ndarray) -> np.ndarray:
    occupied = np.asarray(mask, dtype=bool)
    result = np.zeros(occupied.shape, dtype=bool)
    occupied_columns = np.any(occupied, axis=0)
    if not np.any(occupied_columns):
        return result

    level_count = occupied.shape[0]
    highest_level_by_column = level_count - 1 - np.argmax(occupied[::-1], axis=0)
    latitude_indices, longitude_indices = np.nonzero(occupied_columns)
    result[
        highest_level_by_column[latitude_indices, longitude_indices],
        latitude_indices,
        longitude_indices,
    ] = True
    return result


def kept_size_summary(sizes: np.ndarray) -> dict[str, Any]:
    if sizes.size == 0:
        return {
            "component_count": 0,
            "total_voxels": 0,
            "min_component_voxels": 0,
            "median_component_voxels": 0.0,
            "mean_component_voxels": 0.0,
            "max_component_voxels": 0,
        }
    return {
        "component_count": int(sizes.size),
        "total_voxels": int(np.sum(sizes)),
        "min_component_voxels": int(np.min(sizes)),
        "median_component_voxels": float(np.median(sizes)),
        "mean_component_voxels": float(np.mean(sizes)),
        "max_component_voxels": int(np.max(sizes)),
    }


def filter_full_resolution_by_stddev_side(
    buckets: np.ndarray,
    min_component_voxels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    structure = six_neighbor_structure()
    filtered = np.full(buckets.shape, -1, dtype=np.int8)
    stats: dict[str, Any] = {}

    for group_name, group_buckets in SIDE_GROUPS:
        print(f"labeling 6-neighbor full-resolution components for {group_name}", flush=True)
        mask = np.isin(buckets, group_buckets)
        labels, component_count = label_wrapped_components(mask, structure)
        sizes = (
            np.bincount(labels[labels > 0].ravel(), minlength=component_count + 1)
            if component_count > 0
            else np.zeros(1, dtype=np.int64)
        )
        keep_component = sizes >= int(min_component_voxels)
        keep_component[0] = False
        kept_mask = keep_component[labels] if component_count > 0 else np.zeros(mask.shape, dtype=bool)
        filtered[kept_mask] = buckets[kept_mask]
        kept_sizes = sizes[keep_component]
        bucket_counts = {
            f"bucket_{bucket_index}": int(np.count_nonzero(filtered == bucket_index))
            for bucket_index in group_buckets
        }
        stats[group_name] = {
            "component_buckets": [int(value) for value in group_buckets],
            "component_count_before_filter": int(component_count),
            "component_count_after_filter": int(kept_sizes.size),
            "cell_count_before_filter": int(np.count_nonzero(mask)),
            "cell_count_after_filter": int(np.count_nonzero(kept_mask)),
            "removed_component_count": int(component_count - kept_sizes.size),
            "removed_cell_count": int(np.count_nonzero(mask) - np.count_nonzero(kept_mask)),
            "kept_size_summary": kept_size_summary(kept_sizes),
            "bucket_counts_after_filter_full_resolution": bucket_counts,
        }

    return filtered, stats


def build_variant(
    *,
    filtered_buckets: np.ndarray,
    full_resolution_component_stats: dict[str, Any],
    level_metadata: list[dict[str, Any]],
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamp: str,
    temperature_path: Path,
    climatology_path: Path,
    output_base_dir: Path,
    stats_dir: Path,
    variant_name: str,
    variant_label: str,
    min_component_voxels: int,
    latitude_stride: int,
    longitude_stride: int,
    base_radius: float,
    vertical_span: float,
    omit_top_faces: bool,
) -> None:
    output_dir = output_base_dir / "variants" / variant_name
    clear_output_dir(output_dir)
    frame_dir = output_dir / timestamp_to_slug(timestamp)
    frame_dir.mkdir(parents=True, exist_ok=True)

    structure = six_neighbor_structure()
    class_summaries: dict[str, Any] = {}
    class_counts: dict[str, Any] = {}
    all_mask = np.zeros(filtered_buckets.shape, dtype=bool)
    exposure_mask = filtered_buckets >= 0 if omit_top_faces else None
    top_face_omit_mask = (
        build_highest_column_cell_mask(exposure_mask)
        if exposure_mask is not None
        else None
    )
    total_voxel_count = 0
    total_component_count = 0

    for bucket_index in RETAINED_BUCKETS:
        class_key = f"bucket_{bucket_index}"
        mask = filtered_buckets == bucket_index
        all_mask |= mask
        voxel_count = int(np.count_nonzero(mask))
        summary = component_summary(mask, structure)
        mesh = build_exposed_face_mesh_from_mask_with_face_options(
            keep_mask=mask,
            exposure_mask=exposure_mask if exposure_mask is not None else mask,
            top_face_omit_mask=top_face_omit_mask,
            pressure_levels_hpa=pressure_levels,
            latitudes_deg=latitudes,
            longitudes_deg=longitudes,
        )

        positions_path = frame_dir / f"{class_key}_positions.bin"
        indices_path = frame_dir / f"{class_key}_indices.bin"
        np.asarray(mesh.positions, dtype="<f4").tofile(positions_path)
        maybe_flip_triangle_winding(np.asarray(mesh.indices, dtype=np.uint32)).astype("<u4").tofile(indices_path)

        class_summaries[class_key] = {
            "label": class_label("standard-deviation", bucket_index),
            "bucket_index": int(bucket_index),
            "color": BUCKET_COLORS[bucket_index],
            "voxel_count": voxel_count,
            "component_count": int(summary["component_count"]),
            "largest_component_voxel_count": int(summary["largest_component_voxel_count"]),
            "positions_file": positions_path.relative_to(output_dir).as_posix(),
            "indices_file": indices_path.relative_to(output_dir).as_posix(),
            "vertex_count": int(mesh.vertex_count),
            "index_count": int(mesh.indices.size),
        }
        class_counts[class_key] = {
            "voxel_count": voxel_count,
            "component_count": int(summary["component_count"]),
        }
        total_voxel_count += voxel_count
        total_component_count += int(summary["component_count"])

    score_thresholds_by_pressure_level = [
        {
            "pressure_hpa": float(pressure_levels[index]),
            **level_metadata[index],
        }
        for index in range(len(level_metadata))
    ]
    selection = {
        "keep_top_percent": 0.0,
        "axis_min_abs_zscore": 1.0,
        "score_basis": "per-level_zscore_abs_greater_than_1_side_components_min100000",
        "bridge_gap_levels": 0,
        "min_component_voxels": int(min_component_voxels),
        "min_component_pressure_span_levels": 1,
        "surface_attached_only": False,
        "component_connectivity": "6-neighbor with wrapped longitude",
        "component_grouping": "stddev-side; cold buckets 0-2 connect together and warm buckets 7-9 connect together",
        "component_filter_resolution": "full source grid before frontend subsampling",
        "radial_top_faces_drawn": not omit_top_faces,
        "face_exposure_basis": "all_retained_buckets" if omit_top_faces else "same_bucket",
        "top_face_omit_rule": (
            "omit only the original highest occupied radial cap face per lat/lon column"
            if omit_top_faces
            else "none"
        ),
        "classes": class_entries("standard-deviation"),
    }

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": timestamp,
        "voxel_count": int(total_voxel_count),
        "component_count": int(total_component_count),
        "component_connectivity": selection["component_connectivity"],
        "component_grouping": selection["component_grouping"],
        "component_filter_resolution": selection["component_filter_resolution"],
        "radial_top_faces_drawn": selection["radial_top_faces_drawn"],
        "face_exposure_basis": selection["face_exposure_basis"],
        "top_face_omit_rule": selection["top_face_omit_rule"],
        "class_summaries": class_summaries,
        "pressure_levels_hpa": [float(value) for value in pressure_levels.tolist()],
        "score_thresholds_by_pressure_level": score_thresholds_by_pressure_level,
        "full_resolution_component_stats": full_resolution_component_stats,
        "thermal_axis": {
            "field": "dry_potential_temperature",
            "transform": "matched_gridpoint_climatology_anomaly",
            "scale_by_pressure_level": [],
        },
        "moisture_axis": {
            "field": "none",
            "transform": "none",
            "scale_by_pressure_level": [],
        },
        "selection": selection,
        "smoothing_sigma_cells": 0.0,
        **build_bounds(all_mask, pressure_levels, latitudes, longitudes),
    }
    metadata_path = frame_dir / "metadata.json"
    write_json(metadata_path, metadata)

    manifest = {
        "version": OUTPUT_VERSION,
        "dataset": temperature_path.name,
        "climatology_dataset": climatology_path.name,
        "variant": variant_name,
        "variant_label": variant_label,
        "structure_kind": "theta-anomaly-bucket-shells",
        "geometry_mode": "voxel-faces",
        "variables": {
            "temperature": "t",
            "relative_humidity": "none",
            "specific_humidity": "none",
        },
        "classification": {
            "thermal_axis_field": "dry_potential_temperature",
            "moisture_axis_field": "none",
            "thermal_transform": "matched_gridpoint_climatology_anomaly",
            "moisture_transform": "none",
            "score_basis": selection["score_basis"],
            "keep_top_percent": selection["keep_top_percent"],
            "axis_min_abs_zscore": selection["axis_min_abs_zscore"],
            "bridge_gap_levels": selection["bridge_gap_levels"],
            "min_component_voxels": selection["min_component_voxels"],
            "min_component_pressure_span_levels": selection["min_component_pressure_span_levels"],
            "surface_attached_only": selection["surface_attached_only"],
            "component_connectivity": selection["component_connectivity"],
            "component_grouping": selection["component_grouping"],
            "component_filter_resolution": selection["component_filter_resolution"],
            "radial_top_faces_drawn": selection["radial_top_faces_drawn"],
            "face_exposure_basis": selection["face_exposure_basis"],
            "top_face_omit_rule": selection["top_face_omit_rule"],
            "classes": class_entries("standard-deviation"),
        },
        "sampling": {
            "latitude_stride": int(latitude_stride),
            "longitude_stride": int(longitude_stride),
            "method": "component_filter_full_resolution_then_subsample_centers",
        },
        "pressure_window_hpa": {
            "min": PRESSURE_MIN_HPA,
            "max": PRESSURE_MAX_HPA,
            "level_count": int(pressure_levels.size),
        },
        "globe": {
            "base_radius": float(base_radius),
            "vertical_span": float(vertical_span),
            "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
        },
        "grid": {
            "pressure_level_count": int(pressure_levels.size),
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_step_degrees": coordinate_step_degrees(latitudes),
            "longitude_step_degrees": coordinate_step_degrees(longitudes),
        },
        "timestamps": [
            {
                "timestamp": timestamp,
                "metadata": metadata_path.relative_to(output_dir).as_posix(),
                "voxel_count": int(total_voxel_count),
                "component_count": int(total_component_count),
                "class_counts": class_counts,
            }
        ],
    }
    write_json(output_dir / "index.json", manifest)

    stats = {
        "variant": variant_name,
        "timestamp": timestamp,
        "pressure_window_hpa": {"min": PRESSURE_MIN_HPA, "max": PRESSURE_MAX_HPA},
        "min_component_voxels": int(min_component_voxels),
        "component_connectivity": selection["component_connectivity"],
        "component_grouping": selection["component_grouping"],
        "component_filter_resolution": selection["component_filter_resolution"],
        "radial_top_faces_drawn": selection["radial_top_faces_drawn"],
        "face_exposure_basis": selection["face_exposure_basis"],
        "top_face_omit_rule": selection["top_face_omit_rule"],
        "sampling": manifest["sampling"],
        "full_resolution_component_stats": full_resolution_component_stats,
        "frontend_asset_counts": {
            "voxel_count": int(total_voxel_count),
            "component_count": int(total_component_count),
            "class_counts": class_counts,
        },
    }
    write_json(stats_dir / f"{variant_name}-export-stats.json", stats)

    print(
        "Built theta anomaly stddev side-tail variant:",
        variant_name,
        f"frontend_voxels={total_voxel_count}",
        f"frontend_bucket_components={total_component_count}",
    )


def main() -> None:
    args = parse_args()
    temperature_path = resolve_repo_path(args.temperature)
    climatology_path = resolve_repo_path(args.climatology)
    output_base_dir = resolve_repo_path(args.output_dir)
    stats_dir = resolve_repo_path(args.stats_dir)
    if args.omit_top_faces:
        variant_name = args.variant_name or DEFAULT_NO_TOP_VARIANT_NAME
        variant_label = args.variant_label or DEFAULT_NO_TOP_VARIANT_LABEL
    else:
        variant_name = args.variant_name or DEFAULT_VARIANT_NAME
        variant_label = args.variant_label or DEFAULT_VARIANT_LABEL

    anomaly, pressure_levels, latitudes, longitudes = compute_theta_anomaly(
        temperature_path=temperature_path,
        climatology_path=climatology_path,
        timestamp=args.timestamp,
    )
    bucket_indices, level_metadata = standard_deviation_bucket_indices(anomaly)
    filtered_full_resolution_buckets, full_resolution_component_stats = (
        filter_full_resolution_by_stddev_side(
            bucket_indices,
            min_component_voxels=args.min_component_voxels,
        )
    )
    filtered_buckets, strided_latitudes, strided_longitudes = stride_mask(
        filtered_full_resolution_buckets,
        latitudes,
        longitudes,
        args.latitude_stride,
        args.longitude_stride,
    )
    build_variant(
        filtered_buckets=filtered_buckets,
        full_resolution_component_stats=full_resolution_component_stats,
        level_metadata=level_metadata,
        pressure_levels=pressure_levels,
        latitudes=strided_latitudes,
        longitudes=strided_longitudes,
        timestamp=args.timestamp,
        temperature_path=temperature_path,
        climatology_path=climatology_path,
        output_base_dir=output_base_dir,
        stats_dir=stats_dir,
        variant_name=variant_name,
        variant_label=variant_label,
        min_component_voxels=args.min_component_voxels,
        latitude_stride=args.latitude_stride,
        longitude_stride=args.longitude_stride,
        base_radius=args.base_radius,
        vertical_span=args.vertical_span,
        omit_top_faces=args.omit_top_faces,
    )
    print("Wrote export stats to", repo_path(stats_dir))


if __name__ == "__main__":
    main()
