from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

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
DEFAULT_TEMPERATURE_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY_PATH = Path(
    "data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc"
)
DEFAULT_OUTPUT_DIR = Path("public/air-mass-structures")
DEFAULT_STATS_DIR = Path("tmp/theta-anomaly-bucket-component-stats")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
REFERENCE_PRESSURE_HPA = 1000.0
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
PRESSURE_MIN_HPA = 250.0
PRESSURE_MAX_HPA = 1000.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 18.0

BUCKET_COLORS = {
    0: "#08306b",
    1: "#2171b5",
    2: "#6baed6",
    3: "#c6dbef",
    4: "#f7fbff",
    5: "#fff5f0",
    6: "#fcbba1",
    7: "#fb6a4a",
    8: "#cb181d",
    9: "#67000d",
}
RETAINED_BUCKETS = (0, 1, 2, 7, 8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export theta-climatology anomaly tail-bucket voxel shells as air-mass "
            "frontend variants, plus component-connectivity stats."
        )
    )
    parser.add_argument("--temperature", type=Path, default=DEFAULT_TEMPERATURE_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--latitude-stride", type=int, default=DEFAULT_LATITUDE_STRIDE)
    parser.add_argument("--longitude-stride", type=int, default=DEFAULT_LONGITUDE_STRIDE)
    parser.add_argument("--base-radius", type=float, default=DEFAULT_BASE_RADIUS)
    parser.add_argument("--vertical-span", type=float, default=DEFAULT_VERTICAL_SPAN)
    parser.add_argument(
        "--method",
        choices=("all", "percentile", "standard-deviation"),
        default="all",
    )
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


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


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="m").removesuffix("Z")


def select_time_index(dataset: xr.Dataset, timestamp: str) -> int:
    values = [
        timestamp_to_iso_minute(value)
        for value in np.asarray(dataset.coords["valid_time"].values, dtype="datetime64[m]")
    ]
    if timestamp not in values:
        raise KeyError(f"Timestamp {timestamp} not found in temperature dataset.")
    return values.index(timestamp)


def select_pressure_window(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keep = (pressure_levels_hpa >= PRESSURE_MIN_HPA) & (pressure_levels_hpa <= PRESSURE_MAX_HPA)
    if not np.any(keep):
        raise ValueError("No pressure levels found in the 1000-250 hPa window.")
    return np.asarray(field[keep], dtype=np.float32), np.asarray(pressure_levels_hpa[keep], dtype=np.float32)


def compute_theta_anomaly(
    *,
    temperature_path: Path,
    climatology_path: Path,
    timestamp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(temperature_path) as source, xr.open_dataset(climatology_path) as climatology:
        time_index = select_time_index(source, timestamp)
        pressure_levels = np.asarray(source.coords["pressure_level"].values, dtype=np.float32)
        climatology_pressures = np.asarray(
            climatology.coords["pressure_level"].values,
            dtype=np.float32,
        )
        if not np.array_equal(pressure_levels, climatology_pressures):
            raise ValueError("Temperature and climatology pressure levels do not match.")
        if not np.array_equal(source.coords["latitude"].values, climatology.coords["latitude"].values):
            raise ValueError("Temperature and climatology latitude grids do not match.")
        if not np.array_equal(source.coords["longitude"].values, climatology.coords["longitude"].values):
            raise ValueError("Temperature and climatology longitude grids do not match.")

        temperature = np.asarray(source["t"].isel(valid_time=time_index).values, dtype=np.float32)
        theta_climatology = np.asarray(climatology["theta_climatology_mean"].values, dtype=np.float32)
        temperature, selected_pressures = select_pressure_window(temperature, pressure_levels)
        theta_climatology, _ = select_pressure_window(theta_climatology, pressure_levels)
        latitudes = np.asarray(source.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(source.coords["longitude"].values, dtype=np.float32)

    pressure_scale = np.power(
        REFERENCE_PRESSURE_HPA / selected_pressures[:, None, None],
        POTENTIAL_TEMPERATURE_KAPPA,
        dtype=np.float32,
    )
    theta = np.asarray(temperature * pressure_scale, dtype=np.float32)
    return np.asarray(theta - theta_climatology, dtype=np.float32), selected_pressures, latitudes, longitudes


def stride_mask(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    latitude_stride: int,
    longitude_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_stride = max(int(latitude_stride), 1)
    lon_stride = max(int(longitude_stride), 1)
    return (
        np.asarray(values[:, ::lat_stride, ::lon_stride]),
        np.asarray(latitudes[::lat_stride], dtype=np.float32),
        np.asarray(longitudes[::lon_stride], dtype=np.float32),
    )


def percentile_bucket_indices(anomaly: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    buckets = np.full(anomaly.shape, -1, dtype=np.int8)
    level_metadata: list[dict[str, Any]] = []
    for level_index in range(anomaly.shape[0]):
        level = anomaly[level_index]
        finite = level[np.isfinite(level)]
        edges = np.percentile(finite, np.linspace(0.0, 100.0, 11))
        for edge_index in range(1, edges.size):
            if edges[edge_index] <= edges[edge_index - 1]:
                edges[edge_index] = np.nextafter(edges[edge_index - 1], np.inf)
        level_buckets = np.searchsorted(edges[1:-1], level, side="right").astype(np.int8)
        level_buckets[~np.isfinite(level)] = -1
        buckets[level_index] = level_buckets
        level_metadata.append(
            {
                "bucket_edges_k": [float(value) for value in edges],
                "retained_cell_count_full_resolution": int(
                    np.count_nonzero(np.isin(level_buckets, RETAINED_BUCKETS))
                ),
            }
        )
    return buckets, level_metadata


def standard_deviation_bucket_indices(anomaly: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    z_edges = np.asarray([-np.inf, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf])
    buckets = np.full(anomaly.shape, -1, dtype=np.int8)
    level_metadata: list[dict[str, Any]] = []
    for level_index in range(anomaly.shape[0]):
        level = anomaly[level_index]
        finite = level[np.isfinite(level)]
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        z_score = np.asarray((level - mean) / std, dtype=np.float32)
        level_buckets = np.searchsorted(z_edges[1:-1], z_score, side="right").astype(np.int8)
        level_buckets[~np.isfinite(z_score)] = -1
        level_buckets[np.isfinite(z_score) & (np.abs(z_score) <= 1.0)] = -1
        buckets[level_index] = level_buckets
        level_metadata.append(
            {
                "anomaly_mean_k": mean,
                "anomaly_std_k": std,
                "z_edges": ["-Infinity", -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, "Infinity"],
                "retained_cell_count_full_resolution": int(
                    np.count_nonzero(np.isin(level_buckets, RETAINED_BUCKETS))
                ),
            }
        )
    return buckets, level_metadata


def six_neighbor_structure() -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = 1
    structure[2, 1, 1] = 1
    structure[1, 0, 1] = 1
    structure[1, 2, 1] = 1
    structure[1, 1, 0] = 1
    structure[1, 1, 2] = 1
    return structure


def build_seam_merged_component_info(labels: np.ndarray, seam_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        root_map[max(first_root, second_root)] = min(first_root, second_root)

    for component_id in range(1, component_count + 1):
        root_map[component_id] = find_root(component_id)

    return root_map, np.unique(root_map[1:])


def label_wrapped_components(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0
    longitude_count = occupied.shape[2]
    extended = np.concatenate([occupied, occupied[..., :1]], axis=2)
    labels, component_count = ndimage.label(extended, structure=structure)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0
    seam_pairs = np.column_stack([labels[..., 0].reshape(-1), labels[..., -1].reshape(-1)])
    root_map, unique_root_ids = build_seam_merged_component_info(labels, seam_pairs)
    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_root_ids] = np.arange(1, unique_root_ids.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[..., :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_root_ids.size)


def component_summary(mask: np.ndarray, structure: np.ndarray) -> dict[str, Any]:
    labels, component_count = label_wrapped_components(mask, structure)
    return summarize_labeled_components(labels, component_count)


def histogram_counts(values: np.ndarray, bins: list[int]) -> dict[str, int]:
    if values.size == 0:
        return {
            f"{bins[index]}-{bins[index + 1] - 1}": 0
            for index in range(len(bins) - 1)
        } | {f">={bins[-1]}": 0}

    result: dict[str, int] = {}
    for index in range(len(bins) - 1):
        lower = bins[index]
        upper = bins[index + 1]
        result[f"{lower}-{upper - 1}"] = int(
            np.count_nonzero((values >= lower) & (values < upper))
        )
    result[f">={bins[-1]}"] = int(np.count_nonzero(values >= bins[-1]))
    return result


def summarize_labeled_components(labels: np.ndarray, component_count: int) -> dict[str, Any]:
    if component_count <= 0:
        return {
            "component_count": 0,
            "largest_component_voxel_count": 0,
            "mean_component_voxel_count": 0.0,
            "median_component_voxel_count": 0.0,
            "component_size_quantiles": {},
            "component_size_histogram": {},
            "single_level_component_count": 0,
            "multi_level_component_count": 0,
            "mean_pressure_span_levels": 0.0,
            "pressure_span_histogram": {},
            "small_component_counts": {},
        }

    sizes = np.bincount(labels[labels > 0].ravel(), minlength=component_count + 1)[1:]
    spans = np.zeros(component_count, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        occupied_levels = np.flatnonzero(np.any(labels == label_id, axis=(1, 2)))
        spans[label_id - 1] = int(occupied_levels.size)
    single_level_count = int(np.count_nonzero(spans <= 1))
    quantile_points = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantiles = np.quantile(sizes, quantile_points) if sizes.size else np.zeros(len(quantile_points))
    return {
        "component_count": int(component_count),
        "largest_component_voxel_count": int(sizes.max()) if sizes.size else 0,
        "mean_component_voxel_count": float(np.mean(sizes)) if sizes.size else 0.0,
        "median_component_voxel_count": float(np.median(sizes)) if sizes.size else 0.0,
        "component_size_quantiles": {
            f"p{int(point * 100):02d}": float(value)
            for point, value in zip(quantile_points, quantiles)
        },
        "component_size_histogram": histogram_counts(
            sizes.astype(np.int64),
            [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
        ),
        "single_level_component_count": int(single_level_count),
        "multi_level_component_count": int(component_count - single_level_count),
        "mean_pressure_span_levels": float(np.mean(spans)) if spans.size else 0.0,
        "pressure_span_histogram": histogram_counts(
            spans.astype(np.int64),
            [1, 2, 3, 4, 5, 8, 12, 16, 21],
        ),
        "small_component_counts": {
            "size_1": int(np.count_nonzero(sizes == 1)),
            "size_2_to_4": int(np.count_nonzero((sizes >= 2) & (sizes <= 4))),
            "size_5_to_24": int(np.count_nonzero((sizes >= 5) & (sizes <= 24))),
            "size_25_or_more": int(np.count_nonzero(sizes >= 25)),
        },
    }


def merge_summary_between_connectivities(
    mask: np.ndarray,
    six_structure: np.ndarray,
    full_structure: np.ndarray,
) -> dict[str, Any]:
    six_labels, six_count = label_wrapped_components(mask, six_structure)
    full_labels, full_count = label_wrapped_components(mask, full_structure)
    if six_count <= 0 or full_count <= 0:
        return {
            "six_component_count": int(six_count),
            "twenty_six_component_count": int(full_count),
            "merged_twenty_six_component_count": 0,
            "unchanged_twenty_six_component_count": 0,
            "mean_six_components_per_twenty_six_component": 0.0,
            "median_six_components_per_twenty_six_component": 0.0,
            "max_six_components_merged_into_one_twenty_six_component": 0,
            "six_components_per_twenty_six_component_histogram": {},
            "merged_six_component_size_summary": {},
            "unchanged_six_component_size_summary": {},
            "top_merges": [],
        }

    six_sizes = np.bincount(six_labels[six_labels > 0].ravel(), minlength=six_count + 1)
    full_sizes = np.bincount(full_labels[full_labels > 0].ravel(), minlength=full_count + 1)
    full_to_six: list[set[int]] = [set() for _ in range(full_count + 1)]
    active = (six_labels > 0) & (full_labels > 0)
    pairs = np.column_stack([full_labels[active].ravel(), six_labels[active].ravel()])
    if pairs.size:
        unique_pairs = np.unique(pairs, axis=0)
        for full_id, six_id in unique_pairs:
            full_to_six[int(full_id)].add(int(six_id))

    six_counts_per_full = np.asarray(
        [len(full_to_six[full_id]) for full_id in range(1, full_count + 1)],
        dtype=np.int64,
    )
    merged_full_ids = [
        full_id for full_id in range(1, full_count + 1) if len(full_to_six[full_id]) > 1
    ]
    merged_six_ids = sorted(
        {six_id for full_id in merged_full_ids for six_id in full_to_six[full_id]}
    )
    unchanged_six_ids = sorted(
        {next(iter(full_to_six[full_id])) for full_id in range(1, full_count + 1) if len(full_to_six[full_id]) == 1}
    )

    def size_summary(ids: list[int]) -> dict[str, Any]:
        component_sizes = np.asarray([six_sizes[component_id] for component_id in ids], dtype=np.int64)
        if component_sizes.size == 0:
            return {
                "component_count": 0,
                "mean_voxel_count": 0.0,
                "median_voxel_count": 0.0,
                "histogram": {},
            }
        return {
            "component_count": int(component_sizes.size),
            "mean_voxel_count": float(np.mean(component_sizes)),
            "median_voxel_count": float(np.median(component_sizes)),
            "histogram": histogram_counts(
                component_sizes,
                [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            ),
        }

    top_merges: list[dict[str, Any]] = []
    for full_id in sorted(
        merged_full_ids,
        key=lambda component_id: (
            len(full_to_six[component_id]),
            int(full_sizes[component_id]),
        ),
        reverse=True,
    )[:10]:
        member_sizes = sorted(
            [int(six_sizes[six_id]) for six_id in full_to_six[full_id]],
            reverse=True,
        )
        occupied_levels = np.flatnonzero(np.any(full_labels == full_id, axis=(1, 2)))
        top_merges.append(
            {
                "twenty_six_component_id": int(full_id),
                "twenty_six_component_voxel_count": int(full_sizes[full_id]),
                "six_component_count_merged": int(len(full_to_six[full_id])),
                "largest_member_six_component_voxel_count": int(member_sizes[0]) if member_sizes else 0,
                "member_six_component_voxel_counts_top10": member_sizes[:10],
                "pressure_span_levels": int(occupied_levels.size),
            }
        )

    return {
        "six_component_count": int(six_count),
        "twenty_six_component_count": int(full_count),
        "merged_twenty_six_component_count": int(len(merged_full_ids)),
        "unchanged_twenty_six_component_count": int(full_count - len(merged_full_ids)),
        "mean_six_components_per_twenty_six_component": float(np.mean(six_counts_per_full)),
        "median_six_components_per_twenty_six_component": float(np.median(six_counts_per_full)),
        "max_six_components_merged_into_one_twenty_six_component": int(
            np.max(six_counts_per_full)
        ),
        "six_components_per_twenty_six_component_histogram": histogram_counts(
            six_counts_per_full,
            [1, 2, 3, 5, 10, 25, 50, 100, 250],
        ),
        "merged_six_component_size_summary": size_summary(merged_six_ids),
        "unchanged_six_component_size_summary": size_summary(unchanged_six_ids),
        "top_merges": top_merges,
    }


def class_label(method: str, bucket_index: int) -> str:
    if method == "percentile":
        if bucket_index < 3:
            return f"Percentile bucket {bucket_index} cold tail"
        return f"Percentile bucket {bucket_index} warm tail"
    if bucket_index < 3:
        return f"Std-dev bucket {bucket_index} cold tail"
    return f"Std-dev bucket {bucket_index} warm tail"


def class_entries(method: str) -> list[dict[str, Any]]:
    return [
        {
            "key": f"bucket_{bucket_index}",
            "label": class_label(method, bucket_index),
            "bucket_index": int(bucket_index),
            "color": BUCKET_COLORS[bucket_index],
        }
        for bucket_index in RETAINED_BUCKETS
    ]


def maybe_flip_triangle_winding(indices: np.ndarray) -> np.ndarray:
    flipped = np.asarray(indices, dtype=np.uint32).copy()
    for index in range(0, flipped.size, 3):
        flipped[index + 1], flipped[index + 2] = flipped[index + 2], flipped[index + 1]
    return flipped


def build_bounds(mask: np.ndarray, pressure_levels: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {
            "pressure_min_hpa": float(np.min(pressure_levels)),
            "pressure_max_hpa": float(np.max(pressure_levels)),
            "latitude_min_deg": float(np.min(latitudes)),
            "latitude_max_deg": float(np.max(latitudes)),
            "longitude_min_deg": float(np.min(longitudes)),
            "longitude_max_deg": float(np.max(longitudes)),
        }
    return {
        "pressure_min_hpa": float(np.min(pressure_levels[coords[:, 0]])),
        "pressure_max_hpa": float(np.max(pressure_levels[coords[:, 0]])),
        "latitude_min_deg": float(np.min(latitudes[coords[:, 1]])),
        "latitude_max_deg": float(np.max(latitudes[coords[:, 1]])),
        "longitude_min_deg": float(np.min(longitudes[coords[:, 2]])),
        "longitude_max_deg": float(np.max(longitudes[coords[:, 2]])),
    }


def build_variant(
    *,
    method: str,
    buckets: np.ndarray,
    level_metadata: list[dict[str, Any]],
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamp: str,
    temperature_path: Path,
    climatology_path: Path,
    output_base_dir: Path,
    stats_dir: Path,
    latitude_stride: int,
    longitude_stride: int,
    base_radius: float,
    vertical_span: float,
) -> None:
    variant_name = f"theta-anomaly-{'percentile' if method == 'percentile' else 'stddev'}-tails"
    variant_label = (
        "Theta Percentile Tail Buckets"
        if method == "percentile"
        else "Theta Std Dev Tail Buckets"
    )
    output_dir = output_base_dir / "variants" / variant_name
    clear_output_dir(output_dir)

    frame_dir = output_dir / timestamp_to_slug(timestamp)
    frame_dir.mkdir(parents=True, exist_ok=True)

    six_structure = six_neighbor_structure()
    full_structure = np.ones((3, 3, 3), dtype=np.uint8)
    class_summaries: dict[str, Any] = {}
    class_counts: dict[str, Any] = {}
    connectivity_stats: dict[str, Any] = {
        "method": method,
        "variant": variant_name,
        "timestamp": timestamp,
        "pressure_window_hpa": {"min": PRESSURE_MIN_HPA, "max": PRESSURE_MAX_HPA},
        "grid": {
            "pressure_level_count": int(pressure_levels.size),
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_stride": int(latitude_stride),
            "longitude_stride": int(longitude_stride),
        },
        "buckets": {},
    }

    total_voxel_count = 0
    total_six_component_count = 0
    total_full_component_count = 0
    all_mask = np.zeros(buckets.shape, dtype=bool)

    for bucket_index in RETAINED_BUCKETS:
        class_key = f"bucket_{bucket_index}"
        mask = buckets == bucket_index
        all_mask |= mask
        voxel_count = int(np.count_nonzero(mask))
        six_summary = component_summary(mask, six_structure)
        full_summary = component_summary(mask, full_structure)
        merge_summary = merge_summary_between_connectivities(
            mask,
            six_structure,
            full_structure,
        )
        mesh = build_exposed_face_mesh_from_mask(
            keep_mask=mask,
            pressure_levels_hpa=pressure_levels,
            latitudes_deg=latitudes,
            longitudes_deg=longitudes,
        )

        positions_path = frame_dir / f"{class_key}_positions.bin"
        indices_path = frame_dir / f"{class_key}_indices.bin"
        np.asarray(mesh.positions, dtype="<f4").tofile(positions_path)
        maybe_flip_triangle_winding(np.asarray(mesh.indices, dtype=np.uint32)).astype("<u4").tofile(indices_path)

        class_summaries[class_key] = {
            "label": class_label(method, bucket_index),
            "bucket_index": int(bucket_index),
            "color": BUCKET_COLORS[bucket_index],
            "voxel_count": voxel_count,
            "component_count": int(six_summary["component_count"]),
            "largest_component_voxel_count": int(six_summary["largest_component_voxel_count"]),
            "positions_file": positions_path.relative_to(output_dir).as_posix(),
            "indices_file": indices_path.relative_to(output_dir).as_posix(),
            "vertex_count": int(mesh.vertex_count),
            "index_count": int(mesh.indices.size),
        }
        class_counts[class_key] = {
            "voxel_count": voxel_count,
            "component_count": int(six_summary["component_count"]),
        }
        connectivity_stats["buckets"][class_key] = {
            "bucket_index": int(bucket_index),
            "voxel_count": voxel_count,
            "six_neighbor": six_summary,
            "twenty_six_neighbor": full_summary,
            "merge_summary": merge_summary,
            "component_count_reduction": int(
                six_summary["component_count"] - full_summary["component_count"]
            ),
            "component_count_reduction_fraction": (
                float(
                    (six_summary["component_count"] - full_summary["component_count"])
                    / six_summary["component_count"]
                )
                if six_summary["component_count"] > 0
                else 0.0
            ),
        }
        total_voxel_count += voxel_count
        total_six_component_count += int(six_summary["component_count"])
        total_full_component_count += int(full_summary["component_count"])

    connectivity_stats["total"] = {
        "voxel_count": int(total_voxel_count),
        "six_neighbor_component_count": int(total_six_component_count),
        "twenty_six_neighbor_component_count": int(total_full_component_count),
        "component_count_reduction": int(total_six_component_count - total_full_component_count),
        "component_count_reduction_fraction": (
            float((total_six_component_count - total_full_component_count) / total_six_component_count)
            if total_six_component_count > 0
            else 0.0
        ),
    }
    connectivity_stats["total"]["bucket_count"] = int(len(RETAINED_BUCKETS))

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": timestamp,
        "voxel_count": int(total_voxel_count),
        "component_count": int(total_six_component_count),
        "component_connectivity": "6-neighbor with wrapped longitude; components are labeled separately per bucket.",
        "class_summaries": class_summaries,
        "pressure_levels_hpa": [float(value) for value in pressure_levels.tolist()],
        "score_thresholds_by_pressure_level": [
            {
                "pressure_hpa": float(pressure_levels[index]),
                **level_metadata[index],
            }
            for index in range(len(level_metadata))
        ],
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
        "selection": {
            "keep_top_percent": 60.0,
            "axis_min_abs_zscore": 1.0 if method == "standard-deviation" else 0.0,
            "score_basis": (
                "per-level_percentile_outer_30_percent"
                if method == "percentile"
                else "per-level_zscore_abs_greater_than_1"
            ),
            "bridge_gap_levels": 0,
            "min_component_voxels": 1,
            "min_component_pressure_span_levels": 1,
            "surface_attached_only": False,
            "classes": class_entries(method),
        },
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
            "score_basis": metadata["selection"]["score_basis"],
            "keep_top_percent": 60.0,
            "axis_min_abs_zscore": 1.0 if method == "standard-deviation" else 0.0,
            "bridge_gap_levels": 0,
            "min_component_voxels": 1,
            "min_component_pressure_span_levels": 1,
            "surface_attached_only": False,
            "classes": class_entries(method),
        },
        "sampling": {
            "latitude_stride": int(latitude_stride),
            "longitude_stride": int(longitude_stride),
            "method": "subsample_centers_after_full_resolution_bucket_edges",
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
                "component_count": int(total_six_component_count),
                "class_counts": class_counts,
            }
        ],
    }
    write_json(output_dir / "index.json", manifest)
    write_json(stats_dir / f"{variant_name}-connectivity-stats.json", connectivity_stats)
    print(
        "Built theta anomaly bucket variant:",
        variant_name,
        f"voxels={total_voxel_count}",
        f"components6={total_six_component_count}",
        f"components26={total_full_component_count}",
    )


def main() -> None:
    args = parse_args()
    temperature_path = (REPO_ROOT / args.temperature).resolve() if not args.temperature.is_absolute() else args.temperature.resolve()
    climatology_path = (REPO_ROOT / args.climatology).resolve() if not args.climatology.is_absolute() else args.climatology.resolve()
    output_base_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    stats_dir = (REPO_ROOT / args.stats_dir).resolve() if not args.stats_dir.is_absolute() else args.stats_dir.resolve()

    anomaly, pressure_levels, latitudes, longitudes = compute_theta_anomaly(
        temperature_path=temperature_path,
        climatology_path=climatology_path,
        timestamp=args.timestamp,
    )

    methods = ("percentile", "standard-deviation") if args.method == "all" else (args.method,)
    for method in methods:
        if method == "percentile":
            bucket_indices, level_metadata = percentile_bucket_indices(anomaly)
        else:
            bucket_indices, level_metadata = standard_deviation_bucket_indices(anomaly)
        strided_buckets, strided_latitudes, strided_longitudes = stride_mask(
            bucket_indices,
            latitudes,
            longitudes,
            args.latitude_stride,
            args.longitude_stride,
        )
        build_variant(
            method=method,
            buckets=strided_buckets,
            level_metadata=level_metadata,
            pressure_levels=pressure_levels,
            latitudes=strided_latitudes,
            longitudes=strided_longitudes,
            timestamp=args.timestamp,
            temperature_path=temperature_path,
            climatology_path=climatology_path,
            output_base_dir=output_base_dir,
            stats_dir=stats_dir,
            latitude_stride=args.latitude_stride,
            longitude_stride=args.longitude_stride,
            base_radius=args.base_radius,
            vertical_span=args.vertical_span,
        )

    print("Wrote connectivity stats to", repo_path(stats_dir))


if __name__ == "__main__":
    main()
