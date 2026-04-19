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
PLANAR_COMPONENT_STRUCTURE = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)
OUTPUT_VERSION = 8
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY_PATH = Path(
    "data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc"
)
DEFAULT_OUTPUT_DIR = Path("public/potential-temperature-structures")
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 18.0
DEFAULT_LATITUDE_STRIDE = 4
DEFAULT_LONGITUDE_STRIDE = 4
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0
DEFAULT_KEEP_TOP_PERCENT = 5.0
DEFAULT_SMOOTHING_SIGMA_CELLS = 1.0
DEFAULT_CONNECTION_MODE = "bridge-gap-1"
COMPONENT_CORE_SIGN_GROWTH_MODE = "top10-components-sign-growth"
CONNECTION_MODE_CHOICES = (
    "bridge-gap-1",
    "bridge-gap-2",
    "fill-between-anchors",
    COMPONENT_CORE_SIGN_GROWTH_MODE,
)


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
            "temperature, subtracts a matched climatological dry-potential-"
            "temperature mean field, keeps the hottest and coldest sign-specific "
            "tails on each pressure level, lightly smooths that signed anomaly "
            "field, and exports warm/cold 3D voxel-face structures with "
            "per-vertex anomaly magnitude and theta values."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 pressure-level temperature NetCDF file.",
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        default=DEFAULT_CLIMATOLOGY_PATH,
        help="Path to the dry-potential-temperature climatology NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated potential-temperature assets will be written.",
    )
    parser.add_argument(
        "--keep-top-percent",
        type=float,
        default=DEFAULT_KEEP_TOP_PERCENT,
        help=(
            "Keep the hottest N percent of positive anomalies and the coldest N "
            "percent of negative anomalies at each pressure level before "
            "smoothing. The default 5 means: keep the strongest 5 percent on "
            "each sign tail per level."
        ),
    )
    parser.add_argument(
        "--absolute-anomaly-percentile",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility alias for the percentile cutoff. "
            "For example, 50 here is equivalent to --keep-top-percent 50, while "
            "0 is equivalent to --keep-top-percent 100."
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
        "--connection-mode",
        type=str,
        choices=CONNECTION_MODE_CHOICES,
        default=DEFAULT_CONNECTION_MODE,
        help=(
            "Variant recipe applied after the climatology anomaly is built. "
            "The bridge/fill modes use the existing sign-tail threshold path, "
            "while top10-components-sign-growth uses an exact per-level top-"
            "percent absolute-anomaly core, keeps only the largest per-level "
            "components, and then grows columns upward and downward until the "
            "raw anomaly sign flips."
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


def load_climatology_theta_mean(
    climatology_path: Path,
    longitude_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(climatology_path)
    try:
        theta_mean = reorder_longitude_axis(
            np.asarray(dataset["theta_climatology_mean"].values, dtype=np.float32),
            longitude_order,
        )
        pressure_levels_hpa = np.asarray(
            dataset.coords["pressure_level"].values,
            dtype=np.float32,
        )
        latitudes_deg = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        longitudes_deg = np.asarray(
            dataset.coords["longitude"].values,
            dtype=np.float32,
        )
        return theta_mean, pressure_levels_hpa, latitudes_deg, reorder_longitude_axis(
            longitudes_deg[None, None, :],
            longitude_order,
        ).reshape(-1)
    finally:
        dataset.close()


def build_climatology_mean_anomaly(
    theta_field: np.ndarray,
    climatology_theta_mean: np.ndarray,
) -> np.ndarray:
    return np.asarray(theta_field - climatology_theta_mean, dtype=np.float32)


def resolve_keep_top_percent(
    keep_top_percent: float,
    absolute_anomaly_percentile: float | None,
) -> tuple[float, float]:
    if absolute_anomaly_percentile is not None:
        percentile = float(absolute_anomaly_percentile)
        if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
            raise ValueError(
                "absolute_anomaly_percentile must be a finite value between 0 and 100."
            )
        resolved_keep_top_percent = 100.0 - percentile
    else:
        resolved_keep_top_percent = float(keep_top_percent)

    if (
        not np.isfinite(resolved_keep_top_percent)
        or resolved_keep_top_percent < 0.0
        or resolved_keep_top_percent > 100.0
    ):
        raise ValueError("keep_top_percent must be a finite value between 0 and 100.")

    return resolved_keep_top_percent, 100.0 - resolved_keep_top_percent


def resolve_keep_count(total_count: int, keep_top_percent: float) -> int:
    total = max(int(total_count), 0)
    if total <= 0:
        return 0

    resolved_keep_top_percent, _ = resolve_keep_top_percent(
        keep_top_percent=keep_top_percent,
        absolute_anomaly_percentile=None,
    )
    if resolved_keep_top_percent <= 0.0:
        return 0
    if resolved_keep_top_percent >= 100.0:
        return total

    return max(1, min(total, int(np.rint(total * resolved_keep_top_percent / 100.0))))


def build_top_percent_selected_anomaly(
    anomaly: np.ndarray,
    keep_top_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    resolved_keep_top_percent, absolute_anomaly_percentile = resolve_keep_top_percent(
        keep_top_percent=keep_top_percent,
        absolute_anomaly_percentile=None,
    )

    absolute_anomaly = np.asarray(np.abs(anomaly), dtype=np.float32)
    thresholds = np.asarray(
        np.percentile(absolute_anomaly, absolute_anomaly_percentile, axis=(1, 2)),
        dtype=np.float32,
    )
    keep_mask = np.asarray(
        np.isfinite(anomaly) & (absolute_anomaly >= thresholds[:, None, None]),
        dtype=bool,
    )
    selected_anomaly = np.where(keep_mask, anomaly, 0.0).astype(np.float32)
    return selected_anomaly, keep_mask, thresholds, resolved_keep_top_percent


def build_exact_top_percent_selected_anomaly(
    anomaly: np.ndarray,
    keep_top_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    resolved_keep_top_percent, _ = resolve_keep_top_percent(
        keep_top_percent=keep_top_percent,
        absolute_anomaly_percentile=None,
    )
    absolute_anomaly = np.asarray(np.abs(anomaly), dtype=np.float32)
    keep_mask = np.zeros_like(anomaly, dtype=bool)
    thresholds = np.full(anomaly.shape[0], np.nan, dtype=np.float32)
    selected_cell_counts = np.zeros(anomaly.shape[0], dtype=np.int32)

    for level_index in range(anomaly.shape[0]):
        level_values = np.asarray(absolute_anomaly[level_index], dtype=np.float32)
        finite_indices = np.flatnonzero(np.isfinite(level_values))
        keep_count = resolve_keep_count(finite_indices.size, resolved_keep_top_percent)
        selected_cell_counts[level_index] = keep_count
        if keep_count <= 0:
            continue

        finite_values = level_values.reshape(-1)[finite_indices]
        if keep_count >= finite_indices.size:
            selected_indices = finite_indices
        else:
            partition_indices = np.argpartition(finite_values, -keep_count)[-keep_count:]
            selected_indices = finite_indices[partition_indices]

        flat_mask = keep_mask[level_index].reshape(-1)
        flat_mask[selected_indices] = True
        selected_values = level_values.reshape(-1)[selected_indices]
        thresholds[level_index] = float(np.min(selected_values))

    selected_anomaly = np.where(keep_mask, anomaly, 0.0).astype(np.float32)
    return (
        selected_anomaly,
        keep_mask,
        thresholds,
        selected_cell_counts,
        resolved_keep_top_percent,
    )


def build_sign_tail_selected_anomaly(
    anomaly: np.ndarray,
    keep_top_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    resolved_keep_top_percent, _ = resolve_keep_top_percent(
        keep_top_percent=keep_top_percent,
        absolute_anomaly_percentile=None,
    )
    upper_percentile = 100.0 - resolved_keep_top_percent
    hot_thresholds = np.full(anomaly.shape[0], np.nan, dtype=np.float32)
    cold_thresholds = np.full(anomaly.shape[0], np.nan, dtype=np.float32)
    hot_keep_mask = np.zeros_like(anomaly, dtype=bool)
    cold_keep_mask = np.zeros_like(anomaly, dtype=bool)

    for level_index in range(anomaly.shape[0]):
        level_values = np.asarray(anomaly[level_index], dtype=np.float32)
        finite_mask = np.isfinite(level_values)

        positive_values = level_values[finite_mask & (level_values > 0.0)]
        if positive_values.size > 0:
            hot_threshold = float(np.percentile(positive_values, upper_percentile))
            hot_thresholds[level_index] = hot_threshold
            hot_keep_mask[level_index] = (
                finite_mask & (level_values > 0.0) & (level_values >= hot_threshold)
            )

        negative_values = level_values[finite_mask & (level_values < 0.0)]
        if negative_values.size > 0:
            cold_threshold = float(np.percentile(negative_values, resolved_keep_top_percent))
            cold_thresholds[level_index] = cold_threshold
            cold_keep_mask[level_index] = (
                finite_mask & (level_values < 0.0) & (level_values <= cold_threshold)
            )

    keep_mask = np.asarray(hot_keep_mask | cold_keep_mask, dtype=bool)
    selected_anomaly = np.where(keep_mask, anomaly, 0.0).astype(np.float32)
    return (
        selected_anomaly,
        keep_mask,
        hot_thresholds,
        cold_thresholds,
        np.asarray(np.fmax(np.abs(hot_thresholds), np.abs(cold_thresholds)), dtype=np.float32),
        resolved_keep_top_percent,
    )


def vertical_connection_mode_label(connection_mode: str) -> str:
    labels = {
        "bridge-gap-1": "bridge_one_missing_level",
        "bridge-gap-2": "bridge_up_to_two_missing_levels",
        "fill-between-anchors": "fill_between_same_sign_anchors",
        COMPONENT_CORE_SIGN_GROWTH_MODE: "grow_same_sign_columns_from_top10_component_core",
    }
    return labels.get(connection_mode, connection_mode)


def apply_vertical_connection_mode(
    anomaly: np.ndarray,
    selected_anomaly: np.ndarray,
    connection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if connection_mode not in CONNECTION_MODE_CHOICES:
        raise ValueError(f"Unsupported connection mode: {connection_mode}")

    if connection_mode == "bridge-gap-1":
        max_gap: int | None = 1
    elif connection_mode == "bridge-gap-2":
        max_gap = 2
    else:
        max_gap = None

    filled = np.asarray(selected_anomaly, dtype=np.float32).copy()
    original_mask = np.asarray(np.abs(selected_anomaly) > 0.0, dtype=bool)
    filled_mask = original_mask.copy()
    original_sign = np.sign(selected_anomaly).astype(np.int8)
    raw_sign = np.sign(anomaly).astype(np.int8)

    _, latitude_count, longitude_count = anomaly.shape
    for latitude_index in range(latitude_count):
        for longitude_index in range(longitude_count):
            for sign_value in (-1, 1):
                anchor_levels = np.flatnonzero(
                    original_sign[:, latitude_index, longitude_index] == sign_value
                )
                if anchor_levels.size < 2:
                    continue

                for start_level, end_level in zip(
                    anchor_levels[:-1],
                    anchor_levels[1:],
                    strict=True,
                ):
                    gap_size = int(end_level - start_level - 1)
                    if gap_size <= 0:
                        continue
                    if max_gap is not None and gap_size > max_gap:
                        continue

                    gap_slice = slice(start_level + 1, end_level)
                    gap_signs = raw_sign[gap_slice, latitude_index, longitude_index]
                    if gap_signs.size == 0:
                        continue
                    if np.any(
                        original_sign[gap_slice, latitude_index, longitude_index]
                        == -sign_value
                    ):
                        continue
                    if not np.all(gap_signs == sign_value):
                        continue

                    gap_mask = ~filled_mask[gap_slice, latitude_index, longitude_index]
                    if not np.any(gap_mask):
                        continue

                    filled_values = anomaly[gap_slice, latitude_index, longitude_index]
                    filled_segment = filled[gap_slice, latitude_index, longitude_index]
                    filled_segment[gap_mask] = filled_values[gap_mask]
                    filled[gap_slice, latitude_index, longitude_index] = filled_segment
                    filled_mask[gap_slice, latitude_index, longitude_index] |= gap_mask

    added_mask = np.asarray(filled_mask & ~original_mask, dtype=bool)
    return filled.astype(np.float32), added_mask


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


def remove_single_level_components(
    smoothed_anomaly_values: np.ndarray,
    sign_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    filtered_anomaly_values = np.asarray(smoothed_anomaly_values, dtype=np.float32).copy()
    filtered_sign_field = np.asarray(sign_field, dtype=np.int8).copy()
    summary = {
        "removed_component_count": 0,
        "removed_positive_component_count": 0,
        "removed_negative_component_count": 0,
        "removed_voxel_count": 0,
        "removed_positive_voxel_count": 0,
        "removed_negative_voxel_count": 0,
    }

    for sign_value, component_key, voxel_key in (
        (1, "removed_positive_component_count", "removed_positive_voxel_count"),
        (-1, "removed_negative_component_count", "removed_negative_voxel_count"),
    ):
        labels, component_count = label_wrapped_volume_components(filtered_sign_field == sign_value)
        for label_id in range(1, component_count + 1):
            component_mask = labels == label_id
            if not np.any(component_mask):
                continue

            pressure_span_levels = np.unique(np.argwhere(component_mask)[:, 0]).size
            if pressure_span_levels > 1:
                continue

            removed_voxels = int(np.count_nonzero(component_mask))
            filtered_anomaly_values[component_mask] = 0.0
            filtered_sign_field[component_mask] = 0
            summary["removed_component_count"] += 1
            summary[component_key] += 1
            summary["removed_voxel_count"] += removed_voxels
            summary[voxel_key] += removed_voxels

    return filtered_anomaly_values, filtered_sign_field, summary


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


def label_wrapped_planar_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("Planar component labeling expects a 2D mask.")
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    longitude_count = occupied.shape[1]
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(extended, structure=PLANAR_COMPONENT_STRUCTURE)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    seam_pairs = np.column_stack([labels[:, 0].reshape(-1), labels[:, -1].reshape(-1)])
    root_map, unique_root_ids = build_seam_merged_component_info(labels, seam_pairs)
    if unique_root_ids.size == 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_root_ids] = np.arange(1, unique_root_ids.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[:, :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_root_ids.size)


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


def keep_largest_planar_component_share_per_level(
    anomaly: np.ndarray,
    keep_mask: np.ndarray,
    component_keep_top_percent: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    core_mask = np.zeros_like(keep_mask, dtype=bool)
    level_metadata: list[dict[str, Any]] = []

    for level_index in range(keep_mask.shape[0]):
        level_labels, component_count = label_wrapped_planar_components(keep_mask[level_index])
        keep_component_count = resolve_keep_count(component_count, component_keep_top_percent)

        if component_count > 0:
            component_sizes = np.bincount(level_labels[level_labels > 0].ravel())[1:]
            component_order = np.argsort(-component_sizes, kind="stable")
            kept_component_ids = component_order[:keep_component_count] + 1
            core_mask[level_index] = np.isin(level_labels, kept_component_ids)
            kept_component_threshold = (
                int(component_sizes[component_order[keep_component_count - 1]])
                if keep_component_count > 0
                else 0
            )
            largest_kept_component_size = (
                int(component_sizes[component_order[0]]) if component_order.size > 0 else 0
            )
        else:
            component_sizes = np.zeros(0, dtype=np.int32)
            kept_component_threshold = 0
            largest_kept_component_size = 0

        level_metadata.append(
            {
                "component_count": int(component_count),
                "kept_component_count": int(keep_component_count),
                "kept_component_size_threshold": int(kept_component_threshold),
                "largest_kept_component_size": int(largest_kept_component_size),
                "largest_component_size": int(component_sizes.max()) if component_sizes.size else 0,
            }
        )

    core_anomaly = np.where(core_mask, anomaly, 0.0).astype(np.float32)
    return core_anomaly, core_mask, level_metadata


def grow_core_columns_by_sign(
    anomaly: np.ndarray,
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign_field = np.sign(np.asarray(anomaly, dtype=np.float32)).astype(np.int8)
    grown_mask = np.asarray(core_mask, dtype=bool).copy()
    pressure_count, latitude_count, longitude_count = anomaly.shape

    for latitude_index in range(latitude_count):
        for longitude_index in range(longitude_count):
            core_levels = np.flatnonzero(core_mask[:, latitude_index, longitude_index])
            for level_index in core_levels:
                sign_value = int(sign_field[level_index, latitude_index, longitude_index])
                if sign_value == 0:
                    continue

                for direction in (-1, 1):
                    candidate_level = int(level_index + direction)
                    while 0 <= candidate_level < pressure_count:
                        if int(sign_field[candidate_level, latitude_index, longitude_index]) != sign_value:
                            break
                        grown_mask[candidate_level, latitude_index, longitude_index] = True
                        candidate_level += direction

    added_mask = np.asarray(grown_mask & ~core_mask, dtype=bool)
    grown_anomaly = np.where(grown_mask, anomaly, 0.0).astype(np.float32)
    return grown_anomaly, added_mask


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
    selected_voxel_count_before_connection_fill: int,
    smoothed_anomaly_values: np.ndarray,
    sign_field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    keep_top_percent: float,
    absolute_anomaly_percentile: float,
    smoothing_sigma_cells: float,
    variant_name: str,
    selection_thresholds_by_pressure_level: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    base_radius: float,
    vertical_span: float,
    connection_fill_mask: np.ndarray,
    single_level_removal_summary: dict[str, int],
    extra_metadata: dict[str, Any] | None = None,
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
        "selected_voxel_count_before_connection_fill": int(
            selected_voxel_count_before_connection_fill
        ),
        "connection_fill_voxel_count": int(np.count_nonzero(connection_fill_mask)),
        "selected_voxel_count_before_smoothing": int(
            selected_voxel_count_before_connection_fill
            + np.count_nonzero(connection_fill_mask)
        ),
        "single_level_component_min_span_levels": int(
            selection_summary.get("min_component_pressure_span_levels", 2)
        ),
        **single_level_removal_summary,
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
        "keep_top_percent": float(keep_top_percent),
        "absolute_anomaly_percentile": float(absolute_anomaly_percentile),
        "smoothing_sigma_cells": float(smoothing_sigma_cells),
        "connection_mode": variant_name,
        "selection": {
            "kept_signs": ["negative", "positive"],
            "keep_top_percent": float(keep_top_percent),
            "absolute_anomaly_percentile": float(absolute_anomaly_percentile),
            **selection_summary,
            "thresholds_by_pressure_level": selection_thresholds_by_pressure_level,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return SignedAnomalyShellAssetPayload(
        timestamp=timestamp,
        warm_positions=warm_mesh.positions,
        warm_indices=maybe_flip_triangle_winding(warm_mesh.indices),
        cold_positions=cold_mesh.positions,
        cold_indices=maybe_flip_triangle_winding(cold_mesh.indices),
        voxel_count=total_voxel_count,
        metadata=metadata,
    )


def maybe_flip_triangle_winding(indices: np.ndarray) -> np.ndarray:
    normalized = np.asarray(indices, dtype=np.uint32).copy()
    if OUTPUT_VERSION < 8:
        return normalized

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
    climatology_dataset_name: str,
    entries: list[dict[str, Any]],
    variant_name: str,
    threshold_basis: str,
    keep_top_percent: float,
    absolute_anomaly_percentile: float,
    smoothing_sigma_cells: float,
    selection_summary: dict[str, Any],
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
        "climatology_dataset": climatology_dataset_name,
        "variable": TEMPERATURE_VARIABLE,
        "units": contents.units,
        "variant": variant_name,
        "derived_variable": {
            "name": DERIVED_VARIABLE_NAME,
            "units": DERIVED_UNITS,
            "reference_pressure_hpa": REFERENCE_PRESSURE_HPA,
            "kappa": POTENTIAL_TEMPERATURE_KAPPA,
        },
        "structure_kind": "potential-temperature-climatology-anomaly-shell",
        "geometry_mode": "voxel-faces",
        "selection": {
            "background": "matched_gridpoint_climatological_theta_mean",
            "threshold_basis": threshold_basis,
            "keep_top_percent": float(keep_top_percent),
            "absolute_anomaly_percentile": float(absolute_anomaly_percentile),
            "smoothing_sigma_cells": float(smoothing_sigma_cells),
            "keep_signs": ["negative", "positive"],
            **selection_summary,
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
    variant_name = args.connection_mode
    is_component_core_sign_growth = variant_name == COMPONENT_CORE_SIGN_GROWTH_MODE
    keep_top_percent, absolute_anomaly_percentile = resolve_keep_top_percent(
        keep_top_percent=args.keep_top_percent,
        absolute_anomaly_percentile=args.absolute_anomaly_percentile,
    )
    effective_smoothing_sigma_cells = (
        0.0 if is_component_core_sign_growth else float(args.smoothing_sigma_cells)
    )
    dataset_path = resolve_dataset_path(args.dataset)
    climatology_path = resolve_dataset_path(args.climatology)
    contents = load_dataset_contents(dataset_path)
    (
        climatology_theta_mean,
        climatology_pressure_levels_hpa,
        climatology_latitudes_deg,
        climatology_longitudes_deg,
    ) = load_climatology_theta_mean(
        climatology_path,
        contents.longitude_order,
    )
    if not np.array_equal(climatology_pressure_levels_hpa, contents.pressure_levels_hpa):
        raise ValueError(
            "Climatology pressure levels do not match the source temperature dataset."
        )
    if not np.array_equal(climatology_latitudes_deg, contents.latitudes_deg):
        raise ValueError(
            "Climatology latitudes do not match the source temperature dataset."
        )
    if not np.array_equal(climatology_longitudes_deg, contents.longitudes_deg):
        raise ValueError(
            "Climatology longitudes do not match the source temperature dataset."
        )
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
            climatology_pressure_window, _ = select_pressure_window(
                climatology_theta_mean,
                contents.pressure_levels_hpa,
                pressure_min_hpa=args.pressure_min_hpa,
                pressure_max_hpa=args.pressure_max_hpa,
            )
            anomaly = build_climatology_mean_anomaly(
                theta_field,
                climatology_pressure_window[:, :: max(int(args.latitude_stride), 1), :: max(int(args.longitude_stride), 1)],
            )
            selection_summary = {
                "min_component_pressure_span_levels": 2,
            }
            extra_metadata: dict[str, Any] = {}

            if is_component_core_sign_growth:
                (
                    selected_top_percent_anomaly,
                    selected_top_percent_mask,
                    absolute_anomaly_thresholds_by_level,
                    selected_cell_counts_by_level,
                    resolved_keep_top_percent,
                ) = build_exact_top_percent_selected_anomaly(
                    anomaly,
                    keep_top_percent=keep_top_percent,
                )
                (
                    core_anomaly,
                    core_mask,
                    component_level_metadata,
                ) = keep_largest_planar_component_share_per_level(
                    anomaly,
                    selected_top_percent_mask,
                    component_keep_top_percent=resolved_keep_top_percent,
                )
                selected_anomaly, connection_fill_mask = grow_core_columns_by_sign(
                    anomaly,
                    core_mask,
                )
                smoothed_anomaly_values = np.asarray(selected_anomaly, dtype=np.float32)
                sign_field = np.sign(smoothed_anomaly_values).astype(np.int8)
                single_level_removal_summary = {
                    "removed_component_count": 0,
                    "removed_positive_component_count": 0,
                    "removed_negative_component_count": 0,
                    "removed_voxel_count": 0,
                    "removed_positive_voxel_count": 0,
                    "removed_negative_voxel_count": 0,
                }
                selection_thresholds_by_pressure_level = [
                    {
                        "pressure_hpa": float(pressure_hpa),
                        "absolute_anomaly_threshold": float(absolute_threshold),
                        "selected_cell_count": int(selected_cell_count),
                        "component_count": int(level_meta["component_count"]),
                        "kept_component_count": int(level_meta["kept_component_count"]),
                        "kept_component_size_threshold": int(
                            level_meta["kept_component_size_threshold"]
                        ),
                        "largest_component_size": int(level_meta["largest_component_size"]),
                        "largest_kept_component_size": int(
                            level_meta["largest_kept_component_size"]
                        ),
                    }
                    for pressure_hpa, absolute_threshold, selected_cell_count, level_meta in zip(
                        pressure_window_levels_hpa.tolist(),
                        absolute_anomaly_thresholds_by_level.tolist(),
                        selected_cell_counts_by_level.tolist(),
                        component_level_metadata,
                        strict=True,
                    )
                ]
                selection_summary.update(
                    {
                        "threshold_basis": "per-level_absolute-anomaly_top-percent_then_top-component-share",
                        "component_keep_top_percent": float(resolved_keep_top_percent),
                        "core_component_connectivity": "4-connected-with-longitude-wrap",
                        "vertical_connection_mode": variant_name,
                        "vertical_connection_label": vertical_connection_mode_label(
                            variant_name
                        ),
                        "min_component_pressure_span_levels": 1,
                    }
                )
                extra_metadata.update(
                    {
                        "selected_voxel_count_before_component_filter": int(
                            np.count_nonzero(selected_top_percent_mask)
                        ),
                        "core_voxel_count": int(np.count_nonzero(core_mask)),
                    }
                )
                selected_voxel_count_before_connection_fill = int(np.count_nonzero(core_mask))
            else:
                (
                    selected_top_percent_anomaly,
                    _,
                    hot_anomaly_thresholds_by_level,
                    cold_anomaly_thresholds_by_level,
                    _,
                    resolved_keep_top_percent,
                ) = build_sign_tail_selected_anomaly(
                    anomaly,
                    keep_top_percent=keep_top_percent,
                )
                selected_anomaly, connection_fill_mask = apply_vertical_connection_mode(
                    anomaly,
                    selected_top_percent_anomaly,
                    connection_mode=variant_name,
                )
                smoothed_anomaly_values, sign_field = smooth_selected_sign_anomaly(
                    selected_anomaly,
                    smoothing_sigma_cells=effective_smoothing_sigma_cells,
                )
                (
                    smoothed_anomaly_values,
                    sign_field,
                    single_level_removal_summary,
                ) = remove_single_level_components(
                    smoothed_anomaly_values,
                    sign_field,
                )
                selection_thresholds_by_pressure_level = [
                    {
                        "pressure_hpa": float(pressure_hpa),
                        "hot_anomaly_threshold": float(hot_threshold),
                        "cold_anomaly_threshold": float(cold_threshold),
                    }
                    for pressure_hpa, hot_threshold, cold_threshold in zip(
                        pressure_window_levels_hpa.tolist(),
                        hot_anomaly_thresholds_by_level.tolist(),
                        cold_anomaly_thresholds_by_level.tolist(),
                        strict=True,
                    )
                ]
                selection_summary.update(
                    {
                        "vertical_connection_mode": variant_name,
                        "vertical_connection_label": vertical_connection_mode_label(
                            variant_name
                        ),
                    }
                )
                selected_voxel_count_before_connection_fill = int(
                    np.count_nonzero(np.abs(selected_top_percent_anomaly) > 0.0)
                )
            if not np.any(sign_field):
                print(
                    "Skipped potential temperature frame after smoothing:",
                    timestamp,
                    f"keep_top_percent={resolved_keep_top_percent}",
                    f"absolute_anomaly_percentile={absolute_anomaly_percentile}",
                    f"smoothing_sigma_cells={effective_smoothing_sigma_cells}",
                )
                continue

            payload = build_asset_payload(
                timestamp=timestamp,
                theta_field=theta_field,
                selected_anomaly=selected_anomaly,
                selected_voxel_count_before_connection_fill=selected_voxel_count_before_connection_fill,
                smoothed_anomaly_values=smoothed_anomaly_values,
                sign_field=sign_field,
                pressure_levels_hpa=pressure_window_levels_hpa,
                latitudes_deg=strided_latitudes_deg,
                longitudes_deg=strided_longitudes_deg,
                keep_top_percent=resolved_keep_top_percent,
                absolute_anomaly_percentile=absolute_anomaly_percentile,
                smoothing_sigma_cells=effective_smoothing_sigma_cells,
                variant_name=variant_name,
                selection_thresholds_by_pressure_level=selection_thresholds_by_pressure_level,
                selection_summary=selection_summary,
                base_radius=args.base_radius,
                vertical_span=args.vertical_span,
                connection_fill_mask=connection_fill_mask,
                single_level_removal_summary=single_level_removal_summary,
                extra_metadata=extra_metadata,
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
                f"connection_mode={variant_name}",
                f"filled={payload.metadata['connection_fill_voxel_count']}",
                f"removed_single_level={payload.metadata['removed_component_count']}",
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
        climatology_dataset_name=climatology_path.name,
        entries=entries,
        variant_name=variant_name,
        threshold_basis=(
            "per-level_absolute-anomaly_top-percent_then_top-component-share"
            if is_component_core_sign_growth
            else "per-level_sign-tail_top-percent"
        ),
        keep_top_percent=keep_top_percent,
        absolute_anomaly_percentile=absolute_anomaly_percentile,
        smoothing_sigma_cells=effective_smoothing_sigma_cells,
        selection_summary=(
            {
                "component_keep_top_percent": float(keep_top_percent),
                "core_component_connectivity": "4-connected-with-longitude-wrap",
                "vertical_connection_mode": variant_name,
                "vertical_connection_label": vertical_connection_mode_label(variant_name),
                "min_component_pressure_span_levels": 1,
            }
            if is_component_core_sign_growth
            else {
                "vertical_connection_mode": variant_name,
                "vertical_connection_label": vertical_connection_mode_label(variant_name),
                "min_component_pressure_span_levels": 2,
            }
        ),
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
