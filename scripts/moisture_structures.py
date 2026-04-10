from __future__ import annotations

import json
import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import netCDF4
import xarray as xr
from scipy import ndimage
from skimage import measure, morphology

DATASET_VARIABLE = "q"
OUTPUT_VERSION = 1
DEFAULT_BASE_RADIUS = 100.0
DEFAULT_VERTICAL_SPAN = 12.0
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_MIN_COMPONENT_SIZE = 1_024
DEFAULT_TIME_WINDOW = 7
DEFAULT_GAUSSIAN_SIGMA = 0.6
DEFAULT_THRESHOLD_SAMPLE_STRIDE = 1
DEFAULT_THRESHOLD_TIME_STRIDE = 1
DEFAULT_GEOMETRY_MODE = "marching-cubes"
SIGNED_DISTANCE_GEOMETRY_MODE = "signed-distance"
COLUMN_ENVELOPE_GEOMETRY_MODE = "column-envelope"
DEFAULT_CLOSING_RADIUS_CELLS = 1
DEFAULT_OPENING_RADIUS_CELLS = 1
DEFAULT_SEGMENTATION_MODE = "p95-close"
VOXEL_SHELL_SEGMENTATION_MODE = "p95-close-voxel-shell"
RAW_VOXEL_SHELL_SEGMENTATION_MODE = "p95-raw-voxel-shell"
SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE = "p95-smooth-open1-voxel-shell"
GEOMETRY_SMOOTHED_SEGMENTATION_MODE = "p95-close-smoothmesh"
LEGACY_BRIDGE_PRUNED_SEGMENTATION_MODE = "p95-close-open1"
SMOOTH_SUPPORT_SIGMA = 1.0
LOCAL_ANOMALY_BACKGROUND_SIGMA = 8.0
P97_CLOSE_QUANTILE = 0.97
LOCAL_ANOMALY_RAW_QUANTILE = 0.90
LOCAL_ANOMALY_ANOMALY_QUANTILE = 0.95
BUCKET_SEGMENTATION_MODE = "buckets"
BUCKET_GLOBAL_SEGMENTATION_MODE = "buckets-global"
BUCKET_COUNT = 10
BUCKET_FIELD_LATITUDE_STRIDE = 6
BUCKET_FIELD_LONGITUDE_STRIDE = 6
BUCKET_EDGE_QUANTILES = tuple(index / BUCKET_COUNT for index in range(1, BUCKET_COUNT))
BUCKET_LEVEL_SMOOTHING_SIGMA = 0.75
BUCKET_LATITUDE_SMOOTHING_SIGMA = 3.0
BUCKET_LONGITUDE_SMOOTHING_SIGMA = 3.0
SUPPORTED_SEGMENTATION_MODES = (
    "p95-close",
    VOXEL_SHELL_SEGMENTATION_MODE,
    RAW_VOXEL_SHELL_SEGMENTATION_MODE,
    SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE,
    GEOMETRY_SMOOTHED_SEGMENTATION_MODE,
    LEGACY_BRIDGE_PRUNED_SEGMENTATION_MODE,
    "p95-smooth-open1",
    "p95-local-anomaly",
    "p95-open",
    "p97-close",
    BUCKET_SEGMENTATION_MODE,
    BUCKET_GLOBAL_SEGMENTATION_MODE,
)
SEGMENTATION_GEOMETRY_MODE_OVERRIDES = {
    VOXEL_SHELL_SEGMENTATION_MODE: "voxel-faces",
    RAW_VOXEL_SHELL_SEGMENTATION_MODE: "voxel-faces",
    SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE: "voxel-faces",
    GEOMETRY_SMOOTHED_SEGMENTATION_MODE: "marching-cubes",
}
LABEL_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class BuildConfig:
    dataset_path: Path
    output_dir: Path
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE
    time_window: int = DEFAULT_TIME_WINDOW
    base_radius: float = DEFAULT_BASE_RADIUS
    vertical_span: float = DEFAULT_VERTICAL_SPAN
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA
    threshold_time_stride: int = DEFAULT_THRESHOLD_TIME_STRIDE
    threshold_sample_stride: int = DEFAULT_THRESHOLD_SAMPLE_STRIDE
    geometry_mode: str = DEFAULT_GEOMETRY_MODE
    closing_radius_cells: int = DEFAULT_CLOSING_RADIUS_CELLS
    opening_radius_cells: int = DEFAULT_OPENING_RADIUS_CELLS
    segmentation_mode: str = DEFAULT_SEGMENTATION_MODE
    write_footprints: bool = True
    limit_timestamps: int | None = None
    include_timestamps: tuple[str, ...] | None = None
    mesh_smoothing_iterations: int = 0
    mesh_smoothing_lambda: float = 0.45
    mesh_smoothing_mu: float = -0.47
    mesh_island_min_vertices: int = 0
    voxel_exterior_only: bool = False


@dataclass(frozen=True)
class SegmentationContext:
    segmentation_mode: str
    primary_thresholds: np.ndarray
    threshold_tables: dict[str, np.ndarray]
    closing_radius_cells: int
    opening_radius_cells: int
    threshold_quantile: float
    threshold_kind: str
    recipe_metadata: dict[str, Any]
    bucket_count: int = 0
    latitude_stride: int = 1
    longitude_stride: int = 1


def resolve_geometry_mode(segmentation_mode: str, requested_geometry_mode: str) -> str:
    return SEGMENTATION_GEOMETRY_MODE_OVERRIDES.get(
        segmentation_mode,
        requested_geometry_mode,
    )


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if not text.endswith("Z"):
        return text
    return text[:-1]


def timestamp_to_slug(timestamp: str) -> str:
    return f"{timestamp.replace(':', '-')}-00"


def lat_lon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(-(lon_deg + 270.0))
    x = radius * math.cos(lat) * math.cos(lon)
    y = radius * math.sin(lat)
    z = radius * math.cos(lat) * math.sin(lon)
    return np.array([x, y, z], dtype=np.float32)


def pressure_to_standard_atmosphere_height_m(pressure_hpa: np.ndarray) -> np.ndarray:
    safe_pressure = np.maximum(np.asarray(pressure_hpa, dtype=np.float64), 1.0)
    return 44330.0 * (1.0 - np.power(safe_pressure / 1013.25, 0.1903))


def build_radius_lookup(
    pressure_levels_hpa: np.ndarray,
    base_radius: float = DEFAULT_BASE_RADIUS,
    vertical_span: float = DEFAULT_VERTICAL_SPAN,
) -> np.ndarray:
    heights = pressure_to_standard_atmosphere_height_m(pressure_levels_hpa)
    min_height = float(pressure_to_standard_atmosphere_height_m(np.array([1000.0]))[0])
    max_height = float(pressure_to_standard_atmosphere_height_m(np.array([1.0]))[0])
    scale = vertical_span / max(max_height - min_height, 1e-9)
    return (base_radius + (heights - min_height) * scale).astype(np.float32)


def compute_per_level_thresholds_from_array(
    values: np.ndarray,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    value_mask: np.ndarray | None = None,
) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError("Expected values with shape (time, level, ...).")
    if value_mask is not None and value_mask.shape != values.shape:
        raise ValueError("Expected value_mask to match values.")

    thresholds = []
    for level_index in range(values.shape[1]):
        level_values = np.asarray(values[:, level_index, ...], dtype=np.float32).ravel()
        finite_mask = np.isfinite(level_values)
        if value_mask is not None:
            level_mask = np.asarray(
                value_mask[:, level_index, ...],
                dtype=bool,
            ).ravel()
            finite_mask &= level_mask
        finite_values = level_values[finite_mask]
        if finite_values.size == 0:
            thresholds.append(np.nan)
            continue
        thresholds.append(float(np.quantile(finite_values, quantile)))
    return np.asarray(thresholds, dtype=np.float32)


def compute_per_level_bucket_boundaries_from_array(
    values: np.ndarray,
    quantiles: tuple[float, ...] = BUCKET_EDGE_QUANTILES,
    value_mask: np.ndarray | None = None,
) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError("Expected values with shape (time, level, ...).")
    if value_mask is not None and value_mask.shape != values.shape:
        raise ValueError("Expected value_mask to match values.")

    quantile_values = np.asarray(quantiles, dtype=np.float32)
    boundaries = np.full(
        (values.shape[1], quantile_values.size),
        np.nan,
        dtype=np.float32,
    )

    for level_index in range(values.shape[1]):
        level_values = np.asarray(values[:, level_index, ...], dtype=np.float32).ravel()
        finite_mask = np.isfinite(level_values)
        if value_mask is not None:
            level_mask = np.asarray(
                value_mask[:, level_index, ...],
                dtype=bool,
            ).ravel()
            finite_mask &= level_mask

        finite_values = level_values[finite_mask]
        if finite_values.size == 0:
            continue

        boundaries[level_index] = np.asarray(
            np.quantile(finite_values, quantile_values),
            dtype=np.float32,
        )

    return boundaries


def compute_global_bucket_boundaries_from_array(
    values: np.ndarray,
    quantiles: tuple[float, ...] = BUCKET_EDGE_QUANTILES,
    value_mask: np.ndarray | None = None,
) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError("Expected values with shape (time, level, ...).")
    if value_mask is not None and value_mask.shape != values.shape:
        raise ValueError("Expected value_mask to match values.")

    flattened = np.asarray(values, dtype=np.float32).ravel()
    finite_mask = np.isfinite(flattened)
    if value_mask is not None:
        finite_mask &= np.asarray(value_mask, dtype=bool).ravel()

    finite_values = flattened[finite_mask]
    if finite_values.size == 0:
        return np.full((values.shape[1], len(quantiles)), np.nan, dtype=np.float32)

    shared_boundaries = np.asarray(
        np.quantile(finite_values, np.asarray(quantiles, dtype=np.float32)),
        dtype=np.float32,
    )
    return np.repeat(shared_boundaries[None, :], values.shape[1], axis=0)


def compute_pressure_thresholds(
    specific_humidity: xr.DataArray,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    time_stride: int = DEFAULT_THRESHOLD_TIME_STRIDE,
    sample_stride: int = DEFAULT_THRESHOLD_SAMPLE_STRIDE,
) -> np.ndarray:
    sampled = np.asarray(
        specific_humidity.isel(
            valid_time=slice(None, None, time_stride),
            latitude=slice(None, None, sample_stride),
            longitude=slice(None, None, sample_stride),
        ).values,
        dtype=np.float32,
    )
    return compute_per_level_thresholds_from_array(sampled, quantile=quantile)


def compute_pressure_thresholds_from_variable(
    variable: netCDF4.Variable,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    time_stride: int = DEFAULT_THRESHOLD_TIME_STRIDE,
    sample_stride: int = DEFAULT_THRESHOLD_SAMPLE_STRIDE,
) -> np.ndarray:
    del time_stride, sample_stride
    midpoint_index = variable.shape[0] // 2
    sampled = np.asarray(
        variable[midpoint_index : midpoint_index + 1, :, :, :],
        dtype=np.float32,
    )
    return compute_per_level_thresholds_from_array(sampled, quantile=quantile)


def load_threshold_seed_sample(variable: netCDF4.Variable) -> np.ndarray:
    midpoint_index = variable.shape[0] // 2
    return np.asarray(
        variable[midpoint_index : midpoint_index + 1, :, :, :],
        dtype=np.float32,
    )


def downsample_lat_lon_values(
    values: np.ndarray,
    latitude_stride: int = 1,
    longitude_stride: int = 1,
) -> np.ndarray:
    latitude_stride = max(int(latitude_stride), 1)
    longitude_stride = max(int(longitude_stride), 1)

    indexer = [slice(None)] * values.ndim
    indexer[-2] = slice(None, None, latitude_stride)
    indexer[-1] = slice(None, None, longitude_stride)
    return np.asarray(values[tuple(indexer)], dtype=np.float32)


def coordinate_step_degrees(values: np.ndarray) -> float | None:
    axis = np.asarray(values, dtype=np.float32)
    if axis.size < 2:
        return None
    return float(abs(axis[1] - axis[0]))


def smooth_bucket_values(values: np.ndarray) -> np.ndarray:
    sigma_axes = [0.0] * values.ndim
    sigma_axes[-3] = BUCKET_LEVEL_SMOOTHING_SIGMA
    sigma_axes[-2] = BUCKET_LATITUDE_SMOOTHING_SIGMA
    sigma_axes[-1] = BUCKET_LONGITUDE_SMOOTHING_SIGMA
    return np.asarray(
        ndimage.gaussian_filter(
            np.asarray(values, dtype=np.float32),
            sigma=sigma_axes,
            mode="nearest",
        ),
        dtype=np.float32,
    )


def lat_lon_gaussian_filter(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(values, dtype=np.float32)

    sigma_axes = [0.0] * values.ndim
    sigma_axes[-2] = sigma
    sigma_axes[-1] = sigma
    return np.asarray(
        ndimage.gaussian_filter(
            np.asarray(values, dtype=np.float32),
            sigma=sigma_axes,
            mode="nearest",
        ),
        dtype=np.float32,
    )


def compute_pressure_thresholds_histogram(
    specific_humidity: xr.DataArray,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    time_window: int = DEFAULT_TIME_WINDOW,
    bins: int = 8192,
) -> np.ndarray:
    time_count = specific_humidity.sizes["valid_time"]
    level_count = specific_humidity.sizes["pressure_level"]

    mins = np.full(level_count, np.inf, dtype=np.float64)
    maxs = np.full(level_count, -np.inf, dtype=np.float64)

    for start in range(0, time_count, time_window):
        stop = min(time_count, start + time_window)
        chunk = np.asarray(
            specific_humidity.isel(valid_time=slice(start, stop)).values,
            dtype=np.float32,
        )
        chunk_mins = np.nanmin(chunk, axis=(0, 2, 3))
        chunk_maxs = np.nanmax(chunk, axis=(0, 2, 3))
        mins = np.minimum(mins, chunk_mins)
        maxs = np.maximum(maxs, chunk_maxs)

    hist = np.zeros((level_count, bins), dtype=np.int64)
    totals = np.zeros(level_count, dtype=np.int64)

    for start in range(0, time_count, time_window):
        stop = min(time_count, start + time_window)
        chunk = np.asarray(
            specific_humidity.isel(valid_time=slice(start, stop)).values,
            dtype=np.float32,
        )
        for level_index in range(level_count):
            values = chunk[:, level_index, :, :].ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            lo = float(mins[level_index])
            hi = float(maxs[level_index])
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                hist[level_index, 0] += values.size
                totals[level_index] += values.size
                continue

            counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
            hist[level_index] += counts
            totals[level_index] += values.size

    thresholds = np.zeros(level_count, dtype=np.float32)
    for level_index in range(level_count):
        lo = float(mins[level_index])
        hi = float(maxs[level_index])
        if not math.isfinite(lo) or not math.isfinite(hi) or totals[level_index] == 0:
            thresholds[level_index] = np.nan
            continue

        cumulative = np.cumsum(hist[level_index])
        target = quantile * totals[level_index]
        bin_index = int(np.searchsorted(cumulative, target, side="left"))
        if bins <= 1 or hi <= lo:
            thresholds[level_index] = np.float32(lo)
            continue

        thresholds[level_index] = np.float32(
            lo + (bin_index / max(bins - 1, 1)) * (hi - lo)
        )

    return thresholds


def build_threshold_mask(field: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    threshold_grid = thresholds[:, None, None]
    return np.asarray(field >= threshold_grid, dtype=bool)


def build_closing_structure(radius_cells: int) -> np.ndarray:
    radius = max(int(radius_cells), 0)
    if radius == 0:
        return np.ones((1, 1, 1), dtype=bool)
    width = radius * 2 + 1
    return np.ones((width, width, width), dtype=bool)


def lightly_close_mask(mask: np.ndarray, closing_radius_cells: int = DEFAULT_CLOSING_RADIUS_CELLS) -> np.ndarray:
    if closing_radius_cells <= 0:
        return np.asarray(mask, dtype=bool)
    return ndimage.binary_closing(
        mask,
        structure=build_closing_structure(closing_radius_cells),
    )


def lightly_open_mask(
    mask: np.ndarray,
    opening_radius_cells: int = DEFAULT_OPENING_RADIUS_CELLS,
) -> np.ndarray:
    if opening_radius_cells <= 0:
        return np.asarray(mask, dtype=bool)
    return ndimage.binary_opening(
        mask,
        structure=build_closing_structure(opening_radius_cells),
    )


def apply_binary_postprocess(
    mask: np.ndarray,
    closing_radius_cells: int,
    opening_radius_cells: int,
) -> np.ndarray:
    return lightly_open_mask(
        lightly_close_mask(mask, closing_radius_cells=closing_radius_cells),
        opening_radius_cells=opening_radius_cells,
    )


def iter_wrapped_components(
    mask: np.ndarray,
    min_component_size: int = 0,
) -> list[np.ndarray]:
    extended = np.concatenate([mask, mask[..., :1]], axis=2)
    labels, count = ndimage.label(extended, structure=LABEL_STRUCTURE)
    objects = ndimage.find_objects(labels)
    longitude_count = mask.shape[2]

    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, count + 1):
        component_slices = objects[label_id - 1] if label_id - 1 < len(objects) else None
        if component_slices is None:
            continue

        component_local = labels[component_slices] == label_id
        voxel_count = int(component_local.sum())
        if voxel_count < min_component_size:
            continue

        collapsed = np.zeros(mask.shape, dtype=bool)
        level_slice, latitude_slice, longitude_slice = component_slices
        non_wrapped_stop = min(longitude_slice.stop, longitude_count)
        direct_width = max(0, non_wrapped_stop - longitude_slice.start)
        if direct_width > 0:
            collapsed[
                level_slice,
                latitude_slice,
                longitude_slice.start:non_wrapped_stop,
            ] |= component_local[..., :direct_width]

        wrapped_width = max(0, longitude_slice.stop - longitude_count)
        if wrapped_width > 0:
            collapsed[level_slice, latitude_slice, :wrapped_width] |= component_local[
                ...,
                direct_width : direct_width + wrapped_width,
            ]

        components.append((voxel_count, collapsed))

    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]


def prepare_segmentation_context(
    threshold_seed_sample: np.ndarray,
    config: BuildConfig,
) -> SegmentationContext:
    mode = config.segmentation_mode
    if mode not in SUPPORTED_SEGMENTATION_MODES:
        raise ValueError(f"Unsupported segmentation mode: {mode}")

    if mode in {
        "p95-close",
        VOXEL_SHELL_SEGMENTATION_MODE,
        GEOMETRY_SMOOTHED_SEGMENTATION_MODE,
        LEGACY_BRIDGE_PRUNED_SEGMENTATION_MODE,
    }:
        raw_q95 = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=config.threshold_quantile,
        )
        recipe_name = (
            "bridge-pruned-smoothed-mesh"
            if mode == GEOMETRY_SMOOTHED_SEGMENTATION_MODE
            else "bridge-pruned-voxel-shell"
            if mode == VOXEL_SHELL_SEGMENTATION_MODE
            else "bridge-pruned-baseline"
        )
        geometry_metadata = (
            {
                "mesh_postprocess": {
                    "algorithm": "taubin-laplacian",
                    "iterations": config.mesh_smoothing_iterations,
                    "lambda": config.mesh_smoothing_lambda,
                    "mu": config.mesh_smoothing_mu,
                    "island_min_vertices": config.mesh_island_min_vertices,
                }
            }
            if mode == GEOMETRY_SMOOTHED_SEGMENTATION_MODE
            else {
                "geometry_variant": "voxel-shell",
                "voxel_exterior_only": config.voxel_exterior_only,
            }
            if mode == VOXEL_SHELL_SEGMENTATION_MODE
            else {}
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=raw_q95,
            threshold_tables={"raw_q95": raw_q95},
            closing_radius_cells=config.closing_radius_cells,
            opening_radius_cells=config.opening_radius_cells,
            threshold_quantile=config.threshold_quantile,
            threshold_kind="pressure-relative-quantile",
            recipe_metadata={
                "recipe": recipe_name,
                "thresholds": {"raw_q95": config.threshold_quantile},
                "preprocessing": {},
                "postprocess": {
                    "binary_closing_radius_cells": config.closing_radius_cells,
                    "binary_opening_radius_cells": config.opening_radius_cells,
                },
                **geometry_metadata,
            },
        )

    if mode == RAW_VOXEL_SHELL_SEGMENTATION_MODE:
        raw_q95 = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=config.threshold_quantile,
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=raw_q95,
            threshold_tables={"raw_q95": raw_q95},
            closing_radius_cells=0,
            opening_radius_cells=0,
            threshold_quantile=config.threshold_quantile,
            threshold_kind="pressure-relative-quantile",
            recipe_metadata={
                "recipe": "raw-threshold-voxel-shell",
                "thresholds": {"raw_q95": config.threshold_quantile},
                "preprocessing": {},
                "postprocess": {
                    "binary_closing_radius_cells": 0,
                    "binary_opening_radius_cells": 0,
                },
                "geometry_variant": "voxel-shell",
                "voxel_exterior_only": config.voxel_exterior_only,
            },
        )

    if mode in {"p95-smooth-open1", SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE}:
        smoothed_sample = lat_lon_gaussian_filter(
            threshold_seed_sample,
            sigma=SMOOTH_SUPPORT_SIGMA,
        )
        smoothed_q95 = compute_per_level_thresholds_from_array(
            smoothed_sample,
            quantile=DEFAULT_THRESHOLD_QUANTILE,
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=smoothed_q95,
            threshold_tables={"smoothed_q95": smoothed_q95},
            closing_radius_cells=DEFAULT_CLOSING_RADIUS_CELLS,
            opening_radius_cells=DEFAULT_OPENING_RADIUS_CELLS,
            threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
            threshold_kind="lat-lon-smoothed-quantile",
            recipe_metadata={
                "recipe": (
                    "smoothed-support-voxel-shell"
                    if mode == SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE
                    else "smoothed-support"
                ),
                "thresholds": {"smoothed_q95": DEFAULT_THRESHOLD_QUANTILE},
                "preprocessing": {"lat_lon_gaussian_sigma": SMOOTH_SUPPORT_SIGMA},
                "postprocess": {
                    "binary_closing_radius_cells": DEFAULT_CLOSING_RADIUS_CELLS,
                    "binary_opening_radius_cells": DEFAULT_OPENING_RADIUS_CELLS,
                },
                **(
                    {"geometry_variant": "voxel-shell"}
                    if mode == SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE
                    else {}
                ),
            },
        )

    if mode == "p95-local-anomaly":
        background_sample = lat_lon_gaussian_filter(
            threshold_seed_sample,
            sigma=LOCAL_ANOMALY_BACKGROUND_SIGMA,
        )
        anomaly_sample = threshold_seed_sample - background_sample
        raw_q90 = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=LOCAL_ANOMALY_RAW_QUANTILE,
        )
        anomaly_q95 = compute_per_level_thresholds_from_array(
            anomaly_sample,
            quantile=LOCAL_ANOMALY_ANOMALY_QUANTILE,
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=anomaly_q95,
            threshold_tables={
                "raw_q90": raw_q90,
                "anomaly_q95": anomaly_q95,
            },
            closing_radius_cells=DEFAULT_CLOSING_RADIUS_CELLS,
            opening_radius_cells=DEFAULT_OPENING_RADIUS_CELLS,
            threshold_quantile=LOCAL_ANOMALY_ANOMALY_QUANTILE,
            threshold_kind="local-anomaly-quantile",
            recipe_metadata={
                "recipe": "local-anomaly",
                "thresholds": {
                    "raw_q90": LOCAL_ANOMALY_RAW_QUANTILE,
                    "anomaly_q95": LOCAL_ANOMALY_ANOMALY_QUANTILE,
                },
                "preprocessing": {
                    "background_lat_lon_gaussian_sigma": LOCAL_ANOMALY_BACKGROUND_SIGMA,
                },
                "postprocess": {
                    "binary_closing_radius_cells": DEFAULT_CLOSING_RADIUS_CELLS,
                    "binary_opening_radius_cells": DEFAULT_OPENING_RADIUS_CELLS,
                },
            },
        )

    if mode == "p95-open":
        raw_q95 = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=DEFAULT_THRESHOLD_QUANTILE,
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=raw_q95,
            threshold_tables={"raw_q95": raw_q95},
            closing_radius_cells=0,
            opening_radius_cells=0,
            threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
            threshold_kind="pressure-relative-quantile",
            recipe_metadata={
                "recipe": "legacy-p95-open",
                "thresholds": {"raw_q95": DEFAULT_THRESHOLD_QUANTILE},
                "preprocessing": {},
                "postprocess": {
                    "binary_closing_radius_cells": 0,
                    "binary_opening_radius_cells": 0,
                },
            },
        )

    if mode == "p97-close":
        raw_q97 = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=P97_CLOSE_QUANTILE,
        )
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=raw_q97,
            threshold_tables={"raw_q97": raw_q97},
            closing_radius_cells=DEFAULT_CLOSING_RADIUS_CELLS,
            opening_radius_cells=0,
            threshold_quantile=P97_CLOSE_QUANTILE,
            threshold_kind="pressure-relative-quantile",
            recipe_metadata={
                "recipe": "legacy-p97-close",
                "thresholds": {"raw_q97": P97_CLOSE_QUANTILE},
                "preprocessing": {},
                "postprocess": {
                    "binary_closing_radius_cells": DEFAULT_CLOSING_RADIUS_CELLS,
                    "binary_opening_radius_cells": 0,
                },
            },
        )

    if mode == BUCKET_SEGMENTATION_MODE:
        sampled = downsample_lat_lon_values(
            smooth_bucket_values(threshold_seed_sample),
            latitude_stride=BUCKET_FIELD_LATITUDE_STRIDE,
            longitude_stride=BUCKET_FIELD_LONGITUDE_STRIDE,
        )
        bucket_boundaries = compute_per_level_bucket_boundaries_from_array(sampled)
        median_bucket_edge = bucket_boundaries[:, bucket_boundaries.shape[1] // 2]
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=median_bucket_edge,
            threshold_tables={
                "bucket_boundaries": bucket_boundaries,
                "bucket_reference": median_bucket_edge,
            },
            closing_radius_cells=0,
            opening_radius_cells=0,
            threshold_quantile=0.5,
            threshold_kind="pressure-relative-bucket-heuristic",
            recipe_metadata={
                "recipe": "bucketed-specific-humidity",
                "bucket_count": BUCKET_COUNT,
                "bucket_strategy": "per-pressure-level",
                "thresholds": {
                    "bucket_edge_quantiles": list(BUCKET_EDGE_QUANTILES),
                },
                "preprocessing": {
                    "bucket_gaussian_sigma": {
                        "pressure_level": BUCKET_LEVEL_SMOOTHING_SIGMA,
                        "latitude": BUCKET_LATITUDE_SMOOTHING_SIGMA,
                        "longitude": BUCKET_LONGITUDE_SMOOTHING_SIGMA,
                    },
                    "lat_lon_downsample_stride": {
                        "latitude": BUCKET_FIELD_LATITUDE_STRIDE,
                        "longitude": BUCKET_FIELD_LONGITUDE_STRIDE,
                    },
                    "threshold_seed": "midpoint_time_slice",
                },
                "postprocess": {},
                "bucket_labels": {
                    "lowest_bucket_index": 0,
                    "highest_bucket_index": BUCKET_COUNT - 1,
                },
            },
            bucket_count=BUCKET_COUNT,
            latitude_stride=BUCKET_FIELD_LATITUDE_STRIDE,
            longitude_stride=BUCKET_FIELD_LONGITUDE_STRIDE,
        )

    if mode == BUCKET_GLOBAL_SEGMENTATION_MODE:
        sampled = downsample_lat_lon_values(
            smooth_bucket_values(threshold_seed_sample),
            latitude_stride=BUCKET_FIELD_LATITUDE_STRIDE,
            longitude_stride=BUCKET_FIELD_LONGITUDE_STRIDE,
        )
        bucket_boundaries = compute_global_bucket_boundaries_from_array(sampled)
        median_bucket_edge = bucket_boundaries[:, bucket_boundaries.shape[1] // 2]
        return SegmentationContext(
            segmentation_mode=mode,
            primary_thresholds=median_bucket_edge,
            threshold_tables={
                "bucket_boundaries": bucket_boundaries,
                "bucket_reference": median_bucket_edge,
            },
            closing_radius_cells=0,
            opening_radius_cells=0,
            threshold_quantile=0.5,
            threshold_kind="global-bucket-heuristic",
            recipe_metadata={
                "recipe": "bucketed-specific-humidity",
                "bucket_count": BUCKET_COUNT,
                "bucket_strategy": "global-shared",
                "thresholds": {
                    "bucket_edge_quantiles": list(BUCKET_EDGE_QUANTILES),
                },
                "preprocessing": {
                    "bucket_gaussian_sigma": {
                        "pressure_level": BUCKET_LEVEL_SMOOTHING_SIGMA,
                        "latitude": BUCKET_LATITUDE_SMOOTHING_SIGMA,
                        "longitude": BUCKET_LONGITUDE_SMOOTHING_SIGMA,
                    },
                    "lat_lon_downsample_stride": {
                        "latitude": BUCKET_FIELD_LATITUDE_STRIDE,
                        "longitude": BUCKET_FIELD_LONGITUDE_STRIDE,
                    },
                    "threshold_seed": "midpoint_time_slice",
                },
                "postprocess": {},
                "bucket_labels": {
                    "lowest_bucket_index": 0,
                    "highest_bucket_index": BUCKET_COUNT - 1,
                },
            },
            bucket_count=BUCKET_COUNT,
            latitude_stride=BUCKET_FIELD_LATITUDE_STRIDE,
            longitude_stride=BUCKET_FIELD_LONGITUDE_STRIDE,
        )

    raise ValueError(f"Unsupported segmentation mode: {mode}")


def build_segmentation_mask(
    field: np.ndarray,
    context: SegmentationContext,
) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    mode = context.segmentation_mode

    if mode in {
        "p95-close",
        VOXEL_SHELL_SEGMENTATION_MODE,
        GEOMETRY_SMOOTHED_SEGMENTATION_MODE,
        LEGACY_BRIDGE_PRUNED_SEGMENTATION_MODE,
    }:
        return apply_binary_postprocess(
            build_threshold_mask(field, context.threshold_tables["raw_q95"]),
            closing_radius_cells=context.closing_radius_cells,
            opening_radius_cells=context.opening_radius_cells,
        )

    if mode == RAW_VOXEL_SHELL_SEGMENTATION_MODE:
        return build_threshold_mask(field, context.threshold_tables["raw_q95"])

    if mode in {"p95-smooth-open1", SMOOTHED_VOXEL_SHELL_SEGMENTATION_MODE}:
        smoothed = lat_lon_gaussian_filter(field, sigma=SMOOTH_SUPPORT_SIGMA)
        return apply_binary_postprocess(
            build_threshold_mask(smoothed, context.threshold_tables["smoothed_q95"]),
            closing_radius_cells=context.closing_radius_cells,
            opening_radius_cells=context.opening_radius_cells,
        )

    if mode == "p95-local-anomaly":
        background = lat_lon_gaussian_filter(
            field,
            sigma=LOCAL_ANOMALY_BACKGROUND_SIGMA,
        )
        anomaly = field - background
        raw_support = build_threshold_mask(field, context.threshold_tables["raw_q90"])
        anomaly_support = build_threshold_mask(
            anomaly,
            context.threshold_tables["anomaly_q95"],
        )
        return apply_binary_postprocess(
            raw_support & anomaly_support,
            closing_radius_cells=context.closing_radius_cells,
            opening_radius_cells=context.opening_radius_cells,
        )

    if mode == "p95-open":
        return apply_binary_postprocess(
            build_threshold_mask(field, context.threshold_tables["raw_q95"]),
            closing_radius_cells=context.closing_radius_cells,
            opening_radius_cells=context.opening_radius_cells,
        )

    if mode == "p97-close":
        return apply_binary_postprocess(
            build_threshold_mask(field, context.threshold_tables["raw_q97"]),
            closing_radius_cells=context.closing_radius_cells,
            opening_radius_cells=context.opening_radius_cells,
        )

    if mode in {BUCKET_SEGMENTATION_MODE, BUCKET_GLOBAL_SEGMENTATION_MODE}:
        raise ValueError("Bucket segmentation uses bucket component masks, not a binary mask.")

    raise ValueError(f"Unsupported segmentation mode: {mode}")


def build_bucket_index_field(
    field: np.ndarray,
    bucket_boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(field)
    bucket_indices = np.zeros(field.shape, dtype=np.uint8)

    for edge_index in range(bucket_boundaries.shape[1]):
        threshold_grid = bucket_boundaries[:, edge_index][:, None, None]
        bucket_indices += np.asarray(
            finite_mask & (field >= threshold_grid),
            dtype=np.uint8,
        )

    return bucket_indices, finite_mask


def build_bucket_component_specs(
    field: np.ndarray,
    context: SegmentationContext,
) -> list[dict[str, Any]]:
    if context.segmentation_mode not in {
        BUCKET_SEGMENTATION_MODE,
        BUCKET_GLOBAL_SEGMENTATION_MODE,
    }:
        raise ValueError("Bucket component specs are only available in bucket mode.")

    bucket_boundaries = context.threshold_tables["bucket_boundaries"]
    smoothed_field = smooth_bucket_values(field)
    bucket_indices, finite_mask = build_bucket_index_field(smoothed_field, bucket_boundaries)
    specs: list[dict[str, Any]] = []

    for bucket_index in range(context.bucket_count):
        mask = finite_mask & (bucket_indices == bucket_index)
        if not bool(mask.any()):
            continue

        specs.append(
            {
                "mask": mask,
                "metadata": {
                    "id": bucket_index,
                    "bucket_index": bucket_index,
                },
            }
        )

    return specs


def choose_longitude_roll(component_mask: np.ndarray) -> int:
    active = np.flatnonzero(component_mask.any(axis=(0, 1)))
    if active.size == 0:
        return 0

    ordered = np.sort(active)
    wrapped = np.concatenate([ordered, [ordered[0] + component_mask.shape[2]]])
    gaps = np.diff(wrapped)
    cut_index = int(np.argmax(gaps))
    return int(ordered[(cut_index + 1) % ordered.size])


def roll_longitudes(longitudes: np.ndarray, start_index: int) -> np.ndarray:
    rolled = np.roll(np.asarray(longitudes, dtype=np.float64), -start_index)
    wrap_points = np.where(np.diff(rolled) < 0)[0]
    if wrap_points.size:
        rolled[wrap_points[0] + 1 :] += 360.0
    return rolled


def crop_mask(mask: np.ndarray, pad: int = 1) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot crop an empty component mask.")

    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    cropped = mask[
        mins[0] : maxs[0],
        mins[1] : maxs[1],
        mins[2] : maxs[2],
    ]
    return cropped, tuple(int(x) for x in mins), tuple(int(x) for x in maxs)


def build_axis_bounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        step = 1.0
        return np.array([values[0] - step * 0.5, values[0] + step * 0.5], dtype=np.float64)

    midpoints = (values[:-1] + values[1:]) * 0.5
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]]).astype(np.float64)


def interpolate_pressures(pressure_levels: np.ndarray, fractional_indices: np.ndarray) -> np.ndarray:
    return np.exp(
        np.interp(
            fractional_indices,
            np.arange(pressure_levels.size, dtype=np.float64),
            np.log(np.asarray(pressure_levels, dtype=np.float64)),
        )
    )


def interpolate_radius(
    pressure_levels: np.ndarray,
    radius_lookup: np.ndarray,
    fractional_indices: np.ndarray,
) -> np.ndarray:
    pressures = interpolate_pressures(pressure_levels, fractional_indices)
    return np.interp(
        pressures[::-1],
        pressure_levels[::-1].astype(np.float64),
        radius_lookup[::-1].astype(np.float64),
    )[::-1]


def build_component_metadata(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    rolled_longitudes: np.ndarray,
    roll_start: int,
) -> dict[str, Any]:
    wraps_longitude_seam = bool(component_mask[..., 0].any() and component_mask[..., -1].any())
    component_values = field[component_mask]
    filled_coords = np.argwhere(component_mask)
    pressure_indices = filled_coords[:, 0]
    latitude_indices = filled_coords[:, 1]
    longitude_indices = filled_coords[:, 2]

    rolled_longitude_indices = (longitude_indices - roll_start) % longitudes.size
    longitude_min = float(rolled_longitudes[rolled_longitude_indices].min() % 360.0)
    longitude_max = float(rolled_longitudes[rolled_longitude_indices].max() % 360.0)

    return {
        "voxel_count": int(component_mask.sum()),
        "mean_specific_humidity": float(np.mean(component_values)),
        "max_specific_humidity": float(np.max(component_values)),
        "pressure_min_hpa": float(np.min(pressure_levels[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes[latitude_indices])),
        "longitude_min_deg": longitude_min,
        "longitude_max_deg": longitude_max,
        "wraps_longitude_seam": wraps_longitude_seam,
    }


def build_component_marching_cubes_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_longitudes = roll_longitudes(longitudes, roll_start)

    cropped_mask, mins, _ = crop_mask(rolled_mask, pad=1)

    smoothed = ndimage.gaussian_filter(
        cropped_mask.astype(np.float32),
        sigma=gaussian_sigma,
        mode="nearest",
    )

    if float(smoothed.max()) <= 0.5:
        return None

    vertices, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
    if vertices.size == 0 or faces.size == 0:
        return None

    global_level = vertices[:, 0] + mins[0]
    global_lat = vertices[:, 1] + mins[1]
    global_lon = vertices[:, 2] + mins[2]

    radius_values = interpolate_radius(pressure_levels, radius_lookup, global_level)
    latitude_values = np.interp(
        global_lat,
        np.arange(latitudes.size, dtype=np.float64),
        latitudes.astype(np.float64),
    )
    longitude_values = np.interp(
        global_lon,
        np.arange(rolled_longitudes.size, dtype=np.float64),
        rolled_longitudes,
    )

    positions = np.empty((vertices.shape[0], 3), dtype=np.float32)
    for index in range(vertices.shape[0]):
        positions[index] = lat_lon_to_xyz(
            float(latitude_values[index]),
            float(longitude_values[index]),
            float(radius_values[index]),
        )

    metadata = build_component_metadata(
        component_mask=component_mask,
        field=field,
        pressure_levels=pressure_levels,
        latitudes=latitudes,
        longitudes=longitudes,
        rolled_longitudes=rolled_longitudes,
        roll_start=roll_start,
    )
    return positions, np.asarray(faces, dtype=np.uint32), metadata


def build_component_signed_distance_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_longitudes = roll_longitudes(longitudes, roll_start)

    pad = max(2, int(math.ceil(max(float(gaussian_sigma), 1.0) * 3.0)))
    cropped_mask, mins, _ = crop_mask(rolled_mask, pad=pad)

    inside_distance = ndimage.distance_transform_edt(cropped_mask)
    outside_distance = ndimage.distance_transform_edt(~cropped_mask)
    signed_distance = outside_distance - inside_distance
    smoothed = ndimage.gaussian_filter(
        signed_distance.astype(np.float32),
        sigma=max(float(gaussian_sigma), 0.5),
        mode="nearest",
    )

    if float(np.min(smoothed)) >= 0 or float(np.max(smoothed)) <= 0:
        return None

    vertices, faces, _, _ = measure.marching_cubes(smoothed, level=0.0)
    if vertices.size == 0 or faces.size == 0:
        return None

    global_level = vertices[:, 0] + mins[0]
    global_lat = vertices[:, 1] + mins[1]
    global_lon = vertices[:, 2] + mins[2]

    radius_values = interpolate_radius(pressure_levels, radius_lookup, global_level)
    latitude_values = np.interp(
        global_lat,
        np.arange(latitudes.size, dtype=np.float64),
        latitudes.astype(np.float64),
    )
    longitude_values = np.interp(
        global_lon,
        np.arange(rolled_longitudes.size, dtype=np.float64),
        rolled_longitudes,
    )

    positions = np.empty((vertices.shape[0], 3), dtype=np.float32)
    for index in range(vertices.shape[0]):
        positions[index] = lat_lon_to_xyz(
            float(latitude_values[index]),
            float(longitude_values[index]),
            float(radius_values[index]),
        )

    metadata = build_component_metadata(
        component_mask=component_mask,
        field=field,
        pressure_levels=pressure_levels,
        latitudes=latitudes,
        longitudes=longitudes,
        rolled_longitudes=rolled_longitudes,
        roll_start=roll_start,
    )
    return positions, np.asarray(faces, dtype=np.uint32), metadata


def build_horizontal_closing_structure(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    flat_structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(2, 1),
        radius_cells,
    )
    structure = np.zeros((1, flat_structure.shape[0], flat_structure.shape[1]), dtype=bool)
    structure[0] = flat_structure
    return structure


def fill_component_vertical_columns(
    component_mask: np.ndarray,
    horizontal_closing_radius: int = 0,
    horizontal_dilation_radius: int = 0,
) -> np.ndarray:
    mask = np.asarray(component_mask, dtype=bool)
    if not bool(mask.any()):
        return np.zeros_like(mask, dtype=bool)

    roll_start = choose_longitude_roll(mask)
    rolled_mask = np.roll(mask, -roll_start, axis=2)
    filled = np.zeros_like(rolled_mask, dtype=bool)
    base_footprint = rolled_mask.any(axis=0)
    target_footprint = morphology.convex_hull_image(base_footprint)

    lower_levels = np.full(base_footprint.shape, -1, dtype=np.int32)
    upper_levels = np.full(base_footprint.shape, -1, dtype=np.int32)

    for latitude_index, longitude_index in np.argwhere(base_footprint):
        level_indices = np.flatnonzero(
            rolled_mask[:, latitude_index, longitude_index]
        )
        if level_indices.size == 0:
            continue
        lower_levels[int(latitude_index), int(longitude_index)] = int(level_indices[0])
        upper_levels[int(latitude_index), int(longitude_index)] = int(level_indices[-1])

    _, nearest_indices = ndimage.distance_transform_edt(
        ~base_footprint,
        return_indices=True,
    )

    for latitude_index, longitude_index in np.argwhere(target_footprint):
        latitude_int = int(latitude_index)
        longitude_int = int(longitude_index)
        source_latitude = (
            latitude_int
            if base_footprint[latitude_int, longitude_int]
            else int(nearest_indices[0, latitude_int, longitude_int])
        )
        source_longitude = (
            longitude_int
            if base_footprint[latitude_int, longitude_int]
            else int(nearest_indices[1, latitude_int, longitude_int])
        )
        lower_level = int(lower_levels[source_latitude, source_longitude])
        upper_level = int(upper_levels[source_latitude, source_longitude])
        if lower_level < 0 or upper_level < lower_level:
            continue
        filled[
            lower_level : upper_level + 1,
            latitude_int,
            longitude_int,
        ] = True

    if horizontal_closing_radius > 0:
        filled = ndimage.binary_closing(
            filled,
            structure=build_horizontal_closing_structure(horizontal_closing_radius),
        )
    if horizontal_dilation_radius > 0:
        filled = ndimage.binary_dilation(
            filled,
            structure=build_horizontal_closing_structure(horizontal_dilation_radius),
        )

    return np.roll(filled, roll_start, axis=2)


def build_vertex_neighbors(face_indices: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]

    for a, b, c in np.asarray(face_indices, dtype=np.int64):
        ai = int(a)
        bi = int(b)
        ci = int(c)
        neighbors[ai].update((bi, ci))
        neighbors[bi].update((ai, ci))
        neighbors[ci].update((ai, bi))

    return [
        np.asarray(sorted(vertex_neighbors), dtype=np.int32)
        for vertex_neighbors in neighbors
    ]


def laplacian_smooth_positions(
    positions: np.ndarray,
    neighbors: list[np.ndarray],
    step: float,
) -> np.ndarray:
    current = np.asarray(positions, dtype=np.float32)
    next_positions = current.copy()

    for vertex_index, vertex_neighbors in enumerate(neighbors):
        if vertex_neighbors.size == 0:
            continue
        neighbor_mean = np.mean(current[vertex_neighbors], axis=0)
        next_positions[vertex_index] = current[vertex_index] + step * (
            neighbor_mean - current[vertex_index]
        )

    return next_positions


def taubin_smooth_mesh(
    positions: np.ndarray,
    face_indices: np.ndarray,
    iterations: int,
    lambda_step: float,
    mu_step: float,
) -> np.ndarray:
    if iterations <= 0 or positions.shape[0] == 0 or face_indices.shape[0] == 0:
        return np.asarray(positions, dtype=np.float32)

    neighbors = build_vertex_neighbors(face_indices, positions.shape[0])
    smoothed = np.asarray(positions, dtype=np.float32)

    for _ in range(iterations):
        smoothed = laplacian_smooth_positions(smoothed, neighbors, lambda_step)
        smoothed = laplacian_smooth_positions(smoothed, neighbors, mu_step)

    return smoothed


def prune_small_mesh_islands(
    positions: np.ndarray,
    face_indices: np.ndarray,
    min_vertices: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    if min_vertices <= 0:
        return np.asarray(positions, dtype=np.float32), np.asarray(face_indices, dtype=np.uint32)
    if positions.shape[0] == 0 or face_indices.shape[0] == 0:
        return None

    neighbors = build_vertex_neighbors(face_indices, positions.shape[0])
    seen = np.zeros(positions.shape[0], dtype=bool)
    keep_mask = np.zeros(positions.shape[0], dtype=bool)

    for start_index in range(positions.shape[0]):
        if seen[start_index]:
            continue

        stack = [start_index]
        seen[start_index] = True
        island_vertices: list[int] = []

        while stack:
            vertex_index = stack.pop()
            island_vertices.append(vertex_index)
            for neighbor_index in neighbors[vertex_index]:
                neighbor_int = int(neighbor_index)
                if seen[neighbor_int]:
                    continue
                seen[neighbor_int] = True
                stack.append(neighbor_int)

        if len(island_vertices) >= min_vertices:
            keep_mask[np.asarray(island_vertices, dtype=np.int32)] = True

    if not keep_mask.any():
        return None

    kept_faces_mask = keep_mask[face_indices].all(axis=1)
    kept_faces = np.asarray(face_indices[kept_faces_mask], dtype=np.uint32)
    if kept_faces.shape[0] == 0:
        return None

    kept_vertices = np.flatnonzero(keep_mask)
    remap = np.full(positions.shape[0], -1, dtype=np.int32)
    remap[kept_vertices] = np.arange(kept_vertices.size, dtype=np.int32)

    return (
        np.asarray(positions[kept_vertices], dtype=np.float32),
        np.asarray(remap[kept_faces], dtype=np.uint32),
    )


def append_quad(
    corners: list[tuple[int, int, int]],
    vertex_lookup: dict[tuple[int, int, int], int],
    positions: list[float],
    faces: list[int],
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

    faces.extend(
        [
            quad_indices[0],
            quad_indices[1],
            quad_indices[2],
            quad_indices[0],
            quad_indices[2],
            quad_indices[3],
        ]
    )


def exterior_empty_mask(mask: np.ndarray) -> np.ndarray:
    padded_mask = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    exterior = np.zeros_like(padded_mask, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()

    max_level, max_latitude, max_longitude = (
        padded_mask.shape[0] - 1,
        padded_mask.shape[1] - 1,
        padded_mask.shape[2] - 1,
    )

    def push_if_exterior(level_index: int, latitude_index: int, longitude_index: int) -> None:
        if padded_mask[level_index, latitude_index, longitude_index]:
            return
        if exterior[level_index, latitude_index, longitude_index]:
            return
        exterior[level_index, latitude_index, longitude_index] = True
        queue.append((level_index, latitude_index, longitude_index))

    for level_index in range(padded_mask.shape[0]):
        for latitude_index in range(padded_mask.shape[1]):
            push_if_exterior(level_index, latitude_index, 0)
            push_if_exterior(level_index, latitude_index, max_longitude)

    for level_index in range(padded_mask.shape[0]):
        for longitude_index in range(padded_mask.shape[2]):
            push_if_exterior(level_index, 0, longitude_index)
            push_if_exterior(level_index, max_latitude, longitude_index)

    for latitude_index in range(padded_mask.shape[1]):
        for longitude_index in range(padded_mask.shape[2]):
            push_if_exterior(0, latitude_index, longitude_index)
            push_if_exterior(max_level, latitude_index, longitude_index)

    neighbor_offsets = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )

    while queue:
        level_index, latitude_index, longitude_index = queue.popleft()
        for d_level, d_latitude, d_longitude in neighbor_offsets:
            next_level = level_index + d_level
            next_latitude = latitude_index + d_latitude
            next_longitude = longitude_index + d_longitude
            if not (
                0 <= next_level <= max_level
                and 0 <= next_latitude <= max_latitude
                and 0 <= next_longitude <= max_longitude
            ):
                continue
            push_if_exterior(next_level, next_latitude, next_longitude)

    return exterior[1:-1, 1:-1, 1:-1]


def build_component_voxel_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    exterior_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_longitudes = roll_longitudes(longitudes, roll_start)

    cropped_mask, mins, maxs = crop_mask(rolled_mask, pad=0)
    if not bool(cropped_mask.any()):
        return None

    exterior_mask = exterior_empty_mask(cropped_mask) if exterior_only else None

    radius_bounds = build_axis_bounds(radius_lookup.astype(np.float64))[mins[0] : maxs[0] + 1]
    latitude_bounds = build_axis_bounds(latitudes.astype(np.float64))[mins[1] : maxs[1] + 1]
    longitude_bounds = build_axis_bounds(rolled_longitudes.astype(np.float64))[mins[2] : maxs[2] + 1]

    vertex_lookup: dict[tuple[int, int, int], int] = {}
    positions: list[float] = []
    faces: list[int] = []

    level_count, latitude_count, longitude_count = cropped_mask.shape
    filled_coords = np.argwhere(cropped_mask)

    for level_index, latitude_index, longitude_index in filled_coords:
        r0 = int(level_index)
        r1 = r0 + 1
        a0 = int(latitude_index)
        a1 = a0 + 1
        o0 = int(longitude_index)
        o1 = o0 + 1

        lower_exposed = r0 == 0 or not cropped_mask[r0 - 1, a0, o0]
        if lower_exposed and (exterior_mask is None or (r0 > 0 and exterior_mask[r0 - 1, a0, o0]) or r0 == 0):
            append_quad(
                [(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        upper_exposed = r0 == level_count - 1 or not cropped_mask[r0 + 1, a0, o0]
        if upper_exposed and (
            exterior_mask is None
            or (r0 < level_count - 1 and exterior_mask[r0 + 1, a0, o0])
            or r0 == level_count - 1
        ):
            append_quad(
                [(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        south_exposed = a0 == 0 or not cropped_mask[r0, a0 - 1, o0]
        if south_exposed and (exterior_mask is None or (a0 > 0 and exterior_mask[r0, a0 - 1, o0]) or a0 == 0):
            append_quad(
                [(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        north_exposed = a0 == latitude_count - 1 or not cropped_mask[r0, a0 + 1, o0]
        if north_exposed and (
            exterior_mask is None
            or (a0 < latitude_count - 1 and exterior_mask[r0, a0 + 1, o0])
            or a0 == latitude_count - 1
        ):
            append_quad(
                [(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        west_exposed = o0 == 0 or not cropped_mask[r0, a0, o0 - 1]
        if west_exposed and (exterior_mask is None or (o0 > 0 and exterior_mask[r0, a0, o0 - 1]) or o0 == 0):
            append_quad(
                [(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        east_exposed = o0 == longitude_count - 1 or not cropped_mask[r0, a0, o0 + 1]
        if east_exposed and (
            exterior_mask is None
            or (o0 < longitude_count - 1 and exterior_mask[r0, a0, o0 + 1])
            or o0 == longitude_count - 1
        ):
            append_quad(
                [(r0, a0, o1), (r0, a1, o1), (r1, a1, o1), (r1, a0, o1)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )

    if not positions or not faces:
        return None

    metadata = build_component_metadata(
        component_mask=component_mask,
        field=field,
        pressure_levels=pressure_levels,
        latitudes=latitudes,
        longitudes=longitudes,
        rolled_longitudes=rolled_longitudes,
        roll_start=roll_start,
    )
    return (
        np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        np.asarray(faces, dtype=np.uint32).reshape(-1, 3),
        metadata,
    )


def build_component_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    geometry_mode: str = DEFAULT_GEOMETRY_MODE,
    mesh_smoothing_iterations: int = 0,
    mesh_smoothing_lambda: float = 0.45,
    mesh_smoothing_mu: float = -0.47,
    mesh_island_min_vertices: int = 0,
    voxel_exterior_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if geometry_mode == "voxel-faces":
        surface = build_component_voxel_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            exterior_only=voxel_exterior_only,
        )
    elif geometry_mode == "marching-cubes":
        surface = build_component_marching_cubes_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            gaussian_sigma=gaussian_sigma,
        )
    elif geometry_mode == SIGNED_DISTANCE_GEOMETRY_MODE:
        surface = build_component_signed_distance_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            gaussian_sigma=gaussian_sigma,
        )
    elif geometry_mode == COLUMN_ENVELOPE_GEOMETRY_MODE:
        surface = build_component_signed_distance_surface(
            component_mask=fill_component_vertical_columns(
                component_mask,
                horizontal_closing_radius=max(1, int(round(max(gaussian_sigma, 1.0) / 3.0))),
                horizontal_dilation_radius=max(1, int(round(max(gaussian_sigma, 1.0)))),
            ),
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            gaussian_sigma=gaussian_sigma,
        )
    else:
        raise ValueError(f"Unsupported geometry mode: {geometry_mode}")

    if surface is None:
        return None

    positions, faces, metadata = surface
    smoothed_positions = taubin_smooth_mesh(
        positions=positions,
        face_indices=faces,
        iterations=mesh_smoothing_iterations,
        lambda_step=mesh_smoothing_lambda,
        mu_step=mesh_smoothing_mu,
    )
    pruned_surface = prune_small_mesh_islands(
        positions=smoothed_positions,
        face_indices=faces,
        min_vertices=mesh_island_min_vertices,
    )
    if pruned_surface is None:
        return None

    pruned_positions, pruned_faces = pruned_surface
    return pruned_positions, pruned_faces, metadata


GridPoint = tuple[int, int]
GridEdge = tuple[GridPoint, GridPoint]


def edge_direction(edge: GridEdge) -> GridPoint:
    (x0, y0), (x1, y1) = edge
    return (x1 - x0, y1 - y0)


RIGHT_TURN_PRIORITY: dict[GridPoint, tuple[GridPoint, ...]] = {
    (1, 0): ((0, 1), (1, 0), (0, -1)),
    (0, 1): ((-1, 0), (0, 1), (1, 0)),
    (-1, 0): ((0, -1), (-1, 0), (0, 1)),
    (0, -1): ((1, 0), (0, -1), (-1, 0)),
}


def is_collinear(prev_point: GridPoint, point: GridPoint, next_point: GridPoint) -> bool:
    return (
        prev_point[0] == point[0] == next_point[0]
        or prev_point[1] == point[1] == next_point[1]
    )


def simplify_loop(points: list[GridPoint]) -> list[GridPoint]:
    if len(points) < 4:
        return points

    simplified = points[:-1]
    changed = True
    while changed and len(simplified) >= 3:
        changed = False
        next_points: list[GridPoint] = []
        for index, point in enumerate(simplified):
            prev_point = simplified[index - 1]
            next_point = simplified[(index + 1) % len(simplified)]
            if is_collinear(prev_point, point, next_point):
                changed = True
                continue
            next_points.append(point)
        if next_points:
            simplified = next_points

    return simplified + [simplified[0]]


def boundary_edges_from_footprint_mask(mask: np.ndarray) -> list[GridEdge]:
    latitude_count, longitude_count = mask.shape
    edges: list[GridEdge] = []

    filled_coords = np.argwhere(mask)
    for latitude_index, longitude_index in filled_coords:
        a = int(latitude_index)
        o = int(longitude_index)

        if a == 0 or not mask[a - 1, o]:
            edges.append(((o, a), (o + 1, a)))
        if o == longitude_count - 1 or not mask[a, o + 1]:
            edges.append(((o + 1, a), (o + 1, a + 1)))
        if a == latitude_count - 1 or not mask[a + 1, o]:
            edges.append(((o + 1, a + 1), (o, a + 1)))
        if o == 0 or not mask[a, o - 1]:
            edges.append(((o, a + 1), (o, a)))

    return edges


def trace_boundary_loops(edges: list[GridEdge]) -> list[list[GridPoint]]:
    outgoing: dict[GridPoint, list[GridEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge[0], []).append(edge)

    visited: set[GridEdge] = set()
    loops: list[list[GridPoint]] = []

    for edge in edges:
        if edge in visited:
            continue

        current_edge = edge
        loop = [current_edge[0], current_edge[1]]
        visited.add(current_edge)

        while loop[-1] != loop[0]:
            current_point = loop[-1]
            current_direction = edge_direction(current_edge)
            candidates = [
                candidate
                for candidate in outgoing.get(current_point, [])
                if candidate not in visited and candidate[1] != current_edge[0]
            ]
            if not candidates:
                break

            priority = RIGHT_TURN_PRIORITY.get(current_direction, ())
            next_edge = candidates[0]
            for direction in priority:
                directed = next(
                    (candidate for candidate in candidates if edge_direction(candidate) == direction),
                    None,
                )
                if directed is not None:
                    next_edge = directed
                    break

            current_edge = next_edge
            visited.add(current_edge)
            loop.append(current_edge[1])

        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(simplify_loop(loop))

    return loops


def build_component_footprint(
    component_mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, Any]:
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_longitudes = roll_longitudes(longitudes, roll_start)
    footprint_mask = rolled_mask.any(axis=0)

    latitude_bounds = build_axis_bounds(latitudes.astype(np.float64))
    longitude_bounds = build_axis_bounds(rolled_longitudes.astype(np.float64))
    loops = trace_boundary_loops(boundary_edges_from_footprint_mask(footprint_mask))

    rings: list[list[list[float]]] = []
    for loop in loops:
        ring: list[list[float]] = []
        for longitude_index, latitude_index in loop[:-1]:
            ring.append(
                [
                    float(longitude_bounds[longitude_index]),
                    float(latitude_bounds[latitude_index]),
                ]
            )
        if len(ring) >= 3:
            rings.append(ring)

    filled_coords = np.argwhere(footprint_mask)
    if filled_coords.size == 0:
        latitude_min = latitude_max = float(latitudes[0])
        longitude_min = longitude_max = float(rolled_longitudes[0] % 360.0)
    else:
        latitude_min = float(np.min(latitudes[filled_coords[:, 0]]))
        latitude_max = float(np.max(latitudes[filled_coords[:, 0]]))
        longitude_values = rolled_longitudes[filled_coords[:, 1]]
        longitude_min = float(np.min(longitude_values) % 360.0)
        longitude_max = float(np.max(longitude_values) % 360.0)

    return {
        "occupied_cell_count": int(footprint_mask.sum()),
        "latitude_min_deg": latitude_min,
        "latitude_max_deg": latitude_max,
        "longitude_min_deg": longitude_min,
        "longitude_max_deg": longitude_max,
        "rings": rings,
    }


def build_timestamp_payload(
    field: np.ndarray,
    thresholds: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    closing_radius_cells: int = DEFAULT_CLOSING_RADIUS_CELLS,
    opening_radius_cells: int = DEFAULT_OPENING_RADIUS_CELLS,
    geometry_mode: str = DEFAULT_GEOMETRY_MODE,
    mesh_smoothing_iterations: int = 0,
    mesh_smoothing_lambda: float = 0.45,
    mesh_smoothing_mu: float = -0.47,
    mesh_island_min_vertices: int = 0,
    voxel_exterior_only: bool = False,
    write_footprints: bool = True,
    processed_mask: np.ndarray | None = None,
    component_masks: list[np.ndarray] | None = None,
    component_metadata_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if component_masks is None and processed_mask is None:
        processed_mask = apply_binary_postprocess(
            build_threshold_mask(field, thresholds),
            closing_radius_cells=closing_radius_cells,
            opening_radius_cells=opening_radius_cells,
        )
    elif processed_mask is not None:
        processed_mask = np.asarray(processed_mask, dtype=bool)

    position_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    components: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
    vertex_offset = 0
    index_offset = 0

    if component_masks is None:
        if processed_mask is None:
            raise ValueError("Expected processed_mask when component_masks are not provided.")
        wrapped_components = iter_wrapped_components(
            processed_mask,
            min_component_size=min_component_size,
        )
        component_masks = wrapped_components
    else:
        component_masks = [np.asarray(mask, dtype=bool) for mask in component_masks]

    if component_metadata_overrides is None:
        component_metadata_overrides = [{} for _ in component_masks]
    elif len(component_metadata_overrides) != len(component_masks):
        raise ValueError("Expected component metadata overrides to match component masks.")

    for component_id, component_mask in enumerate(component_masks):
        voxel_count = int(component_mask.sum())

        surface = build_component_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            gaussian_sigma=gaussian_sigma,
            geometry_mode=geometry_mode,
            mesh_smoothing_iterations=mesh_smoothing_iterations,
            mesh_smoothing_lambda=mesh_smoothing_lambda,
            mesh_smoothing_mu=mesh_smoothing_mu,
            mesh_island_min_vertices=mesh_island_min_vertices,
            voxel_exterior_only=voxel_exterior_only,
        )
        if surface is None:
            continue

        positions, faces, metadata = surface
        flat_positions = positions.reshape(-1)
        flat_indices = faces.reshape(-1) + vertex_offset
        metadata_override = component_metadata_overrides[component_id]
        component_record_id = int(metadata_override.get("id", component_id))
        component_metadata = {
            "id": component_record_id,
            "vertex_offset": int(vertex_offset),
            "vertex_count": int(positions.shape[0]),
            "index_offset": int(index_offset),
            "index_count": int(flat_indices.size),
            **metadata,
            **{
                key: value
                for key, value in metadata_override.items()
                if key != "id"
            },
        }

        position_chunks.append(flat_positions)
        index_chunks.append(flat_indices.astype(np.uint32))
        components.append(component_metadata)
        if write_footprints:
            footprints.append(
                {
                    "id": component_record_id,
                    **build_component_footprint(
                        component_mask=component_mask,
                        latitudes=latitudes,
                        longitudes=longitudes,
                    ),
                }
            )

        vertex_offset += positions.shape[0]
        index_offset += flat_indices.size

    positions = (
        np.concatenate(position_chunks).astype(np.float32)
        if position_chunks
        else np.empty((0,), dtype=np.float32)
    )
    indices = (
        np.concatenate(index_chunks).astype(np.uint32)
        if index_chunks
        else np.empty((0,), dtype=np.uint32)
    )

    return {
        "positions": positions,
        "indices": indices,
        "components": components,
        "component_count": len(components),
        "vertex_count": int(positions.size // 3),
        "index_count": int(indices.size),
        "voxel_count": int(
            processed_mask.sum()
            if processed_mask is not None
            else sum(int(mask.sum()) for mask in component_masks)
        ),
        "footprints": footprints,
    }


def build_threshold_table(
    pressure_levels: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    return [
        {
            "pressure_hpa": float(pressure),
            "threshold": float(threshold),
        }
        for pressure, threshold in zip(pressure_levels, thresholds, strict=True)
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


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


def write_timestamp_assets(
    output_dir: Path,
    timestamp: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    positions_path = frame_dir / "positions.bin"
    indices_path = frame_dir / "indices.bin"
    metadata_path = frame_dir / "metadata.json"
    footprints_path = frame_dir / "footprints.json"

    payload["positions"].astype("<f4").tofile(positions_path)
    payload["indices"].astype("<u4").tofile(indices_path)
    if payload.get("footprints"):
        write_json(
            footprints_path,
            {
                "version": OUTPUT_VERSION,
                "timestamp": timestamp,
                "components": payload["footprints"],
            },
        )

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": timestamp,
        "component_count": payload["component_count"],
        "vertex_count": payload["vertex_count"],
        "index_count": payload["index_count"],
        "thresholded_voxel_count": payload["voxel_count"],
        "components": payload["components"],
        "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
    }
    write_json(metadata_path, metadata)

    entry = {
        "timestamp": timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "positions": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "component_count": payload["component_count"],
        "vertex_count": payload["vertex_count"],
        "index_count": payload["index_count"],
    }
    if payload.get("footprints"):
        entry["footprints"] = str(footprints_path.relative_to(output_dir)).replace(
            "\\", "/"
        )
    return entry


def build_assets(config: BuildConfig) -> dict[str, Any]:
    dataset = xr.open_dataset(config.dataset_path, chunks={})
    raw_dataset = netCDF4.Dataset(config.dataset_path, mode="r")
    try:
        effective_geometry_mode = resolve_geometry_mode(
            config.segmentation_mode,
            config.geometry_mode,
        )
        specific_humidity = dataset[DATASET_VARIABLE]
        specific_humidity_var = raw_dataset.variables[DATASET_VARIABLE]
        pressure_levels = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        latitudes = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
        timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(dataset.coords["valid_time"].values)
        ]
        allowed_timestamps = (
            {timestamp for timestamp in config.include_timestamps}
            if config.include_timestamps
            else None
        )

        print(
            f"Preparing segmentation recipe {config.segmentation_mode}...",
            flush=True,
        )
        threshold_seed_sample = load_threshold_seed_sample(specific_humidity_var)
        segmentation_context = prepare_segmentation_context(
            threshold_seed_sample,
            config,
        )
        thresholds = segmentation_context.primary_thresholds
        print("Segmentation recipe ready.", flush=True)
        radius_lookup = build_radius_lookup(
            pressure_levels_hpa=pressure_levels,
            base_radius=config.base_radius,
            vertical_span=config.vertical_span,
        )
        effective_latitudes = latitudes[:: segmentation_context.latitude_stride]
        effective_longitudes = longitudes[:: segmentation_context.longitude_stride]

        preserve_children = {"variants"} if config.segmentation_mode == "p95-close" else set()
        clear_output_dir(config.output_dir, preserve_child_names=preserve_children)

        entries: list[dict[str, Any]] = []
        time_count = specific_humidity.sizes["valid_time"]
        processed_count = 0

        for start in range(0, time_count, config.time_window):
            stop = min(time_count, start + config.time_window)
            print(f"Loading source window {start}:{stop}...", flush=True)
            window = np.asarray(specific_humidity_var[start:stop, :, :, :], dtype=np.float32)

            for local_index in range(window.shape[0]):
                absolute_index = start + local_index
                timestamp = timestamps[absolute_index]
                if allowed_timestamps is not None and timestamp not in allowed_timestamps:
                    continue
                print(f"Building {timestamp}...", flush=True)
                field = downsample_lat_lon_values(
                    window[local_index],
                    latitude_stride=segmentation_context.latitude_stride,
                    longitude_stride=segmentation_context.longitude_stride,
                )
                processed_mask = None
                component_masks = None
                component_metadata_overrides = None
                if config.segmentation_mode in {
                    BUCKET_SEGMENTATION_MODE,
                    BUCKET_GLOBAL_SEGMENTATION_MODE,
                }:
                    bucket_specs = build_bucket_component_specs(field, segmentation_context)
                    component_masks = [spec["mask"] for spec in bucket_specs]
                    component_metadata_overrides = [
                        spec["metadata"] for spec in bucket_specs
                    ]
                else:
                    processed_mask = build_segmentation_mask(
                        field,
                        segmentation_context,
                    )
                payload = build_timestamp_payload(
                    field=field,
                    thresholds=thresholds,
                    pressure_levels=pressure_levels,
                    latitudes=effective_latitudes,
                    longitudes=effective_longitudes,
                    radius_lookup=radius_lookup,
                    min_component_size=config.min_component_size,
                    gaussian_sigma=config.gaussian_sigma,
                    closing_radius_cells=config.closing_radius_cells,
                    opening_radius_cells=config.opening_radius_cells,
                    geometry_mode=effective_geometry_mode,
                    mesh_smoothing_iterations=config.mesh_smoothing_iterations,
                    mesh_smoothing_lambda=config.mesh_smoothing_lambda,
                    mesh_smoothing_mu=config.mesh_smoothing_mu,
                    mesh_island_min_vertices=config.mesh_island_min_vertices,
                    voxel_exterior_only=config.voxel_exterior_only,
                    write_footprints=config.write_footprints,
                    processed_mask=processed_mask,
                    component_masks=component_masks,
                    component_metadata_overrides=component_metadata_overrides,
                )
                entry = write_timestamp_assets(config.output_dir, timestamp, payload)
                entries.append(entry)
                processed_count += 1
                if allowed_timestamps is not None:
                    allowed_timestamps.discard(timestamp)
                    if not allowed_timestamps:
                        break

                if config.limit_timestamps is not None and processed_count >= config.limit_timestamps:
                    break

            if allowed_timestamps is not None and not allowed_timestamps:
                break
            if config.limit_timestamps is not None and processed_count >= config.limit_timestamps:
                break

        manifest = {
            "version": OUTPUT_VERSION,
            "dataset": config.dataset_path.name,
            "variable": DATASET_VARIABLE,
            "units": str(specific_humidity.attrs.get("units", "")),
            "segmentation_mode": config.segmentation_mode,
            "threshold_mode": {
                "kind": segmentation_context.threshold_kind,
                "quantile": segmentation_context.threshold_quantile,
                "minimum_component_size": config.min_component_size,
                "threshold_seed": "midpoint_time_slice",
                "smoothing": {
                    "binary_closing_radius_cells": segmentation_context.closing_radius_cells,
                    "binary_opening_radius_cells": segmentation_context.opening_radius_cells,
                    "gaussian_sigma": config.gaussian_sigma,
                },
                "recipe": segmentation_context.recipe_metadata,
            },
            "geometry_mode": effective_geometry_mode,
            "mesh_postprocess": {
                "iterations": config.mesh_smoothing_iterations,
                "lambda": config.mesh_smoothing_lambda,
                "mu": config.mesh_smoothing_mu,
                "island_min_vertices": config.mesh_island_min_vertices,
            },
            "voxel_exterior_only": config.voxel_exterior_only,
            "globe": {
                "base_radius": config.base_radius,
                "vertical_span": config.vertical_span,
                "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
            },
            "grid": {
                "pressure_level_count": int(pressure_levels.size),
                "latitude_count": int(effective_latitudes.size),
                "longitude_count": int(effective_longitudes.size),
                "latitude_step_degrees": coordinate_step_degrees(effective_latitudes),
                "longitude_step_degrees": coordinate_step_degrees(effective_longitudes),
            },
            "thresholds": build_threshold_table(pressure_levels, thresholds),
            "timestamps": entries,
        }

        write_json(config.output_dir / "index.json", manifest)
        print("Wrote manifest.", flush=True)
        return manifest
    finally:
        raw_dataset.close()
        dataset.close()
