from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import netCDF4
import xarray as xr
from scipy import ndimage
from skimage import measure

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
CLOSING_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
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
    limit_timestamps: int | None = None


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
) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError("Expected values with shape (time, level, ...).")

    thresholds = []
    for level_index in range(values.shape[1]):
        level_values = np.asarray(values[:, level_index, ...], dtype=np.float32).ravel()
        finite_values = level_values[np.isfinite(level_values)]
        if finite_values.size == 0:
            thresholds.append(np.nan)
            continue
        thresholds.append(float(np.quantile(finite_values, quantile)))
    return np.asarray(thresholds, dtype=np.float32)


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


def lightly_close_mask(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_closing(mask, structure=CLOSING_STRUCTURE)


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


def build_component_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    wraps_longitude_seam = bool(component_mask[..., 0].any() and component_mask[..., -1].any())
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_field = np.roll(field, -roll_start, axis=2)
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

    component_values = field[component_mask]
    filled_coords = np.argwhere(component_mask)
    pressure_indices = filled_coords[:, 0]
    latitude_indices = filled_coords[:, 1]
    longitude_indices = filled_coords[:, 2]

    rolled_longitude_indices = (longitude_indices - roll_start) % longitudes.size
    longitude_min = float(rolled_longitudes[rolled_longitude_indices].min() % 360.0)
    longitude_max = float(rolled_longitudes[rolled_longitude_indices].max() % 360.0)

    metadata = {
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
    return positions, np.asarray(faces, dtype=np.uint32), metadata


def build_timestamp_payload(
    field: np.ndarray,
    thresholds: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> dict[str, Any]:
    raw_mask = build_threshold_mask(field, thresholds)
    closed_mask = lightly_close_mask(raw_mask)

    position_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    components: list[dict[str, Any]] = []
    vertex_offset = 0
    index_offset = 0

    wrapped_components = iter_wrapped_components(
        closed_mask,
        min_component_size=min_component_size,
    )

    for component_id, component_mask in enumerate(wrapped_components):
        voxel_count = int(component_mask.sum())

        surface = build_component_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            gaussian_sigma=gaussian_sigma,
        )
        if surface is None:
            continue

        positions, faces, metadata = surface
        flat_positions = positions.reshape(-1)
        flat_indices = faces.reshape(-1) + vertex_offset

        position_chunks.append(flat_positions)
        index_chunks.append(flat_indices.astype(np.uint32))
        components.append(
            {
                "id": component_id,
                "vertex_offset": int(vertex_offset),
                "vertex_count": int(positions.shape[0]),
                "index_offset": int(index_offset),
                "index_count": int(flat_indices.size),
                **metadata,
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
        "voxel_count": int(closed_mask.sum()),
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

    payload["positions"].astype("<f4").tofile(positions_path)
    payload["indices"].astype("<u4").tofile(indices_path)

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

    return {
        "timestamp": timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "positions": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "component_count": payload["component_count"],
        "vertex_count": payload["vertex_count"],
        "index_count": payload["index_count"],
    }


def build_assets(config: BuildConfig) -> dict[str, Any]:
    dataset = xr.open_dataset(config.dataset_path, chunks={})
    raw_dataset = netCDF4.Dataset(config.dataset_path, mode="r")
    try:
        specific_humidity = dataset[DATASET_VARIABLE]
        specific_humidity_var = raw_dataset.variables[DATASET_VARIABLE]
        pressure_levels = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        latitudes = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
        timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(dataset.coords["valid_time"].values)
        ]

        print("Computing moisture thresholds...", flush=True)
        thresholds = compute_pressure_thresholds_from_variable(
            specific_humidity_var,
            quantile=config.threshold_quantile,
            time_stride=config.threshold_time_stride,
            sample_stride=config.threshold_sample_stride,
        )
        print("Thresholds ready.", flush=True)
        radius_lookup = build_radius_lookup(
            pressure_levels_hpa=pressure_levels,
            base_radius=config.base_radius,
            vertical_span=config.vertical_span,
        )

        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

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
                print(f"Building {timestamp}...", flush=True)
                payload = build_timestamp_payload(
                    field=window[local_index],
                    thresholds=thresholds,
                    pressure_levels=pressure_levels,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    radius_lookup=radius_lookup,
                    min_component_size=config.min_component_size,
                    gaussian_sigma=config.gaussian_sigma,
                )
                entry = write_timestamp_assets(config.output_dir, timestamp, payload)
                entries.append(entry)
                processed_count += 1

                if config.limit_timestamps is not None and processed_count >= config.limit_timestamps:
                    break

            if config.limit_timestamps is not None and processed_count >= config.limit_timestamps:
                break

        manifest = {
            "version": OUTPUT_VERSION,
            "dataset": config.dataset_path.name,
            "variable": DATASET_VARIABLE,
            "units": str(specific_humidity.attrs.get("units", "")),
            "threshold_mode": {
                "kind": "pressure-relative-quantile",
                "quantile": config.threshold_quantile,
                "minimum_component_size": config.min_component_size,
                "threshold_seed": "midpoint_time_slice",
                "smoothing": {
                    "binary_closing_radius_cells": 1,
                    "gaussian_sigma": config.gaussian_sigma,
                },
            },
            "globe": {
                "base_radius": config.base_radius,
                "vertical_span": config.vertical_span,
                "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
            },
            "grid": {
                "pressure_level_count": int(pressure_levels.size),
                "latitude_count": int(latitudes.size),
                "longitude_count": int(longitudes.size),
                "latitude_step_degrees": float(abs(latitudes[1] - latitudes[0])),
                "longitude_step_degrees": float(abs(longitudes[1] - longitudes[0])),
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
