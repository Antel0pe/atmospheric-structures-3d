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
DEFAULT_GEOMETRY_MODE = "marching-cubes"
DEFAULT_CLOSING_RADIUS_CELLS = 1
DEFAULT_SEGMENTATION_MODE = "p95-close"
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
    segmentation_mode: str = DEFAULT_SEGMENTATION_MODE
    write_footprints: bool = True
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


def build_component_voxel_surface(
    component_mask: np.ndarray,
    field: np.ndarray,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    roll_start = choose_longitude_roll(component_mask)
    rolled_mask = np.roll(component_mask, -roll_start, axis=2)
    rolled_longitudes = roll_longitudes(longitudes, roll_start)

    cropped_mask, mins, maxs = crop_mask(rolled_mask, pad=0)
    if not bool(cropped_mask.any()):
        return None

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

        if r0 == 0 or not cropped_mask[r0 - 1, a0, o0]:
            append_quad(
                [(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        if r0 == level_count - 1 or not cropped_mask[r0 + 1, a0, o0]:
            append_quad(
                [(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        if a0 == 0 or not cropped_mask[r0, a0 - 1, o0]:
            append_quad(
                [(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        if a0 == latitude_count - 1 or not cropped_mask[r0, a0 + 1, o0]:
            append_quad(
                [(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        if o0 == 0 or not cropped_mask[r0, a0, o0 - 1]:
            append_quad(
                [(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                vertex_lookup,
                positions,
                faces,
                radius_bounds,
                latitude_bounds,
                longitude_bounds,
            )
        if o0 == longitude_count - 1 or not cropped_mask[r0, a0, o0 + 1]:
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if geometry_mode == "voxel-faces":
        return build_component_voxel_surface(
            component_mask=component_mask,
            field=field,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
        )

    if geometry_mode != "marching-cubes":
        raise ValueError(f"Unsupported geometry mode: {geometry_mode}")

    return build_component_marching_cubes_surface(
        component_mask=component_mask,
        field=field,
        pressure_levels=pressure_levels,
        latitudes=latitudes,
        longitudes=longitudes,
        radius_lookup=radius_lookup,
        gaussian_sigma=gaussian_sigma,
    )


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
    geometry_mode: str = DEFAULT_GEOMETRY_MODE,
    write_footprints: bool = True,
) -> dict[str, Any]:
    raw_mask = build_threshold_mask(field, thresholds)
    closed_mask = lightly_close_mask(raw_mask, closing_radius_cells=closing_radius_cells)

    position_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    components: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
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
            geometry_mode=geometry_mode,
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
        if write_footprints:
            footprints.append(
                {
                    "id": component_id,
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
        "voxel_count": int(closed_mask.sum()),
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
                    closing_radius_cells=config.closing_radius_cells,
                    geometry_mode=config.geometry_mode,
                    write_footprints=config.write_footprints,
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
            "segmentation_mode": config.segmentation_mode,
            "threshold_mode": {
                "kind": "pressure-relative-quantile",
                "quantile": config.threshold_quantile,
                "minimum_component_size": config.min_component_size,
                "threshold_seed": "midpoint_time_slice",
                "smoothing": {
                    "binary_closing_radius_cells": config.closing_radius_cells,
                    "gaussian_sigma": config.gaussian_sigma,
                },
            },
            "geometry_mode": config.geometry_mode,
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
