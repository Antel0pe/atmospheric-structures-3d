from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from matplotlib.colors import ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_HOUR = 3600.0
DEFAULT_DATASET = Path(
    "data/era5_source_air_wind_uv_omega_2021-11-05t12_to_2021-11-08t12_3hourly_1000-250hpa.nc"
)
DEFAULT_OUTPUT_DIR = Path("tmp/source-air-trajectories-2021-11-08t12")
DEFAULT_TARGET_TIME = "2021-11-08T12:00"
DEFAULT_LOOKBACK_HOURS = (24, 48, 72)
DEFAULT_STRIDE = 8
DEFAULT_STEP_HOURS = 1.0
DEFAULT_CLUSTER_COUNT = 10
SOURCE_COLOR_CMAP_SIZE = 512


@dataclass(frozen=True)
class WindCube:
    time_hours: np.ndarray
    pressure_hpa: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray
    omega_pa_s: np.ndarray


@dataclass(frozen=True)
class TrajectorySummary:
    lookback_hours: int
    target_pressure_hpa: float
    parcel_count: int
    mean_displacement_km: float
    median_displacement_km: float
    mean_endpoint_pressure_hpa: float
    lower_boundary_fraction: float
    upper_boundary_fraction: float
    source_color_plot: str
    clustered_plot: str


@dataclass(frozen=True)
class ClusterSummary:
    lookback_hours: int
    cluster_id: int
    parcel_count: int
    fraction: float
    center_lon_deg: float
    center_lat_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-color and clustered-source maps from ERA5 pressure-level winds."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    parser.add_argument(
        "--lookback-hours",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOOKBACK_HOURS),
        help="Backward trajectory snapshots to save.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Subsample factor along latitude and longitude before trajectory integration.",
    )
    parser.add_argument(
        "--step-hours",
        type=float,
        default=DEFAULT_STEP_HOURS,
        help="RK4 integration step in hours. The step is negative internally for back trajectories.",
    )
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument(
        "--methods",
        choices=("both", "source-color", "clustered"),
        default="both",
        help="Which plot family to write.",
    )
    parser.add_argument(
        "--max-kmeans-points",
        type=int,
        default=120_000,
        help="Maximum endpoint sample used to fit source-region clusters per lookback.",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="*",
        default=None,
        help="Optional target pressure levels to plot. Defaults to every level in the dataset.",
    )
    return parser.parse_args()


def ensure_plot_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")


def to_repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def normalize_lon_360(lon: np.ndarray) -> np.ndarray:
    return lon % 360.0


def angular_lon_delta_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return ((first - second + 180.0) % 360.0) - 180.0


def great_circle_distance_km(
    lon0_deg: np.ndarray,
    lat0_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat1_deg: np.ndarray,
) -> np.ndarray:
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    lon1 = np.radians(lon1_deg)
    lat1 = np.radians(lat1_deg)
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2
    return (2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))) / 1000.0


def find_nc_name(dataset: Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables or name in dataset.dimensions:
            return name
    raise KeyError(f"Could not find any NetCDF variable or dimension named {candidates}.")


def load_wind_cube(
    path: Path,
    target_time: np.datetime64,
    max_lookback_hours: int,
    stride: int,
    requested_levels: list[float] | None,
) -> WindCube:
    dataset = Dataset(path.expanduser().resolve())
    try:
        time_name = find_nc_name(dataset, ("valid_time", "time"))
        level_name = find_nc_name(dataset, ("pressure_level", "level", "isobaricInhPa"))
        lat_name = find_nc_name(dataset, ("latitude", "lat"))
        lon_name = find_nc_name(dataset, ("longitude", "lon"))
        u_name = find_nc_name(dataset, ("u", "u_component_of_wind"))
        v_name = find_nc_name(dataset, ("v", "v_component_of_wind"))
        omega_name = find_nc_name(dataset, ("w", "omega", "vertical_velocity"))

        start_time = target_time - np.timedelta64(int(max_lookback_hours), "h")
        raw_times = np.asarray(dataset.variables[time_name][:], dtype=np.int64).astype("datetime64[s]")
        time_indices = np.flatnonzero((raw_times >= start_time) & (raw_times <= target_time))
        if time_indices.size < 2:
            raise ValueError("Trajectory integration needs at least two wind time slices.")
        time_indexer = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
        times = raw_times[time_indices]
        if times.size < 2:
            raise ValueError("Trajectory integration needs at least two wind time slices.")
        target_seconds = target_time.astype("datetime64[s]").astype(np.int64)
        time_hours = (times.astype("datetime64[s]").astype(np.int64) - target_seconds) / SECONDS_PER_HOUR

        all_pressures = np.asarray(dataset.variables[level_name][:], dtype=np.float32)
        if requested_levels:
            level_indices = np.asarray(
                [int(np.argmin(np.abs(all_pressures - float(level)))) for level in requested_levels],
                dtype=np.int64,
            )
            level_indices = np.unique(level_indices)
            level_indexer: slice | np.ndarray = level_indices
        else:
            level_indices = np.arange(all_pressures.size, dtype=np.int64)
            level_indexer = slice(None)
        pressures = all_pressures[level_indices]

        all_lats = np.asarray(dataset.variables[lat_name][:], dtype=np.float32)
        all_lons = normalize_lon_360(np.asarray(dataset.variables[lon_name][:], dtype=np.float32))
        lat_indices = np.arange(0, all_lats.size, stride, dtype=np.int64)
        lon_indices = np.arange(0, all_lons.size, stride, dtype=np.int64)
        use_contiguous_grid_read = stride > 1 and len(level_indices) <= 4
        lat_indexer = slice(None) if use_contiguous_grid_read else slice(None, None, stride)
        lon_indexer = slice(None) if use_contiguous_grid_read else slice(None, None, stride)

        selector = (time_indexer, level_indexer, lat_indexer, lon_indexer)
        u = np.asarray(dataset.variables[u_name][selector], dtype=np.float32)
        v = np.asarray(dataset.variables[v_name][selector], dtype=np.float32)
        omega = np.asarray(dataset.variables[omega_name][selector], dtype=np.float32)
        if use_contiguous_grid_read:
            u = u[:, :, lat_indices, :][:, :, :, lon_indices]
            v = v[:, :, lat_indices, :][:, :, :, lon_indices]
            omega = omega[:, :, lat_indices, :][:, :, :, lon_indices]
        lats = all_lats[lat_indices]
        lons = all_lons[lon_indices]

        if np.any(np.diff(pressures) < 0):
            order = np.argsort(pressures)
            pressures = pressures[order]
            u = u[:, order, :, :]
            v = v[:, order, :, :]
            omega = omega[:, order, :, :]

        if np.any(np.diff(lats) < 0):
            lats = lats[::-1]
            u = u[:, :, ::-1, :]
            v = v[:, :, ::-1, :]
            omega = omega[:, :, ::-1, :]

        lon_order = np.argsort(lons)
        lons = lons[lon_order]
        u = u[:, :, :, lon_order]
        v = v[:, :, :, lon_order]
        omega = omega[:, :, :, lon_order]

        return WindCube(
            time_hours=np.asarray(time_hours, dtype=np.float32),
            pressure_hpa=pressures,
            latitude_deg=lats,
            longitude_deg=lons,
            u_ms=u,
            v_ms=v,
            omega_pa_s=omega,
        )
    finally:
        dataset.close()


def trilinear_sample(
    field: np.ndarray,
    pressures: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    pressure_hpa: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    p = np.clip(pressure_hpa, float(pressures[0]), float(pressures[-1]))
    y = np.clip(lat_deg, float(lats[0]), float(lats[-1]))

    p0 = np.searchsorted(pressures, p, side="right") - 1
    p0 = np.clip(p0, 0, len(pressures) - 2)
    p1 = p0 + 1
    wp = (p - pressures[p0]) / np.maximum(1e-6, pressures[p1] - pressures[p0])

    y0 = np.searchsorted(lats, y, side="right") - 1
    y0 = np.clip(y0, 0, len(lats) - 2)
    y1 = y0 + 1
    wy = (y - lats[y0]) / np.maximum(1e-6, lats[y1] - lats[y0])

    lon0 = float(lons[0])
    dx = float(np.median(np.diff(lons)))
    x_float = ((normalize_lon_360(lon_deg) - lon0) % 360.0) / dx
    x0 = np.floor(x_float).astype(np.int64) % len(lons)
    x1 = (x0 + 1) % len(lons)
    wx = x_float - np.floor(x_float)

    c000 = field[p0, y0, x0]
    c001 = field[p0, y0, x1]
    c010 = field[p0, y1, x0]
    c011 = field[p0, y1, x1]
    c100 = field[p1, y0, x0]
    c101 = field[p1, y0, x1]
    c110 = field[p1, y1, x0]
    c111 = field[p1, y1, x1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wp) + c1 * wp


def spatial_interpolation_weights(
    pressures: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    pressure_hpa: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(pressures) == 1:
        p0 = np.zeros(pressure_hpa.shape, dtype=np.int64)
        p1 = p0
        wp = np.zeros(pressure_hpa.shape, dtype=np.float32)
    else:
        p = np.clip(pressure_hpa, float(pressures[0]), float(pressures[-1]))
        p0 = np.searchsorted(pressures, p, side="right") - 1
        p0 = np.clip(p0, 0, len(pressures) - 2)
        p1 = p0 + 1
        wp = ((p - pressures[p0]) / np.maximum(1e-6, pressures[p1] - pressures[p0])).astype(np.float32)

    y = np.clip(lat_deg, float(lats[0]), float(lats[-1]))
    y0 = np.searchsorted(lats, y, side="right") - 1
    y0 = np.clip(y0, 0, len(lats) - 2)
    y1 = y0 + 1
    wy = ((y - lats[y0]) / np.maximum(1e-6, lats[y1] - lats[y0])).astype(np.float32)

    lon0 = float(lons[0])
    dx = float(np.median(np.diff(lons)))
    x_float = ((normalize_lon_360(lon_deg) - lon0) % 360.0) / dx
    x_floor = np.floor(x_float)
    x0 = x_floor.astype(np.int64) % len(lons)
    x1 = (x0 + 1) % len(lons)
    wx = (x_float - x_floor).astype(np.float32)
    return p0, p1, wp, y0, y1, wy, x0, x1, wx


def sample_field_with_weights(
    field: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    wp: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    wy: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    wx: np.ndarray,
) -> np.ndarray:
    if field.shape[0] == 1:
        c00 = field[0, y0, x0]
        c01 = field[0, y1, x0]
        c0 = c00 * (1.0 - wy) + c01 * wy

        c10 = field[0, y0, x1]
        c11 = field[0, y1, x1]
        c1 = c10 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wx) + c1 * wx

    c000 = field[p0, y0, x0]
    c001 = field[p0, y0, x1]
    c010 = field[p0, y1, x0]
    c011 = field[p0, y1, x1]
    c100 = field[p1, y0, x0]
    c101 = field[p1, y0, x1]
    c110 = field[p1, y1, x0]
    c111 = field[p1, y1, x1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wp) + c1 * wp


def sample_wind(cube: WindCube, lon: np.ndarray, lat: np.ndarray, pressure: np.ndarray, time_hour: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = float(np.clip(time_hour, float(cube.time_hours[0]), float(cube.time_hours[-1])))
    t0 = int(np.searchsorted(cube.time_hours, t, side="right") - 1)
    t0 = max(0, min(t0, len(cube.time_hours) - 2))
    t1 = t0 + 1
    wt = (t - float(cube.time_hours[t0])) / max(1e-6, float(cube.time_hours[t1] - cube.time_hours[t0]))

    weights = spatial_interpolation_weights(
        cube.pressure_hpa, cube.latitude_deg, cube.longitude_deg, pressure, lat, lon
    )
    values: list[np.ndarray] = []
    for field in (cube.u_ms, cube.v_ms, cube.omega_pa_s):
        first = sample_field_with_weights(field[t0], *weights)
        second = sample_field_with_weights(field[t1], *weights)
        values.append(first * (1.0 - wt) + second * wt)
    return values[0], values[1], values[2]


def derivatives(cube: WindCube, lon: np.ndarray, lat: np.ndarray, pressure: np.ndarray, time_hour: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, omega = sample_wind(cube, lon, lat, pressure, time_hour)
    lat_rad = np.radians(np.clip(lat, -89.0, 89.0))
    cos_lat = np.maximum(0.02, np.cos(lat_rad))
    dlon_dt = np.degrees(u / (EARTH_RADIUS_M * cos_lat)) * SECONDS_PER_HOUR
    dlat_dt = np.degrees(v / EARTH_RADIUS_M) * SECONDS_PER_HOUR
    dp_dt = (omega / 100.0) * SECONDS_PER_HOUR
    return dlon_dt, dlat_dt, dp_dt


def rk4_step(
    cube: WindCube,
    lon: np.ndarray,
    lat: np.ndarray,
    pressure: np.ndarray,
    time_hour: float,
    step_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1_lon, k1_lat, k1_p = derivatives(cube, lon, lat, pressure, time_hour)
    k2_lon, k2_lat, k2_p = derivatives(
        cube,
        lon + 0.5 * step_hours * k1_lon,
        lat + 0.5 * step_hours * k1_lat,
        pressure + 0.5 * step_hours * k1_p,
        time_hour + 0.5 * step_hours,
    )
    k3_lon, k3_lat, k3_p = derivatives(
        cube,
        lon + 0.5 * step_hours * k2_lon,
        lat + 0.5 * step_hours * k2_lat,
        pressure + 0.5 * step_hours * k2_p,
        time_hour + 0.5 * step_hours,
    )
    k4_lon, k4_lat, k4_p = derivatives(
        cube,
        lon + step_hours * k3_lon,
        lat + step_hours * k3_lat,
        pressure + step_hours * k3_p,
        time_hour + step_hours,
    )

    next_lon = lon + (step_hours / 6.0) * (k1_lon + 2.0 * k2_lon + 2.0 * k3_lon + k4_lon)
    next_lat = lat + (step_hours / 6.0) * (k1_lat + 2.0 * k2_lat + 2.0 * k3_lat + k4_lat)
    next_pressure = pressure + (step_hours / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)

    return (
        normalize_lon_360(next_lon).astype(np.float32),
        np.clip(next_lat, -89.0, 89.0).astype(np.float32),
        np.clip(next_pressure, float(cube.pressure_hpa[0]), float(cube.pressure_hpa[-1])).astype(np.float32),
    )


def build_initial_states(cube: WindCube, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(cube.longitude_deg, cube.latitude_deg)
    flat_lon = lon_grid.reshape(-1).astype(np.float32)
    flat_lat = lat_grid.reshape(-1).astype(np.float32)
    level_count = len(levels)
    cell_count = flat_lon.size

    target_lon = np.tile(flat_lon, level_count)
    target_lat = np.tile(flat_lat, level_count)
    target_pressure = np.repeat(levels.astype(np.float32), cell_count)
    level_index = np.repeat(np.arange(level_count, dtype=np.int32), cell_count)
    cell_index = np.tile(np.arange(cell_count, dtype=np.int32), level_count)
    return target_lon.copy(), target_lat.copy(), target_pressure.copy(), level_index, cell_index, target_pressure.copy()


def run_trajectories(
    cube: WindCube,
    levels: np.ndarray,
    lookback_hours: list[int],
    step_hours: float,
) -> dict[int, dict[str, np.ndarray]]:
    lon, lat, pressure, level_index, cell_index, target_pressure = build_initial_states(cube, levels)
    target_lon = lon.copy()
    target_lat = lat.copy()
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    max_lookback = max(lookback_hours)
    step = -abs(float(step_hours))
    total_steps = int(math.ceil(max_lookback / abs(step)))
    wanted = set(int(item) for item in lookback_hours)

    elapsed = 0.0
    current_time = 0.0
    for _ in range(total_steps):
        remaining = max_lookback - elapsed
        actual_step = -min(abs(step), remaining)
        lon, lat, pressure = rk4_step(cube, lon, lat, pressure, current_time, actual_step)
        current_time += actual_step
        elapsed += abs(actual_step)
        rounded = int(round(elapsed))
        if rounded in wanted and abs(elapsed - rounded) < 1e-5:
            snapshots[rounded] = {
                "source_lon": lon.copy(),
                "source_lat": lat.copy(),
                "source_pressure": pressure.copy(),
                "target_lon": target_lon,
                "target_lat": target_lat,
                "target_pressure": target_pressure,
                "level_index": level_index,
                "cell_index": cell_index,
            }
    return snapshots


def source_rgb(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    hue = normalize_lon_360(lon_deg) / 360.0
    saturation = np.full_like(hue, 0.78, dtype=np.float32)
    value = np.clip(0.28 + 0.72 * ((lat_deg + 89.0) / 178.0), 0.18, 1.0).astype(np.float32)
    return hsv_to_rgb(hue, saturation, value)


def hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    h = (hue % 1.0) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    i_mod = i % 6
    rgb = np.empty(hue.shape + (3,), dtype=np.float32)
    choices = [
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    ]
    for index, channels in enumerate(choices):
        mask = i_mod == index
        for channel_index, channel_values in enumerate(channels):
            rgb[..., channel_index][mask] = channel_values[mask]
    return rgb


def endpoint_features(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)]).astype(np.float32)


def feature_to_lon_lat(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = features / np.maximum(1e-6, np.linalg.norm(features, axis=1, keepdims=True))
    lon = normalize_lon_180(np.degrees(np.arctan2(normalized[:, 1], normalized[:, 0])))
    lat = np.degrees(np.arcsin(np.clip(normalized[:, 2], -1.0, 1.0)))
    return lon, lat


def fit_kmeans(features: np.ndarray, cluster_count: int, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if features.shape[0] > max_points:
        sample_index = rng.choice(features.shape[0], size=max_points, replace=False)
        sample = features[sample_index]
    else:
        sample = features

    centers = np.empty((cluster_count, features.shape[1]), dtype=np.float32)
    centers[0] = sample[rng.integers(0, sample.shape[0])]
    distances = np.sum((sample - centers[0]) ** 2, axis=1)
    for center_index in range(1, cluster_count):
        next_index = int(np.argmax(distances))
        centers[center_index] = sample[next_index]
        distances = np.minimum(distances, np.sum((sample - centers[center_index]) ** 2, axis=1))

    for _ in range(40):
        labels = assign_clusters(sample, centers)
        new_centers = centers.copy()
        for cluster_id in range(cluster_count):
            members = sample[labels == cluster_id]
            if members.size == 0:
                new_centers[cluster_id] = sample[rng.integers(0, sample.shape[0])]
            else:
                mean = members.mean(axis=0)
                new_centers[cluster_id] = mean / max(1e-6, float(np.linalg.norm(mean)))
        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift < 1e-4:
            break
    return centers


def assign_clusters(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(features * features, axis=1, keepdims=True)
        - 2.0 * features @ centers.T
        + np.sum(centers * centers, axis=1)
    )
    return np.argmin(distances, axis=1).astype(np.int32)


def cluster_palette(count: int) -> ListedColormap:
    base = plt.get_cmap("tab10")
    colors = np.array([base(index % 10) for index in range(count)])
    return ListedColormap(colors)


def add_source_key(axis: plt.Axes) -> None:
    lon = np.linspace(-180.0, 180.0, SOURCE_COLOR_CMAP_SIZE)
    lat = np.linspace(-89.0, 89.0, SOURCE_COLOR_CMAP_SIZE)
    key_lon, key_lat = np.meshgrid(lon, lat)
    axis.imshow(source_rgb(key_lon, key_lat), extent=(-180, 180, -89, 89), origin="lower", aspect="auto")
    axis.set_title("Source color key")
    axis.set_xlabel("source lon")
    axis.set_ylabel("source lat")
    axis.set_xlim(-180, 180)
    axis.set_ylim(-89, 89)


def format_map_axis(axis: plt.Axes) -> None:
    axis.set_xlim(-180, 180)
    axis.set_ylim(-89, 89)
    axis.set_xlabel("target lon")
    axis.set_ylabel("target lat")
    axis.set_xticks(np.arange(-180, 181, 60))
    axis.set_yticks(np.arange(-60, 91, 30))
    axis.grid(color="0.75", linewidth=0.4, alpha=0.45)


def plot_lon_order(lon_grid: np.ndarray) -> np.ndarray:
    return np.argsort(normalize_lon_180(lon_grid[0]))


def save_source_color_plot(
    output_path: Path,
    target_lon_grid: np.ndarray,
    target_lat_grid: np.ndarray,
    source_lon_grid: np.ndarray,
    source_lat_grid: np.ndarray,
    lookback_hours: int,
    pressure_hpa: float,
) -> None:
    order = plot_lon_order(target_lon_grid)
    rgb = source_rgb(source_lon_grid[:, order], source_lat_grid[:, order])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].imshow(
        rgb,
        extent=(-180, 180, float(np.nanmin(target_lat_grid)), float(np.nanmax(target_lat_grid))),
        origin="lower",
        aspect="auto",
    )
    axes[0].set_title(f"{lookback_hours}h source-color map, target {pressure_hpa:.0f} hPa")
    format_map_axis(axes[0])
    add_source_key(axes[1])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_cluster_plot(
    output_path: Path,
    target_lon_grid: np.ndarray,
    target_lat_grid: np.ndarray,
    source_lon_grid: np.ndarray,
    source_lat_grid: np.ndarray,
    cluster_grid: np.ndarray,
    centers: np.ndarray,
    lookback_hours: int,
    pressure_hpa: float,
) -> None:
    cluster_count = centers.shape[0]
    cmap = cluster_palette(cluster_count)
    center_lon, center_lat = feature_to_lon_lat(centers)
    order = plot_lon_order(target_lon_grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    image = axes[0].pcolormesh(
        normalize_lon_180(target_lon_grid[:, order]),
        target_lat_grid[:, order],
        cluster_grid[:, order],
        shading="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=cluster_count - 0.5,
    )
    axes[0].set_title(f"{lookback_hours}h clustered source regions, target {pressure_hpa:.0f} hPa")
    format_map_axis(axes[0])
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("source cluster")

    axes[1].scatter(
        normalize_lon_180(source_lon_grid.reshape(-1)),
        source_lat_grid.reshape(-1),
        c=cluster_grid.reshape(-1),
        cmap=cmap,
        vmin=-0.5,
        vmax=cluster_count - 0.5,
        s=3,
        linewidths=0,
        alpha=0.45,
    )
    axes[1].scatter(center_lon, center_lat, c=np.arange(cluster_count), cmap=cmap, s=90, edgecolors="black")
    for cluster_id, (lon, lat) in enumerate(zip(center_lon, center_lat)):
        axes[1].text(lon, lat, str(cluster_id), ha="center", va="center", fontsize=8, color="white")
    axes[1].set_title("Endpoint clusters")
    axes[1].set_xlabel("source lon")
    axes[1].set_ylabel("source lat")
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-89, 89)
    axes[1].grid(color="0.75", linewidth=0.4, alpha=0.45)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def reshape_level_values(values: np.ndarray, level_mask: np.ndarray, cell_count: int) -> np.ndarray:
    grid = np.full(cell_count, np.nan, dtype=np.float32)
    cell_index = level_mask["cell_index"]
    grid[cell_index] = values
    return grid


def build_outputs(
    cube: WindCube,
    snapshots: dict[int, dict[str, np.ndarray]],
    levels: np.ndarray,
    output_dir: Path,
    cluster_count: int,
    max_kmeans_points: int,
    methods: str,
) -> tuple[list[TrajectorySummary], list[ClusterSummary]]:
    lon_grid, lat_grid = np.meshgrid(cube.longitude_deg, cube.latitude_deg)
    flat_shape = lon_grid.shape
    cell_count = lon_grid.size
    summaries: list[TrajectorySummary] = []
    cluster_summaries: list[ClusterSummary] = []

    for lookback_hours in sorted(snapshots):
        snapshot = snapshots[lookback_hours]
        lookback_dir = output_dir / f"{lookback_hours:02d}h"
        source_color_dir = lookback_dir / "source-color-map"
        cluster_dir = lookback_dir / "clustered-source-regions"
        if methods in {"both", "source-color"}:
            source_color_dir.mkdir(parents=True, exist_ok=True)
        if methods in {"both", "clustered"}:
            cluster_dir.mkdir(parents=True, exist_ok=True)

        features = endpoint_features(snapshot["source_lon"], snapshot["source_lat"])
        centers = fit_kmeans(
            features,
            cluster_count=cluster_count,
            max_points=max_kmeans_points,
            seed=10_000 + lookback_hours,
        )
        cluster_labels = assign_clusters(features, centers)
        center_lon, center_lat = feature_to_lon_lat(centers)
        counts = np.bincount(cluster_labels, minlength=cluster_count)
        for cluster_id, count in enumerate(counts):
            cluster_summaries.append(
                ClusterSummary(
                    lookback_hours=lookback_hours,
                    cluster_id=cluster_id,
                    parcel_count=int(count),
                    fraction=float(count / max(1, cluster_labels.size)),
                    center_lon_deg=float(center_lon[cluster_id]),
                    center_lat_deg=float(center_lat[cluster_id]),
                )
            )

        for level_position, pressure_hpa in enumerate(levels):
            mask = snapshot["level_index"] == level_position
            level_data = {
                "cell_index": snapshot["cell_index"][mask],
            }
            source_lon = reshape_level_values(snapshot["source_lon"][mask], level_data, cell_count).reshape(flat_shape)
            source_lat = reshape_level_values(snapshot["source_lat"][mask], level_data, cell_count).reshape(flat_shape)
            source_pressure = reshape_level_values(snapshot["source_pressure"][mask], level_data, cell_count)
            target_lon = reshape_level_values(snapshot["target_lon"][mask], level_data, cell_count)
            target_lat = reshape_level_values(snapshot["target_lat"][mask], level_data, cell_count)
            cluster_grid = reshape_level_values(cluster_labels[mask].astype(np.float32), level_data, cell_count).reshape(flat_shape)

            pressure_slug = f"{int(round(float(pressure_hpa))):04d}hpa"
            source_color_plot = source_color_dir / f"source_color_{pressure_slug}.png"
            clustered_plot = cluster_dir / f"clustered_sources_{pressure_slug}.png"

            if methods in {"both", "source-color"}:
                save_source_color_plot(
                    source_color_plot,
                    lon_grid,
                    lat_grid,
                    source_lon,
                    source_lat,
                    lookback_hours,
                    float(pressure_hpa),
                )
            if methods in {"both", "clustered"}:
                save_cluster_plot(
                    clustered_plot,
                    lon_grid,
                    lat_grid,
                    source_lon,
                    source_lat,
                    cluster_grid,
                    centers,
                    lookback_hours,
                    float(pressure_hpa),
                )

            distance = great_circle_distance_km(target_lon, target_lat, snapshot["source_lon"][mask], snapshot["source_lat"][mask])
            summaries.append(
                TrajectorySummary(
                    lookback_hours=lookback_hours,
                    target_pressure_hpa=float(pressure_hpa),
                    parcel_count=int(mask.sum()),
                    mean_displacement_km=float(np.nanmean(distance)),
                    median_displacement_km=float(np.nanmedian(distance)),
                    mean_endpoint_pressure_hpa=float(np.nanmean(source_pressure)),
                    lower_boundary_fraction=float(np.mean(source_pressure >= cube.pressure_hpa[-1] - 1e-3)),
                    upper_boundary_fraction=float(np.mean(source_pressure <= cube.pressure_hpa[0] + 1e-3)),
                    source_color_plot=to_repo_relative(source_color_plot)
                    if methods in {"both", "source-color"}
                    else "",
                    clustered_plot=to_repo_relative(clustered_plot)
                    if methods in {"both", "clustered"}
                    else "",
                )
            )

    return summaries, cluster_summaries


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    cube: WindCube,
    levels: np.ndarray,
    trajectory_summaries: list[TrajectorySummary],
    cluster_summaries: list[ClusterSummary],
) -> None:
    payload: dict[str, Any] = {
        "dataset": to_repo_relative(args.dataset),
        "target_time": args.target_time,
        "lookback_hours": [int(item) for item in args.lookback_hours],
        "rk4_step_hours": float(args.step_hours),
        "grid_stride": int(args.stride),
        "cluster_count": int(args.cluster_count),
        "methods": args.methods,
        "time_hours_relative_to_target": [float(item) for item in cube.time_hours],
        "target_pressure_levels_hpa": [float(item) for item in levels],
        "latitude_count": int(len(cube.latitude_deg)),
        "longitude_count": int(len(cube.longitude_deg)),
        "trajectory_summaries": [asdict(item) for item in trajectory_summaries],
        "cluster_summaries": [asdict(item) for item in cluster_summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Source Air Trajectory Prototype",
        "",
        f"- dataset: `{payload['dataset']}`",
        f"- target time: `{args.target_time}`",
        f"- lookbacks: `{', '.join(str(item) + 'h' for item in args.lookback_hours)}`",
        f"- RK4 step: `{args.step_hours}h`",
        f"- sampled grid: `{len(cube.latitude_deg)} x {len(cube.longitude_deg)}` from stride `{args.stride}`",
        f"- target levels: `{', '.join(str(int(round(item))) for item in levels)} hPa`",
        f"- cluster count: `{args.cluster_count}` per lookback",
        "",
        "## Output Layout",
        "",
        f"- selected methods: `{args.methods}`",
        "- lookback folders are named `24h/`, `48h/`, and `72h/`",
        "",
        "## Notes",
        "",
        "- Source-color maps color target cells by the trajectory endpoint longitude and latitude.",
        "- Clustered-source maps fit endpoint clusters per lookback, then color target cells by source cluster.",
        "- Pressure motion uses ERA5 vertical velocity and is clamped to the available `250-1000 hPa` domain.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_plot_env()
    args = parse_args()
    target_time = np.datetime64(args.target_time)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cube = load_wind_cube(args.dataset, target_time, max(args.lookback_hours), args.stride, args.levels)
    levels = cube.pressure_hpa.copy()

    snapshots = run_trajectories(cube, levels=levels, lookback_hours=list(args.lookback_hours), step_hours=args.step_hours)
    trajectory_summaries, cluster_summaries = build_outputs(
        cube,
        snapshots,
        levels=levels,
        output_dir=output_dir,
        cluster_count=args.cluster_count,
        max_kmeans_points=args.max_kmeans_points,
        methods=args.methods,
    )
    write_summary(output_dir, args, cube, levels, trajectory_summaries, cluster_summaries)

    print(json.dumps({
        "output_dir": to_repo_relative(output_dir),
        "plot_count": len(trajectory_summaries)
        * (2 if args.methods == "both" else 1),
        "levels": [float(item) for item in levels],
    }, indent=2))


if __name__ == "__main__":
    main()
