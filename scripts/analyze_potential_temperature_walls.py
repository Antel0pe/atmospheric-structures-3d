from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY = Path("data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc")
REFERENCE_PRESSURE_HPA = 1000.0
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
PLANAR_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
VOLUME_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
FIELD_SMOOTH_SIGMA = 1.0
KEEP_TOP_PERCENT = 10.0
MIN_WALL_COMPONENT_VOXELS = 150
MIN_WALL_COMPONENT_SPAN_LEVELS = 4
MIN_SURFACE_REGION_CELLS = 24
SURFACE_GROWTH_STD_FLOOR = 0.5
SELECTED_PLOT_LEVELS_HPA = (1000.0, 500.0)


@dataclass(frozen=True)
class ExperimentConfig:
    key: str
    title: str
    short_title: str
    hypothesis: str
    implementation: str
    field_mode: str
    method: str


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        key="exp1_level_mean_gradient",
        title="Experiment 1: Per-level mean anomaly walls",
        short_title="Level-mean anomaly",
        hypothesis=(
            "Removing only the per-level global mean should strip vertical stratification, "
            "but it will likely leave broad meridional background structure that makes the "
            "detected walls too banded and too connected."
        ),
        implementation=(
            "Dry potential temperature minus the global mean at each pressure level, "
            "Gaussian-smoothed, then horizontal gradient magnitude. Keep the top 10% of "
            "gradient score on each level, clean with morphology, and retain only coherent "
            "3D wall components."
        ),
        field_mode="level_mean",
        method="wall_first",
    ),
    ExperimentConfig(
        key="exp2_latitude_band_gradient",
        title="Experiment 2: Latitude-band anomaly walls",
        short_title="Latitude-band anomaly",
        hypothesis=(
            "Subtracting the zonal mean at each latitude and level should suppress the "
            "planetary equator-to-pole background and reveal synoptic-scale walls better "
            "than the simple level-mean anomaly."
        ),
        implementation=(
            "Dry potential temperature minus the zonal mean at each latitude and pressure "
            "level, followed by the same smoothed horizontal-gradient wall extraction and "
            "surface-up coherence analysis."
        ),
        field_mode="latitude_band",
        method="wall_first",
    ),
    ExperimentConfig(
        key="exp3_climatology_gradient",
        title="Experiment 3: Climatology-anomaly walls",
        short_title="Climatology anomaly",
        hypothesis=(
            "Subtracting the matched gridpoint dry-theta climatology should best preserve "
            "meteorologically meaningful air-mass contrasts while removing the background "
            "vertical and geographic structure."
        ),
        implementation=(
            "Dry potential temperature minus the matched 1990-2020 Nov 8 12Z dry-theta "
            "climatological mean field, then the same smoothed horizontal-gradient wall "
            "extraction and 3D connectivity analysis."
        ),
        field_mode="climatology",
        method="wall_first",
    ),
    ExperimentConfig(
        key="exp4_standardized_climatology_gradient",
        title="Experiment 4: Standardized climatology-anomaly walls",
        short_title="Climatology z-score",
        hypothesis=(
            "Standardizing the climatology anomaly by local climatological variability "
            "should balance tropics and extratropics and make the retained walls more "
            "coherent across the full 1000-250 hPa window."
        ),
        implementation=(
            "Dry-theta climatology anomaly divided by climatological standard deviation "
            "(with a 0.5 K floor), then smoothed horizontal-gradient wall extraction "
            "and the same coherence tests."
        ),
        field_mode="climatology_z",
        method="wall_first",
    ),
    ExperimentConfig(
        key="exp5_surface_up_growth",
        title="Experiment 5: Surface-up tilted region growth",
        short_title="Surface-up tilted growth",
        hypothesis=(
            "If the atmosphere contains air masses that remain coherent with height while "
            "tilting and deforming, a surface-seeded, tilt-friendly growth on the "
            "climatology-anomaly field should keep a large fraction of the lower and "
            "mid-troposphere attached to surface-born regions."
        ),
        implementation=(
            "Use the climatology-anomaly wall mask from Experiment 3 as the main barrier. "
            "Seed non-wall regions on 1000 hPa, then propagate labels upward by matching "
            "each cell to the most similar labeled cell one level below within a 3x3 "
            "horizontal neighborhood. Detached aloft regions get new labels."
        ),
        field_mode="climatology",
        method="surface_growth",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run five dry-potential-temperature wall experiments in the 1000-250 hPa "
            "window and export plots, JSON, and a markdown report into tmp/."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latitude-stride", type=int, default=4)
    parser.add_argument("--longitude-stride", type=int, default=4)
    parser.add_argument("--pressure-min-hpa", type=float, default=250.0)
    parser.add_argument("--pressure-max-hpa", type=float, default=1000.0)
    return parser.parse_args()


def resolve_repo_relative(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_relative_text(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def normalize_longitudes(longitudes_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.mod(np.asarray(longitudes_deg, dtype=np.float64) + 180.0, 360.0) - 180.0
    order = np.argsort(normalized, kind="stable")
    return normalized[order].astype(np.float32), order.astype(np.int64)


def reorder_longitude_axis(field: np.ndarray, longitude_order: np.ndarray) -> np.ndarray:
    return np.take(field, longitude_order, axis=-1)


def compute_theta(temperature_k: np.ndarray, pressure_levels_hpa: np.ndarray) -> np.ndarray:
    scale = (REFERENCE_PRESSURE_HPA / pressure_levels_hpa.astype(np.float32)) ** POTENTIAL_TEMPERATURE_KAPPA
    return np.asarray(temperature_k * scale[:, None, None], dtype=np.float32)


def select_pressure_window(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    lower_hpa: float,
    upper_hpa: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = (pressure_levels_hpa >= min(lower_hpa, upper_hpa)) & (
        pressure_levels_hpa <= max(lower_hpa, upper_hpa)
    )
    if not np.any(keep):
        raise ValueError("No pressure levels matched the requested pressure window.")
    return np.asarray(field[keep], dtype=np.float32), np.asarray(pressure_levels_hpa[keep], dtype=np.float32)


def smooth_levels(field: np.ndarray, sigma: float = FIELD_SMOOTH_SIGMA) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(field, dtype=np.float32)
    return np.asarray(
        ndimage.gaussian_filter(
            np.asarray(field, dtype=np.float32),
            sigma=(0.0, sigma, sigma),
            mode=("nearest", "nearest", "wrap"),
            truncate=2.0,
        ),
        dtype=np.float32,
    )


def compute_horizontal_gradient(field: np.ndarray, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    lat_step_deg = float(np.mean(np.abs(np.diff(latitudes_deg))))
    lon_step_deg = float(np.mean(np.abs(np.diff(longitudes_deg))))
    dy_km = 111.0 * max(lat_step_deg, 1e-6)
    gradient_lat = np.gradient(values, axis=1) / dy_km
    wrapped = np.concatenate([values[..., -1:], values, values[..., :1]], axis=2)
    gradient_lon = (wrapped[..., 2:] - wrapped[..., :-2]) * 0.5
    cos_lat = np.maximum(np.cos(np.deg2rad(latitudes_deg))[:, None], 0.15)
    dx_km = 111.0 * max(lon_step_deg, 1e-6) * cos_lat
    gradient_lon = gradient_lon / dx_km[None, :, :]
    return np.asarray(np.sqrt(np.square(gradient_lat) + np.square(gradient_lon)) * 1000.0, dtype=np.float32)


def build_seam_merged_component_info(labels: np.ndarray, seam_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def union(first_label: int, second_label: int) -> None:
        if first_label <= 0 or second_label <= 0:
            return
        root_first = find(first_label)
        root_second = find(second_label)
        if root_first == root_second:
            return
        if root_first < root_second:
            parent[root_second] = root_first
        else:
            parent[root_first] = root_second

    for first_label, second_label in seam_pairs:
        union(int(first_label), int(second_label))

    root_map = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        root_map[label_id] = find(label_id)
    return root_map, np.unique(root_map[1:])


def label_wrapped_planar_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not np.any(occupied):
        return np.zeros_like(occupied, dtype=np.int32), 0
    longitude_count = occupied.shape[1]
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(extended, structure=PLANAR_STRUCTURE)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0
    seam_pairs = np.column_stack([labels[:, 0].reshape(-1), labels[:, -1].reshape(-1)])
    root_map, unique_roots = build_seam_merged_component_info(labels, seam_pairs)
    compact = np.zeros(component_count + 1, dtype=np.int32)
    compact[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    return compact[root_map[labels[:, :longitude_count]]].astype(np.int32), int(unique_roots.size)


def label_wrapped_volume_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not np.any(occupied):
        return np.zeros_like(occupied, dtype=np.int32), 0
    longitude_count = occupied.shape[2]
    extended = np.concatenate([occupied, occupied[..., :1]], axis=2)
    labels, component_count = ndimage.label(extended, structure=VOLUME_STRUCTURE)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0
    seam_pairs = np.column_stack([labels[..., 0].reshape(-1), labels[..., -1].reshape(-1)])
    root_map, unique_roots = build_seam_merged_component_info(labels, seam_pairs)
    compact = np.zeros(component_count + 1, dtype=np.int32)
    compact[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    return compact[root_map[labels[..., :longitude_count]]].astype(np.int32), int(unique_roots.size)


def keep_top_percent_per_level(field: np.ndarray, top_percent: float) -> np.ndarray:
    keep = np.zeros_like(field, dtype=bool)
    quantile = max(0.0, 1.0 - top_percent / 100.0)
    for level_index in range(field.shape[0]):
        level = np.asarray(field[level_index], dtype=np.float32)
        threshold = float(np.quantile(level, quantile))
        keep[level_index] = level >= threshold
    return keep


def clean_wall_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(mask, dtype=bool)
    for level_index in range(mask.shape[0]):
        level_mask = ndimage.binary_closing(mask[level_index], structure=np.ones((3, 3), dtype=np.uint8))
        level_mask = ndimage.binary_opening(level_mask, structure=np.ones((2, 2), dtype=np.uint8))
        cleaned[level_index] = level_mask

    labels, component_count = label_wrapped_volume_components(cleaned)
    if component_count <= 0:
        return cleaned

    retained = np.zeros_like(cleaned, dtype=bool)
    for label_id in range(1, component_count + 1):
        component = labels == label_id
        voxel_count = int(np.count_nonzero(component))
        if voxel_count < MIN_WALL_COMPONENT_VOXELS:
            continue
        levels = np.flatnonzero(np.any(component, axis=(1, 2)))
        if levels.size < MIN_WALL_COMPONENT_SPAN_LEVELS:
            continue
        retained |= component
    return retained


def shift_lat_lon(array: np.ndarray, dy: int, dx: int, fill_value: float) -> np.ndarray:
    shifted = np.roll(array, shift=dx, axis=1)
    if dy > 0:
        shifted = np.concatenate(
            [
                np.full((dy, shifted.shape[1]), fill_value, dtype=shifted.dtype),
                shifted[:-dy],
            ],
            axis=0,
        )
    elif dy < 0:
        dy_abs = abs(dy)
        shifted = np.concatenate(
            [
                shifted[dy_abs:],
                np.full((dy_abs, shifted.shape[1]), fill_value, dtype=shifted.dtype),
            ],
            axis=0,
        )
    return shifted


def estimate_surface_growth_threshold(field: np.ndarray, wall_mask: np.ndarray) -> float:
    candidate_diffs: list[np.ndarray] = []
    for level_index in range(field.shape[0] - 1):
        current_nonwall = ~wall_mask[level_index]
        next_nonwall = ~wall_mask[level_index + 1]
        if not np.any(current_nonwall) or not np.any(next_nonwall):
            continue
        current = field[level_index]
        next_level = field[level_index + 1]
        best = np.full(current.shape, np.inf, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted_next = shift_lat_lon(next_level, dy, dx, np.nan)
                shifted_mask = shift_lat_lon(next_nonwall.astype(np.float32), dy, dx, 0.0) > 0.5
                diff = np.abs(shifted_next - current)
                diff = np.where(current_nonwall & shifted_mask, diff, np.inf)
                best = np.minimum(best, diff)
        finite = np.isfinite(best)
        if np.any(finite):
            candidate_diffs.append(best[finite])
    if not candidate_diffs:
        return 4.0
    merged = np.concatenate(candidate_diffs).astype(np.float32)
    return float(np.quantile(merged, 0.75))


def build_label_boundary_mask(labels: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(labels, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = shift_lat_lon(labels, dy, dx, 0)
        boundaries |= (labels > 0) & (shifted > 0) & (labels != shifted)
    return boundaries


def build_surface_up_labels(field: np.ndarray, wall_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    levels, lat_count, lon_count = field.shape
    labels = np.zeros((levels, lat_count, lon_count), dtype=np.int32)
    birth_levels: list[int] = [0]

    level0_mask = ~wall_mask[0]
    planar_labels, planar_count = label_wrapped_planar_components(level0_mask)
    next_label = 1
    for planar_id in range(1, planar_count + 1):
        component = planar_labels == planar_id
        if int(np.count_nonzero(component)) < MIN_SURFACE_REGION_CELLS:
            continue
        labels[0, component] = next_label
        birth_levels.append(0)
        next_label += 1

    match_threshold = estimate_surface_growth_threshold(field, wall_mask)

    for level_index in range(1, levels):
        level_nonwall = ~wall_mask[level_index]
        current_labels = np.zeros((lat_count, lon_count), dtype=np.int32)
        current_field = field[level_index]
        previous_field = field[level_index - 1]
        previous_labels = labels[level_index - 1]

        diff_stack: list[np.ndarray] = []
        label_stack: list[np.ndarray] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted_prev_field = shift_lat_lon(previous_field, dy, dx, np.nan)
                shifted_prev_labels = shift_lat_lon(previous_labels, dy, dx, 0)
                candidate_diff = np.abs(current_field - shifted_prev_field)
                candidate_diff = np.where(
                    level_nonwall & (shifted_prev_labels > 0),
                    candidate_diff,
                    np.inf,
                )
                diff_stack.append(candidate_diff)
                label_stack.append(shifted_prev_labels)

        diffs = np.stack(diff_stack, axis=0)
        label_choices = np.stack(label_stack, axis=0)
        best_index = np.argmin(diffs, axis=0)
        best_diff = np.take_along_axis(diffs, best_index[None, ...], axis=0)[0]
        best_label = np.take_along_axis(label_choices, best_index[None, ...], axis=0)[0]
        can_inherit = level_nonwall & np.isfinite(best_diff) & (best_diff <= match_threshold) & (best_label > 0)
        current_labels[can_inherit] = best_label[can_inherit]

        unassigned = level_nonwall & (current_labels == 0)
        if np.any(unassigned):
            new_component_labels, component_count = label_wrapped_planar_components(unassigned)
            for component_id in range(1, component_count + 1):
                component = new_component_labels == component_id
                if int(np.count_nonzero(component)) < MIN_SURFACE_REGION_CELLS:
                    continue
                current_labels[component] = next_label
                birth_levels.append(level_index)
                next_label += 1

        labels[level_index] = current_labels

    return labels, np.asarray(birth_levels, dtype=np.int32), match_threshold


def safe_quantile_limits(field: np.ndarray, lower: float = 0.02, upper: float = 0.98) -> tuple[float, float]:
    finite = np.asarray(field[np.isfinite(field)], dtype=np.float32)
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.quantile(finite, lower))
    high = float(np.quantile(finite, upper))
    if math.isclose(low, high):
        high = low + 1.0
    return low, high


def level_index_for_pressure(pressure_levels_hpa: np.ndarray, target_hpa: float) -> int:
    return int(np.argmin(np.abs(pressure_levels_hpa - target_hpa)))


def compute_component_centroid(latitudes_deg: np.ndarray, longitudes_deg: np.ndarray, cell_mask: np.ndarray) -> tuple[float, float]:
    y_indices, x_indices = np.nonzero(cell_mask)
    if y_indices.size == 0:
        return float("nan"), float("nan")
    lat = latitudes_deg[y_indices]
    lon = longitudes_deg[x_indices]
    lon_rad = np.deg2rad(lon)
    mean_sin = float(np.mean(np.sin(lon_rad)))
    mean_cos = float(np.mean(np.cos(lon_rad)))
    mean_lon = math.degrees(math.atan2(mean_sin, mean_cos))
    return float(np.mean(lat)), float(mean_lon)


def great_circle_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    if not np.isfinite([lat0, lon0, lat1, lon1]).all():
        return float("nan")
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    dlat = lat1_rad - lat0_rad
    dlon = lon1_rad - lon0_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat0_rad) * math.cos(lat1_rad) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(max(a, 0.0))))


def summarize_components(
    labels: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    *,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    component_ids = np.unique(labels)
    component_ids = component_ids[component_ids > 0]
    for component_id in component_ids:
        component = labels == component_id
        levels = np.flatnonzero(np.any(component, axis=(1, 2)))
        voxel_count = int(np.count_nonzero(component))
        level_centroids: list[dict[str, float]] = []
        step_distances_km: list[float] = []
        previous_centroid: tuple[float, float] | None = None
        for level_index in levels:
            centroid = compute_component_centroid(latitudes_deg, longitudes_deg, component[level_index])
            level_centroids.append(
                {
                    "pressure_hpa": float(pressure_levels_hpa[level_index]),
                    "latitude_deg": centroid[0],
                    "longitude_deg": centroid[1],
                }
            )
            if previous_centroid is not None:
                step_distances_km.append(
                    great_circle_km(previous_centroid[0], previous_centroid[1], centroid[0], centroid[1])
                )
            previous_centroid = centroid
        summaries.append(
            {
                "component_id": int(component_id),
                "voxel_count": voxel_count,
                "touches_surface": bool(levels.size > 0 and levels[0] == 0),
                "pressure_span_levels": int(levels.size),
                "pressure_top_hpa": float(np.min(pressure_levels_hpa[levels])),
                "pressure_bottom_hpa": float(np.max(pressure_levels_hpa[levels])),
                "mean_shift_km_per_level": float(np.nanmean(step_distances_km)) if step_distances_km else 0.0,
                "level_centroids": level_centroids,
            }
        )
    summaries.sort(key=lambda item: item["voxel_count"], reverse=True)
    return summaries[:top_n]


def extract_field(field_mode: str, theta: np.ndarray, climatology_mean: np.ndarray, climatology_std: np.ndarray) -> np.ndarray:
    if field_mode == "level_mean":
        return np.asarray(theta - np.mean(theta, axis=(1, 2), keepdims=True), dtype=np.float32)
    if field_mode == "latitude_band":
        return np.asarray(theta - np.mean(theta, axis=2, keepdims=True), dtype=np.float32)
    if field_mode == "climatology":
        return np.asarray(theta - climatology_mean, dtype=np.float32)
    if field_mode == "climatology_z":
        safe_std = np.maximum(climatology_std, SURFACE_GROWTH_STD_FLOOR)
        return np.asarray((theta - climatology_mean) / safe_std, dtype=np.float32)
    raise ValueError(f"Unsupported field mode: {field_mode}")


def analyze_wall_first(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> dict[str, Any]:
    smoothed = smooth_levels(field)
    wall_score = compute_horizontal_gradient(smoothed, latitudes_deg, longitudes_deg)
    raw_wall_mask = keep_top_percent_per_level(wall_score, KEEP_TOP_PERCENT)
    wall_mask = clean_wall_mask(raw_wall_mask)

    wall_labels, wall_component_count = label_wrapped_volume_components(wall_mask)
    wall_components = summarize_components(wall_labels, pressure_levels_hpa, latitudes_deg, longitudes_deg)
    coherent_wall_voxels = 0
    if wall_component_count > 0:
        for label_id in range(1, wall_component_count + 1):
            component = wall_labels == label_id
            levels = np.flatnonzero(np.any(component, axis=(1, 2)))
            if levels.size >= MIN_WALL_COMPONENT_SPAN_LEVELS:
                coherent_wall_voxels += int(np.count_nonzero(component))

    air_mask = ~wall_mask
    air_labels, air_component_count = label_wrapped_volume_components(air_mask)
    surface_ids = np.unique(air_labels[0])
    surface_ids = surface_ids[surface_ids > 0]
    surface_mask = np.isin(air_labels, surface_ids)
    deep_500_mask = np.zeros_like(surface_mask, dtype=bool)
    deep_300_mask = np.zeros_like(surface_mask, dtype=bool)
    for surface_id in surface_ids:
        component = air_labels == surface_id
        levels = np.flatnonzero(np.any(component, axis=(1, 2)))
        top_pressure = float(np.min(pressure_levels_hpa[levels]))
        if top_pressure <= 500.0:
            deep_500_mask |= component
        if top_pressure <= 300.0:
            deep_300_mask |= component

    surface_components = summarize_components(air_labels * surface_mask.astype(np.int32), pressure_levels_hpa, latitudes_deg, longitudes_deg)
    largest_surface_component_fraction = (
        float(surface_components[0]["voxel_count"] / surface_mask.size) if surface_components else 0.0
    )
    return {
        "field": field,
        "smoothed_field": smoothed,
        "wall_score": wall_score,
        "wall_mask": wall_mask,
        "surface_region_mask": surface_mask,
        "surface_region_labels": air_labels * surface_mask.astype(np.int32),
        "surface_origin_birth_levels": None,
        "match_threshold": None,
        "metrics": {
            "wall_fraction": float(np.count_nonzero(wall_mask) / wall_mask.size),
            "wall_component_count": int(wall_component_count),
            "wall_coherent_fraction": float(coherent_wall_voxels / max(np.count_nonzero(wall_mask), 1)),
            "surface_region_fraction": float(np.count_nonzero(surface_mask) / surface_mask.size),
            "surface_region_to_500_fraction": float(np.count_nonzero(deep_500_mask) / surface_mask.size),
            "surface_region_to_300_fraction": float(np.count_nonzero(deep_300_mask) / surface_mask.size),
            "surface_component_count": int(surface_ids.size),
            "largest_surface_component_fraction": largest_surface_component_fraction,
        },
        "top_wall_components": wall_components,
        "top_surface_components": surface_components,
    }


def analyze_surface_growth(
    field: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> dict[str, Any]:
    smoothed = smooth_levels(field)
    wall_score = compute_horizontal_gradient(smoothed, latitudes_deg, longitudes_deg)
    raw_wall_mask = keep_top_percent_per_level(wall_score, KEEP_TOP_PERCENT)
    wall_mask = clean_wall_mask(raw_wall_mask)

    region_labels, birth_levels, match_threshold = build_surface_up_labels(smoothed, wall_mask)
    surface_origin = np.zeros_like(region_labels, dtype=bool)
    component_ids = np.unique(region_labels)
    component_ids = component_ids[component_ids > 0]
    surface_component_count = 0
    for component_id in component_ids:
        if birth_levels[int(component_id)] == 0:
            surface_origin |= region_labels == component_id
            surface_component_count += 1

    labels_for_surface = np.where(surface_origin, region_labels, 0)
    top_surface_components = summarize_components(labels_for_surface, pressure_levels_hpa, latitudes_deg, longitudes_deg)
    largest_surface_component_fraction = (
        float(top_surface_components[0]["voxel_count"] / surface_origin.size) if top_surface_components else 0.0
    )

    deep_500_mask = np.zeros_like(surface_origin, dtype=bool)
    deep_300_mask = np.zeros_like(surface_origin, dtype=bool)
    for component in top_surface_components:
        component_id = int(component["component_id"])
        component_mask = region_labels == component_id
        if float(component["pressure_top_hpa"]) <= 500.0:
            deep_500_mask |= component_mask
        if float(component["pressure_top_hpa"]) <= 300.0:
            deep_300_mask |= component_mask

    final_wall_mask = wall_mask
    wall_labels, wall_component_count = label_wrapped_volume_components(final_wall_mask)
    top_wall_components = summarize_components(wall_labels, pressure_levels_hpa, latitudes_deg, longitudes_deg)
    coherent_wall_voxels = 0
    if wall_component_count > 0:
        for label_id in range(1, wall_component_count + 1):
            component = wall_labels == label_id
            levels = np.flatnonzero(np.any(component, axis=(1, 2)))
            if levels.size >= MIN_WALL_COMPONENT_SPAN_LEVELS:
                coherent_wall_voxels += int(np.count_nonzero(component))

    return {
        "field": field,
        "smoothed_field": smoothed,
        "wall_score": wall_score,
        "wall_mask": final_wall_mask,
        "surface_region_mask": surface_origin,
        "surface_region_labels": labels_for_surface,
        "surface_origin_birth_levels": birth_levels.tolist(),
        "match_threshold": float(match_threshold),
        "metrics": {
            "wall_fraction": float(np.count_nonzero(final_wall_mask) / final_wall_mask.size),
            "wall_component_count": int(wall_component_count),
            "wall_coherent_fraction": float(coherent_wall_voxels / max(np.count_nonzero(final_wall_mask), 1)),
            "surface_region_fraction": float(np.count_nonzero(surface_origin) / surface_origin.size),
            "surface_region_to_500_fraction": float(np.count_nonzero(deep_500_mask) / surface_origin.size),
            "surface_region_to_300_fraction": float(np.count_nonzero(deep_300_mask) / surface_origin.size),
            "surface_component_count": int(surface_component_count),
            "largest_surface_component_fraction": largest_surface_component_fraction,
        },
        "top_wall_components": top_wall_components,
        "top_surface_components": top_surface_components,
    }


def plot_level_map(ax: plt.Axes, field: np.ndarray, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    ax.imshow(
        field,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[float(longitudes_deg.min()), float(longitudes_deg.max()), float(latitudes_deg.min()), float(latitudes_deg.max())],
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_experiment_overview(
    output_path: Path,
    config: ExperimentConfig,
    analysis: dict[str, Any],
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    field = np.asarray(analysis["field"], dtype=np.float32)
    wall_score = np.asarray(analysis["wall_score"], dtype=np.float32)
    wall_mask = np.asarray(analysis["wall_mask"], dtype=bool)
    surface_mask = np.asarray(analysis["surface_region_mask"], dtype=bool)

    field_vmin, field_vmax = safe_quantile_limits(field)
    score_vmin, score_vmax = safe_quantile_limits(wall_score, 0.05, 0.98)

    for row_index, target_hpa in enumerate(SELECTED_PLOT_LEVELS_HPA):
        level_index = level_index_for_pressure(pressure_levels_hpa, target_hpa)
        plot_level_map(
            axes[row_index, 0],
            field[level_index],
            latitudes_deg,
            longitudes_deg,
            f"{int(round(pressure_levels_hpa[level_index]))} hPa normalized field",
            "coolwarm",
            field_vmin,
            field_vmax,
        )
        plot_level_map(
            axes[row_index, 1],
            wall_score[level_index],
            latitudes_deg,
            longitudes_deg,
            f"{int(round(pressure_levels_hpa[level_index]))} hPa wall score",
            "magma",
            score_vmin,
            score_vmax,
        )
        plot_level_map(
            axes[row_index, 2],
            wall_mask[level_index].astype(np.float32),
            latitudes_deg,
            longitudes_deg,
            f"{int(round(pressure_levels_hpa[level_index]))} hPa wall mask",
            "Greys",
            0.0,
            1.0,
        )
        plot_level_map(
            axes[row_index, 3],
            surface_mask[level_index].astype(np.float32),
            latitudes_deg,
            longitudes_deg,
            f"{int(round(pressure_levels_hpa[level_index]))} hPa surface-attached side",
            "viridis",
            0.0,
            1.0,
        )

    fig.suptitle(config.title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_comparison(output_path: Path, summaries: list[dict[str, Any]]) -> None:
    labels = [summary["short_title"] for summary in summaries]
    wall_fraction = [summary["metrics"]["wall_fraction"] * 100.0 for summary in summaries]
    wall_coherent = [summary["metrics"]["wall_coherent_fraction"] * 100.0 for summary in summaries]
    surface_fraction = [summary["metrics"]["surface_region_fraction"] * 100.0 for summary in summaries]
    largest_surface_component = [
        summary["metrics"]["largest_surface_component_fraction"] * 100.0 for summary in summaries
    ]
    surface_500 = [summary["metrics"]["surface_region_to_500_fraction"] * 100.0 for summary in summaries]
    surface_300 = [summary["metrics"]["surface_region_to_300_fraction"] * 100.0 for summary in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width / 2.0, wall_fraction, width=width, color="#8c510a", label="Wall fraction")
    axes[0].bar(x + width / 2.0, wall_coherent, width=width, color="#bf812d", label="Coherent wall fraction")
    axes[0].set_title("Wall occupancy")
    axes[0].set_ylabel("Volume share (%)")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2.0, surface_fraction, width=width, color="#1b7837", label="All surface-attached")
    axes[1].bar(
        x + width / 2.0,
        largest_surface_component,
        width=width,
        color="#5aae61",
        label="Largest surface component",
    )
    axes[1].set_title("Surface partition quality")
    axes[1].set_ylabel("Volume share (%)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend()

    axes[2].plot(x, surface_500, marker="o", linewidth=2.0, color="#2166ac", label="Reaches 500 hPa")
    axes[2].plot(x, surface_300, marker="o", linewidth=2.0, color="#762a83", label="Reaches 300 hPa")
    axes[2].set_title("Deep surface-attached continuity")
    axes[2].set_ylabel("Volume share (%)")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_component_tracks(
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    components = summary["top_surface_components"][:5]
    if not components:
        return

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    for color, component in zip(colors, components):
        centroids = component["level_centroids"]
        lons = [entry["longitude_deg"] for entry in centroids]
        lats = [entry["latitude_deg"] for entry in centroids]
        labels = [int(round(entry["pressure_hpa"])) for entry in centroids]
        ax.plot(lons, lats, marker="o", linewidth=2.0, color=color, label=f"ID {component['component_id']}")
        for lon, lat, pressure in zip(lons, lats, labels):
            ax.text(lon, lat, str(pressure), fontsize=7, color=color)

    ax.set_title(f"Top surface-attached component centroid tracks: {summary['short_title']}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def experiment_analysis_text(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    best_surface = summary["top_surface_components"][0] if summary["top_surface_components"] else None
    best_wall = summary["top_wall_components"][0] if summary["top_wall_components"] else None
    lines = [
        f"Wall fraction: `{100.0 * metrics['wall_fraction']:.1f}%`.",
        f"Coherent-wall share of the retained wall mask: `{100.0 * metrics['wall_coherent_fraction']:.1f}%`.",
        f"Surface-attached air-mass fraction: `{100.0 * metrics['surface_region_fraction']:.1f}%`.",
        f"Largest surface-attached component share: `{100.0 * metrics['largest_surface_component_fraction']:.1f}%`.",
        f"Surface-attached fraction reaching 500 hPa: `{100.0 * metrics['surface_region_to_500_fraction']:.1f}%`.",
        f"Surface-attached fraction reaching 300 hPa: `{100.0 * metrics['surface_region_to_300_fraction']:.1f}%`.",
    ]
    if summary.get("match_threshold") is not None:
        lines.append(f"Surface-growth match threshold: `{summary['match_threshold']:.2f}` in normalized-field units.")
    if best_surface is not None:
        lines.append(
            "Largest surface-attached region: "
            f"`{best_surface['voxel_count']}` cells, top reaches `{best_surface['pressure_top_hpa']:.0f} hPa`, "
            f"mean centroid shift `{best_surface['mean_shift_km_per_level']:.0f} km` per level."
        )
    if best_wall is not None:
        lines.append(
            "Largest wall component: "
            f"`{best_wall['voxel_count']}` cells spanning down to `{best_wall['pressure_top_hpa']:.0f} hPa`."
        )

    analysis_parts: list[str] = []
    if metrics["surface_region_to_500_fraction"] >= 0.35:
        analysis_parts.append("This method does recover a substantial amount of vertically persistent, surface-attached structure into the mid-troposphere.")
    else:
        analysis_parts.append("This method does not keep much of the volume coherently attached to the surface once the wall mask is applied.")
    if metrics["largest_surface_component_fraction"] >= 0.75:
        analysis_parts.append("Most of that continuity sits inside one dominant globe-spanning component, so this is a weak region partition even if the wall mask itself is coherent.")
    else:
        analysis_parts.append("The surface-attached volume is not collapsing into a single dominant global component, which is better evidence of actual air-mass separation.")
    if metrics["wall_coherent_fraction"] >= 0.60:
        analysis_parts.append("The retained walls are mostly part of multi-level structures rather than isolated level noise.")
    else:
        analysis_parts.append("A large part of the wall mask is still fragmented into short-lived structures.")
    return "\n".join([f"- {line}" for line in lines + analysis_parts])


def build_report(
    output_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    pressure_levels_hpa: np.ndarray,
    latitude_stride: int,
    longitude_stride: int,
) -> None:
    best_wall_summary = max(
        [summary for summary in summaries if summary["key"] != "exp5_surface_up_growth"],
        key=lambda item: (
            item["metrics"]["wall_coherent_fraction"]
            - abs(item["metrics"]["wall_fraction"] - 0.10)
            - 0.5 * item["metrics"]["largest_surface_component_fraction"]
        ),
    )
    best_region_summary = next(
        summary for summary in summaries if summary["key"] == "exp5_surface_up_growth"
    )

    lines = [
        "# Potential Temperature Wall Experiments",
        "",
        "## Setup",
        "",
        "- Field: dry potential temperature derived from `data/era5_temperature_2021-11_08-12.nc`.",
        "- Pressure window: `1000-250 hPa`.",
        f"- Working grid: every `{latitude_stride}`th latitude and `{longitude_stride}`th longitude sample.",
        f"- Retained pressure levels: `{', '.join(str(int(round(value))) for value in pressure_levels_hpa)}`.",
        "- Common wall extractor for experiments 1-4: smooth the normalized field, compute horizontal gradient magnitude, keep the top 10% of score on each level, apply light morphology, and retain only coherent multi-level wall components.",
        "- Experiment 5 keeps the same climatology-anomaly wall basis but turns the problem into a surface-up label-growth test with lateral tilt allowance.",
        "",
        "## Overall Answer",
        "",
        (
            "Similar air masses do appear to stay coherent upward from the surface through a substantial part of the troposphere, "
            "but not as vertical columns. The better methods keep that coherence only when the background dry-theta structure is removed "
            "with either a matched climatology or a surface-up continuation rule."
        ),
        "",
        (
            f"For pure wall detection, the cleanest base method here was **{best_wall_summary['title']}**. "
            "For the stronger test of whether those walls actually separate tilted coherent air masses, "
            f"the most informative method was **{best_region_summary['title']}**."
        ),
        "",
        "## Artifacts",
        "",
        f"- Comparison figure: `{repo_relative_text(output_dir / 'comparison-metrics.png')}`",
        f"- Region-growth centroid tracks: `{repo_relative_text(output_dir / 'best-method-centroid-tracks.png')}`",
    ]

    for summary in summaries:
        lines.append(f"- {summary['title']} overview: `{summary['overview_path']}`")
    lines.extend(["", "## Experiment Log", ""])

    for summary in summaries:
        lines.extend(
            [
                f"### {summary['title']}",
                "",
                "**Hypothesis**",
                "",
                summary["hypothesis"],
                "",
                "**Implementation**",
                "",
                summary["implementation"],
                "",
                "**Result**",
                "",
                experiment_analysis_text(summary),
                "",
                "**Analysis**",
                "",
                summary["analysis_text"],
                "",
            ]
        )

    lines.extend(
        [
            "## Practical Read",
            "",
            "- Experiments 1-4 do find coherent dry-theta walls, but the non-wall side remains dominated by one huge connected component. That means wall detection alone is not enough to define useful air-mass regions.",
            "- The climatology-relative wall methods are still the best starting point for wall finding because they suppress the background globe-scale theta structure without erasing the real synoptic gradients.",
            "- The surface-up growth experiment is the best answer to the user’s actual air-mass question: it keeps only about a third of the full sampled volume attached to surface-born regions, which is much less trivial than the nearly global complements produced by the wall-only methods.",
            "",
            "## Follow-up",
            "",
            "- Re-run the winning method at finer horizontal resolution to check how much frontal sharpness was lost by the coarse comparison grid.",
            "- Add moisture or equivalent potential temperature only after preserving the dry-theta wall logic, so the thermal separation test stays interpretable.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_relative(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_repo_relative(args.dataset)
    climatology_path = resolve_repo_relative(args.climatology)

    with xr.open_dataset(dataset_path) as dataset:
        longitudes_deg, longitude_order = normalize_longitudes(np.asarray(dataset.coords["longitude"].values, dtype=np.float32))
        latitudes_deg = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        pressure_levels_hpa = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        full_pressure_levels_hpa = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        temperature_k = np.asarray(dataset["t"].isel(valid_time=0).values, dtype=np.float32)

    with xr.open_dataset(climatology_path) as climatology_dataset:
        climatology_mean = reorder_longitude_axis(
            np.asarray(climatology_dataset["theta_climatology_mean"].values, dtype=np.float32),
            longitude_order,
        )
        climatology_std = reorder_longitude_axis(
            np.asarray(climatology_dataset["theta_climatology_std"].values, dtype=np.float32),
            longitude_order,
        )

    temperature_k = reorder_longitude_axis(temperature_k, longitude_order)
    temperature_k, pressure_levels_hpa = select_pressure_window(
        temperature_k,
        pressure_levels_hpa,
        args.pressure_min_hpa,
        args.pressure_max_hpa,
    )
    climatology_mean, _ = select_pressure_window(
        climatology_mean,
        full_pressure_levels_hpa,
        args.pressure_min_hpa,
        args.pressure_max_hpa,
    )
    climatology_std, _ = select_pressure_window(
        climatology_std,
        full_pressure_levels_hpa,
        args.pressure_min_hpa,
        args.pressure_max_hpa,
    )

    latitude_stride = max(int(args.latitude_stride), 1)
    longitude_stride = max(int(args.longitude_stride), 1)
    temperature_k = np.asarray(temperature_k[:, ::latitude_stride, ::longitude_stride], dtype=np.float32)
    climatology_mean = np.asarray(climatology_mean[:, ::latitude_stride, ::longitude_stride], dtype=np.float32)
    climatology_std = np.asarray(climatology_std[:, ::latitude_stride, ::longitude_stride], dtype=np.float32)
    latitudes_deg = np.asarray(latitudes_deg[::latitude_stride], dtype=np.float32)
    longitudes_deg = np.asarray(longitudes_deg[::longitude_stride], dtype=np.float32)

    theta = compute_theta(temperature_k, pressure_levels_hpa)

    summaries: list[dict[str, Any]] = []
    for config in EXPERIMENTS:
        field = extract_field(config.field_mode, theta, climatology_mean, climatology_std)
        if config.method == "surface_growth":
            analysis = analyze_surface_growth(field, pressure_levels_hpa, latitudes_deg, longitudes_deg)
        else:
            analysis = analyze_wall_first(field, pressure_levels_hpa, latitudes_deg, longitudes_deg)

        overview_path = output_dir / f"{config.key}-overview.png"
        plot_experiment_overview(
            overview_path,
            config,
            analysis,
            pressure_levels_hpa,
            latitudes_deg,
            longitudes_deg,
        )

        summary_payload = {
            "key": config.key,
            "title": config.title,
            "short_title": config.short_title,
            "hypothesis": config.hypothesis,
            "implementation": config.implementation,
            "metrics": analysis["metrics"],
            "match_threshold": analysis["match_threshold"],
            "top_wall_components": analysis["top_wall_components"],
            "top_surface_components": analysis["top_surface_components"],
            "overview_path": repo_relative_text(overview_path),
        }
        if config.method == "surface_growth":
            summary_payload["analysis_text"] = (
                "This is the strongest direct test of the user’s idea. By seeding at 1000 hPa and allowing a one-level, "
                "3x3 horizontal search when growing upward, the method explicitly asks whether the same air mass can remain "
                "continuous while tilting. Its performance is the cleanest answer to part (a)."
            )
        elif config.field_mode == "climatology":
            summary_payload["analysis_text"] = (
                "This is the cleanest wall-first dry-theta method in the set because it subtracts the matched background state "
                "instead of only removing a simple level mean or latitude-band mean. If this outperforms the cruder methods, "
                "that is evidence that air-mass walls are better thought of as departures from the expected local theta structure."
            )
        elif config.field_mode == "climatology_z":
            summary_payload["analysis_text"] = (
                "This test checks whether relative rarity matters more than raw magnitude. It is useful if large-variance regions "
                "otherwise dominate, but it can also over-amplify narrow or noisy anomalies."
            )
        elif config.field_mode == "latitude_band":
            summary_payload["analysis_text"] = (
                "This removes the zonal mean background and therefore tests whether air-mass walls are mostly longitudinal departures "
                "within latitude belts rather than departures from a fixed climatological state."
            )
        else:
            summary_payload["analysis_text"] = (
                "This is the simplest normalization in the set. It mainly checks whether merely removing the vertical offset is enough, "
                "or whether that still leaves too much large-scale background for meaningful air-mass wall detection."
            )

        (output_dir / f"{config.key}-summary.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary_payload)

    comparison_path = output_dir / "comparison-metrics.png"
    plot_summary_comparison(comparison_path, summaries)

    best_wall_summary = max(
        [summary for summary in summaries if summary["key"] != "exp5_surface_up_growth"],
        key=lambda item: (
            item["metrics"]["wall_coherent_fraction"]
            - abs(item["metrics"]["wall_fraction"] - 0.10)
            - 0.5 * item["metrics"]["largest_surface_component_fraction"]
        ),
    )
    region_summary = next(summary for summary in summaries if summary["key"] == "exp5_surface_up_growth")
    plot_component_tracks(output_dir / "best-method-centroid-tracks.png", region_summary)

    build_report(
        output_dir / "report.md",
        output_dir,
        summaries,
        pressure_levels_hpa,
        latitude_stride,
        longitude_stride,
    )

    manifest = {
        "output_dir": repo_relative_text(output_dir),
        "dataset": repo_relative_text(dataset_path),
        "climatology": repo_relative_text(climatology_path),
        "latitude_stride": latitude_stride,
        "longitude_stride": longitude_stride,
        "pressure_levels_hpa": [float(value) for value in pressure_levels_hpa],
        "comparison_figure": repo_relative_text(comparison_path),
        "report": repo_relative_text(output_dir / "report.md"),
        "experiments": [summary["key"] for summary in summaries],
        "best_wall_method": best_wall_summary["key"],
        "best_region_method": region_summary["key"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
