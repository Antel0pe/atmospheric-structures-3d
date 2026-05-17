from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_ROOT / "fontconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, median_filter, uniform_filter


TEMPERATURE_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
CLIMATOLOGY_PATH = Path("data/era5_temperature-climatology_1990-2020_11-08_12.nc")
WIND_PATH = Path("data/era5_source_air_wind_uv_omega_2021-11-05t12_to_2021-11-08t12_1000-250hpa.nc")
HUMIDITY_PATH = Path("data/era5_specific-humidity_2021-11_08-12.nc")
OUTPUT_DIR = Path("tmp/thermal-conflict-method-search")
TIMESTAMP = np.datetime64("2021-11-08T12:00:00")
LEVEL_MIN = 250.0
LEVEL_MAX = 1000.0
KEY_LEVELS = [1000.0, 850.0, 700.0, 500.0, 300.0, 250.0]
EARTH_KM_PER_DEG = 111.2
RADIUS_KM = 6371.0
HEMISPHERES = ("north", "south")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    hypothesis: str
    steps: list[str]
    color: str


METHODS = [
    MethodSpec(
        "vertical_identity_shear",
        "Vertical thermal-identity shear",
        "A sloping frontal/baroclinic zone should show large thermal-identity change between neighboring pressure levels as well as horizontal identity contrast at the level being plotted.",
        [
            "Match raw temperature to same-longitude climatology and compute thermal-displacement score at every 250-1000 hPa level.",
            "For each level, compare that score with the nearest pressure level above and below.",
            "Score cells where vertical thermal-identity shear and horizontal thermal-identity gradient are both high.",
            "Pick the strongest midlatitude ridge by longitude in each hemisphere.",
        ],
        "#9333ea",
    ),
    MethodSpec(
        "thermal_advection_dipole",
        "Warm/cold advection dipole",
        "Hot/cold fighting should be active where winds are advecting tropical-like identity into nearby polar-like identity, or vice versa.",
        [
            "Compute thermal-displacement score and its horizontal gradient.",
            "Read ERA5 wind at the same pressure level and time.",
            "Compute thermal-identity advection, then smooth nearby warm-advection and cold-advection lobes.",
            "Score cells where opposite-signed advection lobes touch near a thermal-identity gradient.",
        ],
        "#ea580c",
    ),
    MethodSpec(
        "thetae_displacement_contact",
        "Theta-e gradient with displacement contrast",
        "Air-mass fights often sharpen equivalent potential temperature, so a theta-e gradient becomes more relevant when it also separates tropical-like and polar-like thermal displacement.",
        [
            "Compute equivalent potential temperature from temperature and specific humidity.",
            "Smooth theta-e about 500 km and compute its horizontal gradient.",
            "Sample thermal-displacement score across that theta-e gradient direction.",
            "Score cells where theta-e changes sharply and the warm/moist side is more tropical-like than the cold/dry side.",
        ],
        "#16a34a",
    ),
    MethodSpec(
        "thermal_wind_baroclinic",
        "Thermal-wind baroclinicity",
        "A real baroclinic boundary should align with vertical wind shear because horizontal temperature contrasts imply thermal-wind shear.",
        [
            "Read wind at every 250-1000 hPa level.",
            "Compute vertical wind-shear magnitude against neighboring pressure levels.",
            "Multiply shear by the local thermal-displacement horizontal gradient.",
            "Pick the strongest shear-supported thermal-identity ridge by longitude.",
        ],
        "#0284c7",
    ),
    MethodSpec(
        "isotherm_normal_displacement_jump",
        "Raw-isotherm-normal displacement jump",
        "Use real smoothed-temperature isotherms for geometry, then ask whether crossing those isotherms also crosses a polar-like to tropical-like displacement jump.",
        [
            "Smooth raw temperature about 500 km and compute its gradient direction.",
            "Sample thermal-displacement score on the warm and cold sides normal to the smoothed isotherms.",
            "Score cells where real isotherms are packed and thermal-displacement score jumps across the isotherm normal.",
            "Pick the strongest real-isotherm/displacement-jump ridge by longitude.",
        ],
        "#dc2626",
    ),
]


def normalize(values: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return np.asarray(np.clip(values / scale, 0.0, 1.0), dtype=np.float32)


def smooth_grid(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    return gaussian_filter(
        np.asarray(values, dtype=np.float32),
        sigma=(sigma_cells, sigma_cells),
        mode=("nearest", "wrap"),
    ).astype(np.float32)


def physical_smooth(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, sigma_km: float) -> np.ndarray:
    if sigma_km <= 0.0:
        return np.asarray(values, dtype=np.float32)
    dlat = float(abs(np.nanmedian(np.diff(latitudes))))
    dlon = float(abs(np.nanmedian(np.diff(longitudes))))
    sigma_lat = sigma_km / (EARTH_KM_PER_DEG * dlat)
    out = gaussian_filter1d(
        np.asarray(values, dtype=np.float32),
        sigma=sigma_lat,
        axis=0,
        mode="nearest",
    )
    smoothed = np.empty_like(out, dtype=np.float32)
    for row, lat in enumerate(latitudes):
        cos_lat = max(float(np.cos(np.deg2rad(lat))), 0.12)
        sigma_lon = sigma_km / (EARTH_KM_PER_DEG * cos_lat * dlon)
        smoothed[row, :] = gaussian_filter1d(out[row, :], sigma=sigma_lon, mode="wrap")
    return smoothed


def match_equivalent_latitude_same_longitude(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    n_lat, n_lon = raw.shape
    matched = np.empty((n_lat, n_lon), dtype=np.float32)
    for lon_index in range(n_lon):
        profile = climatology[:, lon_index]
        order = np.argsort(profile, kind="mergesort")
        sorted_values = profile[order]
        sorted_latitudes = latitudes[order]
        source_values = raw[:, lon_index]
        source_rows = np.arange(n_lat)
        insertion = np.searchsorted(sorted_values, source_values, side="left")
        lower = np.clip(insertion - 1, 0, n_lat - 1)
        upper = np.clip(insertion, 0, n_lat - 1)
        lower_distance = np.abs(source_values - sorted_values[lower])
        upper_distance = np.abs(source_values - sorted_values[upper])
        lower_row_distance = np.abs(order[lower] - source_rows)
        upper_row_distance = np.abs(order[upper] - source_rows)
        choose_upper = (upper_distance < lower_distance) | (
            (upper_distance == lower_distance)
            & (upper_row_distance < lower_row_distance)
        )
        nearest = np.where(choose_upper, upper, lower)
        matched[:, lon_index] = sorted_latitudes[nearest]
    return matched


def thermal_displacement_score(raw: np.ndarray, clim: np.ndarray, latitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matched = match_equivalent_latitude_same_longitude(raw, clim, latitudes)
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes))), 1e-6)
    score_raw = (1.0 - np.abs(matched) / max_abs_latitude) * 100.0
    score = smooth_grid(np.clip(score_raw, 0.0, 100.0), 1.0)
    eq_abs = max_abs_latitude * (1.0 - score / 100.0)
    return score.astype(np.float32), eq_abs.astype(np.float32)


def lat_grid_abs(latitudes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.abs(latitudes)[:, None], shape).astype(np.float32)


def signed_displacement(eq_abs: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    return (lat_grid_abs(latitudes, eq_abs.shape) - eq_abs).astype(np.float32)


def midlatitude_weight(latitudes: np.ndarray) -> np.ndarray:
    abs_lat = np.abs(latitudes)
    low = 1.0 / (1.0 + np.exp(-(abs_lat - 22.0) / 4.0))
    high = 1.0 / (1.0 + np.exp((abs_lat - 68.0) / 4.0))
    return (low * high).astype(np.float32)


def offset_rows(latitudes: np.ndarray, offset_km: float) -> int:
    dlat = float(abs(np.nanmedian(np.diff(latitudes))))
    return max(1, int(round((offset_km / EARTH_KM_PER_DEG) / dlat)))


def physical_gradients(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dlat_rad = np.deg2rad(float(abs(np.nanmedian(np.diff(latitudes)))))
    dlon_rad = np.deg2rad(float(abs(np.nanmedian(np.diff(longitudes)))))
    north = np.empty_like(values)
    south = np.empty_like(values)
    north[0] = values[0]
    north[1:] = values[:-1]
    south[-1] = values[-1]
    south[:-1] = values[1:]
    east = np.roll(values, -1, axis=1)
    west = np.roll(values, 1, axis=1)
    dy = 0.5 * (south - north) / (RADIUS_KM * dlat_rad)
    cos_lat = np.maximum(np.cos(np.deg2rad(latitudes)), 0.12)
    dx_scale = (RADIUS_KM * dlon_rad * cos_lat)[:, None]
    dx = 0.5 * (east - west) / dx_scale
    return dx.astype(np.float32), dy.astype(np.float32)


def gradient_magnitude(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    gx, gy = physical_gradients(values, latitudes, longitudes)
    return np.hypot(gx, gy).astype(np.float32)


def hemisphere_mask(latitudes: np.ndarray, hem: str) -> np.ndarray:
    if hem == "north":
        keep = (latitudes >= 20.0) & (latitudes <= 70.0)
    else:
        keep = (latitudes <= -20.0) & (latitudes >= -70.0)
    return keep[:, None]


def split_hemi(score: np.ndarray, latitudes: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for hem in HEMISPHERES:
        field = np.full(score.shape, np.nan, dtype=np.float32)
        keep = hemisphere_mask(latitudes, hem)
        field[keep[:, 0], :] = score[keep[:, 0], :]
        out[hem] = field
    return out


def side_order_score(score: np.ndarray, eq_abs: np.ndarray, latitudes: np.ndarray, offset_km: float = 750.0) -> dict[str, np.ndarray]:
    disp = signed_displacement(eq_abs, latitudes)
    rows = offset_rows(latitudes, offset_km)
    valid = slice(rows, score.shape[0] - rows)
    center_lats = latitudes[valid]
    north_disp = disp[:-2 * rows]
    south_disp = disp[2 * rows :]
    north_score = score[:-2 * rows]
    south_score = score[2 * rows :]
    tropical_nh = np.clip(south_disp / 20.0, 0.0, 1.0)
    polar_nh = np.clip(-north_disp / 20.0, 0.0, 1.0)
    tropical_sh = np.clip(north_disp / 20.0, 0.0, 1.0)
    polar_sh = np.clip(-south_disp / 20.0, 0.0, 1.0)
    contrast_nh = np.clip((south_score - north_score) / 35.0, 0.0, 1.0)
    contrast_sh = np.clip((north_score - south_score) / 35.0, 0.0, 1.0)
    out = {hem: np.full(score.shape, np.nan, dtype=np.float32) for hem in HEMISPHERES}
    out["north"][valid] = np.where(
        (center_lats >= 0.0)[:, None],
        np.sqrt(tropical_nh * polar_nh) * contrast_nh,
        np.nan,
    )
    out["south"][valid] = np.where(
        (center_lats <= 0.0)[:, None],
        np.sqrt(tropical_sh * polar_sh) * contrast_sh,
        np.nan,
    )
    weight = midlatitude_weight(latitudes)[:, None]
    return {hem: physical_smooth(out[hem] * weight, latitudes, np.arange(score.shape[1]), 250.0) for hem in HEMISPHERES}


def signed_zero_gradient_score(eq_abs: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    disp = signed_displacement(eq_abs, latitudes)
    zero = np.exp(-0.5 * (disp / 9.0) ** 2)
    grad = normalize(gradient_magnitude(disp, latitudes, longitudes), 96.0)
    field = physical_smooth((zero * grad * midlatitude_weight(latitudes)[:, None]).astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def local_copresence_score(score: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    tropical = (score >= 70.0).astype(np.float32)
    polar = (score <= 35.0).astype(np.float32)
    tropical_density = physical_smooth(tropical, latitudes, longitudes, 750.0)
    polar_density = physical_smooth(polar, latitudes, longitudes, 750.0)
    both = np.sqrt(np.clip(tropical_density, 0, 1) * np.clip(polar_density, 0, 1))
    local_mean = uniform_filter(score.astype(np.float32), size=(9, 9), mode=("nearest", "wrap"))
    local_sq_mean = uniform_filter((score.astype(np.float32) ** 2), size=(9, 9), mode=("nearest", "wrap"))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0.0))
    field = both * (0.45 + 0.55 * normalize(local_std, 95.0)) * midlatitude_weight(latitudes)[:, None]
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def raw_isotherm_packing_score(
    raw: np.ndarray,
    score: np.ndarray,
    eq_abs: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    raw_smooth = physical_smooth(raw, latitudes, longitudes, 500.0)
    packing = normalize(gradient_magnitude(raw_smooth, latitudes, longitudes), 97.0)
    side = side_order_score(score, eq_abs, latitudes)
    out = {}
    for hem in HEMISPHERES:
        out[hem] = physical_smooth((packing * normalize(np.nan_to_num(side[hem], nan=0.0), 98.0)).astype(np.float32), latitudes, longitudes, 250.0)
        out[hem][~hemisphere_mask(latitudes, hem)[:, 0], :] = np.nan
    return out


def wind_frontogenesis_score(
    score: np.ndarray,
    eq_abs: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    side: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    u = physical_smooth(u, latitudes, longitudes, 250.0)
    v = physical_smooth(v, latitudes, longitudes, 250.0)
    gx, gy = physical_gradients(score, latitudes, longitudes)
    grad_mag = np.maximum(np.hypot(gx, gy), 1e-8)
    dudx, dudy = physical_gradients(u, latitudes, longitudes)
    dvdx, dvdy = physical_gradients(v, latitudes, longitudes)
    stretch = dudx - dvdy
    shear = dvdx + dudy
    divergence = dudx + dvdy
    deformation = ((gx * gx - gy * gy) * stretch + 2.0 * gx * gy * shear) / grad_mag
    fronto = normalize(np.maximum(0.0, 0.5 * deformation - 0.5 * grad_mag * divergence), 98.5)
    out = {}
    for hem in HEMISPHERES:
        contact = normalize(np.nan_to_num(side[hem], nan=0.0), 98.0)
        field = fronto * (0.25 + 0.75 * contact) * midlatitude_weight(latitudes)[:, None]
        field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 150.0)
        field[~hemisphere_mask(latitudes, hem)[:, 0], :] = np.nan
        out[hem] = field
    return out


def sample_with_offsets(field: np.ndarray, row_offsets: np.ndarray, col_offsets: np.ndarray) -> np.ndarray:
    rows = np.arange(field.shape[0], dtype=np.float32)[:, None] + row_offsets
    cols = np.arange(field.shape[1], dtype=np.float32)[None, :] + col_offsets
    rows = np.clip(rows, 0.0, field.shape[0] - 1.0)
    cols = np.mod(cols, field.shape[1])
    return map_coordinates(field, [rows, cols], order=1, mode="nearest").astype(np.float32)


def sample_score_across_gradient(
    score: np.ndarray,
    geometry_field: np.ndarray,
    distance_cells: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_grad, col_grad = np.gradient(geometry_field.astype(np.float32))
    magnitude = np.maximum(np.hypot(row_grad, col_grad), 1e-6)
    row_unit = row_grad / magnitude
    col_unit = col_grad / magnitude
    plus = sample_with_offsets(score, row_unit * distance_cells, col_unit * distance_cells)
    minus = sample_with_offsets(score, -row_unit * distance_cells, -col_unit * distance_cells)
    return plus, minus, magnitude.astype(np.float32)


def theta_e_simple(temperature_k: np.ndarray, specific_humidity: np.ndarray, pressure_hpa: float) -> np.ndarray:
    theta = temperature_k * (1000.0 / pressure_hpa) ** 0.2854
    exponent = (2.5e6 * np.clip(specific_humidity, 0.0, 0.05)) / (1004.0 * np.maximum(temperature_k, 180.0))
    return (theta * np.exp(np.clip(exponent, 0.0, 0.6))).astype(np.float32)


def vertical_identity_shear_score(
    level: float,
    levels: list[float],
    score_by_level: dict[float, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = levels.index(level)
    neighbors: list[np.ndarray] = []
    if idx > 0:
        neighbors.append(score_by_level[levels[idx - 1]])
    if idx < len(levels) - 1:
        neighbors.append(score_by_level[levels[idx + 1]])
    vertical_change = np.mean([np.abs(score_by_level[level] - neighbor) for neighbor in neighbors], axis=0)
    horizontal_change = gradient_magnitude(score_by_level[level], latitudes, longitudes)
    field = (
        normalize(vertical_change, 96.0)
        * normalize(horizontal_change, 96.0)
        * midlatitude_weight(latitudes)[:, None]
    )
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def thermal_advection_dipole_score(
    score: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    gx, gy = physical_gradients(score, latitudes, longitudes)
    advection = -(u * gx + v * gy)
    warm_advection = physical_smooth(np.maximum(advection, 0.0).astype(np.float32), latitudes, longitudes, 500.0)
    cold_advection = physical_smooth(np.maximum(-advection, 0.0).astype(np.float32), latitudes, longitudes, 500.0)
    contact = np.sqrt(normalize(warm_advection, 97.0) * normalize(cold_advection, 97.0))
    field = contact * normalize(np.hypot(gx, gy), 96.0) * midlatitude_weight(latitudes)[:, None]
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 200.0)
    return split_hemi(field, latitudes)


def thetae_displacement_contact_score(
    thetae: np.ndarray,
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    thetae_smooth = physical_smooth(thetae, latitudes, longitudes, 500.0)
    plus_score, minus_score, _ = sample_score_across_gradient(score, thetae_smooth, distance_cells=14.0)
    displacement_jump = np.maximum(plus_score - minus_score, 0.0)
    thetae_gradient = gradient_magnitude(thetae_smooth, latitudes, longitudes)
    field = (
        normalize(thetae_gradient, 97.0)
        * normalize(displacement_jump, 96.0)
        * midlatitude_weight(latitudes)[:, None]
    )
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def thermal_wind_baroclinic_score(
    level: float,
    levels: list[float],
    score_by_level: dict[float, np.ndarray],
    wind_by_level: dict[float, tuple[np.ndarray, np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = levels.index(level)
    u0, v0 = wind_by_level[level]
    shear_terms: list[np.ndarray] = []
    if idx > 0:
        u1, v1 = wind_by_level[levels[idx - 1]]
        shear_terms.append(np.hypot(u0 - u1, v0 - v1))
    if idx < len(levels) - 1:
        u1, v1 = wind_by_level[levels[idx + 1]]
        shear_terms.append(np.hypot(u0 - u1, v0 - v1))
    shear = np.mean(shear_terms, axis=0)
    identity_gradient = gradient_magnitude(score_by_level[level], latitudes, longitudes)
    field = (
        normalize(shear, 97.0)
        * normalize(identity_gradient, 96.0)
        * midlatitude_weight(latitudes)[:, None]
    )
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def isotherm_normal_displacement_jump_score(
    raw: np.ndarray,
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    raw_smooth = physical_smooth(raw, latitudes, longitudes, 500.0)
    warm_side_score, cold_side_score, _ = sample_score_across_gradient(score, raw_smooth, distance_cells=14.0)
    displacement_jump = np.maximum(warm_side_score - cold_side_score, 0.0)
    raw_gradient = gradient_magnitude(raw_smooth, latitudes, longitudes)
    field = (
        normalize(raw_gradient, 97.0)
        * normalize(displacement_jump, 96.0)
        * midlatitude_weight(latitudes)[:, None]
    )
    field = physical_smooth(field.astype(np.float32), latitudes, longitudes, 250.0)
    return split_hemi(field, latitudes)


def extract_ridges(score_by_hem: dict[str, np.ndarray], latitudes: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for hem, field in score_by_hem.items():
        ridge = np.full(field.shape[1], np.nan, dtype=np.float32)
        strength = np.full(field.shape[1], np.nan, dtype=np.float32)
        keep_rows = np.where(np.isfinite(field).any(axis=1))[0]
        if keep_rows.size == 0:
            result[hem] = {"ridge": ridge, "strength": strength}
            continue
        subset = field[keep_rows, :]
        safe = np.where(np.isfinite(subset), subset, -np.inf)
        indices = np.argmax(safe, axis=0)
        ridge[:] = latitudes[keep_rows[indices]]
        strength[:] = safe[indices, np.arange(field.shape[1])]
        no_data = ~np.isfinite(strength) | (strength == -np.inf)
        ridge[no_data] = np.nan
        strength[no_data] = np.nan
        if np.isfinite(ridge).sum() > 10:
            filled = ridge.copy()
            finite = np.isfinite(filled)
            if not np.all(finite):
                x = np.arange(filled.size)
                filled[~finite] = np.interp(x[~finite], x[finite], filled[finite])
            smoothed = median_filter(filled, size=41, mode="wrap")
            smoothed = gaussian_filter1d(smoothed, sigma=6.0, mode="wrap")
            ridge[finite] = smoothed[finite]
        result[hem] = {"ridge": ridge, "strength": strength}
    return result


def level_metrics(ridges_by_level: dict[float, dict[str, dict[str, np.ndarray]]], levels: list[float]) -> dict[str, float]:
    adjacent_offsets: list[float] = []
    key_offsets: list[float] = []
    smoothness: list[float] = []
    for hem in HEMISPHERES:
        for level in levels:
            ridge = ridges_by_level[level][hem]["ridge"]
            if np.isfinite(ridge).sum() > 10:
                second = np.roll(ridge, -1) - 2 * ridge + np.roll(ridge, 1)
                smoothness.append(float(np.nanmedian(np.abs(second))))
        for lower, upper in zip(levels[:-1], levels[1:], strict=False):
            a = ridges_by_level[lower][hem]["ridge"]
            b = ridges_by_level[upper][hem]["ridge"]
            valid = np.isfinite(a) & np.isfinite(b)
            if np.count_nonzero(valid) > 10:
                adjacent_offsets.append(float(np.nanmedian(np.abs(a[valid] - b[valid]))))
        if 1000.0 in ridges_by_level and 250.0 in ridges_by_level:
            a = ridges_by_level[1000.0][hem]["ridge"]
            b = ridges_by_level[250.0][hem]["ridge"]
            valid = np.isfinite(a) & np.isfinite(b)
            if np.count_nonzero(valid) > 10:
                key_offsets.append(float(np.nanmedian(np.abs(a[valid] - b[valid]))))
    return {
        "median_adjacent_level_shift_deg": float(np.median(adjacent_offsets)) if adjacent_offsets else float("nan"),
        "p90_adjacent_level_shift_deg": float(np.percentile(adjacent_offsets, 90)) if adjacent_offsets else float("nan"),
        "median_1000_to_250_shift_deg": float(np.median(key_offsets)) if key_offsets else float("nan"),
        "median_longitude_waviness_deg": float(np.median(smoothness)) if smoothness else float("nan"),
    }


def grade_from_metrics(metrics: dict[str, float], science_score: int) -> str:
    shift = metrics["median_adjacent_level_shift_deg"]
    p90 = metrics["p90_adjacent_level_shift_deg"]
    waviness = metrics["median_longitude_waviness_deg"]
    if not np.isfinite(shift):
        return "D"
    if 2.0 <= shift <= 8.0 and p90 <= 13.0 and waviness <= 1.2 and science_score >= 4:
        return "A"
    if 1.0 <= shift <= 10.0 and p90 <= 17.0 and waviness <= 2.0 and science_score >= 3:
        return "B"
    if shift <= 14.0 and p90 <= 24.0 and science_score >= 2:
        return "C"
    return "D"


def lon_for_plot(longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon180 = ((longitudes + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lon180[order], order


def plot_method_overlay(
    spec: MethodSpec,
    ridges_by_level: dict[float, dict[str, dict[str, np.ndarray]]],
    longitudes: np.ndarray,
    output_path: Path,
    levels: list[float],
) -> None:
    colors = {
        1000.0: "#b91c1c",
        850.0: "#f97316",
        700.0: "#eab308",
        500.0: "#22c55e",
        300.0: "#2563eb",
        250.0: "#7c3aed",
    }
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor("#f8fafc")
    ax.coastlines(resolution="110m", linewidth=0.55, color="#111827")
    ax.gridlines(draw_labels=False, linewidth=0.25, color="#cbd5e1", alpha=0.65)
    lons, order = lon_for_plot(longitudes)
    for level in levels:
        if level not in ridges_by_level:
            continue
        for hem in HEMISPHERES:
            ridge = ridges_by_level[level][hem]["ridge"][order]
            ax.plot(
                lons,
                ridge,
                transform=ccrs.PlateCarree(),
                color=colors[level],
                linewidth=1.9,
                alpha=0.88,
                label=f"{int(level)} hPa" if hem == "north" else None,
                zorder=4,
            )
    ax.set_title(spec.name, fontsize=15, loc="left")
    ax.legend(ncol=6, loc="lower left", fontsize=9, frameon=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_all_levels_overlay(
    spec: MethodSpec,
    ridges_by_level: dict[float, dict[str, dict[str, np.ndarray]]],
    longitudes: np.ndarray,
    output_path: Path,
    levels: list[float],
) -> None:
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor("#f8fafc")
    ax.coastlines(resolution="110m", linewidth=0.5, color="#111827")
    ax.gridlines(draw_labels=False, linewidth=0.22, color="#cbd5e1", alpha=0.6)
    cmap = plt.get_cmap("turbo_r")
    norm = mcolors.Normalize(vmin=LEVEL_MIN, vmax=LEVEL_MAX)
    lons, order = lon_for_plot(longitudes)
    for level in levels:
        color = cmap(norm(level))
        for hem in HEMISPHERES:
            ridge = ridges_by_level[level][hem]["ridge"][order]
            ax.plot(
                lons,
                ridge,
                transform=ccrs.PlateCarree(),
                color=color,
                linewidth=0.85,
                alpha=0.62,
                zorder=4,
            )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.45, pad=0.04)
    cbar.set_label("Pressure level hPa")
    ax.set_title(f"{spec.name} - all 250-1000 hPa levels", fontsize=15, loc="left")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_panels(
    spec: MethodSpec,
    scores_by_level: dict[float, dict[str, np.ndarray]],
    ridges_by_level: dict[float, dict[str, dict[str, np.ndarray]]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    proj = ccrs.PlateCarree()
    lon_edges = np.linspace(float(longitudes.min()) - 0.125, float(longitudes.max()) + 0.125, longitudes.size + 1)
    lat_edges = np.linspace(float(latitudes.max()) + 0.125, float(latitudes.min()) - 0.125, latitudes.size + 1)
    selected = [1000.0, 850.0, 500.0, 250.0]
    for idx, level in enumerate(selected):
        ax = fig.add_subplot(4, 1, idx + 1, projection=proj)
        combined = np.nanmax(np.stack([np.nan_to_num(scores_by_level[level][hem], nan=0.0) for hem in HEMISPHERES]), axis=0)
        ax.pcolormesh(
            lon_edges,
            lat_edges,
            combined,
            transform=proj,
            shading="flat",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        lons, order = lon_for_plot(longitudes)
        for hem in HEMISPHERES:
            ax.plot(lons, ridges_by_level[level][hem]["ridge"][order], color="#00f5ff", linewidth=1.1, transform=proj)
        ax.coastlines(resolution="110m", linewidth=0.45, color="white")
        ax.set_global()
        ax.set_title(f"{int(level)} hPa", fontsize=10, loc="left", color="#111827")
    fig.suptitle(f"{spec.name}: conflict score and picked ridge", fontsize=15)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    images = [plt.imread(path) for path in paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(14, 6.8 * len(images)), constrained_layout=True)
    if len(images) == 1:
        axes = [axes]
    for ax, image, path in zip(axes, images, paths, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(path.stem, loc="left", fontsize=12)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def pressure_levels_between(levels: np.ndarray) -> list[float]:
    selected = [float(level) for level in levels if LEVEL_MIN <= float(level) <= LEVEL_MAX]
    return sorted(selected, reverse=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temperature_ds = xr.open_dataset(TEMPERATURE_PATH)
    climatology_ds = xr.open_dataset(CLIMATOLOGY_PATH)
    wind_ds = xr.open_dataset(WIND_PATH)
    humidity_ds = xr.open_dataset(HUMIDITY_PATH)
    temperature = temperature_ds["t"].sel(valid_time=TIMESTAMP)
    climatology = climatology_ds["temperature_climatology_mean"]
    humidity = humidity_ds["q"].sel(valid_time=TIMESTAMP)
    humidity = humidity.assign_coords(
        longitude=(((humidity.longitude + 180.0) % 360.0) - 180.0)
    ).sortby("longitude")
    latitudes = np.asarray(temperature.latitude.values, dtype=np.float32)
    longitudes = np.asarray(temperature.longitude.values, dtype=np.float32)
    levels = pressure_levels_between(np.asarray(temperature.pressure_level.values))

    all_scores: dict[str, dict[float, dict[str, np.ndarray]]] = {spec.key: {} for spec in METHODS}
    all_ridges: dict[str, dict[float, dict[str, dict[str, np.ndarray]]]] = {spec.key: {} for spec in METHODS}

    raw_by_level: dict[float, np.ndarray] = {}
    score_by_level: dict[float, np.ndarray] = {}
    eq_abs_by_level: dict[float, np.ndarray] = {}
    wind_by_level: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    thetae_by_level: dict[float, np.ndarray] = {}

    for level in levels:
        print(f"Preparing {level:g} hPa")
        raw = np.asarray(temperature.sel(pressure_level=level).values, dtype=np.float32)
        clim = np.asarray(climatology.sel(pressure_level=level).values, dtype=np.float32)
        score, eq_abs = thermal_displacement_score(raw, clim, latitudes)
        u = np.asarray(wind_ds["u"].sel(valid_time=TIMESTAMP, pressure_level=level).values, dtype=np.float32)
        v = np.asarray(wind_ds["v"].sel(valid_time=TIMESTAMP, pressure_level=level).values, dtype=np.float32)
        q = np.asarray(
            humidity.sel(pressure_level=level)
            .sel(latitude=latitudes, longitude=longitudes)
            .values,
            dtype=np.float32,
        )
        raw_by_level[level] = raw
        score_by_level[level] = score
        eq_abs_by_level[level] = eq_abs
        wind_by_level[level] = (u, v)
        thetae_by_level[level] = theta_e_simple(raw, q, level)

    for level in levels:
        print(f"Scoring {level:g} hPa")
        raw = raw_by_level[level]
        score = score_by_level[level]
        u, v = wind_by_level[level]
        by_method = {
            "vertical_identity_shear": vertical_identity_shear_score(
                level, levels, score_by_level, latitudes, longitudes
            ),
            "thermal_advection_dipole": thermal_advection_dipole_score(
                score, u, v, latitudes, longitudes
            ),
            "thetae_displacement_contact": thetae_displacement_contact_score(
                thetae_by_level[level], score, latitudes, longitudes
            ),
            "thermal_wind_baroclinic": thermal_wind_baroclinic_score(
                level, levels, score_by_level, wind_by_level, latitudes, longitudes
            ),
            "isotherm_normal_displacement_jump": isotherm_normal_displacement_jump_score(
                raw, score, latitudes, longitudes
            ),
        }
        for key, scores in by_method.items():
            all_scores[key][level] = scores
            all_ridges[key][level] = extract_ridges(scores, latitudes)

    science_scores = {
        "vertical_identity_shear": 4,
        "thermal_advection_dipole": 4,
        "thetae_displacement_contact": 4,
        "thermal_wind_baroclinic": 5,
        "isotherm_normal_displacement_jump": 4,
    }
    rows: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    all_level_paths: list[Path] = []
    panel_paths: list[Path] = []
    for spec in METHODS:
        metrics = level_metrics(all_ridges[spec.key], levels)
        grade = grade_from_metrics(metrics, science_scores[spec.key])
        row = {
            "method": spec.key,
            "name": spec.name,
            "grade": grade,
            "science_score_1_to_5": science_scores[spec.key],
            **metrics,
        }
        rows.append(row)
        overlay_path = OUTPUT_DIR / f"{spec.key}-key-level-ridge-overlay.png"
        all_levels_path = OUTPUT_DIR / f"{spec.key}-all-level-ridge-overlay.png"
        panel_path = OUTPUT_DIR / f"{spec.key}-score-panels.png"
        plot_method_overlay(spec, all_ridges[spec.key], longitudes, overlay_path, [level for level in KEY_LEVELS if level in levels])
        plot_all_levels_overlay(spec, all_ridges[spec.key], longitudes, all_levels_path, levels)
        plot_score_panels(spec, all_scores[spec.key], all_ridges[spec.key], latitudes, longitudes, panel_path)
        overlay_paths.append(overlay_path)
        all_level_paths.append(all_levels_path)
        panel_paths.append(panel_path)

    with (OUTPUT_DIR / "method_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "timestamp": str(TIMESTAMP),
        "pressure_levels_hpa": levels,
        "method_metrics": rows,
        "methods": [
            {
                "key": spec.key,
                "name": spec.name,
                "hypothesis": spec.hypothesis,
                "steps": spec.steps,
            }
            for spec in METHODS
        ],
        "pruned_ideas": [
            {
                "idea": "signed displacement zero-crossing",
                "reason": "Already tested in the previous pressure-level loop; it is too close to a normal-for-latitude crossing.",
            },
            {
                "idea": "side-ordered tropical/polar contact",
                "reason": "Already the strongest simple prior candidate, so this pass avoids repeating it as a standalone method.",
            },
            {
                "idea": "local tropical/polar co-presence",
                "reason": "Already tested and found useful as corridor width/confidence rather than line placement.",
            },
            {
                "idea": "thermal-character frontogenesis/confluence",
                "reason": "Already tested as process evidence; this pass uses different wind-process ideas instead.",
            },
            {
                "idea": "middle-80 and zonal-mean climatology displacement",
                "reason": "Prior notes show they remove longitude-specific context and introduce hard-to-defend pressure jumps.",
            },
            {
                "idea": "dry-potential-temperature displacement",
                "reason": "At fixed pressure levels it nearly duplicates raw-temperature displacement for this lookup.",
            },
            {
                "idea": "standalone raw temperature gradient",
                "reason": "Meteorologically real, but prior diagnostics show it is too generic and terrain/polar biased unless gated.",
            },
            {
                "idea": "one global rare histogram bucket",
                "reason": "Useful as a contour proposal, but it does not directly measure spatial hot/cold opposition.",
            },
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(overlay_paths, OUTPUT_DIR / "key-level-ridge-overlays-contact-sheet.png")
    make_contact_sheet(all_level_paths, OUTPUT_DIR / "all-level-ridge-overlays-contact-sheet.png")
    make_contact_sheet(panel_paths, OUTPUT_DIR / "score-panels-contact-sheet.png")

    report_lines = [
        "# Thermal Conflict Method Search",
        "",
        "Valid time: `2021-11-08T12:00`.",
        "Pressure levels: `1000-250 hPa`.",
        "",
        "## Ideas Considered And Pruned",
    ]
    for item in summary["pruned_ideas"]:
        report_lines.append(f"- {item['idea']}: {item['reason']}")
    report_lines += ["", "## Method Metrics", ""]
    for row in sorted(rows, key=lambda item: (str(item["grade"]), float(item["median_adjacent_level_shift_deg"]))):
        report_lines.append(
            f"- `{row['method']}` ({row['grade']}): median adjacent shift "
            f"`{float(row['median_adjacent_level_shift_deg']):.1f}°`, p90 adjacent shift "
            f"`{float(row['p90_adjacent_level_shift_deg']):.1f}°`, 1000-250 shift "
            f"`{float(row['median_1000_to_250_shift_deg']):.1f}°`, waviness "
            f"`{float(row['median_longitude_waviness_deg']):.2f}°`."
        )
    report_lines += ["", "## Simple Implementations", ""]
    for spec in METHODS:
        report_lines.append(f"### {spec.name}")
        report_lines.append("")
        report_lines.append(spec.hypothesis)
        report_lines.append("")
        for idx, step in enumerate(spec.steps, start=1):
            report_lines.append(f"{idx}. {step}")
        report_lines.append("")
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
