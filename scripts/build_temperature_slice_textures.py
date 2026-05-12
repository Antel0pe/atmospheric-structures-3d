from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import xarray as xr
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_voxel_builder import (
    clear_output_dir,
    coordinate_step_degrees,
    timestamp_to_slug,
)


OUTPUT_VERSION = 1
DATASET_VARIABLE = "t"
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_SPECIFIC_HUMIDITY_PATH = Path("data/era5_specific-humidity_2021-11_08-12.nc")
RAW_TEMPERATURE_FIELD_KIND = "raw-temperature"
EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND = "equivalent-potential-temperature"
TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND = "temperature-climatology-anomaly"
POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND = (
    "potential-temperature-climatology-anomaly"
)
RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND = (
    "raw-temperature-vertical-coherence"
)
RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND = (
    "raw-temperature-anomaly-strength"
)
RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND = (
    "raw-temperature-anomaly-agreement"
)
RAW_TEMPERATURE_FRONT_OVERLAY_FIELD_KIND = "raw-temperature-front-overlay"
THERMAL_DISPLACEMENT_LATITUDE_FIELD_KIND = "thermal-displacement-latitude"
THERMAL_DISPLACEMENT_LATITUDE_SMOOTHED_FIELD_KIND = (
    "thermal-displacement-latitude-smoothed"
)
THERMAL_DISPLACEMENT_ZONAL_MEAN_FIELD_KIND = (
    "thermal-displacement-zonal-mean-latitude"
)
THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND = (
    "thermal-displacement-zonal-trimmed-mean-latitude"
)
THERMAL_CONFLICT_NEIGHBORHOOD_FIELD_KIND = "thermal-conflict-neighborhood"
THERMAL_DISPLACEMENT_SMOOTH_SIGMA_CELLS = 20.0
FRONT_OVERLAY_ENCODING = "raw-temperature-uint16-rg-front-mask-b"
THERMAL_CONFLICT_ENCODING = "thermal-conflict-warmness-uint16-rg-conflict-b"
DEFAULT_FRONT_SMOOTH_SIGMA_CELLS = 2.0
DEFAULT_FRONT_GRADIENT_PERCENTILE = 96.0
DEFAULT_FRONT_EXCLUDE_TROPICS_ABS_LAT = 20.0
DEFAULT_FRONT_DILATION_ITERATIONS = 1
DEFAULT_FRONT_MIN_CELLS = 16
DEFAULT_VARIANT_NAME = "raw-temperature"
DEFAULT_VARIANT_LABEL = "Raw Temperature"
DEFAULT_COLOR_SCALE = "global-min-blue-to-global-max-red"
DEFAULT_OUTPUT_DIR = Path("public/temperature-slices/variants") / DEFAULT_VARIANT_NAME
DEFAULT_TEMPERATURE_CLIMATOLOGY_PATH = Path(
    "data/era5_temperature-climatology_1990-2020_11-08_12.nc"
)
DEFAULT_THETA_CLIMATOLOGY_PATH = Path(
    "data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc"
)
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BORDER_GEOJSON_PATH = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)


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
class ClimatologyField:
    dataset_path: Path
    variable_name: str
    values: np.ndarray
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static pressure-level temperature textures for the full-field "
            "temperature slice layer."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 pressure-level temperature NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated temperature variant textures will be written.",
    )
    parser.add_argument(
        "--field-kind",
        type=str,
        default=RAW_TEMPERATURE_FIELD_KIND,
        choices=(
            RAW_TEMPERATURE_FIELD_KIND,
            EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND,
            TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
            POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
            RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND,
            RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND,
            RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND,
            RAW_TEMPERATURE_FRONT_OVERLAY_FIELD_KIND,
            THERMAL_DISPLACEMENT_LATITUDE_FIELD_KIND,
            THERMAL_DISPLACEMENT_LATITUDE_SMOOTHED_FIELD_KIND,
            THERMAL_DISPLACEMENT_ZONAL_MEAN_FIELD_KIND,
            THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND,
            THERMAL_CONFLICT_NEIGHBORHOOD_FIELD_KIND,
        ),
        help="Scalar field exported into the pressure-slice textures.",
    )
    parser.add_argument(
        "--specific-humidity",
        type=Path,
        default=DEFAULT_SPECIFIC_HUMIDITY_PATH,
        help=(
            "Path to the ERA5 pressure-level specific humidity NetCDF file. "
            "Used when --field-kind equivalent-potential-temperature."
        ),
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        default=None,
        help="Optional matched-grid climatology NetCDF file for anomaly field kinds.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=DEFAULT_VARIANT_NAME,
        help="Variant slug written into the manifest and used by the frontend.",
    )
    parser.add_argument(
        "--variant-label",
        type=str,
        default=DEFAULT_VARIANT_LABEL,
        help="Human-readable variant label written into the manifest.",
    )
    parser.add_argument(
        "--color-scale",
        type=str,
        default=DEFAULT_COLOR_SCALE,
        choices=(
            "global-min-blue-to-global-max-red",
            "per-level-min-blue-to-per-level-max-red",
            "global-symmetric-zero-white-blue-red",
        ),
        help="Color-scale recipe described by this variant manifest.",
    )
    parser.add_argument(
        "--pressure-min-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MIN_HPA,
        help="Lowest pressure level to export.",
    )
    parser.add_argument(
        "--pressure-max-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MAX_HPA,
        help="Highest pressure level to export.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default=",".join(DEFAULT_INCLUDE_TIMESTAMPS),
        help="Comma-separated ISO minute timestamps to build.",
    )
    parser.add_argument(
        "--front-smooth-sigma-cells",
        type=float,
        default=DEFAULT_FRONT_SMOOTH_SIGMA_CELLS,
        help="Gaussian smoothing sigma used before front detection.",
    )
    parser.add_argument(
        "--front-gradient-percentile",
        type=float,
        default=DEFAULT_FRONT_GRADIENT_PERCENTILE,
        help="Per-level horizontal temperature-gradient percentile used as the front-candidate support mask.",
    )
    parser.add_argument(
        "--front-exclude-tropics-abs-lat",
        type=float,
        default=DEFAULT_FRONT_EXCLUDE_TROPICS_ABS_LAT,
        help="Exclude front candidates equatorward of this absolute latitude.",
    )
    parser.add_argument(
        "--front-dilation-iterations",
        type=int,
        default=DEFAULT_FRONT_DILATION_ITERATIONS,
        help="Number of 3x3 dilation passes applied to the detected front line mask.",
    )
    parser.add_argument(
        "--front-min-cells",
        type=int,
        default=DEFAULT_FRONT_MIN_CELLS,
        help="Drop connected front-mask fragments smaller than this many grid cells.",
    )
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if text.endswith("Z"):
        return text[:-1]
    return text


def format_display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        home = Path.home()
        try:
            return f"~/{path.relative_to(home).as_posix()}"
        except ValueError:
            return path.name or "<external-path>"


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {format_display_path(resolved)}"
        )
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


def load_climatology_field(path: Path, variable_name: str) -> ClimatologyField:
    dataset = xr.open_dataset(path)
    try:
        variable = dataset[variable_name]
        return ClimatologyField(
            dataset_path=path,
            variable_name=variable_name,
            values=np.asarray(variable.values, dtype=np.float32),
            pressure_levels_hpa=np.asarray(
                variable.coords["pressure_level"].values,
                dtype=np.float32,
            ),
            latitudes_deg=np.asarray(variable.coords["latitude"].values, dtype=np.float32),
            longitudes_deg=np.asarray(
                variable.coords["longitude"].values,
                dtype=np.float32,
            ),
        )
    finally:
        dataset.close()


def resolve_climatology(
    field_kind: str,
    climatology_path: Path | None,
) -> ClimatologyField | None:
    if field_kind in {
        RAW_TEMPERATURE_FIELD_KIND,
        EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND,
        RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND,
        RAW_TEMPERATURE_FRONT_OVERLAY_FIELD_KIND,
    }:
        return None

    if field_kind in {
        TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND,
        THERMAL_DISPLACEMENT_LATITUDE_FIELD_KIND,
        THERMAL_DISPLACEMENT_LATITUDE_SMOOTHED_FIELD_KIND,
        THERMAL_DISPLACEMENT_ZONAL_MEAN_FIELD_KIND,
        THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND,
        THERMAL_CONFLICT_NEIGHBORHOOD_FIELD_KIND,
    }:
        path = climatology_path or DEFAULT_TEMPERATURE_CLIMATOLOGY_PATH
        return load_climatology_field(
            resolve_dataset_path(path),
            "temperature_climatology_mean",
        )

    if field_kind == POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND:
        path = climatology_path or DEFAULT_THETA_CLIMATOLOGY_PATH
        return load_climatology_field(
            resolve_dataset_path(path),
            "theta_climatology_mean",
        )

    raise ValueError(f"Unsupported temperature slice field kind: {field_kind}")


def validate_climatology_grid(
    contents: DatasetContents,
    climatology: ClimatologyField | None,
) -> None:
    if climatology is None:
        return
    if not np.array_equal(contents.pressure_levels_hpa, climatology.pressure_levels_hpa):
        raise ValueError("Source temperature and climatology pressure levels do not match.")
    if not np.array_equal(contents.latitudes_deg, climatology.latitudes_deg):
        raise ValueError("Source temperature and climatology latitude grids do not match.")
    if not np.array_equal(contents.longitudes_deg, climatology.longitudes_deg):
        raise ValueError("Source temperature and climatology longitude grids do not match.")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_target_timestamps(
    all_timestamps: list[str],
    include_timestamps_text: str,
) -> list[str]:
    if not include_timestamps_text.strip():
        return all_timestamps

    requested = {
        value.strip()
        for value in include_timestamps_text.split(",")
        if value.strip()
    }
    return [timestamp for timestamp in all_timestamps if timestamp in requested]


def pressure_level_indices(
    pressure_levels_hpa: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> list[int]:
    lower = min(pressure_min_hpa, pressure_max_hpa)
    upper = max(pressure_min_hpa, pressure_max_hpa)
    return [
        index
        for index, pressure_hpa in enumerate(pressure_levels_hpa)
        if lower <= float(pressure_hpa) <= upper
    ]


def encode_field_to_rgba_uint8(
    values: np.ndarray,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    normalized = np.clip(
        (np.asarray(values, dtype=np.float32) - value_min)
        / max(value_max - value_min, 1e-6),
        0.0,
        1.0,
    )
    encoded16 = np.round(normalized * 65535.0).astype(np.uint16)
    rgba = np.empty((*encoded16.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (encoded16 >> 8).astype(np.uint8)
    rgba[..., 1] = (encoded16 & 255).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = 255
    return rgba


def encode_temperature_with_strength_to_rgba_uint8(
    temperature_k: np.ndarray,
    strength01: np.ndarray,
    temperature_min_k: float,
    temperature_max_k: float,
) -> np.ndarray:
    normalized_temperature = np.clip(
        (np.asarray(temperature_k, dtype=np.float32) - temperature_min_k)
        / max(temperature_max_k - temperature_min_k, 1e-6),
        0.0,
        1.0,
    )
    encoded16 = np.round(normalized_temperature * 65535.0).astype(np.uint16)
    strength8 = np.round(np.clip(strength01, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.empty((*encoded16.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (encoded16 >> 8).astype(np.uint8)
    rgba[..., 1] = (encoded16 & 255).astype(np.uint8)
    rgba[..., 2] = strength8
    rgba[..., 3] = 255
    return rgba


def is_front_overlay_field_kind(field_kind: str) -> bool:
    return field_kind == RAW_TEMPERATURE_FRONT_OVERLAY_FIELD_KIND


def is_signed_saturation_field_kind(field_kind: str) -> bool:
    return field_kind == RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND


def is_thermal_displacement_field_kind(field_kind: str) -> bool:
    return field_kind in {
        THERMAL_DISPLACEMENT_LATITUDE_FIELD_KIND,
        THERMAL_DISPLACEMENT_LATITUDE_SMOOTHED_FIELD_KIND,
        THERMAL_DISPLACEMENT_ZONAL_MEAN_FIELD_KIND,
        THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND,
    }


def is_thermal_conflict_field_kind(field_kind: str) -> bool:
    return field_kind == THERMAL_CONFLICT_NEIGHBORHOOD_FIELD_KIND


def dry_potential_temperature(
    temperature_k: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    return np.asarray(
        temperature_k * (1000.0 / pressure_hpa) ** 0.285906374501992,
        dtype=np.float32,
    )


def equivalent_potential_temperature(
    temperature_k: np.ndarray,
    specific_humidity_kgkg: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=np.float32)
    specific_humidity = np.clip(
        np.asarray(specific_humidity_kgkg, dtype=np.float32),
        0.0,
        0.2,
    )
    pressure = max(float(pressure_hpa), 1.0)
    epsilon = 0.622

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mixing_ratio = specific_humidity / np.maximum(1.0 - specific_humidity, 1.0e-7)
        vapor_pressure_hpa = (
            specific_humidity
            * pressure
            / np.maximum(epsilon + (1.0 - epsilon) * specific_humidity, 1.0e-7)
        )
        vapor_pressure_hpa = np.maximum(vapor_pressure_hpa, 1.0e-6)
        log_ratio = np.log(vapor_pressure_hpa / 6.112)
        dewpoint_c = (243.5 * log_ratio) / np.maximum(17.67 - log_ratio, 1.0e-7)
        dewpoint_k = dewpoint_c + 273.15
        lifting_condensation_temperature_k = (
            1.0
            / (
                1.0 / np.maximum(dewpoint_k - 56.0, 1.0e-7)
                + np.log(np.maximum(temperature, 1.0) / np.maximum(dewpoint_k, 1.0))
                / 800.0
            )
            + 56.0
        )
        theta = temperature * (1000.0 / pressure) ** (
            0.2854 * (1.0 - 0.28 * mixing_ratio)
        )
        theta_e = theta * np.exp(
            (3.376 / np.maximum(lifting_condensation_temperature_k, 1.0) - 0.00254)
            * mixing_ratio
            * 1000.0
            * (1.0 + 0.81 * mixing_ratio)
        )

    return np.asarray(theta_e, dtype=np.float32)


def thermal_displacement_latitude_score(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes))), 1e-6)

    for lon_index in range(raw.shape[1]):
        climatology_column = climatology[:, lon_index]
        valid = np.isfinite(climatology_column)
        if not np.any(valid):
            result[:, lon_index] = np.nan
            continue

        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(climatology_column[valid], kind="mergesort")]
        sorted_temperatures = climatology_column[order]
        source_temperatures = raw[:, lon_index]
        insert_positions = np.searchsorted(sorted_temperatures, source_temperatures)
        right_positions = np.clip(insert_positions, 0, sorted_temperatures.size - 1)
        left_positions = np.clip(insert_positions - 1, 0, sorted_temperatures.size - 1)

        right_indices = order[right_positions]
        left_indices = order[left_positions]
        right_diffs = np.abs(sorted_temperatures[right_positions] - source_temperatures)
        left_diffs = np.abs(sorted_temperatures[left_positions] - source_temperatures)

        row_indices = np.arange(raw.shape[0])
        right_lat_distance = np.abs(right_indices - row_indices)
        left_lat_distance = np.abs(left_indices - row_indices)
        use_left = (left_diffs < right_diffs) | (
            (left_diffs == right_diffs) & (left_lat_distance <= right_lat_distance)
        )
        matched_indices = np.where(use_left, left_indices, right_indices)
        matched_latitudes = latitudes[matched_indices]
        result[:, lon_index] = 1.0 - np.abs(matched_latitudes) / max_abs_latitude
        result[~np.isfinite(source_temperatures), lon_index] = np.nan

    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def climatology_zonal_profile(
    climatology_temperature_k: np.ndarray,
    trim_fraction: float = 0.0,
) -> np.ndarray:
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    if trim_fraction <= 0.0:
        return np.asarray(np.nanmean(climatology, axis=1), dtype=np.float32)

    lower = np.nanquantile(climatology, trim_fraction, axis=1)
    upper = np.nanquantile(climatology, 1.0 - trim_fraction, axis=1)
    middle = np.where(
        (climatology >= lower[:, None]) & (climatology <= upper[:, None]),
        climatology,
        np.nan,
    )
    return np.asarray(np.nanmean(middle, axis=1), dtype=np.float32)


def thermal_displacement_zonal_profile_score(
    raw_temperature_k: np.ndarray,
    climatology_latitude_profile_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    profile = np.asarray(climatology_latitude_profile_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes))), 1e-6)

    valid = np.isfinite(profile)
    if not np.any(valid):
        result[:] = np.nan
        return result

    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(profile[valid], kind="mergesort")]
    sorted_temperatures = profile[order]
    flat_raw = raw.reshape(-1)
    insert_positions = np.searchsorted(sorted_temperatures, flat_raw)
    right_positions = np.clip(insert_positions, 0, sorted_temperatures.size - 1)
    left_positions = np.clip(insert_positions - 1, 0, sorted_temperatures.size - 1)

    right_indices = order[right_positions]
    left_indices = order[left_positions]
    right_diffs = np.abs(sorted_temperatures[right_positions] - flat_raw)
    left_diffs = np.abs(sorted_temperatures[left_positions] - flat_raw)

    row_indices = np.repeat(np.arange(raw.shape[0]), raw.shape[1])
    right_lat_distance = np.abs(right_indices - row_indices)
    left_lat_distance = np.abs(left_indices - row_indices)
    use_left = (left_diffs < right_diffs) | (
        (left_diffs == right_diffs) & (left_lat_distance <= right_lat_distance)
    )
    matched_indices = np.where(use_left, left_indices, right_indices)
    matched_latitudes = latitudes[matched_indices]
    flat_result = 1.0 - np.abs(matched_latitudes) / max_abs_latitude
    flat_result[~np.isfinite(flat_raw)] = np.nan
    result = flat_result.reshape(raw.shape)

    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def smoothstep_array(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    if edge1 >= edge0:
        t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    else:
        t = np.clip((edge0 - value) / max(edge0 - edge1, 1e-6), 0.0, 1.0)
    return np.asarray(t * t * (3.0 - 2.0 * t), dtype=np.float32)


def matched_zonal_mean_latitudes_same_hemisphere(
    raw_temperature_k: np.ndarray,
    climatology_latitude_profile_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    profile = np.asarray(climatology_latitude_profile_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    result = np.full_like(raw, np.nan, dtype=np.float32)

    for source_mask, candidate_mask in (
        (latitudes >= 0.0, latitudes >= 0.0),
        (latitudes < 0.0, latitudes < 0.0),
    ):
        valid = np.isfinite(profile) & candidate_mask
        if not np.any(valid):
            continue

        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(profile[valid], kind="mergesort")]
        sorted_temperatures = profile[order]

        source_rows = np.flatnonzero(source_mask)
        source_values = raw[source_rows, :].reshape(-1)
        insert_positions = np.searchsorted(sorted_temperatures, source_values)
        right_positions = np.clip(insert_positions, 0, sorted_temperatures.size - 1)
        left_positions = np.clip(insert_positions - 1, 0, sorted_temperatures.size - 1)

        right_indices = order[right_positions]
        left_indices = order[left_positions]
        right_diffs = np.abs(sorted_temperatures[right_positions] - source_values)
        left_diffs = np.abs(sorted_temperatures[left_positions] - source_values)
        repeated_rows = np.repeat(source_rows, raw.shape[1])
        right_lat_distance = np.abs(right_indices - repeated_rows)
        left_lat_distance = np.abs(left_indices - repeated_rows)
        use_left = (left_diffs < right_diffs) | (
            (left_diffs == right_diffs) & (left_lat_distance <= right_lat_distance)
        )
        matched_indices = np.where(use_left, left_indices, right_indices)
        result[source_rows, :] = latitudes[matched_indices].reshape(source_rows.size, raw.shape[1])

    return result


def grid_gradient(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    drow = np.gradient(field, axis=0, edge_order=2)
    dcol = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * 0.5
    return np.asarray(np.sqrt(np.square(drow) + np.square(dcol)), dtype=np.float32)


def local_max_filter(values: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return np.asarray(
        ndimage.maximum_filter(
            np.asarray(values, dtype=np.float32),
            size=(size, size),
            mode=("nearest", "wrap"),
        ),
        dtype=np.float32,
    )


def thermal_conflict_neighborhood_fields(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_temperature = smooth_lat_lon(raw_temperature_k, 1.0)
    profile = climatology_zonal_profile(climatology_temperature_k)
    matched_latitudes = matched_zonal_mean_latitudes_same_hemisphere(
        source_temperature,
        profile,
        latitudes_deg,
    )
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes_deg))), 1e-6)
    abs_matched_latitude = np.abs(matched_latitudes)
    warmness = np.asarray(1.0 - abs_matched_latitude / max_abs_latitude, dtype=np.float32)
    warmness[~np.isfinite(abs_matched_latitude)] = np.nan
    warmness = smooth_lat_lon(warmness, 1.0)

    warm_membership = smoothstep_array(55.0, 40.0, abs_matched_latitude)
    cold_membership = smoothstep_array(40.0, 60.0, abs_matched_latitude)
    warm_membership = smooth_lat_lon(warm_membership, 1.0)
    cold_membership = smooth_lat_lon(cold_membership, 1.0)

    local_gradient = grid_gradient(warmness)
    finite_gradient = local_gradient[np.isfinite(local_gradient)]
    gradient_display_max = (
        float(np.nanpercentile(finite_gradient, 99.0))
        if finite_gradient.size
        else 1.0
    )
    local_gradient_norm = np.clip(local_gradient / max(gradient_display_max, 1e-6), 0.0, 1.0)

    best_score = np.zeros_like(warmness, dtype=np.float32)
    for radius in (3, 6, 12):
        warm_near = local_max_filter(warm_membership, radius)
        cold_near = local_max_filter(cold_membership, radius)
        co_presence = np.sqrt(np.clip(warm_near * cold_near, 0.0, 1.0))
        score = co_presence * np.sqrt(local_gradient_norm)
        best_score = np.maximum(best_score, score.astype(np.float32))

    abs_latitudes = np.abs(np.asarray(latitudes_deg, dtype=np.float32))[:, None]
    equatorward_weight = smoothstep_array(20.0, 35.0, abs_latitudes)
    poleward_weight = 1.0 - smoothstep_array(70.0, 82.0, abs_latitudes)
    conflict = np.asarray(best_score * equatorward_weight * poleward_weight, dtype=np.float32)
    conflict[~np.isfinite(warmness)] = 0.0
    return np.asarray(np.clip(warmness, 0.0, 1.0), dtype=np.float32), np.clip(conflict, 0.0, 1.0)


def smooth_raw_temperature_for_thermal_displacement(
    raw_temperature_k: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        gaussian_filter(
            np.asarray(raw_temperature_k, dtype=np.float32),
            sigma=(THERMAL_DISPLACEMENT_SMOOTH_SIGMA_CELLS,) * 2,
            mode=("nearest", "wrap"),
            truncate=3.0,
        ),
        dtype=np.float32,
    )


def smooth_lat_lon(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    sigma = max(float(sigma_cells), 0.0)
    if sigma <= 0.0:
        return np.asarray(values, dtype=np.float32)
    return np.asarray(
        gaussian_filter(
            np.asarray(values, dtype=np.float32),
            sigma=(sigma, sigma),
            mode=("nearest", "wrap"),
            truncate=3.0,
        ),
        dtype=np.float32,
    )


def horizontal_derivatives_per_km(
    values: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    dlat = float(np.mean(np.abs(np.diff(lat_rad))))
    dlon = float(np.mean(np.abs(np.diff(lon_rad))))
    earth_radius_km = 6371.0

    grad_north = (
        np.gradient(values.astype(np.float64), dlat, axis=0, edge_order=1)
        / earth_radius_km
    )
    east = np.roll(values, -1, axis=1)
    west = np.roll(values, 1, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dvalue_dlon = (east - west) / (2.0 * dlon)
        cos_lat = np.cos(lat_rad)[:, None]
        safe_cos_lat = np.where(np.abs(cos_lat) < 1.0e-6, np.nan, cos_lat)
        grad_east = dvalue_dlon / (earth_radius_km * safe_cos_lat)

    return grad_east.astype(np.float32), grad_north.astype(np.float32)


def temperature_front_mask(
    raw_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    smooth_sigma_cells: float,
    gradient_percentile: float,
    exclude_tropics_abs_lat: float,
    dilation_iterations: int,
    min_cells: int,
) -> tuple[np.ndarray, float]:
    smoothed = smooth_lat_lon(raw_temperature_k, smooth_sigma_cells)
    grad_east, grad_north = horizontal_derivatives_per_km(
        smoothed,
        latitudes_deg,
        longitudes_deg,
    )
    gradient_mag = np.sqrt(np.square(grad_east) + np.square(grad_north))
    mag_east, mag_north = horizontal_derivatives_per_km(
        gradient_mag,
        latitudes_deg,
        longitudes_deg,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        tfp = -((mag_east * grad_east) + (mag_north * grad_north)) / np.maximum(
            gradient_mag,
            1.0e-6,
        )

    gradient_k_per_100km = gradient_mag * 100.0
    finite_gradient = gradient_k_per_100km[np.isfinite(gradient_k_per_100km)]
    threshold = (
        float(np.nanpercentile(finite_gradient, gradient_percentile))
        if finite_gradient.size
        else float("nan")
    )
    strong = gradient_k_per_100km >= threshold
    strong &= np.abs(latitudes_deg[:, None]) >= float(exclude_tropics_abs_lat)
    finite_tfp = np.isfinite(tfp)

    east_crossing = finite_tfp & np.roll(finite_tfp, -1, axis=1)
    east_crossing &= (tfp <= 0.0) != (np.roll(tfp, -1, axis=1) <= 0.0)
    west_crossing = finite_tfp & np.roll(finite_tfp, 1, axis=1)
    west_crossing &= (tfp <= 0.0) != (np.roll(tfp, 1, axis=1) <= 0.0)

    north_crossing = np.zeros_like(strong, dtype=bool)
    south_crossing = np.zeros_like(strong, dtype=bool)
    north_crossing[1:, :] = (
        finite_tfp[1:, :]
        & finite_tfp[:-1, :]
        & ((tfp[1:, :] <= 0.0) != (tfp[:-1, :] <= 0.0))
    )
    south_crossing[:-1, :] = (
        finite_tfp[:-1, :]
        & finite_tfp[1:, :]
        & ((tfp[:-1, :] <= 0.0) != (tfp[1:, :] <= 0.0))
    )

    front = strong & (east_crossing | west_crossing | north_crossing | south_crossing)
    for _ in range(max(int(dilation_iterations), 0)):
        padded = np.pad(front, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        padded[1:-1, 0] = front[:, -1]
        padded[1:-1, -1] = front[:, 0]
        front = ndimage.binary_dilation(
            padded,
            structure=np.ones((3, 3), dtype=bool),
        )[1:-1, 1:-1]

    if min_cells > 1 and np.any(front):
        labels, label_count = ndimage.label(front)
        cleaned = np.zeros_like(front, dtype=bool)
        for label_id in range(1, label_count + 1):
            component = labels == label_id
            if int(np.count_nonzero(component)) >= min_cells:
                cleaned |= component
        front = cleaned

    return front.astype(np.float32), threshold


def build_field(
    raw_temperature_k: np.ndarray,
    pressure_hpa: float,
    level_index: int,
    field_kind: str,
    climatology: ClimatologyField | None,
    specific_humidity_kgkg: np.ndarray | None = None,
    latitudes_deg: np.ndarray | None = None,
) -> np.ndarray:
    if field_kind == RAW_TEMPERATURE_FIELD_KIND:
        return raw_temperature_k
    if is_front_overlay_field_kind(field_kind):
        return raw_temperature_k
    if field_kind == EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND:
        if specific_humidity_kgkg is None:
            raise ValueError("Missing specific humidity for equivalent potential temperature.")
        return equivalent_potential_temperature(
            temperature_k=raw_temperature_k,
            specific_humidity_kgkg=specific_humidity_kgkg,
            pressure_hpa=pressure_hpa,
        )
    if climatology is None:
        raise ValueError(f"Missing climatology for field kind: {field_kind}")
    if field_kind == TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND:
        return np.asarray(raw_temperature_k - climatology.values[level_index], dtype=np.float32)
    if field_kind == POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND:
        theta = dry_potential_temperature(raw_temperature_k, pressure_hpa)
        return np.asarray(theta - climatology.values[level_index], dtype=np.float32)
    if is_thermal_displacement_field_kind(field_kind):
        if latitudes_deg is None:
            raise ValueError("Missing latitudes for thermal displacement field.")
        source_temperature = (
            smooth_raw_temperature_for_thermal_displacement(raw_temperature_k)
            if field_kind == THERMAL_DISPLACEMENT_LATITUDE_SMOOTHED_FIELD_KIND
            else raw_temperature_k
        )
        if field_kind in {
            THERMAL_DISPLACEMENT_ZONAL_MEAN_FIELD_KIND,
            THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND,
        }:
            profile = climatology_zonal_profile(
                climatology.values[level_index],
                trim_fraction=0.1
                if field_kind == THERMAL_DISPLACEMENT_ZONAL_TRIMMED_MEAN_FIELD_KIND
                else 0.0,
            )
            return thermal_displacement_zonal_profile_score(
                raw_temperature_k=source_temperature,
                climatology_latitude_profile_k=profile,
                latitudes_deg=latitudes_deg,
            )
        return thermal_displacement_latitude_score(
            raw_temperature_k=source_temperature,
            climatology_temperature_k=climatology.values[level_index],
            latitudes_deg=latitudes_deg,
        )
    raise ValueError(f"Unsupported temperature slice field kind: {field_kind}")


def is_saturation_encoded_field_kind(field_kind: str) -> bool:
    return field_kind in {
        RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND,
    }


def pressure_to_standard_atmosphere_height_m(pressure_hpa: np.ndarray) -> np.ndarray:
    safe_pressure = np.maximum(np.asarray(pressure_hpa, dtype=np.float32), 1.0)
    return np.asarray(
        44330.0 * (1.0 - (safe_pressure / 1013.25) ** 0.1903),
        dtype=np.float32,
    )


def vertical_coherence_kink_strength(
    raw_stack_k: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> np.ndarray:
    heights_m = pressure_to_standard_atmosphere_height_m(pressure_levels_hpa)
    strength = np.zeros_like(raw_stack_k, dtype=np.float32)

    for level_index in range(2, raw_stack_k.shape[0] - 2):
        lower_span_m = heights_m[level_index] - heights_m[level_index - 2]
        upper_span_m = heights_m[level_index + 2] - heights_m[level_index]
        if abs(float(lower_span_m)) < 1e-6 or abs(float(upper_span_m)) < 1e-6:
            continue
        lower_trend = (
            raw_stack_k[level_index] - raw_stack_k[level_index - 2]
        ) / lower_span_m
        upper_trend = (
            raw_stack_k[level_index + 2] - raw_stack_k[level_index]
        ) / upper_span_m
        strength[level_index] = np.abs(upper_trend - lower_trend)

    return strength


def read_raw_temperature_stack(
    raw_dataset: netCDF4.Dataset,
    time_index: int,
    level_indices: list[int],
) -> np.ndarray:
    variable = raw_dataset.variables[DATASET_VARIABLE]
    return np.asarray(variable[time_index, level_indices, :, :], dtype=np.float32)


def uses_specific_humidity(field_kind: str) -> bool:
    return field_kind == EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND


def normalize_longitudes_360(longitudes_deg: np.ndarray) -> np.ndarray:
    return np.asarray(np.mod(longitudes_deg, 360.0), dtype=np.float32)


def longitudes_for_source_grid(
    target_longitudes_deg: np.ndarray,
    source_longitudes_deg: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_longitudes_deg, dtype=np.float32)
    if np.nanmin(source) >= 0.0 and np.nanmax(source) > 180.0:
        return normalize_longitudes_360(target_longitudes_deg)
    return np.asarray(target_longitudes_deg, dtype=np.float32)


def load_specific_humidity_stack(
    specific_humidity_path: Path,
    timestamp: str,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    dataset_path = resolve_dataset_path(specific_humidity_path)
    dataset = xr.open_dataset(dataset_path)
    try:
        variable = dataset["q"]
        selected = variable.sel(
            valid_time=np.datetime64(timestamp),
            pressure_level=xr.DataArray(pressure_levels_hpa, dims="pressure_level"),
            latitude=xr.DataArray(latitudes_deg, dims="latitude"),
            longitude=xr.DataArray(
                longitudes_for_source_grid(
                    target_longitudes_deg=longitudes_deg,
                    source_longitudes_deg=np.asarray(variable.coords["longitude"].values),
                ),
                dims="longitude",
            ),
        )
        return np.asarray(selected.values, dtype=np.float32)
    finally:
        dataset.close()


def validate_specific_humidity_stack(
    specific_humidity_stack: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    if specific_humidity_stack.shape != expected_shape:
        raise ValueError(
            "Specific humidity grid does not match selected temperature grid: "
            f"expected {expected_shape}, got {specific_humidity_stack.shape}."
        )


def saturation_strength_stack(
    raw_stack_k: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    level_indices: list[int],
    field_kind: str,
    climatology: ClimatologyField | None,
) -> np.ndarray:
    if field_kind == RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND:
        return vertical_coherence_kink_strength(raw_stack_k, pressure_levels_hpa)

    if field_kind == RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND:
        if climatology is None:
            raise ValueError(f"Missing climatology for field kind: {field_kind}")
        climatology_stack = climatology.values[level_indices]
        return np.abs(raw_stack_k - climatology_stack).astype(np.float32)

    if field_kind == RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND:
        if climatology is None:
            raise ValueError(f"Missing climatology for field kind: {field_kind}")
        climatology_stack = climatology.values[level_indices]
        return (raw_stack_k - climatology_stack).astype(np.float32)

    raise ValueError(f"Unsupported saturation field kind: {field_kind}")


def compute_field_range(
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    target_time_indices: list[int],
    level_indices: list[int],
    field_kind: str,
    climatology: ClimatologyField | None,
    specific_humidity_path: Path | None = None,
) -> tuple[float, float]:
    if is_thermal_displacement_field_kind(
        field_kind
    ) or is_thermal_conflict_field_kind(field_kind):
        return 0.0, 1.0

    variable = raw_dataset.variables[DATASET_VARIABLE]
    data_min = np.inf
    data_max = -np.inf

    for timestamp, time_index in zip(
        (contents.timestamps[index] for index in target_time_indices),
        target_time_indices,
    ):
        specific_humidity_stack = None
        if uses_specific_humidity(field_kind):
            if specific_humidity_path is None:
                raise ValueError(
                    "Missing specific humidity path for equivalent potential temperature."
                )
            specific_humidity_stack = load_specific_humidity_stack(
                specific_humidity_path=specific_humidity_path,
                timestamp=timestamp,
                pressure_levels_hpa=contents.pressure_levels_hpa[level_indices],
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
            )
            validate_specific_humidity_stack(
                specific_humidity_stack,
                (
                    len(level_indices),
                    contents.latitudes_deg.size,
                    contents.longitudes_deg.size,
                ),
            )

        for stack_index, level_index in enumerate(level_indices):
            pressure_hpa = float(contents.pressure_levels_hpa[level_index])
            raw_temperature = np.asarray(
                variable[time_index, level_index, :, :],
                dtype=np.float32,
            )
            field = build_field(
                raw_temperature,
                pressure_hpa=pressure_hpa,
                level_index=level_index,
                field_kind=field_kind,
                climatology=climatology,
                specific_humidity_kgkg=None
                if specific_humidity_stack is None
                else specific_humidity_stack[stack_index],
                latitudes_deg=contents.latitudes_deg,
            )
            data_min = min(data_min, float(np.nanmin(field)))
            data_max = max(data_max, float(np.nanmax(field)))

    if field_kind in {
        TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
        POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
    }:
        anomaly_abs_max = max(abs(data_min), abs(data_max))
        return -anomaly_abs_max, anomaly_abs_max

    return data_min, data_max


def compute_raw_temperature_range(
    raw_dataset: netCDF4.Dataset,
    target_time_indices: list[int],
    level_indices: list[int],
) -> tuple[float, float]:
    variable = raw_dataset.variables[DATASET_VARIABLE]
    data_min = np.inf
    data_max = -np.inf

    for time_index in target_time_indices:
        raw_stack = np.asarray(variable[time_index, level_indices, :, :], dtype=np.float32)
        data_min = min(data_min, float(np.nanmin(raw_stack)))
        data_max = max(data_max, float(np.nanmax(raw_stack)))

    return data_min, data_max


def compute_saturation_strength_scale(
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    target_time_indices: list[int],
    level_indices: list[int],
    field_kind: str,
    climatology: ClimatologyField | None,
) -> float:
    samples: list[np.ndarray] = []
    pressure_levels_hpa = contents.pressure_levels_hpa[level_indices]

    for time_index in target_time_indices:
        raw_stack = read_raw_temperature_stack(raw_dataset, time_index, level_indices)
        strength = saturation_strength_stack(
            raw_stack_k=raw_stack,
            pressure_levels_hpa=pressure_levels_hpa,
            level_indices=level_indices,
            field_kind=field_kind,
            climatology=climatology,
        )
        valid = strength[np.isfinite(strength)]
        if valid.size:
            samples.append(valid.reshape(-1))

    if not samples:
        return 1.0

    all_values = np.abs(np.concatenate(samples))
    nonzero = all_values[all_values > 0]
    if nonzero.size == 0:
        return 1.0

    return float(max(np.nanpercentile(nonzero, 95), 1e-6))


def build_timestamp_entry(
    output_dir: Path,
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    timestamp: str,
    time_index: int,
    level_indices: list[int],
    field_min: float,
    field_max: float,
    field_kind: str,
    climatology: ClimatologyField | None,
    specific_humidity_stack: np.ndarray | None = None,
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    variable = raw_dataset.variables[DATASET_VARIABLE]
    levels: list[dict[str, str | float]] = []

    if specific_humidity_stack is not None:
        validate_specific_humidity_stack(
            specific_humidity_stack,
            (
                len(level_indices),
                contents.latitudes_deg.size,
                contents.longitudes_deg.size,
            ),
        )

    for stack_index, level_index in enumerate(level_indices):
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        raw_temperature = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
        field = build_field(
            raw_temperature,
            pressure_hpa=pressure_hpa,
            level_index=level_index,
            field_kind=field_kind,
            climatology=climatology,
            specific_humidity_kgkg=None
            if specific_humidity_stack is None
            else specific_humidity_stack[stack_index],
            latitudes_deg=contents.latitudes_deg,
        )
        encoded = encode_field_to_rgba_uint8(
            field,
            value_min=field_min,
            value_max=field_max,
        )
        image_name = f"temperature-{int(round(pressure_hpa))}hpa.png"
        image_path = frame_dir / image_name
        Image.fromarray(encoded, mode="RGBA").save(image_path, optimize=True)
        levels.append(
            {
                "pressure_hpa": pressure_hpa,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "temperature_min_k": 0.0
                if is_thermal_displacement_field_kind(field_kind)
                else float(np.nanmin(field)),
                "temperature_max_k": 1.0
                if is_thermal_displacement_field_kind(field_kind)
                else float(np.nanmax(field)),
                "value_min": float(np.nanmin(field)),
                "value_max": float(np.nanmax(field)),
            }
        )

    levels.sort(key=lambda entry: float(entry["pressure_hpa"]))
    return {
        "timestamp": timestamp,
        "levels": levels,
    }


def build_saturation_timestamp_entry(
    output_dir: Path,
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    timestamp: str,
    time_index: int,
    level_indices: list[int],
    temperature_min_k: float,
    temperature_max_k: float,
    strength_scale: float,
    field_kind: str,
    climatology: ClimatologyField | None,
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    pressure_levels_hpa = contents.pressure_levels_hpa[level_indices]
    raw_stack = read_raw_temperature_stack(raw_dataset, time_index, level_indices)
    strength = saturation_strength_stack(
        raw_stack_k=raw_stack,
        pressure_levels_hpa=pressure_levels_hpa,
        level_indices=level_indices,
        field_kind=field_kind,
        climatology=climatology,
    )
    if is_signed_saturation_field_kind(field_kind):
        normalized_strength = np.clip(
            0.5 + 0.5 * strength / max(strength_scale, 1e-6),
            0.0,
            1.0,
        )
    else:
        normalized_strength = np.clip(strength / max(strength_scale, 1e-6), 0.0, 1.0)

    levels: list[dict[str, str | float]] = []
    for stack_index, level_index in enumerate(level_indices):
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        raw_temperature = raw_stack[stack_index]
        strength_slice = normalized_strength[stack_index]
        encoded = encode_temperature_with_strength_to_rgba_uint8(
            temperature_k=raw_temperature,
            strength01=strength_slice,
            temperature_min_k=temperature_min_k,
            temperature_max_k=temperature_max_k,
        )
        image_name = f"temperature-{int(round(pressure_hpa))}hpa.png"
        image_path = frame_dir / image_name
        Image.fromarray(encoded, mode="RGBA").save(image_path, optimize=True)
        levels.append(
            {
                "pressure_hpa": pressure_hpa,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "temperature_min_k": float(np.nanmin(raw_temperature)),
                "temperature_max_k": float(np.nanmax(raw_temperature)),
                "value_min": float(np.nanmin(raw_temperature)),
                "value_max": float(np.nanmax(raw_temperature)),
                "saturation_strength_min": float(np.nanmin(strength[stack_index])),
                "saturation_strength_max": float(np.nanmax(strength[stack_index])),
            }
        )

    levels.sort(key=lambda entry: float(entry["pressure_hpa"]))
    return {
        "timestamp": timestamp,
        "levels": levels,
    }


def build_front_overlay_timestamp_entry(
    output_dir: Path,
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    timestamp: str,
    time_index: int,
    level_indices: list[int],
    temperature_min_k: float,
    temperature_max_k: float,
    front_smooth_sigma_cells: float,
    front_gradient_percentile: float,
    front_exclude_tropics_abs_lat: float,
    front_dilation_iterations: int,
    front_min_cells: int,
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    variable = raw_dataset.variables[DATASET_VARIABLE]
    levels: list[dict[str, str | float]] = []

    for level_index in level_indices:
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        raw_temperature = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
        front_mask, gradient_threshold = temperature_front_mask(
            raw_temperature_k=raw_temperature,
            latitudes_deg=contents.latitudes_deg,
            longitudes_deg=contents.longitudes_deg,
            smooth_sigma_cells=front_smooth_sigma_cells,
            gradient_percentile=front_gradient_percentile,
            exclude_tropics_abs_lat=front_exclude_tropics_abs_lat,
            dilation_iterations=front_dilation_iterations,
            min_cells=front_min_cells,
        )
        encoded = encode_temperature_with_strength_to_rgba_uint8(
            temperature_k=raw_temperature,
            strength01=front_mask,
            temperature_min_k=temperature_min_k,
            temperature_max_k=temperature_max_k,
        )
        image_name = f"temperature-{int(round(pressure_hpa))}hpa.png"
        image_path = frame_dir / image_name
        Image.fromarray(encoded, mode="RGBA").save(image_path, optimize=True)
        levels.append(
            {
                "pressure_hpa": pressure_hpa,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "temperature_min_k": float(np.nanmin(raw_temperature)),
                "temperature_max_k": float(np.nanmax(raw_temperature)),
                "value_min": float(np.nanmin(raw_temperature)),
                "value_max": float(np.nanmax(raw_temperature)),
                "front_mask_fraction": float(np.count_nonzero(front_mask) / front_mask.size),
                "front_gradient_threshold_k_per_100km": gradient_threshold,
            }
        )

    levels.sort(key=lambda entry: float(entry["pressure_hpa"]))
    return {
        "timestamp": timestamp,
        "levels": levels,
    }


def build_thermal_conflict_timestamp_entry(
    output_dir: Path,
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    timestamp: str,
    time_index: int,
    level_indices: list[int],
    climatology: ClimatologyField,
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    variable = raw_dataset.variables[DATASET_VARIABLE]
    levels: list[dict[str, str | float]] = []

    for level_index in level_indices:
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        raw_temperature = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
        warmness, conflict = thermal_conflict_neighborhood_fields(
            raw_temperature_k=raw_temperature,
            climatology_temperature_k=climatology.values[level_index],
            latitudes_deg=contents.latitudes_deg,
        )
        encoded = encode_temperature_with_strength_to_rgba_uint8(
            temperature_k=warmness,
            strength01=conflict,
            temperature_min_k=0.0,
            temperature_max_k=1.0,
        )
        image_name = f"temperature-{int(round(pressure_hpa))}hpa.png"
        image_path = frame_dir / image_name
        Image.fromarray(encoded, mode="RGBA").save(image_path, optimize=True)
        levels.append(
            {
                "pressure_hpa": pressure_hpa,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "temperature_min_k": 0.0,
                "temperature_max_k": 1.0,
                "value_min": float(np.nanmin(warmness)),
                "value_max": float(np.nanmax(warmness)),
                "conflict_min": float(np.nanmin(conflict)),
                "conflict_max": float(np.nanmax(conflict)),
                "conflict_p95": float(np.nanpercentile(conflict[np.isfinite(conflict)], 95.0)),
                "conflict_p99": float(np.nanpercentile(conflict[np.isfinite(conflict)], 99.0)),
            }
        )

    levels.sort(key=lambda entry: float(entry["pressure_hpa"]))
    return {
        "timestamp": timestamp,
        "levels": levels,
    }


def build_manifest(
    contents: DatasetContents,
    entries: list[dict],
    level_indices: list[int],
    field_min: float,
    field_max: float,
    field_kind: str,
    climatology: ClimatologyField | None,
    variant_name: str,
    variant_label: str,
    color_scale: str,
    saturation_strength_scale: float | None = None,
    front_detection: dict | None = None,
    specific_humidity_path: Path | None = None,
) -> dict:
    pressures = sorted(float(contents.pressure_levels_hpa[index]) for index in level_indices)
    display_units = (
        "equator-to-pole latitude match"
        if is_thermal_displacement_field_kind(field_kind)
        or is_thermal_conflict_field_kind(field_kind)
        else "K"
    )
    if is_thermal_conflict_field_kind(field_kind):
        encoding = THERMAL_CONFLICT_ENCODING
    elif is_front_overlay_field_kind(field_kind):
        encoding = FRONT_OVERLAY_ENCODING
    elif is_signed_saturation_field_kind(field_kind):
        encoding = "raw-temperature-uint16-rg-signed-saturation-b"
    elif saturation_strength_scale is not None:
        encoding = "raw-temperature-uint16-rg-saturation-strength-b"
    else:
        encoding = "normalized-temperature-uint16-packed-rg"

    manifest = {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "display_units": display_units,
        "variant": variant_name,
        "variant_label": variant_label,
        "field_kind": field_kind,
        "climatology_dataset": climatology.dataset_path.name if climatology else None,
        "specific_humidity_dataset": specific_humidity_path.name
        if specific_humidity_path
        else None,
        "rendering": {
            "kind": "full-field-pressure-slice",
            "filtering": "none",
            "color_scale": color_scale,
            "encoding": encoding,
        },
        "temperature_range_k": {
            "min": field_min,
            "max": field_max,
        },
        "value_range": {
            "min": field_min,
            "max": field_max,
        },
        "pressure_window_hpa": {
            "min": min(pressures),
            "max": max(pressures),
        },
        "pressure_levels_hpa": pressures,
        "grid": {
            "latitude_count": int(contents.latitudes_deg.size),
            "longitude_count": int(contents.longitudes_deg.size),
            "latitude_step_degrees": coordinate_step_degrees(contents.latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(contents.longitudes_deg),
            "latitude_min_degrees": float(np.nanmin(contents.latitudes_deg)),
            "latitude_max_degrees": float(np.nanmax(contents.latitudes_deg)),
            "longitude_min_degrees": float(np.nanmin(contents.longitudes_deg)),
            "longitude_max_degrees": float(np.nanmax(contents.longitudes_deg)),
        },
        "timestamps": entries,
    }

    if saturation_strength_scale is not None:
        manifest["saturation_strength_range"] = {
            "min": -saturation_strength_scale
            if is_signed_saturation_field_kind(field_kind)
            else 0.0,
            "max": saturation_strength_scale,
        }

    if front_detection is not None:
        manifest["front_detection"] = front_detection

    if is_thermal_conflict_field_kind(field_kind):
        manifest["thermal_conflict"] = {
            "thermal_identity": "hemisphere-constrained zonal-mean equivalent latitude",
            "warm_full_abs_latitude_degrees": 40.0,
            "warm_zero_abs_latitude_degrees": 55.0,
            "cold_zero_abs_latitude_degrees": 40.0,
            "cold_full_abs_latitude_degrees": 60.0,
            "conflict_method": "neighborhood opposition plus local sharpness",
            "neighborhood_radii_grid_cells": [3, 6, 12],
            "conflict_midlatitude_weight": {
                "fade_in_abs_latitude_degrees": [20.0, 35.0],
                "fade_out_abs_latitude_degrees": [70.0, 82.0],
            },
        }

    if field_kind == EQUIVALENT_POTENTIAL_TEMPERATURE_FIELD_KIND:
        manifest["equivalent_potential_temperature"] = {
            "method": "Bolton-style theta-e from pressure-level temperature and specific humidity",
            "inputs": ["temperature", "specific_humidity", "pressure_level"],
            "specific_humidity_alignment": "selected onto the temperature latitude/longitude grid before calculation",
        }

    return manifest



def split_dateline_segments(points: list[tuple[float, float]]) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for lon, lat in points:
        if previous_lon is not None and abs(lon - previous_lon) > 180:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((lon, lat))
        previous_lon = lon

    if len(segment) >= 2:
        yield segment


def lon_lat_to_pixel(
    lon: float,
    lat: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * (width - 1)
    y = (90.0 - lat) / 180.0 * (height - 1)
    return x, y


def draw_border_ring(
    draw: ImageDraw.ImageDraw,
    ring: list[list[float]],
    width: int,
    height: int,
) -> None:
    points = [(float(lon), float(lat)) for lon, lat, *_ in ring]
    for segment in split_dateline_segments(points):
        pixels = [lon_lat_to_pixel(lon, lat, width, height) for lon, lat in segment]
        draw.line(pixels, fill=(255, 255, 255, 220), width=1, joint="curve")


def write_border_texture(output_dir: Path, width: int, height: int) -> str | None:
    geojson_path = (REPO_ROOT / DEFAULT_BORDER_GEOJSON_PATH).resolve()
    if not geojson_path.exists():
        return None

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue

        for polygon in polygons:
            for ring in polygon:
                draw_border_ring(draw, ring, width, height)

    border_path = output_dir / "borders.png"
    image.save(border_path, optimize=True)
    return str(border_path.relative_to(output_dir)).replace("\\", "/")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    contents = load_dataset_contents(dataset_path)
    climatology = resolve_climatology(args.field_kind, args.climatology)
    validate_climatology_grid(contents, climatology)
    target_timestamps = resolve_target_timestamps(
        contents.timestamps,
        args.include_timestamps,
    )
    if not target_timestamps:
        raise ValueError("No matching timestamps were selected for export.")

    level_indices = pressure_level_indices(
        contents.pressure_levels_hpa,
        pressure_min_hpa=args.pressure_min_hpa,
        pressure_max_hpa=args.pressure_max_hpa,
    )
    if not level_indices:
        raise ValueError("No matching pressure levels were selected for export.")

    target_time_indices = [contents.timestamps.index(timestamp) for timestamp in target_timestamps]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    raw_dataset = netCDF4.Dataset(dataset_path)
    try:
        saturation_strength_scale: float | None = None
        front_detection: dict | None = None
        if is_front_overlay_field_kind(args.field_kind):
            field_min, field_max = compute_raw_temperature_range(
                raw_dataset=raw_dataset,
                target_time_indices=target_time_indices,
                level_indices=level_indices,
            )
            entries = [
                build_front_overlay_timestamp_entry(
                    output_dir=output_dir,
                    raw_dataset=raw_dataset,
                    contents=contents,
                    timestamp=timestamp,
                    time_index=time_index,
                    level_indices=level_indices,
                    temperature_min_k=field_min,
                    temperature_max_k=field_max,
                    front_smooth_sigma_cells=args.front_smooth_sigma_cells,
                    front_gradient_percentile=args.front_gradient_percentile,
                    front_exclude_tropics_abs_lat=args.front_exclude_tropics_abs_lat,
                    front_dilation_iterations=args.front_dilation_iterations,
                    front_min_cells=args.front_min_cells,
                )
                for timestamp, time_index in zip(target_timestamps, target_time_indices)
            ]
            front_detection = {
                "method": "TFP-style zero crossing inside the strongest smoothed horizontal raw-temperature gradients",
                "smooth_sigma_cells": float(args.front_smooth_sigma_cells),
                "gradient_percentile": float(args.front_gradient_percentile),
                "exclude_tropics_abs_lat": float(args.front_exclude_tropics_abs_lat),
                "dilation_iterations": int(args.front_dilation_iterations),
                "min_cells": int(args.front_min_cells),
                "limitation": "This is a temperature-structure proxy for frontal zones, not a synoptic front analysis with winds or time evolution.",
            }
        elif is_thermal_conflict_field_kind(args.field_kind):
            if climatology is None:
                raise ValueError("Missing climatology for thermal conflict field.")
            field_min, field_max = 0.0, 1.0
            entries = [
                build_thermal_conflict_timestamp_entry(
                    output_dir=output_dir,
                    raw_dataset=raw_dataset,
                    contents=contents,
                    timestamp=timestamp,
                    time_index=time_index,
                    level_indices=level_indices,
                    climatology=climatology,
                )
                for timestamp, time_index in zip(target_timestamps, target_time_indices)
            ]
        elif is_saturation_encoded_field_kind(args.field_kind):
            field_min, field_max = compute_raw_temperature_range(
                raw_dataset=raw_dataset,
                target_time_indices=target_time_indices,
                level_indices=level_indices,
            )
            saturation_strength_scale = compute_saturation_strength_scale(
                raw_dataset=raw_dataset,
                contents=contents,
                target_time_indices=target_time_indices,
                level_indices=level_indices,
                field_kind=args.field_kind,
                climatology=climatology,
            )
            entries = [
                build_saturation_timestamp_entry(
                    output_dir=output_dir,
                    raw_dataset=raw_dataset,
                    contents=contents,
                    timestamp=timestamp,
                    time_index=time_index,
                    level_indices=level_indices,
                    temperature_min_k=field_min,
                    temperature_max_k=field_max,
                    strength_scale=saturation_strength_scale,
                    field_kind=args.field_kind,
                    climatology=climatology,
                )
                for timestamp, time_index in zip(target_timestamps, target_time_indices)
            ]
        else:
            field_min, field_max = compute_field_range(
                raw_dataset=raw_dataset,
                contents=contents,
                target_time_indices=target_time_indices,
                level_indices=level_indices,
                field_kind=args.field_kind,
                climatology=climatology,
                specific_humidity_path=args.specific_humidity
                if uses_specific_humidity(args.field_kind)
                else None,
            )
            entries = []
            for timestamp, time_index in zip(target_timestamps, target_time_indices):
                specific_humidity_stack = None
                if uses_specific_humidity(args.field_kind):
                    specific_humidity_stack = load_specific_humidity_stack(
                        specific_humidity_path=args.specific_humidity,
                        timestamp=timestamp,
                        pressure_levels_hpa=contents.pressure_levels_hpa[level_indices],
                        latitudes_deg=contents.latitudes_deg,
                        longitudes_deg=contents.longitudes_deg,
                    )
                entries.append(
                    build_timestamp_entry(
                        output_dir=output_dir,
                        raw_dataset=raw_dataset,
                        contents=contents,
                        timestamp=timestamp,
                        time_index=time_index,
                        level_indices=level_indices,
                        field_min=field_min,
                        field_max=field_max,
                        field_kind=args.field_kind,
                        climatology=climatology,
                        specific_humidity_stack=specific_humidity_stack,
                    )
                )
    finally:
        raw_dataset.close()

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        level_indices=level_indices,
        field_min=field_min,
        field_max=field_max,
        field_kind=args.field_kind,
        climatology=climatology,
        variant_name=args.variant_name,
        variant_label=args.variant_label,
        color_scale=args.color_scale,
        saturation_strength_scale=saturation_strength_scale,
        front_detection=front_detection,
        specific_humidity_path=resolve_dataset_path(args.specific_humidity)
        if uses_specific_humidity(args.field_kind)
        else None,
    )
    border_texture = write_border_texture(
        output_dir=output_dir,
        width=int(contents.longitudes_deg.size),
        height=int(contents.latitudes_deg.size),
    )
    if border_texture is not None:
        manifest["border_texture"] = border_texture
    write_json(output_dir / "index.json", manifest)
    print(
        "Built temperature slice assets:",
        f"{len(entries)} timestamps",
        f"{len(level_indices)} pressure levels",
        f"variant={args.variant_name}",
        f"{field_min:.2f}-{field_max:.2f} K",
        f"-> {format_display_path(output_dir)}",
    )


if __name__ == "__main__":
    main()
