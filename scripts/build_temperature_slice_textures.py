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
RAW_TEMPERATURE_FIELD_KIND = "raw-temperature"
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
            TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
            POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
            RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND,
            RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND,
            RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND,
        ),
        help="Scalar field exported into the pressure-slice textures.",
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
        RAW_TEMPERATURE_VERTICAL_COHERENCE_FIELD_KIND,
    }:
        return None

    if field_kind in {
        TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_STRENGTH_FIELD_KIND,
        RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND,
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


def is_signed_saturation_field_kind(field_kind: str) -> bool:
    return field_kind == RAW_TEMPERATURE_ANOMALY_AGREEMENT_FIELD_KIND


def dry_potential_temperature(
    temperature_k: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    return np.asarray(
        temperature_k * (1000.0 / pressure_hpa) ** 0.285906374501992,
        dtype=np.float32,
    )


def build_field(
    raw_temperature_k: np.ndarray,
    pressure_hpa: float,
    level_index: int,
    field_kind: str,
    climatology: ClimatologyField | None,
) -> np.ndarray:
    if field_kind == RAW_TEMPERATURE_FIELD_KIND:
        return raw_temperature_k
    if climatology is None:
        raise ValueError(f"Missing climatology for field kind: {field_kind}")
    if field_kind == TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND:
        return np.asarray(raw_temperature_k - climatology.values[level_index], dtype=np.float32)
    if field_kind == POTENTIAL_TEMPERATURE_CLIMATOLOGY_ANOMALY_FIELD_KIND:
        theta = dry_potential_temperature(raw_temperature_k, pressure_hpa)
        return np.asarray(theta - climatology.values[level_index], dtype=np.float32)
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
) -> tuple[float, float]:
    variable = raw_dataset.variables[DATASET_VARIABLE]
    data_min = np.inf
    data_max = -np.inf

    for time_index in target_time_indices:
        for level_index in level_indices:
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
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    variable = raw_dataset.variables[DATASET_VARIABLE]
    levels: list[dict[str, str | float]] = []

    for level_index in level_indices:
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        raw_temperature = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
        field = build_field(
            raw_temperature,
            pressure_hpa=pressure_hpa,
            level_index=level_index,
            field_kind=field_kind,
            climatology=climatology,
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
                "temperature_min_k": float(np.nanmin(field)),
                "temperature_max_k": float(np.nanmax(field)),
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
) -> dict:
    pressures = sorted(float(contents.pressure_levels_hpa[index]) for index in level_indices)
    manifest = {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "display_units": "K",
        "variant": variant_name,
        "variant_label": variant_label,
        "field_kind": field_kind,
        "climatology_dataset": climatology.dataset_path.name if climatology else None,
        "rendering": {
            "kind": "full-field-pressure-slice",
            "filtering": "none",
            "color_scale": color_scale,
            "encoding": (
                "raw-temperature-uint16-rg-signed-saturation-b"
                if is_signed_saturation_field_kind(field_kind)
                else "raw-temperature-uint16-rg-saturation-strength-b"
                if saturation_strength_scale is not None
                else "normalized-temperature-uint16-packed-rg"
            ),
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
        draw.line(pixels, fill=(255, 255, 255, 220), width=2, joint="curve")


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
        if is_saturation_encoded_field_kind(args.field_kind):
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
            )
            entries = [
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
                )
                for timestamp, time_index in zip(target_timestamps, target_time_indices)
            ]
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
