from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import xarray as xr

from scripts.build_potential_temperature_structures import (
    apply_vertical_connection_mode,
    build_seam_merged_component_info,
    build_climatology_mean_anomaly,
    build_latitude_mean_anomaly,
    compute_dry_potential_temperature,
    label_wrapped_volume_components,
    stride_spatial_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THETA_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_Q_DATASET = Path("data/era5_specific-humidity_2021-11_08-12.nc")
DEFAULT_RH_DATASET = Path("data/era5_relative-humidity_2021-11_08-12.nc")
DEFAULT_THETA_CLIMATOLOGY = Path(
    "data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc"
)
FIELD_REPORT_TOP_SHARES = (5.0, 10.0, 20.0)
FIELD_REPORT_REFERENCE_TOP_SHARE = 10.0
RELATIVE_DEPARTURE_THRESHOLDS_PERCENT = (1.0, 2.0, 5.0, 10.0)
STANDARDIZED_DEPARTURE_THRESHOLDS_SIGMA = (1.0, 2.0, 3.0)
ABS_LATITUDE_BANDS_DEG = ((0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 90.0))
REPRESENTATIVE_PROFILE_PRESSURES = (1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 250.0)
STRUCTURE_COMPONENT_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
SLICE_COMPONENT_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
COARSE_LATITUDE_STRIDE = 4
COARSE_LONGITUDE_STRIDE = 4
QUICK_COMPONENT_TOP_SHARE_PERCENT = 10.0
QUICK_COMPONENT_SWEEP_TOP_SHARES = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0)
QUICK_COMPONENT_MIN_CELLS = 64
QUICK_COMPONENT_MIN_LEVEL_FRACTION = 0.001

FIELD_ALIASES = {
    "q": {
        "canonical_field": "specific_humidity",
        "variable": "q",
        "dataset": DEFAULT_Q_DATASET,
        "units": "kg kg^-1",
    },
    "specific_humidity": {
        "canonical_field": "specific_humidity",
        "variable": "q",
        "dataset": DEFAULT_Q_DATASET,
        "units": "kg kg^-1",
    },
    "r": {
        "canonical_field": "relative_humidity",
        "variable": "r",
        "dataset": DEFAULT_RH_DATASET,
        "units": "%",
    },
    "rh": {
        "canonical_field": "relative_humidity",
        "variable": "r",
        "dataset": DEFAULT_RH_DATASET,
        "units": "%",
    },
    "relative_humidity": {
        "canonical_field": "relative_humidity",
        "variable": "r",
        "dataset": DEFAULT_RH_DATASET,
        "units": "%",
    },
    "t": {
        "canonical_field": "temperature",
        "variable": "t",
        "dataset": DEFAULT_THETA_DATASET,
        "units": "K",
    },
    "temperature": {
        "canonical_field": "temperature",
        "variable": "t",
        "dataset": DEFAULT_THETA_DATASET,
        "units": "K",
    },
    "theta": {
        "canonical_field": "dry_potential_temperature",
        "variable": "t",
        "dataset": DEFAULT_THETA_DATASET,
        "units": "K",
        "derived": "theta",
    },
}


@dataclass(frozen=True)
class FieldCube:
    canonical_field: str
    requested_field: str
    source_variable: str
    dataset_path: Path
    timestamp: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    values: np.ndarray
    units: str
    transform: dict[str, Any]
    notes: list[str]


def format_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        home = Path.home()
        try:
            return f"~/{resolved.relative_to(home).as_posix()}"
        except ValueError:
            tmp_root = Path('/tmp')
            try:
                return f"tmp/{resolved.relative_to(tmp_root).as_posix()}"
            except ValueError:
                return resolved.name or "<external-path>"


def repo_relative_path(path: Path) -> str:
    return format_display_path(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def slugify(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return lowered or "run"


def normalize_longitudes_with_order(
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.mod(np.asarray(longitudes_deg, dtype=np.float64) + 180.0, 360.0) - 180.0
    order = np.argsort(normalized, kind="stable")
    return normalized[order].astype(np.float32), order.astype(np.int64)


def reorder_longitude_axis(values: np.ndarray, longitude_order: np.ndarray) -> np.ndarray:
    return np.take(np.asarray(values, dtype=np.float32), longitude_order, axis=-1)


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    return text[:-1] if text.endswith("Z") else text


def canonicalize_field_name(field: str) -> str:
    key = field.strip().lower().replace(" ", "_")
    if key not in FIELD_ALIASES:
        raise ValueError(
            "Unsupported field alias. Use one of: "
            + ", ".join(sorted(FIELD_ALIASES.keys()))
        )
    return key


def resolve_field_spec(
    field: str,
    dataset_path: Path | None,
    variable_name: str | None,
    derived: str | None,
) -> dict[str, Any]:
    alias = FIELD_ALIASES[canonicalize_field_name(field)].copy()
    if dataset_path is not None:
        alias["dataset"] = dataset_path
    if variable_name is not None:
        alias["variable"] = variable_name
    if derived is not None:
        alias["derived"] = derived
    return alias


def resolve_timestamp(requested: str | None, available: list[str]) -> str:
    if not available:
        raise ValueError("Dataset does not expose any valid_time entries.")
    if requested is None:
        return available[0]
    if requested in available:
        return requested
    raise ValueError(
        f"Timestamp {requested!r} is not available. Use one of: {', '.join(available[:8])}"
    )


def select_pressure_levels(
    values: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    requested_levels_hpa: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not requested_levels_hpa:
        return np.asarray(values, dtype=np.float32), np.asarray(pressure_levels_hpa, dtype=np.float32)

    requested = np.asarray(requested_levels_hpa, dtype=np.float32)
    selected_indices: list[int] = []
    for level in requested:
        matches = np.flatnonzero(np.isclose(pressure_levels_hpa, level, atol=0.5))
        if matches.size == 0:
            raise ValueError(f"Requested pressure level {level:.1f} hPa is not in the dataset.")
        selected_indices.append(int(matches[0]))
    unique_indices = np.asarray(selected_indices, dtype=np.int64)
    return (
        np.asarray(values[unique_indices], dtype=np.float32),
        np.asarray(pressure_levels_hpa[unique_indices], dtype=np.float32),
    )


def compute_horizontal_gradient_magnitude(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite_values = np.nan_to_num(values, nan=0.0)
    gradient_lat = np.gradient(finite_values, axis=1)
    wrapped = np.concatenate(
        [finite_values[..., -1:], finite_values, finite_values[..., :1]],
        axis=2,
    )
    gradient_lon = (wrapped[..., 2:] - wrapped[..., :-2]) * 0.5
    return np.sqrt(np.square(gradient_lat) + np.square(gradient_lon), dtype=np.float32).astype(
        np.float32
    )


def apply_transform_chain(
    *,
    values: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    longitudes_deg: np.ndarray,
    canonical_field: str,
    source_variable: str,
    derived: str | None,
    anomaly: str,
    smoothing: float,
    climatology_path: Path | None,
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    transformed = np.asarray(values, dtype=np.float32)
    units = "unknown"
    notes: list[str] = []
    resolved_derived = (derived or "").strip().lower() or None
    resolved_anomaly = anomaly.strip().lower()

    if canonical_field in {"temperature", "dry_potential_temperature"}:
        units = "K"
    elif canonical_field == "specific_humidity":
        units = "kg kg^-1"
    elif canonical_field == "relative_humidity":
        units = "%"

    if canonical_field == "dry_potential_temperature" and resolved_derived is None:
        resolved_derived = "theta"

    if resolved_derived == "theta":
        if source_variable != "t":
            raise ValueError("Derived theta requires a temperature source variable.")
        transformed = compute_dry_potential_temperature(transformed, pressure_levels_hpa)
        canonical_field = "dry_potential_temperature"
        units = "K"
        notes.append("Derived dry potential temperature from pressure-level temperature.")
    elif resolved_derived == "gradient":
        transformed = compute_horizontal_gradient_magnitude(transformed)
        units = f"{units} per grid-cell"
        notes.append("Derived horizontal gradient magnitude before anomaly handling.")
    elif resolved_derived not in {None, "", "none"}:
        raise ValueError("Unsupported derived transform. Use one of: theta, gradient.")

    if resolved_anomaly == "none":
        pass
    elif resolved_anomaly == "lat_mean":
        transformed, _ = build_latitude_mean_anomaly(transformed)
        units = units
        notes.append("Subtracted the per-level latitude-band mean.")
    elif resolved_anomaly == "climatology":
        if canonical_field != "dry_potential_temperature":
            raise ValueError(
                "Climatology anomaly is currently only supported for dry potential temperature."
            )
        climatology_dataset = xr.open_dataset((climatology_path or DEFAULT_THETA_CLIMATOLOGY).resolve())
        try:
            full_theta_climatology = np.asarray(
                climatology_dataset["theta_climatology_mean"].values,
                dtype=np.float32,
            )
            climatology_pressures = np.asarray(
                climatology_dataset.coords["pressure_level"].values,
                dtype=np.float32,
            )
            climatology_latitudes = np.asarray(
                climatology_dataset.coords["latitude"].values,
                dtype=np.float32,
            )
            climatology_longitudes, climatology_order = normalize_longitudes_with_order(
                np.asarray(climatology_dataset.coords["longitude"].values, dtype=np.float32)
            )
            theta_climatology = reorder_longitude_axis(full_theta_climatology, climatology_order)
        finally:
            climatology_dataset.close()

        theta_climatology, matched_climatology_pressures = select_pressure_levels(
            theta_climatology,
            climatology_pressures,
            list(np.asarray(pressure_levels_hpa, dtype=np.float32)),
        )
        if not np.allclose(matched_climatology_pressures, pressure_levels_hpa):
            raise ValueError("Climatology pressure levels do not match the selected field.")
        if climatology_latitudes.shape[0] != transformed.shape[1]:
            raise ValueError("Climatology latitudes do not match the selected field.")
        if climatology_longitudes.shape != longitudes_deg.shape or not np.allclose(
            climatology_longitudes,
            longitudes_deg,
        ):
            raise ValueError("Climatology longitudes do not match the selected field.")
        transformed = build_climatology_mean_anomaly(transformed, theta_climatology)
        notes.append("Subtracted the matched gridpoint dry-theta climatology mean.")
    else:
        raise ValueError("Unsupported anomaly transform. Use one of: none, lat_mean, climatology.")

    sigma = max(float(smoothing), 0.0)
    if sigma > 0.0:
        transformed = ndimage.gaussian_filter(
            np.nan_to_num(transformed, nan=0.0).astype(np.float32),
            sigma=(0.0, sigma, sigma),
            mode=("nearest", "nearest", "wrap"),
            truncate=2.0,
        ).astype(np.float32)
        notes.append(f"Applied horizontal Gaussian smoothing with sigma={sigma:.2f} cells.")

    return transformed, units, notes, {
        "derived": resolved_derived or "none",
        "anomaly": resolved_anomaly,
        "smoothing": sigma,
        "climatology_path": (
            format_display_path((climatology_path or DEFAULT_THETA_CLIMATOLOGY).resolve())
            if resolved_anomaly == "climatology"
            else None
        ),
    }


def load_field_cube(
    *,
    field: str,
    dataset_path: Path | None,
    variable_name: str | None,
    timestamp: str | None,
    pressure_levels_hpa: list[float] | None,
    anomaly: str,
    smoothing: float,
    derived: str | None,
    climatology_path: Path | None,
    latitude_stride: int,
    longitude_stride: int,
) -> FieldCube:
    spec = resolve_field_spec(field, dataset_path, variable_name, derived)
    resolved_dataset_path = Path(spec["dataset"]).expanduser().resolve()
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {format_display_path(resolved_dataset_path)}")

    dataset = xr.open_dataset(resolved_dataset_path)
    try:
        variable = dataset[str(spec["variable"])]
        available_timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(variable.coords["valid_time"].values)
        ]
        resolved_timestamp = resolve_timestamp(timestamp, available_timestamps)
        timestamp_index = available_timestamps.index(resolved_timestamp)

        raw_pressure_levels = np.asarray(
            variable.coords["pressure_level"].values,
            dtype=np.float32,
        )
        raw_latitudes = np.asarray(variable.coords["latitude"].values, dtype=np.float32)
        normalized_longitudes, longitude_order = normalize_longitudes_with_order(
            np.asarray(variable.coords["longitude"].values, dtype=np.float32)
        )
        raw_values = reorder_longitude_axis(
            np.asarray(variable.isel(valid_time=timestamp_index).values, dtype=np.float32),
            longitude_order,
        )
        selected_values, selected_pressure_levels = select_pressure_levels(
            raw_values,
            raw_pressure_levels,
            pressure_levels_hpa,
        )
        transformed_values, units, transform_notes, transform_metadata = apply_transform_chain(
            values=selected_values,
            pressure_levels_hpa=selected_pressure_levels,
            longitudes_deg=normalized_longitudes,
            canonical_field=str(spec["canonical_field"]),
            source_variable=str(spec["variable"]),
            derived=spec.get("derived"),
            anomaly=anomaly,
            smoothing=smoothing,
            climatology_path=climatology_path.resolve() if climatology_path else None,
        )
        strided_values, strided_latitudes, strided_longitudes = stride_spatial_axes(
            transformed_values,
            raw_latitudes,
            normalized_longitudes,
            latitude_stride=latitude_stride,
            longitude_stride=longitude_stride,
        )
    finally:
        dataset.close()

    notes = [
        f"Loaded {spec['canonical_field']} from {format_display_path(resolved_dataset_path)}.",
        f"Selected timestamp {resolved_timestamp}.",
        f"Applied spatial stride lat={max(int(latitude_stride), 1)} lon={max(int(longitude_stride), 1)}.",
    ]
    notes.extend(transform_notes)
    return FieldCube(
        canonical_field=str(spec["canonical_field"]),
        requested_field=field,
        source_variable=str(spec["variable"]),
        dataset_path=resolved_dataset_path,
        timestamp=resolved_timestamp,
        pressure_levels_hpa=np.asarray(selected_pressure_levels, dtype=np.float32),
        latitudes_deg=np.asarray(strided_latitudes, dtype=np.float32),
        longitudes_deg=np.asarray(strided_longitudes, dtype=np.float32),
        values=np.asarray(strided_values, dtype=np.float32),
        units=units,
        transform={
            **transform_metadata,
            "latitude_stride": max(int(latitude_stride), 1),
            "longitude_stride": max(int(longitude_stride), 1),
        },
        notes=notes,
    )


def area_weights_for_latitudes(latitudes_deg: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(np.deg2rad(np.asarray(latitudes_deg, dtype=np.float32))), 1e-6, None)


def detect_sign_mode(values: np.ndarray) -> str:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite_values.size == 0:
        return "empty"
    positive_fraction = float(np.mean(finite_values > 0.0))
    negative_fraction = float(np.mean(finite_values < 0.0))
    if positive_fraction > 0.01 and negative_fraction > 0.01:
        return "mixed"
    if positive_fraction >= negative_fraction:
        return "nonnegative"
    return "nonpositive"


def mass_basis(values: np.ndarray) -> tuple[np.ndarray, str]:
    sign_mode = detect_sign_mode(values)
    field = np.asarray(values, dtype=np.float32)
    if sign_mode == "mixed":
        return np.abs(field).astype(np.float32), "absolute_magnitude"
    if sign_mode == "nonpositive":
        return np.abs(field).astype(np.float32), "absolute_negative_magnitude"
    return np.maximum(field, 0.0).astype(np.float32), "raw_positive_magnitude"


def signal_basis(values: np.ndarray) -> tuple[np.ndarray, str]:
    sign_mode = detect_sign_mode(values)
    field = np.asarray(values, dtype=np.float32)
    if sign_mode == "mixed":
        return np.abs(field).astype(np.float32), "absolute_value"
    if sign_mode == "nonpositive":
        return np.abs(field).astype(np.float32), "absolute_negative_value"
    return np.maximum(field, 0.0).astype(np.float32), "positive_value"


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float | None, float | None]:
    broadcast_weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), np.asarray(values).shape)
    finite = np.isfinite(values) & np.isfinite(broadcast_weights) & (broadcast_weights > 0.0)
    if not np.any(finite):
        return None, None
    weighted_values = np.asarray(values[finite], dtype=np.float64)
    weighted_weights = np.asarray(broadcast_weights[finite], dtype=np.float64)
    total_weight = float(weighted_weights.sum())
    if total_weight <= 0.0:
        return None, None
    mean = float(np.sum(weighted_values * weighted_weights) / total_weight)
    variance = float(
        np.sum(weighted_weights * np.square(weighted_values - mean, dtype=np.float64)) / total_weight
    )
    return mean, math.sqrt(max(variance, 0.0))


def safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite_values.size == 0:
        return None
    return float(np.percentile(finite_values, percentile))


def estimate_mode(values: np.ndarray) -> float | None:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite_values.size == 0:
        return None
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if abs(maximum - minimum) <= 1e-12:
        return minimum

    bin_count = max(24, min(80, int(round(math.sqrt(finite_values.size)))))
    counts, edges = np.histogram(finite_values, bins=bin_count)
    if counts.size == 0:
        return None
    index = int(np.argmax(counts))
    return float(0.5 * (edges[index] + edges[index + 1]))


def summarize_profile_direction(
    level_means: np.ndarray,
    pressure_levels_hpa: np.ndarray,
) -> tuple[str, float | None]:
    finite = np.isfinite(level_means)
    if np.count_nonzero(finite) < 2:
        return "mixed", None

    ordered_means = np.asarray(level_means[finite], dtype=np.float64)
    ordered_pressures = np.asarray(pressure_levels_hpa[finite], dtype=np.float64)
    if ordered_pressures[0] < ordered_pressures[-1]:
        ordered_means = ordered_means[::-1]

    diffs = np.diff(ordered_means)
    nonzero_diffs = diffs[np.abs(diffs) > 1e-9]
    if nonzero_diffs.size == 0:
        return "flat", 1.0

    overall = np.sign(ordered_means[-1] - ordered_means[0])
    if overall > 0.0:
        direction = "increases_with_height"
    elif overall < 0.0:
        direction = "decreases_with_height"
    else:
        direction = "mixed"

    if overall == 0.0:
        monotonicity = float(np.mean(np.abs(nonzero_diffs) > 0.0))
    else:
        monotonicity = float(np.mean(np.sign(nonzero_diffs) == overall))
    return direction, monotonicity


def structure_quantity_terms(cube: FieldCube) -> dict[str, str]:
    anomaly = str(cube.transform.get("anomaly") or "none").strip().lower()
    if anomaly == "climatology":
        return {
            "share_noun": "departure magnitude from climatology",
            "share_noun_short": "departure magnitude",
            "dominant_phrase": "largest departures from climatology",
            "chart_label": "departure magnitude",
        }
    if anomaly != "none":
        return {
            "share_noun": "anomaly magnitude",
            "share_noun_short": "anomaly magnitude",
            "dominant_phrase": "largest anomalies",
            "chart_label": "anomaly magnitude",
        }
    return {
        "share_noun": "signal",
        "share_noun_short": "signal",
        "dominant_phrase": "largest values",
        "chart_label": "signal",
    }


def choose_representative_level_indices(pressure_levels_hpa: np.ndarray, limit: int) -> list[int]:
    available = np.asarray(pressure_levels_hpa, dtype=np.float32)
    chosen: list[int] = []
    for target in REPRESENTATIVE_PROFILE_PRESSURES:
        index = int(np.argmin(np.abs(available - target)))
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= limit:
            break
    if not chosen:
        chosen = list(range(min(limit, available.size)))
    return chosen


def compute_spatial_entropy(column_mass: np.ndarray) -> float | None:
    flat = np.asarray(column_mass, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat) & (flat > 0.0)]
    if flat.size <= 1:
        return None
    probabilities = flat / flat.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(probabilities.size))


def compute_normalized_entropy(weights: np.ndarray) -> float | None:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size <= 1:
        return None
    probabilities = values / values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(probabilities.size))


def compute_abs_latitude_thresholds(
    latitudes_deg: np.ndarray,
    zonal_signal_fraction: np.ndarray,
    *,
    targets: tuple[float, ...] = (0.5, 0.8),
) -> dict[str, float | None]:
    latitudes = np.asarray(latitudes_deg, dtype=np.float64)
    fractions = np.asarray(zonal_signal_fraction, dtype=np.float64)
    abs_latitudes = np.abs(latitudes)
    order = np.argsort(abs_latitudes, kind="stable")
    cumulative = np.cumsum(fractions[order])
    ordered_abs_latitudes = abs_latitudes[order]
    results: dict[str, float | None] = {}
    for target in targets:
        indices = np.flatnonzero(cumulative >= target)
        key = f"{int(round(target * 100.0))}_percent"
        results[key] = float(ordered_abs_latitudes[int(indices[0])]) if indices.size else None
    return results


def compute_abs_latitude_band_shares(
    latitudes_deg: np.ndarray,
    zonal_signal_fraction: np.ndarray,
    area_weights_by_lat: np.ndarray,
    *,
    bands_deg: tuple[tuple[float, float], ...] = ABS_LATITUDE_BANDS_DEG,
) -> list[dict[str, float]]:
    latitudes = np.asarray(latitudes_deg, dtype=np.float64)
    signal_fraction = np.asarray(zonal_signal_fraction, dtype=np.float64)
    area_weights = np.asarray(area_weights_by_lat, dtype=np.float64)
    abs_latitudes = np.abs(latitudes)
    total_area = float(area_weights.sum())
    rows: list[dict[str, float]] = []
    for lower, upper in bands_deg:
        if upper >= 90.0:
            mask = (abs_latitudes >= lower) & (abs_latitudes <= upper)
        else:
            mask = (abs_latitudes >= lower) & (abs_latitudes < upper)
        rows.append(
            {
                "lower_deg": float(lower),
                "upper_deg": float(upper),
                "signal_fraction": float(signal_fraction[mask].sum()),
                "area_fraction": float(area_weights[mask].sum() / max(total_area, 1e-12)),
            }
        )
    return rows


def load_climatology_reference_for_cube(cube: FieldCube) -> dict[str, np.ndarray] | None:
    anomaly = str(cube.transform.get("anomaly") or "none").strip().lower()
    if anomaly != "climatology" or cube.canonical_field != "dry_potential_temperature":
        return None

    climatology_path = Path(
        str(cube.transform.get("climatology_path") or format_display_path(DEFAULT_THETA_CLIMATOLOGY))
    ).expanduser()
    if not climatology_path.is_absolute():
        climatology_path = (REPO_ROOT / climatology_path).resolve()

    dataset = xr.open_dataset(climatology_path)
    try:
        longitudes_deg, longitude_order = normalize_longitudes_with_order(
            np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
        )
        theta_mean = reorder_longitude_axis(
            np.asarray(dataset["theta_climatology_mean"].values, dtype=np.float32),
            longitude_order,
        )
        theta_std = reorder_longitude_axis(
            np.asarray(dataset["theta_climatology_std"].values, dtype=np.float32),
            longitude_order,
        )
        theta_count = reorder_longitude_axis(
            np.asarray(dataset["theta_sample_count"].values, dtype=np.float32),
            longitude_order,
        )
        pressures = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        latitudes_deg = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
    finally:
        dataset.close()

    selected_levels = [float(value) for value in np.asarray(cube.pressure_levels_hpa, dtype=np.float32)]
    theta_mean, matched_pressures = select_pressure_levels(theta_mean, pressures, selected_levels)
    theta_std, _ = select_pressure_levels(theta_std, pressures, selected_levels)
    theta_count, _ = select_pressure_levels(theta_count, pressures, selected_levels)
    if not np.allclose(matched_pressures, cube.pressure_levels_hpa):
        raise ValueError("Climatology pressure levels do not match the selected field cube.")

    latitude_stride = max(int(cube.transform.get("latitude_stride") or 1), 1)
    longitude_stride = max(int(cube.transform.get("longitude_stride") or 1), 1)
    theta_mean, strided_latitudes, strided_longitudes = stride_spatial_axes(
        theta_mean,
        latitudes_deg,
        longitudes_deg,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )
    theta_std, _, _ = stride_spatial_axes(
        theta_std,
        latitudes_deg,
        longitudes_deg,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )
    theta_count, _, _ = stride_spatial_axes(
        theta_count,
        latitudes_deg,
        longitudes_deg,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )

    if theta_mean.shape != cube.values.shape:
        raise ValueError("Climatology reference does not match the transformed cube shape.")
    if not np.allclose(strided_latitudes, cube.latitudes_deg):
        raise ValueError("Climatology latitudes do not match the transformed cube.")
    if not np.allclose(strided_longitudes, cube.longitudes_deg):
        raise ValueError("Climatology longitudes do not match the transformed cube.")

    return {
        "mean": np.asarray(theta_mean, dtype=np.float32),
        "std": np.asarray(theta_std, dtype=np.float32),
        "sample_count": np.asarray(theta_count, dtype=np.float32),
    }


def label_wrapped_slice_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    longitude_count = occupied.shape[1]
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(extended, structure=SLICE_COMPONENT_STRUCTURE)
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


def summarize_slice_component_counts(
    selected_mask: np.ndarray,
    level_values: np.ndarray,
    valid_cell_count: int,
    minimum_big_component_cells: int,
) -> dict[str, float | int]:
    def count_components(mask: np.ndarray) -> tuple[int, int, float]:
        labels, component_count = label_wrapped_slice_components(mask)
        component_sizes = (
            np.bincount(labels[labels > 0].ravel(), minlength=component_count + 1)[1:]
            if component_count > 0
            else np.zeros(0, dtype=np.int32)
        )
        big_component_sizes = component_sizes[component_sizes >= minimum_big_component_cells]
        largest_component_fraction = (
            float(component_sizes.max() / max(valid_cell_count, 1))
            if component_sizes.size
            else 0.0
        )
        return int(component_count), int(big_component_sizes.size), largest_component_fraction

    component_count, big_component_count, largest_component_fraction = count_components(selected_mask)
    warm_component_count, warm_big_component_count, _ = count_components(selected_mask & (level_values > 0.0))
    cold_component_count, cold_big_component_count, _ = count_components(selected_mask & (level_values < 0.0))
    return {
        "component_count": component_count,
        "big_component_count": big_component_count,
        "warm_component_count": warm_component_count,
        "warm_big_component_count": warm_big_component_count,
        "cold_component_count": cold_component_count,
        "cold_big_component_count": cold_big_component_count,
        "largest_component_fraction": largest_component_fraction,
    }


def build_quick_top_share_component_metrics(
    cube: FieldCube,
    anomaly_values: np.ndarray,
    finite_mask: np.ndarray,
) -> dict[str, Any]:
    per_level: list[dict[str, Any]] = []

    for level_index, pressure_hpa in enumerate(cube.pressure_levels_hpa):
        level_values = np.asarray(anomaly_values[level_index], dtype=np.float32)
        level_valid_mask = finite_mask[level_index] & np.isfinite(level_values)
        valid_cell_count = int(np.count_nonzero(level_valid_mask))
        level_mask, threshold_value = compute_top_share_mask(
            np.where(level_valid_mask, level_values, np.nan),
            QUICK_COMPONENT_TOP_SHARE_PERCENT,
            tail="absolute",
        )
        level_mask &= level_valid_mask
        exceed_cell_count = int(np.count_nonzero(level_mask))

        cleaned_mask = np.asarray(level_mask, dtype=bool)

        minimum_big_component_cells = max(
            QUICK_COMPONENT_MIN_CELLS,
            int(math.ceil(valid_cell_count * QUICK_COMPONENT_MIN_LEVEL_FRACTION)),
        )
        counts = summarize_slice_component_counts(
            cleaned_mask,
            level_values,
            valid_cell_count,
            minimum_big_component_cells,
        )

        per_level.append(
            {
                "pressure_hpa": float(pressure_hpa),
                "valid_cell_count": valid_cell_count,
                "threshold_value": float(threshold_value) if threshold_value is not None else None,
                "exceed_cell_count": exceed_cell_count,
                "cleaned_cell_count": int(np.count_nonzero(cleaned_mask)),
                **counts,
                "minimum_big_component_cells": int(minimum_big_component_cells),
            }
        )

    big_count_levels = [row for row in per_level if row["big_component_count"] > 0]
    ranked_by_big_count = sorted(
        per_level,
        key=lambda row: (row["big_component_count"], row["component_count"], row["cleaned_cell_count"]),
        reverse=True,
    )

    return {
        "top_share_percent": QUICK_COMPONENT_TOP_SHARE_PERCENT,
        "minimum_big_component_cells_floor": QUICK_COMPONENT_MIN_CELLS,
        "minimum_big_component_fraction_of_level": QUICK_COMPONENT_MIN_LEVEL_FRACTION,
        "per_level": per_level,
        "levels_with_any_big_components": int(len(big_count_levels)),
        "max_big_component_count": int(max((row["big_component_count"] for row in per_level), default=0)),
        "peak_big_component_levels": [
            {
                "pressure_hpa": float(row["pressure_hpa"]),
                "big_component_count": int(row["big_component_count"]),
                "component_count": int(row["component_count"]),
            }
            for row in ranked_by_big_count[:3]
            if row["big_component_count"] > 0
        ],
    }


def build_quick_top_share_sweep_metrics(
    cube: FieldCube,
    anomaly_values: np.ndarray,
    finite_mask: np.ndarray,
) -> dict[str, Any]:
    per_level: list[dict[str, Any]] = []

    for level_index, pressure_hpa in enumerate(cube.pressure_levels_hpa):
        level_values = np.asarray(anomaly_values[level_index], dtype=np.float32)
        level_valid_mask = finite_mask[level_index] & np.isfinite(level_values)
        valid_cell_count = int(np.count_nonzero(level_valid_mask))
        minimum_big_component_cells = max(
            QUICK_COMPONENT_MIN_CELLS,
            int(math.ceil(valid_cell_count * QUICK_COMPONENT_MIN_LEVEL_FRACTION)),
        )
        share_rows: list[dict[str, Any]] = []

        for top_share_percent in QUICK_COMPONENT_SWEEP_TOP_SHARES:
            level_mask, threshold_value = compute_top_share_mask(
                np.where(level_valid_mask, level_values, np.nan),
                top_share_percent,
                tail="absolute",
            )
            level_mask &= level_valid_mask
            share_rows.append(
                {
                    "top_share_percent": float(top_share_percent),
                    "threshold_value": float(threshold_value) if threshold_value is not None else None,
                    "selected_cell_count": int(np.count_nonzero(level_mask)),
                    **summarize_slice_component_counts(
                        level_mask,
                        level_values,
                        valid_cell_count,
                        minimum_big_component_cells,
                    ),
                }
            )

        per_level.append(
            {
                "pressure_hpa": float(pressure_hpa),
                "valid_cell_count": valid_cell_count,
                "minimum_big_component_cells": int(minimum_big_component_cells),
                "shares": share_rows,
            }
        )

    return {
        "top_share_percents": [float(value) for value in QUICK_COMPONENT_SWEEP_TOP_SHARES],
        "minimum_big_component_cells_floor": QUICK_COMPONENT_MIN_CELLS,
        "minimum_big_component_fraction_of_level": QUICK_COMPONENT_MIN_LEVEL_FRACTION,
        "per_level": per_level,
    }


def build_climatology_anomaly_metrics(cube: FieldCube) -> dict[str, Any] | None:
    reference = load_climatology_reference_for_cube(cube)
    if reference is None:
        return None

    values = np.asarray(cube.values, dtype=np.float32)
    abs_values = np.abs(values)
    finite_mask = np.isfinite(values)
    area_weights_2d = area_weights_for_latitudes(cube.latitudes_deg)[:, None].astype(np.float64)
    area_weights_3d = np.broadcast_to(area_weights_2d[None, :, :], values.shape)

    climatology_mean = np.asarray(reference["mean"], dtype=np.float32)
    climatology_std = np.asarray(reference["std"], dtype=np.float32)
    safe_mean = np.where(np.abs(climatology_mean) > 1e-6, np.abs(climatology_mean), np.nan)
    safe_std = np.where(climatology_std > 1e-6, climatology_std, np.nan)

    relative_departure_percent = 100.0 * np.divide(
        abs_values,
        safe_mean,
        out=np.full_like(abs_values, np.nan, dtype=np.float32),
        where=np.isfinite(safe_mean),
    )
    standardized_departure = np.divide(
        abs_values,
        safe_std,
        out=np.full_like(abs_values, np.nan, dtype=np.float32),
        where=np.isfinite(safe_std),
    )

    def build_threshold_summary(
        scaled_values: np.ndarray,
        thresholds: tuple[float, ...],
        unit_label: str,
    ) -> dict[str, Any]:
        valid_mask = finite_mask & np.isfinite(scaled_values)
        valid_weights = np.where(valid_mask, area_weights_3d, 0.0).astype(np.float64)
        total_weight = float(valid_weights.sum())
        total_voxels = int(np.count_nonzero(valid_mask))
        overall_area_fraction: list[float] = []
        overall_voxel_fraction: list[float] = []
        per_level: list[dict[str, Any]] = []

        for threshold in thresholds:
            exceed = valid_mask & (scaled_values >= threshold)
            overall_area_fraction.append(
                float(np.where(exceed, area_weights_3d, 0.0).sum() / max(total_weight, 1e-12))
            )
            overall_voxel_fraction.append(float(np.count_nonzero(exceed) / max(total_voxels, 1)))

        for level_index, pressure in enumerate(cube.pressure_levels_hpa):
            level_valid = valid_mask[level_index]
            level_total_voxels = int(np.count_nonzero(level_valid))
            level_weights = np.where(level_valid, area_weights_2d, 0.0).astype(np.float64)
            level_total_weight = float(level_weights.sum())
            area_fraction_by_threshold: list[float] = []
            voxel_fraction_by_threshold: list[float] = []
            for threshold in thresholds:
                exceed = level_valid & (scaled_values[level_index] >= threshold)
                area_fraction_by_threshold.append(
                    float(np.where(exceed, area_weights_2d, 0.0).sum() / max(level_total_weight, 1e-12))
                )
                voxel_fraction_by_threshold.append(
                    float(np.count_nonzero(exceed) / max(level_total_voxels, 1))
                )
            per_level.append(
                {
                    "pressure_hpa": float(pressure),
                    "area_fraction_by_threshold": area_fraction_by_threshold,
                    "voxel_fraction_by_threshold": voxel_fraction_by_threshold,
                }
            )

        return {
            "unit_label": unit_label,
            "thresholds": [float(value) for value in thresholds],
            "overall_area_fraction": overall_area_fraction,
            "overall_voxel_fraction": overall_voxel_fraction,
            "per_level": per_level,
        }

    return {
        "reference_sample_count_range": {
            "min": float(np.nanmin(reference["sample_count"])),
            "max": float(np.nanmax(reference["sample_count"])),
        },
        "relative_departure_percent": build_threshold_summary(
            relative_departure_percent,
            RELATIVE_DEPARTURE_THRESHOLDS_PERCENT,
            "%",
        ),
        "standardized_departure": build_threshold_summary(
            standardized_departure,
            STANDARDIZED_DEPARTURE_THRESHOLDS_SIGMA,
            "sigma",
        ),
        "quick_top_share_components": build_quick_top_share_component_metrics(
            cube,
            values,
            finite_mask,
        ),
        "quick_top_share_sweep": build_quick_top_share_sweep_metrics(
            cube,
            values,
            finite_mask,
        ),
    }


def cumulative_abs_latitude_curve(
    latitudes_deg: np.ndarray,
    zonal_signal_fraction: np.ndarray,
    area_weights_by_lat: np.ndarray,
) -> dict[str, np.ndarray]:
    latitudes = np.asarray(latitudes_deg, dtype=np.float64)
    signal_fraction = np.asarray(zonal_signal_fraction, dtype=np.float64)
    area_weights = np.asarray(area_weights_by_lat, dtype=np.float64)
    unique_abs_latitudes = np.unique(np.abs(latitudes))
    cumulative_signal = []
    cumulative_area = []
    total_area = float(np.sum(area_weights))
    for abs_latitude in unique_abs_latitudes:
        mask = np.abs(latitudes) <= abs_latitude + 1e-9
        cumulative_signal.append(float(np.sum(signal_fraction[mask])))
        cumulative_area.append(float(np.sum(area_weights[mask]) / max(total_area, 1e-12)))
    return {
        "abs_latitude_deg": unique_abs_latitudes.astype(np.float32),
        "cumulative_signal_fraction": np.asarray(cumulative_signal, dtype=np.float32),
        "cumulative_area_fraction": np.asarray(cumulative_area, dtype=np.float32),
    }


def sample_column_concentration_curve(column_signal: np.ndarray, sample_count: int = 101) -> dict[str, np.ndarray]:
    flat = np.sort(np.asarray(column_signal, dtype=np.float64).ravel())[::-1]
    if flat.size == 0:
        x = np.linspace(0.0, 1.0, num=sample_count, dtype=np.float32)
        return {
            "column_fraction": x,
            "cumulative_signal_fraction": np.zeros_like(x),
        }
    cumulative = np.cumsum(flat)
    cumulative /= max(cumulative[-1], 1e-12)
    x_full = np.arange(1, flat.size + 1, dtype=np.float64) / float(flat.size)
    x_sampled = np.linspace(0.0, 1.0, num=sample_count, dtype=np.float64)
    y_sampled = np.interp(x_sampled, np.concatenate([[0.0], x_full]), np.concatenate([[0.0], cumulative]))
    return {
        "column_fraction": x_sampled.astype(np.float32),
        "cumulative_signal_fraction": y_sampled.astype(np.float32),
    }


def cumulative_pressure_thresholds(
    pressure_levels_hpa: np.ndarray,
    level_signal_fraction: np.ndarray,
    *,
    direction: str,
    targets: tuple[float, ...] = (0.5, 0.75),
) -> dict[str, float | None]:
    pressures = np.asarray(pressure_levels_hpa, dtype=np.float64)
    fractions = np.asarray(level_signal_fraction, dtype=np.float64)
    if direction == "surface_down":
        ordered_pressures = pressures
        ordered_fractions = fractions
    elif direction == "top_down":
        ordered_pressures = pressures[::-1]
        ordered_fractions = fractions[::-1]
    else:
        raise ValueError("Unsupported direction. Use one of: surface_down, top_down.")

    cumulative = np.cumsum(ordered_fractions)
    results: dict[str, float | None] = {}
    for target in targets:
        indices = np.flatnonzero(cumulative >= target)
        key = f"{int(round(target * 100.0))}_percent"
        results[key] = float(ordered_pressures[int(indices[0])]) if indices.size else None
    return results


def narrowest_pressure_band(
    pressure_levels_hpa: np.ndarray,
    level_signal_fraction: np.ndarray,
    *,
    target_fraction: float,
) -> dict[str, float | int | None]:
    pressures = np.asarray(pressure_levels_hpa, dtype=np.float64)
    fractions = np.asarray(level_signal_fraction, dtype=np.float64)
    if pressures.size == 0:
        return {
            "target_fraction": float(target_fraction),
            "highest_pressure_hpa": None,
            "lowest_pressure_hpa": None,
            "level_count": 0,
            "contained_signal_fraction": 0.0,
        }

    best: tuple[int, float, int, int, float] | None = None
    start = 0
    running = 0.0
    for end in range(fractions.size):
        running += float(fractions[end])
        while start <= end and running - float(fractions[start]) >= target_fraction:
            running -= float(fractions[start])
            start += 1
        if running >= target_fraction:
            level_count = end - start + 1
            pressure_span = abs(float(pressures[start] - pressures[end]))
            candidate = (level_count, pressure_span, start, end, running)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

    if best is None:
        return {
            "target_fraction": float(target_fraction),
            "highest_pressure_hpa": float(pressures[0]),
            "lowest_pressure_hpa": float(pressures[-1]),
            "level_count": int(pressures.size),
            "contained_signal_fraction": float(np.sum(fractions)),
        }

    _, _, best_start, best_end, contained_signal = best
    return {
        "target_fraction": float(target_fraction),
        "highest_pressure_hpa": float(pressures[best_start]),
        "lowest_pressure_hpa": float(pressures[best_end]),
        "level_count": int(best_end - best_start + 1),
        "contained_signal_fraction": float(contained_signal),
    }


def compute_top_share_mask(
    values: np.ndarray,
    top_share_percent: float,
    tail: str,
) -> tuple[np.ndarray, float | None]:
    normalized_top_share = float(top_share_percent)
    finite_mask = np.isfinite(values)
    if normalized_top_share <= 0.0 or not np.any(finite_mask):
        return np.zeros_like(values, dtype=bool), None

    tail_lower = tail.lower()
    finite_values = np.asarray(values[finite_mask], dtype=np.float32)
    percentile = max(0.0, min(100.0, 100.0 - normalized_top_share))
    if tail_lower == "absolute":
        metric = np.abs(finite_values)
        threshold = float(np.percentile(metric, percentile))
        mask = finite_mask & (np.abs(values) >= threshold)
    elif tail_lower == "high":
        threshold = float(np.percentile(finite_values, percentile))
        mask = finite_mask & (values >= threshold)
    elif tail_lower == "positive":
        positive_values = finite_values[finite_values > 0.0]
        if positive_values.size == 0:
            return np.zeros_like(values, dtype=bool), None
        threshold = float(np.percentile(positive_values, percentile))
        mask = finite_mask & (values > 0.0) & (values >= threshold)
    elif tail_lower == "negative":
        negative_values = finite_values[finite_values < 0.0]
        if negative_values.size == 0:
            return np.zeros_like(values, dtype=bool), None
        threshold = float(np.percentile(negative_values, normalized_top_share))
        mask = finite_mask & (values < 0.0) & (values <= threshold)
    else:
        raise ValueError("Unsupported tail mode. Use one of: absolute, high, positive, negative.")
    return np.asarray(mask, dtype=bool), threshold


def find_runs_1d(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def compute_sign_flip_rate(values: np.ndarray) -> float | None:
    sign_mode = detect_sign_mode(values)
    if sign_mode != "mixed":
        return None

    flip_rates: list[float] = []
    _, lat_count, lon_count = values.shape
    for lat_index in range(lat_count):
        for lon_index in range(lon_count):
            column = np.asarray(values[:, lat_index, lon_index], dtype=np.float32)
            finite = np.isfinite(column) & (np.abs(column) > 0.0)
            if np.count_nonzero(finite) < 2:
                continue
            signs = np.sign(column[finite]).astype(np.int8)
            flips = np.count_nonzero(signs[1:] != signs[:-1])
            possible = max(signs.size - 1, 1)
            flip_rates.append(flips / possible)
    if not flip_rates:
        return None
    return float(np.mean(flip_rates))


def compute_vertical_coherence(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    lat_count, lon_count = mask.shape[1], mask.shape[2]
    run_lengths: list[int] = []
    coherent_columns = 0
    occupied_columns = 0
    single_level_columns = 0
    for lat_index in range(lat_count):
        for lon_index in range(lon_count):
            runs = find_runs_1d(mask[:, lat_index, lon_index])
            if not runs:
                continue
            occupied_columns += 1
            run_lengths.extend(runs)
            if max(runs) >= 3:
                coherent_columns += 1
            if max(runs) == 1:
                single_level_columns += 1
    return {
        "mean_vertical_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "median_vertical_run_length": float(np.median(run_lengths)) if run_lengths else 0.0,
        "coherent_column_fraction": (
            coherent_columns / max(mask.shape[1] * mask.shape[2], 1)
        ),
        "occupied_column_fraction": occupied_columns / max(mask.shape[1] * mask.shape[2], 1),
        "single_level_column_fraction": (
            single_level_columns / max(occupied_columns, 1)
        ),
    }


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.0f}%"


def summarize_top_components(component_sizes: np.ndarray, total_voxels: int, limit: int = 5) -> list[str]:
    if total_voxels <= 0 or component_sizes.size == 0:
        return []
    ordered = np.sort(np.asarray(component_sizes, dtype=np.int64))[::-1]
    summary = []
    for size in ordered[:limit]:
        summary.append(f"{100.0 * float(size) / float(total_voxels):.0f}%")
    return summary


def update_log_index(
    *,
    skill_root: Path,
    run_id: str,
    summary_title: str,
    field_label: str,
    timestamp: str,
    decision: str,
    report_filename: str,
    json_filename: str,
    headline: str,
    image_filenames: list[str],
) -> None:
    logs_dir = skill_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    index_path = logs_dir / "index.json"
    if index_path.exists():
        entries = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        entries = []

    entries = [
        entry
        for entry in entries
        if not (entry.get("run_id") == run_id and entry.get("skill") == skill_root.name)
    ]
    entries.insert(
        0,
        {
            "skill": skill_root.name,
            "run_id": run_id,
            "summary_title": summary_title,
            "field": field_label,
            "timestamp": timestamp,
            "decision": decision,
            "headline": headline,
            "report": f"runs/{run_id}/{report_filename}",
            "summary_json": f"runs/{run_id}/{json_filename}",
            "images": [f"runs/{run_id}/{name}" for name in image_filenames],
        },
    )
    write_json(index_path, entries)

    toc_lines = [
        f"# {skill_root.name} Log",
        "",
        "Use this table of contents first. Each run keeps a compact Markdown report, a JSON summary, and cheap diagnostic figures.",
        "",
        "| Run | Field | Timestamp | Decision | Headline | Artifacts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        report_link = f"[report]({entry['report']})"
        json_link = f"[json]({entry['summary_json']})"
        image_link = ""
        if entry["images"]:
            image_link = f", [figures]({entry['images'][0]})"
        toc_lines.append(
            "| "
            f"`{entry['run_id']}` | `{entry['field']}` | `{entry['timestamp']}` | "
            f"`{entry['decision']}` | {entry['headline']} | {report_link}, {json_link}{image_link} |"
        )
    (logs_dir / "TOC.md").write_text("\n".join(toc_lines) + "\n", encoding="utf-8")


def render_map_panels(
    *,
    values: np.ndarray,
    threshold_mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    output_path: Path,
    title_prefix: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    level_indices = choose_representative_level_indices(pressure_levels_hpa, limit=3)
    latitudes_for_plot = latitudes_deg[::-1] if latitudes_deg[0] > latitudes_deg[-1] else latitudes_deg
    for column_index, level_index in enumerate(level_indices):
        level_values = values[level_index]
        level_mask = threshold_mask[level_index]
        if latitudes_deg[0] > latitudes_deg[-1]:
            level_values = level_values[::-1]
            level_mask = level_mask[::-1]
        axes[0, column_index].imshow(
            level_values,
            extent=[longitudes_deg[0], longitudes_deg[-1], latitudes_for_plot[0], latitudes_for_plot[-1]],
            aspect="auto",
            cmap="coolwarm",
        )
        axes[0, column_index].set_title(f"{title_prefix} {pressure_levels_hpa[level_index]:.0f} hPa")
        axes[0, column_index].set_xlabel("Lon")
        axes[0, column_index].set_ylabel("Lat")
        axes[1, column_index].imshow(
            level_mask.astype(np.float32),
            extent=[longitudes_deg[0], longitudes_deg[-1], latitudes_for_plot[0], latitudes_for_plot[-1]],
            aspect="auto",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, column_index].set_title(f"Top {FIELD_REPORT_REFERENCE_TOP_SHARE:.0f}% mask")
        axes[1, column_index].set_xlabel("Lon")
        axes[1, column_index].set_ylabel("Lat")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def choose_sample_columns(field_mass: np.ndarray, latitudes_deg: np.ndarray, longitudes_deg: np.ndarray) -> list[tuple[int, int, str]]:
    column_mass = np.asarray(field_mass.sum(axis=0), dtype=np.float32)
    flat_order = np.argsort(column_mass.ravel())[::-1]
    chosen: list[tuple[int, int, str]] = []
    for flat_index in flat_order:
        lat_index, lon_index = np.unravel_index(int(flat_index), column_mass.shape)
        latitude = float(latitudes_deg[lat_index])
        if any(abs(latitude - float(latitudes_deg[existing_lat])) < 20.0 for existing_lat, _, _ in chosen):
            continue
        label = f"{latitude:.0f}°, {float(longitudes_deg[lon_index]):.0f}°"
        chosen.append((lat_index, lon_index, label))
        if len(chosen) >= 3:
            break
    return chosen


def render_profile_panel(
    *,
    values: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    metrics: dict[str, Any],
    quantity_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    level_stats = metrics["vertical_structure"]["level_statistics"]
    pressures = np.asarray([row["pressure_hpa"] for row in level_stats], dtype=np.float32)
    weighted_means = np.asarray([row["weighted_mean"] for row in level_stats], dtype=np.float32)
    p05 = np.asarray([row["p05"] for row in level_stats], dtype=np.float32)
    p95 = np.asarray([row["p95"] for row in level_stats], dtype=np.float32)
    signal_fraction = np.asarray([row["signal_fraction"] for row in level_stats], dtype=np.float32)
    uniform_share = 1.0 / max(len(level_stats), 1)

    axes[0, 0].fill_betweenx(pressures, p05, p95, color="#c4d6f0", alpha=0.9, label="p05-p95")
    axes[0, 0].plot(weighted_means, pressures, color="#15305b", linewidth=2.0, label="area-weighted mean")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Pressure (hPa)")
    axes[0, 0].set_title("Per-level value spread")
    axes[0, 0].legend(fontsize=8)

    if pressures.size >= 2:
        bar_height = float(np.median(np.abs(np.diff(pressures))) * 0.7)
    else:
        bar_height = 25.0
    axes[0, 1].barh(pressures, 100.0 * signal_fraction, height=bar_height, color="#31688e")
    axes[0, 1].axvline(100.0 * uniform_share, color="#9b2226", linestyle="--", linewidth=1.4, label="even-by-level share")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel(f"{quantity_label.capitalize()} share (%)")
    axes[0, 1].set_ylabel("Pressure (hPa)")
    axes[0, 1].set_title(f"{quantity_label.capitalize()} share by pressure level")
    axes[0, 1].legend(fontsize=8)

    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite_values.size:
        bin_count = max(24, min(72, int(round(math.sqrt(finite_values.size)))))
        counts, edges = np.histogram(finite_values, bins=bin_count)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[1, 0].bar(centers, counts, width=np.diff(edges), color="#94d2bd", edgecolor="#0a9396", linewidth=0.4)
        for percentile, color, label in (
            (metrics["distribution"]["p05"], "#ae2012", "p05"),
            (metrics["distribution"]["p50"], "#005f73", "median"),
            (metrics["distribution"]["p95"], "#bb3e03", "p95"),
        ):
            if percentile is not None:
                axes[1, 0].axvline(percentile, color=color, linewidth=1.3, linestyle="--", label=label)
        if metrics["distribution"].get("mode_estimate") is not None:
            axes[1, 0].axvline(
                metrics["distribution"]["mode_estimate"],
                color="#3a0ca3",
                linewidth=1.5,
                label="mode estimate",
            )
        if np.nanmin(finite_values) < 0.0 < np.nanmax(finite_values):
            axes[1, 0].axvline(0.0, color="#6c757d", linewidth=1.0, linestyle=":")
        axes[1, 0].set_xlabel("Value")
        axes[1, 0].set_ylabel("Voxel count")
        axes[1, 0].set_title("Global value distribution")
        axes[1, 0].legend(fontsize=8)

    lat_curve = metrics["horizontal_structure"]["abs_latitude_curve"]
    abs_lat = np.asarray(lat_curve["abs_latitude_deg"], dtype=np.float32)
    cum_signal = np.asarray(lat_curve["cumulative_signal_fraction"], dtype=np.float32)
    cum_area = np.asarray(lat_curve["cumulative_area_fraction"], dtype=np.float32)
    axes[1, 1].plot(abs_lat, 100.0 * cum_signal, color="#005f73", linewidth=2.0, label=f"{quantity_label} within ±lat")
    axes[1, 1].plot(abs_lat, 100.0 * cum_area, color="#bb3e03", linewidth=1.6, linestyle="--", label="area within ±lat")
    axes[1, 1].set_xlabel("Absolute latitude (deg)")
    axes[1, 1].set_ylabel("Cumulative share (%)")
    axes[1, 1].set_title("Equator-to-pole concentration")
    axes[1, 1].legend(fontsize=8)

    concentration_curve = metrics["horizontal_structure"]["column_concentration_curve"]
    column_fraction = np.asarray(concentration_curve["column_fraction"], dtype=np.float32)
    cumulative_signal_fraction = np.asarray(concentration_curve["cumulative_signal_fraction"], dtype=np.float32)
    axes[2, 0].plot(100.0 * column_fraction, 100.0 * cumulative_signal_fraction, color="#7b2cbf", linewidth=2.0, label=f"{quantity_label} concentration")
    axes[2, 0].plot([0.0, 100.0], [0.0, 100.0], color="#6c757d", linestyle="--", linewidth=1.3, label="even baseline")
    axes[2, 0].set_xlabel("Top columns retained (%)")
    axes[2, 0].set_ylabel(f"Cumulative {quantity_label} captured (%)")
    axes[2, 0].set_title(f"Horizontal {quantity_label} concentration")
    axes[2, 0].legend(fontsize=8)

    anomaly_structure = metrics.get("anomaly_structure")
    if anomaly_structure and anomaly_structure.get("relative_departure_percent"):
        threshold_summary = anomaly_structure["relative_departure_percent"]
        fractions = np.asarray(
            [row["area_fraction_by_threshold"] for row in threshold_summary["per_level"]],
            dtype=np.float32,
        )
        image = axes[2, 1].imshow(
            100.0 * fractions,
            aspect="auto",
            cmap="magma",
            origin="upper",
        )
        axes[2, 1].set_xticks(range(len(threshold_summary["thresholds"])))
        axes[2, 1].set_xticklabels([f"{threshold:.0f}%" for threshold in threshold_summary["thresholds"]])
        tick_indices = np.linspace(0, len(pressures) - 1, min(6, len(pressures)), dtype=int)
        axes[2, 1].set_yticks(tick_indices)
        axes[2, 1].set_yticklabels([f"{pressures[index]:.0f}" for index in tick_indices])
        axes[2, 1].set_xlabel("Relative anomaly threshold")
        axes[2, 1].set_ylabel("Pressure (hPa)")
        axes[2, 1].set_title("Area above climatology-departure thresholds")
        colorbar = figure.colorbar(image, ax=axes[2, 1], fraction=0.046, pad=0.04)
        colorbar.set_label("Area fraction (%)")
    else:
        latitude_bands = metrics["horizontal_structure"].get("abs_latitude_band_shares", [])
        labels = [f"{int(row['lower_deg'])}-{int(row['upper_deg'])}°" for row in latitude_bands]
        signal_share = [100.0 * row["signal_fraction"] for row in latitude_bands]
        area_share = [100.0 * row["area_fraction"] for row in latitude_bands]
        y_positions = np.arange(len(labels), dtype=np.float32)
        axes[2, 1].barh(y_positions + 0.17, signal_share, height=0.32, color="#219ebc", label="signal")
        axes[2, 1].barh(y_positions - 0.17, area_share, height=0.32, color="#adb5bd", label="area")
        axes[2, 1].set_yticks(y_positions)
        axes[2, 1].set_yticklabels(labels)
        axes[2, 1].set_xlabel("Share (%)")
        axes[2, 1].set_ylabel("Absolute latitude band")
        axes[2, 1].set_title("Absolute-latitude concentration by band")
        axes[2, 1].legend(fontsize=8)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_component_threshold_sweep_panel(
    *,
    sweep_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True, sharex=True)
    level_rows = sweep_metrics["per_level"]
    top_shares = np.asarray(sweep_metrics["top_share_percents"], dtype=np.float32)
    pressures = np.asarray([row["pressure_hpa"] for row in level_rows], dtype=np.float32)
    norm = matplotlib.colors.LogNorm(vmin=max(1.0, float(np.min(pressures))), vmax=float(np.max(pressures)))
    cmap = matplotlib.cm.viridis_r

    panels = [
        ("component_count", "Total Components"),
        ("warm_component_count", "Warm Components"),
        ("cold_component_count", "Cold Components"),
    ]
    for axis, (metric_key, title) in zip(axes, panels, strict=True):
        for row, pressure_hpa in zip(level_rows, pressures, strict=True):
            values = np.asarray([entry[metric_key] for entry in row["shares"]], dtype=np.float32)
            axis.plot(
                top_shares,
                values,
                color=cmap(norm(float(pressure_hpa))),
                linewidth=1.1,
                alpha=0.95,
            )
        axis.set_title(title)
        axis.set_xlabel("Top-share threshold kept (%)")
        axis.set_ylabel("Component count")
        axis.grid(True, alpha=0.22, linewidth=0.6)

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Pressure (hPa)")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_structure_of_data_metrics(cube: FieldCube) -> dict[str, Any]:
    values = np.asarray(cube.values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    finite_values = np.asarray(values[finite_mask], dtype=np.float64)
    signal_field, signal_basis_name = signal_basis(values)

    area_weights_2d = area_weights_for_latitudes(cube.latitudes_deg)[:, None].astype(np.float64)
    area_weights_3d = area_weights_2d[None, :, :]
    weighted_signal = np.where(finite_mask, signal_field * area_weights_3d, 0.0).astype(np.float64)
    total_signal = float(weighted_signal.sum())

    weighted_positive = np.where(finite_mask, np.maximum(values, 0.0) * area_weights_3d, 0.0).astype(np.float64)
    weighted_negative = np.where(finite_mask, np.abs(np.minimum(values, 0.0)) * area_weights_3d, 0.0).astype(np.float64)
    positive_signal_share = float(weighted_positive.sum() / max(total_signal, 1e-12))
    negative_signal_share = float(weighted_negative.sum() / max(total_signal, 1e-12))

    global_mean, global_std = weighted_mean_std(values, np.where(finite_mask, area_weights_3d, 0.0))
    q01 = safe_percentile(values, 1.0)
    q05 = safe_percentile(values, 5.0)
    q25 = safe_percentile(values, 25.0)
    q50 = safe_percentile(values, 50.0)
    q75 = safe_percentile(values, 75.0)
    q95 = safe_percentile(values, 95.0)
    q99 = safe_percentile(values, 99.0)
    mode_estimate = estimate_mode(values)
    iqr = None if q25 is None or q75 is None else q75 - q25
    central_90_span = None if q05 is None or q95 is None else q95 - q05
    tail_to_core_ratio = None
    if q01 is not None and q99 is not None and iqr is not None and abs(iqr) > 1e-12:
        tail_to_core_ratio = float((q99 - q01) / iqr)
    outlier_fraction = 0.0
    if global_mean is not None and global_std is not None and global_std > 1e-12:
        outlier_fraction = float(np.mean(np.abs(finite_values - global_mean) >= (3.0 * global_std)))

    level_signal = weighted_signal.reshape(weighted_signal.shape[0], -1).sum(axis=1)
    level_signal_fraction = (
        np.asarray(level_signal / max(total_signal, 1e-12), dtype=np.float64)
        if total_signal > 0.0
        else np.zeros(values.shape[0], dtype=np.float64)
    )

    level_stats: list[dict[str, Any]] = []
    level_means: list[float] = []
    level_stds: list[float] = []
    level_total_weights: list[float] = []
    for level_index, pressure in enumerate(cube.pressure_levels_hpa):
        level_values = np.asarray(values[level_index], dtype=np.float32)
        level_weights = np.broadcast_to(area_weights_2d, level_values.shape)
        mean, std = weighted_mean_std(level_values, area_weights_2d)
        level_means.append(float(mean) if mean is not None else np.nan)
        level_stds.append(float(std) if std is not None else np.nan)
        finite_level = np.isfinite(level_values)
        finite_level_values = np.asarray(level_values[finite_level], dtype=np.float64)
        level_total_weights.append(float(level_weights[finite_level].sum()))
        level_stats.append(
            {
                "pressure_hpa": float(pressure),
                "weighted_mean": mean,
                "weighted_std": std,
                "min": float(np.min(finite_level_values)) if finite_level_values.size else None,
                "max": float(np.max(finite_level_values)) if finite_level_values.size else None,
                "p05": float(np.percentile(finite_level_values, 5.0)) if finite_level_values.size else None,
                "p25": float(np.percentile(finite_level_values, 25.0)) if finite_level_values.size else None,
                "p50": float(np.percentile(finite_level_values, 50.0)) if finite_level_values.size else None,
                "p75": float(np.percentile(finite_level_values, 75.0)) if finite_level_values.size else None,
                "p95": float(np.percentile(finite_level_values, 95.0)) if finite_level_values.size else None,
                "signal_fraction": float(level_signal_fraction[level_index]),
            }
        )

    level_means_array = np.asarray(level_means, dtype=np.float64)
    level_stds_array = np.asarray(level_stds, dtype=np.float64)
    finite_level_means = level_means_array[np.isfinite(level_means_array)]
    finite_level_stds = level_stds_array[np.isfinite(level_stds_array)]
    between_level_mean_std = float(np.std(finite_level_means)) if finite_level_means.size else 0.0
    mean_within_level_std = float(np.mean(finite_level_stds)) if finite_level_stds.size else 0.0
    level_mean_separation_ratio = (
        float(between_level_mean_std / max(mean_within_level_std, 1e-12))
        if finite_level_stds.size
        else 0.0
    )

    total_weight = float(np.sum(level_total_weights))
    between_level_variance = 0.0
    within_level_variance = 0.0
    if global_mean is not None and total_weight > 0.0:
        for total_weight_level, mean, std in zip(
            level_total_weights,
            level_means,
            level_stds,
            strict=True,
        ):
            if total_weight_level <= 0.0 or math.isnan(mean):
                continue
            between_level_variance += total_weight_level * (mean - global_mean) ** 2
            if not math.isnan(std):
                within_level_variance += total_weight_level * (std**2)
        between_level_variance /= max(total_weight, 1e-12)
        within_level_variance /= max(total_weight, 1e-12)
    between_level_variance_fraction = (
        float(between_level_variance / max(between_level_variance + within_level_variance, 1e-12))
        if (between_level_variance + within_level_variance) > 0.0
        else 0.0
    )

    profile_direction, profile_monotonicity = summarize_profile_direction(
        level_means_array,
        cube.pressure_levels_hpa,
    )

    column_signal = weighted_signal.sum(axis=0)
    band_metrics = {}
    base_area = float(area_weights_2d.sum() * cube.longitudes_deg.size)
    for band in (20.0, 40.0, 60.0):
        latitude_mask = np.asarray(np.abs(cube.latitudes_deg) <= band, dtype=bool)
        band_metrics[f"within_{int(band)}"] = {
            "signal_fraction": float(column_signal[latitude_mask].sum() / max(total_signal, 1e-12)),
            "area_fraction": float(
                (area_weights_2d[latitude_mask].sum() * cube.longitudes_deg.size) / max(base_area, 1e-12)
            ),
        }

    north_mask = np.asarray(cube.latitudes_deg >= 0.0, dtype=bool)
    south_mask = np.asarray(cube.latitudes_deg < 0.0, dtype=bool)
    zonal_signal = column_signal.sum(axis=1)
    zonal_signal_fraction = np.asarray(zonal_signal / max(total_signal, 1e-12), dtype=np.float64)
    abs_latitude_thresholds = compute_abs_latitude_thresholds(
        cube.latitudes_deg,
        zonal_signal_fraction,
    )
    abs_latitude_curve = cumulative_abs_latitude_curve(
        cube.latitudes_deg,
        zonal_signal_fraction,
        area_weights_for_latitudes(cube.latitudes_deg),
    )
    abs_latitude_band_shares = compute_abs_latitude_band_shares(
        cube.latitudes_deg,
        zonal_signal_fraction,
        area_weights_for_latitudes(cube.latitudes_deg),
    )
    peak_column_index = int(np.argmax(column_signal)) if column_signal.size else 0
    peak_lat_index, peak_lon_index = np.unravel_index(peak_column_index, column_signal.shape)
    positive_columns = np.sort(np.asarray(column_signal.ravel(), dtype=np.float64))[::-1]
    top_column_shares = {}
    for share_percent in (1, 5, 10):
        keep_count = max(1, int(math.ceil(column_signal.size * (share_percent / 100.0))))
        top_column_shares[f"top_{share_percent}_percent_columns_signal_share"] = float(
            positive_columns[:keep_count].sum() / max(total_signal, 1e-12)
        )

    entropy = compute_spatial_entropy(column_signal)
    column_concentration_curve = sample_column_concentration_curve(column_signal)

    representative_indices = choose_representative_level_indices(cube.pressure_levels_hpa, limit=5)
    representative_levels = [level_stats[index] for index in representative_indices]
    representative_spans = [
        row["p95"] - row["p05"]
        for row in representative_levels
        if row["p95"] is not None and row["p05"] is not None and abs(row["p95"] - row["p05"]) > 1e-12
    ]
    representative_span_ratio = (
        float(max(representative_spans) / min(representative_spans))
        if len(representative_spans) >= 2
        else 1.0
    )
    vertical_evenness = compute_normalized_entropy(level_signal_fraction)
    cumulative_from_surface = cumulative_pressure_thresholds(
        cube.pressure_levels_hpa,
        level_signal_fraction,
        direction="surface_down",
    )
    cumulative_from_top = cumulative_pressure_thresholds(
        cube.pressure_levels_hpa,
        level_signal_fraction,
        direction="top_down",
    )
    band_50 = narrowest_pressure_band(
        cube.pressure_levels_hpa,
        level_signal_fraction,
        target_fraction=0.5,
    )
    band_80 = narrowest_pressure_band(
        cube.pressure_levels_hpa,
        level_signal_fraction,
        target_fraction=0.8,
    )

    dominant_level_indices = np.argsort(level_signal_fraction)[::-1][:3]
    dominant_levels = [
        {
            "pressure_hpa": float(cube.pressure_levels_hpa[index]),
            "signal_fraction": float(level_signal_fraction[index]),
            "weighted_mean": level_stats[index]["weighted_mean"],
        }
        for index in dominant_level_indices
    ]
    anomaly_structure = build_climatology_anomaly_metrics(cube)

    return {
        "sign_mode": detect_sign_mode(values),
        "signal_basis": signal_basis_name,
        "finite_voxel_count": int(np.count_nonzero(finite_mask)),
        "distribution": {
            "min": float(np.min(finite_values)) if finite_values.size else None,
            "max": float(np.max(finite_values)) if finite_values.size else None,
            "weighted_mean": global_mean,
            "weighted_std": global_std,
            "p01": q01,
            "p05": q05,
            "p25": q25,
            "p50": q50,
            "p75": q75,
            "p95": q95,
            "p99": q99,
            "mode_estimate": mode_estimate,
            "iqr": iqr,
            "central_90_span": central_90_span,
            "tail_to_core_ratio": tail_to_core_ratio,
            "outlier_fraction_above_3sigma": outlier_fraction,
        },
        "sign_structure": {
            "positive_fraction": float(np.mean(finite_values > 0.0)) if finite_values.size else 0.0,
            "negative_fraction": float(np.mean(finite_values < 0.0)) if finite_values.size else 0.0,
            "near_zero_fraction": float(np.mean(np.abs(finite_values) <= 1e-12)) if finite_values.size else 0.0,
            "positive_signal_share": positive_signal_share,
            "negative_signal_share": negative_signal_share,
        },
        "vertical_structure": {
            "level_signal_fraction": [
                {
                    "pressure_hpa": float(pressure),
                    "signal_fraction": float(fraction),
                }
                for pressure, fraction in zip(cube.pressure_levels_hpa, level_signal_fraction, strict=True)
            ],
            "below_850_hpa_signal_fraction": float(
                level_signal[np.asarray(cube.pressure_levels_hpa >= 850.0, dtype=bool)].sum() / max(total_signal, 1e-12)
            ),
            "below_700_hpa_signal_fraction": float(
                level_signal[np.asarray(cube.pressure_levels_hpa >= 700.0, dtype=bool)].sum() / max(total_signal, 1e-12)
            ),
            "below_500_hpa_signal_fraction": float(
                level_signal[np.asarray(cube.pressure_levels_hpa >= 500.0, dtype=bool)].sum() / max(total_signal, 1e-12)
            ),
            "above_300_hpa_signal_fraction": float(
                level_signal[np.asarray(cube.pressure_levels_hpa <= 300.0, dtype=bool)].sum() / max(total_signal, 1e-12)
            ),
            "between_level_variance_fraction": between_level_variance_fraction,
            "between_level_mean_std": between_level_mean_std,
            "mean_within_level_std": mean_within_level_std,
            "level_mean_separation_ratio": level_mean_separation_ratio,
            "level_mean_range": (
                float(np.max(finite_level_means) - np.min(finite_level_means))
                if finite_level_means.size
                else 0.0
            ),
            "profile_direction": profile_direction,
            "profile_monotonicity_fraction": profile_monotonicity,
            "vertical_evenness": vertical_evenness,
            "cumulative_from_surface_pressure_hpa": cumulative_from_surface,
            "cumulative_from_top_pressure_hpa": cumulative_from_top,
            "narrowest_band_50_percent": band_50,
            "narrowest_band_80_percent": band_80,
            "top_3_levels_signal_share": float(sum(item["signal_fraction"] for item in dominant_levels)),
            "dominant_levels": dominant_levels,
            "level_statistics": level_stats,
        },
        "horizontal_structure": {
            "latitudinal_signal_share": band_metrics,
            "zonal_signal_fraction_by_latitude": [
                {
                    "latitude_deg": float(latitude),
                    "signal_fraction": float(fraction),
                }
                for latitude, fraction in zip(cube.latitudes_deg, zonal_signal_fraction, strict=True)
            ],
            "abs_latitude_thresholds_deg": abs_latitude_thresholds,
            "abs_latitude_curve": {
                key: value.tolist() for key, value in abs_latitude_curve.items()
            },
            "abs_latitude_band_shares": abs_latitude_band_shares,
            "north_hemisphere_signal_share": float(column_signal[north_mask].sum() / max(total_signal, 1e-12)),
            "south_hemisphere_signal_share": float(column_signal[south_mask].sum() / max(total_signal, 1e-12)),
            "spatial_entropy": entropy,
            "column_concentration_curve": {
                key: value.tolist() for key, value in column_concentration_curve.items()
            },
            **top_column_shares,
            "peak_column": {
                "latitude_deg": float(cube.latitudes_deg[peak_lat_index]),
                "longitude_deg": float(cube.longitudes_deg[peak_lon_index]),
                "signal_fraction": float(column_signal[peak_lat_index, peak_lon_index] / max(total_signal, 1e-12)),
            },
        },
        "cross_level_scale": {
            "representative_levels": representative_levels,
            "representative_span_ratio": representative_span_ratio,
        },
        "anomaly_structure": anomaly_structure,
    }


def build_structure_of_data_interpretation(metrics: dict[str, Any], cube: FieldCube) -> dict[str, Any]:
    terms = structure_quantity_terms(cube)
    distribution = metrics["distribution"]
    sign = metrics["sign_structure"]
    vertical = metrics["vertical_structure"]
    horizontal = metrics["horizontal_structure"]
    scale = metrics["cross_level_scale"]
    anomaly_structure = metrics.get("anomaly_structure")
    within20 = horizontal["latitudinal_signal_share"]["within_20"]
    tropics_excess = within20["signal_fraction"] - within20["area_fraction"]
    entropy = horizontal["spatial_entropy"]
    abs_lat_thresholds = horizontal.get(
        "abs_latitude_thresholds_deg",
        {"50_percent": None, "80_percent": None},
    )
    cumulative_from_surface = vertical.get(
        "cumulative_from_surface_pressure_hpa",
        {"50_percent": None, "75_percent": None},
    )
    cumulative_from_top = vertical.get(
        "cumulative_from_top_pressure_hpa",
        {"50_percent": None, "75_percent": None},
    )
    band_50 = vertical.get(
        "narrowest_band_50_percent",
        {"highest_pressure_hpa": None, "lowest_pressure_hpa": None},
    )
    band_80 = vertical.get(
        "narrowest_band_80_percent",
        {"highest_pressure_hpa": None, "lowest_pressure_hpa": None},
    )

    executive_summary: list[str] = []
    imbalance_flags: list[str] = []
    meteorological_interpretation: list[str] = []
    implications: list[str] = []
    recommendations: list[str] = []

    strong_vertical_stratification = (
        vertical["between_level_variance_fraction"] >= 0.55
        or vertical["level_mean_separation_ratio"] >= 1.0
    )
    moderate_vertical_stratification = (
        vertical["between_level_variance_fraction"] >= 0.35
        or vertical["level_mean_separation_ratio"] >= 0.6
    )
    near_surface_dominance = vertical["below_700_hpa_signal_fraction"] >= 0.75
    upper_level_dominance = vertical["above_300_hpa_signal_fraction"] >= 0.45
    vertical_evenness = vertical.get("vertical_evenness")
    top_3_levels_signal_share = vertical.get("top_3_levels_signal_share")
    vertically_even = (
        vertical_evenness is not None
        and vertical_evenness >= 0.9
        and top_3_levels_signal_share is not None
        and top_3_levels_signal_share <= 0.25
        and not near_surface_dominance
        and not upper_level_dominance
    )
    equatorial_concentration = tropics_excess >= 0.12
    regional_localization = (
        horizontal["top_5_percent_columns_signal_share"] >= 0.25
        or (entropy is not None and entropy <= 0.75)
    )
    hemispheric_asymmetry = abs(
        horizontal["north_hemisphere_signal_share"] - horizontal["south_hemisphere_signal_share"]
    ) >= 0.15
    heavy_tails = (
        (distribution["tail_to_core_ratio"] is not None and distribution["tail_to_core_ratio"] >= 6.0)
        or distribution["outlier_fraction_above_3sigma"] >= 0.01
    )
    cross_level_scale_mismatch = scale["representative_span_ratio"] >= 4.0
    sign_balanced = (
        sign["positive_fraction"] >= 0.2
        and sign["negative_fraction"] >= 0.2
        and 0.35 <= sign["positive_signal_share"] <= 0.65
    )

    if strong_vertical_stratification:
        executive_summary.append(
            f"The field is strongly vertically stratified: level-to-level mean shifts explain {format_percent(vertical['between_level_variance_fraction'])} of total variance."
        )
        imbalance_flags.append("strong_vertical_stratification")
    elif moderate_vertical_stratification:
        executive_summary.append(
            f"The field has moderate vertical stratification: level-to-level mean shifts explain {format_percent(vertical['between_level_variance_fraction'])} of total variance."
        )
        imbalance_flags.append("moderate_vertical_stratification")

    if near_surface_dominance:
        lower_75 = cumulative_from_surface["75_percent"]
        executive_summary.append(
            f"The {terms['share_noun_short']} is concentrated in the lower troposphere ({format_percent(vertical['below_700_hpa_signal_fraction'])} of the area-weighted {terms['share_noun']} lies below 700 hPa"
            + (f", and about three quarters lies below {lower_75:.0f} hPa)." if lower_75 is not None else ").")
        )
        imbalance_flags.append("near_surface_dominance")
    elif upper_level_dominance:
        upper_75 = cumulative_from_top["75_percent"]
        executive_summary.append(
            f"The {terms['share_noun_short']} is weighted toward upper levels ({format_percent(vertical['above_300_hpa_signal_fraction'])} lies above 300 hPa"
            + (f", and about three quarters lies above {upper_75:.0f} hPa)." if upper_75 is not None else ").")
        )
        imbalance_flags.append("upper_level_dominance")
    elif vertically_even:
        executive_summary.append(
            f"The {terms['share_noun_short']} is fairly evenly distributed across the available pressure levels; no narrow vertical band dominates."
        )
        imbalance_flags.append("vertically_even")

    if equatorial_concentration:
        half_lat = abs_lat_thresholds["50_percent"]
        executive_summary.append(
            f"The field is concentrated toward the tropics ({format_percent(within20['signal_fraction'])} of the {terms['share_noun_short']} lies within ±20°, versus {format_percent(within20['area_fraction'])} of the area)."
            + (f" Half the {terms['share_noun_short']} lies within about ±{half_lat:.0f}°." if half_lat is not None else "")
        )
        imbalance_flags.append("equatorial_concentration")
    elif abs_lat_thresholds["80_percent"] is not None and abs_lat_thresholds["80_percent"] >= 65.0:
        executive_summary.append(
            f"The field is broad in latitude: about 80% of the {terms['share_noun_short']} is spread across ±{abs_lat_thresholds['80_percent']:.0f}°."
        )
        imbalance_flags.append("latitude_broad")

    if regional_localization:
        executive_summary.append(
            f"The field is geographically uneven: the top 5% of horizontal columns contain {format_percent(horizontal['top_5_percent_columns_signal_share'])} of the {terms['share_noun_short']}."
        )
        imbalance_flags.append("regional_localization")

    if heavy_tails:
        executive_summary.append(
            f"The value distribution has heavy tails (tail/core ratio {distribution['tail_to_core_ratio']:.1f}; {format_percent(distribution['outlier_fraction_above_3sigma'])} beyond 3σ)."
        )
        imbalance_flags.append("heavy_tails")

    if sign_balanced:
        executive_summary.append(
            "Positive and negative departures both matter, so the sign structure is part of the field’s organization."
        )
        imbalance_flags.append("sign_balanced")

    if cross_level_scale_mismatch:
        executive_summary.append(
            f"The horizontal spread differs substantially by level (representative span ratio {scale['representative_span_ratio']:.1f}x)."
        )
        imbalance_flags.append("cross_level_scale_mismatch")

    if not executive_summary:
        executive_summary.append("The field is fairly diffuse and does not show one overwhelming structural imbalance.")

    if cube.canonical_field == "specific_humidity":
        meteorological_interpretation.append(
            "For specific humidity, lower-tropospheric and tropical concentration is physically expected because the warm boundary layer holds most of the water vapor."
        )
    if cube.canonical_field in {"temperature", "dry_potential_temperature"} and cube.transform.get("anomaly") == "none":
        meteorological_interpretation.append(
            "For raw temperature-like fields, strong vertical structure mostly reflects the background thermal stratification of the atmosphere rather than discrete coherent objects."
        )
    if cube.transform.get("anomaly") == "climatology":
        meteorological_interpretation.append(
            "Because this is a climatology-relative anomaly field, the important structure is where departures from the seasonal background concentrate, not the raw background profile itself."
        )
        if anomaly_structure:
            relative = anomaly_structure["relative_departure_percent"]
            sigma = anomaly_structure["standardized_departure"]
            threshold_lookup = {
                float(threshold): index for index, threshold in enumerate(relative["thresholds"])
            }
            sigma_lookup = {
                float(threshold): index for index, threshold in enumerate(sigma["thresholds"])
            }
            rel_5 = relative["overall_area_fraction"][threshold_lookup[5.0]]
            rel_10 = relative["overall_area_fraction"][threshold_lookup[10.0]]
            sig_2 = sigma["overall_area_fraction"][sigma_lookup[2.0]]
            meteorological_interpretation.append(
                "Large climatology departures are spatially selective: "
                f"about {100.0 * rel_5:.1f}% of the area-weighted cube exceeds a 5% departure, "
                f"about {100.0 * rel_10:.1f}% exceeds 10%, and about {100.0 * sig_2:.1f}% exceeds 2σ."
            )
    if cube.transform.get("anomaly") == "lat_mean":
        meteorological_interpretation.append(
            "Because this is a latitude-mean anomaly field, zonally symmetric background structure has already been removed; what remains emphasizes regional departures."
        )
    if vertical["profile_direction"] in {"increases_with_height", "decreases_with_height"} and vertical["profile_monotonicity_fraction"] is not None:
        direction_text = "increases" if vertical["profile_direction"] == "increases_with_height" else "decreases"
        meteorological_interpretation.append(
            f"The area-weighted mean profile mostly {direction_text} with height ({format_percent(vertical['profile_monotonicity_fraction'])} of adjacent level steps follow that direction)."
        )
    if band_50["highest_pressure_hpa"] is not None and band_80["highest_pressure_hpa"] is not None:
        meteorological_interpretation.append(
            f"Using all available levels, the narrowest band holding half the {terms['share_noun_short']} is {band_50['highest_pressure_hpa']:.0f}-{band_50['lowest_pressure_hpa']:.0f} hPa, and the narrowest band holding 80% is {band_80['highest_pressure_hpa']:.0f}-{band_80['lowest_pressure_hpa']:.0f} hPa."
        )
    if hemispheric_asymmetry:
        meteorological_interpretation.append(
            f"There is a hemispheric imbalance ({format_percent(horizontal['north_hemisphere_signal_share'])} NH versus {format_percent(horizontal['south_hemisphere_signal_share'])} SH)."
        )
    if abs_lat_thresholds["50_percent"] is not None and abs_lat_thresholds["80_percent"] is not None:
        meteorological_interpretation.append(
            f"Half the {terms['share_noun_short']} lies within about ±{abs_lat_thresholds['50_percent']:.0f}°, and 80% lies within about ±{abs_lat_thresholds['80_percent']:.0f}°."
        )
    if not meteorological_interpretation:
        meteorological_interpretation.append(
            "No single physical imbalance dominates the field; it is a relatively broad distribution in this sampled cube."
        )

    if (
        strong_vertical_stratification
        and cube.transform.get("anomaly") == "none"
        and cube.canonical_field in {"temperature", "dry_potential_temperature"}
    ):
        implications.append(
            "A raw global threshold will mostly recover the background thermal stratification; remove the background profile or normalize by level before treating this as a 3D object field."
        )
    elif strong_vertical_stratification and cube.canonical_field == "specific_humidity":
        implications.append(
            "A raw global threshold will mostly recover the moist boundary layer and warm-source regions. That is meteorologically real, but it will hide weaker elevated structure."
        )
    if near_surface_dominance:
        implications.append(
            "Any naive 3D rendering will be pulled toward shallow lower-tropospheric structure unless you restrict the pressure window or normalize vertically."
        )
    if upper_level_dominance:
        implications.append(
            "If you want lower or mid-tropospheric structure, clip the pressure window first; otherwise the upper atmosphere will dominate the result."
        )
    if equatorial_concentration:
        implications.append(
            "Global thresholds will overemphasize the tropics relative to mid-latitude structure unless you normalize by level or latitude band."
        )
    if regional_localization:
        implications.append(
            "This field is organized around regional hotspots rather than a globally even background, so regional views may be more informative than global thresholds alone."
        )
    if cross_level_scale_mismatch:
        implications.append(
            "A single raw threshold is not cross-level comparable here; per-level normalization or per-level thresholds are safer."
        )
    if sign_balanced:
        implications.append(
            "Any later extraction should preserve sign, because opposite-signed departures are both meteorologically meaningful."
        )
    if not implications:
        implications.append(
            "The field is structurally simple enough that a first-pass 3D probe can be informative without heavy preprocessing."
        )

    severity = 0
    severity += 2 if strong_vertical_stratification else 1 if moderate_vertical_stratification else 0
    severity += 2 if near_surface_dominance or upper_level_dominance else 0
    severity += 1 if equatorial_concentration else 0
    severity += 1 if regional_localization else 0
    severity += 1 if cross_level_scale_mismatch else 0
    if sign_balanced:
        severity += 1
    if (
        strong_vertical_stratification
        and cube.transform.get("anomaly") == "none"
        and cube.canonical_field in {"temperature", "dry_potential_temperature"}
    ):
        severity += 2

    if severity >= 4:
        verdict = "needs_preconditioning"
        recommendations.extend(
            [
                "Precondition the field before structure_probe: adjust the pressure window and/or normalize by level.",
                "Use the structural imbalances above to decide whether the next transform should be vertical, latitudinal, or sign-aware.",
            ]
        )
    elif severity >= 2:
        verdict = "usable_with_constraints"
        recommendations.extend(
            [
                "A structure_probe is still useful, but keep the listed imbalances in mind when choosing thresholds and pressure windows.",
                "Prefer per-level or sign-aware logic if you want the probe to reflect meteorological structure rather than background bias.",
            ]
        )
    else:
        verdict = "usable_as_is"
        recommendations.append(
            "The field is clean enough for a first structure_probe without major preprocessing."
        )

    return {
        "executive_summary": executive_summary[:6],
        "imbalance_flags": imbalance_flags,
        "meteorological_interpretation": meteorological_interpretation[:6],
        "representation_implications": implications[:6],
        "structure_probe_readiness": {
            "verdict": verdict,
            "recommendations": recommendations[:4],
        },
    }


def build_structure_of_data_chat_report(
    *,
    cube: FieldCube,
    metrics: dict[str, Any],
    interpretation: dict[str, Any],
    artifact_paths: dict[str, str] | None = None,
) -> str:
    terms = structure_quantity_terms(cube)
    distribution = metrics["distribution"]
    vertical = metrics["vertical_structure"]
    horizontal = metrics["horizontal_structure"]
    scale = metrics["cross_level_scale"]
    anomaly_structure = metrics.get("anomaly_structure")
    within20 = horizontal["latitudinal_signal_share"]["within_20"]
    within40 = horizontal["latitudinal_signal_share"]["within_40"]
    within60 = horizontal["latitudinal_signal_share"]["within_60"]
    peak = horizontal["peak_column"]
    abs_lat_thresholds = horizontal["abs_latitude_thresholds_deg"]
    surface_cumulative = vertical["cumulative_from_surface_pressure_hpa"]
    top_cumulative = vertical["cumulative_from_top_pressure_hpa"]
    band_50 = vertical["narrowest_band_50_percent"]
    band_80 = vertical["narrowest_band_80_percent"]
    dominant_levels_text = ", ".join(
        f"{row['pressure_hpa']:.0f} hPa ({100.0 * row['signal_fraction']:.0f}%)"
        for row in vertical["dominant_levels"]
    )
    mode_text = (
        f"`{distribution['mode_estimate']:.3g}`"
        if distribution.get("mode_estimate") is not None
        else "not resolved"
    )
    latitude_band_rows = sorted(
        horizontal.get("abs_latitude_band_shares", []),
        key=lambda row: row["signal_fraction"],
        reverse=True,
    )[:3]
    latitude_band_text = ", ".join(
        f"{int(row['lower_deg'])}-{int(row['upper_deg'])}°: {100.0 * row['signal_fraction']:.0f}% signal vs {100.0 * row['area_fraction']:.0f}% area"
        for row in latitude_band_rows
    ) or "not resolved"

    summary_lines = "\n".join(f"- {item}" for item in interpretation["executive_summary"])
    physics_lines = "\n".join(f"- {item}" for item in interpretation["meteorological_interpretation"])
    implication_lines = "\n".join(f"- {item}" for item in interpretation["representation_implications"])
    recommendation_lines = "\n".join(
        f"- {item}" for item in interpretation["structure_probe_readiness"]["recommendations"]
    )

    horizontal_lines = [
        f"- {terms['share_noun_short'].capitalize()} within ±20° / ±40° / ±60°: `{100.0 * within20['signal_fraction']:.0f}% / {100.0 * within40['signal_fraction']:.0f}% / {100.0 * within60['signal_fraction']:.0f}%`",
        f"- Area within ±20° / ±40° / ±60°: `{100.0 * within20['area_fraction']:.0f}% / {100.0 * within40['area_fraction']:.0f}% / {100.0 * within60['area_fraction']:.0f}%`",
        f"- Strongest absolute-latitude bands: {latitude_band_text}",
    ]
    if abs_lat_thresholds["50_percent"] is not None and abs_lat_thresholds["80_percent"] is not None:
        horizontal_lines.append(
            f"- Half / 80% of the {terms['share_noun_short']} lies within: `±{abs_lat_thresholds['50_percent']:.0f}° / ±{abs_lat_thresholds['80_percent']:.0f}°`"
        )
    horizontal_lines.extend(
        [
            f"- North vs south hemisphere signal: `{100.0 * horizontal['north_hemisphere_signal_share']:.0f}% / {100.0 * horizontal['south_hemisphere_signal_share']:.0f}%`",
            f"- Top 1% / 5% / 10% of columns contain: `{100.0 * horizontal['top_1_percent_columns_signal_share']:.0f}% / {100.0 * horizontal['top_5_percent_columns_signal_share']:.0f}% / {100.0 * horizontal['top_10_percent_columns_signal_share']:.0f}%` of total {terms['share_noun_short']}",
            f"- Peak column location: `{peak['latitude_deg']:.1f}°, {peak['longitude_deg']:.1f}°` carrying `{100.0 * peak['signal_fraction']:.4f}%` of total {terms['share_noun_short']}",
            f"- Spatial entropy: `{horizontal['spatial_entropy']:.2f}`",
        ]
    )
    horizontal_section = "\n".join(horizontal_lines)

    vertical_lines = [
        f"- Quantity summarized here: `{terms['share_noun']}`",
        f"- Area-weighted level-mean range: `{vertical['level_mean_range']:.3g}`",
        f"- Between-level variance fraction: `{100.0 * vertical['between_level_variance_fraction']:.0f}%`",
        f"- Level-mean separation ratio: `{vertical['level_mean_separation_ratio']:.2f}`",
    ]
    if vertical["vertical_evenness"] is not None:
        vertical_lines.append(
            f"- Vertical evenness across all available levels: `{vertical['vertical_evenness']:.2f}`"
        )
    mean_profile_line = f"- Mean profile with height: `{vertical['profile_direction']}`"
    if vertical["profile_monotonicity_fraction"] is not None:
        mean_profile_line += (
            f" (`{100.0 * vertical['profile_monotonicity_fraction']:.0f}%` of adjacent steps agree)"
        )
    vertical_lines.append(mean_profile_line)
    if surface_cumulative["50_percent"] is not None and surface_cumulative["75_percent"] is not None:
        vertical_lines.append(
            f"- Half / 75% of the {terms['share_noun_short']} lies below: `{surface_cumulative['50_percent']:.0f} / {surface_cumulative['75_percent']:.0f} hPa`"
        )
    if top_cumulative["50_percent"] is not None and top_cumulative["75_percent"] is not None:
        vertical_lines.append(
            f"- Half / 75% of the {terms['share_noun_short']} lies above: `{top_cumulative['50_percent']:.0f} / {top_cumulative['75_percent']:.0f} hPa`"
        )
    vertical_lines.extend(
        [
            f"- Narrowest band containing 50% of the {terms['share_noun_short']}: `{band_50['highest_pressure_hpa']:.0f}-{band_50['lowest_pressure_hpa']:.0f} hPa`",
            f"- Narrowest band containing 80% of the {terms['share_noun_short']}: `{band_80['highest_pressure_hpa']:.0f}-{band_80['lowest_pressure_hpa']:.0f} hPa`",
            f"- Top contributing levels: `{dominant_levels_text}`",
            f"- {terms['share_noun_short'].capitalize()} below 850 / 700 / 500 hPa: `{100.0 * vertical['below_850_hpa_signal_fraction']:.0f}% / {100.0 * vertical['below_700_hpa_signal_fraction']:.0f}% / {100.0 * vertical['below_500_hpa_signal_fraction']:.0f}%`",
            f"- {terms['share_noun_short'].capitalize()} above 300 hPa: `{100.0 * vertical['above_300_hpa_signal_fraction']:.0f}%`",
        ]
    )
    vertical_section = "\n".join(vertical_lines)

    anomaly_section = ""
    if anomaly_structure:
        relative = anomaly_structure["relative_departure_percent"]
        sigma = anomaly_structure["standardized_departure"]
        quick_components = anomaly_structure.get("quick_top_share_components")
        rel_lookup = {float(threshold): index for index, threshold in enumerate(relative["thresholds"])}
        sigma_lookup = {float(threshold): index for index, threshold in enumerate(sigma["thresholds"])}
        rel_5_idx = rel_lookup[5.0]
        rel_10_idx = rel_lookup[10.0]
        sigma_2_idx = sigma_lookup[2.0]
        strongest_rel5 = sorted(
            relative["per_level"],
            key=lambda row: row["area_fraction_by_threshold"][rel_5_idx],
            reverse=True,
        )[:3]
        strongest_sigma2 = sorted(
            sigma["per_level"],
            key=lambda row: row["area_fraction_by_threshold"][sigma_2_idx],
            reverse=True,
        )[:3]
        strongest_rel5_text = ", ".join(
            f"{row['pressure_hpa']:.0f} hPa ({100.0 * row['area_fraction_by_threshold'][rel_5_idx]:.1f}%)"
            for row in strongest_rel5
        )
        strongest_sigma2_text = ", ".join(
            f"{row['pressure_hpa']:.0f} hPa ({100.0 * row['area_fraction_by_threshold'][sigma_2_idx]:.1f}%)"
            for row in strongest_sigma2
        )
        quick_component_lines = ""
        if quick_components:
            per_level_quick = quick_components["per_level"]
            count_series = ", ".join(
                (
                    f"{row['pressure_hpa']:.0f}:"
                    f"T{row['component_count']}/{row['big_component_count']}"
                    f" W{row['warm_component_count']}/{row['warm_big_component_count']}"
                    f" C{row['cold_component_count']}/{row['cold_big_component_count']}"
                )
                for row in per_level_quick
            )
            peak_big_levels = quick_components.get("peak_big_component_levels", [])
            peak_big_text = (
                ", ".join(
                    f"{row['pressure_hpa']:.0f} hPa ({row['big_component_count']} big; {row['component_count']} total)"
                    for row in peak_big_levels
                )
                if peak_big_levels
                else "no levels produced big components"
            )
            quick_component_lines = (
                f"- Quick component read uses the per-level top `{quick_components['top_share_percent']:.0f}%` of cells by `abs(climatology anomaly)` with no smoothing or morphology.\n"
                f"- Component counts by level (`T=total/big, W=warm/big, C=cold/big`): `{count_series}`\n"
                f"- Big means at least `max({quick_components['minimum_big_component_cells_floor']} cells, {100.0 * quick_components['minimum_big_component_fraction_of_level']:.1f}% of a level)`; peak levels: `{peak_big_text}`\n"
                f"- Levels with any big components: `{quick_components['levels_with_any_big_components']}` / `{len(per_level_quick)}`\n"
            )
        anomaly_section = (
            "## Anomaly Checks\n"
            f"- Area above 1 / 2 / 5 / 10% climatology departure: `{100.0 * relative['overall_area_fraction'][rel_lookup[1.0]]:.1f}% / {100.0 * relative['overall_area_fraction'][rel_lookup[2.0]]:.1f}% / {100.0 * relative['overall_area_fraction'][rel_5_idx]:.1f}% / {100.0 * relative['overall_area_fraction'][rel_10_idx]:.1f}%`\n"
            f"- Area above 1σ / 2σ / 3σ: `{100.0 * sigma['overall_area_fraction'][sigma_lookup[1.0]]:.1f}% / {100.0 * sigma['overall_area_fraction'][sigma_2_idx]:.1f}% / {100.0 * sigma['overall_area_fraction'][sigma_lookup[3.0]]:.1f}%`\n"
            f"- Levels with the largest 5% departures: `{strongest_rel5_text}`\n"
            f"- Levels with the largest 2σ departures: `{strongest_sigma2_text}`\n\n"
            f"{quick_component_lines}\n"
        )

    plots_section = ""
    if artifact_paths:
        lines = []
        if artifact_paths.get("maps"):
            lines.append(f"- Sample maps: `{artifact_paths['maps']}`")
        if artifact_paths.get("profiles"):
            lines.append(f"- Diagnostic panels: `{artifact_paths['profiles']}`")
        if artifact_paths.get("component_thresholds"):
            lines.append(f"- Component threshold sweep: `{artifact_paths['component_thresholds']}`")
        if lines:
            plots_section = "## Plots\n" + "\n".join(lines) + "\n"

    return (
        f"# structure_of_data: {cube.canonical_field}\n\n"
        f"Field: `{cube.canonical_field}` from `{format_display_path(cube.dataset_path)}` at `{cube.timestamp}`.\n"
        f"Transform: derived=`{cube.transform.get('derived') or 'none'}`, anomaly=`{cube.transform.get('anomaly')}`, smoothing=`{cube.transform.get('smoothing')}`.\n\n"
        "## What Stands Out\n"
        f"{summary_lines}\n\n"
        "## Distribution\n"
        f"- Global range: `{distribution['min']:.3g}` to `{distribution['max']:.3g}`\n"
        f"- Area-weighted mean ± std: `{distribution['weighted_mean']:.3g}` ± `{distribution['weighted_std']:.3g}`\n"
        f"- Mode estimate: {mode_text}\n"
        f"- Middle 90% of values: `{distribution['p05']:.3g}` to `{distribution['p95']:.3g}`\n"
        f"- Tail/core ratio: `{distribution['tail_to_core_ratio']:.2f}`\n"
        f"- Fraction beyond 3σ: `{100.0 * distribution['outlier_fraction_above_3sigma']:.2f}%`\n\n"
        "## Vertical Structure\n"
        f"{vertical_section}\n\n"
        "## Horizontal Structure\n"
        f"{horizontal_section}\n\n"
        f"{anomaly_section}"
        "## Cross-Level Comparability\n"
        f"- Representative per-level spread ratio: `{scale['representative_span_ratio']:.2f}x`\n\n"
        "## Meteorological Read\n"
        f"{physics_lines}\n\n"
        "## What This Means For Representation\n"
        f"{implication_lines}\n\n"
        "## Structure Probe Readiness\n"
        f"- Verdict: `{interpretation['structure_probe_readiness']['verdict']}`\n"
        f"{recommendation_lines}\n\n"
        f"{plots_section}"
    )


def structure_of_data_markdown(
    *,
    cube: FieldCube,
    metrics: dict[str, Any],
    interpretation: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    chat_report = build_structure_of_data_chat_report(
        cube=cube,
        metrics=metrics,
        interpretation=interpretation,
        artifact_paths=artifact_paths,
    )
    return (
        f"{chat_report}\n\n"
        "## Artifacts\n"
        f"- Level panels: `{artifact_paths['maps']}`\n"
        f"- Profile panels: `{artifact_paths['profiles']}`\n"
        + (
            f"- Component threshold sweep: `{artifact_paths['component_thresholds']}`\n"
            if artifact_paths.get("component_thresholds")
            else ""
        )
    )


def run_structure_of_data(
    *,
    skill_root: Path,
    field: str,
    dataset_path: Path | None = None,
    variable_name: str | None = None,
    timestamp: str | None = None,
    pressure_levels_hpa: list[float] | None = None,
    anomaly: str = "none",
    smoothing: float = 0.0,
    derived: str | None = None,
    climatology_path: Path | None = None,
    latitude_stride: int = 2,
    longitude_stride: int = 2,
    artifact_dir: Path | None = None,
    make_plots: bool = True,
    save_summary: bool = False,
) -> dict[str, Any]:
    cube = load_field_cube(
        field=field,
        dataset_path=dataset_path,
        variable_name=variable_name,
        timestamp=timestamp,
        pressure_levels_hpa=pressure_levels_hpa,
        anomaly=anomaly,
        smoothing=smoothing,
        derived=derived,
        climatology_path=climatology_path,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )

    metrics = build_structure_of_data_metrics(cube)
    interpretation = build_structure_of_data_interpretation(metrics, cube)

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    label = f"{cube.canonical_field}-{cube.timestamp}"
    run_id = f"{now}__{slugify(label)}"
    run_dir = artifact_dir.expanduser().resolve() if artifact_dir else (CACHE_ROOT / "structure-of-data" / run_id)

    artifact_paths: dict[str, str] = {}
    if make_plots or save_summary:
        run_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        signal_field, _ = signal_basis(cube.values)
        quantity_terms = structure_quantity_terms(cube)
        anomaly_structure = metrics.get("anomaly_structure")
        reference_mask, _ = compute_top_share_mask(
            signal_field,
            FIELD_REPORT_REFERENCE_TOP_SHARE,
            tail="high",
        )
        maps_path = run_dir / "maps.png"
        profiles_path = run_dir / "profiles.png"
        component_thresholds_path = run_dir / "component-thresholds.png"
        render_map_panels(
            values=cube.values,
            threshold_mask=reference_mask,
            pressure_levels_hpa=cube.pressure_levels_hpa,
            latitudes_deg=cube.latitudes_deg,
            longitudes_deg=cube.longitudes_deg,
            output_path=maps_path,
            title_prefix=cube.canonical_field,
        )
        render_profile_panel(
            values=cube.values,
            pressure_levels_hpa=cube.pressure_levels_hpa,
            metrics=metrics,
            quantity_label=quantity_terms["chart_label"],
            output_path=profiles_path,
        )
        if anomaly_structure and anomaly_structure.get("quick_top_share_sweep"):
            render_component_threshold_sweep_panel(
                sweep_metrics=anomaly_structure["quick_top_share_sweep"],
                output_path=component_thresholds_path,
            )
        artifact_paths.update(
            {
                "maps": repo_relative_path(maps_path),
                "profiles": repo_relative_path(profiles_path),
            }
        )
        if anomaly_structure and anomaly_structure.get("quick_top_share_sweep"):
            artifact_paths["component_thresholds"] = repo_relative_path(component_thresholds_path)

    chat_report = build_structure_of_data_chat_report(
        cube=cube,
        metrics=metrics,
        interpretation=interpretation,
        artifact_paths=artifact_paths or None,
    )
    summary = {
        "skill": "structure_of_data",
        "run_id": run_id,
        "generated_at": now,
        "input": {
            "field": cube.requested_field,
            "canonical_field": cube.canonical_field,
            "source_variable": cube.source_variable,
            "dataset_path": format_display_path(cube.dataset_path),
            "timestamp": cube.timestamp,
            "pressure_levels_hpa": [float(value) for value in cube.pressure_levels_hpa],
            "transform": cube.transform,
            "notes": cube.notes,
            "units": cube.units,
        },
        "metrics": metrics,
        "interpretation": interpretation,
        "chat_report": chat_report,
        "artifacts": artifact_paths,
    }

    if save_summary:
        report_text = structure_of_data_markdown(
            cube=cube,
            metrics=metrics,
            interpretation=interpretation,
            artifact_paths=artifact_paths,
        )
        report_path = run_dir / "summary.md"
        report_path.write_text(report_text, encoding="utf-8")
        summary_path = run_dir / "summary.json"
        summary["artifacts"].update(
            {
                "summary_markdown": repo_relative_path(report_path),
                "summary_json": repo_relative_path(summary_path),
            }
        )
        write_json(summary_path, summary)

    return summary


def build_field_report_metrics(cube: FieldCube) -> dict[str, Any]:
    return build_structure_of_data_metrics(cube)


def build_field_report_interpretation(metrics: dict[str, Any], cube: FieldCube) -> dict[str, Any]:
    return build_structure_of_data_interpretation(metrics, cube)


def field_report_markdown(
    *,
    cube: FieldCube,
    metrics: dict[str, Any],
    interpretation: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    return structure_of_data_markdown(
        cube=cube,
        metrics=metrics,
        interpretation=interpretation,
        artifact_paths=artifact_paths,
    )


def run_field_report(
    *,
    skill_root: Path,
    field: str,
    dataset_path: Path | None = None,
    variable_name: str | None = None,
    timestamp: str | None = None,
    pressure_levels_hpa: list[float] | None = None,
    anomaly: str = "none",
    smoothing: float = 0.0,
    derived: str | None = None,
    climatology_path: Path | None = None,
    latitude_stride: int = 2,
    longitude_stride: int = 2,
) -> dict[str, Any]:
    return run_structure_of_data(
        skill_root=skill_root,
        field=field,
        dataset_path=dataset_path,
        variable_name=variable_name,
        timestamp=timestamp,
        pressure_levels_hpa=pressure_levels_hpa,
        anomaly=anomaly,
        smoothing=smoothing,
        derived=derived,
        climatology_path=climatology_path,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )


def default_threshold_tail(sign_mode: str) -> str:
    return "absolute" if sign_mode == "mixed" else "high"


def bridge_mask(
    values: np.ndarray,
    occupied_mask: np.ndarray,
    bridge_levels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    gap_limit = max(int(bridge_levels), 0)
    if gap_limit <= 0:
        return np.asarray(occupied_mask, dtype=bool), {
            "bridge_levels": 0,
            "added_voxel_count": 0,
            "component_count_before": None,
            "component_count_after": None,
        }

    sign_mode = detect_sign_mode(values)
    sign_field = np.ones_like(values, dtype=np.int8)
    if sign_mode == "mixed":
        sign_field = np.sign(np.asarray(values, dtype=np.float32)).astype(np.int8)
    filled_mask = np.asarray(occupied_mask, dtype=bool).copy()

    for lat_index in range(values.shape[1]):
        for lon_index in range(values.shape[2]):
            column_mask = filled_mask[:, lat_index, lon_index]
            occupied_levels = np.flatnonzero(column_mask)
            if occupied_levels.size < 2:
                continue
            for start_level, end_level in zip(occupied_levels[:-1], occupied_levels[1:], strict=True):
                gap_size = int(end_level - start_level - 1)
                if gap_size <= 0 or gap_size > gap_limit:
                    continue
                if sign_mode == "mixed":
                    start_sign = sign_field[start_level, lat_index, lon_index]
                    end_sign = sign_field[end_level, lat_index, lon_index]
                    if start_sign == 0 or start_sign != end_sign:
                        continue
                    gap_signs = sign_field[start_level + 1 : end_level, lat_index, lon_index]
                    if np.any(gap_signs == -start_sign):
                        continue
                    if not np.all((gap_signs == 0) | (gap_signs == start_sign)):
                        continue
                filled_mask[start_level + 1 : end_level, lat_index, lon_index] = True

    before_labels, before_count = label_wrapped_volume_components(occupied_mask)
    after_labels, after_count = label_wrapped_volume_components(filled_mask)
    return filled_mask, {
        "bridge_levels": gap_limit,
        "added_voxel_count": int(np.count_nonzero(filled_mask & ~occupied_mask)),
        "component_count_before": int(before_count),
        "component_count_after": int(after_count),
    }


def apply_binary_morphology(mask: np.ndarray, mode: str) -> np.ndarray:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "none":
        return np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    if normalized_mode == "open":
        return ndimage.binary_opening(mask, structure=structure)
    if normalized_mode == "close":
        return ndimage.binary_closing(mask, structure=structure)
    raise ValueError("Unsupported morphology mode. Use one of: none, open, close.")


def resolve_grow_rule(grow_rule: str | None, sign_mode: str) -> tuple[str, float | None]:
    if not grow_rule:
        if sign_mode == "mixed":
            return "same-sign-relaxed-half", 0.5
        return "relaxed-half", 0.5
    normalized = grow_rule.strip().lower()
    if normalized in {"same-sign-relaxed-half", "relaxed-half"}:
        return normalized, 0.5
    if normalized in {"same-sign-relaxed-quarter", "relaxed-quarter"}:
        return normalized, 0.25
    if normalized in {"same-sign-above-zero", "above-zero"}:
        return normalized, None
    raise ValueError(
        "Unsupported grow_rule. Use one of: same-sign-relaxed-half, same-sign-relaxed-quarter, same-sign-above-zero, relaxed-half, relaxed-quarter, above-zero."
    )


def keep_seed_connected_components(eligible_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    labels, component_count = label_wrapped_volume_components(eligible_mask)
    if component_count <= 0:
        return np.zeros_like(eligible_mask, dtype=bool)
    seed_labels = np.unique(labels[seed_mask & (labels > 0)])
    if seed_labels.size == 0:
        return np.zeros_like(eligible_mask, dtype=bool)
    return np.isin(labels, seed_labels)


def build_seed_grow_mask(
    values: np.ndarray,
    *,
    threshold_percent: float,
    tail: str,
    grow_rule: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sign_mode = detect_sign_mode(values)
    seed_mask, seed_threshold = compute_top_share_mask(values, threshold_percent, tail=tail)
    resolved_rule, factor = resolve_grow_rule(grow_rule, sign_mode)
    if np.count_nonzero(seed_mask) == 0:
        return seed_mask, {
            "grow_rule": resolved_rule,
            "seed_threshold": seed_threshold,
            "seed_voxel_count": 0,
            "grown_voxel_count": 0,
        }

    if resolved_rule in {"same-sign-above-zero", "above-zero"}:
        if sign_mode == "mixed":
            positive = keep_seed_connected_components((values > 0.0), seed_mask & (values > 0.0))
            negative = keep_seed_connected_components((values < 0.0), seed_mask & (values < 0.0))
            grown_mask = positive | negative
        else:
            grown_mask = keep_seed_connected_components(values > 0.0, seed_mask)
    else:
        assert factor is not None
        if tail == "absolute" and sign_mode == "mixed":
            eligible_positive = (values > 0.0) & (np.abs(values) >= abs(float(seed_threshold)) * factor)
            eligible_negative = (values < 0.0) & (np.abs(values) >= abs(float(seed_threshold)) * factor)
            positive = keep_seed_connected_components(eligible_positive, seed_mask & (values > 0.0))
            negative = keep_seed_connected_components(eligible_negative, seed_mask & (values < 0.0))
            grown_mask = positive | negative
        else:
            threshold_value = float(seed_threshold or 0.0)
            eligible_mask = values >= threshold_value * factor
            grown_mask = keep_seed_connected_components(eligible_mask, seed_mask)

    return np.asarray(grown_mask, dtype=bool), {
        "grow_rule": resolved_rule,
        "seed_threshold": float(seed_threshold) if seed_threshold is not None else None,
        "seed_voxel_count": int(np.count_nonzero(seed_mask)),
        "grown_voxel_count": int(np.count_nonzero(grown_mask)),
    }


def build_threshold_mask_for_structure(
    values: np.ndarray,
    *,
    method: str,
    threshold_percent: float,
    tail: str,
    grow_rule: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_method = method.strip().lower()
    if normalized_method == "threshold":
        keep_mask, threshold_value = compute_top_share_mask(values, threshold_percent, tail=tail)
        return keep_mask, {
            "method": normalized_method,
            "tail": tail,
            "threshold_value": float(threshold_value) if threshold_value is not None else None,
            "threshold_percent": float(threshold_percent),
        }
    if normalized_method == "seed_grow":
        keep_mask, metadata = build_seed_grow_mask(
            values,
            threshold_percent=threshold_percent,
            tail=tail,
            grow_rule=grow_rule,
        )
        return keep_mask, {
            "method": normalized_method,
            "tail": tail,
            "threshold_percent": float(threshold_percent),
            **metadata,
        }
    if normalized_method == "gradient":
        gradient_values = compute_horizontal_gradient_magnitude(values)
        keep_mask, threshold_value = compute_top_share_mask(gradient_values, threshold_percent, tail="high")
        return keep_mask, {
            "method": normalized_method,
            "tail": "high",
            "threshold_value": float(threshold_value) if threshold_value is not None else None,
            "threshold_percent": float(threshold_percent),
        }
    raise ValueError("Unsupported extraction method. Use one of: threshold, seed_grow, gradient.")


def component_sizes_from_labels(labels: np.ndarray, component_count: int) -> np.ndarray:
    if component_count <= 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(labels[labels > 0].ravel())
    return np.asarray(counts[1:], dtype=np.int64)


def component_vertical_spans(labels: np.ndarray, component_count: int) -> np.ndarray:
    spans = np.zeros(component_count, dtype=np.int64)
    for label_id in range(1, component_count + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        spans[label_id - 1] = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    return spans


def compute_top_down_occlusion(mask: np.ndarray) -> dict[str, float]:
    column_counts = np.count_nonzero(mask, axis=0)
    total_voxels = int(np.count_nonzero(mask))
    visible_voxels = int(np.count_nonzero(column_counts > 0))
    hidden_voxels = max(total_voxels - visible_voxels, 0)
    return {
        "top_down_occlusion_score": hidden_voxels / max(total_voxels, 1),
        "visible_footprint_fraction": visible_voxels / max(total_voxels, 1),
    }


def exposed_face_count(mask: np.ndarray) -> int:
    occupied = np.asarray(mask, dtype=bool)
    if not occupied.any():
        return 0
    faces = 0
    faces += int(np.count_nonzero(occupied & ~np.roll(occupied, 1, axis=2)))
    faces += int(np.count_nonzero(occupied & ~np.roll(occupied, -1, axis=2)))
    faces += int(np.count_nonzero(occupied & ~np.pad(occupied[:, 1:, :], ((0, 0), (0, 1), (0, 0)))))
    faces += int(np.count_nonzero(occupied & ~np.pad(occupied[:, :-1, :], ((0, 0), (1, 0), (0, 0)))))
    faces += int(np.count_nonzero(occupied & ~np.pad(occupied[1:, :, :], ((0, 1), (0, 0), (0, 0)))))
    faces += int(np.count_nonzero(occupied & ~np.pad(occupied[:-1, :, :], ((1, 0), (0, 0), (0, 0)))))
    return faces


def largest_component_shape_metrics(labels: np.ndarray, component_count: int) -> dict[str, Any]:
    if component_count <= 0:
        return {
            "anisotropy_score": 0.0,
            "shape_class": "none",
            "surface_to_volume_ratio": 0.0,
        }
    sizes = component_sizes_from_labels(labels, component_count)
    largest_label = int(np.argmax(sizes) + 1)
    coords = np.argwhere(labels == largest_label)
    if coords.shape[0] < 3:
        return {
            "anisotropy_score": 0.0,
            "shape_class": "speckle",
            "surface_to_volume_ratio": float(exposed_face_count(labels == largest_label) / max(coords.shape[0], 1)),
        }
    centered = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    ratio_21 = float(eigenvalues[1] / eigenvalues[0])
    ratio_32 = float(eigenvalues[2] / eigenvalues[1])
    anisotropy = float(1.0 - eigenvalues[2] / eigenvalues[0])
    if ratio_21 < 0.35:
        shape_class = "filament-like"
    elif ratio_32 < 0.35:
        shape_class = "sheet-like"
    else:
        shape_class = "blob-like"
    return {
        "anisotropy_score": anisotropy,
        "shape_class": shape_class,
        "surface_to_volume_ratio": float(
            exposed_face_count(labels == largest_label) / max(np.count_nonzero(labels == largest_label), 1)
        ),
    }


def classify_structure_type(
    *,
    method: str,
    num_components: int,
    largest_component_fraction: float,
    mean_vertical_span: float,
    single_level_fraction: float,
    shape_class: str,
    surface_to_volume_ratio: float,
) -> str:
    if method == "gradient" or surface_to_volume_ratio >= 5.0:
        return "boundary-like structures"
    if largest_component_fraction >= 0.7:
        return "single dominant mass"
    if num_components >= 40 and largest_component_fraction <= 0.2:
        return "fragmented speckle"
    if mean_vertical_span <= 2.0 or single_level_fraction >= 0.6 or shape_class == "sheet-like":
        return "layered sheets"
    return "coherent volumetric bodies"


def neighboring_threshold_sensitivity(
    values: np.ndarray,
    *,
    threshold_percent: float,
    tail: str,
) -> dict[str, Any]:
    shares = sorted(
        {
            max(1.0, threshold_percent * 0.5),
            float(threshold_percent),
            min(40.0, threshold_percent * 2.0),
        }
    )
    observations = []
    for share in shares:
        mask, _ = compute_top_share_mask(values, share, tail=tail)
        labels, component_count = label_wrapped_volume_components(mask)
        sizes = component_sizes_from_labels(labels, component_count)
        largest_fraction = float(sizes.max() / max(np.count_nonzero(mask), 1)) if sizes.size else 0.0
        observations.append(
            {
                "top_share_percent": float(share),
                "component_count": int(component_count),
                "largest_component_fraction": largest_fraction,
            }
        )
    return {"observations": observations}


def diagnose_structure_failure(
    *,
    method: str,
    largest_component_fraction: float,
    num_components: int,
    postprocess_inflation: float,
    bridge_added_fraction: float,
    sensitivity: dict[str, Any],
) -> dict[str, str]:
    if postprocess_inflation >= 0.35:
        return {
            "primary_cause": "postprocessing",
            "summary": "Most of the apparent structure is being created by smoothing or morphology rather than the raw extraction.",
        }
    if bridge_added_fraction >= 0.25:
        return {
            "primary_cause": "connectivity_rule",
            "summary": "Connectivity rules materially change the object, so the current structure depends on bridge logic more than on raw field support.",
        }

    observations = sensitivity["observations"]
    if largest_component_fraction >= 0.75:
        stricter = observations[0]
        if stricter["largest_component_fraction"] >= 0.65:
            return {
                "primary_cause": "field",
                "summary": "Even stricter thresholds stay blob-dominant, so the field itself is broad and smooth rather than the threshold simply being too loose.",
            }
        return {
            "primary_cause": "threshold",
            "summary": "A stricter threshold breaks the giant mass, so the current failure is mainly caused by a loose threshold choice.",
        }
    if num_components >= 40:
        looser = observations[-1]
        if looser["component_count"] >= 30:
            return {
                "primary_cause": "field",
                "summary": "Looser neighboring thresholds still stay fragmented, so the field itself lacks coherent volumetric support.",
            }
        return {
            "primary_cause": "threshold",
            "summary": "A slightly looser threshold improves connectivity, so the current speckle is mainly threshold-driven.",
        }
    return {
        "primary_cause": "field",
        "summary": "The current extraction is behaving roughly as the field would suggest; no stronger threshold or postprocess failure dominates.",
    }


def render_structure_panels(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    occupancy_count = np.count_nonzero(mask, axis=0).astype(np.float32)
    if latitudes_deg[0] > latitudes_deg[-1]:
        occupancy_count = occupancy_count[::-1]
    axes[0, 0].imshow(
        occupancy_count,
        extent=[longitudes_deg[0], longitudes_deg[-1], latitudes_deg.min(), latitudes_deg.max()],
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 0].set_title("Top-down occupied-level count")
    axes[0, 0].set_xlabel("Lon")
    axes[0, 0].set_ylabel("Lat")

    component_projection = np.max(labels, axis=0).astype(np.float32)
    if latitudes_deg[0] > latitudes_deg[-1]:
        component_projection = component_projection[::-1]
    axes[0, 1].imshow(
        component_projection,
        extent=[longitudes_deg[0], longitudes_deg[-1], latitudes_deg.min(), latitudes_deg.max()],
        aspect="auto",
        cmap="tab20",
    )
    axes[0, 1].set_title("Component-colored top projection")
    axes[0, 1].set_xlabel("Lon")
    axes[0, 1].set_ylabel("Lat")

    zonal_occupancy = np.mean(mask.astype(np.float32), axis=2)
    axes[1, 0].imshow(
        zonal_occupancy,
        extent=[latitudes_deg.min(), latitudes_deg.max(), pressure_levels_hpa.max(), pressure_levels_hpa.min()],
        aspect="auto",
        cmap="magma",
    )
    axes[1, 0].set_title("Latitude-pressure occupancy fraction")
    axes[1, 0].set_xlabel("Lat")
    axes[1, 0].set_ylabel("Pressure (hPa)")

    sizes = component_sizes_from_labels(labels, int(labels.max()))
    if sizes.size:
        ordered = np.sort(sizes)[::-1][:8]
        axes[1, 1].bar(np.arange(ordered.size), ordered, color="#24476b")
        axes[1, 1].set_xticks(np.arange(ordered.size), [str(index + 1) for index in range(ordered.size)])
        axes[1, 1].set_title("Largest component sizes")
        axes[1, 1].set_xlabel("Component rank")
        axes[1, 1].set_ylabel("Voxel count")
    else:
        axes[1, 1].text(0.5, 0.5, "No surviving voxels", ha="center", va="center")
        axes[1, 1].set_axis_off()

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_structure_probe_metrics(
    *,
    values: np.ndarray,
    method: str,
    threshold_percent: float,
    tail: str,
    grow_rule: str | None,
    bridge_levels: int,
    morphology: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    initial_mask, extraction_metadata = build_threshold_mask_for_structure(
        values,
        method=method,
        threshold_percent=threshold_percent,
        tail=tail,
        grow_rule=grow_rule,
    )
    after_bridge_mask, bridge_metadata = bridge_mask(values, initial_mask, bridge_levels)
    after_morphology_mask = apply_binary_morphology(after_bridge_mask, morphology)

    baseline_voxels = int(np.count_nonzero(initial_mask))
    final_voxels = int(np.count_nonzero(after_morphology_mask))
    postprocess_inflation = (
        (final_voxels - baseline_voxels) / max(baseline_voxels, 1)
        if baseline_voxels > 0
        else 0.0
    )
    labels, component_count = label_wrapped_volume_components(after_morphology_mask)
    sizes = component_sizes_from_labels(labels, component_count)
    spans = component_vertical_spans(labels, component_count)
    largest_component_fraction = float(sizes.max() / max(final_voxels, 1)) if sizes.size else 0.0

    single_level_fraction = float(
        np.count_nonzero(spans <= 1) / max(spans.size, 1)
    ) if spans.size else 0.0
    mean_vertical_span = float(np.mean(spans)) if spans.size else 0.0
    occlusion = compute_top_down_occlusion(after_morphology_mask)
    shape_metrics = largest_component_shape_metrics(labels, component_count)
    structure_type = classify_structure_type(
        method=method,
        num_components=int(component_count),
        largest_component_fraction=largest_component_fraction,
        mean_vertical_span=mean_vertical_span,
        single_level_fraction=single_level_fraction,
        shape_class=shape_metrics["shape_class"],
        surface_to_volume_ratio=shape_metrics["surface_to_volume_ratio"],
    )
    sensitivity = neighboring_threshold_sensitivity(
        values,
        threshold_percent=threshold_percent,
        tail=tail,
    )
    diagnosis = diagnose_structure_failure(
        method=method,
        largest_component_fraction=largest_component_fraction,
        num_components=int(component_count),
        postprocess_inflation=postprocess_inflation,
        bridge_added_fraction=bridge_metadata["added_voxel_count"] / max(final_voxels, 1),
        sensitivity=sensitivity,
    )

    metrics = {
        "component_structure": {
            "num_components": int(component_count),
            "largest_component_fraction": largest_component_fraction,
            "component_size_distribution": summarize_top_components(sizes, final_voxels),
        },
        "vertical_structure": {
            "mean_vertical_span": mean_vertical_span,
            "median_vertical_span": float(np.median(spans)) if spans.size else 0.0,
            "vertical_span_distribution_top_components": [int(value) for value in np.sort(spans)[::-1][:5]],
            "single_level_fraction": single_level_fraction,
        },
        "occlusion_visibility": occlusion,
        "connectivity_behavior": {
            "bridging_effect": bridge_metadata,
            "postprocess_inflation": postprocess_inflation,
        },
        "shape_heuristics": shape_metrics,
        "structure_type_classification": structure_type,
        "extraction_metadata": extraction_metadata,
        "sensitivity": sensitivity,
        "diagnosis": diagnosis,
    }
    return metrics, after_morphology_mask, labels


def build_structure_probe_interpretation(metrics: dict[str, Any]) -> dict[str, Any]:
    component = metrics["component_structure"]
    vertical = metrics["vertical_structure"]
    occlusion = metrics["occlusion_visibility"]
    connectivity = metrics["connectivity_behavior"]
    diagnosis = metrics["diagnosis"]

    executive_summary: list[str] = []
    failures: list[str] = []
    recommendations: list[str] = []

    executive_summary.append(
        f"Produces {component['num_components']} component(s), with the largest occupying {100.0 * component['largest_component_fraction']:.0f}% of kept voxels."
    )
    executive_summary.append(
        f"Mean vertical span is {vertical['mean_vertical_span']:.1f} levels and {100.0 * vertical['single_level_fraction']:.0f}% of components are single-level."
    )
    executive_summary.append(
        f"Top-down occlusion hides {100.0 * occlusion['top_down_occlusion_score']:.0f}% of kept voxels."
    )
    executive_summary.append(
        f"Structure reads as {metrics['structure_type_classification']}."
    )

    if component["largest_component_fraction"] >= 0.75:
        failures.append("One dominant component will read as a giant mass rather than inspectable bodies.")
        recommendations.append("Try a stricter threshold or a sign-aware/boundary extraction.")
    if component["num_components"] >= 40:
        failures.append("Component count is in speckle territory.")
        recommendations.append("Relax the threshold or switch to seed-grow.")
    if vertical["single_level_fraction"] >= 0.6:
        failures.append("Most surviving structure is effectively single-level, so depth will not read honestly.")
        recommendations.append("Use bridge_levels or seed-grow only if the raw field actually supports it.")
    if connectivity["postprocess_inflation"] >= 0.35:
        failures.append("Postprocess is manufacturing a large share of the visible structure.")
        recommendations.append("Reduce smoothing or morphology before trusting the geometry.")
    if occlusion["top_down_occlusion_score"] >= 0.7:
        failures.append("A top layer will occlude most of the volume from common viewpoints.")
        recommendations.append("Lower upper-level dominance or render as sparse boundaries instead of filled volume.")

    if not failures:
        failures.append("No dominant failure mode surfaced in the coarse probe.")
        recommendations.append("Promote this variant to a richer viewer only if the meteorological story matches the structure-of-data summary.")

    decision = "promote_variant" if (
        component["largest_component_fraction"] <= 0.7
        and component["num_components"] <= 25
        and vertical["single_level_fraction"] <= 0.5
        and connectivity["postprocess_inflation"] <= 0.3
    ) else "do_not_promote_yet"
    if decision == "do_not_promote_yet":
        recommendations.insert(0, diagnosis["summary"])

    return {
        "executive_summary": executive_summary[:6],
        "failure_modes": failures[:5],
        "extraction_diagnosis": diagnosis,
        "promotion_decision": {
            "decision": decision,
            "recommendations": recommendations[:5],
        },
    }


def structure_probe_markdown(
    *,
    input_config: dict[str, Any],
    metrics: dict[str, Any],
    interpretation: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    component = metrics["component_structure"]
    vertical = metrics["vertical_structure"]
    occlusion = metrics["occlusion_visibility"]
    connectivity = metrics["connectivity_behavior"]
    shape = metrics["shape_heuristics"]
    summary_lines = "\n".join(f"- {line}" for line in interpretation["executive_summary"])
    failure_lines = "\n".join(f"- {line}" for line in interpretation["failure_modes"])
    recommendation_lines = "\n".join(
        f"- {line}" for line in interpretation["promotion_decision"]["recommendations"]
    )

    return (
        f"# structure_probe: {input_config['field']}\n\n"
        "This is a fast extraction diagnostic, not a final renderer.\n\n"
        "## Executive Summary\n"
        f"{summary_lines}\n\n"
        "## Core Metrics\n"
        f"- Structure type: `{metrics['structure_type_classification']}`\n"
        f"- Components: `{component['num_components']}`\n"
        f"- Largest component fraction: `{100.0 * component['largest_component_fraction']:.0f}%`\n"
        f"- Component size distribution: `{', '.join(component['component_size_distribution']) or 'n/a'}`\n"
        f"- Mean vertical span: `{vertical['mean_vertical_span']:.2f}` levels\n"
        f"- Median vertical span: `{vertical['median_vertical_span']:.2f}` levels\n"
        f"- Single-level fraction: `{100.0 * vertical['single_level_fraction']:.0f}%`\n"
        f"- Top-down occlusion score: `{100.0 * occlusion['top_down_occlusion_score']:.0f}%`\n"
        f"- Visible footprint fraction: `{100.0 * occlusion['visible_footprint_fraction']:.0f}%`\n"
        f"- Bridging added voxels: `{connectivity['bridging_effect']['added_voxel_count']}`\n"
        f"- Postprocess inflation: `{100.0 * connectivity['postprocess_inflation']:.0f}%`\n"
        f"- Shape heuristic: `{shape['shape_class']}`\n"
        f"- Anisotropy score: `{shape['anisotropy_score']:.2f}`\n"
        f"- Surface-to-volume ratio: `{shape['surface_to_volume_ratio']:.2f}`\n\n"
        "## Failure Modes\n"
        f"{failure_lines}\n\n"
        "## Extraction Diagnosis\n"
        f"- Primary cause: `{interpretation['extraction_diagnosis']['primary_cause']}`\n"
        f"- Summary: {interpretation['extraction_diagnosis']['summary']}\n\n"
        "## Promotion Decision\n"
        f"- Decision: `{interpretation['promotion_decision']['decision']}`\n"
        f"{recommendation_lines}\n\n"
        "## Artifacts\n"
        f"- Structure panels: `{artifact_paths['overview']}`\n"
    )


def run_structure_probe(
    *,
    skill_root: Path,
    field: str,
    dataset_path: Path | None = None,
    variable_name: str | None = None,
    timestamp: str | None = None,
    pressure_levels_hpa: list[float] | None = None,
    anomaly: str = "none",
    smoothing: float = 0.0,
    derived: str | None = None,
    climatology_path: Path | None = None,
    method: str = "threshold",
    threshold_percent: float = 10.0,
    grow_rule: str | None = None,
    bridge_levels: int = 0,
    morphology: str = "none",
    resolution: str = "coarse",
    threshold_tail: str | None = None,
    structure_of_data_summary_path: Path | None = None,
) -> dict[str, Any]:
    if structure_of_data_summary_path is not None:
        summary = json.loads(structure_of_data_summary_path.read_text(encoding="utf-8"))
        input_payload = summary["input"]
        field = str(input_payload["field"])
        dataset_path = REPO_ROOT / input_payload["dataset_path"]
        variable_name = input_payload["source_variable"]
        timestamp = input_payload["timestamp"]
        pressure_levels_hpa = [float(value) for value in input_payload["pressure_levels_hpa"]]
        anomaly = input_payload["transform"]["anomaly"]
        smoothing = float(input_payload["transform"]["smoothing"])
        derived = input_payload["transform"]["derived"]
        if input_payload["transform"].get("climatology_path"):
            climatology_path = REPO_ROOT / input_payload["transform"]["climatology_path"]

    normalized_resolution = resolution.strip().lower()
    if normalized_resolution == "coarse":
        latitude_stride = COARSE_LATITUDE_STRIDE
        longitude_stride = COARSE_LONGITUDE_STRIDE
    elif normalized_resolution == "full":
        latitude_stride = 1
        longitude_stride = 1
    else:
        raise ValueError("Unsupported resolution. Use one of: coarse, full.")

    cube = load_field_cube(
        field=field,
        dataset_path=dataset_path,
        variable_name=variable_name,
        timestamp=timestamp,
        pressure_levels_hpa=pressure_levels_hpa,
        anomaly=anomaly,
        smoothing=smoothing,
        derived=derived,
        climatology_path=climatology_path,
        latitude_stride=latitude_stride,
        longitude_stride=longitude_stride,
    )
    tail = threshold_tail or default_threshold_tail(detect_sign_mode(cube.values))
    metrics, mask, labels = build_structure_probe_metrics(
        values=cube.values,
        method=method,
        threshold_percent=threshold_percent,
        tail=tail,
        grow_rule=grow_rule,
        bridge_levels=bridge_levels,
        morphology=morphology,
    )
    interpretation = build_structure_probe_interpretation(metrics)

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    label = f"{cube.canonical_field}-{method}-{cube.timestamp}"
    run_id = f"{now}__{slugify(label)}"
    run_dir = skill_root / "logs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    overview_path = run_dir / "overview.png"
    render_structure_panels(
        mask=mask,
        labels=labels,
        pressure_levels_hpa=cube.pressure_levels_hpa,
        latitudes_deg=cube.latitudes_deg,
        longitudes_deg=cube.longitudes_deg,
        output_path=overview_path,
    )
    artifact_paths = {"overview": repo_relative_path(overview_path)}

    input_config = {
        "field": cube.canonical_field,
        "dataset_path": format_display_path(cube.dataset_path),
        "timestamp": cube.timestamp,
        "transform": cube.transform,
        "extraction": {
            "method": method,
            "threshold_percent": float(threshold_percent),
            "tail": tail,
            "grow_rule": grow_rule,
            "bridge_levels": int(bridge_levels),
            "morphology": morphology,
            "resolution": normalized_resolution,
        },
    }

    report_text = structure_probe_markdown(
        input_config=input_config,
        metrics=metrics,
        interpretation=interpretation,
        artifact_paths=artifact_paths,
    )
    report_path = run_dir / "summary.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "skill": "structure_probe",
        "run_id": run_id,
        "generated_at": now,
        "input": input_config,
        "metrics": metrics,
        "interpretation": interpretation,
        "artifacts": {
            **artifact_paths,
            "summary_markdown": repo_relative_path(report_path),
        },
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)

    headline = interpretation["executive_summary"][0]
    update_log_index(
        skill_root=skill_root,
        run_id=run_id,
        summary_title=f"structure_probe {cube.canonical_field}",
        field_label=f"{cube.canonical_field}:{method}",
        timestamp=cube.timestamp,
        decision=interpretation["promotion_decision"]["decision"],
        report_filename=report_path.name,
        json_filename=summary_path.name,
        headline=headline,
        image_filenames=[overview_path.name],
    )
    return summary
