from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY_PATH = Path(
    "data/era5_temperature-climatology_1990-2020_11-08_12.nc"
)
DEFAULT_OUTPUT_DIR = Path("tmp/thermal-displacement-reference")
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"
CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "matplotlib").mkdir(exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))


@dataclass(frozen=True)
class SelectedBucket:
    lower: float
    upper: float
    center: float
    count: int
    middle_60_lower: float
    middle_60_upper: float


@dataclass(frozen=True)
class ThermalDisplacementLevel:
    matched_latitudes_deg: np.ndarray
    score_points_unsmoothed: np.ndarray
    score_points: np.ndarray
    bucket_counts: np.ndarray
    bucket_edges: np.ndarray
    selected_bucket: SelectedBucket


def match_same_longitude_climatology_latitudes(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    same_hemisphere: bool = True,
) -> np.ndarray:
    """Match each raw cell to the closest climatology latitude at the same longitude.

    This is the core equivalent-latitude lookup used by Thermal Displacement:
    for every source grid cell, keep pressure level and longitude fixed, then
    find the same-hemisphere climatology latitude whose temperature is closest
    to the raw cell's temperature. Exact temperature-distance ties are broken by
    choosing the climatology latitude row closest to the source cell's own
    latitude row.
    """

    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)

    if raw.shape != climatology.shape:
        raise ValueError("Raw and climatology level slices must have the same shape.")
    if raw.ndim != 2:
        raise ValueError("Expected 2D level slices shaped as latitude x longitude.")
    if latitudes.shape[0] != raw.shape[0]:
        raise ValueError("Latitude count does not match the level slice rows.")

    n_lat, n_lon = raw.shape
    matched = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    if same_hemisphere:
        search_groups = (
            (latitudes >= 0.0, latitudes >= 0.0),
            (latitudes < 0.0, latitudes < 0.0),
        )
    else:
        all_rows = np.ones(n_lat, dtype=bool)
        search_groups = ((all_rows, all_rows),)

    for lon_index in range(n_lon):
        profile = climatology[:, lon_index]
        valid_profile = np.isfinite(profile)
        source_column = raw[:, lon_index]

        for source_mask, candidate_mask in search_groups:
            source_rows = np.flatnonzero(source_mask)
            candidate_rows = np.flatnonzero(valid_profile & candidate_mask)
            if source_rows.size == 0 or candidate_rows.size == 0:
                continue

            order = candidate_rows[
                np.argsort(profile[candidate_rows], kind="mergesort")
            ]
            sorted_values = profile[order]
            source_values = source_column[source_rows]
            finite_source = np.isfinite(source_values)

            insertion = np.searchsorted(sorted_values, source_values)
            lower = np.clip(insertion - 1, 0, sorted_values.size - 1)
            upper = np.clip(insertion, 0, sorted_values.size - 1)

            lower_distance = np.abs(source_values - sorted_values[lower])
            upper_distance = np.abs(source_values - sorted_values[upper])
            lower_row_distance = np.abs(order[lower] - source_rows)
            upper_row_distance = np.abs(order[upper] - source_rows)
            choose_upper = (upper_distance < lower_distance) | (
                (upper_distance == lower_distance)
                & (upper_row_distance < lower_row_distance)
            )
            nearest = np.where(choose_upper, upper, lower)
            matched[source_rows[finite_source], lon_index] = latitudes[
                order[nearest[finite_source]]
            ]

    return matched


def score_points_from_matched_latitudes(
    matched_latitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    """Convert matched latitude to 0..100 Thermal Displacement score points.

    0 means polar-like, 100 means equator-like. The sign of matched latitude is
    intentionally collapsed with abs(latitude), so north and south are treated
    as the same thermal-source distance from the equator.
    """

    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes_deg))), 1.0e-6)
    score = 1.0 - np.abs(np.asarray(matched_latitudes_deg, dtype=np.float32)) / max_abs_latitude
    return np.asarray(np.clip(score, 0.0, 1.0) * 100.0, dtype=np.float32)


def smooth_score_after_matching(
    score_points: np.ndarray,
    sigma_cells: float = 1.0,
) -> np.ndarray:
    """Smooth Thermal Displacement score after matching, not raw temperature first."""

    sigma = max(float(sigma_cells), 0.0)
    if sigma <= 0.0:
        return np.asarray(score_points, dtype=np.float32)
    return np.asarray(
        gaussian_filter(
            np.asarray(score_points, dtype=np.float32),
            sigma=(sigma, sigma),
            mode=("nearest", "wrap"),
        ),
        dtype=np.float32,
    )


def one_point_score_histogram(score_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bucket score points into centered 1-point bins from 0 through 100."""

    centers = np.arange(0.0, 101.0, 1.0, dtype=np.float32)
    edges = np.concatenate(([centers[0] - 0.5], centers + 0.5)).astype(np.float32)
    finite = np.asarray(score_points[np.isfinite(score_points)], dtype=np.float32)
    counts, edges = np.histogram(finite, bins=edges)
    return counts.astype(np.int64), edges.astype(np.float32)


def choose_rarest_middle_60_bucket(score_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, SelectedBucket]:
    """Select the nonzero bucket with the fewest cells in the range-based middle 60%.

    The middle 60% here is not a percentile range. It is the middle 60% of this
    pressure level's numeric score range: min + 20% of range through max - 20%
    of range. This preserves the historical Thermal Displacement map behavior.
    """

    finite = np.asarray(score_points[np.isfinite(score_points)], dtype=np.float32)
    if finite.size == 0:
        raise ValueError("Cannot choose a bucket from an empty score field.")

    counts, edges = one_point_score_histogram(finite)
    centers = 0.5 * (edges[:-1] + edges[1:])
    value_min = float(np.nanmin(finite))
    value_max = float(np.nanmax(finite))
    middle_60_lower = value_min + 0.20 * (value_max - value_min)
    middle_60_upper = value_max - 0.20 * (value_max - value_min)

    middle_mask = (
        (centers >= middle_60_lower)
        & (centers <= middle_60_upper)
        & (counts > 0)
    )
    candidates = np.flatnonzero(middle_mask)
    if candidates.size == 0:
        candidates = np.flatnonzero(counts > 0)
    if candidates.size == 0:
        candidates = np.arange(counts.size)

    selected_index = int(candidates[np.argmin(counts[candidates])])
    lower = float(edges[selected_index])
    upper = float(edges[selected_index + 1])
    selected = SelectedBucket(
        lower=lower,
        upper=upper,
        center=0.5 * (lower + upper),
        count=int(counts[selected_index]),
        middle_60_lower=float(middle_60_lower),
        middle_60_upper=float(middle_60_upper),
    )
    return counts, edges, selected


def compute_thermal_displacement_level(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    *,
    score_smooth_sigma_cells: float = 1.0,
    same_hemisphere: bool = True,
) -> ThermalDisplacementLevel:
    """Compute the canonical same-longitude Thermal Displacement method.

    Ordering matters:
    1. Match raw temperature to same-pressure, same-longitude, same-hemisphere
       climatology latitude.
    2. Convert matched latitude to 0..100 score points.
    3. Smooth that score field with longitude wrapping and nearest latitude edges.
    4. Build 1-point score buckets and select the rarest nonzero bucket in the
       level's range-based middle 60%.
    """

    matched = match_same_longitude_climatology_latitudes(
        raw_temperature_k,
        climatology_temperature_k,
        latitudes_deg,
        same_hemisphere=same_hemisphere,
    )
    score_unsmoothed = score_points_from_matched_latitudes(matched, latitudes_deg)
    score = smooth_score_after_matching(
        score_unsmoothed,
        sigma_cells=score_smooth_sigma_cells,
    )
    counts, edges, selected = choose_rarest_middle_60_bucket(score)
    return ThermalDisplacementLevel(
        matched_latitudes_deg=matched,
        score_points_unsmoothed=score_unsmoothed,
        score_points=score,
        bucket_counts=counts,
        bucket_edges=edges,
        selected_bucket=selected,
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        pass
    return path.name


def _choose_timestamp(temperature, timestamp_text: str) -> np.datetime64:
    valid_times = np.asarray(temperature.coords["valid_time"].values)
    requested = np.datetime64(timestamp_text)
    if requested in valid_times:
        return requested
    nearest_index = int(np.argmin(np.abs(valid_times - requested)))
    return np.datetime64(valid_times[nearest_index])


def _parse_requested_levels(text: str, available_levels: np.ndarray) -> list[float]:
    if not text.strip():
        return [float(level) for level in available_levels]
    available = np.asarray(available_levels, dtype=np.float64)
    selected = []
    for piece in text.split(","):
        if not piece.strip():
            continue
        requested = float(piece.strip())
        selected.append(float(available[int(np.argmin(np.abs(available - requested)))]))
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference implementation for canonical Thermal Displacement."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pressure-levels", type=str, default="1000,850,500,250")
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument(
        "--allow-cross-hemisphere",
        action="store_true",
        help="Use the legacy behavior that searches both hemispheres for matches.",
    )
    parser.add_argument(
        "--write-arrays",
        action="store_true",
        help="Also write matched-latitude and score arrays as .npy files.",
    )
    return parser.parse_args()


def main() -> None:
    import xarray as xr

    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "arrays"
    if args.write_arrays:
        arrays_dir.mkdir(exist_ok=True)

    temperature_ds = xr.open_dataset(args.dataset)
    climatology_ds = xr.open_dataset(args.climatology)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = _choose_timestamp(temperature, args.timestamp)
    pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = _parse_requested_levels(args.pressure_levels, pressure_levels)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)

    rows = []
    for level_hpa in selected_levels:
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        climatology_level = (
            climatology.sel(pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        result = compute_thermal_displacement_level(
            raw_level,
            climatology_level,
            latitudes,
            score_smooth_sigma_cells=args.score_smooth_sigma_cells,
            same_hemisphere=not args.allow_cross_hemisphere,
        )

        row = {
            "pressure_level_hpa": float(level_hpa),
            "white_center": result.selected_bucket.center,
            "selected_bucket_lower": result.selected_bucket.lower,
            "selected_bucket_upper": result.selected_bucket.upper,
            "selected_bucket_count": result.selected_bucket.count,
            "middle_60_score_lower": result.selected_bucket.middle_60_lower,
            "middle_60_score_upper": result.selected_bucket.middle_60_upper,
        }
        if args.write_arrays:
            slug = f"{level_hpa:g}".replace(".", "p").replace("-", "m") + "hpa"
            matched_path = arrays_dir / f"matched_latitudes_deg_{slug}.npy"
            score_raw_path = arrays_dir / f"score_points_unsmoothed_{slug}.npy"
            score_path = arrays_dir / f"score_points_{slug}.npy"
            np.save(matched_path, result.matched_latitudes_deg)
            np.save(score_raw_path, result.score_points_unsmoothed)
            np.save(score_path, result.score_points)
            row.update(
                {
                    "matched_latitudes_deg_npy": _display_path(matched_path),
                    "score_points_unsmoothed_npy": _display_path(score_raw_path),
                    "score_points_npy": _display_path(score_path),
                }
            )
        rows.append(row)

    with (output_dir / "selected_buckets.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "pressure_level_hpa",
            "white_center",
            "selected_bucket_lower",
            "selected_bucket_upper",
            "selected_bucket_count",
            "middle_60_score_lower",
            "middle_60_score_upper",
        ]
        if args.write_arrays:
            fieldnames.extend(
                [
                    "matched_latitudes_deg_npy",
                    "score_points_unsmoothed_npy",
                    "score_points_npy",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "canonical same-longitude same-hemisphere Thermal Displacement",
        "dataset": _display_path(args.dataset),
        "climatology": _display_path(args.climatology),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "score_smooth_sigma_cells": args.score_smooth_sigma_cells,
        "same_hemisphere_matching": not args.allow_cross_hemisphere,
        "arrays_written": bool(args.write_arrays),
        "ordering": (
            "match raw temperature within same pressure/longitude/hemisphere, "
            "convert matched latitude to score, then smooth score"
        ),
        "score_definition": "score_points = (1 - abs(matched_latitude) / max_abs_latitude) * 100",
        "bucket_rule": "rarest nonzero 1-point score bucket inside each level's range-based middle 60%",
        "levels": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {_display_path(output_dir)}")


if __name__ == "__main__":
    main()
