from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY = Path("data/era5_temperature-climatology_1990-2020_11-08_12.nc")
DEFAULT_BORDER_GEOJSON = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
DEFAULT_OUTPUT_DIR = Path(
    "tmp/temperature-equivalent-latitude-process/output/"
    "matched-score-smoothed-range-tiebreak"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match each raw-temperature grid cell to the climatology latitude at "
            "the same pressure level and longitude with the nearest temperature, "
            "then plot histograms and equivalent-latitude maps for every level."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smooth-sigma-cells",
        type=float,
        default=1.0,
        help=(
            "Gaussian smoothing sigma applied to the thermal-displacement score "
            "after same-longitude climatology matching."
        ),
    )
    parser.add_argument(
        "--border-geojson",
        type=Path,
        default=DEFAULT_BORDER_GEOJSON,
        help="GeoJSON country polygon file used for land/border outlines.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output plot DPI.",
    )
    parser.add_argument(
        "--pressure-levels",
        type=str,
        default="",
        help=(
            "Optional comma-separated hPa levels. Empty means every pressure "
            "level in the raw temperature file."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = (Path.cwd() / expanded).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(f"Could not find {path.as_posix()}")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        pass
    if resolved.is_relative_to(Path("/tmp")):
        return resolved.as_posix()
    return path.name


def choose_timestamp(temperature: xr.DataArray, timestamp_text: str) -> np.datetime64:
    if "valid_time" not in temperature.coords:
        raise ValueError("Raw temperature data has no valid_time coordinate.")
    valid_times = np.asarray(temperature.coords["valid_time"].values)
    requested = np.datetime64(timestamp_text)
    if requested in valid_times:
        return requested
    nearest_index = int(np.argmin(np.abs(valid_times - requested)))
    return np.datetime64(valid_times[nearest_index])


def parse_requested_levels(text: str, available_levels: np.ndarray) -> list[float]:
    if not text.strip():
        return [float(level) for level in available_levels]

    requested = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    available = np.asarray(available_levels, dtype=np.float64)
    selected: list[float] = []
    for level in requested:
        nearest_index = int(np.argmin(np.abs(available - level)))
        selected.append(float(available[nearest_index]))
    return selected


def validate_matching_grid(temperature: xr.DataArray, climatology: xr.DataArray) -> None:
    for coord_name in ("pressure_level", "latitude", "longitude"):
        source_coord = np.asarray(temperature.coords[coord_name].values)
        climatology_coord = np.asarray(climatology.coords[coord_name].values)
        if not np.array_equal(source_coord, climatology_coord):
            raise ValueError(
                f"Raw temperature and climatology {coord_name} coordinates differ."
            )


def target_longitude(lon: float, longitudes: np.ndarray) -> float:
    lon_min = float(np.nanmin(longitudes))
    lon_max = float(np.nanmax(longitudes))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def split_target_longitude_segments(
    points: list[tuple[float, float]],
    longitudes: np.ndarray,
) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for lon, lat in points:
        mapped_lon = target_longitude(float(lon), longitudes)
        if previous_lon is not None and abs(mapped_lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((mapped_lon, float(lat)))
        previous_lon = mapped_lon

    if len(segment) >= 2:
        yield segment


def load_border_segments(
    geojson_path: Path,
    longitudes: np.ndarray,
) -> list[list[tuple[float, float]]]:
    if not geojson_path.exists():
        return []

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    segments: list[list[tuple[float, float]]] = []

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
                points = [(float(lon), float(lat)) for lon, lat, *_ in ring]
                segments.extend(split_target_longitude_segments(points, longitudes))

    return segments


def match_equivalent_latitude_same_longitude(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)

    if raw.shape != climatology.shape:
        raise ValueError("Raw and climatology level slices must have the same shape.")

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


def smooth_wrapped_lon(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return np.asarray(field, dtype=np.float32)
    return gaussian_filter(
        np.asarray(field, dtype=np.float32),
        sigma=(sigma_cells, sigma_cells),
        mode=("nearest", "wrap"),
    ).astype(np.float32)


def equivalent_latitude_score_points(
    matched_latitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes_deg))), 1e-6)
    score = 1.0 - np.abs(matched_latitudes_deg) / max_abs_latitude
    return np.asarray(np.clip(score, 0.0, 1.0) * 100.0, dtype=np.float32)


def choose_middle_60_sparse_bucket(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float, float]:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite_values.size == 0:
        raise ValueError("Cannot choose a bucket from an empty field.")

    centers = np.arange(0.0, 101.0, 1.0, dtype=np.float32)
    bins = np.concatenate(([centers[0] - 0.5], centers + 0.5)).astype(np.float32)
    counts, edges = np.histogram(finite_values, bins=bins)
    value_min = float(np.nanmin(finite_values))
    value_max = float(np.nanmax(finite_values))
    middle_min = value_min + 0.20 * (value_max - value_min)
    middle_max = value_max - 0.20 * (value_max - value_min)
    middle_mask = (centers >= middle_min) & (centers <= middle_max) & (counts > 0)
    candidate_indices = np.flatnonzero(middle_mask)
    if candidate_indices.size == 0:
        candidate_indices = np.flatnonzero(counts > 0)
    if candidate_indices.size == 0:
        candidate_indices = np.arange(len(counts))

    selected_index = int(candidate_indices[np.argmin(counts[candidate_indices])])
    selected_lower = float(edges[selected_index])
    selected_upper = float(edges[selected_index + 1])
    selected_center = 0.5 * (selected_lower + selected_upper)
    return (
        counts,
        edges,
        float(middle_min),
        float(middle_max),
        selected_index,
        selected_lower,
        selected_center,
    )


def make_diverging_norm(
    values: np.ndarray,
    center: float,
) -> mcolors.TwoSlopeNorm:
    finite_values = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if not (vmin < center < vmax):
        vmin = min(vmin, center - 1.0)
        vmax = max(vmax, center + 1.0)
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def plot_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    middle_min: float,
    middle_max: float,
    selected_index: int,
    selected_lower: float,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.full(len(counts), "#4d7fb8", dtype=object)
    colors[(centers >= middle_min) & (centers <= middle_max)] = "#8aa9c7"
    colors[selected_index] = "#f2c744"

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.axvspan(middle_min, middle_max, color="#f6d365", alpha=0.18, zorder=0)
    ax.bar(centers, counts, width=0.92, color=colors, edgecolor="none")
    ax.axvline(middle_min, color="#262626", linewidth=1.2, linestyle="--", label="middle 60% lower")
    ax.axvline(middle_max, color="#262626", linewidth=1.2, linestyle=":", label="middle 60% upper")
    ax.set_xlim(-0.5, 100.5)
    ax.set_xlabel("Thermal-displacement score bucket center")
    ax.set_ylabel("Cell count")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement score buckets; selected white bucket "
        f"{selected_lower:g} to {selected_lower + 1:g}"
    )
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_equivalent_latitude_map(
    values: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    center: float,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    norm = make_diverging_norm(values, center=center)

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        values,
        cmap="bwr",
        norm=norm,
        shading="auto",
        rasterized=True,
    )

    for segment in border_segments:
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color="#171717", linewidth=0.35, alpha=0.75)

    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa match-then-smoothed thermal-displacement score "
        f"(white centered at {center:.1f})"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Thermal-displacement score points; 0 = polar-like, 100 = equator-like")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def slug_for_level(level_hpa: float) -> str:
    return f"{level_hpa:g}".replace(".", "p").replace("-", "m") + "hpa"


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(border_path, longitudes)

    histogram_dir = output_dir / "histograms"
    map_dir = output_dir / "maps"
    array_dir = output_dir / "arrays"
    histogram_dir.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    array_dir.mkdir(exist_ok=True)

    levels_summary: list[dict[str, object]] = []

    for level_hpa in selected_levels:
        slug = slug_for_level(level_hpa)
        print(f"Processing {level_hpa:g} hPa")

        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        clim_level = (
            climatology.sel(pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )

        matched = match_equivalent_latitude_same_longitude(raw_level, clim_level, latitudes)
        matched = np.clip(matched, -90.0, 90.0).astype(np.float32)
        score_points_raw = equivalent_latitude_score_points(matched, latitudes)
        score_points = smooth_wrapped_lon(score_points_raw, sigma_cells=args.smooth_sigma_cells)

        counts, edges, middle_min, middle_max, selected_index, selected_lower, selected_center = (
            choose_middle_60_sparse_bucket(score_points)
        )

        histogram_path = histogram_dir / f"histogram_{slug}.png"
        map_path = map_dir / f"map_{slug}.png"
        matched_latitude_path = array_dir / f"matched_latitude_{slug}.npy"
        raw_score_path = array_dir / f"thermal_displacement_score_points_unsmoothed_{slug}.npy"
        score_path = array_dir / f"thermal_displacement_score_points_{slug}.npy"

        np.save(matched_latitude_path, matched)
        np.save(raw_score_path, score_points_raw)
        np.save(score_path, score_points)
        plot_histogram(
            counts=counts,
            edges=edges,
            middle_min=middle_min,
            middle_max=middle_max,
            selected_index=selected_index,
            selected_lower=selected_lower,
            level_hpa=level_hpa,
            output_path=histogram_path,
            dpi=args.dpi,
        )
        plot_equivalent_latitude_map(
            values=score_points,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            center=selected_center,
            level_hpa=level_hpa,
            output_path=map_path,
            dpi=args.dpi,
        )

        levels_summary.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "histogram_png": display_path(histogram_path),
                "map_png": display_path(map_path),
                "matched_latitude_npy": display_path(matched_latitude_path),
                "thermal_displacement_score_points_unsmoothed_npy": display_path(raw_score_path),
                "thermal_displacement_score_points_npy": display_path(score_path),
                "middle_60_score_range": {
                    "lower": middle_min,
                    "upper": middle_max,
                },
                "selected_bucket": {
                    "lower": selected_lower,
                    "upper": selected_lower + 1.0,
                    "center_for_map_white": selected_center,
                    "count": int(counts[selected_index]),
                },
                "matched_latitude_min": float(np.nanmin(matched)),
                "matched_latitude_max": float(np.nanmax(matched)),
                "score_unsmoothed_min": float(np.nanmin(score_points_raw)),
                "score_unsmoothed_max": float(np.nanmax(score_points_raw)),
                "score_min": float(np.nanmin(score_points)),
                "score_max": float(np.nanmax(score_points)),
            }
        )

    summary = {
        "process": "same-longitude climatology thermal-displacement score smoothed after matching",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "climatology_variable": CLIMATOLOGY_VARIABLE,
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "score_smooth_sigma_cells": args.smooth_sigma_cells,
        "smoothing_order": "match raw temperature first, convert matched latitude to score, then smooth score",
        "matched_latitude_tie_breaker": (
            "When left and right climatology temperatures are equally close, "
            "choose the matched climatology latitude row closest to the source cell row."
        ),
        "score_definition": "score_points = (1 - abs(matched_latitude) / max_abs_latitude) * 100",
        "bucket_edges": "1-point score buckets centered on integer values from 0 to 100",
        "middle_60_definition": (
            "Middle 60% of each level's score range: min + 20% of range through "
            "max - 20% of range."
        ),
        "selected_bucket_rule": (
            "Fewest nonzero score bucket whose integer center falls inside the "
            "level's range-based middle 60%."
        ),
        "map_color_scale": (
            "blue-white-red TwoSlopeNorm over each level's score min/max, "
            "white at selected score bucket center"
        ),
        "border_geojson": display_path(border_path),
        "levels": levels_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "selected_buckets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pressure_level_hpa",
                "selected_bucket_lower",
                "selected_bucket_upper",
                "white_center",
                "selected_bucket_count",
                "middle_60_score_lower",
                "middle_60_score_upper",
                "score_min",
                "score_max",
                "score_unsmoothed_min",
                "score_unsmoothed_max",
                "matched_latitude_min",
                "matched_latitude_max",
            ]
        )
        for item in levels_summary:
            bucket = item["selected_bucket"]
            middle_range = item["middle_60_score_range"]
            writer.writerow(
                [
                    item["pressure_level_hpa"],
                    bucket["lower"],
                    bucket["upper"],
                    bucket["center_for_map_white"],
                    bucket["count"],
                    middle_range["lower"],
                    middle_range["upper"],
                    item["score_min"],
                    item["score_max"],
                    item["score_unsmoothed_min"],
                    item["score_unsmoothed_max"],
                    item["matched_latitude_min"],
                    item["matched_latitude_max"],
                ]
            )
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
