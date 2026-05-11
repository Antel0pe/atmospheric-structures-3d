from __future__ import annotations

import argparse
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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_CLIMATOLOGY = Path("data/era5_temperature-climatology_1990-2020_11-08_12.nc")
DEFAULT_OUTPUT_DIR = Path("./tmp/thermal-displacement-gradient")
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"
DEFAULT_BORDER_GEOJSON_PATH = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot thermal-displacement equivalent latitude and its horizontal "
            "gradient on ERA5 pressure levels. The gradient uses real Earth "
            "distances, not raw grid-cell counts."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--matching-mode",
        type=str,
        choices=("same-longitude", "zonal-mean", "zonal-middle-50"),
        default="same-longitude",
        help=(
            "Use same-longitude to match against each climatology longitude column, "
            "zonal-mean to average climatology over longitude before matching, "
            "or zonal-middle-50 to average only the middle 50% of climatology "
            "longitudes at each latitude."
        ),
    )
    parser.add_argument(
        "--hemisphere-constrained",
        action="store_true",
        help=(
            "Only match a source cell against climatology latitudes in the same "
            "hemisphere as that source cell."
        ),
    )
    parser.add_argument(
        "--latitude-match-method",
        type=str,
        choices=("nearest", "interpolate"),
        default="nearest",
        help=(
            "Use nearest to snap to the closest climatology temperature, or "
            "interpolate to linearly estimate equivalent latitude between the "
            "two bracketing climatology temperatures."
        ),
    )
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--levels-hpa",
        type=str,
        default=",".join(str(level) for level in DEFAULT_LEVELS_HPA),
        help="Comma-separated pressure levels in hPa.",
    )
    parser.add_argument(
        "--temperature-smooth-sigma-cells",
        type=float,
        default=1.0,
        help=(
            "Gaussian smoothing sigma applied to raw temperature before thermal "
            "displacement is computed."
        ),
    )
    parser.add_argument(
        "--gradient-smooth-sigma-cells",
        type=float,
        default=1.0,
        help=(
            "Gaussian smoothing sigma applied to absolute matched latitude before "
            "the gradient is computed."
        ),
    )
    parser.add_argument(
        "--border-geojson",
        type=Path,
        default=DEFAULT_BORDER_GEOJSON_PATH,
        help="GeoJSON country/land-border source drawn over each map.",
    )
    parser.add_argument(
        "--gradient-display-percentile",
        type=float,
        default=99.5,
        help=(
            "Upper gradient color limit percentile. Use 100 for strict true-max "
            "scaling. Values above the limit are also yellow."
        ),
    )
    parser.add_argument(
        "--gradient-component",
        type=str,
        choices=("horizontal", "meridional"),
        default="horizontal",
        help=(
            "Use horizontal for sqrt(dx^2 + dy^2), or meridional for the "
            "absolute north-south derivative only."
        ),
    )
    parser.add_argument(
        "--gradient-distance-basis",
        type=str,
        choices=("physical", "grid-neighbor"),
        default="physical",
        help=(
            "Use physical to scale by Earth distance, or grid-neighbor to measure "
            "equivalent-latitude change per native neighboring grid cell."
        ),
    )
    return parser.parse_args()


def format_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        home = Path.home()
        try:
            return f"~/{path.resolve().relative_to(home).as_posix()}"
        except ValueError:
            return path.name or "<external-path>"


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        return resolved
    repo_relative = (Path.cwd() / path).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(f"File not found: {format_display_path(resolved)}")


def parse_levels(text: str) -> list[float]:
    levels = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not levels:
        raise ValueError("At least one pressure level is required.")
    return levels


def choose_timestamp(temperature: xr.DataArray, timestamp_text: str) -> np.datetime64:
    valid_times = np.asarray(temperature.coords["valid_time"].values)
    requested = np.datetime64(timestamp_text)
    if requested in valid_times:
        return requested
    nearest_index = int(np.argmin(np.abs(valid_times - requested)))
    return np.datetime64(valid_times[nearest_index])


def validate_matching_grid(temperature: xr.DataArray, climatology: xr.DataArray) -> None:
    for coord_name in ("pressure_level", "latitude", "longitude"):
        source_coord = np.asarray(temperature.coords[coord_name].values)
        climatology_coord = np.asarray(climatology.coords[coord_name].values)
        if not np.array_equal(source_coord, climatology_coord):
            raise ValueError(f"Source temperature and climatology {coord_name} grids differ.")


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
        target_lon = target_longitude(float(lon), longitudes)
        if previous_lon is not None and abs(target_lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((target_lon, float(lat)))
        previous_lon = target_lon

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


def match_profile_latitudes_deg(
    source_temperatures_k: np.ndarray,
    source_row_indices: np.ndarray,
    climatology_profile_k: np.ndarray,
    latitudes_deg: np.ndarray,
    hemisphere_constrained: bool,
    latitude_match_method: str,
) -> np.ndarray:
    source = np.asarray(source_temperatures_k, dtype=np.float32).reshape(-1)
    rows = np.asarray(source_row_indices, dtype=np.int32).reshape(-1)
    profile = np.asarray(climatology_profile_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    result = np.full(source.shape, np.nan, dtype=np.float32)

    if hemisphere_constrained:
        source_groups = (latitudes[rows] >= 0.0, latitudes[rows] < 0.0)
        candidate_groups = (latitudes >= 0.0, latitudes < 0.0)
    else:
        source_groups = (np.ones(source.shape, dtype=bool),)
        candidate_groups = (np.ones(latitudes.shape, dtype=bool),)

    for source_group, candidate_group in zip(source_groups, candidate_groups):
        valid_candidates = np.isfinite(profile) & candidate_group
        valid_sources = np.isfinite(source) & source_group
        if not np.any(valid_candidates) or not np.any(valid_sources):
            continue

        candidate_indices = np.flatnonzero(valid_candidates)
        order = candidate_indices[
            np.argsort(profile[valid_candidates], kind="mergesort")
        ]
        sorted_temperatures = profile[order]

        source_indices = np.flatnonzero(valid_sources)
        source_values = source[source_indices]
        insert_positions = np.searchsorted(sorted_temperatures, source_values)
        right_positions = np.clip(insert_positions, 0, sorted_temperatures.size - 1)
        left_positions = np.clip(insert_positions - 1, 0, sorted_temperatures.size - 1)

        right_indices = order[right_positions]
        left_indices = order[left_positions]
        if latitude_match_method == "interpolate":
            left_temperatures = sorted_temperatures[left_positions]
            right_temperatures = sorted_temperatures[right_positions]
            denominator = right_temperatures - left_temperatures
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = (source_values - left_temperatures) / denominator
            weight = np.clip(weight, 0.0, 1.0)
            equal_temperature = np.abs(denominator) < 1.0e-6
            left_latitudes = latitudes[left_indices]
            right_latitudes = latitudes[right_indices]
            interpolated = left_latitudes + weight * (right_latitudes - left_latitudes)
            if np.any(equal_temperature):
                row_indices = rows[source_indices]
                right_lat_distance = np.abs(right_indices - row_indices)
                left_lat_distance = np.abs(left_indices - row_indices)
                use_left = left_lat_distance <= right_lat_distance
                interpolated[equal_temperature] = np.where(
                    use_left[equal_temperature],
                    left_latitudes[equal_temperature],
                    right_latitudes[equal_temperature],
                )
            result[source_indices] = interpolated.astype(np.float32)
            continue

        if latitude_match_method != "nearest":
            raise ValueError(f"Unsupported latitude match method: {latitude_match_method}")

        right_diffs = np.abs(sorted_temperatures[right_positions] - source_values)
        left_diffs = np.abs(sorted_temperatures[left_positions] - source_values)
        row_indices = rows[source_indices]
        right_lat_distance = np.abs(right_indices - row_indices)
        left_lat_distance = np.abs(left_indices - row_indices)
        use_left = (left_diffs < right_diffs) | (
            (left_diffs == right_diffs) & (left_lat_distance <= right_lat_distance)
        )
        matched_indices = np.where(use_left, left_indices, right_indices)
        result[source_indices] = latitudes[matched_indices]

    return result


def matched_climatology_latitude_deg(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    hemisphere_constrained: bool,
    latitude_match_method: str,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    result = np.full_like(raw, np.nan, dtype=np.float32)
    row_indices = np.arange(raw.shape[0], dtype=np.int32)

    for lon_index in range(raw.shape[1]):
        result[:, lon_index] = match_profile_latitudes_deg(
            raw[:, lon_index],
            row_indices,
            climatology[:, lon_index],
            latitudes_deg,
            hemisphere_constrained,
            latitude_match_method,
        )

    return result


def climatology_zonal_mean_profile(
    climatology_temperature_k: np.ndarray,
) -> np.ndarray:
    return np.asarray(np.nanmean(climatology_temperature_k, axis=1), dtype=np.float32)


def climatology_zonal_middle_profile(
    climatology_temperature_k: np.ndarray,
    trim_fraction: float,
) -> np.ndarray:
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    lower = np.nanquantile(climatology, trim_fraction, axis=1)
    upper = np.nanquantile(climatology, 1.0 - trim_fraction, axis=1)
    middle = np.where(
        (climatology >= lower[:, None]) & (climatology <= upper[:, None]),
        climatology,
        np.nan,
    )
    return np.asarray(np.nanmean(middle, axis=1), dtype=np.float32)


def matched_zonal_mean_climatology_latitude_deg(
    raw_temperature_k: np.ndarray,
    climatology_latitude_profile_k: np.ndarray,
    latitudes_deg: np.ndarray,
    hemisphere_constrained: bool,
    latitude_match_method: str,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    row_indices = np.repeat(np.arange(raw.shape[0], dtype=np.int32), raw.shape[1])
    matched = match_profile_latitudes_deg(
        raw.reshape(-1),
        row_indices,
        climatology_latitude_profile_k,
        latitudes_deg,
        hemisphere_constrained,
        latitude_match_method,
    )
    return matched.reshape(raw.shape)


def smooth_lat_lon(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    sigma = max(float(sigma_cells), 0.0)
    if sigma == 0.0:
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


def displacement_gradient_per_100km(
    values: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    component: str,
) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    lat_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))

    dvalue_d_lat_rad = np.gradient(field, lat_rad, axis=0, edge_order=2)
    dvalue_dy = dvalue_d_lat_rad / EARTH_RADIUS_M

    mean_dlon_rad = float(np.mean(np.abs(np.diff(lon_rad))))
    east = np.roll(field, -1, axis=1)
    west = np.roll(field, 1, axis=1)
    dvalue_d_lon_rad = (east - west) / (2.0 * mean_dlon_rad)
    cos_lat = np.cos(lat_rad)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        dvalue_dx = dvalue_d_lon_rad / (EARTH_RADIUS_M * cos_lat)

    if component == "horizontal":
        gradient = np.sqrt(np.square(dvalue_dx) + np.square(dvalue_dy)) * 100_000.0
    elif component == "meridional":
        gradient = np.abs(dvalue_dy) * 100_000.0
    else:
        raise ValueError(f"Unsupported gradient component: {component}")

    gradient[~np.isfinite(gradient)] = np.nan
    return np.asarray(gradient, dtype=np.float32)


def displacement_gradient_per_grid_cell(
    values: np.ndarray,
    component: str,
) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    dvalue_drow = np.gradient(field, axis=0, edge_order=2)
    east = np.roll(field, -1, axis=1)
    west = np.roll(field, 1, axis=1)
    dvalue_dcol = (east - west) * 0.5

    if component == "horizontal":
        gradient = np.sqrt(np.square(dvalue_dcol) + np.square(dvalue_drow))
    elif component == "meridional":
        gradient = np.abs(dvalue_drow)
    else:
        raise ValueError(f"Unsupported gradient component: {component}")

    gradient[~np.isfinite(gradient)] = np.nan
    return np.asarray(gradient, dtype=np.float32)


def black_to_yellow_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "black_to_yellow",
        ["#000000", "#ffff00"],
    )


def plot_field_map(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str | mcolors.Colormap,
    vmin: float,
    vmax: float,
    border_segments: list[list[tuple[float, float]]] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="upper",
        extent=(
            float(np.nanmin(longitudes)),
            float(np.nanmax(longitudes)),
            float(np.nanmin(latitudes)),
            float(np.nanmax(latitudes)),
        ),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(longitudes)), float(np.nanmax(longitudes)))
    ax.set_ylim(float(np.nanmin(latitudes)), float(np.nanmax(latitudes)))
    if border_segments:
        for segment in border_segments:
            xs = [point[0] for point in segment]
            ys = [point[1] for point in segment]
            ax.plot(xs, ys, color=(1.0, 1.0, 1.0, 0.7), linewidth=0.75, zorder=3)
            ax.plot(xs, ys, color=(0.0, 0.0, 0.0, 0.82), linewidth=0.35, zorder=4)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_geojson_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_levels = parse_levels(args.levels_hpa)
    summary: dict[str, object] = {
        "method": {
            "field": f"{args.matching_mode} thermal-displacement equivalent latitude",
            "matching_mode": args.matching_mode,
            "hemisphere_constrained": bool(args.hemisphere_constrained),
            "latitude_match_method": args.latitude_match_method,
            "thermal_displacement": (
                "Lightly smooth raw temperature first, then choose the climatology "
                "latitude that best matches the source temperature. In same-longitude "
                "mode the lookup uses each pressure/longitude climatology column. In "
                "zonal-mean mode the lookup first averages climatology over all "
                "longitudes for each latitude. In zonal-middle-50 mode it discards "
                "the coldest and warmest 25% of climatology longitudes at each "
                "latitude, then averages the middle 50%. Hemisphere-constrained mode "
                "only uses candidate climatology latitudes in the source cell's "
                "hemisphere. Nearest mode snaps to the closest climatology "
                "temperature; interpolate mode estimates equivalent latitude between "
                "the two bracketing climatology temperatures."
            ),
            "temperature_smoothing": {
                "type": "Gaussian",
                "sigma_native_grid_cells": float(args.temperature_smooth_sigma_cells),
            },
            "thermal_displacement_plot": (
                "inferno color scale over 1 - abs(matched latitude) / max(abs(latitude)); "
                "larger values are more equator-like, smaller values are more pole-like"
            ),
            "gradient": {
                "field": (
                    "abs(matched climatology latitude), north-south derivative only"
                    if args.gradient_component == "meridional"
                    else "abs(matched climatology latitude), horizontal gradient magnitude"
                ),
                "units": (
                    "degrees equivalent latitude / native grid cell"
                    if args.gradient_distance_basis == "grid-neighbor"
                    else "degrees equivalent latitude / 100 km"
                ),
                "distance_basis": (
                    "native latitude/longitude grid-neighbor steps, with no physical-distance scaling"
                    if args.gradient_distance_basis == "grid-neighbor"
                    else "real Earth distances between latitude/longitude grid-cell centers"
                ),
                "component": args.gradient_component,
                "distance_mode": args.gradient_distance_basis,
                "derivative": (
                    "centered finite difference in latitude only"
                    if args.gradient_component == "meridional"
                    else (
                        "centered finite difference using adjacent native grid cells "
                        "with longitude wrapping; native row/column steps are treated "
                        "as equal unit distances"
                        if args.gradient_distance_basis == "grid-neighbor"
                        else "centered finite difference using adjacent native grid cells; longitude wraps at dateline"
                    )
                ),
                "smoothing": {
                    "type": "Gaussian",
                    "sigma_native_grid_cells": float(args.gradient_smooth_sigma_cells),
                },
                "color_scale": {
                    "minimum": 0.0,
                    "maximum": (
                        "true finite maximum"
                        if float(args.gradient_display_percentile) >= 100.0
                        else f"finite p{float(args.gradient_display_percentile):g}; larger values clipped to yellow"
                    ),
                },
            },
        },
        "dataset": format_display_path(dataset_path),
        "climatology": format_display_path(climatology_path),
        "border_geojson": format_display_path(border_geojson_path),
        "requested_timestamp": args.timestamp,
        "levels": [],
    }

    with xr.open_dataset(dataset_path) as source, xr.open_dataset(climatology_path) as clim:
        if TEMPERATURE_VARIABLE not in source:
            raise KeyError(f"Expected variable {TEMPERATURE_VARIABLE!r}.")
        if CLIMATOLOGY_VARIABLE not in clim:
            raise KeyError(f"Expected variable {CLIMATOLOGY_VARIABLE!r}.")

        temperature = source[TEMPERATURE_VARIABLE]
        climatology = clim[CLIMATOLOGY_VARIABLE]
        validate_matching_grid(temperature, climatology)

        timestamp = choose_timestamp(temperature, args.timestamp)
        selected_time = temperature.sel(valid_time=timestamp)
        pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float32)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(border_geojson_path, longitudes)
        max_abs_latitude = max(float(np.nanmax(np.abs(latitudes))), 1e-6)

        summary["actual_timestamp"] = np.datetime_as_string(timestamp, unit="m")
        summary["grid"] = {
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_step_degrees": float(abs(latitudes[1] - latitudes[0])),
            "longitude_step_degrees": float(abs(longitudes[1] - longitudes[0])),
            "nominal_latitude_spacing_km": float(
                EARTH_RADIUS_M * np.deg2rad(abs(latitudes[1] - latitudes[0])) / 1000.0
            ),
            "nominal_equator_longitude_spacing_km": float(
                EARTH_RADIUS_M * np.deg2rad(abs(longitudes[1] - longitudes[0])) / 1000.0
            ),
        }

        for requested_level in requested_levels:
            level_slice = selected_time.sel(pressure_level=requested_level, method="nearest")
            actual_level = float(level_slice.coords["pressure_level"].item())
            level_index = int(np.argmin(np.abs(pressure_levels - actual_level)))
            raw_temperature = np.asarray(level_slice.values, dtype=np.float32)
            source_temperature = smooth_lat_lon(
                raw_temperature,
                args.temperature_smooth_sigma_cells,
            )
            climatology_temperature = np.asarray(climatology.values[level_index], dtype=np.float32)
            zonal_profile = (
                climatology_zonal_middle_profile(climatology_temperature, trim_fraction=0.25)
                if args.matching_mode == "zonal-middle-50"
                else climatology_zonal_mean_profile(climatology_temperature)
            )

            if args.matching_mode in {"zonal-mean", "zonal-middle-50"}:
                matched_latitude = matched_zonal_mean_climatology_latitude_deg(
                    source_temperature,
                    zonal_profile,
                    latitudes,
                    args.hemisphere_constrained,
                    args.latitude_match_method,
                )
            else:
                matched_latitude = matched_climatology_latitude_deg(
                    source_temperature,
                    climatology_temperature,
                    latitudes,
                    args.hemisphere_constrained,
                    args.latitude_match_method,
                )
            abs_matched_latitude = np.abs(matched_latitude)
            displacement_score = np.asarray(
                1.0 - abs_matched_latitude / max_abs_latitude,
                dtype=np.float32,
            )
            displacement_score[~np.isfinite(matched_latitude)] = np.nan

            smoothed_abs_latitude = smooth_lat_lon(
                abs_matched_latitude,
                args.gradient_smooth_sigma_cells,
            )
            if args.gradient_distance_basis == "grid-neighbor":
                gradient = displacement_gradient_per_grid_cell(
                    smoothed_abs_latitude,
                    args.gradient_component,
                )
                gradient_units_label = "deg / native grid cell"
                gradient_summary_suffix = "deg_per_grid_cell"
            else:
                gradient = displacement_gradient_per_100km(
                    smoothed_abs_latitude,
                    latitudes,
                    longitudes,
                    args.gradient_component,
                )
                gradient_units_label = "deg / 100 km"
                gradient_summary_suffix = "deg_per_100km"

            finite_gradient = gradient[np.isfinite(gradient)]
            if finite_gradient.size == 0:
                raise ValueError(f"No finite gradients for {actual_level:.0f} hPa.")
            if float(args.gradient_display_percentile) >= 100.0:
                gradient_vmax = float(np.nanmax(finite_gradient))
            else:
                gradient_vmax = float(
                    np.nanpercentile(finite_gradient, args.gradient_display_percentile)
                )

            level_slug = f"{int(round(actual_level)):04d}hpa"
            climatology_image_name = (
                f"climatology-zonal-middle-50-{level_slug}.png"
                if args.matching_mode == "zonal-middle-50"
                else f"climatology-zonal-mean-{level_slug}.png"
            )
            displacement_image_name = f"thermal-displacement-{level_slug}.png"
            gradient_image_name = (
                f"thermal-displacement-gradient-meridional-{level_slug}.png"
                if args.gradient_component == "meridional"
                else (
                    f"thermal-displacement-gradient-grid-neighbor-{level_slug}.png"
                    if args.gradient_distance_basis == "grid-neighbor"
                    else f"thermal-displacement-gradient-{level_slug}.png"
                )
            )

            if args.matching_mode in {"zonal-mean", "zonal-middle-50"}:
                climatology_profile_map = np.repeat(
                    zonal_profile[:, None],
                    longitudes.size,
                    axis=1,
                )
                finite_profile = climatology_profile_map[
                    np.isfinite(climatology_profile_map)
                ]
                plot_field_map(
                    climatology_profile_map,
                    latitudes,
                    longitudes,
                    output_dir / climatology_image_name,
                    (
                        f"{actual_level:.0f} hPa "
                        f"{'middle-50% ' if args.matching_mode == 'zonal-middle-50' else ''}"
                        "zonal temperature climatology, "
                        f"{np.datetime_as_string(timestamp, unit='m')} UTC"
                    ),
                    (
                        "Middle-50% longitude-mean temperature climatology (K)"
                        if args.matching_mode == "zonal-middle-50"
                        else "Longitude-mean temperature climatology (K)"
                    ),
                    "inferno",
                    float(np.nanmin(finite_profile)),
                    float(np.nanmax(finite_profile)),
                    border_segments,
                )

            plot_field_map(
                displacement_score,
                latitudes,
                longitudes,
                output_dir / displacement_image_name,
                (
                    f"{actual_level:.0f} hPa thermal displacement, "
                    f"{np.datetime_as_string(timestamp, unit='m')} UTC"
                ),
                "Thermal displacement score (equator-like 1, pole-like 0)",
                "inferno",
                0.0,
                1.0,
                border_segments,
            )
            plot_field_map(
                gradient,
                latitudes,
                longitudes,
                output_dir / gradient_image_name,
                (
                    f"{actual_level:.0f} hPa "
                    f"{'north-south ' if args.gradient_component == 'meridional' else ''}"
                    "thermal-displacement gradient, "
                    f"{np.datetime_as_string(timestamp, unit='m')} UTC"
                ),
                (
                    f"North-south gradient of abs(matched latitude) ({gradient_units_label})"
                    if args.gradient_component == "meridional"
                    else f"Gradient of abs(matched latitude) ({gradient_units_label})"
                ),
                black_to_yellow_colormap(),
                0.0,
                gradient_vmax,
                border_segments,
            )

            finite_abs_latitude = abs_matched_latitude[np.isfinite(abs_matched_latitude)]
            summary["levels"].append(
                {
                    "requested_pressure_hpa": float(requested_level),
                    "actual_pressure_hpa": actual_level,
                    "climatology_zonal_profile_plot": (
                        climatology_image_name
                        if args.matching_mode in {"zonal-mean", "zonal-middle-50"}
                        else None
                    ),
                    "thermal_displacement_plot": displacement_image_name,
                    "gradient_plot": gradient_image_name,
                    "abs_matched_latitude_p10_deg": float(
                        np.nanpercentile(finite_abs_latitude, 10.0)
                    ),
                    "abs_matched_latitude_p50_deg": float(
                        np.nanpercentile(finite_abs_latitude, 50.0)
                    ),
                    "abs_matched_latitude_p90_deg": float(
                        np.nanpercentile(finite_abs_latitude, 90.0)
                    ),
                    f"gradient_min_{gradient_summary_suffix}": float(np.nanmin(finite_gradient)),
                    f"gradient_p50_{gradient_summary_suffix}": float(
                        np.nanpercentile(finite_gradient, 50.0)
                    ),
                    f"gradient_p90_{gradient_summary_suffix}": float(
                        np.nanpercentile(finite_gradient, 90.0)
                    ),
                    f"gradient_p95_{gradient_summary_suffix}": float(
                        np.nanpercentile(finite_gradient, 95.0)
                    ),
                    f"gradient_p99_{gradient_summary_suffix}": float(
                        np.nanpercentile(finite_gradient, 99.0)
                    ),
                    f"gradient_max_{gradient_summary_suffix}": float(np.nanmax(finite_gradient)),
                    f"gradient_plot_vmax_{gradient_summary_suffix}": gradient_vmax,
                }
            )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote thermal-displacement maps to {format_display_path(output_dir)}")


if __name__ == "__main__":
    main()
