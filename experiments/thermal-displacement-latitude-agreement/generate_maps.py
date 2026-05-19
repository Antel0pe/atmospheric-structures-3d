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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT_DIR = Path("tmp/thermal-displacement-latitude-agreement/output")
DEFAULT_LEVELS = "1000,850,500,250"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thermal-displacement maps and maps marking cells whose "
            "source latitude is close to the matched climatology latitude."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--matching-mode",
        choices=("global-latitudes", "same-hemisphere"),
        default="global-latitudes",
        help=(
            "global-latitudes compares against all climatology latitudes at the "
            "same longitude. same-hemisphere compares northern source cells only "
            "against climatology latitudes >= 0 and southern source cells only "
            "against climatology latitudes < 0."
        ),
    )
    parser.add_argument("--smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument(
        "--matched-latitude-smooth-sigma-cells",
        type=float,
        default=0.0,
        help=(
            "Optional Gaussian smoothing applied to the signed matched latitude "
            "field before converting it to score and before the latitude "
            "agreement comparison."
        ),
    )
    parser.add_argument("--agreement-degrees", type=float, default=5.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write CSVs and histogram diagnostics.",
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


def match_equivalent_latitude_same_longitude_same_hemisphere(
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
    source_rows = np.arange(n_lat)
    hemisphere_masks = (latitudes >= 0.0, latitudes < 0.0)

    for lon_index in range(n_lon):
        profile = climatology[:, lon_index]
        source_values = raw[:, lon_index]

        for hemisphere_mask in hemisphere_masks:
            source_indices = np.flatnonzero(hemisphere_mask)
            if source_indices.size == 0:
                continue

            candidate_rows = np.flatnonzero(hemisphere_mask)
            candidate_values = profile[candidate_rows]
            order = np.argsort(candidate_values, kind="mergesort")
            sorted_candidate_rows = candidate_rows[order]
            sorted_values = candidate_values[order]
            sorted_latitudes = latitudes[sorted_candidate_rows]
            hemisphere_source_values = source_values[source_indices]

            insertion = np.searchsorted(
                sorted_values,
                hemisphere_source_values,
                side="left",
            )
            lower = np.clip(insertion - 1, 0, sorted_values.size - 1)
            upper = np.clip(insertion, 0, sorted_values.size - 1)

            lower_distance = np.abs(hemisphere_source_values - sorted_values[lower])
            upper_distance = np.abs(hemisphere_source_values - sorted_values[upper])
            lower_row_distance = np.abs(sorted_candidate_rows[lower] - source_rows[source_indices])
            upper_row_distance = np.abs(sorted_candidate_rows[upper] - source_rows[source_indices])
            choose_upper = (upper_distance < lower_distance) | (
                (upper_distance == lower_distance)
                & (upper_row_distance < lower_row_distance)
            )
            nearest = np.where(choose_upper, upper, lower)
            matched[source_indices, lon_index] = sorted_latitudes[nearest]

    return matched


def match_equivalent_latitude(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
    matching_mode: str,
) -> np.ndarray:
    if matching_mode == "global-latitudes":
        return match_equivalent_latitude_same_longitude(
            raw_temperature_k,
            climatology_temperature_k,
            latitudes_deg,
        )
    if matching_mode == "same-hemisphere":
        return match_equivalent_latitude_same_longitude_same_hemisphere(
            raw_temperature_k,
            climatology_temperature_k,
            latitudes_deg,
        )
    raise ValueError(f"Unsupported matching mode: {matching_mode}")


def smooth_wrapped_lon(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return np.asarray(field, dtype=np.float32)
    return gaussian_filter(
        np.asarray(field, dtype=np.float32),
        sigma=(sigma_cells, sigma_cells),
        mode=("nearest", "wrap"),
    ).astype(np.float32)


def thermal_displacement_score_points(
    matched_latitudes_deg: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes_deg))), 1e-6)
    score = 1.0 - np.abs(matched_latitudes_deg) / max_abs_latitude
    return np.asarray(np.clip(score, 0.0, 1.0) * 100.0, dtype=np.float32)


def signed_latitude_from_same_hemisphere_score(
    score_points: np.ndarray,
    source_latitudes_deg: np.ndarray,
) -> np.ndarray:
    max_abs_latitude = max(float(np.nanmax(np.abs(source_latitudes_deg))), 1e-6)
    absolute_latitude = (1.0 - np.clip(score_points, 0.0, 100.0) / 100.0) * max_abs_latitude
    source_sign = np.where(source_latitudes_deg[:, np.newaxis] >= 0.0, 1.0, -1.0)
    return np.asarray(absolute_latitude * source_sign, dtype=np.float32)


def draw_borders(
    ax: plt.Axes,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    for segment in border_segments:
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color="#151515", linewidth=0.35, alpha=0.75)


def plot_base_map(
    values: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
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
    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal displacement; blue polar-like, red equator-like"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Thermal-displacement score; 0 = polar-like, 100 = equator-like")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_agreement_map(
    values: np.ndarray,
    agreement_mask: np.ndarray,
    agreement_degrees: float,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
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

    masked_agreement = np.ma.masked_where(~agreement_mask, agreement_mask)
    green_cmap = mcolors.ListedColormap(["#13b84a"])
    ax.pcolormesh(
        longitudes,
        latitudes,
        masked_agreement,
        cmap=green_cmap,
        shading="auto",
        rasterized=True,
        alpha=0.86,
    )

    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa green cells: source latitude within "
        f"{agreement_degrees:g} deg of matched climatology latitude"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Thermal-displacement score where not green")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_latitude_difference_histogram(
    signed_difference_deg: np.ndarray,
    agreement_degrees: float,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite_difference = signed_difference_deg[np.isfinite(signed_difference_deg)]
    centers = np.arange(-180.0, 181.0, 1.0, dtype=np.float32)
    bins = np.concatenate(([centers[0] - 0.5], centers + 0.5)).astype(np.float32)
    counts, edges = np.histogram(finite_difference, bins=bins)
    bar_centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.full(len(counts), "#557fae", dtype=object)
    colors[np.abs(bar_centers) <= agreement_degrees] = "#13b84a"

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.bar(bar_centers, counts, width=0.94, color=colors, edgecolor="none")
    ax.axvline(-agreement_degrees, color="#171717", linewidth=1.1, linestyle="--")
    ax.axvline(agreement_degrees, color="#171717", linewidth=1.1, linestyle="--")
    ax.axvline(0.0, color="#171717", linewidth=1.2)
    ax.set_xlim(-180.5, 180.5)
    ax.set_xlabel("Source latitude minus matched climatology latitude (degrees)")
    ax.set_ylabel("Cell count")
    ax.set_title(
        f"{level_hpa:g} hPa latitude difference histogram; "
        f"green is within +/-{agreement_degrees:g} deg"
    )
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return counts, edges


def plot_score_histogram(
    score_points: np.ndarray,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite_scores = score_points[np.isfinite(score_points)]
    centers = np.arange(0.0, 101.0, 1.0, dtype=np.float32)
    bins = np.concatenate(([centers[0] - 0.5], centers + 0.5)).astype(np.float32)
    counts, edges = np.histogram(finite_scores, bins=bins)
    bar_centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.bar(bar_centers, counts, width=0.94, color="#5f87b8", edgecolor="none")
    ax.axvline(50.0, color="#171717", linewidth=1.2)
    ax.set_xlim(-0.5, 100.5)
    ax.set_xlabel("Smoothed thermal-displacement score")
    ax.set_ylabel("Cell count")
    ax.set_title(f"{level_hpa:g} hPa smoothed thermal-displacement score histogram")
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return counts, edges


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

    thermal_map_dir = output_dir / "thermal-displacement-maps"
    agreement_map_dir = output_dir / "latitude-agreement-green-maps"
    thermal_map_dir.mkdir(exist_ok=True)
    agreement_map_dir.mkdir(exist_ok=True)
    if args.write_diagnostics:
        histogram_dir = output_dir / "latitude-difference-histograms"
        score_histogram_dir = output_dir / "score-histograms"
        histogram_dir.mkdir(exist_ok=True)
        score_histogram_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []

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

        matched_latitude = match_equivalent_latitude(
            raw_level,
            clim_level,
            latitudes,
            args.matching_mode,
        )
        matched_latitude = np.clip(matched_latitude, -90.0, 90.0).astype(np.float32)
        matched_latitude_for_comparison = smooth_wrapped_lon(
            matched_latitude,
            args.matched_latitude_smooth_sigma_cells,
        )
        score_unsmoothed = thermal_displacement_score_points(
            matched_latitude_for_comparison,
            latitudes,
        )
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, args.smooth_sigma_cells)
        source_latitudes = latitudes[:, np.newaxis]
        matched_latitude_after_score_smoothing = signed_latitude_from_same_hemisphere_score(
            score_smoothed,
            latitudes,
        )
        signed_latitude_difference = source_latitudes - matched_latitude_after_score_smoothing
        absolute_latitude_difference = np.abs(signed_latitude_difference)
        agreement_mask = absolute_latitude_difference <= args.agreement_degrees

        thermal_map_path = thermal_map_dir / f"thermal_displacement_{slug}.png"
        agreement_map_path = agreement_map_dir / f"latitude_agreement_green_{slug}.png"
        plot_base_map(
            values=score_smoothed,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            level_hpa=level_hpa,
            output_path=thermal_map_path,
            dpi=args.dpi,
        )
        plot_agreement_map(
            values=score_smoothed,
            agreement_mask=agreement_mask,
            agreement_degrees=args.agreement_degrees,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            level_hpa=level_hpa,
            output_path=agreement_map_path,
            dpi=args.dpi,
        )

        diagnostic_paths: dict[str, str] = {}
        if args.write_diagnostics:
            histogram_path = histogram_dir / f"latitude_difference_histogram_{slug}.png"
            score_histogram_path = score_histogram_dir / f"score_histogram_{slug}.png"
            difference_counts, difference_edges = plot_latitude_difference_histogram(
                signed_difference_deg=signed_latitude_difference,
                agreement_degrees=args.agreement_degrees,
                level_hpa=level_hpa,
                output_path=histogram_path,
                dpi=args.dpi,
            )
            score_counts, score_edges = plot_score_histogram(
                score_points=score_smoothed,
                level_hpa=level_hpa,
                output_path=score_histogram_path,
                dpi=args.dpi,
            )
            diagnostic_paths = {
                "latitude_difference_histogram": display_path(histogram_path),
                "score_histogram": display_path(score_histogram_path),
            }

            with (histogram_dir / f"latitude_difference_histogram_{slug}.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["bucket_lower", "bucket_upper", "bucket_center", "cell_count"])
                for index, count in enumerate(difference_counts):
                    writer.writerow(
                        [
                            float(difference_edges[index]),
                            float(difference_edges[index + 1]),
                            float(0.5 * (difference_edges[index] + difference_edges[index + 1])),
                            int(count),
                        ]
                    )

            with (score_histogram_dir / f"score_histogram_{slug}.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["bucket_lower", "bucket_upper", "bucket_center", "cell_count"])
                for index, count in enumerate(score_counts):
                    writer.writerow(
                        [
                            float(score_edges[index]),
                            float(score_edges[index + 1]),
                            float(0.5 * (score_edges[index] + score_edges[index + 1])),
                            int(count),
                        ]
                    )

        rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "thermal_displacement_map": display_path(thermal_map_path),
                "latitude_agreement_map": display_path(agreement_map_path),
                **diagnostic_paths,
                "agreement_cell_count": int(np.count_nonzero(agreement_mask)),
                "total_cell_count": int(agreement_mask.size),
                "agreement_fraction": float(np.count_nonzero(agreement_mask) / agreement_mask.size),
                "latitude_difference_mean": float(np.nanmean(signed_latitude_difference)),
                "latitude_difference_median": float(np.nanmedian(signed_latitude_difference)),
                "latitude_difference_abs_mean": float(np.nanmean(absolute_latitude_difference)),
                "latitude_difference_abs_median": float(np.nanmedian(absolute_latitude_difference)),
                "latitude_difference_min": float(np.nanmin(signed_latitude_difference)),
                "latitude_difference_max": float(np.nanmax(signed_latitude_difference)),
                "score_min": float(np.nanmin(score_smoothed)),
                "score_max": float(np.nanmax(score_smoothed)),
            }
        )

    summary = {
        "process": "thermal displacement plus source-latitude agreement overlay",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [row["pressure_level_hpa"] for row in rows],
        "score_definition": "score = (1 - abs(matched_climatology_latitude) / max_abs_latitude) * 100",
        "matching": (
            "For each raw cell, compare its raw temperature with climatology "
            "temperatures at the same longitude and pressure level; keep the "
            "latitude with closest climatology temperature."
        ),
        "matching_mode": args.matching_mode,
        "same_hemisphere_rule": (
            "When matching_mode is same-hemisphere, source latitudes >= 0 only "
            "compare against climatology latitudes >= 0, and source latitudes "
            "< 0 only compare against climatology latitudes < 0."
        ),
        "matched_latitude_tie_breaker": (
            "Exact temperature-distance ties choose the climatology latitude row "
            "closest to the source cell row."
        ),
        "matched_latitude_smoothing": (
            f"Gaussian sigma={args.matched_latitude_smooth_sigma_cells:g} native "
            "grid cell on signed matched latitude before score conversion; "
            "longitude wraps, latitude uses nearest edge."
        ),
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cell on score "
            "after matched-latitude smoothing; longitude wraps, latitude uses "
            "nearest edge."
        ),
        "base_map_color_scale": "fixed blue-white-red score scale: 0 blue, 50 white, 100 red",
        "agreement_rule": (
            "green where the source grid latitude is within "
            f"{args.agreement_degrees:g} degrees of the same-hemisphere matched "
            "latitude reconstructed from the smoothed thermal-displacement score"
        ),
        "latitude_difference_histogram": (
            "Signed source_grid_latitude minus the same-hemisphere matched latitude "
            "reconstructed from the smoothed thermal-displacement score, in "
            "1-degree buckets from -180 to 180; the +/- agreement band is colored green."
        ),
        "border_geojson": display_path(border_path),
        "outputs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "agreement_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pressure_level_hpa",
                "agreement_cell_count",
                "total_cell_count",
                "agreement_fraction",
                "latitude_difference_mean",
                "latitude_difference_median",
                "latitude_difference_abs_mean",
                "latitude_difference_abs_median",
                "latitude_difference_min",
                "latitude_difference_max",
                "score_min",
                "score_max",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["pressure_level_hpa"],
                    row["agreement_cell_count"],
                    row["total_cell_count"],
                    row["agreement_fraction"],
                    row["latitude_difference_mean"],
                    row["latitude_difference_median"],
                    row["latitude_difference_abs_mean"],
                    row["latitude_difference_abs_median"],
                    row["latitude_difference_min"],
                    row["latitude_difference_max"],
                    row["score_min"],
                    row["score_max"],
                ]
            )

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
