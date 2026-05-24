from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_latitude_score_lines import slug_for_level  # noqa: E402
from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_BORDER_GEOJSON,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    load_border_segments,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-heatmaps-score-contours-step5-250-1000-north-south-contour-paths"
)


@dataclass(frozen=True)
class ContourLine:
    contour_id: tuple[float, int]
    level: float
    vertices: np.ndarray


@dataclass(frozen=True)
class ClosestPoint:
    contour_id: tuple[float, int]
    level: float
    point: np.ndarray
    distance: float


@dataclass(frozen=True)
class PathAlgorithm:
    name: str
    description: str
    south_only: bool
    exclude_visited: bool
    min_south_step_degrees: float
    relaxed_min_south_step_degrees: float
    south_pole_fallback: bool
    bidirectional_offset: bool = False


ALGORITHMS: dict[str, PathAlgorithm] = {
    "iteration_01_nearest-different": PathAlgorithm(
        name="iteration_01_nearest-different",
        description=(
            "Original rule: choose the nearest point on any contour segment except "
            "the segment the path is currently on."
        ),
        south_only=False,
        exclude_visited=False,
        min_south_step_degrees=0.0,
        relaxed_min_south_step_degrees=0.0,
        south_pole_fallback=False,
    ),
    "iteration_02_southward": PathAlgorithm(
        name="iteration_02_southward",
        description=(
            "Choose the nearest different contour segment whose closest point is "
            "south of the current point by at least a small progress threshold."
        ),
        south_only=True,
        exclude_visited=False,
        min_south_step_degrees=0.25,
        relaxed_min_south_step_degrees=0.01,
        south_pole_fallback=False,
    ),
    "iteration_03_southward-unvisited-to-pole": PathAlgorithm(
        name="iteration_03_southward-unvisited-to-pole",
        description=(
            "Choose the nearest southward point on a contour segment that has not "
            "already been visited by this path. If no southward contour remains, "
            "finish by connecting to latitude -90."
        ),
        south_only=True,
        exclude_visited=True,
        min_south_step_degrees=0.25,
        relaxed_min_south_step_degrees=0.01,
        south_pole_fallback=True,
    ),
    "iteration_04_bidirectional-offset-5": PathAlgorithm(
        name="iteration_04_bidirectional-offset-5",
        description=(
            "Draw north-pole paths every 5 degrees from -180 to 180 and "
            "south-pole paths offset by 2.5 degrees. Each path chooses the "
            "nearest unvisited contour segment in its pole-to-pole direction, "
            "then finishes to the opposite pole when no farther contour remains."
        ),
        south_only=True,
        exclude_visited=True,
        min_south_step_degrees=0.25,
        relaxed_min_south_step_degrees=0.01,
        south_pole_fallback=True,
        bidirectional_offset=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate global thermal-displacement score contour maps and overlay "
            "north-pole seeded nearest-different-contour paths."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--longitude-step", type=float, default=25.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--algorithm",
        choices=tuple(ALGORITHMS.keys()),
        default="iteration_01_nearest-different",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def sorted_global_domain(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_signed = ((longitudes + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon_signed)
    lat_order = np.argsort(-latitudes)
    return lat_order, lon_order, latitudes[lat_order], lon_signed[lon_order]


def build_contour_lines(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    score: np.ndarray,
    contour_levels: np.ndarray,
) -> list[ContourLine]:
    fig, ax = plt.subplots()
    contours = ax.contour(longitudes, latitudes, score, levels=contour_levels)
    lines: list[ContourLine] = []
    for level, segments in zip(contours.levels, contours.allsegs):
        for segment_index, segment in enumerate(segments):
            if len(segment) < 2:
                continue
            vertices = np.asarray(segment, dtype=np.float32)
            if not np.all(np.isfinite(vertices)):
                continue
            lines.append(
                ContourLine(
                    contour_id=(float(level), segment_index),
                    level=float(level),
                    vertices=vertices,
                )
            )
    plt.close(fig)
    return lines


def closest_point_on_polyline(point: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, float]:
    starts = vertices[:-1]
    ends = vertices[1:]
    deltas = ends - starts
    lengths_squared = np.einsum("ij,ij->i", deltas, deltas)
    valid = lengths_squared > 0.0
    if not np.any(valid):
        first = np.asarray(vertices[0], dtype=np.float32)
        return first, float(np.linalg.norm(first - point))

    starts = starts[valid]
    deltas = deltas[valid]
    lengths_squared = lengths_squared[valid]
    offsets = point - starts
    t = np.clip(np.einsum("ij,ij->i", offsets, deltas) / lengths_squared, 0.0, 1.0)
    candidates = starts + deltas * t[:, np.newaxis]
    distances_squared = np.einsum("ij,ij->i", candidates - point, candidates - point)
    best_index = int(np.argmin(distances_squared))
    best_point = np.asarray(candidates[best_index], dtype=np.float32)
    return best_point, float(np.sqrt(distances_squared[best_index]))


def find_closest_different_contour(
    point: np.ndarray,
    contour_lines: list[ContourLine],
    current_contour_id: tuple[float, int] | None,
    visited_contour_ids: set[tuple[float, int]],
    algorithm: PathAlgorithm,
    direction_sign: float,
    min_south_step_degrees: float | None = None,
) -> ClosestPoint | None:
    best: ClosestPoint | None = None
    south_step = (
        algorithm.min_south_step_degrees
        if min_south_step_degrees is None
        else min_south_step_degrees
    )
    for contour_line in contour_lines:
        if contour_line.contour_id == current_contour_id:
            continue
        if algorithm.exclude_visited and contour_line.contour_id in visited_contour_ids:
            continue
        candidate_point, distance = closest_point_on_polyline(point, contour_line.vertices)
        if algorithm.south_only:
            if direction_sign < 0.0 and candidate_point[1] > point[1] - south_step:
                continue
            if direction_sign > 0.0 and candidate_point[1] < point[1] + south_step:
                continue
        if best is None or distance < best.distance:
            best = ClosestPoint(
                contour_id=contour_line.contour_id,
                level=contour_line.level,
                point=candidate_point,
                distance=distance,
            )
    return best


def seed_longitudes(longitude_step: float, include_dateline: bool = False) -> np.ndarray:
    if longitude_step <= 0.0:
        raise ValueError("--longitude-step must be positive")
    if include_dateline:
        return np.arange(-180.0, 180.0 + 0.5 * longitude_step, longitude_step)
    western_seed = -175.0
    eastern_seed = 175.0
    return np.arange(western_seed, eastern_seed + 0.5 * longitude_step, longitude_step)


def offset_seed_longitudes(longitude_step: float) -> np.ndarray:
    if longitude_step <= 0.0:
        raise ValueError("--longitude-step must be positive")
    return np.arange(
        -180.0 + longitude_step / 2.0,
        180.0,
        longitude_step,
    )


def build_seed_path(
    seed_lon: float,
    contour_lines: list[ContourLine],
    max_iterations: int,
    algorithm: PathAlgorithm,
    start_latitude: float = 90.0,
    target_latitude: float = -90.0,
    seed_pole: str = "north",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    direction_sign = -1.0 if target_latitude < start_latitude else 1.0
    current_point = np.asarray([seed_lon, start_latitude], dtype=np.float32)
    current_contour_id: tuple[float, int] | None = None
    visited_contour_ids: set[tuple[float, int]] = set()
    path_points = [current_point.copy()]
    rows: list[dict[str, object]] = []

    for iteration in range(1, max_iterations + 1):
        closest = find_closest_different_contour(
            current_point,
            contour_lines,
            current_contour_id,
            visited_contour_ids,
            algorithm,
            direction_sign,
        )
        if closest is None and algorithm.south_only:
            closest = find_closest_different_contour(
                current_point,
                contour_lines,
                current_contour_id,
                visited_contour_ids,
                algorithm,
                direction_sign,
                min_south_step_degrees=algorithm.relaxed_min_south_step_degrees,
            )
        if closest is None:
            break

        previous_point = current_point
        current_point = closest.point
        previous_contour_id = current_contour_id
        current_contour_id = closest.contour_id
        visited_contour_ids.add(current_contour_id)
        path_points.append(current_point.copy())

        rows.append(
            {
                "seed_longitude": float(seed_lon),
                "seed_pole": seed_pole,
                "target_pole": "south" if target_latitude < start_latitude else "north",
                "iteration": iteration,
                "from_longitude": float(previous_point[0]),
                "from_latitude": float(previous_point[1]),
                "to_longitude": float(current_point[0]),
                "to_latitude": float(current_point[1]),
                "distance_degrees": closest.distance,
                "target_score_contour": closest.level,
                "target_contour_segment": int(closest.contour_id[1]),
                "previous_score_contour": (
                    None if previous_contour_id is None else float(previous_contour_id[0])
                ),
                "previous_contour_segment": (
                    None if previous_contour_id is None else int(previous_contour_id[1])
                ),
                "moved_north": bool(current_point[1] > previous_point[1] + 1e-5),
                "moved_against_direction": bool(
                    (current_point[1] - previous_point[1]) * direction_sign < -1e-5
                ),
                "is_south_pole_finish": False,
                "is_pole_finish": False,
            }
        )

    needs_pole_finish = (
        direction_sign < 0.0 and path_points[-1][1] > target_latitude
    ) or (
        direction_sign > 0.0 and path_points[-1][1] < target_latitude
    )
    if algorithm.south_pole_fallback and needs_pole_finish:
        previous_point = path_points[-1]
        pole_point = np.asarray([float(previous_point[0]), target_latitude], dtype=np.float32)
        path_points.append(pole_point)
        rows.append(
            {
                "seed_longitude": float(seed_lon),
                "seed_pole": seed_pole,
                "target_pole": "south" if target_latitude < start_latitude else "north",
                "iteration": len(rows) + 1,
                "from_longitude": float(previous_point[0]),
                "from_latitude": float(previous_point[1]),
                "to_longitude": float(pole_point[0]),
                "to_latitude": float(pole_point[1]),
                "distance_degrees": float(abs(previous_point[1] - pole_point[1])),
                "target_score_contour": None,
                "target_contour_segment": None,
                "previous_score_contour": (
                    None if current_contour_id is None else float(current_contour_id[0])
                ),
                "previous_contour_segment": (
                    None if current_contour_id is None else int(current_contour_id[1])
                ),
                "moved_north": bool(pole_point[1] > previous_point[1] + 1e-5),
                "moved_against_direction": False,
                "is_south_pole_finish": bool(target_latitude < start_latitude),
                "is_pole_finish": True,
            }
        )

    return np.vstack(path_points), rows


def detect_ping_pong_rows(rows: list[dict[str, object]], tolerance_degrees: float = 0.05) -> int:
    by_seed: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(float(row["seed_longitude"]), []).append(row)

    ping_pong_count = 0
    for seed_rows in by_seed.values():
        for index in range(2, len(seed_rows)):
            current = seed_rows[index]
            previous = seed_rows[index - 1]
            before_previous = seed_rows[index - 2]
            if (
                current.get("target_score_contour") is None
                or previous.get("previous_score_contour") is None
            ):
                continue
            contour_backtrack = (
                current["target_score_contour"] == previous["previous_score_contour"]
                and current["target_contour_segment"] == previous["previous_contour_segment"]
            )
            point_backtrack = np.hypot(
                float(current["to_longitude"]) - float(before_previous["to_longitude"]),
                float(current["to_latitude"]) - float(before_previous["to_latitude"]),
            )
            if contour_backtrack and point_backtrack <= tolerance_degrees:
                ping_pong_count += 1
    return ping_pong_count


def path_success_count(paths: list[np.ndarray]) -> int:
    count = 0
    for path in paths:
        if len(path) < 2:
            continue
        first_latitude = float(path[0, 1])
        last_latitude = float(path[-1, 1])
        if (
            abs(abs(first_latitude) - 90.0) <= 1e-5
            and abs(last_latitude + first_latitude) <= 1e-5
        ):
            count += 1
    return count


def draw_borders_clipped(
    ax: plt.Axes,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    for segment in border_segments:
        points = [
            (lon, lat)
            for lon, lat in segment
            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
        ]
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color="#111111", linewidth=0.45, alpha=0.82)


def plot_map(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    paths: list[np.ndarray],
    path_seed_poles: list[str],
    level_hpa: float,
    contour_step: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap="bwr",
        norm=mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0),
        shading="auto",
        rasterized=True,
    )
    contour_levels = np.arange(contour_step, 100.0, contour_step, dtype=np.float32)
    contours = ax.contour(
        longitudes,
        latitudes,
        score,
        levels=contour_levels,
        colors="#1b1b1b",
        linewidths=0.65,
        alpha=0.82,
    )
    ax.clabel(contours, inline=True, fmt="%g", fontsize=6)
    draw_borders_clipped(ax, border_segments)

    for path, seed_pole in zip(paths, path_seed_poles):
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#00d15f" if seed_pole == "north" else "#00a8ff",
            linewidth=1.05,
            alpha=0.9,
            solid_capstyle="round",
            zorder=8,
        )

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement score contours; "
        "pole paths choose nearest allowed contour"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    algorithm = ALGORITHMS[args.algorithm]
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = (args.output_dir.expanduser() / algorithm.name).resolve()
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    source_latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    source_longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    lat_indices, lon_indices, selected_lats, selected_lons = sorted_global_domain(
        source_latitudes,
        source_longitudes,
    )
    border_segments = load_border_segments(
        border_path,
        np.asarray([-180.0, 180.0], dtype=np.float32),
    )
    contour_levels = np.arange(args.contour_step, 100.0, args.contour_step, dtype=np.float32)
    if algorithm.bidirectional_offset:
        north_seeded_longitudes = seed_longitudes(args.longitude_step, include_dateline=True)
        south_seeded_longitudes = offset_seed_longitudes(args.longitude_step)
        seeded_longitudes = np.concatenate((north_seeded_longitudes, south_seeded_longitudes))
    else:
        north_seeded_longitudes = seed_longitudes(args.longitude_step)
        south_seeded_longitudes = np.asarray([], dtype=np.float32)
        seeded_longitudes = north_seeded_longitudes

    all_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

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
            source_latitudes,
            "same-hemisphere",
        )
        matched_latitude = np.clip(matched_latitude, -90.0, 90.0).astype(np.float32)
        score_unsmoothed = thermal_displacement_score_points(
            matched_latitude,
            source_latitudes,
        )
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, args.smooth_sigma_cells)
        score_global = score_smoothed[np.ix_(lat_indices, lon_indices)]

        contour_lines = build_contour_lines(
            selected_lons,
            selected_lats,
            score_global,
            contour_levels,
        )
        paths: list[np.ndarray] = []
        path_seed_poles: list[str] = []
        level_rows: list[dict[str, object]] = []
        for seed_lon in north_seeded_longitudes:
            path, rows = build_seed_path(
                float(seed_lon),
                contour_lines,
                args.max_iterations,
                algorithm,
                start_latitude=90.0,
                target_latitude=-90.0,
                seed_pole="north",
            )
            paths.append(path)
            path_seed_poles.append("north")
            for row in rows:
                row["pressure_level_hpa"] = float(level_hpa)
            level_rows.extend(rows)
            all_rows.extend(rows)

        for seed_lon in south_seeded_longitudes:
            path, rows = build_seed_path(
                float(seed_lon),
                contour_lines,
                args.max_iterations,
                algorithm,
                start_latitude=-90.0,
                target_latitude=90.0,
                seed_pole="south",
            )
            paths.append(path)
            path_seed_poles.append("south")
            for row in rows:
                row["pressure_level_hpa"] = float(level_hpa)
            level_rows.extend(rows)
            all_rows.extend(rows)

        output_path = heatmap_dir / f"heatmap_{slug}.png"
        plot_map(
            score=score_global,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            paths=paths,
            path_seed_poles=path_seed_poles,
            level_hpa=level_hpa,
            contour_step=args.contour_step,
            output_path=output_path,
            dpi=args.dpi,
        )

        output_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "map": display_path(output_path),
                "contour_line_count": len(contour_lines),
                "path_count": len(paths),
                "successful_path_count": path_success_count(paths),
                "path_step_count": len(level_rows),
                "northward_step_count": int(sum(bool(row["moved_north"]) for row in level_rows)),
                "against_direction_step_count": int(
                    sum(bool(row["moved_against_direction"]) for row in level_rows)
                ),
                "ping_pong_like_step_count": detect_ping_pong_rows(level_rows),
                "south_pole_finish_count": int(
                    sum(bool(row["is_south_pole_finish"]) for row in level_rows)
                ),
                "pole_finish_count": int(sum(bool(row["is_pole_finish"]) for row in level_rows)),
            }
        )

    diagnostics_path = output_dir / "path_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "pressure_level_hpa",
            "seed_longitude",
            "seed_pole",
            "target_pole",
            "iteration",
            "from_longitude",
            "from_latitude",
            "to_longitude",
            "to_latitude",
            "distance_degrees",
            "target_score_contour",
            "target_contour_segment",
            "previous_score_contour",
            "previous_contour_segment",
            "moved_north",
            "moved_against_direction",
            "is_south_pole_finish",
            "is_pole_finish",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    summary = {
        "process": "global thermal-displacement score contours plus nearest-different-contour paths",
        "algorithm": algorithm.name,
        "algorithm_description": algorithm.description,
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "matching_mode": "same-hemisphere",
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cells on score; "
            "longitude wraps and latitude uses nearest edge."
        ),
        "contour_levels": [float(level) for level in contour_levels],
        "path_rule": (
            "Each path starts at one pole, repeatedly connects to the closest "
            "allowed point on a score-contour segment in its pole-to-pole "
            "direction, and finishes to the opposite pole when no farther "
            "allowed contour point remains."
        ),
        "path_constraints_implemented": [
            "do not select the same contour segment as the current segment",
            f"stop after at most {args.max_iterations} nearest-contour iterations",
            f"south_only={algorithm.south_only}",
            f"exclude_visited={algorithm.exclude_visited}",
            f"south_pole_fallback={algorithm.south_pole_fallback}",
            f"bidirectional_offset={algorithm.bidirectional_offset}",
        ],
        "path_constraints_not_implemented_in_this_pass": [
            "prevent two-point ping-pong explicitly beyond no-revisiting contour segments",
        ],
        "seed_longitudes": [float(lon) for lon in seeded_longitudes],
        "north_seed_longitudes": [float(lon) for lon in north_seeded_longitudes],
        "south_seed_longitudes": [float(lon) for lon in south_seeded_longitudes],
        "diagnostics": display_path(diagnostics_path),
        "outputs": output_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
