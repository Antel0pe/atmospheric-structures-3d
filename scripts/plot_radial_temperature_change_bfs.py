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
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_ROOT / "fontconfig"))
os.environ.setdefault("FONTCONFIG_FILE", str(CACHE_ROOT / "fontconfig/fonts.conf"))
(CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "fontconfig/fonts.conf").write_text(
    """<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <cachedir>/tmp/atmospheric-structures-3d-cache/fontconfig-cache</cachedir>
</fontconfig>
""",
    encoding="utf-8",
)
(CACHE_ROOT / "fontconfig-cache").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_BORDER_GEOJSON = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
DEFAULT_OUTPUT_DIR = Path("tmp/radial-temperature-change-bfs")
DEFAULT_LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
DEFAULT_TIMESTAMP = "2021-11-08T12:00"
TEMPERATURE_VARIABLE = "t"
NEIGHBOR_OFFSETS = tuple(
    (di, dj)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    if not (di == 0 and dj == 0)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot outward BFS-ring sums of absolute raw-temperature change from "
            "the grid cell nearest 0 latitude, 0 longitude."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--levels-hpa",
        type=str,
        default=",".join(str(level) for level in DEFAULT_LEVELS_HPA),
        help="Comma-separated pressure levels in hPa.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=100.0,
        help=(
            "Upper color limit percentile for the selected scale mode. "
            "Use 100 for the true maximum."
        ),
    )
    parser.add_argument(
        "--scale-mode",
        choices=("per-level", "global"),
        default="per-level",
        help=(
            "Use each pressure level's own color max, or one shared max across "
            "all requested levels."
        ),
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = (Path.cwd() / expanded).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(path.as_posix())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slug(level: float) -> str:
    return f"{int(round(level)):04d}hpa"


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
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    segments: list[list[tuple[float, float]]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "Polygon":
            polygons = [coordinates]
        elif geometry.get("type") == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        for polygon in polygons:
            for ring in polygon:
                points = [(float(lon), float(lat)) for lon, lat, *_ in ring]
                segments.extend(split_target_longitude_segments(points, longitudes))
    return segments


def bfs_ring_distance(
    shape: tuple[int, int],
    seed_index: tuple[int, int],
) -> np.ndarray:
    nlat, nlon = shape
    seed_i, seed_j = seed_index
    lat_distance = np.abs(np.arange(nlat, dtype=np.int32)[:, None] - seed_i)
    lon_delta = np.abs(np.arange(nlon, dtype=np.int32)[None, :] - seed_j)
    lon_distance = np.minimum(lon_delta, nlon - lon_delta)
    return np.maximum(lat_distance, lon_distance).astype(np.int16)


def radial_abs_change_score(
    temperature_k: np.ndarray,
    ring_distance: np.ndarray,
) -> np.ndarray:
    values = np.asarray(temperature_k, dtype=np.float32)
    if values.shape != ring_distance.shape:
        raise ValueError("temperature and ring distance arrays must have the same shape.")

    nlat, nlon = values.shape
    score = np.zeros((nlat, nlon), dtype=np.float32)
    source_j = np.arange(nlon, dtype=np.int32)

    for di, dj in NEIGHBOR_OFFSETS:
        if di < 0:
            source_i = np.arange(1, nlat, dtype=np.int32)
        elif di > 0:
            source_i = np.arange(0, nlat - 1, dtype=np.int32)
        else:
            source_i = np.arange(0, nlat, dtype=np.int32)
        target_i = source_i + di
        target_j = (source_j + dj) % nlon

        source_distance = ring_distance[np.ix_(source_i, source_j)]
        target_distance = ring_distance[np.ix_(target_i, target_j)]
        outward = target_distance == source_distance + 1
        if not np.any(outward):
            continue

        source_values = values[np.ix_(source_i, source_j)]
        target_values = values[np.ix_(target_i, target_j)]
        finite = np.isfinite(source_values) & np.isfinite(target_values)
        mask = outward & finite
        if not np.any(mask):
            continue

        diff = np.abs(target_values - source_values)
        target_i_grid = np.broadcast_to(target_i[:, None], mask.shape)
        target_j_grid = np.broadcast_to(target_j[None, :], mask.shape)
        np.add.at(score, (target_i_grid[mask], target_j_grid[mask]), diff[mask])

    return score


def white_to_red_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "white_to_red_temperature_change",
        ["#ffffff", "#ff0000"],
    )


def plot_score_map(
    score_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    seed_lat: float,
    seed_lon: float,
    title: str,
    vmax: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score_k,
        cmap=white_to_red_colormap(),
        norm=mcolors.Normalize(vmin=0.0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )
    for segment in border_segments:
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color="#171717", linewidth=0.35, alpha=0.72)

    ax.scatter(
        [seed_lon],
        [seed_lat],
        s=28,
        facecolor="#ffffff",
        edgecolor="#111111",
        linewidth=0.85,
        zorder=5,
    )
    ax.set_xlim(float(np.nanmin(longitudes)), float(np.nanmax(longitudes)))
    ax.set_ylim(float(np.nanmin(latitudes)), float(np.nanmax(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Outward BFS sum of absolute raw-temperature change (K)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def toy_proof() -> dict[str, object]:
    toy_temperature = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 3, 4, 5, 1],
            [1, 2, 10, 8, 1],
            [1, 3, 4, 6, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    ring_distance = bfs_ring_distance(toy_temperature.shape, (2, 2))
    score = radial_abs_change_score(toy_temperature, ring_distance)

    corner_score = (
        abs(float(toy_temperature[1, 0] - toy_temperature[0, 0]))
        + abs(float(toy_temperature[0, 1] - toy_temperature[0, 0]))
        + abs(float(toy_temperature[1, 1] - toy_temperature[0, 0]))
    )
    return {
        "temperature": toy_temperature.astype(float).tolist(),
        "bfs_ring_distance": ring_distance.astype(int).tolist(),
        "score": score.astype(float).tolist(),
        "proof_note": (
            "The top-left level-2 cell receives contributions from the three "
            "touching level-1 cells only: |1-1| + |1-1| + |3-1|."
        ),
        "top_left_expected_score": corner_score,
        "top_left_actual_score": float(score[0, 0]),
    }


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    border_geojson = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_levels = parse_levels(args.levels_hpa)
    level_outputs: list[dict[str, object]] = []
    score_by_level: list[tuple[float, np.ndarray]] = []

    with xr.open_dataset(dataset_path) as dataset:
        if TEMPERATURE_VARIABLE not in dataset:
            raise KeyError(f"Expected variable {TEMPERATURE_VARIABLE!r}.")

        temperature = dataset[TEMPERATURE_VARIABLE]
        timestamp = choose_timestamp(temperature, args.timestamp)
        selected_time = temperature.sel(valid_time=timestamp)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)

        seed_lat_index = int(np.argmin(np.abs(latitudes)))
        seed_lon_index = int(np.argmin(np.abs(longitudes)))
        seed_lat = float(latitudes[seed_lat_index])
        seed_lon = float(longitudes[seed_lon_index])
        ring_distance = bfs_ring_distance(
            (latitudes.size, longitudes.size),
            (seed_lat_index, seed_lon_index),
        )

        for requested_level in requested_levels:
            level_slice = selected_time.sel(pressure_level=requested_level, method="nearest")
            actual_level = float(level_slice.coords["pressure_level"].item())
            raw_temperature = np.asarray(level_slice.values, dtype=np.float32)
            score = radial_abs_change_score(raw_temperature, ring_distance)
            score_by_level.append((actual_level, score))

    finite_values_by_level = [
        (actual_level, score[np.isfinite(score)].ravel()) for actual_level, score in score_by_level
    ]
    if not finite_values_by_level or all(values.size == 0 for _, values in finite_values_by_level):
        raise ValueError("No finite score values were computed.")

    global_values = np.concatenate([values for _, values in finite_values_by_level if values.size])

    def choose_vmax(values: np.ndarray) -> tuple[float, str]:
        if float(args.display_percentile) >= 100.0:
            selected = float(np.nanmax(values))
            basis = "true finite maximum"
        else:
            selected = float(np.nanpercentile(values, float(args.display_percentile)))
            basis = f"finite p{float(args.display_percentile):g}"
        return (selected if selected > 0.0 else 1.0), basis

    global_vmax, global_basis = choose_vmax(global_values)

    border_segments = load_border_segments(border_geojson, longitudes)
    actual_timestamp_text = np.datetime_as_string(timestamp, unit="m")

    for actual_level, score in score_by_level:
        image_name = f"radial-temperature-change-bfs-{slug(actual_level)}.png"
        data_name = f"radial-temperature-change-bfs-{slug(actual_level)}.npz"
        finite_score = score[np.isfinite(score)]
        if args.scale_mode == "global":
            vmax = global_vmax
            vmax_basis = f"{global_basis} across requested levels"
        else:
            vmax, per_level_basis = choose_vmax(finite_score)
            vmax_basis = f"{per_level_basis} for this pressure level"
        plot_score_map(
            score,
            latitudes,
            longitudes,
            border_segments,
            seed_lat,
            seed_lon,
            (
                f"{actual_level:.0f} hPa outward BFS raw-temperature change, "
                f"{actual_timestamp_text} UTC"
            ),
            vmax,
            output_dir / image_name,
            args.dpi,
        )
        np.savez_compressed(
            output_dir / data_name,
            score_k=score,
            latitude_deg=latitudes,
            longitude_deg=longitudes,
            pressure_hpa=np.asarray(actual_level, dtype=np.float32),
            seed_latitude_deg=np.asarray(seed_lat, dtype=np.float32),
            seed_longitude_deg=np.asarray(seed_lon, dtype=np.float32),
            bfs_ring_distance=ring_distance,
        )
        level_outputs.append(
            {
                "actual_pressure_hpa": actual_level,
                "plot": image_name,
                "data": data_name,
                "score_min_k": float(np.nanmin(finite_score)),
                "score_p50_k": float(np.nanpercentile(finite_score, 50.0)),
                "score_p90_k": float(np.nanpercentile(finite_score, 90.0)),
                "score_p95_k": float(np.nanpercentile(finite_score, 95.0)),
                "score_p99_k": float(np.nanpercentile(finite_score, 99.0)),
                "score_max_k": float(np.nanmax(finite_score)),
                "plot_vmax_k": vmax,
                "plot_vmax_basis": vmax_basis,
            }
        )

    toy = toy_proof()
    (output_dir / "toy-proof.json").write_text(json.dumps(toy, indent=2) + "\n", encoding="utf-8")

    summary = {
        "method": {
            "field": "raw temperature",
            "seed": "grid cell nearest latitude 0, longitude 0",
            "movement": "8-neighbor outward BFS rings; longitude wraps, latitude does not",
            "score": (
                "For each directed neighbor edge, add abs(T_source - T_target) "
                "to the target only when the target BFS ring is exactly one "
                "farther from the seed than the source ring."
            ),
            "backtracking": (
                "Edges to earlier rings and edges within the same ring are ignored; "
                "all same-ring frontier contributions to a new cell are summed."
            ),
            "units": "K",
            "color_scale": {
                "minimum": 0.0,
                "mode": args.scale_mode,
                "display_percentile": float(args.display_percentile),
                "global_maximum": global_vmax,
                "global_maximum_basis": f"{global_basis} across requested levels",
                "ramp": "white to red",
            },
        },
        "dataset": display_path(dataset_path),
        "border_geojson": display_path(border_geojson),
        "requested_timestamp": args.timestamp,
        "actual_timestamp": actual_timestamp_text,
        "requested_levels_hpa": requested_levels,
        "seed": {
            "latitude_deg": seed_lat,
            "longitude_deg": seed_lon,
            "latitude_index": seed_lat_index,
            "longitude_index": seed_lon_index,
        },
        "grid": {
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_step_degrees": float(abs(latitudes[1] - latitudes[0])),
            "longitude_step_degrees": float(abs(longitudes[1] - longitudes[0])),
            "max_bfs_ring": int(np.max(ring_distance)),
        },
        "levels": level_outputs,
        "toy_proof": "toy-proof.json",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote radial BFS temperature-change plots to {display_path(output_dir)}")


if __name__ == "__main__":
    main()
