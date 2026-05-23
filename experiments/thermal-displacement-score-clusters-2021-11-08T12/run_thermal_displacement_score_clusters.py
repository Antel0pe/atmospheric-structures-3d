from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from scipy import ndimage

import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.thermal_displacement import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    compute_thermal_displacement_level,
)


DEFAULT_BORDER_GEOJSON = REPO_ROOT / "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
DEFAULT_OUTPUT_DIR = Path("tmp/thermal-displacement-score-clusters-2021-11-08T12")
DEFAULT_LEVELS = "250,500,850,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute canonical Thermal Displacement scores and cluster contiguous "
            "score regions for selected ERA5 pressure levels."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--kmeans-score-classes", type=int, default=7)
    parser.add_argument(
        "--min-cluster-cells",
        type=int,
        default=1200,
        help="Connected score-region components smaller than this are treated as noise.",
    )
    parser.add_argument("--dpi", type=int, default=165)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = (REPO_ROOT / expanded).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(f"Could not find {path.as_posix()}")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        pass
    return path.name


def choose_timestamp(temperature: xr.DataArray, timestamp_text: str) -> np.datetime64:
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
        selected.append(float(available[int(np.argmin(np.abs(available - level)))]))
    return selected


def target_longitude(lon: float, longitudes: np.ndarray) -> float:
    lon_min = float(np.nanmin(longitudes))
    lon_max = float(np.nanmax(longitudes))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def split_target_longitude_segments(
    points: list[tuple[float, float]],
    longitudes: np.ndarray,
) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for lon, lat in points:
        mapped_lon = target_longitude(float(lon), longitudes)
        if previous_lon is not None and abs(mapped_lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                segments.append(segment)
            segment = []
        segment.append((mapped_lon, float(lat)))
        previous_lon = mapped_lon

    if len(segment) >= 2:
        segments.append(segment)
    return segments


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


def draw_borders(
    ax: plt.Axes,
    border_segments: list[list[tuple[float, float]]],
) -> None:
    for segment in border_segments:
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color="#171717", linewidth=0.35, alpha=0.78, zorder=5)


def slug_for_level(level_hpa: float) -> str:
    return f"{level_hpa:04.0f}hpa"


def one_dimensional_kmeans(values: np.ndarray, cluster_count: int) -> tuple[np.ndarray, np.ndarray]:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite_values.size == 0:
        raise ValueError("Cannot cluster an empty score field.")

    k = min(max(int(cluster_count), 2), int(np.unique(finite_values).size))
    quantiles = np.linspace(0.0, 1.0, k + 2, dtype=np.float64)[1:-1]
    centers = np.quantile(finite_values, quantiles).astype(np.float32)

    for _ in range(40):
        distances = np.abs(finite_values[:, np.newaxis] - centers[np.newaxis, :])
        labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for index in range(k):
            members = finite_values[labels == index]
            if members.size:
                next_centers[index] = np.mean(members, dtype=np.float64)
        if np.allclose(next_centers, centers, rtol=0.0, atol=1.0e-4):
            centers = next_centers
            break
        centers = next_centers

    order = np.argsort(centers)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(k)
    labels_full = np.full(values.shape, -1, dtype=np.int16)
    finite_mask = np.isfinite(values)
    final_labels = inverse_order[np.argmin(np.abs(values[finite_mask, np.newaxis] - centers[np.newaxis, :]), axis=1)]
    labels_full[finite_mask] = final_labels.astype(np.int16)
    return labels_full, centers[order].astype(np.float32)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size + 1, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        if left <= 0 or right <= 0:
            return
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def relabel_wrapped_components(
    labeled: np.ndarray,
    component_count: int,
) -> np.ndarray:
    if component_count <= 1:
        return labeled

    union_find = UnionFind(component_count)
    n_lat = labeled.shape[0]
    for lat_index in range(n_lat):
        left_label = int(labeled[lat_index, 0])
        for neighbor_lat in (lat_index - 1, lat_index, lat_index + 1):
            if 0 <= neighbor_lat < n_lat:
                union_find.union(left_label, int(labeled[neighbor_lat, -1]))

    roots = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        roots[label_id] = union_find.find(label_id)
    return roots[labeled]


def extract_score_region_clusters(
    score_class_labels: np.ndarray,
    *,
    min_cluster_cells: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    cluster_labels = np.full(score_class_labels.shape, -1, dtype=np.int32)
    raw_clusters: list[dict[str, object]] = []
    next_cluster_id = 0
    structure = np.ones((3, 3), dtype=np.uint8)

    for score_class in sorted(int(value) for value in np.unique(score_class_labels) if value >= 0):
        mask = score_class_labels == score_class
        labeled, component_count = ndimage.label(mask, structure=structure)
        labeled = relabel_wrapped_components(labeled.astype(np.int32), int(component_count))
        component_ids, counts = np.unique(labeled[labeled > 0], return_counts=True)

        for component_id, count in zip(component_ids, counts):
            if int(count) < min_cluster_cells:
                continue
            component_mask = labeled == int(component_id)
            cluster_labels[component_mask] = next_cluster_id
            raw_clusters.append(
                {
                    "cluster_id": int(next_cluster_id),
                    "score_class": int(score_class),
                    "cell_count": int(count),
                }
            )
            next_cluster_id += 1

    return cluster_labels, sorted(raw_clusters, key=lambda item: item["cell_count"], reverse=True)


def build_cluster_colormap(cluster_count: int) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    base = plt.colormaps["turbo"](np.linspace(0.04, 0.96, max(cluster_count, 1)))
    colors = np.vstack((np.array([[0.86, 0.86, 0.86, 1.0]]), base))
    cmap = mcolors.ListedColormap(colors, name="thermal_score_clusters")
    boundaries = np.arange(-1.5, cluster_count + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def plot_score_map(
    score: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_blue_white_red",
        ["#1f5fbf", "#f8f8f8", "#c9272c"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{level_hpa:g} hPa Thermal Displacement score")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Score points; blue = polar-like, red = equator-like")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_cluster_map(
    cluster_labels: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    cluster_count = int(np.max(cluster_labels)) + 1 if np.any(cluster_labels >= 0) else 0
    cmap, norm = build_cluster_colormap(cluster_count)
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        cluster_labels,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{level_hpa:g} hPa contiguous Thermal Displacement score clusters")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Cluster membership; gray = component below size cutoff")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_contact_sheet(
    image_paths: list[Path],
    title: str,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for axis, image_path in zip(axes.flat, image_paths):
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_axis_off()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_map_dir = output_dir / "score-blue-white-red-maps"
    cluster_map_dir = output_dir / "cluster-membership-maps"
    arrays_dir = output_dir / "arrays"
    score_map_dir.mkdir(exist_ok=True)
    cluster_map_dir.mkdir(exist_ok=True)
    arrays_dir.mkdir(exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = choose_timestamp(temperature, args.timestamp)
    pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, pressure_levels)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(border_path, longitudes)

    rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    score_contact_paths: list[Path] = []
    cluster_contact_paths: list[Path] = []

    for level_hpa in selected_levels:
        slug = slug_for_level(level_hpa)
        print(f"Processing {level_hpa:g} hPa")
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
            same_hemisphere=True,
        )
        score = result.score_points.astype(np.float32)
        score_class_labels, score_class_centers = one_dimensional_kmeans(
            score,
            args.kmeans_score_classes,
        )
        cluster_labels, clusters = extract_score_region_clusters(
            score_class_labels,
            min_cluster_cells=args.min_cluster_cells,
        )

        score_path = score_map_dir / f"thermal-displacement-score-{slug}.png"
        cluster_path = cluster_map_dir / f"thermal-displacement-score-clusters-{slug}.png"
        score_npy_path = arrays_dir / f"thermal_displacement_score_{slug}.npy"
        score_class_npy_path = arrays_dir / f"score_kmeans_class_{slug}.npy"
        cluster_npy_path = arrays_dir / f"score_region_cluster_{slug}.npy"
        np.save(score_npy_path, score)
        np.save(score_class_npy_path, score_class_labels)
        np.save(cluster_npy_path, cluster_labels)

        plot_score_map(
            score=score,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            level_hpa=level_hpa,
            output_path=score_path,
            dpi=args.dpi,
        )
        plot_cluster_map(
            cluster_labels=cluster_labels,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            level_hpa=level_hpa,
            output_path=cluster_path,
            dpi=args.dpi,
        )
        score_contact_paths.append(score_path)
        cluster_contact_paths.append(cluster_path)

        for cluster in clusters:
            cluster_rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "score_class": int(cluster["score_class"]),
                    "cluster_id": int(cluster["cluster_id"]),
                    "cell_count": int(cluster["cell_count"]),
                }
            )

        rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "score_map_png": display_path(score_path),
                "cluster_map_png": display_path(cluster_path),
                "score_array_npy": display_path(score_npy_path),
                "score_kmeans_class_array_npy": display_path(score_class_npy_path),
                "cluster_array_npy": display_path(cluster_npy_path),
                "score_min": float(np.nanmin(score)),
                "score_max": float(np.nanmax(score)),
                "score_mean": float(np.nanmean(score)),
                "selected_white_bucket": asdict(result.selected_bucket),
                "kmeans_score_class_centers": [float(value) for value in score_class_centers],
                "retained_cluster_count": int(len(clusters)),
                "retained_cluster_cell_count": int(np.count_nonzero(cluster_labels >= 0)),
                "noise_or_small_component_cell_count": int(np.count_nonzero(cluster_labels < 0)),
                "largest_clusters": clusters[:15],
            }
        )

    score_sheet_path = output_dir / "thermal-displacement-score-contact-sheet.png"
    cluster_sheet_path = output_dir / "thermal-displacement-score-clusters-contact-sheet.png"
    write_contact_sheet(
        score_contact_paths,
        "Thermal Displacement score maps",
        score_sheet_path,
        dpi=args.dpi,
    )
    write_contact_sheet(
        cluster_contact_paths,
        "Contiguous Thermal Displacement score clusters",
        cluster_sheet_path,
        dpi=args.dpi,
    )

    with (output_dir / "cluster_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pressure_level_hpa", "score_class", "cluster_id", "cell_count"],
        )
        writer.writeheader()
        writer.writerows(cluster_rows)

    summary = {
        "process": "canonical Thermal Displacement score maps plus score-region clustering",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "score_method": (
            "Canonical same-longitude same-hemisphere Thermal Displacement: match raw ERA5 "
            "temperature to closest climatology latitude at the same pressure and longitude, "
            "convert matched latitude to 0..100 score points, then smooth the score."
        ),
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "score_color_scale": "fixed blue-white-red: 0 blue, 50 white, 100 red",
        "clustering_method": (
            "For each pressure level, run 1D k-means on every finite ERA5 cell's "
            "Thermal Displacement score, then extract 8-neighbor connected components "
            "within each score class. Longitude edge components are merged across the dateline."
        ),
        "kmeans_score_classes": int(args.kmeans_score_classes),
        "min_cluster_cells": int(args.min_cluster_cells),
        "cluster_map_color_rule": (
            "Each retained connected component gets its own categorical color. Components "
            "below min_cluster_cells are gray."
        ),
        "border_geojson": display_path(border_path),
        "score_contact_sheet": display_path(score_sheet_path),
        "cluster_contact_sheet": display_path(cluster_sheet_path),
        "outputs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
