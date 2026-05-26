from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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

sys.path.insert(0, str(REPO_ROOT))

from scripts.thermal_displacement import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    compute_thermal_displacement_level,
)


DEFAULT_BORDER_GEOJSON = (
    REPO_ROOT
    / "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
DEFAULT_OUTPUT_DIR = Path("tmp/thermal-displacement-score-two-clusters")
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster canonical Thermal Displacement score values into two "
            "non-spatial score clusters."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default="")
    parser.add_argument("--pressure-min-hpa", type=float, default=DEFAULT_PRESSURE_MIN_HPA)
    parser.add_argument("--pressure-max-hpa", type=float, default=DEFAULT_PRESSURE_MAX_HPA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument(
        "--fit-scope",
        choices=("global", "per-level"),
        default="global",
        help=(
            "global fits one shared two-cluster score split across all selected "
            "levels; per-level fits a separate split for each pressure level."
        ),
    )
    parser.add_argument("--min-cluster-fraction", type=float, default=0.10)
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


def select_pressure_levels(
    text: str,
    available_levels: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> list[float]:
    available = np.asarray(available_levels, dtype=np.float64)
    if text.strip():
        selected: list[float] = []
        for piece in text.split(","):
            if not piece.strip():
                continue
            requested = float(piece.strip())
            selected.append(float(available[int(np.argmin(np.abs(available - requested)))]))
        return selected

    lower = min(float(pressure_min_hpa), float(pressure_max_hpa))
    upper = max(float(pressure_min_hpa), float(pressure_max_hpa))
    selected = available[(available >= lower) & (available <= upper)]
    return [float(level) for level in np.sort(selected)]


def slug_for_level(level_hpa: float) -> str:
    return f"{level_hpa:04.0f}hpa"


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


def two_cluster_kmeans(values: np.ndarray) -> tuple[float, np.ndarray]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32).ravel()
    if finite.size == 0:
        raise ValueError("Cannot cluster an empty score field.")

    centers = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0]).astype(np.float32)
    for _ in range(60):
        distances = np.abs(finite[:, np.newaxis] - centers[np.newaxis, :])
        labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for index in range(2):
            members = finite[labels == index]
            if members.size:
                next_centers[index] = np.mean(members, dtype=np.float64)
        if np.allclose(next_centers, centers, rtol=0.0, atol=1.0e-5):
            centers = next_centers
            break
        centers = next_centers

    centers = np.sort(centers.astype(np.float64))
    threshold = float(0.5 * (centers[0] + centers[1]))
    return threshold, centers.astype(np.float32)


def label_with_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(values.shape, -1, dtype=np.int8)
    finite_mask = np.isfinite(values)
    labels[finite_mask] = (values[finite_mask] > threshold).astype(np.int8)
    return labels


def cluster_stats(score: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    finite_count = int(np.count_nonzero(labels >= 0))
    for cluster_id, name in ((0, "low_score_polar_like"), (1, "high_score_equator_like")):
        mask = labels == cluster_id
        values = score[mask]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": name,
                "cell_count": int(values.size),
                "cell_fraction": float(values.size / finite_count) if finite_count else 0.0,
                "score_min": float(np.nanmin(values)) if values.size else None,
                "score_max": float(np.nanmax(values)) if values.size else None,
                "score_mean": float(np.nanmean(values)) if values.size else None,
                "score_median": float(np.nanmedian(values)) if values.size else None,
            }
        )
    return rows


def plot_cluster_map(
    labels: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    cmap = mcolors.ListedColormap(["#285ea8", "#c8382d"], name="two_score_clusters")
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        labels,
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
    ax.set_title(f"{level_hpa:g} hPa Thermal Displacement score split into 2 clusters")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["low / polar-like", "high / equator-like"])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_score_histogram(
    score: np.ndarray,
    threshold: float,
    level_hpa: float | None,
    output_path: Path,
    dpi: int,
) -> None:
    finite = np.asarray(score[np.isfinite(score)], dtype=np.float32).ravel()
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.hist(finite, bins=np.linspace(0.0, 100.0, 101), color="#777777", edgecolor="none")
    ax.axvline(threshold, color="#111111", linewidth=2.0)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Thermal Displacement score")
    ax.set_ylabel("Grid-cell count")
    if level_hpa is None:
        ax.set_title("All selected levels score distribution with 2-cluster split")
    else:
        ax.set_title(f"{level_hpa:g} hPa score distribution with shared 2-cluster split")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_contact_sheets(
    image_paths: list[Path],
    output_dir: Path,
    prefix: str,
    *,
    dpi: int,
    columns: int = 3,
) -> list[Path]:
    sheet_paths: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(image_paths), columns * columns), start=1):
        chunk = image_paths[start : start + columns * columns]
        rows = int(np.ceil(len(chunk) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(5.8 * columns, 3.2 * rows), constrained_layout=True)
        axes_array = np.atleast_1d(axes).ravel()
        for axis, image_path in zip(axes_array, chunk):
            axis.imshow(plt.imread(image_path))
            axis.set_axis_off()
        for axis in axes_array[len(chunk) :]:
            axis.set_axis_off()
        sheet_path = output_dir / f"{prefix}_contact_sheet_{sheet_index:02d}.png"
        fig.savefig(sheet_path, dpi=dpi)
        plt.close(fig)
        sheet_paths.append(sheet_path)
    return sheet_paths


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = (
        (REPO_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "cluster-membership-maps"
    histogram_dir = output_dir / "histograms"
    arrays_dir = output_dir / "arrays"
    findings_dir = output_dir / "findings"
    map_dir.mkdir(exist_ok=True)
    histogram_dir.mkdir(exist_ok=True)
    arrays_dir.mkdir(exist_ok=True)
    findings_dir.mkdir(exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = choose_timestamp(temperature, args.timestamp)
    pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = select_pressure_levels(
        args.pressure_levels,
        pressure_levels,
        args.pressure_min_hpa,
        args.pressure_max_hpa,
    )
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(border_path, longitudes)

    scores_by_level: dict[float, np.ndarray] = {}
    flat_scores: list[np.ndarray] = []
    for level_hpa in selected_levels:
        print(f"Computing score {level_hpa:g} hPa")
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
        scores_by_level[float(level_hpa)] = score
        flat_scores.append(score[np.isfinite(score)].ravel())

    all_scores = np.concatenate(flat_scores).astype(np.float32)
    if args.fit_scope == "global":
        global_threshold, global_centers = two_cluster_kmeans(all_scores)
        all_labels = (all_scores > global_threshold).astype(np.int8)
        all_counts = np.bincount(all_labels, minlength=2)
        min_fraction = float(np.min(all_counts) / np.sum(all_counts))
        if min_fraction < float(args.min_cluster_fraction):
            raise RuntimeError(
                "2-cluster result is too imbalanced: "
                f"min fraction {min_fraction:.4f} below required {args.min_cluster_fraction:.4f}"
            )
    else:
        global_threshold = None
        global_centers = None
        all_counts = None
        min_fraction = None

    combined_histogram_path = histogram_dir / "all-selected-levels-score-histogram.png"
    plot_score_histogram(
        all_scores,
        float(global_threshold) if global_threshold is not None else float(np.nanmedian(all_scores)),
        None,
        combined_histogram_path,
        args.dpi,
    )

    summary_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    map_paths: list[Path] = []
    histogram_paths: list[Path] = []
    for level_hpa in selected_levels:
        score = scores_by_level[float(level_hpa)]
        if args.fit_scope == "global":
            threshold = float(global_threshold)
            centers = np.asarray(global_centers, dtype=np.float32)
        else:
            threshold, centers = two_cluster_kmeans(score)
        labels = label_with_threshold(score, threshold)
        level_stats = cluster_stats(score, labels)
        level_min_fraction = min(float(row["cell_fraction"]) for row in level_stats)
        if level_min_fraction < float(args.min_cluster_fraction):
            raise RuntimeError(
                f"{level_hpa:g} hPa split is too imbalanced: "
                f"min fraction {level_min_fraction:.4f} below required "
                f"{args.min_cluster_fraction:.4f}"
            )

        slug = slug_for_level(level_hpa)
        map_path = map_dir / f"thermal-displacement-score-two-clusters-{slug}.png"
        histogram_path = histogram_dir / f"score-histogram-two-clusters-{slug}.png"
        score_path = arrays_dir / f"thermal_displacement_score_{slug}.npy"
        label_path = arrays_dir / f"thermal_displacement_score_two_clusters_{slug}.npy"
        np.save(score_path, score)
        np.save(label_path, labels)
        plot_cluster_map(labels, longitudes, latitudes, border_segments, level_hpa, map_path, args.dpi)
        plot_score_histogram(score, threshold, level_hpa, histogram_path, args.dpi)

        map_paths.append(map_path)
        histogram_paths.append(histogram_path)
        for row in level_stats:
            stats_rows.append({"pressure_level_hpa": float(level_hpa), **row})
        summary_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "cluster_map_png": display_path(map_path),
                "histogram_png": display_path(histogram_path),
                "score_array_npy": display_path(score_path),
                "cluster_label_array_npy": display_path(label_path),
                "score_min": float(np.nanmin(score)),
                "score_max": float(np.nanmax(score)),
                "score_mean": float(np.nanmean(score)),
                "cluster_centers_score_points": [float(value) for value in centers],
                "score_threshold": float(threshold),
                "cluster_stats": level_stats,
                "minimum_cluster_fraction": level_min_fraction,
            }
        )

    map_contact_sheets = write_contact_sheets(
        map_paths,
        output_dir,
        "thermal_displacement_score_two_cluster_maps",
        dpi=args.dpi,
    )
    histogram_contact_sheets = write_contact_sheets(
        histogram_paths,
        output_dir,
        "thermal_displacement_score_two_cluster_histograms",
        dpi=args.dpi,
    )

    with (output_dir / "cluster_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pressure_level_hpa",
                "cluster_id",
                "cluster_name",
                "cell_count",
                "cell_fraction",
                "score_min",
                "score_max",
                "score_mean",
                "score_median",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    summary = {
        "process": "canonical Thermal Displacement scores clustered into two non-spatial value clusters",
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
        "clustering_method": (
            "Run 1D k-means with k=2 on Thermal Displacement score values. "
            "Cluster membership is only by score value; cells do not need to "
            "touch spatially."
        ),
        "fit_scope": args.fit_scope,
        "cluster_centers_score_points": (
            [float(value) for value in global_centers]
            if global_centers is not None
            else None
        ),
        "shared_score_threshold": (
            float(global_threshold) if global_threshold is not None else None
        ),
        "minimum_allowed_cluster_fraction": float(args.min_cluster_fraction),
        "combined_cluster_counts": (
            {
                "low_score_polar_like": int(all_counts[0]),
                "high_score_equator_like": int(all_counts[1]),
            }
            if all_counts is not None
            else None
        ),
        "combined_cluster_fractions": (
            {
                "low_score_polar_like": float(all_counts[0] / np.sum(all_counts)),
                "high_score_equator_like": float(all_counts[1] / np.sum(all_counts)),
            }
            if all_counts is not None
            else None
        ),
        "combined_minimum_cluster_fraction": (
            float(min_fraction) if min_fraction is not None else None
        ),
        "combined_histogram_png": display_path(combined_histogram_path),
        "map_contact_sheets": [display_path(path) for path in map_contact_sheets],
        "histogram_contact_sheets": [display_path(path) for path in histogram_contact_sheets],
        "outputs": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    findings = [
        "# Thermal Displacement Score Two-Cluster Experiment",
        "",
        "## Method",
        "",
        "- Compute canonical same-longitude, same-hemisphere Thermal Displacement scores.",
        f"- Use score smoothing sigma `{args.score_smooth_sigma_cells:g}` native grid cells after matching.",
        f"- Fit scope: `{args.fit_scope}`.",
        "- The clusters are not connected-component regions.",
        f"- Require each cluster to contain at least `{args.min_cluster_fraction:.0%}` of finite cells overall and per level.",
        "",
        "## Result",
        "",
        (
            f"- Shared split threshold: `{global_threshold:.2f}` score points."
            if global_threshold is not None
            else "- Per-level split thresholds are listed in `summary.json` and `cluster_stats.csv`."
        ),
        (
            f"- Cluster centers: `{float(global_centers[0]):.2f}` and `{float(global_centers[1]):.2f}` score points."
            if global_centers is not None
            else "- Each pressure level has its own two fitted score centers."
        ),
        (
            f"- Combined fractions: low/polar-like `{all_counts[0] / np.sum(all_counts):.3f}`, "
            f"high/equator-like `{all_counts[1] / np.sum(all_counts):.3f}`."
            if all_counts is not None
            else "- Balance was checked separately at every pressure level."
        ),
        "- Both clusters pass the balance guard; no singleton or tiny cluster was accepted.",
        "",
        "## Interpretation",
        "",
        "This is a value split of the Thermal Displacement identity field, not a spatial object extraction. "
        "The same cluster can appear in many disconnected places because location was not part of the clustering.",
    ]
    (findings_dir / "two_cluster_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    print(f"Wrote {display_path(output_dir)}")
    if global_threshold is not None and all_counts is not None:
        print(f"Shared threshold: {global_threshold:.3f}")
        print(f"Combined counts: low={int(all_counts[0])}, high={int(all_counts[1])}")
    else:
        thresholds = [
            float(row["score_threshold"])
            for row in summary_rows
        ]
        print(
            "Per-level thresholds: "
            f"min={min(thresholds):.3f}, max={max(thresholds):.3f}"
        )


if __name__ == "__main__":
    main()
