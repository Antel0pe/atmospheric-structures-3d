from __future__ import annotations

import argparse
import csv
import importlib.util
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

import imageio.v2 as imageio
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
    TEMPERATURE_VARIABLE,
    compute_thermal_displacement_level,
)

BASE_SCRIPT_PATH = Path(__file__).with_name("run_two_cluster_scores.py")
BASE_SPEC = importlib.util.spec_from_file_location("two_cluster_base", BASE_SCRIPT_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT_PATH.as_posix()}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)


DEFAULT_DATASET_PATH = Path(
    "data/era5_temperature_2021-11-08t15_to_2021-11-10t00_3hourly_250-1000hpa.nc"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-score-two-clusters/"
    "time-series-2021-11-08t15-to-2021-11-10t00"
)
DEFAULT_PRESSURE_GIF_LEVELS = "250,500,850,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Thermal Displacement two-cluster score split across a "
            "3-hourly temperature time series and build requested GIFs."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH)
    parser.add_argument("--timestamps", type=str, default="")
    parser.add_argument("--pressure-levels", type=str, default="")
    parser.add_argument("--pressure-min-hpa", type=float, default=250.0)
    parser.add_argument("--pressure-max-hpa", type=float, default=1000.0)
    parser.add_argument("--pressure-gif-levels", type=str, default=DEFAULT_PRESSURE_GIF_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--border-geojson", type=Path, default=BASE.DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.10)
    parser.add_argument("--gif-frame-duration-seconds", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_timestamp_selection(text: str, available_times: np.ndarray) -> list[np.datetime64]:
    if not text.strip():
        return [np.datetime64(value) for value in available_times]

    selected: list[np.datetime64] = []
    for piece in text.split(","):
        if piece.strip():
            selected.append(np.datetime64(piece.strip()))
    available = {np.datetime64(value) for value in available_times}
    missing = [value for value in selected if value not in available]
    if missing:
        missing_text = ", ".join(np.datetime_as_string(value, unit="s") for value in missing)
        raise ValueError(f"Requested timestamp(s) not in dataset: {missing_text}")
    return selected


def timestamp_slug(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="m").replace(":", "").replace("-", "").lower()


def timestamp_label(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="m").replace("T", " UTC ")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def parse_pressure_list(text: str, available_levels: np.ndarray) -> list[float]:
    if not text.strip():
        return []
    selected: list[float] = []
    available = np.asarray(available_levels, dtype=np.float64)
    for piece in text.split(","):
        if not piece.strip():
            continue
        requested = float(piece.strip())
        selected.append(float(available[int(np.argmin(np.abs(available - requested)))]))
    return selected


def plot_cluster_map(
    labels: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    timestamp: np.datetime64,
    threshold: float,
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
    BASE.draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{timestamp_label(timestamp)} | {level_hpa:g} hPa | "
        f"2 score clusters, threshold {threshold:.2f}"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["low / polar-like", "high / equator-like"])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_gif(frame_paths: list[Path], output_path: Path, duration_seconds: float) -> None:
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(output_path, frames, duration=float(duration_seconds), loop=0)


def main() -> None:
    args = parse_args()
    dataset_path = BASE.resolve_path(args.dataset)
    climatology_path = BASE.resolve_path(args.climatology)
    border_path = BASE.resolve_path(args.border_geojson)
    output_dir = resolve_output_dir(args.output_dir)
    frames_dir = output_dir / "frames"
    gifs_dir = output_dir / "gifs"
    findings_dir = output_dir / "findings"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_times = parse_timestamp_selection(
        args.timestamps,
        np.asarray(temperature.coords["valid_time"].values),
    )
    pressure_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = BASE.select_pressure_levels(
        args.pressure_levels,
        pressure_levels,
        args.pressure_min_hpa,
        args.pressure_max_hpa,
    )
    pressure_gif_levels = parse_pressure_list(args.pressure_gif_levels, pressure_levels)
    selected_level_set = {float(level) for level in selected_levels}
    pressure_gif_levels = [level for level in pressure_gif_levels if float(level) in selected_level_set]

    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = BASE.load_border_segments(border_path, longitudes)

    rows: list[dict[str, object]] = []
    frames_by_time: dict[str, list[Path]] = {}
    frames_by_pressure: dict[float, list[Path]] = {float(level): [] for level in pressure_gif_levels}

    for selected_time in selected_times:
        time_slug = timestamp_slug(selected_time)
        time_frame_dir = frames_dir / f"time-{time_slug}"
        time_frame_dir.mkdir(exist_ok=True)
        frames_by_time[time_slug] = []

        for level_hpa in selected_levels:
            print(f"Computing {timestamp_label(selected_time)} {level_hpa:g} hPa")
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
            threshold, centers = BASE.two_cluster_kmeans(score)
            labels = BASE.label_with_threshold(score, threshold)
            level_stats = BASE.cluster_stats(score, labels)
            level_min_fraction = min(float(row["cell_fraction"]) for row in level_stats)
            if level_min_fraction < float(args.min_cluster_fraction):
                raise RuntimeError(
                    f"{timestamp_label(selected_time)} {level_hpa:g} hPa split is too imbalanced: "
                    f"min fraction {level_min_fraction:.4f} below required "
                    f"{args.min_cluster_fraction:.4f}"
                )

            frame_path = (
                time_frame_dir
                / f"thermal-displacement-score-two-clusters-{time_slug}-{BASE.slug_for_level(level_hpa)}.png"
            )
            plot_cluster_map(
                labels,
                longitudes,
                latitudes,
                border_segments,
                level_hpa,
                selected_time,
                threshold,
                frame_path,
                args.dpi,
            )
            frames_by_time[time_slug].append(frame_path)
            if float(level_hpa) in frames_by_pressure:
                frames_by_pressure[float(level_hpa)].append(frame_path)

            for stat in level_stats:
                rows.append(
                    {
                        "valid_time": np.datetime_as_string(selected_time, unit="s"),
                        "pressure_level_hpa": float(level_hpa),
                        "score_threshold": float(threshold),
                        "cluster_center_low_score": float(centers[0]),
                        "cluster_center_high_score": float(centers[1]),
                        "minimum_cluster_fraction": float(level_min_fraction),
                        "frame_png": display_path(frame_path),
                        **stat,
                    }
                )

    time_gifs: list[Path] = []
    for selected_time in selected_times:
        time_slug = timestamp_slug(selected_time)
        gif_path = gifs_dir / f"thermal-displacement-score-two-clusters-{time_slug}-levels-250-to-1000-0p5s.gif"
        write_gif(frames_by_time[time_slug], gif_path, args.gif_frame_duration_seconds)
        time_gifs.append(gif_path)

    pressure_gifs: list[Path] = []
    for level_hpa in pressure_gif_levels:
        gif_path = gifs_dir / (
            f"thermal-displacement-score-two-clusters-{BASE.slug_for_level(level_hpa)}-"
            "20211108t1500-to-20211110t0000-0p5s.gif"
        )
        write_gif(frames_by_pressure[float(level_hpa)], gif_path, args.gif_frame_duration_seconds)
        pressure_gifs.append(gif_path)

    with (output_dir / "cluster_stats_by_time_level.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "valid_time",
            "pressure_level_hpa",
            "score_threshold",
            "cluster_center_low_score",
            "cluster_center_high_score",
            "minimum_cluster_fraction",
            "frame_png",
            "cluster_id",
            "cluster_name",
            "cell_count",
            "cell_fraction",
            "score_min",
            "score_max",
            "score_mean",
            "score_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "process": "canonical Thermal Displacement scores clustered into two non-spatial value clusters per time and pressure level",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "valid_times": [np.datetime_as_string(value, unit="s") for value in selected_times],
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "pressure_level_gifs_hpa": [float(level) for level in pressure_gif_levels],
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "clustering_method": (
            "For each time and pressure level, run 1D k-means with k=2 on "
            "Thermal Displacement score values. Cluster membership is only by "
            "score value; cells do not need to touch spatially."
        ),
        "gif_frame_duration_seconds": float(args.gif_frame_duration_seconds),
        "time_gifs": [display_path(path) for path in time_gifs],
        "pressure_gifs": [display_path(path) for path in pressure_gifs],
        "stats_csv": display_path(output_dir / "cluster_stats_by_time_level.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    findings = [
        "# Thermal Displacement Score Two-Cluster Time Series",
        "",
        "## Method",
        "",
        "- Compute canonical same-longitude, same-hemisphere Thermal Displacement scores.",
        f"- Use score smoothing sigma `{args.score_smooth_sigma_cells:g}` native grid cells after matching.",
        "- Fit a separate 2-cluster 1D k-means split for each `(valid_time, pressure_level)` score field.",
        "- The clusters are value clusters, not connected-component regions.",
        f"- Use `{args.gif_frame_duration_seconds:g}s` per GIF frame.",
        "",
        "## Inputs",
        "",
        f"- Temperature: `{display_path(dataset_path)}`.",
        f"- Climatology: `{display_path(climatology_path)}`.",
        "",
        "## Outputs",
        "",
        f"- Time-step GIFs: `{display_path(gifs_dir)}`.",
        f"- Cluster frame PNGs: `{display_path(frames_dir)}`.",
        f"- Stats: `{display_path(output_dir / 'cluster_stats_by_time_level.csv')}`.",
        "",
        "## Interpretation",
        "",
        "This extends the two-score-cluster diagnostic over time. Because each map fits its own split, "
        "cluster labels mean lower-score/polar-like versus higher-score/equator-like within that one "
        "time and pressure level.",
    ]
    (findings_dir / "two_cluster_time_series_findings.md").write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {display_path(output_dir)}")
    print(f"Time GIFs: {len(time_gifs)}")
    print(f"Pressure GIFs: {len(pressure_gifs)}")


if __name__ == "__main__":
    main()
