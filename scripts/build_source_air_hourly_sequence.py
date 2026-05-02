from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from build_source_air_trajectory_plots import (
    assign_clusters,
    build_initial_states,
    cluster_palette,
    endpoint_features,
    feature_to_lon_lat,
    fit_kmeans,
    format_map_axis,
    load_wind_cube,
    normalize_lon_180,
    rk4_step,
    to_repo_relative,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "data/era5_source_air_wind_uv_omega_2021-11-05t12_to_2021-11-08t12_1000-250hpa.nc"
)
DEFAULT_OUTPUT_DIR = Path("tmp/source-air-hourly-sequence-2021-11-08t12-250hpa")
DEFAULT_TARGET_TIME = "2021-11-08T12:00"


@dataclass(frozen=True)
class SequenceFrameSummary:
    lookback_hours: int
    frame_path: str
    parcel_count: int
    dominant_cluster_id: int
    dominant_cluster_fraction: float
    mean_source_lon_deg: float
    mean_source_lat_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an hour-by-hour fixed-cluster source-region sequence for one target pressure level."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME)
    parser.add_argument("--level", type=float, default=250.0)
    parser.add_argument("--max-lookback-hours", type=int, default=72)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--step-hours", type=float, default=1.0)
    parser.add_argument("--cluster-count", type=int, default=10)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    return parser.parse_args()


def plot_lon_order(lon_grid: np.ndarray) -> np.ndarray:
    return np.argsort(normalize_lon_180(lon_grid[0]))


def capture_hourly_snapshots(
    dataset: Path,
    target_time: np.datetime64,
    level: float,
    max_lookback_hours: int,
    stride: int,
    step_hours: float,
) -> tuple[object, dict[int, dict[str, np.ndarray]]]:
    cube = load_wind_cube(
        dataset,
        target_time,
        max_lookback_hours,
        stride,
        requested_levels=[level],
    )
    lon, lat, pressure, level_index, cell_index, target_pressure = build_initial_states(
        cube, cube.pressure_hpa
    )
    target_lon = lon.copy()
    target_lat = lat.copy()

    snapshots: dict[int, dict[str, np.ndarray]] = {
        0: {
            "source_lon": lon.copy(),
            "source_lat": lat.copy(),
            "source_pressure": pressure.copy(),
            "target_lon": target_lon,
            "target_lat": target_lat,
            "target_pressure": target_pressure,
            "level_index": level_index,
            "cell_index": cell_index,
        }
    }

    current_time = 0.0
    elapsed = 0
    step = -abs(float(step_hours))
    while elapsed < max_lookback_hours:
        actual_step = -min(abs(step), float(max_lookback_hours - elapsed))
        lon, lat, pressure = rk4_step(cube, lon, lat, pressure, current_time, actual_step)
        current_time += actual_step
        elapsed = int(round(abs(current_time)))
        snapshots[elapsed] = {
            "source_lon": lon.copy(),
            "source_lat": lat.copy(),
            "source_pressure": pressure.copy(),
            "target_lon": target_lon,
            "target_lat": target_lat,
            "target_pressure": target_pressure,
            "level_index": level_index,
            "cell_index": cell_index,
        }
    return cube, snapshots


def fit_sequence_clusters(
    snapshots: dict[int, dict[str, np.ndarray]], cluster_count: int
) -> np.ndarray:
    all_lon = np.concatenate([snapshots[hour]["source_lon"] for hour in sorted(snapshots)])
    all_lat = np.concatenate([snapshots[hour]["source_lat"] for hour in sorted(snapshots)])
    features = endpoint_features(all_lon, all_lat)
    return fit_kmeans(
        features,
        cluster_count=cluster_count,
        max_points=min(250_000, features.shape[0]),
        seed=25_000 + cluster_count,
    )


def save_frame(
    output_path: Path,
    cube: object,
    snapshot: dict[str, np.ndarray],
    centers: np.ndarray,
    lookback_hours: int,
    level_hpa: float,
) -> SequenceFrameSummary:
    lon_grid, lat_grid = np.meshgrid(cube.longitude_deg, cube.latitude_deg)
    grid_shape = lon_grid.shape
    order = plot_lon_order(lon_grid)

    features = endpoint_features(snapshot["source_lon"], snapshot["source_lat"])
    cluster_labels = assign_clusters(features, centers)
    cluster_grid = cluster_labels.reshape(grid_shape)

    cluster_count = centers.shape[0]
    cmap = cluster_palette(cluster_count)
    center_lon, center_lat = feature_to_lon_lat(centers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    image = axes[0].pcolormesh(
        normalize_lon_180(lon_grid[:, order]),
        lat_grid[:, order],
        cluster_grid[:, order],
        shading="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=cluster_count - 0.5,
    )
    axes[0].set_title(f"{lookback_hours:02d}h back source clusters, target {level_hpa:.0f} hPa")
    format_map_axis(axes[0])
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("fixed source cluster")

    axes[1].scatter(
        normalize_lon_180(snapshot["source_lon"]),
        snapshot["source_lat"],
        c=cluster_labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=cluster_count - 0.5,
        s=3,
        linewidths=0,
        alpha=0.45,
    )
    axes[1].scatter(center_lon, center_lat, c=np.arange(cluster_count), cmap=cmap, s=90, edgecolors="black")
    for cluster_id, (lon, lat) in enumerate(zip(center_lon, center_lat)):
        axes[1].text(lon, lat, str(cluster_id), ha="center", va="center", fontsize=8, color="white")
    axes[1].set_title("Current-hour endpoints with fixed centers")
    axes[1].set_xlabel("source lon")
    axes[1].set_ylabel("source lat")
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-89, 89)
    axes[1].grid(color="0.75", linewidth=0.4, alpha=0.45)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    counts = np.bincount(cluster_labels, minlength=cluster_count)
    dominant = int(np.argmax(counts))
    return SequenceFrameSummary(
        lookback_hours=int(lookback_hours),
        frame_path=to_repo_relative(output_path),
        parcel_count=int(cluster_labels.size),
        dominant_cluster_id=dominant,
        dominant_cluster_fraction=float(counts[dominant] / max(1, cluster_labels.size)),
        mean_source_lon_deg=float(np.mean(normalize_lon_180(snapshot["source_lon"]))),
        mean_source_lat_deg=float(np.mean(snapshot["source_lat"])),
    )


def write_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> None:
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, frames, duration=duration_ms / 1000.0, loop=0)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    target_time = np.datetime64(args.target_time)
    cube, snapshots = capture_hourly_snapshots(
        args.dataset,
        target_time,
        args.level,
        args.max_lookback_hours,
        args.stride,
        args.step_hours,
    )
    centers = fit_sequence_clusters(snapshots, args.cluster_count)

    frame_summaries: list[SequenceFrameSummary] = []
    frame_paths: list[Path] = []
    for lookback_hours in range(args.max_lookback_hours, -1, -1):
        frame_path = frame_dir / f"source_clusters_{lookback_hours:03d}h.png"
        frame_summaries.append(
            save_frame(
                frame_path,
                cube,
                snapshots[lookback_hours],
                centers,
                lookback_hours,
                float(cube.pressure_hpa[0]),
            )
        )
        frame_paths.append(frame_path)

    gif_path = output_dir / f"source_clusters_{int(round(float(cube.pressure_hpa[0]))):04d}hpa_072h_to_000h.gif"
    write_gif(frame_paths, gif_path, args.gif_duration_ms)

    center_lon, center_lat = feature_to_lon_lat(centers)
    payload = {
        "dataset": to_repo_relative(args.dataset),
        "target_time": args.target_time,
        "target_level_hpa": float(cube.pressure_hpa[0]),
        "max_lookback_hours": args.max_lookback_hours,
        "stride": args.stride,
        "rk4_step_hours": args.step_hours,
        "cluster_count": args.cluster_count,
        "cluster_centers": [
            {"cluster_id": int(index), "lon_deg": float(lon), "lat_deg": float(lat)}
            for index, (lon, lat) in enumerate(zip(center_lon, center_lat))
        ],
        "gif": to_repo_relative(gif_path),
        "frames": [asdict(item) for item in frame_summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hourly Source-Cluster Sequence",
                "",
                f"- target time: `{args.target_time}`",
                f"- target level: `{float(cube.pressure_hpa[0]):.0f} hPa`",
                f"- frames: `{args.max_lookback_hours}h` back to `0h`",
                f"- trajectory mode: hourly RK4, step `{args.step_hours}h`",
                f"- target grid stride: `{args.stride}`",
                f"- fixed cluster count: `{args.cluster_count}`",
                f"- animation: `{to_repo_relative(gif_path)}`",
                "",
                "The cluster centers are fitted once across all hourly endpoints, so color IDs are stable across the whole sequence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": to_repo_relative(output_dir), "frames": len(frame_paths), "gif": to_repo_relative(gif_path)}, indent=2))


if __name__ == "__main__":
    main()
