from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import ndimage
from skimage import measure
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("/tmp/iter-850hpa-temperature")
DEFAULT_PRESSURE_LEVEL_HPA = 850.0
TEMPERATURE_VARIABLE = "t"
DEFAULT_THRESHOLDS_K = (1.0, 2.0, 4.0)
DEFAULT_SEED_BIN_WIDTH_K = 0.25
PLANAR_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple iterative plotting script. For the first iteration, load ERA5 "
            "temperature, extract the 850 hPa slice on the native 0.25 degree grid, "
            "and write a color plot plus summary metadata to /tmp."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 temperature NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the plot and summary will be written.",
    )
    parser.add_argument(
        "--pressure-level-hpa",
        type=float,
        default=DEFAULT_PRESSURE_LEVEL_HPA,
        help="Pressure level to plot.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="ISO timestamp to plot. Defaults to the first time in the dataset.",
    )
    parser.add_argument(
        "--thresholds-k",
        type=str,
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS_K),
        help="Comma-separated temperature thresholds in K for the seed-grow contour runs.",
    )
    parser.add_argument(
        "--seed-bin-width-k",
        type=float,
        default=DEFAULT_SEED_BIN_WIDTH_K,
        help=(
            "Round seed temperatures to this bin width before building seed-grow groups. "
            "This keeps the native-grid pass tractable while staying close to the data."
        ),
    )
    return parser.parse_args()


def format_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        tmp_root = Path("/tmp")
        try:
            return f"tmp/{resolved.relative_to(tmp_root).as_posix()}"
        except ValueError:
            pass
        home = Path.home()
        try:
            return f"~/{resolved.relative_to(home).as_posix()}"
        except ValueError:
            return resolved.name or "<external-path>"


def resolve_existing_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {format_display_path(resolved)}")
    return resolved


def parse_thresholds(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in text.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("At least one threshold is required.")
    if any(value <= 0.0 for value in values):
        raise ValueError("All thresholds must be positive.")
    return tuple(values)


def choose_timestamp(temperature: xr.DataArray, requested_timestamp: str | None) -> np.datetime64:
    valid_times = temperature.coords["valid_time"].values
    if valid_times.size == 0:
        raise ValueError("Temperature dataset has no valid_time entries.")
    if requested_timestamp is None:
        return np.datetime64(valid_times[0])
    requested = np.datetime64(requested_timestamp)
    if requested not in valid_times:
        available = ", ".join(np.datetime_as_string(valid_times, unit="m").tolist())
        raise ValueError(
            f"Requested timestamp {requested_timestamp} is not available. "
            f"Available timestamps: {available}"
        )
    return requested


def load_temperature_slice(
    dataset_path: Path,
    pressure_level_hpa: float,
    requested_timestamp: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    dataset = xr.open_dataset(dataset_path)
    try:
        if TEMPERATURE_VARIABLE not in dataset:
            raise KeyError(
                f"Expected variable '{TEMPERATURE_VARIABLE}' in {format_display_path(dataset_path)}."
            )

        temperature = dataset[TEMPERATURE_VARIABLE]
        timestamp = choose_timestamp(temperature, requested_timestamp)
        selected = temperature.sel(valid_time=timestamp)
        level_slice = selected.sel(
            pressure_level=pressure_level_hpa,
            method="nearest",
        )

        actual_pressure_level = float(level_slice.coords["pressure_level"].item())
        latitudes = np.asarray(level_slice.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(level_slice.coords["longitude"].values, dtype=np.float32)
        values_kelvin = np.asarray(level_slice.values, dtype=np.float32)

        metadata = {
            "timestamp": np.datetime_as_string(timestamp, unit="m"),
            "requested_pressure_level_hpa": float(pressure_level_hpa),
            "actual_pressure_level_hpa": actual_pressure_level,
            "units": str(level_slice.attrs.get("units", "K")),
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_step_degrees": float(abs(latitudes[1] - latitudes[0])),
            "longitude_step_degrees": float(abs(longitudes[1] - longitudes[0])),
            "temperature_min_kelvin": float(np.nanmin(values_kelvin)),
            "temperature_max_kelvin": float(np.nanmax(values_kelvin)),
            "temperature_mean_kelvin": float(np.nanmean(values_kelvin)),
        }
        return latitudes, longitudes, values_kelvin, metadata
    finally:
        dataset.close()


def label_wrapped_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("Wrapped component labeling expects a 2D mask.")
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    row_count, longitude_count = occupied.shape
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(extended, structure=PLANAR_CONNECTIVITY)
    if component_count <= 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    parent = np.arange(component_count + 1, dtype=np.int32)

    def find(label_id: int) -> int:
        root = label_id
        while parent[root] != root:
            root = int(parent[root])
        while parent[label_id] != label_id:
            next_label = int(parent[label_id])
            parent[label_id] = root
            label_id = next_label
        return root

    def union(first: int, second: int) -> None:
        if first <= 0 or second <= 0:
            return
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if root_first < root_second:
            parent[root_second] = root_first
        else:
            parent[root_first] = root_second

    for row_index in range(row_count):
        for row_offset in (-1, 0, 1):
            other_row = row_index + row_offset
            if other_row < 0 or other_row >= row_count:
                continue
            union(int(labels[row_index, 0]), int(labels[other_row, -1]))

    root_map = np.zeros(component_count + 1, dtype=np.int32)
    for label_id in range(1, component_count + 1):
        root_map[label_id] = find(label_id)

    unique_roots = np.unique(root_map[1:])
    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[:, :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_roots.size)


def digest_mask(mask: np.ndarray) -> str:
    indices = np.flatnonzero(mask).astype(np.int32, copy=False)
    return hashlib.blake2b(indices.tobytes(), digest_size=16).hexdigest()


def build_seed_group_contours(
    *,
    values_kelvin: np.ndarray,
    threshold_k: float,
    seed_bin_width_k: float,
) -> dict[str, object]:
    binned_seed_temperature = (
        np.round(values_kelvin / seed_bin_width_k).astype(np.float32) * seed_bin_width_k
    ).astype(np.float32)
    unique_seed_centers = np.unique(binned_seed_temperature)

    seen_group_digests: set[str] = set()
    unique_group_masks: list[np.ndarray] = []
    unique_group_sizes: list[int] = []
    unique_group_seed_counts: list[int] = []
    assigned_seed_cells = 0
    per_center_component_total = 0

    for seed_center in unique_seed_centers:
        seed_mask = binned_seed_temperature == seed_center
        candidate_mask = (
            (values_kelvin > seed_center - threshold_k)
            & (values_kelvin < seed_center + threshold_k)
        )
        labels, component_count = label_wrapped_components(candidate_mask)
        if component_count <= 0:
            continue

        per_center_component_total += component_count
        used_component_ids = np.unique(labels[seed_mask & (labels > 0)])
        for component_id in used_component_ids:
            component_mask = labels == component_id
            seed_count = int(np.count_nonzero(seed_mask & component_mask))
            if seed_count <= 0:
                continue
            assigned_seed_cells += seed_count
            digest = digest_mask(component_mask)
            if digest in seen_group_digests:
                continue
            seen_group_digests.add(digest)
            unique_group_masks.append(component_mask)
            unique_group_sizes.append(int(np.count_nonzero(component_mask)))
            unique_group_seed_counts.append(seed_count)

    return {
        "threshold_k": float(threshold_k),
        "seed_bin_width_k": float(seed_bin_width_k),
        "seed_center_count": int(unique_seed_centers.size),
        "per_center_component_total": int(per_center_component_total),
        "assigned_seed_cells": int(assigned_seed_cells),
        "unique_group_masks": unique_group_masks,
        "unique_group_count": int(len(unique_group_masks)),
        "unique_group_size_min_cells": int(min(unique_group_sizes)) if unique_group_sizes else 0,
        "unique_group_size_median_cells": float(np.median(unique_group_sizes)) if unique_group_sizes else 0.0,
        "unique_group_size_max_cells": int(max(unique_group_sizes)) if unique_group_sizes else 0,
        "top_group_sizes_cells": sorted(unique_group_sizes, reverse=True)[:12],
        "top_group_seed_counts": sorted(unique_group_seed_counts, reverse=True)[:12],
    }


def write_plot(
    *,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values_kelvin: np.ndarray,
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 9), constrained_layout=True)
    mesh = axis.pcolormesh(
        longitudes,
        latitudes,
        values_kelvin,
        shading="auto",
        cmap="coolwarm",
    )
    axis.set_title(
        (
            f"ERA5 Temperature at {metadata['actual_pressure_level_hpa']:.0f} hPa\n"
            f"{metadata['timestamp']} on native {metadata['latitude_step_degrees']:.2f} degree grid"
        ),
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    axis.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    axis.grid(True, color="#d0d0d0", linewidth=0.35, alpha=0.4)
    axis.text(
        0.01,
        0.01,
        (
            f"Range: {metadata['temperature_min_kelvin']:.2f} to "
            f"{metadata['temperature_max_kelvin']:.2f} K\n"
            f"Mean: {metadata['temperature_mean_kelvin']:.2f} K\n"
            f"Grid: {metadata['latitude_count']} x {metadata['longitude_count']}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#c9c9c9"},
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.01)
    colorbar.set_label("Temperature (K)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_threshold_group_plot(
    *,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values_kelvin: np.ndarray,
    metadata: dict[str, object],
    group_summary: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 9), constrained_layout=True)
    mesh = axis.pcolormesh(
        longitudes,
        latitudes,
        values_kelvin,
        shading="auto",
        cmap="coolwarm",
    )

    contour_segments: list[np.ndarray] = []
    longitude_index = np.arange(longitudes.size, dtype=np.float32)
    latitude_index = np.arange(latitudes.size, dtype=np.float32)
    for component_mask in group_summary["unique_group_masks"]:
        contour_paths = measure.find_contours(component_mask.astype(np.float32), level=0.5)
        for contour_path in contour_paths:
            if contour_path.shape[0] < 2:
                continue
            y_coords = np.interp(contour_path[:, 0], latitude_index, latitudes)
            x_coords = np.interp(contour_path[:, 1], longitude_index, longitudes)
            contour_segments.append(np.column_stack([x_coords, y_coords]))

    if contour_segments:
        axis.add_collection(
            LineCollection(
                contour_segments,
                colors="#111111",
                linewidths=0.22,
                alpha=0.24,
            )
        )

    threshold_k = float(group_summary["threshold_k"])
    axis.set_title(
        (
            f"850 hPa Temperature With Seed-Grow Group Contours, +/-{threshold_k:.1f} K\n"
            f"{metadata['timestamp']} on native {metadata['latitude_step_degrees']:.2f} degree grid"
        ),
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    axis.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    axis.grid(True, color="#d0d0d0", linewidth=0.35, alpha=0.35)
    axis.text(
        0.01,
        0.01,
        (
            f"Unique contour groups: {group_summary['unique_group_count']:,}\n"
            f"Assigned seed cells: {group_summary['assigned_seed_cells']:,}\n"
            f"Seed temp bin width: {group_summary['seed_bin_width_k']:.2f} K\n"
            f"Group size range: {group_summary['unique_group_size_min_cells']:,} to "
            f"{group_summary['unique_group_size_max_cells']:,} cells"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#c9c9c9"},
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.01)
    colorbar.set_label("Temperature (K)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_existing_path(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_k = parse_thresholds(args.thresholds_k)

    latitudes, longitudes, values_kelvin, metadata = load_temperature_slice(
        dataset_path=dataset_path,
        pressure_level_hpa=args.pressure_level_hpa,
        requested_timestamp=args.timestamp,
    )

    actual_pressure_level_hpa = int(round(float(metadata["actual_pressure_level_hpa"])))
    plot_path = output_dir / f"temperature-{actual_pressure_level_hpa}hpa-native-0p25deg.png"
    write_plot(
        latitudes=latitudes,
        longitudes=longitudes,
        values_kelvin=values_kelvin,
        metadata=metadata,
        output_path=plot_path,
    )
    print(f"base_plot={format_display_path(plot_path)}", flush=True)

    summary = {
        "iteration": 1,
        "description": "Plot native-grid ERA5 temperature at 850 hPa.",
        "dataset_path": format_display_path(dataset_path),
        "plot_path": format_display_path(plot_path),
        **metadata,
    }

    threshold_runs: list[dict[str, object]] = []
    for threshold_k in thresholds_k:
        print(f"building_threshold={threshold_k:.2f}K", flush=True)
        group_summary = build_seed_group_contours(
            values_kelvin=values_kelvin,
            threshold_k=threshold_k,
            seed_bin_width_k=args.seed_bin_width_k,
        )
        threshold_plot_path = (
            output_dir / f"temperature-850hpa-seed-groups-threshold-{threshold_k:.1f}k.png"
        )
        write_threshold_group_plot(
            latitudes=latitudes,
            longitudes=longitudes,
            values_kelvin=values_kelvin,
            metadata=metadata,
            group_summary=group_summary,
            output_path=threshold_plot_path,
        )
        print(
            "finished_threshold="
            f"{threshold_k:.2f}K unique_groups={group_summary['unique_group_count']} "
            f"plot={format_display_path(threshold_plot_path)}",
            flush=True,
        )
        threshold_runs.append(
            {
                "threshold_k": group_summary["threshold_k"],
                "seed_bin_width_k": group_summary["seed_bin_width_k"],
                "seed_center_count": group_summary["seed_center_count"],
                "per_center_component_total": group_summary["per_center_component_total"],
                "assigned_seed_cells": group_summary["assigned_seed_cells"],
                "unique_group_count": group_summary["unique_group_count"],
                "unique_group_size_min_cells": group_summary["unique_group_size_min_cells"],
                "unique_group_size_median_cells": group_summary["unique_group_size_median_cells"],
                "unique_group_size_max_cells": group_summary["unique_group_size_max_cells"],
                "top_group_sizes_cells": group_summary["top_group_sizes_cells"],
                "top_group_seed_counts": group_summary["top_group_seed_counts"],
                "plot_path": format_display_path(threshold_plot_path),
                "threshold_rule": (
                    "From each starting cell, grow through 8-neighbor cells while the "
                    "candidate temperature stays strictly within seed_temperature +/- threshold."
                ),
                "implementation_note": (
                    "Seeds are grouped by rounded starting temperature bins so the native "
                    "0.25 degree field can be processed without running one flood-fill per cell."
                ),
            }
        )

    summary["iteration"] = 2
    summary["description"] = (
        "Plot native-grid ERA5 temperature at 850 hPa and compare threshold-based "
        "8-neighbor seed-grow contour groups."
    )
    summary["threshold_group_runs"] = threshold_runs

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"dataset={format_display_path(dataset_path)}")
    print(f"output_dir={output_dir}")
    print(f"plot={plot_path}")
    for threshold_run in threshold_runs:
        print(
            "threshold_plot="
            f"{threshold_run['plot_path']} "
            f"(threshold_k={threshold_run['threshold_k']}, "
            f"unique_groups={threshold_run['unique_group_count']})"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
