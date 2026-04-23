from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import ndimage
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = Path(
    "data/global-850-hpa-wind-uv-for-established-repo-timestamp_2021-11_p850.nc"
)
DEFAULT_OUTPUT_DIR = Path("TMP/wind-direction-groups-2021-11-08t1200-850")
DEFAULT_SPEED_THRESHOLD_MS = 5.0
DEFAULT_COARSEN_FACTOR = 4
DEFAULT_QUIVER_STRIDE = 6
DEFAULT_MIN_GROUP_SIZE = 60
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class DatasetFields:
    timestamp: str
    pressure_level_hpa: float
    latitude: np.ndarray
    longitude: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class GroupSummary:
    group_id: int
    size: int
    representative_direction_deg: float
    coverage_fraction: float
    anchor_label: str


@dataclass(frozen=True)
class MethodSummary:
    slug: str
    title: str
    description: str
    parameters: dict[str, Any]
    active_cell_count: int
    grouped_cell_count: int
    grouped_fraction_of_active: float
    total_group_count: int
    plotted_group_count: int
    top_groups: list[GroupSummary]
    plot_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 850 hPa wind-direction grouping experiments and save quiver-plus-"
            "contour plots into a repo-local TMP folder."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 wind UV NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_MS,
        help="Minimum wind speed in m/s before direction is treated as meaningful.",
    )
    parser.add_argument(
        "--coarsen-factor",
        type=int,
        default=DEFAULT_COARSEN_FACTOR,
        help="Block-average factor applied to latitude and longitude before grouping.",
    )
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=DEFAULT_QUIVER_STRIDE,
        help="Arrow stride on the coarsened grid.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help="Drop connected groups smaller than this many coarsened cells.",
    )
    parser.add_argument(
        "--max-plotted-groups",
        type=int,
        default=18,
        help="Maximum number of groups to contour on each plot.",
    )
    return parser.parse_args()


def ensure_plot_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")


def to_repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    return text[:-1] if text.endswith("Z") else text


def angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def flow_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.mod(np.degrees(np.arctan2(v, u)), 360.0)


def load_dataset(path: Path, coarsen_factor: int) -> DatasetFields:
    dataset = xr.open_dataset(path.expanduser().resolve())
    try:
        u_field = dataset["u"].isel(valid_time=0, pressure_level=0)
        v_field = dataset["v"].isel(valid_time=0, pressure_level=0)
        if coarsen_factor > 1:
            u_field = u_field.coarsen(
                latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
            ).mean()
            v_field = v_field.coarsen(
                latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
            ).mean()

        return DatasetFields(
            timestamp=timestamp_to_iso_minute(
                dataset["valid_time"].isel(valid_time=0).values.astype("datetime64[m]")
            ),
            pressure_level_hpa=float(dataset["pressure_level"].isel(pressure_level=0).item()),
            latitude=np.asarray(u_field.latitude.values, dtype=np.float32),
            longitude=np.asarray(u_field.longitude.values, dtype=np.float32),
            u=np.asarray(u_field.values, dtype=np.float32),
            v=np.asarray(v_field.values, dtype=np.float32),
        )
    finally:
        dataset.close()


def component_sizes(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros(1, dtype=np.int32)
    return np.bincount(labels.reshape(-1))


def label_wrapped_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("Wrapped planar labeling expects a 2D mask.")
    if not occupied.any():
        return np.zeros_like(occupied, dtype=np.int32), 0

    row_count, longitude_count = occupied.shape
    extended = np.concatenate([occupied, occupied[:, :1]], axis=1)
    labels, component_count = ndimage.label(extended, structure=CONNECTIVITY_STRUCTURE)
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
    if unique_roots.size == 0:
        return np.zeros_like(occupied, dtype=np.int32), 0

    compact_root_ids = np.zeros(component_count + 1, dtype=np.int32)
    compact_root_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    compact_labels = compact_root_ids[root_map[labels[:, :longitude_count]]]
    return compact_labels.astype(np.int32), int(unique_roots.size)


def build_groups_from_sector_labels(
    *,
    active_mask: np.ndarray,
    sector_labels: np.ndarray,
    representative_angles_deg: np.ndarray,
    min_group_size: int,
) -> tuple[np.ndarray, list[GroupSummary]]:
    final_labels = np.zeros_like(sector_labels, dtype=np.int32)
    groups: list[GroupSummary] = []
    next_group_id = 1

    for sector_label in np.unique(sector_labels[active_mask]):
        mask = active_mask & (sector_labels == sector_label)
        labeled_components, component_count = label_wrapped_components(mask)
        if component_count <= 0:
            continue

        sizes = component_sizes(labeled_components)
        for component_id in range(1, component_count + 1):
            size = int(sizes[component_id])
            if size < min_group_size:
                continue
            component_mask = labeled_components == component_id
            final_labels[component_mask] = next_group_id
            groups.append(
                GroupSummary(
                    group_id=next_group_id,
                    size=size,
                    representative_direction_deg=float(representative_angles_deg[sector_label]),
                    coverage_fraction=0.0,
                    anchor_label=f"sector-{int(sector_label)}",
                )
            )
            next_group_id += 1

    return final_labels, groups


def raw_sector_method(
    direction_deg: np.ndarray,
    active_mask: np.ndarray,
    min_group_size: int,
) -> tuple[np.ndarray, list[GroupSummary], dict[str, Any]]:
    sector_width_deg = 15.0
    sector_count = int(round(360.0 / sector_width_deg))
    sector_labels = np.floor(((direction_deg + sector_width_deg / 2.0) % 360.0) / sector_width_deg).astype(
        np.int16
    )
    representative_angles = (np.arange(sector_count, dtype=np.float32) * sector_width_deg) % 360.0
    labels, groups = build_groups_from_sector_labels(
        active_mask=active_mask,
        sector_labels=sector_labels,
        representative_angles_deg=representative_angles,
        min_group_size=min_group_size,
    )
    return labels, groups, {"sector_width_deg": sector_width_deg}


def smoothed_sector_method(
    *,
    u: np.ndarray,
    v: np.ndarray,
    direction_deg: np.ndarray,
    active_mask: np.ndarray,
    min_group_size: int,
) -> tuple[np.ndarray, list[GroupSummary], dict[str, Any]]:
    gaussian_sigma = 1.0
    smoothed_u = ndimage.gaussian_filter(u, sigma=gaussian_sigma, mode=("nearest", "wrap"))
    smoothed_v = ndimage.gaussian_filter(v, sigma=gaussian_sigma, mode=("nearest", "wrap"))
    smoothed_speed = np.hypot(smoothed_u, smoothed_v)
    smoothed_direction_deg = flow_direction_deg(smoothed_u, smoothed_v)

    sector_width_deg = 20.0
    sector_count = int(round(360.0 / sector_width_deg))
    sector_labels = np.floor(
        ((smoothed_direction_deg + sector_width_deg / 2.0) % 360.0) / sector_width_deg
    ).astype(np.int16)
    representative_angles = (np.arange(sector_count, dtype=np.float32) * sector_width_deg) % 360.0
    coherence_tolerance_deg = 12.0
    refined_active = (
        active_mask
        & (smoothed_speed >= 5.0)
        & (angle_diff_deg(direction_deg, smoothed_direction_deg) <= coherence_tolerance_deg)
    )
    labels, groups = build_groups_from_sector_labels(
        active_mask=refined_active,
        sector_labels=sector_labels,
        representative_angles_deg=representative_angles,
        min_group_size=min_group_size,
    )
    return labels, groups, {
        "sector_width_deg": sector_width_deg,
        "gaussian_sigma_cells": gaussian_sigma,
        "coherence_tolerance_deg": coherence_tolerance_deg,
    }


def anchored_seed_grow_method(
    direction_deg: np.ndarray,
    active_mask: np.ndarray,
    min_group_size: int,
) -> tuple[np.ndarray, list[GroupSummary], dict[str, Any]]:
    anchor_step_deg = 5.0
    seed_tolerance_deg = 10.0
    max_overlap_fraction = 0.65
    min_new_cells = max(20, min_group_size // 2)

    candidate_groups: list[tuple[int, float, np.ndarray]] = []
    for anchor_deg in np.arange(0.0, 360.0, anchor_step_deg, dtype=np.float32):
        candidate_mask = active_mask & (angle_diff_deg(direction_deg, anchor_deg) <= seed_tolerance_deg)
        labeled_components, component_count = label_wrapped_components(candidate_mask)
        if component_count <= 0:
            continue
        sizes = component_sizes(labeled_components)
        for component_id in range(1, component_count + 1):
            size = int(sizes[component_id])
            if size < min_group_size:
                continue
            candidate_groups.append((size, float(anchor_deg), labeled_components == component_id))

    candidate_groups.sort(key=lambda item: item[0], reverse=True)
    final_labels = np.zeros_like(direction_deg, dtype=np.int32)
    claimed = np.zeros_like(active_mask, dtype=bool)
    groups: list[GroupSummary] = []
    next_group_id = 1

    for size, anchor_deg, component_mask in candidate_groups:
        overlap_count = int(np.count_nonzero(component_mask & claimed))
        new_mask = component_mask & ~claimed
        new_count = int(np.count_nonzero(new_mask))
        if new_count < min_new_cells:
            continue
        if overlap_count / max(size, 1) > max_overlap_fraction:
            continue

        final_labels[new_mask] = next_group_id
        claimed |= component_mask
        groups.append(
            GroupSummary(
                group_id=next_group_id,
                size=new_count,
                representative_direction_deg=anchor_deg,
                coverage_fraction=0.0,
                anchor_label=f"anchor-{anchor_deg:03.0f}",
            )
        )
        next_group_id += 1

    return final_labels, groups, {
        "anchor_step_deg": anchor_step_deg,
        "seed_tolerance_deg": seed_tolerance_deg,
        "max_overlap_fraction": max_overlap_fraction,
        "min_new_cells_after_overlap_prune": min_new_cells,
    }


def finalize_groups(
    active_mask: np.ndarray,
    labels: np.ndarray,
    groups: list[GroupSummary],
) -> tuple[np.ndarray, list[GroupSummary]]:
    if not groups:
        return labels, []

    active_count = int(np.count_nonzero(active_mask))
    sizes = component_sizes(labels)
    refreshed: list[GroupSummary] = []
    for group in groups:
        size = int(sizes[group.group_id]) if group.group_id < sizes.size else 0
        if size <= 0:
            continue
        refreshed.append(
            GroupSummary(
                group_id=group.group_id,
                size=size,
                representative_direction_deg=group.representative_direction_deg,
                coverage_fraction=size / max(active_count, 1),
                anchor_label=group.anchor_label,
            )
        )

    refreshed.sort(key=lambda item: item.size, reverse=True)
    remapped = np.zeros_like(labels, dtype=np.int32)
    remapped_groups: list[GroupSummary] = []
    for new_group_id, group in enumerate(refreshed, start=1):
        remapped[labels == group.group_id] = new_group_id
        remapped_groups.append(
            GroupSummary(
                group_id=new_group_id,
                size=group.size,
                representative_direction_deg=group.representative_direction_deg,
                coverage_fraction=group.coverage_fraction,
                anchor_label=group.anchor_label,
            )
        )
    return remapped, remapped_groups


def plot_direction_groups(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    direction_deg: np.ndarray,
    active_mask: np.ndarray,
    labels: np.ndarray,
    groups: list[GroupSummary],
    title: str,
    subtitle: str,
    quiver_stride: int,
    max_plotted_groups: int,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 9), constrained_layout=True)
    cmap = plt.colormaps["hsv"].copy()
    cmap.set_bad("#f2f2f2")
    masked_direction = np.ma.masked_where(~active_mask, direction_deg)
    mesh = axis.pcolormesh(
        longitude,
        latitude,
        masked_direction,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=360.0,
    )
    sampled_rows = slice(0, latitude.size, quiver_stride)
    sampled_cols = slice(0, longitude.size, quiver_stride)
    sampled_active = active_mask[sampled_rows, sampled_cols]
    lon_grid, lat_grid = np.meshgrid(longitude[sampled_cols], latitude[sampled_rows])
    axis.quiver(
        lon_grid[sampled_active],
        lat_grid[sampled_active],
        u[sampled_rows, sampled_cols][sampled_active],
        v[sampled_rows, sampled_cols][sampled_active],
        color="#151515",
        alpha=0.65,
        scale=850.0,
        width=0.0016,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.5,
    )

    normalizer = Normalize(vmin=0.0, vmax=360.0)
    plotted_groups = groups[:max_plotted_groups]
    for group in plotted_groups:
        mask = labels == group.group_id
        if not mask.any():
            continue
        color = plt.colormaps["hsv"](normalizer(group.representative_direction_deg))
        axis.contour(
            longitude,
            latitude,
            mask.astype(np.float32),
            levels=[0.5],
            colors=["#111111"],
            linewidths=2.4,
            alpha=0.7,
        )
        axis.contour(
            longitude,
            latitude,
            mask.astype(np.float32),
            levels=[0.5],
            colors=[color],
            linewidths=1.5,
            alpha=0.95,
        )

    wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=110))
    axis.set_title(
        f"{title}\n{wrapped_subtitle}",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(float(longitude.min()), float(longitude.max()))
    axis.set_ylim(float(latitude.min()), float(latitude.max()))
    axis.grid(True, color="#d0d0d0", linewidth=0.4, alpha=0.45)

    summary_lines = [
        f"Active cells: {int(np.count_nonzero(active_mask)):,}",
        f"Grouped cells: {int(np.count_nonzero(labels)):,}",
        f"Groups shown: {min(len(groups), max_plotted_groups)} of {len(groups)}",
    ]
    if plotted_groups:
        summary_lines.append(
            "Top sizes: " + ", ".join(f"{group.size}" for group in plotted_groups[:5])
        )
    axis.text(
        0.01,
        0.01,
        "\n".join(summary_lines),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#c9c9c9"},
    )

    colorbar = figure.colorbar(mesh, ax=axis, pad=0.01)
    colorbar.set_label("Flow direction from u/v (deg toward)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def overview_plot(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    direction_deg: np.ndarray,
    active_mask: np.ndarray,
    quiver_stride: int,
    output_path: Path,
    timestamp: str,
    pressure_level_hpa: float,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 9), constrained_layout=True)
    cmap = plt.colormaps["hsv"].copy()
    cmap.set_bad("#f2f2f2")
    masked_direction = np.ma.masked_where(~active_mask, direction_deg)
    mesh = axis.pcolormesh(
        longitude,
        latitude,
        masked_direction,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=360.0,
    )
    sampled_rows = slice(0, latitude.size, quiver_stride)
    sampled_cols = slice(0, longitude.size, quiver_stride)
    sampled_active = active_mask[sampled_rows, sampled_cols]
    lon_grid, lat_grid = np.meshgrid(longitude[sampled_cols], latitude[sampled_rows])
    axis.quiver(
        lon_grid[sampled_active],
        lat_grid[sampled_active],
        u[sampled_rows, sampled_cols][sampled_active],
        v[sampled_rows, sampled_cols][sampled_active],
        color="#151515",
        alpha=0.7,
        scale=850.0,
        width=0.0016,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.5,
    )
    axis.set_title(
        f"850 hPa Wind Flow Direction Overview\n{timestamp} UTC-equivalent frame, 1° block-mean grid",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(float(longitude.min()), float(longitude.max()))
    axis.set_ylim(float(latitude.min()), float(latitude.max()))
    axis.grid(True, color="#d0d0d0", linewidth=0.4, alpha=0.45)
    axis.text(
        0.01,
        0.01,
        f"Pressure level: {pressure_level_hpa:.0f} hPa\nActive cells (>= 5 m/s): {int(np.count_nonzero(active_mask)):,}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#c9c9c9"},
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.01)
    colorbar.set_label("Flow direction from u/v (deg toward)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summarize_method(
    *,
    slug: str,
    title: str,
    description: str,
    parameters: dict[str, Any],
    active_mask: np.ndarray,
    labels: np.ndarray,
    groups: list[GroupSummary],
    max_plotted_groups: int,
    plot_path: Path,
) -> MethodSummary:
    active_count = int(np.count_nonzero(active_mask))
    grouped_count = int(np.count_nonzero(labels))
    return MethodSummary(
        slug=slug,
        title=title,
        description=description,
        parameters=parameters,
        active_cell_count=active_count,
        grouped_cell_count=grouped_count,
        grouped_fraction_of_active=grouped_count / max(active_count, 1),
        total_group_count=len(groups),
        plotted_group_count=min(len(groups), max_plotted_groups),
        top_groups=groups[:max_plotted_groups],
        plot_path=to_repo_relative(plot_path),
    )


def write_summary_markdown(
    *,
    output_path: Path,
    dataset_path: Path,
    timestamp: str,
    pressure_level_hpa: float,
    coarsen_factor: int,
    speed_threshold: float,
    methods: list[MethodSummary],
    overview_path: Path,
) -> None:
    lines = [
        "# Wind Direction Grouping Experiment",
        "",
        f"- Dataset: `{to_repo_relative(dataset_path)}`",
        f"- Timestamp: `{timestamp}`",
        f"- Pressure level: `{pressure_level_hpa:.0f} hPa`",
        f"- Grouping grid: `{coarsen_factor}x{coarsen_factor}` block mean from the native `0.25°` grid to `1°`",
        f"- Calm-wind cutoff: `{speed_threshold:.1f} m/s`",
        f"- Direction convention: flow direction derived directly from `(u, v)` as degrees **toward**, not meteorological **from** direction",
        f"- Overview plot: `{to_repo_relative(overview_path)}`",
        "",
    ]

    for method in methods:
        lines.extend(
            [
                f"## {method.title}",
                "",
                f"- Plot: `{method.plot_path}`",
                f"- Description: {method.description}",
                f"- Grouped cells: `{method.grouped_cell_count:,}` / `{method.active_cell_count:,}` active cells "
                f"(`{100.0 * method.grouped_fraction_of_active:.1f}%`)",
                f"- Total surviving groups: `{method.total_group_count}`",
                f"- Parameters: `{json.dumps(method.parameters, sort_keys=True)}`",
            ]
        )
        if method.top_groups:
            lines.append("- Largest groups:")
            for group in method.top_groups[:8]:
                lines.append(
                    f"  - id `{group.group_id}`: `{group.size}` cells, "
                    f"`{group.representative_direction_deg:.1f}°`, `{group.anchor_label}`"
                )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    ensure_plot_env()
    args = parse_args()
    dataset_fields = load_dataset(args.dataset, args.coarsen_factor)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    speed = np.hypot(dataset_fields.u, dataset_fields.v)
    direction_deg = flow_direction_deg(dataset_fields.u, dataset_fields.v)
    active_mask = speed >= args.speed_threshold

    overview_path = output_dir / "overview_direction_quiver.png"
    overview_plot(
        latitude=dataset_fields.latitude,
        longitude=dataset_fields.longitude,
        u=dataset_fields.u,
        v=dataset_fields.v,
        direction_deg=direction_deg,
        active_mask=active_mask,
        quiver_stride=args.quiver_stride,
        output_path=overview_path,
        timestamp=dataset_fields.timestamp,
        pressure_level_hpa=dataset_fields.pressure_level_hpa,
    )

    method_specs = [
        (
            "raw-sector-components",
            "Raw Sector Components",
            "Connected components after binning each active 1° cell into 15° flow-direction sectors.",
            lambda: raw_sector_method(direction_deg, active_mask, args.min_group_size),
        ),
        (
            "smoothed-sector-components",
            "Smoothed Sector Components",
            "Connected components on 20° sectors after Gaussian smoothing in u/v space, keeping only cells still within 12° of their smoothed local flow.",
            lambda: smoothed_sector_method(
                u=dataset_fields.u,
                v=dataset_fields.v,
                direction_deg=direction_deg,
                active_mask=active_mask,
                min_group_size=args.min_group_size,
            ),
        ),
        (
            "anchored-seed-grow",
            "Anchored Seed Grow",
            "Approximation of the requested seed-and-grow idea: grow 8-neighbor groups inside a strict ±10° band around fixed 5° seed anchors, then greedily keep large groups while pruning overlaps.",
            lambda: anchored_seed_grow_method(direction_deg, active_mask, args.min_group_size),
        ),
    ]

    method_summaries: list[MethodSummary] = []
    for slug, title, description, builder in method_specs:
        labels, groups, parameters = builder()
        labels, groups = finalize_groups(active_mask, labels, groups)
        plot_path = output_dir / f"{slug}.png"
        plot_direction_groups(
            latitude=dataset_fields.latitude,
            longitude=dataset_fields.longitude,
            u=dataset_fields.u,
            v=dataset_fields.v,
            direction_deg=direction_deg,
            active_mask=active_mask,
            labels=labels,
            groups=groups,
            title=title,
            subtitle=description,
            quiver_stride=args.quiver_stride,
            max_plotted_groups=args.max_plotted_groups,
            output_path=plot_path,
        )
        method_summaries.append(
            summarize_method(
                slug=slug,
                title=title,
                description=description,
                parameters=parameters,
                active_mask=active_mask,
                labels=labels,
                groups=groups,
                max_plotted_groups=args.max_plotted_groups,
                plot_path=plot_path,
            )
        )

    summary_payload = {
        "dataset": to_repo_relative(args.dataset),
        "timestamp": dataset_fields.timestamp,
        "pressure_level_hpa": dataset_fields.pressure_level_hpa,
        "coarsen_factor": args.coarsen_factor,
        "speed_threshold_ms": args.speed_threshold,
        "overview_plot": to_repo_relative(overview_path),
        "methods": [asdict(summary) for summary in method_summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        dataset_path=args.dataset,
        timestamp=dataset_fields.timestamp,
        pressure_level_hpa=dataset_fields.pressure_level_hpa,
        coarsen_factor=args.coarsen_factor,
        speed_threshold=args.speed_threshold,
        methods=method_summaries,
        overview_path=overview_path,
    )

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
