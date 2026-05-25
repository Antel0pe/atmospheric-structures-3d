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
import matplotlib.patheffects as patheffects
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


DEFAULT_LEVELS_250_TO_1000 = (
    "250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,"
    "900,925,950,975,1000"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/thermal-displacement-latitude-agreement/output/"
    "global-score-contours-step5-single-wrap-filter-250-1000"
)


@dataclass(frozen=True)
class ContourSegment:
    level: float
    segment_index: int
    vertices: np.ndarray


@dataclass(frozen=True)
class SegmentDecision:
    keep: bool
    reason: str
    left_contact_runs: int
    right_contact_runs: int
    seam_latitude_gap_degrees: float | None


@dataclass(frozen=True)
class DecidedSegment:
    segment: ContourSegment
    decision: SegmentDecision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate global thermal-displacement score contour maps, plus a "
            "filtered variant that keeps only connected contours with one west "
            "and one east dateline contact."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS_250_TO_1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--edge-tolerance-degrees", type=float, default=0.375)
    parser.add_argument("--max-seam-latitude-gap-degrees", type=float, default=3.0)
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


def contiguous_true_runs(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0
    starts = values & np.concatenate(([True], ~values[:-1]))
    return int(np.count_nonzero(starts))


def side_name(point: np.ndarray, lon_min: float, lon_max: float, tolerance: float) -> str | None:
    lon = float(point[0])
    if lon <= lon_min + tolerance:
        return "left"
    if lon >= lon_max - tolerance:
        return "right"
    return None


def build_contour_segments(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    score: np.ndarray,
    contour_levels: np.ndarray,
) -> list[ContourSegment]:
    fig, ax = plt.subplots()
    contours = ax.contour(longitudes, latitudes, score, levels=contour_levels)
    segments: list[ContourSegment] = []
    for level, level_segments in zip(contours.levels, contours.allsegs):
        for segment_index, segment in enumerate(level_segments):
            vertices = np.asarray(segment, dtype=np.float32)
            if vertices.shape[0] < 2 or not np.all(np.isfinite(vertices)):
                continue
            segments.append(
                ContourSegment(
                    level=float(level),
                    segment_index=int(segment_index),
                    vertices=vertices,
                )
            )
    plt.close(fig)
    return segments


def decide_single_wrap_segment(
    segment: ContourSegment,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    edge_tolerance_degrees: float,
    max_seam_latitude_gap_degrees: float,
) -> SegmentDecision:
    vertices = segment.vertices
    latitudes = vertices[:, 1]
    if np.nanmin(latitudes) < lat_min - edge_tolerance_degrees:
        return SegmentDecision(False, "beyond_south_latitude", 0, 0, None)
    if np.nanmax(latitudes) > lat_max + edge_tolerance_degrees:
        return SegmentDecision(False, "beyond_north_latitude", 0, 0, None)

    left_mask = vertices[:, 0] <= lon_min + edge_tolerance_degrees
    right_mask = vertices[:, 0] >= lon_max - edge_tolerance_degrees
    left_runs = contiguous_true_runs(left_mask)
    right_runs = contiguous_true_runs(right_mask)
    endpoint_sides = [
        side_name(vertices[0], lon_min, lon_max, edge_tolerance_degrees),
        side_name(vertices[-1], lon_min, lon_max, edge_tolerance_degrees),
    ]

    if endpoint_sides.count("left") != 1 or endpoint_sides.count("right") != 1:
        return SegmentDecision(
            False,
            "endpoints_do_not_span_west_and_east_edges",
            left_runs,
            right_runs,
            None,
        )
    if left_runs != 1 or right_runs != 1:
        return SegmentDecision(
            False,
            "multiple_dateline_contacts",
            left_runs,
            right_runs,
            None,
        )

    left_endpoint = vertices[0] if endpoint_sides[0] == "left" else vertices[-1]
    right_endpoint = vertices[0] if endpoint_sides[0] == "right" else vertices[-1]
    seam_gap = float(abs(float(left_endpoint[1]) - float(right_endpoint[1])))
    if seam_gap > max_seam_latitude_gap_degrees:
        return SegmentDecision(
            False,
            "west_east_endpoints_do_not_meet_after_wrap",
            left_runs,
            right_runs,
            seam_gap,
        )

    return SegmentDecision(True, "kept_single_wrap", left_runs, right_runs, seam_gap)


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


def add_segment_label(ax: plt.Axes, segment: ContourSegment) -> None:
    vertices = segment.vertices
    label_index = int(np.clip(len(vertices) // 2, 0, len(vertices) - 1))
    lon, lat = vertices[label_index]
    ax.text(
        float(lon),
        float(lat),
        f"{segment.level:g}",
        color="#1b1b1b",
        fontsize=6,
        ha="center",
        va="center",
        path_effects=[
            patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.9)
        ],
    )


def plot_all_contours(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    contour_step: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap="bwr",
        norm=norm,
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
        linewidths=0.7,
        alpha=0.84,
    )
    ax.clabel(contours, inline=True, fmt="%g", fontsize=6)
    draw_borders_clipped(ax, border_segments)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement score heatmap with all 5-point contours"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_filtered_contours(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    kept_segments: list[ContourSegment],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap="bwr",
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    for segment in kept_segments:
        vertices = segment.vertices
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            color="#161616",
            linewidth=0.9,
            alpha=0.88,
            solid_capstyle="round",
        )
        add_segment_label(ax, segment)

    draw_borders_clipped(ax, border_segments)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement contours filtered to one dateline wrap"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_filter_review_overlay(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    decided_segments: list[DecidedSegment],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score,
        cmap="bwr",
        norm=norm,
        shading="auto",
        rasterized=True,
        alpha=0.55,
    )
    draw_borders_clipped(ax, border_segments)
    for decided in decided_segments:
        vertices = decided.segment.vertices
        if decided.decision.keep:
            color = "#111111"
            linewidth = 0.9
            alpha = 0.9
            zorder = 8
        else:
            color = "#00b050"
            linewidth = 0.85
            alpha = 0.86
            zorder = 7
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )

    kept_count = sum(1 for item in decided_segments if item.decision.keep)
    rejected_count = len(decided_segments) - kept_count
    ax.plot([], [], color="#111111", linewidth=1.4, label=f"kept: {kept_count}")
    ax.plot([], [], color="#00b050", linewidth=1.4, label=f"filtered out: {rejected_count}")
    ax.legend(loc="lower left", framealpha=0.92, fontsize=8)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa contour filter review; black kept, green filtered out"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def validate_kept_segments(
    decided_segments: list[DecidedSegment],
    *,
    max_seam_latitude_gap_degrees: float,
) -> dict[str, int]:
    failures = {
        "kept_not_marked_single_wrap": 0,
        "kept_left_contact_run_failures": 0,
        "kept_right_contact_run_failures": 0,
        "kept_missing_seam_gap": 0,
        "kept_seam_gap_failures": 0,
    }
    for decided in decided_segments:
        if not decided.decision.keep:
            continue
        if decided.decision.reason != "kept_single_wrap":
            failures["kept_not_marked_single_wrap"] += 1
        if decided.decision.left_contact_runs != 1:
            failures["kept_left_contact_run_failures"] += 1
        if decided.decision.right_contact_runs != 1:
            failures["kept_right_contact_run_failures"] += 1
        if decided.decision.seam_latitude_gap_degrees is None:
            failures["kept_missing_seam_gap"] += 1
        elif decided.decision.seam_latitude_gap_degrees > max_seam_latitude_gap_degrees:
            failures["kept_seam_gap_failures"] += 1
    return failures


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    all_contour_dir = output_dir / "all_contours"
    filtered_contour_dir = output_dir / "single_wrap_contours"
    review_overlay_dir = output_dir / "filter_review_overlays"
    all_contour_dir.mkdir(parents=True, exist_ok=True)
    filtered_contour_dir.mkdir(parents=True, exist_ok=True)
    review_overlay_dir.mkdir(parents=True, exist_ok=True)

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
    lon_min = float(np.min(selected_lons))
    lon_max = float(np.max(selected_lons))
    lat_min = float(np.min(selected_lats))
    lat_max = float(np.max(selected_lats))

    summary_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
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

        all_output_path = all_contour_dir / f"heatmap_{slug}.png"
        filtered_output_path = filtered_contour_dir / f"heatmap_{slug}.png"
        review_overlay_path = review_overlay_dir / f"heatmap_{slug}.png"
        plot_all_contours(
            score=score_global,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            level_hpa=level_hpa,
            contour_step=args.contour_step,
            output_path=all_output_path,
            dpi=args.dpi,
        )

        segments = build_contour_segments(
            selected_lons,
            selected_lats,
            score_global,
            contour_levels,
        )
        kept_segments: list[ContourSegment] = []
        decided_segments: list[DecidedSegment] = []
        rejected_by_reason: dict[str, int] = {}
        for segment in segments:
            decision = decide_single_wrap_segment(
                segment,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                edge_tolerance_degrees=args.edge_tolerance_degrees,
                max_seam_latitude_gap_degrees=args.max_seam_latitude_gap_degrees,
            )
            if decision.keep:
                kept_segments.append(segment)
            else:
                rejected_by_reason[decision.reason] = (
                    rejected_by_reason.get(decision.reason, 0) + 1
                )
            decided_segments.append(DecidedSegment(segment=segment, decision=decision))
            segment_rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "score_contour": float(segment.level),
                    "segment_index": int(segment.segment_index),
                    "kept": bool(decision.keep),
                    "decision": decision.reason,
                    "left_contact_runs": int(decision.left_contact_runs),
                    "right_contact_runs": int(decision.right_contact_runs),
                    "seam_latitude_gap_degrees": (
                        "" if decision.seam_latitude_gap_degrees is None else decision.seam_latitude_gap_degrees
                    ),
                    "point_count": int(segment.vertices.shape[0]),
                }
            )
        validation_failures = validate_kept_segments(
            decided_segments,
            max_seam_latitude_gap_degrees=args.max_seam_latitude_gap_degrees,
        )

        plot_filtered_contours(
            score=score_global,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            kept_segments=kept_segments,
            level_hpa=level_hpa,
            output_path=filtered_output_path,
            dpi=args.dpi,
        )
        plot_filter_review_overlay(
            score=score_global,
            latitudes=selected_lats,
            longitudes=selected_lons,
            border_segments=border_segments,
            decided_segments=decided_segments,
            level_hpa=level_hpa,
            output_path=review_overlay_path,
            dpi=args.dpi,
        )
        summary_rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "all_contours_map": display_path(all_output_path),
                "single_wrap_contours_map": display_path(filtered_output_path),
                "filter_review_overlay_map": display_path(review_overlay_path),
                "contour_segment_count": int(len(segments)),
                "kept_single_wrap_segment_count": int(len(kept_segments)),
                "rejected_segment_count": int(len(segments) - len(kept_segments)),
                "rejected_by_reason": rejected_by_reason,
                "kept_validation_failures": validation_failures,
            }
        )

    segment_csv_path = output_dir / "contour_segment_filter_decisions.csv"
    with segment_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pressure_level_hpa",
                "score_contour",
                "segment_index",
                "kept",
                "decision",
                "left_contact_runs",
                "right_contact_runs",
                "seam_latitude_gap_degrees",
                "point_count",
            ],
        )
        writer.writeheader()
        writer.writerows(segment_rows)

    summary = {
        "process": "global thermal-displacement score contours with single-wrap contour filter",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "matching_mode": "same-hemisphere",
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cells on score; "
            "longitude wraps and latitude uses nearest edge."
        ),
        "color_scheme": "blue-white-red with 0 blue, 50 white, 100 red",
        "contour_levels": [float(level) for level in contour_levels],
        "single_wrap_filter": {
            "kept_definition": (
                "A connected Matplotlib contour segment is kept only when its two "
                "endpoints touch opposite longitude edges, it has exactly one "
                "contiguous contact with each longitude edge, both endpoints meet "
                "after wrapping within the configured latitude gap, and all "
                "vertices remain inside the latitude domain."
            ),
            "edge_tolerance_degrees": float(args.edge_tolerance_degrees),
            "max_seam_latitude_gap_degrees": float(args.max_seam_latitude_gap_degrees),
            "longitude_domain_degrees": [lon_min, lon_max],
            "latitude_domain_degrees": [lat_min, lat_max],
        },
        "segment_decisions_csv": display_path(segment_csv_path),
        "outputs": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
