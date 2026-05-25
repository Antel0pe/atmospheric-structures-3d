from __future__ import annotations

import argparse
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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_latitude_score_lines import slug_for_level  # noqa: E402
from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)
from generate_single_wrap_score_contours import (  # noqa: E402
    DEFAULT_LEVELS_250_TO_1000,
    DEFAULT_OUTPUT_DIR as DEFAULT_SINGLE_WRAP_OUTPUT_DIR,
    ContourSegment,
    build_contour_segments,
    decide_single_wrap_segment,
    sorted_global_domain,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SINGLE_WRAP_OUTPUT_DIR / "shape_overlays"


@dataclass(frozen=True)
class KeptContour:
    pressure_level_hpa: float
    score_contour: float
    segment_index: int
    hemisphere: str
    median_latitude: float
    vertices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot kept single-wrap thermal-displacement score contours as "
            "longitude-latitude line overlays, split by hemisphere."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS_250_TO_1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--contour-step", type=float, default=5.0)
    parser.add_argument("--edge-tolerance-degrees", type=float, default=0.375)
    parser.add_argument("--max-seam-latitude-gap-degrees", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def west_to_east_vertices(vertices: np.ndarray) -> np.ndarray:
    ordered = np.asarray(vertices, dtype=np.float32)
    if float(ordered[0, 0]) > float(ordered[-1, 0]):
        ordered = ordered[::-1].copy()
    return ordered


def contour_hemisphere(vertices: np.ndarray) -> tuple[str, float]:
    median_latitude = float(np.nanmedian(vertices[:, 1]))
    hemisphere = "northern" if median_latitude >= 0.0 else "southern"
    return hemisphere, median_latitude


def compute_kept_contours(
    temperature: xr.DataArray,
    climatology: xr.DataArray,
    selected_time: np.datetime64,
    selected_levels: list[float],
    source_latitudes: np.ndarray,
    selected_lats: np.ndarray,
    selected_lons: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    contour_levels: np.ndarray,
    edge_tolerance_degrees: float,
    max_seam_latitude_gap_degrees: float,
    smooth_sigma_cells: float,
) -> list[KeptContour]:
    lon_min = float(np.min(selected_lons))
    lon_max = float(np.max(selected_lons))
    lat_min = float(np.min(selected_lats))
    lat_max = float(np.max(selected_lats))
    kept_contours: list[KeptContour] = []

    for level_hpa in selected_levels:
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
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, smooth_sigma_cells)
        score_global = score_smoothed[np.ix_(lat_indices, lon_indices)]
        segments = build_contour_segments(
            selected_lons,
            selected_lats,
            score_global,
            contour_levels,
        )
        for segment in segments:
            decision = decide_single_wrap_segment(
                segment,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                edge_tolerance_degrees=edge_tolerance_degrees,
                max_seam_latitude_gap_degrees=max_seam_latitude_gap_degrees,
            )
            if not decision.keep:
                continue
            vertices = west_to_east_vertices(segment.vertices)
            hemisphere, median_latitude = contour_hemisphere(vertices)
            kept_contours.append(
                KeptContour(
                    pressure_level_hpa=float(level_hpa),
                    score_contour=float(segment.level),
                    segment_index=int(segment.segment_index),
                    hemisphere=hemisphere,
                    median_latitude=median_latitude,
                    vertices=vertices,
                )
            )

    return kept_contours


def plot_shape_overlay(
    contours: list[KeptContour],
    *,
    hemisphere: str,
    level_hpa: float | None,
    centered: bool,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    if not contours:
        ax.text(0.5, 0.5, "No kept contours", ha="center", va="center", transform=ax.transAxes)
    cmap = plt.get_cmap("viridis")
    scores = np.asarray([contour.score_contour for contour in contours], dtype=np.float32)
    score_min = float(np.nanmin(scores)) if scores.size else 0.0
    score_max = float(np.nanmax(scores)) if scores.size else 100.0
    denominator = max(score_max - score_min, 1.0)

    for contour in contours:
        vertices = contour.vertices
        y = vertices[:, 1] - contour.median_latitude if centered else vertices[:, 1]
        normalized_score = (contour.score_contour - score_min) / denominator
        label = None
        ax.plot(
            vertices[:, 0],
            y,
            color=cmap(float(normalized_score)),
            linewidth=0.85,
            alpha=0.58 if level_hpa is None else 0.72,
            label=label,
        )

    ax.axhline(0.0, color="#333333", linewidth=0.6, alpha=0.55)
    ax.set_xlim(-180.0, 180.0)
    if centered:
        ax.set_ylim(-35.0, 35.0)
        ax.set_ylabel("Latitude offset from this contour's median (degrees)")
    else:
        ax.set_ylim(0.0, 90.0) if hemisphere == "northern" else ax.set_ylim(-90.0, 0.0)
        ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
    level_text = "all 250-1000 hPa levels" if level_hpa is None else f"{level_hpa:g} hPa"
    mode_text = "median-centered shape" if centered else "actual-latitude shape"
    ax.set_title(
        f"{hemisphere.capitalize()} kept single-wrap contours, {mode_text}, {level_text}"
    )
    ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.75)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(score_min, score_max))
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score contour")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def contour_sort_key(contour: KeptContour) -> tuple[float, float, float, int]:
    return (
        contour.pressure_level_hpa,
        contour.median_latitude,
        contour.score_contour,
        contour.segment_index,
    )


def plot_stacked_rows(
    contours: list[KeptContour],
    *,
    hemisphere: str,
    level_hpa: float | None,
    output_path: Path,
    dpi: int,
) -> None:
    sorted_contours = sorted(contours, key=contour_sort_key)
    fig_height = min(max(6.0, 0.22 * max(len(sorted_contours), 1) + 1.8), 24.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    scores = np.asarray([contour.score_contour for contour in sorted_contours], dtype=np.float32)
    score_min = float(np.nanmin(scores)) if scores.size else 0.0
    score_max = float(np.nanmax(scores)) if scores.size else 100.0
    denominator = max(score_max - score_min, 1.0)
    row_scale = 0.055

    if not sorted_contours:
        ax.text(0.5, 0.5, "No kept contours", ha="center", va="center", transform=ax.transAxes)

    plotted_min_y = 0.0
    plotted_max_y = 0.0
    for row_index, contour in enumerate(sorted_contours):
        vertices = contour.vertices
        y = row_index + (vertices[:, 1] - contour.median_latitude) * row_scale
        plotted_min_y = min(plotted_min_y, float(np.nanmin(y)))
        plotted_max_y = max(plotted_max_y, float(np.nanmax(y)))
        normalized_score = (contour.score_contour - score_min) / denominator
        ax.plot(
            vertices[:, 0],
            y,
            color=cmap(float(normalized_score)),
            linewidth=0.95,
            alpha=0.88,
        )
        ax.axhline(row_index, color="#e4e4e4", linewidth=0.35, zorder=0)

    ax.set_xlim(-180.0, 180.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("One kept contour per row; wiggle is scaled latitude deviation")
    if sorted_contours:
        ax.set_ylim(plotted_min_y - 0.4, plotted_max_y + 0.4)
    level_text = "all 250-1000 hPa levels" if level_hpa is None else f"{level_hpa:g} hPa"
    ax.set_title(
        f"{hemisphere.capitalize()} kept single-wrap contour rows, {level_text}"
    )
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.75)
    if len(sorted_contours) <= 36:
        ax.set_yticks(np.arange(len(sorted_contours)))
        ax.set_yticklabels(
            [
                f"{contour.pressure_level_hpa:g}hPa s{contour.score_contour:g} "
                f"m{contour.median_latitude:.1f}"
                for contour in sorted_contours
            ],
            fontsize=6,
        )
    else:
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(score_min, score_max))
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.01)
    colorbar.set_label("Thermal-displacement score contour")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "actual_latitude"
    centered_dir = output_dir / "median_centered"
    stacked_dir = output_dir / "stacked_rows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    centered_dir.mkdir(parents=True, exist_ok=True)
    stacked_dir.mkdir(parents=True, exist_ok=True)

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
    contour_levels = np.arange(args.contour_step, 100.0, args.contour_step, dtype=np.float32)
    kept_contours = compute_kept_contours(
        temperature=temperature,
        climatology=climatology,
        selected_time=selected_time,
        selected_levels=selected_levels,
        source_latitudes=source_latitudes,
        selected_lats=selected_lats,
        selected_lons=selected_lons,
        lat_indices=lat_indices,
        lon_indices=lon_indices,
        contour_levels=contour_levels,
        edge_tolerance_degrees=args.edge_tolerance_degrees,
        max_seam_latitude_gap_degrees=args.max_seam_latitude_gap_degrees,
        smooth_sigma_cells=args.smooth_sigma_cells,
    )

    summary_rows: list[dict[str, object]] = []
    for hemisphere in ("northern", "southern"):
        hemisphere_contours = [
            contour for contour in kept_contours if contour.hemisphere == hemisphere
        ]
        raw_output = raw_dir / f"{hemisphere}_all_levels.png"
        centered_output = centered_dir / f"{hemisphere}_all_levels.png"
        stacked_output = stacked_dir / f"{hemisphere}_all_levels.png"
        plot_shape_overlay(
            hemisphere_contours,
            hemisphere=hemisphere,
            level_hpa=None,
            centered=False,
            output_path=raw_output,
            dpi=args.dpi,
        )
        plot_shape_overlay(
            hemisphere_contours,
            hemisphere=hemisphere,
            level_hpa=None,
            centered=True,
            output_path=centered_output,
            dpi=args.dpi,
        )
        plot_stacked_rows(
            hemisphere_contours,
            hemisphere=hemisphere,
            level_hpa=None,
            output_path=stacked_output,
            dpi=args.dpi,
        )
        summary_rows.append(
            {
                "pressure_level_hpa": "all",
                "hemisphere": hemisphere,
                "kept_contour_count": len(hemisphere_contours),
                "actual_latitude_plot": display_path(raw_output),
                "median_centered_plot": display_path(centered_output),
                "stacked_rows_plot": display_path(stacked_output),
            }
        )

        for level_hpa in selected_levels:
            level_contours = [
                contour
                for contour in hemisphere_contours
                if contour.pressure_level_hpa == float(level_hpa)
            ]
            slug = slug_for_level(level_hpa)
            raw_output = raw_dir / f"{hemisphere}_{slug}.png"
            centered_output = centered_dir / f"{hemisphere}_{slug}.png"
            stacked_output = stacked_dir / f"{hemisphere}_{slug}.png"
            plot_shape_overlay(
                level_contours,
                hemisphere=hemisphere,
                level_hpa=float(level_hpa),
                centered=False,
                output_path=raw_output,
                dpi=args.dpi,
            )
            plot_shape_overlay(
                level_contours,
                hemisphere=hemisphere,
                level_hpa=float(level_hpa),
                centered=True,
                output_path=centered_output,
                dpi=args.dpi,
            )
            plot_stacked_rows(
                level_contours,
                hemisphere=hemisphere,
                level_hpa=float(level_hpa),
                output_path=stacked_output,
                dpi=args.dpi,
            )
            summary_rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "hemisphere": hemisphere,
                    "kept_contour_count": len(level_contours),
                    "actual_latitude_plot": display_path(raw_output),
                    "median_centered_plot": display_path(centered_output),
                    "stacked_rows_plot": display_path(stacked_output),
                }
            )

    summary = {
        "process": "shape overlays for kept single-wrap thermal-displacement score contours",
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
        "hemisphere_rule": "median latitude >= 0 is northern; otherwise southern",
        "plot_families": {
            "actual_latitude": (
                "x is longitude and y is real map latitude, preserving where each line starts."
            ),
            "median_centered": (
                "x is longitude and y is latitude minus that contour's median latitude, "
                "so curves can be compared on top of each other."
            ),
            "stacked_rows": (
                "x is longitude and each contour gets one row; the drawn wiggle is "
                "a scaled latitude deviation around that contour's median latitude."
            ),
        },
        "kept_contour_count": len(kept_contours),
        "outputs": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
