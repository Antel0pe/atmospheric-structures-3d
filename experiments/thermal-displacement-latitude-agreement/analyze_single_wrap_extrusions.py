from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
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

from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    parse_requested_levels,
    resolve_path,
    validate_matching_grid,
)
from generate_single_wrap_score_contours import (  # noqa: E402
    DEFAULT_LEVELS_250_TO_1000,
    DEFAULT_OUTPUT_DIR as DEFAULT_SINGLE_WRAP_OUTPUT_DIR,
    sorted_global_domain,
)
from generate_single_wrap_shape_overlays import (  # noqa: E402
    KeptContour,
    compute_kept_contours,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SINGLE_WRAP_OUTPUT_DIR / "extrusion_analysis"


@dataclass(frozen=True)
class ContourMetrics:
    pressure_level_hpa: float
    hemisphere: str
    score_contour: float
    segment_index: int
    median_latitude: float
    min_latitude: float
    max_latitude: float
    latitude_std_deg: float
    robust_amplitude_deg: float
    max_abs_deviation_deg: float
    total_variation_deg: float
    curvature_rms: float
    longitude_reversal_count: int
    poleward_extent_deg: float
    poleward_peak_longitude: float
    poleward_peak_latitude: float
    equatorward_extent_deg: float
    equatorward_peak_longitude: float
    equatorward_peak_latitude: float
    poleward_area_deg2: float
    equatorward_area_deg2: float
    deformation_score: float
    cold_extrusion_score: float
    hot_extrusion_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute shape and extrusion metrics for kept single-wrap "
            "thermal-displacement score contours."
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


def sorted_profile_vertices(contour: KeptContour) -> np.ndarray:
    vertices = np.asarray(contour.vertices, dtype=np.float32)
    order = np.argsort(vertices[:, 0], kind="mergesort")
    return vertices[order]


def unique_longitude_profile(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitudes = np.asarray(vertices[:, 0], dtype=np.float64)
    latitudes = np.asarray(vertices[:, 1], dtype=np.float64)
    rounded_lon = np.round(longitudes, 4)
    unique_lons: list[float] = []
    profile_lats: list[float] = []
    for lon in np.unique(rounded_lon):
        mask = rounded_lon == lon
        unique_lons.append(float(np.mean(longitudes[mask])))
        profile_lats.append(float(np.mean(latitudes[mask])))
    return np.asarray(unique_lons), np.asarray(profile_lats)


def compute_metrics(contour: KeptContour) -> ContourMetrics:
    vertices = sorted_profile_vertices(contour)
    lons, lats = unique_longitude_profile(vertices)
    median_lat = float(contour.median_latitude)
    deviations = lats - median_lat
    abs_deviation = np.abs(deviations)
    robust_amplitude = float(np.nanpercentile(lats, 95) - np.nanpercentile(lats, 5))
    max_abs_deviation = float(np.nanmax(abs_deviation))
    total_variation = float(np.nansum(np.abs(np.diff(lats))))
    if len(lats) >= 3:
        curvature_rms = float(np.sqrt(np.nanmean(np.diff(lats, n=2) ** 2)))
    else:
        curvature_rms = 0.0
    raw_longitudes = np.asarray(contour.vertices[:, 0], dtype=np.float32)
    longitude_reversal_count = int(np.count_nonzero(np.diff(raw_longitudes) < -0.05))

    if contour.hemisphere == "northern":
        poleward_values = lats - median_lat
        equatorward_values = median_lat - lats
    else:
        poleward_values = median_lat - lats
        equatorward_values = lats - median_lat

    poleward_extent = float(np.nanmax(poleward_values))
    equatorward_extent = float(np.nanmax(equatorward_values))
    poleward_index = int(np.nanargmax(poleward_values))
    equatorward_index = int(np.nanargmax(equatorward_values))
    poleward_positive = np.maximum(poleward_values, 0.0)
    equatorward_positive = np.maximum(equatorward_values, 0.0)
    poleward_area = float(np.trapezoid(poleward_positive, lons))
    equatorward_area = float(np.trapezoid(equatorward_positive, lons))

    score = float(contour.score_contour)
    polar_weight = (100.0 - score) / 100.0
    equator_weight = score / 100.0
    deformation_score = robust_amplitude + 0.018 * total_variation + 14.0 * curvature_rms
    cold_extrusion_score = polar_weight * (
        equatorward_extent + 0.006 * equatorward_area + 0.25 * robust_amplitude
    )
    hot_extrusion_score = equator_weight * (
        poleward_extent + 0.006 * poleward_area + 0.25 * robust_amplitude
    )

    return ContourMetrics(
        pressure_level_hpa=float(contour.pressure_level_hpa),
        hemisphere=contour.hemisphere,
        score_contour=score,
        segment_index=int(contour.segment_index),
        median_latitude=median_lat,
        min_latitude=float(np.nanmin(lats)),
        max_latitude=float(np.nanmax(lats)),
        latitude_std_deg=float(np.nanstd(lats)),
        robust_amplitude_deg=robust_amplitude,
        max_abs_deviation_deg=max_abs_deviation,
        total_variation_deg=total_variation,
        curvature_rms=curvature_rms,
        longitude_reversal_count=longitude_reversal_count,
        poleward_extent_deg=poleward_extent,
        poleward_peak_longitude=float(lons[poleward_index]),
        poleward_peak_latitude=float(lats[poleward_index]),
        equatorward_extent_deg=equatorward_extent,
        equatorward_peak_longitude=float(lons[equatorward_index]),
        equatorward_peak_latitude=float(lats[equatorward_index]),
        poleward_area_deg2=poleward_area,
        equatorward_area_deg2=equatorward_area,
        deformation_score=float(deformation_score),
        cold_extrusion_score=float(cold_extrusion_score),
        hot_extrusion_score=float(hot_extrusion_score),
    )


def write_metrics_csv(metrics: list[ContourMetrics], output_path: Path) -> None:
    rows = [asdict(metric) for metric in metrics]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def top_metrics(
    metrics: list[ContourMetrics],
    *,
    hemisphere: str,
    key: str,
    limit: int = 15,
) -> list[ContourMetrics]:
    subset = [metric for metric in metrics if metric.hemisphere == hemisphere]
    return sorted(subset, key=lambda metric: getattr(metric, key), reverse=True)[:limit]


def plot_ranked_bars(
    metrics: list[ContourMetrics],
    *,
    key: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    ordered = sorted(metrics, key=lambda metric: getattr(metric, key), reverse=True)
    labels = [
        f"{metric.pressure_level_hpa:g}hPa s{metric.score_contour:g} "
        f"m{metric.median_latitude:.1f}"
        for metric in ordered
    ]
    values = [float(getattr(metric, key)) for metric in ordered]
    colors = ["#386cb0" if metric.hemisphere == "northern" else "#fdb462" for metric in ordered]
    fig_height = max(4.8, 0.34 * max(len(ordered), 1) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    y_positions = np.arange(len(ordered))
    ax.barh(y_positions, values, color=colors, edgecolor="none", alpha=0.88)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(key.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.75)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_highlighted_contours(
    contours: list[KeptContour],
    ranked_metrics: list[ContourMetrics],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    metric_ids = {
        (
            metric.pressure_level_hpa,
            metric.hemisphere,
            metric.score_contour,
            metric.segment_index,
        )
        for metric in ranked_metrics
    }
    highlighted = [
        contour
        for contour in contours
        if (
            contour.pressure_level_hpa,
            contour.hemisphere,
            contour.score_contour,
            contour.segment_index,
        )
        in metric_ids
    ]
    fig, ax = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for index, contour in enumerate(highlighted):
        vertices = contour.vertices
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            linewidth=1.25,
            alpha=0.9,
            color=cmap(index % 10),
            label=(
                f"{contour.pressure_level_hpa:g}hPa "
                f"{contour.hemisphere[:1].upper()} s{contour.score_contour:g} "
                f"m{contour.median_latitude:.1f}"
            ),
        )
    ax.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.6)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def metric_to_summary(metric: ContourMetrics) -> dict[str, object]:
    return {
        "pressure_level_hpa": metric.pressure_level_hpa,
        "hemisphere": metric.hemisphere,
        "score_contour": metric.score_contour,
        "median_latitude": metric.median_latitude,
        "robust_amplitude_deg": metric.robust_amplitude_deg,
        "poleward_extent_deg": metric.poleward_extent_deg,
        "poleward_peak": {
            "longitude": metric.poleward_peak_longitude,
            "latitude": metric.poleward_peak_latitude,
        },
        "equatorward_extent_deg": metric.equatorward_extent_deg,
        "equatorward_peak": {
            "longitude": metric.equatorward_peak_longitude,
            "latitude": metric.equatorward_peak_latitude,
        },
        "deformation_score": metric.deformation_score,
        "cold_extrusion_score": metric.cold_extrusion_score,
        "hot_extrusion_score": metric.hot_extrusion_score,
    }


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    metrics = [compute_metrics(contour) for contour in kept_contours]
    metrics_csv = output_dir / "single_wrap_contour_extrusion_metrics.csv"
    write_metrics_csv(metrics, metrics_csv)

    ranking_specs = [
        ("deformation_score", "Most deformed kept single-wrap contours"),
        ("cold_extrusion_score", "Best polar-like equatorward extrusion candidates"),
        ("hot_extrusion_score", "Best equator-like poleward extrusion candidates"),
    ]
    ranking_summary: dict[str, dict[str, list[dict[str, object]]]] = {}
    for key, title in ranking_specs:
        ranking_summary[key] = {}
        for hemisphere in ("northern", "southern"):
            ranked = top_metrics(metrics, hemisphere=hemisphere, key=key, limit=15)
            ranking_summary[key][hemisphere] = [metric_to_summary(metric) for metric in ranked]
            plot_ranked_bars(
                ranked,
                key=key,
                title=f"{title}: {hemisphere}",
                output_path=plots_dir / f"{key}_{hemisphere}_top15.png",
                dpi=args.dpi,
            )
            plot_highlighted_contours(
                kept_contours,
                ranked[:8],
                title=f"{title}: {hemisphere} top 8",
                output_path=plots_dir / f"{key}_{hemisphere}_top8_contours.png",
                dpi=args.dpi,
            )

    summary = {
        "process": "extrusion and deformation metrics for kept single-wrap contours",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [float(level) for level in selected_levels],
        "matching_mode": "same-hemisphere",
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cells on score; "
            "longitude wraps and latitude uses nearest edge."
        ),
        "interpretation": {
            "polar_like_low_score": "Low Thermal Displacement score is polar-like.",
            "equator_like_high_score": "High Thermal Displacement score is equator-like.",
            "cold_extrusion_score": (
                "Heuristic ranking for polar-like contours that bulge equatorward. "
                "It combines equatorward extent, equatorward area, contour amplitude, "
                "and a low-score weight."
            ),
            "hot_extrusion_score": (
                "Heuristic ranking for equator-like contours that bulge poleward. "
                "It combines poleward extent, poleward area, contour amplitude, "
                "and a high-score weight."
            ),
            "shape_only_caveat": (
                "These rankings are based on contour geometry and Thermal Displacement "
                "identity only. They do not prove parcel motion or a full synoptic front."
            ),
        },
        "kept_contour_count": len(metrics),
        "metrics_csv": display_path(metrics_csv),
        "plots_dir": display_path(plots_dir),
        "rankings": ranking_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
