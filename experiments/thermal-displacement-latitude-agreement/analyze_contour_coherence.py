from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)
from generate_latitude_score_lines import select_sorted_longitudes, slug_for_level  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-score-smoothed-sigma20-smoothed-agreement/"
    "contour-coherence-analysis-all-lons"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Good-faith statistical probes for thermal-displacement contour coherence."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--lon-min", type=float, default=-125.0)
    parser.add_argument("--lon-max", type=float, default=-50.0)
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=89.0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def selected_domain(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_indices, lon_signed = select_sorted_longitudes(longitudes, lon_min, lon_max)
    lat_indices = np.flatnonzero((latitudes >= lat_min) & (latitudes <= lat_max))
    lat_indices = lat_indices[np.argsort(-latitudes[lat_indices])]
    return lat_indices, lon_indices, latitudes[lat_indices], lon_signed


def crossing_latitude(
    latitudes_desc: np.ndarray,
    score_profile: np.ndarray,
    threshold: float,
) -> float:
    values = score_profile.astype(np.float64)
    lats = latitudes_desc.astype(np.float64)
    signed = values - threshold
    crossing_indices = np.flatnonzero(signed[:-1] * signed[1:] <= 0.0)
    if crossing_indices.size == 0:
        return np.nan
    candidates: list[tuple[float, float]] = []
    for index in crossing_indices:
        y0 = values[index]
        y1 = values[index + 1]
        x0 = lats[index]
        x1 = lats[index + 1]
        if y1 == y0:
            lat = 0.5 * (x0 + x1)
            local_slope = 0.0
        else:
            fraction = (threshold - y0) / (y1 - y0)
            lat = x0 + fraction * (x1 - x0)
            local_slope = abs((y1 - y0) / max(abs(x1 - x0), 1e-6))
        candidates.append((local_slope, float(lat)))
    return max(candidates, key=lambda item: item[0])[1]


def contour_latitudes(
    latitudes_desc: np.ndarray,
    score_subset: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    out = np.full((thresholds.size, score_subset.shape[1]), np.nan, dtype=np.float64)
    for threshold_index, threshold in enumerate(thresholds):
        for lon_index in range(score_subset.shape[1]):
            out[threshold_index, lon_index] = crossing_latitude(
                latitudes_desc,
                score_subset[:, lon_index],
                float(threshold),
            )
    return out


def pearson_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(np.count_nonzero(mask))
    if n < 10:
        return np.nan, n
    av = a[mask]
    bv = b[mask]
    if np.nanstd(av) == 0.0 or np.nanstd(bv) == 0.0:
        return np.nan, n
    return float(np.corrcoef(av, bv)[0, 1]), n


def otsu_threshold(values: np.ndarray, bins: int = 100) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    counts, edges = np.histogram(finite, bins=bins, range=(0.0, 100.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(np.sum(counts))
    if total == 0.0:
        return {"threshold": np.nan, "separation_ratio": np.nan, "cold_fraction": np.nan, "hot_fraction": np.nan}
    prob = counts / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros_like(denom)
    valid = denom > 0.0
    between[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    total_var = float(np.sum(prob * (centers - mu_total) ** 2))
    best = int(np.argmax(between))
    threshold = float(centers[best])
    ratio = float(between[best] / total_var) if total_var > 0.0 else np.nan
    cold_fraction = float(np.mean(finite <= threshold))
    return {
        "threshold": threshold,
        "separation_ratio": ratio,
        "cold_fraction": cold_fraction,
        "hot_fraction": 1.0 - cold_fraction,
    }


def plot_field_correlation_matrix(
    levels: np.ndarray,
    score_by_level: dict[float, np.ndarray],
    output_path: Path,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(levels)
    corr = np.full((n, n), np.nan)
    rmse = np.full((n, n), np.nan)
    for i, level_a in enumerate(levels):
        a = score_by_level[float(level_a)].ravel()
        for j, level_b in enumerate(levels):
            b = score_by_level[float(level_b)].ravel()
            corr[i, j], _ = pearson_pair(a, b)
            mask = np.isfinite(a) & np.isfinite(b)
            rmse[i, j] = float(np.sqrt(np.nanmean((a[mask] - b[mask]) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    labels = [f"{level:g}" for level in levels]
    im0 = axes[0].imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0].set_title("Full-field score correlation")
    im1 = axes[1].imshow(rmse, cmap="magma_r")
    axes[1].set_title("Full-field score RMSE")
    for ax in axes:
        ax.set_xticks(np.arange(n), labels=labels)
        ax.set_yticks(np.arange(n), labels=labels)
        ax.set_xlabel("hPa")
        ax.set_ylabel("hPa")
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
            axes[1].text(j, i, f"{rmse[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im0, ax=axes[0], shrink=0.82)
    fig.colorbar(im1, ax=axes[1], shrink=0.82)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return corr, rmse


def plot_cross_level_contour_metrics(
    levels: np.ndarray,
    thresholds: np.ndarray,
    contour_by_level: dict[float, np.ndarray],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, float | str]]:
    pairs: list[tuple[float, float]] = []
    for i, level_a in enumerate(levels):
        for level_b in levels[i + 1 :]:
            pairs.append((float(level_a), float(level_b)))
    corr_matrix = np.full((thresholds.size, len(pairs)), np.nan)
    sep_matrix = np.full_like(corr_matrix, np.nan)
    rows: list[dict[str, float | str]] = []
    for t_index, threshold in enumerate(thresholds):
        for p_index, (level_a, level_b) in enumerate(pairs):
            a = contour_by_level[level_a][t_index]
            b = contour_by_level[level_b][t_index]
            corr, n = pearson_pair(a, b)
            mask = np.isfinite(a) & np.isfinite(b)
            sep = float(np.nanmean(np.abs(a[mask] - b[mask]))) if np.any(mask) else np.nan
            corr_matrix[t_index, p_index] = corr
            sep_matrix[t_index, p_index] = sep
            rows.append(
                {
                    "score_contour": float(threshold),
                    "level_pair": f"{level_a:g}-{level_b:g}",
                    "correlation": corr,
                    "mean_abs_latitude_separation": sep,
                    "n_longitudes": n,
                }
            )

    pair_labels = [f"{a:g}-{b:g}" for a, b in pairs]
    for matrix, name, cmap, vmin, vmax, label in [
        (corr_matrix, "cross_level_same_score_contour_correlation", "coolwarm", -1.0, 1.0, "Pearson r"),
        (sep_matrix, "cross_level_same_score_contour_separation", "viridis", None, None, "Mean absolute latitude separation"),
    ]:
        fig, ax = plt.subplots(figsize=(10.6, 7.4), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(pair_labels)), labels=pair_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(thresholds.size), labels=[f"{t:g}" for t in thresholds])
        ax.set_xlabel("Pressure-level pair")
        ax.set_ylabel("Score contour")
        ax.set_title(name.replace("_", " "))
        fig.colorbar(im, ax=ax, pad=0.01, label=label)
        fig.savefig(output_dir / f"{name}.png", dpi=dpi)
        plt.close(fig)
    return rows


def plot_contour_latitudes_and_spacing(
    level: float,
    longitudes: np.ndarray,
    thresholds: np.ndarray,
    contours: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> dict[str, float]:
    slug = slug_for_level(level)
    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("turbo")
    norm = mcolors.Normalize(vmin=float(np.min(thresholds)), vmax=float(np.max(thresholds)))
    for t_index, threshold in enumerate(thresholds):
        ax.plot(
            longitudes,
            contours[t_index],
            color=cmap(norm(float(threshold))),
            linewidth=1.0,
            alpha=0.88,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude of score contour")
    ax.set_title(f"{level:g} hPa contour-latitude curves by score")
    ax.grid(color="#d6d6d6", linewidth=0.6, alpha=0.75)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, label="Score contour")
    fig.savefig(output_dir / f"contour_latitudes_{slug}.png", dpi=dpi)
    plt.close(fig)

    spacing = np.abs(np.diff(contours, axis=0))
    band_centers = 0.5 * (thresholds[:-1] + thresholds[1:])
    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    im = ax.imshow(
        spacing,
        aspect="auto",
        origin="lower",
        extent=[float(longitudes[0]), float(longitudes[-1]), float(band_centers[0]), float(band_centers[-1])],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Score band center")
    ax.set_title(f"{level:g} hPa contour spacing; dark = grouped/compressed")
    fig.colorbar(im, ax=ax, pad=0.01, label="Latitude degrees between adjacent 5-point contours")
    fig.savefig(output_dir / f"contour_spacing_compression_{slug}.png", dpi=dpi)
    plt.close(fig)

    mean_spacing_by_band = np.nanmean(spacing, axis=1)
    tight_index = int(np.nanargmin(mean_spacing_by_band))
    loose_index = int(np.nanargmax(mean_spacing_by_band))
    return {
        "pressure_level_hpa": float(level),
        "tightest_score_band_center": float(band_centers[tight_index]),
        "tightest_mean_spacing_deg": float(mean_spacing_by_band[tight_index]),
        "loosest_score_band_center": float(band_centers[loose_index]),
        "loosest_mean_spacing_deg": float(mean_spacing_by_band[loose_index]),
        "overall_mean_spacing_deg": float(np.nanmean(spacing)),
    }


def plot_vertical_std(
    thresholds: np.ndarray,
    longitudes: np.ndarray,
    contour_by_level: dict[float, np.ndarray],
    output_path: Path,
    dpi: int,
) -> np.ndarray:
    stack = np.stack([value for value in contour_by_level.values()], axis=0)
    std = np.nanstd(stack, axis=0)
    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    im = ax.imshow(
        std,
        aspect="auto",
        origin="lower",
        extent=[float(longitudes[0]), float(longitudes[-1]), float(thresholds[0]), float(thresholds[-1])],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Score contour")
    ax.set_title("Cross-pressure spread of same-score contour latitude")
    fig.colorbar(im, ax=ax, pad=0.01, label="Std dev across pressure levels, latitude degrees")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return std


def plot_otsu_histogram(
    level: float,
    values: np.ndarray,
    stats: dict[str, float],
    output_path: Path,
    dpi: int,
) -> None:
    finite = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.hist(finite, bins=np.arange(0.0, 101.0, 1.0), color="#668fbd", edgecolor="none")
    ax.axvline(stats["threshold"], color="#d22", linewidth=1.7, label=f"Otsu threshold {stats['threshold']:.1f}")
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Thermal-displacement score")
    ax.set_ylabel("Cell count")
    ax.set_title(
        f"{level:g} hPa score distribution; separation ratio {stats['separation_ratio']:.2f}"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.75)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = output_dir / "plots"
    hist_dir = output_dir / "score-histograms-otsu"
    plot_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(dataset_path)
    clim_ds = xr.open_dataset(climatology_path)
    temperature = ds[TEMPERATURE_VARIABLE]
    climatology = clim_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)
    selected_time = choose_timestamp(temperature, args.timestamp)
    levels = np.asarray(parse_requested_levels(args.pressure_levels, np.asarray(temperature.pressure_level.values)), dtype=np.float64)
    latitudes = np.asarray(temperature.latitude.values, dtype=np.float32)
    longitudes = np.asarray(temperature.longitude.values, dtype=np.float32)
    lat_indices, lon_indices, selected_lats, selected_lons = selected_domain(
        latitudes,
        longitudes,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    thresholds = np.arange(5.0, 100.0, 5.0, dtype=np.float64)
    score_by_level: dict[float, np.ndarray] = {}
    contour_by_level: dict[float, np.ndarray] = {}
    compression_rows: list[dict[str, float]] = []
    otsu_rows: list[dict[str, float]] = []

    for level in levels:
        print(f"Processing {level:g} hPa")
        raw = (
            temperature.sel(valid_time=selected_time, pressure_level=float(level))
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        clim = (
            climatology.sel(pressure_level=float(level))
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        matched = match_equivalent_latitude(raw, clim, latitudes, "same-hemisphere")
        score = thermal_displacement_score_points(np.clip(matched, -90.0, 90.0), latitudes)
        score = smooth_wrapped_lon(score, args.smooth_sigma_cells)
        subset = score[np.ix_(lat_indices, lon_indices)]
        score_by_level[float(level)] = subset
        contours = contour_latitudes(selected_lats, subset, thresholds)
        contour_by_level[float(level)] = contours
        compression_rows.append(
            plot_contour_latitudes_and_spacing(
                float(level),
                selected_lons,
                thresholds,
                contours,
                plot_dir,
                args.dpi,
            )
        )
        otsu = otsu_threshold(subset)
        otsu_rows.append({"pressure_level_hpa": float(level), **otsu})
        plot_otsu_histogram(
            float(level),
            subset,
            otsu,
            hist_dir / f"score_histogram_otsu_{slug_for_level(float(level))}.png",
            args.dpi,
        )

    corr, rmse = plot_field_correlation_matrix(
        levels,
        score_by_level,
        plot_dir / "full_field_pressure_level_correlation_rmse.png",
        args.dpi,
    )
    cross_rows = plot_cross_level_contour_metrics(
        levels,
        thresholds,
        contour_by_level,
        plot_dir,
        args.dpi,
    )
    vertical_std = plot_vertical_std(
        thresholds,
        selected_lons,
        contour_by_level,
        plot_dir / "cross_pressure_contour_latitude_spread_by_score_longitude.png",
        args.dpi,
    )

    with (output_dir / "cross_level_contour_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cross_rows)
    with (output_dir / "within_level_contour_spacing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compression_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compression_rows)
    with (output_dir / "otsu_score_split_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(otsu_rows[0].keys()))
        writer.writeheader()
        writer.writerows(otsu_rows)
    with (output_dir / "field_correlation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pressure_level_hpa"] + [f"{level:g}" for level in levels])
        for level, row in zip(levels, corr):
            writer.writerow([float(level)] + [float(value) for value in row])

    best_cross = sorted(
        [row for row in cross_rows if np.isfinite(row["correlation"])],
        key=lambda row: abs(float(row["correlation"])),
        reverse=True,
    )[:8]
    tightest = sorted(compression_rows, key=lambda row: row["tightest_mean_spacing_deg"])
    summary = {
        "process": "Contour coherence and separability diagnostics for same-hemisphere sigma-20 thermal-displacement score.",
        "domain": {
            "longitude": [args.lon_min, args.lon_max],
            "latitude": [args.lat_max, args.lat_min],
        },
        "pressure_levels_hpa": [float(level) for level in levels],
        "score_smoothing_sigma_cells": args.smooth_sigma_cells,
        "score_contours": [float(value) for value in thresholds],
        "best_cross_level_same_score_correlations": best_cross,
        "within_level_spacing_summary": compression_rows,
        "otsu_score_split_metrics": otsu_rows,
        "cross_pressure_contour_spread_mean_deg": float(np.nanmean(vertical_std)),
        "cross_pressure_contour_spread_min_deg": float(np.nanmin(vertical_std)),
        "cross_pressure_contour_spread_max_deg": float(np.nanmax(vertical_std)),
        "outputs": {
            "plots": "plots/",
            "otsu_histograms": "score-histograms-otsu/",
            "cross_level_metrics_csv": "cross_level_contour_metrics.csv",
            "within_level_spacing_csv": "within_level_contour_spacing_summary.csv",
            "otsu_metrics_csv": "otsu_score_split_metrics.csv",
            "field_correlation_csv": "field_correlation_matrix.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Contour Coherence Analysis",
        "",
        "This is a probe, not a classifier. It tests whether simple score contours show coherent pressure-level relationships, within-level contour grouping, or clean score separability.",
        "",
        "## Strongest cross-level same-score contour correlations",
        "",
    ]
    for row in best_cross:
        report_lines.append(
            f"- score {row['score_contour']:.0f}, levels {row['level_pair']}: "
            f"r={row['correlation']:.2f}, mean separation={row['mean_abs_latitude_separation']:.1f} deg"
        )
    report_lines.extend(["", "## Within-level contour spacing", ""])
    for row in compression_rows:
        report_lines.append(
            f"- {row['pressure_level_hpa']:.0f} hPa: tightest band near score "
            f"{row['tightest_score_band_center']:.1f} with mean spacing "
            f"{row['tightest_mean_spacing_deg']:.2f} deg; loosest near "
            f"{row['loosest_score_band_center']:.1f} with {row['loosest_mean_spacing_deg']:.2f} deg."
        )
    report_lines.extend(["", "## Simple score split test", ""])
    for row in otsu_rows:
        report_lines.append(
            f"- {row['pressure_level_hpa']:.0f} hPa: Otsu threshold {row['threshold']:.1f}, "
            f"between-class separation ratio {row['separation_ratio']:.2f}, "
            f"cold/low-score fraction {row['cold_fraction']:.2f}."
        )
    report_lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "A high same-score contour correlation means similarly shaped contour paths across pressure levels, but it does not by itself prove a discrete hot/cold object. Tight contour spacing means rapid score transition; loose spacing means diffuse transition. The Otsu threshold is only a one-dimensional split test on score distribution.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir.relative_to(Path.cwd()).as_posix()}")


if __name__ == "__main__":
    main()
