from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from generate_maps import (
    CLIMATOLOGY_VARIABLE,
    DEFAULT_BORDER_GEOJSON,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    display_path,
    draw_borders,
    load_border_segments,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    slug_for_level,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-score-smoothed-sigma20-displacement-from-actual-latitude"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate same-hemisphere thermal-displacement maps colored by "
            "displacement from each cell's actual-latitude score."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def plot_displacement_map(
    displacement: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> tuple[float, float, float]:
    finite = displacement[np.isfinite(displacement)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    symmetric_extent = max(abs(vmin), abs(vmax), 1.0)
    norm = mcolors.TwoSlopeNorm(
        vmin=-symmetric_extent,
        vcenter=0.0,
        vmax=symmetric_extent,
    )

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        displacement,
        cmap="bwr",
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa thermal-displacement score minus actual-latitude score"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label(
        "Score-point displacement; blue = more polar-like, red = more equator-like"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return vmin, vmax, symmetric_extent


def plot_displacement_histogram(
    displacement: np.ndarray,
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = displacement[np.isfinite(displacement)]
    max_abs = max(float(np.nanmax(np.abs(finite))), 1.0)
    lower = np.floor(-max_abs) - 0.5
    upper = np.ceil(max_abs) + 0.5
    edges = np.arange(lower, upper + 1.0, 1.0, dtype=np.float32)
    counts, edges = np.histogram(finite, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.where(centers < 0.0, "#4d78b5", "#bd5a4f").astype(object)
    colors[np.abs(centers) <= 0.5] = "#d9d9d9"

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.bar(centers, counts, width=0.92, color=colors, edgecolor="none")
    ax.axvline(0.0, color="#171717", linewidth=1.3)
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel("Smoothed score minus actual-latitude score")
    ax.set_ylabel("Cell count")
    ax.set_title(f"{level_hpa:g} hPa score-displacement histogram")
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return counts, edges


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    map_dir = output_dir / "score-displacement-maps"
    histogram_dir = output_dir / "score-displacement-histograms"
    map_dir.mkdir(exist_ok=True)
    histogram_dir.mkdir(exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(border_path, longitudes)
    actual_score = thermal_displacement_score_points(latitudes[:, np.newaxis], latitudes)

    rows: list[dict[str, object]] = []
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
            latitudes,
            "same-hemisphere",
        )
        matched_score = thermal_displacement_score_points(matched_latitude, latitudes)
        smoothed_score = smooth_wrapped_lon(matched_score, args.smooth_sigma_cells)
        score_displacement = smoothed_score - actual_score

        map_path = map_dir / f"score_displacement_{slug}.png"
        histogram_path = histogram_dir / f"score_displacement_histogram_{slug}.png"
        histogram_csv_path = histogram_dir / f"score_displacement_histogram_{slug}.csv"

        vmin, vmax, symmetric_extent = plot_displacement_map(
            displacement=score_displacement,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            level_hpa=level_hpa,
            output_path=map_path,
            dpi=args.dpi,
        )
        counts, edges = plot_displacement_histogram(
            displacement=score_displacement,
            level_hpa=level_hpa,
            output_path=histogram_path,
            dpi=args.dpi,
        )

        with histogram_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bucket_lower", "bucket_upper", "bucket_center", "cell_count"])
            for index, count in enumerate(counts):
                writer.writerow(
                    [
                        float(edges[index]),
                        float(edges[index + 1]),
                        float(0.5 * (edges[index] + edges[index + 1])),
                        int(count),
                    ]
                )

        rows.append(
            {
                "pressure_level_hpa": float(level_hpa),
                "map_png": display_path(map_path),
                "histogram_png": display_path(histogram_path),
                "histogram_csv": display_path(histogram_csv_path),
                "displacement_min": vmin,
                "displacement_max": vmax,
                "map_symmetric_color_extent": symmetric_extent,
                "displacement_mean": float(np.nanmean(score_displacement)),
                "displacement_median": float(np.nanmedian(score_displacement)),
                "displacement_abs_mean": float(np.nanmean(np.abs(score_displacement))),
                "displacement_abs_median": float(np.nanmedian(np.abs(score_displacement))),
            }
        )

    summary = {
        "process": "same-hemisphere thermal-displacement score colored by displacement from actual-latitude score",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": np.datetime_as_string(selected_time, unit="s"),
        "pressure_levels_hpa": [row["pressure_level_hpa"] for row in rows],
        "matching_mode": "same-hemisphere",
        "same_hemisphere_rule": (
            "source latitudes >= 0 only compare against climatology latitudes >= 0; "
            "source latitudes < 0 only compare against climatology latitudes < 0"
        ),
        "score_definition": "score = (1 - abs(latitude) / max_abs_latitude) * 100",
        "latitude_grid": {
            "min": float(np.nanmin(latitudes)),
            "max": float(np.nanmax(latitudes)),
            "max_abs_latitude": float(np.nanmax(np.abs(latitudes))),
            "step_degrees": float(abs(latitudes[1] - latitudes[0])),
        },
        "score_smoothing": (
            f"Gaussian sigma={args.smooth_sigma_cells:g} native grid cells on "
            "matched thermal-displacement score; longitude wraps, latitude uses nearest edge"
        ),
        "displacement_definition": "score_displacement = smoothed_matched_score - actual_latitude_score",
        "color_scale": (
            "blue-white-red centered on zero per pressure level; red means more "
            "equator-like than actual latitude, blue means more polar-like"
        ),
        "outputs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "score_displacement_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pressure_level_hpa",
                "displacement_min",
                "displacement_max",
                "map_symmetric_color_extent",
                "displacement_mean",
                "displacement_median",
                "displacement_abs_mean",
                "displacement_abs_median",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["pressure_level_hpa"],
                    row["displacement_min"],
                    row["displacement_max"],
                    row["map_symmetric_color_extent"],
                    row["displacement_mean"],
                    row["displacement_median"],
                    row["displacement_abs_mean"],
                    row["displacement_abs_median"],
                ]
            )

    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
