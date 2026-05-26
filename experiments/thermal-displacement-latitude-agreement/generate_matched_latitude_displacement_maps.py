from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

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
    parse_requested_levels,
    resolve_path,
    slug_for_level,
    validate_matching_grid,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-matched-latitude-displacement-from-actual-latitude"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate maps of how far the closest Thermal Displacement "
            "climatology latitude is from the cell's actual latitude."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def actual_minus_matched_abs_latitude_degrees(
    matched_latitudes: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    actual_abs = np.abs(np.asarray(latitudes, dtype=np.float32))[:, np.newaxis]
    matched_abs = np.abs(np.asarray(matched_latitudes, dtype=np.float32))
    return np.asarray(actual_abs - matched_abs, dtype=np.float32)


def plot_displacement_map(
    *,
    displacement_degrees: np.ndarray,
    score: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    level_hpa: float,
    output_path: Path,
    dpi: int,
) -> dict[str, float]:
    finite = displacement_degrees[np.isfinite(displacement_degrees)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    color_extent = max(abs(vmin), abs(vmax), 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-color_extent, vcenter=0.0, vmax=color_extent)

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        displacement_degrees,
        cmap="bwr",
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    ax.contour(
        longitudes,
        latitudes,
        score,
        levels=np.arange(10.0, 100.0, 10.0),
        colors="#171717",
        linewidths=0.22,
        alpha=0.32,
    )
    draw_borders(ax, border_segments)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level_hpa:g} hPa closest climatology latitude displacement from actual latitude"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label(
        "Actual |latitude| - matched climatology |latitude| (degrees)"
    )
    ax.text(
        0.01,
        0.012,
        "Red: matched climatology latitude is more equatorward than the cell. "
        "Blue: more poleward. Black lines: Thermal Displacement score.",
        transform=ax.transAxes,
        fontsize=8,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 3},
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return {
        "displacement_min_degrees": vmin,
        "displacement_max_degrees": vmax,
        "map_symmetric_color_extent_degrees": color_extent,
        "displacement_mean_degrees": float(np.nanmean(displacement_degrees)),
        "displacement_median_degrees": float(np.nanmedian(displacement_degrees)),
        "displacement_abs_mean_degrees": float(np.nanmean(np.abs(displacement_degrees))),
        "displacement_abs_median_degrees": float(np.nanmedian(np.abs(displacement_degrees))),
    }


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    map_dir = output_dir / "matched-latitude-displacement-maps"
    findings_dir = output_dir / "findings"
    map_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(dataset_path) as temperature_ds, xr.open_dataset(climatology_path) as climatology_ds:
        temperature = temperature_ds[TEMPERATURE_VARIABLE]
        climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
        validate_matching_grid(temperature, climatology)

        selected_time = choose_timestamp(temperature, args.timestamp)
        level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
        selected_levels = parse_requested_levels(args.pressure_levels, level_values)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(border_path, longitudes)

        rows: list[dict[str, object]] = []
        for level_hpa in selected_levels:
            slug = slug_for_level(level_hpa)
            print(f"Processing {level_hpa:g} hPa")
            raw_level = np.asarray(
                temperature.sel(valid_time=selected_time, pressure_level=level_hpa).load().values,
                dtype=np.float32,
            )
            climatology_level = np.asarray(
                climatology.sel(pressure_level=level_hpa).load().values,
                dtype=np.float32,
            )
            result = compute_thermal_displacement_level(
                raw_level,
                climatology_level,
                latitudes,
                score_smooth_sigma_cells=args.score_smooth_sigma_cells,
                same_hemisphere=True,
            )
            displacement = actual_minus_matched_abs_latitude_degrees(
                result.matched_latitudes_deg,
                latitudes,
            )
            map_path = map_dir / f"matched_latitude_displacement_{slug}.png"
            stats = plot_displacement_map(
                displacement_degrees=displacement,
                score=result.score_points,
                longitudes=longitudes,
                latitudes=latitudes,
                border_segments=border_segments,
                level_hpa=level_hpa,
                output_path=map_path,
                dpi=args.dpi,
            )
            rows.append(
                {
                    "pressure_level_hpa": float(level_hpa),
                    "map_png": display_path(map_path),
                    "selected_white_center": float(result.selected_bucket.center),
                    "selected_bucket_count": int(result.selected_bucket.count),
                    **stats,
                }
            )

    summary = {
        "process": "same-hemisphere Thermal Displacement matched climatology latitude displacement from actual latitude",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": args.timestamp,
        "pressure_levels_hpa": [row["pressure_level_hpa"] for row in rows],
        "matching_mode": "same-longitude, same-hemisphere closest climatology temperature",
        "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
        "displacement_definition": "actual_abs_latitude_degrees - matched_climatology_abs_latitude_degrees",
        "color_scale": "blue-white-red centered on zero per pressure level; red is matched more equatorward than actual latitude, blue is matched more poleward",
        "outputs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    findings = [
        "# Matched Latitude Displacement From Actual Latitude",
        "",
        "## Method",
        "",
        "- Compute canonical same-longitude, same-hemisphere Thermal Displacement from raw temperature.",
        "- Keep the closest climatology latitude returned by that lookup.",
        "- Plot `abs(actual latitude) - abs(matched climatology latitude)` in degrees.",
        "- Red means the matched climatology latitude is closer to the equator than the cell's actual latitude.",
        "- Blue means the matched climatology latitude is closer to the pole than the cell's actual latitude.",
        "- Black contours show the smoothed Thermal Displacement score for orientation.",
        "",
        "## Outputs",
        "",
    ]
    for row in rows:
        findings.append(
            f"- `{row['pressure_level_hpa']:g} hPa`: `{row['map_png']}` "
            f"(mean `{row['displacement_mean_degrees']:.2f} deg`, "
            f"median `{row['displacement_median_degrees']:.2f} deg`, "
            f"mean absolute `{row['displacement_abs_mean_degrees']:.2f} deg`)"
        )
    (findings_dir / "matched-latitude-displacement.md").write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
