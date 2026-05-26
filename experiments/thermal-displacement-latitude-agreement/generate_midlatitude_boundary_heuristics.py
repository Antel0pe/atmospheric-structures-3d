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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

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
    validate_matching_grid,
)
from generate_matched_latitude_displacement_maps import (
    actual_minus_matched_abs_latitude_degrees,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-matched-latitude-displacement-from-actual-latitude/"
    "midlatitude-boundary-heuristics"
)


HEURISTICS = {
    "soft_midlatitude_displacement_gradient": (
        "Soft midlatitude-weighted displacement gradient"
    ),
    "red_blue_copresence": "Nearby red/blue displacement co-presence",
    "equator_red_pole_blue_opposition": "Equator-side red, pole-side blue opposition",
    "score_transition_gradient": "Thermal Displacement score-transition gradient",
    "climatology_baroclinic_contact": "Climatology baroclinic weighting plus red/blue contact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test quick midlatitude boundary heuristics from canonical same-hemisphere "
            "Thermal Displacement."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-smooth-sigma-cells", type=float, default=1.0)
    parser.add_argument("--field-smooth-sigma-cells", type=float, default=4.0)
    parser.add_argument("--neighbor-sigma-cells", type=float, default=10.0)
    parser.add_argument("--opposition-offset-degrees", type=float, default=5.0)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def gradient_magnitude(field: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    dlat = float(np.nanmedian(np.abs(np.diff(latitudes))))
    dlon = float(np.nanmedian(np.abs(np.diff(longitudes))))
    gy, gx = np.gradient(np.asarray(field, dtype=np.float32), dlat, dlon)
    return np.asarray(np.hypot(gx, gy), dtype=np.float32)


def soft_midlatitude_weight(latitudes: np.ndarray) -> np.ndarray:
    abs_lat = np.abs(np.asarray(latitudes, dtype=np.float32))
    gaussian = np.exp(-0.5 * ((abs_lat - 45.0) / 18.0) ** 2)
    equator_damper = 1.0 / (1.0 + np.exp(-(abs_lat - 18.0) / 4.0))
    return np.asarray(gaussian * equator_damper, dtype=np.float32)[:, np.newaxis]


def robust_unit(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.nanpercentile(finite, percentile)), 1.0e-6)
    return np.asarray(np.clip(values / scale, 0.0, 1.0), dtype=np.float32)


def normalized_positive(values: np.ndarray) -> np.ndarray:
    return robust_unit(np.maximum(values, 0.0))


def normalized_negative_magnitude(values: np.ndarray) -> np.ndarray:
    return robust_unit(np.maximum(-values, 0.0))


def shifted_by_abs_latitude_direction(
    field: np.ndarray,
    latitudes: np.ndarray,
    *,
    offset_degrees: float,
    toward_equator: bool,
) -> np.ndarray:
    step = float(np.nanmedian(np.abs(np.diff(latitudes))))
    offset_rows = max(int(round(offset_degrees / step)), 1)
    rows = np.arange(len(latitudes))
    north = latitudes >= 0.0
    source_rows = rows.copy()
    if toward_equator:
        source_rows[north] = rows[north] + offset_rows
        source_rows[~north] = rows[~north] - offset_rows
    else:
        source_rows[north] = rows[north] - offset_rows
        source_rows[~north] = rows[~north] + offset_rows
    source_rows = np.clip(source_rows, 0, len(latitudes) - 1)
    return field[source_rows, :]


def compute_heuristics(
    *,
    displacement: np.ndarray,
    score: np.ndarray,
    climatology_temperature: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    field_smooth_sigma: float,
    neighbor_sigma: float,
    opposition_offset_degrees: float,
) -> dict[str, np.ndarray]:
    midlat = soft_midlatitude_weight(latitudes)
    smooth_displacement = gaussian_filter(
        displacement,
        sigma=(field_smooth_sigma, field_smooth_sigma),
        mode=("nearest", "wrap"),
    )
    smooth_score = gaussian_filter(
        score,
        sigma=(field_smooth_sigma, field_smooth_sigma),
        mode=("nearest", "wrap"),
    )

    positive = normalized_positive(smooth_displacement)
    negative = normalized_negative_magnitude(smooth_displacement)
    positive_near = gaussian_filter(positive, sigma=(neighbor_sigma, neighbor_sigma), mode=("nearest", "wrap"))
    negative_near = gaussian_filter(negative, sigma=(neighbor_sigma, neighbor_sigma), mode=("nearest", "wrap"))
    contact = np.sqrt(np.maximum(positive_near, 0.0) * np.maximum(negative_near, 0.0))

    displacement_gradient = robust_unit(gradient_magnitude(smooth_displacement, latitudes, longitudes))
    score_gradient = robust_unit(gradient_magnitude(smooth_score, latitudes, longitudes))
    score_mid_transition = np.exp(-0.5 * ((smooth_score - 50.0) / 14.0) ** 2)

    equator_side = shifted_by_abs_latitude_direction(
        smooth_displacement,
        latitudes,
        offset_degrees=opposition_offset_degrees,
        toward_equator=True,
    )
    pole_side = shifted_by_abs_latitude_direction(
        smooth_displacement,
        latitudes,
        offset_degrees=opposition_offset_degrees,
        toward_equator=False,
    )
    opposition = np.sqrt(
        normalized_positive(equator_side) * normalized_negative_magnitude(pole_side)
    )

    climatology_smooth = gaussian_filter(
        climatology_temperature,
        sigma=(field_smooth_sigma, field_smooth_sigma),
        mode=("nearest", "wrap"),
    )
    climatology_meridional_gradient = robust_unit(
        np.abs(np.gradient(climatology_smooth, axis=0))
    )

    outputs = {
        "soft_midlatitude_displacement_gradient": midlat * displacement_gradient,
        "red_blue_copresence": midlat * robust_unit(contact),
        "equator_red_pole_blue_opposition": midlat * robust_unit(opposition),
        "score_transition_gradient": midlat * score_mid_transition * score_gradient,
        "climatology_baroclinic_contact": midlat * robust_unit(contact) * climatology_meridional_gradient,
    }
    return {key: np.asarray(value, dtype=np.float32) for key, value in outputs.items()}


def top_fraction_stats(
    values: np.ndarray,
    latitudes: np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "top_threshold": 0.0,
            "top_cell_count": 0.0,
            "top_mean_abs_latitude": 0.0,
            "top_midlatitude_25_65_fraction": 0.0,
            "top_equatorward_0_20_fraction": 0.0,
            "top_poleward_65_90_fraction": 0.0,
        }
    threshold = float(np.nanquantile(finite, 1.0 - top_fraction))
    top_mask = values >= threshold
    abs_lat_grid = np.broadcast_to(np.abs(latitudes)[:, np.newaxis], values.shape)
    top_abs_lat = abs_lat_grid[top_mask]
    return {
        "top_threshold": threshold,
        "top_cell_count": float(np.count_nonzero(top_mask)),
        "top_mean_abs_latitude": float(np.nanmean(top_abs_lat)),
        "top_midlatitude_25_65_fraction": float(np.mean((top_abs_lat >= 25.0) & (top_abs_lat <= 65.0))),
        "top_equatorward_0_20_fraction": float(np.mean(top_abs_lat <= 20.0)),
        "top_poleward_65_90_fraction": float(np.mean(top_abs_lat >= 65.0)),
    }


def plot_contact_sheet(
    *,
    heuristic_name: str,
    values_by_level: dict[float, np.ndarray],
    displacement_by_level: dict[float, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    levels = sorted(values_by_level)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.0), sharex=True, sharey=True)
    last_mesh = None
    for ax, level_hpa in zip(axes.ravel(), levels):
        values = values_by_level[level_hpa]
        displacement = displacement_by_level[level_hpa]
        finite = values[np.isfinite(values)]
        vmax = max(float(np.nanpercentile(finite, 99.5)), 1.0e-6)
        last_mesh = ax.pcolormesh(
            longitudes,
            latitudes,
            values,
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        ax.contour(
            longitudes,
            latitudes,
            displacement,
            levels=[-5.0, 5.0],
            colors=["#4e6fe3", "#d44737"],
            linewidths=0.35,
            alpha=0.60,
        )
        draw_borders(ax, border_segments)
        ax.set_title(f"{level_hpa:g} hPa")
        ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
        ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))

    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude")
    fig.suptitle(heuristic_name, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.92, bottom=0.13, hspace=0.22, wspace=0.05)
    if last_mesh is not None:
        cax = fig.add_axes((0.28, 0.055, 0.44, 0.023))
        colorbar = fig.colorbar(last_mesh, cax=cax, orientation="horizontal")
        colorbar.set_label(
            "Candidate strength, per-panel 99.5th percentile scale. Blue/red contours: +/-5 deg latitude displacement."
        )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_findings(
    *,
    output_dir: Path,
    contact_sheets: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    findings_dir = output_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Midlatitude Boundary Heuristics",
        "",
        "## User Observation",
        "",
        "- The latitude-displacement field should not be filtered by a rigid exact area because weather is dynamic.",
        "- The strongest displacement change is not automatically the desired air-mass boundary; at `1000 hPa`, the largest change can appear near the equator.",
        "- The target is an intuitive way to trace the midlatitude hot-versus-cold air zone, while accepting that a location can be normal for its latitude and still sit on a boundary.",
        "",
        "## Quick Heuristics",
        "",
        "- `soft_midlatitude_displacement_gradient`: smoothed latitude-displacement gradient multiplied by a broad midlatitude weight.",
        "- `red_blue_copresence`: nearby red and blue displacement are both present, using a broad neighborhood and a soft midlatitude weight.",
        "- `equator_red_pole_blue_opposition`: the equatorward side is red while the poleward side is blue about `5 deg` away.",
        "- `score_transition_gradient`: strong gradient near Thermal Displacement score `50`, softly weighted toward midlatitudes.",
        "- `climatology_baroclinic_contact`: red/blue co-presence multiplied by the climatology's meridional temperature gradient.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in contact_sheets.items():
        lines.append(f"- `{key}`: `{display_path(path)}`")
    lines.extend(
        [
            "",
            "## First-Pass Read",
            "",
            "- These are screening plots, not a front classification.",
            "- A useful heuristic should keep most top-scoring cells in the broad `25-65 deg` latitude belt without collapsing into a fixed latitude stripe.",
            "- Compare `top_midlatitude_25_65_fraction` and `top_equatorward_0_20_fraction` in `heuristic_summary.csv` before trusting a visually strong map.",
            "",
            "## Run Settings",
            "",
            f"- `field_smooth_sigma_cells`: `{args.field_smooth_sigma_cells:g}`",
            f"- `neighbor_sigma_cells`: `{args.neighbor_sigma_cells:g}`",
            f"- `opposition_offset_degrees`: `{args.opposition_offset_degrees:g}`",
        ]
    )
    (findings_dir / "midlatitude-boundary-heuristics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    border_path = resolve_path(args.border_geojson)
    output_dir = args.output_dir.expanduser().resolve()
    sheets_dir = output_dir / "contact-sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    values_by_heuristic: dict[str, dict[float, np.ndarray]] = {key: {} for key in HEURISTICS}
    displacement_by_level: dict[float, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

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

        for level_hpa in selected_levels:
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
            displacement_by_level[level_hpa] = displacement
            heuristics = compute_heuristics(
                displacement=displacement,
                score=result.score_points,
                climatology_temperature=climatology_level,
                latitudes=latitudes,
                longitudes=longitudes,
                field_smooth_sigma=args.field_smooth_sigma_cells,
                neighbor_sigma=args.neighbor_sigma_cells,
                opposition_offset_degrees=args.opposition_offset_degrees,
            )
            for heuristic_key, values in heuristics.items():
                values_by_heuristic[heuristic_key][level_hpa] = values
                stats = top_fraction_stats(values, latitudes)
                summary_rows.append(
                    {
                        "heuristic": heuristic_key,
                        "heuristic_name": HEURISTICS[heuristic_key],
                        "pressure_level_hpa": float(level_hpa),
                        **stats,
                    }
                )

    contact_sheets: dict[str, Path] = {}
    for heuristic_key, values_by_level in values_by_heuristic.items():
        output_path = sheets_dir / f"{heuristic_key}.png"
        plot_contact_sheet(
            heuristic_name=HEURISTICS[heuristic_key],
            values_by_level=values_by_level,
            displacement_by_level=displacement_by_level,
            latitudes=latitudes,
            longitudes=longitudes,
            border_segments=border_segments,
            output_path=output_path,
            dpi=args.dpi,
        )
        contact_sheets[heuristic_key] = output_path

    csv_path = output_dir / "heuristic_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "process": "five quick midlatitude boundary heuristics from canonical same-hemisphere Thermal Displacement",
        "dataset": display_path(dataset_path),
        "climatology": display_path(climatology_path),
        "timestamp": args.timestamp,
        "pressure_levels_hpa": sorted({row["pressure_level_hpa"] for row in summary_rows}),
        "heuristics": HEURISTICS,
        "settings": {
            "score_smooth_sigma_cells": float(args.score_smooth_sigma_cells),
            "field_smooth_sigma_cells": float(args.field_smooth_sigma_cells),
            "neighbor_sigma_cells": float(args.neighbor_sigma_cells),
            "opposition_offset_degrees": float(args.opposition_offset_degrees),
        },
        "outputs": {
            "heuristic_summary_csv": display_path(csv_path),
            "contact_sheets": {
                key: display_path(path) for key, path in contact_sheets.items()
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_findings(
        output_dir=output_dir,
        contact_sheets=contact_sheets,
        args=args,
    )
    print(f"Wrote {display_path(output_dir)}")


if __name__ == "__main__":
    main()
