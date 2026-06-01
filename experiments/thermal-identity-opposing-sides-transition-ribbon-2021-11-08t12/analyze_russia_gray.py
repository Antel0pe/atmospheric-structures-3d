from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

from make_plots import (
    BORDER_GEOJSON,
    CLIMATOLOGY,
    CLIMATOLOGY_VARIABLE,
    DATASET,
    LEVELS_HPA,
    OUTPUT_DIR,
    SCORE_SIGMA_CELLS,
    TEMPERATURE_VARIABLE,
    TIMESTAMP,
    draw_borders,
    equivalent_abs_latitude_from_score,
    load_border_segments,
    opposing_sides_transition_ribbon,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic-russia-gray-reasons"
RUSSIA_SECTOR = {
    "longitude_degrees_east": [30.0, 180.0],
    "latitude_degrees_north": [45.0, 70.0],
}
WARM_CORE_MAX_DEG = 35.0
COLD_CORE_MIN_DEG = 55.0
PRIMARY_RADIUS_DEG = 12.0
PRIMARY_ANGLE_DEG = 120.0
RADIUS_SWEEP_DEG = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
ANGLE_SWEEP_DEG = (90.0, 105.0, 120.0, 135.0, 150.0)

REASON_COLORS = {
    "validated_transition": "#ffd92f",
    "warm_core": "#c9413a",
    "cold_core": "#2367b3",
    "warm_core_too_far": "#e78ac3",
    "cold_core_too_far": "#66c2a5",
    "both_cores_too_far": "#8da0cb",
    "nearby_but_not_opposed": "#a6d854",
}
REASON_LABELS = {
    "validated_transition": "Validated yellow transition",
    "warm_core": "Warm core",
    "cold_core": "Cold core",
    "warm_core_too_far": "Gray: warm core farther than radius",
    "cold_core_too_far": "Gray: cold core farther than radius",
    "both_cores_too_far": "Gray: both cores farther than radius",
    "nearby_but_not_opposed": "Gray: nearby cores, angle below threshold",
}
REASON_CODES = {
    "cold_core": 0,
    "both_cores_too_far": 1,
    "cold_core_too_far": 2,
    "nearby_but_not_opposed": 3,
    "warm_core_too_far": 4,
    "validated_transition": 5,
    "warm_core": 6,
}


def display_path(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR.parents[1]).as_posix()


def sector_mask(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    return (
        (longitude_grid >= RUSSIA_SECTOR["longitude_degrees_east"][0])
        & (longitude_grid <= RUSSIA_SECTOR["longitude_degrees_east"][1])
        & (latitude_grid >= RUSSIA_SECTOR["latitude_degrees_north"][0])
        & (latitude_grid <= RUSSIA_SECTOR["latitude_degrees_north"][1])
    )


def classify_reasons(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intermediate = result["intermediate"]
    warm_too_far = result["nearest_warm_distance_deg"] > PRIMARY_RADIUS_DEG
    cold_too_far = result["nearest_cold_distance_deg"] > PRIMARY_RADIUS_DEG
    nearby_but_not_opposed = (
        intermediate
        & (~warm_too_far)
        & (~cold_too_far)
        & (result["nearest_core_opposition_angle_deg"] < PRIMARY_ANGLE_DEG)
    )
    return {
        "validated_transition": result["ribbon"],
        "warm_core": result["warm_core"],
        "cold_core": result["cold_core"],
        "warm_core_too_far": intermediate & warm_too_far & (~cold_too_far),
        "cold_core_too_far": intermediate & (~warm_too_far) & cold_too_far,
        "both_cores_too_far": intermediate & warm_too_far & cold_too_far,
        "nearby_but_not_opposed": nearby_but_not_opposed,
    }


def mask_fraction(mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = int(np.count_nonzero(denominator_mask))
    if denominator == 0:
        return 0.0
    return float(np.count_nonzero(mask & denominator_mask) / denominator)


def finite_percentiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {"median": 0.0, "p90": 0.0}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
    }


def reason_stats(
    result: dict[str, np.ndarray],
    reasons: dict[str, np.ndarray],
    region: np.ndarray,
) -> dict[str, object]:
    intermediate = result["intermediate"] & region
    unresolved = intermediate & (~result["ribbon"])
    return {
        "sector_cell_count": int(np.count_nonzero(region)),
        "core_and_transition_cell_fraction_of_sector": {
            key: mask_fraction(mask, region)
            for key, mask in reasons.items()
            if key in ("warm_core", "cold_core", "validated_transition")
        },
        "intermediate_cell_fraction_of_sector": mask_fraction(intermediate, region),
        "unresolved_gray_cell_fraction_of_sector": mask_fraction(unresolved, region),
        "unresolved_gray_cell_fraction_of_intermediate": mask_fraction(unresolved, intermediate),
        "exclusive_gray_failure_fraction_of_unresolved_gray": {
            key: mask_fraction(mask, unresolved)
            for key, mask in reasons.items()
            if key
            in (
                "warm_core_too_far",
                "cold_core_too_far",
                "both_cores_too_far",
                "nearby_but_not_opposed",
            )
        },
        "unresolved_gray_metrics": {
            "nearest_warm_core_distance_degrees": finite_percentiles(
                result["nearest_warm_distance_deg"][unresolved]
            ),
            "nearest_cold_core_distance_degrees": finite_percentiles(
                result["nearest_cold_distance_deg"][unresolved]
            ),
            "warm_cold_bearing_opposition_angle_degrees": finite_percentiles(
                result["nearest_core_opposition_angle_deg"][unresolved]
            ),
        },
    }


def plot_reason_map(
    *,
    level_hpa: float,
    reasons: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    values = np.full(reasons["warm_core"].shape, np.nan, dtype=np.float32)
    ordered_keys = (
        "cold_core",
        "both_cores_too_far",
        "cold_core_too_far",
        "nearby_but_not_opposed",
        "warm_core_too_far",
        "validated_transition",
        "warm_core",
    )
    for key in ordered_keys:
        values[reasons[key]] = REASON_CODES[key]

    color_list = [REASON_COLORS[key] for key in ordered_keys]
    cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(color_list) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(14.2, 5.8))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.87)
    ax.pcolormesh(
        longitudes,
        latitudes,
        values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    draw_borders(ax, border_segments)
    ax.set_xlim(*RUSSIA_SECTOR["longitude_degrees_east"])
    ax.set_ylim(*RUSSIA_SECTOR["latitude_degrees_north"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(30.0, 181.0, 30.0))
    ax.set_yticks(np.arange(45.0, 71.0, 5.0))
    ax.grid(color="#5d6470", linewidth=0.4, alpha=0.25, linestyle="--")
    ax.set_title(f"{level_hpa:g} hPa Russia-sector unresolved-gray reasons")
    ax.legend(
        handles=[
            Patch(facecolor=REASON_COLORS[key], label=REASON_LABELS[key])
            for key in (
                "warm_core",
                "validated_transition",
                "warm_core_too_far",
                "cold_core_too_far",
                "both_cores_too_far",
                "nearby_but_not_opposed",
                "cold_core",
            )
        ],
        loc="lower left",
        framealpha=0.95,
        fontsize=8,
        ncol=2,
    )
    fig.text(
        0.5,
        0.04,
        "Russia-sector diagnostic box: 30-180 E, 45-70 N. Gray reasons are exclusive under the primary 12 deg / 120 deg rule.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def unresolved_fraction(
    equivalent_abs_latitude: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    region: np.ndarray,
    *,
    radius_deg: float,
    angle_deg: float,
) -> float:
    result = opposing_sides_transition_ribbon(
        equivalent_abs_latitude,
        latitudes,
        longitudes,
        warm_core_max_deg=WARM_CORE_MAX_DEG,
        cold_core_min_deg=COLD_CORE_MIN_DEG,
        search_radius_deg=radius_deg,
        minimum_opposition_angle_deg=angle_deg,
    )
    intermediate = result["intermediate"] & region
    return mask_fraction(intermediate & (~result["ribbon"]), intermediate)


def plot_sensitivity(
    *,
    rows: dict[float, dict[str, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    colors = {
        250.0: "#e31a1c",
        500.0: "#33a02c",
        850.0: "#1f78b4",
        1000.0: "#6a3d9a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.85, wspace=0.22)
    for level_hpa in LEVELS_HPA:
        radius_rows = rows[level_hpa]["radius_sweep"]
        axes[0].plot(
            [row["radius_degrees"] for row in radius_rows],
            [100.0 * row["unresolved_fraction_of_intermediate"] for row in radius_rows],
            marker="o",
            color=colors[level_hpa],
            label=f"{level_hpa:g} hPa",
        )
        angle_rows = rows[level_hpa]["angle_sweep"]
        axes[1].plot(
            [row["angle_degrees"] for row in angle_rows],
            [100.0 * row["unresolved_fraction_of_intermediate"] for row in angle_rows],
            marker="o",
            color=colors[level_hpa],
            label=f"{level_hpa:g} hPa",
        )
    axes[0].axvline(PRIMARY_RADIUS_DEG, color="#333333", linestyle="--", linewidth=0.9)
    axes[0].set_title("Radius sweep at fixed 120 deg opposition")
    axes[0].set_xlabel("Search radius (great-circle degrees)")
    axes[1].axvline(PRIMARY_ANGLE_DEG, color="#333333", linestyle="--", linewidth=0.9)
    axes[1].set_title("Opposition-angle sweep at fixed 12 deg radius")
    axes[1].set_xlabel("Minimum bearing separation (degrees)")
    for ax in axes:
        ax.set_ylabel("Unresolved gray share of intermediate air (%)")
        ax.grid(color="#5d6470", linewidth=0.4, alpha=0.35, linestyle="--")
        ax.legend(framealpha=0.95)
    fig.suptitle("Russia-sector gray sensitivity by pressure level", fontsize=14)
    fig.text(
        0.5,
        0.035,
        "Diagnostic box: 30-180 E, 45-70 N. Lower values mean more intermediate cells promoted from unresolved gray to yellow.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(DATASET) as dataset, xr.open_dataset(CLIMATOLOGY) as climatology:
        temperature = dataset[TEMPERATURE_VARIABLE]
        reference = climatology[CLIMATOLOGY_VARIABLE]
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(BORDER_GEOJSON, longitudes)
        region = sector_mask(latitudes, longitudes)
        level_rows: list[dict[str, object]] = []
        sensitivity_rows: dict[float, dict[str, list[dict[str, float]]]] = {}

        for level_hpa in LEVELS_HPA:
            raw_level = np.asarray(
                temperature.sel(valid_time=TIMESTAMP, pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            climatology_level = np.asarray(
                reference.sel(pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            displacement = compute_thermal_displacement_level(
                raw_level,
                climatology_level,
                latitudes,
                score_smooth_sigma_cells=SCORE_SIGMA_CELLS,
                same_hemisphere=True,
            )
            equivalent_abs_latitude = equivalent_abs_latitude_from_score(
                displacement.score_points,
                latitudes,
            )
            result = opposing_sides_transition_ribbon(
                equivalent_abs_latitude,
                latitudes,
                longitudes,
                warm_core_max_deg=WARM_CORE_MAX_DEG,
                cold_core_min_deg=COLD_CORE_MIN_DEG,
                search_radius_deg=PRIMARY_RADIUS_DEG,
                minimum_opposition_angle_deg=PRIMARY_ANGLE_DEG,
            )
            reasons = classify_reasons(result)
            output_path = DIAGNOSTIC_DIR / f"gray_failure_reasons_{int(level_hpa):04d}hpa.png"
            plot_reason_map(
                level_hpa=level_hpa,
                reasons=reasons,
                latitudes=latitudes,
                longitudes=longitudes,
                border_segments=border_segments,
                output_path=output_path,
            )
            sensitivity_rows[level_hpa] = {
                "radius_sweep": [
                    {
                        "radius_degrees": radius_deg,
                        "unresolved_fraction_of_intermediate": unresolved_fraction(
                            equivalent_abs_latitude,
                            latitudes,
                            longitudes,
                            region,
                            radius_deg=radius_deg,
                            angle_deg=PRIMARY_ANGLE_DEG,
                        ),
                    }
                    for radius_deg in RADIUS_SWEEP_DEG
                ],
                "angle_sweep": [
                    {
                        "angle_degrees": angle_deg,
                        "unresolved_fraction_of_intermediate": unresolved_fraction(
                            equivalent_abs_latitude,
                            latitudes,
                            longitudes,
                            region,
                            radius_deg=PRIMARY_RADIUS_DEG,
                            angle_deg=angle_deg,
                        ),
                    }
                    for angle_deg in ANGLE_SWEEP_DEG
                ],
            }
            level_rows.append(
                {
                    "pressure_level_hpa": level_hpa,
                    "plot": display_path(output_path),
                    "global_primary_rule_stats": reason_stats(
                        result,
                        reasons,
                        np.ones_like(region, dtype=bool),
                    ),
                    "primary_rule_sector_stats": reason_stats(result, reasons, region),
                    **sensitivity_rows[level_hpa],
                }
            )

    sensitivity_plot = DIAGNOSTIC_DIR / "gray_threshold_sensitivity.png"
    plot_sensitivity(rows=sensitivity_rows, output_path=sensitivity_plot)
    summary = {
        "diagnostic": "russia-sector-gray-failure-reasons",
        "date_run": "2026-05-31",
        "timestamp": "2021-11-08T12:00 UTC",
        "sector": RUSSIA_SECTOR,
        "sector_note": "A simple Eurasia/Russia-sector bounding box, not a country-border mask.",
        "identity_field": "Canonical Thermal Displacement converted to absolute matched climatological latitude.",
        "warm_core_max_abs_equivalent_latitude_degrees": WARM_CORE_MAX_DEG,
        "cold_core_min_abs_equivalent_latitude_degrees": COLD_CORE_MIN_DEG,
        "primary_search_radius_degrees": PRIMARY_RADIUS_DEG,
        "primary_minimum_opposition_angle_degrees": PRIMARY_ANGLE_DEG,
        "gray_reason_rule": (
            "Exclusive unresolved-gray reasons are assigned after the warm-core, cold-core, "
            "and validated-yellow masks: warm core too far, cold core too far, both too far, "
            "or both nearby but bearing separation below the opposition-angle threshold."
        ),
        "threshold_sensitivity_plot": display_path(sensitivity_plot),
        "levels": level_rows,
    }
    (DIAGNOSTIC_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
