from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from make_plots import (
    CLIMATOLOGY,
    CLIMATOLOGY_VARIABLE,
    DATASET,
    LEVELS_HPA,
    OUTPUT_DIR,
    RENDERED_ABS_LATITUDE_BAND_DEG,
    SCORE_SIGMA_CELLS,
    TEMPERATURE_VARIABLE,
    TIMESTAMP,
    equivalent_abs_latitude_from_score,
    opposing_sides_transition_ribbon,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic-no-bearing-angle-radius-sweep"
WARM_CORE_MAX_DEG = 35.0
COLD_CORE_MIN_DEG = 55.0
CURRENT_RADIUS_DEG = 12.0
RADIUS_SWEEP_DEG = np.arange(0.0, 31.0, 1.0, dtype=np.float32)
DEGREES_TO_APPROX_KM = 111.2


def display_path(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR.parents[1]).as_posix()


def elbow_radius(
    radii_deg: np.ndarray,
    gray_fraction: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(radii_deg, dtype=np.float64)
    y = np.asarray(gray_fraction, dtype=np.float64)
    x_norm = (x - x[0]) / max(float(x[-1] - x[0]), 1.0e-12)
    y_range = float(np.nanmax(y) - np.nanmin(y))
    if y_range <= 1.0e-12:
        return {
            "radius_degrees": float(x[0]),
            "radius_approx_km": float(x[0] * DEGREES_TO_APPROX_KM),
            "normalized_distance_from_endpoint_line": 0.0,
        }
    y_norm = (y - y[-1]) / y_range
    endpoint_line = 1.0 - x_norm
    distance = endpoint_line - y_norm
    index = int(np.nanargmax(distance))
    return {
        "radius_degrees": float(x[index]),
        "radius_approx_km": float(x[index] * DEGREES_TO_APPROX_KM),
        "normalized_distance_from_endpoint_line": float(distance[index]),
    }


def level_sweep(
    equivalent_abs_latitude: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, object]:
    result = opposing_sides_transition_ribbon(
        equivalent_abs_latitude,
        latitudes,
        longitudes,
        warm_core_max_deg=WARM_CORE_MAX_DEG,
        cold_core_min_deg=COLD_CORE_MIN_DEG,
        search_radius_deg=CURRENT_RADIUS_DEG,
        minimum_opposition_angle_deg=None,
    )
    rendered_rows = (
        (np.abs(latitudes) >= RENDERED_ABS_LATITUDE_BAND_DEG[0])
        & (np.abs(latitudes) <= RENDERED_ABS_LATITUDE_BAND_DEG[1])
    )
    rendered = np.broadcast_to(rendered_rows[:, None], result["ribbon"].shape)
    intermediate = result["intermediate"] & rendered
    required_radius = np.maximum(
        result["nearest_warm_distance_deg"],
        result["nearest_cold_distance_deg"],
    )
    intermediate_count = max(int(np.count_nonzero(intermediate)), 1)
    rendered_count = max(int(np.count_nonzero(rendered)), 1)
    rows: list[dict[str, float]] = []
    for radius_deg in RADIUS_SWEEP_DEG:
        unresolved = intermediate & (required_radius > radius_deg)
        gray_count = int(np.count_nonzero(unresolved))
        rows.append(
            {
                "radius_degrees": float(radius_deg),
                "radius_approx_km": float(radius_deg * DEGREES_TO_APPROX_KM),
                "remaining_gray_cell_fraction_of_rendered_band": float(
                    gray_count / rendered_count
                ),
                "remaining_gray_cell_fraction_of_intermediate_air": float(
                    gray_count / intermediate_count
                ),
                "yellow_cell_fraction_of_rendered_band": float(
                    (intermediate_count - gray_count) / rendered_count
                ),
            }
        )
    gray_of_intermediate = np.asarray(
        [row["remaining_gray_cell_fraction_of_intermediate_air"] for row in rows],
        dtype=np.float64,
    )
    gray_of_rendered_band = np.asarray(
        [row["remaining_gray_cell_fraction_of_rendered_band"] for row in rows],
        dtype=np.float64,
    )
    return {
        "intermediate_cell_fraction_of_rendered_band": float(
            intermediate_count / rendered_count
        ),
        "radius_sweep": rows,
        "elbow_candidate_from_gray_share_of_intermediate_air": elbow_radius(
            RADIUS_SWEEP_DEG,
            gray_of_intermediate,
        ),
        "elbow_candidate_from_gray_share_of_rendered_band": elbow_radius(
            RADIUS_SWEEP_DEG,
            gray_of_rendered_band,
        ),
    }


def plot_sweep(
    *,
    levels: list[dict[str, object]],
    output_path: Path,
) -> None:
    colors = {
        250.0: "#e31a1c",
        500.0: "#33a02c",
        850.0: "#1f78b4",
        1000.0: "#6a3d9a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.83, wspace=0.22)
    for row in levels:
        level_hpa = float(row["pressure_level_hpa"])
        sweep = row["radius_sweep"]
        radii = [point["radius_degrees"] for point in sweep]
        gray_band = [
            100.0 * point["remaining_gray_cell_fraction_of_rendered_band"]
            for point in sweep
        ]
        gray_intermediate = [
            100.0 * point["remaining_gray_cell_fraction_of_intermediate_air"]
            for point in sweep
        ]
        elbow = row["elbow_candidate_from_gray_share_of_intermediate_air"]
        elbow_radius_deg = elbow["radius_degrees"]
        elbow_index = radii.index(elbow_radius_deg)
        color = colors[level_hpa]
        label = f"{level_hpa:g} hPa"
        axes[0].plot(radii, gray_band, marker="o", markersize=3, color=color, label=label)
        axes[1].plot(
            radii,
            gray_intermediate,
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        axes[1].scatter(
            [elbow_radius_deg],
            [gray_intermediate[elbow_index]],
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            s=55,
            zorder=5,
        )
    for ax in axes:
        ax.axvline(
            CURRENT_RADIUS_DEG,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label="Current 12 deg" if ax is axes[0] else None,
        )
        ax.set_xlim(float(RADIUS_SWEEP_DEG[0]), float(RADIUS_SWEEP_DEG[-1]))
        ax.set_xlabel("Nearby-core search radius (great-circle degrees)")
        ax.set_ylabel("Remaining unresolved gray (%)")
        ax.grid(color="#5d6470", linewidth=0.4, alpha=0.35, linestyle="--")
        ax.legend(framealpha=0.95)
    axes[0].set_title("Gray share of the rendered 20-70 deg latitude bands")
    axes[1].set_title("Gray share of intermediate-identity air")
    fig.suptitle("No-bearing-angle gray sensitivity to nearby-core search radius", fontsize=14)
    fig.text(
        0.5,
        0.04,
        (
            "Black dashed line: current 12 deg radius. Outlined dots: geometric elbow "
            "candidates from curve shape, not meteorologically proven optima."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    levels: list[dict[str, object]] = []
    with xr.open_dataset(DATASET) as dataset, xr.open_dataset(CLIMATOLOGY) as climatology:
        temperature = dataset[TEMPERATURE_VARIABLE]
        reference = climatology[CLIMATOLOGY_VARIABLE]
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        for level_hpa in LEVELS_HPA:
            raw = np.asarray(
                temperature.sel(valid_time=TIMESTAMP, pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            matched_climatology = np.asarray(
                reference.sel(pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            displacement = compute_thermal_displacement_level(
                raw,
                matched_climatology,
                latitudes,
                score_smooth_sigma_cells=SCORE_SIGMA_CELLS,
                same_hemisphere=True,
            )
            equivalent_abs_latitude = equivalent_abs_latitude_from_score(
                displacement.score_points,
                latitudes,
            )
            levels.append(
                {
                    "pressure_level_hpa": level_hpa,
                    **level_sweep(
                        equivalent_abs_latitude,
                        latitudes,
                        longitudes,
                    ),
                }
            )

    plot_path = DIAGNOSTIC_DIR / "remaining_gray_by_search_radius.png"
    plot_sweep(levels=levels, output_path=plot_path)
    summary = {
        "diagnostic": "global-no-bearing-angle-radius-sweep",
        "date_run": "2026-05-31",
        "timestamp": "2021-11-08T12:00 UTC",
        "levels_hpa": list(LEVELS_HPA),
        "rendered_absolute_latitude_band_degrees": list(RENDERED_ABS_LATITUDE_BAND_DEG),
        "warm_core_max_abs_equivalent_latitude_degrees": WARM_CORE_MAX_DEG,
        "cold_core_min_abs_equivalent_latitude_degrees": COLD_CORE_MIN_DEG,
        "bearing_angle_gate_applied": False,
        "current_search_radius_degrees": CURRENT_RADIUS_DEG,
        "radius_sweep_degrees": [float(value) for value in RADIUS_SWEEP_DEG],
        "elbow_note": (
            "Elbow candidates use maximum normalized distance from the straight line "
            "joining the first and last sweep points. They identify curve-shape bends, "
            "not meteorologically proven optimal radii."
        ),
        "plot": display_path(plot_path),
        "levels": levels,
    }
    (DIAGNOSTIC_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
