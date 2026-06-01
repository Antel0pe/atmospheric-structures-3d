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
    RENDERED_ABS_LATITUDE_BAND_DEG,
    SCORE_SIGMA_CELLS,
    TEMPERATURE_VARIABLE,
    TIMESTAMP,
    draw_borders,
    equivalent_abs_latitude_from_score,
    load_border_segments,
    opposing_sides_transition_ribbon,
)
from scripts.thermal_displacement import compute_thermal_displacement_level


DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic-no-bearing-angle-gray-reasons"
WARM_CORE_MAX_DEG = 35.0
COLD_CORE_MIN_DEG = 55.0
SEARCH_RADIUS_DEG = 12.0
CAUSE_KEYS = (
    "warm_core_too_far",
    "cold_core_too_far",
    "both_cores_too_far",
)
COLORS = {
    "warm_core": "#c9413a",
    "nearby_core_transition": "#ffd92f",
    "warm_core_too_far": "#e78ac3",
    "cold_core_too_far": "#66c2a5",
    "both_cores_too_far": "#8da0cb",
    "cold_core": "#2367b3",
}
LABELS = {
    "warm_core": "Warm core",
    "nearby_core_transition": "Yellow: both nearest cores within 12 deg",
    "warm_core_too_far": "Gray: warm core farther than 12 deg",
    "cold_core_too_far": "Gray: cold core farther than 12 deg",
    "both_cores_too_far": "Gray: both cores farther than 12 deg",
    "cold_core": "Cold core",
}


def display_path(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR.parents[1]).as_posix()


def classify(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intermediate = result["intermediate"]
    warm_too_far = result["nearest_warm_distance_deg"] > SEARCH_RADIUS_DEG
    cold_too_far = result["nearest_cold_distance_deg"] > SEARCH_RADIUS_DEG
    return {
        "warm_core": result["warm_core"],
        "nearby_core_transition": result["ribbon"],
        "warm_core_too_far": intermediate & warm_too_far & (~cold_too_far),
        "cold_core_too_far": intermediate & (~warm_too_far) & cold_too_far,
        "both_cores_too_far": intermediate & warm_too_far & cold_too_far,
        "cold_core": result["cold_core"],
    }


def finite_percentiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {"median": 0.0, "p90": 0.0}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
    }


def stats(
    result: dict[str, np.ndarray],
    classes: dict[str, np.ndarray],
) -> dict[str, object]:
    rendered_rows = (
        (np.abs(LATITUDES) >= RENDERED_ABS_LATITUDE_BAND_DEG[0])
        & (np.abs(LATITUDES) <= RENDERED_ABS_LATITUDE_BAND_DEG[1])
    )
    rendered = np.broadcast_to(rendered_rows[:, None], result["ribbon"].shape)
    intermediate = result["intermediate"] & rendered
    unresolved = intermediate & (~result["ribbon"])

    def fraction(mask: np.ndarray, denominator: np.ndarray) -> float:
        count = int(np.count_nonzero(denominator))
        return float(np.count_nonzero(mask & denominator) / count) if count else 0.0

    return {
        "rendered_band_cell_count": int(np.count_nonzero(rendered)),
        "intermediate_cell_fraction_of_rendered_band": fraction(intermediate, rendered),
        "yellow_cell_fraction_of_rendered_band": fraction(result["ribbon"], rendered),
        "unresolved_gray_cell_fraction_of_rendered_band": fraction(unresolved, rendered),
        "unresolved_gray_cell_fraction_of_intermediate": fraction(unresolved, intermediate),
        "exclusive_gray_failure_fraction_of_unresolved_gray": {
            key: fraction(classes[key], unresolved) for key in CAUSE_KEYS
        },
        "unresolved_gray_metrics": {
            "nearest_warm_core_distance_degrees": finite_percentiles(
                result["nearest_warm_distance_deg"][unresolved]
            ),
            "nearest_cold_core_distance_degrees": finite_percentiles(
                result["nearest_cold_distance_deg"][unresolved]
            ),
        },
    }


def plot_map(
    *,
    level_hpa: float,
    classes: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    keys = (
        "cold_core",
        "both_cores_too_far",
        "cold_core_too_far",
        "warm_core_too_far",
        "nearby_core_transition",
        "warm_core",
    )
    values = np.full(classes["warm_core"].shape, np.nan, dtype=np.float32)
    for code, key in enumerate(keys):
        values[classes[key]] = code
    cmap = mcolors.ListedColormap([COLORS[key] for key in keys])
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(keys) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(15.5, 7.9))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.145, top=0.89)
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
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-89.0, 89.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_yticks(np.arange(-60.0, 61.0, 30.0))
    ax.grid(color="#5d6470", linewidth=0.4, alpha=0.25, linestyle="--")
    ax.set_title(f"{level_hpa:g} hPa no-bearing-angle gray reasons")
    ax.legend(
        handles=[Patch(facecolor=COLORS[key], label=LABELS[key]) for key in keys[::-1]],
        loc="lower left",
        framealpha=0.95,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.028,
        "No bearing-angle gate: unresolved gray means nearest warm core, nearest cold core, or both are farther than 12 deg.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    global LATITUDES
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with xr.open_dataset(DATASET) as dataset, xr.open_dataset(CLIMATOLOGY) as climatology:
        temperature = dataset[TEMPERATURE_VARIABLE]
        reference = climatology[CLIMATOLOGY_VARIABLE]
        LATITUDES = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(BORDER_GEOJSON, longitudes)
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
                LATITUDES,
                score_smooth_sigma_cells=SCORE_SIGMA_CELLS,
                same_hemisphere=True,
            )
            equivalent_abs_latitude = equivalent_abs_latitude_from_score(
                displacement.score_points,
                LATITUDES,
            )
            result = opposing_sides_transition_ribbon(
                equivalent_abs_latitude,
                LATITUDES,
                longitudes,
                warm_core_max_deg=WARM_CORE_MAX_DEG,
                cold_core_min_deg=COLD_CORE_MIN_DEG,
                search_radius_deg=SEARCH_RADIUS_DEG,
                minimum_opposition_angle_deg=None,
            )
            classes = classify(result)
            output_path = DIAGNOSTIC_DIR / f"no_bearing_angle_gray_reasons_{int(level_hpa):04d}hpa.png"
            plot_map(
                level_hpa=level_hpa,
                classes=classes,
                latitudes=LATITUDES,
                longitudes=longitudes,
                border_segments=border_segments,
                output_path=output_path,
            )
            rows.append(
                {
                    "pressure_level_hpa": level_hpa,
                    "plot": display_path(output_path),
                    "stats": stats(result, classes),
                }
            )

    summary = {
        "diagnostic": "global-no-bearing-angle-gray-reasons",
        "date_run": "2026-05-31",
        "timestamp": "2021-11-08T12:00 UTC",
        "levels_hpa": list(LEVELS_HPA),
        "rendered_absolute_latitude_band_degrees": list(RENDERED_ABS_LATITUDE_BAND_DEG),
        "warm_core_max_abs_equivalent_latitude_degrees": WARM_CORE_MAX_DEG,
        "cold_core_min_abs_equivalent_latitude_degrees": COLD_CORE_MIN_DEG,
        "search_radius_degrees": SEARCH_RADIUS_DEG,
        "bearing_angle_gate_applied": False,
        "gray_reason_rule": (
            "An unresolved gray intermediate cell has nearest warm core farther than 12 degrees, "
            "nearest cold core farther than 12 degrees, or both. Distances are spherical great-circle "
            "angular distances. No bearing-angle condition is applied."
        ),
        "levels": rows,
    }
    (DIAGNOSTIC_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
