from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.thermal_displacement import (
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    compute_thermal_displacement_level,
)


EXPERIMENT_DIR = Path("tmp/thermal-displacement-longitude-hemisphere-lines")
PLOTS_DIR = EXPERIMENT_DIR / "plots-random-sets-850hpa"
SUMMARY_PATH = EXPERIMENT_DIR / "random_850hpa_summary.json"
LEVEL_HPA = 850
LONGITUDES_PER_PLOT = 5
PLOT_COUNT = 3
RANDOM_SEED = 20260528
SCORE_SMOOTH_SIGMA_CELLS = 1.0


def display_path(path: Path) -> str:
    return path.as_posix()


def normalize_longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    value = f"{abs(normalized):.2f}".rstrip("0").rstrip(".")
    return f"{value}°{hemisphere}"


def choose_nonoverlapping_longitude_sets(longitudes: np.ndarray) -> list[np.ndarray]:
    needed = PLOT_COUNT * LONGITUDES_PER_PLOT
    if needed > longitudes.size:
        raise ValueError("Not enough longitudes for non-overlapping plot sets.")

    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(longitudes.size, size=needed, replace=False)
    groups = []
    for plot_index in range(PLOT_COUNT):
        start = plot_index * LONGITUDES_PER_PLOT
        stop = start + LONGITUDES_PER_PLOT
        groups.append(np.sort(chosen[start:stop]))
    return groups


def pole_to_equator_axis(latitudes: np.ndarray, hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    if hemisphere == "north":
        mask = latitudes >= 0.0
    elif hemisphere == "south":
        mask = latitudes < 0.0
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")

    row_indices = np.flatnonzero(mask)
    x_abs_lat = np.abs(latitudes[row_indices])
    order = np.argsort(-x_abs_lat, kind="stable")
    return row_indices[order], x_abs_lat[order]


def plot_group(
    *,
    group_number: int,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
) -> dict:
    colors = plt.get_cmap("tab10").colors[:LONGITUDES_PER_PLOT]
    north_rows, north_x = pole_to_equator_axis(latitudes, "north")
    south_rows, south_x = pole_to_equator_axis(latitudes, "south")

    fig, ax = plt.subplots(figsize=(12.5, 7))
    fig.subplots_adjust(right=0.82)
    for sample_number, lon_index in enumerate(longitude_indices):
        color = colors[sample_number]
        ax.plot(
            north_x,
            score_points[north_rows, lon_index],
            color=color,
            linewidth=2.0,
            alpha=0.95,
        )
        ax.plot(
            south_x,
            score_points[south_rows, lon_index],
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
        )

    max_abs_lat = float(np.nanmax(np.abs(latitudes)))
    ax.set_xlim(max_abs_lat, 0.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks([90, 75, 60, 45, 30, 15, 0])
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Thermal Displacement score")
    ax.set_title(
        f"850 hPa Thermal Displacement profiles, random longitude set {group_number}\n"
        "solid = Northern Hemisphere, dashed = Southern Hemisphere"
    )
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=2.6,
            label=normalize_longitude_label(float(longitudes[lon_index])),
        )
        for i, lon_index in enumerate(longitude_indices)
    ]
    hemisphere_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.0, label="Northern"),
        Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="--", label="Southern"),
    ]
    longitude_legend = ax.legend(
        handles=longitude_handles,
        title="Longitude",
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=True,
    )
    ax.add_artist(longitude_legend)
    ax.legend(
        handles=hemisphere_handles,
        title="Hemisphere",
        loc="lower left",
        frameon=True,
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "group_number": group_number,
        "plot": display_path(output_path),
        "sampled_longitudes_degrees": [
            float(longitudes[index]) for index in longitude_indices
        ],
        "sampled_longitude_labels": [
            normalize_longitude_label(float(longitudes[index]))
            for index in longitude_indices
        ],
    }


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(DEFAULT_DATASET_PATH)
    climatology_ds = xr.open_dataset(DEFAULT_CLIMATOLOGY_PATH)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = np.datetime64(DEFAULT_TIMESTAMP)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    longitude_groups = choose_nonoverlapping_longitude_sets(longitudes)

    raw_level = (
        temperature.sel(valid_time=selected_time, pressure_level=float(LEVEL_HPA))
        .load()
        .to_numpy()
        .astype(np.float32)
    )
    climatology_level = (
        climatology.sel(pressure_level=float(LEVEL_HPA))
        .load()
        .to_numpy()
        .astype(np.float32)
    )
    result = compute_thermal_displacement_level(
        raw_level,
        climatology_level,
        latitudes,
        score_smooth_sigma_cells=SCORE_SMOOTH_SIGMA_CELLS,
        same_hemisphere=True,
    )

    plot_summaries = []
    for group_number, longitude_indices in enumerate(longitude_groups, start=1):
        output_path = PLOTS_DIR / f"thermal_displacement_850hpa_random_longitude_set_{group_number:02d}.png"
        plot_summaries.append(
            plot_group(
                group_number=group_number,
                score_points=result.score_points,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=output_path,
            )
        )

    summary = {
        "experiment": "thermal-displacement-longitude-hemisphere-lines",
        "iteration": "random non-overlapping 850 hPa longitude sets",
        "method": "canonical same-longitude same-hemisphere Thermal Displacement",
        "dataset": display_path(DEFAULT_DATASET_PATH),
        "climatology": display_path(DEFAULT_CLIMATOLOGY_PATH),
        "timestamp": DEFAULT_TIMESTAMP,
        "pressure_level_hpa": LEVEL_HPA,
        "score_smooth_sigma_cells": SCORE_SMOOTH_SIGMA_CELLS,
        "random_seed": RANDOM_SEED,
        "plot_count": PLOT_COUNT,
        "longitudes_per_plot": LONGITUDES_PER_PLOT,
        "line_count_per_plot": LONGITUDES_PER_PLOT * 2,
        "x_axis": "absolute latitude, ordered from pole to equator separately for each hemisphere",
        "line_encoding": "color is longitude; solid is Northern Hemisphere; dashed is Southern Hemisphere",
        "plots": plot_summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
