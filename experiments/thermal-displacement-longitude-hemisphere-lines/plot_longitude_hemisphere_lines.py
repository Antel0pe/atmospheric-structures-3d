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
PLOTS_DIR = EXPERIMENT_DIR / "plots"
LEVELS_HPA = [250, 500, 850, 1000]
LONGITUDE_SAMPLE_COUNT = 10
SCORE_SMOOTH_SIGMA_CELLS = 1.0


def display_path(path: Path) -> str:
    return path.as_posix()


def normalize_longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    return f"{abs(normalized):.0f}°{hemisphere}"


def select_even_longitude_indices(longitudes: np.ndarray, count: int) -> np.ndarray:
    if count > longitudes.size:
        raise ValueError("Requested more longitude samples than available columns.")
    indices = np.floor(np.linspace(0, longitudes.size, count, endpoint=False)).astype(int)
    return np.unique(indices)


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


def plot_level(
    *,
    level_hpa: int,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
) -> dict:
    colors = plt.get_cmap("tab10").colors
    north_rows, north_x = pole_to_equator_axis(latitudes, "north")
    south_rows, south_x = pole_to_equator_axis(latitudes, "south")

    fig, ax = plt.subplots(figsize=(13.5, 7))
    fig.subplots_adjust(right=0.84)
    for sample_number, lon_index in enumerate(longitude_indices):
        color = colors[sample_number % len(colors)]
        lon = float(longitudes[lon_index])
        ax.plot(
            north_x,
            score_points[north_rows, lon_index],
            color=color,
            linewidth=1.7,
            alpha=0.95,
        )
        ax.plot(
            south_x,
            score_points[south_rows, lon_index],
            color=color,
            linewidth=1.7,
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
        f"{level_hpa} hPa Thermal Displacement profiles by longitude\n"
        "solid = Northern Hemisphere, dashed = Southern Hemisphere"
    )
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            linewidth=2.4,
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
        "pressure_level_hpa": level_hpa,
        "plot": display_path(output_path),
        "score_min": float(np.nanmin(score_points)),
        "score_max": float(np.nanmax(score_points)),
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
    longitude_indices = select_even_longitude_indices(longitudes, LONGITUDE_SAMPLE_COUNT)

    level_summaries = []
    for level_hpa in LEVELS_HPA:
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=float(level_hpa))
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        climatology_level = (
            climatology.sel(pressure_level=float(level_hpa))
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
        output_path = PLOTS_DIR / f"thermal_displacement_longitude_hemisphere_lines_{level_hpa:04d}hpa.png"
        level_summaries.append(
            plot_level(
                level_hpa=level_hpa,
                score_points=result.score_points,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=output_path,
            )
        )

    summary = {
        "experiment": "thermal-displacement-longitude-hemisphere-lines",
        "method": "canonical same-longitude same-hemisphere Thermal Displacement",
        "dataset": display_path(DEFAULT_DATASET_PATH),
        "climatology": display_path(DEFAULT_CLIMATOLOGY_PATH),
        "timestamp": DEFAULT_TIMESTAMP,
        "levels_hpa": LEVELS_HPA,
        "score_smooth_sigma_cells": SCORE_SMOOTH_SIGMA_CELLS,
        "longitude_sample_count": int(longitude_indices.size),
        "sampled_longitudes_degrees": [
            float(longitudes[index]) for index in longitude_indices
        ],
        "x_axis": "absolute latitude, ordered from pole to equator separately for each hemisphere",
        "line_encoding": "color is longitude; solid is Northern Hemisphere; dashed is Southern Hemisphere",
        "plots": level_summaries,
    }
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
