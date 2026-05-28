from __future__ import annotations

import json
from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
OUTPUT_ROOT = EXPERIMENT_DIR / "organized-profile-map-pairs"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
SCORE_SMOOTH_SIGMA_CELLS = 1.0


def display_path(path: Path) -> str:
    return path.as_posix()


def longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    value = f"{abs(normalized):.2f}".rstrip("0").rstrip(".")
    return f"{value}{hemisphere}"


def nearest_longitude_indices(longitudes: np.ndarray, requested_lons: list[float]) -> np.ndarray:
    indices = []
    for requested_lon in requested_lons:
        indices.append(int(np.argmin(np.abs(longitudes - requested_lon))))
    indices_array = np.asarray(indices, dtype=np.int64)
    if np.unique(indices_array).size != indices_array.size:
        raise ValueError("Requested longitude samples collapsed onto duplicate grid columns.")
    return indices_array


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


def compute_score_fields() -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    temperature_ds = xr.open_dataset(DEFAULT_DATASET_PATH)
    climatology_ds = xr.open_dataset(DEFAULT_CLIMATOLOGY_PATH)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = np.datetime64(DEFAULT_TIMESTAMP)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    needed_levels = [250, 500, 850, 1000]
    score_by_level = {}

    for level_hpa in needed_levels:
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
        score_by_level[int(level_hpa)] = result.score_points
    return score_by_level, latitudes, longitudes


def load_run_specs() -> list[dict]:
    specs = []

    even_summary = json.loads((EXPERIMENT_DIR / "summary.json").read_text(encoding="utf-8"))
    even_longitudes = [float(lon) for lon in even_summary["sampled_longitudes_degrees"]]
    for level in even_summary["levels_hpa"]:
        specs.append(
            {
                "family": "even_10_longitudes",
                "slug": f"{int(level):04d}hpa",
                "title": f"Even 10 longitudes, {int(level)} hPa",
                "level_hpa": int(level),
                "longitudes": even_longitudes,
                "map_extent": [-180.0, 180.0, -90.0, 90.0],
                "window": None,
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/summary.json",
            }
        )

    random_summary = json.loads((EXPERIMENT_DIR / "random_850hpa_summary.json").read_text(encoding="utf-8"))
    for plot in random_summary["plots"]:
        specs.append(
            {
                "family": "random_850hpa",
                "slug": f"set_{int(plot['group_number']):02d}",
                "title": f"Random 850 hPa set {int(plot['group_number'])}",
                "level_hpa": 850,
                "longitudes": [float(lon) for lon in plot["sampled_longitudes_degrees"]],
                "map_extent": [-180.0, 180.0, -90.0, 90.0],
                "window": None,
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/random_850hpa_summary.json",
            }
        )

    regional_summary = json.loads((EXPERIMENT_DIR / "regional_windows_850hpa_summary.json").read_text(encoding="utf-8"))
    for plot in regional_summary["plots"]:
        specs.append(
            {
                "family": "regional_850hpa",
                "slug": plot["region"].lower().replace(" ", "_"),
                "title": f"{plot['region']} 850 hPa longitude window",
                "level_hpa": 850,
                "longitudes": [float(lon) for lon in plot["sampled_longitudes_degrees"]],
                "map_extent": [float(value) for value in plot["map_extent"]],
                "window": [float(value) for value in plot["window_degrees"]],
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/regional_windows_850hpa_summary.json",
            }
        )

    return specs


def plot_profile(
    *,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    north_rows, north_x = pole_to_equator_axis(latitudes, "north")
    south_rows, south_x = pole_to_equator_axis(latitudes, "south")

    fig, ax = plt.subplots(figsize=(12.5, 7))
    for sample_number, lon_index in enumerate(longitude_indices):
        color = colors[sample_number]
        ax.plot(
            north_x,
            score_points[north_rows, lon_index],
            color=color,
            linewidth=1.8,
            alpha=0.95,
        )
        ax.plot(
            south_x,
            score_points[south_rows, lon_index],
            color=color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
        )

    ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks([90, 75, 60, 45, 30, 15, 0])
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Thermal Displacement score")
    ax.set_title(f"{title}\nsolid = Northern Hemisphere, dashed = Southern Hemisphere")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=2.6,
            label=longitude_label(float(longitudes[lon_index])),
        )
        for i, lon_index in enumerate(longitude_indices)
    ]
    hemisphere_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.0, label="Northern"),
        Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="--", label="Southern"),
    ]
    longitude_legend = ax.legend(handles=longitude_handles, title="Longitude", loc="upper left", frameon=True)
    ax.add_artist(longitude_legend)
    ax.legend(handles=hemisphere_handles, title="Hemisphere", loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_source_map(
    *,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    requested_longitudes: list[float],
    title: str,
    map_extent: list[float],
    window: list[float] | None,
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10").colors[: len(requested_longitudes)]
    transform = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12.5, 7))
    ax = fig.add_subplot(1, 1, 1, projection=transform)

    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        score_points,
        cmap="coolwarm",
        vmin=0.0,
        vmax=100.0,
        shading="auto",
        transform=transform,
        zorder=1,
    )
    ax.set_extent(map_extent, crs=transform)
    ax.coastlines(resolution="110m", linewidth=0.65, color="#111111", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.32, edgecolor="#222222", zorder=4)
    gl = ax.gridlines(
        crs=transform,
        draw_labels=True,
        linewidth=0.35,
        color="#666666",
        alpha=0.45,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    lat_min, lat_max = float(map_extent[2]), float(map_extent[3])
    if window is not None:
        for boundary_lon in window:
            ax.plot(
                [boundary_lon, boundary_lon],
                [lat_min, lat_max],
                color="#111111",
                linewidth=2.6,
                transform=transform,
                zorder=7,
            )

    for sample_number, sampled_lon in enumerate(requested_longitudes):
        ax.plot(
            [sampled_lon, sampled_lon],
            [lat_min, lat_max],
            color=colors[sample_number],
            linewidth=2.2,
            transform=transform,
            zorder=8,
            label=longitude_label(sampled_lon),
        )

    ax.set_title(f"{title}\nThermal Displacement map with profile-source meridians")
    ax.legend(title="Extracted longitude", loc="lower left", frameon=True, ncols=2 if len(requested_longitudes) > 5 else 1)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Thermal Displacement score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    score_by_level, latitudes, longitudes = compute_score_fields()
    specs = load_run_specs()
    manifest_entries = []

    for spec in specs:
        output_dir = OUTPUT_ROOT / spec["family"] / spec["slug"]
        output_dir.mkdir(parents=True, exist_ok=True)
        longitude_indices = nearest_longitude_indices(longitudes, spec["longitudes"])
        score_points = score_by_level[int(spec["level_hpa"])]
        profile_path = output_dir / "line_profile.png"
        map_path = output_dir / "thermal_displacement_map.png"
        metadata_path = output_dir / "metadata.json"

        plot_profile(
            score_points=score_points,
            latitudes=latitudes,
            longitudes=longitudes,
            longitude_indices=longitude_indices,
            title=spec["title"],
            output_path=profile_path,
        )
        plot_source_map(
            score_points=score_points,
            latitudes=latitudes,
            longitudes=longitudes,
            requested_longitudes=[float(longitudes[index]) for index in longitude_indices],
            title=spec["title"],
            map_extent=spec["map_extent"],
            window=spec["window"],
            output_path=map_path,
        )

        metadata = {
            "title": spec["title"],
            "family": spec["family"],
            "pressure_level_hpa": int(spec["level_hpa"]),
            "timestamp": DEFAULT_TIMESTAMP,
            "line_profile": display_path(profile_path),
            "thermal_displacement_map": display_path(map_path),
            "source_summary": spec["source_summary"],
            "requested_longitudes_degrees": spec["longitudes"],
            "matched_grid_longitudes_degrees": [float(longitudes[index]) for index in longitude_indices],
            "matched_grid_longitude_labels": [longitude_label(float(longitudes[index])) for index in longitude_indices],
            "map_extent": spec["map_extent"],
            "window": spec["window"],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest_entries.append(metadata)

    manifest = {
        "experiment": "thermal-displacement-longitude-hemisphere-lines",
        "iteration": "organized profile and Thermal Displacement source-map pairs",
        "dataset": display_path(DEFAULT_DATASET_PATH),
        "climatology": display_path(DEFAULT_CLIMATOLOGY_PATH),
        "timestamp": DEFAULT_TIMESTAMP,
        "score_smooth_sigma_cells": SCORE_SMOOTH_SIGMA_CELLS,
        "output_root": display_path(OUTPUT_ROOT),
        "pairs": manifest_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
