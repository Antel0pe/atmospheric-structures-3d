from __future__ import annotations

import json
from pathlib import Path
import shutil
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
PLOTS_DIR = EXPERIMENT_DIR / "plots"
RANDOM_DIR = EXPERIMENT_DIR / "plots-random-sets-850hpa"
REGIONAL_DIR = EXPERIMENT_DIR / "plots-regional-windows-850hpa"
ORGANIZED_DIR = EXPERIMENT_DIR / "organized-profile-map-pairs"
SUMMARY_PATH = EXPERIMENT_DIR / "all_hemisphere_profile_map_summary.json"
SCORE_SMOOTH_SIGMA_CELLS = 1.0
LATITUDE_TICKS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)


def display_path(path: Path) -> str:
    return path.as_posix()


def longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    value = f"{abs(normalized):.2f}".rstrip("0").rstrip(".")
    return f"{value}°{hemisphere}"


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("°", "")
        .replace("/", "_")
    )


def clean_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for png in path.glob("*.png"):
        png.unlink()


def reset_organized_dir() -> None:
    if ORGANIZED_DIR.exists():
        shutil.rmtree(ORGANIZED_DIR)
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)


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


def map_extent_for_spec(spec: dict, hemisphere: str) -> list[float]:
    if spec["map_extent"] is not None:
        lon_min, lon_max, lat_min, lat_max = spec["map_extent"]
        if hemisphere == "north":
            return [float(lon_min), float(lon_max), float(lat_min), float(lat_max)]
        return [float(lon_min), float(lon_max), -float(lat_max), -float(lat_min)]

    if hemisphere == "north":
        return [-180.0, 180.0, 0.0, 90.0]
    return [-180.0, 180.0, -90.0, 0.0]


def marker_latitudes_for_hemisphere(hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return LATITUDE_TICKS_ABS
    if hemisphere == "south":
        return -LATITUDE_TICKS_ABS
    raise ValueError(f"Unknown hemisphere: {hemisphere}")


def compute_score_fields(levels_hpa: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    temperature_ds = xr.open_dataset(DEFAULT_DATASET_PATH)
    climatology_ds = xr.open_dataset(DEFAULT_CLIMATOLOGY_PATH)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]

    selected_time = np.datetime64(DEFAULT_TIMESTAMP)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
    score_by_level = {}

    for level_hpa in sorted(set(int(level) for level in levels_hpa)):
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
        score_by_level[level_hpa] = result.score_points

    return score_by_level, latitudes, longitudes


def load_specs() -> list[dict]:
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
                "map_extent": None,
                "window": None,
                "plot_dir": PLOTS_DIR,
                "filename_base": f"thermal_displacement_longitude_lines_{int(level):04d}hpa",
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/summary.json",
            }
        )

    random_summary = json.loads((EXPERIMENT_DIR / "random_850hpa_summary.json").read_text(encoding="utf-8"))
    for plot in random_summary["plots"]:
        group_number = int(plot["group_number"])
        specs.append(
            {
                "family": "random_850hpa",
                "slug": f"set_{group_number:02d}",
                "title": f"Random 850 hPa set {group_number}",
                "level_hpa": 850,
                "longitudes": [float(lon) for lon in plot["sampled_longitudes_degrees"]],
                "map_extent": None,
                "window": None,
                "plot_dir": RANDOM_DIR,
                "filename_base": f"thermal_displacement_850hpa_random_longitude_set_{group_number:02d}",
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/random_850hpa_summary.json",
            }
        )

    regional_summary = json.loads((EXPERIMENT_DIR / "regional_windows_850hpa_summary.json").read_text(encoding="utf-8"))
    seen_regions = {}
    for plot in regional_summary["plots"]:
        region = plot["region"]
        if region in seen_regions:
            continue
        seen_regions[region] = plot
        specs.append(
            {
                "family": "regional_850hpa",
                "slug": slugify(region),
                "title": f"{region} 850 hPa longitude window",
                "level_hpa": 850,
                "longitudes": [float(lon) for lon in plot["sampled_longitudes_degrees"]],
                "map_extent": [float(value) for value in plot["map_extent"]],
                "window": [float(value) for value in plot["window_degrees"]],
                "plot_dir": REGIONAL_DIR,
                "filename_base": f"thermal_displacement_850hpa_{slugify(region)}_longitude_window",
                "source_summary": "tmp/thermal-displacement-longitude-hemisphere-lines/regional_windows_850hpa_summary.json",
            }
        )

    return specs


def draw_profile_axis(
    ax,
    *,
    spec: dict,
    hemisphere: str,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    colors,
) -> None:
    rows, x_axis = pole_to_equator_axis(latitudes, hemisphere)
    for sample_number, lon_index in enumerate(longitude_indices):
        ax.plot(
            x_axis,
            score_points[rows, lon_index],
            color=colors[sample_number],
            linewidth=2.0,
            alpha=0.95,
        )

    ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Thermal Displacement score")
    ax.set_title(f"{spec['title']}, {hemisphere.title()} Hemisphere profile")
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
    ax.legend(handles=longitude_handles, title="Longitude", loc="upper left", frameon=True)


def draw_thermal_displacement_map_axis(
    ax,
    *,
    spec: dict,
    hemisphere: str,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sampled_longitudes: list[float],
    colors,
) -> None:
    transform = ccrs.PlateCarree()
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

    map_extent = map_extent_for_spec(spec, hemisphere)
    ax.set_extent(map_extent, crs=transform)
    ax.set_aspect("auto")
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

    lat_min, lat_max = map_extent[2], map_extent[3]
    marker_lats = marker_latitudes_for_hemisphere(hemisphere)
    marker_lats = marker_lats[(marker_lats >= min(lat_min, lat_max)) & (marker_lats <= max(lat_min, lat_max))]

    if spec["window"] is not None:
        for boundary_lon in spec["window"]:
            ax.plot(
                [boundary_lon, boundary_lon],
                [lat_min, lat_max],
                color="#111111",
                linewidth=2.4,
                transform=transform,
                zorder=7,
            )

    for sample_number, sampled_lon in enumerate(sampled_longitudes):
        color = colors[sample_number]
        ax.plot(
            [sampled_lon, sampled_lon],
            [lat_min, lat_max],
            color=color,
            linewidth=2.2,
            transform=transform,
            zorder=8,
            label=longitude_label(sampled_lon),
        )
        ax.scatter(
            np.full(marker_lats.shape, sampled_lon),
            marker_lats,
            s=28,
            facecolors=color,
            edgecolors="#111111",
            linewidths=0.55,
            transform=transform,
            zorder=9,
        )

    ax.set_title(f"{spec['level_hpa']} hPa Thermal Displacement map, {hemisphere.title()} Hemisphere")
    ax.legend(title="Extracted longitude", loc="lower left", frameon=True, ncols=2 if len(sampled_longitudes) > 5 else 1)
    return mesh


def write_combined_plot(
    *,
    spec: dict,
    hemisphere: str,
    score_points: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
) -> None:
    sampled_longitudes = [float(longitudes[index]) for index in longitude_indices]
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]

    fig = plt.figure(figsize=(16.4, 7.2))
    profile_ax = fig.add_subplot(1, 2, 1)
    map_ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{spec['title']}, {hemisphere.title()} Hemisphere\n"
        "profile = Thermal Displacement score; map = Thermal Displacement with extraction meridians and 15-degree markers",
        fontsize=15,
    )

    draw_profile_axis(
        profile_ax,
        spec=spec,
        hemisphere=hemisphere,
        score_points=score_points,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        colors=colors,
    )
    mesh = draw_thermal_displacement_map_axis(
        map_ax,
        spec=spec,
        hemisphere=hemisphere,
        score_points=score_points,
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_longitudes=sampled_longitudes,
        colors=colors,
    )
    colorbar = fig.colorbar(mesh, ax=map_ax, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Thermal Displacement score")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    clean_pngs(PLOTS_DIR)
    clean_pngs(RANDOM_DIR)
    clean_pngs(REGIONAL_DIR)
    reset_organized_dir()

    specs = load_specs()
    score_by_level, latitudes, longitudes = compute_score_fields([spec["level_hpa"] for spec in specs])
    output_entries = []

    for spec in specs:
        score_points = score_by_level[int(spec["level_hpa"])]
        longitude_indices = nearest_longitude_indices(longitudes, spec["longitudes"])
        matched_longitudes = [float(longitudes[index]) for index in longitude_indices]

        for hemisphere in ("north", "south"):
            plot_path = spec["plot_dir"] / f"{spec['filename_base']}_{hemisphere}.png"
            write_combined_plot(
                spec=spec,
                hemisphere=hemisphere,
                score_points=score_points,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=plot_path,
            )

            organized_leaf = ORGANIZED_DIR / spec["family"] / spec["slug"] / hemisphere
            organized_leaf.mkdir(parents=True, exist_ok=True)
            organized_plot_path = organized_leaf / "profile_and_thermal_displacement_map.png"
            write_combined_plot(
                spec=spec,
                hemisphere=hemisphere,
                score_points=score_points,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=organized_plot_path,
            )

            metadata = {
                "title": spec["title"],
                "family": spec["family"],
                "hemisphere": hemisphere,
                "pressure_level_hpa": int(spec["level_hpa"]),
                "timestamp": DEFAULT_TIMESTAMP,
                "profile_and_thermal_displacement_map": display_path(organized_plot_path),
                "matching_plot_folder_output": display_path(plot_path),
                "source_summary": spec["source_summary"],
                "requested_longitudes_degrees": spec["longitudes"],
                "matched_grid_longitudes_degrees": matched_longitudes,
                "matched_grid_longitude_labels": [longitude_label(lon) for lon in matched_longitudes],
                "latitude_marker_degrees": marker_latitudes_for_hemisphere(hemisphere).tolist(),
                "map_extent": map_extent_for_spec(spec, hemisphere),
                "window": spec["window"],
                "map_field": "thermal_displacement_score",
            }
            (organized_leaf / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            output_entries.append(metadata)

    summary = {
        "experiment": "thermal-displacement-longitude-hemisphere-lines",
        "iteration": "all outputs split by hemisphere with paired Thermal Displacement source maps",
        "dataset": display_path(DEFAULT_DATASET_PATH),
        "climatology": display_path(DEFAULT_CLIMATOLOGY_PATH),
        "timestamp": DEFAULT_TIMESTAMP,
        "score_smooth_sigma_cells": SCORE_SMOOTH_SIGMA_CELLS,
        "map_field": "thermal_displacement_score",
        "thermal_displacement_maps_only": True,
        "latitude_marker_degrees_abs": LATITUDE_TICKS_ABS.tolist(),
        "plot_dirs": [
            display_path(PLOTS_DIR),
            display_path(RANDOM_DIR),
            display_path(REGIONAL_DIR),
            display_path(ORGANIZED_DIR),
        ],
        "output_count": len(output_entries),
        "outputs": output_entries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ORGANIZED_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    regional_outputs = [
        entry for entry in output_entries if entry["family"] == "regional_850hpa"
    ]
    regional_summary = {
        "experiment": "thermal-displacement-longitude-hemisphere-lines",
        "iteration": "regional 850 hPa longitude windows split by hemisphere with Thermal Displacement maps",
        "method": "canonical same-longitude same-hemisphere Thermal Displacement",
        "dataset": display_path(DEFAULT_DATASET_PATH),
        "climatology": display_path(DEFAULT_CLIMATOLOGY_PATH),
        "timestamp": DEFAULT_TIMESTAMP,
        "pressure_level_hpa": 850,
        "score_smooth_sigma_cells": SCORE_SMOOTH_SIGMA_CELLS,
        "window_width_degrees": 50.0,
        "longitude_increment_degrees": 10.0,
        "longitudes_per_region": 5,
        "plots_per_region": 2,
        "line_count_per_plot": 5,
        "x_axis": "absolute latitude, ordered from pole to equator for the selected hemisphere",
        "line_encoding": "color is longitude",
        "map_field": "thermal_displacement_score",
        "map_encoding": "black vertical lines are 50-degree window boundaries; colored vertical lines are sampled longitudes; dot markers are plotted every 15 latitude degrees",
        "plots": [
            {
                "region": entry["title"].replace(" 850 hPa longitude window", ""),
                "hemisphere": entry["hemisphere"],
                "plot": entry["matching_plot_folder_output"],
                "window_degrees": entry["window"],
                "sampled_longitudes_degrees": entry["matched_grid_longitudes_degrees"],
                "sampled_longitude_labels": entry["matched_grid_longitude_labels"],
                "map_extent": entry["map_extent"],
                "map_field": entry["map_field"],
                "latitude_marker_degrees": entry["latitude_marker_degrees"],
            }
            for entry in regional_outputs
        ],
    }
    (EXPERIMENT_DIR / "regional_windows_850hpa_summary.json").write_text(
        json.dumps(regional_summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
