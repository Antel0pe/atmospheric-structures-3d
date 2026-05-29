from __future__ import annotations

import json
from pathlib import Path
import shutil

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path("tmp/raw-temperature-longitude-hemisphere-lines")
PLOTS_DIR = EXPERIMENT_DIR / "plots"
RANDOM_DIR = EXPERIMENT_DIR / "plots-random-sets-850hpa"
REGIONAL_DIR = EXPERIMENT_DIR / "plots-regional-windows-850hpa"
ORGANIZED_DIR = EXPERIMENT_DIR / "organized-profile-map-pairs"
SUMMARY_PATH = EXPERIMENT_DIR / "all_hemisphere_profile_map_summary.json"

DATASET_PATH = REPO_ROOT / "data/era5_temperature_2021-11_08-12.nc"
DATASET_DISPLAY_PATH = "data/era5_temperature_2021-11_08-12.nc"
TEMPERATURE_VARIABLE = "t"
TIMESTAMP = "2021-11-08T12:00"
LEVELS_HPA = [250, 500, 850, 1000]
LONGITUDE_SAMPLE_COUNT = 10
RANDOM_SEED = 20260528
RANDOM_PLOT_COUNT = 3
RANDOM_LONGITUDES_PER_PLOT = 5
REGIONAL_WINDOWS = [
    {
        "region": "North America",
        "slug": "north_america",
        "window": [-130.0, -80.0],
        "longitudes": [-130.0, -120.0, -110.0, -100.0, -90.0],
        "map_extent": [-142.0, -68.0, 5.0, 78.0],
    },
    {
        "region": "Europe",
        "slug": "europe",
        "window": [-10.0, 40.0],
        "longitudes": [-10.0, 0.0, 10.0, 20.0, 30.0],
        "map_extent": [-22.0, 52.0, 25.0, 75.0],
    },
    {
        "region": "Russia",
        "slug": "russia",
        "window": [60.0, 110.0],
        "longitudes": [60.0, 70.0, 80.0, 90.0, 100.0],
        "map_extent": [48.0, 122.0, 35.0, 82.0],
    },
]
LATITUDE_TICKS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)


def display_path(path: Path) -> str:
    return path.as_posix()


def longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    value = f"{abs(normalized):.2f}".rstrip("0").rstrip(".")
    return f"{value}\N{DEGREE SIGN}{hemisphere}"


def clean_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for png in path.glob("*.png"):
        png.unlink()


def reset_organized_dir() -> None:
    if ORGANIZED_DIR.exists():
        shutil.rmtree(ORGANIZED_DIR)
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)


def select_even_longitude_indices(longitudes: np.ndarray, count: int) -> np.ndarray:
    if count > longitudes.size:
        raise ValueError("Requested more longitude samples than available columns.")
    indices = np.floor(np.linspace(0, longitudes.size, count, endpoint=False)).astype(int)
    return np.unique(indices)


def choose_nonoverlapping_longitude_sets(longitudes: np.ndarray) -> list[np.ndarray]:
    needed = RANDOM_PLOT_COUNT * RANDOM_LONGITUDES_PER_PLOT
    if needed > longitudes.size:
        raise ValueError("Not enough longitudes for non-overlapping plot sets.")

    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(longitudes.size, size=needed, replace=False)
    groups = []
    for plot_index in range(RANDOM_PLOT_COUNT):
        start = plot_index * RANDOM_LONGITUDES_PER_PLOT
        stop = start + RANDOM_LONGITUDES_PER_PLOT
        groups.append(np.sort(chosen[start:stop]))
    return groups


def nearest_longitude_indices(longitudes: np.ndarray, requested_lons: list[float]) -> np.ndarray:
    indices = [int(np.argmin(np.abs(longitudes - requested_lon))) for requested_lon in requested_lons]
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


def load_temperature_fields(levels_hpa: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    with xr.open_dataset(DATASET_PATH) as dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        selected_time = np.datetime64(TIMESTAMP)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        temperature_by_level = {}

        for level_hpa in sorted(set(int(level) for level in levels_hpa)):
            raw_level = (
                temperature.sel(valid_time=selected_time, pressure_level=float(level_hpa))
                .load()
                .to_numpy()
                .astype(np.float32)
            )
            temperature_by_level[level_hpa] = raw_level

    return temperature_by_level, latitudes, longitudes


def level_stats(raw_temperature_k: np.ndarray) -> dict:
    return {
        "temperature_min_k": float(np.nanmin(raw_temperature_k)),
        "temperature_max_k": float(np.nanmax(raw_temperature_k)),
        "temperature_mean_k": float(np.nanmean(raw_temperature_k)),
    }


def load_specs(longitudes: np.ndarray) -> tuple[list[dict], dict, dict]:
    specs = []
    even_indices = select_even_longitude_indices(longitudes, LONGITUDE_SAMPLE_COUNT)
    even_longitudes = [float(longitudes[index]) for index in even_indices]

    even_summary_plots = []
    for level in LEVELS_HPA:
        filename_base = f"raw_temperature_longitude_lines_{int(level):04d}hpa"
        plot_path = PLOTS_DIR / f"{filename_base}.png"
        even_summary_plots.append(
            {
                "pressure_level_hpa": int(level),
                "plot": display_path(plot_path),
            }
        )
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
                "filename_base": filename_base,
                "source_summary": "tmp/raw-temperature-longitude-hemisphere-lines/summary.json",
            }
        )

    random_summary_plots = []
    for group_number, longitude_indices in enumerate(choose_nonoverlapping_longitude_sets(longitudes), start=1):
        sampled_longitudes = [float(longitudes[index]) for index in longitude_indices]
        filename_base = f"raw_temperature_850hpa_random_longitude_set_{group_number:02d}"
        plot_path = RANDOM_DIR / f"{filename_base}.png"
        random_summary_plots.append(
            {
                "group_number": group_number,
                "plot": display_path(plot_path),
                "sampled_longitudes_degrees": sampled_longitudes,
                "sampled_longitude_labels": [longitude_label(lon) for lon in sampled_longitudes],
            }
        )
        specs.append(
            {
                "family": "random_850hpa",
                "slug": f"set_{group_number:02d}",
                "title": f"Random 850 hPa set {group_number}",
                "level_hpa": 850,
                "longitudes": sampled_longitudes,
                "map_extent": None,
                "window": None,
                "plot_dir": RANDOM_DIR,
                "filename_base": filename_base,
                "source_summary": "tmp/raw-temperature-longitude-hemisphere-lines/random_850hpa_summary.json",
            }
        )

    for region in REGIONAL_WINDOWS:
        filename_base = f"raw_temperature_850hpa_{region['slug']}_longitude_window"
        specs.append(
            {
                "family": "regional_850hpa",
                "slug": region["slug"],
                "title": f"{region['region']} 850 hPa longitude window",
                "level_hpa": 850,
                "longitudes": region["longitudes"],
                "map_extent": region["map_extent"],
                "window": region["window"],
                "plot_dir": REGIONAL_DIR,
                "filename_base": filename_base,
                "source_summary": "tmp/raw-temperature-longitude-hemisphere-lines/regional_windows_850hpa_summary.json",
            }
        )

    even_summary = {
        "experiment": "raw-temperature-longitude-hemisphere-lines",
        "method": "raw ERA5 pressure-level temperature",
        "dataset": DATASET_DISPLAY_PATH,
        "timestamp": TIMESTAMP,
        "levels_hpa": LEVELS_HPA,
        "temperature_unit": "K",
        "longitude_sample_count": len(even_longitudes),
        "sampled_longitudes_degrees": even_longitudes,
        "x_axis": "absolute latitude, ordered from pole to equator separately for each hemisphere",
        "line_encoding": "color is longitude; solid is Northern Hemisphere; dashed is Southern Hemisphere in the one-panel summary plots",
        "plots": even_summary_plots,
    }
    random_summary = {
        "experiment": "raw-temperature-longitude-hemisphere-lines",
        "iteration": "random non-overlapping 850 hPa longitude sets",
        "method": "raw ERA5 pressure-level temperature",
        "dataset": DATASET_DISPLAY_PATH,
        "timestamp": TIMESTAMP,
        "pressure_level_hpa": 850,
        "temperature_unit": "K",
        "random_seed": RANDOM_SEED,
        "plot_count": RANDOM_PLOT_COUNT,
        "longitudes_per_plot": RANDOM_LONGITUDES_PER_PLOT,
        "line_count_per_plot": RANDOM_LONGITUDES_PER_PLOT * 2,
        "x_axis": "absolute latitude, ordered from pole to equator separately for each hemisphere",
        "line_encoding": "color is longitude; solid is Northern Hemisphere; dashed is Southern Hemisphere in the one-panel summary plots",
        "plots": random_summary_plots,
    }
    return specs, even_summary, random_summary


def draw_profile_axis(
    ax,
    *,
    spec: dict,
    hemisphere: str,
    raw_temperature_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    colors,
    y_limits: tuple[float, float],
) -> None:
    rows, x_axis = pole_to_equator_axis(latitudes, hemisphere)
    for sample_number, lon_index in enumerate(longitude_indices):
        ax.plot(
            x_axis,
            raw_temperature_k[rows, lon_index],
            color=colors[sample_number],
            linewidth=2.0,
            alpha=0.95,
        )

    ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
    ax.set_ylim(*y_limits)
    ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Raw temperature (K)")
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


def draw_raw_temperature_map_axis(
    ax,
    *,
    spec: dict,
    hemisphere: str,
    raw_temperature_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sampled_longitudes: list[float],
    colors,
    color_limits: tuple[float, float],
) -> None:
    transform = ccrs.PlateCarree()
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        raw_temperature_k,
        cmap="coolwarm",
        vmin=color_limits[0],
        vmax=color_limits[1],
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

    ax.set_title(f"{spec['level_hpa']} hPa raw temperature map, {hemisphere.title()} Hemisphere")
    ax.legend(title="Extracted longitude", loc="lower left", frameon=True, ncols=2 if len(sampled_longitudes) > 5 else 1)
    return mesh


def write_combined_plot(
    *,
    spec: dict,
    hemisphere: str,
    raw_temperature_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
    level_range: dict,
) -> None:
    sampled_longitudes = [float(longitudes[index]) for index in longitude_indices]
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    y_pad = max(1.0, 0.03 * (level_range["temperature_max_k"] - level_range["temperature_min_k"]))
    y_limits = (
        level_range["temperature_min_k"] - y_pad,
        level_range["temperature_max_k"] + y_pad,
    )
    color_limits = (
        level_range["temperature_min_k"],
        level_range["temperature_max_k"],
    )

    fig = plt.figure(figsize=(16.4, 7.2))
    profile_ax = fig.add_subplot(1, 2, 1)
    map_ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{spec['title']}, {hemisphere.title()} Hemisphere\n"
        "profile = raw temperature in K; map = raw temperature with extraction meridians and 15-degree markers",
        fontsize=15,
    )

    draw_profile_axis(
        profile_ax,
        spec=spec,
        hemisphere=hemisphere,
        raw_temperature_k=raw_temperature_k,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        colors=colors,
        y_limits=y_limits,
    )
    mesh = draw_raw_temperature_map_axis(
        map_ax,
        spec=spec,
        hemisphere=hemisphere,
        raw_temperature_k=raw_temperature_k,
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_longitudes=sampled_longitudes,
        colors=colors,
        color_limits=color_limits,
    )
    colorbar = fig.colorbar(mesh, ax=map_ax, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Raw temperature (K)")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    clean_pngs(PLOTS_DIR)
    clean_pngs(RANDOM_DIR)
    clean_pngs(REGIONAL_DIR)
    reset_organized_dir()

    temperature_by_level, latitudes, longitudes = load_temperature_fields(LEVELS_HPA)
    specs, even_summary, random_summary = load_specs(longitudes)
    level_ranges = {level: level_stats(field) for level, field in temperature_by_level.items()}
    output_entries = []

    for spec in specs:
        raw_temperature_k = temperature_by_level[int(spec["level_hpa"])]
        longitude_indices = nearest_longitude_indices(longitudes, spec["longitudes"])
        matched_longitudes = [float(longitudes[index]) for index in longitude_indices]

        for hemisphere in ("north", "south"):
            plot_path = spec["plot_dir"] / f"{spec['filename_base']}_{hemisphere}.png"
            write_combined_plot(
                spec=spec,
                hemisphere=hemisphere,
                raw_temperature_k=raw_temperature_k,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=plot_path,
                level_range=level_ranges[int(spec["level_hpa"])],
            )

            organized_leaf = ORGANIZED_DIR / spec["family"] / spec["slug"] / hemisphere
            organized_leaf.mkdir(parents=True, exist_ok=True)
            organized_plot_path = organized_leaf / "profile_and_raw_temperature_map.png"
            write_combined_plot(
                spec=spec,
                hemisphere=hemisphere,
                raw_temperature_k=raw_temperature_k,
                latitudes=latitudes,
                longitudes=longitudes,
                longitude_indices=longitude_indices,
                output_path=organized_plot_path,
                level_range=level_ranges[int(spec["level_hpa"])],
            )

            metadata = {
                "title": spec["title"],
                "family": spec["family"],
                "hemisphere": hemisphere,
                "pressure_level_hpa": int(spec["level_hpa"]),
                "timestamp": TIMESTAMP,
                "profile_and_raw_temperature_map": display_path(organized_plot_path),
                "matching_plot_folder_output": display_path(plot_path),
                "source_summary": spec["source_summary"],
                "requested_longitudes_degrees": spec["longitudes"],
                "matched_grid_longitudes_degrees": matched_longitudes,
                "matched_grid_longitude_labels": [longitude_label(lon) for lon in matched_longitudes],
                "latitude_marker_degrees": marker_latitudes_for_hemisphere(hemisphere).tolist(),
                "map_extent": map_extent_for_spec(spec, hemisphere),
                "window": spec["window"],
                "map_field": "raw_temperature_k",
                "temperature_unit": "K",
                "color_scale": "per pressure level raw temperature min/max",
                **level_ranges[int(spec["level_hpa"])],
            }
            (organized_leaf / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            output_entries.append(metadata)

    for plot in even_summary["plots"]:
        plot.update(level_ranges[int(plot["pressure_level_hpa"])])
    even_summary["plot_dirs"] = [display_path(PLOTS_DIR)]
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(even_summary, indent=2), encoding="utf-8")

    random_summary["plot_dirs"] = [display_path(RANDOM_DIR)]
    random_summary.update(level_ranges[850])
    (EXPERIMENT_DIR / "random_850hpa_summary.json").write_text(json.dumps(random_summary, indent=2), encoding="utf-8")

    summary = {
        "experiment": "raw-temperature-longitude-hemisphere-lines",
        "iteration": "all outputs split by hemisphere with paired raw-temperature source maps",
        "method": "raw ERA5 pressure-level temperature, no derived equivalent-latitude transform",
        "dataset": DATASET_DISPLAY_PATH,
        "timestamp": TIMESTAMP,
        "temperature_unit": "K",
        "map_field": "raw_temperature_k",
        "latitude_marker_degrees_abs": LATITUDE_TICKS_ABS.tolist(),
        "color_scale": "per pressure level raw temperature min/max",
        "level_ranges": {str(level): stats for level, stats in sorted(level_ranges.items())},
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
        "experiment": "raw-temperature-longitude-hemisphere-lines",
        "iteration": "regional 850 hPa longitude windows split by hemisphere with raw-temperature maps",
        "method": "raw ERA5 pressure-level temperature",
        "dataset": DATASET_DISPLAY_PATH,
        "timestamp": TIMESTAMP,
        "pressure_level_hpa": 850,
        "temperature_unit": "K",
        "window_width_degrees": 50.0,
        "longitude_increment_degrees": 10.0,
        "longitudes_per_region": 5,
        "plots_per_region": 2,
        "line_count_per_plot": 5,
        "x_axis": "absolute latitude, ordered from pole to equator for the selected hemisphere",
        "line_encoding": "color is longitude",
        "map_field": "raw_temperature_k",
        "map_encoding": "black vertical lines are 50-degree window boundaries; colored vertical lines are sampled longitudes; dot markers are plotted every 15 latitude degrees",
        "color_scale": "per pressure level raw temperature min/max",
        **level_ranges[850],
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
