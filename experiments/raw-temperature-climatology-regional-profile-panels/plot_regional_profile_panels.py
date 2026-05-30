from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("CARTOPY_DATA_DIR", str(CACHE_ROOT / "cartopy"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr


EXPERIMENT_DIR = Path("tmp/raw-temperature-climatology-regional-profile-panels")
PLOTS_DIR = EXPERIMENT_DIR / "plots"
SUMMARY_PATH = EXPERIMENT_DIR / "summary.json"

RAW_DATASET_PATH = REPO_ROOT / "data/era5_temperature_2021-11_08-12.nc"
RAW_DATASET_DISPLAY_PATH = "data/era5_temperature_2021-11_08-12.nc"
CLIMATOLOGY_DATASET_PATH = (
    REPO_ROOT
    / "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
CLIMATOLOGY_DATASET_DISPLAY_PATH = (
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
TEMPERATURE_VARIABLE = "t"
TIMESTAMP = "2021-11-08T12:00"
LEVELS_HPA = [250, 500, 850, 1000]
LATITUDE_TICKS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)

REGIONAL_WINDOWS = [
    {
        "region": "North America",
        "slug": "north_america",
        "window": [-130.0, -80.0],
        "longitudes": [-130.0, -110.0, -90.0],
        "map_extent": [-142.0, -68.0, 5.0, 78.0],
    },
    {
        "region": "Europe",
        "slug": "europe",
        "window": [-10.0, 40.0],
        "longitudes": [-10.0, 10.0, 30.0],
        "map_extent": [-22.0, 52.0, 25.0, 75.0],
    },
    {
        "region": "Russia",
        "slug": "russia",
        "window": [60.0, 110.0],
        "longitudes": [60.0, 80.0, 100.0],
        "map_extent": [48.0, 122.0, 35.0, 82.0],
    },
]


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


def map_extent_for_region(region: dict, hemisphere: str) -> list[float]:
    lon_min, lon_max, lat_min, lat_max = region["map_extent"]
    if hemisphere == "north":
        return [float(lon_min), float(lon_max), float(lat_min), float(lat_max)]
    return [float(lon_min), float(lon_max), -float(lat_max), -float(lat_min)]


def marker_latitudes_for_hemisphere(hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return LATITUDE_TICKS_ABS
    if hemisphere == "south":
        return -LATITUDE_TICKS_ABS
    raise ValueError(f"Unknown hemisphere: {hemisphere}")


def field_stats(field_k: np.ndarray) -> dict[str, float]:
    return {
        "temperature_min_k": float(np.nanmin(field_k)),
        "temperature_max_k": float(np.nanmax(field_k)),
        "temperature_mean_k": float(np.nanmean(field_k)),
    }


def combined_limits(*stats: dict[str, float]) -> tuple[float, float]:
    minimum = min(item["temperature_min_k"] for item in stats)
    maximum = max(item["temperature_max_k"] for item in stats)
    pad = max(1.0, 0.03 * (maximum - minimum))
    return minimum - pad, maximum + pad


def load_raw_fields() -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    with xr.open_dataset(RAW_DATASET_PATH) as dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        selected_time = np.datetime64(TIMESTAMP)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        raw_by_level = {}

        for level_hpa in LEVELS_HPA:
            raw_by_level[level_hpa] = (
                temperature.sel(valid_time=selected_time, pressure_level=float(level_hpa))
                .load()
                .to_numpy()
                .astype(np.float32)
            )

    return raw_by_level, latitudes, longitudes


def load_climatology_fields() -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, int]:
    with xr.open_dataset(CLIMATOLOGY_DATASET_PATH) as dataset:
        temperature = dataset[TEMPERATURE_VARIABLE]
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        valid_time_count = int(dataset.sizes["valid_time"])
        climatology_by_level = {}

        for level_hpa in LEVELS_HPA:
            climatology_by_level[level_hpa] = (
                temperature.sel(pressure_level=float(level_hpa))
                .mean(dim="valid_time", skipna=True)
                .load()
                .to_numpy()
                .astype(np.float32)
            )

    return climatology_by_level, latitudes, longitudes, valid_time_count


def assert_matching_grid(
    raw_latitudes: np.ndarray,
    raw_longitudes: np.ndarray,
    climatology_latitudes: np.ndarray,
    climatology_longitudes: np.ndarray,
) -> None:
    if raw_latitudes.shape != climatology_latitudes.shape or not np.allclose(raw_latitudes, climatology_latitudes):
        raise ValueError("Raw temperature and climatology latitude grids do not match.")
    if raw_longitudes.shape != climatology_longitudes.shape or not np.allclose(raw_longitudes, climatology_longitudes):
        raise ValueError("Raw temperature and climatology longitude grids do not match.")


def draw_profile_axis(
    ax,
    *,
    region: dict,
    level_hpa: int,
    hemisphere: str,
    raw_temperature_k: np.ndarray,
    climatology_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    colors,
    y_limits: tuple[float, float],
) -> None:
    rows, x_axis = pole_to_equator_axis(latitudes, hemisphere)
    for sample_number, lon_index in enumerate(longitude_indices):
        color = colors[sample_number]
        ax.plot(
            x_axis,
            raw_temperature_k[rows, lon_index],
            color=color,
            linewidth=2.1,
            alpha=0.95,
        )
        ax.plot(
            x_axis,
            climatology_k[rows, lon_index],
            color=color,
            linewidth=2.1,
            alpha=0.95,
            linestyle="--",
        )

    ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
    ax.set_ylim(*y_limits)
    ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"{region['region']} {level_hpa} hPa {hemisphere.title()} Hemisphere profiles")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=2.8,
            label=longitude_label(float(longitudes[lon_index])),
        )
        for i, lon_index in enumerate(longitude_indices)
    ]
    style_handles = [
        Line2D([0], [0], color="#222222", linewidth=2.4, linestyle="-", label="Raw temperature"),
        Line2D([0], [0], color="#222222", linewidth=2.4, linestyle="--", label="Climatology"),
    ]
    first_legend = ax.legend(handles=longitude_handles, title="Longitude", loc="upper left", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=style_handles, title="Field", loc="lower left", frameon=True)


def draw_temperature_map_axis(
    ax,
    *,
    title: str,
    region: dict,
    hemisphere: str,
    field_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sampled_longitudes: list[float],
    colors,
    color_limits: tuple[float, float],
):
    transform = ccrs.PlateCarree()
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        field_k,
        cmap="coolwarm",
        vmin=color_limits[0],
        vmax=color_limits[1],
        shading="auto",
        transform=transform,
        zorder=1,
    )

    map_extent = map_extent_for_region(region, hemisphere)
    ax.set_extent(map_extent, crs=transform)
    ax.set_aspect("auto")
    ax.coastlines(resolution="110m", linewidth=0.65, color="#111111", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.32, edgecolor="#222222", zorder=4)
    gridlines = ax.gridlines(
        crs=transform,
        draw_labels=True,
        linewidth=0.35,
        color="#666666",
        alpha=0.45,
        linestyle=":",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}

    lat_min, lat_max = map_extent[2], map_extent[3]
    marker_lats = marker_latitudes_for_hemisphere(hemisphere)
    marker_lats = marker_lats[(marker_lats >= min(lat_min, lat_max)) & (marker_lats <= max(lat_min, lat_max))]

    for boundary_lon in region["window"]:
        ax.plot(
            [boundary_lon, boundary_lon],
            [lat_min, lat_max],
            color="#111111",
            linewidth=2.2,
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
            s=26,
            facecolors=color,
            edgecolors="#111111",
            linewidths=0.55,
            transform=transform,
            zorder=9,
        )

    ax.set_title(title)
    return mesh


def write_combined_plot(
    *,
    level_hpa: int,
    region: dict,
    hemisphere: str,
    raw_temperature_k: np.ndarray,
    climatology_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
    raw_stats: dict[str, float],
    climatology_stats: dict[str, float],
) -> None:
    sampled_longitudes = [float(longitudes[index]) for index in longitude_indices]
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    y_limits = combined_limits(raw_stats, climatology_stats)
    raw_color_limits = (raw_stats["temperature_min_k"], raw_stats["temperature_max_k"])
    climatology_color_limits = (
        climatology_stats["temperature_min_k"],
        climatology_stats["temperature_max_k"],
    )

    fig = plt.figure(figsize=(17.2, 10.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.965,
        bottom=0.085,
        top=0.88,
        wspace=0.17,
        hspace=0.28,
    )
    profile_ax = fig.add_subplot(grid[:, 0])
    raw_map_ax = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    climatology_map_ax = fig.add_subplot(grid[1, 1], projection=ccrs.PlateCarree())

    fig.suptitle(
        f"{region['region']} {level_hpa} hPa longitude window, {hemisphere.title()} Hemisphere\n"
        "left = solid raw profiles plus dashed climatology profiles; right = raw map over climatology map",
        fontsize=15,
    )

    draw_profile_axis(
        profile_ax,
        region=region,
        level_hpa=level_hpa,
        hemisphere=hemisphere,
        raw_temperature_k=raw_temperature_k,
        climatology_k=climatology_k,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        colors=colors,
        y_limits=y_limits,
    )
    raw_mesh = draw_temperature_map_axis(
        raw_map_ax,
        title="Raw temperature map",
        region=region,
        hemisphere=hemisphere,
        field_k=raw_temperature_k,
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_longitudes=sampled_longitudes,
        colors=colors,
        color_limits=raw_color_limits,
    )
    climatology_mesh = draw_temperature_map_axis(
        climatology_map_ax,
        title="Raw temperature climatology map",
        region=region,
        hemisphere=hemisphere,
        field_k=climatology_k,
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_longitudes=sampled_longitudes,
        colors=colors,
        color_limits=climatology_color_limits,
    )

    raw_colorbar = fig.colorbar(raw_mesh, ax=raw_map_ax, orientation="horizontal", fraction=0.07, pad=0.075)
    raw_colorbar.set_label("Raw temperature (K)")
    climatology_colorbar = fig.colorbar(
        climatology_mesh,
        ax=climatology_map_ax,
        orientation="horizontal",
        fraction=0.07,
        pad=0.075,
    )
    climatology_colorbar.set_label("Raw temperature climatology (K)")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    clean_pngs(PLOTS_DIR)

    raw_by_level, raw_latitudes, raw_longitudes = load_raw_fields()
    climatology_by_level, climatology_latitudes, climatology_longitudes, valid_time_count = load_climatology_fields()
    assert_matching_grid(raw_latitudes, raw_longitudes, climatology_latitudes, climatology_longitudes)

    raw_ranges = {level: field_stats(field) for level, field in raw_by_level.items()}
    climatology_ranges = {level: field_stats(field) for level, field in climatology_by_level.items()}
    output_entries = []

    for level_hpa in LEVELS_HPA:
        raw_temperature_k = raw_by_level[level_hpa]
        climatology_k = climatology_by_level[level_hpa]

        for region in REGIONAL_WINDOWS:
            longitude_indices = nearest_longitude_indices(raw_longitudes, region["longitudes"])
            matched_longitudes = [float(raw_longitudes[index]) for index in longitude_indices]

            for hemisphere in ("north", "south"):
                plot_path = (
                    PLOTS_DIR
                    / f"raw_temperature_climatology_regional_profile_panel_{level_hpa:04d}hpa_{region['slug']}_{hemisphere}.png"
                )
                print(f"plotting {level_hpa} hPa {region['region']} {hemisphere}", flush=True)
                write_combined_plot(
                    level_hpa=level_hpa,
                    region=region,
                    hemisphere=hemisphere,
                    raw_temperature_k=raw_temperature_k,
                    climatology_k=climatology_k,
                    latitudes=raw_latitudes,
                    longitudes=raw_longitudes,
                    longitude_indices=longitude_indices,
                    output_path=plot_path,
                    raw_stats=raw_ranges[level_hpa],
                    climatology_stats=climatology_ranges[level_hpa],
                )

                output_entries.append(
                    {
                        "region": region["region"],
                        "hemisphere": hemisphere,
                        "pressure_level_hpa": level_hpa,
                        "plot": display_path(plot_path),
                        "window_degrees": region["window"],
                        "requested_longitudes_degrees": region["longitudes"],
                        "matched_grid_longitudes_degrees": matched_longitudes,
                        "matched_grid_longitude_labels": [longitude_label(lon) for lon in matched_longitudes],
                        "map_extent": map_extent_for_region(region, hemisphere),
                        "latitude_marker_degrees": marker_latitudes_for_hemisphere(hemisphere).tolist(),
                        "profile_line_count": 6,
                        "profile_line_encoding": "same color per longitude; solid is raw temperature; dashed is climatology",
                        "raw_map_field": "raw_temperature_k",
                        "climatology_map_field": "raw_temperature_climatology_k",
                        "temperature_unit": "K",
                        "raw_color_scale": "per pressure level raw temperature min/max",
                        "climatology_color_scale": "per pressure level climatology min/max",
                    }
                )

    summary = {
        "experiment": "raw-temperature-climatology-regional-profile-panels",
        "date": "2026-05-29",
        "method": "regional longitude-window profile-map panels combining raw ERA5 temperature with valid_time-mean raw temperature climatology",
        "raw_dataset": RAW_DATASET_DISPLAY_PATH,
        "raw_timestamp": TIMESTAMP,
        "climatology_dataset": CLIMATOLOGY_DATASET_DISPLAY_PATH,
        "climatology_method": "mean across valid_time",
        "climatology_valid_time_count": valid_time_count,
        "variable": TEMPERATURE_VARIABLE,
        "levels_hpa": LEVELS_HPA,
        "regions": [region["region"] for region in REGIONAL_WINDOWS],
        "hemispheres": ["north", "south"],
        "window_width_degrees": 50.0,
        "longitudes_per_region": 3,
        "profile_line_count_per_plot": 6,
        "x_axis": "absolute latitude, ordered from pole to equator for the selected hemisphere",
        "line_encoding": "color is longitude; solid is raw temperature; dashed is climatology",
        "map_encoding": "black vertical lines are 50-degree window boundaries; colored vertical lines are sampled longitudes; dot markers are plotted every 15 latitude degrees",
        "temperature_unit": "K",
        "script": display_path(EXPERIMENT_DIR / "plot_regional_profile_panels.py"),
        "plots_dir": display_path(PLOTS_DIR),
        "plot_count": len(output_entries),
        "raw_level_ranges": {str(level): stats for level, stats in sorted(raw_ranges.items())},
        "climatology_level_ranges": {
            str(level): stats for level, stats in sorted(climatology_ranges.items())
        },
        "plots": output_entries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
