from __future__ import annotations

import json
import os
from pathlib import Path

TMP_DIR = Path("tmp")
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_DIR / "runtime-cache"))
os.environ.setdefault("CARTOPY_DATA_DIR", str(TMP_DIR / "cartopy"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr


DATASET_PATH = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
EXPERIMENT_DIR = Path("experiments/raw-temperature-climatology-maps")
PLOTS_DIR = EXPERIMENT_DIR / "plots-regional-windows"
SUMMARY_PATH = EXPERIMENT_DIR / "summary.json"
PRESSURE_LEVELS_HPA = [250, 500, 850, 1000]
LATITUDE_TICKS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)
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


def repo_relative(path: Path) -> str:
    return path.as_posix()


def longitude_label(lon_deg: float) -> str:
    normalized = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, -180.0):
        normalized = 180.0
    hemisphere = "E" if normalized >= 0.0 else "W"
    value = f"{abs(normalized):.2f}".rstrip("0").rstrip(".")
    return f"{value}\N{DEGREE SIGN}{hemisphere}"


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


def level_stats(climatology_k: np.ndarray) -> dict[str, float]:
    return {
        "temperature_min_k": float(np.nanmin(climatology_k)),
        "temperature_max_k": float(np.nanmax(climatology_k)),
        "temperature_mean_k": float(np.nanmean(climatology_k)),
    }


def load_climatology_fields() -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, int]:
    with xr.open_dataset(DATASET_PATH) as dataset:
        temperature = dataset["t"]
        latitudes = np.asarray(dataset["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(dataset["longitude"].values, dtype=np.float32)
        valid_time_count = int(dataset.sizes["valid_time"])
        climatology_by_level = {}

        for level_hpa in PRESSURE_LEVELS_HPA:
            climatology_by_level[level_hpa] = (
                temperature.sel(pressure_level=float(level_hpa))
                .mean(dim="valid_time", skipna=True)
                .load()
                .to_numpy()
                .astype(np.float32)
            )

    return climatology_by_level, latitudes, longitudes, valid_time_count


def draw_profile_axis(
    ax,
    *,
    title: str,
    hemisphere: str,
    climatology_k: np.ndarray,
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
            climatology_k[rows, lon_index],
            color=colors[sample_number],
            linewidth=2.0,
            alpha=0.95,
        )

    ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
    ax.set_ylim(*y_limits)
    ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
    ax.set_xlabel("Latitude path, pole to equator")
    ax.set_ylabel("Raw temperature climatology (K)")
    ax.set_title(f"{title}, {hemisphere.title()} Hemisphere profile")
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


def draw_climatology_map_axis(
    ax,
    *,
    level_hpa: int,
    region: dict,
    hemisphere: str,
    climatology_k: np.ndarray,
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
        climatology_k,
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

    ax.set_title(f"{level_hpa} hPa raw temperature climatology map, {hemisphere.title()} Hemisphere")
    ax.legend(title="Extracted longitude", loc="lower left", frameon=True)
    return mesh


def write_combined_plot(
    *,
    level_hpa: int,
    region: dict,
    hemisphere: str,
    climatology_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    output_path: Path,
    stats: dict[str, float],
) -> None:
    sampled_longitudes = [float(longitudes[index]) for index in longitude_indices]
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    temperature_range = stats["temperature_max_k"] - stats["temperature_min_k"]
    y_pad = max(1.0, 0.03 * temperature_range)
    y_limits = (stats["temperature_min_k"] - y_pad, stats["temperature_max_k"] + y_pad)
    color_limits = (stats["temperature_min_k"], stats["temperature_max_k"])
    title = f"{region['region']} {level_hpa} hPa climatology longitude window"

    fig = plt.figure(figsize=(16.4, 7.2))
    profile_ax = fig.add_subplot(1, 2, 1)
    map_ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{title}, {hemisphere.title()} Hemisphere\n"
        "profile = raw temperature climatology in K; map = climatology with extraction meridians and 15-degree markers",
        fontsize=15,
    )

    draw_profile_axis(
        profile_ax,
        title=title,
        hemisphere=hemisphere,
        climatology_k=climatology_k,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        colors=colors,
        y_limits=y_limits,
    )
    mesh = draw_climatology_map_axis(
        map_ax,
        level_hpa=level_hpa,
        region=region,
        hemisphere=hemisphere,
        climatology_k=climatology_k,
        latitudes=latitudes,
        longitudes=longitudes,
        sampled_longitudes=sampled_longitudes,
        colors=colors,
        color_limits=color_limits,
    )
    colorbar = fig.colorbar(mesh, ax=map_ax, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Raw temperature climatology (K)")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for png in PLOTS_DIR.glob("*.png"):
        png.unlink()

    climatology_by_level, latitudes, longitudes, valid_time_count = load_climatology_fields()
    level_ranges = {level: level_stats(field) for level, field in climatology_by_level.items()}
    output_entries = []

    for level_hpa in PRESSURE_LEVELS_HPA:
        climatology_k = climatology_by_level[level_hpa]
        for region in REGIONAL_WINDOWS:
            longitude_indices = nearest_longitude_indices(longitudes, region["longitudes"])
            matched_longitudes = [float(longitudes[index]) for index in longitude_indices]

            for hemisphere in ("north", "south"):
                plot_path = (
                    PLOTS_DIR
                    / f"raw_temperature_climatology_{level_hpa:04d}hpa_{region['slug']}_longitude_window_{hemisphere}.png"
                )
                print(f"plotting {level_hpa} hPa {region['region']} {hemisphere}", flush=True)
                write_combined_plot(
                    level_hpa=level_hpa,
                    region=region,
                    hemisphere=hemisphere,
                    climatology_k=climatology_k,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    longitude_indices=longitude_indices,
                    output_path=plot_path,
                    stats=level_ranges[level_hpa],
                )

                output_entries.append(
                    {
                        "region": region["region"],
                        "hemisphere": hemisphere,
                        "pressure_level_hpa": level_hpa,
                        "plot": repo_relative(plot_path),
                        "window_degrees": region["window"],
                        "sampled_longitudes_degrees": matched_longitudes,
                        "sampled_longitude_labels": [longitude_label(lon) for lon in matched_longitudes],
                        "map_extent": map_extent_for_region(region, hemisphere),
                        "latitude_marker_degrees": marker_latitudes_for_hemisphere(hemisphere).tolist(),
                        "map_field": "raw_temperature_climatology_k",
                        "temperature_unit": "K",
                        "color_scale": "per pressure level climatology min/max",
                        **level_ranges[level_hpa],
                    }
                )

    summary = {
        "experiment": "raw-temperature-climatology-maps",
        "date": "2026-05-29",
        "iteration": "regional longitude-window profile-map plots using raw temperature climatology",
        "source_dataset": repo_relative(DATASET_PATH),
        "variable": "t",
        "method": "mean raw pressure-level temperature across valid_time entries",
        "valid_time_count": valid_time_count,
        "levels_hpa": PRESSURE_LEVELS_HPA,
        "regions": [region["region"] for region in REGIONAL_WINDOWS],
        "hemispheres": ["north", "south"],
        "window_width_degrees": 50.0,
        "longitude_increment_degrees": 10.0,
        "longitudes_per_region": 5,
        "line_count_per_plot": 5,
        "x_axis": "absolute latitude, ordered from pole to equator for the selected hemisphere",
        "line_encoding": "color is longitude",
        "map_field": "raw_temperature_climatology_k",
        "map_encoding": "black vertical lines are 50-degree window boundaries; colored vertical lines are sampled longitudes; dot markers are plotted every 15 latitude degrees",
        "color_scale": "per pressure level climatology min/max",
        "temperature_unit": "K",
        "script": repo_relative(EXPERIMENT_DIR / "plot_raw_temperature_climatology_maps.py"),
        "plots_dir": repo_relative(PLOTS_DIR),
        "plot_count": len(output_entries),
        "level_ranges": {str(level): stats for level, stats in sorted(level_ranges.items())},
        "plots": output_entries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
