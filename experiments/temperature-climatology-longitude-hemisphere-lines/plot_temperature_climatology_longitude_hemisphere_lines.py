from __future__ import annotations

import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr


EXPERIMENT_DIR = Path("tmp/temperature-climatology-longitude-hemisphere-lines")
PLOTS_DIR = EXPERIMENT_DIR / "even-5-longitudes"
PAIRED_PLOTS_DIR = EXPERIMENT_DIR / "even-5-longitudes-profile-raw-temperature-map"
RAW_TEMPERATURE_STACK_PATH = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
RAW_TEMPERATURE_VARIABLE = "t"
LEVELS_HPA = [250, 500, 850, 1000]
LONGITUDE_SAMPLE_COUNT = 5
LATITUDE_MARKERS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)


def display_path(path: Path) -> str:
    return path.as_posix()


def longitude_label(lon_deg: float) -> str:
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


def hemisphere_rows(latitudes: np.ndarray, hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        mask = latitudes >= 0.0
    elif hemisphere == "south":
        mask = latitudes < 0.0
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    rows = np.flatnonzero(mask)
    return rows[np.argsort(latitudes[rows], kind="stable")]


def hemisphere_extent(hemisphere: str) -> list[float]:
    if hemisphere == "north":
        return [-180.0, 180.0, 0.0, 90.0]
    if hemisphere == "south":
        return [-180.0, 180.0, -90.0, 0.0]
    raise ValueError(f"Unknown hemisphere: {hemisphere}")


def marker_latitudes(hemisphere: str) -> np.ndarray:
    if hemisphere == "north":
        return LATITUDE_MARKERS_ABS
    if hemisphere == "south":
        return -LATITUDE_MARKERS_ABS
    raise ValueError(f"Unknown hemisphere: {hemisphere}")


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    padding = max((upper - lower) * 0.08, 1.0)
    return lower - padding, upper + padding


def plot_profile_axis(
    ax,
    *,
    level_hpa: int,
    temperature_level: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    hemisphere: str,
    y_limits: tuple[float, float],
    colors,
) -> None:
    rows = hemisphere_rows(latitudes, hemisphere)

    for sample_number, lon_index in enumerate(longitude_indices):
        ax.plot(
            latitudes[rows],
            temperature_level[rows, lon_index],
            color=colors[sample_number % len(colors)],
            linewidth=2.0,
            alpha=0.95,
        )

    if hemisphere == "north":
        ax.set_xlim(0.0, float(np.nanmax(latitudes[rows])))
    else:
        ax.set_xlim(float(np.nanmin(latitudes[rows])), 0.0)

    ax.set_ylim(*y_limits)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Raw temperature climatology mean (K)")
    ax.set_title(f"{level_hpa} hPa profile, {hemisphere.title()} Hemisphere")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[sample_number % len(colors)],
            linewidth=2.6,
            label=longitude_label(float(longitudes[lon_index])),
        )
        for sample_number, lon_index in enumerate(longitude_indices)
    ]
    ax.legend(handles=longitude_handles, title="Longitude", loc="upper left", frameon=True)


def plot_temperature_map_axis(
    ax,
    *,
    level_hpa: int,
    temperature_level: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    hemisphere: str,
    color_limits: tuple[float, float],
    colors,
):
    transform = ccrs.PlateCarree()
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        temperature_level,
        cmap="coolwarm",
        vmin=color_limits[0],
        vmax=color_limits[1],
        shading="auto",
        transform=transform,
        zorder=1,
    )

    lon_min, lon_max, lat_min, lat_max = hemisphere_extent(hemisphere)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=transform)
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

    lats_for_markers = marker_latitudes(hemisphere)
    lats_for_markers = lats_for_markers[
        (lats_for_markers >= min(lat_min, lat_max))
        & (lats_for_markers <= max(lat_min, lat_max))
    ]

    for sample_number, lon_index in enumerate(longitude_indices):
        sampled_lon = float(longitudes[lon_index])
        color = colors[sample_number % len(colors)]
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
            np.full(lats_for_markers.shape, sampled_lon),
            lats_for_markers,
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


def plot_paired_hemisphere(
    *,
    level_hpa: int,
    temperature_level: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    hemisphere: str,
    y_limits: tuple[float, float],
    color_limits: tuple[float, float],
) -> dict:
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    fig = plt.figure(figsize=(16.4, 7.2))
    profile_ax = fig.add_subplot(1, 2, 1)
    map_ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{level_hpa} hPa temperature climatology profiles and source map, {hemisphere.title()} Hemisphere\n"
        "profile = raw temperature climatology along sampled meridians; map = raw temperature climatology with extraction meridians",
        fontsize=15,
    )

    plot_profile_axis(
        profile_ax,
        level_hpa=level_hpa,
        temperature_level=temperature_level,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        hemisphere=hemisphere,
        y_limits=y_limits,
        colors=colors,
    )
    mesh = plot_temperature_map_axis(
        map_ax,
        level_hpa=level_hpa,
        temperature_level=temperature_level,
        latitudes=latitudes,
        longitudes=longitudes,
        longitude_indices=longitude_indices,
        hemisphere=hemisphere,
        color_limits=color_limits,
        colors=colors,
    )
    colorbar = fig.colorbar(mesh, ax=map_ax, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Raw temperature climatology mean (K)")

    output_path = (
        PAIRED_PLOTS_DIR
        / f"temperature_climatology_profile_raw_temperature_map_{level_hpa:04d}hpa_{hemisphere}.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    rows = hemisphere_rows(latitudes, hemisphere)
    selected_values = temperature_level[rows[:, None], longitude_indices[None, :]]
    return {
        "pressure_level_hpa": level_hpa,
        "hemisphere": hemisphere,
        "plot": display_path(output_path),
        "temperature_min_k": float(np.nanmin(selected_values)),
        "temperature_max_k": float(np.nanmax(selected_values)),
        "map_color_min_k": color_limits[0],
        "map_color_max_k": color_limits[1],
    }


def plot_hemisphere(
    *,
    level_hpa: int,
    temperature_level: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    longitude_indices: np.ndarray,
    hemisphere: str,
    y_limits: tuple[float, float],
) -> dict:
    rows = hemisphere_rows(latitudes, hemisphere)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.subplots_adjust(right=0.83)

    for sample_number, lon_index in enumerate(longitude_indices):
        ax.plot(
            latitudes[rows],
            temperature_level[rows, lon_index],
            color=colors[sample_number % len(colors)],
            linewidth=2.1,
            alpha=0.95,
        )

    if hemisphere == "north":
        ax.set_xlim(0.0, float(np.nanmax(latitudes[rows])))
        ax.set_title(f"{level_hpa} hPa temperature climatology profiles: Northern Hemisphere")
    else:
        ax.set_xlim(float(np.nanmin(latitudes[rows])), 0.0)
        ax.set_title(f"{level_hpa} hPa temperature climatology profiles: Southern Hemisphere")

    ax.set_ylim(*y_limits)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Raw temperature climatology mean (K)")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    longitude_handles = [
        Line2D(
            [0],
            [0],
            color=colors[sample_number % len(colors)],
            linewidth=2.8,
            label=longitude_label(float(longitudes[lon_index])),
        )
        for sample_number, lon_index in enumerate(longitude_indices)
    ]
    ax.legend(
        handles=longitude_handles,
        title="Longitude",
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=True,
    )

    output_path = (
        PLOTS_DIR
        / f"temperature_climatology_longitude_lines_{level_hpa:04d}hpa_{hemisphere}.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    selected_values = temperature_level[rows[:, None], longitude_indices[None, :]]
    return {
        "pressure_level_hpa": level_hpa,
        "hemisphere": hemisphere,
        "plot": display_path(output_path),
        "temperature_min_k": float(np.nanmin(selected_values)),
        "temperature_max_k": float(np.nanmax(selected_values)),
    }


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAIRED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    paired_summaries = []
    with xr.open_dataset(RAW_TEMPERATURE_STACK_PATH) as dataset:
        raw_temperature = dataset[RAW_TEMPERATURE_VARIABLE]
        latitudes = np.asarray(raw_temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(raw_temperature.coords["longitude"].values, dtype=np.float32)
        valid_times = raw_temperature.coords["valid_time"].values
        longitude_indices = select_even_longitude_indices(longitudes, LONGITUDE_SAMPLE_COUNT)

        for level_hpa in LEVELS_HPA:
            temperature_level = (
                raw_temperature.sel(pressure_level=float(level_hpa))
                .mean(dim="valid_time")
                .load()
                .to_numpy()
                .astype(np.float32)
            )
            y_limits = padded_limits(temperature_level[:, longitude_indices])
            color_limits = padded_limits(temperature_level)
            for hemisphere in ("north", "south"):
                summaries.append(
                    plot_hemisphere(
                        level_hpa=level_hpa,
                        temperature_level=temperature_level,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        longitude_indices=longitude_indices,
                        hemisphere=hemisphere,
                        y_limits=y_limits,
                    )
                )
                paired_summaries.append(
                    plot_paired_hemisphere(
                        level_hpa=level_hpa,
                        temperature_level=temperature_level,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        longitude_indices=longitude_indices,
                        hemisphere=hemisphere,
                        y_limits=y_limits,
                        color_limits=color_limits,
                    )
                )

        summary = {
            "experiment": "temperature-climatology-longitude-hemisphere-lines",
            "method": "raw temperature climatology latitude profiles at evenly sampled global longitudes, computed as mean(t) over valid_time",
            "input": display_path(RAW_TEMPERATURE_STACK_PATH),
            "variable": RAW_TEMPERATURE_VARIABLE,
            "aggregation": "mean over valid_time",
            "valid_time_count": int(valid_times.size),
            "first_valid_time": str(np.datetime_as_string(valid_times[0], unit="s")),
            "last_valid_time": str(np.datetime_as_string(valid_times[-1], unit="s")),
            "levels_hpa": LEVELS_HPA,
            "longitude_sample_count": int(longitude_indices.size),
            "sampled_longitudes_degrees": [
                float(longitudes[index]) for index in longitude_indices
            ],
            "variant": "even-5-longitudes",
            "x_axis": "actual latitude in degrees, split by hemisphere",
            "y_axis": "mean raw t over valid_time in Kelvin",
            "line_only_plots": summaries,
            "paired_profile_raw_temperature_map_plots": paired_summaries,
            "map_encoding": "raw temperature climatology with coastlines, borders, extracted longitude meridians, and 15-degree latitude markers",
        }

    (EXPERIMENT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
