from __future__ import annotations

import json
import os
from pathlib import Path

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


DATASET_PATH = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
EXPERIMENT_DIR = Path("tmp/raw-temperature-climatology-longitude-derivative")
SUMMARY_PATH = EXPERIMENT_DIR / "summary.json"
PRESSURE_LEVELS_HPA = [250, 500, 850, 1000]
LATITUDE_TICKS_ABS = np.asarray([90, 75, 60, 45, 30, 15, 0], dtype=np.float32)
SMOOTH_SIGMA_SAMPLES = 2.0
PIECEWISE_SMOOTH_SIGMA_SAMPLES = 4.0
PIECEWISE_MIN_SEGMENT_DEGREES = 12.0
PIECEWISE_MIN_SEGMENTS = 2
PIECEWISE_MAX_SEGMENTS = 4
PIECEWISE_PENALTY_WEIGHT = 1.25
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


def gaussian_smooth_1d(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    radius = max(1, int(np.ceil(sigma_samples * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(values, dtype=np.float32), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def segment_stats(x_values: np.ndarray, y_values: np.ndarray, start: int, end: int) -> tuple[float, float, float]:
    x = x_values[start:end].astype(np.float64)
    y = y_values[start:end].astype(np.float64)
    n = float(end - start)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xx = float(np.sum(x * x))
    sum_xy = float(np.sum(x * y))
    sum_yy = float(np.sum(y * y))
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1.0e-12:
        slope = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    sse = sum_yy + slope * slope * sum_xx + n * intercept * intercept + 2.0 * slope * intercept * sum_x - 2.0 * slope * sum_xy - 2.0 * intercept * sum_y
    return float(max(sse, 0.0)), float(slope), float(intercept)


def fit_piecewise_linear(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    min_segment_degrees: float,
    min_segments: int,
    max_segments: int,
    penalty_weight: float,
) -> dict:
    x = np.asarray(x_values, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < 8:
        raise ValueError("Need at least 8 samples for piecewise fitting.")

    median_step = float(np.nanmedian(np.abs(np.diff(x))))
    min_points = max(3, int(np.ceil(min_segment_degrees / median_step)))
    n = int(x.size)
    max_segments = min(max_segments, n // min_points)
    min_segments = min(min_segments, max_segments)
    if max_segments < 1:
        raise ValueError("Minimum segment width is too large for this profile.")

    segment_cache: dict[tuple[int, int], tuple[float, float, float]] = {}

    def cached_segment(start: int, end: int) -> tuple[float, float, float]:
        key = (start, end)
        if key not in segment_cache:
            segment_cache[key] = segment_stats(x, y, start, end)
        return segment_cache[key]

    candidates = []
    for segment_count in range(min_segments, max_segments + 1):
        dp = np.full((segment_count + 1, n + 1), np.inf, dtype=np.float64)
        prev = np.full((segment_count + 1, n + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0

        for used_segments in range(1, segment_count + 1):
            min_end = used_segments * min_points
            max_end = n - (segment_count - used_segments) * min_points
            for end in range(min_end, max_end + 1):
                start_min = (used_segments - 1) * min_points
                start_max = end - min_points
                best_value = np.inf
                best_start = -1
                for start in range(start_min, start_max + 1):
                    previous = dp[used_segments - 1, start]
                    if not np.isfinite(previous):
                        continue
                    sse, _, _ = cached_segment(start, end)
                    value = previous + sse
                    if value < best_value:
                        best_value = value
                        best_start = start
                dp[used_segments, end] = best_value
                prev[used_segments, end] = best_start

        if not np.isfinite(dp[segment_count, n]):
            continue

        boundaries = [n]
        end = n
        for used_segments in range(segment_count, 0, -1):
            start = int(prev[used_segments, end])
            boundaries.append(start)
            end = start
        boundaries = list(reversed(boundaries))

        segments = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            sse, slope, intercept = cached_segment(int(start), int(end))
            segments.append(
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "start_latitude_abs": float(x[int(start)]),
                    "end_latitude_abs": float(x[int(end - 1)]),
                    "slope_k_per_degree": float(slope),
                    "intercept_k": float(intercept),
                    "sse": float(sse),
                }
            )

        total_sse = float(dp[segment_count, n])
        parameter_count = 2 * segment_count + (segment_count - 1)
        score = n * np.log(total_sse / max(n, 1) + 1.0e-9) + penalty_weight * parameter_count * np.log(n)
        candidates.append({"segment_count": segment_count, "score": float(score), "sse": total_sse, "segments": segments})

    if not candidates:
        raise ValueError("No valid piecewise fit candidates were found.")
    return min(candidates, key=lambda candidate: candidate["score"])


def profile_series(
    *,
    climatology_k: np.ndarray,
    latitudes: np.ndarray,
    hemisphere: str,
    lon_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, x_axis = pole_to_equator_axis(latitudes, hemisphere)
    return x_axis, climatology_k[rows, lon_index]


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
    ax.legend(handles=longitude_legend_handles(longitudes, longitude_indices, colors), title="Longitude", loc="upper left", frameon=True)


def longitude_legend_handles(longitudes: np.ndarray, longitude_indices: np.ndarray, colors) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=2.6,
            label=longitude_label(float(longitudes[lon_index])),
        )
        for i, lon_index in enumerate(longitude_indices)
    ]


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


def write_original_plot(
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


def write_smoothed_derivative_plot(
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
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    title = f"{region['region']} {level_hpa} hPa climatology longitude window"
    temperature_range = stats["temperature_max_k"] - stats["temperature_min_k"]
    y_pad = max(1.0, 0.03 * temperature_range)
    y_limits = (stats["temperature_min_k"] - y_pad, stats["temperature_max_k"] + y_pad)
    smoothed_lines = []
    derivative_lines = []

    for lon_index in longitude_indices:
        x_axis, raw_values = profile_series(
            climatology_k=climatology_k,
            latitudes=latitudes,
            hemisphere=hemisphere,
            lon_index=int(lon_index),
        )
        smoothed = gaussian_smooth_1d(raw_values, SMOOTH_SIGMA_SAMPLES)
        derivative = np.gradient(smoothed, x_axis).astype(np.float32)
        smoothed_lines.append((x_axis, smoothed))
        derivative_lines.append((x_axis, derivative))

    all_derivatives = np.concatenate([derivative for _, derivative in derivative_lines])
    derivative_abs = float(np.nanmax(np.abs(all_derivatives)))
    derivative_limit = max(0.05, derivative_abs * 1.08)

    fig, (smooth_ax, derivative_ax) = plt.subplots(1, 2, figsize=(16.4, 7.2))
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{title}, {hemisphere.title()} Hemisphere\n"
        f"left = lightly smoothed profile, sigma {SMOOTH_SIGMA_SAMPLES:g} grid samples; right = first derivative of smoothed profile",
        fontsize=15,
    )

    for sample_number, ((x_axis, smoothed), (_, derivative)) in enumerate(zip(smoothed_lines, derivative_lines, strict=True)):
        smooth_ax.plot(
            x_axis,
            smoothed,
            color=colors[sample_number],
            linewidth=2.0,
            alpha=0.95,
        )
        derivative_ax.plot(
            x_axis,
            derivative,
            color=colors[sample_number],
            linewidth=2.0,
            alpha=0.95,
        )

    for ax in (smooth_ax, derivative_ax):
        ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
        ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
        ax.set_xlabel("Latitude path, pole to equator")
        ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    smooth_ax.set_ylim(*y_limits)
    smooth_ax.set_ylabel("Smoothed raw temperature climatology (K)")
    smooth_ax.set_title(f"{title}, {hemisphere.title()} Hemisphere smoothed profile")
    smooth_ax.legend(handles=longitude_legend_handles(longitudes, longitude_indices, colors), title="Longitude", loc="upper left", frameon=True)

    derivative_ax.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.75)
    derivative_ax.set_ylim(-derivative_limit, derivative_limit)
    derivative_ax.set_ylabel("First derivative (K per degree latitude path)")
    derivative_ax.set_title("Derivative of smoothed profile")
    derivative_ax.legend(handles=longitude_legend_handles(longitudes, longitude_indices, colors), title="Longitude", loc="upper left", frameon=True)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_piecewise_plot(
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
) -> list[dict]:
    colors = plt.get_cmap("tab10").colors[: len(longitude_indices)]
    title = f"{region['region']} {level_hpa} hPa climatology longitude window"
    temperature_range = stats["temperature_max_k"] - stats["temperature_min_k"]
    y_pad = max(1.0, 0.03 * temperature_range)
    y_limits = (stats["temperature_min_k"] - y_pad, stats["temperature_max_k"] + y_pad)
    fits = []

    for lon_index in longitude_indices:
        x_axis, raw_values = profile_series(
            climatology_k=climatology_k,
            latitudes=latitudes,
            hemisphere=hemisphere,
            lon_index=int(lon_index),
        )
        smoothed = gaussian_smooth_1d(raw_values, PIECEWISE_SMOOTH_SIGMA_SAMPLES)
        fit = fit_piecewise_linear(
            x_axis,
            smoothed,
            min_segment_degrees=PIECEWISE_MIN_SEGMENT_DEGREES,
            min_segments=PIECEWISE_MIN_SEGMENTS,
            max_segments=PIECEWISE_MAX_SEGMENTS,
            penalty_weight=PIECEWISE_PENALTY_WEIGHT,
        )
        fits.append({"x_axis": x_axis, "smoothed": smoothed, "fit": fit, "lon_index": int(lon_index)})

    fig, (smooth_ax, piecewise_ax) = plt.subplots(1, 2, figsize=(16.4, 7.2))
    fig.subplots_adjust(left=0.055, right=0.95, bottom=0.1, top=0.86, wspace=0.18)
    fig.suptitle(
        f"{title}, {hemisphere.title()} Hemisphere\n"
        "left = smoothed profile; right = independent per-longitude piecewise linear slope regimes with breakpoints",
        fontsize=15,
    )

    for sample_number, fit_entry in enumerate(fits):
        color = colors[sample_number]
        x_axis = fit_entry["x_axis"]
        smoothed = fit_entry["smoothed"]
        fit = fit_entry["fit"]

        smooth_ax.plot(x_axis, smoothed, color=color, linewidth=2.0, alpha=0.95)
        piecewise_ax.plot(x_axis, smoothed, color=color, linewidth=1.0, alpha=0.24)

        for segment in fit["segments"]:
            start = int(segment["start_index"])
            end = int(segment["end_index"])
            x_segment = x_axis[start:end]
            y_fit = segment["slope_k_per_degree"] * x_segment + segment["intercept_k"]
            piecewise_ax.plot(x_segment, y_fit, color=color, linewidth=3.0, alpha=0.98)

        for segment in fit["segments"][:-1]:
            breakpoint_index = int(segment["end_index"])
            breakpoint_x = float(x_axis[breakpoint_index])
            breakpoint_y = float(smoothed[breakpoint_index])
            piecewise_ax.scatter(
                [breakpoint_x],
                [breakpoint_y],
                s=48,
                color=color,
                edgecolors="#111111",
                linewidths=0.7,
                zorder=6,
            )
            piecewise_ax.axvline(breakpoint_x, color=color, linewidth=0.8, alpha=0.18)

    for ax in (smooth_ax, piecewise_ax):
        ax.set_xlim(float(np.nanmax(np.abs(latitudes))), 0.0)
        ax.set_ylim(*y_limits)
        ax.set_xticks(LATITUDE_TICKS_ABS.tolist())
        ax.set_xlabel("Latitude path, pole to equator")
        ax.set_ylabel("Raw temperature climatology (K)")
        ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    smooth_ax.set_title(f"{title}, {hemisphere.title()} Hemisphere smoothed profile")
    smooth_ax.legend(handles=longitude_legend_handles(longitudes, longitude_indices, colors), title="Longitude", loc="upper left", frameon=True)
    piecewise_ax.set_title("Piecewise linear slope regimes")
    piecewise_ax.legend(handles=longitude_legend_handles(longitudes, longitude_indices, colors), title="Longitude", loc="upper left", frameon=True)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    metadata = []
    for fit_entry in fits:
        lon = float(longitudes[int(fit_entry["lon_index"])])
        metadata.append(
            {
                "longitude_degrees": lon,
                "longitude_label": longitude_label(lon),
                "segment_count": int(fit_entry["fit"]["segment_count"]),
                "score": float(fit_entry["fit"]["score"]),
                "sse": float(fit_entry["fit"]["sse"]),
                "segments": fit_entry["fit"]["segments"],
                "breakpoint_latitudes_abs": [
                    float(segment["end_latitude_abs"]) for segment in fit_entry["fit"]["segments"][:-1]
                ],
            }
        )
    return metadata


def clean_output_dirs() -> None:
    for level_hpa in PRESSURE_LEVELS_HPA:
        for variant in ("original", "smoothed-derivative", "piecewise-segments"):
            output_dir = EXPERIMENT_DIR / f"{level_hpa:04d}hpa" / variant
            output_dir.mkdir(parents=True, exist_ok=True)
            for png in output_dir.glob("*.png"):
                png.unlink()


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    clean_output_dirs()

    climatology_by_level, latitudes, longitudes, valid_time_count = load_climatology_fields()
    level_ranges = {level: level_stats(field) for level, field in climatology_by_level.items()}
    output_entries = []

    for level_hpa in PRESSURE_LEVELS_HPA:
        climatology_k = climatology_by_level[level_hpa]
        level_dir = EXPERIMENT_DIR / f"{level_hpa:04d}hpa"
        for region in REGIONAL_WINDOWS:
            longitude_indices = nearest_longitude_indices(longitudes, region["longitudes"])
            matched_longitudes = [float(longitudes[index]) for index in longitude_indices]

            for hemisphere in ("north", "south"):
                base_name = f"raw_temperature_climatology_{level_hpa:04d}hpa_{region['slug']}_longitude_window_{hemisphere}"
                original_path = level_dir / "original" / f"{base_name}.png"
                derivative_path = level_dir / "smoothed-derivative" / f"{base_name}_smoothed_profile_derivative.png"
                piecewise_path = level_dir / "piecewise-segments" / f"{base_name}_piecewise_segments.png"

                print(f"plotting {level_hpa} hPa {region['region']} {hemisphere}", flush=True)
                write_original_plot(
                    level_hpa=level_hpa,
                    region=region,
                    hemisphere=hemisphere,
                    climatology_k=climatology_k,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    longitude_indices=longitude_indices,
                    output_path=original_path,
                    stats=level_ranges[level_hpa],
                )
                write_smoothed_derivative_plot(
                    level_hpa=level_hpa,
                    region=region,
                    hemisphere=hemisphere,
                    climatology_k=climatology_k,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    longitude_indices=longitude_indices,
                    output_path=derivative_path,
                    stats=level_ranges[level_hpa],
                )
                piecewise_metadata = write_piecewise_plot(
                    level_hpa=level_hpa,
                    region=region,
                    hemisphere=hemisphere,
                    climatology_k=climatology_k,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    longitude_indices=longitude_indices,
                    output_path=piecewise_path,
                    stats=level_ranges[level_hpa],
                )

                output_entries.append(
                    {
                        "region": region["region"],
                        "hemisphere": hemisphere,
                        "pressure_level_hpa": level_hpa,
                        "original_plot": repo_relative(original_path),
                        "smoothed_derivative_plot": repo_relative(derivative_path),
                        "piecewise_segments_plot": repo_relative(piecewise_path),
                        "window_degrees": region["window"],
                        "sampled_longitudes_degrees": matched_longitudes,
                        "sampled_longitude_labels": [longitude_label(lon) for lon in matched_longitudes],
                        "map_extent": map_extent_for_region(region, hemisphere),
                        "latitude_marker_degrees": marker_latitudes_for_hemisphere(hemisphere).tolist(),
                        "map_field": "raw_temperature_climatology_k",
                        "temperature_unit": "K",
                        "color_scale": "per pressure level climatology min/max",
                        "smoothing": {
                            "method": "1D gaussian along pole-to-equator profile",
                            "sigma_grid_samples": SMOOTH_SIGMA_SAMPLES,
                            "edge_mode": "edge padding",
                        },
                        "derivative": "numpy gradient of smoothed profile against absolute latitude path degrees",
                        "piecewise_segments": piecewise_metadata,
                        **level_ranges[level_hpa],
                    }
                )

    summary = {
        "experiment": "raw-temperature-climatology-longitude-derivative",
        "date": "2026-05-29",
        "source_example": "experiments/raw-temperature-climatology-maps/plots-regional-windows",
        "source_dataset": repo_relative(DATASET_PATH),
        "variable": "t",
        "method": "mean raw pressure-level temperature across valid_time entries",
        "valid_time_count": valid_time_count,
        "levels_hpa": PRESSURE_LEVELS_HPA,
        "regions": [region["region"] for region in REGIONAL_WINDOWS],
        "hemispheres": ["north", "south"],
        "outputs_per_level": {
            str(level): {
                "original": repo_relative(EXPERIMENT_DIR / f"{level:04d}hpa" / "original"),
                "smoothed_derivative": repo_relative(EXPERIMENT_DIR / f"{level:04d}hpa" / "smoothed-derivative"),
                "piecewise_segments": repo_relative(EXPERIMENT_DIR / f"{level:04d}hpa" / "piecewise-segments"),
            }
            for level in PRESSURE_LEVELS_HPA
        },
        "original_plot_count": len(output_entries),
        "smoothed_derivative_plot_count": len(output_entries),
        "piecewise_segments_plot_count": len(output_entries),
        "line_count_per_plot": 5,
        "x_axis": "absolute latitude, ordered from pole to equator for the selected hemisphere",
        "line_encoding": "color is longitude; smoothed profile and derivative use matching colors",
        "smoothing": {
            "method": "1D gaussian along pole-to-equator profile",
            "sigma_grid_samples": SMOOTH_SIGMA_SAMPLES,
            "edge_mode": "edge padding",
        },
        "derivative": "numpy gradient of smoothed profile against absolute latitude path degrees",
        "piecewise_segments": {
            "method": "independent per-longitude dynamic-programming piecewise linear fit",
            "fit_input": "smoothed raw temperature profile",
            "smoothing_sigma_grid_samples": PIECEWISE_SMOOTH_SIGMA_SAMPLES,
            "minimum_segment_width_degrees": PIECEWISE_MIN_SEGMENT_DEGREES,
            "minimum_segments": PIECEWISE_MIN_SEGMENTS,
            "maximum_segments": PIECEWISE_MAX_SEGMENTS,
            "selection": "penalized residual error, with extra-breakpoint penalty",
            "penalty_weight": PIECEWISE_PENALTY_WEIGHT,
        },
        "script": repo_relative(EXPERIMENT_DIR / "plot_raw_temperature_climatology_longitude_derivatives.py"),
        "plot_count_total": len(output_entries) * 3,
        "level_ranges": {str(level): stats for level, stats in sorted(level_ranges.items())},
        "plots": output_entries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
