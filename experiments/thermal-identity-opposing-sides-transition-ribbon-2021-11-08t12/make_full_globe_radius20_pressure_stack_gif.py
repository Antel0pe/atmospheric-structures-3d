from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))
os.environ.setdefault("FONTCONFIG_PATH", str(CACHE_ROOT / "fontconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import xarray as xr

from scripts.plot_moist_thermal_displacement import draw_borders, load_border_segments
from scripts.thermal_displacement import compute_thermal_displacement_level


DATASET = REPO_ROOT / "data/era5_temperature_2021-11_08-12.nc"
CLIMATOLOGY = REPO_ROOT / "data/era5_temperature-climatology_1990-2020_11-08_12.nc"
BORDER_GEOJSON = (
    REPO_ROOT
    / "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12"
    / "preferred-no-bearing-angle-radius20-full-globe-pressure-stack"
)
SOURCE_MAP_DIR = OUTPUT_DIR / "source-level-maps"
TIMESTAMP = np.datetime64("2021-11-08T12:00")
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"
MIN_PRESSURE_HPA = 250.0
MAX_PRESSURE_HPA = 1000.0
SCORE_SIGMA_CELLS = 1.0
WARM_CORE_MAX_ABS_EQUIVALENT_LAT_DEG = 35.0
COLD_CORE_MIN_ABS_EQUIVALENT_LAT_DEG = 55.0
SEARCH_RADIUS_DEG = 20.0
GIF_WIDTH_PX = 1200
GIF_SOURCE_LEVEL_HOLD_MS = 500
GIF_TRANSITION_FRAME_COUNT = 4
GIF_TRANSITION_FRAME_MS = 100

CORE_MAP = mcolors.ListedColormap(["#2367b3", "#ececec", "#c9413a"])
CORE_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CORE_MAP.N)
RIBBON_COLOR = "#ffd92f"
RIBBON_EDGE = "#4a3a00"


def display_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def equivalent_abs_latitude_from_score(
    score: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    max_abs_latitude = float(np.nanmax(np.abs(latitudes)))
    return np.asarray((1.0 - score / 100.0) * max_abs_latitude, dtype=np.float32)


def unit_sphere_points(
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    latitude_rad = np.deg2rad(np.asarray(latitudes_deg, dtype=np.float64))
    longitude_rad = np.deg2rad(np.asarray(longitudes_deg, dtype=np.float64))
    cos_latitude = np.cos(latitude_rad)
    return np.column_stack(
        (
            cos_latitude * np.cos(longitude_rad),
            cos_latitude * np.sin(longitude_rad),
            np.sin(latitude_rad),
        )
    )


def angular_distance_deg(chord_distance: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(chord_distance, dtype=np.float64) / 2.0, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(clipped))


def classify_full_globe(
    equivalent_abs_latitude_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    warm_core = equivalent_abs_latitude_deg <= WARM_CORE_MAX_ABS_EQUIVALENT_LAT_DEG
    cold_core = equivalent_abs_latitude_deg >= COLD_CORE_MIN_ABS_EQUIVALENT_LAT_DEG
    intermediate = (~warm_core) & (~cold_core)
    longitude_grid, latitude_grid = np.meshgrid(longitudes_deg, latitudes_deg)
    sphere_points = unit_sphere_points(latitude_grid.ravel(), longitude_grid.ravel())
    sphere_points_grid = sphere_points.reshape((*equivalent_abs_latitude_deg.shape, 3))
    nearest_warm_distance_deg = np.full_like(
        equivalent_abs_latitude_deg,
        np.nan,
        dtype=np.float32,
    )
    nearest_cold_distance_deg = np.full_like(
        equivalent_abs_latitude_deg,
        np.nan,
        dtype=np.float32,
    )
    nearby_core_transition = np.zeros_like(intermediate, dtype=bool)

    for hemisphere_rows in (latitudes_deg >= 0.0, latitudes_deg < 0.0):
        candidate_mask = intermediate & hemisphere_rows[:, None]
        warm_mask = warm_core & hemisphere_rows[:, None]
        cold_mask = cold_core & hemisphere_rows[:, None]
        if not np.any(candidate_mask) or not np.any(warm_mask) or not np.any(cold_mask):
            continue
        candidate_points = sphere_points_grid[candidate_mask]
        warm_distance, _ = cKDTree(sphere_points_grid[warm_mask]).query(
            candidate_points,
            k=1,
        )
        cold_distance, _ = cKDTree(sphere_points_grid[cold_mask]).query(
            candidate_points,
            k=1,
        )
        warm_distance_deg = angular_distance_deg(warm_distance)
        cold_distance_deg = angular_distance_deg(cold_distance)
        nearest_warm_distance_deg[candidate_mask] = warm_distance_deg
        nearest_cold_distance_deg[candidate_mask] = cold_distance_deg
        nearby_core_transition[candidate_mask] = (
            (warm_distance_deg <= SEARCH_RADIUS_DEG)
            & (cold_distance_deg <= SEARCH_RADIUS_DEG)
        )

    classes = np.zeros_like(equivalent_abs_latitude_deg, dtype=np.float32)
    classes[cold_core] = -1.0
    classes[warm_core] = 1.0
    return {
        "classes": classes,
        "warm_core": warm_core,
        "cold_core": cold_core,
        "intermediate": intermediate,
        "nearby_core_transition": nearby_core_transition,
        "nearest_warm_distance_deg": nearest_warm_distance_deg,
        "nearest_cold_distance_deg": nearest_cold_distance_deg,
    }


def add_latitude_guides(ax: plt.Axes) -> None:
    for latitude in (-60.0, -30.0, 0.0, 30.0, 60.0):
        ax.axhline(
            latitude,
            color="#5d6470",
            linewidth=0.45,
            alpha=0.3,
            linestyle="--",
            zorder=2,
        )


def plot_level(
    *,
    level_hpa: float,
    result: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.9))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.145, top=0.89)
    ax.pcolormesh(
        longitudes,
        latitudes,
        result["classes"],
        cmap=CORE_MAP,
        norm=CORE_NORM,
        shading="auto",
        rasterized=True,
    )
    ribbon = result["nearby_core_transition"]
    ax.pcolormesh(
        longitudes,
        latitudes,
        np.where(ribbon, 1.0, np.nan),
        cmap=mcolors.ListedColormap([RIBBON_COLOR]),
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        alpha=0.9,
        rasterized=True,
        zorder=4,
    )
    ax.contour(
        longitudes,
        latitudes,
        ribbon.astype(np.float32),
        levels=[0.5],
        colors=[RIBBON_EDGE],
        linewidths=0.65,
        zorder=5,
    )
    add_latitude_guides(ax)
    draw_borders(ax, border_segments)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-89.0, 89.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_yticks(np.arange(-60.0, 61.0, 30.0))
    ax.set_title(f"{level_hpa:g} hPa full-globe nearby-core thermal regimes", fontsize=14)
    ax.legend(
        handles=[
            Patch(
                facecolor="#c9413a",
                label=(
                    "Warm core: equivalent latitude "
                    f"<= {WARM_CORE_MAX_ABS_EQUIVALENT_LAT_DEG:g} deg"
                ),
            ),
            Patch(
                facecolor=RIBBON_COLOR,
                edgecolor=RIBBON_EDGE,
                label=(
                    "Nearby-core transition: intermediate air with warm and cold "
                    f"cores each within {SEARCH_RADIUS_DEG:g} deg"
                ),
            ),
            Patch(
                facecolor="#ececec",
                label="Unresolved intermediate: nearby warm/cold-core evidence is insufficient",
            ),
            Patch(
                facecolor="#2367b3",
                label=(
                    "Cold core: equivalent latitude "
                    f">= {COLD_CORE_MIN_ABS_EQUIVALENT_LAT_DEG:g} deg"
                ),
            ),
        ],
        loc="lower left",
        framealpha=0.95,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.028,
        (
            "Full globe: yellow requires nearest warm and cold cores within "
            f"{SEARCH_RADIUS_DEG:g} deg great-circle distance. No bearing-angle gate."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def result_stats(result: dict[str, np.ndarray]) -> dict[str, float]:
    cell_count = max(int(result["classes"].size), 1)
    return {
        "warm_core_cell_fraction": float(np.count_nonzero(result["warm_core"]) / cell_count),
        "nearby_core_transition_cell_fraction": float(
            np.count_nonzero(result["nearby_core_transition"]) / cell_count
        ),
        "unresolved_intermediate_cell_fraction": float(
            np.count_nonzero(
                result["intermediate"] & (~result["nearby_core_transition"])
            )
            / cell_count
        ),
        "cold_core_cell_fraction": float(np.count_nonzero(result["cold_core"]) / cell_count),
    }


def resize_for_gif(image: Image.Image) -> Image.Image:
    if image.width == GIF_WIDTH_PX:
        return image
    height = round(image.height * GIF_WIDTH_PX / image.width)
    return image.resize((GIF_WIDTH_PX, height), Image.Resampling.LANCZOS)


def make_crossfade_gif(
    source_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    frames: list[Image.Image] = []
    durations_ms: list[int] = []
    previous: Image.Image | None = None
    for path in source_paths:
        current = resize_for_gif(Image.open(path).convert("RGB"))
        if previous is not None:
            for step in range(1, GIF_TRANSITION_FRAME_COUNT + 1):
                alpha = step / (GIF_TRANSITION_FRAME_COUNT + 1)
                frames.append(Image.blend(previous, current, alpha))
                durations_ms.append(GIF_TRANSITION_FRAME_MS)
        frames.append(current)
        durations_ms.append(GIF_SOURCE_LEVEL_HOLD_MS)
        previous = current
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
    )
    return {
        "source_level_frame_count": len(source_paths),
        "interpolated_transition_frame_count": len(frames) - len(source_paths),
        "total_frame_count": len(frames),
        "source_level_hold_ms": GIF_SOURCE_LEVEL_HOLD_MS,
        "transition_frame_count_between_source_levels": GIF_TRANSITION_FRAME_COUNT,
        "transition_frame_duration_ms": GIF_TRANSITION_FRAME_MS,
        "animation_duration_seconds": float(sum(durations_ms) / 1000.0),
        "interpolation": (
            "Visual crossfade between independently computed source-level maps. "
            "Blended frames are presentation interpolation, not inferred atmospheric levels."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(DATASET) as dataset, xr.open_dataset(CLIMATOLOGY) as climatology:
        temperature = dataset[TEMPERATURE_VARIABLE]
        reference = climatology[CLIMATOLOGY_VARIABLE]
        raw_levels = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
        climatology_levels = np.asarray(
            reference.coords["pressure_level"].values,
            dtype=np.float64,
        )
        levels = [
            float(level)
            for level in raw_levels
            if (
                MIN_PRESSURE_HPA <= float(level) <= MAX_PRESSURE_HPA
                and np.any(np.isclose(climatology_levels, float(level)))
            )
        ]
        levels = sorted(levels)
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(BORDER_GEOJSON, longitudes)
        level_rows: list[dict[str, object]] = []
        source_paths: list[Path] = []
        for level_hpa in levels:
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
                latitudes,
                score_smooth_sigma_cells=SCORE_SIGMA_CELLS,
                same_hemisphere=True,
            )
            equivalent_abs_latitude = equivalent_abs_latitude_from_score(
                displacement.score_points,
                latitudes,
            )
            result = classify_full_globe(
                equivalent_abs_latitude,
                latitudes,
                longitudes,
            )
            output_path = SOURCE_MAP_DIR / f"full_globe_nearby_core_{int(level_hpa):04d}hpa.png"
            plot_level(
                level_hpa=level_hpa,
                result=result,
                latitudes=latitudes,
                longitudes=longitudes,
                border_segments=border_segments,
                output_path=output_path,
            )
            source_paths.append(output_path)
            level_rows.append(
                {
                    "pressure_level_hpa": level_hpa,
                    "plot": display_path(output_path),
                    "stats": result_stats(result),
                }
            )

    gif_path = OUTPUT_DIR / "full_globe_nearby_core_pressure_stack_0250_to_1000hpa.gif"
    gif_summary = make_crossfade_gif(source_paths, gif_path)
    summary = {
        "variant": "preferred-no-bearing-angle-radius20-full-globe-pressure-stack",
        "date_run": "2026-06-01",
        "timestamp": "2021-11-08T12:00 UTC",
        "inputs": {
            "raw_temperature": display_path(DATASET),
            "temperature_climatology": display_path(CLIMATOLOGY),
        },
        "domain": "Full globe: all available latitude and longitude cells.",
        "levels_hpa": levels,
        "method": (
            "Canonical same-longitude, same-pressure, same-hemisphere Thermal Displacement "
            "converted to absolute matched climatological latitude. Warm core is <=35 degrees; "
            "cold core is >=55 degrees; yellow is intermediate identity with nearest warm and "
            "cold cores each within 20 great-circle degrees. No bearing-angle gate."
        ),
        "gif": display_path(gif_path),
        "gif_settings": gif_summary,
        "source_maps": level_rows,
        "intermediate_arrays_written": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
