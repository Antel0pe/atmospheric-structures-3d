from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
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
OUTPUT_DIR = REPO_ROOT / "tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12"
TIMESTAMP = np.datetime64("2021-11-08T12:00")
LEVELS_HPA = (250.0, 500.0, 850.0, 1000.0)
SCORE_SIGMA_CELLS = 1.0
RENDERED_ABS_LATITUDE_BAND_DEG = (20.0, 70.0)
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"

CORE_MAP = mcolors.ListedColormap(["#2367b3", "#ececec", "#c9413a"])
CORE_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CORE_MAP.N)
RIBBON_COLOR = "#ffd92f"
RIBBON_EDGE = "#4a3a00"


@dataclass(frozen=True)
class Variant:
    slug: str
    title: str
    warm_core_max_abs_equivalent_lat_deg: float
    cold_core_min_abs_equivalent_lat_deg: float
    search_radius_deg: float
    minimum_opposition_angle_deg: float | None


VARIANTS = (
    Variant(
        slug="primary-nearby-opposing-cores",
        title="Primary nearby opposing-core test",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=12.0,
        minimum_opposition_angle_deg=120.0,
    ),
    Variant(
        slug="sensitivity-broader-search-radius",
        title="Sensitivity: broader search radius",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=18.0,
        minimum_opposition_angle_deg=120.0,
    ),
    Variant(
        slug="sensitivity-stricter-opposition-angle",
        title="Sensitivity: stricter opposing-side angle",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=12.0,
        minimum_opposition_angle_deg=150.0,
    ),
    Variant(
        slug="sensitivity-no-bearing-angle",
        title="Sensitivity: nearby cores only, no bearing-angle gate",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=12.0,
        minimum_opposition_angle_deg=None,
    ),
    Variant(
        slug="sensitivity-no-bearing-angle-radius15",
        title="Sensitivity: nearby cores only, 15 deg radius",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=15.0,
        minimum_opposition_angle_deg=None,
    ),
    Variant(
        slug="sensitivity-no-bearing-angle-radius20",
        title="Sensitivity: nearby cores only, 20 deg radius",
        warm_core_max_abs_equivalent_lat_deg=35.0,
        cold_core_min_abs_equivalent_lat_deg=55.0,
        search_radius_deg=20.0,
        minimum_opposition_angle_deg=None,
    ),
)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().relative_to(REPO_ROOT).as_posix()


def add_latitude_guides(ax: plt.Axes) -> None:
    for latitude in (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0):
        ax.axhline(
            latitude,
            color="#5d6470",
            linewidth=0.45,
            alpha=0.3,
            linestyle="--",
            zorder=2,
        )


def format_map(ax: plt.Axes, border_segments: list[list[tuple[float, float]]]) -> None:
    add_latitude_guides(ax)
    draw_borders(ax, border_segments)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-89.0, 89.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_yticks(np.arange(-60.0, 61.0, 30.0))


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


def opposition_angle_deg(
    target_points: np.ndarray,
    warm_points: np.ndarray,
    cold_points: np.ndarray,
) -> np.ndarray:
    warm_tangent = warm_points - np.sum(warm_points * target_points, axis=1)[:, None] * target_points
    cold_tangent = cold_points - np.sum(cold_points * target_points, axis=1)[:, None] * target_points
    warm_norm = np.linalg.norm(warm_tangent, axis=1)
    cold_norm = np.linalg.norm(cold_tangent, axis=1)
    valid = (warm_norm > 1.0e-12) & (cold_norm > 1.0e-12)
    cosine = np.ones(target_points.shape[0], dtype=np.float64)
    cosine[valid] = np.sum(
        warm_tangent[valid] * cold_tangent[valid],
        axis=1,
    ) / (warm_norm[valid] * cold_norm[valid])
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def opposing_sides_transition_ribbon(
    equivalent_abs_latitude_deg: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    *,
    warm_core_max_deg: float,
    cold_core_min_deg: float,
    search_radius_deg: float,
    minimum_opposition_angle_deg: float | None,
) -> dict[str, np.ndarray]:
    abs_actual_lat = np.abs(latitudes_deg)
    same_band = (
        (abs_actual_lat >= RENDERED_ABS_LATITUDE_BAND_DEG[0])
        & (abs_actual_lat <= RENDERED_ABS_LATITUDE_BAND_DEG[1])
    )
    warm_core = equivalent_abs_latitude_deg <= warm_core_max_deg
    cold_core = equivalent_abs_latitude_deg >= cold_core_min_deg
    intermediate = (~warm_core) & (~cold_core)

    longitude_grid, latitude_grid = np.meshgrid(longitudes_deg, latitudes_deg)
    sphere_points = unit_sphere_points(latitude_grid.ravel(), longitude_grid.ravel())
    sphere_points_grid = sphere_points.reshape((*equivalent_abs_latitude_deg.shape, 3))
    nearest_warm_distance_deg = np.full_like(equivalent_abs_latitude_deg, np.nan, dtype=np.float32)
    nearest_cold_distance_deg = np.full_like(equivalent_abs_latitude_deg, np.nan, dtype=np.float32)
    nearest_core_opposition_angle_deg = np.full_like(
        equivalent_abs_latitude_deg,
        np.nan,
        dtype=np.float32,
    )
    ribbon = np.zeros_like(intermediate, dtype=bool)

    for hemisphere_rows in (latitudes_deg >= 0.0, latitudes_deg < 0.0):
        candidate_mask = intermediate & same_band[:, None] & hemisphere_rows[:, None]
        warm_mask = warm_core & hemisphere_rows[:, None]
        cold_mask = cold_core & hemisphere_rows[:, None]
        if not np.any(candidate_mask) or not np.any(warm_mask) or not np.any(cold_mask):
            continue

        candidate_points = sphere_points_grid[candidate_mask]
        warm_points = sphere_points_grid[warm_mask]
        cold_points = sphere_points_grid[cold_mask]
        warm_chord_distance, warm_index = cKDTree(warm_points).query(candidate_points, k=1)
        cold_chord_distance, cold_index = cKDTree(cold_points).query(candidate_points, k=1)
        warm_distance_deg = angular_distance_deg(warm_chord_distance)
        cold_distance_deg = angular_distance_deg(cold_chord_distance)
        opposition_angle = opposition_angle_deg(
            candidate_points,
            warm_points[warm_index],
            cold_points[cold_index],
        )
        validated = (warm_distance_deg <= search_radius_deg) & (
            cold_distance_deg <= search_radius_deg
        )
        if minimum_opposition_angle_deg is not None:
            validated &= opposition_angle >= minimum_opposition_angle_deg
        nearest_warm_distance_deg[candidate_mask] = warm_distance_deg
        nearest_cold_distance_deg[candidate_mask] = cold_distance_deg
        nearest_core_opposition_angle_deg[candidate_mask] = opposition_angle
        ribbon[candidate_mask] = validated

    classes = np.full_like(equivalent_abs_latitude_deg, np.nan, dtype=np.float32)
    classes[same_band[:, None] & cold_core] = -1.0
    classes[same_band[:, None] & intermediate] = 0.0
    classes[same_band[:, None] & warm_core] = 1.0
    return {
        "classes": classes,
        "warm_core": warm_core & same_band[:, None],
        "cold_core": cold_core & same_band[:, None],
        "intermediate": intermediate & same_band[:, None],
        "nearest_warm_distance_deg": nearest_warm_distance_deg,
        "nearest_cold_distance_deg": nearest_cold_distance_deg,
        "nearest_core_opposition_angle_deg": nearest_core_opposition_angle_deg,
        "ribbon": ribbon,
    }


def ribbon_stats(result: dict[str, np.ndarray], latitudes: np.ndarray) -> dict[str, object]:
    ribbon = result["ribbon"]
    spacing_deg = float(np.nanmedian(np.abs(np.diff(latitudes))))
    abs_lat = np.abs(latitudes)
    rows_in_band = (
        (abs_lat >= RENDERED_ABS_LATITUDE_BAND_DEG[0])
        & (abs_lat <= RENDERED_ABS_LATITUDE_BAND_DEG[1])
    )
    stats: dict[str, object] = {
        "rendered_band_cell_fraction": float(
            ribbon[rows_in_band].sum() / max(int(rows_in_band.sum() * ribbon.shape[1]), 1)
        ),
    }
    if np.any(ribbon):
        stats["validated_cell_medians"] = {
            "nearest_warm_core_distance_degrees": float(
                np.nanmedian(result["nearest_warm_distance_deg"][ribbon])
            ),
            "nearest_cold_core_distance_degrees": float(
                np.nanmedian(result["nearest_cold_distance_deg"][ribbon])
            ),
            "warm_cold_bearing_opposition_angle_degrees": float(
                np.nanmedian(result["nearest_core_opposition_angle_deg"][ribbon])
            ),
        }
    for hemisphere, row_mask in (
        ("north", rows_in_band & (latitudes >= 0.0)),
        ("south", rows_in_band & (latitudes < 0.0)),
    ):
        widths_deg = ribbon[row_mask].sum(axis=0).astype(np.float32) * spacing_deg
        active = widths_deg > 0.0
        stats[hemisphere] = {
            "longitude_coverage_fraction": float(np.mean(active)),
            "median_ribbon_width_deg_when_present": (
                float(np.median(widths_deg[active])) if np.any(active) else 0.0
            ),
            "p90_ribbon_width_deg_when_present": (
                float(np.percentile(widths_deg[active], 90.0)) if np.any(active) else 0.0
            ),
        }
    return stats


def plot_level(
    *,
    level_hpa: float,
    variant: Variant,
    result: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    if variant.minimum_opposition_angle_deg is None:
        yellow_label = "Nearby-core transition: intermediate air with both cores in range"
        gray_label = "Unresolved intermediate: nearby warm/cold-core evidence is insufficient"
        rule_text = (
            f"{variant.title}: yellow requires nearest warm and cold cores within "
            f"{variant.search_radius_deg:g} deg. No bearing-angle gate is applied."
        )
    else:
        yellow_label = "Validated transition: intermediate air between nearby opposing cores"
        gray_label = "Unresolved intermediate: local opposing-core evidence is insufficient"
        rule_text = (
            f"{variant.title}: yellow requires nearest warm and cold cores within "
            f"{variant.search_radius_deg:g} deg and their bearings from the cell to differ "
            f"by at least {variant.minimum_opposition_angle_deg:g} deg."
        )
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
    ribbon_overlay = np.where(result["ribbon"], 1.0, np.nan)
    ax.pcolormesh(
        longitudes,
        latitudes,
        ribbon_overlay,
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
        result["ribbon"].astype(np.float32),
        levels=[0.5],
        colors=[RIBBON_EDGE],
        linewidths=0.65,
        zorder=5,
    )
    format_map(ax, border_segments)
    ax.set_title(
        f"{level_hpa:g} hPa orientation-independent warm / transition / cold regimes",
        fontsize=14,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#c9413a", label=f"Warm core: equivalent latitude <= {variant.warm_core_max_abs_equivalent_lat_deg:g} deg"),
            Patch(facecolor=RIBBON_COLOR, edgecolor=RIBBON_EDGE, label=yellow_label),
            Patch(facecolor="#ececec", label=gray_label),
            Patch(facecolor="#2367b3", label=f"Cold core: equivalent latitude >= {variant.cold_core_min_abs_equivalent_lat_deg:g} deg"),
        ],
        loc="lower left",
        framealpha=0.95,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.028,
        rule_text,
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all_level_overlay(
    *,
    variant: Variant,
    ribbons_by_level: dict[float, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    colors = {
        1000.0: "#6a3d9a",
        850.0: "#1f78b4",
        500.0: "#33a02c",
        250.0: "#e31a1c",
    }
    fig, ax = plt.subplots(figsize=(15.5, 7.9))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92)
    ax.set_facecolor("#f5f5f5")
    for level_hpa in reversed(LEVELS_HPA):
        ax.contour(
            longitudes,
            latitudes,
            ribbons_by_level[level_hpa].astype(np.float32),
            levels=[0.5],
            colors=[colors[level_hpa]],
            linewidths=1.25,
            alpha=0.9,
            zorder=5,
        )
    format_map(ax, border_segments)
    ax.legend(
        handles=[
            Line2D([0], [0], color=colors[level], lw=2.0, label=f"{level:g} hPa")
            for level in reversed(LEVELS_HPA)
        ],
        loc="lower left",
        framealpha=0.95,
        title="Validated ribbon edges",
    )
    if variant.minimum_opposition_angle_deg is None:
        title = "Nearby-core transition-ribbon overlay: level-to-level displacement is the tilt signal"
        footer = (
            "Each outline uses the orientation-independent nearby-core rule without "
            "a bearing-angle gate. Curved and closed interfaces remain visible."
        )
    else:
        title = "Opposing-sides transition-ribbon overlay: level-to-level displacement is the tilt signal"
        footer = (
            "Each outline uses the same orientation-independent nearby opposing-core "
            "rule. Curved and closed interfaces remain visible."
        )
    ax.set_title(title, fontsize=13)
    fig.text(
        0.5,
        0.025,
        footer,
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    with xr.open_dataset(DATASET) as dataset, xr.open_dataset(CLIMATOLOGY) as climatology:
        temperature = dataset[TEMPERATURE_VARIABLE]
        reference = climatology[CLIMATOLOGY_VARIABLE]
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        border_segments = load_border_segments(BORDER_GEOJSON, longitudes)
        equivalent_abs_latitude_by_level: dict[float, np.ndarray] = {}

        for level_hpa in LEVELS_HPA:
            raw_level = np.asarray(
                temperature.sel(valid_time=TIMESTAMP, pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            climatology_level = np.asarray(
                reference.sel(pressure_level=level_hpa).values,
                dtype=np.float32,
            )
            displacement = compute_thermal_displacement_level(
                raw_level,
                climatology_level,
                latitudes,
                score_smooth_sigma_cells=SCORE_SIGMA_CELLS,
                same_hemisphere=True,
            )
            equivalent_abs_latitude_by_level[level_hpa] = equivalent_abs_latitude_from_score(
                displacement.score_points,
                latitudes,
            )

    summary: dict[str, object] = {
        "experiment": "thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12",
        "date_run": "2026-05-31",
        "timestamp": "2021-11-08T12:00 UTC",
        "inputs": {
            "raw_temperature": display_path(DATASET),
            "temperature_climatology": display_path(CLIMATOLOGY),
        },
        "identity_field": (
            "Canonical same-longitude, same-pressure, same-hemisphere Thermal "
            "Displacement converted back to absolute matched climatological latitude."
        ),
        "rule": (
            "Warm cores resemble a conservative low-latitude climatological source band. "
            "Cold cores resemble a conservative high-latitude climatological source band. "
            "A validated transition cell has intermediate identity, a nearby nearest warm core, "
            "a nearby nearest cold core, and substantially opposing bearings toward those cores. "
            "The test does not assume that warm air is equatorward or cold air is poleward. "
            "One sensitivity variant removes the bearing-angle gate and requires nearby cores only."
        ),
        "levels_hpa": list(LEVELS_HPA),
        "rendered_absolute_latitude_band_degrees": list(RENDERED_ABS_LATITUDE_BAND_DEG),
        "intermediate_arrays_written": False,
        "variants": [],
    }

    for variant in VARIANTS:
        variant_dir = OUTPUT_DIR / variant.slug
        variant_dir.mkdir(parents=True, exist_ok=True)
        ribbons_by_level: dict[float, np.ndarray] = {}
        level_rows: list[dict[str, object]] = []
        for level_hpa in LEVELS_HPA:
            result = opposing_sides_transition_ribbon(
                equivalent_abs_latitude_by_level[level_hpa],
                latitudes,
                longitudes,
                warm_core_max_deg=variant.warm_core_max_abs_equivalent_lat_deg,
                cold_core_min_deg=variant.cold_core_min_abs_equivalent_lat_deg,
                search_radius_deg=variant.search_radius_deg,
                minimum_opposition_angle_deg=variant.minimum_opposition_angle_deg,
            )
            ribbons_by_level[level_hpa] = result["ribbon"]
            plot_path = variant_dir / f"opposing_sides_transition_ribbon_{int(level_hpa):04d}hpa.png"
            plot_level(
                level_hpa=level_hpa,
                variant=variant,
                result=result,
                latitudes=latitudes,
                longitudes=longitudes,
                border_segments=border_segments,
                output_path=plot_path,
            )
            level_rows.append(
                {
                    "pressure_level_hpa": level_hpa,
                    "plot": display_path(plot_path),
                    "ribbon_stats": ribbon_stats(result, latitudes),
                }
            )

        overlay_path = variant_dir / "opposing_sides_transition_ribbon_all_level_overlay.png"
        plot_all_level_overlay(
            variant=variant,
            ribbons_by_level=ribbons_by_level,
            latitudes=latitudes,
            longitudes=longitudes,
            border_segments=border_segments,
            output_path=overlay_path,
        )
        summary["variants"].append(
            {
                "slug": variant.slug,
                "title": variant.title,
                "warm_core_max_abs_equivalent_latitude_degrees": (
                    variant.warm_core_max_abs_equivalent_lat_deg
                ),
                "cold_core_min_abs_equivalent_latitude_degrees": (
                    variant.cold_core_min_abs_equivalent_lat_deg
                ),
                "search_radius_degrees": variant.search_radius_deg,
                "minimum_opposition_angle_degrees": variant.minimum_opposition_angle_deg,
                "levels": level_rows,
                "all_level_overlay": display_path(overlay_path),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
