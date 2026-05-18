from __future__ import annotations

import argparse
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

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, label, percentile_filter


TEMPERATURE_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
CLIMATOLOGY_PATH = Path("data/era5_temperature-climatology_1990-2020_11-08_12.nc")
OUTPUT_DIR = Path("experiments/fuzzy-local-thermal-interface/output")
TIMESTAMP = np.datetime64("2021-11-08T12:00:00")
KEY_LEVELS = [1000.0, 850.0, 500.0, 250.0]
TEMPERATURE_VARIABLE = "t"
CLIMATOLOGY_VARIABLE = "temperature_climatology_mean"


@dataclass(frozen=True)
class ClusterAnchors:
    cold_center: float
    warm_center: float
    midpoint: float
    softness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find fuzzy local warm/cold thermal-identity interfaces from Thermal "
            "Displacement without assuming one monotonic north-south boundary."
        )
    )
    parser.add_argument("--temperature", type=Path, default=TEMPERATURE_PATH)
    parser.add_argument("--climatology", type=Path, default=CLIMATOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--levels", type=str, default=",".join(str(int(v)) for v in KEY_LEVELS))
    parser.add_argument(
        "--window-deg",
        type=float,
        default=16.0,
        help="Full local percentile window width in degrees on the native grid.",
    )
    parser.add_argument("--low-percentile", type=float, default=15.0)
    parser.add_argument("--high-percentile", type=float, default=85.0)
    parser.add_argument(
        "--min-local-range",
        type=float,
        default=12.0,
        help="Thermal Displacement score range where local cold/warm co-presence starts contributing.",
    )
    parser.add_argument(
        "--full-local-range",
        type=float,
        default=40.0,
        help="Thermal Displacement score range where local cold/warm co-presence reaches full strength.",
    )
    parser.add_argument(
        "--transition-width",
        type=float,
        default=0.32,
        help="Fuzzy membership half-width around the local 50/50 transition. Larger keeps broader slow boundaries.",
    )
    parser.add_argument(
        "--sharpness-floor",
        type=float,
        default=0.35,
        help="Minimum multiplier before adding gradient sharpness. Higher values favor broad slow transitions.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(path.as_posix())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home()).as_posix()}"
    except ValueError:
        pass
    if resolved.is_relative_to(Path("/tmp")):
        return resolved.as_posix()
    return resolved.name


def parse_levels(text: str) -> list[float]:
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def match_equivalent_latitude_same_longitude(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_temperature_k, dtype=np.float32)
    climatology = np.asarray(climatology_temperature_k, dtype=np.float32)
    latitudes = np.asarray(latitudes_deg, dtype=np.float32)
    n_lat, n_lon = raw.shape
    matched = np.empty((n_lat, n_lon), dtype=np.float32)

    for lon_index in range(n_lon):
        profile = climatology[:, lon_index]
        order = np.argsort(profile, kind="mergesort")
        sorted_values = profile[order]
        sorted_latitudes = latitudes[order]
        source_values = raw[:, lon_index]
        source_rows = np.arange(n_lat)

        insertion = np.searchsorted(sorted_values, source_values, side="left")
        lower = np.clip(insertion - 1, 0, n_lat - 1)
        upper = np.clip(insertion, 0, n_lat - 1)

        lower_distance = np.abs(source_values - sorted_values[lower])
        upper_distance = np.abs(source_values - sorted_values[upper])
        lower_row_distance = np.abs(order[lower] - source_rows)
        upper_row_distance = np.abs(order[upper] - source_rows)
        choose_upper = (upper_distance < lower_distance) | (
            (upper_distance == lower_distance)
            & (upper_row_distance < lower_row_distance)
        )
        nearest = np.where(choose_upper, upper, lower)
        matched[:, lon_index] = sorted_latitudes[nearest]

    return matched


def thermal_displacement_score(
    raw_temperature_k: np.ndarray,
    climatology_temperature_k: np.ndarray,
    latitudes_deg: np.ndarray,
) -> np.ndarray:
    matched = match_equivalent_latitude_same_longitude(
        raw_temperature_k,
        climatology_temperature_k,
        latitudes_deg,
    )
    max_abs_latitude = max(float(np.nanmax(np.abs(latitudes_deg))), 1e-6)
    score = (1.0 - np.abs(matched) / max_abs_latitude) * 100.0
    return gaussian_filter(
        np.clip(score, 0.0, 100.0).astype(np.float32),
        sigma=(1.0, 1.0),
        mode=("nearest", "wrap"),
    ).astype(np.float32)


def local_percentile_lon_wrap(
    values: np.ndarray,
    percentile: float,
    window_cells: int,
) -> np.ndarray:
    if window_cells % 2 == 0:
        window_cells += 1
    lon_pad = window_cells // 2
    padded = np.concatenate([values[:, -lon_pad:], values, values[:, :lon_pad]], axis=1)
    filtered = percentile_filter(
        padded,
        percentile=percentile,
        size=(window_cells, window_cells),
        mode="nearest",
    )
    return filtered[:, lon_pad:-lon_pad].astype(np.float32)


def fit_two_cluster_anchors(score: np.ndarray, latitudes: np.ndarray) -> ClusterAnchors:
    lat_mask = (np.abs(latitudes) >= 15.0) & (np.abs(latitudes) <= 80.0)
    sample = score[lat_mask, :].reshape(-1)
    sample = sample[np.isfinite(sample)]
    cold = float(np.nanpercentile(sample, 25.0))
    warm = float(np.nanpercentile(sample, 75.0))

    for _ in range(20):
        midpoint = 0.5 * (cold + warm)
        cold_values = sample[sample <= midpoint]
        warm_values = sample[sample > midpoint]
        if cold_values.size == 0 or warm_values.size == 0:
            break
        next_cold = float(np.nanmean(cold_values))
        next_warm = float(np.nanmean(warm_values))
        if abs(next_cold - cold) + abs(next_warm - warm) < 0.01:
            cold, warm = next_cold, next_warm
            break
        cold, warm = next_cold, next_warm

    if cold > warm:
        cold, warm = warm, cold
    midpoint = 0.5 * (cold + warm)
    softness = max((warm - cold) / 8.0, 3.5)
    return ClusterAnchors(cold, warm, midpoint, softness)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def fuzzy_membership(values: np.ndarray, anchors: ClusterAnchors) -> tuple[np.ndarray, np.ndarray]:
    warm = sigmoid((values - anchors.midpoint) / anchors.softness)
    cold = 1.0 - warm
    return cold.astype(np.float32), warm.astype(np.float32)


def gradient_strength(score: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    dlat = float(abs(np.nanmedian(np.diff(latitudes))))
    dlon = float(abs(np.nanmedian(np.diff(longitudes))))
    north = np.empty_like(score)
    south = np.empty_like(score)
    north[0] = score[0]
    north[1:] = score[:-1]
    south[-1] = score[-1]
    south[:-1] = score[1:]
    west = np.roll(score, 1, axis=1)
    east = np.roll(score, -1, axis=1)

    dy = 0.5 * (south - north) / max(dlat, 1e-6)
    cos_lat = np.maximum(np.cos(np.deg2rad(latitudes)), 0.15)
    dx = 0.5 * (east - west) / (max(dlon, 1e-6) * cos_lat[:, None])
    magnitude = np.hypot(dx, dy)
    finite = magnitude[np.isfinite(magnitude)]
    scale = float(np.nanpercentile(finite, 98.0)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def connected_component_summary(mask: np.ndarray, latitudes: np.ndarray) -> dict[str, object]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = label(mask, structure=structure)
    if count == 0:
        return {"component_count": 0, "largest_component_cells": 0, "midlatitude_share": 0.0}

    sizes = np.bincount(labels.reshape(-1))[1:]
    largest = int(sizes.max()) if sizes.size else 0
    lat_grid = np.broadcast_to(latitudes[:, None], mask.shape)
    selected = mask & np.isfinite(lat_grid)
    midlat = selected & (np.abs(lat_grid) >= 25.0) & (np.abs(lat_grid) <= 70.0)
    return {
        "component_count": int(count),
        "largest_component_cells": largest,
        "midlatitude_share": float(midlat.sum() / max(int(selected.sum()), 1)),
    }


def analyze_level(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    window_cells: int,
    low_percentile: float,
    high_percentile: float,
    min_local_range: float,
    full_local_range: float,
    transition_width: float,
    sharpness_floor: float,
) -> dict[str, np.ndarray | float | dict[str, object]]:
    local_low = local_percentile_lon_wrap(score, low_percentile, window_cells)
    local_high = local_percentile_lon_wrap(score, high_percentile, window_cells)
    local_range = np.maximum(local_high - local_low, 1e-3)

    anchors = fit_two_cluster_anchors(score, latitudes)
    cold_low, _ = fuzzy_membership(local_low, anchors)
    _, warm_high = fuzzy_membership(local_high, anchors)

    local_membership = np.clip((score - local_low) / local_range, 0.0, 1.0).astype(np.float32)
    transition = np.exp(-((local_membership - 0.5) / max(transition_width, 1e-3)) ** 4).astype(np.float32)
    range_span = max(full_local_range - min_local_range, 1e-3)
    range_strength = np.clip((local_range - min_local_range) / range_span, 0.0, 1.0).astype(np.float32)
    co_presence = (cold_low * warm_high * range_strength).astype(np.float32)
    sharpness = gradient_strength(score, latitudes, longitudes)

    # Gradient sharpness should brighten sharp boundaries, but broad slow transitions
    # remain visible when local cold/warm regimes are well separated.
    sharpness_floor = float(np.clip(sharpness_floor, 0.0, 1.0))
    conflict = co_presence * transition * (sharpness_floor + (1.0 - sharpness_floor) * sharpness)
    conflict = gaussian_filter(conflict, sigma=(1.0, 1.0), mode=("nearest", "wrap")).astype(np.float32)

    corridor = conflict >= max(float(np.nanpercentile(conflict, 94.0)), 0.12)
    high_confidence = conflict >= max(float(np.nanpercentile(conflict, 98.0)), 0.18)

    return {
        "local_low": local_low,
        "local_high": local_high,
        "local_membership": local_membership,
        "co_presence": co_presence,
        "sharpness": sharpness,
        "conflict": conflict,
        "corridor": corridor,
        "high_confidence": high_confidence,
        "anchors": {
            "cold_center": anchors.cold_center,
            "warm_center": anchors.warm_center,
            "midpoint": anchors.midpoint,
            "softness": anchors.softness,
        },
        "summary": {
            "local_range_mean": float(np.nanmean(local_range)),
            "local_range_p90": float(np.nanpercentile(local_range, 90.0)),
            "conflict_p94": float(np.nanpercentile(conflict, 94.0)),
            "conflict_p98": float(np.nanpercentile(conflict, 98.0)),
            "corridor_grid_share": float(np.nanmean(corridor)),
            "high_confidence_grid_share": float(np.nanmean(high_confidence)),
            "corridor_components": connected_component_summary(corridor, latitudes),
            "high_confidence_components": connected_component_summary(high_confidence, latitudes),
        },
    }


def add_map_features(ax: plt.Axes) -> None:
    ax.coastlines(linewidth=0.45, color="#1f2937")
    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color="#94a3b8",
        alpha=0.55,
        linestyle="-",
    )


def plot_level_map(
    output_path: Path,
    level: float,
    score: np.ndarray,
    result: dict[str, np.ndarray | float | dict[str, object]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    dpi: int,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.5, 7.0), dpi=dpi)
    ax = plt.axes(projection=projection)
    if extent is not None:
        ax.set_extent(extent, crs=projection)
    else:
        ax.set_global()
    add_map_features(ax)

    lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    cmap = plt.get_cmap("coolwarm")
    ax.pcolormesh(
        lon2d,
        lat2d,
        score,
        transform=projection,
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=100.0),
        shading="auto",
        alpha=0.9,
    )

    conflict = result["conflict"]
    assert isinstance(conflict, np.ndarray)
    overlay = np.ma.masked_where(conflict < max(float(np.nanpercentile(conflict, 91.0)), 0.08), conflict)
    ax.pcolormesh(
        lon2d,
        lat2d,
        overlay,
        transform=projection,
        cmap="Wistia",
        norm=mcolors.Normalize(vmin=float(np.nanpercentile(conflict, 90.0)), vmax=float(np.nanpercentile(conflict, 99.4))),
        shading="auto",
        alpha=0.62,
    )

    local_membership = result["local_membership"]
    co_presence = result["co_presence"]
    assert isinstance(local_membership, np.ndarray)
    assert isinstance(co_presence, np.ndarray)
    masked_membership = np.ma.masked_where(co_presence < max(float(np.nanpercentile(co_presence, 75.0)), 0.20), local_membership)
    ax.contour(
        lon2d,
        lat2d,
        masked_membership,
        levels=[0.5],
        transform=projection,
        colors=["#111827"],
        linewidths=0.65,
        alpha=0.75,
    )

    title_suffix = "" if extent is None else " Greenland/N Atlantic crop"
    ax.set_title(
        f"{int(level)} hPa fuzzy local warm/cold thermal-identity interface{title_suffix}\n"
        "background = Thermal Displacement; yellow/white = fuzzy local conflict corridor; black = local 50/50 transition",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(output_path: Path, image_paths: list[Path], title: str, dpi: int) -> None:
    images = [plt.imread(path) for path in image_paths]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=dpi)
    fig.suptitle(title, fontsize=16)
    for ax, image, path in zip(axes.reshape(-1), images, image_paths):
        ax.imshow(image)
        ax.set_title(path.stem.replace("-", " "), fontsize=10)
        ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    temperature_path = resolve_path(args.temperature)
    climatology_path = resolve_path(args.climatology)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = parse_levels(args.levels)
    temperature_ds = xr.open_dataset(temperature_path)
    climatology_ds = xr.open_dataset(climatology_path)
    try:
        temperature = temperature_ds[TEMPERATURE_VARIABLE].sel(valid_time=TIMESTAMP)
        climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
        latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)
        grid_step = float(abs(np.nanmedian(np.diff(latitudes))))
        window_cells = max(5, int(round(args.window_deg / grid_step)))
        if window_cells % 2 == 0:
            window_cells += 1

        summaries: dict[str, object] = {
            "method": "fuzzy_local_thermal_identity_interface",
            "valid_time": np.datetime_as_string(TIMESTAMP, unit="m"),
            "temperature": display_path(temperature_path),
            "climatology": display_path(climatology_path),
            "levels_hpa": levels,
            "window_degrees": args.window_deg,
            "window_cells": window_cells,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "min_local_range": args.min_local_range,
            "full_local_range": args.full_local_range,
            "transition_width": args.transition_width,
            "sharpness_floor": args.sharpness_floor,
            "interpretation": (
                "Conflict is local co-presence of cold-like and warm-like thermal "
                "identity plus a fuzzy transition between those local regimes. "
                "Gradient sharpness brightens the signal but is not required."
            ),
            "level_summaries": {},
        }

        global_maps: list[Path] = []
        crop_maps: list[Path] = []
        for level in levels:
            raw = np.asarray(temperature.sel(pressure_level=level, method="nearest").values, dtype=np.float32)
            clim = np.asarray(climatology.sel(pressure_level=level, method="nearest").values, dtype=np.float32)
            actual_level = float(temperature.sel(pressure_level=level, method="nearest").coords["pressure_level"].item())
            score = thermal_displacement_score(raw, clim, latitudes)
            result = analyze_level(
                score,
                latitudes,
                longitudes,
                window_cells,
                float(args.low_percentile),
                float(args.high_percentile),
                float(args.min_local_range),
                float(args.full_local_range),
                float(args.transition_width),
                float(args.sharpness_floor),
            )

            level_key = f"{int(actual_level):04d}hpa"
            summary = result["summary"]
            assert isinstance(summary, dict)
            summaries["level_summaries"][level_key] = {
                "anchors": result["anchors"],
                **summary,
            }

            global_path = output_dir / f"fuzzy-local-interface-{level_key}.png"
            crop_path = output_dir / f"fuzzy-local-interface-greenland-crop-{level_key}.png"
            plot_level_map(global_path, actual_level, score, result, latitudes, longitudes, args.dpi)
            plot_level_map(
                crop_path,
                actual_level,
                score,
                result,
                latitudes,
                longitudes,
                args.dpi,
                extent=(-85.0, -5.0, 35.0, 75.0),
            )
            global_maps.append(global_path)
            crop_maps.append(crop_path)

        contact_sheet = output_dir / "fuzzy-local-interface-contact-sheet.png"
        plot_contact_sheet(
            contact_sheet,
            global_maps,
            "Fuzzy Local Thermal-Identity Interface",
            args.dpi,
        )
        crop_contact_sheet = output_dir / "fuzzy-local-interface-greenland-crop-contact-sheet.png"
        plot_contact_sheet(
            crop_contact_sheet,
            crop_maps,
            "Fuzzy Local Thermal-Identity Interface - Greenland / North Atlantic Crop",
            args.dpi,
        )
        summaries["plots"] = {
            "contact_sheet": display_path(contact_sheet),
            "greenland_crop_contact_sheet": display_path(crop_contact_sheet),
            "global_maps": [display_path(path) for path in global_maps],
            "greenland_crop_maps": [display_path(path) for path in crop_maps],
        }

        (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps({"summary": display_path(output_dir / "summary.json"), "contact_sheet": display_path(contact_sheet)}, indent=2))
    finally:
        temperature_ds.close()
        climatology_ds.close()


if __name__ == "__main__":
    main()
