from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CACHE_ROOT = Path("/tmp/atmospheric-structures-3d-cache")
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_maps import (  # noqa: E402
    CLIMATOLOGY_VARIABLE,
    DEFAULT_CLIMATOLOGY,
    DEFAULT_DATASET,
    DEFAULT_LEVELS,
    DEFAULT_TIMESTAMP,
    TEMPERATURE_VARIABLE,
    choose_timestamp,
    match_equivalent_latitude,
    parse_requested_levels,
    resolve_path,
    smooth_wrapped_lon,
    thermal_displacement_score_points,
    validate_matching_grid,
)


DEFAULT_OUTPUT_DIR = Path(
    "tmp/thermal-displacement-latitude-agreement/output/"
    "same-hemisphere-score-smoothed-sigma20-smoothed-agreement/"
    "longitude-latitude-line-plots"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot thermal-displacement score versus latitude for every longitude "
            "in a requested window."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument("--pressure-levels", type=str, default=DEFAULT_LEVELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-sigma-cells", type=float, default=20.0)
    parser.add_argument("--lon-min", type=float, default=-125.0)
    parser.add_argument("--lon-max", type=float, default=-50.0)
    parser.add_argument("--longitude-stride", type=int, default=1)
    parser.add_argument(
        "--red-ramp",
        choices=("dark", "bright", "stepwise"),
        default="dark",
    )
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def slug_for_level(level_hpa: float) -> str:
    return f"{level_hpa:g}".replace(".", "p").replace("-", "m") + "hpa"


def longitude_to_signed(longitudes: np.ndarray) -> np.ndarray:
    return ((longitudes + 180.0) % 360.0) - 180.0


def select_sorted_longitudes(
    longitudes: np.ndarray,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed = longitude_to_signed(longitudes.astype(np.float64))
    mask = (signed >= lon_min) & (signed <= lon_max)
    indices = np.flatnonzero(mask)
    order = np.argsort(signed[indices])
    return indices[order], signed[indices][order]


def plot_level_lines(
    score: np.ndarray,
    latitudes: np.ndarray,
    longitudes_signed: np.ndarray,
    longitude_indices: np.ndarray,
    latitude_indices: np.ndarray,
    level_hpa: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_path: Path,
    dpi: int,
    red_ramp: str,
) -> None:
    selected_lats = latitudes[latitude_indices]
    selected_scores = score[np.ix_(latitude_indices, longitude_indices)]
    norm = mcolors.Normalize(vmin=lon_min, vmax=lon_max)
    if red_ramp == "stepwise":
        colors = plt.get_cmap("tab20").colors
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(len(longitudes_signed) + 1) - 0.5, len(colors))
    else:
        ramp_colors = ["#4b0000", "#ffb0a4"]
        if red_ramp == "bright":
            ramp_colors = ["#d00000", "#ffb3a8"]
        cmap = mcolors.LinearSegmentedColormap.from_list("west_to_east_reds", ramp_colors)
        norm = mcolors.Normalize(vmin=lon_min, vmax=lon_max)

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for column_index, longitude in enumerate(longitudes_signed):
        color_value = column_index if red_ramp == "stepwise" else longitude
        ax.plot(
            selected_lats,
            selected_scores[:, column_index],
            color=cmap(norm(color_value)),
            linewidth=1.05 if red_ramp == "stepwise" else 0.55,
            alpha=0.92 if red_ramp == "stepwise" else 0.58,
        )

    ax.set_xlim(lat_max, lat_min)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Thermal-displacement score")
    ax.set_title(
        f"{level_hpa:g} hPa score by latitude, longitudes {lon_min:g} to {lon_max:g}"
    )
    ax.grid(color="#d6d6d6", linewidth=0.6, alpha=0.75)
    if red_ramp == "stepwise":
        handles = []
        for column_index, longitude in enumerate(longitudes_signed):
            line = plt.Line2D(
                [0],
                [0],
                color=cmap(norm(column_index)),
                linewidth=2.0,
                label=f"{longitude:g}",
            )
            handles.append(line)
        ax.legend(
            handles=handles,
            title="Longitude",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
        )
    else:
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.01)
        colorbar.set_label("Longitude; dark red west, light red east")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    climatology_path = resolve_path(args.climatology)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_ds = xr.open_dataset(dataset_path)
    climatology_ds = xr.open_dataset(climatology_path)
    temperature = temperature_ds[TEMPERATURE_VARIABLE]
    climatology = climatology_ds[CLIMATOLOGY_VARIABLE]
    validate_matching_grid(temperature, climatology)

    selected_time = choose_timestamp(temperature, args.timestamp)
    level_values = np.asarray(temperature.coords["pressure_level"].values, dtype=np.float64)
    selected_levels = parse_requested_levels(args.pressure_levels, level_values)
    latitudes = np.asarray(temperature.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(temperature.coords["longitude"].values, dtype=np.float32)

    longitude_indices, longitudes_signed = select_sorted_longitudes(
        longitudes,
        args.lon_min,
        args.lon_max,
    )
    if args.longitude_stride < 1:
        raise ValueError("--longitude-stride must be >= 1")
    longitude_indices = longitude_indices[:: args.longitude_stride]
    longitudes_signed = longitudes_signed[:: args.longitude_stride]
    latitude_indices = np.flatnonzero(
        (latitudes >= args.lat_min) & (latitudes <= args.lat_max)
    )
    latitude_indices = latitude_indices[np.argsort(-latitudes[latitude_indices])]

    for level_hpa in selected_levels:
        slug = slug_for_level(level_hpa)
        print(f"Processing {level_hpa:g} hPa")
        raw_level = (
            temperature.sel(valid_time=selected_time, pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )
        clim_level = (
            climatology.sel(pressure_level=level_hpa)
            .load()
            .to_numpy()
            .astype(np.float32)
        )

        matched_latitude = match_equivalent_latitude(
            raw_level,
            clim_level,
            latitudes,
            "same-hemisphere",
        )
        matched_latitude = np.clip(matched_latitude, -90.0, 90.0).astype(np.float32)
        score_unsmoothed = thermal_displacement_score_points(matched_latitude, latitudes)
        score_smoothed = smooth_wrapped_lon(score_unsmoothed, args.smooth_sigma_cells)
        output_path = output_dir / f"score_by_latitude_{slug}.png"
        plot_level_lines(
            score=score_smoothed,
            latitudes=latitudes,
            longitudes_signed=longitudes_signed,
            longitude_indices=longitude_indices,
            latitude_indices=latitude_indices,
            level_hpa=level_hpa,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            output_path=output_path,
            dpi=args.dpi,
            red_ramp=args.red_ramp,
        )

    print(f"Wrote {output_dir.relative_to(Path.cwd()).as_posix()}")


if __name__ == "__main__":
    main()
