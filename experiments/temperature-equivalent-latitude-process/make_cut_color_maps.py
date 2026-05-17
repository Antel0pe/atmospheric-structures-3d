from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

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
from PIL import Image
from scipy.ndimage import gaussian_filter1d


ROOT = Path("tmp/temperature-equivalent-latitude-process/output")
DEFAULT_OUTPUT = ROOT / "cut-color-analysis"
DEFAULT_DATASET = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_BORDER_GEOJSON = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw histogram-derived cut-color equivalent-latitude maps."
    )
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--border-geojson", type=Path, default=DEFAULT_BORDER_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hist-smooth-sigma-bins", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--gif-width", type=int, default=1200)
    parser.add_argument(
        "--map-scale",
        type=str,
        choices=("global", "per-level"),
        default="global",
        help=(
            "Use global for the original -90..90 map normalization, or per-level "
            "to stretch each pressure map to that level's own min/max."
        ),
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF generation. Useful when only static map plots are needed.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = (Path.cwd() / expanded).resolve()
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(path.as_posix())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slug(level: float) -> str:
    return f"{level:g}".replace(".", "p").replace("-", "m") + "hpa"


def load_levels(source_root: Path) -> list[float]:
    levels: list[float] = []
    with (source_root / "selected_buckets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            levels.append(float(row["pressure_level_hpa"]))
    levels.sort(reverse=True)
    return levels


def target_longitude(lon: float, longitudes: np.ndarray) -> float:
    lon_min = float(np.nanmin(longitudes))
    lon_max = float(np.nanmax(longitudes))
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def split_target_longitude_segments(
    points: list[tuple[float, float]],
    longitudes: np.ndarray,
) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None
    for lon, lat in points:
        mapped_lon = target_longitude(float(lon), longitudes)
        if previous_lon is not None and abs(mapped_lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((mapped_lon, float(lat)))
        previous_lon = mapped_lon
    if len(segment) >= 2:
        yield segment


def load_border_segments(
    geojson_path: Path,
    longitudes: np.ndarray,
) -> list[list[tuple[float, float]]]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    segments: list[list[tuple[float, float]]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "Polygon":
            polygons = [coordinates]
        elif geometry.get("type") == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        for polygon in polygons:
            for ring in polygon:
                points = [(float(lon), float(lat)) for lon, lat, *_ in ring]
                segments.extend(split_target_longitude_segments(points, longitudes))
    return segments


def choose_cut_summary(
    values: np.ndarray,
    bins: np.ndarray,
    smooth_sigma_bins: float,
) -> dict[str, object]:
    counts, edges = np.histogram(values, bins=bins)
    smoothed = gaussian_filter1d(
        counts.astype(np.float64),
        sigma=smooth_sigma_bins,
        mode="nearest",
    )
    q03, q20, q30, q70, q80, q97 = np.percentile(
        values,
        [3.0, 20.0, 30.0, 70.0, 80.0, 97.0],
    )

    def valley(low: float, high: float) -> tuple[float, float, int]:
        lo = min(low, high)
        hi = max(low, high)
        indices = np.flatnonzero((edges[:-1] >= lo) & (edges[1:] <= hi))
        if indices.size == 0:
            indices = np.arange(len(counts))
        index = int(indices[np.argmin(smoothed[indices])])
        center = float(0.5 * (edges[index] + edges[index + 1]))
        return center, float(edges[index]), int(counts[index])

    cold_outer_center, cold_outer_lower, cold_outer_count = valley(q03, q30)
    white_center, white_lower, white_count = valley(q20, q80)
    warm_outer_center, warm_outer_lower, warm_outer_count = valley(q70, q97)

    warm_shoulder_center = float(round(q80 * 2.0) / 2.0)

    return {
        "reference_pressure_range_hpa": [1000.0, 250.0],
        "histogram_smooth_sigma_bins": smooth_sigma_bins,
        "percentiles": {
            "p03": float(q03),
            "p20": float(q20),
            "p30": float(q30),
            "p70": float(q70),
            "p80": float(q80),
            "p97": float(q97),
        },
        "cuts": [
            {
                "name": "cold_outer_density_dip",
                "value": cold_outer_center,
                "bucket_lower": cold_outer_lower,
                "bucket_count": cold_outer_count,
                "role": "dark blue to blue",
            },
            {
                "name": "white_middle_60_density_dip",
                "value": white_center,
                "bucket_lower": white_lower,
                "bucket_count": white_count,
                "role": "blue/red transition white point",
            },
            {
                "name": "warm_middle_60_shoulder",
                "value": warm_shoulder_center,
                "source": "rounded 80th percentile of reference distribution",
                "role": "pale red to red shoulder",
            },
            {
                "name": "warm_outer_density_dip",
                "value": warm_outer_center,
                "bucket_lower": warm_outer_lower,
                "bucket_count": warm_outer_count,
                "role": "red to dark red",
            },
        ],
    }


def make_cmap(cuts: dict[str, float]) -> mcolors.LinearSegmentedColormap:
    stops = [
        (-90.0, "#061536"),
        (cuts["cold_outer_density_dip"], "#1764b5"),
        (cuts["white_middle_60_density_dip"], "#f7f7f2"),
        (cuts["warm_middle_60_shoulder"], "#f3a0a0"),
        (cuts["warm_outer_density_dip"], "#cd2a2a"),
        (90.0, "#4a0000"),
    ]
    normalized = [((value + 90.0) / 180.0, color) for value, color in stops]
    return mcolors.LinearSegmentedColormap.from_list("histogram_cut_bwr", normalized)


def make_level_scaled_cmap(
    base_cmap: mcolors.Colormap,
    color_stops: list[tuple[float, str]],
    vmin: float,
    vmax: float,
) -> mcolors.LinearSegmentedColormap:
    if not vmin < vmax:
        vmax = vmin + 1.0

    points: list[tuple[float, object]] = [(0.0, base_cmap((vmin + 90.0) / 180.0))]
    for value, color in color_stops:
        if vmin < value < vmax:
            points.append(((value - vmin) / (vmax - vmin), color))
    points.append((1.0, base_cmap((vmax + 90.0) / 180.0)))
    points.sort(key=lambda item: item[0])
    return mcolors.LinearSegmentedColormap.from_list(
        "histogram_cut_bwr_level_scaled",
        points,
    )


def plot_aggregate_histogram(
    values: np.ndarray,
    bins: np.ndarray,
    cut_summary: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    smoothed = gaussian_filter1d(
        counts.astype(np.float64),
        sigma=float(cut_summary["histogram_smooth_sigma_bins"]),
        mode="nearest",
    )
    cuts = cut_summary["cuts"]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(edges[:-1], counts, width=1.0, align="edge", color="#6e8fb5", edgecolor="none", alpha=0.55)
    ax.plot(0.5 * (edges[:-1] + edges[1:]), smoothed, color="#16202a", linewidth=1.7)
    for cut in cuts:
        ax.axvline(float(cut["value"]), color="#ffd21f", linewidth=2.4)
        ax.text(
            float(cut["value"]),
            ax.get_ylim()[1] * 0.93,
            f"{cut['value']:.1f}",
            rotation=90,
            va="top",
            ha="right",
            color="#7d6500",
            fontsize=8,
        )
    ax.set_title("Aggregate 1000-250 hPa smoothed equivalent-latitude histogram with selected cuts")
    ax.set_xlabel("Smoothed equivalent latitude")
    ax.set_ylabel("Cell count")
    ax.set_xlim(-90, 90)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_level_histogram(
    values: np.ndarray,
    level: float,
    bins: np.ndarray,
    cut_values: list[float],
    smooth_sigma_bins: float,
    output_path: Path,
    dpi: int,
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma_bins, mode="nearest")

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(edges[:-1], counts, width=1.0, align="edge", color="#6e8fb5", edgecolor="none", alpha=0.55)
    ax.plot(0.5 * (edges[:-1] + edges[1:]), smoothed, color="#16202a", linewidth=1.6)
    for value in cut_values:
        ax.axvline(value, color="#ffd21f", linewidth=2.2)
    ax.set_title(f"{level:g} hPa histogram with fixed global cut values")
    ax.set_xlabel("Smoothed equivalent latitude")
    ax.set_ylabel("Cell count")
    ax.set_xlim(-90, 90)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_map(
    values: np.ndarray,
    level: float,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    border_segments: list[list[tuple[float, float]]],
    cmap: mcolors.Colormap,
    cut_values: list[float],
    vmin: float,
    vmax: float,
    map_scale: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        values,
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        shading="auto",
        rasterized=True,
    )
    for segment in border_segments:
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color="#171717", linewidth=0.35, alpha=0.75)
    ax.set_xlim(float(np.min(longitudes)), float(np.max(longitudes)))
    ax.set_ylim(float(np.min(latitudes)), float(np.max(latitudes)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{level:g} hPa equivalent latitude with fixed cuts, "
        f"{map_scale} color scale [{vmin:.1f}, {vmax:.1f}]"
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.88)
    colorbar.set_label("Smoothed matched climatology latitude")
    for value in cut_values:
        if vmin <= value <= vmax:
            colorbar.ax.axhline(value, color="#ffd21f", linewidth=1.8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_contact_sheet(map_dir: Path, levels: list[float], output_path: Path) -> None:
    key_levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50, 20, 10, 1]
    images = []
    for level in key_levels:
        if level not in levels:
            continue
        image = Image.open(map_dir / f"cut_color_map_{slug(level)}.png").convert("RGB")
        image.thumbnail((450, 210), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 245), "white")
        canvas.paste(image, (5, 25))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"{level:g} hPa", fill=(0, 0, 0))
        images.append(canvas)
    cols = 3
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows * 245), (245, 245, 245))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % cols) * 460, (index // cols) * 245))
    sheet.save(output_path)


def make_gif(map_dir: Path, levels: list[float], output_path: Path, width: int) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    previous: Image.Image | None = None
    for level in levels:
        image = Image.open(map_dir / f"cut_color_map_{slug(level)}.png").convert("RGB")
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        if previous is not None:
            for step in range(1, 6):
                frames.append(Image.blend(previous, image, step / 6.0))
                durations.append(80)
        frames.append(image)
        durations.append(160)
        previous = image
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    histogram_dir = output_dir / "histograms"
    map_dir = output_dir / "maps"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    levels = load_levels(source_root)
    arrays = {
        level: np.load(source_root / "arrays" / f"equivalent_latitude_smoothed_{slug(level)}.npy")
        for level in levels
    }
    reference_levels = [level for level in levels if 250.0 <= level <= 1000.0]
    reference_values = np.concatenate([arrays[level].ravel() for level in reference_levels])
    bins = np.arange(-90.0, 91.0, 1.0)
    cut_summary = choose_cut_summary(
        reference_values,
        bins=bins,
        smooth_sigma_bins=args.hist_smooth_sigma_bins,
    )
    cuts_by_name = {str(cut["name"]): float(cut["value"]) for cut in cut_summary["cuts"]}
    cut_values = [float(cut["value"]) for cut in cut_summary["cuts"]]
    base_cmap = make_cmap(cuts_by_name)
    color_stops = [
        (-90.0, "#061536"),
        (cuts_by_name["cold_outer_density_dip"], "#1764b5"),
        (cuts_by_name["white_middle_60_density_dip"], "#f7f7f2"),
        (cuts_by_name["warm_middle_60_shoulder"], "#f3a0a0"),
        (cuts_by_name["warm_outer_density_dip"], "#cd2a2a"),
        (90.0, "#4a0000"),
    ]

    ds = xr.open_dataset(resolve_path(args.dataset))
    latitudes = np.asarray(ds.coords["latitude"].values, dtype=np.float32)
    longitudes = np.asarray(ds.coords["longitude"].values, dtype=np.float32)
    border_segments = load_border_segments(resolve_path(args.border_geojson), longitudes)

    plot_aggregate_histogram(
        reference_values,
        bins=bins,
        cut_summary=cut_summary,
        output_path=output_dir / "aggregate_histogram_with_yellow_cuts.png",
        dpi=args.dpi,
    )

    for level in levels:
        values = arrays[level]
        level_vmin = float(np.nanmin(values))
        level_vmax = float(np.nanmax(values))
        map_vmin = -90.0 if args.map_scale == "global" else level_vmin
        map_vmax = 90.0 if args.map_scale == "global" else level_vmax
        map_cmap = (
            base_cmap
            if args.map_scale == "global"
            else make_level_scaled_cmap(
                base_cmap,
                color_stops=color_stops,
                vmin=level_vmin,
                vmax=level_vmax,
            )
        )
        plot_level_histogram(
            values=values.ravel(),
            level=level,
            bins=bins,
            cut_values=cut_values,
            smooth_sigma_bins=args.hist_smooth_sigma_bins,
            output_path=histogram_dir / f"histogram_yellow_cuts_{slug(level)}.png",
            dpi=args.dpi,
        )
        plot_map(
            values=values,
            level=level,
            longitudes=longitudes,
            latitudes=latitudes,
            border_segments=border_segments,
            cmap=map_cmap,
            cut_values=cut_values,
            vmin=map_vmin,
            vmax=map_vmax,
            map_scale=args.map_scale,
            output_path=map_dir / f"cut_color_map_{slug(level)}.png",
            dpi=args.dpi,
        )

    make_contact_sheet(map_dir, levels, output_dir / "cut_color_key_levels_contact_sheet.png")
    if not args.skip_gif:
        make_gif(
            map_dir,
            levels,
            output_dir / "cut_color_maps_high_to_low_pressure.gif",
            width=args.gif_width,
        )

    summary = {
        **cut_summary,
        "method_note": (
            "Reference histogram is all smoothed equivalent-latitude cells from "
            "1000-250 hPa. Density dips are chosen after Gaussian smoothing the "
            "whole-degree histogram. The warm shoulder is the rounded 80th "
            "percentile because the red side did not contain a robust central "
            "density valley before the warm outer tail dip."
        ),
        "map_scale": args.map_scale,
        "color_stops": [
            {"value": -90.0, "color": "#061536", "meaning": "deep cold-side equivalent latitude"},
            {"value": cuts_by_name["cold_outer_density_dip"], "color": "#1764b5", "meaning": "cold-side density dip"},
            {"value": cuts_by_name["white_middle_60_density_dip"], "color": "#f7f7f2", "meaning": "global middle-60 density minimum"},
            {"value": cuts_by_name["warm_middle_60_shoulder"], "color": "#f3a0a0", "meaning": "warm-side shoulder"},
            {"value": cuts_by_name["warm_outer_density_dip"], "color": "#cd2a2a", "meaning": "warm-side density dip"},
            {"value": 90.0, "color": "#4a0000", "meaning": "deep warm-side equivalent latitude"},
        ],
        "outputs": {
            "aggregate_histogram": display_path(output_dir / "aggregate_histogram_with_yellow_cuts.png"),
            "histograms": display_path(histogram_dir),
            "maps": display_path(map_dir),
            "contact_sheet": display_path(output_dir / "cut_color_key_levels_contact_sheet.png"),
            "gif": None
            if args.skip_gif
            else display_path(output_dir / "cut_color_maps_high_to_low_pressure.gif"),
        },
    }
    (output_dir / "cut_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)
    print(json.dumps(summary["color_stops"], indent=2))


if __name__ == "__main__":
    main()
