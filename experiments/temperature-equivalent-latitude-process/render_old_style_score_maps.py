from __future__ import annotations

import csv
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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image, ImageDraw


SOURCE_DIR = Path("tmp/temperature-equivalent-latitude-process/output/raw-smoothed-score-range-tiebreak")
OUTPUT_DIR = SOURCE_DIR / "old-style-cartopy-maps"
DATASET = Path("data/era5_temperature_2021-11_08-12.nc")


def slug(level: float) -> str:
    return f"{level:g}".replace(".", "p").replace("-", "m") + "hpa"


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    deltas = np.diff(values)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = values[:-1] + deltas / 2.0
    edges[0] = values[0] - deltas[0] / 2.0
    edges[-1] = values[-1] + deltas[-1] / 2.0
    return edges


def load_white_centers() -> dict[float, float]:
    centers: dict[float, float] = {}
    with (SOURCE_DIR / "selected_buckets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            centers[float(row["pressure_level_hpa"])] = float(row["white_center"])
    return centers


def render_map(
    level: float,
    values: np.ndarray,
    white_center: float,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    output_path: Path,
) -> None:
    field = values[::-1, :]
    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "thermal_displacement_centered_bwr",
        ["#1f5eff", "#ffffff", "#d7191c"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=white_center, vmax=vmax)

    fig = plt.figure(figsize=(13.5, 6.7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=1,
    )
    ax.coastlines(resolution="110m", linewidth=0.55, color="#1b1b1b", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, edgecolor="#404040", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#9e9e9e", alpha=0.55)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(
        f"{level:g} hPa Thermal Displacement, old-style Cartopy render\n"
        f"white centered at range-middle rare bucket {white_center:.0f}"
    )
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, shrink=0.82)
    colorbar.set_label("Thermal displacement score points (0 polar-like, 100 equator-like)")
    colorbar.set_ticks([vmin, white_center, vmax])
    colorbar.set_ticklabels([f"{vmin:.1f}", f"{white_center:.0f}", f"{vmax:.1f}"])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_contact_sheet(levels: list[float]) -> None:
    key = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50, 20, 10, 1]
    images = []
    for level in key:
        if float(level) not in levels:
            continue
        path = OUTPUT_DIR / f"old_style_score_map_{slug(float(level))}.png"
        image = Image.open(path).convert("RGB")
        image.thumbnail((450, 210), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 245), "white")
        canvas.paste(image, (5, 25))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"{level:g} hPa", fill=(0, 0, 0))
        images.append(canvas)
    cols = 3
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows * 245), (245, 245, 245))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % cols) * 460, (index // cols) * 245))
    sheet.save(OUTPUT_DIR / "old_style_score_key_levels_contact_sheet.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(DATASET)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_edges = cell_edges(lat[::-1])
    lon_edges = cell_edges(lon)
    centers = load_white_centers()
    levels = sorted(centers.keys(), reverse=True)
    for level in levels:
        values = np.load(SOURCE_DIR / "arrays" / f"thermal_displacement_score_points_{slug(level)}.npy")
        render_map(
            level=level,
            values=values,
            white_center=centers[level],
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            output_path=OUTPUT_DIR / f"old_style_score_map_{slug(level)}.png",
        )
    make_contact_sheet(levels)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
