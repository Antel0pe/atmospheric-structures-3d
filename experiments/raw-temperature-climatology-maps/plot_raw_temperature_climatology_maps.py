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
from cartopy.util import add_cyclic_point
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DATASET_PATH = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
OUTPUT_ROOT = Path("tmp/raw-temperature-climatology-maps")
PLOTS_DIR = OUTPUT_ROOT / "plots"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
PRESSURE_LEVELS_HPA = [250, 500, 850, 1000]


def repo_relative(path: Path) -> str:
    return path.as_posix()


def temperature_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "raw_temperature_blue_white_red",
        ["#1d4ed8", "#f8fafc", "#b91c1c"],
    )


def plot_level(
    *,
    level_hpa: int,
    climatology_k: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_path: Path,
) -> dict[str, float | int | str]:
    finite = climatology_k[np.isfinite(climatology_k)]
    if finite.size == 0:
        raise ValueError(f"No finite temperature values at {level_hpa} hPa")

    min_k = float(np.nanmin(finite))
    max_k = float(np.nanmax(finite))
    mean_k = float(np.nanmean(finite))
    midpoint_k = (min_k + max_k) / 2.0
    cyclic_values, cyclic_longitudes = add_cyclic_point(climatology_k, coord=longitudes)

    fig = plt.figure(figsize=(13.5, 6.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_facecolor("white")

    mesh = ax.pcolormesh(
        cyclic_longitudes,
        latitudes,
        cyclic_values,
        transform=ccrs.PlateCarree(),
        cmap=temperature_cmap(),
        norm=mcolors.Normalize(vmin=min_k, vmax=max_k),
        shading="auto",
        zorder=1,
    )
    ax.coastlines(resolution="110m", linewidth=0.55, color="#171717", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, edgecolor="#404040", zorder=3)
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        color="#737373",
        alpha=0.45,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    ax.set_title(
        (
            f"{level_hpa} hPa raw temperature climatology | "
            "Nov 8 12Z, 1990-2020 mean"
        ),
        fontsize=13,
    )
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.055, fraction=0.045)
    colorbar.set_label("Raw temperature climatology (K)")
    colorbar.ax.axvline(midpoint_k, color="#111111", linewidth=0.7, alpha=0.75)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "pressure_level_hpa": level_hpa,
        "plot": repo_relative(output_path),
        "min_k": min_k,
        "mean_k": mean_k,
        "max_k": max_k,
        "color_scale": "per pressure level min/max; blue low, white midpoint, red high",
    }


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    level_summaries: list[dict[str, float | int | str]] = []
    with xr.open_dataset(DATASET_PATH) as dataset:
        temperature = dataset["t"]
        latitudes = np.asarray(dataset["latitude"].values, dtype=np.float64)
        longitudes = np.asarray(dataset["longitude"].values, dtype=np.float64)
        valid_times = np.asarray(dataset["valid_time"].values)

        for level_hpa in PRESSURE_LEVELS_HPA:
            level_values = (
                temperature.sel(pressure_level=float(level_hpa))
                .mean(dim="valid_time", skipna=True)
                .values.astype(np.float32)
            )
            output_path = PLOTS_DIR / f"raw-temperature-climatology-map-{level_hpa:04d}hpa.png"
            print(f"plotting {level_hpa} hPa", flush=True)
            level_summaries.append(
                plot_level(
                    level_hpa=level_hpa,
                    climatology_k=level_values,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    output_path=output_path,
                )
            )

    summary = {
        "experiment": "raw-temperature-climatology-maps",
        "date": "2026-05-28",
        "source_dataset": repo_relative(DATASET_PATH),
        "variable": "t",
        "method": "mean raw pressure-level temperature across valid_time entries",
        "valid_time_count": int(valid_times.size),
        "levels_hpa": PRESSURE_LEVELS_HPA,
        "domain": "global latitude/longitude",
        "unit": "K",
        "script": repo_relative(Path("tmp/raw-temperature-climatology-maps/plot_raw_temperature_climatology_maps.py")),
        "plots_dir": repo_relative(PLOTS_DIR),
        "levels": level_summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
