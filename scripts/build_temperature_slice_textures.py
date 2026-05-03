from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import xarray as xr
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_voxel_builder import (
    clear_output_dir,
    coordinate_step_degrees,
    timestamp_to_slug,
)


OUTPUT_VERSION = 1
DATASET_VARIABLE = "t"
DEFAULT_DATASET_PATH = Path("data/era5_temperature_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/temperature-slices")
DEFAULT_PRESSURE_MIN_HPA = 250.0
DEFAULT_PRESSURE_MAX_HPA = 1000.0
DEFAULT_INCLUDE_TIMESTAMPS = ("2021-11-08T12:00",)
DEFAULT_BORDER_GEOJSON_PATH = Path(
    "node_modules/three-globe/example/country-polygons/ne_110m_admin_0_countries.geojson"
)


@dataclass(frozen=True)
class DatasetContents:
    dataset_path: Path
    variable_name: str
    units: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    timestamps: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static pressure-level temperature textures for the full-field "
            "temperature slice layer."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 pressure-level temperature NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated temperature textures will be written.",
    )
    parser.add_argument(
        "--pressure-min-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MIN_HPA,
        help="Lowest pressure level to export.",
    )
    parser.add_argument(
        "--pressure-max-hpa",
        type=float,
        default=DEFAULT_PRESSURE_MAX_HPA,
        help="Highest pressure level to export.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default=",".join(DEFAULT_INCLUDE_TIMESTAMPS),
        help="Comma-separated ISO minute timestamps to build.",
    )
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if text.endswith("Z"):
        return text[:-1]
    return text


def format_display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        home = Path.home()
        try:
            return f"~/{path.relative_to(home).as_posix()}"
        except ValueError:
            return path.name or "<external-path>"


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {format_display_path(resolved)}"
        )
    return resolved


def load_dataset_contents(dataset_path: Path) -> DatasetContents:
    dataset = xr.open_dataset(dataset_path)
    try:
        variable = dataset[DATASET_VARIABLE]
        timestamps = [
            timestamp_to_iso_minute(value)
            for value in np.asarray(variable.coords["valid_time"].values)
        ]
        return DatasetContents(
            dataset_path=dataset_path,
            variable_name=DATASET_VARIABLE,
            units=str(variable.attrs.get("units", "")),
            pressure_levels_hpa=np.asarray(
                variable.coords["pressure_level"].values,
                dtype=np.float32,
            ),
            latitudes_deg=np.asarray(variable.coords["latitude"].values, dtype=np.float32),
            longitudes_deg=np.asarray(
                variable.coords["longitude"].values,
                dtype=np.float32,
            ),
            timestamps=timestamps,
        )
    finally:
        dataset.close()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_target_timestamps(
    all_timestamps: list[str],
    include_timestamps_text: str,
) -> list[str]:
    if not include_timestamps_text.strip():
        return all_timestamps

    requested = {
        value.strip()
        for value in include_timestamps_text.split(",")
        if value.strip()
    }
    return [timestamp for timestamp in all_timestamps if timestamp in requested]


def pressure_level_indices(
    pressure_levels_hpa: np.ndarray,
    pressure_min_hpa: float,
    pressure_max_hpa: float,
) -> list[int]:
    lower = min(pressure_min_hpa, pressure_max_hpa)
    upper = max(pressure_min_hpa, pressure_max_hpa)
    return [
        index
        for index, pressure_hpa in enumerate(pressure_levels_hpa)
        if lower <= float(pressure_hpa) <= upper
    ]


def encode_temperature_to_rgba_uint8(
    temperature_k: np.ndarray,
    temperature_min_k: float,
    temperature_max_k: float,
) -> np.ndarray:
    normalized = np.clip(
        (np.asarray(temperature_k, dtype=np.float32) - temperature_min_k)
        / max(temperature_max_k - temperature_min_k, 1e-6),
        0.0,
        1.0,
    )
    encoded16 = np.round(normalized * 65535.0).astype(np.uint16)
    rgba = np.empty((*encoded16.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (encoded16 >> 8).astype(np.uint8)
    rgba[..., 1] = (encoded16 & 255).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = 255
    return rgba


def compute_temperature_range(
    raw_dataset: netCDF4.Dataset,
    target_time_indices: list[int],
    level_indices: list[int],
) -> tuple[float, float]:
    variable = raw_dataset.variables[DATASET_VARIABLE]
    data_min = np.inf
    data_max = -np.inf

    for time_index in target_time_indices:
        for level_index in level_indices:
            field = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
            data_min = min(data_min, float(np.nanmin(field)))
            data_max = max(data_max, float(np.nanmax(field)))

    return data_min, data_max


def build_timestamp_entry(
    output_dir: Path,
    raw_dataset: netCDF4.Dataset,
    contents: DatasetContents,
    timestamp: str,
    time_index: int,
    level_indices: list[int],
    temperature_min_k: float,
    temperature_max_k: float,
) -> dict:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)
    variable = raw_dataset.variables[DATASET_VARIABLE]
    levels: list[dict[str, str | float]] = []

    for level_index in level_indices:
        pressure_hpa = float(contents.pressure_levels_hpa[level_index])
        field = np.asarray(variable[time_index, level_index, :, :], dtype=np.float32)
        encoded = encode_temperature_to_rgba_uint8(
            field,
            temperature_min_k=temperature_min_k,
            temperature_max_k=temperature_max_k,
        )
        image_name = f"temperature-{int(round(pressure_hpa))}hpa.png"
        image_path = frame_dir / image_name
        Image.fromarray(encoded, mode="RGBA").save(image_path, optimize=True)
        levels.append(
            {
                "pressure_hpa": pressure_hpa,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "temperature_min_k": float(np.nanmin(field)),
                "temperature_max_k": float(np.nanmax(field)),
            }
        )

    levels.sort(key=lambda entry: float(entry["pressure_hpa"]))
    return {
        "timestamp": timestamp,
        "levels": levels,
    }


def build_manifest(
    contents: DatasetContents,
    entries: list[dict],
    level_indices: list[int],
    temperature_min_k: float,
    temperature_max_k: float,
) -> dict:
    pressures = sorted(float(contents.pressure_levels_hpa[index]) for index in level_indices)
    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "display_units": "K",
        "rendering": {
            "kind": "full-field-pressure-slice",
            "filtering": "none",
            "color_scale": "global-min-blue-to-global-max-red",
            "encoding": "normalized-temperature-uint16-packed-rg",
        },
        "temperature_range_k": {
            "min": temperature_min_k,
            "max": temperature_max_k,
        },
        "pressure_window_hpa": {
            "min": min(pressures),
            "max": max(pressures),
        },
        "pressure_levels_hpa": pressures,
        "grid": {
            "latitude_count": int(contents.latitudes_deg.size),
            "longitude_count": int(contents.longitudes_deg.size),
            "latitude_step_degrees": coordinate_step_degrees(contents.latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(contents.longitudes_deg),
            "latitude_min_degrees": float(np.nanmin(contents.latitudes_deg)),
            "latitude_max_degrees": float(np.nanmax(contents.latitudes_deg)),
            "longitude_min_degrees": float(np.nanmin(contents.longitudes_deg)),
            "longitude_max_degrees": float(np.nanmax(contents.longitudes_deg)),
        },
        "timestamps": entries,
    }


def split_dateline_segments(points: list[tuple[float, float]]) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for lon, lat in points:
        if previous_lon is not None and abs(lon - previous_lon) > 180:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((lon, lat))
        previous_lon = lon

    if len(segment) >= 2:
        yield segment


def lon_lat_to_pixel(
    lon: float,
    lat: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * (width - 1)
    y = (90.0 - lat) / 180.0 * (height - 1)
    return x, y


def draw_border_ring(
    draw: ImageDraw.ImageDraw,
    ring: list[list[float]],
    width: int,
    height: int,
) -> None:
    points = [(float(lon), float(lat)) for lon, lat, *_ in ring]
    for segment in split_dateline_segments(points):
        pixels = [lon_lat_to_pixel(lon, lat, width, height) for lon, lat in segment]
        draw.line(pixels, fill=(255, 255, 255, 220), width=2, joint="curve")


def write_border_texture(output_dir: Path, width: int, height: int) -> str | None:
    geojson_path = (REPO_ROOT / DEFAULT_BORDER_GEOJSON_PATH).resolve()
    if not geojson_path.exists():
        return None

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue

        for polygon in polygons:
            for ring in polygon:
                draw_border_ring(draw, ring, width, height)

    border_path = output_dir / "borders.png"
    image.save(border_path, optimize=True)
    return str(border_path.relative_to(output_dir)).replace("\\", "/")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    contents = load_dataset_contents(dataset_path)
    target_timestamps = resolve_target_timestamps(
        contents.timestamps,
        args.include_timestamps,
    )
    if not target_timestamps:
        raise ValueError("No matching timestamps were selected for export.")

    level_indices = pressure_level_indices(
        contents.pressure_levels_hpa,
        pressure_min_hpa=args.pressure_min_hpa,
        pressure_max_hpa=args.pressure_max_hpa,
    )
    if not level_indices:
        raise ValueError("No matching pressure levels were selected for export.")

    target_time_indices = [contents.timestamps.index(timestamp) for timestamp in target_timestamps]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    raw_dataset = netCDF4.Dataset(dataset_path)
    try:
        temperature_min_k, temperature_max_k = compute_temperature_range(
            raw_dataset=raw_dataset,
            target_time_indices=target_time_indices,
            level_indices=level_indices,
        )
        entries = [
            build_timestamp_entry(
                output_dir=output_dir,
                raw_dataset=raw_dataset,
                contents=contents,
                timestamp=timestamp,
                time_index=time_index,
                level_indices=level_indices,
                temperature_min_k=temperature_min_k,
                temperature_max_k=temperature_max_k,
            )
            for timestamp, time_index in zip(target_timestamps, target_time_indices)
        ]
    finally:
        raw_dataset.close()

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        level_indices=level_indices,
        temperature_min_k=temperature_min_k,
        temperature_max_k=temperature_max_k,
    )
    border_texture = write_border_texture(
        output_dir=output_dir,
        width=int(contents.longitudes_deg.size),
        height=int(contents.latitudes_deg.size),
    )
    if border_texture is not None:
        manifest["border_texture"] = border_texture
    write_json(output_dir / "index.json", manifest)
    print(
        "Built temperature slice assets:",
        f"{len(entries)} timestamps",
        f"{len(level_indices)} pressure levels",
        f"{temperature_min_k:.2f}-{temperature_max_k:.2f} K",
        f"-> {format_display_path(output_dir)}",
    )


if __name__ == "__main__":
    main()
