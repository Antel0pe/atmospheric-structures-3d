from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simple_voxel_builder import (
    clear_output_dir,
    coordinate_step_degrees,
    timestamp_to_slug,
)


OUTPUT_VERSION = 1
DATASET_VARIABLE = "tp"
DEFAULT_DATASET_PATH = Path("data/era5_total-precipitation_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/precipitation-radar")
DEFAULT_MAX_MM = 20.0
DEFAULT_THRESHOLDS_MM = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class DatasetContents:
    dataset_path: Path
    variable_name: str
    units: str
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    timestamps: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static precipitation radar textures from ERA5 total "
            "precipitation."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the source ERA5 total precipitation NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated radar textures will be written.",
    )
    parser.add_argument(
        "--max-mm",
        type=float,
        default=DEFAULT_MAX_MM,
        help="Upper display bound in mm before encoded values saturate.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default="",
        help=(
            "Optional comma-separated ISO minute timestamps to build. "
            "Builds every timestamp when omitted."
        ),
    )
    return parser.parse_args()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="m")
    if not text.endswith("Z"):
        return text
    return text[:-1]


def resolve_dataset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
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


def build_timestamp_entry(
    output_dir: Path,
    timestamp: str,
    encoded_uint8: np.ndarray,
) -> dict[str, str]:
    slug = timestamp_to_slug(timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    image_path = frame_dir / "radar.png"
    Image.fromarray(encoded_uint8, mode="L").save(image_path, optimize=True)

    return {
        "timestamp": timestamp,
        "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
    }


def encode_precipitation_slice_to_uint8(
    precipitation_m: np.ndarray,
    max_mm: float,
) -> np.ndarray:
    precipitation_mm = np.maximum(
        np.asarray(precipitation_m, dtype=np.float32) * 1000.0,
        0.0,
    )
    scaled = np.clip(precipitation_mm / max(max_mm, 1e-6), 0.0, 1.0)
    encoded = np.sqrt(scaled)
    return np.round(encoded * 255.0).astype(np.uint8)


def build_manifest(
    contents: DatasetContents,
    entries: list[dict[str, str]],
    max_mm: float,
) -> dict:
    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "display_units": "mm",
        "encoding": {
            "curve": "sqrt",
            "max_mm": max_mm,
        },
        "thresholds_mm": [float(value) for value in DEFAULT_THRESHOLDS_MM],
        "grid": {
            "latitude_count": int(contents.latitudes_deg.size),
            "longitude_count": int(contents.longitudes_deg.size),
            "latitude_step_degrees": coordinate_step_degrees(contents.latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(contents.longitudes_deg),
        },
        "timestamps": entries,
    }


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

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    raw_dataset = netCDF4.Dataset(dataset_path)
    try:
        variable = raw_dataset.variables[DATASET_VARIABLE]
        entries: list[dict[str, str]] = []
        for time_index, timestamp in enumerate(contents.timestamps):
            if timestamp not in target_timestamps:
                continue

            field = np.asarray(variable[time_index, :, :], dtype=np.float32)
            encoded_uint8 = encode_precipitation_slice_to_uint8(
                precipitation_m=field,
                max_mm=args.max_mm,
            )
            entries.append(
                build_timestamp_entry(
                    output_dir=output_dir,
                    timestamp=timestamp,
                    encoded_uint8=encoded_uint8,
                )
            )
    finally:
        raw_dataset.close()

    manifest = build_manifest(
        contents=contents,
        entries=entries,
        max_mm=args.max_mm,
    )
    write_json(output_dir / "index.json", manifest)
    print(
        "Built precipitation radar assets:",
        f"{len(entries)} timestamps",
        f"-> {output_dir}",
    )


if __name__ == "__main__":
    main()
