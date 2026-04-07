from __future__ import annotations

"""
Simple voxel builder scaffold.

This file is intentionally small in scope.
Right now it only handles the "boilerplate" layer:

1. figure out which dataset file to open
2. open the dataset in a way that matches the existing builder
3. expose the basic coordinate arrays and metadata we will need later
4. optionally read a single 3D time slice for inspection

The goal is to make the data-loading path easy to read before we add any
thresholding, component labeling, or voxel-face generation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr


DATASET_VARIABLE = "q"
DEFAULT_DATASET_PATH = Path("data/era5_specific-humidity_2021-11_08-12.nc")
DEFAULT_OUTPUT_DIR = Path("public/moisture-structures/variants/simple-voxel-shell")
DEFAULT_DEBUG_TIMESTAMP = "2021-11-08T12:00"
OUTPUT_VERSION = 1


@dataclass(frozen=True)
class DatasetHandles:
    """
    The two dataset objects we keep open while working.

    Why keep both?
    - xarray is nice for coordinates and metadata.
    - netCDF4 gives us straightforward low-level slicing for future steps.
    """

    dataset: xr.Dataset
    raw_dataset: netCDF4.Dataset


@dataclass(frozen=True)
class DatasetContents:
    """
    The small set of arrays and metadata we care about at the start.

    This is the "what did we find in the file?" object.
    Later steps can build on this without reopening or re-parsing the dataset.
    """

    dataset_path: Path
    variable_name: str
    units: str
    pressure_levels_hpa: np.ndarray
    latitudes_deg: np.ndarray
    longitudes_deg: np.ndarray
    timestamps: list[str]
    time_count: int
    level_count: int
    latitude_count: int
    longitude_count: int


@dataclass(frozen=True)
class NaiveVoxelMesh:
    """
    The most literal mesh representation of the kept voxels.

    For now we are doing the naive thing:
    - every occupied cell becomes a full cube
    - we do not try to remove shared interior faces
    - we do not merge neighboring cubes

    The arrays are intentionally shaped to resemble what the frontend already
    knows how to consume later:
    - positions: flat float32 XYZ values
    - indices: flat uint32 triangle indices
    """

    positions: np.ndarray
    indices: np.ndarray
    cube_count: int
    vertex_count: int
    triangle_count: int


@dataclass(frozen=True)
class NaiveVoxelAssetPayload:
    """
    Everything needed to write one debug asset frame.

    We keep this separate from the raw mesh because the written asset also needs:
    - metadata for the frontend
    - manifest information
    - the exact timestamp string
    """

    timestamp: str
    positions: np.ndarray
    indices: np.ndarray
    threshold_value: float
    voxel_count: int
    component_metadata: dict


def main() -> None:
    """
    Read top-to-bottom:

    Step 1: parse the command-line inputs.
    Step 2: resolve the dataset path and make sure it exists.
    Step 3: open the source file.
    Step 4: pull out the coordinates and lightweight metadata we need.
    Step 5: print a compact summary so we can confirm the file is what we expect.
    Step 6: optionally read one 3D field if we want to inspect actual values.
    Step 7: optionally threshold that field and build a simple boolean mask.
    Step 8: optionally write one debug asset frame for the frontend.

    All of the helper functions used here are defined below.
    """

    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)

    handles = open_dataset_handles(dataset_path)
    try:
        contents = load_dataset_contents(handles, dataset_path)
        print_dataset_summary(contents)

        if args.preview_time_index is not None:
            field = read_field_at_time_index(handles, time_index=args.preview_time_index)
            print_field_summary(field, time_index=args.preview_time_index)

            keep_mask, threshold_value = build_top_percent_mask(
                field,
                keep_quantile=args.keep_quantile,
            )
            print_mask_summary(
                keep_mask,
                threshold_value=threshold_value,
                keep_quantile=args.keep_quantile,
            )

            naive_mesh = build_exposed_face_mesh_from_mask(
                keep_mask=keep_mask,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
            )
            print_naive_mesh_summary(naive_mesh)

        if args.write_assets:
            target_timestamp = args.timestamp
            time_index = timestamp_to_time_index(contents.timestamps, target_timestamp)
            field = read_field_at_time_index(handles, time_index=time_index)
            keep_mask, threshold_value = build_top_percent_mask(
                field,
                keep_quantile=args.keep_quantile,
            )
            naive_mesh = build_exposed_face_mesh_from_mask(
                keep_mask=keep_mask,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
            )
            payload = build_single_component_asset_payload(
                timestamp=target_timestamp,
                field=field,
                keep_mask=keep_mask,
                mesh=naive_mesh,
                threshold_value=threshold_value,
                pressure_levels_hpa=contents.pressure_levels_hpa,
                latitudes_deg=contents.latitudes_deg,
                longitudes_deg=contents.longitudes_deg,
            )
            output_dir = args.output_dir.expanduser().resolve()
            write_single_timestamp_assets(
                output_dir=output_dir,
                payload=payload,
                contents=contents,
                keep_quantile=args.keep_quantile,
            )
            print(f"Wrote debug assets for {target_timestamp} -> {output_dir}")
    finally:
        close_dataset_handles(handles)


def parse_args() -> argparse.Namespace:
    """
    Keep the CLI small for now.

    We only need:
    - the dataset path
    - an optional time index to prove we can read one field cleanly
    - a quantile that says how aggressive our threshold should be
    """

    parser = argparse.ArgumentParser(
        description=(
            "Readable scaffold for the voxel builder. "
            "This version only opens the ERA5 specific humidity file and "
            "summarizes its contents."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the source ERA5 specific humidity NetCDF file.",
    )
    parser.add_argument(
        "--preview-time-index",
        type=int,
        default=None,
        help=(
            "Optional time index to read as a single 3D field. "
            "Useful for sanity-checking the file before we build voxels."
        ),
    )
    parser.add_argument(
        "--keep-quantile",
        type=float,
        default=0.95,
        help=(
            "Keep values at or above this quantile for one timestamp. "
            "The default 0.95 means: keep the top 5 percent of finite values."
        ),
    )
    parser.add_argument(
        "--write-assets",
        action="store_true",
        help=(
            "Write one debug asset frame in the same public asset format the "
            "frontend moisture layer already consumes."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=DEFAULT_DEBUG_TIMESTAMP,
        help=(
            "Timestamp to write when --write-assets is enabled. "
            "Defaults to the viewer's dev-mode start timestamp."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the debug frontend assets.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: Path) -> Path:
    """
    Convert the user input into an absolute path and fail early if it is missing.

    Doing this up front keeps the rest of the script simpler:
    after this point, we can assume the path is real.
    """

    dataset_path = dataset_arg.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    return dataset_path


def open_dataset_handles(dataset_path: Path) -> DatasetHandles:
    """
    Open the same file in two ways.

    This mirrors the existing production builder on purpose, because we will
    probably want the same access pattern later when we start reading time windows.
    """

    dataset = xr.open_dataset(dataset_path, chunks={})
    raw_dataset = netCDF4.Dataset(dataset_path, mode="r")
    return DatasetHandles(dataset=dataset, raw_dataset=raw_dataset)


def load_dataset_contents(
    handles: DatasetHandles,
    dataset_path: Path,
    variable_name: str = DATASET_VARIABLE,
) -> DatasetContents:
    """
    Pull out the coordinates and metadata we will need in later steps.

    Important detail:
    this does not read the full 4D humidity variable into memory.
    It only reads the smaller coordinate arrays and basic metadata.
    """

    dataset = handles.dataset
    variable = dataset[variable_name]

    pressure_levels_hpa = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
    latitudes_deg = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
    longitudes_deg = np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
    timestamps = [
        timestamp_to_iso_minute(value)
        for value in np.asarray(dataset.coords["valid_time"].values)
    ]

    return DatasetContents(
        dataset_path=dataset_path,
        variable_name=variable_name,
        units=str(variable.attrs.get("units", "")),
        pressure_levels_hpa=pressure_levels_hpa,
        latitudes_deg=latitudes_deg,
        longitudes_deg=longitudes_deg,
        timestamps=timestamps,
        time_count=int(variable.sizes["valid_time"]),
        level_count=int(variable.sizes["pressure_level"]),
        latitude_count=int(variable.sizes["latitude"]),
        longitude_count=int(variable.sizes["longitude"]),
    )


def print_dataset_summary(contents: DatasetContents) -> None:
    """
    Print the facts that are useful when orienting ourselves in a new file.

    This is deliberately plain text so it is easy to scan in the terminal.
    """

    print("Opened dataset:")
    print(f"  file: {contents.dataset_path}")
    print(f"  variable: {contents.variable_name}")
    print(f"  units: {contents.units or '(missing)'}")
    print(
        "  grid shape:"
        f" time={contents.time_count},"
        f" levels={contents.level_count},"
        f" lat={contents.latitude_count},"
        f" lon={contents.longitude_count}"
    )
    print(
        "  coordinate ranges:"
        f" pressure={contents.pressure_levels_hpa.min():.1f}..{contents.pressure_levels_hpa.max():.1f} hPa,"
        f" latitude={contents.latitudes_deg.min():.1f}..{contents.latitudes_deg.max():.1f} deg,"
        f" longitude={contents.longitudes_deg.min():.1f}..{contents.longitudes_deg.max():.1f} deg"
    )
    if contents.timestamps:
        print(f"  first timestamp: {contents.timestamps[0]}")
        print(f"  last timestamp:  {contents.timestamps[-1]}")


def read_field_at_time_index(
    handles: DatasetHandles,
    time_index: int,
    variable_name: str = DATASET_VARIABLE,
) -> np.ndarray:
    """
    Read one 3D field from disk as a NumPy array.

    Shape of the returned array:
    - (pressure_level, latitude, longitude)

    We are not using this for voxel building yet.
    It is here because reading one field cleanly is the next basic building block.
    """

    variable = handles.raw_dataset.variables[variable_name]
    time_count = int(variable.shape[0])

    if time_index < 0 or time_index >= time_count:
        raise IndexError(
            f"time_index must be between 0 and {time_count - 1}, got {time_index}"
        )

    return np.asarray(variable[time_index, :, :, :], dtype=np.float32)


def print_field_summary(field: np.ndarray, time_index: int) -> None:
    """
    Print a quick sanity-check summary for one loaded 3D field.

    This helps us answer:
    - did the slice load?
    - is the shape what we expected?
    - are the values finite and in a reasonable range?
    """

    finite_mask = np.isfinite(field)
    finite_values = field[finite_mask]

    print(f"Preview field at time index {time_index}:")
    print(f"  shape: {field.shape}")
    print(f"  finite values: {int(finite_mask.sum())} / {field.size}")

    if finite_values.size == 0:
        print("  all values are non-finite")
        return

    print(f"  min: {float(finite_values.min()):.6g}")
    print(f"  max: {float(finite_values.max()):.6g}")
    print(f"  mean: {float(finite_values.mean()):.6g}")


def build_top_percent_mask(
    field: np.ndarray,
    keep_quantile: float = 0.95,
) -> tuple[np.ndarray, float]:
    """
    Build the first real "voxel candidate" mask for one timestamp.

    What this function does:
    - look at every finite humidity value in the 3D field
    - compute the chosen quantile across that whole timestamp
    - keep only values at or above that cutoff

    With the default keep_quantile=0.95:
    - values in the top 5 percent are kept
    - values in the bottom 95 percent are discarded

    Output:
    - keep_mask: boolean array with the same shape as field
    - threshold_value: the actual humidity value used as the cutoff

    We are deliberately keeping this very simple for now:
    - one timestamp only
    - one global threshold across the whole 3D field
    - no per-pressure thresholds yet
    - no morphology yet
    """

    if not 0.0 <= keep_quantile <= 1.0:
        raise ValueError(
            f"keep_quantile must be between 0 and 1, got {keep_quantile}"
        )

    finite_mask = np.isfinite(field)
    finite_values = field[finite_mask]

    if finite_values.size == 0:
        raise ValueError("Cannot build a threshold mask because the field has no finite values.")

    threshold_value = float(np.quantile(finite_values, keep_quantile))
    keep_mask = np.asarray(finite_mask & (field >= threshold_value), dtype=bool)
    return keep_mask, threshold_value


def print_mask_summary(
    keep_mask: np.ndarray,
    threshold_value: float,
    keep_quantile: float,
) -> None:
    """
    Print the threshold result in a way that is easy to reason about.

    This is the first place where we can answer:
    - what cutoff value did the quantile produce?
    - how many voxels survived?
    - what fraction of the 3D field is still "on"?
    """

    kept_count = int(keep_mask.sum())
    total_count = int(keep_mask.size)
    kept_fraction = kept_count / total_count if total_count else 0.0

    print("Threshold mask:")
    print(f"  keep quantile: {keep_quantile:.2f}")
    print(f"  threshold value: {threshold_value:.6g}")
    print(f"  kept voxels: {kept_count} / {total_count}")
    print(f"  kept fraction: {kept_fraction:.3%}")


def build_exposed_face_mesh_from_mask(
    keep_mask: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> NaiveVoxelMesh:
    """
    Turn the kept mask into an exposed-face voxel shell.

    Mental model:
    - each humidity sample sits at the center of one 3D grid cell
    - we build the cell edges by going halfway toward neighboring sample centers
    - if the sample is kept by the threshold mask, it contributes only the faces
      that touch empty space

    This is the simplest useful "deduplicate faces" pass:
    - shared interior faces are removed
    - the outer shell stays exactly voxel-aligned
    - no smoothing or fancy meshing is applied
    """

    radius_values = build_radius_lookup_from_pressure_levels(pressure_levels_hpa)
    radius_bounds = build_axis_bounds(radius_values)
    latitude_bounds = build_axis_bounds(latitudes_deg)
    longitude_bounds = build_axis_bounds(longitudes_deg)

    level_count, latitude_count, longitude_count = keep_mask.shape
    occupied_cells = np.argwhere(keep_mask)
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    positions: list[float] = []
    indices: list[int] = []

    for level_index, latitude_index, longitude_index in occupied_cells:
        r0 = int(level_index)
        r1 = r0 + 1
        a0 = int(latitude_index)
        a1 = a0 + 1
        o0 = int(longitude_index)
        o1 = o0 + 1

        west_neighbor = (o0 - 1) % longitude_count
        east_neighbor = (o0 + 1) % longitude_count

        if r0 == 0 or not keep_mask[r0 - 1, a0, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a1, o0), (r0, a1, o1), (r0, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if r0 == level_count - 1 or not keep_mask[r0 + 1, a0, o0]:
            append_quad(
                corners=[(r1, a0, o0), (r1, a0, o1), (r1, a1, o1), (r1, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == 0 or not keep_mask[r0, a0 - 1, o0]:
            append_quad(
                corners=[(r0, a0, o0), (r0, a0, o1), (r1, a0, o1), (r1, a0, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if a0 == latitude_count - 1 or not keep_mask[r0, a0 + 1, o0]:
            append_quad(
                corners=[(r0, a1, o0), (r1, a1, o0), (r1, a1, o1), (r0, a1, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not keep_mask[r0, a0, west_neighbor]:
            append_quad(
                corners=[(r0, a0, o0), (r1, a0, o0), (r1, a1, o0), (r0, a1, o0)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )
        if not keep_mask[r0, a0, east_neighbor]:
            append_quad(
                corners=[(r0, a0, o1), (r0, a1, o1), (r1, a1, o1), (r1, a0, o1)],
                vertex_lookup=vertex_lookup,
                positions=positions,
                indices=indices,
                radius_bounds=radius_bounds,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            )

    positions_array = np.asarray(positions, dtype=np.float32)
    indices_array = np.asarray(indices, dtype=np.uint32)

    return NaiveVoxelMesh(
        positions=positions_array,
        indices=indices_array,
        cube_count=int(occupied_cells.shape[0]),
        vertex_count=int(positions_array.size // 3),
        triangle_count=int(indices_array.size // 3),
    )


def append_quad(
    corners: list[tuple[int, int, int]],
    vertex_lookup: dict[tuple[int, int, int], int],
    positions: list[float],
    indices: list[int],
    radius_bounds: np.ndarray,
    latitude_bounds: np.ndarray,
    longitude_bounds: np.ndarray,
) -> None:
    """
    Append one quad and reuse matching corner vertices when possible.
    """

    quad_indices: list[int] = []

    for key in corners:
        vertex_index = vertex_lookup.get(key)
        if vertex_index is None:
            radius_index, latitude_index, longitude_index = key
            vertex = lat_lon_to_xyz(
                float(latitude_bounds[latitude_index]),
                float(longitude_bounds[longitude_index]),
                float(radius_bounds[radius_index]),
            )
            vertex_index = len(positions) // 3
            positions.extend(float(value) for value in vertex)
            vertex_lookup[key] = vertex_index
        quad_indices.append(vertex_index)

    indices.extend(
        [
            quad_indices[0],
            quad_indices[1],
            quad_indices[2],
            quad_indices[0],
            quad_indices[2],
            quad_indices[3],
        ]
    )


def print_naive_mesh_summary(mesh: NaiveVoxelMesh) -> None:
    """
    Print the size of the naive output mesh.

    This is useful because it gives us an immediate feel for whether the fully
    naive cube approach is tiny, manageable, or obviously too large.
    """

    print("Voxel shell mesh:")
    print(f"  cubes: {mesh.cube_count}")
    print(f"  vertices: {mesh.vertex_count}")
    print(f"  triangles: {mesh.triangle_count}")
    print(f"  positions bytes: {mesh.positions.nbytes:,}")
    print(f"  indices bytes: {mesh.indices.nbytes:,}")


def build_single_component_asset_payload(
    timestamp: str,
    field: np.ndarray,
    keep_mask: np.ndarray,
    mesh: NaiveVoxelMesh,
    threshold_value: float,
    pressure_levels_hpa: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> NaiveVoxelAssetPayload:
    """
    Wrap the naive mesh in the metadata shape expected by the current frontend.

    We intentionally publish everything as a single component.
    That keeps this debug path simple:
    - one component
    - one mesh
    - no footprints
    - no component splitting or pruning
    """

    occupied_coords = np.argwhere(keep_mask)
    if occupied_coords.size == 0:
        raise ValueError("Cannot write assets because no voxels survived the threshold.")

    pressure_indices = occupied_coords[:, 0]
    latitude_indices = occupied_coords[:, 1]
    longitude_indices = occupied_coords[:, 2]
    kept_values = field[keep_mask]

    component_metadata = {
        "id": 0,
        "vertex_offset": 0,
        "vertex_count": int(mesh.vertex_count),
        "index_offset": 0,
        "index_count": int(mesh.indices.size),
        "voxel_count": int(keep_mask.sum()),
        "mean_specific_humidity": float(np.mean(kept_values)),
        "max_specific_humidity": float(np.max(kept_values)),
        "pressure_min_hpa": float(np.min(pressure_levels_hpa[pressure_indices])),
        "pressure_max_hpa": float(np.max(pressure_levels_hpa[pressure_indices])),
        "latitude_min_deg": float(np.min(latitudes_deg[latitude_indices])),
        "latitude_max_deg": float(np.max(latitudes_deg[latitude_indices])),
        "longitude_min_deg": float(np.min(longitudes_deg[longitude_indices])),
        "longitude_max_deg": float(np.max(longitudes_deg[longitude_indices])),
        "wraps_longitude_seam": bool(keep_mask[..., 0].any() and keep_mask[..., -1].any()),
    }

    return NaiveVoxelAssetPayload(
        timestamp=timestamp,
        positions=mesh.positions,
        indices=mesh.indices,
        threshold_value=threshold_value,
        voxel_count=int(keep_mask.sum()),
        component_metadata=component_metadata,
    )


def lat_lon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    """
    Convert one spherical coordinate on the globe into XYZ.

    This matches the coordinate convention used by the existing viewer code.
    """

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(-(lon_deg + 270.0))
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.sin(lat)
    z = radius * np.cos(lat) * np.sin(lon)
    return np.array([x, y, z], dtype=np.float32)


def pressure_to_standard_atmosphere_height_m(pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Convert pressure levels into approximate height in meters.

    We are not aiming for perfect physical realism here.
    We just need a monotonic mapping so each pressure level gets a radius.
    """

    safe_pressure = np.maximum(np.asarray(pressure_hpa, dtype=np.float64), 1.0)
    return 44330.0 * (1.0 - np.power(safe_pressure / 1013.25, 0.1903))


def build_radius_lookup_from_pressure_levels(
    pressure_levels_hpa: np.ndarray,
    base_radius: float = 100.0,
    vertical_span: float = 12.0,
) -> np.ndarray:
    """
    Map pressure levels onto globe radii.

    This gives us one rendered radius for each pressure level center.
    Later we turn those center values into cell edges using build_axis_bounds().
    """

    heights_m = pressure_to_standard_atmosphere_height_m(pressure_levels_hpa)
    min_height_m = float(pressure_to_standard_atmosphere_height_m(np.array([1000.0]))[0])
    max_height_m = float(pressure_to_standard_atmosphere_height_m(np.array([1.0]))[0])
    scale = vertical_span / max(max_height_m - min_height_m, 1e-9)
    return (base_radius + (heights_m - min_height_m) * scale).astype(np.float32)


def build_axis_bounds(values: np.ndarray) -> np.ndarray:
    """
    Turn cell-center coordinates into cell-edge coordinates.

    This is exactly the midpoint idea you described:
    - the edge between two neighboring sample centers sits halfway between them
    - the outermost edges are extrapolated outward by half a grid step

    Example:
    - centers: [0.00, 0.25, 0.50]
    - edges:   [-0.125, 0.125, 0.375, 0.625]
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 1:
        step = 1.0
        return np.array(
            [values[0] - step * 0.5, values[0] + step * 0.5],
            dtype=np.float64,
        )

    midpoints = (values[:-1] + values[1:]) * 0.5
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]]).astype(np.float64)


def coordinate_step_degrees(values: np.ndarray) -> float | None:
    """
    Return the step size for a regularly spaced lat or lon axis.

    The manifest uses this only as lightweight descriptive metadata.
    """

    axis = np.asarray(values, dtype=np.float32)
    if axis.size < 2:
        return None
    return float(abs(axis[1] - axis[0]))


def write_single_timestamp_assets(
    output_dir: Path,
    payload: NaiveVoxelAssetPayload,
    contents: DatasetContents,
    keep_quantile: float,
) -> None:
    """
    Write one debug timestamp in the same general format as the production assets.

    Output layout:
    - index.json
    - YYYY-MM-DDTHH-MM-SS/metadata.json
    - YYYY-MM-DDTHH-MM-SS/positions.bin
    - YYYY-MM-DDTHH-MM-SS/indices.bin

    We omit footprints for this debug path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)

    entry = write_single_timestamp_frame(output_dir, payload)
    manifest = build_single_timestamp_manifest(
        contents=contents,
        entry=entry,
        keep_quantile=keep_quantile,
        threshold_value=payload.threshold_value,
    )
    write_json(output_dir / "index.json", manifest)


def write_single_timestamp_frame(
    output_dir: Path,
    payload: NaiveVoxelAssetPayload,
) -> dict:
    """
    Write the binary and metadata files for one frame.
    """

    slug = timestamp_to_slug(payload.timestamp)
    frame_dir = output_dir / slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    positions_path = frame_dir / "positions.bin"
    indices_path = frame_dir / "indices.bin"
    metadata_path = frame_dir / "metadata.json"

    payload.positions.astype("<f4").tofile(positions_path)
    payload.indices.astype("<u4").tofile(indices_path)

    metadata = {
        "version": OUTPUT_VERSION,
        "timestamp": payload.timestamp,
        "component_count": 1,
        "vertex_count": int(payload.positions.size // 3),
        "index_count": int(payload.indices.size),
        "thresholded_voxel_count": payload.voxel_count,
        "components": [payload.component_metadata],
        "positions_file": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices_file": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
    }
    write_json(metadata_path, metadata)

    return {
        "timestamp": payload.timestamp,
        "metadata": str(metadata_path.relative_to(output_dir)).replace("\\", "/"),
        "positions": str(positions_path.relative_to(output_dir)).replace("\\", "/"),
        "indices": str(indices_path.relative_to(output_dir)).replace("\\", "/"),
        "component_count": 1,
        "vertex_count": int(payload.positions.size // 3),
        "index_count": int(payload.indices.size),
    }


def build_single_timestamp_manifest(
    contents: DatasetContents,
    entry: dict,
    keep_quantile: float,
    threshold_value: float,
) -> dict:
    """
    Build a minimal manifest that the existing frontend can load.

    Notes:
    - thresholds are repeated per pressure level because the frontend expects
      one pressure entry per level for color/radius banding
    - this is a global single-timestamp quantile, not the production
      pressure-relative threshold recipe
    """

    return {
        "version": OUTPUT_VERSION,
        "dataset": contents.dataset_path.name,
        "variable": contents.variable_name,
        "units": contents.units,
        "segmentation_mode": "simple-voxel-shell",
        "threshold_mode": {
            "kind": "global-single-timestamp-quantile",
            "quantile": keep_quantile,
            "minimum_component_size": 0,
            "threshold_seed": "single_timestamp",
            "smoothing": {
                "binary_closing_radius_cells": 0,
                "binary_opening_radius_cells": 0,
                "gaussian_sigma": 0.0,
            },
            "recipe": {
                "recipe": "simple-exposed-face-voxel-shell",
                "thresholds": {"global_q": keep_quantile},
                "preprocessing": {},
                "postprocess": {},
                "geometry_variant": "exposed-face-voxel-shell",
            },
        },
        "geometry_mode": "voxel-faces",
        "globe": {
            "base_radius": 100.0,
            "vertical_span": 12.0,
            "reference_pressure_hpa": {"min": 1.0, "max": 1000.0},
        },
        "grid": {
            "pressure_level_count": int(contents.level_count),
            "latitude_count": int(contents.latitude_count),
            "longitude_count": int(contents.longitude_count),
            "latitude_step_degrees": coordinate_step_degrees(contents.latitudes_deg),
            "longitude_step_degrees": coordinate_step_degrees(contents.longitudes_deg),
        },
        "thresholds": [
            {
                "pressure_hpa": float(pressure_hpa),
                "threshold": float(threshold_value),
            }
            for pressure_hpa in contents.pressure_levels_hpa
        ],
        "timestamps": [entry],
    }


def write_json(path: Path, payload: dict) -> None:
    """
    Small JSON writer helper so the output format stays explicit.
    """

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_output_dir(output_dir: Path) -> None:
    """
    Remove previous debug files from the target output folder.
    """

    if not output_dir.exists():
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested_dir in sorted(
                [path for path in child.rglob("*") if path.is_dir()],
                reverse=True,
            ):
                nested_dir.rmdir()
            child.rmdir()
        else:
            child.unlink()


def timestamp_to_time_index(timestamps: list[str], target_timestamp: str) -> int:
    """
    Convert a timestamp string into the matching time index in the file.
    """

    try:
        return timestamps.index(target_timestamp)
    except ValueError as exc:
        raise ValueError(f"Timestamp not found in dataset: {target_timestamp}") from exc


def close_dataset_handles(handles: DatasetHandles) -> None:
    """
    Close both dataset objects.

    Keeping this in one helper makes the cleanup step obvious in main().
    """

    handles.dataset.close()
    handles.raw_dataset.close()


def timestamp_to_iso_minute(value: np.datetime64) -> str:
    """
    Match the existing builder's timestamp formatting.

    Example:
    - input:  2021-11-08T12:00:00.000000000
    - output: 2021-11-08T12:00
    """

    text = np.datetime_as_string(value, unit="m")
    if text.endswith("Z"):
        return text[:-1]
    return text


def timestamp_to_slug(timestamp: str) -> str:
    """
    Match the existing moisture asset folder naming.
    """

    return f"{timestamp.replace(':', '-')}-00"


if __name__ == "__main__":
    main()
