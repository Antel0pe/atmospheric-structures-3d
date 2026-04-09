from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.moisture_structures import (
    BuildConfig,
    SUPPORTED_SEGMENTATION_MODES,
    build_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build browser-ready 3D moisture structure assets from ERA5 specific humidity."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/era5_specific-humidity_2021-11_08-12.nc"),
        help="Path to the source specific humidity NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public/moisture-structures"),
        help="Directory where the generated manifest and binary assets will be written.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Per-pressure quantile used to classify humid voxels.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=1_024,
        help="Minimum voxel count required for a connected component to be kept.",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=7,
        help="Number of timestamps to process together when reading the source file.",
    )
    parser.add_argument(
        "--base-radius",
        type=float,
        default=100.0,
        help="Base world radius of the globe mesh.",
    )
    parser.add_argument(
        "--vertical-span",
        type=float,
        default=12.0,
        help="World units spanning 1000 hPa to 1 hPa.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.6,
        help="Gaussian smoothing sigma applied before marching cubes.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("marching-cubes", "voxel-faces"),
        default="marching-cubes",
        help="Mesh generation strategy for each connected moisture component.",
    )
    parser.add_argument(
        "--closing-radius-cells",
        type=int,
        default=1,
        help="Radius in grid cells for the binary closing pass. Use 0 to disable it.",
    )
    parser.add_argument(
        "--opening-radius-cells",
        type=int,
        default=1,
        help="Radius in grid cells for the binary opening pass after closing. Use 0 to disable it.",
    )
    parser.add_argument(
        "--segmentation-mode",
        type=str,
        default="p95-close",
        choices=SUPPORTED_SEGMENTATION_MODES,
        help="Label written into the manifest to identify the segmentation variant.",
    )
    parser.add_argument(
        "--write-footprints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to emit per-component footprint assets alongside the mesh payloads.",
    )
    parser.add_argument(
        "--limit-timestamps",
        type=int,
        default=None,
        help="Optional cap on the number of timestamps to process. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--include-timestamps",
        type=str,
        default="",
        help="Comma-separated ISO minute timestamps to build, for example 2021-11-08T12:00.",
    )
    parser.add_argument(
        "--mesh-smoothing-iterations",
        type=int,
        default=0,
        help="Number of Taubin mesh smoothing iterations applied after marching cubes.",
    )
    parser.add_argument(
        "--mesh-smoothing-lambda",
        type=float,
        default=0.45,
        help="Taubin smoothing lambda step.",
    )
    parser.add_argument(
        "--mesh-smoothing-mu",
        type=float,
        default=-0.47,
        help="Taubin smoothing mu step.",
    )
    parser.add_argument(
        "--mesh-island-min-vertices",
        type=int,
        default=0,
        help=(
            "Drop disconnected marching-cubes mesh islands smaller than this many "
            "vertices after smoothing. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--voxel-exterior-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For voxel-face geometry, emit only faces adjacent to outside-connected "
            "empty space. This removes interior cavity walls."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        threshold_quantile=args.threshold_quantile,
        min_component_size=args.min_component_size,
        time_window=args.time_window,
        base_radius=args.base_radius,
        vertical_span=args.vertical_span,
        gaussian_sigma=args.gaussian_sigma,
        geometry_mode=args.geometry_mode,
        closing_radius_cells=args.closing_radius_cells,
        opening_radius_cells=args.opening_radius_cells,
        segmentation_mode=args.segmentation_mode,
        write_footprints=args.write_footprints,
        limit_timestamps=args.limit_timestamps,
        include_timestamps=tuple(
            value.strip()
            for value in args.include_timestamps.split(",")
            if value.strip()
        )
        or None,
        mesh_smoothing_iterations=args.mesh_smoothing_iterations,
        mesh_smoothing_lambda=args.mesh_smoothing_lambda,
        mesh_smoothing_mu=args.mesh_smoothing_mu,
        mesh_island_min_vertices=args.mesh_island_min_vertices,
        voxel_exterior_only=args.voxel_exterior_only,
    )
    manifest = build_assets(config)
    print(
        "Built moisture structures:",
        f"{len(manifest['timestamps'])} timestamps",
        f"-> {config.output_dir}",
    )


if __name__ == "__main__":
    main()
