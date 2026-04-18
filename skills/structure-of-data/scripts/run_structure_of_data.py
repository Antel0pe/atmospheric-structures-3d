from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.representation_lab import run_structure_of_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repo-local structure-of-data skill. This is a meteorological "
            "diagnostic for understanding how a field is organized in value, height, "
            "latitude/longitude, and imbalance before attempting a 3D extraction."
        )
    )
    parser.add_argument("--field", required=True, help="Field alias: q, rh, t, theta, etc.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional NetCDF dataset path.")
    parser.add_argument("--variable", type=str, default=None, help="Optional dataset variable name.")
    parser.add_argument("--timestamp", type=str, default=None, help="ISO minute timestamp, e.g. 2021-11-08T12:00.")
    parser.add_argument(
        "--pressure-levels",
        type=str,
        default="",
        help="Comma-separated pressure levels to keep, e.g. 1000,925,850,700,500.",
    )
    parser.add_argument(
        "--anomaly",
        choices=("none", "lat_mean", "climatology"),
        default="none",
        help="Optional anomaly transform applied before the structural analysis.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Horizontal Gaussian smoothing sigma in grid cells.",
    )
    parser.add_argument(
        "--derived",
        type=str,
        default=None,
        help="Optional derived field, e.g. theta or gradient.",
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        default=None,
        help="Optional climatology dataset for anomaly=climatology.",
    )
    parser.add_argument("--latitude-stride", type=int, default=2, help="Spatial stride for latitude.")
    parser.add_argument("--longitude-stride", type=int, default=2, help="Spatial stride for longitude.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for plots and any explicitly saved summary files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and return a text-only chat summary.",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Also write summary.md and summary.json into the artifact directory or cache run directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary JSON instead of the chat-style report.",
    )
    return parser.parse_args()


def parse_pressure_levels(value: str) -> list[float] | None:
    if not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    skill_root = Path(__file__).resolve().parents[1]
    summary = run_structure_of_data(
        skill_root=skill_root,
        field=args.field,
        dataset_path=args.dataset,
        variable_name=args.variable,
        timestamp=args.timestamp,
        pressure_levels_hpa=parse_pressure_levels(args.pressure_levels),
        anomaly=args.anomaly,
        smoothing=args.smoothing,
        derived=args.derived,
        climatology_path=args.climatology,
        latitude_stride=args.latitude_stride,
        longitude_stride=args.longitude_stride,
        artifact_dir=args.artifact_dir,
        make_plots=not args.no_plots,
        save_summary=args.save_summary,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(summary["chat_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
