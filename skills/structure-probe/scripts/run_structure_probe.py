from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.representation_lab import run_structure_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repo-local structure_probe skill. This is a fast 3D extraction "
            "diagnostic for understanding whether a representation becomes a blob, "
            "sheet, speckle field, or something worth promoting."
        )
    )
    parser.add_argument("--field", default="q", help="Field alias: q, rh, t, theta, etc.")
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
        help="Fast anomaly transform applied before probing.",
    )
    parser.add_argument("--smoothing", type=float, default=0.0, help="Horizontal Gaussian smoothing sigma.")
    parser.add_argument("--derived", type=str, default=None, help="Optional derived field, e.g. theta or gradient.")
    parser.add_argument("--climatology", type=Path, default=None, help="Optional climatology dataset.")
    parser.add_argument(
        "--method",
        choices=("threshold", "seed_grow", "gradient"),
        default="threshold",
        help="Extraction method to probe.",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=10.0,
        help="Top-share percent used by threshold, seed, or gradient extraction.",
    )
    parser.add_argument(
        "--threshold-tail",
        choices=("absolute", "high", "positive", "negative"),
        default=None,
        help="Optional threshold tail override. Defaults depend on field sign.",
    )
    parser.add_argument(
        "--grow-rule",
        type=str,
        default=None,
        help="Seed-grow rule, e.g. same-sign-relaxed-half or same-sign-above-zero.",
    )
    parser.add_argument(
        "--bridge-levels",
        type=int,
        default=0,
        help="Fill gaps up to this many levels within a vertical column.",
    )
    parser.add_argument(
        "--morphology",
        choices=("none", "open", "close"),
        default="none",
        help="Binary morphology applied after extraction.",
    )
    parser.add_argument(
        "--resolution",
        choices=("coarse", "full"),
        default="coarse",
        help="Coarse uses fast spatial striding. Full keeps the whole grid.",
    )
    parser.add_argument(
        "--structure-of-data",
        dest="structure_of_data",
        type=Path,
        default=None,
        help="Optional structure_of_data summary.json. Reuses the exact field config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary JSON instead of a compact terminal summary.",
    )
    return parser.parse_args()


def parse_pressure_levels(value: str) -> list[float] | None:
    if not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    skill_root = Path(__file__).resolve().parents[1]
    summary = run_structure_probe(
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
        method=args.method,
        threshold_percent=args.threshold_percent,
        grow_rule=args.grow_rule,
        bridge_levels=args.bridge_levels,
        morphology=args.morphology,
        resolution=args.resolution,
        threshold_tail=args.threshold_tail,
        structure_of_data_summary_path=args.structure_of_data,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    interpretation = summary["interpretation"]
    print(
        f"structure_probe {summary['input']['field']} "
        f"{summary['input']['extraction']['method']} "
        f"{summary['input']['timestamp']}"
    )
    for bullet in interpretation["executive_summary"]:
        print(f"- {bullet}")
    print(f"decision: {interpretation['promotion_decision']['decision']}")
    print(f"report: {summary['artifacts']['summary_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
