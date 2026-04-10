from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray as xr

from scripts.moisture_structures import (
    BuildConfig,
    build_segmentation_mask,
    build_threshold_mask,
    iter_wrapped_components,
    load_threshold_seed_sample,
    prepare_segmentation_context,
)
from scripts.simple_voxel_builder import (
    build_top_percent_mask,
    load_dataset_contents,
    open_dataset_handles,
    read_field_at_time_index,
)

MOISTURE_GLOBE_CLEARANCE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a moisture view-debug case against generated assets and raw data."
    )
    parser.add_argument("--debug-case", required=True, type=Path)
    parser.add_argument("--capture-context", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_relative(path: Path) -> str:
    return path.as_posix()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_longitude(lon_deg: float) -> float:
    return lon_deg % 360.0


def xyz_to_lat_lon_radius(position: dict[str, float]) -> tuple[float, float, float]:
    x = float(position["x"])
    y = float(position["y"])
    z = float(position["z"])
    radius = math.sqrt(x * x + y * y + z * z)
    if radius <= 1e-9:
        raise ValueError("Cannot invert a zero-length world position.")
    lat = math.degrees(math.asin(max(-1.0, min(1.0, y / radius))))
    lon = normalize_longitude(-math.degrees(math.atan2(z, x)) - 270.0)
    return lat, lon, radius


def world_radius_to_source_radius(
    world_radius: float,
    *,
    base_radius: float,
    vertical_exaggeration: float,
    clearance: float = MOISTURE_GLOBE_CLEARANCE,
) -> float:
    if vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be positive.")
    radial_offset = max(world_radius - base_radius - clearance, 0.0) / vertical_exaggeration
    return base_radius + radial_offset


def source_radius_to_pressure_hpa(
    source_radius: float,
    *,
    base_radius: float,
    vertical_span: float,
) -> float:
    min_height = 44330.0 * (1.0 - (1000.0 / 1013.25) ** 0.1903)
    max_height = 44330.0 * (1.0 - (1.0 / 1013.25) ** 0.1903)
    scale = vertical_span / max(max_height - min_height, 1e-9)
    height_m = min_height + (source_radius - base_radius) / max(scale, 1e-9)
    height_ratio = max(0.0, min(0.999999, height_m / 44330.0))
    return 1013.25 * (1.0 - height_ratio) ** (1.0 / 0.1903)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - target)))


def wrapped_lon_window(
    values: np.ndarray,
    center_index: int,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    lon_count = values.shape[1]
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int32)
    lon_indices = (center_index + offsets) % lon_count
    return values[:, lon_indices], lon_indices


def build_public_asset_root(segmentation_mode: str) -> Path:
    if segmentation_mode == "p95-close":
        return Path("public/moisture-structures")
    return Path("public/moisture-structures/variants") / segmentation_mode


def load_manifest_and_entry(segmentation_mode: str, timestamp: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = build_public_asset_root(segmentation_mode)
    manifest = load_json(root / "index.json")
    entry = next((item for item in manifest["timestamps"] if item["timestamp"] == timestamp), None)
    if entry is None:
        raise ValueError(f"No asset entry found for {timestamp} in {segmentation_mode}.")
    return manifest, entry


def find_time_index(dataset: xr.Dataset, timestamp: str) -> int:
    timestamps = [
        value if not value.endswith("Z") else value[:-1]
        for value in np.datetime_as_string(
            np.asarray(dataset.coords["valid_time"].values),
            unit="m",
        )
    ]
    if timestamp not in timestamps:
        raise ValueError(f"Timestamp {timestamp} is not present in the source dataset.")
    return timestamps.index(timestamp)


def load_case_field(dataset_path: Path, timestamp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(dataset_path, chunks={})
    try:
        time_index = find_time_index(dataset, timestamp)
        field = np.asarray(dataset["q"].isel(valid_time=time_index).values, dtype=np.float32)
        pressure_levels = np.asarray(dataset.coords["pressure_level"].values, dtype=np.float32)
        latitudes = np.asarray(dataset.coords["latitude"].values, dtype=np.float32)
        longitudes = np.asarray(dataset.coords["longitude"].values, dtype=np.float32)
    finally:
        dataset.close()
    return field, pressure_levels, latitudes, longitudes


def reconstruct_masks(
    segmentation_mode: str,
    *,
    dataset_path: Path,
    timestamp: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if segmentation_mode == "simple-voxel-shell":
        handles = open_dataset_handles(dataset_path)
        try:
            contents = load_dataset_contents(handles, dataset_path)
            time_index = contents.timestamps.index(timestamp)
            field = read_field_at_time_index(handles, time_index)
            keep_mask, threshold_value = build_top_percent_mask(
                field,
                keep_quantile=float(manifest["threshold_mode"]["quantile"]),
            )
            component_masks = [keep_mask] if bool(keep_mask.any()) else []
            raw_mask = keep_mask.copy()
            processed_mask = keep_mask.copy()
            thresholds = np.asarray(
                [float(entry["threshold"]) for entry in manifest["thresholds"]],
                dtype=np.float32,
            )
            return {
                "field": field,
                "pressure_levels": contents.pressure_levels_hpa,
                "latitudes": contents.latitudes_deg,
                "longitudes": contents.longitudes_deg,
                "raw_mask": raw_mask,
                "processed_mask": processed_mask,
                "component_masks": component_masks,
                "thresholds": thresholds,
                "global_threshold": threshold_value,
                "postprocess_added_fraction": 0.0,
            }
        finally:
            handles.dataset.close()
            handles.raw_dataset.close()

    field, pressure_levels, latitudes, longitudes = load_case_field(dataset_path, timestamp)
    raw_dataset = netCDF4.Dataset(dataset_path, mode="r")
    try:
        threshold_seed_sample = load_threshold_seed_sample(raw_dataset.variables["q"])
        threshold_mode = manifest["threshold_mode"]
        config = BuildConfig(
            dataset_path=dataset_path,
            output_dir=Path("tmp/view-debug-unused"),
            threshold_quantile=float(threshold_mode["quantile"]),
            min_component_size=int(threshold_mode["minimum_component_size"]),
            closing_radius_cells=int(threshold_mode["smoothing"]["binary_closing_radius_cells"]),
            opening_radius_cells=int(
                threshold_mode["smoothing"].get("binary_opening_radius_cells", 0)
            ),
            gaussian_sigma=float(threshold_mode["smoothing"]["gaussian_sigma"]),
            segmentation_mode=segmentation_mode,
        )
        context = prepare_segmentation_context(threshold_seed_sample, config)
        processed_mask = build_segmentation_mask(field, context)
        component_masks = iter_wrapped_components(
            processed_mask,
            min_component_size=config.min_component_size,
        )

        if segmentation_mode in {
            "p95-close",
            "p95-close-voxel-shell",
            "p95-raw-voxel-shell",
            "p95-close-smoothmesh",
            "p95-close-open1",
            "p95-open",
        }:
            raw_mask = build_threshold_mask(field, context.threshold_tables["raw_q95"])
        elif segmentation_mode == "p97-close":
            raw_mask = build_threshold_mask(field, context.threshold_tables["raw_q97"])
        elif segmentation_mode in {"p95-smooth-open1", "p95-smooth-open1-voxel-shell"}:
            smoothed = context.threshold_tables["smoothed_q95"]
            raw_mask = build_threshold_mask(field, smoothed)
        else:
            raw_mask = build_threshold_mask(field, context.primary_thresholds)

        postprocess_added = processed_mask & ~raw_mask
        processed_count = int(processed_mask.sum())
        postprocess_added_fraction = (
            float(postprocess_added.sum()) / processed_count if processed_count else 0.0
        )

        return {
            "field": field,
            "pressure_levels": pressure_levels,
            "latitudes": latitudes,
            "longitudes": longitudes,
            "raw_mask": raw_mask,
            "processed_mask": processed_mask,
            "component_masks": component_masks,
            "thresholds": np.asarray(context.primary_thresholds, dtype=np.float32),
            "global_threshold": None,
            "postprocess_added_fraction": postprocess_added_fraction,
        }
    finally:
        raw_dataset.close()


def choose_nearest_component_cell(
    component_mask: np.ndarray,
    *,
    pressure_levels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    pressure_hpa: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[int, int, int]:
    coords = np.argwhere(component_mask)
    if coords.size == 0:
        raise ValueError("Component mask is empty.")

    pressure_index = nearest_index(pressure_levels, pressure_hpa)
    lat_index = nearest_index(latitudes, lat_deg)
    lon_index = nearest_index(longitudes, lon_deg)

    lon_count = len(longitudes)
    best_distance = float("inf")
    best_coord = tuple(int(value) for value in coords[0])
    for level_idx, lat_idx, lon_idx in coords:
        lon_delta = min(
            abs(int(lon_idx) - lon_index),
            lon_count - abs(int(lon_idx) - lon_index),
        )
        distance = (
            (int(level_idx) - pressure_index) ** 2
            + (int(lat_idx) - lat_index) ** 2
            + lon_delta**2
        )
        if distance < best_distance:
            best_distance = float(distance)
            best_coord = (int(level_idx), int(lat_idx), int(lon_idx))

    return best_coord


def summarize_target(
    *,
    hit: dict[str, Any],
    scenario_lookup: dict[str, dict[str, Any]],
    debug_case: dict[str, Any],
    manifest: dict[str, Any],
    reconstructed: dict[str, Any],
    target_index: int,
) -> dict[str, Any]:
    if not hit.get("didHit"):
        return {
            "targetIndex": target_index,
            "target": debug_case["targets"][target_index],
            "didHit": False,
            "primaryCause": "uncertain",
            "summary": "The target did not hit a moisture surface in the baseline scenario.",
        }

    pressure_levels = reconstructed["pressure_levels"]
    latitudes = reconstructed["latitudes"]
    longitudes = reconstructed["longitudes"]
    field = reconstructed["field"]
    raw_mask = reconstructed["raw_mask"]
    processed_mask = reconstructed["processed_mask"]
    thresholds = reconstructed["thresholds"]

    base_radius = float(manifest["globe"]["base_radius"])
    vertical_span = float(manifest["globe"]["vertical_span"])
    vertical_exaggeration = float(
        debug_case["layerState"]["moistureStructureLayer"]["verticalExaggeration"]
    )

    lat_deg, lon_deg, world_radius = xyz_to_lat_lon_radius(hit["worldPosition"])
    source_radius = world_radius_to_source_radius(
        world_radius,
        base_radius=base_radius,
        vertical_exaggeration=vertical_exaggeration,
    )
    pressure_hpa = source_radius_to_pressure_hpa(
        source_radius,
        base_radius=base_radius,
        vertical_span=vertical_span,
    )

    component_id = hit.get("componentId")
    component_mask = None
    if isinstance(component_id, int) and component_id < len(reconstructed["component_masks"]):
        component_mask = reconstructed["component_masks"][component_id]
    if component_mask is None or not bool(component_mask.any()):
        component_mask = processed_mask

    level_idx, lat_idx, lon_idx = choose_nearest_component_cell(
        component_mask,
        pressure_levels=pressure_levels,
        latitudes=latitudes,
        longitudes=longitudes,
        pressure_hpa=pressure_hpa,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )

    local_component_column = component_mask[:, lat_idx, lon_idx]
    local_processed_column = processed_mask[:, lat_idx, lon_idx]
    local_raw_column = raw_mask[:, lat_idx, lon_idx]
    raw_value = float(field[level_idx, lat_idx, lon_idx])
    threshold_value = (
        float(thresholds[level_idx])
        if level_idx < thresholds.shape[0]
        else float(reconstructed["global_threshold"] or 0.0)
    )
    raw_margin = raw_value - threshold_value
    component_vertical_thickness = int(local_component_column.sum())
    processed_vertical_thickness = int(local_processed_column.sum())
    raw_vertical_thickness = int(local_raw_column.sum())

    lat_window = 24
    lon_window = 24
    level_window = 6
    lat_start = clamp(lat_idx - lat_window, 0, len(latitudes) - 1)
    lat_stop = clamp(lat_idx + lat_window + 1, 1, len(latitudes))
    level_start = clamp(level_idx - level_window, 0, len(pressure_levels) - 1)
    level_stop = clamp(level_idx + level_window + 1, 1, len(pressure_levels))
    horizontal_field = field[level_idx, lat_start:lat_stop, :]
    horizontal_component = component_mask[level_idx, lat_start:lat_stop, :]
    horizontal_raw = raw_mask[level_idx, lat_start:lat_stop, :]
    horizontal_field_wrapped, lon_indices = wrapped_lon_window(horizontal_field, lon_idx, lon_window)
    horizontal_component_wrapped, _ = wrapped_lon_window(horizontal_component, lon_idx, lon_window)
    horizontal_raw_wrapped, _ = wrapped_lon_window(horizontal_raw, lon_idx, lon_window)
    lon_values_wrapped = longitudes[lon_indices]

    lat_section = field[level_start:level_stop, :, lon_idx]
    lat_section_component = component_mask[level_start:level_stop, :, lon_idx]
    lat_section_raw = raw_mask[level_start:level_stop, :, lon_idx]
    lon_section = field[level_start:level_stop, lat_idx, :]
    lon_section_component = component_mask[level_start:level_stop, lat_idx, :]
    lon_section_raw = raw_mask[level_start:level_stop, lat_idx, :]
    lon_section_wrapped, lon_section_indices = wrapped_lon_window(
        lon_section,
        lon_idx,
        lon_window,
    )
    lon_section_component_wrapped, _ = wrapped_lon_window(
        lon_section_component,
        lon_idx,
        lon_window,
    )
    lon_section_raw_wrapped, _ = wrapped_lon_window(lon_section_raw, lon_idx, lon_window)
    lon_section_values = longitudes[lon_section_indices]

    output_dir = Path(debug_case["_outputDir"])
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    horizontal_plot = plots_dir / f"target-{target_index + 1}-horizontal.png"
    plt.figure(figsize=(9, 4))
    plt.imshow(horizontal_field_wrapped, aspect="auto", origin="lower")
    plt.contour(horizontal_component_wrapped.astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    plt.contour(horizontal_raw_wrapped.astype(float), levels=[0.5], colors=["black"], linewidths=0.8)
    plt.title(
        f"Horizontal slice near {pressure_levels[level_idx]:.0f} hPa\n"
        "white=processed component, black=raw threshold mask"
    )
    plt.tight_layout()
    plt.savefig(horizontal_plot, dpi=150)
    plt.close()

    lat_section_plot = plots_dir / f"target-{target_index + 1}-lat-pressure.png"
    plt.figure(figsize=(9, 4))
    plt.imshow(lat_section, aspect="auto", origin="lower")
    plt.contour(lat_section_component.astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    plt.contour(lat_section_raw.astype(float), levels=[0.5], colors=["black"], linewidths=0.8)
    plt.title("Latitude/pressure section at target longitude")
    plt.tight_layout()
    plt.savefig(lat_section_plot, dpi=150)
    plt.close()

    lon_section_plot = plots_dir / f"target-{target_index + 1}-lon-pressure.png"
    plt.figure(figsize=(9, 4))
    plt.imshow(lon_section_wrapped, aspect="auto", origin="lower")
    plt.contour(lon_section_component_wrapped.astype(float), levels=[0.5], colors=["white"], linewidths=1.2)
    plt.contour(lon_section_raw_wrapped.astype(float), levels=[0.5], colors=["black"], linewidths=0.8)
    plt.title("Longitude/pressure section through target latitude")
    plt.tight_layout()
    plt.savefig(lon_section_plot, dpi=150)
    plt.close()

    component_touches_boundary = bool(
        component_mask[0].any()
        or component_mask[-1].any()
        or component_mask[:, 0, :].any()
        or component_mask[:, -1, :].any()
    )

    def get_scenario_hit(label: str) -> dict[str, Any] | None:
        scenario = scenario_lookup.get(label)
        if not scenario:
            return None
        hits = scenario.get("hits")
        if not isinstance(hits, list) or target_index >= len(hits):
            return None
        target_hit = hits[target_index]
        return target_hit if isinstance(target_hit, dict) else None

    simplified_hit = get_scenario_hit("simplified-render")
    voxel_hit = get_scenario_hit("simple-voxel-shell")
    visible_in_simple_render = bool(simplified_hit and simplified_hit.get("didHit"))
    visible_in_simple_voxel = bool(voxel_hit and voxel_hit.get("didHit"))

    if visible_in_simple_render and visible_in_simple_voxel:
        rendering_primary = False
    else:
        rendering_primary = True

    if rendering_primary and not visible_in_simple_render:
        primary_cause = "rendering_amplified_existing_wall"
        summary = "The target weakens or disappears under simplified rendering, so rendering/view settings materially amplify the wall read."
    elif (
        processed_vertical_thickness >= raw_vertical_thickness + 2
        or (
            raw_vertical_thickness > 0
            and processed_vertical_thickness / raw_vertical_thickness >= 1.25
        )
    ):
        primary_cause = "threshold_postprocess_behavior"
        summary = "The processed occupancy is materially thicker than the raw threshold support near the target, so threshold postprocess contributes to the wall-like surface."
    elif component_touches_boundary:
        primary_cause = "mesh_extraction_behavior"
        summary = "The target component touches a data boundary, so open-surface or extraction behavior can contribute to the freestanding wall read."
    elif (
        raw_margin > 0
        and processed_vertical_thickness == raw_vertical_thickness
        and visible_in_simple_render
    ):
        primary_cause = "raw_field_structure"
        summary = "The raw threshold support and processed occupancy match at the target, and the wall read persists under simplified rendering, so the wall mainly reflects the thresholded raw field rather than a rendering artifact."
    elif processed_vertical_thickness <= 2 and raw_margin > 0 and visible_in_simple_voxel:
        primary_cause = "raw_field_structure"
        summary = "The target stays wall-like in voxel-shell mode and the local occupied column is only 1-2 levels thick, which points to a genuinely thin raw moisture slab."
    else:
        primary_cause = "mixed_or_uncertain"
        summary = "The target wall likely reflects multiple factors, with no single layer clearly dominating from the available evidence."

    return {
        "targetIndex": target_index,
        "target": debug_case["targets"][target_index],
        "didHit": True,
        "componentId": component_id,
        "estimatedLatDeg": lat_deg,
        "estimatedLonDeg": lon_deg,
        "estimatedPressureHpa": pressure_hpa,
        "nearestCell": {
            "levelIndex": level_idx,
            "latitudeIndex": lat_idx,
            "longitudeIndex": lon_idx,
            "pressureHpa": float(pressure_levels[level_idx]),
            "latitudeDeg": float(latitudes[lat_idx]),
            "longitudeDeg": float(longitudes[lon_idx]),
        },
        "rawValue": raw_value,
        "thresholdValue": threshold_value,
        "rawMargin": raw_margin,
        "verticalThickness": {
            "rawThresholdColumn": raw_vertical_thickness,
            "processedColumn": processed_vertical_thickness,
            "componentColumn": component_vertical_thickness,
        },
        "componentTouchesBoundary": component_touches_boundary,
        "postprocessAddedFraction": reconstructed["postprocess_added_fraction"],
        "visibleInSimplifiedRender": visible_in_simple_render,
        "visibleInSimpleVoxelShell": visible_in_simple_voxel,
        "plots": [
            repo_relative(horizontal_plot.relative_to(Path.cwd())),
            repo_relative(lat_section_plot.relative_to(Path.cwd())),
            repo_relative(lon_section_plot.relative_to(Path.cwd())),
        ],
        "primaryCause": primary_cause,
        "summary": summary,
    }


def build_report_markdown(
    *,
    debug_case: dict[str, Any],
    capture_context: dict[str, Any],
    analysis: dict[str, Any],
    output_dir: Path,
) -> str:
    lines = [
        "# View Debug Report",
        "",
        f"- Case: `{debug_case['title']}`",
        f"- Analyzer: `{debug_case['analyzer']}`",
        f"- Timestamp: `{debug_case['timestamp']}`",
        f"- Baseline segmentation: `{debug_case['layerState']['moistureStructureLayer']['segmentationMode']}`",
        f"- Legibility: `{debug_case['layerState']['moistureStructureLayer']['legibilityExperiment']}`",
        f"- Output dir: `{repo_relative(output_dir.relative_to(Path.cwd()))}`",
        "",
        "## Captures",
    ]

    for scenario in capture_context["scenarios"]:
        lines.append(f"- `{scenario['label']}`: `{scenario['screenshot']}`")

    lines.extend(["", "## Findings"])

    for target in analysis["targets"]:
        lines.append(
            f"- Target {target['targetIndex'] + 1}: `{target['primaryCause']}`"
        )
        lines.append(f"  {target['summary']}")
        if target.get("didHit"):
            lines.append(
                "  "
                f"Estimated lat/lon/pressure: "
                f"{target['estimatedLatDeg']:.2f}, "
                f"{target['estimatedLonDeg']:.2f}, "
                f"{target['estimatedPressureHpa']:.1f} hPa."
            )
            lines.append(
                "  "
                f"Raw value vs threshold: "
                f"{target['rawValue']:.6g} vs {target['thresholdValue']:.6g} "
                f"(margin {target['rawMargin']:.6g})."
            )
            lines.append(
                "  "
                f"Vertical thickness raw/processed/component: "
                f"{target['verticalThickness']['rawThresholdColumn']}/"
                f"{target['verticalThickness']['processedColumn']}/"
                f"{target['verticalThickness']['componentColumn']} levels."
            )
            lines.append(
            "  "
                f"Plots: {', '.join(f'`{plot}`' for plot in target['plots'])}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    debug_case = load_json(args.debug_case)
    capture_context = load_json(args.capture_context)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_case["_outputDir"] = str(output_dir)

    baseline = next(
        (scenario for scenario in capture_context["scenarios"] if scenario["label"] == "baseline"),
        None,
    )
    if baseline is None:
        raise ValueError("capture-context.json is missing the baseline scenario.")

    baseline_state = baseline["appliedState"]["analyzers"].get("moisture-structure")
    if not baseline_state or not baseline_state.get("frame"):
        raise ValueError("The baseline scenario does not expose moisture debug state.")

    manifest, entry = load_manifest_and_entry(
        baseline_state["resolvedLayer"]["segmentationMode"],
        debug_case["timestamp"],
    )
    asset_root = build_public_asset_root(baseline_state["resolvedLayer"]["segmentationMode"])
    dataset_path = Path("data") / manifest["dataset"]
    reconstructed = reconstruct_masks(
        baseline_state["resolvedLayer"]["segmentationMode"],
        dataset_path=dataset_path,
        timestamp=debug_case["timestamp"],
        manifest=manifest,
    )

    scenario_lookup = {scenario["label"]: scenario for scenario in capture_context["scenarios"]}
    targets = []
    for target_index, hit in enumerate(baseline["hits"]):
        targets.append(
            summarize_target(
                hit=hit,
                scenario_lookup=scenario_lookup,
                debug_case=debug_case,
                manifest=manifest,
                reconstructed=reconstructed,
                target_index=target_index,
            )
        )

    analysis = {
        "version": 1,
        "caseTitle": debug_case["title"],
        "segmentationMode": baseline_state["resolvedLayer"]["segmentationMode"],
        "dataset": manifest["dataset"],
        "geometryMode": manifest.get("geometry_mode"),
        "targets": targets,
    }

    analysis_path = output_dir / "moisture-analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    report_text = build_report_markdown(
        debug_case=debug_case,
        capture_context=capture_context,
        analysis=analysis,
        output_dir=output_dir,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "analysis": repo_relative(analysis_path.relative_to(Path.cwd())),
                "report": repo_relative(report_path.relative_to(Path.cwd())),
            }
        )
    )


if __name__ == "__main__":
    main()
