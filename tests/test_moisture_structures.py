from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import json
import xarray as xr

from scripts.moisture_structures import (
    BUCKET_COUNT,
    BUCKET_LATITUDE_SMOOTHING_SIGMA,
    BUCKET_LEVEL_SMOOTHING_SIGMA,
    BUCKET_LONGITUDE_SMOOTHING_SIGMA,
    BuildConfig,
    SegmentationContext,
    build_assets,
    build_bucket_component_specs,
    prepare_segmentation_context,
    build_radius_lookup,
    build_segmentation_mask,
    build_threshold_mask,
    build_timestamp_payload,
    compute_per_level_thresholds_from_array,
    iter_wrapped_components,
    DEFAULT_OPENING_RADIUS_CELLS,
    resolve_geometry_mode,
    write_timestamp_assets,
)


def make_segmentation_context(
    segmentation_mode: str,
    *,
    primary_thresholds: np.ndarray,
    threshold_tables: dict[str, np.ndarray],
    closing_radius_cells: int = 1,
    opening_radius_cells: int = 1,
) -> SegmentationContext:
    return SegmentationContext(
        segmentation_mode=segmentation_mode,
        primary_thresholds=np.asarray(primary_thresholds, dtype=np.float32),
        threshold_tables={
            key: np.asarray(value, dtype=np.float32)
            for key, value in threshold_tables.items()
        },
        closing_radius_cells=closing_radius_cells,
        opening_radius_cells=opening_radius_cells,
        threshold_quantile=0.95,
        threshold_kind="test",
        recipe_metadata={},
    )


class MoistureStructuresTests(unittest.TestCase):
    def test_prepare_segmentation_context_uses_configured_quantile_for_p95_close(self) -> None:
        threshold_seed_sample = np.array(
            [
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ]
            ],
            dtype=np.float32,
        )
        config = BuildConfig(
            dataset_path=Path("test.nc"),
            output_dir=Path("tmp/out"),
            threshold_quantile=0.75,
            segmentation_mode="p95-close",
        )

        context = prepare_segmentation_context(threshold_seed_sample, config)
        expected = compute_per_level_thresholds_from_array(
            threshold_seed_sample,
            quantile=0.75,
        )

        np.testing.assert_allclose(context.primary_thresholds, expected)
        self.assertEqual(context.threshold_quantile, 0.75)
        self.assertEqual(context.recipe_metadata["thresholds"]["raw_q95"], 0.75)

    def test_prepare_segmentation_context_uses_configured_radii_for_p95_close(self) -> None:
        threshold_seed_sample = np.ones((1, 1, 2, 2), dtype=np.float32)
        config = BuildConfig(
            dataset_path=Path("test.nc"),
            output_dir=Path("tmp/out"),
            closing_radius_cells=0,
            opening_radius_cells=2,
            segmentation_mode="p95-close",
        )

        context = prepare_segmentation_context(threshold_seed_sample, config)

        self.assertEqual(context.closing_radius_cells, 0)
        self.assertEqual(context.opening_radius_cells, 2)
        self.assertEqual(
            context.recipe_metadata["postprocess"]["binary_closing_radius_cells"],
            0,
        )
        self.assertEqual(
            context.recipe_metadata["postprocess"]["binary_opening_radius_cells"],
            2,
        )

    def test_prepare_segmentation_context_supports_smoothed_mesh_variant(self) -> None:
        threshold_seed_sample = np.array(
            [[[[0.1, 0.2], [0.3, 0.4]]]],
            dtype=np.float32,
        )
        config = BuildConfig(
            dataset_path=Path("test.nc"),
            output_dir=Path("tmp/out"),
            segmentation_mode="p95-close-smoothmesh",
            mesh_smoothing_iterations=8,
            mesh_smoothing_lambda=0.41,
            mesh_smoothing_mu=-0.43,
        )

        context = prepare_segmentation_context(threshold_seed_sample, config)

        self.assertEqual(context.segmentation_mode, "p95-close-smoothmesh")
        self.assertEqual(context.recipe_metadata["recipe"], "bridge-pruned-smoothed-mesh")
        self.assertEqual(
            context.recipe_metadata["mesh_postprocess"]["iterations"],
            8,
        )
        self.assertEqual(context.recipe_metadata["mesh_postprocess"]["lambda"], 0.41)
        self.assertEqual(context.recipe_metadata["mesh_postprocess"]["mu"], -0.43)

    def test_prepare_segmentation_context_supports_voxel_shell_variant(self) -> None:
        threshold_seed_sample = np.array(
            [[[[0.1, 0.2], [0.3, 0.4]]]],
            dtype=np.float32,
        )
        config = BuildConfig(
            dataset_path=Path("test.nc"),
            output_dir=Path("tmp/out"),
            segmentation_mode="p95-close-voxel-shell",
        )

        context = prepare_segmentation_context(threshold_seed_sample, config)

        self.assertEqual(context.segmentation_mode, "p95-close-voxel-shell")
        self.assertEqual(context.recipe_metadata["recipe"], "bridge-pruned-voxel-shell")
        self.assertEqual(context.recipe_metadata["geometry_variant"], "voxel-shell")

    def test_prepare_segmentation_context_supports_smoothed_voxel_shell_variant(self) -> None:
        threshold_seed_sample = np.array(
            [[[[0.1, 0.2], [0.3, 0.4]]]],
            dtype=np.float32,
        )
        config = BuildConfig(
            dataset_path=Path("test.nc"),
            output_dir=Path("tmp/out"),
            segmentation_mode="p95-smooth-open1-voxel-shell",
        )

        context = prepare_segmentation_context(threshold_seed_sample, config)

        self.assertEqual(context.segmentation_mode, "p95-smooth-open1-voxel-shell")
        self.assertEqual(context.recipe_metadata["recipe"], "smoothed-support-voxel-shell")
        self.assertEqual(
            context.recipe_metadata["preprocessing"]["lat_lon_gaussian_sigma"],
            1.0,
        )
        self.assertEqual(context.recipe_metadata["geometry_variant"], "voxel-shell")

    def test_resolve_geometry_mode_overrides_variant_specific_modes(self) -> None:
        self.assertEqual(
            resolve_geometry_mode("p95-close-voxel-shell", "marching-cubes"),
            "voxel-faces",
        )
        self.assertEqual(
            resolve_geometry_mode("p95-smooth-open1-voxel-shell", "marching-cubes"),
            "voxel-faces",
        )
        self.assertEqual(
            resolve_geometry_mode("p95-close-smoothmesh", "voxel-faces"),
            "marching-cubes",
        )
        self.assertEqual(
            resolve_geometry_mode("p95-close", "marching-cubes"),
            "marching-cubes",
        )

    def test_wraparound_components_merge_across_longitude_seam(self) -> None:
        mask = np.zeros((2, 2, 8), dtype=bool)
        mask[0, 0, 0] = True
        mask[0, 0, -1] = True

        components = iter_wrapped_components(mask)
        large_components = [component for component in components if component.sum() >= 2]

        self.assertEqual(len(large_components), 1)
        self.assertEqual(int(large_components[0].sum()), 2)

    def test_pressure_relative_thresholds_keep_upper_level_signal(self) -> None:
        values = np.array(
            [
                [
                    [[0.010, 0.011], [0.012, 0.013]],
                    [[0.00035, 0.00040], [0.00045, 0.00050]],
                ],
                [
                    [[0.011, 0.012], [0.013, 0.014]],
                    [[0.00040, 0.00045], [0.00050, 0.00055]],
                ],
            ],
            dtype=np.float32,
        )

        thresholds = compute_per_level_thresholds_from_array(values, quantile=0.75)
        upper_level = values[1, 1]
        mask = build_threshold_mask(upper_level[None, ...], thresholds[1:2])

        self.assertGreater(float(thresholds[0]), float(thresholds[1]))
        self.assertTrue(mask.any())

    def test_timestamp_payload_writes_consistent_geometry(self) -> None:
        pressure_levels = np.array([1000.0, 700.0, 500.0], dtype=np.float32)
        latitudes = np.array([10.0, 0.0, -10.0], dtype=np.float32)
        longitudes = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
        radius_lookup = build_radius_lookup(pressure_levels)

        field = np.zeros((3, 3, 4), dtype=np.float32)
        field[0:2, 1, 1:3] = 0.8
        field[1:3, 1, 1:3] = 0.85
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        for geometry_mode in ("marching-cubes", "voxel-faces"):
            payload = build_timestamp_payload(
                field=field,
                thresholds=thresholds,
                pressure_levels=pressure_levels,
                latitudes=latitudes,
                longitudes=longitudes,
                radius_lookup=radius_lookup,
                min_component_size=1,
                gaussian_sigma=0.4,
                opening_radius_cells=0,
                geometry_mode=geometry_mode,
            )

            self.assertGreater(payload["component_count"], 0)
            self.assertGreater(payload["vertex_count"], 0)
            self.assertGreater(payload["index_count"], 0)
            self.assertTrue(np.isfinite(payload["positions"]).all())
            self.assertLess(int(payload["indices"].max()), payload["vertex_count"])

            with tempfile.TemporaryDirectory() as tmp_dir:
                entry = write_timestamp_assets(
                    Path(tmp_dir),
                    "2021-11-08T00:00",
                    payload,
                )

                self.assertEqual(entry["component_count"], payload["component_count"])
                self.assertTrue((Path(tmp_dir) / entry["metadata"]).exists())
                self.assertTrue((Path(tmp_dir) / entry["positions"]).exists())
                self.assertTrue((Path(tmp_dir) / entry["indices"]).exists())

    def test_closing_radius_changes_component_extent(self) -> None:
        pressure_levels = np.array([1000.0, 850.0, 700.0], dtype=np.float32)
        latitudes = np.array([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32)
        longitudes = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        radius_lookup = build_radius_lookup(pressure_levels)

        field = np.zeros((3, 5, 5), dtype=np.float32)
        field[1, 2, 1] = 0.9
        field[1, 2, 3] = 0.9
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        opened = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=0,
            opening_radius_cells=0,
            geometry_mode="voxel-faces",
        )
        closed = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=1,
            opening_radius_cells=0,
            geometry_mode="voxel-faces",
        )

        self.assertEqual(opened["component_count"], 2)
        self.assertEqual(closed["component_count"], 1)

    def test_opening_radius_breaks_thin_bridge(self) -> None:
        pressure_levels = np.array([1000.0, 850.0, 700.0], dtype=np.float32)
        latitudes = np.array([3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float32)
        longitudes = np.arange(8.0, dtype=np.float32)
        radius_lookup = build_radius_lookup(pressure_levels)

        field = np.zeros((3, 5, 8), dtype=np.float32)
        field[:, 1:4, 0:3] = 0.9
        field[:, 1:4, 5:8] = 0.9
        field[:, 2:3, 3:5] = 0.9
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        connected = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=0,
            opening_radius_cells=0,
            geometry_mode="voxel-faces",
        )
        pruned = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=0,
            opening_radius_cells=1,
            geometry_mode="voxel-faces",
        )

        self.assertEqual(connected["component_count"], 1)
        self.assertEqual(pruned["component_count"], 2)
        self.assertLess(pruned["voxel_count"], connected["voxel_count"])

    def test_default_opening_radius_uses_bridge_pruned_processing(self) -> None:
        pressure_levels = np.array([1000.0, 850.0, 700.0], dtype=np.float32)
        latitudes = np.array([3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float32)
        longitudes = np.arange(8.0, dtype=np.float32)
        radius_lookup = build_radius_lookup(pressure_levels)

        field = np.zeros((3, 5, 8), dtype=np.float32)
        field[:, 1:4, 0:3] = 0.9
        field[:, 1:4, 5:8] = 0.9
        field[:, 2:3, 3:5] = 0.9
        thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        default_payload = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=0,
            geometry_mode="voxel-faces",
        )

        self.assertEqual(DEFAULT_OPENING_RADIUS_CELLS, 1)
        self.assertEqual(default_payload["component_count"], 2)

    def test_smoothed_support_merges_a_tiny_gap_without_capturing_a_far_blob(self) -> None:
        field = np.zeros((3, 7, 10), dtype=np.float32)
        field[:, 2:5, 0:3] = 1.0
        field[:, 2:5, 4:7] = 1.0
        field[:, 0:2, 8:10] = 1.0

        context = make_segmentation_context(
            "p95-smooth-open1",
            primary_thresholds=np.array([0.45, 0.45, 0.45], dtype=np.float32),
            threshold_tables={
                "smoothed_q95": np.array([0.45, 0.45, 0.45], dtype=np.float32),
            },
            closing_radius_cells=0,
            opening_radius_cells=0,
        )

        mask = build_segmentation_mask(field, context)
        components = iter_wrapped_components(mask, min_component_size=1)

        self.assertEqual(len(components), 2)
        self.assertTrue(mask[:, 2:5, 3].any())
        self.assertTrue(mask[:, 0:2, 8:10].any())

    def test_local_anomaly_suppresses_broad_background_humidity(self) -> None:
        field = np.full((3, 7, 10), 0.82, dtype=np.float32)
        field[:, 2:5, 4:7] = 0.96

        context = make_segmentation_context(
            "p95-local-anomaly",
            primary_thresholds=np.array([0.03, 0.03, 0.03], dtype=np.float32),
            threshold_tables={
                "raw_q90": np.array([0.80, 0.80, 0.80], dtype=np.float32),
                "anomaly_q95": np.array([0.03, 0.03, 0.03], dtype=np.float32),
            },
            closing_radius_cells=0,
            opening_radius_cells=0,
        )

        mask = build_segmentation_mask(field, context)

        self.assertTrue(mask[:, 2:5, 4:7].any())
        self.assertFalse(mask[:, 0:2, 0:3].any())
        self.assertLess(int(mask.sum()), int(np.prod(field.shape)))

    def test_timestamp_assets_write_footprints_when_present(self) -> None:
        pressure_levels = np.array([1000.0, 700.0], dtype=np.float32)
        latitudes = np.array([20.0, 10.0, 0.0, -10.0], dtype=np.float32)
        longitudes = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
        radius_lookup = build_radius_lookup(pressure_levels)

        field = np.zeros((2, 4, 4), dtype=np.float32)
        field[:, 1:3, 1:3] = 0.85
        thresholds = np.array([0.5, 0.5], dtype=np.float32)
        payload = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            closing_radius_cells=0,
            opening_radius_cells=0,
            geometry_mode="voxel-faces",
            write_footprints=True,
        )

        self.assertGreater(len(payload["footprints"]), 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            entry = write_timestamp_assets(
                Path(tmp_dir),
                "2021-11-08T00:00",
                payload,
            )
            self.assertIn("footprints", entry)
            footprint_path = Path(tmp_dir) / entry["footprints"]
            self.assertTrue(footprint_path.exists())
            footprint_payload = json.loads(footprint_path.read_text(encoding="utf-8"))
            self.assertEqual(footprint_payload["timestamp"], "2021-11-08T00:00")
            self.assertEqual(
                [component["id"] for component in footprint_payload["components"]],
                [component["id"] for component in payload["footprints"]],
            )

    def test_bucket_segmentation_context_uses_per_level_edges(self) -> None:
        threshold_seed_sample = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ]
            ],
            dtype=np.float32,
        )
        context = prepare_segmentation_context(
            threshold_seed_sample,
            BuildConfig(
                dataset_path=Path("test.nc"),
                output_dir=Path("tmp/out"),
                segmentation_mode="buckets",
            ),
        )

        self.assertEqual(context.bucket_count, BUCKET_COUNT)
        self.assertEqual(context.recipe_metadata["bucket_strategy"], "per-pressure-level")
        self.assertEqual(
            context.recipe_metadata["preprocessing"]["bucket_gaussian_sigma"],
            {
                "pressure_level": BUCKET_LEVEL_SMOOTHING_SIGMA,
                "latitude": BUCKET_LATITUDE_SMOOTHING_SIGMA,
                "longitude": BUCKET_LONGITUDE_SMOOTHING_SIGMA,
            },
        )
        self.assertEqual(context.recipe_metadata["bucket_labels"]["lowest_bucket_index"], 0)
        self.assertEqual(
            context.recipe_metadata["bucket_labels"]["highest_bucket_index"],
            BUCKET_COUNT - 1,
        )
        self.assertEqual(
            context.threshold_tables["bucket_boundaries"].shape,
            (2, BUCKET_COUNT - 1),
        )
        self.assertLess(
            float(context.threshold_tables["bucket_boundaries"][0, -1]),
            float(context.threshold_tables["bucket_boundaries"][1, 0]),
        )

    def test_global_bucket_segmentation_context_uses_shared_edges(self) -> None:
        threshold_seed_sample = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ]
            ],
            dtype=np.float32,
        )
        context = prepare_segmentation_context(
            threshold_seed_sample,
            BuildConfig(
                dataset_path=Path("test.nc"),
                output_dir=Path("tmp/out"),
                segmentation_mode="buckets-global",
            ),
        )

        self.assertEqual(context.recipe_metadata["bucket_strategy"], "global-shared")
        self.assertEqual(
            context.recipe_metadata["preprocessing"]["bucket_gaussian_sigma"],
            {
                "pressure_level": BUCKET_LEVEL_SMOOTHING_SIGMA,
                "latitude": BUCKET_LATITUDE_SMOOTHING_SIGMA,
                "longitude": BUCKET_LONGITUDE_SMOOTHING_SIGMA,
            },
        )
        np.testing.assert_allclose(
            context.threshold_tables["bucket_boundaries"][0],
            context.threshold_tables["bucket_boundaries"][1],
        )

    def test_bucket_component_specs_assign_one_component_per_bucket(self) -> None:
        gradient = np.linspace(0.0, 1.0, 120, dtype=np.float32).reshape(2, 3, 20)
        field = gradient.copy()
        context = make_segmentation_context(
            "buckets",
            primary_thresholds=np.array([0.5, 0.5], dtype=np.float32),
            threshold_tables={
                "bucket_boundaries": np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    ],
                    dtype=np.float32,
                ),
            },
            closing_radius_cells=0,
            opening_radius_cells=0,
        )
        context = SegmentationContext(
            **{**context.__dict__, "bucket_count": BUCKET_COUNT}
        )

        specs = build_bucket_component_specs(field, context)
        bucket_indices = [spec["metadata"]["bucket_index"] for spec in specs]

        self.assertEqual(bucket_indices, sorted(bucket_indices))
        self.assertGreaterEqual(bucket_indices[0], 0)
        self.assertLessEqual(bucket_indices[-1], BUCKET_COUNT - 1)
        self.assertGreater(bucket_indices[-1], bucket_indices[0])
        self.assertGreaterEqual(len(specs), BUCKET_COUNT // 2)
        self.assertGreater(int(specs[0]["mask"].sum()), 0)
        self.assertGreater(int(specs[-1]["mask"].sum()), 0)

    def test_build_assets_allows_singleton_downsampled_grid_steps_in_bucket_mode(self) -> None:
        values = np.linspace(0.01, 0.24, 2 * 2 * 5 * 4, dtype=np.float32).reshape(2, 2, 5, 4)
        dataset = xr.Dataset(
            data_vars={
                "q": (
                    ("valid_time", "pressure_level", "latitude", "longitude"),
                    values,
                )
            },
            coords={
                "valid_time": np.array(
                    ["2021-11-08T00:00", "2021-11-08T06:00"],
                    dtype="datetime64[m]",
                ),
                "pressure_level": np.array([1000.0, 850.0], dtype=np.float32),
                "latitude": np.array([20.0, 10.0, 0.0, -10.0, -20.0], dtype=np.float32),
                "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
            },
        )
        dataset["q"].attrs["units"] = "kg kg-1"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "tiny.nc"
            output_dir = Path(tmp_dir) / "out"
            dataset.to_netcdf(dataset_path)

            manifest = build_assets(
                BuildConfig(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    segmentation_mode="buckets",
                    geometry_mode="voxel-faces",
                    min_component_size=1,
                    limit_timestamps=1,
                )
            )

            self.assertEqual(manifest["grid"]["latitude_count"], 1)
            self.assertEqual(manifest["grid"]["longitude_count"], 1)
            self.assertIsNone(manifest["grid"]["latitude_step_degrees"])
            self.assertIsNone(manifest["grid"]["longitude_step_degrees"])
            self.assertEqual(len(manifest["timestamps"]), 1)
            self.assertTrue((output_dir / "index.json").exists())

    def test_build_assets_forces_voxel_geometry_for_voxel_shell_segmentation(self) -> None:
        values = np.zeros((1, 2, 4, 4), dtype=np.float32)
        values[:, :, 1:3, 1:3] = 0.9
        dataset = xr.Dataset(
            data_vars={
                "q": (
                    ("valid_time", "pressure_level", "latitude", "longitude"),
                    values,
                )
            },
            coords={
                "valid_time": np.array(["2021-11-08T00:00"], dtype="datetime64[m]"),
                "pressure_level": np.array([1000.0, 850.0], dtype=np.float32),
                "latitude": np.array([20.0, 10.0, 0.0, -10.0], dtype=np.float32),
                "longitude": np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
            },
        )
        dataset["q"].attrs["units"] = "kg kg-1"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "tiny.nc"
            output_dir = Path(tmp_dir) / "out"
            dataset.to_netcdf(dataset_path)

            manifest = build_assets(
                BuildConfig(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    segmentation_mode="p95-close-voxel-shell",
                    geometry_mode="marching-cubes",
                    min_component_size=1,
                    limit_timestamps=1,
                )
            )

            self.assertEqual(manifest["geometry_mode"], "voxel-faces")


if __name__ == "__main__":
    unittest.main()
