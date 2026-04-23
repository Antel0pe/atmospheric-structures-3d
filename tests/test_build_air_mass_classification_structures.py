from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.build_air_mass_classification_structures import (
    CLASS_PROXY_LABELS,
    VARIANT_RECIPES,
    apply_bridge_gap,
    build_latitude_mean_anomaly,
    build_manifest,
    build_score_mask,
    classify_quadrants,
    filter_surface_attached_components,
    filter_components,
    standardize_per_level,
    DatasetContents,
)


class BuildAirMassClassificationStructuresTests(unittest.TestCase):
    def test_build_latitude_mean_anomaly_subtracts_rowwise_mean(self) -> None:
        field = np.array(
            [
                [
                    [1.0, 3.0, 5.0],
                    [2.0, 4.0, 6.0],
                ]
            ],
            dtype=np.float32,
        )

        anomaly, latitude_mean = build_latitude_mean_anomaly(field)

        np.testing.assert_allclose(latitude_mean[..., 0], [[3.0, 4.0]])
        np.testing.assert_allclose(
            anomaly,
            [[[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]]],
        )

    def test_standardize_per_level_uses_levelwise_std(self) -> None:
        anomaly = np.array(
            [
                [[-2.0, 0.0, 2.0]],
                [[-1.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )

        standardized, scale = standardize_per_level(anomaly)

        np.testing.assert_allclose(scale, np.array([1.6329932, 0.8164966]))
        np.testing.assert_allclose(
            standardized,
            np.array(
                [
                    [[-1.2247449, 0.0, 1.2247449]],
                    [[-1.2247449, 0.0, 1.2247449]],
                ],
                dtype=np.float32,
            ),
            rtol=1e-5,
        )

    def test_build_score_mask_requires_both_axes_and_keeps_top_percent(self) -> None:
        thermal = np.array([[[2.0, 0.4, -1.2, -2.0]]], dtype=np.float32)
        moisture = np.array([[[-2.0, 1.4, -1.0, 2.5]]], dtype=np.float32)

        keep_mask, thresholds = build_score_mask(
            thermal,
            moisture,
            keep_top_percent=50.0,
            axis_min_abs_zscore=1.0,
        )

        np.testing.assert_array_equal(
            keep_mask,
            np.array([[[True, False, False, True]]], dtype=bool),
        )
        self.assertAlmostEqual(float(thresholds[0]["score_threshold"]), 1.5477226, places=5)
        self.assertEqual(int(thresholds[0]["kept_cell_count"]), 2)

    def test_classify_quadrants_splits_four_proxy_classes(self) -> None:
        keep_mask = np.ones((1, 1, 4), dtype=bool)
        thermal = np.array([[[1.0, 1.0, -1.0, -1.0]]], dtype=np.float32)
        moisture = np.array([[[-1.0, 1.0, -1.0, 1.0]]], dtype=np.float32)

        classified = classify_quadrants(keep_mask, thermal, moisture)

        np.testing.assert_array_equal(
          classified["warm_dry"], np.array([[[True, False, False, False]]], dtype=bool)
        )
        np.testing.assert_array_equal(
          classified["warm_moist"], np.array([[[False, True, False, False]]], dtype=bool)
        )
        np.testing.assert_array_equal(
          classified["cold_dry"], np.array([[[False, False, True, False]]], dtype=bool)
        )
        np.testing.assert_array_equal(
          classified["cold_moist"], np.array([[[False, False, False, True]]], dtype=bool)
        )

    def test_apply_bridge_gap_and_filter_components_remove_single_level_noise(self) -> None:
        mask = np.zeros((4, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        mask[2, 0, 0] = True
        expanded = np.pad(mask, ((0, 0), (0, 1), (0, 1)))
        expanded[1, 2, 2] = True

        bridged = apply_bridge_gap(expanded, 1)
        filtered, summary = filter_components(
            bridged,
            min_component_voxels=3,
            min_component_pressure_span_levels=2,
        )

        np.testing.assert_array_equal(
            filtered[:, 0, 0],
            np.array([True, True, True, False], dtype=bool),
        )
        self.assertFalse(bool(filtered[1, 2, 2]))
        self.assertEqual(summary["component_count"], 1)
        self.assertEqual(summary["largest_component_voxel_count"], 3)

    def test_filter_surface_attached_components_keeps_only_components_touching_1000_hpa(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 0, 0] = True
        mask[2, 2, 2] = True
        pressure_levels_hpa = np.array([1000.0, 850.0, 700.0], dtype=np.float32)

        filtered, summary = filter_surface_attached_components(mask, pressure_levels_hpa)

        np.testing.assert_array_equal(
            filtered,
            np.array(
                [
                    [[True, False, False], [False, False, False], [False, False, False]],
                    [[True, False, False], [False, False, False], [False, False, False]],
                    [[False, False, False], [False, False, False], [False, False, False]],
                ],
                dtype=bool,
            ),
        )
        self.assertEqual(summary["surface_attached_component_count"], 1)
        self.assertEqual(summary["largest_surface_attached_component_voxel_count"], 2)

    def test_build_manifest_includes_proxy_classification_summary(self) -> None:
        contents = DatasetContents(
            dataset_path=Path("data/example.nc"),
            variable_names={
                "temperature": "t",
                "relative_humidity": "r",
                "specific_humidity": "q",
            },
            pressure_levels_hpa=np.array([1000.0, 850.0], dtype=np.float32),
            latitudes_deg=np.array([10.0, 0.0], dtype=np.float32),
            longitudes_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float32),
            longitude_order=np.array([0, 1, 2], dtype=np.int64),
            timestamps=["2021-11-08T12:00"],
        )

        manifest = build_manifest(
            contents=contents,
            recipe=VARIANT_RECIPES["theta-rh-latmean"],
            entries=[
                {
                    "timestamp": "2021-11-08T12:00",
                    "metadata": "2021-11-08T12-00-00/metadata.json",
                    "voxel_count": 12,
                    "component_count": 4,
                    "class_counts": {
                        key: {"voxel_count": 3, "component_count": 1}
                        for key in CLASS_PROXY_LABELS
                    },
                }
            ],
            pressure_levels_hpa=contents.pressure_levels_hpa,
            latitudes_deg=contents.latitudes_deg,
            longitudes_deg=contents.longitudes_deg,
            base_radius=100.0,
            vertical_span=18.0,
            latitude_stride=4,
            longitude_stride=4,
        )

        self.assertEqual(manifest["variant"], "theta-rh-latmean")
        self.assertEqual(
            manifest["classification"]["thermal_axis_field"],
            "dry_potential_temperature",
        )
        self.assertFalse(manifest["classification"]["surface_attached_only"])
        self.assertEqual(
            manifest["classification"]["classes"][0]["label"],
            CLASS_PROXY_LABELS["warm_dry"],
        )
        self.assertEqual(manifest["timestamps"][0]["component_count"], 4)


if __name__ == "__main__":
    unittest.main()
