from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.build_potential_temperature_structures import (
    DatasetContents,
    apply_vertical_connection_mode,
    build_climatology_mean_anomaly,
    build_latitude_mean_anomaly,
    build_manifest,
    build_sign_tail_selected_anomaly,
    build_top_percent_selected_anomaly,
    compute_dry_potential_temperature,
    label_wrapped_volume_components,
    maybe_flip_triangle_winding,
    remove_single_level_components,
    resolve_keep_top_percent,
    stride_spatial_axes,
)


class BuildPotentialTemperatureStructuresTests(unittest.TestCase):
    def test_compute_dry_potential_temperature_uses_reference_pressure_scaling(self) -> None:
        temperature = np.array(
            [
                [[300.0]],
                [[280.0]],
            ],
            dtype=np.float32,
        )
        pressure_levels_hpa = np.array([1000.0, 500.0], dtype=np.float32)

        theta = compute_dry_potential_temperature(temperature, pressure_levels_hpa)

        self.assertAlmostEqual(float(theta[0, 0, 0]), 300.0, places=4)
        self.assertGreater(float(theta[1, 0, 0]), 340.0)

    def test_build_latitude_mean_anomaly_subtracts_per_level_latitude_band_mean(self) -> None:
        theta_field = np.array(
            [
                [
                    [300.0, 304.0, 308.0],
                    [290.0, 292.0, 294.0],
                ]
            ],
            dtype=np.float32,
        )

        anomaly, latitude_band_mean = build_latitude_mean_anomaly(theta_field)

        expected_mean = np.array(
            [
                [
                    [304.0],
                    [292.0],
                ]
            ],
            dtype=np.float32,
        )
        expected_anomaly = np.array(
            [
                [
                    [-4.0, 0.0, 4.0],
                    [-2.0, 0.0, 2.0],
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(latitude_band_mean, expected_mean)
        np.testing.assert_allclose(anomaly, expected_anomaly)

    def test_build_climatology_mean_anomaly_subtracts_matched_theta_mean(self) -> None:
        theta_field = np.array(
            [
                [
                    [300.0, 304.0],
                    [290.0, 292.0],
                ]
            ],
            dtype=np.float32,
        )
        climatology_theta_mean = np.array(
            [
                [
                    [296.0, 301.0],
                    [291.0, 289.0],
                ]
            ],
            dtype=np.float32,
        )

        anomaly = build_climatology_mean_anomaly(theta_field, climatology_theta_mean)

        np.testing.assert_allclose(
            anomaly,
            np.array(
                [
                    [
                        [4.0, 3.0],
                        [-1.0, 3.0],
                    ]
                ],
                dtype=np.float32,
            ),
        )

    def test_resolve_keep_top_percent_treats_percentile_alias_as_inverse(self) -> None:
        keep_top_percent, percentile = resolve_keep_top_percent(
            keep_top_percent=50.0,
            absolute_anomaly_percentile=20.0,
        )

        self.assertEqual(keep_top_percent, 80.0)
        self.assertEqual(percentile, 20.0)

    def test_build_top_percent_selected_anomaly_uses_top_share_semantics(self) -> None:
        anomaly = np.array(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ],
            dtype=np.float32,
        )

        selected_50, keep_mask_50, thresholds_50, keep_top_percent_50 = (
            build_top_percent_selected_anomaly(anomaly, keep_top_percent=50.0)
        )
        selected_100, keep_mask_100, thresholds_100, keep_top_percent_100 = (
            build_top_percent_selected_anomaly(anomaly, keep_top_percent=100.0)
        )

        self.assertEqual(keep_top_percent_50, 50.0)
        self.assertEqual(float(thresholds_50[0]), 2.5)
        np.testing.assert_array_equal(
            keep_mask_50,
            np.array(
                [
                    [
                        [False, False],
                        [True, True],
                    ]
                ],
                dtype=bool,
            ),
        )
        np.testing.assert_array_equal(
            selected_50,
            np.array(
                [
                    [
                        [0.0, 0.0],
                        [3.0, 4.0],
                    ]
                ],
                dtype=np.float32,
            ),
        )

        self.assertEqual(keep_top_percent_100, 100.0)
        self.assertEqual(float(thresholds_100[0]), 1.0)
        np.testing.assert_array_equal(keep_mask_100, np.ones_like(keep_mask_100, dtype=bool))
        np.testing.assert_array_equal(selected_100, anomaly)

    def test_build_sign_tail_selected_anomaly_keeps_each_sign_tail_separately(self) -> None:
        anomaly = np.array(
            [
                [
                    [-8.0, -6.0, -2.0, 1.0, 3.0, 9.0],
                ]
            ],
            dtype=np.float32,
        )

        (
            selected,
            keep_mask,
            hot_thresholds,
            cold_thresholds,
            _,
            keep_top_percent,
        ) = build_sign_tail_selected_anomaly(anomaly, keep_top_percent=50.0)

        self.assertEqual(keep_top_percent, 50.0)
        self.assertEqual(float(hot_thresholds[0]), 3.0)
        self.assertEqual(float(cold_thresholds[0]), -6.0)
        np.testing.assert_array_equal(
            keep_mask,
            np.array([[[True, True, False, False, True, True]]], dtype=bool),
        )
        np.testing.assert_array_equal(
            selected,
            np.array([[[-8.0, -6.0, 0.0, 0.0, 3.0, 9.0]]], dtype=np.float32),
        )

    def test_maybe_flip_triangle_winding_swaps_each_triangle_tail(self) -> None:
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        flipped = maybe_flip_triangle_winding(indices)

        np.testing.assert_array_equal(
            flipped,
            np.array([0, 2, 1, 2, 0, 3], dtype=np.uint32),
        )

    def test_apply_vertical_connection_mode_bridges_one_missing_level(self) -> None:
        anomaly = np.array([[[5.0]], [[1.0]], [[4.0]]], dtype=np.float32)
        selected = np.array([[[5.0]], [[0.0]], [[4.0]]], dtype=np.float32)

        filled, fill_mask = apply_vertical_connection_mode(
            anomaly,
            selected,
            connection_mode="bridge-gap-1",
        )

        np.testing.assert_array_equal(filled, anomaly)
        np.testing.assert_array_equal(
            fill_mask,
            np.array([[[False]], [[True]], [[False]]], dtype=bool),
        )

    def test_apply_vertical_connection_mode_respects_gap_limit(self) -> None:
        anomaly = np.array([[[5.0]], [[1.0]], [[2.0]], [[4.0]]], dtype=np.float32)
        selected = np.array([[[5.0]], [[0.0]], [[0.0]], [[4.0]]], dtype=np.float32)

        strict_filled, strict_mask = apply_vertical_connection_mode(
            anomaly,
            selected,
            connection_mode="bridge-gap-1",
        )
        relaxed_filled, relaxed_mask = apply_vertical_connection_mode(
            anomaly,
            selected,
            connection_mode="bridge-gap-2",
        )

        np.testing.assert_array_equal(strict_filled, selected)
        np.testing.assert_array_equal(strict_mask, np.zeros_like(strict_mask, dtype=bool))
        np.testing.assert_array_equal(relaxed_filled, anomaly)
        np.testing.assert_array_equal(
            relaxed_mask,
            np.array([[[False]], [[True]], [[True]], [[False]]], dtype=bool),
        )

    def test_apply_vertical_connection_mode_fills_between_same_sign_anchors_only(self) -> None:
        anomaly = np.array(
            [[[6.0]], [[1.0]], [[2.0]], [[3.0]], [[5.0]]],
            dtype=np.float32,
        )
        selected = np.array(
            [[[6.0]], [[0.0]], [[0.0]], [[0.0]], [[5.0]]],
            dtype=np.float32,
        )

        filled, fill_mask = apply_vertical_connection_mode(
            anomaly,
            selected,
            connection_mode="fill-between-anchors",
        )

        np.testing.assert_array_equal(filled, anomaly)
        np.testing.assert_array_equal(
            fill_mask,
            np.array([[[False]], [[True]], [[True]], [[True]], [[False]]], dtype=bool),
        )

        blocked_filled, blocked_mask = apply_vertical_connection_mode(
            np.array(
                [[[6.0]], [[1.0]], [[-2.0]], [[3.0]], [[5.0]]],
                dtype=np.float32,
            ),
            selected,
            connection_mode="fill-between-anchors",
        )

        np.testing.assert_array_equal(blocked_filled, selected)
        np.testing.assert_array_equal(
            blocked_mask,
            np.zeros_like(blocked_mask, dtype=bool),
        )

    def test_remove_single_level_components_drops_same_sign_single_level_fragments(self) -> None:
        smoothed_anomaly = np.zeros((3, 3, 4), dtype=np.float32)
        sign_field = np.zeros((3, 3, 4), dtype=np.int8)
        smoothed_anomaly[0, 0, 0] = 2.0
        sign_field[0, 0, 0] = 1
        smoothed_anomaly[1, 2, 2] = 3.0
        sign_field[1, 2, 2] = 1
        smoothed_anomaly[2, 2, 2] = 4.0
        sign_field[2, 2, 2] = 1

        filtered_anomaly, filtered_sign, summary = remove_single_level_components(
            smoothed_anomaly,
            sign_field,
        )

        self.assertEqual(int(filtered_sign[0, 0, 0]), 0)
        self.assertEqual(int(filtered_sign[1, 2, 2]), 1)
        self.assertEqual(int(filtered_sign[2, 2, 2]), 1)
        self.assertEqual(summary["removed_component_count"], 1)
        self.assertEqual(summary["removed_positive_component_count"], 1)
        self.assertEqual(summary["removed_negative_component_count"], 0)
        self.assertEqual(summary["removed_voxel_count"], 1)

    def test_stride_spatial_axes_subsamples_latitudes_and_longitudes(self) -> None:
        field = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)
        latitudes = np.array([50.0, 40.0, 30.0, 20.0, 10.0], dtype=np.float32)
        longitudes = np.array(
            [-180.0, -170.0, -160.0, -150.0, -140.0, -130.0, -120.0],
            dtype=np.float32,
        )

        strided_field, strided_latitudes, strided_longitudes = stride_spatial_axes(
            field,
            latitudes,
            longitudes,
            latitude_stride=2,
            longitude_stride=3,
        )

        np.testing.assert_array_equal(strided_field, field[:, ::2, ::3])
        np.testing.assert_array_equal(strided_latitudes, latitudes[::2])
        np.testing.assert_array_equal(strided_longitudes, longitudes[::3])

    def test_label_wrapped_volume_components_merges_longitude_seam(self) -> None:
        mask = np.array(
            [
                [[True, False, True]],
            ],
            dtype=bool,
        )

        labels, component_count = label_wrapped_volume_components(mask)

        self.assertEqual(component_count, 1)
        self.assertEqual(int(labels[0, 0, 0]), int(labels[0, 0, 2]))

    def test_build_manifest_reports_top_percent_selection_metadata(self) -> None:
        contents = DatasetContents(
            dataset_path=Path("data/example.nc"),
            units="K",
            pressure_levels_hpa=np.array([1000.0, 850.0], dtype=np.float32),
            latitudes_deg=np.array([10.0, 0.0], dtype=np.float32),
            longitudes_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float32),
            longitude_order=np.array([0, 1, 2], dtype=np.int64),
            timestamps=["2021-11-08T12:00"],
        )

        manifest = build_manifest(
            contents=contents,
            climatology_dataset_name="climatology.nc",
            entries=[
                {
                    "timestamp": "2021-11-08T12:00",
                    "metadata": "2021-11-08t12-00/metadata.json",
                    "warm_positions": "2021-11-08t12-00/warm_positions.bin",
                    "warm_indices": "2021-11-08t12-00/warm_indices.bin",
                    "cold_positions": "2021-11-08t12-00/cold_positions.bin",
                    "cold_indices": "2021-11-08t12-00/cold_indices.bin",
                    "voxel_count": 12,
                    "component_count": 2,
                    "positive_component_count": 1,
                    "negative_component_count": 1,
                }
            ],
            keep_top_percent=50.0,
            absolute_anomaly_percentile=50.0,
            smoothing_sigma_cells=1.0,
            connection_mode="bridge-gap-1",
            pressure_min_hpa=250.0,
            pressure_max_hpa=1000.0,
            latitude_stride=4,
            longitude_stride=4,
            latitudes_deg=contents.latitudes_deg,
            longitudes_deg=contents.longitudes_deg,
            pressure_levels_hpa=contents.pressure_levels_hpa,
            base_radius=100.0,
            vertical_span=12.0,
        )

        self.assertEqual(
            manifest["selection"]["threshold_basis"],
            "per-level_sign-tail_top-percent",
        )
        self.assertEqual(manifest["variant"], "bridge-gap-1")
        self.assertEqual(manifest["selection"]["keep_top_percent"], 50.0)
        self.assertEqual(manifest["selection"]["absolute_anomaly_percentile"], 50.0)
        self.assertEqual(manifest["selection"]["vertical_connection_mode"], "bridge-gap-1")
        self.assertEqual(manifest["selection"]["min_component_pressure_span_levels"], 2)


if __name__ == "__main__":
    unittest.main()
