from __future__ import annotations

import unittest

import numpy as np

from scripts.build_precipitable_water_proxy import (
    build_precipitable_water_proxy_mask,
    build_pressure_window_mask,
    filter_small_wrapped_components,
    keep_only_runs_with_min_length,
    specific_humidity_quantile_from_top_percent,
)


class BuildPrecipitableWaterProxyTests(unittest.TestCase):
    def test_specific_humidity_top_percent_maps_to_upper_tail_quantile(self) -> None:
        self.assertAlmostEqual(specific_humidity_quantile_from_top_percent(40.0), 0.6)

    def test_build_pressure_window_mask_is_inclusive(self) -> None:
        pressure_levels_hpa = np.array([1000.0, 700.0, 500.0, 450.0], dtype=np.float32)
        mask = build_pressure_window_mask(
            pressure_levels_hpa,
            min_pressure_hpa=500.0,
            max_pressure_hpa=1000.0,
        )
        np.testing.assert_array_equal(mask, np.array([True, True, True, False]))

    def test_keep_only_runs_with_min_length_preserves_entire_qualifying_run(self) -> None:
        mask = np.zeros((5, 2, 1), dtype=bool)
        mask[0:3, 0, 0] = True
        mask[1:3, 1, 0] = True

        kept = keep_only_runs_with_min_length(mask, minimum_adjacent_levels=3)

        expected = np.zeros_like(mask)
        expected[0:3, 0, 0] = True
        np.testing.assert_array_equal(kept, expected)

    def test_filter_small_wrapped_components_drops_components_smaller_than_ten(self) -> None:
        keep_mask = np.zeros((2, 3, 10), dtype=bool)
        keep_mask[0, 0, 0:9] = True
        keep_mask[:, 2, 0:5] = True

        filtered_mask, metadata = filter_small_wrapped_components(
            keep_mask,
            min_component_size=10,
        )

        expected = np.zeros_like(keep_mask)
        expected[:, 2, 0:5] = True

        np.testing.assert_array_equal(filtered_mask, expected)
        self.assertEqual(metadata["component_count_before_filter"], 2)
        self.assertEqual(metadata["component_count_after_filter"], 1)
        self.assertEqual(metadata["removed_component_count"], 1)
        self.assertEqual(metadata["removed_voxel_count"], 9)

    def test_precipitable_water_proxy_mask_applies_pressure_q_rh_and_depth_gates(self) -> None:
        pressure_levels_hpa = np.array(
            [1000.0, 975.0, 950.0, 925.0, 500.0, 450.0],
            dtype=np.float32,
        )
        specific_humidity_thresholds = np.array(
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            dtype=np.float32,
        )
        specific_humidity_field = np.array(
            [
                [[11.0]],
                [[12.0]],
                [[13.0]],
                [[9.0]],
                [[14.0]],
                [[99.0]],
            ],
            dtype=np.float32,
        )
        relative_humidity_field = np.array(
            [
                [[90.0]],
                [[91.0]],
                [[92.0]],
                [[93.0]],
                [[70.0]],
                [[99.0]],
            ],
            dtype=np.float32,
        )

        kept, gate_counts = build_precipitable_water_proxy_mask(
            specific_humidity_field=specific_humidity_field,
            relative_humidity_field=relative_humidity_field,
            specific_humidity_thresholds=specific_humidity_thresholds,
            pressure_levels_hpa=pressure_levels_hpa,
            relative_humidity_threshold=85.0,
            min_pressure_hpa=500.0,
            max_pressure_hpa=1000.0,
            minimum_adjacent_levels=3,
        )

        expected = np.zeros_like(kept)
        expected[0:3, 0, 0] = True

        np.testing.assert_array_equal(kept, expected)
        self.assertEqual(gate_counts["finite_pressure_window_voxel_count"], 5)
        self.assertEqual(gate_counts["specific_humidity_gate_voxel_count"], 4)
        self.assertEqual(gate_counts["relative_humidity_gate_voxel_count"], 4)
        self.assertEqual(gate_counts["combined_gate_voxel_count"], 3)
        self.assertEqual(gate_counts["depth_gate_voxel_count"], 3)


if __name__ == "__main__":
    unittest.main()
