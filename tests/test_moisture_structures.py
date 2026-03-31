from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.moisture_structures import (
    build_radius_lookup,
    build_threshold_mask,
    build_timestamp_payload,
    compute_per_level_thresholds_from_array,
    iter_wrapped_components,
    write_timestamp_assets,
)


class MoistureStructuresTests(unittest.TestCase):
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

        payload = build_timestamp_payload(
            field=field,
            thresholds=thresholds,
            pressure_levels=pressure_levels,
            latitudes=latitudes,
            longitudes=longitudes,
            radius_lookup=radius_lookup,
            min_component_size=1,
            gaussian_sigma=0.4,
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


if __name__ == "__main__":
    unittest.main()
