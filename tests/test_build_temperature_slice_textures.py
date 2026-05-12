from __future__ import annotations

import unittest

import numpy as np

from scripts.build_temperature_slice_textures import (
    equivalent_potential_temperature,
    longitudes_for_source_grid,
    normalize_longitudes_360,
)


class BuildTemperatureSliceTexturesTests(unittest.TestCase):
    def test_equivalent_potential_temperature_matches_dry_limit(self) -> None:
        temperature = np.array([[300.0]], dtype=np.float32)
        specific_humidity = np.array([[0.0]], dtype=np.float32)

        theta_e = equivalent_potential_temperature(
            temperature_k=temperature,
            specific_humidity_kgkg=specific_humidity,
            pressure_hpa=1000.0,
        )

        self.assertAlmostEqual(float(theta_e[0, 0]), 300.0, places=4)

    def test_equivalent_potential_temperature_increases_with_moisture(self) -> None:
        temperature = np.array([[300.0, 300.0]], dtype=np.float32)
        specific_humidity = np.array([[0.004, 0.016]], dtype=np.float32)

        theta_e = equivalent_potential_temperature(
            temperature_k=temperature,
            specific_humidity_kgkg=specific_humidity,
            pressure_hpa=1000.0,
        )

        self.assertGreater(float(theta_e[0, 1]), float(theta_e[0, 0]))
        self.assertGreater(float(theta_e[0, 1]), 335.0)

    def test_normalize_longitudes_360_preserves_temperature_grid_order(self) -> None:
        longitudes = np.array([-180.0, -179.75, 0.0, 179.75], dtype=np.float32)

        normalized = normalize_longitudes_360(longitudes)

        np.testing.assert_allclose(
            normalized,
            np.array([180.0, 180.25, 0.0, 179.75], dtype=np.float32),
        )

    def test_longitudes_for_source_grid_keeps_negative_grid_when_available(self) -> None:
        target = np.array([-180.0, -179.75, 0.0, 179.75], dtype=np.float32)
        source = np.array([-180.0, -179.75, 0.0, 179.75], dtype=np.float32)

        selected = longitudes_for_source_grid(target, source)

        np.testing.assert_allclose(selected, target)


if __name__ == "__main__":
    unittest.main()
