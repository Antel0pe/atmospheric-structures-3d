from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from scripts.build_potential_temperature_climatology import (
    build_climatology,
    compute_dry_potential_temperature,
)


class BuildPotentialTemperatureClimatologyTests(unittest.TestCase):
    def test_compute_dry_potential_temperature_scales_by_pressure_level(self) -> None:
        temperature = xr.DataArray(
            np.array(
                [
                    [
                        [[300.0]],
                        [[280.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            dims=("valid_time", "pressure_level", "latitude", "longitude"),
            coords={
                "valid_time": np.array(
                    ["1990-11-08T12:00:00"],
                    dtype="datetime64[s]",
                ),
                "pressure_level": np.array([1000.0, 500.0], dtype=np.float32),
                "latitude": np.array([45.0], dtype=np.float32),
                "longitude": np.array([-75.0], dtype=np.float32),
            },
        )

        theta = compute_dry_potential_temperature(temperature)

        self.assertAlmostEqual(float(theta.values[0, 0, 0, 0]), 300.0, places=4)
        self.assertGreater(float(theta.values[0, 1, 0, 0]), 340.0)

    def test_build_climatology_returns_mean_std_and_counts(self) -> None:
        theta = xr.DataArray(
            np.array(
                [
                    [[[300.0]], [[320.0]]],
                    [[[302.0]], [[326.0]]],
                    [[[304.0]], [[332.0]]],
                ],
                dtype=np.float32,
            ),
            dims=("valid_time", "pressure_level", "latitude", "longitude"),
            coords={
                "valid_time": np.array(
                    [
                        "1990-11-08T12:00:00",
                        "1991-11-08T12:00:00",
                        "1992-11-08T12:00:00",
                    ],
                    dtype="datetime64[s]",
                ),
                "pressure_level": np.array([1000.0, 500.0], dtype=np.float32),
                "latitude": np.array([45.0], dtype=np.float32),
                "longitude": np.array([-75.0], dtype=np.float32),
            },
            name="theta",
        )

        climatology = build_climatology(theta)

        np.testing.assert_allclose(
            climatology["theta_climatology_mean"].values[:, 0, 0],
            np.array([302.0, 326.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            climatology["theta_climatology_std"].values[:, 0, 0],
            np.array([2.0, 6.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            climatology["theta_sample_count"].values[:, 0, 0],
            np.array([3, 3], dtype=np.int16),
        )


if __name__ == "__main__":
    unittest.main()
