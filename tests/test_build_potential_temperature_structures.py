from __future__ import annotations

import unittest

import numpy as np

from scripts.build_potential_temperature_structures import (
    build_exposed_face_mesh_from_mask,
    build_top_percent_mask,
    compute_dry_potential_temperature,
    keep_components_touching_opposite,
    label_wrapped_components,
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

    def test_build_top_percent_mask_keeps_global_top_tail(self) -> None:
        field = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ],
            dtype=np.float32,
        )

        keep_mask, threshold_value = build_top_percent_mask(field, top_percent=20.0)

        self.assertEqual(threshold_value, 66.0)
        np.testing.assert_array_equal(
            keep_mask,
            np.array(
                [
                    [[False, False], [False, False]],
                    [[False, False], [True, True]],
                ],
                dtype=bool,
            ),
        )

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

    def test_label_wrapped_components_merges_longitude_seam(self) -> None:
        mask = np.array(
            [
                [[True, False, True]],
            ],
            dtype=bool,
        )

        labels, component_count = label_wrapped_components(mask)

        self.assertEqual(component_count, 1)
        self.assertEqual(int(labels[0, 0, 0]), int(labels[0, 0, 2]))

    def test_keep_components_touching_opposite_keeps_only_gradient_touching_components(self) -> None:
        mask = np.array(
            [
                [[True, False, False, True, False]],
            ],
            dtype=bool,
        )
        opposite = np.array(
            [
                [[False, True, False, False, False]],
            ],
            dtype=bool,
        )

        keep_mask, component_count, touching_component_count = keep_components_touching_opposite(
            mask,
            opposite,
        )

        self.assertEqual(component_count, 2)
        self.assertEqual(touching_component_count, 1)
        np.testing.assert_array_equal(
            keep_mask,
            np.array(
                [
                    [[True, False, False, False, False]],
                ],
                dtype=bool,
            ),
        )

    def test_build_exposed_face_mesh_from_mask_wraps_longitude_seam(self) -> None:
        keep_mask = np.array(
            [
                [[True, False, True]],
            ],
            dtype=bool,
        )
        pressure_levels_hpa = np.array([1000.0], dtype=np.float32)
        latitudes = np.array([0.0], dtype=np.float32)
        longitudes = np.array([-180.0, -60.0, 60.0], dtype=np.float32)

        mesh = build_exposed_face_mesh_from_mask(
            source_mask=keep_mask,
            pressure_levels_hpa=pressure_levels_hpa,
            latitudes_deg=latitudes,
            longitudes_deg=longitudes,
            base_radius=100.0,
            vertical_span=12.0,
        )

        self.assertEqual(mesh.voxel_count, 2)
        self.assertGreater(mesh.positions.size, 0)
        self.assertGreater(mesh.indices.size, 0)

    def test_build_exposed_face_mesh_can_suppress_interface_faces(self) -> None:
        source_mask = np.array(
            [
                [[False, True, False]],
            ],
            dtype=bool,
        )
        occupancy_mask = np.array(
            [
                [[False, True, True]],
            ],
            dtype=bool,
        )
        pressure_levels_hpa = np.array([1000.0], dtype=np.float32)
        latitudes = np.array([0.0], dtype=np.float32)
        longitudes = np.array([-180.0, -60.0, 60.0], dtype=np.float32)

        mesh = build_exposed_face_mesh_from_mask(
            source_mask=source_mask,
            occupancy_mask=occupancy_mask,
            pressure_levels_hpa=pressure_levels_hpa,
            latitudes_deg=latitudes,
            longitudes_deg=longitudes,
            base_radius=100.0,
            vertical_span=12.0,
        )

        self.assertEqual(mesh.voxel_count, 1)
        self.assertEqual(int(mesh.indices.size), 30)


if __name__ == "__main__":
    unittest.main()
