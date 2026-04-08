from __future__ import annotations

import numpy as np

from scripts.moisture_structures import build_radius_lookup, lat_lon_to_xyz
from scripts.view_debug_moisture import (
    choose_nearest_component_cell,
    source_radius_to_pressure_hpa,
    world_radius_to_source_radius,
    xyz_to_lat_lon_radius,
)


def test_xyz_to_lat_lon_radius_round_trips_viewer_coordinates() -> None:
    xyz = lat_lon_to_xyz(12.5, 45.0, 123.0)
    lat_deg, lon_deg, radius = xyz_to_lat_lon_radius(
        {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
    )

    np.testing.assert_allclose(radius, 123.0, atol=1e-5)
    np.testing.assert_allclose(lat_deg, 12.5, atol=1e-5)
    np.testing.assert_allclose(lon_deg, 45.0, atol=1e-5)


def test_pressure_conversion_matches_radius_lookup_levels() -> None:
    pressure_levels = np.asarray([1000.0, 850.0, 700.0, 500.0], dtype=np.float32)
    radii = build_radius_lookup(pressure_levels, base_radius=100.0, vertical_span=12.0)

    for pressure_hpa, source_radius in zip(pressure_levels, radii, strict=True):
        reconstructed = source_radius_to_pressure_hpa(
            float(source_radius),
            base_radius=100.0,
            vertical_span=12.0,
        )
        np.testing.assert_allclose(reconstructed, float(pressure_hpa), atol=1.0)


def test_world_radius_to_source_radius_reverses_vertical_exaggeration() -> None:
    source_radius = 104.5
    world_radius = 100.0 + 10.0 + (source_radius - 100.0) * 2.35

    reconstructed = world_radius_to_source_radius(
        world_radius,
        base_radius=100.0,
        vertical_exaggeration=2.35,
    )

    np.testing.assert_allclose(reconstructed, source_radius, atol=1e-6)


def test_choose_nearest_component_cell_wraps_longitude_distance() -> None:
    component_mask = np.zeros((2, 2, 8), dtype=bool)
    component_mask[1, 1, 7] = True

    coord = choose_nearest_component_cell(
        component_mask,
        pressure_levels=np.asarray([1000.0, 850.0], dtype=np.float32),
        latitudes=np.asarray([10.0, 0.0], dtype=np.float32),
        longitudes=np.asarray([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=np.float32),
        pressure_hpa=850.0,
        lat_deg=0.0,
        lon_deg=1.0,
    )

    assert coord == (1, 1, 7)
