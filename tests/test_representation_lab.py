from __future__ import annotations

import unittest

import numpy as np

from scripts.representation_lab import (
    bridge_mask,
    build_structure_of_data_interpretation,
    build_seed_grow_mask,
    build_structure_probe_interpretation,
    build_structure_probe_metrics,
    classify_structure_type,
    compute_top_share_mask,
    compute_vertical_coherence,
    detect_sign_mode,
    mass_basis,
)


class RepresentationLabTests(unittest.TestCase):
    def test_mass_basis_uses_absolute_magnitude_for_mixed_field(self) -> None:
        field = np.array([[[-2.0, 1.0], [3.0, -4.0]]], dtype=np.float32)
        mass, basis = mass_basis(field)
        self.assertEqual(basis, "absolute_magnitude")
        np.testing.assert_array_equal(mass, np.abs(field))

    def test_detect_sign_mode_ignores_tiny_negative_artifact_fraction(self) -> None:
        field = np.ones((1, 10, 10), dtype=np.float32)
        field[0, 0, 0] = -1e-6
        self.assertEqual(detect_sign_mode(field), "nonnegative")

    def test_compute_vertical_coherence_detects_short_runs(self) -> None:
        mask = np.zeros((4, 2, 1), dtype=bool)
        mask[0, 0, 0] = True
        mask[2, 0, 0] = True
        mask[0:3, 1, 0] = True
        metrics = compute_vertical_coherence(mask)
        self.assertAlmostEqual(metrics["mean_vertical_run_length"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["coherent_column_fraction"], 0.5)

    def test_bridge_mask_fills_short_gap(self) -> None:
        values = np.array([[[3.0]], [[2.0]], [[4.0]]], dtype=np.float32)
        occupied = np.array([[[True]], [[False]], [[True]]], dtype=bool)
        filled, metadata = bridge_mask(values, occupied, bridge_levels=1)
        np.testing.assert_array_equal(filled, np.ones_like(occupied, dtype=bool))
        self.assertEqual(metadata["added_voxel_count"], 1)

    def test_seed_grow_expands_from_seed_into_relaxed_region(self) -> None:
        values = np.array(
            [
                [[9.0, 5.0, 4.0]],
            ],
            dtype=np.float32,
        )
        grown, metadata = build_seed_grow_mask(
            values,
            threshold_percent=67.0,
            tail="high",
            grow_rule="relaxed-half",
        )
        self.assertEqual(metadata["seed_voxel_count"], 2)
        self.assertEqual(metadata["grown_voxel_count"], 3)
        np.testing.assert_array_equal(grown, np.array([[[True, True, True]]], dtype=bool))

    def test_structure_of_data_interpretation_flags_strong_imbalances(self) -> None:
        class Cube:
            canonical_field = "specific_humidity"
            transform = {"anomaly": "none"}

        metrics = {
            "distribution": {
                "tail_to_core_ratio": 8.5,
                "outlier_fraction_above_3sigma": 0.03,
            },
            "sign_structure": {
                "positive_fraction": 0.95,
                "negative_fraction": 0.0,
                "positive_signal_share": 1.0,
            },
            "vertical_structure": {
                "below_700_hpa_signal_fraction": 0.92,
                "above_300_hpa_signal_fraction": 0.01,
                "between_level_variance_fraction": 0.68,
                "level_mean_separation_ratio": 1.5,
                "profile_direction": "decreases_with_height",
                "profile_monotonicity_fraction": 0.94,
            },
            "horizontal_structure": {
                "latitudinal_signal_share": {
                    "within_20": {"signal_fraction": 0.82, "area_fraction": 0.34},
                },
                "top_5_percent_columns_signal_share": 0.41,
                "north_hemisphere_signal_share": 0.63,
                "south_hemisphere_signal_share": 0.37,
                "spatial_entropy": 0.22,
            },
            "cross_level_scale": {"representative_span_ratio": 5.0},
        }
        interpretation = build_structure_of_data_interpretation(metrics, Cube())
        self.assertEqual(
            interpretation["structure_probe_readiness"]["verdict"],
            "needs_preconditioning",
        )
        self.assertIn("strong_vertical_stratification", interpretation["imbalance_flags"])
        self.assertIn("near_surface_dominance", interpretation["imbalance_flags"])

    def test_structure_type_classifies_sheet_like_case(self) -> None:
        structure_type = classify_structure_type(
            method="threshold",
            num_components=6,
            largest_component_fraction=0.35,
            mean_vertical_span=1.4,
            single_level_fraction=0.75,
            shape_class="sheet-like",
            surface_to_volume_ratio=3.2,
        )
        self.assertEqual(structure_type, "layered sheets")

    def test_structure_probe_metrics_identify_single_dominant_mass(self) -> None:
        values = np.ones((3, 4, 4), dtype=np.float32)
        metrics, _, _ = build_structure_probe_metrics(
            values=values,
            method="threshold",
            threshold_percent=100.0,
            tail="high",
            grow_rule=None,
            bridge_levels=0,
            morphology="none",
        )
        self.assertEqual(metrics["component_structure"]["num_components"], 1)
        self.assertGreaterEqual(metrics["component_structure"]["largest_component_fraction"], 0.99)
        interpretation = build_structure_probe_interpretation(metrics)
        self.assertEqual(
            interpretation["promotion_decision"]["decision"],
            "do_not_promote_yet",
        )


if __name__ == "__main__":
    unittest.main()
