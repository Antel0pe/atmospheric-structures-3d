from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_relative_humidity_voxel_shell import (
    clear_output_dir,
    filter_small_wrapped_components,
)


class BuildRelativeHumidityVoxelShellTests(unittest.TestCase):
    def test_filter_small_wrapped_components_merges_seam_connected_labels(self) -> None:
        keep_mask = np.zeros((1, 1, 4), dtype=bool)
        keep_mask[0, 0, 0] = True
        keep_mask[0, 0, -1] = True
        keep_mask[0, 0, 1] = True

        filtered_mask, metadata = filter_small_wrapped_components(
            keep_mask,
            min_component_size=3,
        )

        np.testing.assert_array_equal(filtered_mask, keep_mask)
        self.assertEqual(metadata["component_count_before_filter"], 1)
        self.assertEqual(metadata["component_count_after_filter"], 1)
        self.assertEqual(metadata["removed_component_count"], 0)
        self.assertEqual(metadata["removed_voxel_count"], 0)

    def test_clear_output_dir_preserves_variants_for_baseline_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "index.json").write_text("baseline", encoding="utf-8")
            (output_dir / "2021-11-08t1200" / "metadata.json").parent.mkdir(parents=True)
            (output_dir / "2021-11-08t1200" / "metadata.json").write_text(
                "frame",
                encoding="utf-8",
            )
            preserved = output_dir / "variants" / "min-component-10" / "index.json"
            preserved.parent.mkdir(parents=True)
            preserved.write_text("variant", encoding="utf-8")

            clear_output_dir(output_dir, preserve_child_names={"variants"})

            self.assertFalse((output_dir / "index.json").exists())
            self.assertFalse((output_dir / "2021-11-08t1200").exists())
            self.assertTrue(preserved.exists())


if __name__ == "__main__":
    unittest.main()
