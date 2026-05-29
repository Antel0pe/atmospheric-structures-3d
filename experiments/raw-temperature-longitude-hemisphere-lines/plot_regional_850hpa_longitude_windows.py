from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).with_name("regenerate_all_hemisphere_profile_map_plots.py")
    runpy.run_path(str(script), run_name="__main__")
