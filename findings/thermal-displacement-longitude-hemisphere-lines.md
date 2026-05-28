# Thermal Displacement Longitude Hemisphere Lines

Date: 2026-05-27

This experiment samples 10 longitudes and plots Thermal Displacement score along each meridian, split into Northern and Southern Hemisphere profiles. Both hemispheres use a pole-to-equator x-axis so the north and south profiles can be compared directly.

Method:
- Canonical same-longitude, same-hemisphere Thermal Displacement from `thermal_displacement.md`.
- Raw temperature: `data/era5_temperature_2021-11_08-12.nc`
- Climatology: `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Timestamp: `2021-11-08T12:00`
- Levels: `250`, `500`, `850`, `1000 hPa`
- Score smoothing: sigma `1` native grid cell, after matching.
- Sampled longitudes: `180°E`, `144°W`, `108°W`, `72°W`, `36°W`, `0°E`, `36°E`, `72°E`, `108°E`, `144°E`

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/plot_longitude_hemisphere_lines.py`
- Summary: `tmp/thermal-displacement-longitude-hemisphere-lines/summary.json`
- Plots: `tmp/thermal-displacement-longitude-hemisphere-lines/plots/`

Plot encoding:
- x-axis: latitude path from pole to equator, separately for each hemisphere.
- y-axis: Thermal Displacement score, where `0` is polar-like and `100` is equator-like.
- color: sampled longitude.
- solid line: Northern Hemisphere.
- dashed line: Southern Hemisphere.

## Iteration: Random 850 hPa Longitude Sets

Date: 2026-05-28

User asked to try 5 longitudes per plot instead: 10 lines per plot after splitting north/south, 3 random plots, no longitude overlap between plots, only `850 hPa`, same data timestamp.

Method:
- Same canonical Thermal Displacement method and same timestamp: `2021-11-08T12:00`.
- Pressure level: `850 hPa`.
- Random seed: `20260528`.
- Each plot has 5 sampled longitudes, with north/south split into 10 lines.
- Longitudes do not overlap across the 3 plots.

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/plot_random_850hpa_longitude_sets.py`
- Summary: `tmp/thermal-displacement-longitude-hemisphere-lines/random_850hpa_summary.json`
- Plots: `tmp/thermal-displacement-longitude-hemisphere-lines/plots-random-sets-850hpa/`

Sampled sets:
- Set 1: `115.75°W`, `102°W`, `12°E`, `27°E`, `159.5°E`
- Set 2: `125.25°W`, `113°W`, `56°W`, `42.25°W`, `38.5°W`
- Set 3: `72.5°W`, `70.75°E`, `96.5°E`, `128.25°E`, `169.75°E`

## Iteration: Regional 850 hPa Longitude Windows

Date: 2026-05-28

User asked to repeat the 5-longitude, 10-line version but choose nearby longitudes instead of random ones. Use three 50-degree windows for North America, Europe, and Russia, sample 5 longitudes at 10-degree increments, keep `850 hPa`, and keep the same data timestamp.

Method:
- Same canonical Thermal Displacement method and same timestamp: `2021-11-08T12:00`.
- Pressure level: `850 hPa`.
- Each regional plot has a profile chart plus a Thermal Displacement map.
- The map marks the 50-degree longitude window in black and the 5 sampled meridians in the same colors as the profile lines.

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/plot_regional_850hpa_longitude_windows.py`
- Summary: `tmp/thermal-displacement-longitude-hemisphere-lines/regional_windows_850hpa_summary.json`
- Plots: `tmp/thermal-displacement-longitude-hemisphere-lines/plots-regional-windows-850hpa/`

Windows and sampled longitudes:
- North America window: `130°W` to `80°W`; sampled `130°W`, `120°W`, `110°W`, `100°W`, `90°W`
- Europe window: `10°W` to `40°E`; sampled `10°W`, `0°E`, `10°E`, `20°E`, `30°E`
- Russia window: `60°E` to `110°E`; sampled `60°E`, `70°E`, `80°E`, `90°E`, `100°E`

## Iteration: Organized Profile And Source Map Pairs

Date: 2026-05-28

User asked to add Thermal Displacement maps with lines showing where each profile was extracted, including the previous outputs, and organize the runs so each line plot has an associated map.

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/organize_profiles_with_source_maps.py`
- Manifest: `tmp/thermal-displacement-longitude-hemisphere-lines/organized-profile-map-pairs/manifest.json`
- Organized root: `tmp/thermal-displacement-longitude-hemisphere-lines/organized-profile-map-pairs/`

Structure:
- `even_10_longitudes/<level>hpa/line_profile.png`
- `even_10_longitudes/<level>hpa/thermal_displacement_map.png`
- `random_850hpa/set_<number>/line_profile.png`
- `random_850hpa/set_<number>/thermal_displacement_map.png`
- `regional_850hpa/<region>/line_profile.png`
- `regional_850hpa/<region>/thermal_displacement_map.png`

Each pair folder also has `metadata.json` with the matched grid longitudes, timestamp, pressure level, and source summary file.

## Superseded Iteration: Regional Windows Split By Hemisphere With Raw Temperature Maps

Date: 2026-05-28

User asked to regenerate `plots-regional-windows-850hpa/` so the 3 regional plots become 6 plots: North and South Hemisphere split separately for each region/window. The longitude lines stay the same as the previous regional run; only the hemisphere split and matching raw-temperature maps changed.

Superseded 2026-05-28: user corrected that the companion maps should be Thermal Displacement, not raw temperature. The generated plot folder and summary were replaced by the next iteration.

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/plot_regional_850hpa_longitude_windows.py`
- Summary: `tmp/thermal-displacement-longitude-hemisphere-lines/regional_windows_850hpa_summary.json`
- Regenerated plots: `tmp/thermal-displacement-longitude-hemisphere-lines/plots-regional-windows-850hpa/`

Plot layout:
- Left: Thermal Displacement score profile for one hemisphere only.
- Right: raw `850 hPa` temperature map in Kelvin for the matching longitude/hemisphere window.
- Colored vertical lines on the raw-temperature map match the profile longitudes.
- Black vertical lines mark the 50-degree longitude window boundaries.

## Iteration: All Outputs Split By Hemisphere With Thermal Displacement Maps

Date: 2026-05-28

User asked to apply the same split-hemisphere paired style to all folders generated in this chat and corrected that the companion maps should be Thermal Displacement, not raw temperature. Regenerated the original even-longitude plots, random 850 hPa sets, regional 850 hPa windows, and organized output tree.

Outputs:
- Script: `tmp/thermal-displacement-longitude-hemisphere-lines/regenerate_all_hemisphere_profile_map_plots.py`
- Summary: `tmp/thermal-displacement-longitude-hemisphere-lines/all_hemisphere_profile_map_summary.json`
- Organized manifest: `tmp/thermal-displacement-longitude-hemisphere-lines/organized-profile-map-pairs/manifest.json`
- Regenerated folders: `tmp/thermal-displacement-longitude-hemisphere-lines/plots/`, `tmp/thermal-displacement-longitude-hemisphere-lines/plots-random-sets-850hpa/`, `tmp/thermal-displacement-longitude-hemisphere-lines/plots-regional-windows-850hpa/`, and `tmp/thermal-displacement-longitude-hemisphere-lines/organized-profile-map-pairs/`

Current plot rules:
- Every PNG is one hemisphere only: north or south.
- Every PNG has a Thermal Displacement score line plot on the left and a Thermal Displacement map on the right.
- Extracted longitude meridians on the map use the same colors as the line plot.
- Dot markers on the meridians are placed every `15°` latitude, matching the line-plot x-axis ticks: `90`, `75`, `60`, `45`, `30`, `15`, `0`.
- Regional windows keep their 50-degree window boundaries as black vertical lines.

## Iteration: Fix Short Global Hemisphere Map Panels

Date: 2026-05-28

User noticed some global/random hemisphere maps rendered as short strips while regional maps filled the panel. Cause: the map axis preserved geographic aspect ratio, so a `360° x 90°` hemisphere extent rendered much shorter than the line-plot panel. Regenerated all paired outputs with the map axis stretched to fill the right-hand plot area.

Checked examples:
- `plots-random-sets-850hpa/thermal_displacement_850hpa_random_longitude_set_02_south.png`
- `plots-random-sets-850hpa/thermal_displacement_850hpa_random_longitude_set_01_south.png`
- `organized-profile-map-pairs/even_10_longitudes/0250hpa/north/profile_and_thermal_displacement_map.png`
