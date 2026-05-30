# Raw Temperature Climatology Maps

Date: 2026-05-28

Purpose:
- Plot raw temperature climatology directly on global maps, using the raw 1990-2020 pressure-level temperature stack rather than a derived Thermal Displacement or equivalent-latitude score.

Method:
- Input: `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`
- Variable: `t`
- Operation: mean across the dataset's `valid_time` entries.
- Levels: `250`, `500`, `850`, `1000 hPa`
- Domain: global latitude/longitude
- Unit: Kelvin
- Map styling: blue-white-red temperature colors, per-level min/max scaling, coastlines, borders, and gridlines.

Outputs:
- Script: `tmp/raw-temperature-climatology-maps/plot_raw_temperature_climatology_maps.py`
- Summary: `tmp/raw-temperature-climatology-maps/summary.json`
- Plots: `tmp/raw-temperature-climatology-maps/plots/`

Update 2026-05-29:
- Adapted the climatology experiment to the regional longitude-window paired design from `experiments/raw-temperature-longitude-hemisphere-lines/plots-regional-windows-850hpa/`.
- Kept the plotted field as raw temperature climatology from `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`, averaged across `valid_time`.
- Generated paired plots with profile lines on the left and a regional climatology temperature map with extraction meridians on the right.
- Levels: `250`, `500`, `850`, `1000 hPa`.
- Regional windows: North America, Europe, Russia.
- Hemispheres: north and south.
- Outputs: `experiments/raw-temperature-climatology-maps/plots-regional-windows/`
- Summary: `experiments/raw-temperature-climatology-maps/summary.json`

Update 2026-05-29 later:
- Ran a tmp experiment that mirrors the `experiments/raw-temperature-climatology-maps/plots-regional-windows/` style, data, layout, regional windows, pressure levels, and longitude samples, but organizes the outputs by pressure level.
- Added a second plot for each level/region/hemisphere: left panel is the raw temperature climatology longitude profile lightly smoothed along the pole-to-equator latitude path, and right panel is the first derivative of that smoothed line. The same longitude colors are used on both sides.
- Smoothing: 1D Gaussian along each longitude profile, sigma `2` grid samples, edge-padded.
- Derivative: gradient of the smoothed profile against absolute latitude path degrees.
- Levels: `250`, `500`, `850`, `1000 hPa`.
- Regional windows: North America, Europe, Russia.
- Hemispheres: north and south.
- Outputs: `tmp/raw-temperature-climatology-longitude-derivative/<level>hpa/original/` and `tmp/raw-temperature-climatology-longitude-derivative/<level>hpa/smoothed-derivative/`.
- Summary: `tmp/raw-temperature-climatology-longitude-derivative/summary.json`.

Update 2026-05-29 piecewise iteration:
- The derivative view was too local/noisy for the user's goal. The target is broad slope-regime structure, such as weak change from `90-45`, stronger warming from `45-15`, then leveling from `15-0`, while still allowing small wiggles.
- Added a `piecewise-segments` variant for every level, regional window, and hemisphere.
- Each longitude line is fit independently, with no shared breakpoints across the five longitudes.
- Method: smooth each profile with a 1D Gaussian sigma `4` grid samples, then fit independent piecewise linear segments with dynamic programming. Segment width must be at least `12` latitude degrees, and extra breakpoints are penalized so the fit favors distinct broad slope-regime changes rather than small wiggles.
- The plot layout keeps the smoothed profile on the left and shows the per-longitude piecewise fit on the right. Breakpoints are shown as colored dots plus faint vertical guides in the same longitude color.
- Outputs: `tmp/raw-temperature-climatology-longitude-derivative/<level>hpa/piecewise-segments/`.
- Summary metadata now includes each line's segment count, breakpoint latitudes, and segment slopes in `tmp/raw-temperature-climatology-longitude-derivative/summary.json`.
