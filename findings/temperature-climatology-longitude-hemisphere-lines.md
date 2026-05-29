# Temperature Climatology Longitude Hemisphere Lines

Date: 2026-05-28

This experiment mirrors the Thermal Displacement longitude-profile idea, but plots raw temperature climatology values instead of Thermal Displacement score. It evenly samples 5 global longitude columns and splits the profiles into separate Northern and Southern Hemisphere plots.

Method:
- Input: `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`
- Variable: `t`
- Aggregation: mean over `valid_time`
- Sample count: `31` yearly `1990-11-08T12:00:00` through `2020-11-08T12:00:00`
- Levels: `250`, `500`, `850`, `1000 hPa`
- Sampled longitudes: `180°E`, `108°W`, `36°W`, `36°E`, `108°E`
- Variant: `even-5-longitudes`
- Values stay in Kelvin.

Outputs:
- Script: `tmp/temperature-climatology-longitude-hemisphere-lines/plot_temperature_climatology_longitude_hemisphere_lines.py`
- Summary: `tmp/temperature-climatology-longitude-hemisphere-lines/summary.json`
- Plots: `tmp/temperature-climatology-longitude-hemisphere-lines/even-5-longitudes/`

Plot encoding:
- x-axis: actual latitude in degrees, split by hemisphere.
- y-axis: raw temperature climatology mean in Kelvin.
- color: sampled longitude.
- one PNG per pressure level and hemisphere.

## Iteration: Paired Profile And Raw Temperature Map

Date: 2026-05-28

User asked to follow the previous Thermal Displacement example layout: line plot on the left, raw temperature color plot on the right. Regenerated paired panels for the same 5 evenly sampled global longitudes and the same `250`, `500`, `850`, `1000 hPa` levels.

Outputs:
- Script: `tmp/temperature-climatology-longitude-hemisphere-lines/plot_temperature_climatology_longitude_hemisphere_lines.py`
- Summary: `tmp/temperature-climatology-longitude-hemisphere-lines/summary.json`
- Paired plots: `tmp/temperature-climatology-longitude-hemisphere-lines/even-5-longitudes-profile-raw-temperature-map/`

Current plot rules:
- Left panel: temperature climatology line profiles by latitude.
- Right panel: raw temperature climatology map in Kelvin with coastlines and borders.
- Colored meridians and dot markers on the map show where each profile was extracted.

## Correction: Use Raw Temperature Stack Directly

Date: 2026-05-28

User asked to double-check that the field is truly raw temperature climatology: climatology should mean the average across all years of raw temperature. The first run used `data/era5_temperature-climatology_1990-2020_11-08_12.nc`, which is temperature reconstructed from dry-potential-temperature climatology. Regenerated from the raw temperature stack directly by computing `t.mean("valid_time")` for each pressure level.

Check:
- Raw stack variable: `t(valid_time, pressure_level, latitude, longitude)`
- Aggregation used now: mean over `valid_time`
- The precomputed reconstructed file was numerically almost identical at the requested levels because the pressure-level conversion is linear at fixed pressure, but the current outputs now match the requested raw-temperature-climatology definition directly.
