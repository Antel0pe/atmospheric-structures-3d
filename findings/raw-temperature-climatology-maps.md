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
