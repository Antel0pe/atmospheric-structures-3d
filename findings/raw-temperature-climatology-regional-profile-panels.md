# Raw Temperature Climatology Regional Profile Panels

Date: 2026-05-29

Purpose:
- Combine the regional longitude-window profile/map layout with a direct raw-vs-climatology temperature comparison.
- Each output puts three views in one figure: profile comparison on the left, raw temperature map on the upper right, and raw temperature climatology map on the lower right.

Method:
- Raw input: `data/era5_temperature_2021-11_08-12.nc`
- Raw timestamp: `2021-11-08T12:00`
- Climatology input: `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`
- Climatology operation: mean across `valid_time`.
- Variable: `t`
- Unit: Kelvin
- Levels: `250`, `500`, `850`, `1000 hPa`
- Regional windows: North America, Europe, Russia.
- Hemispheres: north and south.
- Longitude samples: three per regional window, downsampled from the prior five-line regional windows.
- Profile encoding: same color per longitude; solid lines are raw temperature, dashed lines are climatology.
- Map encoding: black vertical lines are the regional window boundaries; colored meridians and 15-degree markers match the profile longitudes.

Outputs:
- Script: `tmp/raw-temperature-climatology-regional-profile-panels/plot_regional_profile_panels.py`
- Summary: `tmp/raw-temperature-climatology-regional-profile-panels/summary.json`
- Plots: `tmp/raw-temperature-climatology-regional-profile-panels/plots/`
- Generated `24` PNGs: `4` levels x `3` regions x `2` hemispheres.

Verification:
- `summary.json` validates with `python -m json.tool`.
- PNG count is `24`.
- Sample PNG dimensions are `2752 x 1600`.
