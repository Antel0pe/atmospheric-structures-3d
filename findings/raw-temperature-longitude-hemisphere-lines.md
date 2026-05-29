# Raw Temperature Longitude Hemisphere Lines

Date: 2026-05-28

Purpose:
- Replicate the longitude/hemisphere profile-map experiment from `experiments/thermal-displacement-longitude-hemisphere-lines/`, but use raw ERA5 pressure-level temperature everywhere instead of the derived equivalent-latitude field.

Method:
- Raw temperature file: `data/era5_temperature_2021-11_08-12.nc`
- Variable: `t`
- Timestamp: `2021-11-08T12:00`
- Unit: Kelvin
- Levels: `250`, `500`, `850`, `1000 hPa`
- No climatology lookup, equivalent-latitude matching, derived scoring, or smoothing.
- Profiles use raw temperature along sampled meridians, split into north/south hemisphere plots with the same pole-to-equator x-axis convention as the source experiment.
- Maps use raw temperature with per-pressure-level min/max color scaling.

Outputs:
- Script root: `tmp/raw-temperature-longitude-hemisphere-lines/`
- Main generator: `tmp/raw-temperature-longitude-hemisphere-lines/regenerate_all_hemisphere_profile_map_plots.py`
- Summary: `tmp/raw-temperature-longitude-hemisphere-lines/all_hemisphere_profile_map_summary.json`
- Even-longitude plots: `tmp/raw-temperature-longitude-hemisphere-lines/plots/`
- Random 850 hPa plots: `tmp/raw-temperature-longitude-hemisphere-lines/plots-random-sets-850hpa/`
- Regional 850 hPa plots: `tmp/raw-temperature-longitude-hemisphere-lines/plots-regional-windows-850hpa/`
- Organized paired tree: `tmp/raw-temperature-longitude-hemisphere-lines/organized-profile-map-pairs/`

Generated:
- `40` PNGs total: `20` plot-folder outputs plus `20` matching organized-profile-map pair outputs.
- JSON summaries and metadata validate with `python -m json.tool`.
- Representative PNGs were checked as valid `2624 x 1152` images.

Raw temperature ranges:
- `250 hPa`: `200.88` to `235.71 K`
- `500 hPa`: `228.27` to `271.86 K`
- `850 hPa`: `243.51` to `305.06 K`
- `1000 hPa`: `247.79` to `314.84 K`
