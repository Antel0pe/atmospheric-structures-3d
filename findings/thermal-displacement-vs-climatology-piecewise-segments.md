# Thermal Displacement Vs Climatology Piecewise Segments

Date: 2026-05-30

Purpose:
- Repeat the raw-temperature-vs-climatology piecewise-segment experiment using canonical Thermal Displacement score instead of temperature in Kelvin.
- Compare the observed `2021-11-08T12:00` Thermal Displacement profile against a climatological equivalent-latitude baseline.

Method:
- Raw temperature input: `data/era5_temperature_2021-11_08-12.nc`
- Raw timestamp: `2021-11-08T12:00`
- Climatology temperature input: `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Thermal Displacement: canonical same-pressure, same-longitude, same-hemisphere method from `scripts/thermal_displacement.py`.
- Observed TD: raw `2021-11-08T12:00` temperature matched against climatology temperature.
- Climatology TD baseline: climatology temperature matched against itself. This is the folded equivalent-latitude baseline, not a separately archived TD climatology product.
- Canonical score smoothing: Gaussian sigma `1` native grid cell after matching.
- Levels: `250`, `500`, `850`, `1000 hPa`.
- Regional windows: North America, Europe, Russia.
- Hemispheres: north and south.
- Longitude samples: five per regional window, matching the source experiment.
- Piecewise fit: additionally smooth each pole-to-equator TD score profile with 1D Gaussian sigma `4` grid samples, then fit independent broad piecewise linear segments with a `12`-degree minimum segment width and penalized extra breakpoints.

Plot layout:
- Top left: smoothed climatology-baseline TD profiles.
- Top right: climatology-baseline TD piecewise segments.
- Bottom left: smoothed observed TD profiles at `2021-11-08T12:00`.
- Bottom right: observed TD piecewise segments.

Outputs:
- Script: `tmp/thermal-displacement-vs-climatology-piecewise-segments/plot_thermal_displacement_vs_climatology_piecewise_segments.py`
- Summary: `tmp/thermal-displacement-vs-climatology-piecewise-segments/summary.json`
- Plots: `tmp/thermal-displacement-vs-climatology-piecewise-segments/<level>hpa/piecewise-segments/`
- No `original/`, `smoothed-derivative/`, `.npy`, or `.csv` outputs are created.

Verification:
- Generated `24` PNGs: `4` levels x `3` regions x `2` hemispheres.
- All PNGs are `2624 x 2080`.
- `summary.json` validates with `python -m json.tool`.
- Observed canonical TD checks match the reference values: `250 hPa: 47/count 6172`, `500 hPa: 37/count 5954`, `850 hPa: 50/count 8000`, `1000 hPa: 60/count 8251`.
- The output tree contains no `original/`, `smoothed-derivative/`, `.npy`, or `.csv` artifacts.
