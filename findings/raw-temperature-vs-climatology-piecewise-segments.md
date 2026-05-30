# Raw Temperature Vs Climatology Piecewise Segments

Date: 2026-05-30

Purpose:
- Compare the existing climatology piecewise latitude-profile view directly against raw temperature from `2021-11-08T12:00`.
- Keep the existing pressure-level and regional-window organization, but generate only the requested piecewise-segment plots.

Method:
- Raw input: `data/era5_temperature_2021-11_08-12.nc`
- Raw timestamp: `2021-11-08T12:00`
- Climatology input: `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`
- Climatology operation: mean raw temperature across `valid_time`.
- Levels: `250`, `500`, `850`, `1000 hPa`.
- Regional windows: North America, Europe, Russia.
- Hemispheres: north and south.
- Longitude samples: five per regional window, matching the source experiment.
- Piecewise fit: matching the source experiment: smooth each pole-to-equator profile with 1D Gaussian sigma `4` grid samples, then fit independent broad piecewise linear segments with a `12`-degree minimum segment width and penalized extra breakpoints.

Plot layout:
- Top left: smoothed raw temperature climatology profiles.
- Top right: climatology piecewise segments.
- Bottom left: smoothed raw temperature profiles at `2021-11-08T12:00`.
- Bottom right: raw temperature piecewise segments.

Outputs:
- Script: `tmp/raw-temperature-vs-climatology-piecewise-segments/plot_raw_temperature_vs_climatology_piecewise_segments.py`
- Summary: `tmp/raw-temperature-vs-climatology-piecewise-segments/summary.json`
- Plots: `tmp/raw-temperature-vs-climatology-piecewise-segments/<level>hpa/piecewise-segments/`
- No `original/` or `smoothed-derivative/` output folders are created.

Verification:
- Generated `24` PNGs: `4` levels x `3` regions x `2` hemispheres.
- All PNGs are `2624 x 2080`.
- `summary.json` validates with `python -m json.tool`.
- The output tree contains no `original/` or `smoothed-derivative/` directories.
