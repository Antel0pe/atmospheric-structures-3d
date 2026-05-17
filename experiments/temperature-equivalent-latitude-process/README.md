# Temperature Equivalent-Latitude Process

Scratch workflow for converting raw ERA5 temperature into same-longitude
climatology-equivalent thermal-displacement score maps.

Run from the repository root:

```bash
conda run -n atmospheric-structures-3d python tmp/temperature-equivalent-latitude-process/process_temperature_equivalent_latitude.py
```

Current default output:

- `output/matched-score-smoothed-range-tiebreak/histograms/`: one score-bucket histogram per pressure level
- `output/matched-score-smoothed-range-tiebreak/maps/`: one land-border score map per pressure level
- `output/matched-score-smoothed-range-tiebreak/arrays/`: matched-latitude and score arrays as `.npy`
- `output/matched-score-smoothed-range-tiebreak/summary.json`: selected bucket and plot paths for every level

Method:

1. Load raw temperature from `data/era5_temperature_2021-11_08-12.nc`.
2. Load matched-grid temperature climatology from
   `data/era5_temperature-climatology_1990-2020_11-08_12.nc`.
3. For each raw cell, use its pressure level, longitude, and raw temperature value.
   At the same pressure level and longitude in the climatology, find the latitude
   whose climatological temperature is closest to the raw temperature.
   If two climatology latitudes are tied, choose the one whose latitude row is
   closest to the source cell's own latitude row.
4. Store that matched latitude, then convert it into a score:
   `score_points = (1 - abs(matched_latitude) / max_abs_latitude) * 100`.
   `0` is polar-like; `100` is equator-like.
5. Smooth the score field with Gaussian `sigma=1` native grid cell. Longitude
   wraps; latitude edges use nearest-neighbor extension.
6. Histogram the smoothed score into 1-point buckets centered on integer values from
   `0` to `100`.
7. Use the middle 60% of each level's score range:
   `min + 20% of range` through `max - 20% of range`.
8. Within nonzero buckets inside that middle range, choose the bucket with the fewest
   cells. That bucket is colored yellow in the histogram.
9. Plot the thermal-displacement score with a blue-white-red scale, with
   white centered on the selected bucket center.
