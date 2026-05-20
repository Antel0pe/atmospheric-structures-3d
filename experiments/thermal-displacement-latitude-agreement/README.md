# Thermal Displacement Latitude Agreement

Scratch experiment for `2021-11-08T12:00` at `1000`, `850`, `500`, and `250 hPa`.

## Method

1. Read raw temperature from `data/era5_temperature_2021-11_08-12.nc`.
2. Read temperature climatology from `data/era5_temperature-climatology_1990-2020_11-08_12.nc`.
3. For each selected pressure level and grid cell, compare the raw temperature to the climatology temperature profile at the same longitude and pressure level.
4. Keep the climatology latitude whose temperature is closest. Exact ties prefer the climatology latitude row closest to the source grid row.
5. Convert the matched latitude to thermal-displacement score:

   ```text
   score = (1 - abs(matched_latitude) / max_abs_latitude) * 100
   ```

   `0` is polar-like and `100` is equator-like.

6. Optionally smooth the signed matched-latitude field before scoring and before the latitude-agreement comparison. Longitude wraps; latitude edge uses nearest.
7. Smooth the score with Gaussian sigma `1` native grid cell by default. Longitude wraps; latitude edge uses nearest.
8. Generate base thermal-displacement maps with land/country borders using a fixed blue-white-red scale: `0` blue, `50` white, `100` red.
9. Generate agreement maps with the same base color scale. For the score-smoothed same-hemisphere run, reconstruct a same-hemisphere matched latitude from the smoothed score, then color cells green where:

   ```text
   abs(source_grid_latitude - smoothed_score_implied_matched_latitude) <= 5 degrees
   ```

9. Generate latitude-difference histograms by pressure level using signed difference:

   ```text
   source_grid_latitude - smoothed_score_implied_matched_latitude
   ```

   Histograms use 1-degree buckets from `-180` to `180`; the `-5` to `+5` agreement band is colored green.

## Outputs

- `output/thermal-displacement-maps/`: base thermal-displacement maps.
- `output/latitude-agreement-green-maps/`: green latitude-agreement overlays.
- `output/latitude-difference-histograms/`: signed latitude-difference histograms and bucket CSVs.
- `output/summary.json`: run metadata.
- `output/agreement_counts.csv`: green-cell counts and latitude-difference summary stats by pressure level.

## Same-Hemisphere Variant

The generator supports `--matching-mode same-hemisphere`. In that mode:

- source latitudes `>= 0` only compare against climatology latitudes `>= 0`
- source latitudes `< 0` only compare against climatology latitudes `< 0`

The same-hemisphere run is saved under `output/same-hemisphere/`.

The current score-smoothed same-hemisphere run is saved under
`output/same-hemisphere-score-smoothed-sigma20-smoothed-agreement/`. It smooths
the final `0-100` thermal-displacement score with Gaussian sigma `20` native
grid cells before drawing both the base thermal-displacement maps and the green
agreement overlay maps. The green agreement mask and latitude-difference
histograms are computed from the smoothed score field. With the native `0.25°`
grid, sigma `20` is about `5°`.

## Score-Displacement Maps

The no-green score-displacement run is saved under
`output/same-hemisphere-score-smoothed-sigma20-displacement-from-actual-latitude/`.
It uses same-hemisphere matching and sigma `20` smoothing on the matched
thermal-displacement score, then colors:

```text
actual_latitude_score = (1 - abs(source_grid_latitude) / max_abs_latitude) * 100
score_displacement = smoothed_matched_score - actual_latitude_score
```

Positive values are red and mean the matched air is more equator-like than the
cell's actual latitude. Negative values are blue and mean the matched air is
more polar-like than the cell's actual latitude. White is zero displacement.
Each pressure level uses a symmetric blue-white-red range centered on zero.

Run:

```bash
conda run -n atmospheric-structures-3d python tmp/thermal-displacement-latitude-agreement/generate_maps.py
conda run -n atmospheric-structures-3d python tmp/thermal-displacement-latitude-agreement/generate_maps.py --matching-mode same-hemisphere --output-dir tmp/thermal-displacement-latitude-agreement/output/same-hemisphere
conda run -n atmospheric-structures-3d python tmp/thermal-displacement-latitude-agreement/generate_maps.py --matching-mode same-hemisphere --smooth-sigma-cells 20 --write-diagnostics --output-dir tmp/thermal-displacement-latitude-agreement/output/same-hemisphere-score-smoothed-sigma20-smoothed-agreement
conda run -n atmospheric-structures-3d python tmp/thermal-displacement-latitude-agreement/generate_score_displacement_maps.py
```

Light smoothed matched-latitude run:

```bash
conda run -n atmospheric-structures-3d python tmp/thermal-displacement-latitude-agreement/generate_maps.py \
  --matched-latitude-smooth-sigma-cells 4 \
  --smooth-sigma-cells 0 \
  --output-dir tmp/thermal-displacement-latitude-agreement/output-smoothed-matched-latitude-sigma4
```
