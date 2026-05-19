# Thermal Displacement

Reference implementation: [`scripts/thermal_displacement.py`](scripts/thermal_displacement.py)

When the user says "thermal displacement", start from this method unless they
explicitly ask for a different experiment. The method is an equivalent-latitude
temperature lookup: it asks what climatological latitude has the closest
temperature to the raw cell, while keeping pressure level, longitude, and
hemisphere fixed.

## Canonical Method

1. Load raw ERA5 pressure-level temperature, usually
   `data/era5_temperature_2021-11_08-12.nc`.
2. Load matched-grid temperature climatology, usually
   `data/era5_temperature-climatology_1990-2020_11-08_12.nc`.
3. For each pressure level and each raw grid cell, take the raw temperature,
   source latitude row, and longitude.
4. At the same pressure level and longitude in the climatology, search only
   climatology latitudes in the source cell's same hemisphere for the
   climatology temperature closest to the raw cell temperature. Northern
   Hemisphere source rows search `latitude >= 0`; Southern Hemisphere source
   rows search `latitude < 0`.
5. If two climatology latitude rows are exactly tied in temperature distance,
   choose the climatology latitude row closest to the source cell's own latitude
   row. This tie breaker reduces row-to-row jumpiness.
6. Store the matched climatology latitude in degrees. This intermediate field
   can range from roughly `-90..90`.
7. Convert matched latitude into Thermal Displacement score points:

   ```text
   score_points = (1 - abs(matched_latitude) / max_abs_latitude) * 100
   ```

   `0` means polar-like. `100` means equator-like. North and south are
   intentionally collapsed together through `abs(latitude)`.
8. Smooth the score field after matching with a light Gaussian, usually
   `sigma=1` native grid cell. Longitude wraps; latitude edges use nearest.
9. Define centered 1-point score buckets from `0` through `100`, such as
   `39.5-40.5`, `40.5-41.5`, etc.
10. For each pressure level, find that level's numeric score range, then take
    the middle 60% of that range:

    ```text
    lower = score_min + 0.20 * (score_max - score_min)
    upper = score_max - 0.20 * (score_max - score_min)
    ```

    This is range-based, not percentile-based.
11. Inside that range-based middle 60%, choose the nonzero 1-point bucket with
    the fewest cells. Blue-white-red maps can use that bucket center as the
    white point for that pressure level.
12. Draw maps with a continuous blue-white-red scale over each level's own
    score min/max. Blue is polar-like, red is equator-like, and white is the
    selected bucket center.

## Critical Ordering

Do not smooth raw temperature first for the canonical method.

The historical smooth Thermal Displacement maps match this ordering:

```text
raw temperature
-> same-longitude, same-hemisphere climatology latitude match
-> 0..100 score
-> smooth score with sigma=1
-> histogram buckets / selected white point
```

The alternative ordering below produces different white centers and should only
be used when the user explicitly asks for a raw-smoothed experiment:

```text
raw temperature
-> smooth raw temperature
-> same-longitude, same-hemisphere climatology latitude match
-> 0..100 score
```

The legacy cross-hemisphere behavior searched both hemispheres at the same
pressure level and longitude. Use it only for reproducing older output:

```bash
conda run -n atmospheric-structures-3d python scripts/thermal_displacement.py \
  --allow-cross-hemisphere \
  --output-dir tmp/thermal-displacement-cross-hemisphere
```

## Known Default Check

For `2021-11-08T12:00`, using the default temperature and climatology files,
the same-hemisphere default with `sigma=1` score smoothing should reproduce
these selected white centers and bucket counts:

| Pressure | White center | Bucket count |
| --- | ---: | ---: |
| `1000 hPa` | `60` | `8251` |
| `850 hPa` | `50` | `8000` |
| `500 hPa` | `37` | `5954` |
| `250 hPa` | `47` | `6172` |

If those four values change unexpectedly, first check the hemisphere restriction,
smoothing order, bucket definition, tie breaker, selected timestamp, and
climatology variable.

For legacy cross-hemisphere reproduction, the same four levels should be:

| Pressure | White center | Bucket count |
| --- | ---: | ---: |
| `1000 hPa` | `53` | `7858` |
| `850 hPa` | `50` | `6707` |
| `500 hPa` | `40` | `6181` |
| `250 hPa` | `54` | `5958` |

## How To Use The Script

Use the repo conda environment:

```bash
conda run -n atmospheric-structures-3d python scripts/thermal_displacement.py
```

Useful options:

```bash
conda run -n atmospheric-structures-3d python scripts/thermal_displacement.py \
  --pressure-levels 1000,850,500,250 \
  --score-smooth-sigma-cells 1 \
  --output-dir tmp/thermal-displacement-reference
```

By default the script writes only summary outputs:

- `selected_buckets.csv`
- `summary.json`

Use `--write-arrays` only when an experiment needs reusable `.npy` arrays:

```bash
conda run -n atmospheric-structures-3d python scripts/thermal_displacement.py \
  --write-arrays \
  --output-dir tmp/thermal-displacement-reference-with-arrays
```

For map and histogram rendering, use this script's functions as the source of
truth for the computed arrays and bucket centers. Plotting style can vary by
experiment, but the data transform above should not drift silently.

## Interpretation Notes

Thermal Displacement is not a direct temperature anomaly and not a literal air
parcel trajectory. It is a temperature-equivalent latitude score against the
matched climatological temperature field.

The score is useful for showing whether a cell's temperature is more polar-like
or equator-like for its longitude and pressure level. It does not by itself
identify where hot and cold air are fighting. For thermal-conflict work, treat
Thermal Displacement as the identity field, then add a separate spatial conflict
metric such as gradient strength, isotherm compression, or poleward/equatorward
opposition.
