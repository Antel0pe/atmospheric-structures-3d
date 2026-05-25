# Thermal Displacement Closest-Climatology Temperature Delta

## Experiment

Output folder: `experiments/thermal-displacement-closest-climatology-temperature-delta/`

Plots:
- `experiments/thermal-displacement-closest-climatology-temperature-delta/overview-raw-minus-matched-climatology.png`
- `experiments/thermal-displacement-closest-climatology-temperature-delta/raw-minus-matched-climatology-0250hpa.png`
- `experiments/thermal-displacement-closest-climatology-temperature-delta/raw-minus-matched-climatology-0500hpa.png`
- `experiments/thermal-displacement-closest-climatology-temperature-delta/raw-minus-matched-climatology-0850hpa.png`
- `experiments/thermal-displacement-closest-climatology-temperature-delta/raw-minus-matched-climatology-1000hpa.png`

Method:
- Run canonical same-longitude, same-hemisphere Thermal Displacement for `250`, `500`, `850`, and `1000 hPa`.
- For each cell, color `raw ERA5 temperature - climatology temperature at the matched latitude row`.
- Blue means the raw cell is colder than the closest climatology temperature found by the lookup. Red means it is hotter.
- Black contours show the smoothed `0-100` Thermal Displacement score.

## Finding

This experiment shows that the raw temperature and the closest matched climatology temperature usually differ only a little, with the largest visible range roughly around `-8 K` to `+8 K` across the four pressure levels. The strongest cold tail in this run is a little past that, near `-9 K` at `250 hPa`, but the practical read is still that most cells are near zero because the lookup is explicitly choosing the closest climatology temperature.

The larger residuals rarely seem to mark the significant Thermal Displacement boundaries themselves. They often appear where the score is already maxed out toward the equator-like or polar-like end of the scale, especially near polar regions. In those cases, the Thermal Displacement interpretation is already saturated, so the residual is not a major problem for reading the boundary.

The useful takeaway is that where there is both a big residual and a strong Thermal Displacement score near a boundary, it is mostly confirming that the matched temperature is imperfect in a saturated part of the score field. Since the score is already maxed out there, the difference does not substantially change the boundary interpretation.

## Scale Notes

Summary residual ranges from `summary.csv`:

| Pressure level | Minimum delta | Maximum delta | Mean absolute delta |
| --- | ---: | ---: | ---: |
| `250 hPa` | `-9.38 K` | `+2.98 K` | `0.49 K` |
| `500 hPa` | `-8.15 K` | `+3.41 K` | `0.30 K` |
| `850 hPa` | `-6.81 K` | `+5.87 K` | `0.23 K` |
| `1000 hPa` | `-4.34 K` | `+5.89 K` | `0.15 K` |

Read this as a closest-match residual diagnostic, not a same-location climatology anomaly.
