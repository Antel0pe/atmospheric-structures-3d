# Thermal Displacement And GPH Contours

## Experiment

Output folder: `tmp/thermal-displacement-gph-contours-2021-11-08T12/`

Plots:
- `tmp/thermal-displacement-gph-contours-2021-11-08T12/thermal_displacement/`
- `tmp/thermal-displacement-gph-contours-2021-11-08T12/thermal_displacement_white/`
- `tmp/thermal-displacement-gph-contours-2021-11-08T12/gph/`

Method:
- Thermal Displacement uses the canonical same-longitude, same-hemisphere method from `scripts/thermal_displacement.py`.
- Thermal Displacement contours are drawn every `5` score points for `250`, `500`, `850`, and `1000 hPa`.
- ERA5 geopotential `z` is converted to geopotential height in meters with `z / 9.80665`.
- GPH is raw ERA5 geopotential height; it is not smoothed.

## Finding

GPH does some things similar to Thermal Displacement and generally lines up, but raw GPH is not expected to match every Thermal Displacement feature. Treat the overlap as useful broad structure, not a one-to-one validation of every contour.

The white contour-only Thermal Displacement variant is hard to trace by eye. Color fill is helpful for following individual contour families, even when the contour lines are the main diagnostic.
