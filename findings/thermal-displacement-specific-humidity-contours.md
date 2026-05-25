# Thermal Displacement Contours Over Specific Humidity

## Experiment

Output folder: `tmp/thermal-displacement-contour-paths-specific-humidity/`

Plots:
- `tmp/thermal-displacement-contour-paths-specific-humidity/maps/thermal_displacement_contour_paths_q_250hpa.png`
- `tmp/thermal-displacement-contour-paths-specific-humidity/maps/thermal_displacement_contour_paths_q_500hpa.png`
- `tmp/thermal-displacement-contour-paths-specific-humidity/maps/thermal_displacement_contour_paths_q_850hpa.png`
- `tmp/thermal-displacement-contour-paths-specific-humidity/maps/thermal_displacement_contour_paths_q_1000hpa.png`

Method:
- Color the full map by same-level ERA5 specific humidity `q` from `data/era5_specific-humidity_2021-11_08-12.nc`.
- Overlay black `5`-point Thermal Displacement score contours.
- Thermal Displacement uses same-hemisphere matching and `sigma=20` score smoothing, matching the existing global contour-path plotting convention.

## Finding

At lower levels, especially `1000`, `850`, and to a lesser extent `500 hPa`, there is visible moisture reaching out toward places where hot extrusion appears in the Thermal Displacement geometry. In these areas, the black contours compress poleward while the humidity field still extends into or near that compressed-contour zone.

At `250 hPa`, this relationship is not very apparent. The humidity is much weaker overall and is mostly concentrated near the equator rather than clearly extending into the poleward-compressed hot-extrusion contour regions.

## Scale Caveat

The color scale changes substantially by pressure level because each panel is scaled to that level's own specific-humidity range. The maxima are:

| Pressure level | Maximum `q` |
| --- | ---: |
| `250 hPa` | `0.000446 kg kg^-1` |
| `500 hPa` | `0.006291 kg kg^-1` |
| `850 hPa` | `0.019852 kg kg^-1` |
| `1000 hPa` | `0.022491 kg kg^-1` |

So a saturated-looking blue at `250 hPa` is still far drier in absolute terms than strong blue at `850` or `1000 hPa`. The main result should be read as within-level spatial structure, not equal absolute moisture across pressure levels.
