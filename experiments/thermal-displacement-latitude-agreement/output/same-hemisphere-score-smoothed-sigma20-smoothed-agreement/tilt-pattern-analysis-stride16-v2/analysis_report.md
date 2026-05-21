# Tilt Pattern Analysis

Input: same-hemisphere thermal-displacement score, Gaussian score smoothing sigma `20`, longitude window `-125°..-50°`, northern latitude window `89°N..0°`, every 16th longitude.

## Methods

- `score50_crossing`: interpolated latitude where score crosses `50`; if multiple crossings exist, choose the crossing with strongest local slope.
- `max_gradient`: latitude of strongest meridional score gradient between `15°N` and `75°N`.
- `gradient_centroid`: gradient-weighted transition latitude where score is between `20` and `80`.
- `profile_shift_correlation`: latitude shift needed to align each longitude profile against the westernmost longitude profile.

## Main Read

The clearest within-level longitudinal tilt appears at `250 hPa` and `500 hPa`.

- `250 hPa` is strongest by `score50_crossing`: slope `+0.42 latitude degrees per longitude degree`, `R2 0.66`.
- `250 hPa` is also strong by `gradient_centroid`: slope `+0.25`, `R2 0.71`.
- `500 hPa` is very coherent by `gradient_centroid`: slope `+0.10`, `R2 0.95`, and moderately coherent by `score50_crossing`: slope `+0.14`, `R2 0.59`.
- `1000 hPa` and `850 hPa` are weaker and more method-dependent.

Positive longitude slope means the estimated transition latitude moves poleward as longitude increases eastward across `-125°..-50°`.

The cross-pressure signal is moderate, not decisive:

- `gradient_centroid` cross-pressure plane: longitude slope `+0.10`, height-proxy slope `-2.22`, `R2 0.44`.
- `score50_crossing` cross-pressure plane: longitude slope `+0.18`, height-proxy slope `-3.54`, `R2 0.39`.
- `max_gradient` cross-pressure plane: longitude slope `+0.18`, height-proxy slope `+10.32`, `R2 0.27`.

Interpretation: there is a real-ish longitudinal tilt in the score transition, strongest aloft and at midlevel. The pressure-level connection exists visually in the pressure waterfalls, but with only four levels it should be treated as a coarse diagnostic rather than proof of a vertically coherent sloping structure.

## Outputs

- `plots/within_level_fit_score50_crossing.png`
- `plots/within_level_fit_max_gradient.png`
- `plots/within_level_fit_gradient_centroid.png`
- `plots/within_level_fit_profile_shift_correlation.png`
- `plots/cross_pressure_transition_latitude_score50_crossing.png`
- `plots/cross_pressure_transition_latitude_max_gradient.png`
- `plots/cross_pressure_transition_latitude_gradient_centroid.png`
- `pressure-waterfalls/pressure_waterfall_longitude_mean.png`
- `pressure-waterfalls/pressure_waterfall_lon_m125.png`
- `pressure-waterfalls/pressure_waterfall_lon_m109.png`
- `pressure-waterfalls/pressure_waterfall_lon_m89.png`
- `pressure-waterfalls/pressure_waterfall_lon_m69.png`
- `pressure-waterfalls/pressure_waterfall_lon_m53.png`
- `tilt_metrics.csv`
