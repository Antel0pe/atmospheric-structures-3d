# Thermal Displacement Score Two-Cluster Time Series

## Method

- Compute canonical same-longitude, same-hemisphere Thermal Displacement scores.
- Use score smoothing sigma `1` native grid cells after matching.
- Fit a separate 2-cluster 1D k-means split for each `(valid_time, pressure_level)` score field.
- The clusters are value clusters, not connected-component regions.
- Use `0.5s` per GIF frame.

## Inputs

- Temperature: `.worktree-shared/data/era5_temperature_2021-11-08t15_to_2021-11-10t00_3hourly_250-1000hpa.nc`.
- Climatology: `.worktree-shared/data/era5_temperature-climatology_1990-2020_11-08_12.nc`.

## Outputs

- Time-step GIFs: `experiments/thermal-displacement-score-two-clusters/time-series-2021-11-08t15-to-2021-11-10t00/gifs`.
- Cluster frame PNGs: `experiments/thermal-displacement-score-two-clusters/time-series-2021-11-08t15-to-2021-11-10t00/frames`.
- Stats: `experiments/thermal-displacement-score-two-clusters/time-series-2021-11-08t15-to-2021-11-10t00/cluster_stats_by_time_level.csv`.

## Interpretation

This extends the two-score-cluster diagnostic over time. Because each map fits its own split, cluster labels mean lower-score/polar-like versus higher-score/equator-like within that one time and pressure level.
