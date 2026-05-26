# Thermal Displacement Score Two-Cluster Experiment

## Method

- Compute canonical same-longitude, same-hemisphere Thermal Displacement scores.
- Use score smoothing sigma `1` native grid cells after matching.
- Fit one shared 2-cluster 1D k-means split across all selected pressure-level score values.
- Apply that score-value split back to each pressure level. The clusters are not connected-component regions.
- Require each cluster to contain at least `10%` of finite cells overall and per level.

## Result

- Shared split threshold: `47.97` score points.
- Cluster centers: `22.11` and `73.83` score points.
- Combined fractions: low/polar-like `0.462`, high/equator-like `0.538`.
- Both clusters pass the balance guard; no singleton or tiny cluster was accepted.

## Interpretation

This is a value split of the Thermal Displacement identity field, not a spatial object extraction. The same cluster can appear in many disconnected places because location was not part of the clustering.
