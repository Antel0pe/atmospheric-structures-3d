# Thermal Displacement Unsmoothed Contour-Only Maps

This experiment tested whether raw, unsmoothed same-hemisphere Thermal Displacement score contours can stand on their own as a line-only global map. It used the standard `250`, `500`, `850`, and `1000 hPa` levels, global longitude/latitude, no score smoothing, no color fill, score contours every `5` points, land/country borders, and a tiny-segment filter.

## Outputs

- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-unsmoothed-contours-only-minlen4-250-500-850-1000/heatmaps/heatmap_250hpa.png`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-unsmoothed-contours-only-minlen4-250-500-850-1000/heatmaps/heatmap_500hpa.png`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-unsmoothed-contours-only-minlen4-250-500-850-1000/heatmaps/heatmap_850hpa.png`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-unsmoothed-contours-only-minlen4-250-500-850-1000/heatmaps/heatmap_1000hpa.png`

## Iterations

- 2026-05-25: Initial line-only variant used no score smoothing, no color fill, and `5`-point score contours.
- 2026-05-25: Tiny contour segments were filtered more aggressively with `min length = 8 deg` and `min vertices = 24`, overwriting the same four PNGs in place.

## User Feedback

- 2026-05-25: No smoothing is horrible with Thermal Displacement. The line-only/no-color version is also very hard to understand. Future Thermal Displacement contour maps should use smoothing and retain contextual color or another readable background unless explicitly testing line-only output.
