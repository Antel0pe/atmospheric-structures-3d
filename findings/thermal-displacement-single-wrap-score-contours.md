# Thermal Displacement Single-Wrap Score Contours

## Experiment

Generate global same-hemisphere Thermal Displacement score heatmaps for all available `250-1000 hPa` levels, with 5-point score contours.

Outputs:
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/all_contours/`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/single_wrap_contours/`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/filter_review_overlays/`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/shape_overlays/`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/summary.json`
- `experiments/thermal-displacement-latitude-agreement/output/global-score-contours-step5-single-wrap-filter-250-1000/contour_segment_filter_decisions.csv`

## Method

- Score method: same-longitude, same-hemisphere Thermal Displacement.
- Score smoothing: Gaussian sigma `20` native grid cells, with longitude wrapping.
- Color scale: fixed blue-white-red score scale, `0` blue, `50` white, `100` red.
- Contours: score levels `5, 10, ..., 95`.

The filtered set keeps a connected contour segment only if:
- its endpoints touch opposite longitude edges
- it has exactly one contiguous contact with the west edge and one with the east edge
- the west/east endpoints meet after wrapping within `3°` latitude
- all vertices stay inside the latitude domain

This removes closed regional loops and same-side dateline fragments while preserving globe-wrapping score bands.

## Quick Read

For `650 hPa`, the unfiltered set found `55` contour segments. The single-wrap filter kept `25` and rejected `30`, all because their endpoints did not span both longitude edges.

Across all `21` pressure levels, the filter kept `545` of `1161` contour segments and rejected `616`. The validation check found `0` kept-line failures: every retained segment had exactly one west-edge contact run, exactly one east-edge contact run, and a west/east seam latitude gap within the configured threshold.

`filter_review_overlays/` is the main visual audit: black lines are retained, green lines are filtered out. At `650 hPa`, the visible green high-latitude pieces touch only one longitude edge or the same edge twice, so they are correctly rejected under the single-wrap rule.

The filter is intentionally geometry-only: it does not decide whether a retained band is meteorologically important, only whether the contour behaves like one connected line around the globe.

## Shape Overlays

`shape_overlays/` redraws only the kept single-wrap contour segments as line charts, split into northern and southern groups by median latitude.

Plot families:
- `actual_latitude/`: x is longitude and y is real map latitude, so each line remains where it was on the map.
- `median_centered/`: x is longitude and y is latitude offset from that contour's own median, so curvature can be compared directly.
- `stacked_rows/`: each kept contour gets one row, and the line wiggle is a scaled latitude deviation around that row; this is the clearest plot for judging individual contour shapes without the lines covering each other.

For `650 hPa`, the stacked-row plots contain `11` northern contours and `14` southern contours. The southern rows are smoother and more parallel overall, while the northern rows show stronger localized bends and a few loop-like bulges around the Atlantic/European longitudes.

## Extrusion Metrics

`extrusion_analysis/` scores the kept single-wrap contours by shape:
- `deformation_score`: large if the line has a large latitude range, lots of wiggle, and local curvature.
- `cold_extrusion_score`: heuristic for polar-like contours bulging equatorward. Low score contours get more weight.
- `hot_extrusion_score`: heuristic for equator-like contours bulging poleward. High score contours get more weight.

These are contour-geometry heuristics, not parcel trajectories or front classifications.

Strong northern polar-like/equatorward candidates:
- `300 hPa`, score `15`, median latitude `74.2°N`; strongest equatorward dip near `123°W`, down to `51.8°N`.
- `250 hPa`, score `40`, median latitude `59.7°N`; strongest equatorward dip near `125.5°W`, down to `35.2°N`.
- `300 hPa`, score `20`, median latitude `69.1°N`; strongest equatorward dip near `122°W`, down to `49.2°N`.

Strong southern polar-like/equatorward candidates:
- `550 hPa`, score `15`, median latitude `71.5°S`; strongest equatorward bulge near `116.5°W`, up to `53.8°S`.
- `650 hPa`, score `15`, median latitude `72.2°S`; strongest equatorward bulge near `118°W`, up to `52.2°S`.
- `250 hPa`, score `25`, median latitude `60.4°S`; strongest equatorward bulge near `142°W`, up to `41.6°S`.

Strong northern equator-like/poleward candidates:
- `700 hPa`, score `85`, median latitude `11.5°N`; strongest poleward bulge near `98.25°W`, up to `30.3°N`.
- `700 hPa`, score `80`, median latitude `20.3°N`; strongest poleward bulge near `98.5°W`, up to `34.0°N`.
- `350 hPa`, score `55`, median latitude `41.0°N`; strongest poleward bulge near `2.25°W`, up to `63.7°N`.

Strong southern equator-like/poleward candidates:
- `250 hPa`, score `90`, median latitude `12.1°S`; strongest poleward bulge near `110°E`, down to `26.0°S`.
- `700 hPa`, score `85`, median latitude `14.8°S`; strongest poleward bulge near `40.75°E`, down to `27.7°S`.
- `875 hPa`, score `80`, median latitude `16.5°S`; strongest poleward bulge near `39.5°E`, down to `29.4°S`.

The cleanest cold-like examples are the low-score equatorward dips around `116-123°W` in both hemispheres. The cleanest hot-like examples are the high-score poleward bulges at `700 hPa` in the Northern Hemisphere and `250/700/875 hPa` in the Southern Hemisphere. Some top-ranked high-score southern contours cross or approach the equator, so treat those as shape/identity candidates first and require separate meteorological checks before calling them actual hot-air intrusions.
