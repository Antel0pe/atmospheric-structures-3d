# Piecewise Transition Air Search

Date: 2026-05-29

Purpose:
- Search whether independent piecewise latitude-profile segments can suggest broad transition air between cold-side and warm-side regimes.
- This is a food-for-thought search-space pass, not a final boundary method.

Method:
- Input: `data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc`
- Variable: `t`
- Operation: mean raw temperature across `valid_time`.
- Levels: `250`, `500`, `850`, `1000 hPa`.
- Hemispheres: north and south.
- Sampling: longitude every `4` degrees, latitude every `1` degree.
- Each longitude profile is smoothed along the pole-to-equator latitude path, then independently fit with broad piecewise linear segments.
- Candidate readings:
  - primary ramp: steepest broad equatorward-warming segment per longitude
  - all steep segments: secondary plausible transition segments
  - entry breakpoint: poleward edge where the profile enters the strongest ramp

Outputs:
- Script: `tmp/piecewise-transition-air-search/run_piecewise_transition_air_search.py`
- Summary: `tmp/piecewise-transition-air-search/summary.json`
- Candidate maps: `tmp/piecewise-transition-air-search/transition-candidate-maps/`
- Slope-regime heatmaps: `tmp/piecewise-transition-air-search/slope-regime-heatmaps/`
- Overview: `tmp/piecewise-transition-air-search/overview/primary_ramp_centers_and_widths.png`

Quick observations:
- The cleanest interpretation is not a line; it is a broad meridional ramp/ribbon.
- Northern `250 hPa` looks like the most intuitive case: a broad ramp centered near `32.5` degrees absolute latitude with median width near `18` degrees.
- Northern `500/850/1000 hPa` shifts the primary ramp poleward and becomes more regionally jagged, suggesting the best transition-air product may need confidence and width rather than a single global boundary.
- Southern levels are less stable in this raw-climatology framing. Several levels jump between midlatitude and high-latitude ramps, which suggests the steepest raw-temperature ramp can latch onto polar/frontal-zone structure that may or may not be the user's intended hot/cold transition.
- Multiple steep segments are common, especially at `500 hPa` south and lower levels. That supports the user's note that transition structure is not always a single cold-transition-warm sequence.

Working interpretation:
- Piecewise segments are promising as a "where does the profile change regimes?" detector.
- On their own, they identify slope-regime transitions, not necessarily hot/cold air-mass conflict.
- A better next step may be to pair the piecewise ramp with an identity field: use the piecewise segment to propose the transition ribbon, then require polar-like air on one side and tropical-like air on the other side.
