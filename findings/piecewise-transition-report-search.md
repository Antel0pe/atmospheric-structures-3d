# Piecewise Transition Report Search

Date: 2026-05-30

Purpose:
- Run three search-space experiments for report-level thinking about whether piecewise latitude-profile segments can help define transition air between warm/tropical-like and cold/polar-like regimes.
- This is a brainstorming pass; the outputs are meant to support discussion more than serve as final product assets.

Inputs:
- Raw temperature: `data/era5_temperature_2021-11_08-12.nc`
- Temperature climatology: `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Levels: `250`, `500`, `850`, `1000 hPa`
- Hemispheres: north and south
- Sampling: longitude every `4` degrees, latitude every `1` degree

Experiments:
- `01-raw-temperature-ramp`: fit broad piecewise segments to smoothed actual raw temperature profiles. This asks where the observed temperature profile changes regime fastest.
- `02-thermal-identity-ramp`: fit broad piecewise segments to canonical same-hemisphere Thermal Displacement score. This asks where the profile changes from polar-like to equator-like identity.
- `03-identity-validated-ribbon`: use raw-temperature ramp geometry, then score it by whether Thermal Displacement is more polar-like on the poleward side and more equator-like on the equatorward side.

Outputs:
- Script: `tmp/piecewise-transition-report-search/run_three_search_space_experiments.py`
- Summary: `tmp/piecewise-transition-report-search/summary.json`
- Report notes: `tmp/piecewise-transition-report-search/report-thoughts.md`
- Raw ramp plots: `tmp/piecewise-transition-report-search/01-raw-temperature-ramp/`
- Thermal identity ramp plots: `tmp/piecewise-transition-report-search/02-thermal-identity-ramp/`
- Identity-validated ribbon plots: `tmp/piecewise-transition-report-search/03-identity-validated-ribbon/`
- Overview: `tmp/piecewise-transition-report-search/overview/three_experiment_comparison.png`

Quick observations:
- The clean conceptual split is geometry vs identity. Raw temperature segments are good at finding broad ramp geometry. Thermal Displacement is better for asking whether the ramp separates polar-like and equator-like air.
- The strongest reportable candidate is not the steepest raw ramp alone. It is the raw ramp plus a side-order check: poleward side lower Thermal Displacement score, equatorward side higher score.
- The side-order version keeps transition air as a ribbon between two breakpoints instead of a single boundary line.
- For the sampled longitudes, the side-order check passes nearly everywhere, but confidence varies. That suggests this is useful as a confidence/width visualization, not a binary mask.
- `500 hPa` north is a clear example: raw ramp median center is near `45` degrees absolute latitude, Thermal Displacement side-order pass fraction is `1.00`, and median confidence is about `0.86`.

Report thought:
- A defensible story is: piecewise segments find where the meridional profile changes regime; Thermal Displacement tells whether the two sides are actually cold-side and warm-side. The transition air is the segment between those breakpoints.
