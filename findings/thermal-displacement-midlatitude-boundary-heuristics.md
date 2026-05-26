# Thermal Displacement Midlatitude Boundary Heuristics

Date: 2026-05-26

Quick experiment to test whether matched-latitude Thermal Displacement can help trace a midlatitude hot-versus-cold air zone without a rigid latitude mask.

Levels: `250`, `500`, `850`, `1000 hPa`.
Domain: global latitude/longitude.
Base field: canonical same-longitude, same-hemisphere Thermal Displacement from raw temperature and matched climatology.

## User Observation

- The latitude-displacement field should not be filtered by an exact rigid area because weather is dynamic.
- The strongest displacement change is not automatically the desired air-mass boundary; at `1000 hPa`, the largest change can appear near the equator.
- The target is an intuitive way to trace the midlatitude hot-versus-cold air zone, while accepting that a location can be normal for its latitude and still sit on a boundary.

## Variants

- `soft_midlatitude_displacement_gradient`: smoothed latitude-displacement gradient multiplied by a broad midlatitude weight.
- `red_blue_copresence`: nearby equatorward-like and poleward-like displacement are both present.
- `equator_red_pole_blue_opposition`: the equatorward side is red while the poleward side is blue about `5 deg` away.
- `score_transition_gradient`: strong gradient near Thermal Displacement score `50`, softly weighted toward midlatitudes.
- `climatology_baroclinic_contact`: red/blue co-presence multiplied by climatological meridional temperature gradient.

## Outputs

- Contact sheets: `tmp/thermal-displacement-latitude-agreement/output/same-hemisphere-matched-latitude-displacement-from-actual-latitude/midlatitude-boundary-heuristics/contact-sheets/`
- Summary: `tmp/thermal-displacement-latitude-agreement/output/same-hemisphere-matched-latitude-displacement-from-actual-latitude/midlatitude-boundary-heuristics/summary.json`
- Detailed note: `tmp/thermal-displacement-latitude-agreement/output/same-hemisphere-matched-latitude-displacement-from-actual-latitude/midlatitude-boundary-heuristics/findings/midlatitude-boundary-heuristics.md`

## First Read

- `score_transition_gradient` is the cleanest latitude-wise: about `98.5%` of top cells land in `25-65 deg`, with essentially no equatorward top cells. It may be too tied to the built-in score transition, so it can trace a neat belt without proving hot/cold conflict.
- `red_blue_copresence` is the most intuitive direct use of the displacement field. It highlights places where equatorward-like and poleward-like displacement are nearby, but reads as broad zones rather than a single line.
- `climatology_baroclinic_contact` is similar to co-presence but more selective and spotty. It does not obviously solve the boundary question by itself.
- `equator_red_pole_blue_opposition` is the strictest hot/cold ordering test. It is more physically pointed but patchier and lets more poleward top cells through.
- `soft_midlatitude_displacement_gradient` avoids the `1000 hPa` equator trap, but mostly answers where displacement changes quickly, not necessarily where two meaningful air identities meet.
