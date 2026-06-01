# Fixed Midlatitude Thermal-Identity Cut

## 2026-05-31 Initial Run

Purpose:
- Make a small, easy-to-audit four-level plot set for a real global ERA5 frame.
- Show a meteorologically interpretable polar-like / equator-like temperature divide without treating any local gradient maximum as the answer.
- Keep the same definition at every pressure level so horizontal displacement with height can be inspected as a tilt signal.

Requirements extracted from `findings.md`:
- Use the standard quick-iteration levels: `250`, `500`, `850`, and `1000 hPa`.
- Keep the plots simple enough to understand and audit directly.
- Give the cut a meteorological reason for being where it is, rather than choosing an arbitrary algorithmic line.
- Target genuinely polar-like versus equator-like air in the midlatitudes, including normal-for-location air rather than anomaly-only air.
- Make vertical displacement, tilt, loops, and branches visible without forcing the atmosphere into a connected line.
- Represent diffuse versus compressed transitions with a ribbon rather than implying a perfectly sharp wall everywhere.

Method:
- Raw temperature: `data/era5_temperature_2021-11_08-12.nc`
- Reference climatology: `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Timestamp: `2021-11-08T12:00 UTC`
- Levels: `250`, `500`, `850`, and `1000 hPa`
- Domain: global ERA5 grid
- Identity field: canonical same-longitude, same-pressure, same-hemisphere Thermal Displacement score, lightly smoothed after matching with Gaussian sigma `1` native grid cell
- Fixed cut: score `50`
- Transition ribbon: scores `45-55`
- Rendered score and cut band: `20-70°` absolute latitude in both hemispheres, across all longitudes
- Interpretation: blue is polar-like, red is equator-like, the yellow line is the score-`50` midpoint, and the thin black lines bound the transition ribbon.

Outputs:
- Plots: `tmp/thermal-displacement-fixed-midlatitude-cut-2021-11-08t12/plots/`
- Reproducer: `tmp/thermal-displacement-fixed-midlatitude-cut-2021-11-08t12/make_plots.py`
- Summary: `tmp/thermal-displacement-fixed-midlatitude-cut-2021-11-08t12/summary.json`

Notes:
- The score-`50` contour is a fixed thermal-identity midpoint, not a detected front and not a parcel trajectory.
- The `20-70°` rendered band is an explicit midlatitude focus, not a change to the underlying score calculation.
- Ribbon width can be inspected as a simple diffuse-versus-compressed transition cue.
- The all-level overlay retains loops and branches instead of forcing one connected path.

Initial visual read:
- The `1000` and `850 hPa` midpoint traces are relatively broad, zonal belts with regional bends.
- The `500 hPa` trace deforms more strongly, and the `250 hPa` trace has much larger excursions, branches, and loops.
- The shared-definition overlay makes vertical displacement easy to see, but it does not support claiming one clean material wall through all four levels.

## 2026-05-31 Iteration: Ordered Transition Ribbon

Reason for iteration:
- The fixed score-`50` line is easy to audit but not meteorologically justified as the boundary. It means a temperature matching roughly `44.5°` climatological absolute latitude, not necessarily warm and cold air meeting.
- Replace the unsupported midpoint with an explicit three-state regime test: warm core, cold core, and locally validated transition air.

New method:
- Keep canonical same-longitude, same-pressure, same-hemisphere Thermal Displacement as the identity field.
- Convert its score back to absolute matched climatological latitude so the rule can be stated in geographic source-region terms.
- Primary warm core: temperature resembling climatological air at or equatorward of `35°` absolute latitude.
- Primary cold core: temperature resembling climatological air at or poleward of `55°` absolute latitude.
- Validated transition air: an intermediate-identity cell with warm-core evidence within `18°` equatorward and cold-core evidence within `18°` poleward at the same longitude.
- Render intermediate cells without the required side evidence as neutral gray rather than silently treating them as boundary air.
- Add a stricter `30°/60°` core sensitivity variant to check whether the result depends too strongly on the selected source-region anchors.
- Add a narrower `12°` side-search sensitivity variant to check whether the ribbon relies too strongly on the primary `18°` neighborhood scale.

Outputs:
- Plots and overlay: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/`
- Reproducer: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/make_plots.py`
- Summary: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/summary.json`

Initial iteration read:
- This is a more defensible baseline than score `50`: yellow now requires ordered evidence from distinct warm and cold source-region-like cores.
- The primary `18°` search gives a broad transition-air ribbon; the `12°` sensitivity run preserves recognizable geometry but exposes gaps and is the better high-confidence view.
- See `findings/thermal-identity-ordered-transition-ribbon-2021-11-08t12.md` for the coverage table and interpretation limits.
