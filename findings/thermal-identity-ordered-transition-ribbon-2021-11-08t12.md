# Ordered Thermal-Identity Transition Ribbon

## 2026-05-31 Initial Run

Purpose:
- Replace the unsupported fixed Thermal Displacement score-`50` boundary with a simple three-state classification.
- Identify warm-core air, cold-core air, and transition air only where the expected warm-equatorward / cold-poleward side ordering is locally present.

Method:
- Inputs: `data/era5_temperature_2021-11_08-12.nc` and `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Timestamp: `2021-11-08T12:00 UTC`
- Levels: `250`, `500`, `850`, and `1000 hPa`
- Domain: global longitudes and `20-70°` absolute latitude in both hemispheres
- Identity field: canonical same-longitude, same-pressure, same-hemisphere Thermal Displacement, converted back to absolute matched climatological latitude

Primary rule:
- Warm core: temperature resembling climatological air at or equatorward of `35°` absolute latitude
- Cold core: temperature resembling climatological air at or poleward of `55°` absolute latitude
- Validated transition: intermediate-identity air with warm-core evidence within `18°` equatorward and cold-core evidence within `18°` poleward at the same longitude
- Neutral gray: intermediate identity that does not pass the local side-order test

Sensitivity variants:
- Conservative source cores: `30°/60°` instead of `35°/55°`
- Narrower side search: `12°` instead of `18°`

Outputs:
- Plots and overlays: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/`
- Reproducer: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/make_plots.py`
- Summary: `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/summary.json`

Interpretation limits:
- The warm and cold core bands are conservative source-region anchors, not formal air-mass taxonomy.
- Yellow means locally ordered thermal-regime transition air. It does not by itself prove active frontogenesis or parcel origin.
- The side-search distance is an explicit scale choice, so the narrower sensitivity variant should be checked before promoting the rule.

Initial visual read:
- The primary `35°/55°`, `18°`-search variant produces broad, mostly globe-spanning ribbons while retaining gray intermediate regions that lack the required local side evidence.
- The `30°/60°` conservative-core variant preserves the broad geography but reduces coverage, especially in the Northern Hemisphere at `500 hPa`.
- The `12°` narrower-search variant preserves recognizable ribbons while exposing more honest gaps. It is the better high-confidence view; the `18°` version is the broader transition-air interpretation.
- Level-to-level displacement remains visible, especially at `250 hPa`, without forcing one shared centerline through the four levels.

Coverage summary:

| Variant | Level | NH longitude coverage | SH longitude coverage | NH median width | SH median width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary `35°/55°`, search `18°` | `250 hPa` | `100%` | `100%` | `10.25°` | `7.50°` |
| Primary `35°/55°`, search `18°` | `500 hPa` | `98%` | `100%` | `12.00°` | `12.25°` |
| Primary `35°/55°`, search `18°` | `850 hPa` | `93%` | `98%` | `13.50°` | `12.50°` |
| Primary `35°/55°`, search `18°` | `1000 hPa` | `94%` | `100%` | `14.00°` | `14.25°` |
| Narrow search `12°` | `250 hPa` | `94%` | `93%` | `8.00°` | `5.25°` |
| Narrow search `12°` | `500 hPa` | `69%` | `74%` | `8.00°` | `7.25°` |
| Narrow search `12°` | `850 hPa` | `76%` | `87%` | `7.75°` | `8.25°` |
| Narrow search `12°` | `1000 hPa` | `69%` | `81%` | `7.00°` | `5.75°` |

## 2026-05-31 Requested Iteration

User feedback:
- The equatorward-warm / poleward-cold assumption is too restrictive. A warm plume can curl poleward while cold air wraps equatorward.

Iteration:
- Supersede the same-longitude side-order test with an orientation-independent nearby opposing-core test.
- Keep this ordered experiment as the simpler baseline comparison.
- New outputs: `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/`
- New findings: `findings/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12.md`
