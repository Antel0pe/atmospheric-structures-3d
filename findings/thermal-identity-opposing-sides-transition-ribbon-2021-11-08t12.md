# Orientation-Independent Thermal-Identity Transition Ribbon

## 2026-05-31 Initial Run

Purpose:
- Iterate on `tmp/thermal-identity-ordered-transition-ribbon-2021-11-08t12/`.
- Keep the simple warm-core / intermediate / cold-core identity field while allowing transition ribbons to curve, curl, and close around embedded intrusions.
- Remove the assumption that warm air must be equatorward of cold air at each longitude.

Method:
- Inputs: `data/era5_temperature_2021-11_08-12.nc` and `data/era5_temperature-climatology_1990-2020_11-08_12.nc`
- Timestamp: `2021-11-08T12:00 UTC`
- Levels: `250`, `500`, `850`, and `1000 hPa`
- Domain: global longitudes and `20-70°` absolute latitude in both hemispheres
- Identity field: canonical same-longitude, same-pressure, same-hemisphere Thermal Displacement, converted back to absolute matched climatological latitude

Primary rule:
- Warm core: temperature resembling climatological air at or equatorward of `35°` absolute latitude
- Cold core: temperature resembling climatological air at or poleward of `55°` absolute latitude
- Candidate transition: intermediate thermal identity between the two core thresholds
- Validated transition: the nearest warm and cold cores are each within `12°` great-circle angular distance, and their bearings from the candidate cell differ by at least `120°`
- Gray: unresolved intermediate identity. It does not pass the local opposing-sides test, but it is not proven to be non-transition air.

Sensitivity variants:
- Broader radius: `18°` instead of `12°`
- Stricter opposition: `150°` instead of `120°`

Outputs:
- Plots and overlays: `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/`
- Reproducer: `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/make_plots.py`
- Summary: `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/summary.json`

User feedback that motivated this iteration:
- Equatorward warm air is not always the correct geometry. A warm plume can curl poleward while cold air wraps equatorward, especially in deformed or occluded systems.

Initial visual read:
- The primary `12° / 120°` rule preserves recognizable midlatitude ribbons and adds folded or closed geometry that the same-longitude rule could miss.
- The `18°` radius expands the mask substantially and is better read as broad transition context than as the default.
- The stricter `150°` opposition angle exposes a narrower high-confidence subset but becomes visibly fragmented.
- The primary ribbon is not forced to exist at every longitude. Honest gaps remain where the nearby opposing-core evidence is absent.
- Level-to-level displacement is still visible in the overlay, so the four slices continue to demonstrate tilt.

Coverage summary:

| Variant | Level | NH longitude coverage | SH longitude coverage | NH median width | SH median width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary `12° / 120°` | `250 hPa` | `100%` | `100%` | `9.50°` | `4.50°` |
| Primary `12° / 120°` | `500 hPa` | `94%` | `89%` | `8.75°` | `8.00°` |
| Primary `12° / 120°` | `850 hPa` | `93%` | `100%` | `9.25°` | `7.50°` |
| Primary `12° / 120°` | `1000 hPa` | `90%` | `98%` | `8.50°` | `6.75°` |
| Stricter `12° / 150°` | `250 hPa` | `98%` | `96%` | `5.50°` | `2.62°` |
| Stricter `12° / 150°` | `500 hPa` | `82%` | `71%` | `6.00°` | `6.50°` |
| Stricter `12° / 150°` | `850 hPa` | `83%` | `91%` | `5.12°` | `4.75°` |
| Stricter `12° / 150°` | `1000 hPa` | `73%` | `75%` | `5.25°` | `5.00°` |

Interpretation limits:
- The `35°/55°` matched-latitude bands remain explicit source-region-like anchors, not universal air-mass constants.
- The nearest-core test is a local geometry test, not a parcel-origin calculation.
- Yellow does not prove frontogenesis, frontal activity, or a formal synoptic front.
- Nearest-core selection intentionally keeps the explanation simple. It may miss complicated junctions where a farther core is more representative than the nearest one.

## 2026-05-31 Russia-Sector Gray Diagnostic

User feedback:
- The amount of gray is generally acceptable in the primary `12° / 120°` maps, but a persistent open gray area over Russia needs explanation.
- Question: should the search radius or opposition angle vary by pressure level?

Diagnostic:
- Added `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/diagnostic-russia-gray-reasons/`.
- Uses an explicit `30-180°E, 45-70°N` Eurasia/Russia-sector box, not a country-border mask.
- Splits unresolved gray into exclusive reasons: warm core too far, cold core too far, both cores too far, or nearby cores with insufficient opposing angle.
- Adds a radius and angle sensitivity plot by pressure level.

Main result:
- At `500`, `850`, and `1000 hPa`, the Russia-sector gray area is mainly a broad intermediate-to-cold shoulder with no warm-core air within `12°`.
- At `250 hPa`, insufficient bearing opposition is slightly more common than excessive warm-core distance, consistent with more folded upper-level geometry.

| Level | Gray share of intermediate air | Warm core too far | Cold core too far | Both too far | Nearby but angle `<120°` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `250 hPa` | `62%` | `46%` | `0%` | `0%` | `54%` |
| `500 hPa` | `78%` | `78%` | `10%` | `3%` | `10%` |
| `850 hPa` | `81%` | `76%` | `1%` | `0%` | `23%` |
| `1000 hPa` | `87%` | `82%` | `3%` | `3%` | `12%` |

Interpretation:
- Keep a fixed `12° / 120°` baseline for clean vertical comparison.
- There is no simple meteorological rule saying the radius or angle should monotonically increase or decrease with pressure.
- Radius changes can be shown as an explicit broad-context sensitivity. Angle changes are better treated as confidence sensitivities.
- Do not loosen thresholds per level merely to erase gray. Gray is carrying information about broad or unresolved intermediate regimes.

### Global `250 hPa` Gray Breakdown

- For the primary `12° / 120°` global `250 hPa` map, `45%` of intermediate cells remain unresolved gray and `55%` are promoted to yellow.
- Exclusive failure causes among the unresolved gray cells:
  - `66%`: nearest warm and cold cores are both within `12°`, but their bearing separation is below `120°`
  - `23%`: nearest warm core is farther than `12°`
  - `10%`: nearest cold core is farther than `12°`
  - `0%`: both cores are farther than `12°`
- This differs from the Russia-sector lower-level pattern. Globally at `250 hPa`, the opposition-angle assumption is the main confidence filter.

## 2026-05-31 No-Bearing-Angle Variant

User request:
- Try a variant without the bearing-angle requirement.

Method:
- Keep the `35°/55°` matched-latitude core anchors and `12°` great-circle search radius.
- Promote an intermediate cell to yellow when its nearest warm and cold cores are both within `12°`.
- Do not require the two bearings to differ by at least `120°`.

Output:
- `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/sensitivity-no-bearing-angle/`

Comparison:

| Level | Primary `12° / 120°` yellow share | No-angle `12°` yellow share |
| --- | ---: | ---: |
| `250 hPa` | `15.1%` | `23.4%` |
| `500 hPa` | `14.6%` | `17.9%` |
| `850 hPa` | `15.3%` | `21.8%` |
| `1000 hPa` | `13.8%` | `19.0%` |

Initial read:
- Removing the angle gate has the largest effect at `250 hPa`, as expected from the global gray breakdown.
- The no-angle `250 hPa` map fills many folded and side-adjacent upper-level regions. It is a useful broad nearby-core view, but yellow no longer specifically means that intermediate air lies between opposing sides.
- The large Russia gray areas at `500-1000 hPa` remain substantially visible because many fail the `12°` warm-core distance requirement, not the angle requirement.

## 2026-05-31 Accepted No-Angle Gray Analysis

User feedback:
- The no-bearing-angle `1000 hPa` map reads well.
- Analyze the remaining gray area by pressure level.

Diagnostic:
- Added `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/diagnostic-no-bearing-angle-gray-reasons/`.
- Global domain within the rendered `20-70°` absolute-latitude bands.
- With the angle gate removed, gray has only three possible causes: nearest warm core farther than `12°`, nearest cold core farther than `12°`, or both farther than `12°`.

| Level | Intermediate share of rendered band | Yellow share | Remaining gray share | Gray share of intermediate air |
| --- | ---: | ---: | ---: | ---: |
| `250 hPa` | `27.6%` | `23.4%` | `4.2%` | `15.3%` |
| `500 hPa` | `38.9%` | `17.9%` | `20.9%` | `53.9%` |
| `850 hPa` | `35.8%` | `21.8%` | `14.0%` | `39.0%` |
| `1000 hPa` | `41.9%` | `19.0%` | `22.9%` | `54.6%` |

Exclusive causes among remaining gray cells:

| Level | Warm core too far | Cold core too far | Both too far |
| --- | ---: | ---: | ---: |
| `250 hPa` | `69.8%` | `30.1%` | `0.0%` |
| `500 hPa` | `47.4%` | `50.9%` | `1.5%` |
| `850 hPa` | `47.0%` | `52.8%` | `0.1%` |
| `1000 hPa` | `52.1%` | `46.8%` | `1.0%` |

Interpretation:
- `250 hPa` is largely resolved by removing the angle gate. Its remaining gray is mostly on the side with no warm core within `12°`.
- At `500`, `850`, and `1000 hPa`, remaining gray is not dominated globally by one side. Warm-too-far and cold-too-far regions are both substantial and form broad shoulders around the yellow nearby-core corridor.
- Cells with both cores farther than `12°` are rare at every level. Gray mostly means one-sided unresolved intermediate air, not total isolation from both thermal regimes.

## 2026-05-31 No-Angle Radius Sweep

User request:
- Plot how much gray remains as the nearby warm/cold-core distance requirement expands beyond `12°`.
- Look for a sharp turn that might suggest a useful distance scale.

Diagnostic:
- Added `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/diagnostic-no-bearing-angle-radius-sweep/`.
- Sweeps radius from `0°` through `30°` in `1°` steps.
- Plots unresolved gray as a share of the rendered `20-70°` latitude bands and as a share of intermediate-identity air.
- Marks geometric elbow candidates from curve shape. These are not meteorologically proven optimal radii.

Geometric elbow candidates:

| Level | Radius | Approximate distance |
| --- | ---: | ---: |
| `250 hPa` | `11°` | `1220 km` |
| `500 hPa` | `20°` | `2220 km` |
| `850 hPa` | `16°` | `1780 km` |
| `1000 hPa` | `19°` | `2110 km` |

Selected gray shares of intermediate air:

| Level | `12°` | `16°` | `18°` | `20°` |
| --- | ---: | ---: | ---: | ---: |
| `250 hPa` | `15.3%` | `6.1%` | `3.8%` | `2.3%` |
| `500 hPa` | `53.8%` | `28.5%` | `19.1%` | `11.8%` |
| `850 hPa` | `39.0%` | `19.1%` | `13.1%` | `8.2%` |
| `1000 hPa` | `54.5%` | `26.5%` | `16.8%` | `10.2%` |

Interpretation:
- There is no single shared sharp turn across pressure levels.
- The current `12°` radius is close to the `250 hPa` curve bend but conservative for `500-1000 hPa`.
- A common radius near `16-18°` would substantially reduce lower-level gray while leaving some visible unresolved shoulder. It would also broaden the meaning of "nearby" from roughly `1330 km` to `1780-2000 km`.
- Do not treat curve geometry alone as meteorological proof. The next useful comparison is visual: generate common-radius no-angle maps around `16°` and `18°` and judge whether yellow still reads as meaningful transition air.

## 2026-05-31 No-Angle Radius `15°` And `20°` Maps

User request:
- Render no-bearing-angle comparison maps with `15°` and `20°` nearby-core radii.

Outputs:
- `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/sensitivity-no-bearing-angle-radius15/`
- `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/sensitivity-no-bearing-angle-radius20/`

Yellow rendered-band coverage:

| Level | `12°` | `15°` | `20°` |
| --- | ---: | ---: | ---: |
| `250 hPa` | `23.4%` | `25.4%` | `26.9%` |
| `500 hPa` | `17.9%` | `25.6%` | `34.3%` |
| `850 hPa` | `21.8%` | `27.6%` | `32.8%` |
| `1000 hPa` | `19.0%` | `28.2%` | `37.6%` |

Initial visual read:
- `15°` substantially reduces lower-level open gray areas while preserving visible unresolved shoulders.
- `20°` nearly fills the intermediate band at `1000 hPa`. Yellow starts to read more like a broad intermediate reservoir than a localized nearby-core transition corridor.
- `250 hPa` changes only modestly because it was already mostly resolved by the `12°` no-angle rule.
- Of these two new comparisons, `15°` is the more balanced candidate for a common-radius default.

## 2026-05-31 Preferred Full-Globe `20°` Pressure Stack

User decision:
- Prefer the no-bearing-angle `20°` nearby-core method despite the earlier initial read favoring `15°`.
- Render the full globe, not only the earlier `20-70°` diagnostic latitude bands.
- Apply the method to every available ERA5 pressure level from `1000` through `250 hPa` and create an interpolated GIF.

Output:
- `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/preferred-no-bearing-angle-radius20-full-globe-pressure-stack/`
- Source-level PNGs: `source-level-maps/`
- GIF: `full_globe_nearby_core_pressure_stack_1000_to_0250hpa.gif`
- Summary: `summary.json`

Method:
- Use canonical same-pressure, same-longitude, same-hemisphere Thermal Displacement matching.
- Red is warm-core-like air with equivalent absolute climatological latitude `<= 35°`.
- Blue is cold-core-like air with equivalent absolute climatological latitude `>= 55°`.
- Yellow is intermediate-identity air whose nearest same-hemisphere red and blue cores are each within `20°` great-circle distance.
- Gray is intermediate-identity air that does not meet that nearby-core test.
- No bearing-angle requirement is applied.

Pressure levels:
- `1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250 hPa`

Animation:
- `21` independently computed source-level maps.
- Each real pressure level is held for `0.75s`.
- Four `0.10s` visual crossfade frames connect each adjacent pair of source levels.
- Total: `101` GIF frames and `23.75s`.
- Crossfades are presentation interpolation only, not inferred atmospheric pressure levels.

## 2026-06-01 Revised Animation Direction And Cadence

User request:
- Regenerate the preferred full-globe GIF in the `250 -> 1000 hPa` direction.
- Hold each independently computed source-level map for `0.5s`.

Updated GIF:
- `tmp/thermal-identity-opposing-sides-transition-ribbon-2021-11-08t12/preferred-no-bearing-angle-radius20-full-globe-pressure-stack/full_globe_nearby_core_pressure_stack_0250_to_1000hpa.gif`
- Pressure order: `250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa`
- `21` real source-level frames held for `0.5s` each.
- Four `0.10s` visual crossfade frames still connect each adjacent pressure pair.
- Total: `101` GIF frames and `18.5s`.
- The prior `1000 -> 250 hPa` GIF is superseded.
