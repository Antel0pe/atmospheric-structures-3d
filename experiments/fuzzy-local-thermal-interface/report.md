# Fuzzy Local Thermal-Identity Interface

Valid time: `2021-11-08T12:00`.

Purpose: test a simple replacement for rare-bucket conflict-zone selection.

## Question

Can a cold/warm conflict zone be identified without assuming:

- one global temperature or Thermal Displacement value
- the sharpest gradient is the correct boundary
- a north-to-south sequence of only cold, transition, warm

The motivating failure case is an embedded warm blob near southern Greenland at
`250 hPa`: a valid warm feature can sit inside colder air, so the method must
allow closed local interfaces and multiple cold/warm transitions along a
latitude or longitude transect.

## Method

For each pressure level:

1. Compute the existing same-longitude Thermal Displacement score.
2. Fit two fuzzy score anchors over the broad non-tropical/non-polar field:
   cold-like and warm-like thermal identity.
3. In a local 2D window, compute low and high Thermal Displacement percentiles.
4. Require local co-presence: the low side must be cold-like, the high side must
   be warm-like, and the local range must be large enough.
5. Define fuzzy local membership between the local low/high regimes.
6. Mark conflict corridors where membership is near the local `50/50`
   transition.
7. Use Thermal Displacement gradient only as a brightness/sharpness cue, not as
   the definition of the boundary.

## Outputs

- `output/fuzzy-local-interface-contact-sheet.png`
- `output/fuzzy-local-interface-greenland-crop-contact-sheet.png`
- `output/summary.json`
- `output-window24/fuzzy-local-interface-contact-sheet.png`
- `output-window24/fuzzy-local-interface-greenland-crop-contact-sheet.png`
- `output-window24/summary.json`

## First Read

The `16 degree` local-window run is the better first diagnostic. It usually
places corridors along broad midlatitude cold/warm transition zones, and at
`250 hPa` it draws a closed transition around the warm Greenland/North Atlantic
blob instead of forcing a single north-south boundary.

The `24 degree` local-window comparison is smoother globally but too broad in
the Greenland crop. It starts coloring too much surrounding gradual transition
air as conflict, so larger-scale versions should use a cheaper sector or
downsampled clustering approach rather than brute-force native-grid sliding
percentiles.

## Interpretation

This is more logically defensible than the rare-bucket method because it asks
whether cold-like and warm-like thermal regimes both exist nearby, then finds
their local fuzzy interface. It still inherits the caveat that Thermal
Displacement is a proxy for thermal identity, not an academic air-mass variable.

Best next change: separate the output into three visual products:

- broad fuzzy transition corridor
- high-confidence centerline or contour
- embedded warm/cold island boundaries

That would make slow boundaries and sharp closed features readable without
forcing one mask to do all jobs.

## Knobs

Most important knobs in this first method:

- `window-deg`: physical scale of the local neighborhood used to decide whether
  cold-like and warm-like regimes both exist nearby.
- `low-percentile` / `high-percentile`: how far apart the local cold-side and
  warm-side regime anchors are inside that window.
- `sharpness-floor`: how much broad slow transition survives before gradient
  sharpness brightens it.
- `transition-width`: how wide the fuzzy `50/50` transition band is.
- `min-local-range` / `full-local-range`: how much local Thermal Displacement
  spread is required before the method believes cold/warm co-presence exists.

## Tuning Experiments

Three simple sweeps were run after the first pass:

1. Window scale:
   - `output-window10/`: `10 degree` local window
   - `output/`: `16 degree` local window
   - `output-window24/`: `24 degree` local window
2. Local regime anchor spread:
   - `output-percentile10-90/`: local `10th/90th` percentiles
   - `output/`: local `15th/85th` percentiles
   - `output-percentile25-75/`: local `25th/75th` percentiles
3. Sharpness dependence:
   - `output-sharpness-floor010/`: mostly gradient-dependent
   - `output/`: balanced default
   - `output-sharpness-floor065/`: broad-boundary-friendly

## Tuning Read

Window scale is the strongest visual knob. `10 degrees` is narrow and selective;
it preserves embedded features but drops too much broad low/midlevel transition.
`24 degrees` is smoother globally but over-expands gradual transition air,
especially in the Greenland crop. `16 degrees` remains the best first default.

Percentile spread is the cleanest science knob. `10/90` sees more broadly
separated local regimes and fills wider corridors. `25/75` is stricter and less
overfilled, but risks missing slow cold/warm transitions because the local
anchors are not far enough apart. Keep `15/85` as the default compromise.

Sharpness dependence controls philosophy more than placement. A low floor
turns the method back toward a gradient detector and suppresses broad
boundaries. A high floor keeps slow transitions visible but can make broad
bands too fat. The default `0.35` is reasonable for now because the method
should not be "sharpest boundary wins."

Recommended next default:

```text
window-deg = 16
low/high percentiles = 15 / 85
sharpness-floor = 0.35
transition-width = 0.32
min/full local range = 12 / 40
```

The next useful tuning pass should focus on `transition-width` and
`min-local-range`, because those are likely to separate "broad but valid" from
"too much weak transition air" better than further changing the window size.
