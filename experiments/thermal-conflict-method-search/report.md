# Thermal Conflict Method Search

Valid time: `2021-11-08T12:00`.
Pressure levels: `1000-250 hPa`.

## Ideas Considered And Pruned
- signed displacement zero-crossing: Already tested in the previous pressure-level loop; it is too close to a normal-for-latitude crossing.
- side-ordered tropical/polar contact: Already the strongest simple prior candidate, so this pass avoids repeating it as a standalone method.
- local tropical/polar co-presence: Already tested and found useful as corridor width/confidence rather than line placement.
- thermal-character frontogenesis/confluence: Already tested as process evidence; this pass uses different wind-process ideas instead.
- middle-80 and zonal-mean climatology displacement: Prior notes show they remove longitude-specific context and introduce hard-to-defend pressure jumps.
- dry-potential-temperature displacement: At fixed pressure levels it nearly duplicates raw-temperature displacement for this lookup.
- standalone raw temperature gradient: Meteorologically real, but prior diagnostics show it is too generic and terrain/polar biased unless gated.
- one global rare histogram bucket: Useful as a contour proposal, but it does not directly measure spatial hot/cold opposition.

## Method Metrics

- `thermal_wind_baroclinic` (B): median adjacent shift `1.1°`, p90 adjacent shift `1.8°`, 1000-250 shift `10.3°`, waviness `0.00°`.
- `vertical_identity_shear` (B): median adjacent shift `1.6°`, p90 adjacent shift `2.5°`, 1000-250 shift `12.3°`, waviness `0.00°`.
- `isotherm_normal_displacement_jump` (C): median adjacent shift `0.5°`, p90 adjacent shift `0.9°`, 1000-250 shift `11.0°`, waviness `0.00°`.
- `thetae_displacement_contact` (C): median adjacent shift `0.5°`, p90 adjacent shift `0.9°`, 1000-250 shift `10.5°`, waviness `0.00°`.
- `thermal_advection_dipole` (C): median adjacent shift `0.9°`, p90 adjacent shift `1.6°`, 1000-250 shift `11.1°`, waviness `0.00°`.

## Simple Implementations

### Vertical thermal-identity shear

A sloping frontal/baroclinic zone should show large thermal-identity change between neighboring pressure levels as well as horizontal identity contrast at the level being plotted.

1. Match raw temperature to same-longitude climatology and compute thermal-displacement score at every 250-1000 hPa level.
2. For each level, compare that score with the nearest pressure level above and below.
3. Score cells where vertical thermal-identity shear and horizontal thermal-identity gradient are both high.
4. Pick the strongest midlatitude ridge by longitude in each hemisphere.

### Warm/cold advection dipole

Hot/cold fighting should be active where winds are advecting tropical-like identity into nearby polar-like identity, or vice versa.

1. Compute thermal-displacement score and its horizontal gradient.
2. Read ERA5 wind at the same pressure level and time.
3. Compute thermal-identity advection, then smooth nearby warm-advection and cold-advection lobes.
4. Score cells where opposite-signed advection lobes touch near a thermal-identity gradient.

### Theta-e gradient with displacement contrast

Air-mass fights often sharpen equivalent potential temperature, so a theta-e gradient becomes more relevant when it also separates tropical-like and polar-like thermal displacement.

1. Compute equivalent potential temperature from temperature and specific humidity.
2. Smooth theta-e about 500 km and compute its horizontal gradient.
3. Sample thermal-displacement score across that theta-e gradient direction.
4. Score cells where theta-e changes sharply and the warm/moist side is more tropical-like than the cold/dry side.

### Thermal-wind baroclinicity

A real baroclinic boundary should align with vertical wind shear because horizontal temperature contrasts imply thermal-wind shear.

1. Read wind at every 250-1000 hPa level.
2. Compute vertical wind-shear magnitude against neighboring pressure levels.
3. Multiply shear by the local thermal-displacement horizontal gradient.
4. Pick the strongest shear-supported thermal-identity ridge by longitude.

### Raw-isotherm-normal displacement jump

Use real smoothed-temperature isotherms for geometry, then ask whether crossing those isotherms also crosses a polar-like to tropical-like displacement jump.

1. Smooth raw temperature about 500 km and compute its gradient direction.
2. Sample thermal-displacement score on the warm and cold sides normal to the smoothed isotherms.
3. Score cells where real isotherms are packed and thermal-displacement score jumps across the isotherm normal.
4. Pick the strongest real-isotherm/displacement-jump ridge by longitude.

