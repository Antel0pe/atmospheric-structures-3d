---
name: structure-of-data
description: Meteorological structural diagnosis for understanding how a field is organized in value, height, latitude/longitude, and imbalance before any 3D extraction.
---

# structure_of_data

Use this skill before building a new 3D extraction rule.

This skill exists to answer:

- What is the underlying structure of this field?
- How vertically stratified is it?
- How concentrated is it by latitude, longitude, or region?
- How uneven is it across pressure levels?
- Is the value distribution tight, broad, or dominated by tails and extremes?
- What would a world-class meteorologist say about the field before any visualization work begins?

This is not mainly a threshold triage skill. It is a structural diagnosis skill.

## Principles

- Prioritize meteorological truth over extraction convenience.
- Describe the field in physically meaningful language.
- Make vertical and horizontal imbalances explicit.
- Surface whether the field is dominated by background stratification, regional hotspots, or broad diffuse structure.
- Always produce a readable in-chat report; logs are supporting material.

## Chat Summary Requirement

After running this skill, always give the user the chat-style report directly.

Assume the user may never open the generated files. The logs exist for follow-up inspection, not as the primary deliverable.

The report should explain:

- the main structural takeaways in plain language
- vertical structure and whether the field is strongly height-stratified
- horizontal structure and whether the field is equatorward, hemispherically asymmetric, or regionally localized
- value distribution and whether the field is tight, broad, or heavy-tailed
- what these imbalances imply for any future thresholding, normalization, or pressure-window choices

## Default Data Conventions

- `q` / `specific_humidity` defaults to `data/era5_specific-humidity_2021-11_08-12.nc`
- `rh` / `relative_humidity` defaults to `data/era5_relative-humidity_2021-11_08-12.nc`
- `t` / `temperature` defaults to `data/era5_temperature_2021-11_08-12.nc`
- `theta` derives dry potential temperature from the temperature dataset

## Commands

Analyze the structure of specific humidity:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field q \
  --timestamp 2021-11-08T12:00
```

Analyze latitude-mean dry-theta anomalies:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field theta \
  --anomaly lat_mean \
  --timestamp 2021-11-08T12:00
```

Analyze climatological dry-theta anomalies in the intended lower/mid-tropospheric window:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field theta \
  --anomaly climatology \
  --pressure-levels 1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,500,450,400,350,300,250 \
  --timestamp 2021-11-08T12:00
```

## What It Analyzes

The skill is expected to comment on at least these structural questions:

- vertical stratification:
  how much of the field is explained by height-dependent background structure versus horizontal variability at a given level
- pressure-level concentration:
  whether the field is near-surface weighted, upper-level weighted, or relatively depth-distributed
- latitudinal concentration:
  whether the field is disproportionately tropical, subtropical, polar, or hemispherically asymmetric
- regional localization:
  whether a small fraction of columns or hotspots dominate the field
- value distribution:
  overall range, middle spread, tail heaviness, and whether outliers matter
- cross-level comparability:
  whether a single raw threshold means the same thing at different pressure levels

## Outputs

Each run writes a compact bundle under `skills/structure-of-data/logs/runs/<run-id>/`:

- `summary.md`: full structural report
- `summary.json`: machine-readable summary
- `maps.png`: representative raw and high-signal maps
- `profiles.png`: global and sample-column profiles

Use `skills/structure-of-data/logs/TOC.md` first when you want prior runs.

These artifacts are for the agent and for deeper inspection. The user-facing output is the chat-style report.

## Transition Rule

Use this skill before `structure_probe`.

Only move on to `structure_probe` once you understand:

- whether the field is dominated by vertical background structure
- whether the field needs a pressure window
- whether it needs per-level normalization
- whether sign matters
- whether the field is physically broad or dominated by narrow hotspots
