---
name: structure-of-data
description: Fast meteorological diagnostics for understanding how a field is distributed in value, height, and geography before any 3D extraction.
---

# structure_of_data

Use this skill before building or revising a 3D representation idea.

This skill exists to answer, quickly and with data:

- What does this field mostly consist of?
- Where is it concentrated vertically?
- Where is it concentrated geographically?
- Are the values broad, tight, heavy-tailed, or dominated by a narrow background?
- If this is an anomaly field, where are the meaningful departures and how rare are they?
- What would matter before trying to threshold or mesh this field in 3D?

This is a fast structural diagnosis skill, not a polished report writer.

## Operating Rules

- The chat summary is the deliverable.
- Do not treat saved logs as part of the workflow.
- Run quick code against the data and report only what the diagnostics support.
- Keep the summary plain-English and meteorologist-useful.
- Include plots when they answer a concrete question.
- Avoid filler. Every statistic or chart should earn its place.

## What Good Output Looks Like

A good run should tell the user things like:

- most of the specific-humidity signal is confined to the lower troposphere
- the field is broad by longitude but concentrated within specific latitude belts
- raw temperature is mostly background stratification, not a clean object field
- dry-theta climatology departures are strongest on these pressure levels
- only a small fraction of the sampled area exceeds 5% or 10% climatology departure
- a single raw threshold is or is not comparable across levels

## Default Data Conventions

- `q` / `specific_humidity` defaults to `data/era5_specific-humidity_2021-11_08-12.nc`
- `rh` / `relative_humidity` defaults to `data/era5_relative-humidity_2021-11_08-12.nc`
- `t` / `temperature` defaults to `data/era5_temperature_2021-11_08-12.nc`
- `theta` derives dry potential temperature from the temperature dataset

## Commands

Analyze specific humidity with the default fast plots:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field q \
  --timestamp 2021-11-08T12:00
```

Analyze dry potential temperature as a latitude-mean anomaly:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field theta \
  --anomaly lat_mean \
  --timestamp 2021-11-08T12:00
```

Analyze dry potential temperature against climatology across all available levels first:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field theta \
  --anomaly climatology \
  --timestamp 2021-11-08T12:00
```

If a later comparison needs a pressure window, choose that only after the all-level diagnostic shows why.

Get machine-readable output for a follow-on step:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field theta \
  --anomaly climatology \
  --timestamp 2021-11-08T12:00 \
  --json
```

If a later step explicitly needs files, save them deliberately instead of relying on repo logs:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure-of-data/scripts/run_structure_of_data.py \
  --field q \
  --timestamp 2021-11-08T12:00 \
  --artifact-dir /tmp/structure-of-data-q \
  --save-summary
```

## What It Should Check

Every run should cover the structural questions that matter for the chosen field.

Usually that means:

- value distribution: range, middle spread, tails, and mode
- vertical structure: where the signal sits by pressure level and whether a narrow layer dominates
- horizontal structure: tropical bias, latitude-band concentration, hotspot concentration, and hemispheric imbalance
- cross-level comparability: whether one raw threshold means the same thing at different levels
- anomaly severity when relevant: how much area exceeds simple relative-departure or sigma-style thresholds

For dry potential temperature climatology anomalies, explicitly surface:

- area above simple percent departures from climatology
- area above 1σ / 2σ / 3σ
- which pressure levels dominate those departures
- a quick-and-dirty per-level component count above a strong departure threshold so chat can say how many big 2D blobs would show up on each pressure level

## Plot Expectations

The default plots should stay lightweight and diagnostic.

Good plots include:

- per-level mean and spread versus pressure
- signal share by pressure level
- global value histogram
- latitude concentration curve
- horizontal concentration curve
- anomaly-threshold heatmap when climatology anomalies are active
- when useful, a simple pressure-vs-component-count view for quick per-level blob-count intuition

## Required Agent Behavior

After running the command:

- summarize the result directly in chat
- lead with the structural takeaways, not the command or the files
- mention the plots only if they add useful evidence
- do not tell the user to open a repo log or TOC
- do not invent meteorology; only say what the diagnostics support

## Transition Rule

Use this skill before `structure-probe`.

Only move on when you understand:

- whether the field is mostly background stratification or real localized structure
- whether the field needs a pressure window
- whether it needs per-level normalization
- whether sign matters
- whether the interesting part of the field is broad, rare, or hotspot-driven
