---
name: structure-probe
description: Fast 3D extraction diagnostic for understanding what kind of object a field and extraction rule will actually produce.
---

# structure_probe

Use this skill after `structure_of_data` when the field looks promising enough to test a cheap 3D extraction.

This skill exists to answer:

- Does this become blobs, sheets, speckle, or one giant mass?
- Is the failure caused by the field, the threshold, the connectivity rule, or postprocess?
- Is this variant worth promoting to a real viewer layer?

This is not a final renderer. It is a fast extraction reality check.

## Principles

- Run fast, defaulting to coarse resolution.
- Isolate extraction problems from rendering problems.
- Save concise run summaries so future users and LLMs can avoid repeating bad probes.

## Commands

Probe a simple threshold extraction on specific humidity:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure_probe/scripts/run_structure_probe.py \
  --field q \
  --method threshold \
  --threshold-percent 10
```

Probe a seed-grow variant on a structure_of_data configuration:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure_probe/scripts/run_structure_probe.py \
  --structure-of-data skills/structure-of-data/logs/runs/<run-id>/summary.json \
  --method seed_grow \
  --threshold-percent 10 \
  --grow-rule same-sign-relaxed-half \
  --bridge-levels 1
```

Probe a boundary-like gradient extraction:

```bash
conda run -n atmospheric-structures-3d \
  python skills/structure_probe/scripts/run_structure_probe.py \
  --field theta \
  --anomaly lat_mean \
  --method gradient \
  --threshold-percent 10
```

## Outputs

Each run writes a compact bundle under `skills/structure_probe/logs/runs/<run-id>/`:

- `summary.md`: human-readable report
- `summary.json`: machine-readable summary
- `overview.png`: cheap top-down, cross-section, and component-size panels

Use `skills/structure_probe/logs/TOC.md` first. It is the table of contents for prior probes.

## Diagnosis Goal

Every report must say whether the current failure is mainly caused by:

- the field
- the threshold
- the connectivity rule
- the postprocess

Only promote variants that survive that diagnosis.
