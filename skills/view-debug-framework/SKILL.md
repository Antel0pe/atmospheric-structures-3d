---
name: view-debug-framework
description: Restore exact viewer debug cases, pick targets by normalized screen point, capture comparison screenshots, and run the moisture analyzer against generated assets and raw NetCDF data.
---

# View Debug Framework

Use this skill when you need to debug a specific feature inside a saved view rather than just capture a screenshot.

The workflow:

1. Restore a saved view or an existing debug-case JSON.
2. Freeze the current layer/render state into a reusable debug case.
3. Hit-test one or more normalized screen targets.
4. Capture the baseline scene plus analyzer-defined comparison states.
5. Run the moisture analyzer to trace the feature through:
   - rendered scene
   - generated mesh/component metadata
   - processed occupancy
   - raw NetCDF data

## Commands

Create and run a debug case from a saved-view file:

```bash
bun run debug:view -- \
  --saved-view-path saved-views/2026-04-07T23-45-54-970Z-random-walls-5f0848f5.json \
  --target 0.50,0.50
```

Run from a saved-view title already stored in the app:

```bash
bun run debug:view -- \
  --saved-view-title "random walls" \
  --target 0.50,0.50
```

Replay an existing debug-case JSON:

```bash
bun run debug:view -- --case-file tmp/view-debug/random-walls/debug-case.json
```

## Artifacts

The workflow writes a compact bundle under `tmp/view-debug/...`:

- `debug-case.json`
- `capture-context.json`
- comparison screenshots
- `moisture-analysis.json`
- `report.md`
- analyzer plots

Use `view_image` on the generated screenshots when you want to inspect them visually.
