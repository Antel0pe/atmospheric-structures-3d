---
name: run-diagnostic-experiment
description: Run repo-local atmospheric diagnostic experiments when the user asks to "run an experiment", "conduct an experiment", "try a variant", "test a method", or generate pressure-level plots/maps for an exploratory analysis. Use for tmp-based experiments involving ERA5, pressure levels, thermal displacement, raw temperature, climatology, maps, contours, method comparisons, or plot variants in this repository.
---

# Run Diagnostic Experiment

## Core Contract

Conduct the experiment in the repo's `tmp/` folder, keep outputs easy to compare, and avoid surprise data dumps.

Default pressure levels are:

```text
250, 500, 850, 1000 hPa
```

Use different levels only when the user specifies them.

## Output Layout

Create one experiment folder under `tmp/`:

```text
tmp/<experiment-slug>/
  findings/
    <experiment-slug>.md
  <subexperiment-or-variant-slug>/
    <plot-name>_<level>hpa.png
  <another-subexperiment-or-variant-slug>/
    <plot-name>_<level>hpa.png
```

Rules:

- If the user specifies methods, variants, smoothing choices, thresholds, windows, or fields, make each distinct method/variant a subfolder.
- If there is only one method, use a clear subfolder like `base/`, `plots/`, or the method name.
- Use one PNG per requested pressure level unless the user explicitly asks for an overview, panel, GIF, or combined figure.
- Do not make per-level subfolders unless the user asks; put level-specific PNGs directly in the relevant subexperiment folder.
- Always create a brief experiment-specific Markdown note in `tmp/<experiment-slug>/findings/`.
- Keep names repo-relative and portable. Never write full local filesystem paths into generated files, notes, reports, or scripts intended to be shared.

## Artifact Rules

By default, generate only:

- scripts needed to run the experiment, usually in `tmp/<experiment-slug>/`
- PNG plots, one per requested level
- a brief Markdown findings note in `tmp/<experiment-slug>/findings/`
- a small README or JSON manifest only when it materially helps reproduce the run

Do **not** generate `.npy` or `.csv` files unless the user explicitly requests them.

If intermediate arrays are needed, keep them in memory or recompute them. If persistence is unavoidable, ask before writing binary/tabular intermediates.

## Findings Note

After running an experiment, create or update:

```text
tmp/<experiment-slug>/findings/<experiment-slug>.md
```

Keep this note quick and dirty. It should preserve the user's thoughts and the experiment lineage, not become a polished report.

Include:

- title
- one or two sentences describing what the experiment was about
- levels, domain, and variants/methods in brief
- output folders or representative plots, using repo-relative paths
- user feedback or analysis from later conversation
- requested iterations and what each iteration changes
- date of the experiment run

When the user gives analysis, observations, correlations, interpretations, or "I see..." feedback, append it to this experiment note unless the user says otherwise. Do not expand the analysis beyond making it comprehensible. Preserve the user's meaning and keep it brief.

When the user asks for an iteration, append a short iteration entry to the same note with the date, then add any new variant/output paths after running it.

## Plot Rules

For geographic plots:

- default to global latitude/longitude unless the user specifies a crop, region, or map window
- include land/coast/country borders by default
- include any user-specified contours, colors, smoothing, crop windows, annotations, masks, or overlays
- preserve the user-specified map window exactly; do not infer it from an output folder name
- use the repo's established plotting and smoothing helpers when a matching experiment family already exists

Color defaults:

- for temperature plots, use blue to white to red from lowest to highest values
- for precipitation and humidity plots, use white to blue from lowest to highest values
- if the user specifies a different color range, palette, midpoint, or normalization, follow that instead

For pressure-level plots:

- produce one plot per requested level
- put the pressure level in the filename and visible title
- order levels in the user's requested order; otherwise use `250, 500, 850, 1000`

## Workflow

1. Read the user's request and identify:
   - field or diagnostic
   - pressure levels, defaulting to `250, 500, 850, 1000`
   - map window or domain, defaulting to global latitude/longitude
   - subexperiments/variants/methods
   - required overlays, borders, contours, smoothing, and units
2. Check repo context before coding:
   - `notes/instructions.md`
   - `notes/index.md`
   - relevant current notes if the experiment touches product direction or known technical pitfalls
   - `thermal_displacement.md` and `scripts/thermal_displacement.py` when the request involves thermal displacement, equivalent-latitude matching, or polar-like/equator-like temperature score
3. Create `tmp/<experiment-slug>/` and subfolders for each method/variant.
4. Write a small, reproducible script under the experiment folder when needed.
5. Run Python with the conda environment named `atmospheric-structures-3d`.
6. Set writable plotting caches before matplotlib/cartopy work when needed:

```bash
MPLCONFIGDIR=/tmp/atmospheric-structures-3d-cache/matplotlib \
XDG_CACHE_HOME=/tmp/atmospheric-structures-3d-cache \
conda run -n atmospheric-structures-3d python tmp/<experiment-slug>/<script>.py
```

7. Verify the expected PNG count and dimensions after the run.
8. Create or update the Markdown findings note.
9. Summarize the outputs with repo-relative paths and mention any skipped artifacts.

## Implementation Notes

- Prefer temporary scripts over heredoc-driven `conda run` snippets; heredoc stdin has been unreliable in this environment.
- Use repo-local `data/` inputs directly when the experiment is based on available ERA5 or climatology data.
- Keep raw ERA5 temperature in Kelvin unless the user asks for conversion.
- Label rough heuristics as rough; do not present them as physical models.
- When variants look similar, explain what actually changed rather than just listing files.

## Final Check

Before finishing:

- Are all outputs under `tmp/<experiment-slug>/`?
- Did each requested level get its own plot?
- Are land/coast/country borders included on maps unless the user said otherwise?
- Were `.npy` and `.csv` files avoided unless explicitly requested?
- Does `tmp/<experiment-slug>/findings/` contain a brief Markdown note for the experiment?
- Are all paths in generated artifacts repo-relative?
