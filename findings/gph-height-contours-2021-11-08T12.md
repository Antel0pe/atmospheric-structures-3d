# GPH Height Contours

## 2026-05-31 Initial Run

Global ERA5 geopotential-height contour maps for the established repo frame
`2021-11-08T12:00 UTC`. This first pass deliberately reuses the locally
available four-level context file rather than downloading the full
`250-1000 hPa` pressure-level stack.

- Levels: `250`, `500`, `850`, and `1000 hPa`
- Domain: global ERA5 grid
- Variant: raw ERA5 geopotential `z`, converted to geopotential height in
  meters with `z / 9.80665`
- Map styling: filled geopotential-height colors, labeled height contours,
  coastlines, country borders, and gridlines
- Input: `data/four-level-geopotential-height-context_2021-11_p250-500-850-1000.nc`
- Output folder: `tmp/gph-height-contours-2021-11-08T12/plots/`
- Reproducer: `tmp/gph-height-contours-2021-11-08T12/make_plots.py`
- Summary: `tmp/gph-height-contours-2021-11-08T12/summary.json`

## Requested Iterations

- Start with the existing four pressure levels before deciding whether the
  full `250-1000 hPa` ERA5 geopotential stack is worth downloading.

## 2026-05-31 Raw Temperature Comparison

Added matching raw ERA5 temperature maps beside the GPH plots for the same
`2021-11-08T12:00 UTC` frame and the same four pressure levels. The
temperature variant uses blue-white-red filled colors with labeled Kelvin
contours so its broad thermal geometry can be compared directly against the
raw geopotential-height contour geometry.

- Temperature input: `data/era5_temperature_2021-11_08-12.nc`
- Output folder: `tmp/gph-height-contours-2021-11-08T12/plots/`
- Temperature files: `raw_temperature_contours_<level>hpa.png`
