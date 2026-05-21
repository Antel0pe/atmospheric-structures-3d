# Contour Coherence Analysis

This is a probe, not a classifier. It tests whether simple score contours show coherent pressure-level relationships, within-level contour grouping, or clean score separability.

## Strongest cross-level same-score contour correlations

- score 10, levels 850-250: r=1.00, mean separation=15.2 deg
- score 10, levels 850-500: r=1.00, mean separation=1.9 deg
- score 15, levels 1000-250: r=-0.97, mean separation=5.0 deg
- score 50, levels 1000-850: r=0.96, mean separation=1.3 deg
- score 55, levels 1000-850: r=0.96, mean separation=1.1 deg
- score 35, levels 1000-850: r=0.94, mean separation=1.7 deg
- score 30, levels 1000-850: r=0.94, mean separation=4.7 deg
- score 45, levels 1000-850: r=0.94, mean separation=1.5 deg

## Within-level contour spacing

- 1000 hPa: tightest band near score 52.5 with mean spacing 3.96 deg; loosest near 82.5 with 9.67 deg.
- 850 hPa: tightest band near score 12.5 with mean spacing 2.87 deg; loosest near 87.5 with 11.22 deg.
- 500 hPa: tightest band near score 17.5 with mean spacing 2.97 deg; loosest near 27.5 with 9.25 deg.
- 250 hPa: tightest band near score 52.5 with mean spacing 2.67 deg; loosest near 17.5 with 7.55 deg.

## Simple score split test

- 1000 hPa: Otsu threshold 53.5, between-class separation ratio 0.81, cold/low-score fraction 0.48.
- 850 hPa: Otsu threshold 53.5, between-class separation ratio 0.83, cold/low-score fraction 0.48.
- 500 hPa: Otsu threshold 55.5, between-class separation ratio 0.80, cold/low-score fraction 0.54.
- 250 hPa: Otsu threshold 49.5, between-class separation ratio 0.80, cold/low-score fraction 0.50.

## Interpretation guardrail

A high same-score contour correlation means similarly shaped contour paths across pressure levels, but it does not by itself prove a discrete hot/cold object. Tight contour spacing means rapid score transition; loose spacing means diffuse transition. The Otsu threshold is only a one-dimensional split test on score distribution.
