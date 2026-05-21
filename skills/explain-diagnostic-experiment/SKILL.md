---
name: explain-diagnostic-experiment
description: Explain meteorology/data diagnostic experiments, plots, maps, histograms, heatmaps, waterfalls, regressions, metrics, or scripts in plain traceable terms. Use when the user asks "explain this plot", "explain what you did", "what is this method doing", "how are x/y/color/points determined", "what does this statistic mean", or asks for the method behind an experiment, especially for ERA5, climatology, thermal displacement, pressure-level, longitude/latitude, or atmospheric analysis outputs.
---

# Explain Diagnostic Experiment

## Core Rule

Explain the **data-to-mark pipeline**. Do not start with generic visualization talk. Start from one plotted thing and say exactly how it was produced from the data.

The explanation that worked well here had this shape:

1. Name what one mark represents.
2. Say how its x coordinate is chosen.
3. Say how its y coordinate is chosen.
4. Say how color/line/fit/subplot is chosen.
5. Walk through one concrete data item.
6. State what the plot can and cannot prove.
7. Anticipate the confusion the user is likely having.

## Default Structure

Use these sections when they fit:

- **What One Mark Means**: dot, line, pixel, cell, contour, bar, lane, or fitted line.
- **How Coordinates Are Determined**: x/y axes in plain English, not only variable names.
- **Step-By-Step For One Example**: one longitude, pressure level, cell, bucket, contour, or profile.
- **Common Confusions**: explicitly say what the axis/mark is *not*.
- **How To Read The Result**: what positive slope, darker color, higher lane, or crossing means.
- **Difference Between Methods**: if comparing methods, state the different question each method answers.

Keep the answer short enough to read, but do not compress away the coordinate semantics.

## Required Moves

Always identify the unit of summarization.

Examples:

- "Each dot is not one grid cell. Each dot summarizes an entire north-south profile at one longitude."
- "Each pixel is one latitude-longitude bin; color is the score at that bin."
- "Each lane is a longitude; the wiggle inside the lane is the score profile."
- "Each bar is a histogram bucket; height is the number of cells in that bucket."

Always separate **data value**, **display coordinate**, and **statistical summary**.

Example:

```text
x coordinate = longitude
y coordinate = estimated transition latitude
color = pressure level
red line = linear fit through the estimated latitudes
```

Do not say "the plot shows a trend" until after the mechanics are clear.

## Method Explanation Pattern

When explaining a derived method, use this pattern:

```text
For each [group]:
1. Take [source field] over [domain].
2. Compute [intermediate thing].
3. Reduce it to one number by [rule].
4. Plot that number at x=[x rule], y=[y rule].
```

Then add:

```text
Common confusion:
- [thing] is not [likely mistaken interpretation].
- [line/fit/color] does not mean [overclaim].
```

## Plot-Specific Guidance

For line plots:

- Say whether one line is one longitude, one pressure level, one time, or one profile.
- Say what the line shape means before discussing color.
- If color is carrying order, say whether the order is physical or just visual.

For heatmaps:

- Say x and y are physical coordinates.
- Say color is the actual measured/derived value.
- If contours are present, say they are constant-value lines and explain one contour level.

For waterfalls/ridgelines:

- Say the vertical placement is mostly an artificial lane/offset.
- Say the wiggle inside each lane is the data value.
- Warn that y is not purely the data value.

For fitted-line/statistical plots:

- Say what each dot summarizes.
- Say what the red/fitted line is fitting.
- Explain slope direction in the user's coordinate language.
- Say R2 means "how line-like the dot cloud is", not physical proof.

## Tone And Wording

Prefer concrete phrases:

- "A dot at `45` means..."
- "Moving right means moving east."
- "This is the latitude where the profile crosses score `50`."
- "This averages the steep parts of the transition zone."

Avoid vague phrases:

- "This visualizes the relationship between variables."
- "The plot captures trends."
- "The metric characterizes the structure."

Use formulas only after plain English, and only when they disambiguate.

## Example Style

For a plot of transition latitude by longitude:

```text
Both plots reduce each longitude profile to one latitude number, then ask:
"As longitude moves west-to-east, does that estimated transition latitude move north or south?"

x coordinate of each dot = longitude
y coordinate of each dot = estimated transition latitude

Each dot is not one grid cell. Each dot summarizes an entire north-south line/profile at one longitude.
```

For a threshold-crossing method:

```text
For each longitude:
1. Take the score profile from north to south.
2. Look for where it crosses score 50.
3. Interpolate between neighboring latitude cells.
4. If there are multiple crossings, keep the sharpest one.
5. Plot that crossing latitude as the dot's y value.
```

For a centroid method:

```text
This does not find one exact contour. It finds the center of mass of the transition zone.
It weights latitudes more heavily where the score changes faster.
If there are two separate transition zones, the centroid can land between them.
```

## Final Check

Before answering, verify the explanation answers these:

- Could the user point to one dot/line/pixel/bar and know what it represents?
- Could the user reconstruct x/y/color from the source data?
- Did you name at least one likely misreading?
- Did you distinguish the plotted statistic from the physical phenomenon?
