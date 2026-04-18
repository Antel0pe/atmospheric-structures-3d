# Example Questions

Questions and requests from a real `structure-of-data` exploration, cleaned lightly and grouped by similar intent.

The groups are ordered by first appearance in the conversation. The questions inside each group stay in conversation order.

## Run And Field Choice

- Run `structure-of-data`.
- On potential temperature relative to climatology.

## Interpreting Report Metrics

- Cross-level spread is actually fairly comparable here (`~1.27x` representative spread ratio), so the bigger issue is the vertical weighting, not per-level scaling. What does this mean?

## Per-Level Components And Threshold Meaning

- When I say `10%` anomaly, I mean take the difference of the cell's potential temperature from the climatology mean, then find the top `10%` of differences. `10%` means find the top `10%` of cells that diverge most from the climatology mean. It’s just taking all the cells and eliminating the `90%` with the smallest differences.
- Divide it into how many are warm vs cold for each of the surviving ones.

## Threshold Sweeps

- Give me a threshold graph so if I go from `1%` to `5%` to `10%` to `15%` to `20%` to `25%`, what do the counts look like? Give me a line graph for each level. It can be a single graph with all pressure levels in different colors or something.

## Vertical Coherence

- Look at the data and look at a metric like vertical coherence. For a given lat/lon, how many levels at `10%` anomaly are above the threshold and present there? How many levels are continuous? How many levels are the same sign? Are there any particular trends, like there are more continuous levels around certain levels, or rarely continuous around certain levels? What does the vertical coherence look like?
- What kinds of thresholds or selection parameters maximize vertical coherence?

## Full 2D Level Plots

- In the repo `tmp` folder, generate plots of all `37` levels with `10%` anomaly for me.

## Component Size Distributions

- What are the sizes of the components at each level? The range, maybe a line graph again for each pressure level? Would a line graph make sense? I want to see all the ranges. I guess I want the numbers too. How would you show me this? I want to see the distribution of component size per pressure level at `10%` anomaly.

## Component-Filtered 2D Plots

- Generate the `2D` plots again, but this time with top `10%` anomaly and top `10%` component sizes.

## Vertical Coherence On The Component-Filtered Mask

- Do the vertical coherence thing on the top `10%` anomaly, top `10%` component size per anomaly, and do the analysis like you did previously about vertical coherence. You can reference the example questions.
- For the vertical coherence, how many continuous levels end in a level that does not pass the threshold for `10%` anomaly? I want to differentiate between cases where a run ends because it runs into the opposite-sign anomaly versus just empty space.
- Specify the top `10%` anomaly and top `10%` component size. Then at the end of each continuous run, look one level directly above and below. If it ends in empty space, check whether the anomaly in that cell has the same sign. If so, include that cell in the mask regardless of anomaly threshold, and continue up and down until you end in an anomaly with a different sign. Make the plots for each pressure level again and tell me what the vertical coherence stats are.
- What is the distribution of the anomaly percentage now, given that we start with top `10%` anomaly but then include anything with the same sign? Give me the quantile chart again if that makes sense.
