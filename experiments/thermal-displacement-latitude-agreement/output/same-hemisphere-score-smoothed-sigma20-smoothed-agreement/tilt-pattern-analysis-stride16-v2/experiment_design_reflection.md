# Why The Two Transition-Fit Experiments Worked

This note is about two specific plots:

- `plots/within_level_fit_score50_crossing.png`
- `plots/within_level_fit_gradient_centroid.png`

They worked because they stopped trying to make a prettier version of the existing line plot and instead converted the visual intuition into a measurable geometric question:

> At each longitude, where is the north-south thermal-displacement transition located, and does that transition latitude move systematically as longitude changes?

## What Produced The Good Thinking

The useful context was not just "make a plot." It was the user's frustration with line tracing:

- many colored longitude lines were visually hard to follow
- color was carrying too much meaning
- the user wanted to see whether a tilted feature existed through longitude
- the waterfall view hinted at a diagonal pattern but did not cleanly prove it

The key move was to stop encoding every profile directly. Instead, each longitude profile was reduced to one interpretable latitude estimate. That created a simple plot where every dot had a clear provenance:

```text
x = longitude
y = estimated transition latitude from that longitude's score-vs-latitude profile
```

That is why the plots were legible. The axes answered the real question.

## Why `score50_crossing` Was A Good Experiment

This method asks the most literal possible question:

> Where does the profile cross the middle of the thermal-displacement scale?

For each pressure level and longitude:

1. Take the score profile from north to equator.
2. Find the latitude where score crosses `50`.
3. If multiple crossings exist, choose the crossing with the strongest local slope.
4. Plot longitude versus that crossing latitude.
5. Fit a line only as a summary of directional tilt.

This is good because it is easy to explain and easy to falsify. A future Codex should prefer this kind of first statistic when the user is trying to understand a visual pattern: pick a simple, inspectable landmark in the field and track where it moves.

Its weakness is also clear: score `50` is arbitrary. That weakness is acceptable because the method is transparent.

## Why `gradient_centroid` Was A Good Pair

This method was the right companion because it asks the same question without relying on one arbitrary score value.

For each longitude:

1. Look only at the broad transition band, score `20..80`.
2. Compute where the score changes fastest with latitude.
3. Treat stronger gradients as more important.
4. Return the weighted center latitude of that transition zone.

This made the experiment robust to profiles where the score-50 crossing was noisy, nonrepresentative, or one of several crossings. It was not a random second method; it was a deliberately different estimator of the same physical/visual idea.

That pairing mattered:

- `score50_crossing` gave a literal contour-like answer.
- `gradient_centroid` gave a transition-zone answer.
- Agreement between them increased confidence.
- Disagreement identified where the pattern was method-dependent.

## The General Recipe To Replicate This Quality

Future Codex experiments should use this pattern:

1. State the user's visual intuition as a measurable question.

   Bad: "make a better visualization."

   Good: "Does the transition latitude shift eastward with longitude?"

2. Define the unit of evidence.

   Here the unit was:

   ```text
   one longitude profile at one pressure level
   ```

3. Reduce each unit to one meaningful statistic.

   Here:

   ```text
   profile -> transition latitude
   ```

4. Make every plotted coordinate explainable in one sentence.

   Here:

   ```text
   x is longitude; y is the transition latitude estimated from that longitude's profile.
   ```

5. Pair one simple/literal method with one robust/structural method.

   Here:

   ```text
   score50_crossing + gradient_centroid
   ```

6. Add a fit only after the dots exist.

   The red line was not the experiment. The dots were the experiment. The fit summarized whether those dots formed a coherent tilt.

7. Include failure modes in the method, not after the fact.

   For these plots:

   - score-50 crossing can be arbitrary
   - multiple crossings can occur
   - gradient centroid can average two separate transition zones
   - low `R2` means the apparent tilt is weak or non-linear

8. Anticipate axis confusion.

   These plots are easy to misunderstand unless the explanation explicitly says:

   ```text
   y is latitude, not score
   each dot summarizes a whole north-south profile
   the red line is a statistical summary, not a physical boundary
   ```

## What To Tell Future Codex

Use this instruction when asking for similar experiments:

> Do not start by making a prettier plot. First translate the visual hunch into a measurable geometric question. Define what one observation is, reduce each observation to one physically interpretable statistic, plot that statistic directly, and pair a simple threshold method with a gradient/centroid method that tests the same idea differently. Explain exactly how each dot's x and y coordinates are computed, and state the failure modes.

That is the core reason these two plots worked.

They were not successful because they were complicated. They were successful because they made the hidden question explicit:

> Where is the transition, and does that location move systematically?
