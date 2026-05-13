something that worked well
- top 10% anomaly relative to climatology per pressure level, top 10% component size
    - treat this as core, then expand upwards or downward as long as it is the same anomaly sign
        - this is good at identifying the big air masses but i believe creates straight vertical boundaries rather than tilted as might be more reasonably in general
    - experimenting with expanding upward and downward within horizontal radius of 1-3 cells given it is above certain anomaly threshold
        - might help show more of the tilt
        - doesnt really interestingly, not super sure why - likely code correctness error

- experimenting with looking at gradients between air masses more

- potential temperature anomaly and raw temperature anomaly are basically the same thing
- plots in tmp/temperature-cardinal-contrast where you sum up the temperature change in the cardinal directions look really freakin cool
- probably want to look towards air masses of similar thermodynamic vertical character
- obviously cant look only straight up for vertical coherence, look within window like 3x3
- std dev and percentile based bucketing seems to work well across levels tmp/potential-temperature-anomaly-discrete-buckets-all-levels/2021-11-08T12-00/
- only showing anomalously cold/warm air masses means maybe you dont see where anomalous air pushes up against normal air. can implicitly see it if the air front is tilted or something?

- interpolate between standard levels somehow so altitude cutaway is better

- make a google earth human kinda feature where you can place it somewhere and see a 3d cross section easily? maybe just click a spot, select which level to get to then vertical cutaway in straight line?

what is the stated goal?
- true to atmosphere - is not good enough. 
- understand 3d thermal structure faster than flipping through 2d maps?
- connected volumnes where interior is much more similar than outside?
- view that takes temperature and emphasizes other information it typically tells you like pressure, etc
- does it even make sense to group the atmosphere temperature wise vertically? is the structure just 2d slabs?
- see weather systems like a meteorologist?
- learned meteorologists might not even think in 3d as much as recognizing phenomena and if x,y,z then this happens intuition

requirements:
- if vertical structure exists should show vertical coherence, tilted, tubes, whatever the case
- persist across timesteps
- meteorologically reasonable
- make things like fronts super obvious visually

target thermal representation
- some element of temporal continuity
- can investigate origin of air, divide plot into say 10 even cells then color cell based on which cell area it came from
- investigate if the assumption of tilted, winding air is meteorologically true
- look at gradients? or 2nd derivatives?
- looking at skew t plot with a smooth variation in temperature, is it possible to detect different air masses by how differently the skew t line changes
- some element of 3d

- maybe show correlated fields like look at pressure at specific level and have pop up that says oh temperature high, divergence low correlated with this field. select area by circling area on screen


GOAL: i look at 2d temperature maps, see a warm blob and wonder hm i wonder what that blob looks like at different pressure levels. it is a smooth sphere, tilted, what happens to it? i want to make it easier and better to answer this question for myself.

- currently white part is the middle of the color scale between blue and red. come up with some way to make white the part where genuinely cold and warm air masses collide around mid latitude. NOT temperature gradient or fronts as that happens everywhere but genuinely cold and warm air colliding is where white happens. might be hard to make this a uniform feature across the globe. 

finding meridional value
- smoothing at scale of 500km helps
- median is better than mean but not significantly
- isotherms pretty good at least at higher levels, lower levels are a mess. maybe smoothing first?
- my assumption is there should be a broadly connected line across the globe (obviously not straight, but generally connected) - this assumption might be wrong
- good idea of not needing to identify coherent path but can identify high collision regions then connecting line between them
- raw temp seems to show straight line while climatology methods find more curving areas. raw temp might find general region and anomaly the specific air mass contours?

have 2d map click region on it that selects a 2d region then can click button and as you slide the altitude slider it creates a 3d object and then you can fly around it and look at it

- having codex run experiments for me with goals and constraints isn't very useful largely because
    - i cant understand the experiments it runs and why and what i can learn from them
        - this could be because it doesnt really have great direction on what i want to learn. i give it the goal of hey this is what im looking for. could be more specific honestly it could be running shit experiments because i havent constrained it enough. it's missing some context. but even so i just dont understand the experiments well enough to learn from them. maybe it's not so much i cant learn from it but i couldnt follow it's thinking or reasoning for why this expriment makes sense to learn. i dont have a question that this experiment is answering so it doesnt fit into my knowledge because i wasnt wondering about it. hm maybe it isn't even that but i cant evaluate the results well enough. like i see plots but im missing some context or the larger flow into understanding why this might be an interesting result. i tell it to run 10 experiments at a time and react to the previous ones so codex iterates logically but i just dont see that iteration, just the end results so it looks like disconnected experiments.
    - example report: atmospheric-structures-3d/tmp/temperature-pressure-level-boundary-method-loop-3/report.md


additional constraings and thoughts
- I want the boundary line to have a clear meteorological reason for being where it is, not just be an algorithmic line that moved around.
- The real goal is identifying genuinely warm/tropical-like and cold/polar-like air masses, especially where they interact near the midlatitudes.
- The line should be connected around the globe because warm and cold regimes are always meeting somewhere, even if the transition is sharp in some places and diffuse in others.
- I care less about “how did the line change?” and more about “why is this line meteorologically defensible here and not somewhere else?”
- Warm/cold air does not always need to be anomalous for its latitude. Normal warm-side air meeting normal cold-side air can still be a valid boundary.
- Simplicity is important. I prefer simple heuristics over complicated multi-term hybrids that may overfit or become too specific.
- Possible next idea: identify cold and warm regimes first, then infer the transition/ribbon between them rather than directly detecting a one-cell line.
- The transition should probably be a variable-width ribbon, since some warm/cold boundaries are sharp and others are broad or gradual.
- the hard problem is defining “hot/warm” and “cold/chilly” in a meteorologically meaningful way.
- The structure is not always a simple cold-transition-warm sequence; warm anomalies can be embedded inside cold regions, and there can be multiple local regimes.
- Scale matters: I need to define whether I’m looking for hot vs cold, warm vs chilly, or smaller local contrasts.
- The pole-cold / equator-warm assumption may work broadly, especially aloft, but may be weaker near the surface.
- I’m wondering whether temperature alone is making this problem unnecessarily hard.
- At upper levels, especially around 250 hPa, jet-stream structure may make the warm/cold divide much easier and more meteorologically grounded.

the smallest bucket thing probably works best because it finds areas of rapid change like an isoterm does
- however given this single range globally to find areas of hot + cold air is maybe not accurate because different air masses collide in different ways
- smoothing the temperature buckets before picking minima is good 
- equivalent latitude isotherms are terrible perhaps because of the rapid precise changes? 