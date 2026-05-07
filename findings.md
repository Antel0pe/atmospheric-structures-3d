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
