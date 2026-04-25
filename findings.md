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