# HAPLO-CARE — fixed 20-seed confirmatory simulation study

All 20 seeds were fixed before this run. This analysis is **not** independent human-data validation.

## Primary endpoint

Mean paired error improvement, reference + graph minus ML + graph, calculated per simulated genealogy and tested across 20 independent genealogies.

- Mean reference + graph error: 9.278%
- Mean ML + graph error: 8.904%
- Mean absolute improvement: +0.3741 percentage points
- Relative improvement: +4.03%
- 95% CI (independent-seed Student-t): [+0.2538, +0.4945] percentage points
- Two-sided p (seed-level paired comparison): 3.1166e-06
- Seeds with positive gain: 19/20 (sign test p=4.00543212890625e-05)
- Predeclared simulation-level rule met: True

## Important boundaries

The comparator is **population reference + the same graph decoder**, not Beagle or SHAPEIT5. The simulator uses a simplified, fixed two-split demography with known population labels and regularly binned common SNPs. Accordingly these data support at most **preliminary reproducibility under the simulated conditions**. Do not claim comprehensive methodological validation, ancestry generalizability, calibrated uncertainty, or real-world superiority.
