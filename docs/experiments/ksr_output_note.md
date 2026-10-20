# KEEP–SPLIT–RECOVER experiment: archived aggregate only

The repository includes `results/ksr/aggregate_summary.csv` and `protocol.json` from a controlled-error experiment on 2,269 adjacent pairs. The protocol used an 8% injected switch-error rate and assessed thresholds 0.70, 0.85, and 0.95. The comparison labels came from published statistical phase, not independently verified molecular truth.

At the prespecified 0.85 threshold, the archived table reports:

| Rule | Coverage | Residual error among retained pairs | Recovery precision |
| --- | ---: | ---: | ---: |
| Keep all | 100.0% | 8.02% | — |
| Split only | 66.55% | 0.07% | — |
| Reference recovery | 74.13% | 2.08% | 80.23% |
| Fixed fusion | 76.82% | 1.95% | 81.14% |
| Adaptive recovery | 81.97% | 2.37% | 78.35% |

This shows an accuracy–coverage trade-off in that particular controlled-error setting. It is not evidence that these decision rules improve real-world phasing.

The saved materials did not include the script that generated the per-edge actions and probabilities. For that reason, the repository keeps only the aggregate table and protocol; it does not claim that the KSR results can be regenerated from the included source. Per-individual rows and sample identifiers are also not included. The result should be treated as a recorded exploratory output until the original decision-generation code and a fully specified evaluation can be recovered.
