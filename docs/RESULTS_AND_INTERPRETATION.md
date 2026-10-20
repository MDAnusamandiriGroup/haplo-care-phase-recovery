# Results and interpretation

## What HAPLO-CARE currently does

The core model predicts whether adjacent heterozygous variants are in cis or trans. It builds reference-informed features, including allele-frequency and linkage-disequilibrium summaries, and uses a histogram gradient-boosted classifier. Training features are cross-fitted so the target person's haplotypes are not used as their own reference. The graph extension predicts local phase relationships for heterozygote pairs up to three steps apart and uses dynamic programming to choose a consistent local phase path.

This is still a research prototype. Pairwise prediction and local graph decoding are not equivalent to validated, chromosome-scale phasing. The graph objective treats correlated edges as separate pseudo-likelihood factors and can over-count evidence.

## Fixed 20-seed simulation

The included confirmatory experiment used 20 fixed, independently simulated genealogies, 240 markers per 1 Mb, and a simplified three-population demographic model. For each seed, it compared the population-reference predictor plus graph decoder with the ML predictor plus the same graph decoder.

- Mean reference-plus-graph disagreement: 9.278%
- Mean ML-plus-graph disagreement: 8.904%
- Mean difference: 0.374 percentage points (4.03% relative improvement)
- 95% seed-level t interval: 0.254 to 0.494 percentage points
- Two-sided seed-level p-value: 3.12 × 10⁻⁶
- Positive gain in 19 of 20 seeds

The independent unit is the simulated genealogy, not the many linked pairs within it. This helps avoid treating linked markers as independent replicates. The evidence supports repeatability under this one simulator. It does not show generalisation across realistic evolutionary settings or performance on human genomes.

## Exploratory chromosome 22 analysis

The real-data pilot uses the 1000 Genomes high-coverage phased panel for chr22:30–35 Mb (GRCh38), selecting five 1 Mb windows and common SNP markers. It used 60 samples in each of AFR, EUR, and EAS groups, with held-out evaluation on 45 individuals.

- Mean reference-plus-graph disagreement: 9.603%
- Mean ML-plus-graph disagreement: 8.984%
- Absolute difference: 0.619 percentage points (6.45% relative to the reference comparator)
- Exploratory individual-bootstrap 95% interval for the absolute difference: 0.186 to 1.033 percentage points

The reference panel's phase labels are statistically inferred. This analysis measures agreement with those labels, not accuracy against independently established truth. The sample is small, only common markers were included, and relatedness may remain. The result is exploratory and is not a benchmark against SHAPEIT5, Beagle, or Eagle.

## Earlier synthetic pilot

The first toy pilot used a deliberately simplified founder-mosaic generator (950 people, 72 sites). It helped test the model and graph-decoding code. Most of the reduction in disagreement came from adding graph consistency; the ML model did not reliably improve on the reference-prior graph comparator across a handful of toy seeds. Those results are kept as an engineering record, not as evidence of validated performance.

## What remains to be done

1. Compare against established phasing tools on the same data and with compatible definitions of switch error.
2. Evaluate on trios or another source of independently supported phase truth.
3. Test different demographic scenarios, admixture, missingness, rare variants, and population-label uncertainty.
4. Calibrate reliability estimates and assess whether selective splitting or recovery improves the accuracy–coverage trade-off.
5. Run a literature review before making any novelty claim.

The existing benchmark references include [SHAPEIT5 documentation](https://odelaneau.github.io/shapeit/) and its [switch-error documentation](https://odelaneau.github.io/shapeit/docs/documentation/switch/).
