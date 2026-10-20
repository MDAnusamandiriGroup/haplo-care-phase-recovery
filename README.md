# HAPLO-CARE

A research prototype exploring reference-informed haplotype phase prediction, local graph consistency, and selective phase decisions. The code is intended to make the experiments inspectable and repeatable. It is not a validated clinical tool or a replacement for established phasing software.

## What is in this repository

- A pairwise cis/trans predictor that uses reference-panel features and a gradient-boosted model.
- A finite-range graph decoder that turns pairwise probabilities into a locally consistent phase path.
- A fixed 20-seed simulation study using `msprime`.
- An exploratory 5 Mb chromosome 22 analysis using 1000 Genomes phase labels.
- Aggregate outputs from an earlier KEEP–SPLIT–RECOVER experiment. The code that generated its per-edge decisions was not included in the saved project files, so that result is documented as output-only and is not presented as reproducible here.

## Current status

The strongest result so far is from a fixed 20-seed simulation. The machine-learning-plus-graph condition had a mean adjacent-heterozygote disagreement rate of 8.90%, compared with 9.28% for the population-reference-plus-same-graph comparator. This is a 0.374 percentage-point difference, or 4.03% relative improvement, across the 20 simulated genealogies (two-sided seed-level p = 3.12 × 10⁻⁶). The simulation uses a simplified demographic model. This is evidence of repeatability under that simulator, not evidence of better performance on real genomes.

In the chromosome 22 pilot, the corresponding mean disagreement was 8.98% for the ML-plus-graph condition and 9.60% for the reference-plus-graph condition across 45 held-out individuals and five selected 1 Mb windows. The panel phase labels are statistically inferred, so this measures agreement with those labels, not accuracy against independent truth. The analysis is exploratory and does not compare HAPLO-CARE against SHAPEIT5, Beagle, or Eagle.

See [`docs/RESULTS_AND_INTERPRETATION.md`](docs/RESULTS_AND_INTERPRETATION.md) for the methods, results, and limits.

## Run the tests and toy pilot

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py --output results/synthetic_pilot
python run_graph_experiment.py --output results/graph_pilot
```

The pilot uses a simplified synthetic founder-mosaic generator with known phase. It is useful for checking the code path, but it is not a realistic population-genetics simulator.

## Run the 20-seed simulation

Install the additional simulation dependency and run:

```bash
python -m pip install -r requirements-confirmatory.txt
python run_confirmatory.py
```

The run is checkpointed by seed. It writes simulation outputs under `results/confirmatory_240markers_20seeds/`. The version of the aggregate output included in this repository is in the same folder.

## Run the chromosome 22 pilot

Download the two files listed in [`data/README.md`](data/README.md), keep them in `data/`, then run:

```bash
python run_real_chr22_5mb.py \
  --vcf data/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \
  --panel data/20130606_g1k_3202_samples_ped_population.txt \
  --output results/real_chr22_5mb_run
```

This analysis reads a selected subset of common markers from the five windows. It does not benchmark every variant on the chromosome. Do not upload the VCF, sample-level predictions, or individual-level result files to this public repository.

Compatibility: Python 3.10 or newer.

## Important limitations

- The predictor estimates pairwise phase relationships; it does not by itself reconstruct a fully validated chromosome-scale haplotype.
- The graph decoder optimizes a finite-range pairwise pseudo-likelihood. Correlated edges may double-count evidence, so the returned path is not necessarily a calibrated posterior estimate.
- The 20-seed experiment uses one simplified simulation design and known population labels.
- The real-data pilot evaluates agreement with inferred phase labels, not independent ground truth.
- No comparison against SHAPEIT5, Beagle, or Eagle has been completed. No claim of novelty, clinical usefulness, or superiority is being made.
- The archived KEEP–SPLIT–RECOVER table is an aggregate output from a controlled-error experiment. The original decision-generation script was not present in the saved files; see [`docs/experiments/ksr_output_note.md`](docs/experiments/ksr_output_note.md).

## Citation and contact

This is an independent exploratory proof of concept. Licensed under the MIT License. See LICENSE.
