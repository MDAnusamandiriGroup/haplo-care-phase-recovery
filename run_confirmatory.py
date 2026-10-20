#!/usr/bin/env python3
"""Fixed, independent-seed replication of the previously developed HAPLO-CARE pilot.

PRIMARY comparison: macro donor-level adjacent-heterozygote switch disagreement,
reference+graph minus ML+graph. Positive means improvement with ML.
Statistical unit: independently msprime-generated genealogies (one per seed).
This is simulation-level confirmation, NOT biological / external validation.
"""
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy import stats
from simulate_msprime import simulate

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results' / 'confirmatory_240markers_20seeds'
SEEDS = list(range(11001, 11021)) # FROZEN before seeing these simulation results
N_SITES = 240
PEOPLE_PER_POP = 80
N_SAMPLES = 3 * PEOPLE_PER_POP
MAX_HET_SKIP = 3
ALPHA = .05

PROTOCOL = {
    'protocol_name': 'HAPLO-CARE fixed-seed coalescent simulation confirmation',
    'source_code': 'same simulator/ML+graph/reference+graph from Step2; no hyperparameter retuning',
    'data': 'msprime two-split toy demographic model, 3 labelled populations; 1Mb, 240 common SNPs',
    'seed_set': SEEDS, 'n_independent_genomic_replicates':len(SEEDS),
    'people_per_population': PEOPLE_PER_POP, 'n_sites': N_SITES,
    'max_heterozygous_skip': MAX_HET_SKIP,
    'test_fraction': 0.25,
    'primary_endpoint':'mean donor adjacent-heterozygote disagreement, ref+graph minus ML+graph, unweighted donors within replicate; mean across independently simulated replicates',
    'statistical_test':'two-sided one-sample t test on 20 independent replicate differences, versus 0',
    'confidence_interval':'two-sided 95% Student-t across independent simulations',
    'secondary_check':'two-sided binomial sign test on improvement sign across replicates',
    'population_analyses':'exploratory/descriptive only; no significance claims',
    'decision_rule':'mean gain > 0 and two-sided p < .05 at the primary endpoint; does NOT imply human-data validation',
    'limitations':['toy demography is not empirically calibrated','common evenly binned SNPs only','true simulation population membership is known','reference haplotypes are prephased','no Beagle/SHAPEIT5 comparator','no external real-genome validation'],
}


def evaluate_one(seed):
    prefix=OUT / f'seed_{seed}'
    sim=prefix/'msprime'
    graph=prefix/'graph'
    compfile=graph/'graph_comparison.csv'
    donorfile=graph/'donor_comparison.csv'
    manifest=sim/'simulation_manifest.json'
    if compfile.exists() and donorfile.exists() and manifest.exists():
        m=json.loads(manifest.read_text())
        if m.get('seed') != seed or m.get('variants') != N_SITES:
            raise RuntimeError(f'Output mismatch detected for seed {seed}; do not reuse this folder')
        print(f'Seed {seed}: checkpoint found; skipping recomputation',flush=True)
    else:
        print(f'\nSeed {seed}: simulate + train + compare',flush=True)
        vcf,meta=simulate(sim, seed=seed, people_per_pop=PEOPLE_PER_POP, n_sites=N_SITES)
        subprocess.run([sys.executable,str(BASE/'run_graph_experiment.py'),
            '--vcf',str(vcf),'--population-tsv',str(meta),'--output',str(graph),
            '--samples',str(N_SAMPLES),'--variants',str(N_SITES),
            '--max-step',str(MAX_HET_SKIP),'--seed',str(seed)],check=True,cwd=str(BASE))
    df=pd.read_csv(donorfile)
    df['seed']=seed
    wide=df.pivot(index=['donor','population'],columns='method',values='error')
    if not {'population-prior-graph','ml-graph-consensus'}.issubset(wide.columns):
        raise ValueError('Missing required methods for seed '+str(seed))
    wide['gain']=wide['population-prior-graph']-wide['ml-graph-consensus']
    gain=float(wide['gain'].mean())
    row={'seed':seed,'n_test_donors_evaluable':int(len(wide)),
         'ref_graph_error':float(wide['population-prior-graph'].mean()),
         'ml_graph_error':float(wide['ml-graph-consensus'].mean()),
         'gain_pp':gain*100}
    per_pop=wide.reset_index().groupby('population').agg(
        n_evaluable=('gain','size'),gain_pp=('gain',lambda v:100*float(v.mean())),
        ref_error=('population-prior-graph','mean'),ml_error=('ml-graph-consensus','mean')).reset_index()
    per_pop.insert(0,'seed',seed)
    print(f"RESULT seed {seed}: ML gain = {row['gain_pp']:+.3f} pp, n={row['n_test_donors_evaluable']}",flush=True)
    return row,per_pop


def summarize(rows, poprows):
    tab=pd.DataFrame(rows).sort_values('seed')
    pops=pd.concat(poprows,ignore_index=True)
    tab.to_csv(OUT/'seed_level_primary.csv',index=False)
    pops.to_csv(OUT/'per_population_exploratory.csv',index=False)
    v=tab['gain_pp'].to_numpy()
    n=len(v)
    mu=float(v.mean())
    sd=float(v.std(ddof=1))
    se=float(sd/np.sqrt(n))
    tcrit=float(stats.t.ppf(1-ALPHA/2,n-1))
    ci=[mu-tcrit*se,mu+tcrit*se]
    test=stats.ttest_1samp(v,0,alternative='two-sided')
    signs=int((v>0).sum())
    sign_p=float(stats.binomtest(signs,n,0.5,alternative='two-sided').pvalue) if not np.any(v==0) else None
    result={
        'analysis':'Independent-seed simulation-only confirmation of fixed primary comparison',
        'n_independent_seeds':n,
        'mean_reference_graph_error_pct':float(100*tab['ref_graph_error'].mean()),
        'mean_ml_graph_error_pct':float(100*tab['ml_graph_error'].mean()),
        'mean_gain_percentage_points':mu,
        'relative_gain_pct': float(100*mu/(100*tab['ref_graph_error'].mean())),
        'gain_sd_across_seeds_pp':sd,
        'gain_ci95_t_pp':ci,
        'gain_pvalue_two_sided_seed_level_ttest':float(test.pvalue),
        'positive_seed_count':signs,
        'sign_test_pvalue_two_sided':sign_p,
        'success_under_predeclared_simulation_rule':bool(mu>0 and float(test.pvalue)<ALPHA),
        'caution':'Even success here is only replication under one toy demographic simulator, not externally validated phasing or established novelty.',
    }
    (OUT/'statistical_summary.json').write_text(json.dumps(result,indent=2))
    report=f'''# HAPLO-CARE — fixed 20-seed confirmatory simulation study

All 20 seeds were fixed before this run. This analysis is **not** independent human-data validation.

## Primary endpoint

Mean paired error improvement, reference + graph minus ML + graph, calculated per simulated genealogy and tested across 20 independent genealogies.

- Mean reference + graph error: {result['mean_reference_graph_error_pct']:.3f}%
- Mean ML + graph error: {result['mean_ml_graph_error_pct']:.3f}%
- Mean absolute improvement: {mu:+.4f} percentage points
- Relative improvement: {result['relative_gain_pct']:+.2f}%
- 95% CI (independent-seed Student-t): [{ci[0]:+.4f}, {ci[1]:+.4f}] percentage points
- Two-sided p (seed-level paired comparison): {float(test.pvalue):.5g}
- Seeds with positive gain: {signs}/{n} (sign test p={sign_p})
- Predeclared simulation-level rule met: {result['success_under_predeclared_simulation_rule']}

## Important boundaries

The comparator is **population reference + the same graph decoder**, not Beagle or SHAPEIT5. The simulator uses a simplified, fixed two-split demography with known population labels and regularly binned common SNPs. Accordingly these data support at most **preliminary reproducibility under the simulated conditions**. Do not claim comprehensive methodological validation, ancestry generalizability, calibrated uncertainty, or real-world superiority.
'''
    (OUT/'SUMMARY_README.md').write_text(report)
    return result


def package():
    zfile=BASE/'HAPLO_CARE_CONFIRMATORY_RESULTS_20_SEEDS.zip'
    with zipfile.ZipFile(zfile,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('PROTOCOL_FROZEN.json','seed_level_primary.csv','per_population_exploratory.csv',
                     'statistical_summary.json','SUMMARY_README.md'):
            z.write(OUT/name,name)
        for file in OUT.rglob('*'):
            if file.is_file() and file.name in ('graph_report.json','graph_comparison.csv','donor_comparison.csv','simulation_manifest.json'):
                z.write(file,file.relative_to(OUT))
    print('\nRESULT FILE TO UPLOAD:',zfile, '\nsize:',round(zfile.stat().st_size/1e6,2),'MB')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'PROTOCOL_FROZEN.json').write_text(json.dumps(PROTOCOL,indent=2))
    started=time.time()
    all_rows=[];all_pops=[]
    for s in SEEDS:
        row,pop=evaluate_one(s)
        all_rows.append(row);all_pops.append(pop)
        # checkpoints include partial result summaries
        pd.DataFrame(all_rows).to_csv(OUT/'IN_PROGRESS.csv',index=False)
    summary=summarize(all_rows,all_pops)
    package()
    print('\n--- FINAL RESULT ---\n',json.dumps(summary,indent=2))
    print('Total seconds:',round(time.time()-started,1))

if __name__=='__main__':
    main()
