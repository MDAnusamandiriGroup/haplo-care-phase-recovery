#!/usr/bin/env python3
"""Experimental extension: learn local pairwise phase edges and decode a consistent path."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.haplocare import synthetic_haplotypes,read_phased_vcf,split_individuals,crossfit_train,make_examples
from src.graph import decode_consistent_phase,chained_phase

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vcf')
    p.add_argument('--population-tsv')
    p.add_argument('--output',default='results/graph')
    p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--samples',type=int,default=950)
    p.add_argument('--variants',type=int,default=72)
    p.add_argument('--max-step',type=int,default=3)
    a=p.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    if a.vcf:
        h,pops,positions,sid,_=read_phased_vcf(a.vcf,a.samples,a.variants,a.population_tsv)
        source='prephased cohort agreement (not independent true-phase accuracy)'
    else:
        h,pops,positions,sid=synthetic_haplotypes(a.samples,a.variants,seed=a.seed)
        source='simplified founder-mosaic synthetic simulation'
    train,test=split_individuals(pops,seed=a.seed)
    xtr,ytr,_=crossfit_train(h,pops,positions,train,seed=a.seed,max_het_step=a.max_step)
    xte,yte,ids,edges=make_examples(h,pops,positions,test,train,True,max_het_step=a.max_step,return_edges=True)
    ml=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=70,
        l2_regularization=2,learning_rate=.045,random_state=a.seed)
    ml.fit(xtr,ytr)
    mlprob=ml.predict_proba(xte)[:,1]
    methods={'population-prior-chain':xte.reference_cis_population.to_numpy(),
             'population-prior-graph':xte.reference_cis_population.to_numpy(),
             'global-prior-graph':xte.reference_cis_global.to_numpy(),
             'ml-adjacent-chain':mlprob,'ml-graph-consensus':mlprob}
    rows=[]
    for donor in test:
        di=np.flatnonzero(ids==donor)
        hetcount=int(np.sum(h[donor].sum(axis=0)==1))
        if hetcount<2:continue
        phase_truth=h[donor,0,np.flatnonzero(h[donor].sum(axis=0)==1)].astype(int)
        true_relation=phase_truth[:-1]==phase_truth[1:]
        for method, probs in methods.items():
            lookup={tuple(map(int,edges[k])):float(probs[k]) for k in di}
            if method.endswith('-graph') or method=='ml-graph-consensus': bits=decode_consistent_phase(hetcount,lookup,a.max_step)
            else:bits=chained_phase(hetcount,lookup)
            predicted=(bits[:-1]==bits[1:])
            err=np.mean(predicted!=true_relation)
            rows.append(dict(donor=int(donor),population=str(pops[donor]),method=method,
                             n_het_pairs=len(true_relation),error=float(err)))
    detail=pd.DataFrame(rows)
    detail.to_csv(dest/'donor_comparison.csv',index=False)
    comparison=detail.groupby('method').apply(lambda g: pd.Series({
        'n_donors':len(g),'mean_donor_switch_disagreement':g.error.mean(),
        'pooled_adjacent_pair_disagreement':np.average(g.error,weights=g.n_het_pairs)}),include_groups=False).reset_index()
    comparison.to_csv(dest/'graph_comparison.csv',index=False)
    rng=np.random.default_rng(a.seed)
    wide=detail.pivot(index='donor',columns='method',values='error')
    dd=wide['ml-adjacent-chain']-wide['ml-graph-consensus']
    ml_vs_prior=wide['population-prior-graph']-wide['ml-graph-consensus']
    mlboot=np.array([rng.choice(ml_vs_prior.values,size=len(ml_vs_prior),replace=True).mean() for _ in range(600)])
    boot=np.array([rng.choice(dd.values,size=len(dd),replace=True).mean() for _ in range(600)])
    bonus={'mean_macro_improvement_adjacent_minus_graph':float(dd.mean()),
            'ci95_low':float(np.percentile(boot,2.5)),
            'ci95_high':float(np.percentile(boot,97.5)),
            'positive_favors_graph':True,
            'ml_graph_vs_prior_graph_mean_improvement':float(ml_vs_prior.mean()),
            'ml_graph_vs_prior_graph_ci95_low':float(np.percentile(mlboot,2.5)),
            'ml_graph_vs_prior_graph_ci95_high':float(np.percentile(mlboot,97.5))}
    (dest/'graph_report.json').write_text(json.dumps({'data_source':source,'n_train':len(train),
         'n_test':len(test),'maximum_heterozygote_skip':a.max_step,'graph_vs_adjacent':bonus},indent=2))
    print(comparison.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('Graph vs adjacent paired-donor bootstrap',bonus)

if __name__=='__main__':main()
