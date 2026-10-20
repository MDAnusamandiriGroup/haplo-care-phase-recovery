#!/usr/bin/env python3
"""Run leakage-aware synthetic or prephased regional VCF pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from src.haplocare import (synthetic_haplotypes,read_phased_vcf,split_individuals,
    crossfit_train,make_examples,metric_summary,reliability,phase_switch_error)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vcf',default=None,help='OPTIONAL SMALL phased regional VCF/VCF.gz')
    ap.add_argument('--population-tsv',default=None)
    ap.add_argument('--samples',type=int,default=950)
    ap.add_argument('--variants',type=int,default=72)
    ap.add_argument('--seed',type=int,default=20261018)
    ap.add_argument('--output',default='results')
    args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    if args.vcf:
        hap,pop,pos,sid,variants=read_phased_vcf(args.vcf,max_samples=args.samples,
            max_variants=args.variants,populations=args.population_tsv)
        source=f'public/published prephased dataset: {Path(args.vcf).name} (not independent phasing truth)'
    else:
        hap,pop,pos,sid=synthetic_haplotypes(args.samples,args.variants,seed=args.seed)
        source='simplified synthetic founder-mosaic simulation (known phase truth)'
    train,test=split_individuals(pop,seed=args.seed)
    print('Data:',source,'individuals=',len(sid),'sites=',len(pos),'train=',len(train),'test=',len(test),flush=True)
    print('Constructing cross-fitted training features...',flush=True)
    xtr,ytr,idtr=crossfit_train(hap,pop,pos,train)
    print('Constructing held-out individual test features...',flush=True)
    xte,yte,idte=make_examples(hap,pop,pos,test,train,True)
    if not len(xte) or len(set(ytr))<2:raise RuntimeError('Insufficient pairs/classes. Use more donors or sites.')
    model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=70,
        l2_regularization=2.0,learning_rate=.045,random_state=args.seed)
    model.fit(xtr,ytr)
    result={}
    pred={
      'population-reference-prior':xte['reference_cis_population'].to_numpy(),
      'global-reference-prior':xte['reference_cis_global'].to_numpy(),
      'haplocare-ml-pilot':model.predict_proba(xte)[:,1]
    }
    for label,p in pred.items():
        m=metric_summary(yte,p)
        donor_ser=phase_switch_error(yte,p,idte)
        m['macro_mean_adjacent_het_pair_error']=float(np.mean(donor_ser))
        m['median_donor_adjacent_het_pair_error']=float(np.median(donor_ser))
        result[label]=m
        reliability(yte,p).to_csv(dest/f'calibration_{label}.csv',index=False)
    tbl=pd.DataFrame.from_dict(result,orient='index').reset_index(names='model')
    tbl.to_csv(dest/'comparison.csv',index=False)
    pd.DataFrame({'individual':idte,'truth_cis':yte,**pred}).to_csv(dest/'heldout_pair_predictions.csv',index=False)
    # Donor-level paired-bootstrap uncertainty for difference in errors relative to prior.
    rng=np.random.default_rng(args.seed)
    test_donors=np.unique(idte)
    unique_err={k:np.array([np.mean((v[idte==d]>=.5)!=yte[idte==d]) for d in test_donors]) for k,v in pred.items()}
    diff=unique_err['population-reference-prior']-unique_err['haplocare-ml-pilot']
    boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(600)])
    result['bootstrap_difference_population_prior_minus_ml_macro_error']={
        'point_estimate':float(diff.mean()),'ci95_low':float(np.percentile(boot,2.5)),
        'ci95_high':float(np.percentile(boot,97.5)),
        'unit':'absolute fraction; positive favors ML; donors resampled (not variant pairs)'}
    with (dest/'metrics.json').open('w') as f:json.dump({'data_source':source,'n_individuals':len(sid),
       'n_sites':len(pos),'n_train_individuals':len(train),'n_test_individuals':len(test),
       'n_train_pairs':len(ytr),'n_test_pairs':len(yte),'metrics':result},f,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    ax=axes[0]; ax.plot([0,1],[0,1],'--',alpha=.5,label='ideal')
    for label in pred:
        r=reliability(yte,pred[label]);ax.plot(r.mean_pred,r.observed,'o-',label=label,ms=4)
    ax.set(xlabel='Predicted cis probability',ylabel='Observed cis fraction',title='Held-out calibration',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=7)
    ax=axes[1];names=list(pred)
    vals=[result[k]['macro_mean_adjacent_het_pair_error'] for k in names]
    ax.barh(names,vals);ax.invert_yaxis();ax.set_xlabel('Mean donor-level pair error (lower is better)')
    ax.set_title('Held-out adjacent heterozygote pairs')
    fig.tight_layout();fig.savefig(dest/'pilot_performance.png',dpi=175);plt.close(fig)
    print(tbl.to_string(index=False,float_format=lambda a:f'{a:.4f}'))
    print('Donor-bootstrap difference in error (prior - ML):',result['bootstrap_difference_population_prior_minus_ml_macro_error'])
    print('Written:',str(dest.resolve()),flush=True)

if __name__=='__main__':main()
