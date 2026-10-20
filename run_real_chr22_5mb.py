"""5-Mb exploratory pilot, 1000 Genomes chr22 GRCh38 phased panel.

IMPORTANT: compares agreement with statistically inferred published phase,
not independent molecular truth. Not validation against SHAPEIT5/Beagle.

Requires: numpy pandas scipy scikit-learn; standard-library gzip, zipfile.
No bcftools, pysam or matplotlib needed on Windows.
"""
from __future__ import annotations
import argparse, gzip, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from src.haplocare import crossfit_train,make_examples
from src.graph import decode_consistent_phase

GRPS={
    'AFR': ['YRI','LWK','GWD','MSL','ESN'],
    'EUR': ['GBR','FIN','CEU','TSI','IBS'],
    'EAS': ['CHB','CHS','JPT','CDX','KHV'],
}


def metadata_map(path: Path):
    # Pedigree/population metadata has SampleID Population FatherID MotherID.
    df=pd.read_csv(path,sep=r'\s+',dtype=str,comment='#')
    required={'SampleID','Population','FatherID','MotherID'}
    if not required.issubset(df.columns):
        raise ValueError(f'Metadata requires columns {required}. Found {list(df.columns)}')
    df=df[(df.FatherID=='0')&(df.MotherID=='0')].copy()
    df['group']=df['Population'].map({p:g for g,ps in GRPS.items() for p in ps})
    df=df[df['group'].notna()].copy()
    # Additional exclusion of individuals from the same pedigree when available.
    if 'FamilyID' in df.columns:
        family=df['FamilyID'].fillna('')
        keep=[];seen=set()
        for i,(f,s) in enumerate(zip(family,df.SampleID)):
            key=f if f and f not in ('0','NA','.') else s
            if key in seen: continue
            seen.add(key);keep.append(i)
        df=df.iloc[keep]
    return df.set_index('SampleID')['group'].to_dict()


def select_vcf_samples(vcf_header_ids, sample_popmap, n_per_group=60, seed=20261018):
    # Fixed selection before reading variant records; deterministic and independent of allele content.
    rng=np.random.default_rng(seed)
    selected=[];groups=[]
    for group in ('AFR','EUR','EAS'):
        candidates=np.array([s for s in vcf_header_ids if sample_popmap.get(s)==group],dtype=object)
        if len(candidates)<n_per_group:
            raise ValueError(f'Not enough unrelated {group} samples: {len(candidates)}; requested {n_per_group}')
        choices=np.sort(rng.choice(candidates,size=n_per_group,replace=False))
        selected.extend(choices.tolist());groups.extend([group]*len(choices))
    idx_by_id={s:i for i,s in enumerate(vcf_header_ids)}
    return np.asarray(selected),np.asarray(groups),np.array([idx_by_id[s] for s in selected],dtype=int)


def scan_5mb_vcf(vcf, metadata, start=30000000, region_bp=5000000,
                 n_per_group=60, bin_bp=5000, seed=20261018):
    """Stream compressed VCF, retain <=1 fully phased common SNP per bin.
    The fixed 5 Mb span is partitioned into five non-overlapping 1-Mb windows.
    AF filter uses ONLY training individuals, not held-out test individuals.
    """
    end=start+region_bp-1
    data=[[] for _ in range(5)]; pos=[[] for _ in range(5)]
    selected=None; pops=None; columns=None
    split_rng=seed
    start_t=time.time(); scanned=0
    header_ids=None
    with gzip.open(vcf,'rt',encoding='utf-8',errors='replace') as handle:
        for line in handle:
            if line.startswith('##'): continue
            if line.startswith('#CHROM'):
                header_ids=line.rstrip('\r\n').split('\t')[9:]
                selected,pops,columns=select_vcf_samples(header_ids,metadata_map(metadata),n_per_group,seed)
                indices=np.arange(len(selected))
                tr,te=train_test_split(indices,test_size=.25,random_state=seed,stratify=pops)
                tr=np.sort(tr);te=np.sort(te)
                print(f'Selected: {len(selected)} (AFR/EUR/EAS each {n_per_group}), training={len(tr)} heldout={len(te)}',flush=True)
                continue
            if line.startswith('#'):continue
            if header_ids is None:raise RuntimeError('VCF header not found')
            fields=line.rstrip('\r\n').split('\t')
            chrom=fields[0]
            if chrom not in ('22','chr22'):
                if len(data[0])==0:continue
                break
            p=int(fields[1])
            if p<start:continue
            if p>end:break
            scanned+=1
            window=(p-start)//1000000
            if window<0 or window>=5:continue
            # Only one site per ~5 kb interval, providing up to 200 markers/Mb.
            bin_num=(p-(start+window*1000000))//bin_bp
            if pos[window] and ((pos[window][-1]-(start+window*1000000))//bin_bp)==bin_num:continue
            ref,alt=fields[3],fields[4]
            if len(ref)!=1 or len(alt)!=1 or ref not in 'ACGT' or alt not in 'ACGT':continue
            fmt=fields[8].split(':')
            if 'GT' not in fmt:continue
            gi=fmt.index('GT')
            alleles=np.empty((len(selected),2),dtype=np.int8)
            valid=True
            for k,col in enumerate(columns):
                v=fields[9+col]
                gt=(v.split(':',gi+1)[gi] if gi else v.split(':',1)[0])
                if len(gt)!=3 or gt[1]!='|' or gt[0] not in '01' or gt[2] not in '01':
                    valid=False;break
                alleles[k]=[ord(gt[0])-48,ord(gt[2])-48]
            if not valid:continue
            af=alleles[tr].mean()
            if not (0.05<=af<=0.95):continue
            data[window].append(alleles)
            pos[window].append(p)
            if scanned%15000==0:print(f'Processed {scanned:,} regional VCF rows',flush=True)
    if selected is None:raise RuntimeError('No VCF #CHROM header found')
    windows=[]
    for i,(d,pp) in enumerate(zip(data,pos)):
        if len(d)<35:raise RuntimeError(f'Only {len(d)} usable SNPs in window {i+1}; lower bin_bp or change region')
        h=np.stack(d,axis=2)
        windows.append((h,np.asarray(pp,dtype=np.int64)))
        print(f'Window {i+1}: chr22:{start+i*1000000:,}-{start+(i+1)*1000000-1:,}; {len(pp)} SNPs',flush=True)
    print(f'Loaded {scanned:,} in-range VCF rows in {time.time()-start_t:.1f} seconds',flush=True)
    return windows,selected,pops,tr,te


def run_window(h,pops,positions,train,test,window_idx,seed,max_step=3):
    tic=time.time()
    xtr,ytr,_=crossfit_train(h,pops,positions,train,n_folds=4,seed=seed,max_het_step=max_step)
    xte,yte,ids,edges=make_examples(h,pops,positions,test,train,True,max_het_step=max_step,return_edges=True)
    if not len(ytr) or len(set(ytr))!=2 or not len(ids):raise RuntimeError(f'Insufficient training/evaluation pairs for window {window_idx}')
    ml=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=70,
        l2_regularization=2,learning_rate=.045,random_state=seed)
    ml.fit(xtr,ytr)
    probs=ml.predict_proba(xte)[:,1]
    prior=xte.reference_cis_population.to_numpy()
    rows=[]
    for donor in test:
        keep=np.flatnonzero(ids==donor)
        het=np.flatnonzero(h[donor].sum(axis=0)==1)
        if len(het)<2:continue
        truth=h[donor,0,het]
        rel=(truth[:-1]==truth[1:])
        res={'donor':int(donor),'population':str(pops[donor]),'window':int(window_idx),
             'n_pairs':int(len(rel))}
        for name,pv in (('reference_graph',prior),('ml_graph',probs)):
            lk={tuple(map(int,edges[k])):float(pv[k]) for k in keep}
            bits=decode_consistent_phase(len(het),lk,max_step)
            predicted=(bits[:-1]==bits[1:])
            res[name]=float(np.mean(predicted!=rel))
        res['gain']=res['reference_graph']-res['ml_graph']
        rows.append(res)
    print(f'Window {window_idx}: train edges {len(xtr):,}, test edges {len(xte):,}; '
          f'gain {np.mean([r["gain"] for r in rows]):+.5f} in {time.time()-tic:.1f}s',flush=True)
    return rows


def summarize(all_rows, sample_ids, outdir, settings):
    rows=pd.DataFrame(all_rows)
    if rows.empty:raise RuntimeError('No test donors could be evaluated')
    rows['sample_id']=rows.donor.map({i:str(s) for i,s in enumerate(sample_ids)})
    outdir.mkdir(exist_ok=True,parents=True)
    rows.to_csv(outdir/'donor_window_results.csv',index=False)
    # Equal donor weighting; windows contribute equally unless some missing.
    donor=rows.groupby(['donor','population'],as_index=False)[['reference_graph','ml_graph','gain']].mean()
    donor.to_csv(outdir/'donor_summary.csv',index=False)
    rng=np.random.default_rng(settings['seed'])
    v=donor.gain.to_numpy()
    boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)])
    summary={**settings,'n_evaluated':int(len(donor)),
             'n_total_windows':int(rows.window.nunique()),
             'mean_reference_error':float(donor.reference_graph.mean()),
             'mean_ml_error':float(donor.ml_graph.mean()),
             'mean_improvement':float(donor.gain.mean()),
             'bootstrap_ci95_lower':float(np.quantile(boot,.025)),
             'bootstrap_ci95_upper':float(np.quantile(boot,.975)),
             'per_window':rows.groupby('window')[['reference_graph','ml_graph','gain']].mean().reset_index().to_dict('records'),
             'per_population':donor.groupby('population')[['reference_graph','ml_graph','gain']].mean().reset_index().to_dict('records'),
             'truth_caveat':'Published 1000 Genomes phased haplotypes are inferred statistical phase labels, NOT independently verified truth.',
             'statistical_caveat':'Bootstrap over individuals is exploratory; population/LD and shared phasing pipeline may induce dependence.',
             'extraction_caveat':'Sparse common SNP marker subset across each 1Mb window; not analysis of every site in 5Mb.',
             'relatedness_caveat':'Both reported father and mother are 0; at most one per known family ID. Does not exclude all cryptic relatedness.'}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vcf',required=True,type=Path)
    p.add_argument('--panel',required=True,type=Path)
    p.add_argument('--output',default='results/real_chr22_5mb',type=Path)
    p.add_argument('--start',type=int,default=30000000)
    p.add_argument('--n-per-group',type=int,default=60)
    p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--bin-bp',type=int,default=5000)
    p.add_argument('--max-step',type=int,default=3)
    args=p.parse_args()
    if not args.vcf.is_file():raise FileNotFoundError(args.vcf)
    if not args.panel.is_file():raise FileNotFoundError(args.panel)
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'start':args.start,'region_bp':5000000,'n_per_group':args.n_per_group,'seed':args.seed,
              'bin_bp':args.bin_bp,'max_step':args.max_step,'vcf_filename':args.vcf.name,
              'panel_filename':args.panel.name}
    windows,samples,pops,train,test=scan_5mb_vcf(args.vcf,args.panel,args.start,5000000,args.n_per_group,args.bin_bp,args.seed)
    # Emit data audit, no underlying phased/genotype matrix in output ZIP.
    pd.DataFrame({'sample':samples,'population':pops,'split':['train' if i in set(train) else 'test' for i in range(len(samples))]}).to_csv(args.output/'sample_split.csv',index=False)
    all_rows=[]
    for wi,(h,positions) in enumerate(windows,1):
        all_rows.extend(run_window(h,pops,positions,train,test,wi,args.seed,args.max_step))
        pd.DataFrame(all_rows).to_csv(args.output/'partial_donor_window_results.csv',index=False)
    summarize(all_rows,samples,args.output,settings)
    (args.output/'DONE.txt').write_text('DONE: all five windows completed.\n')

if __name__=='__main__':main()
