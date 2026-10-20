"""HAPLO-CARE: reference-informed pairwise haplotype phase proof of concept.
All code works offline with numpy, pandas, scipy, scikit-learn and matplotlib.
This is *not* a chromosome-scale phasing algorithm.
"""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['reference_cis_global','reference_cis_population','reference_logodds_global',
            'reference_logodds_population','reference_r2_global','reference_r2_population',
            'maf_i_global','maf_j_global','marker_distance','heterozygous_between',
            'homozygous_alt_between','homozygous_ref_between','local_alt_fraction',
            'ref_support_population','ref_support_global']

def synthetic_haplotypes(n_people=1000, n_sites=72, n_pops=4, n_founders=20, seed=20261018):
    """Generate correlated binary haplotypes via population-specific founder mosaics.
    Simplified simulator, NOT msprime or validated evolutionary genetics.
    """
    rng = np.random.default_rng(seed)
    root_maf = rng.uniform(.12,.48,n_sites)
    pop_founders = np.zeros((n_pops,n_founders,n_sites),dtype=np.int8)
    for p in range(n_pops):
        drift = rng.normal(0,.13,n_sites)
        freq = np.clip(root_maf + drift,.025,.975)
        # correlated founder alleles: copy from 6 historical templates, with rare mutations
        templates=(rng.random((6,n_sites))<freq).astype(np.int8)
        for f in range(n_founders):
            t=int(rng.integers(6))
            for j in range(n_sites):
                if j and rng.random()<.025: t=int(rng.integers(6))
                val=templates[t,j]
                if rng.random()<.007: val=1-val
                pop_founders[p,f,j]=val
    pops=np.tile(np.arange(n_pops), int(np.ceil(n_people/n_pops)))[:n_people]
    rng.shuffle(pops)
    hap=np.empty((n_people,2,n_sites),dtype=np.int8)
    for n in range(n_people):
        founders=pop_founders[pops[n]]
        for h in range(2):
            founder=int(rng.integers(n_founders))
            for j in range(n_sites):
                if j and rng.random()<.018: founder=int(rng.integers(n_founders))
                hap[n,h,j]=founders[founder,j]
    return hap,np.asarray([f'POP{x}' for x in pops]),np.arange(n_sites)*1200, np.array([f'SIM{n:05d}' for n in range(n_people)])

def read_phased_vcf(path, max_samples=1000,max_variants=100,populations=None):
    """Read a SMALL phased, biallelic SNP VCF slice in memory.
    Real cohort phase labels may be statistical estimates, not molecular ground truth.
    """
    opener=gzip.open if str(path).endswith('.gz') else open
    sid=None; variants=[]; sites=[]; positions=[]
    with opener(path,'rt') as h:
        for line in h:
            if line.startswith('##'):continue
            if line.startswith('#CHROM'):
                sid=np.array(line.rstrip('\n').split('\t')[9:9+max_samples]); continue
            if sid is None: raise ValueError('No #CHROM header in VCF')
            fields=line.rstrip('\n').split('\t')
            if len(fields)<9+len(sid):raise ValueError('Insufficient sample columns')
            chrom,pos,_,ref,alt,_,_,_,fmt=fields[:9]
            if len(ref)!=1 or len(alt)!=1 or alt not in 'ACGT' or ref not in 'ACGT':continue
            fmtfields=fmt.split(':')
            if 'GT' not in fmtfields:continue
            gt_idx=fmtfields.index('GT')
            alleles=[];valid=True
            for sample in fields[9:9+len(sid)]:
                part=sample.split(':')
                if gt_idx>=len(part):valid=False;break
                gt=part[gt_idx]
                if len(gt)!=3 or gt[1]!='|' or gt[0] not in '01' or gt[2] not in '01':
                    valid=False;break
                alleles.append((int(gt[0]),int(gt[2])))
            if not valid:continue
            arr=np.asarray(alleles,dtype=np.int8)
            af=arr.mean()
            if not .03<=af<=.97:continue
            sites.append(arr)
            positions.append(int(pos))
            variants.append(f'{chrom}:{pos}:{ref}:{alt}')
            if len(sites)>=max_variants:break
    if len(sites)<8:raise ValueError('Too few fully phased, common, biallelic SNPs. Use a pre-filtered regional VCF.')
    haps=np.stack(sites,axis=2)
    popmap={}
    if populations:
        pdf=pd.read_csv(populations,sep='\t',dtype=str)
        if not {'sample','population'}.issubset(pdf.columns):
            raise ValueError('Population TSV needs sample and population columns')
        popmap=dict(zip(pdf['sample'],pdf['population']))
    pops=np.array([popmap.get(s,'ALL') for s in sid])
    return haps,pops,np.asarray(positions),sid,np.asarray(variants)

def split_individuals(pops, test_fraction=.25, seed=7):
    from sklearn.model_selection import train_test_split
    all_idx=np.arange(len(pops))
    counts=pd.Series(pops).value_counts()
    strat=pops if min(counts)>=2 else None
    a,b=train_test_split(all_idx,test_size=test_fraction,random_state=seed,stratify=strat)
    return np.sort(a),np.sort(b)

def _refstats(hap,pop,indices):
    hapref=hap[indices]
    groups=pop[indices]
    global_ref=hapref.reshape(-1,hap.shape[2])
    by_pop={p:hapref[groups==p].reshape(-1,hap.shape[2]) for p in np.unique(groups)}
    return global_ref,by_pop

def _pair_metrics(haps,i,j):
    if len(haps)==0: return np.array([.5,0.,.5,.5,0.]),0
    aa=haps[:,i];bb=haps[:,j]
    counts=np.bincount((aa*2+bb).astype(int),minlength=4).astype(float)+.5
    probs=counts/counts.sum();p00,p01,p10,p11=probs
    pcis=(p00*p11)/(p00*p11+p01*p10)
    pi=p10+p11;pj=p01+p11
    r=(p11-pi*pj)/np.sqrt(max(pi*(1-pi)*pj*(1-pj),1e-9))
    return np.array([pcis, np.clip(r*r,0,1),min(pi,1-pi),min(pj,1-pj),np.log(pcis/(1-pcis))]),len(haps)

def make_examples(hap,pop,positions,query_indices,reference_indices,return_id=False, max_het_step=1, return_edges=False):
    ref,group_ref=_refstats(hap,pop,reference_indices)
    genotypes=hap.sum(axis=1)
    records=[];ys=[];identifiers=[];edge_ordinals=[]
    # cache reference statistics once per site pair, population; no truth from the query used
    paircache={}
    for q in query_indices:
        hets=np.flatnonzero(genotypes[q]==1)
        for t in range(len(hets)):
          for step in range(1,min(max_het_step,len(hets)-t-1)+1):
            i,j=int(hets[t]),int(hets[t+step])
            p=pop[q]
            key=(p,int(i),int(j))
            if key not in paircache:
                g,ng=_pair_metrics(ref,i,j)
                pref=group_ref.get(p,ref)
                if len(pref)<30: pref=ref
                ps,ns=_pair_metrics(pref,i,j)
                paircache[key]=(g,ng,ps,ns)
            g,ng,ps,ns=paircache[key]
            mid=genotypes[q,i+1:j]
            left=max(0,i-5);right=min(hap.shape[2],j+6)
            local=genotypes[q,left:right]
            row=[g[0],ps[0],g[4],ps[4],g[1],ps[1],g[2],g[3],
                 float((positions[j]-positions[i])/10000), float(np.sum(mid==1)),
                 float(np.sum(mid==2)),float(np.sum(mid==0)),float(np.mean(local)/2),
                 np.log1p(ns),np.log1p(ng)]
            # y invariant to exchange of maternal/paternal labels
            y=int(hap[q,0,i]==hap[q,0,j])
            records.append(row);ys.append(y);identifiers.append(int(q));edge_ordinals.append((t,t+step))
    frame=pd.DataFrame(records,columns=FEATURES)
    if return_id and return_edges:return frame,np.asarray(ys,dtype=np.int8),np.asarray(identifiers),np.asarray(edge_ordinals,dtype=int).reshape(-1,2)
    if return_id:return frame,np.asarray(ys,dtype=np.int8),np.asarray(identifiers)
    return frame,np.asarray(ys,dtype=np.int8)

def crossfit_train(hap,pop,positions,train_idx,n_folds=4,seed=8,max_het_step=1):
    from sklearn.model_selection import KFold
    folds=KFold(n_splits=n_folds,shuffle=True,random_state=seed)
    frames=[];ys=[];ids=[]
    for tr_pos,q_pos in folds.split(train_idx):
        x,y,ident=make_examples(hap,pop,positions,train_idx[q_pos],train_idx[tr_pos],True,max_het_step=max_het_step)
        frames.append(x);ys.append(y);ids.append(ident)
    return pd.concat(frames,ignore_index=True),np.concatenate(ys),np.concatenate(ids)

def reliability(y,p,bins=10):
    df=pd.DataFrame({'truth':y,'probability':p})
    df['bin']=np.minimum((df.probability*bins).astype(int),bins-1)
    return df.groupby('bin',as_index=False).agg(n=('truth','size'),mean_pred=('probability','mean'),observed=('truth','mean'))

def metric_summary(y,p):
    from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss,log_loss
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return dict(n_pairs=int(len(y)),accuracy=float(accuracy_score(y,p>=.5)),
                auc=float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,
                brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))

def phase_switch_error(y,p,donor_idx):
    """Adjacent-heterozygote pair disagreement rate. Equivalent to SER *only*
    for a full local phase assignment at ordered heterozygotes, not direct SHAPEIT comparison.
    """
    per_donor=[]
    for d in np.unique(donor_idx):
        mask=donor_idx==d
        per_donor.append(float(np.mean((p[mask]>=.5)!=y[mask])))
    return np.array(per_donor)
