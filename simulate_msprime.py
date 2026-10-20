"""Three-population coalescent pilot for HAPLO-CARE; not a validated human demography model."""
import argparse, json, time
from pathlib import Path
import numpy as np

def simulate(outdir, seed=20261018, people_per_pop=80, n_sites=80, seq_len=1_000_000):
    import msprime
    from importlib.metadata import version
    tic=time.perf_counter()
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    dem=msprime.Demography()
    for name,n in [('AFR',15000),('EUR',10000),('EAS',10000),('OOA',5000),('ANC',15000)]:
        dem.add_population(name=name,initial_size=n)
    dem.add_population_split(time=700,derived=['EUR','EAS'],ancestral='OOA')
    dem.add_population_split(time=2000,derived=['OOA','AFR'],ancestral='ANC')
    ts=msprime.sim_ancestry(samples={'AFR':people_per_pop,'EUR':people_per_pop,'EAS':people_per_pop},
        demography=dem,sequence_length=seq_len,recombination_rate=1.0e-8,
        ploidy=2,random_seed=seed)
    mts=msprime.sim_mutations(ts,rate=1.25e-8,random_seed=seed+1,model=msprime.BinaryMutationModel())
    snodes=mts.samples()
    if len(snodes)%2: raise ValueError('Expected diploid even number of haploid samples')
    n_people=len(snodes)//2
    popnames={i:p.name for i,p in enumerate(dem.populations)}
    groups=[]
    for k in range(n_people):
        p0=mts.node(int(snodes[2*k])).population
        p1=mts.node(int(snodes[2*k+1])).population
        if p0 != p1: raise ValueError('Diploid sample nodes not paired by population')
        groups.append(popnames[p0])
    bins={}
    # A regularly spaced, common-biallelic SNP panel; NOT full chromosome phasing.
    for variant in mts.variants(samples=snodes):
        gt=np.asarray(variant.genotypes)
        if (len(variant.alleles)!=2 or gt.min()<0 or gt.max()>1):continue
        maf=min(gt.mean(),1-gt.mean())
        if maf < .05:continue
        b=min(int(variant.site.position * n_sites/seq_len), n_sites-1)
        if b in bins:continue
        bins[b]=(int(variant.site.position)+1,gt.copy())
        if len(bins)==n_sites:break
    if len(bins)<20:raise ValueError(f'Only {len(bins)} eligible SNPs; increase seq_len')
    ordered=sorted(bins.values(),key=lambda x:x[0])
    sampids=[f'{groups[i]}_{i:04d}' for i in range(n_people)]
    vcf=outdir/'msprime_pilot_phased.vcf'
    with vcf.open('w') as f:
        f.write('##fileformat=VCFv4.2\n##source=HAPLO_CARE_msprime_pilot\n')
        f.write('##contig=<ID=sim1,length=%d>\n'%seq_len)
        f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(sampids)+'\n')
        for pos,gt in ordered:
            calls=[f'{gt[2*i]}|{gt[2*i+1]}' for i in range(n_people)]
            f.write('\t'.join(['sim1',str(pos),'.','A','C','.','PASS','.','GT']+calls)+'\n')
    meta=outdir/'population.tsv'
    meta.write_text('sample\tpopulation\n'+''.join(f'{s}\t{p}\n' for s,p in zip(sampids,groups)))
    info={'source':'msprime two-split demographic toy scenario; not empirically calibrated',
          'individuals':n_people,'populations':{x:groups.count(x) for x in sorted(set(groups))},
          'variants':len(ordered),'sequence_length':seq_len,'seed':seed,
          'msprime_version':version('msprime'),'elapsed_seconds':round(time.perf_counter()-tic,2)}
    (outdir/'simulation_manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
    return vcf,meta

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--outdir',default='results/msprime')
    pa.add_argument('--seed',type=int,default=20261018)
    pa.add_argument('--people-per-pop',type=int,default=80)
    pa.add_argument('--n-sites',type=int,default=80)
    a=pa.parse_args()
    simulate(a.outdir,a.seed,a.people_per_pop,a.n_sites)
