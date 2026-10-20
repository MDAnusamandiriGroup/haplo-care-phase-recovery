"""Exact finite-range binary-chain graph decoding of pairwise cis probabilities.
Assumes pairwise edge probabilities are independent pseudo-likelihood factors;
this assumption need not hold and may double-count LD evidence.
"""
from __future__ import annotations
import numpy as np

def decode_consistent_phase(n_hets, edge_probabilities, max_step=3):
    """MAP phase bits with first haplotype fixed to 0 (gauge invariance).
    edge_probabilities: {(i,j): P(cis)}, i<j, j-i<=max_step.
    Uses dynamic programming on last `max_step` bits; globally optimal under
    sum of pairwise log pseudo-likelihoods. Returns bits 0/1 per het.
    """
    if n_hets==0:return np.array([],dtype=int)
    if n_hets==1:return np.array([0],dtype=int)
    states={(0,):(0.,(0,))}
    for j in range(1,n_hets):
        updated={}
        for last,(score,path) in states.items():
            for bit in (0,1):
                increment=0.
                for i in range(max(0,j-max_step),j):
                    p=edge_probabilities.get((i,j),.5)
                    p=float(np.clip(p,1e-5,1-1e-5))
                    increment+=np.log(p if path[i]==bit else 1-p)
                newpath=path+(bit,)
                key=newpath[-max_step:]
                newscore=score+increment
                if key not in updated or newscore>updated[key][0]:
                    updated[key]=(newscore,newpath)
        states=updated
    return np.asarray(max(states.values(),key=lambda row:row[0])[1],dtype=int)

def chained_phase(n_hets,edge_probabilities):
    z=np.zeros(n_hets,dtype=int)
    for j in range(1,n_hets):
        z[j]=z[j-1] if edge_probabilities.get((j-1,j),.5)>=.5 else 1-z[j-1]
    return z
