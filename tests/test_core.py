import unittest
import numpy as np
from src.haplocare import synthetic_haplotypes, split_individuals,make_examples,_pair_metrics

class PilotTests(unittest.TestCase):
    def test_cis_trans_symmetry(self):
        hap=np.array([[[0,0],[1,1]],[[0,1],[1,0]]],dtype=np.int8)
        pop=np.array(['A','A']);pos=np.array([10,20]);
        x,y,ids=make_examples(hap,pop,pos,[0,1],[0,1],True)
        np.testing.assert_array_equal(y,np.array([1,0]))
        flip=hap[:,::-1,:]
        _,yf,_=make_examples(flip,pop,pos,[0,1],[0,1],True)
        np.testing.assert_array_equal(y,yf)
    def test_training_test_individual_disjoint(self):
        h,p,pos,sids=synthetic_haplotypes(100,32,seed=99)
        train,test=split_individuals(p,seed=1)
        self.assertEqual(len(set(train)&set(test)),0)
    def test_cis_prior_bounded(self):
        r,n=_pair_metrics(np.zeros((4,2),dtype=np.int8),0,1)
        self.assertGreater(r[0],0);self.assertLess(r[0],1)


from src.graph import decode_consistent_phase,chained_phase

class GraphTests(unittest.TestCase):
    def test_consistent_chain(self):
        edges={(0,1):.9,(1,2):.1,(0,2):.1}
        got=decode_consistent_phase(3,edges)
        self.assertTrue(np.array_equal(got,[0,0,1]))
        self.assertTrue(np.array_equal(chained_phase(3,edges),[0,0,1]))
    def test_discordant_graph_returns_valid_path(self):
        edges={(0,1):.8,(1,2):.8,(0,2):.01,(2,3):.9}
        pred=decode_consistent_phase(4,edges)
        self.assertEqual(len(pred),4)
        self.assertEqual(pred[0],0)
        self.assertTrue(set(pred).issubset({0,1}))

    def test_dynamic_program_matches_brute_force(self):
        import itertools
        from math import log
        rng=np.random.default_rng(11)
        n=7; k=3
        edges={(i,j):float(rng.uniform(.05,.95)) for i in range(n) for j in range(i+1,min(n,i+k+1))}
        actual=decode_consistent_phase(n,edges,k)
        def score(z):
            return sum(log(p if z[i]==z[j] else 1-p) for (i,j),p in edges.items())
        allz=[(0,)+rest for rest in itertools.product([0,1],repeat=n-1)]
        best=max(score(z) for z in allz)
        self.assertAlmostEqual(score(actual),best,places=10)

if __name__=='__main__':unittest.main()
