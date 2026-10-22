#!/usr/bin/env python3
"""Local anomalous amplitudes, periodic bond phases and LDOS threshold tests."""
from pathlib import Path
import os
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from me_postprocess import KB_EV
from me_postprocess.backend import Backend
from me_postprocess.local_profiles import (onsite_and_bonds,correlation_blocks,
    uniform_bulk_blocks,shell_average,threshold_half_gap)


class Archive(dict):
    @property
    def files(self):
        return list(self)


class ProfileTests(unittest.TestCase):
    def test_bcc_periodic_bonds(self):
        shape=np.array([4,4,4]);cells=np.indices(shape).reshape(3,-1).T[1:]
        left,right,images,steps=onsite_and_bonds(cells,shape)
        self.assertEqual(len(left),248)
        np.testing.assert_array_equal(cells[right]+images*shape-cells[left],steps)
        lattice=(np.ones((3,3))-2*np.eye(3))/2
        np.testing.assert_allclose(np.linalg.norm(steps @ lattice.T,axis=1),np.sqrt(3)/2)

    def test_shell_counts_and_ranges(self):
        r,n,mean,low,high=shell_average([1,1+1e-9,2],np.array([[1,2],[3,4],[8,9]]))
        np.testing.assert_array_equal(n,[2,1])
        np.testing.assert_allclose(mean,[[2,3],[8,9]])
        np.testing.assert_allclose(low,[[1,2],[8,9]])
        np.testing.assert_allclose(high,[[3,4],[8,9]])

    def test_threshold_definition_and_zero_background(self):
        e=np.linspace(-.004,.004,401)
        half,onsets=threshold_half_gap(e,np.abs(e),.1,.003)
        np.testing.assert_allclose(half,[.0003],atol=1e-15)
        np.testing.assert_allclose(onsets,[[.0003,.0003]],atol=1e-15)
        half,_=threshold_half_gap(e,np.ones_like(e),.1,.003)
        np.testing.assert_allclose(half,[0.],atol=1e-15)
        with self.assertRaises(ValueError):
            threshold_half_gap(e,np.ones_like(e),0.)

    def test_uniform_chain_site_bond_and_boundary_phase(self):
        backend=Backend('cuda' if os.environ.get('ME_TEST_GPU')=='1' else 'cpu',
                        int(os.environ.get('ME_TEST_GPU_DEVICE','0')))
        q=np.arange(4)/4
        k=(q[:,None]+np.arange(2)[None,:])/2
        xi=-.003+.004*np.cos(2*np.pi*k)
        u=np.exp(2j*np.pi*k[:,None,:]*np.arange(2)[None,:,None])/np.sqrt(2)
        delta=.002;t=2.
        h=np.zeros((4,4,4),complex)
        for b in range(4):
            h[b,:2,:2]=np.diag(xi[b]);h[b,2:,2:]=-np.diag(xi[b])
            h[b,:2,2:]=h[b,2:,:2]=delta*np.eye(2)
        twists=np.column_stack((q,np.zeros((4,2))))
        data=Archive(projected_bdg_2_ev=h,twists_2=twists)
        basis=Archive(vectors_2=u,twists_2=twists)
        left=np.array([0,0,1,0]);right=np.array([0,1,0,1])
        images=np.array([[0,0,0],[0,0,0],[1,0,0],[-1,0,0]])
        blocks=correlation_blocks(data,basis,left,right,images,t,4,norb=1,backend=backend,chunk=2)
        energy=np.sqrt(xi.ravel()**2+delta**2)
        f=delta/(2*energy)*np.tanh(energy/(2*KB_EV*t))
        f0=f.mean();f1=np.mean(np.exp(-2j*np.pi*k.ravel())*f)
        np.testing.assert_allclose(blocks[:,0,0],[f0,f1,f1,f1],atol=1e-13)
        primitive=Archive(normal_density=np.zeros(1),total_points=np.array(8),
            xi_1=xi.ravel()[:,None],vectors_1=np.ones((8,1,1),complex),
            twists_1=np.column_stack((k.ravel(),np.zeros((8,2)))))
        reference=uniform_bulk_blocks(primitive,delta,[[0,0,0],[1,0,0],[-1,0,0]],t,backend,chunk=3)
        np.testing.assert_allclose(reference[:,0,0],[f0,f1,f1],atol=1e-13)


if __name__=='__main__':
    unittest.main()
