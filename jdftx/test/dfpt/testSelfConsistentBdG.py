#!/usr/bin/env python3
"""Analytic gap, projected-channel variation, and exact vacancy deletion tests."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from me_postprocess import KB_EV
from me_postprocess.backend import Backend,host
from me_postprocess.selfconsistent import ProjectedSpectrum,supercell_matrix,solve_gap


class BdGTests(unittest.TestCase):
    def backend(self):
        return Backend('cuda' if os.environ.get('ME_TEST_GPU')=='1' else 'cpu',
                       int(os.environ.get('ME_TEST_GPU_DEVICE','0')))

    def test_one_level_gap_density_and_iteration(self):
        backend = self.backend();xp = backend.xp
        xi,delta,t = .002,.003,2.
        spectrum = ProjectedSpectrum(1,1,.02,t,backend)
        spectrum.add(xp.array([[xi]]),xp.ones((1,1,1),complex),[[0,0,0]])
        spectrum.finalize()
        energy = np.sqrt(xi**2+delta**2)
        f = delta/(2*energy)*np.tanh(energy/(2*KB_EV*t))
        result = spectrum.evaluate([delta],observables=True)
        np.testing.assert_allclose(result['anomalous'],[f],atol=1e-14)
        np.testing.assert_allclose(result['density'],[1-xi/energy*np.tanh(energy/(2*KB_EV*t))],atol=1e-14)
        attraction = delta/f
        for scale in (.5,1.5):
            solution = solve_gap(spectrum,[attraction],[scale*delta],t,tolerance_ev=1e-12)
            np.testing.assert_allclose(solution['delta_ev'],[delta],atol=1e-10)
            self.assertLess(solution['grand_potential_difference_ev'],0.)

    def test_projected_variational_gradient_and_restart(self):
        backend = self.backend();xp = backend.xp
        rng = np.random.default_rng(35)
        basis = np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
        spectrum = ProjectedSpectrum(3,2,.01,2.,backend)
        xi = np.array([[-.003,.002,.1],[-.003,.002,.1]])
        spectrum.add(xp.asarray(xi),xp.asarray(np.array([basis,basis.conj()])),[[.13,0,0],[-.13,0,0]])
        spectrum.finalize()
        self.assertEqual(list(spectrum.groups),[2])
        delta = np.array([.001,.002,.003]);u = np.array([.01,.02,.015])
        result = spectrum.evaluate(delta,observables=True)
        def omega(d):
            return np.sum(d*d/u)+spectrum.evaluate(d,observables=True)['electronic_grand_potential_ev']
        numerical=[]
        for a in range(3):
            shift=np.eye(3)[a]*1e-7
            numerical.append((omega(delta+shift)-omega(delta-shift))/(2e-7))
        np.testing.assert_allclose(numerical,2*(delta/u-result['anomalous']),atol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'basis.npz';spectrum.save_basis(path)
            loaded=ProjectedSpectrum.load_basis(path,backend)
            np.testing.assert_allclose(loaded.evaluate(delta)['anomalous'],result['anomalous'],atol=1e-14)

    def test_exact_supercell_folding_and_deleted_orbitals(self):
        backend=self.backend();xp=backend.xp
        class Chain:
            nw=1
            mu=0.
            def h(self,k):
                k=xp.asarray(k)
                return (-2*xp.cos(2*np.pi*k[...,0]))[...,None,None].astype(complex)
        model=Chain();model.xp=xp
        for twist in (.0,.137,.5):
            q=np.array([[twist,0,0]])
            full,_=supercell_matrix(model,[4,1,1],q,(),1.)
            expected=np.sort(-2*np.cos(2*np.pi*(twist+np.arange(4))/4))
            np.testing.assert_allclose(host(xp.linalg.eigvalsh(full))[0],expected,atol=1e-13)
            vacancy,sites=supercell_matrix(model,[4,1,1],q,(0,),1.)
            np.testing.assert_array_equal(sites,[1,2,3])
            np.testing.assert_allclose(host(vacancy)[0],[[0,-1,0],[-1,0,-1],[0,-1,0]],atol=1e-13)

    def test_pristine_supercell_anomalous_density_normalization(self):
        backend=self.backend();xp=backend.xp
        class Chain:
            nw=1;mu=0.
            def h(self,k):
                k=xp.asarray(k)
                return (-2*xp.cos(2*np.pi*k[...,0]))[...,None,None].astype(complex)
        model=Chain();model.xp=xp
        q=np.array([[.137,0,0]])
        matrix,_=supercell_matrix(model,[4,1,1],q,(),1.)
        xi,vectors=xp.linalg.eigh(matrix)
        large=ProjectedSpectrum(4,1,3.,2.,backend)
        large.add(xi,vectors,q);large.finalize()
        k=np.zeros((4,3));k[:,0]=(.137+np.arange(4))/4
        small=ProjectedSpectrum(1,4,3.,2.,backend)
        small.add(model.h(k)[:,0,:].real,xp.ones((4,1,1),complex),k);small.finalize()
        a=large.evaluate(np.full(4,.003),observables=True)
        b=small.evaluate([.003],observables=True)
        np.testing.assert_allclose(a['anomalous'],np.repeat(b['anomalous'],4),atol=1e-13)
        np.testing.assert_allclose(a['density'],np.repeat(b['density'],4),atol=1e-13)


if __name__=='__main__':
    unittest.main()
