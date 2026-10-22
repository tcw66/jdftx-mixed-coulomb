#!/usr/bin/env python3
"""Analytic spectra, signed-frequency equations, gauge and native-layout checks.

Run: python jdftx/test/dfpt/testMigdalEliashberg.py
No JDFTx executable or external calculation files are required.
"""
from pathlib import Path
import os
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import numpy as np
from scipy.linalg import eigvals

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from me_postprocess import HARTREE_EV, KB_EV
from me_postprocess.wannier import WannierModel, mesh_points, indices
from me_postprocess.spectral import moments, histogram, calculate
from me_postprocess.eliashberg import IsotropicME, low_frequency_fit
from me_postprocess.bdg import (isotropic_singlet, export_arrays, negative_partners,
                               coherent_reduce, nambu_normal, charge_conjugate)
from me_postprocess.backend import Backend, host


class BackendChecks(unittest.TestCase):
    def test_cpu_and_invalid_configuration(self):
        self.assertIs(Backend('cpu').xp, np)
        with self.assertRaisesRegex(ValueError, 'backend must be'):
            Backend('invalid')
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            Backend('cuda', -1)

    def test_optional_cuda_missing(self):
        with mock.patch.dict(sys.modules, {'cupy': None}):
            with self.assertRaisesRegex(ValueError, 'CUDA backend unavailable'):
                Backend('cuda')
            fallback = Backend('auto')
            self.assertEqual(fallback.name, 'cpu')
            self.assertIn('fallback_reason', fallback.info)


class MEChecks(unittest.TestCase):
    def setUp(self):
        self.omega = np.array([0.02])
        self.weight = np.array([0.01])  # Einstein mode lambda=1
        self.solver = IsotropicME(self.omega, self.weight, 0.1, 0.4, 0.2)

    def test_einstein_moments_and_spectral_area(self):
        m = moments(self.omega, self.weight)
        self.assertAlmostEqual(m['lambda_total'], 1.)
        self.assertAlmostEqual(m['omega_log_ev'], 0.02)
        self.assertAlmostEqual(m['omega_2_ev'], 0.02)
        hist = histogram(self.omega, self.weight)
        self.assertAlmostEqual(hist[-1, 2], 1.)
        self.assertAlmostEqual(self.solver.kernel(np.array([0.]))[0], 1.)

    def test_positive_frequency_reduction_against_signed_sum(self):
        t = 12.
        wn, pi_t, kz, pairing, zn = self.solver.matrices(t)
        signed = np.r_[-wn[::-1], wn]
        kernel = self.solver.kernel(signed[:, None]-signed[None, :])
        # Test an arbitrary nonzero even gap against the full +/- sum.
        delta = 0.003/(1+(wn/0.04)**2)
        d_signed = np.r_[delta[::-1], delta]
        denom = np.sqrt(signed**2+d_signed**2)
        z_finite = 1+pi_t/signed*(kernel @ (signed/denom))
        z_finite_normal = 1+pi_t/signed*(kernel @ np.sign(signed))
        z_corrected = z_finite[-len(wn):]+zn-z_finite_normal[-len(wn):]
        np.testing.assert_allclose(z_corrected, zn+pi_t/wn*(kz @ (wn/np.sqrt(wn**2+delta**2)-1)), atol=1e-14)
        pair_signed = kernel-self.solver.mu_star*(np.abs(signed)[None, :] < 0.2)
        rhs = pi_t*pair_signed @ (d_signed/denom)
        np.testing.assert_allclose(rhs[-len(wn):], pi_t*pairing @ (delta/np.sqrt(wn**2+delta**2)), atol=1e-16)
        # Independent signed-frequency eigenproblem has the same even solution.
        z_signed = np.r_[zn[::-1], zn]
        matrix = pi_t*pair_signed/(z_signed[:, None]*np.abs(signed)[None, :])
        rho = max(eigvals(matrix).real)
        self.assertAlmostEqual(rho, self.solver.eigenvalue(t), places=9)

    def test_nonlinear_equations_and_tc(self):
        tc = self.solver.critical_temperature(2., 60.)
        self.assertGreater(tc, 10.)
        self.assertLess(tc, 40.)
        self.assertAlmostEqual(self.solver.eigenvalue(tc), 1., delta=1e-3)
        solution = self.solver.solve(5.)
        wn, pi_t, kz, pairing, zn = self.solver.matrices(5.)
        delta = solution['delta_ev']
        norm = np.sqrt(wn**2+delta**2)
        np.testing.assert_allclose(solution['z']*delta, pi_t*(pairing @ (delta/norm)), atol=3e-10)
        self.assertGreater(low_frequency_fit(solution)['z0'], 1.)
        self.assertLess(float(solution['free_energy_per_dos_ev2']), 0.)
        normal = self.solver.solve(60.)
        np.testing.assert_array_equal(normal['delta_ev'], 0.)

    def test_energy_temperature_scaling(self):
        a = self.solver.solve(5.)
        other = IsotropicME(2*self.omega, 2*self.weight, 0.1, 0.8, 0.4)
        b = other.solve(10.)
        np.testing.assert_allclose(b['z'], a['z'], atol=2e-7)
        np.testing.assert_allclose(b['delta_ev'], 2*a['delta_ev'], atol=2e-9)


class BdGChecks(unittest.TestCase):
    def setUp(self):
        self.shape = np.array([4, 1, 1])
        self.k = mesh_points(self.shape)
        r = np.array([[1., 0.3], [-0.2, 0.5]])
        phase = np.exp(2j*np.pi*self.k[:, 0])
        self.h = np.diag([0.1, -0.2])[None]+phase[:, None, None]*r+phase.conj()[:, None, None]*r.T
        self.partner = negative_partners(self.k)

    def test_singlet_poles_and_fourier_round_trip(self):
        bdg, s = isotropic_singlet(self.h, self.k, 0.2, 2., 0.006)
        output, checks = export_arrays(bdg, self.k, self.shape)
        self.assertLess(max(checks.values()), 1e-14)
        energy = (np.linalg.eigvalsh(self.h)-0.2)/2
        expected = np.repeat(np.sqrt(energy**2+0.003**2), 2, axis=-1)
        np.testing.assert_allclose(np.linalg.eigvalsh(bdg)[:, 4:], np.sort(expected), atol=1e-14)
        phase = np.exp(2j*np.pi*self.k @ output['cells'].T)
        np.testing.assert_allclose(np.einsum('kr,rab->kab', phase, output['hopping_ev']), output['h_qp_ev'], atol=1e-14)
        np.testing.assert_allclose(np.einsum('kr,rab->kab', phase, output['pairing_ev']), output['delta_qp_ev'], atol=1e-14)
        self.assertAlmostEqual(np.max(np.abs(output['pairing_ev'][1:])), 0.)

    def test_full_metric_and_k_dependent_gauge_covariance(self):
        base, _ = isotropic_singlet(self.h, self.k, 0.2, 1., 0.006)
        n = base.shape[-1]//2
        metric = np.broadcast_to(2*np.eye(2*n), base.shape).copy()
        metric[:, :n, n:] = 0.1*np.eye(n)
        metric[:, n:, :n] = 0.1*np.eye(n)
        reduced, _, _ = coherent_reduce(base, metric, self.partner)
        rng = np.random.default_rng(721)
        u = np.array([np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0] for _ in self.k])
        w = np.zeros_like(base)
        w[:, :n, :n], w[:, n:, n:] = u, u[self.partner].conj()
        dagger = w.conj().swapaxes(-1, -2)
        rotated, _, _ = coherent_reduce(w @ base @ dagger, w @ metric @ dagger, self.partner)
        np.testing.assert_allclose(rotated, w @ reduced @ dagger, atol=1e-14)
        # The anomalous block transforms with U^T(-k), not U^dagger(-k).
        np.testing.assert_allclose((w @ base @ dagger)[:, :n, n:],
                                  u @ base[:, :n, n:] @ u[self.partner].swapaxes(-1, -2), atol=1e-14)

    def test_invalid_metric_and_pairing_rejected(self):
        h = nambu_normal(self.h, 0.2, self.partner)
        metric = np.broadcast_to(-np.eye(4), h.shape)
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            coherent_reduce(h, metric, self.partner)
        h[:, :2, 2:] = 0.001*np.eye(2)
        h[:, 2:, :2] = 0.001*np.eye(2)
        with self.assertRaisesRegex(ValueError, 'Nambu constraints'):
            coherent_reduce(h, -metric, self.partner)

    def test_matrix_cli_export(self):
        expected, _ = isotropic_singlet(self.h, self.k, 0.2, 2., 0.006)
        static, _ = isotropic_singlet(self.h, self.k, 0.2, 1., 0.006)
        normal = nambu_normal(np.kron(self.h, np.eye(2)), 0.2, self.partner)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.npz', Path(directory)/'bdg.npz'
            np.savez(source, kpoints=self.k, h_ev=np.kron(self.h, np.eye(2)),
                     mu_ev=0.2, sigma0_ev=static-normal,
                     metric=np.broadcast_to(2*np.eye(8), static.shape))
            script = Path(__file__).resolve().parents[2]/'scripts/migdalEliashberg'
            result = subprocess.run([sys.executable, str(script), 'matrix-bdg',
                                     str(source), str(output), '--mesh', '4', '1', '1'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(output) as arrays:
                np.testing.assert_allclose(arrays['bdg_ev'], expected, atol=1e-14)
                self.assertEqual(arrays['hopping_ev'].shape, (4,4,4))


class NativeChecks(unittest.TestCase):
    def test_binary_layout_cell_weights_and_zero_point_factor(self):
        with tempfile.TemporaryDirectory() as directory:
            p = str(Path(directory)/'w.')
            cells = np.array([[-1,0,0], [0,0,0], [1,0,0]])
            fold = np.array([2,1,1])
            for name in ('mlwfCellMap', 'mlwfCellMapPh'):
                np.savetxt(p+name, cells)
            np.savetxt(p+'phononCellMap', [[0,0,0]])
            Path(p+'phononBasis').write_text('X 0 1 0 0 1\nX 0 0 1 0 1\nX 0 0 0 1 1\n')
            h = np.array([np.diag([0.1,0.2]), np.diag([0.3,0.4])])
            h.swapaxes(-1,-2).tofile(p+'mlwfH')
            weights = np.ones((3,2,2))
            weights[0] *= 0.5
            weights[2] *= 0.5
            weights.swapaxes(-1,-2).tofile(p+'mlwfCellWeights')
            wph = np.array([[[0.2,0.3]], [[1.,1.]], [[0.8,0.7]]])
            wph.swapaxes(-1,-2).tofile(p+'mlwfCellWeightsPh')
            omega_ha = np.array([0.001,0.002,0.003])
            np.diag(omega_ha**2)[None].tofile(p+'phononOmegaSq')
            raw = (np.arange(2*2*3*2*2).reshape(2,2,3,2,2)+1.)*1e-4
            raw.swapaxes(-1,-2).tofile(p+'mlwfHePh')
            model = WannierModel(p, p, fold, fold, 0.15*HARTREE_EV)
            k, q = np.array([[0.13,0,0], [0.27,0,0]]), np.array([0.21,0,0])
            # Independent explicit two-real-space-cell expansion.
            rid = indices(cells, fold)
            full = raw[rid][:,rid]
            w = np.repeat(wph, 3, axis=1)
            full = full*w[:,None,:,:,None]*w[None,:,:,None,:]
            cart = np.einsum('ki,kj,ijxab->kxab', np.exp(-2j*np.pi*(k+q) @ cells.T),
                             np.exp(2j*np.pi*k @ cells.T), full)
            expected = cart/np.sqrt(2*omega_ha)[None,:,None,None]*HARTREE_EV
            np.testing.assert_allclose(model.vertex(k,q), expected, atol=1e-14)
            expected_h = np.einsum('kr,rab->kab', np.exp(2j*np.pi*k @ cells.T), weights*h[rid])*HARTREE_EV
            np.testing.assert_allclose(model.h(k), expected_h, atol=1e-14)
            if os.environ.get('ME_TEST_GPU') == '1':
                gpu = WannierModel(p, p, fold, fold, 0.15*HARTREE_EV, backend='cuda')
                np.testing.assert_allclose(host(gpu.vertex(k,q)), expected, atol=1e-13)
                np.testing.assert_allclose(host(gpu.h(k)), expected_h, atol=1e-13)

    def test_double_delta_normalization(self):
        class ConstantModel:
            mu = 0.
            nmodes = 1
            frequency_floor = 1e-6
            def electrons(self, k):
                return np.zeros((len(k),1)), np.ones((len(k),1,1))
            def phonons(self,q):
                return np.array([0.02]), np.ones((1,1))
            def vertex(self,k,q,*args):
                return np.full((len(k),1,1,1), 0.01)
        sample = mesh_points([2,2,1])
        data = calculate(ConstantModel(), sample, sample, 0.1, chunk=3)
        dos = 1/(0.1*np.sqrt(2*np.pi))
        self.assertAlmostEqual(float(data['dos_per_spin_ev']), dos)
        self.assertAlmostEqual(moments(data['omega_ev'], data['line_weight_ev'])['lambda_total'],
                               2*0.01**2*dos/0.02)


@unittest.skipUnless(os.environ.get('ME_TEST_GPU') == '1', 'set ME_TEST_GPU=1 to require CUDA checks')
class GPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Requested GPU tests must fail, rather than skip, if CUDA is broken.
        cls.backend = Backend('cuda', 0)

    def test_me_kernels_eigenvalues_and_solutions(self):
        args = ([0.015,0.025], [0.004,0.007], 0.1, 0.4, 0.2)
        cpu, gpu = IsotropicME(*args), IsotropicME(*args, backend=self.backend)
        for a, b in zip(cpu.matrices(5.), gpu.matrices(5.)):
            np.testing.assert_allclose(host(b), a, atol=1e-13)
        for t in (5., 60.):
            self.assertAlmostEqual(cpu.eigenvalue(t), gpu.eigenvalue(t), places=10)
            a, b = cpu.solve(t), gpu.solve(t)
            for key in ('z', 'delta_ev', 'phi_ev', 'free_energy_per_dos_ev2'):
                np.testing.assert_allclose(b[key], a[key], atol=2e-12, rtol=1e-9)
        self.assertAlmostEqual(cpu.critical_temperature(2.,60.), gpu.critical_temperature(2.,60.), places=6)

    def test_coherent_metric_fourier_and_poles(self):
        k = mesh_points([4,1,1])
        p = negative_partners(k)
        hopping = np.array([[1.,0.3],[-0.2,0.5]])
        phase = np.exp(2j*np.pi*k[:,0,None,None])
        h = phase*hopping+phase.conj()*hopping.T
        static, _ = isotropic_singlet(h,k,0.2,1.,0.006)
        metric = np.broadcast_to(2*np.eye(8),static.shape).copy()
        metric[:,:4,4:] = metric[:,4:,:4] = 0.1*np.eye(4)
        a, _, _ = coherent_reduce(static,metric,p)
        b, _, _ = coherent_reduce(static,metric,p,backend=self.backend)
        np.testing.assert_allclose(b,a,atol=1e-13)
        arrays_a, _ = export_arrays(a,k,[4,1,1])
        arrays_b, checks = export_arrays(b,k,[4,1,1],self.backend)
        self.assertLess(max(checks.values()),1e-12)
        for key in ('hopping_ev','pairing_ev','eigenvalues_ev'):
            np.testing.assert_allclose(arrays_b[key],arrays_a[key],atol=1e-13)

    def test_device_selection_and_errors(self):
        cp = self.backend.xp
        if cp.cuda.runtime.getDeviceCount() > 1:
            other = Backend('cuda', 1)
            self.assertEqual(other.info['gpu_device'],1)
            cp.cuda.Device(0).use()
        with self.assertRaisesRegex(ValueError,'CUDA backend unavailable'):
            Backend('cuda',cp.cuda.runtime.getDeviceCount())
        with self.assertRaisesRegex(ValueError,'backend must be'):
            Backend('invalid')


if __name__ == '__main__':
    unittest.main()
