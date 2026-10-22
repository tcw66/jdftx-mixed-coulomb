#!/usr/bin/env python3
"""Embedding versus direct finite-system resolvents and thermal BdG sums."""
import os
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from me_postprocess.embedded import (BulkHost, dyson, remove_orbitals,
    matsubara_quadrature, prepare_embedding, solve_embedding, anomalous_matrix,
    perturbed_green, project_diagonal, SpatialWannier)
from me_postprocess.backend import host, select_backend
from me_postprocess import KB_EV


class EmbeddedTests(unittest.TestCase):
    def test_discrete_quadrature(self):
        w, v = matsubara_quadrature(2., order=12, exact=16, maximum=4096)
        exact = (2*np.arange(4096)+1)*np.pi*KB_EV*2
        for energy in [.002, .1, 2., 20., 100.]:
            np.testing.assert_allclose(np.sum(v/(w*w+energy**2)),
                                       np.sum(1/(exact*exact+energy**2)), rtol=2e-10)

    def test_complex_cavity_and_dyson(self):
        rng = np.random.default_rng(19)
        a = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8)); h = a+a.conj().T
        z = .4+.2j; g = np.linalg.inv(z*np.eye(8)-h)
        keep = np.array([0, 2, 3, 5, 6, 7]); cavity = remove_orbitals(g, [1, 4])
        expected = np.linalg.inv(z*np.eye(6)-h[np.ix_(keep, keep)])
        np.testing.assert_allclose(cavity[np.ix_(keep, keep)], expected, atol=2e-14)
        v = np.diag(np.linspace(-.1, .1, 8))
        np.testing.assert_allclose(dyson(g, v), np.linalg.inv(z*np.eye(8)-h-v), atol=2e-14)

    def test_chain_self_consistency(self):
        backend = 'cuda' if os.environ.get('ME_TEST_GPU') else 'cpu'
        device = int(os.environ.get('ME_TEST_GPU_DEVICE', 0))
        translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        hopping = np.array([.003, -.002, -.002])[:, None, None]
        bulk = BulkHost(translations, hopping, [32, 2, 2], 0., 1., .001, 2., backend, device)
        cells = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        surviving = cells[[0, 2]]
        w, weights = matsubara_quadrature(2., maximum=4096)
        cache, ref = prepare_embedding(bulk, cells, w, return_reference=True)
        delta, f, history = solve_embedding(bulk, cache, (w, weights), tolerance=1e-11)
        matrix = anomalous_matrix(bulk, surviving, delta, cache, ref, weights)
        np.testing.assert_allclose(np.diag(matrix), f, atol=1e-12)
        # The Fourier quadrature is exactly a 32-site periodic reference host.
        # Only the vacancy and two local fields change: compare a full solve.
        h = np.diag(np.full(32, .003))
        for i in range(32):
            h[i, (i+1) % 32] = h[(i+1) % 32, i] = -.002
        retained = np.arange(1, 32); h = h[np.ix_(retained, retained)]
        fields = np.full(31, bulk.gap); fields[[30, 0]] = delta
        bdg = np.block([[h, np.diag(fields)], [np.diag(fields), -h]])
        e, vectors = np.linalg.eigh(bdg); positive = e > 0
        full_f = (vectors[:31, positive]*np.tanh(e[positive]/(2*KB_EV*2)))@vectors[31:, positive].conj().T
        np.testing.assert_allclose(np.diag(full_f)[[30, 0]]*bulk.attraction, delta, atol=2e-11)
        g, combined = perturbed_green(bulk, cells, surviving, delta, .0007+.0001j)
        expected = np.linalg.inv((.0007+.0001j)*np.eye(62)-bdg)
        # Observation includes the removed site as exact zeros.
        np.testing.assert_allclose(host(g)[np.ix_([0, 2, 3, 5], [0, 2, 3, 5])],
            expected[np.ix_([30, 0, 61, 31], [30, 0, 61, 31])], atol=1e-9)

    def test_continuum_cross_terms(self):
        w = np.array([[1., 1j], [2., .5]])
        a = np.array([[2., .3+.2j], [.3-.2j, 1.]])
        np.testing.assert_allclose(project_diagonal(w, a), np.diag(w@a@w.conj().T))
        np.testing.assert_allclose(project_diagonal(w, a, False), np.diag(w@a@w.T))
        backend = select_backend('cuda' if os.environ.get('ME_TEST_GPU') else 'cpu',
                                 int(os.environ.get('ME_TEST_GPU_DEVICE', 0)))
        for conjugate in [True, False]:
            expected = np.diag(w@a@(w.conj().T if conjugate else w.T))
            np.testing.assert_allclose(project_diagonal(w, backend.xp.asarray(a), conjugate,
                                                        backend=backend, batch_size=1), expected)

    def test_skew_wannier_fourier_normalization(self):
        lattice = np.array([[5., 1., 0.], [0., 4., 1.], [.5, 0., 6.]])
        gt = 2*np.pi*np.linalg.inv(lattice).T
        ig = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        coefficient = np.array([1., .2j, -.2j, .3, .3], complex)
        coefficient /= np.sqrt(np.sum(abs(coefficient)**2)*abs(np.linalg.det(lattice)))
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp)/'w.')
            coefficient.tofile(prefix+'mlwfC')
            with open(prefix+'mlwfC.header', 'w') as stream:
                stream.write('1 5\n'+' '.join(map(str, gt.ravel()))+'\n')
                np.savetxt(stream, ig, fmt='%d')
            w = SpatialWannier(prefix, grid_factor=4); w.prepare()
            np.testing.assert_allclose(w.norms, 1, atol=1e-14)
            fraction = np.array([[1, 2, 0], [3, 1, 1]])/np.asarray(w.shape)
            positions = w.origin+fraction@lattice.T
            np.testing.assert_allclose(w.evaluate(0, positions), w.exact(0, positions), atol=1e-14)
            np.testing.assert_allclose(w.evaluate(0, positions+lattice[:, 2]), w.exact(0, positions), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
