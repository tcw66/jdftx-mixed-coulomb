#!/usr/bin/env python3
"""Analytic chain, complex multiatom phase, and binary layout regressions."""
import importlib.machinery
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np

loader = importlib.machinery.SourceFileLoader('phononDFPT', sys.argv[1])
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)

cells = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
force = np.array([-np.eye(3), 2*np.eye(3), -np.eye(3)])
qs = np.array([[0, 0, 0], [0.125, 0.2, 0], [0.5, 0, 0], [1.125, 0.2, 0]])
dyn = module.interpolate(cells, force, qs)
expected = 4*np.sin(np.pi*qs[:, 0])**2
np.testing.assert_allclose(dyn, expected[:, None, None]*np.eye(3), atol=1e-14)
np.testing.assert_allclose(module.frequencies(dyn)[1], np.sqrt(expected[1])*219474.6313705)
# Nonsymmetric real off-site blocks produce complex Hermitian D(q).
b = np.array([[1., 2.], [3., 4.]])
blocks = np.array([b.T, 5*np.eye(2), b])
result = module.interpolate(cells, blocks, qs)
np.testing.assert_allclose(result[1], result[3], atol=1e-14)
np.testing.assert_allclose(result[1], result[1].conj().T)
assert abs(result[1, 0, 1].imag) > 0.1
with tempfile.TemporaryDirectory() as directory:
    prefix = str(Path(directory)/'test.')
    np.savetxt(prefix+'phononCellMap', cells)
    Path(prefix+'phononBasis').write_text('X 0 1 0 0 1\nX 0 0 1 0 1\n')
    blocks.transpose(0, 2, 1).tofile(prefix+'phononOmegaSq')
    loaded_cells, loaded_blocks = module.load(prefix)
    np.testing.assert_array_equal(loaded_cells, cells)
    np.testing.assert_array_equal(loaded_blocks, blocks)
# Fourier grid reconstruction checks normalization and signs.
qs = np.array([[i/3, 0, 0] for i in range(3)])
coarse = module.interpolate(cells, blocks, qs)
back = np.einsum('rq,qij->rij', np.exp(-2j*np.pi*cells@qs.T), coarse)/3
np.testing.assert_allclose(back, blocks, atol=1e-14)
# Instabilities retain their sign rather than being dropped.
np.testing.assert_allclose(module.frequencies(np.array([np.diag([-4., 0., 9.])])),
                           [[-2*219474.6313705, 0., 3*219474.6313705]])
print('DFPT interpolation checks passed')
