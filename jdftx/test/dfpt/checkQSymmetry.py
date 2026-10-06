#!/usr/bin/env python3
"""Compare symmetry-reconstructed silicon force matrices with explicit q solves."""
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
interpolator = Path(__file__).resolve().parents[2]/'scripts/phononDFPT'
loader = importlib.machinery.SourceFileLoader('phononDFPT', str(interpolator))
spec = importlib.util.spec_from_loader(loader.name, loader)
tool = importlib.util.module_from_spec(spec)
loader.exec_module(tool)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''lattice face-centered Cubic 10.26
coords-type lattice
ion-species {pseudo}
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
symmetries none
elec-cutoff 8
kpoint-folding 2 2 2
elec-n-bands 4
electronic-minimize nIterations 500 energyDiffThreshold 1e-13
'''


def run(directory, text):
    (directory/'input.in').write_text(text)
    subprocess.run([str(binary), '-c', '1', '-d', '-i', 'input.in', '-o', 'output.out'],
                   cwd=directory, env=env, check=True, capture_output=True, text=True)
    log = (directory/'output.out').read_text()
    assert 'Done!' in log, log
    return log


def read_force_q(path):
    rows = path.read_text().splitlines()
    nq, nm = map(int, rows[1].split())
    data = np.fromstring(' '.join(rows[2:]), sep=' ').reshape(nq, 3+2*nm*nm)
    entries = data[:, 3:].reshape(nq, nm, nm, 2)
    return data[:, :3], entries[..., 0]+1j*entries[..., 1]


with tempfile.TemporaryDirectory(prefix='jdftx-q-symmetry-') as name:
    directory = Path(name)
    run(directory, common+'electronic-scf nIterations 200 nEigSteps 0 '
        'energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10\n'
        'dump-name scf.$VAR\ndump End State Fillings\n')
    matrices = []
    for reduction in ('no', 'yes'):
        # An anisotropic q grid exercises the subgroup preserving the entire
        # mesh, including non-self-inverse q and fractional atom translations.
        prefix = f'{directory}/{reduction}.'
        log = run(directory, common+'initial-state scf.$VAR\n'
            'perturb-minimize nIterations 500 residualTol 1e-7 residualDiffThreshold 0\n'
            'phonon-dfpt 4 2 2\n'
            f'phonon-dfpt-symmetries {reduction}\ndump-name {reduction}.$VAR\ndump End None\n')
        representatives, nq = map(int, re.search(r'DFPT q symmetry reduction: (\d+)/(\d+)', log).groups())
        assert nq == 16
        assert representatives == nq if reduction == 'no' else representatives < nq
        solves = int(re.search(r'representatives \((\d+) displacement solves\)', log)[1])
        assert len(re.findall('DFPT final relative residual:', log)) == solves
        if reduction == 'no':
            assert solves == 6*representatives
        else:
            assert solves < 6*representatives
        qs, force = read_force_q(Path(prefix+'phononForceMatrixQ'))
        cells, real_space = tool.load(prefix)
        basis = tool.read_text_array(Path(prefix+'phononBasis'), dtype=str)
        masses = basis[:, -1].astype(float)*1822.88839
        reconstructed = tool.interpolate(cells, real_space, qs)*np.sqrt(masses[:, None]*masses[None, :])
        np.testing.assert_allclose(reconstructed, force, atol=1e-11)
        matrices.append(force)
    error = np.linalg.norm(matrices[0]-matrices[1])/np.linalg.norm(matrices[0])
    assert error < 2e-4, f'q symmetry reconstruction relative error {error:g}'
    assert np.max(np.abs(matrices[0].imag)) > 1e-4, 'test lost its finite-q Bloch phases'
    print(f'Q symmetry versus full grid: {representatives}/{nq} representatives, relative error {error:.3g}')
