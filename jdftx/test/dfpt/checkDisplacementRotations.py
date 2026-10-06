#!/usr/bin/env python3
"""Check displacement reconstruction for rotations mixing Cartesian directions."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

binary, boron, nitrogen = [Path(p).resolve() for p in sys.argv[1:4]]
environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
# Hexagonal BN exercises 120-degree Cartesian rotations, which cannot be
# reconstructed by copying signed Cartesian columns or permuting their axes.
common = f'''lattice Hexagonal 4.72 12
coords-type lattice
ion-species {boron}
ion-species {nitrogen}
ion B 0 0 0 1
ion N 0.3333333333333333 0.6666666666666667 0 1
symmetries none
elec-cutoff 8
kpoint-folding 1 1 1
elec-n-bands 4
electronic-minimize nIterations 500 energyDiffThreshold 1e-13
'''


def run(directory, text):
    (directory / 'input.in').write_text(text)
    result = subprocess.run([str(binary), '-c', '1', '-d', '-i', 'input.in', '-o', 'output.out'],
                            cwd=directory, env=environment, capture_output=True, text=True, timeout=120)
    log = (directory / 'output.out').read_text()
    assert result.returncode == 0 and 'Done!' in log, log + result.stderr
    return log


with tempfile.TemporaryDirectory(prefix='jdftx-displacement-rotations-') as name:
    directory = Path(name)
    run(directory, common + 'electronic-scf nIterations 200 nEigSteps 0 energyDiffThreshold 0 '
        'eigDiffThreshold 0 residualThreshold 1e-10\ndump-name scf.$VAR\ndump End State Fillings\n')
    matrices = []
    for enabled in ('no', 'yes'):
        log = run(directory, common + 'initial-state scf.$VAR\nphonon-dfpt 1 1 1\n'
                  'perturb-minimize nIterations 500 residualTol 1e-7 residualDiffThreshold 0\n'
                  f'phonon-dfpt-displacement-symmetries {enabled}\ndump-name {enabled}.$VAR\ndump End None\n')
        directions = int(re.search(r'DFPT displacement symmetry reduction at q 1: (\d+)/6', log)[1])
        assert directions == (6 if enabled == 'no' else 4), log
        rows = (directory / f'{enabled}.phononForceMatrixQ').read_text().splitlines()
        data = np.fromstring(' '.join(rows[2:]), sep=' ')[3:].reshape(6, 6, 2)
        matrices.append(data[..., 0] + 1j*data[..., 1])
    error = np.linalg.norm(matrices[0] - matrices[1]) / np.linalg.norm(matrices[0])
    assert error < 2e-5, f'Cartesian-mixing displacement reconstruction error {error:g}'
    print(f'Hexagonal BN: 4/6 displacement directions, force-matrix relative error {error:.3g}')
