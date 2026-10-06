#!/usr/bin/env python3
"""Check MPI q groups against k-only parallelism, including idle groups."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
launcher = sys.argv[3]
rank_flag = sys.argv[4] if len(sys.argv)>4 else '-n'
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
prefix_flags = shlex.split(os.environ.get('DFPT_MPI_PREFLAGS',''))
common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 0 0 3.3 1
ion H 0 0 4.7 1
symmetries none
core-overlap-check none
elec-cutoff 8
kpoint-folding 1 1 3
elec-n-bands 1
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''


def run(directory, text, ranks=1, success=True):
    (directory/'input.in').write_text(text)
    command = [str(binary), '-c', '1', '-d', '-i', 'input.in', '-o', 'output.out']
    if ranks>1:
        command = [launcher]+prefix_flags+[rank_flag, str(ranks)]+command
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=120)
    log = (directory/'output.out').read_text()
    if success:
        assert result.returncode == 0 and 'Done!' in log, log+result.stderr
    else:
        assert result.returncode != 0, log
    return log if success else log+result.stderr


def matrices(path):
    lines = path.read_text().splitlines()
    nq, nm = map(int, lines[1].split())
    data = np.fromstring(' '.join(lines[2:]), sep=' ').reshape(nq, 3+2*nm*nm)
    z = data[:, 3:].reshape(nq, nm, nm, 2)
    return z[..., 0]+1j*z[..., 1]


with tempfile.TemporaryDirectory(prefix='jdftx-q-groups-') as name:
    directory = Path(name)
    run(directory, common+'electronic-scf nIterations 200 nEigSteps 0 '
        'energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10\n'
        'dump-name scf.$VAR\ndump End State Fillings\n')
    response = common+'initial-state scf.$VAR\n'
    response += 'perturb-minimize nIterations 300 residualTol 1e-8 residualDiffThreshold 0\nphonon-dfpt 1 1 3\n'
    response += 'phonon-dfpt-displacement-symmetries no\n'
    reference = None
    for groups, ranks in ((1, 2), (2, 4), (3, 4), (4, 4)):
        log = run(directory, response+f'phonon-dfpt-q-groups {groups}\ndump-name result.$VAR\ndump End State Fillings\n', ranks)
        calculated = matrices(directory/'result.phononForceMatrixQ')
        if groups == 1:
            reference = calculated
        else:
            summary = re.search(r'DFPT q groups completed: (\d+) response solves; maximum relative residual: (\S+)', log)
            assert summary and int(summary[1]) == 12, log
            assert np.isfinite(float(summary[2])) and float(summary[2])<1e-7, log
            error = np.linalg.norm(calculated-reference)/np.linalg.norm(reference)
            assert error<1e-5, f'{groups} q groups differ from reference by {error:g}'
            print(f'{groups} q groups / {ranks} ranks: force-matrix relative difference {error:.3g}')
        # End-state dumps must still represent the entire k mesh and be written once.
        assert (directory/'result.fillings').stat().st_size == 3*8
        assert (directory/'result.wfns').stat().st_size == (directory/'scf.wfns').stat().st_size
    log = run(directory, response+'phonon-dfpt-q-groups 3\ndump End None\n',2,success=False)
    assert 'no more groups than MPI ranks' in log, log
    failed = response.replace('perturb-minimize nIterations 300','perturb-minimize nIterations 1')
    log = run(directory, failed+'phonon-dfpt-q-groups 2\ndump End None\n',4,success=False)
    assert 'DFPT phonon response did not converge' in log, log
