#!/usr/bin/env python3
"""Metallic aluminum: finite-difference, reduced restart, offset reuse and MPI.
Usage: checkMetallicAl.py JDFTX_BINARY AL_UPF [MPI_LAUNCHER RANK_FLAG]
"""
from contextlib import nullcontext
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''ion-species {pseudo}
coords-type cartesian
core-overlap-check none
elec-cutoff 8
elec-ex-corr gga-PBE
elec-smearing Fermi 0.01
electronic-minimize nIterations 500 energyDiffThreshold 1e-14
electronic-scf nIterations 150 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
'''
primitive = '''lattice face-centered Cubic 7.65
fftbox 32 32 32
ion Al 0 0 0 1
kpoint-folding 2 2 2
elec-n-bands 8
'''
response = '''perturb-minimize nIterations 500 residualTol 1e-9 residualDiffThreshold 0
phonon-dfpt 1 1 2
dump-name result.$VAR
dump End None
'''


def run(directory, text, launcher=(), dry=False, success=True):
    directory.mkdir()
    (directory/'input.in').write_text(text)
    result = subprocess.run([*launcher, str(binary), '-i', 'input.in', '-o', 'output.out', '-c', '1']+
                            (['-n'] if dry else []), cwd=directory, env=env, stdout=subprocess.DEVNULL)
    assert (result.returncode == 0) == success, (directory/'output.out').read_text()[-3000:]
    return (directory/'output.out').read_text()


def load(directory):
    raw = np.fromstring(' '.join((directory/'result.phononForceMatrixQ').read_text().splitlines()[2:]), sep=' ')
    nq = int((directory/'result.phononForceMatrixQ').read_text().splitlines()[1].split()[0])
    raw = raw.reshape(nq, 3+2*3*3)[:, 3:].reshape(nq, 3, 3, 2)
    return raw[..., 0]+1j*raw[..., 1]


with (nullcontext(os.environ['DFPT_TEST_DIR']) if 'DFPT_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-metal-al-')) as name:
    root = Path(name)
    root.mkdir(parents=True, exist_ok=True)
    full = root/'full'
    log = run(full, common+primitive+'symmetries none\n'+response)
    assert 'Metallic DFPT:' in log and 'Fermi-level shift' in log
    full_force = load(full)
    # Cubic degeneracies exercise the finite divided-difference limit.
    assert np.isfinite(full_force).all()
    # A two-cell distortion at q=(0,0,1/2); preserve the physical k mesh.
    # Avoid amplifying inner SCF eigenstate noise in these small metallic forces.
    h = 2e-3
    forces = []
    for sign in (-1, 1):
        directory = root/f'displaced-{sign}'
        dx = sign*h
        text = common+f'''symmetries none
lattice 0 3.825 7.65 3.825 0 7.65 3.825 3.825 0
fftbox 32 32 64
kpoint-folding 2 2 1
elec-n-bands 16
ion Al {dx:.16g} 0 0 1
ion Al {3.825-dx:.16g} 3.825 0 1
dump-name result.$VAR
dump End Forces
'''
        run(directory, text)
        rows = [line.split() for line in (directory/'result.force').read_text().splitlines()
                if line.startswith('force ')]
        forces.append(np.array([float(v) for v in rows[0][2:5]]))
    measured = -(forces[1]-forces[0])/(2*h)
    error = np.linalg.norm(measured-full_force[1, :, 0])/np.linalg.norm(measured)
    assert error < 3e-3, f'aluminum metallic DFPT / finite difference error {error:g}'
    print(f'Aluminum metallic DFPT / supercell forces: relative error {error:.3g}', flush=True)
    source = root/'scf'
    source_text = common+primitive+'symmetries automatic\ndump-name scf.$VAR\ndump End State Fillings\n'
    run(source, source_text)
    restart = f'phonon-dfpt-initial-state {source}/input.in {source}/scf.$VAR\n'
    reduced = root/'reduced'
    log = run(reduced, common+primitive+'symmetries none\n'+restart+response)
    assert 'DFPT ground-state expansion:' in log
    np.testing.assert_allclose(load(reduced), full_force, atol=2e-6, rtol=2e-4)
    no_reuse = root/'no-reuse'
    run(no_reuse, common+primitive+'symmetries none\n'+restart+response+
        'phonon-dfpt-reuse-offset-states no\nphonon-dfpt-displacement-symmetries no\n')
    np.testing.assert_allclose(load(no_reuse), full_force, atol=2e-6, rtol=2e-4)
    print('Metallic reduced SCF restart, offset reuse and displacement symmetry agree', flush=True)
    off_mesh = root/'off-mesh'
    log = run(off_mesh, common+primitive+'symmetries none\n'+restart+
              response.replace('phonon-dfpt 1 1 2', 'phonon-dfpt 1 1 3'))
    force = load(off_mesh)
    assert 'DFPT offset-state reuse: 0/16' in log
    assert np.isfinite(force).all()
    np.testing.assert_allclose(force[1], force[2].conj(), atol=1e-9)
    print('Metallic off-mesh q response converges and preserves reciprocal reality', flush=True)
    # Input errors must fail explicitly before any response work.
    log = run(root/'mismatched-smearing', (common+primitive+'symmetries none\n'+restart+response).replace(
        'elec-smearing Fermi 0.01', 'elec-smearing Gauss 0.01'), dry=True, success=False)
    assert 'must match smearing' in log
    for mode in ('MP1', 'Cold'):
        log = run(root/f'unsupported-{mode}', (common+primitive+'symmetries none\n'+response).replace(
            'elec-smearing Fermi 0.01', f'elec-smearing {mode} 0.01'), dry=True, success=False)
        assert 'requires Fermi or Gauss smearing' in log
    log = run(root/'too-few-bands', (common+primitive+'symmetries none\n'+response).replace(
        'elec-n-bands 8', 'elec-n-bands 7').replace(
        'elec-smearing Fermi 0.01', 'elec-smearing Fermi 0.04'), success=False)
    assert 'highest-band occupation exceeds' in log
    if len(sys.argv) > 3:
        directory = root/'mpi'
        launcher = [sys.argv[3], sys.argv[4], '4']
        run(directory, common+primitive+'symmetries none\n'+restart+response+
            'phonon-dfpt-q-groups 2\n', launcher=launcher)
        np.testing.assert_allclose(load(directory), full_force, atol=2e-6, rtol=2e-4)
        print('Metallic response with two MPI q groups matches serial', flush=True)
