#!/usr/bin/env python3
"""Compare full-mesh KI with space-group and time-reversal reduced SCF states."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import numpy as np

build = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
launcher = sys.argv[3] if len(sys.argv) > 3 else None
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(folder, executable, text, ranks=1):
    (folder/'input.in').write_text(text)
    command = [str(build/executable), '-c', '1', '-i', 'input.in', '-o', 'output.out']
    if ranks > 1:
        command = [launcher, '-n', str(ranks)]+command
    subprocess.run(command, cwd=folder, env=env, check=True, stdout=subprocess.DEVNULL)
    output = (folder/'output.out').read_text()
    assert 'Done!' in output and 'Stack trace:' not in output, output[-2500:]
    return output


def bands(prefix):
    data = np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2)
    coordinates = np.round(data[:, :3] % 1., 9) % 1.
    order = np.lexsort(coordinates.T[::-1])
    return np.column_stack((coordinates[order], data[order, 3:]))


with tempfile.TemporaryDirectory(prefix='jdftx-ki-symmetry-') as name:
    root = Path(name)
    for fold in ('1 1 3', '2 2 2'):
        reference = {}
        for mode, sym, inversion in (('full', 'none', 'no'),
                                     ('time-reversal', 'none', 'yes'),
                                     ('space-group', 'automatic', 'yes')):
            folder = root/(fold.replace(' ', '-')+'-'+mode)
            folder.mkdir()
            # Non-origin bond centers also exercise symmetry translation phases.
            common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 3.1 4.2 3.3 1
ion H 3.1 4.2 4.7 1
symmetries {sym}
kpoint-reduce-inversion {inversion}
kpoint-folding {fold}
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr lda-PZ
elec-n-bands 2
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
            run(folder, 'jdftx', common+'dump-name scf.$VAR\ndump End State BandEigs\n')
            # The lowest 3D empty band is degenerate at M; selecting one
            # member gives different subspaces in independent SCF runs.
            # Test its unique occupied manifold and both isolated 1D bands.
            for empty in ((False,) if fold == '2 2 2' else (False, True)):
                prefix = folder/('empty' if empty else 'valence')
                text = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier-center Gaussian 3.1 4.2 3.3
wannier bStart {int(empty)} saveMomenta yes
wannier-minimize nIterations 100 knormThreshold 1e-8
wannier-koopmans 0 400 1e-9
'''
                output = run(folder, 'wannier', text)
                assert 'full-mesh k points reconstructed from' in output
                screening = np.loadtxt(str(prefix)+'.koopmansScreening')
                spectrum = bands(prefix)
                if mode == 'full':
                    reference[empty] = (screening, spectrum)
                else:
                    np.testing.assert_allclose(screening, reference[empty][0], atol=3e-6, rtol=3e-5)
                    np.testing.assert_allclose(spectrum, reference[empty][1], atol=3e-6, rtol=3e-5)
                if launcher and mode == 'space-group':
                    run(folder, 'wannier', text, ranks=2)
                    np.testing.assert_allclose(np.loadtxt(str(prefix)+'.koopmansScreening'), screening, atol=1e-10)
                    np.testing.assert_allclose(bands(prefix), spectrum, atol=1e-10)
                print(f'{fold}: {mode}, {"empty" if empty else "occupied"}, alpha={screening[4]:.9g}', flush=True)
    print('KI symmetry-reduced SCF, time reversal, translation phases, and state restoration passed.')
