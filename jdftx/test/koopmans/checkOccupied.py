#!/usr/bin/env python3
"""Compare occupied-only and full-column KI responses at frozen rotations.
Usage: checkOccupied.py BUILD_DIR H_NC_UPF [MPIEXEC]
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import numpy as np

build = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
launcher = sys.argv[3] if len(sys.argv) > 3 else None
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')


def run(folder, executable, text, ranks=1):
    (folder/'input.in').write_text(text)
    command = [str(build/(executable+suffix)), '-c', '1', '-i', 'input.in', '-o', 'output.out']
    if ranks > 1:
        command = [launcher, '-n', str(ranks)]+command
    subprocess.run(command, cwd=folder, env=env, check=True, stdout=subprocess.DEVNULL)
    log = (folder/'output.out').read_text()
    assert 'Done!' in log and 'Stack trace:' not in log, log[-2000:]
    return log


def read(prefix):
    kind, _, _ = Path(str(prefix)+'.koopmansFormat').read_text().split()
    dtype = np.float64 if kind == 'real64' else np.complex128
    return (
        np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2),
        np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2),
        np.fromfile(str(prefix)+'.mlwfHki', dtype=dtype),
        np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype),
        # These subsequent exports require restoration of the stored band count
        # and loaded ground-state arrays after the temporary response state.
        np.fromfile(str(prefix)+'.mlwfH', dtype=np.float64),
        np.fromfile(str(prefix)+'.mlwfP', dtype=np.float64),
    )


with tempfile.TemporaryDirectory(prefix='jdftx-ki-occupied-') as tmp:
    root = Path(tmp)
    for case, folding, centers, xc, sym, inversion in (
        ('gamma', '1 1 1', (4.,), 'lda-PZ', 'none', 'no'),
        ('wrapping-gga', '1 1 3', (2., 6.), 'gga-PBE', 'none', 'no'),
        ('reduced-lda', '1 1 3', (2., 6.), 'lda-PZ', 'automatic', 'yes'),
    ):
        folder = root/case
        folder.mkdir()
        nocc = len(centers)
        common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
symmetries {sym}
kpoint-reduce-inversion {inversion}
kpoint-folding {folding}
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr {xc}
elec-n-bands 8
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
        for x in centers:
            common += f'ion H {x} 4 3.3 1\nion H {x} 4 4.7 1\n'
        run(folder, 'jdftx', common+'dump-name scf.$VAR\ndump End State BandEigs\n')
        eigs = np.fromfile(folder/'scf.eigenvals', dtype=np.float64).reshape(-1, 8)
        gap_midpoint = (eigs[:, nocc-1].max()+eigs[:, nocc].min())/2
        for empty in (False, True):
            prefix = folder/('empty' if empty else 'valence')
            # An isolated pair of the lowest empty H4 bands may have no overlap
            # with bond trials. Disentangle its empty manifold without mixing
            # occupied states, as in the silicon conduction calculation.
            subspace = (f'outerWindow {gap_midpoint:.15g} {eigs.max()+0.1:.15g}'
                        if empty and nocc > 1 else f'bStart {nocc if empty else 0}')
            text = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier {subspace} saveMomenta yes
wannier-minimize nIterations 100 knormThreshold 1e-8
wannier-koopmans 0 400 1e-9 no 1e-6 no
'''
            for x in centers:
                text += f'wannier-center Gaussian {x} 4 3.3\n'
            run(folder, 'wannier', text)
            # Fix the rotations for both paths, avoiding any localization changes.
            frozen = (text.replace('saveMomenta yes', 'saveMomenta yes loadRotations yes')
                      .replace('nIterations 100 knormThreshold', 'nIterations 0 knormThreshold'))
            full_log = run(folder, 'wannier', frozen)
            reference = read(prefix)
            compact = frozen.replace('1e-9 no 1e-6 no', '1e-9 no 1e-6 yes')
            compact_log = run(folder, 'wannier', compact)
            assert f'KI response columns: {nocc}/8 stored bands (occupied only)' in compact_log
            assert 'KI response columns: 8/8 stored bands (all stored bands)' in full_log
            for a, b in zip(read(prefix), reference):
                np.testing.assert_allclose(a, b, atol=3e-9, rtol=3e-8)
            full_count = re.search(r'KI response solves: (\d+)/(\d+)', full_log).groups()
            assert re.search(r'KI response solves: (\d+)/(\d+)', compact_log).groups() == full_count
            if launcher:
                run(folder, 'wannier', compact, ranks=2)
                for a, b in zip(read(prefix), reference):
                    np.testing.assert_allclose(a, b, atol=3e-9, rtol=3e-8)
            # Also check the default compact path together with symmetry reuse.
            run(folder, 'wannier', frozen.replace('1e-9 no 1e-6 no', '1e-9 yes 1e-6'))
            for a, b in zip(read(prefix), reference):
                np.testing.assert_allclose(a, b, atol=3e-8, rtol=3e-7)
            print(f'{case}, {"empty" if empty else "occupied"}: {nocc}/8 columns; screening, spectra, complex matrices and restored exports match', flush=True)
    print('Occupied-only KI response validation passed.')
