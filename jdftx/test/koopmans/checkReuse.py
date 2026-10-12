#!/usr/bin/env python3
"""Compare reduced and exhaustive KI responses at identical frozen Wannier rotations.
Usage: checkReuse.py BUILD_DIR H_NC_UPF [MPIEXEC]
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


def run(folder, executable, text, ranks=1):
    (folder/'input.in').write_text(text)
    (folder/'output.out').unlink(missing_ok=True)
    command = [str(build/executable), '-c', '1', '-i', 'input.in', '-o', 'output.out']
    if ranks > 1:
        command = [launcher, '-n', str(ranks)]+command
    subprocess.run(command, cwd=folder, env=env, check=True, stdout=subprocess.DEVNULL)
    log = (folder/'output.out').read_text()
    assert 'Done!' in log and 'Stack trace:' not in log, log[-2000:]
    return log


def read(prefix):
    fmt, cells, centers = Path(str(prefix)+'.koopmansFormat').read_text().split()
    dtype = np.float64 if fmt == 'real64' else np.complex128
    return (np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2),
            np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2),
            np.fromfile(str(prefix)+'.mlwfHki', dtype=dtype),
            np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype))


with tempfile.TemporaryDirectory(prefix='jdftx-ki-reuse-') as tmp:
    root = Path(tmp)
    for case, folding, centers, expect in (
        ('wrapping', '1 1 3', (4.,), 2),
        ('rotations', '3 3 1', (4.,), 3),
        ('orbital-translation', '1 1 3', (2., 6.), 2),
    ):
        folder = root/case
        folder.mkdir()
        ncent = len(centers)
        common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
symmetries automatic
kpoint-reduce-inversion yes
kpoint-folding {folding}
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr lda-PZ
elec-n-bands {2*ncent}
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
        for x in centers:
            common += f'ion H {x} 4 3.3 1\nion H {x} 4 4.7 1\n'
        run(folder, 'jdftx', common+'dump-name scf.$VAR\ndump End State BandEigs\n')
        for empty in ((False,) if case == 'rotations' else (False, True)):
            prefix = folder/('empty' if empty else 'valence')
            text = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier bStart {ncent if empty else 0} saveMomenta yes
wannier-minimize nIterations 100 knormThreshold 1e-8
wannier-koopmans 0 400 1e-9 no
'''
            for x in centers:
                text += f'wannier-center Gaussian {x} 4 3.3\n'
            full_log = run(folder, 'wannier', text)
            reference = read(prefix)
            # Freeze the same rotations to distinguish response-reuse error from
            # changes in the localization minimum or gauge.
            frozen = (text.replace('saveMomenta yes', 'saveMomenta yes loadRotations yes')
                      .replace('nIterations 100 knormThreshold', 'nIterations 0 knormThreshold')
                      .replace('1e-9 no', '1e-9 yes 1e-6'))
            reduced_log = run(folder, 'wannier', frozen)
            reduced = read(prefix)
            for a, b in zip(reduced, reference):
                np.testing.assert_allclose(a, b, atol=3e-8, rtol=3e-7)
            solved, total = map(int, re.search(r'KI response solves: (\d+)/(\d+)', reduced_log).groups())
            assert solved < total, reduced_log[-2000:]
            if not empty:
                assert solved <= expect, (case, solved, expect)
            print(f'{case}, {"empty" if empty else "occupied"}: {solved}/{total} response solves; matrices match', flush=True)
            if launcher:
                mpi_log = run(folder, 'wannier', frozen, ranks=2)
                for a, b in zip(read(prefix), reduced):
                    np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-9)
                assert re.search(r'KI response solves: (\d+)/(\d+)', mpi_log).groups() == (str(solved), str(total))
            if case == 'orbital-translation' and not empty:
                # Break equivalence of the two localized densities while keeping
                # the same occupied manifold. Geometry alone must not identify them.
                rotation_file = Path(str(prefix)+'.mlwfU')
                original_bytes = rotation_file.read_bytes()
                matrix_bytes = len(reference[1])*ncent*(2*ncent)*16
                rotations = np.frombuffer(original_bytes[:matrix_bytes], dtype=np.complex128).reshape(-1, ncent, 2*ncent).swapaxes(1, 2)
                angle = 0.27
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                altered = rotations@rotation
                rotation_file.write_bytes(altered.swapaxes(1, 2).tobytes()+original_bytes[matrix_bytes:])
                run(folder, 'wannier', frozen.replace('1e-9 yes 1e-6', '1e-9 no 1e-6'))
                asymmetric_reference = read(prefix)
                log = run(folder, 'wannier', frozen)
                for a, b in zip(read(prefix), asymmetric_reference):
                    np.testing.assert_allclose(a, b, atol=3e-8, rtol=3e-7)
                count = int(re.search(r'KI response solves: (\d+)/', log).group(1))
                assert count > solved, ('Asymmetric orbital densities incorrectly treated as equivalent', count, solved)
                print(f'asymmetric occupied orbitals: {count}/{total} solves; invalid orbital exchange rejected', flush=True)
    print('KI response reuse: time reversal, rotations, orbital translations, asymmetric orbitals, and full complex Hamiltonians passed.')
