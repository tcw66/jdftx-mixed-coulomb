#!/usr/bin/env python3
"""Compare q-group and k-only MPI responses at identical Wannier rotations.
Usage: checkQGroups.py BUILD_DIR H_NC_UPF MPIEXEC
Set JDFTX_TEST_EXECUTABLE_SUFFIX=_gpu to exercise CUDA.
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
launcher = sys.argv[3]
suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(folder, executable, text, ranks=1, success=True,
        expected_error='qGroups must be between 1 and the MPI rank count'):
    (folder/'input.in').write_text(text)
    # All ranks open the CLI output file before secondary-rank logging is
    # disabled. Avoid the native -d truncation race in an MPI launch.
    (folder/'output.out').unlink(missing_ok=True)
    command = [str(build/(executable+suffix)), '-c', '1', '-i', 'input.in', '-o', 'output.out']
    if ranks > 1:
        command = [launcher, '-n', str(ranks)]+command
    result = subprocess.run(command, cwd=folder, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, timeout=180)
    log = (folder/'output.out').read_text()
    if success:
        assert result.returncode == 0 and 'Done!' in log and 'Stack trace:' not in log, log[-3000:]+result.stderr
        if 'KI response solves:' in log:
            nq, groups, pairs, sectors = map(int, re.search(
                r'KI response workload: (\d+) independent q points across (\d+) groups; '
                r'(\d+) independent orbital-q pairs, (\d+) sector solves', log).groups())
            assert pairs == solves(log)[0] and sectors == 2*pairs
            prefix = re.search(r'wannier-dump-name (.*)\.\$VAR', text)[1]
            logs = [log]+[Path(prefix+f'.koopmansQGroup{i}.out').read_text() for i in range(1, groups)]
            assigned = 0
            for group_log in logs:
                group, count = map(int, re.search(
                    r'KI group (\d+)/\d+: (\d+) independent q points assigned', group_log).groups())
                assigned += count
                seen = set()
                for index, total, iq in re.findall(
                        r'KI group \d+/\d+: computed q (\d+)/(\d+), mesh q (\d+)/', group_log):
                    seen.add(int(iq))
                    assert int(index) == len(seen) and int(total) == count
                assert len(seen) == count
            assert assigned == nq
    else:
        assert result.returncode != 0 and expected_error in log+result.stderr, log[-2000:]+result.stderr
    return log


def read(prefix):
    kind, _, _ = Path(str(prefix)+'.koopmansFormat').read_text().split()
    dtype = np.float64 if kind == 'real64' else np.complex128
    return (
        np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2),
        np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2),
        np.fromfile(str(prefix)+'.mlwfHki', dtype=dtype),
        np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype),
        np.fromfile(str(prefix)+'.mlwfP', dtype=np.float64),
    )


def compare(prefix, reference):
    for value, expected in zip(read(prefix), reference):
        np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)


def solves(log):
    return tuple(map(int, re.search(r'KI response solves: (\d+)/(\d+)', log).groups()))


with tempfile.TemporaryDirectory(prefix='jdftx-ki-qgroups-') as tmp:
    root = Path(tmp)
    for case, folding, centers, xc, sym in (
        ('wrapping-gga', '1 1 3', (2., 6.), 'gga-PBE', 'none'),
        ('reduced-lda', '3 3 1', (4.,), 'lda-PZ', 'automatic'),
        ('gamma', '1 1 1', (4.,), 'lda-PZ', 'automatic'),
    ):
        folder = root/case
        folder.mkdir()
        nocc = len(centers)
        common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
symmetries {sym}
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
        eigs = np.fromfile(folder/'scf.eigenvals').reshape(-1, 8)
        gap = (eigs[:, nocc-1].max()+eigs[:, nocc].min())/2
        trials = ''.join(f'wannier-center Gaussian {x} 4 3.3\n' for x in centers)
        # Test combined frozen occupied + new empty centers as well as separate
        # manifolds. The Gamma Ewald term can belong to a non-world-head group.
        for manifold in ('valence', 'empty', 'combined'):
            prefix = folder/manifold
            if manifold == 'valence':
                subspace = 'bStart 0'
            elif manifold == 'empty':
                subspace = f'outerWindow {gap:.15g} {eigs.max()+0.1:.15g}'
            else:
                subspace = (f'frozenCenters {nocc} {folder}/valence.mlwfU '
                            f'innerWindow -10 {gap:.15g} outerWindow -10 {eigs.max()+0.1:.15g}')
            base = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier {subspace} saveMomenta yes
wannier-minimize nIterations 100 knormThreshold 1e-8
'''+trials
            run(folder, 'wannier', base+'wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 no\n')
            frozen = (base.replace('saveMomenta yes', 'saveMomenta yes loadRotations yes')
                      .replace('nIterations 100 knormThreshold', 'nIterations 0 knormThreshold'))
            # Reference uses the original repeated XC setup and split complex
            # potential operator. Compare all optimized paths to this result.
            serial_log = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 no\n')
            reference = read(prefix)
            optimized_log = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 yes\n')
            compare(prefix, reference)
            assert solves(optimized_log) == solves(serial_log)
            log = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 no 1e-6 yes 2\n', ranks=2)
            compare(prefix, reference)
            assert solves(log) == solves(serial_log)
            active = 1 if case == 'gamma' else 2
            assert f'2 response groups requested, {active} active' in log
            if active == 2:
                group_log = Path(str(prefix)+'.koopmansQGroup1.out')
                assert re.search(r'KI group \d+/\d+: .*orbital \d+/\d+:', group_log.read_text())
            # Unequal group sizes exercise subgroup k distribution and cached
            # GGA grid ranges. Multi-rank groups exercise nested collectives.
            for ranks in (3, 4) if case == 'wrapping-gga' else (4,):
                reuse_serial = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 yes 1e-6 yes 1\n', ranks=ranks)
                compare(prefix, reference)
                reuse_parallel = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 yes 1e-6 yes 2\n', ranks=ranks)
                compare(prefix, reference)
                assert solves(reuse_parallel) == solves(reuse_serial)
            if case == 'wrapping-gga' and manifold == 'empty':
                log = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 no 1e-6 no 2\n', ranks=2)
                compare(prefix, reference)
                assert '8/8 stored bands (all stored bands)' in log
                log = run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 no 1e-6 yes 3\n', ranks=4)
                compare(prefix, reference)
                assert '3 response groups requested, 3 active' in log
                assert Path(str(prefix)+'.koopmansQGroup2.out').exists()
                # A failure within a subgroup must terminate the full job,
                # rather than strand other groups at the final world sum.
                run(folder, 'wannier', frozen+'wannier-koopmans 2 1 1e-9 no 1e-6 yes 2\n', ranks=2,
                    success=False, expected_error='Koopmans screening response did not converge')
            print(f'{case}, {manifold}: screening, spectra, matrices and restored exports match', flush=True)
        run(folder, 'wannier', frozen+'wannier-koopmans 2 400 1e-9 yes 1e-6 yes 3\n', ranks=2, success=False)
    print('KI q-group MPI validation passed.', flush=True)
