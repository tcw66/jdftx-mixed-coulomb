#!/usr/bin/env python3
"""Split/resume/merge DFPT on CPU or CUDA, including common-gauge el-ph export."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import nullcontext

import numpy as np

binary = Path(sys.argv[1]).resolve()
batch_binary = Path(os.environ.get('DFPT_BATCH_BINARY', binary)).resolve()
pseudo = Path(sys.argv[2]).resolve()
pseudo_input = os.environ.get('DFPT_PSEUDO_INPUT', str(pseudo))
metallic = '--metallic' in sys.argv
phonon_only = '--phonon-only' in sys.argv
ibz_storage = os.environ.get('DFPT_IBZ_STORAGE') == '1'
ibz_nscf = os.environ.get('DFPT_IBZ_NSCF') == '1'
mpi = [a for a in sys.argv[3:] if a not in ('--metallic', '--phonon-only')]
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(directory, text, launch=(), success=True, executable=binary):
    directory.mkdir(exist_ok=True)
    (directory/'input.in').write_text(text)
    (directory/'output.out').unlink(missing_ok=True)
    result = subprocess.run([*launch, str(executable), '-c', '1', '-i', 'input.in', '-o', 'output.out'],
                            cwd=directory, env=env, capture_output=True, text=True, timeout=180)
    log = (directory/'output.out').read_text()+result.stderr
    assert (result.returncode == 0) == success, log[-5000:]
    if success:
        assert 'Done!' in log, log[-5000:]
    return log


with (nullcontext(os.environ['DFPT_TEST_DIR']) if 'DFPT_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-q-batches-')) as name:
    root = Path(name)
    root.mkdir(exist_ok=True)
    common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo_input}
ion H 0 0 3.3 1
ion H 0 0 4.7 1
symmetries none
core-overlap-check none
elec-cutoff 8
kpoint-folding 1 1 {3 if phonon_only else 4}
elec-n-bands {8 if metallic else 1}
electronic-minimize nIterations 500 energyDiffThreshold 1e-14
electronic-scf nIterations 200 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
'''
    if metallic:
        common += 'elec-smearing Fermi 0.025\n'
    run(root/'scf', (common.replace('symmetries none', 'symmetries automatic') if ibz_storage else common)
        +'dump-name scf.$VAR\ndump End State Fillings ElecDensity\n')
    initial = (f'phonon-dfpt-initial-state {root}/scf/input.in {root}/scf/scf.$VAR\n' if ibz_storage
               else f'initial-state {root}/scf/scf.$VAR\n')
    response = common+initial+f'''
perturb-minimize nIterations 500 residualTol 1e-8 residualDiffThreshold 0
phonon-dfpt 1 1 4
phonon-dfpt-electron-phonon {'no' if phonon_only else 'yes'}
phonon-dfpt-solver nested
phonon-dfpt-mixing 0.4 8 100
dump-name result.$VAR
dump End None
'''
    run(root/'reference', response)
    checkpoints = root/'checkpoints'
    checkpointed = response+f'phonon-dfpt-checkpoint {checkpoints}\n'
    if 'DFPT_K_BATCH_SIZE' in os.environ:
        checkpointed += f"phonon-dfpt-k-batch-size {int(os.environ['DFPT_K_BATCH_SIZE'])}\n"
    if ibz_storage:
        if ibz_nscf:
            nscf = common.replace('symmetries none', 'symmetries automatic')
            nscf = '\n'.join(line for line in nscf.splitlines() if not line.startswith('electronic-scf'))
            nscf += f'\nfix-electron-density {root}/scf/scf.$VAR\n'
            (root/'nscf-source.in').write_text(nscf)
            checkpointed = checkpointed.replace(f'{root}/scf/input.in',f'{root}/nscf-source.in')
        checkpointed += 'phonon-dfpt-ibz-storage yes\n'
    batch = checkpointed+'phonon-dfpt-q-batch 1 2\n'
    groups = 'phonon-dfpt-q-groups 2\n' if mpi else ''
    # The first MPI batch writes two q checkpoints concurrently. Subsequent
    # restarts use different rank/group decompositions of the same k mesh.
    first_launch = [mpi[0], '-n', '4'] if mpi else []
    log = run(root/'batch1', batch+groups, first_launch)
    assert 'DFPT q batch 1/2 complete' in log
    if ibz_storage:
        assert 'Deferring full-mesh ground-state reconstruction' in log and 'DFPT IBZ storage ready:' in log
        invalid = checkpointed.replace(f"phonon-dfpt-k-batch-size {int(os.environ['DFPT_K_BATCH_SIZE'])}",
                                       'phonon-dfpt-k-batch-size 0')
        rejected = run(root/'invalid-k-batch', invalid, success=False)
        assert 'positive phonon-dfpt-k-batch-size' in rejected
        rejected = run(root/'invalid-dump', checkpointed.replace('dump End None','dump End State'), success=False)
        assert 'requires dump End None' in rejected
    assert not list((root/'batch1').glob('result.*'))
    files = list(checkpoints.glob('q-*.bin'))
    assert len(files) == 2, files  # Gamma and 1/2; time reversal reduces 1/4 and -1/4.
    snapshots = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    log = run(root/'batch1', batch)
    assert log.count('DFPT checkpoint loaded:') == 2 and 'DFPT phonon q ' not in log, log[-4000:]
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == snapshots[p.name] for p in files)
    # Simulate interruption before the second q in this batch was published.
    last = sorted(files)[-1]
    last.unlink()
    log = run(root/'batch1', batch)
    assert log.count('DFPT checkpoint loaded:') == 1 and 'DFPT phonon q 3/4' in log
    first = sorted(files)[0]
    assert (first.stat().st_mtime_ns, first.read_bytes()) == snapshots[first.name]

    merge = checkpointed+'phonon-dfpt-merge yes\n'
    log = run(root/'missing', merge, success=False)
    assert 'Missing DFPT checkpoint' in log
    assert not list((root/'missing').glob('result.*'))
    # An interrupted write must never masquerade as a completed q checkpoint.
    (checkpoints/'q-000002.bin.tmp-interrupted').write_bytes(b'incomplete')
    launch = [mpi[0], '-n', '2'] if mpi else []
    run(root/'batch2', checkpointed+'phonon-dfpt-q-batch 2 2\n'+groups, launch, executable=batch_binary)
    assert len(list(checkpoints.glob('q-*.bin'))) == 3
    log = run(root/'merged', merge+groups, launch)
    assert log.count('DFPT checkpoint loaded:') >= 1 and 'DFPT phonon q ' not in log

    def force(path):
        lines = path.read_text().splitlines()
        return np.fromstring(' '.join(lines[2:]), sep=' ')

    np.testing.assert_allclose(force(root/'merged/result.phononForceMatrixQ'),
                               force(root/'reference/result.phononForceMatrixQ'), rtol=2e-5, atol=2e-7)
    arrays = [('phononOmegaSq', np.float64)]
    if not phonon_only:
        arrays.append(('phononHsub', np.complex128))
    for suffix, dtype in arrays:
        actual = np.fromfile(root/f'merged/result.{suffix}', dtype=dtype)
        expected = np.fromfile(root/f'reference/result.{suffix}', dtype=dtype)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        # Eigenvector phases can differ across independent initializations. Their
        # magnitudes, and the matching eigenvalues, provide gauge-invariant checks.
        if not (ibz_storage and suffix == 'phononHsub'):
            np.testing.assert_allclose(np.abs(actual), np.abs(expected), rtol=5e-4, atol=3e-7)
        if suffix == 'phononHsub':
            bands = 8 if metallic else 1
            actual = actual.reshape(6, 4, 4, bands, bands).swapaxes(-1, -2)
            expected = expected.reshape(6, 4, 4, bands, bands).swapaxes(-1, -2)
            if ibz_storage:
                # Polishing the IBZ and full mesh can rotate degenerate bands.
                # Compare the same operators using overlaps of the exported
                # eigenstates, rather than assuming a phase-only gauge change.
                wavefunctions = []
                for directory in ('reference', 'merged'):
                    raw = np.fromfile(root/f'{directory}/result.wfns', dtype=np.complex128)
                    states = []
                    offset = 0
                    box = np.stack(np.meshgrid(*([np.arange(-8, 9)]*3), indexing='ij'), axis=-1).reshape(-1, 3)
                    for k in (0., .25, .5, -.25):
                        count = np.count_nonzero(.5*((box+[0, 0, k])**2).sum(axis=1)*(2*np.pi/8)**2 <= 8)
                        states.append(raw[offset:offset+bands*count].reshape(bands, count))
                        offset += bands*count
                    assert offset == raw.size
                    wavefunctions.append(states)
                gauges = [512*(r.conj()@a.T) for r, a in zip(*wavefunctions)]
                points = np.loadtxt(root/'merged/result.phononKpts')
                indices = np.rint(points[:, 2]*4).astype(int) % 4
                energies = np.fromfile(root/'merged/result.eigenvals').reshape(4, bands)
                for target, kt in enumerate(indices):
                    for source, ks in enumerate(indices):
                        expected[:, target, source] = gauges[kt].conj().T@expected[:, target, source]@gauges[ks]
                        # The highest empty band can cut through a degenerate
                        # eigenspace: independent solves may retain different
                        # vectors there. Compare complete lower eigenspaces.
                        nt = np.count_nonzero(energies[kt] < energies[kt, -1]-1e-6) if metallic else bands
                        ns = np.count_nonzero(energies[ks] < energies[ks, -1]-1e-6) if metallic else bands
                        np.testing.assert_allclose(actual[:, target, source, :nt, :ns],
                                                   expected[:, target, source, :nt, :ns], atol=3e-7, rtol=5e-4)
            np.testing.assert_allclose(actual, actual.conj().transpose(0, 2, 1, 4, 3), atol=3e-7, rtol=5e-4)
            # Gamma diagonal elements have no eigenvector phase ambiguity.
            for k in (() if ibz_storage else range(4)):
                np.testing.assert_allclose(np.diagonal(actual[:, k, k], axis1=-2, axis2=-1),
                                           np.diagonal(expected[:, k, k], axis1=-2, axis2=-1),
                                           atol=3e-7, rtol=5e-4)
    for suffix in (() if phonon_only else ('eigenvals', 'fillings')):
        np.testing.assert_allclose(np.fromfile(root/f'merged/result.{suffix}'),
                                   np.fromfile(root/f'reference/result.{suffix}'), atol=2e-8, rtol=1e-6)
    if not phonon_only:
        assert (root/'merged/result.wfns').stat().st_size == (root/'reference/result.wfns').stat().st_size
    else:
        assert not (root/'merged/result.phononHsub').exists()
    # Ordinary runs with a checkpoint directory also resume, then export.
    resumed = checkpointed
    if 'DFPT_K_BATCH_SIZE' in os.environ:
        # Batch size is a runtime memory choice, independent of checkpoint physics.
        resumed = resumed.replace(f"phonon-dfpt-k-batch-size {int(os.environ['DFPT_K_BATCH_SIZE'])}\n",
                                  'phonon-dfpt-k-batch-size 0\n')
        if ibz_nscf:
            resumed = resumed.replace(f'{root}/nscf-source.in',f'{root}/scf/input.in')
        resumed = resumed.replace('phonon-dfpt-ibz-storage yes\n','')
    log = run(root/'resumed', resumed+'phonon-dfpt-source-cache-size 0\nphonon-dfpt-eph-cache-size 0\n')
    assert log.count('DFPT checkpoint loaded:') == 3 and 'DFPT phonon q ' not in log
    np.testing.assert_allclose(force(root/'resumed/result.phononForceMatrixQ'),
                               force(root/'merged/result.phononForceMatrixQ'), atol=1e-12)
    log = run(root/'empty', checkpointed+'phonon-dfpt-q-batch 4 4\n')
    assert 'DFPT phonon q ' not in log and not list((root/'empty').glob('result.*'))
    log = run(root/'incompatible', merge.replace('residualTol 1e-8', 'residualTol 1e-7'), success=False)
    assert 'Incompatible DFPT checkpoint' in log, log[-4000:]
    target = next(checkpoints.glob('q-*.bin'))
    original = target.read_bytes()
    damaged = bytearray(original)
    damaged[-16] ^= 1
    target.write_bytes(damaged)
    log = run(root/'corrupt', merge, success=False)
    assert 'Corrupt DFPT checkpoint' in log
    target.write_bytes(original[:100])
    log = run(root/'truncated', merge, success=False)
    assert 'Truncated DFPT checkpoint' in log
    target.write_bytes(original)
    print(f'{binary.name}: q-batch selection, restart, merge, output comparisons, missing/incompatible/corrupt rejection passed'
          + (' (metallic)' if metallic else ' (insulating)')
          + (' (IBZ storage)' if ibz_storage else '')
          + (' (phonon-only, off-mesh q)' if phonon_only else ' (el-ph)'), flush=True)
