#!/usr/bin/env python3
"""Compare symmetry-exported complex operators with explicit DFPT responses."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('pseudo', type=Path)
parser.add_argument('--species', choices=('Si', 'Nb', 'BN'), default='Si')
parser.add_argument('--nitrogen', type=Path)
parser.add_argument('--launcher')
parser.add_argument('--rank-flag', default='-n')
parser.add_argument('--ranks', type=int, default=3)
parser.add_argument('--nested', action='store_true')
parser.add_argument('--ibz-storage', action='store_true')
parser.add_argument('--reuse-reference', action='store_true',
                    help='reuse a completed explicit reference and SCF in DFPT_SYMMETRY_TEST_DIR')
args = parser.parse_args()
if args.reuse_reference and 'DFPT_SYMMETRY_TEST_DIR' not in os.environ:
    parser.error('--reuse-reference requires DFPT_SYMMETRY_TEST_DIR')
binary, pseudo = args.binary.resolve(), args.pseudo.resolve()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
timings = {}
if args.species == 'Nb':
    geometry = 'lattice body-centered Cubic 6.28\nion Nb 0 0 0 1\n'
    settings = 'kpoint-folding 2 2 2\nelec-n-bands 20\nelec-smearing Fermi 0.02\n'
    grid, nm, bands = '2 2 2', 3, 20
elif args.species == 'BN':
    assert args.nitrogen, '--nitrogen is required for BN'
    geometry = ('lattice Hexagonal 4.72 12\n'
                f'ion-species {args.nitrogen.resolve()}\n'
                'ion B 0 0 0 1\nion N 0.3333333333333333 0.6666666666666667 0 1\n')
    settings = 'kpoint-folding 1 1 1\nelec-n-bands 4\n'
    grid, nm, bands = '1 1 1', 6, 4
else:
    geometry = 'lattice face-centered Cubic 10.26\nion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n'
    # Non-self-inverse q points plus nonsymmorphic atom translations.
    # IBZ density symmetrization must preserve the electronic integration mesh.
    settings = f'kpoint-folding {"2 2 2" if args.ibz_storage else "4 2 2"}\nelec-n-bands 4\n'
    grid, nm, bands = ('2 2 2' if args.ibz_storage else '4 2 2'), 6, 4
common = f'''coords-type lattice
ion-species {pseudo}
{geometry}{settings}symmetries none
elec-cutoff 8
elec-ex-corr gga-PBE
electronic-minimize nIterations 500 energyDiffThreshold 1e-13
electronic-scf nIterations 200 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
'''
nq = int(np.prod([int(n) for n in grid.split()]))


def run(directory, text, launch=(), log_name='scf.out'):
    (directory/'input.in').write_text(text)
    # All ranks open -o before MPI initialization; removing the old log avoids
    # concurrent truncation by --no-append on a shared file.
    (directory/log_name).unlink(missing_ok=True)
    # Exercise the old single-column GPU FFT path as the explicit reference.
    run_env = dict(env, JDFTX_FFT_BATCH_SIZE='0') if log_name in ('explicit.out', 'reference.out') else env
    started = time.perf_counter()
    result = subprocess.run([*launch, str(binary), '-c', '1', '-i', 'input.in', '-o', log_name],
                            cwd=directory, env=run_env, capture_output=True, text=True, timeout=300)
    log = (directory/log_name).read_text()
    assert result.returncode == 0 and 'Done!' in log, log[-6000:] + result.stderr
    timings[log_name] = time.perf_counter() - started
    return log


def relative(actual, reference):
    return np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-30)


def eigenstates(directory, prefix, log, sizes=None):
    # The first offset setup is Gamma: its alternating +/- basis dimensions
    # coincide with the full electronic mesh saved in the native checkpoint.
    if sizes is None:
        pairs = [int(n) for n in re.findall(r'nbasis = (\d+) for k =', log)][:2*nq]
        assert len(pairs) == 2*nq and pairs[::2] == pairs[1::2]
        sizes = pairs[::2]
    raw = np.fromfile(directory/f'{prefix}.wfns', dtype=np.complex128)
    states, offset = [], 0
    for nbasis in sizes:
        count = nbasis*bands
        state = raw[offset:offset+count].reshape(bands, nbasis).T.copy()
        state /= np.linalg.norm(state, axis=0)
        states.append(state)
        offset += count
    assert offset == raw.size
    return states


def complex_operator_error(h, actual, reference, reference_h, energies):
    # MPI summation changes can rotate degenerate eigenvectors. Use overlaps
    # of the exported states, rather than comparing only scattering norms.
    rotations = []
    for k, (a, r) in enumerate(zip(actual, reference)):
        # A finite band count may cut the last degenerate empty manifold.
        # Its chosen subspace can differ between runs; compare all complete
        # lower manifolds, including the occupied and partially occupied bands.
        count = int(np.count_nonzero(energies[k] < energies[k, -1]-1e-6)) if args.species == 'Nb' else bands
        assert count >= (7 if args.species == 'Nb' else bands)
        left, values, right = np.linalg.svd(a.conj().T@r[:, :count], full_matrices=False)
        np.testing.assert_allclose(values, 1., atol=2e-5)
        rotations.append(left@right)
    squared_error, squared_norm = 0., 0.
    for target in range(nq):
        for source in range(nq):
            aligned = rotations[target].conj().T@h[:, target, source]@rotations[source]
            expected = reference_h[:, target, source, :aligned.shape[-2], :aligned.shape[-1]]
            squared_error += np.linalg.norm(aligned-expected)**2
            squared_norm += np.linalg.norm(expected)**2
    return np.sqrt(squared_error/max(squared_norm, 1e-30))


with (nullcontext(os.environ['DFPT_SYMMETRY_TEST_DIR']) if 'DFPT_SYMMETRY_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-eph-symmetry-')) as name:
    directory = Path(name)
    directory.mkdir(parents=True, exist_ok=True)
    if args.reuse_reference:
        assert all((directory/name).is_file() for name in (
            'scf.wfns', 'scf.fillings', 'explicit.out', 'explicit.phononHsub',
            'explicit.phononForceMatrixQ', 'explicit.wfns', 'explicit.eigenvals'))
        if (directory/'timings.json').exists():
            timings.update(json.loads((directory/'timings.json').read_text()))
    else:
        run(directory, common+'dump-name scf.$VAR\ndump End State Fillings\n')
    if args.ibz_storage:
        source = common.replace('symmetries none','symmetries automatic')+'dump-name ibz-scf.$VAR\ndump End State Fillings\n'
        (directory/'ibz-source.in').write_text(source)
        run(directory,source,log_name='ibz-scf.out')
    variants = [('explicit', 'no', 'no', ())]
    if args.species != 'BN' and not args.reuse_reference and not args.ibz_storage:
        variants.append(('q-only', 'yes', 'no', ()))
    variants.append(('reference', 'yes', 'yes', ()))
    variants.append(('reduced', 'yes', 'yes', ()))
    variants.append(('optimized', 'yes', 'yes', ()))
    variants.append(('optimized-uncached', 'yes', 'yes', ()))
    if args.launcher:
        variants.append(('mpi', 'yes', 'yes', (args.launcher, args.rank_flag, str(args.ranks))))
        variants.append(('optimized-mpi', 'yes', 'yes', (args.launcher, args.rank_flag, str(args.ranks))))
    if args.ibz_storage:
        variants = [variants[0], ('ibz', 'yes', 'yes',
                    (args.launcher,args.rank_flag,str(args.ranks)) if args.launcher else ())]
    reference = None
    for prefix, qsym, dsym, launch in variants:
        if prefix == 'explicit' and args.reuse_reference:
            log = (directory/'explicit.out').read_text()
            assert 'Done!' in log
        else:
            initial = ('phonon-dfpt-initial-state ibz-source.in ibz-scf.$VAR\n'
                       'phonon-dfpt-k-batch-size 1\nphonon-dfpt-ibz-storage yes\n' if prefix == 'ibz'
                       else 'initial-state scf.$VAR\n')
            log = run(directory, common+initial+f'''
perturb-minimize nIterations 500 residualTol 1e-7 residualDiffThreshold 0
phonon-dfpt {grid}
phonon-dfpt-electron-phonon yes
phonon-dfpt-symmetries {qsym}
phonon-dfpt-displacement-symmetries {dsym}
phonon-dfpt-k-symmetries {'yes' if prefix.startswith('optimized') else 'no'}
phonon-dfpt-optimize {'no' if prefix in ('explicit', 'reference') else 'yes'}
phonon-dfpt-eph-cache-size {0 if prefix == 'optimized-uncached' else 512}
phonon-dfpt-eph-reciprocity {'no' if prefix in ('explicit', 'reference', 'optimized-uncached') else 'yes'}
dump-name {prefix}.$VAR
dump End None
'''+('phonon-dfpt-q-groups 2\n' if launch else '')
            +('phonon-dfpt-solver nested\n' if args.nested else ''), launch, prefix+'.out')
        representatives, total, solves = map(int, re.search(
            r'DFPT q symmetry reduction: (\d+)/(\d+) representatives \((\d+) displacement solves\)', log).groups())
        assert total == nq
        if launch:
            assert int(re.search(r'DFPT q groups completed: (\d+) response solves', log)[1]) == solves
        else:
            assert len(re.findall(r'DFPT final relative residual(?: bound)?:', log)) == solves
        if prefix == 'explicit':
            assert representatives == nq and solves == nq*nm
        else:
            assert solves < nq*nm, log
        if prefix == 'optimized':
            contractions = [int(n) for n in re.findall(r'column \d+: (\d+)/\d+ contractions', log)]
            assert len(contractions) == nq*nm
            assert sum(contractions) == nm*nq*(nq+1)//2, 'Hermitian pair reduction was not applied'
            if args.species == 'Nb':
                assert 'DFPT electronic response reduction:' in log, 'response k reduction was not exercised'
        raw = np.fromfile(directory/f'{prefix}.phononHsub', dtype=np.complex128)
        assert raw.size == nm*nq*nq*bands*bands and np.isfinite(raw).all()
        h = raw.reshape(nm, nq, nq, bands, bands).swapaxes(-1, -2)
        reciprocity = relative(h, h.conj().transpose(0, 2, 1, 4, 3))
        assert reciprocity < 3e-4, f'{prefix}: reciprocity error {reciprocity:g}'
        rows = (directory/f'{prefix}.phononForceMatrixQ').read_text().splitlines()
        force = np.fromstring(' '.join(rows[2:]), sep=' ').reshape(nq, 3+2*nm*nm)[:, 3:]
        # Q groups need not assign Gamma to the logging group. Ground-state
        # bases are fixed, so use dimensions from the serial explicit run.
        states = eigenstates(directory, prefix, log, None if reference is None
                             else [state.shape[0] for state in reference[2]])
        if reference is None:
            energies = np.fromfile(directory/f'{prefix}.eigenvals', dtype=np.float64).reshape(nq, bands)
            reference = h, force, states, energies
        else:
            error = complex_operator_error(h, states, reference[2], reference[0], reference[3])
            force_error = relative(force, reference[1])
            assert error < 3e-4, f'{prefix}: complex operator error {error:g}'
            assert force_error < 3e-4, f'{prefix}: force matrix error {force_error:g}'
            print(f'{args.species} {prefix}: {representatives}/{nq} q points, {solves}/{nq*nm} responses; '
                  f'complex operator error {error:.3g}, force error {force_error:.3g}; '
                  f'{timings[prefix+".out"]:.2f} s', flush=True)
    (directory/'timings.json').write_text(json.dumps(timings, indent=2)+'\n')
