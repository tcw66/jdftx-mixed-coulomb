#!/usr/bin/env python3
"""Compare DFPT caches, symmetry reduction, solvers and balanced q groups.

Use Nb to exercise partially occupied bands and the Gamma charge projection;
Si exercises multiple atoms, cell phases, and off-mesh offset eigenstates.
Timings are recorded, but correctness tests never assert a speedup.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

import numpy as np


def matrices(path):
    lines = path.read_text().splitlines()
    nq, nm = map(int, lines[1].split())
    data = np.fromstring(' '.join(lines[2:]), sep=' ').reshape(nq, 3 + 2*nm*nm)
    values = data[:, 3:].reshape(nq, nm, nm, 2)
    return values[..., 0] + 1j*values[..., 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('pseudo', type=Path)
    parser.add_argument('--species', choices=['Si', 'Nb'], default='Si')
    parser.add_argument('--launcher')
    parser.add_argument('--rank-flag', default='-n')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--main-speedups', action='store_true',
                        help='compare k symmetry and nested response solves against the coupled full mesh')
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--k-only', action='store_true',
                        help='with --main-speedups, compare only the full and reduced k meshes')
    args = parser.parse_args()
    binary, pseudo = args.binary.resolve(), args.pseudo.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.species == 'Nb':
        geometry = 'lattice body-centered Cubic 6.27510830215844\nion Nb 0 0 0 1\n'
        electronic = ('elec-smearing Fermi 0.01\nelec-n-bands 14\n'
                      'converge-empty-states yes\n')
        mesh = '2 2 2'
    else:
        geometry = ('lattice face-centered Cubic 10.26\n'
                    'ion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n')
        electronic = 'elec-n-bands 4\n'
        mesh = '4 1 1'
    common = (f'ion-species {pseudo}\ncoords-type lattice\n{geometry}'
              f'elec-cutoff 8\nkpoint-folding {args.fold} {args.fold} {args.fold}\n{electronic}'
              'symmetries none\nelectronic-minimize nIterations 500 energyDiffThreshold 1e-13\n')
    scf = ('electronic-scf nIterations 200 nEigSteps 30 energyDiffThreshold 0 '
           'eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4\n')
    response = (f'perturb-minimize nIterations 500 algorithm MINRES residualTol 1e-8 residualDiffThreshold 0\n'
                f'phonon-dfpt {mesh}\nphonon-dfpt-displacement-symmetries no\n')
    timings = {}
    comparisons = {}
    context = (tempfile.TemporaryDirectory(prefix='jdftx-frozen-cache-') if args.out is None
               else nullcontext(str(args.out.resolve())))
    with context as name:
        directory = Path(name)
        directory.mkdir(parents=True, exist_ok=True)

        def run(name, text, ranks=1):
            (directory / f'{name}.in').write_text(text)
            command = [str(binary), '-c', '1', '-i', f'{name}.in', '-o', f'{name}.out']
            if ranks > 1:
                command = ([args.launcher] + shlex.split(os.environ.get('DFPT_MPI_PREFLAGS', ''))
                           + [args.rank_flag, str(ranks)] + command)
            started = time.monotonic()
            result = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                    text=True, timeout=240)
            timings[name] = time.monotonic() - started
            output = directory / f'{name}.out'
            assert output.exists(), result.stdout + result.stderr
            log = output.read_text()
            assert result.returncode == 0 and 'Done!' in log, log + result.stderr
            return log

        run('scf', common + scf + 'dump-name scf.$VAR\ndump End State Fillings\n')
        reference = None
        cases = [('uncached', 'no', 128., 1), ('potentials', 'yes', 0., 1),
                 ('bounded', 'yes', 0.001, 1), ('cached', 'yes', 128., 1)]
        if args.main_speedups:
            cases = [('cached', 'yes', 128., 1), ('k-reduced', 'yes', 128., 1),
                     ('nested', 'yes', 128., 1), ('combined', 'yes', 128., 1)]
            if args.k_only:
                cases = cases[:2]
        if args.launcher:
            cases.append(('mpi-cached', 'yes', 128., 3))
        for name, cache, budget, ranks in cases:
            text = common + 'initial-state scf.$VAR\n' + response
            text += f'phonon-dfpt-cache {cache}\nphonon-dfpt-source-cache-size {budget}\n'
            reduced = args.main_speedups and name in ('k-reduced', 'combined', 'mpi-cached')
            nested = args.main_speedups and name in ('nested', 'combined', 'mpi-cached')
            # Exercise displacement reconstruction in serial. Keep explicit
            # directions in the MPI case so the balanced first group owns
            # Gamma and its charge diagnostic appears in the visible log.
            if args.main_speedups and name == 'combined':
                text = text.replace('phonon-dfpt-displacement-symmetries no',
                                    'phonon-dfpt-displacement-symmetries yes')
            text += f'phonon-dfpt-k-symmetries {"yes" if reduced else "no"}\n'
            text += f'phonon-dfpt-solver {"nested" if nested else "coupled"}\n'
            if ranks > 1:
                text += 'phonon-dfpt-q-groups 2\n'
            text += f'dump-name {name}.$VAR\ndump End None\n'
            log = run(name, text, ranks)
            calculated = matrices(directory / f'{name}.phononForceMatrixQ')
            if reference is None:
                reference = calculated
            error = np.linalg.norm(calculated - reference) / max(1e-30, np.linalg.norm(reference))
            # Symmetry reconstruction mixes cancellation-limited entries.
            # Scale their absolute floor to the force matrix; also bound the
            # complete matrix's relative error below, including near zeros.
            absolute_tol = max(2e-9, 1e-7 * np.max(np.abs(reference)))
            np.testing.assert_allclose(calculated, reference, atol=absolute_tol, rtol=2e-6)
            assert error < 2e-6, error
            comparisons[name] = float(error)
            rows = re.findall(r'DFPT q cache: (\d+) retained sources, (\d+) hits, (\d+) source builds, '
                              r'(\d+) bare potential pairs, maximum rank storage (\S+) MiB', log)
            assert rows, log
            if cache == 'no':
                assert all(int(retained) == int(hits) == int(pairs) == 0
                           for retained, hits, builds, pairs, storage in rows)
            elif budget < 0.01:
                assert all(int(retained) == int(hits) == 0 for retained, hits, builds, pairs, storage in rows)
                assert any(int(pairs) > 0 for retained, hits, builds, pairs, storage in rows)
            else:
                if not (reduced and args.species == 'Nb'):
                    # Displacement reconstruction contracts the retained
                    # sources directly and need not perform another lookup.
                    if name == 'combined':
                        assert any(int(retained) > 0 for retained, hits, builds, pairs, storage in rows)
                    else:
                        assert any(int(hits) > 0 for retained, hits, builds, pairs, storage in rows)
                assert any(int(pairs) > 0 for retained, hits, builds, pairs, storage in rows)
                assert all(float(storage) <= budget for retained, hits, builds, pairs, storage in rows)
            if args.species == 'Nb':
                charges = [float(x) for x in re.findall(r'Metallic DFPT Gamma charge response: ([0-9.eE+-]+)', log)]
                assert charges and max(charges) < 1e-9, charges
            if ranks > 1:
                assert 'DFPT q groups completed:' in log
                assert 'capacity 1.' in log and 'capacity 2.' in log
            if reduced:
                assert 'DFPT electronic response reduction:' in log, log
            if nested:
                assert 'DFPT nested iteration' in log, log
                assert 'DFPT independent-band solve:' in log, log
            print(f'{args.species} {name}: {timings[name]:.3f} s; force-matrix relative error {error:.3g}', flush=True)
        (directory / 'cache-validation.json').write_text(json.dumps(
            {'species': args.species, 'seconds': timings, 'relative_errors': comparisons}, indent=2) + '\n')


if __name__ == '__main__':
    main()
