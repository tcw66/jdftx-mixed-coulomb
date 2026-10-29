#!/usr/bin/env python3
"""Compare complete fixed-center silicon minimizations from identical atomic trials.

Uses one SCF state for all methods. All writable files live in --work (or a
temporary directory). An optional --scf-prefix is read only. Failed convergence
is retained in summary.json and makes the benchmark fail, rather than counting
an iteration-limit trajectory as a speedup.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--scf-prefix', type=Path)
    parser.add_argument('--methods', nargs='+', choices=['CG', 'LBFGS', 'SQP'],
                        default=['CG', 'LBFGS', 'SQP'])
    parser.add_argument('--history', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--fd-test', action='store_true')
    args = parser.parse_args()
    if args.history < 1 or args.iterations < 1 or args.threshold <= 0 or args.np < 1:
        parser.error('history, iterations, threshold, and np must be positive')
    build = args.build.resolve()
    temporary = None
    if args.work:
        root = args.work.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix='fixed-center-optimizers-')
        root = Path(temporary.name)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    launcher = ['mpirun', '-np', str(args.np)] if args.np > 1 else []

    def run(executable, directory, text):
        (directory/'input.in').write_text(text)
        start = time.monotonic()
        result = subprocess.run(
            launcher+[str(build/executable), '-i', 'input.in', '-o', 'output.out'],
            cwd=directory, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True)
        output = (directory/'output.out').read_text()
        if result.returncode:
            raise RuntimeError(f'{executable} failed in {directory}\n'
                               +output[-4000:]+'\n'+result.stderr)
        return output, time.monotonic()-start

    common = f'''ion-species {build}/pseudopotentials/SG15/Si_ONCV_PBE.upf
lattice Face-Centered Cubic 10.26
coords-type lattice
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
elec-cutoff 5
elec-n-bands 12
kpoint-folding 2 2 2
'''
    if args.scf_prefix:
        scf = args.scf_prefix.resolve()
        for suffix in ('.wfns', '.eigenvals'):
            if not Path(str(scf)+suffix).is_file():
                raise FileNotFoundError(str(scf)+suffix)
    else:
        directory = root/'scf'
        directory.mkdir()
        run('jdftx', directory, common+'''dump-name scf.$VAR
dump End State BandEigs
electronic-minimize nIterations 70
''')
        scf = directory/'scf'
    results = {}
    summary = dict(system='Si, eight fixed s/p atomic trials, 12 source bands, 2x2x2 mesh',
                   measure='TruncatedDC', inner_window_hartree=[-100, -.15],
                   outer_window_hartree=[-100, 100], source_scf=str(scf), ranks=args.np,
                   history=args.history, gradient_threshold=args.threshold,
                   center_tolerance_bohr=1e-8, fd_test=args.fd_test, results=results)
    pattern = (r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+)\s+'
               r'\|grad\|_K:\s*(\S+)\s+centerError:\s*(\S+)\s+t\[s\]:\s*(\S+)')
    for method in dict.fromkeys(args.methods):
        directory = root/method.lower()
        directory.mkdir()
        text = common+f'''wannier-initial-state {scf}.$VAR
wannier-dump-name wannier.$VAR
wannier localizationMeasure TruncatedDC fixCenters yes innerWindow -100 -0.15 outerWindow -100 100 loadRotations no saveWfns no
wannier-center-constraint-solver MatrixFree
wannier-center-constraint-method {method}
wannier-minimize nIterations {args.iterations} energyDiffThreshold 0 knormThreshold {args.threshold} history {args.history} fdTest {'yes' if args.fd_test else 'no'}
'''
        text += ''.join(f'wannier-center Si {atom} {orbital}\n'
                        for atom in (1, 2) for orbital in ('s', 'px', 'py', 'pz'))
        out, wall = run('wannier', directory, text)
        rows = re.findall(pattern, out)
        if not rows:
            raise AssertionError('Missing optimization trajectory in '+str(directory))
        trajectory = [dict(iteration=int(r[0]), objective=float(r[1]), gradient=float(r[2]),
                           center_error_bohr=float(r[3]), seconds=float(r[4])) for r in rows]
        centers = np.loadtxt(directory/'wannier.mlwfCenters', ndmin=2)
        expected = np.repeat([[0., 0., 0.], [.25, .25, .25]], 4, axis=0)
        lattice = 5.13*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        center_error = float(np.max(np.linalg.norm((centers[:, 3:6]-expected)@lattice.T, axis=1)))
        header = (directory/'wannier.mlwfState').read_text().splitlines()[0].split()
        nk = len(header[17:])//3
        nb = int(header[2])
        U = np.fromfile(directory/'wannier.mlwfU', dtype=np.complex128,
                        count=nk*nb*8).reshape(nk, 8, nb).swapaxes(1, 2)
        orth = float(np.max(np.linalg.norm(U.conj().swapaxes(1, 2)@U-np.eye(8), axis=(1, 2))))
        invariants = re.search(r'orthonormality error (\S+); inner-window inclusion error (\S+)', out)
        if not invariants:
            raise AssertionError('Missing inner-window check')
        inclusion = float(invariants[2])
        converged = 'WannierMinimize: Converged' in out and trajectory[-1]['gradient'] < args.threshold
        result = dict(iterations=trajectory[-1]['iteration'], converged=converged,
                      initial_objective=trajectory[0]['objective'], objective=trajectory[-1]['objective'],
                      gradient=trajectory[-1]['gradient'], maximum_center_error_bohr=center_error,
                      exported_orthonormality_error=orth, inner_window_inclusion_error=inclusion,
                      iteration_seconds=trajectory[-1]['seconds']-trajectory[0]['seconds'],
                      wall_seconds=wall, trajectory=trajectory)
        for key, regex in (
                ('line_search_trials', r'line-search trials (\d+)'),
                ('gradient_evaluations', r'; gradient evaluations (\d+)'),
                ('objective_only_evaluations', r'objective-only evaluations (\d+)'),
                ('normal_products', r'matrix-free normal products (\d+)'),
                ('hessian_products', r'SQP Hessian products (\d+)')):
            match = re.search(regex, out)
            if match:
                result[key] = int(match[1])
        results[method] = result
        (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        assert center_error < 1e-8 and max(orth, inclusion) < 1e-8, result
        assert max(r['center_error_bohr'] for r in trajectory) < 1e-8
        assert np.max(np.diff([r['objective'] for r in trajectory]), initial=0.) <= 1e-9
        if args.fd_test:
            assert 'fdTest feasible derivative error' in out
        print(method, json.dumps({k: v for k, v in result.items() if k != 'trajectory'}), flush=True)
    initial = [r['initial_objective'] for r in results.values()]
    assert max(initial)-min(initial) < 5e-8, initial
    failed = [m for m, r in results.items() if not r['converged']]
    assert not failed, f'Methods did not converge: {failed}; results in {root}/summary.json'
    print('Complete fixed-center benchmark passed in', root, flush=True)
    if temporary:
        temporary.cleanup()


if __name__ == '__main__':
    main()
