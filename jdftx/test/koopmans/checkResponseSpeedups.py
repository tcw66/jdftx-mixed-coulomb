#!/usr/bin/env python3
"""Compare KI operator caches, star scheduling and nested solvers at fixed rotations.

The same test can use CUDA with JDFTX_TEST_EXECUTABLE_SUFFIX=_gpu. All inputs,
checkpoints and results live in a temporary directory, or --work-dir.
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
    parser.add_argument('pseudo', type=Path)
    parser.add_argument('--launcher')
    parser.add_argument('--baseline-build', type=Path, help='also time the previous optimized build')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--silicon', action='store_true')
    parser.add_argument('--benchmark', action='store_true', help='time reference, coupled and both nested solvers')
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--case', choices=['gamma', 'wrapping-gga', 'reduced-lda'],
                        help='run one hydrogen regression fixture')
    args = parser.parse_args()
    build, pseudo = args.build.resolve(), args.pseudo.resolve()
    suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    temporary = tempfile.TemporaryDirectory(prefix='jdftx-ki-response-') if not args.work_dir else None
    root = args.work_dir.resolve() if args.work_dir else Path(temporary.name)
    root.mkdir(parents=True, exist_ok=True)
    timings = []

    def run(folder, executable, text, label, ranks=1, binary_build=None, overrides=None):
        binary_build = binary_build.resolve() if binary_build else build
        if executable == 'wannier' and binary_build == build and re.search(r'^wannier-koopmans ', text, re.M):
            # This benchmark extracts per-iteration diagnostics; normal KI
            # calculations now default to concise logging.
            text += 'wannier-koopmans-verbose yes\n'
        inp, out = folder/(label+'.in'), folder/(label+'.out')
        inp.write_text(text)
        out.unlink(missing_ok=True)
        command = [str(binary_build/(executable+suffix)), '-c', '1', '-i', inp.name, '-o', out.name]
        if ranks > 1:
            command = [args.launcher, '-n', str(ranks)]+command
        start = time.monotonic()
        run_env = dict(env, **(overrides or {}))
        run_env['LD_LIBRARY_PATH'] = str(binary_build)+os.pathsep+run_env.get('LD_LIBRARY_PATH', '')
        result = subprocess.run(command, cwd=folder, env=run_env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, timeout=600)
        wall = time.monotonic()-start
        log = out.read_text()
        assert result.returncode == 0 and 'Done!' in log and 'Stack trace:' not in log, log[-5000:]+result.stderr
        if executable == 'wannier' and re.search(r'^wannier-koopmans ', text, re.M):
            residuals = [float(x) for x in re.findall(r'DFPT final relative residual: (\S+)', log)]
            tolerance = float(re.search(r'^wannier-koopmans \S+ \S+ (\S+)', text, re.M)[1])
            assert residuals and max(residuals) <= 10*tolerance, residuals
            counts = re.search(r'KI response solves: (\d+)/(\d+)', log)
            timings.append(dict(case=folder.name, label=label, ranks=ranks, seconds=wall,
                                solves=int(counts[1]), total=int(counts[2]),
                                independent_cg='DFPT independent-band solve:' in log,
                                one_sided='KI time-reversal response: one branch' in log,
                                fft_correlations='KI k-mesh correlations: FFT sources' in log,
                                empty_preconditioner='KI empty-state preconditioner:' in log and 'columns,' in log,
                                recycled_potentials=log.count('KI recycled screening potential:'),
                                band_applications=sum(int(x) for x in re.findall(r'(\d+) band Hamiltonian applications', log)),
                                outer_iterations=len(re.findall(r'DFPT nested iteration', log)),
                                response_tolerance=tolerance, max_relative_residual=max(residuals)))
        return log

    def read(prefix):
        kind, _, _ = Path(str(prefix)+'.koopmansFormat').read_text().split()
        dtype = np.float64 if kind == 'real64' else np.complex128
        return [np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2),
                np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2),
                np.fromfile(str(prefix)+'.mlwfHki', dtype=dtype),
                np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype),
                np.fromfile(str(prefix)+'.mlwfP', dtype=np.float64)]

    cases = [('gamma', '1 1 1', (4.,), 'lda-PZ', 'none'),
             ('wrapping-gga', '1 1 3', (2., 6.), 'gga-PBE', 'none'),
             ('reduced-lda', '3 3 1', (4.,), 'lda-PZ', 'automatic')]
    if args.silicon:
        cases = [('silicon', f'{args.fold} {args.fold} {args.fold}', (), 'gga-PBE', 'automatic')]
    if args.case:
        if args.silicon:
            parser.error('--case and --silicon select different fixtures')
        cases = [case for case in cases if case[0] == args.case]
    for case, folding, centers, xc, sym in cases:
        folder = root/case
        folder.mkdir(exist_ok=True)
        common = f'''ion-species {pseudo}
symmetries {sym}
kpoint-folding {folding}
elec-ex-corr {xc}
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
        if args.silicon:
            nocc, nbands = 4, 24
            common += '''lattice Face-Centered Cubic 10.26
elec-cutoff 30
elec-n-bands 24
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
'''
            trials = ''.join(f'wannier-center Gaussian {x} {y} {z}\n' for x, y, z in
                             [(0.125, 0.125, 0.125), (0.625, 0.125, 0.125),
                              (0.125, 0.625, 0.125), (0.125, 0.125, 0.625)])
        else:
            nocc, nbands = len(centers), 8
            common += '''lattice Cubic 8
coords-type cartesian
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-n-bands 8
'''
            common += ''.join(f'ion H {x} 4 {z} 1\n' for x in centers for z in (3.3, 4.7))
            trials = ''.join(f'wannier-center Gaussian {x} 4 3.3\n' for x in centers)
        run(folder, 'jdftx', common+'dump-name scf.$VAR\ndump End State BandEigs\n', 'scf')
        eigs = np.fromfile(folder/'scf.eigenvals').reshape(-1, nbands)
        midpoint = (eigs[:, nocc-1].max()+eigs[:, nocc].min())/2
        for manifold in ('valence', 'empty', 'combined'):
            prefix = folder/manifold
            if manifold == 'valence':
                subspace = 'bStart 0'
            elif manifold == 'empty':
                subspace = f'outerWindow {midpoint:.15g} {eigs.max()+0.1:.15g}'
            else:
                subspace = (f'frozenCenters {nocc} {folder}/valence.mlwfU '
                            f'innerWindow -10 {midpoint:.15g} outerWindow -10 {eigs.max()+0.1:.15g}')
            base = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier {subspace} saveMomenta yes saveWfns no
wannier-minimize nIterations 100 knormThreshold 1e-8
'''+trials
            response = 'wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 '
            run(folder, 'wannier', base, manifold+'-localize')
            frozen = (base.replace('saveMomenta yes', 'saveMomenta yes loadRotations yes')
                      .replace('nIterations 100 knormThreshold', 'nIterations 0 knormThreshold'))
            run(folder, 'wannier', frozen+response+'no\n', manifold+'-reference')
            reference = read(prefix)
            if args.baseline_build:
                run(folder, 'wannier', frozen+response+'yes\n', manifold+'-previous-coupled',
                    binary_build=args.baseline_build)
                for value, expected in zip(read(prefix), reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
                if args.benchmark:
                    run(folder, 'wannier', frozen+response+'yes\nwannier-koopmans-response yes yes\n',
                        manifold+'-previous-nested-cg', binary_build=args.baseline_build)
                    for value, expected in zip(read(prefix), reference):
                        np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
            configurations = [
                ('coupled', ''),
                ('two-branches', ''),
                ('direct-correlations', ''),
                ('uncached-projectors', ''),
                ('no-caches', 'no yes 0.4 8 80 0 0 0'),
                ('bounded-caches', 'no yes 0.4 8 80 0.25 0.25 0.25'),
                ('nested-minres', 'yes no'),
                ('nested-cg', 'yes yes'),
                ('nested-cg-no-recycling', 'yes yes'),
                ('nested-cg-coarse-only', 'yes yes'),
                ('nested-cg-dielectric-only', 'yes yes'),
                ('nested-cg-bounded-recycling', 'yes yes'),
                ('nested-cg-recycle-two-branches', 'yes yes'),
                ('nested-cg-individual-projectors', 'yes yes'),
            ]
            if args.benchmark:
                configurations = [(label, config) for label, config in configurations
                                  if label in ('coupled', 'nested-minres', 'nested-cg', 'nested-cg-no-recycling',
                                               'nested-cg-coarse-only', 'nested-cg-dielectric-only')
                                  or (suffix == '_gpu' and label == 'nested-cg-individual-projectors')]
            elif case == 'wrapping-gga':
                configurations.append(('one-branch-finite-q', 'yes yes'))
            for label, config in configurations:
                text = frozen+response+'yes\n'
                if config:
                    text += 'wannier-koopmans-response '+config+'\n'
                if label == 'uncached-projectors':
                    text += 'cache-projectors no\n'
                if label == 'one-branch-finite-q':
                    # Use the production response tolerance so the verified
                    # ground-state sewing error admits reduction away from Gamma.
                    text = text.replace('1e-9 no', '1e-7 no')
                overrides = {'two-branches': {'JDFTX_KI_TIME_REVERSAL': '0'},
                             'direct-correlations': {'JDFTX_KI_FFT_CORRELATION': '0'},
                             'nested-cg-no-recycling': {'JDFTX_KI_EMPTY_PRECOND_BANDS': '0', 'JDFTX_KI_DIELECTRIC_RECYCLE_MIB': '0',
                                                        'JDFTX_KI_REUSE_BARE_HESSIAN': '0'},
                             'nested-cg-coarse-only': {'JDFTX_KI_DIELECTRIC_RECYCLE_MIB': '0'},
                             'nested-cg-dielectric-only': {'JDFTX_KI_EMPTY_PRECOND_BANDS': '0'},
                             'nested-cg-bounded-recycling': {'JDFTX_KI_EMPTY_PRECOND_MIB': '0.0001', 'JDFTX_KI_DIELECTRIC_RECYCLE_MIB': '0.0001'},
                             'nested-cg-recycle-two-branches': {'JDFTX_KI_TIME_REVERSAL': '0'},
                             'nested-cg-individual-projectors': {'JDFTX_KI_BATCHED_PROJECTORS': '0'}}.get(label)
                log = run(folder, 'wannier', text, manifold+'-'+label, overrides=overrides)
                for value, expected in zip(read(prefix), reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
                if label == 'nested-cg':
                    assert 'DFPT independent-band solve:' in log, log[-3000:]
                    assert 'KI occupied Hamiltonian is not diagonal' not in log
                    if manifold != 'valence':
                        assert 'KI empty-state preconditioner:' in log
                    if manifold == 'combined':
                        # Alternating sectors must preserve separate histories.
                        assert 'source norm captured (charge sector)' in log
                        assert 'source norm captured (magnetic sector)' in log
                if label in ('nested-cg-no-recycling', 'nested-cg-bounded-recycling'):
                    assert not re.search(r'KI empty-state preconditioner: \d+ columns', log)
                    assert 'KI recycled screening potential:' not in log
                if label == 'one-branch-finite-q':
                    assert 'KI time-reversal response: one branch' in log
                    assert 'DFPT independent-band solve:' in log
                if label.startswith('nested'):
                    assert 'DFPT nested iteration' in log
                    assert 'KI nested screening reached its outer iteration limit' not in log
                if label == 'coupled':
                    assert 'KI frozen orbital cache:' in log
                    assert 'KI k-mesh correlations: FFT sources' in log
                    if case == 'gamma':
                        assert 'KI time-reversal response: one branch' in log, log[-3000:]
                print(f'{case}/{manifold}/{label}: outputs agree, {timings[-1]["seconds"]:.3f} s', flush=True)
            # Exhaustive and symmetry-reused calculations at identical rotations,
            # including cache eviction between stars and a cache too small to insert.
            for label, config in ([] if args.benchmark else
                                  [('star-cache', ''), ('zero-star-cache', 'no yes 0.4 8 80 2048 0')]):
                text = frozen+response.replace('1e-9 no', '1e-9 yes')+'yes\n'
                if config:
                    text += 'wannier-koopmans-response '+config+'\n'
                run(folder, 'wannier', text, manifold+'-'+label)
                for value, expected in zip(read(prefix), reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
            if args.launcher and case != 'gamma':
                text = frozen+response.replace('yes 1 ', 'yes 2 ')+'yes\n'
                text += 'wannier-koopmans-response yes yes\n'
                run(folder, 'wannier', text, manifold+'-nested-mpi', ranks=2)
                for value, expected in zip(read(prefix), reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
                # Keep both ranks in one response group to exercise occupied
                # sewing broadcasts and distributed inner-band convergence.
                text = frozen+response+'yes\nwannier-koopmans-response yes yes\n'
                if case == 'wrapping-gga':
                    text = text.replace('1e-9 no', '1e-7 no')
                log = run(folder, 'wannier', text, manifold+'-nested-intragroup-mpi', ranks=2)
                assert 'DFPT independent-band solve:' in log
                if case == 'wrapping-gga':
                    assert 'KI time-reversal response: one branch' in log
                for value, expected in zip(read(prefix), reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
        (root/'timings.json').write_text(json.dumps(timings, indent=2)+'\n')
    print('KI correlations, time reversal, projector caches, GPU batches and nested solver validation passed.', flush=True)
    if args.work_dir:
        print(f'Logs and timings: {root}', flush=True)


if __name__ == '__main__':
    main()
