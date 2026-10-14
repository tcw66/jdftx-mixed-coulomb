#!/usr/bin/env python3
"""Prepare/run occupied-only, self-consistent Berry finite-field silicon tensors."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def minimized(output, prefix):
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    return bool(lines) and 'Converged' in lines[-1]


def field_gap_ratios(fields, axes, mesh, lattice, gap):
    """PRL heuristic N_i |E.a_i| / E_gap, in Hartree atomic units."""
    import numpy as np
    lattice = np.asarray(lattice)
    result = []
    for h in fields:
        for axis in axes:
            ratios = np.asarray(mesh)*np.abs(h*lattice['xyz'.index(axis), :])/gap
            result.append(dict(amplitude=h, axis=axis, directional_ratios=ratios.tolist(), maximum=float(ratios.max())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path(__file__).resolve().parents[2] / 'build')
    parser.add_argument('--work', type=Path, default=Path('runs'))
    parser.add_argument('--mesh', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--fields', type=float, nargs='+', default=[0.0002, 0.0001])
    parser.add_argument('--cutoff', type=float, default=30.)
    parser.add_argument('--axes', choices=['x', 'y', 'z'], nargs='+', default=['x', 'y', 'z'])
    parser.add_argument('--relax-ions', action='store_true', help='also compute relaxed-ion static tensor at fixed strain')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--electronic-threshold', type=float, default=1e-9)
    parser.add_argument('--force-threshold', type=float, default=1e-7)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='use jdftx_gpu')
    parser.add_argument('--run', action='store_true', help='execute; otherwise only prepare inputs')
    parser.add_argument('--resume', action='store_true', help='reuse converged outputs only when the input is identical')
    args = parser.parse_args()
    if any(n < 3 for n in args.mesh) or any(h <= 0 for h in args.fields):
        parser.error('use meshes >= 3 and positive weak-field amplitudes')
    if (args.threads < 1 or args.ranks < 1 or
            any(not math.isfinite(v) or v <= 0 for v in
                args.fields + [args.cutoff, args.electronic_threshold, args.force_threshold])):
        parser.error('cutoff, fields, thresholds, thread and rank counts must be finite and positive')
    build = args.build.resolve()
    pseudo = build / 'pseudopotentials/SG15/Si_ONCV_PBE.upf'
    if not pseudo.is_file():
        parser.error(f'missing silicon pseudopotential: {pseudo}')
    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1')
    command = [str(build / ('jdftx_gpu' if args.gpu else 'jdftx'))]
    if args.ranks > 1:
        command = ['mpiexec', '-n', str(args.ranks)] + command
    for n in args.mesh:
        work = root / f'k{n}'
        work.mkdir(exist_ok=True)
        common = (f'ion-species {pseudo}\n'
                  'lattice Face-Centered Cubic 10.26\ncoords-type lattice\n'
                  'ion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n'
                  'symmetries none\nkpoint-reduce-inversion no\ncoulomb-interaction periodic\n'
                  f'spintype no-spin\nelec-ex-corr gga-PBE\nelec-cutoff {args.cutoff:g}\n'
                  f'kpoint-folding {n} {n} {n}\n')
        common_path = work / 'common.in'
        common_unchanged = common_path.is_file() and common_path.read_text() == common
        common_path.write_text(common)
        minimize = ('electronic-minimize nIterations 1000 energyDiffThreshold 0 '
                    f'knormThreshold {args.electronic_threshold:g}\n')
        jobs = [('zero', 'include common.in\nelec-n-bands 4\nberry-phase-field 0 0 0\n' + minimize +
                 'dump-name zero.$VAR\ndump End State ElecDensity BandEigs Forces Ecomponents\n')]
        # Independent gap check at the zero-field density. Empty bands are used
        # only here, never in the electric-enthalpy occupied subspace.
        jobs.append(('gap', 'include common.in\nelec-n-bands 8\nwavefunction read zero.wfns 4\n'
                     'fix-electron-density zero.$VAR\ndavidson-band-ratio 2\n'
                     'electronic-minimize nIterations 300 energyDiffThreshold 1e-10\n'
                     'dump-name gap.$VAR\ndump End BandEigs\n'))
        metadata = dict(mesh=[n]*3, cutoff=args.cutoff, fields=args.fields, axes=args.axes,
                        relaxed=args.relax_ions, electronic_threshold=args.electronic_threshold,
                        force_threshold=args.force_threshold,
                        lattice=[[0., 5.13, 5.13], [5.13, 0., 5.13], [5.13, 5.13, 0.]],
                        jobs=[])
        for h in args.fields:
            for axis in args.axes:
                for sign in [-1, 1]:
                    field = [0.] * 3
                    field['xyz'.index(axis)] = sign * h
                    for mode in ['clamped', 'relaxed'] if args.relax_ions else ['clamped']:
                        name = f'{mode}-{axis}-{h:g}-{"plus" if sign > 0 else "minus"}'
                        text = ('include common.in\nelec-n-bands 4\nwavefunction read zero.wfns\n'
                                f'berry-phase-field {field[0]:g} {field[1]:g} {field[2]:g}\n' + minimize)
                        if mode == 'relaxed':
                            text += f'ionic-minimize nIterations 100 energyDiffThreshold 0 knormThreshold {args.force_threshold:g}\n'
                        text += f'dump-name {name}.$VAR\ndump End State BandEigs Forces Ecomponents\n'
                        jobs.append((name, text))
                        metadata['jobs'].append(dict(name=name, mode=mode, axis=axis, amplitude=h, sign=sign))
        (work / 'workflow.json').write_text(json.dumps(metadata, indent=2) + '\n')
        reusable = set()
        for name, contents in jobs:
            path = work / f'{name}.in'
            if path.is_file() and path.read_text() == contents:
                reusable.add(name)
            path.write_text(contents)
        print(f'Prepared {len(jobs)} calculations in {work}', flush=True)
        if args.run:
            ground_reused = False
            for name, _ in jobs:
                output_path = work / f'{name}.out'
                output = output_path.read_text().split('Start date and time:')[-1] if output_path.is_file() else ''
                required = ['gap.eigenvals'] if name == 'gap' else [name + '.polarization']
                if name == 'zero':
                    required += ['zero.wfns', 'zero.n']
                converged = ('BandDavidson: None of the convergence criteria' not in output and
                             output.count('BandDavidson: Converged') == n**3) if name == 'gap' else (
                                 minimized(output, 'ElecMinimize:'))
                if name.startswith('relaxed'):
                    converged = converged and minimized(output, 'IonicMinimize:')
                reuse = (args.resume and common_unchanged and name in reusable and converged and 'Done!' in output
                         and (name == 'zero' or ground_reused)
                         and all((work / file).is_file() for file in required))
                if name == 'zero':
                    ground_reused = reuse
                print(f'{"Reusing" if reuse else "Running"} k={n} {name}', flush=True)
                if not reuse:
                    subprocess.run(command + ['-d', '-i', f'{name}.in', '-o', f'{name}.out'],
                                   cwd=work, env=env, check=True)
                output = (work / f'{name}.out').read_text().split('Start date and time:')[-1]
                if name == 'gap':
                    import numpy as np
                    if 'None of the convergence criteria' in output or output.count('BandDavidson: Converged') != n**3:
                        raise RuntimeError('zero-field gap eigenproblem did not converge at every k point')
                    eigenvalues = np.fromfile(work / 'gap.eigenvals').reshape(-1, 8)
                    gap = eigenvalues[:, 4].min() - eigenvalues[:, 3].max()
                    if not math.isfinite(float(gap)) or gap <= 0:
                        raise RuntimeError(f'zero-field mesh is metallic (gap {gap:g} Hartree)')
                    print(f'Zero-field sampled gap: {gap * 27.211386245988:.6f} eV', flush=True)
                    ratios = field_gap_ratios(args.fields,args.axes,metadata['mesh'],metadata['lattice'],gap)
                    metadata['field_gap_ratios'] = ratios
                    (work / 'workflow.json').write_text(json.dumps(metadata,indent=2)+'\n')
                    largest = max(row['maximum'] for row in ratios)
                    print(f'PRL mesh/gap ratio max N_i |E.a_i|/gap: {largest:.6f} (heuristic threshold ~1)',flush=True)
                    if largest >= 1:
                        raise RuntimeError('field exceeds the PRL mesh/gap estimate; reduce field amplitude or mesh density')
                elif not minimized(output, 'ElecMinimize:'):
                    raise RuntimeError(f'{name}: electronic minimization did not converge')
                if name.startswith('relaxed') and not minimized(output, 'IonicMinimize:'):
                    raise RuntimeError(f'{name}: ionic minimization did not converge')
            subprocess.run(['python3', str(Path(__file__).with_name('analyze.py')), str(work)], check=True)


if __name__ == '__main__':
    main()
