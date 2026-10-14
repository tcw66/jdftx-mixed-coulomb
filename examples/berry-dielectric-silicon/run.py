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
    parser.add_argument('--lattice-constant', type=float, default=10.26, help='initial conventional silicon lattice constant in bohr')
    parser.add_argument('--initial-wavefunction', type=Path, help='optional occupied zero-field seed for a single mesh')
    parser.add_argument('--initial-cutoff', type=float, help='cutoff defining the seed G lists in the initial cell, for wavefunction conversion')
    parser.add_argument('--axes', choices=['x', 'y', 'z'], nargs='+', default=['x', 'y', 'z'])
    parser.add_argument('--relax-ions', action='store_true', help='also compute relaxed-ion static tensor at fixed strain')
    parser.add_argument('--relax-cell', action='store_true', help='also relax ions and cell at fixed Cartesian field; first equilibrate zero-field cell')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--electronic-threshold', type=float, default=1e-9)
    parser.add_argument('--force-threshold', type=float, default=1e-7)
    parser.add_argument('--lattice-threshold', type=float, default=1e-6, help='maximum preconditioned joint lattice/ionic gradient component')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='use jdftx_gpu')
    parser.add_argument('--run', action='store_true', help='execute; otherwise only prepare inputs')
    parser.add_argument('--resume', action='store_true', help='reuse converged outputs only when the input is identical')
    args = parser.parse_args()
    if any(n < 3 for n in args.mesh) or any(h <= 0 for h in args.fields):
        parser.error('use meshes >= 3 and positive weak-field amplitudes')
    if (args.threads < 1 or args.ranks < 1 or
            any(not math.isfinite(v) or v <= 0 for v in
                args.fields + [args.cutoff, args.lattice_constant, args.electronic_threshold, args.force_threshold, args.lattice_threshold])):
        parser.error('cutoff, fields, thresholds, thread and rank counts must be finite and positive')
    if args.initial_wavefunction and (len(args.mesh) != 1 or not args.initial_wavefunction.is_file()):
        parser.error('initial-wavefunction requires one mesh and an existing occupied wavefunction file')
    if args.initial_cutoff is not None and (not args.initial_wavefunction or not math.isfinite(args.initial_cutoff) or args.initial_cutoff <= 0):
        parser.error('initial-cutoff requires initial-wavefunction and a finite positive value')
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
        lattice_line = f'lattice Face-Centered Cubic {args.lattice_constant:.16g}\n'
        seed = (f'wavefunction read {args.initial_wavefunction.resolve()} 4 '
                f'{args.initial_cutoff if args.initial_cutoff is not None else args.cutoff:.16g}\n') if args.initial_wavefunction else ''
        common = (f'ion-species {pseudo}\n' + lattice_line + 'coords-type lattice\n'
                  'ion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n'
                  'symmetries none\nkpoint-reduce-inversion no\ncoulomb-interaction periodic\n'
                  f'spintype no-spin\nelec-ex-corr gga-PBE\nelec-cutoff {args.cutoff:g}\n'
                  f'kpoint-folding {n} {n} {n}\n')
        geometry = lattice_line + 'ion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n'
        if args.relax_cell:
            common = common.replace(lattice_line, '').replace(
                'ion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\n', '')
        relaxed = args.relax_ions or args.relax_cell
        modes = ['clamped'] + (['relaxed'] if relaxed else []) + (['cell-relaxed'] if args.relax_cell else [])
        branch = 'berry-phase-branch -1 -1 -1\n' if args.relax_cell else ''
        reference_geometry = 'include equilibrate.lattice\ninclude equilibrate.ionpos\n' if args.relax_cell else ''
        lattice_minimize = ('latt-move-scale 8 8 8\n'
                            'lattice-minimize nIterations 100 energyDiffThreshold 0 '
                            f'knormThreshold {args.lattice_threshold:g} maxThreshold yes\n')
        common_path = work / 'common.in'
        common_unchanged = common_path.is_file() and common_path.read_text() == common
        common_path.write_text(common)
        minimize = ('electronic-minimize nIterations 1000 energyDiffThreshold 0 '
                    f'knormThreshold {args.electronic_threshold:g}\n')
        jobs = []
        if args.relax_cell:
            # Rebuild the cutoff basis after equilibrating the cell, rather than
            # reading a binary wavefunction whose G list belongs to the old cell.
            jobs.append(('equilibrate', 'include common.in\n' + geometry +
                         'elec-n-bands 4\nberry-phase-field 0 0 0\n' + branch + seed + minimize + lattice_minimize +
                         'dump-name equilibrate.$VAR\ndump End State Stress\n'))
        jobs.append(('zero', 'include common.in\n' + reference_geometry +
                     'elec-n-bands 4\nberry-phase-field 0 0 0\n' + branch + ('' if args.relax_cell else seed) + minimize +
                     'dump-name zero.$VAR\ndump End State ElecDensity BandEigs Forces Ecomponents Lattice IonicPositions\n'))
        # Independent gap check at the zero-field density. Empty bands are used
        # only here, never in the electric-enthalpy occupied subspace.
        jobs.append(('gap', 'include common.in\n' + reference_geometry + 'elec-n-bands 8\nwavefunction read zero.wfns 4\n'
                     'fix-electron-density zero.$VAR\ndavidson-band-ratio 2\n'
                     'electronic-minimize nIterations 300 energyDiffThreshold 1e-10\n'
                     'dump-name gap.$VAR\ndump End BandEigs\n'))
        if args.relax_cell:
            # Cell-zero and finite-field cell runs start with the same G lists.
            # Their common fixed basis avoids cutoff-reset errors in curvature.
            jobs.append(('cell-zero', 'include common.in\n' + reference_geometry +
                         'elec-n-bands 4\nwavefunction read zero.wfns\nberry-phase-field 0 0 0\n' +
                         branch + minimize + lattice_minimize +
                         'dump-name cell-zero.$VAR\ndump End State Stress Forces Ecomponents\n'))
        metadata = dict(mesh=[n]*3, cutoff=args.cutoff, fields=args.fields, axes=args.axes,
                        initial_lattice_constant=args.lattice_constant,
                        initial_wavefunction=str(args.initial_wavefunction.resolve()) if args.initial_wavefunction else None,
                        initial_cutoff=args.initial_cutoff,
                        relaxed=relaxed, cell_relaxed=args.relax_cell, modes=modes,
                        electronic_threshold=args.electronic_threshold,
                        force_threshold=args.force_threshold,
                        lattice_threshold=args.lattice_threshold,
                        lattice=[[0., args.lattice_constant/2, args.lattice_constant/2],
                                 [args.lattice_constant/2, 0., args.lattice_constant/2],
                                 [args.lattice_constant/2, args.lattice_constant/2, 0.]],
                        jobs=[])
        for h in args.fields:
            for axis in args.axes:
                for sign in [-1, 1]:
                    field = [0.] * 3
                    field['xyz'.index(axis)] = sign * h
                    for mode in modes:
                        name = f'{mode}-{axis}-{h:g}-{"plus" if sign > 0 else "minus"}'
                        source = 'zero' if mode == 'clamped' else f'clamped-{axis}-{h:g}-{"plus" if sign > 0 else "minus"}'
                        text = ('include common.in\n' + reference_geometry + f'elec-n-bands 4\nwavefunction read {source}.wfns\n'
                                f'berry-phase-field {field[0]:g} {field[1]:g} {field[2]:g}\n' + branch + minimize)
                        if mode == 'relaxed':
                            text += f'ionic-minimize nIterations 100 energyDiffThreshold 0 knormThreshold {args.force_threshold:g}\n'
                        if mode == 'cell-relaxed':
                            text += lattice_minimize
                        text += f'dump-name {name}.$VAR\ndump End State BandEigs Forces Ecomponents Lattice\n'
                        if args.relax_cell:
                            text += 'dump End Stress\n'
                        jobs.append((name, text))
                        metadata['jobs'].append(dict(name=name, mode=mode, axis=axis, amplitude=h, sign=sign,
                                                     initial_state=source))
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
            structure_reused = not args.relax_cell
            job_reused = {}
            for name, _ in jobs:
                output_path = work / f'{name}.out'
                output = output_path.read_text().split('Start date and time:')[-1] if output_path.is_file() else ''
                required = ['gap.eigenvals'] if name == 'gap' else [name + '.polarization']
                if name == 'zero':
                    required += ['zero.wfns', 'zero.n']
                if name == 'equilibrate':
                    required += ['equilibrate.lattice', 'equilibrate.ionpos']
                converged = ('BandDavidson: None of the convergence criteria' not in output and
                             output.count('BandDavidson: Converged') == n**3) if name == 'gap' else (
                                 minimized(output, 'ElecMinimize:'))
                if name.startswith('relaxed'):
                    converged = converged and minimized(output, 'IonicMinimize:')
                if name in ['equilibrate', 'cell-zero'] or name.startswith('cell-relaxed'):
                    converged = converged and minimized(output, 'LatticeMinimize:')
                source = next((job['initial_state'] for job in metadata['jobs'] if job['name'] == name), 'zero')
                reuse = (args.resume and common_unchanged and name in reusable and converged and 'Done!' in output
                         and (name == 'equilibrate' or (structure_reused if name == 'zero' else ground_reused))
                         and (source == 'zero' or job_reused.get(source, False))
                         and all((work / file).is_file() for file in required))
                if name == 'zero':
                    ground_reused = reuse
                if name == 'equilibrate':
                    structure_reused = reuse
                job_reused[name] = reuse
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
                if (name in ['equilibrate', 'cell-zero'] or name.startswith('cell-relaxed')) and not minimized(output, 'LatticeMinimize:'):
                    raise RuntimeError(f'{name}: joint cell/ionic minimization did not converge')
                if name == 'zero':
                    import numpy as np
                    from analyze import read
                    metadata['lattice'] = read(work / 'zero.polarization')['lattice'].reshape(3, 3).tolist()
                    (work / 'workflow.json').write_text(json.dumps(metadata, indent=2)+'\n')
            subprocess.run(['python3', str(Path(__file__).with_name('analyze.py')), str(work)], check=True)


if __name__ == '__main__':
    main()
