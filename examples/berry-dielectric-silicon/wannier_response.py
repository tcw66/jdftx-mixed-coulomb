#!/usr/bin/env python3
"""Wannierize converged occupied silicon states and finite-difference polarization."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze import aligned_change, minimized, read


CENTERS = np.array([[.125, .125, .125], [-.375, .125, .125],
                    [.125, -.375, .125], [.125, .125, -.375]])


def fingerprint(paths):
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def lattice(path):
    text = path.read_text().split('#')[0].replace('\\', '').split()
    if text[0] != 'lattice' or len(text) != 10:
        raise ValueError(f'{path}: expected a dumped 3x3 lattice')
    return np.array([float(v) for v in text[1:]]).reshape(3, 3)


def compatible_bases(initial, final, mesh, cutoff):
    """Compare every G-vector membership, not just total binary file sizes.

    Lattice minimization preserves the initial G lists. Wannier rebuilds them
    at the final cell, so even a length-preserving change would corrupt a read.
    This helper is specific to the uniform Gamma-centered silicon workflow.
    """
    bounds = [int(np.ceil(np.sqrt(2*cutoff)*max(np.linalg.norm(initial[:, d]),
              np.linalg.norm(final[:, d]))/(2*np.pi) + .5)) + 1 for d in range(3)]
    gs = np.array(list(itertools.product(*(range(-b, b+1) for b in bounds))))
    ks = np.array(list(itertools.product(*(((np.arange(n)+n//2) % n - n//2)/n
                                          for n in mesh))))
    metrics = []
    for r in (initial, final):
        b = 2*np.pi*np.linalg.inv(r).T
        metrics.append(b.T @ b)
    count = 0
    for start in range(0, len(ks), 32):
        q = gs[None, :, :] + ks[start:start+32, None, :]
        masks = [np.einsum('...i,ij,...j->...', q, metric, q)/2 <= cutoff
                 for metric in metrics]
        if not np.array_equal(*masks):
            raise ValueError('Final-cell cutoff basis differs from saved initial G lists. '
                             'Convert wavefunctions with explicit G-vector mapping before Wannierization.')
        count += np.count_nonzero(masks[0])
    return count*4*16  # Four occupied complex-double bands, no spin polarization.


def matched_centers(centers, reference, r):
    """Match labels and primitive-cell images to the weak-field reference."""
    offsets = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    delta = centers[:, None, :] - reference[None, :, :]
    shifts = -np.rint(delta).astype(int)[:, :, None, :] + offsets
    displacement = (delta[:, :, None, :] + shifts) @ r.T
    norms = np.sum(displacement**2, axis=-1)
    best = np.argmin(norms, axis=-1)
    rows, cols = linear_sum_assignment(np.min(norms, axis=-1))
    aligned = np.empty_like(centers)
    images = np.empty(centers.shape, dtype=int)
    order = np.empty(len(centers), dtype=int)
    for row, col in zip(rows, cols):
        order[col] = row
        images[col] = shifts[row, col, best[row, col]]
        aligned[col] = centers[row] + images[col]
    if np.max(np.linalg.norm((aligned-reference) @ r.T, axis=1)) > .25:
        raise ValueError('Wannier centers moved too far for the weak-field branch; reduce field')
    return aligned, images, order


def polarization(centers, ions, r, branch, spin_weight=2):
    """Physical positive-ion convention, charges in e, lengths in bohr."""
    volume = float(np.linalg.det(r))
    if volume <= 0:
        raise ValueError('Expected a right-handed cell')
    electronic = -spin_weight * (r @ np.sum(centers, axis=0))/volume
    ionic = 4 * (r @ np.sum(ions, axis=0))/volume  # SG15 Si valence charge
    offset = r @ branch/volume
    return electronic, ionic, electronic + ionic + offset


def ionic_positions(path):
    positions = []
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == 'ion':
            if words[1] != 'Si':
                raise ValueError('This workflow requires two SG15 silicon atoms')
            positions.append([float(x) for x in words[2:5]])
    if len(positions) != 2:
        raise ValueError('Expected two Si ions')
    return np.array(positions)


def positions_file(directory, name):
    path = directory/f'{name}.ionpos'
    # Clamped runs never change the zero-field positions and need not dump them.
    return directory/'zero.ionpos' if name.startswith('clamped-') and not path.exists() else path


def check_source(directory, name):
    log = (directory/f'{name}.out').read_text().split('Start date and time:')[-1]
    if not minimized(log, 'ElecMinimize:'):
        raise ValueError(f'{name}: source electrons did not converge')
    structural = 'LatticeMinimize:' if name.startswith('cell-') else (
                 'IonicMinimize:' if name.startswith('relaxed-') else None)
    if structural and not minimized(log, structural):
        raise ValueError(f'{name}: source structure did not converge')


def analyze(directory, output, metadata, modes):
    states = {}
    for mode in modes:
        reference = 'cell-zero' if mode == 'cell-relaxed' else 'zero'
        jobs = [job for job in metadata['jobs'] if job['mode'] == mode]
        ref_data = read(directory / f'{reference}.polarization')
        np.testing.assert_allclose(ref_data['field'], np.zeros(3), atol=1e-15)
        ref_centers = np.loadtxt(output / f'{reference}.mlwfCenters', ndmin=2)[:, 3:6]
        for name in [reference] + [j['name'] for j in jobs]:
            if name in states:
                continue
            check_source(directory, name)
            identity = output/f'{name}.sources.json'
            if not identity.exists():
                raise ValueError(f'{name}: run this workflow to verify the Wannier source state first')
            expected = json.loads(identity.read_text())['files']
            if fingerprint([Path(p) for p in expected]) != expected:
                raise ValueError(f'{name}: source state changed; rerun Wannierization')
            log = (output / f'{name}.out').read_text().split('Start date and time:')[-1]
            if not minimized(log, 'WannierMinimize:') or 'Done!' not in log:
                raise ValueError(f'{name}: Wannier minimization did not converge')
            rows = np.loadtxt(output / f'{name}.mlwfCenters', ndmin=2)
            if rows.shape != (4, 7) or not np.all(np.isfinite(rows)):
                raise ValueError(f'{name}: expected four full-precision occupied centers')
            r = lattice(directory / f'{name}.lattice')
            np.testing.assert_allclose(rows[:, :3], rows[:, 3:6] @ r.T, atol=1e-12)
            centers, images, order = matched_centers(rows[:, 3:6], ref_centers, r)
            data = read(directory / f'{name}.polarization')
            np.testing.assert_allclose(data['lattice'].reshape(3, 3), r, atol=1e-12)
            np.testing.assert_allclose(data['spin_weight'], [2])
            branch = data.get('polarization_branch', np.zeros(3))
            np.testing.assert_array_equal(branch, ref_data.get('polarization_branch', np.zeros(3)))
            ions = ionic_positions(positions_file(directory, name))
            el, ion, total = polarization(centers, ions, r, branch)
            np.testing.assert_allclose(ion, data['ionic'], atol=1e-12)
            gradient_lines = [s for s in log.splitlines() if '|grad|_K:' in s]
            states[name] = dict(centers_lattice=centers.tolist(), center_image_shifts=images.tolist(),
                               center_source_indices=order.tolist(),
                               spreads_bohr2=rows[order, 6].tolist(), lattice=r.tolist(),
                               ionic_positions_lattice=ions.tolist(), spin_weight=2,
                               volume_bohr3=float(np.linalg.det(r)), electronic=el.tolist(),
                               ionic=ion.tolist(), branch=branch.tolist(), total=total.tolist(),
                               field=data['field'].tolist(), last_iteration=gradient_lines[-1],
                               gradient_norm=float(gradient_lines[-1].split('|grad|_K:')[1].split()[0]))
    results = []
    for mode in modes:
        reference = 'cell-zero' if mode == 'cell-relaxed' else 'zero'
        ref = read(directory / f'{reference}.polarization')
        for h in metadata['fields']:
            tensor = np.full((3, 3), np.nan)
            berry = tensor.copy()
            for axis in metadata['axes']:
                d = 'xyz'.index(axis)
                names = [next(j['name'] for j in metadata['jobs'] if j['mode'] == mode
                              and j['amplitude'] == h and j['axis'] == axis and j['sign'] == sign)
                         for sign in (-1, 1)]
                for name, sign in zip(names, (-1, 1)):
                    np.testing.assert_allclose(states[name]['field'], sign*h*np.eye(3)[d], atol=1e-15)
                tensor[:, d] = np.eye(3)[:, d] + 4*np.pi*(np.array(states[names[1]]['total'])
                                 - np.array(states[names[0]]['total']))/(2*h)
                deltas = [aligned_change(read(directory / f'{n}.polarization'), ref,
                                         np.array(metadata['lattice']), np.array(metadata['mesh']))[0]
                          for n in names]
                berry[:, d] = np.eye(3)[:, d] + 4*np.pi*(deltas[1]-deltas[0])/(2*h)
            def values(a):
                return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
            results.append(dict(mode=mode, amplitude=h, reference=reference,
                                dielectric=values(tensor), berry_dielectric=values(berry),
                                wannier_minus_berry=values(tensor-berry)))
            print(f'{mode}, h={h:g}: Wannier epsilon\n{tensor}\nBerry epsilon\n{berry}')
    result = dict(mesh=metadata['mesh'], cutoff=metadata['cutoff'],
                  localization_measure='FiniteDifference',
                  polarization_units='e/bohr^2', field_units='Hartree/(e bohr)',
                  formula='P = (4 sum_Si R_I - 2 sum_occupied r_n + R branch)/volume',
                  difference='epsilon_ab = delta_ab + 4*pi*(P_a(+h)-P_a(-h))/(2*h)',
                  states=states, response=results)
    (output / 'dielectric-wannier.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='completed run.py kN directory')
    parser.add_argument('--build', type=Path, default=Path(__file__).resolve().parents[2]/'build')
    parser.add_argument('--output', type=Path, help='default: DIRECTORY/wannier-fd')
    parser.add_argument('--modes', choices=['clamped', 'relaxed', 'cell-relaxed'], nargs='+')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--spread-threshold', type=float, default=0.)
    parser.add_argument('--gradient-threshold', type=float, default=5e-7)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--run', action='store_true', help='execute prepared inputs and analyze')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse identical converged inputs')
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 1 or args.spread_threshold < 0 or args.gradient_threshold <= 0:
        parser.error('require positive threads, iterations and gradient threshold; nonnegative spread threshold')
    directory = args.directory.resolve()
    output = args.output.resolve() if args.output else directory/'wannier-fd'
    metadata = json.loads((directory/'workflow.json').read_text())
    modes = args.modes or metadata.get('modes', ['clamped'])
    if any(m not in metadata.get('modes', ['clamped']) for m in modes):
        parser.error('requested mode was not computed')
    if args.analyze_only:
        analyze(directory, output, metadata, modes)
        return
    output.mkdir(exist_ok=True, parents=True)
    suffix = '_gpu' if args.gpu else ''
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=str(args.threads))
    def execute(binary, name):
        subprocess.run([str(args.build.resolve()/(binary+suffix)), '-i', str(output/f'{name}.in'),
                        '-o', str(output/f'{name}.out'), '-c', str(args.threads)],
                       cwd=output, env=env, check=True)
    names = sorted(set(['cell-zero' if m == 'cell-relaxed' else 'zero' for m in modes]
                       + [j['name'] for j in metadata['jobs'] if j['mode'] in modes]))
    for name in names:
        check_source(directory, name)
        r = lattice(directory/f'{name}.lattice')
        size = compatible_bases(np.array(metadata['lattice']), r, metadata['mesh'], metadata['cutoff'])
        if (directory/f'{name}.wfns').stat().st_size != size:
            raise ValueError(f'{name}: incompatible occupied wavefunction dimensions')
        common = (f'include {directory}/common.in\ninclude {directory}/{name}.lattice\n'
                  f'include {positions_file(directory, name)}\nelec-n-bands 4\n')
        sources = [directory/f'{name}.wfns', directory/f'{name}.polarization',
                   directory/f'{name}.lattice', positions_file(directory, name), directory/'common.in']
        eigs = directory/f'{name}.eigenvals'
        options = []
        if not eigs.exists():
            # Obtain projected DFT eigenvalues without optimizing the field state.
            eig_name = name+'-eigs'
            data = read(directory/f'{name}.polarization')
            field = ' '.join(f'{v:.17g}' for v in data['field'])
            branch = ' '.join(str(int(v)) for v in data.get('polarization_branch', np.zeros(3)))
            eig_input = (common + f'wavefunction read {directory}/{name}.wfns\n'
                         f'berry-phase-field {field}\nberry-phase-branch {branch}\n'
                         f'electronic-minimize nIterations 0\ndump-name {output}/{eig_name}.$VAR\n'
                         'dump End BandEigs\n')
            eig_inp = output/f'{eig_name}.in'
            unchanged = eig_inp.exists() and eig_inp.read_text() == eig_input
            eig_inp.write_text(eig_input)
            eigs = output/f'{eig_name}.eigenvals'
            if args.run and (not unchanged or not eigs.exists()
                             or max(p.stat().st_mtime_ns for p in sources) > eigs.stat().st_mtime_ns):
                execute('jdftx', eig_name)
            options += ['eigsOverride', str(eigs)]
        text = (common + f'wannier-initial-state {directory}/{name}.$VAR\n'
                f'wannier-dump-name {output}/{name}.$VAR\n')
        if args.resume:
            options += ['loadRotations', 'yes']
        if options:
            text += 'wannier '+' '.join(options)+'\n'
        for center in CENTERS:
            text += 'wannier-center Gaussian '+' '.join(f'{v:g}' for v in center)+'\n'
        text += (f'wannier-minimize nIterations {args.iterations} energyDiffThreshold '
                 f'{args.spread_threshold:g} knormThreshold {args.gradient_threshold:g}\n')
        inp = output/f'{name}.in'
        old = inp.read_text() if inp.exists() else None
        log_path = output/f'{name}.out'
        signature_path = output/f'{name}.sources.json'
        signature = dict(input=text, files=fingerprint(sources+[eigs])) if args.run else None
        previous = json.loads(signature_path.read_text()) if signature_path.exists() else None
        done = (old == text and log_path.exists() and (output/f'{name}.mlwfCenters').exists()
                and signature == previous and minimized(log_path.read_text(), 'WannierMinimize:')
                and 'Done!' in log_path.read_text())
        inp.write_text(text)
        if args.run and not (args.resume and done):
            execute('wannier', name)
            if not minimized(log_path.read_text(), 'WannierMinimize:') or 'Done!' not in log_path.read_text():
                raise ValueError(f'{name}: Wannier minimization did not converge')
            signature_path.write_text(json.dumps(signature, indent=2)+'\n')
        status = 'reused' if args.resume and done else 'converged' if args.run else 'prepared'
        print(f'{name}: {status}')
    if args.run:
        analyze(directory, output, metadata, modes)


if __name__ == '__main__':
    main()
