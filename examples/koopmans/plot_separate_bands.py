#!/usr/bin/env python3
"""Overlay separate KI and original DFT Wannier bands and find their gaps.

DFT is recovered as H_DFT = H_KI - Delta_H_KI from each run, preserving
complex hoppings. No additional electronic calculation is needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Small Wannier matrices benefit from a single BLAS thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from interpolate import load, interpolate

ROOT = Path(__file__).resolve().parent
EV = 27.211386245988
SUFFIXES = ('koopmansFormat', 'mlwfHki', 'mlwfKI', 'mlwfCellMap',
            'mlwfCellWeights', 'koopmansBands', 'koopmansScreening')


def filename(prefix, suffix):
    return Path(str(prefix).rstrip('.') + '.' + suffix)


def evaluate(model, points):
    """Bound the Fourier-transform memory use, including dense gap searches."""
    points = np.asarray(points).reshape(-1, 3)
    return np.concatenate([interpolate(*model, points[i:i+128]) * EV
                           for i in range(0, len(points), 128)])


def edges(valence, conduction, points):
    vi = int(np.argmax(valence[:, -1]))
    ci = int(np.argmin(conduction[:, 0]))
    return {'gap_eV': float(conduction[ci, 0] - valence[vi, -1]),
            'VBM_eV': float(valence[vi, -1]), 'CBM_eV': float(conduction[ci, 0]),
            'VBM_k_fractional': points[vi].tolist(),
            'CBM_k_fractional': points[ci].tolist(),
            'minimum_sampled_direct_gap_eV': float(np.min(conduction[:, 0]-valence[:, -1]))}


def search_gap(models, points, refine):
    """Search the full periodic reciprocal cell, then refine multiple seeds."""
    values = [evaluate(model, points) for model in models]
    result = edges(*values, points)
    result['sampled_mesh_and_path_gap_eV'] = result['gap_eV']
    if refine:
        for model, bands, column, sign, label in zip(
                models, values, (-1, 0), (-1, 1), ('VBM', 'CBM')):
            energy = bands[:, column]
            best = int(np.argmin(sign*energy))
            best_value, best_k = float(energy[best]), points[best]
            seeds = []
            # Avoid using all starts on neighboring points in the same pocket.
            for index in np.argsort(sign*energy):
                candidate = points[index]
                if all(np.linalg.norm((candidate-k+.5) % 1-.5) > .12 for k in seeds):
                    seeds.append(candidate)
                if len(seeds) == 8:
                    break
            successes = 0
            for seed in seeds:
                optimum = minimize(
                    lambda k: sign*float(evaluate(model, [k])[0, column]),
                    seed, method='L-BFGS-B',
                    options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 200})
                successes += int(optimum.success)
                candidate_value = sign*float(optimum.fun)
                if np.isfinite(candidate_value) and sign*candidate_value < sign*best_value:
                    best_value, best_k = candidate_value, optimum.x
            result[label+'_eV'] = best_value
            result[label+'_k_fractional'] = (np.asarray(best_k) % 1).tolist()
            result[label+'_refinement_successes'] = successes
            result[label+'_refinement_starts'] = len(seeds)
        result['gap_eV'] = result['CBM_eV']-result['VBM_eV']
    return result


def read_models(prefix, folding):
    for suffix in SUFFIXES:
        path = filename(prefix, suffix)
        if not path.is_file():
            raise ValueError(f'Missing {path}. Wait for the {prefix.name} KI run to finish, '
                             'or select a completed dump prefix.')
    # Accept manual native logs as well as the workflow launcher's logs.
    logs = [filename(prefix.parent.parent/prefix.name, 'out'),
            filename(prefix.parent.parent/'logs'/prefix.name, 'out'),
            filename(prefix, 'out')]
    logs = [p for p in logs if p.is_file()]
    if logs:
        latest = max(logs, key=lambda p: p.stat().st_mtime)
        with latest.open('rb') as stream:
            stream.seek(max(0, latest.stat().st_size-4096))
            if not stream.read().rstrip().endswith(b'Done!'):
                raise ValueError(f'{latest}: calculation has not finished (no final Done!).')
    ki = load(prefix, folding, 'ki')
    pbe = load(prefix, folding, 'dft')
    native = np.loadtxt(filename(prefix, 'koopmansBands'), ndmin=2)
    calculated = evaluate(ki, native[:, :3])/EV
    if calculated.shape != native[:, 3:].shape:
        raise ValueError(f'{prefix}: native band count does not match the Hamiltonian')
    error = float(np.max(np.abs(calculated-native[:, 3:])))
    if error > 1e-8:
        raise ValueError(f'{prefix}: interpolation disagrees with native KI bands by {error:g} Ha')
    screening = np.loadtxt(filename(prefix, 'koopmansScreening'), ndmin=2)
    metadata = filename(prefix, 'koopmansScreening').read_text()
    spin = 'single_spin_charge_and_magnetic' if 'single_spin_charge_and_magnetic' in \
        metadata else 'unspecified/legacy'
    epsilon = next((float(line.split()[2].rstrip(';')) for line in metadata.splitlines()
                    if line.startswith('# epsilon_infinity ')), None)
    screening_mesh = next(([int(value.rstrip(';')) for value in line.split()[2:5]]
                           for line in metadata.splitlines()
                           if line.startswith('# screening_q_mesh ')), None)
    return ki, pbe, {'native_KI_max_error_Ha': error, 'spin_treatment': spin,
                     'functional_approximation': 'KI(2)',
                     'epsilon_infinity': epsilon,
                     'screening_q_mesh': screening_mesh,
                     'image_correction_applied': bool(np.any(np.abs(screening[:, 5]) > 1e-12)),
                     'negative_relaxed_curvature_orbitals':
                     screening[screening[:, 3] < 0, 0].astype(int).tolist()}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valence', type=Path, default=ROOT/'results/valence', help='Valence dump prefix')
    parser.add_argument('--conduction', type=Path, default=ROOT/'results/conduction', help='Conduction dump prefix')
    parser.add_argument('--fold', type=int, nargs=3, help='Electronic mesh; default: common.in')
    parser.add_argument('--gap-mesh', type=int, nargs=3, default=(20, 20, 20), metavar=('NX', 'NY', 'NZ'),
                        help='Uniform BZ search mesh, followed by local refinement (default: 20 20 20)')
    parser.add_argument('--no-refine', action='store_true', help='Report sampled BZ extrema without local refinement')
    parser.add_argument('--path', type=Path, default=ROOT/'paths/path.kpoints')
    parser.add_argument('--distance', type=Path, default=ROOT/'paths/path-distance.dat')
    parser.add_argument('--ticks', type=Path, default=ROOT/'paths/path-ticks.dat')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--dft-label', choices=('PBE','LDA-PW','LDA-PZ','DFT'), default='PBE',
                        help='Base functional used by these checkpoints (default PBE)')
    return parser.parse_args()


def main():
    args = arguments()
    dft_label = args.dft_label
    folding = args.fold
    if folding is None:
        folding = next((list(map(int, line.split()[1:4])) for line in
                        (ROOT/'common.in').read_text().splitlines()
                        if line.strip().startswith('kpoint-folding ')), None)
    if folding is None or any(n <= 0 for n in (*folding, *args.gap_mesh)):
        raise ValueError('Electronic and gap-search meshes must have three positive dimensions')
    points = np.loadtxt(args.path, ndmin=2)
    distance = np.loadtxt(args.distance, ndmin=1)
    ticks = np.loadtxt(args.ticks, dtype=str, skiprows=1, ndmin=2)
    if points.shape != (len(distance), 3) or np.any(np.diff(distance) < 0):
        raise ValueError('Path points and monotonically increasing distances must match')
    indices = ticks[:, 0].astype(int)
    np.testing.assert_allclose(points[indices], ticks[:, 3:6].astype(float), atol=1e-10, rtol=0)
    np.testing.assert_allclose(distance[indices], ticks[:, 1].astype(float), atol=1e-10, rtol=0)
    models, checks = {}, {}
    for name, prefix in [('valence', args.valence), ('conduction', args.conduction)]:
        ki, pbe, checks[name] = read_models(prefix, folding)
        models[name] = {'KI': ki, 'PBE': pbe}
    if checks['valence']['spin_treatment'] != checks['conduction']['spin_treatment']:
        raise ValueError('Valence and conduction results use different KI spin treatments')
    for setting in ('screening_q_mesh', 'epsilon_infinity'):
        valence_setting = checks['valence'][setting]
        conduction_setting = checks['conduction'][setting]
        # Older native dumps may lack metadata; retain their existing support.
        if valence_setting is None or conduction_setting is None:
            continue
        differs = (valence_setting != conduction_setting if setting == 'screening_q_mesh' else
                   not np.isclose(valence_setting, conduction_setting, rtol=1e-10, atol=1e-12))
        if differs:
            raise ValueError(f'Valence/conduction {setting} differs: {valence_setting} vs {conduction_setting}. '
                             'Select matching completed runs before reporting a combined gap.')
    axes = [np.arange(n)/n for n in args.gap_mesh]
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    search_points = np.concatenate((mesh, points))
    bands, gaps = {}, {}
    for method in ('KI', 'PBE'):
        pair = [models[name][method] for name in ('valence', 'conduction')]
        bands[method] = [evaluate(model, points) for model in pair]
        if not all(np.isfinite(data).all() for data in bands[method]):
            raise ValueError(f'Nonfinite {method} path energies')
        print(f'Searching interpolated {"KI(2)" if method == "KI" else dft_label} band edges on {tuple(args.gap_mesh)}...', flush=True)
        gaps[method] = {'path': edges(*bands[method], points),
                        'BZ_search': search_gap(pair, search_points, not args.no_refine)}
    reference = gaps['PBE']['BZ_search']['VBM_eV']
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for method in ('PBE', 'KI'):
        for name, data in zip(('valence', 'conduction'), bands[method]):
            for i in range(data.shape[1]):
                ax.plot(distance, data[:, i]-reference,
                        color='#666666' if method == 'PBE' else
                        ('#245da8' if name == 'valence' else '#cf592d'),
                        ls='--' if method == 'PBE' else '-', lw=1.2 if method == 'PBE' else 1.5,
                        label=(dft_label+' (original Wannier)' if method == 'PBE' else 'KI(2) '+name)
                        if i == 0 and (method != 'PBE' or name == 'valence') else None)
    for position in ticks[:, 1].astype(float):
        ax.axvline(position, color='#dddddd', lw=.7, zorder=0)
    ax.axhline(0, color='#aaaaaa', lw=.7, zorder=0)
    ax.set_xticks(ticks[:, 1].astype(float),
                 [r'$\Gamma$' if label == 'Gamma' else label for label in ticks[:, 2]])
    ax.set_xlim(distance[0], distance[-1])
    ax.set_xlabel('Wave vector')
    ax.set_ylabel(f'Energy relative to interpolated {dft_label} VBM (eV)')
    ax.set_title(f'Separate valence/conduction KI(2) and {dft_label} bands — '
                 + ' × '.join(map(str, folding))+' electronic mesh')
    ax.legend(loc='upper left')
    annotation = 'Interpolated BZ-search gaps\n' + '\n'.join(
        f'{"KI(2)" if method == "KI" else dft_label}: {gaps[method]["BZ_search"]["gap_eV"]:.3f} eV '
        f'(path: {gaps[method]["path"]["gap_eV"]:.3f} eV)' for method in ('KI', 'PBE'))
    ax.text(.98, .02, annotation, transform=ax.transAxes, ha='right', va='bottom',
            bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': .95, 'pad': 5})
    stem = output/'koopmans-bands-comparison'
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(stem)+suffix, dpi=220)
    plt.close(fig)
    counts = [data.shape[1] for data in bands['KI']]
    np.savetxt(str(stem)+'.dat', np.column_stack((distance, points,
        *(data-reference for method in ('KI', 'PBE') for data in bands[method]))),
        header=f'distance_bohr^-1 kx ky kz KI_valence({counts[0]}) KI_conduction({counts[1]}) '
               f'{dft_label}_valence({counts[0]}) {dft_label}_conduction({counts[1]}); '
               f'energies_eV; common_zero_eV={reference:.15g}')
    sources = [filename(prefix, suffix) for prefix in (args.valence, args.conduction) for suffix in SUFFIXES]
    sources += [args.path, args.distance, args.ticks]
    report = {'folding': folding, 'gap_search_mesh': list(args.gap_mesh),
              'KI_approximation': 'One-shot second-order KI(2); DFPT infinitesimal-occupation screening',
              'local_refinement': not args.no_refine, 'path_points': len(points),
              'gap_search_note': 'Mesh and path seeded BZ search; converge the search mesh and the underlying Wannier model.',
              'DFT_functional': dft_label,
              dft_label+'_source': f'Original projected {dft_label}: H_DFT = mlwfHki - mlwfKI; no extra DFT run',
              'energy_zero': f'Interpolated {dft_label} BZ-search VBM; same shift for all bands',
              'energy_zero_eV': reference,
              'gaps': {dft_label if name == 'PBE' else name: values for name, values in gaps.items()}, 'checks': checks,
              'sources_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    Path(str(stem)+'.json').write_text(json.dumps(report, indent=2)+'\n')
    for method in ('KI', 'PBE'):
        print(f'{dft_label if method == "PBE" else "KI(2)"}: interpolated BZ-search gap {gaps[method]["BZ_search"]["gap_eV"]:.6f} eV; '
              f'path gap {gaps[method]["path"]["gap_eV"]:.6f} eV')
    print(f'Plots, band table and gap report: {stem}.{{png,pdf,dat,json}}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f'Cannot plot bands: {error}')
