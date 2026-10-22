"""Command-line stages and reproducible JSON configuration."""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
from . import KB_EV
from .wannier import WannierModel, mesh_points, read_metadata
from .spectral import calculate, moments, histogram
from .eliashberg import IsotropicME, allen_dynes, low_frequency_fit
from .bdg import isotropic_singlet, export_arrays, coherent_reduce, nambu_normal, negative_partners
from .backend import select_backend, host


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ('wannier_prefix', 'phonon_prefix', 'metadata_log', 'wannier_log', 'output'):
        if key in config:
            config[key] = str((path.parent/config[key]).resolve())
    required = ('wannier_prefix', 'phonon_prefix', 'metadata_log', 'phonon_supercell',
                'kmesh', 'qmesh', 'sigma_ev', 'mu_star', 'output')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'missing configuration keys: {missing}')
    return config


def model_from_config(config):
    kfold, chemical = read_metadata(config['metadata_log'])
    chemical = config.get('mu_ev', chemical)
    if chemical is None:
        raise ValueError('no chemical potential in log; set mu_ev in configuration')
    native = Path(config['phonon_prefix']+'phononHsub')
    coupling = Path(config['wannier_prefix']+'mlwfHePh')
    if native.exists() and coupling.stat().st_mtime_ns < native.stat().st_mtime_ns:
        raise ValueError('Wannier electron–phonon export predates phononHsub; regenerate it from the current checkpoint')
    if 'wannier_log' in config:
        wfold, _ = read_metadata(config['wannier_log'])
        if not np.array_equal(kfold, wfold):
            raise ValueError('Wannier and DFPT electronic meshes differ')
        text = Path(config['wannier_log']).read_text(errors='replace')
        matches = re.findall(r'phononSupercell\s+(\d+)\s+(\d+)\s+(\d+)', text)
        if not matches or not np.array_equal(np.array(matches[-1], int), config['phonon_supercell']):
            raise ValueError('Wannier phononSupercell differs from configured supercell')
    return WannierModel(config['wannier_prefix'], config['phonon_prefix'], kfold,
                        config['phonon_supercell'], chemical,
                        config.get('frequency_floor_ev', 1e-5),
                        config.get('imaginary_tolerance_ev', 1e-4),
                        config.get('backend', 'cpu'), config.get('gpu_device', 0))


def check_saved_spectrum(config):
    """Keep independently rerunnable stages tied to the inputs they consumed."""
    summary = json.loads((Path(config['output'])/'spectral-summary.json').read_text())
    saved = summary['configuration']
    keys = ('wannier_prefix', 'phonon_prefix', 'metadata_log', 'phonon_supercell',
            'kmesh', 'qmesh', 'sigma_ev', 'mu_ev')
    defaults = {'frequency_floor_ev': 1e-5, 'imaginary_tolerance_ev': 1e-4}
    for key in (*keys, *defaults):
        if config.get(key, defaults.get(key)) != saved.get(key, defaults.get(key)):
            raise ValueError(f'{key} changed since the spectrum was computed; rerun spectrum')
    for item in summary['input_files']:
        stat = Path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'{item["path"]} changed since the spectrum was computed; rerun spectrum')


def spectrum_stage(config):
    started = time.perf_counter()
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    model = model_from_config(config)
    print(f'Compute backend: {model.backend.info}', flush=True)
    k, q = mesh_points(config['kmesh']), mesh_points(config['qmesh'])
    def progress(i, total):
        if i == 1 or i == total or i % max(1, total//10) == 0:
            print(f'Fermi-surface integration: q {i}/{total}', flush=True)
    data = calculate(model, k, q, config['sigma_ev'], config.get('chunk_size', 64), progress)
    np.savez_compressed(out/'spectrum.npz', **data)
    np.savetxt(out/'a2f.dat', histogram(data['omega_ev'], data['line_weight_ev'], config.get('spectral_bins', 300)),
               header='phonon_energy_eV alpha2F_dimensionless cumulative_lambda; histogram only, solver uses exact discrete lines')
    summary = moments(data['omega_ev'], data['line_weight_ev'])
    summary.update(dos_per_spin_per_cell_ev=float(data['dos_per_spin_ev']),
                   lambda_by_sorted_mode=data['mode_lambda'].mean(axis=0).tolist(),
                   excluded_modes=int(data['excluded_modes']),
                   tc_allen_dynes_k=allen_dynes(data['omega_ev'], data['line_weight_ev'], config['mu_star']),
                   nw_orbitals=model.nw, nk=len(k), nq=len(q),
                   input_files=model.provenance(), configuration=config,
                   execution=dict(**model.backend.info, elapsed_seconds=time.perf_counter()-started))
    write_json(out/'spectral-summary.json', summary)
    print(f'lambda={summary["lambda_total"]:.6g}; Allen–Dynes Tc={summary["tc_allen_dynes_k"]:.6g} K', flush=True)
    return data


def solve_stage(config):
    started = time.perf_counter()
    out = Path(config['output'])
    check_saved_spectrum(config)
    with np.load(out/'spectrum.npz') as archive:
        omega, weight = archive['omega_ev'], archive['line_weight_ev']
        dos = float(archive['dos_per_spin_ev'])
        chemical = float(archive['mu_ev'])
    solver = IsotropicME(omega, weight, config['mu_star'], config.get('matsubara_cutoff_ev', 0.5),
                         config.get('coulomb_cutoff_ev', 0.2), config.get('max_frequencies', 2048),
                         config.get('backend', 'cpu'), config.get('gpu_device', 0))
    print(f'Compute backend: {solver.backend.info}', flush=True)
    low, high = config.get('tc_bracket_k', [1., 30.])
    tc = solver.critical_temperature(low, high, config.get('tc_tolerance_k', 0.01))
    print(f'Linearized ME Tc={tc:.6g} K', flush=True)
    rows = []
    for temperature in config.get('temperatures_k', [2., 4., 6.]):
        solution = solver.solve(temperature, config.get('mixing', 0.35),
                                config.get('nonlinear_tolerance_ev', 1e-10),
                                config.get('max_iterations', 6000))
        fit = low_frequency_fit(solution, config.get('fit_points', 4))
        solution.update(mu_star=np.array(config['mu_star']), mu_ev=np.array(chemical),
                        matsubara_cutoff_ev=np.array(solver.cutoff),
                        coulomb_cutoff_ev=np.array(solver.coulomb_cutoff))
        label = f'{temperature:g}'
        np.savez_compressed(out/f'me-T{label}.npz', **solution)
        row = dict(temperature_k=temperature, **fit,
                   delta_first_ev=float(solution['delta_ev'][0]),
                   free_energy_ev_per_cell=float(solution['free_energy_per_dos_ev2']*dos),
                   positive_matsubara_frequencies=len(solution['wn_ev']),
                   iterations=int(solution['iterations']), residual_ev=float(solution['residual_ev']),
                   two_delta_fit_over_kbtc=2*fit['delta0_ev']/(KB_EV*tc),
                   fit_frequency_range_ev=solution['wn_ev'][:config.get('fit_points', 4)].tolist())
        rows.append(row)
        print(f'T={temperature:g} K: Delta(iw0)={row["delta_first_ev"]*1e3:.6g} meV; Z fit={fit["z0"]:.6g}', flush=True)
    summary = dict(tc_me_k=tc, tc_allen_dynes_k=allen_dynes(omega, weight, config['mu_star']),
                   mu_star=config['mu_star'], matsubara_cutoff_ev=solver.cutoff,
                   coulomb_cutoff_ev=solver.coulomb_cutoff, temperatures=rows,
                   static_estimator='linear fit in positive Matsubara frequency squared; no real-axis continuation',
                   configuration=config,
                   execution=dict(**solver.backend.info, elapsed_seconds=time.perf_counter()-started))
    write_json(out/'me-summary.json', summary)
    return summary


def bdg_stage(config):
    out = Path(config['output'])
    check_saved_spectrum(config)
    model = model_from_config(config)
    k = mesh_points(config.get('bdg_mesh', config['kmesh']))
    h = host(model.h(k))
    summary = json.loads((out/'me-summary.json').read_text())
    for key, default in [('mu_star', None), ('matsubara_cutoff_ev', 0.5),
                         ('coulomb_cutoff_ev', 0.2), ('fit_points', 4)]:
        if config.get(key, default) != summary['configuration'].get(key, default):
            raise ValueError(f'{key} changed since the ME solution; rerun solve')
    for row in summary['temperatures']:
        t = row['temperature_k']
        label = f'{t:g}'
        bdg, residue_sqrt = isotropic_singlet(h, k, model.mu, row['z0'], row['phi0_ev'], model.backend)
        arrays, checks = export_arrays(bdg, k, config.get('bdg_mesh', config['kmesh']), model.backend)
        arrays.update(residue_sqrt=residue_sqrt, h_bare_orbital_ev=h,
                      mu_ev=np.array(model.mu), z0=np.array(row['z0']),
                      phi0_ev=np.array(row['phi0_ev']), temperature_k=np.array(t),
                      basis_order=np.array('electrons: orb0 up, orb0 down, ...; then matching holes at -k'),
                      pairing_model=np.array('assumed local orbital singlet in real Wannier TR gauge'),
                      static_estimator=np.array(summary['static_estimator']))
        np.savez_compressed(out/f'bdg-T{label}.npz', **arrays)
        write_json(out/f'bdg-T{label}-checks.json', checks)
        print(f'BdG T={t:g} K: dimension {bdg.shape[-1]}, max symmetry error {max(checks.values()):.3g} eV', flush=True)


def matrix_bdg_stage(source, destination, shape, backend='cpu', gpu_device=0):
    backend = select_backend(backend, gpu_device)
    with np.load(source) as data:
        k, h = data['kpoints'], data['h_ev']
        mu = float(data['mu_ev'])
        partner = negative_partners(k)
        static = nambu_normal(h, mu, partner)+data['sigma0_ev']
        metric = data['metric']
        bdg, s, min_metric = coherent_reduce(static, metric, partner, backend=backend)
    arrays, checks = export_arrays(bdg, k, shape, backend)
    arrays.update(residue_sqrt=s, mu_ev=np.array(mu),
                  basis_order=np.array('user-specified spinful Wannier order; matching holes at -k'))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    checks['minimum_metric_eigenvalue'] = min_metric
    checks['execution'] = backend.info
    write_json(str(destination)+'.json', checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'spectrum', 'solve', 'bdg'):
        stage = sub.add_parser(command)
        stage.add_argument('configuration', type=Path)
        stage.add_argument('--backend', choices=('cpu', 'cuda', 'auto'))
        stage.add_argument('--gpu-device', type=int)
        stage.add_argument('--output', type=Path, help='override the output directory')
        stage.add_argument('--chunk-size', type=int, help='k points per interpolation batch')
    stage = sub.add_parser('matrix-bdg', help='coherent reduction of supplied full Nambu self-energy and metric')
    stage.add_argument('input', type=Path)
    stage.add_argument('output', type=Path)
    stage.add_argument('--mesh', nargs=3, type=int, required=True)
    stage.add_argument('--backend', choices=('cpu', 'cuda', 'auto'), default='cpu')
    stage.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == 'matrix-bdg':
            matrix_bdg_stage(args.input, args.output, args.mesh, args.backend, args.gpu_device)
        else:
            config = load_config(args.configuration)
            for name in ('backend', 'gpu_device', 'chunk_size'):
                value = getattr(args, name)
                if value is not None:
                    config[name] = value
            if args.output is not None:
                config['output'] = str(args.output.resolve())
            for command, stage in [('spectrum', spectrum_stage), ('solve', solve_stage), ('bdg', bdg_stage)]:
                if args.command in ('run', command):
                    stage(config)
    except (OSError, ValueError, KeyError, MemoryError, np.linalg.LinAlgError) as error:
        parser.exit(1, f'migdalEliashberg: {error}\n')
