#!/usr/bin/env python3
"""Self-consistent isolated Nb vacancy and continuum Wannier maps at 2 K."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'jdftx/scripts'))
from me_postprocess.embedded import (BulkHost, read_hopping, matsubara_quadrature,
    prepare_embedding, solve_embedding, anomalous_matrix, perturbed_green,
    SpatialWannier, project_diagonal)
from me_postprocess.embedded import matsubara_tail
from me_postprocess import KB_EV
from me_postprocess.backend import host
from me_postprocess.wannier import read_metadata
from me_postprocess.cli import write_json
from plot_bands import path_definition


def sphere(lattice, radius_angstrom):
    bound = int(np.ceil(radius_angstrom/np.linalg.svd(lattice*.529177210903)[1].min()))+1
    cells = np.indices((2*bound+1,)*3).reshape(3, -1).T-bound
    return cells[np.linalg.norm(cells@lattice.T*.529177210903, axis=1) <= radius_angstrom+1e-8]


def fingerprint(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def continuum_figures(out, axis, pair, pair_bulk, ldos, clean_ldos, energies, eta, radius, mesh,
                      atomic_positions_angstrom=None):
    """Replot the saved continuum.npz arrays without rerunning Green functions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shape = (len(axis), len(axis))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    pair_scale = max(float(abs(pair).max()), float(abs(pair_bulk).max()), 1e-20)
    for a, field, title in zip(axes[0], [abs(pair).reshape(shape), abs(pair_bulk).reshape(shape),
                                       abs(pair).reshape(shape)-abs(pair_bulk).reshape(shape)],
                               ['Vacancy |F(r,r)|', 'Bulk |F(r,r)|', 'Change in |F(r,r)|']):
        change = 'Change' in title
        limit = max(float(abs(field).max()), 1e-20) if change else pair_scale
        image = a.pcolormesh(axis, axis, field.T, shading='auto', cmap='coolwarm' if change else 'viridis',
                            vmin=-limit if change else 0, vmax=limit)
        fig.colorbar(image, ax=a, label='bohr⁻³'); a.set_title(title)
    for a, energy, field in zip(axes[1], energies, ldos-clean_ldos):
        limit = max(float(abs(field).max()), 1e-20)
        image = a.pcolormesh(axis, axis, field.T, shading='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
        fig.colorbar(image, ax=a, label='eV⁻¹ bohr⁻³ / spin'); a.set_title(f'LDOS change, E = {energy*1000:g} meV')
    for a in axes.ravel():
        a.set(xlabel='x (Å)', ylabel='y (Å)', aspect='equal')
        if atomic_positions_angstrom is not None:
            positions = np.asarray(atomic_positions_angstrom)
            inside = ((abs(positions[:, 2]) < 1e-6) & (abs(positions[:, 0]) <= axis[-1]) &
                      (abs(positions[:, 1]) <= axis[-1]) & (np.linalg.norm(positions, axis=1) > 1e-6))
            a.scatter(positions[inside, 0], positions[inside, 1], marker='o', s=32,
                      facecolors='none', edgecolors='black', linewidths=.8)
        a.plot(0, 0, 'kx', ms=6)
        a.set_xlim(axis[0], axis[-1]); a.set_ylim(axis[0], axis[-1])
    fig.suptitle('Embedded Nb vacancy at 2 K — coherent electron fields, Cartesian z = 0\n'
                 f'Pairing radius {radius:g} Å; host integration {mesh}³; LDOS η = {eta*1000:g} meV\n'
                 '○ Nb sites in the slice; × vacancy')
    fig.savefig(out/'continuum-profiles.png', dpi=180); fig.savefig(out/'continuum-profiles.pdf'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, default=Path('/home/tyler/Nb_DFPT/wannier.'))
    p.add_argument('--lattice', type=Path, default=Path('/home/tyler/Nb_DFPT/Nb.lattice'))
    p.add_argument('--calibration', type=Path, default=HERE/'results-vacancy-T2-final/pairing-calibration.json')
    p.add_argument('--output', type=Path, default=HERE/'results-vacancy-embedded-T2')
    p.add_argument('--continuum-output', type=Path, help='separate directory for a new viewing window')
    p.add_argument('--mesh', type=int, default=128)
    p.add_argument('--radius', type=float, default=6., help='self-consistent pairing radius in angstrom')
    p.add_argument('--observation-radius', type=float, default=9.)
    p.add_argument('--quadrature-order', type=int, default=10)
    p.add_argument('--tolerance', type=float, default=1e-8)
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--gpu-device', type=int, default=1)
    p.add_argument('--reuse', action='store_true', help='reuse a matching converged solution')
    p.add_argument('--skip-continuum', action='store_true')
    p.add_argument('--pixels', type=int, default=81)
    p.add_argument('--extent', type=float, default=2., help='Cartesian slice half width in angstrom')
    p.add_argument('--wannier-cutoff', type=float, default=10., help='localized Wannier cutoff in bohr')
    p.add_argument('--wannier-grid-factor', type=int, default=2)
    args = p.parse_args(); started = time.time()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())
    lattice = np.asarray(path_definition(args.lattice)['primitive_lattice_columns_bohr'])
    kfold, _ = read_metadata(str(args.prefix)+'out')
    cells, hopping = read_hopping(args.prefix, kfold)
    signatures = fingerprint([Path(str(args.prefix)+s) for s in
                              ('mlwfH', 'mlwfCellMap', 'mlwfCellWeights', 'mlwfU')]+[args.calibration, args.lattice])
    configuration = dict(mesh=args.mesh, radius_angstrom=args.radius, temperature_k=2.,
                         quadrature_order=args.quadrature_order, source_sha256=signatures)
    if (out/'configuration.json').exists() and json.loads((out/'configuration.json').read_text()) != configuration:
        raise ValueError('output contains a different embedded model; choose a new output directory')
    bulk = BulkHost(cells, hopping, [args.mesh]*3, calibration['mu_ev'], calibration['z0'],
                    calibration['bulk_gap_ev'], 2., args.backend, args.gpu_device)
    xp = bulk.backend.xp
    support = sphere(lattice, args.radius)
    surviving = support[np.any(support != 0, axis=1)]
    if len(surviving) == 0:
        raise ValueError('pairing radius must include surviving sites')
    frequencies, weights = matsubara_quadrature(2., args.quadrature_order)
    print(f'Embedded Nb: {len(surviving)} self-consistent sites; {args.mesh}^3 host integration; '
          f'{len(frequencies)} Matsubara nodes; {bulk.backend.info}', flush=True)
    saved = out/'solution.npz'
    reuse = args.reuse and saved.exists() and (out/'configuration.json').exists()
    if reuse and json.loads((out/'configuration.json').read_text()) != configuration:
        raise ValueError('saved embedded configuration differs; choose a new output directory')
    if reuse:
        with np.load(saved) as data:
            delta, f, history = data['delta_ev'], data['anomalous'], data['residual_ev']
        if history[-1] > args.tolerance:
            raise ValueError('saved solution does not satisfy requested tolerance')
    else:
        def progress(done, total):
            if done % 10 == 0 or done == total:
                print(f'Host cavity cache {done}/{total}', flush=True)
        cavity, reference = prepare_embedding(bulk, support, frequencies, progress, return_reference=True)
        delta, fdiag, history = solve_embedding(bulk, cavity, (frequencies, weights),
            tolerance=args.tolerance,
            progress=lambda i, r: print(f'Gap iteration {i}: residual {r:.6g} eV', flush=True))
        delta = delta.reshape(len(surviving), bulk.norb)
        f = anomalous_matrix(bulk, surviving, delta, cavity, reference, weights)
        error = float(np.max(abs(np.diag(f)-fdiag)))
        if error > 1e-8:
            raise RuntimeError(f'anomalous matrix diagonal mismatch: {error}')
        np.savez_compressed(saved, cells=surviving, delta_ev=delta, anomalous=f,
                            attraction_ev=bulk.attraction, residual_ev=history,
                            frequencies_ev=frequencies, frequency_weights=weights)
        write_json(out/'configuration.json', configuration)
        del cavity, reference
        if args.backend == 'cuda':
            xp.get_default_memory_pool().free_all_blocks()
    summary = dict(configuration, backend=bulk.backend.info, bulk_gap_ev=bulk.gap,
        mu_ev=calibration['mu_ev'], z0=bulk.z0,
        attraction_ev=bulk.attraction.tolist(), pairing_regularization='full bandwidth onsite; U refitted to same bulk gap',
        mean_gap_mev=float(delta.mean()*1000), gap_range_mev=[float(delta.min()*1000), float(delta.max()*1000)],
        final_residual_ev=float(history[-1]), iterations=len(history), sites=len(surviving),
        boundary='infinite superconducting host; finite pairing-update region',
        convergence_status='finite host mesh, pairing radius, quadrature and Wannier tails require convergence studies',
        fixed_parameters='bulk mu and Z0; no Hartree, structural relaxation or modified phonons')
    radius = np.linalg.norm(surviving@lattice.T*.529177210903, axis=1)
    np.savetxt(out/'site-profile.dat', np.column_stack((surviving, radius, delta*1000)),
               header='cell0 cell1 cell2 radius_A Delta_s_meV Delta_dz2_meV Delta_dxy_meV Delta_dxz_meV Delta_dx2y2_meV Delta_dyz_meV')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for a, label in enumerate(['s', 'dz2', 'dxy', 'dxz', 'dx2-y2', 'dyz']):
        ax.scatter(radius, delta[:, a]*1000, label=label, s=20)
    ax.axhline(bulk.gap*1000, color='black', ls='--', label='bulk')
    ax.set(xlabel='Distance from vacancy (Å)', ylabel='Onsite pairing field (meV)',
           title=f'Embedded Nb vacancy, 2 K — pairing radius {args.radius:g} Å')
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(out/'pairing-profile.png', dpi=180); fig.savefig(out/'pairing-profile.pdf'); plt.close(fig)
    if not args.skip_continuum:
        view_out = args.continuum_output or out
        view_out.mkdir(parents=True, exist_ok=True)
        if args.observation_radius < args.radius:
            raise ValueError('observation radius must contain pairing support')
        if args.observation_radius < np.sqrt(2)*args.extent+args.wannier_cutoff*.529177210903:
            raise ValueError('observation sphere must cover slice plus Wannier tail cutoff')
        orbitals = SpatialWannier(args.prefix, args.backend, args.gpu_device, args.wannier_grid_factor)
        if orbitals.norb != bulk.norb:
            raise ValueError('Wannier amplitudes and Hamiltonian orbital counts differ')
        summary['mlwfC_sha256'] = fingerprint([Path(str(args.prefix)+'mlwfC'), Path(str(args.prefix)+'mlwfC.header')])
        summary['wannier_physical_norms'] = orbitals.norms.tolist()
        print('Reconstructing 3D Wannier amplitudes', flush=True)
        orbitals.prepare()
        probe = np.array([[.3, -.7, 1.1], [1., 1., 0.]])
        summary['wannier_interpolation_max_abs_error'] = float(max(np.max(abs(orbitals.evaluate(a, probe)-orbitals.exact(a, probe))) for a in range(bulk.norb)))
        observation = sphere(lattice, args.observation_radius)
        if args.observation_radius < args.radius:
            raise ValueError('observation radius must contain pairing support')
        if args.observation_radius < np.sqrt(2)*args.extent+args.wannier_cutoff*.529177210903:
            raise ValueError('observation sphere must cover slice plus Wannier tail cutoff')
        axis = np.linspace(-args.extent, args.extent, args.pixels)
        x, y = np.meshgrid(axis, axis, indexing='ij')
        positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))/.529177210903
        w = orbitals.map(positions, observation, lattice, args.wannier_cutoff)
        del orbitals.splines
        n = len(observation)*bulk.norb
        fobs = bulk.pair_matrix(observation)
        print(f'Continuum pair amplitude: {len(observation)} observation cells, {len(frequencies)} nodes', flush=True)
        for j, (frequency, weight) in enumerate(zip(frequencies, weights)):
            g, combined, clean = perturbed_green(bulk, observation, surviving, delta, 1j*frequency, return_clean=True)
            if not np.array_equal(combined, observation):
                raise RuntimeError('observation support ordering mismatch')
            fobs -= 2*KB_EV*bulk.temperature*weight*(g[:n, n:]-clean[:n, n:]).real
            if j % 10 == 0:
                print(f'Continuum F {j+1}/{len(frequencies)}', flush=True)
        leading = np.zeros(n)
        lookup = {tuple(c): i for i, c in enumerate(observation)}
        for cell, field in zip(surviving, delta):
            i = lookup[tuple(cell)]*bulk.norb
            leading[i:i+bulk.norb] = field-bulk.gap
        vacancy_indices = lookup[(0, 0, 0)]*bulk.norb+np.arange(bulk.norb)
        leading[vacancy_indices] = -bulk.gap
        ix = xp.arange(n)
        fobs[ix, ix] += matsubara_tail(bulk.temperature, weights)*xp.asarray(leading)
        # The excluded orbitals have identically zero correlations.
        fobs[xp.asarray(vacancy_indices), :] = 0
        fobs[:, xp.asarray(vacancy_indices)] = 0
        retained_indices = np.concatenate([lookup[tuple(c)]*bulk.norb+np.arange(bulk.norb) for c in surviving])
        fcheck = host(fobs[xp.ix_(xp.asarray(retained_indices), xp.asarray(retained_indices))])
        summary['continuum_support_anomalous_error'] = float(np.max(abs(fcheck-f)))
        if summary['continuum_support_anomalous_error'] > 1e-8:
            raise RuntimeError('observation and pairing-support anomalous matrices disagree')
        f0 = bulk.pair_matrix(observation)
        pair = project_diagonal(w, fobs, False, backend=bulk.backend)/bulk.z0
        pair_bulk = project_diagonal(w, f0, False, backend=bulk.backend)/bulk.z0
        energies = np.array([0., .0015, .0025]); eta = .00005
        ldos, clean_ldos = [], []
        for energy in energies:
            g, _, clean = perturbed_green(bulk, observation, surviving, delta, energy+1j*eta, return_clean=True)
            for matrix, destination in ((g, ldos), (clean, clean_ldos)):
                gee = matrix[:n, :n]
                spectral = -(gee-gee.conj().T)/(2j*np.pi*bulk.z0)
                destination.append(project_diagonal(w, spectral, backend=bulk.backend).real.reshape(x.shape))
        ldos, clean_ldos = np.asarray(ldos), np.asarray(clean_ldos)
        atomic_positions = observation@lattice.T*.529177210903
        np.savez_compressed(view_out/'continuum.npz', x_angstrom=axis, y_angstrom=axis,
            z_angstrom=0., energy_ev=energies, eta_ev=eta, ldos=ldos, bulk_ldos=clean_ldos,
            pair_amplitude=pair.reshape(x.shape), bulk_pair_amplitude=pair_bulk.reshape(x.shape),
            observation_cells=observation, atomic_positions_angstrom=atomic_positions,
            wannier_cutoff_bohr=args.wannier_cutoff)
        continuum_figures(view_out, axis, pair, pair_bulk, ldos, clean_ldos, energies, eta, args.radius, args.mesh,
                          atomic_positions_angstrom=atomic_positions)
        summary.update(observation_radius_angstrom=args.observation_radius, wannier_cutoff_bohr=args.wannier_cutoff,
                       wannier_grid_factor=args.wannier_grid_factor,
                       viewing_extent_angstrom=args.extent, viewing_pixels=args.pixels,
                       solution_source=str(saved.resolve()),
                       ldos_eta_ev=eta, continuum_units='F: bohr^-3; LDOS: eV^-1 bohr^-3 per spin; coherent weights /Z0')
    summary['wall_seconds'] = time.time()-started
    write_json((args.continuum_output if args.continuum_output and not args.skip_continuum else out)/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
