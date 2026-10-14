#!/usr/bin/env python3
"""Regenerate the silicon convergence figure from the committed numerical records."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def fit_limit(mesh, values, minimum=6, powers=(1,)):
    n = np.asarray(mesh, dtype=float)
    y = np.asarray(values, dtype=float)
    selected = n >= minimum
    x = 1/n[selected]
    design = np.column_stack([np.ones(len(x))]+[x**p for p in powers])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y[selected], rcond=None)
    if rank != len(powers)+1 or len(x) <= rank:
        raise ValueError('Extrapolation requires more points than fit parameters')
    residual = design @ coefficients-y[selected]
    return dict(mesh=n[selected].astype(int).tolist(), powers=list(powers),
                coefficients=coefficients.tolist(), epsilon_infinity=float(coefficients[0]),
                residual_rms=float(np.sqrt(np.mean(residual**2))))


def main():
    directory = Path(__file__).resolve().parent
    amplitude = 1e-4
    with (directory/'validation.csv').open(newline='') as stream:
        rows = sorted((r for r in csv.DictReader(stream) if float(r['field_au']) == amplitude),
                      key=lambda r: int(r['mesh']))
    cell = json.loads((directory/'validation-cell-k16.json').read_text())
    assert cell['mesh'] == [16, 16, 16]
    clamped = next(r for r in cell['responses']
                   if r['mode'] == 'clamped' and r['amplitude'] == amplitude)
    relaxed = next(r for r in cell['responses']
                   if r['mode'] == 'relaxed' and r['amplitude'] == amplitude)
    mesh = [int(row['mesh']) for row in rows]
    electronic = [float(row['epsilon_electronic_zz']) for row in rows]
    curvature = [float(row['epsilon_from_enthalpy_zz']) for row in rows]
    static = [float(row['epsilon_static_zz']) for row in rows]
    mesh.append(16)
    electronic.append(clamped['dielectric'][2][2])
    curvature.append(clamped['curvature']['z'])
    static.append(relaxed['dielectric'][2][2])
    inverse_mesh = 1/np.array(mesh, dtype=float)
    fits = {name: fit_limit(mesh, values) for name, values in
            [('electronic', electronic), ('enthalpy_curvature', curvature), ('static', static)]}
    sensitivity = [dict(label=label, **fit_limit(ns, ys, minimum, powers))
                   for label, ns, ys, minimum, powers in [
                       ('linear 1/N, all meshes', mesh, static, 4, (1,)),
                       ('linear 1/N, N >= 8', mesh, static, 8, (1,)),
                       ('linear 1/N, same geometry N=4..12', mesh[:-1], static[:-1], 4, (1,)),
                       ('quadratic 1/N, N >= 6', mesh, static, 6, (1, 2)),
                       ('linear 1/N^2, N >= 6', mesh, static, 6, (2,))]]
    result = dict(model='epsilon(N) = epsilon_infinity + A/N', amplitude_au=amplitude,
                  primary=fits, sensitivity=sensitivity,
                  caveat='Empirical fit: reference geometry differs at N=16; field and asymptotic mesh convergence are not established.')
    (directory/'validation-extrapolation.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, fit in fits.items():
        print(f'{name}: epsilon_infinity = {fit["epsilon_infinity"]:.8f}, residual RMS = {fit["residual_rms"]:.5f}')

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 18, 'axes.labelsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=.09, right=.975, top=.89, bottom=.18)
    ax.plot(inverse_mesh, electronic, '-o', color='#245c9f', linewidth=2.2, markersize=7,
            label='Electronic: polarization slope')
    ax.plot(inverse_mesh, curvature, 'o', color='#dd770b', markerfacecolor='none',
            markersize=10, markeredgewidth=1.6, label='Electronic: enthalpy curvature')
    ax.plot(inverse_mesh, static, 'x', color='#a93d45', markersize=8, markeredgewidth=1.5,
            label='Relaxed ions, fixed cell')
    fit = fits['static']
    grid = np.linspace(0, 1/6, 200)
    ax.plot(grid, fit['coefficients'][0]+fit['coefficients'][1]*grid,
            '--', color='#7a489a', linewidth=1.8, label='Linear 1/N fit, N ≥ 6')
    ax.plot(0, fit['epsilon_infinity'], 'D', color='#7a489a', markersize=7, clip_on=False)
    ax.annotate(rf'$N\to\infty$: $\epsilon_{{zz}}\approx{fit["epsilon_infinity"]:.3f}$ (linear fit)',
                xy=(0, fit['epsilon_infinity']), xytext=(.045, 13.90),
                fontsize=11, arrowprops={'arrowstyle': '-', 'color': '#7a489a'})
    ax.axhline(12, color='#686868', linestyle='--', linewidth=1.5,
               label='Experimental reference ≈ 12')
    ax.set_title('Self-consistent dielectric response of silicon', pad=14)
    ax.set_xlabel('Inverse mesh size 1/N (N × N × N k points)')
    ax.set_ylabel(r'Relative dielectric constant $\epsilon_{zz}$')
    sizes = [16, 12, 8, 6, 4]
    ax.set_xticks([0]+[1/n for n in sizes])
    ax.set_xticklabels(['0']+[f'1/{n}' for n in sizes])
    ax.set_xlim(0, .265)
    ax.set_ylim(9.35, 14.15)
    ax.grid(axis='y', color='#e8e8e8', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False, fontsize=10.5)
    fig.text(.09, .045,
             '|Ez| = 0.0001 a.u.; PBE/SG15; cutoff = 30 Hartree. '
             'Fixed-cell ionic relaxation.\n'
             'Empirical extrapolation; the reference lattice differs at N = 16.',
             fontsize=9.5, color='#555555', va='bottom')
    for extension in ('png', 'pdf'):
        path = directory/f'validation-convergence.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
