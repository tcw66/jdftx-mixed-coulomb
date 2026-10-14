#!/usr/bin/env python3
"""Regenerate the silicon convergence figure from the committed numerical records."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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
    ax.axhline(12, color='#686868', linestyle='--', linewidth=1.5,
               label='Experimental reference ≈ 12')
    ax.set_title('Self-consistent dielectric response of silicon', pad=14)
    ax.set_xlabel('Inverse mesh size 1/N (N × N × N k points)')
    ax.set_ylabel(r'Relative dielectric constant $\epsilon_{zz}$')
    sizes = [16, 12, 8, 6, 4]
    ax.set_xticks([0]+[1/n for n in sizes])
    ax.set_xticklabels(['0']+[f'1/{n}' for n in sizes])
    ax.set_xlim(0, .265)
    ax.set_ylim(9.35, 13.18)
    ax.grid(axis='y', color='#e8e8e8', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False, fontsize=10.5)
    fig.text(.09, .045,
             '|Ez| = 0.0001 a.u.; PBE/SG15; cutoff = 30 Hartree. '
             'All static points use fixed-cell ionic relaxation.',
             fontsize=9.5, color='#555555', va='bottom')
    for extension in ('png', 'pdf'):
        path = directory/f'validation-convergence.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
