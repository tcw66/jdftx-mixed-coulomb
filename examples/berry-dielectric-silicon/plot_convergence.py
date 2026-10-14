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
    wannier = json.loads((directory/'validation-wannier-k16.json').read_text())
    assert cell['mesh'] == wannier['mesh'] == [16, 16, 16]
    berry_response = next(r for r in cell['responses']
                          if r['mode'] == 'cell-relaxed' and r['amplitude'] == amplitude)
    wannier_response = next(r for r in wannier['response']
                            if r['mode'] == 'cell-relaxed' and r['amplitude'] == amplitude)
    berry_value = berry_response['dielectric'][2][2]
    wannier_value = wannier_response['dielectric'][2][2]
    np.testing.assert_allclose(wannier_response['berry_dielectric'][2][2], berry_value, atol=1e-12)
    r = np.array(berry_response['reference_lattice'])
    lattice_constant = (4*np.linalg.det(r))**(1/3)
    mesh = [int(row['mesh']) for row in rows]
    electronic = [float(row['epsilon_electronic_zz']) for row in rows]
    curvature = [float(row['epsilon_from_enthalpy_zz']) for row in rows]
    static = [float(row['epsilon_static_zz']) for row in rows]

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 18, 'axes.labelsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=.09, right=.975, top=.89, bottom=.18)
    ax.plot(mesh, electronic, '-o', color='#245c9f', linewidth=2.2, markersize=7,
            label='Electronic: polarization slope')
    ax.plot(mesh, curvature, 'o', color='#dd770b', markerfacecolor='none',
            markersize=10, markeredgewidth=1.6, label='Electronic: enthalpy curvature')
    ax.plot(mesh, static, 'x', color='#a93d45', markersize=8, markeredgewidth=1.5,
            label='Relaxed ions, fixed cell')
    # This point uses a different reference geometry; keep it off the fixed-cell line.
    ax.plot(16, berry_value, '*', color='#7a489a', markersize=14,
            label='16³ Berry: relaxed cell + ions', zorder=5)
    ax.plot(16, wannier_value, 'D', color='#24876b', markerfacecolor='none',
            markersize=9, markeredgewidth=1.6,
            label='16³ Wannier: relaxed cell + ions', zorder=6)
    ax.axhline(12, color='#686868', linestyle='--', linewidth=1.5,
               label='Experimental reference ≈ 12')
    ax.annotate(f'16³, relaxed cell + ions\nBerry: {berry_value:.5f}\nWannier: {wannier_value:.5f}',
                xy=(16, berry_value), xytext=(10.3, 12.92), fontsize=11,
                va='top', arrowprops={'arrowstyle': '-', 'color': '#777777'},
                bbox={'boxstyle': 'round,pad=.4', 'facecolor': 'white', 'edgecolor': '#dddddd'})
    ax.set_title('Self-consistent dielectric response of silicon', pad=14)
    ax.set_xlabel('Uniform k mesh (N × N × N)')
    ax.set_ylabel(r'Relative dielectric constant $\epsilon_{zz}$')
    ax.set_xticks(mesh+[16])
    ax.set_xlim(3.4, 17.1)
    ax.set_ylim(9.35, 13.08)
    ax.grid(axis='y', color='#e8e8e8', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=10.5)
    fig.text(.09, .045,
             f'4³–12³: fixed a = 10.26 bohr; 16³: relaxed a = {lattice_constant:.6f} bohr.\n'
             '|Ez| = 0.0001 a.u.; PBE/SG15; cutoff = 30 Hartree. '
             'Different geometries: not a mesh-only comparison.',
             fontsize=9.5, color='#555555', va='bottom')
    for extension in ('png', 'pdf'):
        path = directory/f'validation-convergence.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
