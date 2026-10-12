#!/usr/bin/env python3
"""Plot graphene DFPT phonons along Gamma-M-K-Gamma from native force matrices."""
import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

BOHR_ANGSTROM = 0.529177210903
HARTREE_MEV = 27211.386245988
HARTREE_CM = 219474.6313705


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default=str(here/'phonon.'))
    parser.add_argument('--lattice', type=Path, default=here/'latMin.lattice')
    parser.add_argument('--output', type=Path, default=here/'results_dispersion')
    parser.add_argument('--points-per-segment', type=int, default=150)
    args = parser.parse_args()
    if args.points_per_segment < 2:
        parser.error('--points-per-segment must be at least 2')
    loader = importlib.machinery.SourceFileLoader(
        'phononDFPT', str(here.parents[1]/'jdftx/scripts/phononDFPT'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    phonon = importlib.util.module_from_spec(spec)
    loader.exec_module(phonon)
    # JDFTx lattice files store Cartesian vectors as matrix columns in bohrs.
    tokens = ' '.join(line.split('#')[0] for line in args.lattice.read_text().splitlines()).replace('\\', ' ').split()
    if tokens[0] != 'lattice' or len(tokens) != 10:
        raise ValueError('expected an explicit 3x3 JDFTx lattice matrix')
    lattice = np.array([float(x) for x in tokens[1:]]).reshape(3, 3)
    reciprocal = 2*np.pi*np.linalg.inv(lattice*BOHR_ANGSTROM).T
    vertices = np.array([[0., 0., 0.], [.5, 0., 0.], [1/3, 1/3, 0.], [0., 0., 0.]])
    qs = np.concatenate([np.linspace(a, b, args.points_per_segment, endpoint=False)
                         for a, b in zip(vertices[:-1], vertices[1:])] + [vertices[-1:]])
    cartesian = qs @ reciprocal.T
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    ticks = np.arange(4)*args.points_per_segment
    cells, force = phonon.load(args.prefix)
    dyn = phonon.interpolate(cells, force, qs)
    eigs, vectors = np.linalg.eigh(dyn)
    signed_omega = np.sign(eigs)*np.sqrt(np.abs(eigs))
    energy = signed_omega*HARTREE_MEV
    frequency = signed_omega*HARTREE_CM
    normal = np.cross(lattice[:, 0], lattice[:, 1])
    normal /= np.linalg.norm(normal)
    # Eigenvectors are mass-weighted Cartesian polarizations; graphene atoms
    # have equal masses, so normal components directly classify ZA/ZO motion.
    polarizations = vectors.reshape(len(qs), -1, 3, vectors.shape[-1])
    out_of_plane = np.sum(np.abs(np.einsum('d,qadv->qav', normal, polarizations))**2, axis=1)
    fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    colors = ['#2368ad', '#d77620']
    for band in range(energy.shape[1]):
        points = np.column_stack((distance, energy[:, band]))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        is_normal = (out_of_plane[:-1, band]+out_of_plane[1:, band]) > 1.
        ax.add_collection(LineCollection(segments, colors=np.where(is_normal, colors[1], colors[0]), linewidths=1.7, capstyle='round'))
    ax.set_xlim(distance[0], distance[-1])
    ax.set_ylim(min(-4., energy.min()-4.), energy.max()+10.)
    ax.set_xticks(distance[ticks], [r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
    ax.set_ylabel('Phonon energy (meV)')
    ax.set_title('Graphene phonon dispersion', loc='left', fontsize=14, weight='bold', pad=12)
    for x in distance[ticks]:
        ax.axvline(x, color='0.8', linewidth=.8, zorder=0)
    ax.axhline(0., color='0.55', linewidth=.8, zorder=0)
    ax.grid(axis='y', color='0.91', linewidth=.7)
    ax.set_axisbelow(True)
    right = ax.secondary_yaxis('right', functions=(lambda x: x*HARTREE_CM/HARTREE_MEV,
                                                 lambda x: x*HARTREE_MEV/HARTREE_CM))
    right.set_ylabel(r'Frequency (cm$^{-1}$)')
    ax.legend([Line2D([0], [0], color=c, lw=2) for c in colors],
              ['In-plane', 'Out-of-plane'], loc='upper center', bbox_to_anchor=(.5, -.09), ncol=2, frameon=False)
    fig.text(.13, -.12, 'DFPT: 6×6×1 q mesh · 12×12×1 electronic mesh · slab Coulomb embedding',
             fontsize=9, color='0.4')
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'graphene_phonon_dispersion.{suffix}', dpi=220, bbox_inches='tight')
    np.savetxt(args.output/'qpath.dat', qs, header='Gamma-M-K-Gamma; fractional reciprocal q')
    np.savetxt(args.output/'dispersion.dat', np.column_stack((distance, qs, energy, frequency)),
               header='distance[1/Angstrom] q0 q1 q2 energies[meV, six bands] frequencies[cm^-1, six bands]; negative denotes imaginary')
    np.savez_compressed(args.output/'dispersion.npz', qpoints=qs, distance=distance,
                        dynamical_matrices=dyn, omega_squared=eigs, eigenvectors=vectors,
                        energies_meV=energy, frequencies_cm=frequency, out_of_plane_weight=out_of_plane,
                        tick_indices=ticks, vertices=vertices)
    print(f'Wrote PNG, PDF, dispersion tables and eigenvectors to {args.output}')
    print(f'Minimum signed frequency: {frequency.min():.6g} cm^-1')


if __name__ == '__main__':
    main()
