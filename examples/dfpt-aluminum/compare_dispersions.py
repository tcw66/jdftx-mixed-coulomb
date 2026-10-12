#!/usr/bin/env python3
"""Interpolate and compare saved silicon DFPT and finite-difference phonons in meV."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import load_tool


# Match jdftx/core/Units.h; sqrt(phononOmegaSq) is an atomic-unit energy.
HARTREE_MEV = 27211.38505


def calculate(directory, tool):
    files = sorted(directory.glob('*phononOmegaSq'))
    if len(files) != 1:
        raise ValueError(f'expected one phononOmegaSq file in {directory}')
    prefix = str(files[0])[:-len('phononOmegaSq')]
    qpoints = tool.read_text_array(directory / 'qpath.dat')
    path = tool.read_text_array(directory / 'qpath-distance.dat')
    if path.shape != (len(qpoints), 4) or not np.allclose(path[:, 1:], qpoints, atol=1e-12, rtol=0):
        raise ValueError(f'qpath-distance.dat does not match qpath.dat in {directory}')
    cells, force = tool.load(prefix)
    dynamical = tool.interpolate(cells, force, qpoints)
    eig = np.linalg.eigvalsh(dynamical)
    energies = np.sign(eig) * np.sqrt(np.abs(eig)) * HARTREE_MEV
    basis = tool.read_text_array(prefix + 'phononBasis', str)
    return path, energies, basis, len(cells), prefix


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dfpt', type=Path, default=base / 'results_dfpt')
    parser.add_argument('--fd', type=Path, default=base / 'results_FD')
    parser.add_argument('--metadata', type=Path, default=base / 'qpath.json')
    parser.add_argument('--out', type=Path, default=base / 'results_comparison')
    args = parser.parse_args()
    tool = load_tool(base.parents[1] / 'jdftx/scripts/phononDFPT')
    path, dfpt, basis, nc_dfpt, prefix_dfpt = calculate(args.dfpt.resolve(), tool)
    path_fd, fd, basis_fd, nc_fd, prefix_fd = calculate(args.fd.resolve(), tool)
    if path.shape != path_fd.shape or not np.allclose(path, path_fd, atol=1e-12, rtol=0):
        raise ValueError('the two calculations must use the same q path and distances')
    if dfpt.shape != fd.shape or not np.array_equal(basis[:, :2], basis_fd[:, :2]):
        raise ValueError('the two calculations must have matching phonon bases')
    metadata = json.loads(args.metadata.read_text())
    positions, labels = [], []
    for tick in metadata['ticks']:
        i, label = tick['index'], tick['label']
        if not np.allclose(path[i, 1:], metadata['points'][label], atol=1e-12, rtol=0):
            raise ValueError('path metadata does not match the supplied q points')
        if not np.isclose(path[i, 0], tick['distance'], atol=1e-12, rtol=0):
            raise ValueError('path metadata does not match the supplied path distances')
        text = r'$\Gamma$' if label == 'G' else label
        if positions and abs(path[i, 0] - positions[-1]) < 1e-12:
            labels[-1] += ' | ' + text
        else:
            positions.append(path[i, 0])
            labels.append(text)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    columns = ' '.join(f'mode{i+1}[meV]' for i in range(dfpt.shape[1]))
    header = 'distance[bohr^-1] q0 q1 q2 ' + columns + '; negative denotes an imaginary mode'
    for name, energies in [('dfpt', dfpt), ('finite-difference', fd)]:
        np.savetxt(out / f'{name}-dispersion-meV.dat', np.column_stack((path, energies)), header=header)
    difference = dfpt - fd
    np.savetxt(out / 'difference-meV.dat', np.column_stack((path, difference)),
               header='distance[bohr^-1] q0 q1 q2 DFPT-minus-FD: ' + columns)
    worst = np.unravel_index(np.abs(difference).argmax(), difference.shape)
    report = {
        'dfpt_prefix': prefix_dfpt, 'finite_difference_prefix': prefix_fd,
        'q_points': len(path), 'modes': dfpt.shape[1],
        'real_space_cells': {'dfpt': nc_dfpt, 'finite_difference': nc_fd},
        'hartree_to_meV': HARTREE_MEV,
        'branch_comparison': 'eigenvalues sorted independently at every q point',
        'rms_difference_meV': float(np.sqrt(np.mean(difference**2))),
        'max_absolute_difference_meV': float(np.abs(difference[worst])),
        'max_difference_q_fractional': path[worst[0], 1:].tolist(),
        'max_difference_mode_number': int(worst[1] + 1),
        'gamma_energies_meV': {'dfpt': dfpt[0].tolist(), 'finite_difference': fd[0].tolist()},
    }
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained')
    colors = ('#1764ab', '#d85b15')
    breaks = metadata['breaks'] + [len(path)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        ax.plot(path[start:stop, 0], dfpt[start:stop], color=colors[0], linewidth=1.6)
        ax.plot(path[start:stop, 0], fd[start:stop], color=colors[1], linewidth=1.3, linestyle='--')
    for position in positions:
        ax.axvline(position, color='0.83', linewidth=0.7, zorder=0)
    ax.axhline(0, color='0.45', linewidth=0.7, zorder=0)
    ax.set_xticks(positions, labels)
    ax.set_xlim(path[0, 0], path[-1, 0])
    ax.set_ylim(min(-1, float(min(dfpt.min(), fd.min())) - 1), max(dfpt.max(), fd.max()) + 5)
    ax.set_xlabel('Wavevector path')
    ax.set_ylabel(r'Phonon energy $\hbar\omega$ (meV)')
    ax.set_title('Aluminum phonon dispersion')
    ax.legend(handles=[Line2D([0], [0], color=colors[0], lw=1.6, label='DFPT'),
                       Line2D([0], [0], color=colors[1], lw=1.3, ls='--', label='Finite difference')],
              loc='lower center', framealpha=0.95)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'phonon-comparison-meV.{extension}', dpi=220)
    plt.close(fig)
    print(f'Wrote comparison plots and dispersion tables to {out}')
    print(f'RMS difference: {report["rms_difference_meV"]:.6f} meV; '
          f'maximum: {report["max_absolute_difference_meV"]:.6f} meV')


if __name__ == '__main__':
    main()
