#!/usr/bin/env python3
"""Plot BCC niobium phonons from the saved inputs used by compare_dispersions.py.

Requires NumPy and Matplotlib only; no imports from other JDFTx scripts.
Run with no arguments to plot the phonon.* files beside this script along
the BCC path Gamma-H-N-Gamma-P-H | P-N, using Nb.lattice for distances.
  python plot_dispersion.py
  python plot_dispersion.py --dfpt results_dfpt --fd results_FD
  python plot_dispersion.py results_dfpt --metadata qpath.json

Each calculation directory must contain one *phononOmegaSq file, its matching
*phononCellMap and *phononBasis. With --metadata, also supply qpath.dat and
qpath-distance.dat in each calculation directory, as in compare_dispersions.py.
Outputs are PNG/PDF plots and interpolated dispersion tables in meV.
Negative energies represent imaginary modes.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np


# Match compare_dispersions.py and jdftx/core/Units.h.
HARTREE_MEV = 27211.38505


def bcc_path(lattice_file, intervals):
    """Build a BCC path in the reciprocal basis of the actual Nb lattice.

    Cartesian points in units 2*pi/a: H=(0,1,0), N=(.5,.5,0),
    P=(.5,.5,.5). See https://doi.org/10.1038/s41524-019-0235-7.
    """
    text = '\n'.join(line.split('#', 1)[0] for line in lattice_file.read_text().splitlines())
    match = re.fullmatch(r'\s*lattice\s+([\s\S]+)', text)
    if not match:
        raise ValueError('expected an explicit lattice matrix in Nb.lattice')
    values = np.fromstring(match[1].replace('\\', ' '), sep=' ')
    if values.size != 9 or not np.isfinite(values).all():
        raise ValueError('Nb.lattice must contain nine finite matrix entries in bohr')
    lattice = values.reshape(3, 3)
    # JDFTx stores primitive direct vectors as matrix columns.
    a = 2 * np.mean(np.abs(lattice))
    expected = 0.5 * a * (np.ones((3, 3)) - 2 * np.eye(3))
    if a <= 0 or not np.allclose(lattice, expected, atol=1e-8, rtol=1e-6):
        raise ValueError('expected the BCC primitive lattice a/2*[[-1,1,1],[1,-1,1],[1,1,-1]]')
    cartesian = {'G': [0., 0., 0.], 'H': [0., 1., 0.],
                 'N': [.5, .5, 0.], 'P': [.5, .5, .5]}
    points = {name: (np.array(point) @ lattice / a).tolist()
              for name, point in cartesian.items()}
    reciprocal = 2 * np.pi * np.linalg.inv(lattice)
    paths = [['G', 'H', 'N', 'G', 'P', 'H'], ['P', 'N']]
    qs, distances, ticks, breaks = [], [], [], []
    total = 0.0
    for route in paths:
        breaks.append(len(qs))
        qs.append(points[route[0]])
        distances.append(total)
        ticks.append({'index': len(qs) - 1, 'label': route[0], 'distance': total})
        for start, stop in zip(route[:-1], route[1:]):
            q0, q1 = np.array(points[start]), np.array(points[stop])
            step = np.linalg.norm((q1 - q0) @ reciprocal) / intervals
            for i in range(1, intervals + 1):
                qs.append(q0 + (q1 - q0) * i / intervals)
                total += step
                distances.append(total)
            ticks.append({'index': len(qs) - 1, 'label': stop, 'distance': total})
    metadata = {'points': points, 'paths': paths, 'ticks': ticks, 'breaks': breaks,
                'lattice_bohr': a, 'primitive_lattice_columns_bohr': lattice.tolist(),
                'distance_units': 'bohr^-1', 'points_per_segment': intervals}
    return np.column_stack((distances, qs)), metadata


def read_text_array(path, dtype=float):
    lines = [line for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not lines:
        raise ValueError(f'no data in {path}')
    return np.loadtxt(lines, dtype=dtype, ndmin=2)


def calculate(directory, generated_path=None):
    files = sorted(directory.glob('*phononOmegaSq'))
    if len(files) != 1:
        raise ValueError(f'expected one phononOmegaSq file in {directory}')
    prefix = str(files[0])[:-len('phononOmegaSq')]
    if generated_path is None:
        qpoints = read_text_array(directory / 'qpath.dat')
        path = read_text_array(directory / 'qpath-distance.dat')
    else:
        path = generated_path
        qpoints = path[:, 1:]
    if qpoints.shape[1] != 3 or not np.isfinite(qpoints).all():
        raise ValueError('q points must have three finite coordinates')
    if (path.shape != (len(qpoints), 4) or not np.isfinite(path).all()
            or not np.allclose(path[:, 1:], qpoints, atol=1e-12, rtol=0)):
        raise ValueError(f'qpath-distance.dat does not match qpath.dat in {directory}')
    if np.any(np.diff(path[:, 0]) < 0) or path[-1, 0] <= path[0, 0]:
        raise ValueError('path distances must be nondecreasing with nonzero extent')

    cells = read_text_array(prefix + 'phononCellMap')[:, :3]
    basis = read_text_array(prefix + 'phononBasis', str)
    if len(basis) != 3 or not np.all(basis[:, 0] == 'Nb'):
        raise ValueError('expected three displacement modes for one primitive-cell Nb atom')
    nmodes = len(basis)
    raw = np.fromfile(prefix + 'phononOmegaSq', dtype='<f8')
    if raw.size != len(cells) * nmodes * nmodes:
        raise ValueError('phononOmegaSq size does not match cell map and basis')
    if (cells.shape[1] != 3 or not np.isfinite(cells).all()
            or not np.allclose(cells, np.rint(cells))):
        raise ValueError('cell map must contain integer lattice coordinates')
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite force constants')
    # JDFTx writes each real-space matrix in column-major order.
    matrices = raw.reshape(len(cells), nmodes, nmodes).transpose(0, 2, 1)
    phase = np.exp(2j * np.pi * qpoints @ cells.T)
    dynamical = np.einsum('qr,rij->qij', phase, matrices)
    adjoint = dynamical.conj().transpose(0, 2, 1)
    error = np.max(np.abs(dynamical - adjoint))
    scale = max(1e-30, np.max(np.abs(dynamical)))
    if error > 1e-6 * scale:
        raise ValueError(f'interpolated dynamical matrix is not Hermitian '
                         f'(relative error {error / scale:g})')
    eig = np.linalg.eigvalsh(0.5 * (dynamical + adjoint))
    energies = np.sign(eig) * np.sqrt(np.abs(eig)) * HARTREE_MEV
    return path, energies, basis


def path_labels(path, metadata):
    positions, labels = [], []
    for tick in metadata['ticks']:
        index, label = tick['index'], tick['label']
        if not isinstance(index, int) or not 0 <= index < len(path):
            raise ValueError('metadata tick index is outside the q path')
        if not np.allclose(path[index, 1:], metadata['points'][label], atol=1e-12, rtol=0):
            raise ValueError('path metadata does not match the supplied q points')
        if not np.isclose(path[index, 0], tick['distance'], atol=1e-12, rtol=0):
            raise ValueError('path metadata does not match the supplied path distances')
        text = r'$\Gamma$' if label == 'G' else label
        if positions and abs(path[index, 0] - positions[-1]) < 1e-12:
            labels[-1] += ' | ' + text
        else:
            positions.append(path[index, 0])
            labels.append(text)
    breaks = metadata['breaks']
    if (not breaks or breaks[0] != 0
            or any(not isinstance(i, int) or not 0 <= i < len(path) for i in breaks)
            or any(b <= a for a, b in zip(breaks, breaks[1:]))):
        raise ValueError('metadata breaks must be increasing indices starting at zero')
    return positions, labels, breaks + [len(path)]


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', nargs='?', type=Path,
                        help='single calculation directory (default: script directory)')
    parser.add_argument('--dfpt', type=Path, help='DFPT calculation directory')
    parser.add_argument('--fd', type=Path, help='finite-difference calculation directory')
    parser.add_argument('--metadata', type=Path,
                        help='use existing q-path files and JSON instead of generating a BCC path')
    parser.add_argument('--lattice', type=Path, default=base / 'Nb.lattice',
                        help='BCC primitive lattice file in bohr (default: adjacent Nb.lattice)')
    parser.add_argument('--points-per-segment', type=int, default=60)
    parser.add_argument('--out', type=Path, default=base / 'results_dispersion',
                        help='output directory (default: results_dispersion beside script)')
    parser.add_argument('--title', default='Niobium (BCC) phonon dispersion')
    args = parser.parse_args()
    if args.directory and (args.dfpt or args.fd):
        parser.error('use directory alone, or --dfpt/--fd')
    if args.points_per_segment < 1:
        parser.error('--points-per-segment must be positive')
    inputs = []
    if args.dfpt:
        inputs.append(('dfpt', 'DFPT', args.dfpt))
    if args.fd:
        inputs.append(('finite-difference', 'Finite difference', args.fd))
    if not inputs:
        directory = args.directory or base
        inputs.append(('phonon', 'Nb', directory))

    try:
        if args.metadata:
            generated_path = None
            metadata = json.loads(args.metadata.read_text())
        else:
            generated_path, metadata = bcc_path(args.lattice, args.points_per_segment)
        datasets = [(name, label, calculate(directory.resolve(), generated_path))
                    for name, label, directory in inputs]
        path, _, basis = datasets[0][2]
        for _, _, (other_path, other_energies, other_basis) in datasets[1:]:
            if (path.shape != other_path.shape
                    or not np.allclose(path, other_path, atol=1e-12, rtol=0)):
                raise ValueError('the calculations must use the same q path and distances')
            if (other_energies.shape != datasets[0][2][1].shape
                    or not np.array_equal(basis[:, :2], other_basis[:, :2])):
                raise ValueError('the calculations must have matching phonon bases')
        positions, labels, breaks = path_labels(path, metadata)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained')
        handles = []
        minimum, maximum = 0.0, 0.0
        styles = [('#1764ab', '-'), ('#d85b15', '--')]
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / 'qpath.dat', path[:, 1:], header='fractional q0 q1 q2')
        np.savetxt(out / 'qpath-distance.dat', path, header='distance[bohr^-1] q0 q1 q2')
        (out / 'qpath.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for (name, label, (_, energies, _)), (color, linestyle) in zip(datasets, styles):
            for start, stop in zip(breaks[:-1], breaks[1:]):
                ax.plot(path[start:stop, 0], energies[start:stop], color=color,
                        linewidth=1.5, linestyle=linestyle)
            handles.append(Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=label))
            minimum = min(minimum, float(energies.min()))
            maximum = max(maximum, float(energies.max()))
            columns = ' '.join(f'mode{i+1}[meV]' for i in range(energies.shape[1]))
            np.savetxt(out / f'{name}-dispersion-meV.dat', np.column_stack((path, energies)),
                       header='distance[bohr^-1] q0 q1 q2 ' + columns
                       + '; negative denotes an imaginary mode')
        for position in positions:
            ax.axvline(position, color='0.83', linewidth=0.7, zorder=0)
        ax.axhline(0, color='0.45', linewidth=0.7, zorder=0)
        ax.set_xticks(positions, labels)
        ax.set_xlim(path[0, 0], path[-1, 0])
        ax.set_ylim(min(-1, minimum - 1), maximum + 5)
        ax.set_xlabel('Wavevector path')
        ax.set_ylabel(r'Phonon energy $\hbar\omega$ (meV)')
        ax.set_title(args.title)
        ax.legend(handles=handles, loc='lower center', framealpha=0.95)
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'phonon-dispersion-meV.{extension}', dpi=220)
        plt.close(fig)
        print(f'Wrote dispersion plots and tables to {out}')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
