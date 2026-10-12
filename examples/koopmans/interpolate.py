#!/usr/bin/env python3
"""Interpolate native KI Hamiltonians, preserving complex Wannier hoppings.
Example: python interpolate.py conduction path.kpoints --fold 2 2 2 --ev -o bands.dat
Input kpoints: one row of three fractional reciprocal coordinates per point.
"""
import argparse
from pathlib import Path
import numpy as np


def load(prefix, folding, hamiltonian="ki"):
    prefix = str(prefix).rstrip('.')+'.'
    kind, cells, bands = Path(prefix+'koopmansFormat').read_text().split()
    cells, bands = int(cells), int(bands)
    if kind not in ('real64', 'complex128'):
        raise ValueError('Unknown KI Hamiltonian dtype: '+kind)
    if any(n <= 0 for n in folding) or np.prod(folding) != cells:
        raise ValueError('--fold must match the electronic kpoint-folding of the calculation')
    dtype = np.float64 if kind == 'real64' else np.complex128
    if hamiltonian == 'bare':
        # The legacy spinless dump is real64 even when KI is complex128.
        path = Path(prefix+'mlwfH')
        size = cells*bands*bands
        if path.stat().st_size == 8*size:
            dtype = np.float64
        elif path.stat().st_size == 16*size:
            dtype = np.complex128
        else:
            raise ValueError('Bare Hamiltonian size does not match the mesh and center count')
        reduced = np.fromfile(path, dtype=dtype).reshape(cells, bands, bands).swapaxes(1, 2)
    else:
        reduced = np.fromfile(prefix+'mlwfHki', dtype=dtype).reshape(cells, bands, bands).swapaxes(1, 2)
    if hamiltonian == 'dft':
        # Recover the frozen DFT block without losing complex hoppings in
        # the legacy spinless mlwfH export: H_DFT = H_KI - Delta_H_KI.
        correction = np.fromfile(prefix+'mlwfKI', dtype=dtype).reshape(cells, bands, bands).swapaxes(1, 2)
        reduced = reduced-correction
    elif hamiltonian not in ('ki', 'bare'):
        raise ValueError('Hamiltonian must be ki, dft or bare')
    cell_map = np.loadtxt(prefix+'mlwfCellMap', ndmin=2)[:, :3].astype(int)
    weights = np.fromfile(prefix+'mlwfCellWeights', dtype=np.float64).reshape(len(cell_map), bands, bands).swapaxes(1, 2)
    stride = np.array([folding[1]*folding[2], folding[2], 1])
    indices = (cell_map % np.array(folding))@stride
    return cell_map, weights*reduced[indices]


def interpolate(cell_map, hamiltonian, points):
    h = np.tensordot(np.exp(2j*np.pi*np.asarray(points)@cell_map.T), hamiltonian, axes=1)
    if not np.allclose(h, h.swapaxes(1, 2).conj(), atol=1e-9, rtol=1e-9):
        raise ValueError('Interpolated KI Hamiltonian is not Hermitian; check the mesh and file set')
    return np.linalg.eigvalsh(0.5*(h+h.swapaxes(1, 2).conj()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', help='Wannier dump prefix, e.g. conduction')
    parser.add_argument('kpoints', type=Path)
    parser.add_argument('--fold', nargs=3, type=int, required=True)
    parser.add_argument('--hamiltonian', choices=('ki', 'dft', 'bare'), default='ki',
                        help='KI, frozen DFT recovered as mlwfHki minus mlwfKI, or the dumped bare mlwfH')
    parser.add_argument('--ev', action='store_true', help='write energies in eV instead of Hartree')
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    points = np.loadtxt(args.kpoints, ndmin=2, usecols=(0, 1, 2))
    cells, h = load(args.prefix, args.fold, args.hamiltonian)
    values = interpolate(cells, h, points)*(27.211386245988 if args.ev else 1.)
    import sys
    np.savetxt(args.output if args.output else sys.stdout, np.column_stack((points, values)),
               header='kx ky kz followed by '+args.hamiltonian.upper()+' energies in '+('eV' if args.ev else 'Hartree'))
