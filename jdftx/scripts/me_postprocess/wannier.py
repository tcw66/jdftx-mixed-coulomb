"""Read native JDFTx Wannier exports without expanding the two-cell vertex."""
from pathlib import Path
import re
import numpy as np
from . import HARTREE_EV
from .backend import Backend, array_module


def grid(value):
    result = np.asarray(value, dtype=int)
    if result.shape != (3,) or np.any(result < 1) or not np.all(result == value):
        raise ValueError('mesh must contain three positive integers')
    return result


def mesh_points(shape):
    shape = grid(shape)
    return np.indices(shape).reshape(3, -1).T / shape


def indices(cells, shape):
    return (cells % shape) @ np.array([shape[1]*shape[2], shape[2], 1])


def cells_from(path):
    data = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    if not np.isfinite(data).all() or not np.allclose(data, np.rint(data), atol=1e-10):
        raise ValueError(f'{path}: cell coordinates must be integers')
    return data.astype(int)


def binary(path, shape, complex_allowed=False):
    """Native column-major matrix entries, C-ordered outer indices."""
    count = int(np.prod(shape))
    size = Path(path).stat().st_size
    dtype = '<c16' if complex_allowed and size == 16*count else '<f8'
    if size != np.dtype(dtype).itemsize*count:
        raise ValueError(f'{path}: unexpected size {size}; shape {shape}, dtype {dtype}')
    result = np.memmap(path, dtype=dtype, mode='r', shape=tuple(shape))
    if not np.isfinite(result).all():
        raise ValueError(f'{path}: contains nonfinite entries')
    return result


def hermitian(matrix, label, rtol=1e-6, atol=1e-12):
    xp = array_module(matrix)
    error = float(xp.max(xp.abs(matrix-matrix.conj().swapaxes(-1, -2))))
    if error > atol+rtol*float(xp.max(xp.abs(matrix))):
        raise ValueError(f'{label}: non-Hermitian matrix (max error {error:g})')
    return (matrix+matrix.conj().swapaxes(-1, -2))/2


def read_metadata(log):
    text = Path(log).read_text(errors='replace')
    # Logs can contain appended runs; use the last complete command list.
    text = text[text.rfind('Input parsed successfully'): ] if 'Input parsed successfully' in text else text
    match = re.findall(r'^kpoint-folding\s+(\d+)\s+(\d+)\s+(\d+)', text, re.M)
    if not match:
        raise ValueError(f'{log}: missing kpoint-folding')
    spin = re.findall(r'^spintype\s+(\S+)', text, re.M)
    if not spin or spin[-1] != 'no-spin':
        raise ValueError('isotropic native-input path requires explicit spintype no-spin')
    chemical = re.findall(r'Metallic DFPT: Fermi energy\s+([-+\d.eE]+)', text)
    if not chemical:
        chemical = re.findall(r'FillingsUpdate:\s+mu:\s+([-+\d.eE]+)', text)
    return np.array(match[-1], int), (float(chemical[-1])*HARTREE_EV if chemical else None)


class WannierModel:
    """Scalar-relativistic, spin-degenerate orbitals; no implicit spin factor.

    g(q,k)[nu,a,b] has final Wannier index a at k+q and initial b at k.
    It includes sqrt(1/(2 omega_Ha)) exactly once, after phonon rotation.
    The native mass weighting is already present in mlwfHePh.
    """

    def __init__(self, wannier_prefix, phonon_prefix, kfold, phonon_supercell,
                 mu_ev, frequency_floor_ev=1e-5, imaginary_tolerance_ev=1e-4,
                 backend='cpu', gpu_device=0):
        self.backend = Backend(backend, gpu_device)
        self.xp = self.backend.xp
        self.wp, self.pp = str(wannier_prefix), str(phonon_prefix)
        self.kfold, self.phonon_supercell = grid(kfold), grid(phonon_supercell)
        if np.any(self.kfold % self.phonon_supercell):
            raise ValueError('electronic mesh must contain the native phonon supercell')
        if not np.isfinite(mu_ev):
            raise ValueError('chemical potential must be finite')
        if (not np.isfinite(frequency_floor_ev) or not np.isfinite(imaginary_tolerance_ev)
                or frequency_floor_ev <= 0 or imaginary_tolerance_ev < 0):
            raise ValueError('invalid phonon floor or imaginary tolerance')
        self.mu = float(mu_ev)
        self.frequency_floor = frequency_floor_ev
        self.imaginary_tolerance = imaginary_tolerance_ev
        self.cells = cells_from(self.wp+'mlwfCellMap')
        size = Path(self.wp+'mlwfCellWeights').stat().st_size
        self.nw = int(np.sqrt(size/(8*len(self.cells))))
        weights = binary(self.wp+'mlwfCellWeights', (len(self.cells), self.nw, self.nw)).swapaxes(-1, -2)
        reduced = binary(self.wp+'mlwfH', (int(np.prod(self.kfold)), self.nw, self.nw), True).swapaxes(-1, -2)
        self.h_r = weights*reduced[indices(self.cells, self.kfold)]*HARTREE_EV
        lines = [line for line in Path(self.pp+'phononBasis').read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        basis = np.loadtxt(lines, dtype=str, ndmin=2)
        self.nmodes = len(basis)
        if self.nmodes % 3 or basis.shape[1] != 6:
            raise ValueError('phononBasis must list three Cartesian directions per atom')
        self.phonon_cells = cells_from(self.pp+'phononCellMap')
        self.dyn_r = binary(self.pp+'phononOmegaSq',
                            (len(self.phonon_cells), self.nmodes, self.nmodes)).swapaxes(-1, -2)
        self.eph_cells = cells_from(self.wp+'mlwfCellMapPh')
        weights = binary(self.wp+'mlwfCellWeightsPh',
                         (len(self.eph_cells), self.nw, self.nmodes//3)).swapaxes(-1, -2)
        self.eph_weights = np.repeat(weights, 3, axis=1)
        self.eph_indices = indices(self.eph_cells, self.phonon_supercell)
        self.nred = int(np.prod(self.phonon_supercell))
        self.eph = binary(self.wp+'mlwfHePh',
                          (self.nred, self.nred, self.nmodes, self.nw, self.nw), True).swapaxes(-1, -2)
        if self.backend.name == 'cuda':
            # Transfer static data once. Chunking bounds interpolation temporaries.
            for name in ('cells', 'h_r', 'phonon_cells', 'dyn_r', 'eph_cells',
                         'eph_weights', 'eph_indices', 'eph'):
                setattr(self, name, self.xp.ascontiguousarray(self.xp.asarray(getattr(self, name))))

    def h(self, k):
        xp = self.xp
        phase = xp.exp(2j*np.pi*xp.asarray(k) @ self.cells.T)
        return hermitian(xp.einsum('...r,rab->...ab', phase, self.h_r), 'Wannier H(k)')

    def electrons(self, k):
        return self.xp.linalg.eigh(self.h(k))

    def phonons(self, q):
        xp = self.xp
        phase = xp.exp(2j*np.pi*xp.asarray(q) @ self.phonon_cells.T)
        dyn = hermitian(xp.einsum('...r,rab->...ab', phase, self.dyn_r), 'dynamical matrix')
        squared, vectors = xp.linalg.eigh(dyn)
        minimum = float(squared.min())
        if minimum < -(self.imaginary_tolerance/HARTREE_EV)**2:
            raise ValueError(f'imaginary phonon energy {np.sqrt(-minimum)*HARTREE_EV:g} eV')
        return xp.sqrt(xp.maximum(squared, 0))*HARTREE_EV, vectors

    def weighted_phases(self, k, sign):
        xp = self.xp
        phase = xp.exp(sign*2j*np.pi*xp.asarray(k) @ self.eph_cells.T)
        result = xp.zeros((len(k), self.nred, self.nmodes, self.nw), complex)
        weighted = phase[:, :, None, None]*self.eph_weights
        if self.backend.name == 'cuda':
            # CuPy scatter-add supports float64 but not complex128 atomics.
            xp.add.at(result.real.swapaxes(0, 1), self.eph_indices, weighted.real.swapaxes(0, 1))
            xp.add.at(result.imag.swapaxes(0, 1), self.eph_indices, weighted.imag.swapaxes(0, 1))
        else:
            np.add.at(result.swapaxes(0, 1), self.eph_indices, weighted.swapaxes(0, 1))
        return result

    def vertex(self, k, q, omega=None, vectors=None):
        """Paired k -> k+q interpolation, without a Cartesian product of k's."""
        xp = self.xp
        k, q = xp.atleast_2d(xp.asarray(k)), xp.asarray(q)
        if omega is None:
            omega, vectors = self.phonons(q)
        left, right = self.weighted_phases(k+q, -1), self.weighted_phases(k, 1)
        cart = xp.empty((len(k), self.nmodes, self.nw, self.nw), complex)
        for x in range(self.nmodes):
            intermediate = xp.einsum('kua,uvab->kvab', left[:, :, x], self.eph[:, :, x], optimize=True)
            cart[:, x] = xp.einsum('kvab,kvb->kab', intermediate, right[:, :, x], optimize=True)
        norm = xp.where(omega > self.frequency_floor,
                        HARTREE_EV/(2*xp.maximum(omega, self.frequency_floor)), 0.)
        return xp.einsum('xv,kxab->kvab', vectors, cart)*xp.sqrt(norm)[None, :, None, None]*HARTREE_EV

    def provenance(self):
        files = [self.wp+s for s in ('mlwfH', 'mlwfCellMap', 'mlwfCellWeights',
                 'mlwfHePh', 'mlwfCellMapPh', 'mlwfCellWeightsPh')]
        files += [self.pp+s for s in ('phononBasis', 'phononCellMap', 'phononOmegaSq')]
        return [{'path': str(Path(p).resolve()), 'bytes': Path(p).stat().st_size,
                 'mtime_ns': Path(p).stat().st_mtime_ns} for p in files]
