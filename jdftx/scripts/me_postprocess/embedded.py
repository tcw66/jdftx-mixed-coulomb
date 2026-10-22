"""Infinite-host Nambu embedding with finite-support onsite pairing changes.

Reduced singlet basis (c_up, c_down^dagger), real nonmagnetic Wannier gauge.
The host is integrated in k space, never obtained by cutting hopping bonds.
Units: eV, kelvin; F=-kBT sum_n G_eh(iw_n) in this Nambu convention.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import polygamma
from . import KB_EV, HARTREE_EV
from .backend import select_backend, host
from .wannier import binary, cells_from, indices


def matsubara_quadrature(temperature, order=10, exact=32, maximum=262144):
    """Gaussian quadrature of the DISCRETE positive Matsubara sum.

    Uniform discrete-measure (Gram polynomial) recurrence in dyadic blocks;
    weights count Matsubara indices, not an approximation by an integral.
    """
    if temperature <= 0 or order < 2 or exact < 1 or maximum < exact:
        raise ValueError('invalid Matsubara quadrature')
    nodes, weights = [np.arange(exact, dtype=float)], [np.ones(exact)]
    start = exact
    while start < maximum:
        count = min(start, maximum-start)
        if count <= order:
            nodes.append(start+np.arange(count)); weights.append(np.ones(count))
        else:
            j = np.arange(1, order, dtype=float)
            off = .5*j*np.sqrt((count**2-j*j)/(4*j*j-1))
            x, v = eigh_tridiagonal(np.full(order, (count-1)/2), off)
            nodes.append(start+x); weights.append(count*v[0]**2)
        start += count
    return (2*np.concatenate(nodes)+1)*np.pi*KB_EV*temperature, np.concatenate(weights)


def dyson(g, perturbation):
    """Fully resum a Hermitian perturbation; works on CPU or CUDA arrays."""
    xp = np if isinstance(g, np.ndarray) else __import__('cupy')
    return xp.linalg.solve(xp.eye(len(g))-g@perturbation, g)


def matsubara_tail(temperature, weights):
    """2 kBT sum_{n=N}^infinity 1/omega_n^2, for the leading local tail."""
    return float(polygamma(1, float(np.sum(weights))+.5)/(2*np.pi**2*KB_EV*temperature))


def remove_orbitals(g, removed):
    """Exact hard cavity, following wannier_scattering.cavity's Schur solve."""
    xp = np if isinstance(g, np.ndarray) else __import__('cupy')
    ids = xp.asarray(removed, dtype=int)
    return g-g[:, ids]@xp.linalg.solve(g[xp.ix_(ids, ids)], g[ids, :])


class BulkHost:
    """Uniform real singlet superconductor on a Fourier integration mesh.

    Full-bandwidth local attractive model. Fit U_a=Delta_bulk/F_bulk,a on
    this mesh; do NOT reuse U from a spectrally projected vacancy model.
    """
    def __init__(self, cells, hopping_ev, mesh, mu, z0, gap, temperature,
                 backend='cpu', device=0):
        self.backend = select_backend(backend, device); xp = self.backend.xp
        self.mesh = tuple(int(n) for n in mesh)
        self.norb = np.asarray(hopping_ev).shape[-1]
        self.gap, self.temperature, self.z0 = float(gap), float(temperature), float(z0)
        if min(self.mesh) < 2 or min(gap, temperature, z0) <= 0:
            raise ValueError('positive host parameters and mesh >=2 required')
        if np.max(abs(np.asarray(hopping_ev).imag)) > 1e-9:
            raise ValueError('embedding requires a real time-reversal Wannier gauge')
        # Distinct native translations can coincide on a smaller Fourier mesh.
        idx = indices(np.asarray(cells), np.asarray(mesh))
        np_h = np.zeros((int(np.prod(mesh)), self.norb, self.norb), complex)
        np.add.at(np_h, idx, np.asarray(hopping_ev))
        h = xp.asarray(np_h.reshape(*mesh, self.norb, self.norb)); del np_h
        h = xp.fft.ifftn(h, axes=(0, 1, 2))*np.prod(mesh)
        error = float(xp.max(abs(h-h.conj().swapaxes(-1, -2))))
        if error > 1e-7:
            raise ValueError(f'non-Hermitian host: {error}')
        h = ((h+h.conj().swapaxes(-1, -2))/2).reshape(-1, self.norb, self.norb)
        self.xi = xp.empty((len(h), self.norb))
        self.vectors = xp.empty_like(h)
        for start in range(0, len(h), 32768):
            e, u = xp.linalg.eigh(h[start:start+32768])
            self.xi[start:start+len(e)] = (e-mu)/z0
            self.vectors[start:start+len(e)] = u
        del h
        self.energy = xp.sqrt(self.xi**2+gap**2)
        weight = gap/(2*self.energy)*xp.tanh(self.energy/(2*KB_EV*temperature))
        self.bulk_f = host(xp.einsum('kan,kn,kbn->ab', self.vectors, weight,
                                    self.vectors.conj(), optimize=True)/len(self.xi))
        self.attraction = gap/np.diag(self.bulk_f).real

    def green(self, z, cells):
        """Full host G(z) on requested cells, ordered all electrons, all holes."""
        xp = self.backend.xp
        cells = np.asarray(cells, int)
        if np.any(2*np.ptp(cells, axis=0)+1 > np.asarray(self.mesh)):
            raise ValueError('host Fourier mesh aliases observation displacements')
        displacement = cells[None, :, :]-cells[:, None, :]
        ix = tuple(xp.asarray(displacement[..., a] % self.mesh[a]) for a in range(3))
        n = len(cells)*self.norb
        result = xp.empty((2*n, 2*n), complex)
        den = z*z-self.energy**2
        for row, col, w in ((0, 0, (z+self.xi)/den), (1, 1, (z-self.xi)/den),
                            (0, 1, self.gap/den)):
            matrix = xp.einsum('kan,kn,kbn->kab', self.vectors, w,
                               self.vectors.conj(), optimize=True)
            grid = xp.fft.fftn(matrix.reshape(*self.mesh, self.norb, self.norb),
                               axes=(0, 1, 2))/len(self.xi)
            block = grid[ix].transpose(0, 2, 1, 3).reshape(n, n)
            result[row*n:(row+1)*n, col*n:(col+1)*n] = block
            if row != col:
                result[n:, :n] = block
        return result

    def pair_matrix(self, cells):
        """Exact thermal host F blocks, ordered spatial orbitals only."""
        xp = self.backend.xp
        weight = self.gap/(2*self.energy)*xp.tanh(self.energy/(2*KB_EV*self.temperature))
        matrix = xp.einsum('kan,kn,kbn->kab', self.vectors, weight,
                           self.vectors.conj(), optimize=True)
        grid = xp.fft.fftn(matrix.reshape(*self.mesh, self.norb, self.norb),
                           axes=(0, 1, 2))/len(self.xi)
        displacement = np.asarray(cells)[None, :, :]-np.asarray(cells)[:, None, :]
        ix = tuple(xp.asarray(displacement[..., a] % self.mesh[a]) for a in range(3))
        n = len(cells)*self.norb
        return grid[ix].transpose(0, 2, 1, 3).reshape(n, n)


def read_hopping(prefix, kfold):
    cells = cells_from(str(prefix)+'mlwfCellMap')
    nw = int(round(np.sqrt(Path(str(prefix)+'mlwfCellWeights').stat().st_size/(8*len(cells)))))
    weights = binary(str(prefix)+'mlwfCellWeights', (len(cells), nw, nw)).swapaxes(-1, -2)
    reduced = binary(str(prefix)+'mlwfH', (int(np.prod(kfold)), nw, nw), True).swapaxes(-1, -2)
    return cells, weights*reduced[indices(cells, np.asarray(kfold))]*HARTREE_EV


def prepare_embedding(bulk, cells, frequencies, progress=None, return_reference=False):
    """Cache the finite support after deleting the origin's Nambu orbitals."""
    xp = bulk.backend.xp; nw = bulk.norb
    origin = np.flatnonzero(np.all(np.asarray(cells) == 0, axis=1))
    if len(origin) != 1:
        raise ValueError('exactly one vacancy cell at the origin required')
    n = len(cells)*nw
    removed = np.r_[origin[0]*nw+np.arange(nw), n+origin[0]*nw+np.arange(nw)]
    keep = np.setdiff1d(np.arange(2*n), removed)
    cache = xp.empty((len(frequencies), len(keep), len(keep)), complex)
    reference = xp.empty((len(frequencies), len(keep)//2, len(keep)//2), complex) if return_reference else None
    for i, w in enumerate(frequencies):
        clean = bulk.green(1j*w, cells)
        if return_reference:
            reference[i] = clean[xp.ix_(xp.asarray(keep[:len(keep)//2]),
                                        xp.asarray(keep[len(keep)//2:]))]
        g = remove_orbitals(clean, removed)
        cache[i] = g[xp.ix_(xp.asarray(keep), xp.asarray(keep))]
        if progress:
            progress(i+1, len(frequencies))
    return (cache, reference) if return_reference else cache


def anomalous_matrix(bulk, cells, delta, cavity, reference, weights):
    """Full F on surviving support, including orbital interference blocks."""
    xp = bulk.backend.xp; n = cavity.shape[1]//2; ix = xp.arange(n)
    v = xp.zeros((2*n, 2*n), complex)
    v[ix, ix+n] = xp.asarray(delta).ravel()-bulk.gap
    v[ix+n, ix] = v[ix, ix+n].conj()
    f = bulk.pair_matrix(cells)
    for j, weight in enumerate(weights):
        g = dyson(cavity[j], v)
        f -= 2*KB_EV*bulk.temperature*weight*(g[:n, n:]-reference[j]).real
    f[ix, ix] += matsubara_tail(bulk.temperature, weights)*(xp.asarray(delta).ravel()-bulk.gap)
    return host(f)


def solve_embedding(bulk, cavity, weights, *, initial=None, mixing=.5,
                    tolerance=1e-8, iterations=200, progress=None):
    """Self-consistent full-band onsite model, analytic bulk + Matsubara difference.

    Subtracting the bulk anomalous sum leaves a rapidly convergent defect
    correction; the pristine reference is evaluated analytically over bands.
    """
    xp = bulk.backend.xp
    n = cavity.shape[1]//2; nw = bulk.norb
    delta = xp.full(n, bulk.gap) if initial is None else xp.asarray(initial).reshape(n).copy()
    u = xp.tile(xp.asarray(bulk.attraction), n//nw)
    f0 = xp.tile(xp.asarray(np.diag(bulk.bulk_f).real), n//nw)
    ix = xp.arange(n); perturbation = xp.zeros((2*n, 2*n), complex)
    history = []
    # Host onsite G_eh is cell independent and analytic on the same mesh.
    reference = xp.stack([-bulk.gap*xp.einsum('kan,kn->a', abs(bulk.vectors)**2,
                          1/(w*w+bulk.energy**2))/len(bulk.xi)
                          for w in weights[0]])
    for iteration in range(iterations):
        perturbation[ix, ix+n] = delta-bulk.gap
        perturbation[ix+n, ix] = (delta-bulk.gap).conj()
        f = f0.astype(complex)
        for j, weight in enumerate(weights[1]):
            g = dyson(cavity[j], perturbation)
            correction = g[ix, ix+n]-xp.tile(reference[j], n//nw)
            # For real, TR-invariant onsite singlet fields, negative frequencies
            # supply the conjugate. No extra spin degeneracy enters F.
            f -= 2*KB_EV*bulk.temperature*weight*correction.real
        f += matsubara_tail(bulk.temperature, weights[1])*(delta-bulk.gap)
        target = u*f.real
        residual = float(xp.max(abs(target-delta)))
        history.append(residual)
        if progress:
            progress(iteration+1, residual)
        if residual < tolerance:
            return host(delta), host(f), np.asarray(history)
        delta = (1-mixing)*delta+mixing*target
    raise RuntimeError(f'embedded gap did not converge: residual {history[-1]:g} eV')


def perturbed_green(bulk, cells, delta_cells, delta, z, return_clean=False):
    """Observation + support G with cavity and self-consistent pairing scattering."""
    cells, delta_cells = np.asarray(cells, int), np.asarray(delta_cells, int)
    combined = np.unique(np.vstack((cells, delta_cells, np.zeros((1, 3), int))), axis=0)
    xp = bulk.backend.xp; nw = bulk.norb; n = len(combined)*nw
    origin = np.flatnonzero(np.all(combined == 0, axis=1))[0]
    removed = np.r_[origin*nw+np.arange(nw), n+origin*nw+np.arange(nw)]
    clean = bulk.green(z, combined)
    g = remove_orbitals(clean, removed)
    # Dyson inversion only on pairing support, as in the QPI T-matrix routine.
    lookup = {tuple(c): i for i, c in enumerate(combined)}
    e = np.concatenate([lookup[tuple(c)]*nw+np.arange(nw) for c in delta_cells])
    support = xp.asarray(np.r_[e, n+e]); m = len(e)
    v = xp.zeros((2*m, 2*m), complex); ix = xp.arange(m)
    v[ix, ix+m] = xp.asarray(delta).ravel()-bulk.gap
    v[ix+m, ix] = v[ix, ix+m].conj()
    gd = g[xp.ix_(support, support)]
    t = xp.linalg.solve(xp.eye(2*m)-v@gd, v)
    g += g[:, support]@t@g[support, :]
    return (g, combined, clean) if return_clean else (g, combined)


class SpatialWannier:
    """3D mlwfC amplitudes for arbitrary (including skew) lattices.

    JDFTx's coefficients already include plane-wave volume normalization;
    volume*sum_G |C_G|^2 must equal one for each normalized Wannier orbital.
    Unlike a plane-specific interpolation, translations may have z components.
    """
    def __init__(self, prefix, backend='cpu', device=0, grid_factor=2):
        prefix = str(prefix)
        with open(prefix+'mlwfC.header') as stream:
            self.norb, nbasis = map(int, stream.readline().split()[:2])
            gt = np.array(stream.readline().split()[:9], float).reshape(3, 3)
            self.ig = np.loadtxt(stream, dtype=int)
        if self.ig.shape != (nbasis, 3):
            raise ValueError('mlwfC header basis mismatch')
        self.lattice = 2*np.pi*np.linalg.inv(gt).T
        self.volume = abs(np.linalg.det(self.lattice))
        self.origin = -.5*self.lattice.sum(axis=1)
        if not isinstance(grid_factor, int) or grid_factor < 1:
            raise ValueError('Wannier grid factor must be a positive integer')
        self.shape = tuple(grid_factor*(2*np.max(abs(self.ig), axis=0)+1))
        if Path(prefix+'mlwfC').stat().st_size != self.norb*nbasis*16:
            raise ValueError('mlwfC file size mismatch')
        self.coefficients = np.memmap(prefix+'mlwfC', mode='r', dtype=complex,
                                      shape=(self.norb, nbasis))
        self.norms = self.volume*np.sum(abs(self.coefficients)**2, axis=1)
        if not np.allclose(self.norms, 1, atol=1e-5):
            raise ValueError(f'mlwfC physical normalization mismatch: {self.norms}')
        self.backend = select_backend(backend, device)
        self.splines = []

    def prepare(self):
        xp = self.backend.xp
        ix = tuple(xp.asarray(self.ig[:, a] % self.shape[a]) for a in range(3))
        for coefficient in self.coefficients:
            grid = xp.zeros(self.shape, complex)
            grid[ix] = xp.asarray(coefficient)
            amplitude = host(xp.fft.ifftn(grid))*np.prod(self.shape)
            self.splines.append(spline_filter(amplitude.real, mode='grid-wrap')+
                                1j*spline_filter(amplitude.imag, mode='grid-wrap'))

    def evaluate(self, orbital, positions):
        frac = (np.asarray(positions)-self.origin)@np.linalg.inv(self.lattice).T
        return map_coordinates(self.splines[orbital], (frac*np.asarray(self.shape)).T,
                               order=3, mode='grid-wrap', prefilter=False)

    def exact(self, orbital, positions):
        frac = (np.asarray(positions)-self.origin)@np.linalg.inv(self.lattice).T
        return np.exp(2j*np.pi*frac@self.ig.T)@self.coefficients[orbital]

    def map(self, positions, cells, lattice, cutoff_bohr=14., taper_bohr=2.):
        """Dense point-by-orbital amplitudes with explicit localized tail cutoff."""
        if not 0 < taper_bohr < cutoff_bohr:
            raise ValueError('require 0 < taper < cutoff')
        result = np.zeros((len(positions), len(cells)*self.norb), complex)
        for i, cell in enumerate(cells):
            local = positions-np.asarray(cell)@np.asarray(lattice).T
            radius = np.linalg.norm(local, axis=1)
            keep = radius < cutoff_bohr
            window = .5*(1+np.cos(np.pi*np.clip((radius[keep]-cutoff_bohr+taper_bohr)/taper_bohr, 0, 1)))
            for a in range(self.norb):
                result[keep, i*self.norb+a] = self.evaluate(a, local[keep])*window
        return result


def project_diagonal(w, matrix, conjugate_right=True, backend='cpu', batch_size=1024):
    """Diagonal of W M W^dagger or W F W^T; batch points on CPU or CUDA.

    The output is a NumPy array; batching bounds the temporary for wide views.
    A Backend instance can select an already active GPU.
    """
    if batch_size < 1 or w.shape[1] != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('incompatible projection dimensions or batch size')
    compute = select_backend(backend); xp = compute.xp
    matrix = xp.asarray(matrix)
    result = np.empty(len(w), complex)
    for start in range(0, len(w), batch_size):
        block = xp.asarray(w[start:start+batch_size])
        right = block.conj() if conjugate_right else block
        result[start:start+len(block)] = host(xp.einsum('pi,pi->p', block@matrix, right, optimize=True))
    return result
