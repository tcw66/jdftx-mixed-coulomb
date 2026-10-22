"""Coherent Wannier BdG construction, symmetry checks and real-space export."""
import numpy as np
from .wannier import hermitian, grid
from .backend import select_backend, host


def negative_partners(kpoints):
    kpoints = np.asarray(kpoints, float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or not np.isfinite(kpoints).all():
        raise ValueError('kpoints must have shape (Nk,3) and be finite')
    def key(k):
        return tuple(np.rint((k % 1)*1e9).astype(np.int64) % 1000000000)
    lookup = {key(k): i for i, k in enumerate(kpoints)}
    if len(lookup) != len(kpoints):
        raise ValueError('duplicate periodic k points')
    try:
        return np.array([lookup[key(-k)] for k in kpoints])
    except KeyError as error:
        raise ValueError('k mesh must contain the -k partner of every point') from error


def charge_conjugate(matrices):
    n = matrices.shape[-1]//2
    order = np.r_[np.arange(n, 2*n), np.arange(n)]
    return matrices.conj()[..., order, :][..., :, order]


def symmetry_errors(hamiltonian, partner):
    n = hamiltonian.shape[-1]//2
    delta = hamiltonian[:, :n, n:]
    return dict(hermiticity_ev=float(np.max(np.abs(hamiltonian-hamiltonian.conj().swapaxes(-1, -2)))),
                particle_hole_ev=float(np.max(np.abs(hamiltonian+charge_conjugate(hamiltonian[partner])))),
                pairing_antisymmetry_ev=float(np.max(np.abs(delta+delta[partner].swapaxes(-1, -2)))))


def coherent_reduce(hstatic, metric, partner, tolerance=1e-8, backend='cpu'):
    """Eq. (36): A^-1/2 Hstatic A^-1/2, retaining anomalous derivatives.

    hstatic = H0_Nambu + Hermitian Sigma^R(0), in eV.
    metric = I - d_omega Hermitian Sigma^R(0), dimensionless.
    Input must already satisfy the Nambu constraints up to tolerance.
    """
    hstatic, metric = np.asarray(hstatic), np.asarray(metric)
    if (hstatic.ndim != 3 or hstatic.shape != metric.shape or
            hstatic.shape[-1] != hstatic.shape[-2] or hstatic.shape[-1] % 2 or
            len(partner) != len(hstatic) or not np.isfinite(hstatic).all() or
            not np.isfinite(metric).all()):
        raise ValueError('invalid Nambu static matrix or metric')
    hstatic = hermitian(hstatic, 'static Nambu matrix', rtol=0, atol=tolerance)
    metric = hermitian(metric, 'Nambu frequency metric', rtol=0, atol=tolerance)
    h_error = np.max(np.abs(hstatic+charge_conjugate(hstatic[partner])))
    a_error = np.max(np.abs(metric-charge_conjugate(metric[partner])))
    if max(h_error, a_error) > tolerance:
        raise ValueError(f'Nambu constraints violated: H error {h_error:g} eV, metric error {a_error:g}')
    hstatic = (hstatic-charge_conjugate(hstatic[partner]))/2
    metric = (metric+charge_conjugate(metric[partner]))/2
    xp = select_backend(backend).xp
    values, vectors = xp.linalg.eigh(xp.asarray(metric))
    minimum = float(values.min())
    if minimum <= 0:
        raise ValueError('frequency metric must be positive definite for coherent reduction')
    s = (vectors/xp.sqrt(values)[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
    result = host(s @ xp.asarray(hstatic) @ s)
    result = (result-charge_conjugate(result[partner]))/2
    return hermitian(result, 'coherent BdG'), host(s), minimum


def nambu_normal(h, mu_ev, partner):
    h = hermitian(np.asarray(h), 'spinful normal Hamiltonian')
    xi = h-mu_ev*np.eye(h.shape[-1])
    n = h.shape[-1]
    result = np.zeros((len(h), 2*n, 2*n), complex)
    result[:, :n, :n] = xi
    result[:, n:, n:] = -xi[partner].swapaxes(-1, -2)
    return result


def isotropic_singlet(h_orbital, kpoints, mu_ev, z0, phi0_ev, backend='cpu'):
    """Explicit local orbital singlet assumption in a real Wannier TR gauge.

    Electron order: (orb0 up, orb0 down, orb1 up, orb1 down, ...).
    Full Nambu dimension is 4*Norb. Sigma_normal(0)=0 (PH-symmetric ME).
    """
    partner = negative_partners(kpoints)
    if not (np.isfinite(z0) and z0 > 0 and np.isfinite(phi0_ev)):
        raise ValueError('Z0 must be positive and Phi0 finite')
    h_orbital = np.asarray(h_orbital)
    if np.max(np.abs(h_orbital[partner]-h_orbital.conj())) > 1e-8:
        raise ValueError('local singlet model requires a real Wannier time-reversal gauge')
    h_spin = np.kron(h_orbital, np.eye(2))
    static = nambu_normal(h_spin, mu_ev, partner)
    n = h_spin.shape[-1]
    isigma_y = np.array([[0, 1], [-1, 0]])
    phi = phi0_ev*np.kron(np.eye(h_orbital.shape[-1]), isigma_y)
    static[:, :n, n:] = phi
    static[:, n:, :n] = phi.conj().T
    metric = np.broadcast_to(z0*np.eye(2*n), static.shape)
    result, s, _ = coherent_reduce(static, metric, partner, backend=backend)
    return result, s


def real_space(matrices, kpoints, shape, backend='cpu'):
    """Uniform Gamma-centered discrete inverse of H(k)=sum_R exp(+ikR)t(R).

    Coefficients live in the periodic mesh supercell. In particular, -R at a
    Nyquist boundary is understood modulo that supercell.
    """
    shape = grid(shape)
    if len(kpoints) != np.prod(shape):
        raise ValueError('real-space export requires every point of the specified mesh')
    scaled = np.asarray(kpoints)*shape
    if not np.allclose(scaled, np.rint(scaled), atol=1e-7, rtol=0):
        raise ValueError('real-space export requires an unshifted uniform mesh')
    coords = np.rint(scaled).astype(int) % shape
    order = np.ravel_multi_index(coords.T, shape)
    if len(np.unique(order)) != len(order):
        raise ValueError('duplicate grid points')
    sorted_matrices = np.empty_like(matrices)
    sorted_matrices[order] = matrices
    tensor = sorted_matrices.reshape((*shape, *matrices.shape[1:]))
    xp = select_backend(backend).xp
    transformed = host(xp.fft.fftn(xp.asarray(tensor), axes=(0, 1, 2))/len(order))
    axes = [np.rint(np.fft.fftfreq(n)*n).astype(int) for n in shape]
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return cells, transformed.reshape(matrices.shape)


def export_arrays(hamiltonian, kpoints, shape, backend='cpu'):
    backend = select_backend(backend)
    xp = backend.xp
    partner = negative_partners(kpoints)
    checks = symmetry_errors(hamiltonian, partner)
    if max(checks.values()) > 1e-8:
        raise ValueError(f'invalid BdG Hamiltonian: {checks}')
    n = hamiltonian.shape[-1]//2
    h, delta = hamiltonian[:, :n, :n], hamiltonian[:, :n, n:]
    cells, t = real_space(h, kpoints, shape, backend)
    _, pairing = real_space(delta, kpoints, shape, backend)
    # Check the real-space conjugation relations, including mesh aliasing.
    lookup = {tuple(r % shape): i for i, r in enumerate(cells)}
    inv = np.array([lookup[tuple(-r % shape)] for r in cells])
    checks['real_space_hopping_ev'] = float(np.max(np.abs(t-t[inv].conj().swapaxes(-1, -2))))
    checks['real_space_pairing_ev'] = float(np.max(np.abs(pairing+pairing[inv].swapaxes(-1, -2))))
    return dict(kpoints=kpoints, mesh=np.array(shape), h_qp_ev=h, delta_qp_ev=delta,
                bdg_ev=hamiltonian, eigenvalues_ev=host(xp.linalg.eigvalsh(xp.asarray(hamiltonian))),
                cells=cells, hopping_ev=t, pairing_ev=pairing), checks
