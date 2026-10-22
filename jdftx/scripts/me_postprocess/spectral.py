"""Double-delta Fermi-surface coupling and discrete Eliashberg spectrum."""
import numpy as np
from .backend import array_module, host


def fermi_delta(energy, mu, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('electronic Gaussian sigma must be positive, in eV')
    xp = array_module(energy)
    return xp.exp(-0.5*((energy-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)


def calculate(model, kpoints, qpoints, sigma_ev, chunk=128, progress=None):
    """Uniform BZ samples. DOS is per spin per primitive cell per eV.

    gamma uses Eq. (2) in the supplied note (EPW half-width convention).
    No electronic spin multiplicity is inserted in either DOS or numerator.
    """
    nk, nq = len(kpoints), len(qpoints)
    if nk == 0 or nq == 0 or chunk < 1:
        raise ValueError('empty mesh or invalid chunk size')
    xp = getattr(model, 'xp', np)
    k_device, q_device = xp.asarray(kpoints), xp.asarray(qpoints)
    dos = 0.
    for start in range(0, nk, chunk):
        energy, _ = model.electrons(k_device[start:start+chunk])
        dos += fermi_delta(energy, model.mu, sigma_ev).sum()/nk
    dos = float(dos)
    if dos <= 1e-12 or not np.isfinite(dos):
        raise ValueError('no Fermi-level DOS; check chemical potential, bands, mesh and smearing')
    omega = xp.empty((nq, model.nmodes))
    scattering = xp.zeros_like(omega)
    for iq, q in enumerate(q_device):
        omega[iq], vectors = model.phonons(q)
        for start in range(0, nk, chunk):
            k = k_device[start:start+chunk]
            e0, u0 = model.electrons(k)
            e1, u1 = model.electrons(k+q)
            g = model.vertex(k, q, omega[iq], vectors)
            bands = xp.einsum('kam,kvab,kbn->kvmn', u1.conj(), g, u0, optimize=True)
            d0, d1 = fermi_delta(e0, model.mu, sigma_ev), fermi_delta(e1, model.mu, sigma_ev)
            scattering[iq] += xp.einsum('km,kn,kvmn->v', d1, d0, xp.abs(bands)**2)/nk
        if progress:
            progress(iq+1, nq)
    active = omega > model.frequency_floor
    mode_lambda = xp.where(active, 2*scattering/(dos*xp.maximum(omega, model.frequency_floor)), 0.)
    # Represent alpha²F as positive spectral lines: sum_j A_j delta(E-Omega_j).
    line_weight = xp.where(active, scattering/(dos*nq), 0.)
    result = dict(kpoints=kpoints, qpoints=qpoints, omega_ev=omega,
                line_weight_ev=line_weight, mode_lambda=mode_lambda,
                gamma_ev=2*np.pi*omega*scattering, dos_per_spin_ev=np.array(dos),
                sigma_ev=np.array(sigma_ev), mu_ev=np.array(model.mu),
                excluded_modes=xp.count_nonzero(~active))
    return {key: host(value) for key, value in result.items()}


def moments(omega, weight):
    omega, weight = np.asarray(omega).ravel(), np.asarray(weight).ravel()
    if omega.shape != weight.shape or not np.isfinite(omega).all() or not np.isfinite(weight).all():
        raise ValueError('invalid discrete spectrum')
    if np.any(weight < 0) or np.any((weight > 0) & (omega <= 0)):
        raise ValueError('spectrum requires nonnegative weights at positive energies')
    use = weight > 0
    omega, weight = omega[use], weight[use]
    contribution = 2*weight/omega
    coupling = contribution.sum()
    if coupling <= 0:
        raise ValueError('zero electron–phonon coupling')
    return dict(lambda_total=float(coupling),
                omega_log_ev=float(np.exp(np.sum(contribution*np.log(omega))/coupling)),
                omega_2_ev=float(np.sqrt(np.sum(contribution*omega**2)/coupling)))


def histogram(omega, weight, bins=300):
    if bins < 2:
        raise ValueError('at least two spectral bins are required')
    edges = np.linspace(0, np.max(omega)*1.001, bins+1)
    area, _ = np.histogram(omega, bins=edges, weights=weight)
    per_line = np.divide(2*weight, omega, out=np.zeros_like(weight), where=omega > 0)
    contribution, _ = np.histogram(omega, bins=edges, weights=per_line)
    return np.column_stack(((edges[:-1]+edges[1:])/2, area/np.diff(edges),
                            np.cumsum(contribution)))
