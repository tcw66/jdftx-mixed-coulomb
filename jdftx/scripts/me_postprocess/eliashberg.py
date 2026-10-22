"""Isotropic, even-frequency, constant-DOS imaginary-axis ME equations."""
import numpy as np
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs, ArpackNoConvergence, LinearOperator
from scipy.optimize import brentq
from . import KB_EV
from .spectral import moments
from .backend import select_backend, host


def allen_dynes(omega, weight, mu_star):
    if not np.isfinite(mu_star) or mu_star < 0:
        raise ValueError('mu* must be finite and nonnegative')
    m = moments(omega, weight)
    lam, wlog, w2 = m['lambda_total'], m['omega_log_ev'], m['omega_2_ev']
    denom = lam-mu_star*(1+0.62*lam)
    if denom <= 0:
        return 0.
    l1 = 2.46*(1+3.8*mu_star)
    ratio = w2/wlog
    l2 = 1.82*(1+6.3*mu_star)*ratio
    f1 = (1+(lam/l1)**1.5)**(1/3)
    f2 = 1+(ratio-1)*lam**2/(lam**2+l2**2)
    return float(f1*f2*wlog/(1.2*KB_EV)*np.exp(-1.04*(1+lam)/denom))


class IsotropicME:
    def __init__(self, omega, weight, mu_star, cutoff_ev, coulomb_cutoff_ev,
                 max_frequencies=2048, backend='cpu', gpu_device=0):
        self.backend = select_backend(backend, gpu_device)
        self.xp = self.backend.xp
        moments(omega, weight)  # validate before filtering zero weights
        omega, weight = np.asarray(omega).ravel(), np.asarray(weight).ravel()
        self.omega, self.weight = omega[weight > 0], weight[weight > 0]
        if not np.isfinite(mu_star) or mu_star < 0:
            raise ValueError('mu* must be finite and nonnegative')
        if not (np.isfinite(cutoff_ev) and np.isfinite(coulomb_cutoff_ev)
                and cutoff_ev >= coulomb_cutoff_ev > 0):
            raise ValueError('require Matsubara cutoff >= Coulomb cutoff > 0 (eV)')
        if cutoff_ev < 3*self.omega.max():
            raise ValueError('Matsubara cutoff must be at least three times the maximum phonon energy')
        self.mu_star, self.cutoff, self.coulomb_cutoff = mu_star, cutoff_ev, coulomb_cutoff_ev
        self.max_frequencies = max_frequencies
        self._omega, self._weight = self.xp.asarray(self.omega), self.xp.asarray(self.weight)

    def kernel(self, transfer):
        xp = self.xp
        transfer = xp.asarray(transfer)
        if self.backend.name == 'cuda':
            result = xp.zeros_like(transfer, dtype=float)
            # Bound spectral-line temporaries; keep the kernel on the device.
            for start in range(0, len(self.omega), 256):
                shape = (-1,)+(1,)*transfer.ndim
                om = self._omega[start:start+256].reshape(shape)
                area = self._weight[start:start+256].reshape(shape)
                result += xp.sum(2*area*om/(om**2+transfer[None]**2), axis=0)
            return result
        # Sum one line at a time to avoid a spectral-lines x N² temporary.
        result = np.zeros_like(transfer, dtype=float)
        for om, area in zip(self.omega, self.weight):
            result += 2*area*om/(om**2+transfer**2)
        return result

    def matrices(self, temperature):
        xp = self.xp
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive in K')
        pi_t = np.pi*KB_EV*temperature
        nfreq = max(16, int(np.ceil((self.cutoff/pi_t-1)/2))+1)
        if nfreq > self.max_frequencies:
            raise ValueError(f'{nfreq} positive Matsubara frequencies needed; increase --max-frequencies')
        n = xp.arange(nfreq)
        wn = (2*n+1)*pi_t
        kernel = self.kernel(2*pi_t*xp.arange(2*nfreq))
        minus = kernel[xp.abs(n[:, None]-n[None, :])]
        plus = kernel[n[:, None]+n[None, :]+1]
        # Exact infinite-sum normal-state Z for the constant-DOS approximation.
        # This removes the leading normal-state finite-Matsubara-tail error.
        zn = 1+(2*xp.cumsum(kernel[:nfreq])-kernel[0])/(2*n+1)
        attraction = minus+plus
        pairing = attraction-2*self.mu_star*(wn[None, :] < self.coulomb_cutoff)
        return wn, pi_t, minus-plus, pairing, zn

    def eigenvalue(self, temperature):
        wn, pi_t, _, pairing, zn = self.matrices(temperature)
        matrix = pi_t*pairing/(zn[:, None]*wn[None, :])
        if len(wn) < 40:
            values = eigvals(host(matrix))
            value = values[np.argmax(values.real)]
        else:
            try:
                operator = matrix
                if self.backend.name == 'cuda':
                    # ARPACK controls the nonsymmetric eigenproblem on CPU;
                    # its dense matrix-vector products stay on GPU.
                    operator = LinearOperator(matrix.shape, dtype=np.float64,
                        matvec=lambda v: host(matrix @ self.xp.asarray(v)))
                value = eigs(operator, k=1, which='LR', v0=np.ones(len(wn)),
                             return_eigenvectors=False, tol=1e-9)[0]
            except ArpackNoConvergence as error:
                raise ValueError('linearized ME eigensolver did not converge') from error
        if abs(value.imag) > 1e-7:
            raise ValueError('leading pairing eigenvalue is not real')
        return float(value.real)

    def critical_temperature(self, low, high, tolerance=0.01):
        if not (0 < low < high) or tolerance <= 0:
            raise ValueError('invalid Tc bracket or tolerance')
        lower, upper = self.eigenvalue(low), self.eigenvalue(high)
        if not lower > 1 > upper:
            raise ValueError(f'Tc not bracketed: rho({low:g} K)={lower:g}, rho({high:g} K)={upper:g}; adjust bounds')
        return float(brentq(lambda t: self.eigenvalue(t)-1, low, high, xtol=tolerance))

    def solve(self, temperature, mixing=0.35, tolerance_ev=1e-10, max_iterations=6000):
        xp = self.xp
        if not 0 < mixing <= 1 or tolerance_ev <= 0 or max_iterations < 1:
            raise ValueError('invalid nonlinear iteration controls')
        wn, pi_t, normal_kernel, pairing, zn = self.matrices(temperature)
        rho = self.eigenvalue(temperature)
        delta = xp.zeros_like(wn)
        z = zn.copy()
        residual = 0.
        iterations = 0
        if rho > 1:
            delta[:] = min(0.002, self.omega.max()/10)
            for iterations in range(1, max_iterations+1):
                norm = xp.sqrt(wn**2+delta**2)
                z = zn+pi_t/wn*(normal_kernel @ (wn/norm-1))
                new = pi_t*(pairing @ (delta/norm))/z
                residual = float(xp.max(xp.abs(new-delta)))
                delta += mixing*(new-delta)
                if residual < tolerance_ev:
                    break
            else:
                raise ValueError(f'nonlinear ME did not converge at {temperature:g} K; residual={residual:g} eV')
            if abs(delta[0]) < 100*tolerance_ev:
                raise ValueError('iteration collapsed near Tc; lower T or tighten tolerance')
            if delta[0] < 0:
                delta *= -1
        norm = xp.sqrt(wn**2+delta**2)
        z = zn+pi_t/wn*(normal_kernel @ (wn/norm-1))
        # Delta F / N(0) has units eV², with a per-spin N(0).
        free_energy_per_dos = -2*pi_t*xp.sum((norm-wn)*(z-zn*wn/norm))
        result = dict(temperature_k=np.array(temperature), wn_ev=wn, z=z, z_normal=zn,
                    delta_ev=delta, phi_ev=z*delta, eigenvalue=np.array(rho),
                    iterations=np.array(iterations), residual_ev=np.array(residual),
                    free_energy_per_dos_ev2=free_energy_per_dos)
        return {key: host(value) for key, value in result.items()}


def low_frequency_fit(solution, count=4):
    """Even polynomial extrapolation on the imaginary axis, not continuation.

    Returns coherent parameters estimated from Z(iw), Phi(iw). These are
    approximations to zero real frequency, suitable only at sufficiently low T.
    """
    wn, z, phi = solution['wn_ev'], solution['z'], solution['phi_ev']
    if count < 2 or count > len(wn):
        raise ValueError('invalid low-frequency fit point count')
    x = (wn[:count]/wn[count-1])**2
    z0 = float(np.polynomial.polynomial.polyfit(x, z[:count], 1)[0])
    phi0 = float(np.polynomial.polynomial.polyfit(x, phi[:count], 1)[0])
    if not z0 > 0:
        raise ValueError('nonpositive extrapolated Z')
    return dict(z0=z0, phi0_ev=phi0, delta0_ev=phi0/z0,
                first_matsubara_ev=float(wn[0]), fit_points=count)
