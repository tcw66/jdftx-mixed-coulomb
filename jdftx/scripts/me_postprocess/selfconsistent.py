"""Static onsite-singlet BdG with a specified normal-state pairing window.

The canonical low-energy subspace is selected AFTER removing vacancy orbitals.
All matrix elements of the pairing field inside that subspace are retained;
this is not a band-diagonal (Anderson) approximation. Hole eigenvectors use
the time-reversed normal electron basis, fixing their relative gauge.
"""
import numpy as np
from . import KB_EV
from .backend import host, select_backend
from .wannier import mesh_points, indices, hermitian


class ProjectedSpectrum:
    """Twist-integrated normal eigenstates, grouped by pairing-subspace size."""
    def __init__(self, norbitals, total_points, cutoff_ev, temperature_k, backend='cpu'):
        if (norbitals<1 or total_points<1 or not np.isfinite(cutoff_ev) or cutoff_ev<=0
                or not np.isfinite(temperature_k) or temperature_k<=0):
            raise ValueError('invalid spectrum dimensions, pairing window or temperature')
        self.backend = select_backend(backend)
        self.xp = self.backend.xp
        self.norbitals, self.total_points = norbitals, total_points
        self.cutoff, self.temperature = cutoff_ev, temperature_k
        self.groups = {}
        self.normal_density = self.xp.zeros(norbitals)
        self.counts = []
        self.all_energies, self.all_twists = [], []

    def add(self, xi, vectors, twists):
        xp = self.xp
        weight = 1/self.total_points
        f = 0.5*(1-xp.tanh(xi/(2*KB_EV*self.temperature)))
        self.normal_density += 2*weight*xp.einsum('bip,bp->i',xp.abs(vectors)**2,f)
        mask = np.abs(host(xi)) < self.cutoff
        counts = mask.sum(axis=1)
        self.counts.extend(counts.tolist())
        self.all_energies.append(host(xi))
        self.all_twists.append(np.asarray(twists))
        for size in np.unique(counts):
            if size==0:
                continue
            rows = np.flatnonzero(counts==size)
            bands = np.stack([np.flatnonzero(mask[row]) for row in rows])
            r, b = xp.asarray(rows), xp.asarray(bands)
            group = self.groups.setdefault(int(size),dict(xi=[],vectors=[],twists=[]))
            group['xi'].append(xi[r[:,None],b])
            group['vectors'].append(vectors[r[:,None,None],xp.arange(self.norbitals)[None,:,None],b[:,None,:]])
            group['twists'].append(np.asarray(twists)[rows])

    def finalize(self):
        for group in self.groups.values():
            for key in ('xi','vectors'):
                group[key] = self.xp.concatenate(group[key])
            group['twists'] = np.concatenate(group['twists'])
        if len(self.counts)!=self.total_points:
            raise ValueError('incomplete Brillouin-zone sampling')
        if not self.groups:
            raise ValueError('pairing window contains no normal states; refine k sampling')
        self.all_energies = np.concatenate(self.all_energies)
        self.all_twists = np.concatenate(self.all_twists)
        return self

    def evaluate(self, delta, temperature=None, observables=False, ldos_grid=None, eta_ev=5e-5):
        """F_i = sum_positive u_i v_i* tanh(E/2kT), with normalized k weights.

        A normal-state projector defines the cutoff; ALL positive states of
        the projected BdG matrix are included. No second BdG-energy cutoff.
        """
        xp = self.xp
        t = self.temperature if temperature is None else temperature
        delta = xp.asarray(delta)
        if delta.shape!=(self.norbitals,):
            raise ValueError('pairing field has the wrong orbital dimension')
        f_local = xp.zeros(self.norbitals,dtype=complex)
        density = xp.zeros(self.norbitals)
        # Full density is available at the construction temperature. Outside-
        # window eigenvectors are discarded, so do not claim a full density at
        # another temperature when validating only gaps or grand potentials.
        condensation_electronic = xp.asarray(0.)
        minimum, phs_error = np.inf, 0.
        saved = {}
        ldos = None if ldos_grid is None else xp.zeros((self.norbitals,len(ldos_grid)))
        grid = None if ldos_grid is None else xp.asarray(ldos_grid)
        for size, group in self.groups.items():
            xi, basis = group['xi'],group['vectors']
            pairing = xp.einsum('bip,i,biq->bpq',basis.conj(),delta,basis,optimize=True)
            matrix = xp.zeros((len(xi),2*size,2*size),complex)
            diagonal = xp.arange(size)
            matrix[:,diagonal,diagonal] = xi
            matrix[:,size+diagonal,size+diagonal] = -xi
            matrix[:,:size,size:] = pairing
            matrix[:,size:,:size] = pairing.conj().swapaxes(-1,-2)
            energy, states = xp.linalg.eigh(matrix)
            positive = energy[:,size:]
            u = basis @ states[:,:size,size:]
            v = basis @ states[:,size:,size:]
            thermal = xp.tanh(positive/(2*KB_EV*t))
            f_local += xp.einsum('bip,bip,bp->i',u,v.conj(),thermal)/self.total_points
            if observables:
                occupation = (1-thermal)/2
                paired = 2*xp.einsum('bip,bp->i',xp.abs(u)**2,occupation)
                paired += 2*xp.einsum('bip,bp->i',xp.abs(v)**2,1-occupation)
                normal = 2*xp.einsum('bip,bp->i',xp.abs(basis)**2,
                                    0.5*(1-xp.tanh(xi/(2*KB_EV*t))))
                density += (paired-normal)/self.total_points
                log_cosh = xp.logaddexp(positive/(2*KB_EV*t),-positive/(2*KB_EV*t))
                normal_log_cosh = xp.logaddexp(xi/(2*KB_EV*t),-xi/(2*KB_EV*t))
                condensation_electronic += -2*KB_EV*t*xp.sum(log_cosh-normal_log_cosh)/self.total_points
                minimum = min(minimum,float(positive.min()))
                phs_error = max(phs_error,float(xp.max(xp.abs(energy+energy[:,::-1]))))
                saved[f'energies_{size}_ev'] = host(energy)
                saved[f'twists_{size}'] = group['twists']
                saved[f'projected_bdg_{size}_ev'] = host(matrix)
            if grid is not None:
                # Per-spin quasiparticle DOS inside the projected window.
                for start in range(0,len(grid),128):
                    window = grid[start:start+128]
                    particle = eta_ev/(np.pi*((window[None,None]-positive[:,:,None])**2+eta_ev**2))
                    hole = eta_ev/(np.pi*((window[None,None]+positive[:,:,None])**2+eta_ev**2))
                    ldos[:,start:start+128] += (
                        xp.einsum('bip,bpe->ie',xp.abs(u)**2,particle,optimize=True)+
                        xp.einsum('bip,bpe->ie',xp.abs(v)**2,hole,optimize=True))/self.total_points
        imaginary = float(xp.max(xp.abs(f_local.imag)))
        if imaginary>1e-9:
            raise ValueError(f'anomalous density has an unexpected imaginary component {imaginary:g}')
        result = dict(anomalous=host(f_local.real),imaginary_anomalous=imaginary)
        if observables:
            result.update(electronic_grand_potential_ev=float(condensation_electronic),
                minimum_energy_ev=minimum,particle_hole_error_ev=phs_error,spectra=saved)
            if t==self.temperature:
                result['density'] = host(self.normal_density+density)
        if ldos is not None:
            result['ldos_qp_per_spin_ev'] = host(ldos)
        return result

    def save_basis(self,path):
        data = dict(normal_density=host(self.normal_density),normal_eigenvalues_ev=self.all_energies,
                    all_twists=self.all_twists,cutoff_ev=self.cutoff,total_points=self.total_points,
                    temperature_k=self.temperature)
        for size,g in self.groups.items():
            data[f'xi_{size}'] = host(g['xi'])
            data[f'vectors_{size}'] = host(g['vectors'])
            data[f'twists_{size}'] = g['twists']
        np.savez_compressed(path,**data)

    @classmethod
    def load_basis(cls,path,backend='cpu'):
        with np.load(path) as data:
            result = cls(len(data['normal_density']),int(data['total_points']),
                         float(data['cutoff_ev']),float(data['temperature_k']),backend)
            result.normal_density = result.xp.asarray(data['normal_density'])
            result.all_energies, result.all_twists = data['normal_eigenvalues_ev'],data['all_twists']
            result.counts = np.sum(np.abs(result.all_energies)<result.cutoff,axis=1).tolist()
            for key in data.files:
                if key.startswith('xi_'):
                    size = int(key.split('_')[1])
                    result.groups[size] = dict(xi=result.xp.asarray(data[key]),
                        vectors=result.xp.asarray(data[f'vectors_{size}']),twists=data[f'twists_{size}'])
        return result


def bulk_spectrum(model,shape,z0,cutoff_ev,temperature,chunk=512,progress=None):
    kpoints = mesh_points(shape)
    result = ProjectedSpectrum(model.nw,len(kpoints),cutoff_ev,temperature,model.backend)
    for start in range(0,len(kpoints),chunk):
        k = kpoints[start:start+chunk]
        energy,vectors = model.electrons(k)
        result.add((energy-model.mu)/z0,vectors,k)
        if progress:
            progress(min(start+chunk,len(kpoints)),len(kpoints))
    return result.finalize()


def supercell_matrix(model,shape,twists,removed_sites=(),z0=1.):
    """Exact Fourier folding, including ALL native Wannier hopping ranges.

    H_q(r,r') = exp[2πiq.(r-r')/L] ifft_j h((q+j)/L)[r-r'].
    Deleted sites are removed from both matrix indices before diagonalization.
    """
    xp = model.xp
    shape = np.asarray(shape,int)
    cells = np.indices(shape).reshape(3,-1).T
    sites = np.flatnonzero(~np.isin(np.arange(len(cells)),np.asarray(removed_sites,int)))
    positions = cells[sites]
    difference = positions[:,None]-positions[None,:]
    lookup = indices(difference.reshape(-1,3),shape).reshape(len(sites),len(sites))
    twists = np.atleast_2d(twists)
    folded = (twists[:,None]+cells[None])/shape
    h = model.h(folded.reshape(-1,3)).reshape(len(twists),*shape,model.nw,model.nw)
    blocks = xp.fft.ifftn(h,axes=(1,2,3)).reshape(len(twists),len(cells),model.nw,model.nw)
    phase = xp.exp(2j*np.pi*xp.einsum('bd,ijd->bij',xp.asarray(twists),xp.asarray(difference/shape)))
    matrix = blocks[:,xp.asarray(lookup)]*phase[:,:,:,None,None]
    matrix = matrix.transpose(0,1,3,2,4).reshape(len(twists),len(sites)*model.nw,len(sites)*model.nw)
    matrix = hermitian(matrix,'supercell Hamiltonian',rtol=1e-10)
    return (matrix-model.mu*xp.eye(matrix.shape[-1]))/z0,sites


def vacancy_spectrum(model,shape,twist_mesh,z0,cutoff_ev,temperature,removed_sites=(0,),
                     chunk=4,progress=None):
    twists = mesh_points(twist_mesh)
    norbitals = (int(np.prod(shape))-len(removed_sites))*model.nw
    result = ProjectedSpectrum(norbitals,len(twists),cutoff_ev,temperature,model.backend)
    for start in range(0,len(twists),chunk):
        q = twists[start:start+chunk]
        matrix, sites = supercell_matrix(model,shape,q,removed_sites,z0)
        if model.backend.name=='cuda' and matrix.shape[-1]>32:
            # CuPy's large stacked matrices use Jacobi; the 2D cuSOLVER
            # divide-and-conquer path is much faster for these normal blocks.
            pairs = [model.xp.linalg.eigh(block) for block in matrix]
            energy = model.xp.stack([pair[0] for pair in pairs])
            vectors = model.xp.stack([pair[1] for pair in pairs])
        else:
            energy,vectors = model.xp.linalg.eigh(matrix)
        result.add(energy,vectors,q)
        if progress:
            progress(min(start+chunk,len(twists)),len(twists))
    return result.finalize(),sites


def solve_gap(spectrum,attraction_ev,initial_delta_ev,temperature,tolerance_ev=1e-9,
              mixing=0.4,max_iterations=1000,progress=None):
    if (not 0<mixing<=1 or tolerance_ev<=0 or max_iterations<1
            or not np.isfinite(temperature) or temperature<=0):
        raise ValueError('invalid self-consistency controls or temperature')
    attraction = np.broadcast_to(np.asarray(attraction_ev,float),(spectrum.norbitals,))
    if np.any(attraction<=0) or not np.isfinite(attraction).all():
        raise ValueError('onsite attractions must be finite and positive')
    delta = np.broadcast_to(np.asarray(initial_delta_ev,float),(spectrum.norbitals,)).copy()
    history = []
    for iteration in range(1,max_iterations+1):
        target = attraction*spectrum.evaluate(delta,temperature)['anomalous']
        residual = float(np.max(np.abs(target-delta)))
        history.append([iteration,residual,float(delta.min()),float(delta.max()),float(delta.mean())])
        if progress:
            progress(iteration,residual,delta)
        if residual<tolerance_ev:
            break
        delta += mixing*(target-delta)
    else:
        raise ValueError(f'BdG gap iteration failed: residual {residual:g} eV after {max_iterations} iterations')
    observables = spectrum.evaluate(delta,temperature,observables=True)
    final_residual = float(np.max(np.abs(attraction*observables['anomalous']-delta)))
    observables.update(delta_ev=delta,iterations=iteration,residual_ev=final_residual,
        history=np.array(history),grand_potential_difference_ev=float(
            np.sum(delta**2/attraction)+observables['electronic_grand_potential_ev']))
    return observables
