# Audit against Souza, Iniguez, and Vanderbilt, PRL 89, 117602 (2002)

Reviewed the supplied official article,
[First-Principles Approach to Insulators in Finite Electric Fields](https://doi.org/10.1103/PhysRevLett.89.117602).
The occupied-only, norm-conserving, weak-field formulation agrees with the
paper. Agreement with the experimental silicon dielectric constant was not
used to establish correctness.

| Article requirement | Implementation and assessment |
|---|---|
| Eq. (1): `F = EDFT - Omega E.Pmac` | The full DFT internal energy is evaluated on field-polarized orbitals. `EfieldBerry` adds the electronic and ionic field coupling; occupied eigenvalue sums are not used as the total energy. |
| Eq. (2): occupied-overlap determinants on closed strings | `M = Omega C_k^dagger T C_next` includes JDFTx's coefficient normalization. Electronic polarization is `f/(2*pi*Omega) sum_i a_i phi_i`, with the phase averaged over transverse strings. The reciprocal-boundary mapping is `u_(k+G) = exp(-i G.r) u_k`. |
| Eq. (4): neighbor-coupled orbital derivative | The outgoing term is `+i f Omega (E.a_i)/(4*pi*Ntransverse) T C_plus M_plus^-1`. The incoming term is its backward-neighbor counterpart with negative sign and the inverse adjoint. This matches the article after translating its normalized-orbital metric to JDFTx coefficients and its derivative to `2 Re Tr(G^dagger dC)`. |
| Self-consistency among k points and in the density | Each variational evaluation recomputes density, Hartree, and XC terms, then includes the Berry derivative in the orbital gradient and kinetic preconditioner. A nonzero-field fixed-Hamiltonian or independent-k SCF calculation is rejected. |
| Occupied-gauge invariance | Final subspace diagonalization rotates only occupied states; it does not replace the polarized subspace with independent-k ground-state eigenvectors. Gauge-rotation regressions pass. |
| Exclude independent-k cleanup and zero-field response Hessians | The review found that `converge-empty-states yes` calls an independent-k band minimizer even when all stored bands are occupied. This could undo finite-field polarization after convergence and is now rejected. `perturb-minimize` and `phonon-dfpt` combinations are also rejected because those Hessians lack the Berry coupling. |
| Force discussion, page 3 | For norm-conserving orbitals at fixed lattice, the electronic Berry term has no explicit ion-position derivative. Standard DFT forces on polarized states plus `Z_I E` therefore give the stationary enthalpy forces. The existing implementation is correct; ultrasoft augmentation remains unsupported. |
| Strain and electrical boundary conditions, Eqs. (5)–(6) | Cell relaxation keeps Cartesian E fixed. Its explicit enthalpy strain derivative is `-Omega sym(E outer P_total)`, added to the ordinary DFT stress evaluated on polarized orbitals. JDFTx varies symmetric strain, so the rotational torque is projected out. The branch offset `sum_i n_i a_i/Omega` deforms with the lattice; silicon uses its nonpolar branch. All six strain derivatives are checked numerically for both the field term and full enthalpy. |
| Finite mesh and local minima, pages 2–3 | A finite mesh is essential. The workflow now reports `N_i |E.a_i| / Egap` and rejects values at or above one, using an independently computed sampled gap. This is an estimate, not a rigorous breakdown criterion. |
| Secular branch of the line-search energy, Fig. 1 | The original implementation kept the initial string references permanently. It was locally correct for the tested small fields, but could reset a phase at a fixed cut for larger cumulative changes. References now advance at accepted iterates, remain fixed during trials, and permit continuation across polarization quanta. Large phase changes and singular trial overlaps trigger backtracking. |
| Fractional quantum on a single string, page 3 | The analyzer originally rounded only the total polarization to whole lattice quanta. It now aligns individual string phases before averaging and applies the corresponding exact dipole correction to enthalpy. This handles a string shift of `f a_i/(Omega*Ntransverse)`. |
| Symmetry and minimization method | Keeping the full BZ is valid but redundant: the paper permits time reversal and field-preserving symmetries. JDFTx uses block conjugate gradients rather than the paper's band-by-band updates; the same functional and constrained gradient determine the local stationary state. |

The small-field silicon results remain unchanged within numerical convergence
after the branch corrections. A fresh 4³ calculation gives electronic and
relaxed-ion `epsilon_zz = 9.49996036` at `h=1e-4` Hartree atomic units.
For the earlier 12³ mesh, the maximum mesh/gap estimate is 0.3014 at `h=1e-4`
and 0.6028 at `h=2e-4`, below the paper's approximate threshold of one.

## Independent Quantum ESPRESSO comparison

Quantum ESPRESSO 7.5 `lelfield=.true.` was run using the same SG15 Si UPF,
PBE functional, two-atom cell, 10.26-bohr conventional lattice constant,
Gamma-centered 4³ mesh, and 30-Hartree / 60-Ry orbital cutoff. Both signs
started from the same converged zero-field state. The density threshold was
`1e-14` Ry, with `nberrycyc=10` and conjugate-gradient orbital updates.

QE's finite-field and dipole units differ from Hartree atomic units:
`E_QE = sqrt(2) E_Hartree` and `mu_QE = sqrt(2) mu_Hartree`. The conversions
cancel when computing `4*pi*dmu_QE/(Omega*dE_QE)`.

| z column of relative dielectric tensor | JDFTx | QE 7.5 | Difference, QE − JDFTx |
|---|---:|---:|---:|
| εxz | 0.10615082 | 0.10612184 | −0.00002899 |
| εyz | 0.10615083 | 0.10613391 | −0.00001691 |
| εzz | 9.49996036 | 9.50000531 | +0.00004495 |

The εzz difference is approximately 4.7 parts per million relative to its
value. The finite-mesh off-diagonal response is reproduced by the independent
implementation, supporting the interpretation as a discretization effect.
Neither code was tuned to obtain an experimental value near 12.

[Machine-readable comparison](prl-comparison.json). Exact inputs and QE logs
are preserved in `results-prl-qe`; fresh JDFTx inputs and outputs are in
`results-prl-jdftx/k4`.

## Additional checks and remaining limits

Nine Berry-specific CTest regressions pass on CPU, GPU, MPI and combined
GPU/MPI. GPU/MPI state broadcasts use ordinary host staging to avoid MPI
shared-memory access to migrating managed allocations. A 16³ fixed-state
comparison also passes on two GPUs. New tests
walk electronic polarization forward and backward through multiple quanta,
verify rejected trials leave the accepted branch unchanged, exercise a 3×4×5
mesh, and check fractional-string branch corrections to both polarization
and enthalpy. Finite-difference gradient, gauge, boundary-phase, and ionic
force tests continue to pass.
Input guards are tested for independent-k empty-state cleanup and unsupported
finite-field DFPT combinations.

The native occupied-only command cannot establish a conduction-band gap on
its own. The example workflow checks it independently. A nonsingular overlap
does not by itself prove stability below Zener breakdown. These calculations
target local weak-field insulating states; convergence versus field, mesh,
cutoff, and ionic force tolerance remains necessary. The 12³ result of
12.2689 is still a finite-mesh value. Norm-conserving internal-ion and cell
relaxation are supported at fixed Cartesian field. Ultrasoft/PAW augmentation,
finite-field DFPT, lattice dynamics and rigid-rotation torque response remain
outside the implemented scope.
