# Silicon KI implementation audit — 12 October 2026

The follow-up [direct QE implementation comparison](qe-implementation-comparison.md)
includes source checks, fresh cross-code curvature/Hamiltonian tests, the
additional reduced-screening-mesh Ewald correction, and the exact official
silicon tutorial input.

The saved calculation gives a **1.522083 eV** interpolated Brillouin-zone gap
and a **1.522991 eV** sampled-path gap. The original projected PBE gap is
0.545067 eV. Recovering PBE as `mlwfHki - mlwfKI` and comparing interpolation
with the native mesh spectra gives errors below 2.2e-14 Ha. The discrepancy
with published KI values is therefore present in the native Hamiltonian.

## Implementation correction

The charge-channel GGA Hessian used relative finite differences in
`sigma = |gradient(n)|^2`. At zero gradient both sigma samples coincided,
and division by sigma generated NaNs. Near zero, cancellation also made the
mixed derivative unreliable. An independent potential-difference regression
reproduced the NaN for uniform-density PBE before the correction.

`ExCorr::getSecondDerivatives` now uses a density-scaled lower bound for the
sigma perturbation, keeps sigma samples nonnegative, and obtains the mixed
derivative from density differences of the analytic sigma derivative.
The relative density step is 1e-5 to reduce cancellation. The CPU and GPU
regressions cover charge and magnetic response for LDA/PBE, uniform,
nearly uniform and inhomogeneous densities, Gamma and finite q, and both
kernel application paths. The new `koopmansSpinKIPBE` test independently
checks bare curvature against full Hxc potential differences and screened
curvature against relaxed spin-polarized supercell SCFs. It also checks
occupied/empty corrections, reciprocal wrapping and Ewald scaling.

Native screening outputs now identify the KI(2) approximation, dielectric
constant and electrostatic treatment. Plots label the correction KI(2), and
their JSON records whether an image correction was applied. These metadata
changes do not change eigenvalues.

Validation completed: all nine selected CTest checks passed
(`koopmansSpinKernel`, `koopmansSpinKernelGPU`, `koopmansKI`,
`koopmansSpinKI`, `koopmansSpinKIPBE`, `koopmansScreeningMesh`,
`koopmansOccupied`, `koopmansReuse`, `koopmansResponseSpeedups`).
The independent PBE supercell test also passed with two MPI ranks, including
agreement of serial/MPI screening and spectra. CPU/GPU Wannier binaries were
rebuilt. Python compilation, plot metadata and whitespace checks passed.

## Checks on the saved silicon state

Source-only audits used copies of the saved valence and conduction rotations,
the full 8x8x8 electronic/screening mesh, and the corrected GPU kernel.
No response solves or production output replacements were performed by
these audits. The maximum change in bare curvature was:

| Manifold | Maximum difference (Ha) |
| --- | ---: |
| Four occupied Wanniers | 2.99e-12 |
| Six empty Wanniers | 3.12e-12 |

Thus the fixed stationary-density defect has negligible impact on the
**bare curvatures of this saved state**. This does not establish that all
screening responses or the resulting gap are unchanged. The complete
8x8x8 screened Hamiltonians have not been recomputed after the kernel fix.

An additional conduction response calculation kept the same 8x8x8 electronic
mesh and frozen rotations and solved the charge and magnetic sectors at
Gamma. Its unweighted Gamma response contributions agree with the saved
full-screening run within 1.28e-12 Ha. This checks response as well as bare
curvature at that point. Its Gamma-only screening coefficients and spectra
are diagnostic outputs and must not replace the production results.

Isolated numerical artifacts are in
[`archive/implementation-audit-20261012/`](archive/implementation-audit-20261012/),
including source audit logs, curvature comparisons and regenerated plots
of the existing full-mesh spectra. Production SCF and KI files are preserved.

## Comparison with reported values

The native equations were checked against Colonna et al., JCTC 18,
5435 (2022), Eqs. 11, 18, 20, 22–24 and SI Eqs. 5–6:
finite-q orbital density normalization, reciprocal wrapping, screened/bare
curvature ratio, constant occupied shift, empty-state potential contractions,
column screening and Gamma image correction. Independent numerical tests
validate the kernel and response; no empirical scaling or gap offset was added.

The calculations are not identical to the common silicon references:

* [De Gennaro et al., PRB 106, 035106 (2022)](https://doi.org/10.1103/PhysRevB.106.035106)
  uses full nonlinear KI and finite-occupation constrained screening.
  Table I reports 1.18 eV **including** a -0.05 eV zero-point correction,
  corresponding to 1.23 eV before that correction.
* [Colonna et al., JCTC 18, 5435 (2022)](https://doi.org/10.1021/acs.jctc.2c00161)
  implements KI(2). Its silicon validation uses LDA, a 4x4x4 mesh and
  four antibonding conduction Wanniers; its 1.41 eV validation gap is explicitly
  described as not converged in Brillouin-zone sampling.
* The present setup uses PBE, six conduction Wanniers, KI(2),
  infinitesimal-occupation screening and neutralizing-background
  electrostatics (`epsilonInfinity=0`). Published electrostatic corrections
  and finite-occupation screening cannot be substituted by choosing a
  dielectric constant or rescaling alpha to fit the desired gap.

A numerical match to a particular reported KIC/KI value needs its exact
reference and computational setup. The remaining difference has not been
assigned to a single cause by this audit.
