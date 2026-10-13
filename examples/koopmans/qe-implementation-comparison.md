# JDFTx KI comparison with official QE implementations

Audit date: 12 October 2026. The native implementation is **KI(2)**, matching
the perturbative formulation in QE's **KCW** module. The linked silicon
tutorial runs **full KI with finite-occupation DSCF screening** through the
separate **KCP** backend. Those two calculations need not produce the same
gap, even with identical ground-state XC and Wannier orbitals.

## Reference code inspected

The numerical reference is the local QE 7.5 CPU build, including a newly
built `kcw.x`. Six relevant local sources were checked byte-for-byte against
the official `qe-7.5` tag: `rho_of_q.f90`, `bare_pot.f90`,
`screen_coeff.f90`, `solve_linter_koop_mod.f90`, `koopmans_ham.f90`,
and `coulomb.f90`. All match. Checksums are stored with the audit artifacts.

For the full-KI comparison, source inspection used `epfl-theos/koopmans`
commit `023fe26857466c5fae2520f5794dd5a86c416a84` and its pinned KCP submodule
commit `dd88e3209ce0e97d58591f5689cb16c05ac574fd`.

| Component | Official source | Native result |
| --- | --- | --- |
| Finite-q orbital density | [rho_of_q.f90](https://github.com/QEF/q-e/blob/qe-7.5/KCW/src/rho_of_q.f90) | Both use a full-mesh average of periodic Bloch products and reciprocal wrapping. QE's wavefunctions use volume normalization; native wavefunctions carry the corresponding real-space normalization. |
| Orbital Hxc kernel and spin | [bare_pot.f90](https://github.com/QEF/q-e/blob/qe-7.5/KCW/src/bare_pot.f90) | QE injects a unit orbital density into one spin channel. Native charge plus longitudinal magnetic response is the equivalent transformation at zero magnetization. No extra factor of two is required. |
| Screening | [screen_coeff.f90](https://github.com/QEF/q-e/blob/qe-7.5/KCW/src/screen_coeff.f90) | Both use relaxed/bare curvature. The driven response excludes the constant Hartree Gamma term; the macroscopic screened correction is added separately. |
| Hamiltonian | [koopmans_ham.f90](https://github.com/QEF/q-e/blob/qe-7.5/KCW/src/koopmans_ham.f90) | Both subtract half the bare curvature, add the empty-manifold potential contraction, screen each upper-triangle entry by its column's alpha, and complete the Hermitian matrix. The active QE routine includes diagonal empty-state contractions; its older unused `hamilt.f90` is not the reference. |
| Coulomb singularity | [coulomb.f90](https://github.com/QEF/q-e/blob/qe-7.5/KCW/src/coulomb.f90) | Isotropic point-charge Ewald and QE Gygi–Baldereschi corrections agree for the tested cell. Native screening on a reduced q mesh needed its own supercell correction; this defect is corrected. |
| Full-KI screening | [official DSCF workflow](https://github.com/epfl-theos/koopmans/blob/023fe26857466c5fae2520f5794dd5a86c416a84/src/koopmans/workflows/_koopmans_dscf.py) | Full KI compares constrained N±1 total-energy differences with orbital eigenvalues. Native infinitesimal-response KI(2) does not perform those finite-occupation calculations. |

QE reports curvatures in **Ry**, native output in **Ha**: divide QE values
by two before comparison. Alpha is dimensionless. Ground-state spin
degeneracy must not double the normalized single-spin orbital source.

## Corrections made

The earlier charge-channel PBE Hessian correction remains in place: a
relative finite difference of `sigma=|gradient(n)|²` had generated 0/0 at
zero gradient and cancellation near zero. See
[implementation-audit.md](implementation-audit.md) for its independent
potential-difference tests and the checks on the saved silicon state.

The additional defect found during this QE audit occurs when
`wannier-koopmans-screening-mesh` is coarser than the Wannier mesh and
`epsilonInfinity` enables Ewald correction. Previously, screening reused
the fine-mesh Gamma kernel and changed only its quadrature weight. The
kernel itself depends on the screening supercell: for a cubic mesh its
weighted image correction scales as **1/L**, not simply as the inverse
number of q points.

The corrected code builds a second Ewald kernel for the screening mesh.
The full-mesh kernel still enters the spectral Hamiltonian. Bare screening
receives its own image term D, and relaxed screening receives D/epsilon.
`koopmansResponse` retains its original first nine columns and appends
`screening_image_Ha`; `image_Ha` continues to describe the full-mesh term.
The test now compares both terms with an independent real/reciprocal-space
point-charge Ewald sum, rather than reconstructing the same implementation's
weights alone.
For example, in the regression's 3x3x1 electronic mesh with Gamma-only
screening, the old image curvature would be 0.737667310396 Ha; the corrected
screening value is 0.354662184935 Ha, agreeing with the cubic point-charge sum.

This additional fix **does not affect the saved 1.522083 eV silicon result**:
that run used the full screening mesh and `epsilonInfinity=0`. Neither fix
has been shown to explain that gap's difference from reported full-KI gaps.
No screening coefficient was fitted or silently clipped.

## Direct numerical comparison

`jdftx/test/koopmans/checkQEKI.py` runs both programs from fresh state with
the same H2 geometry, NC hydrogen UPF, 6 Ha/12 Ry cutoff, 24³ density grid,
two bands, and Gamma sampling. The single occupied and single empty state
have no nontrivial orbital mixing at Gamma. Native uses a no-spin ground
state with single-spin KI response; QE uses `nspin=2, tot_magnetization=0`.
The fixture explicitly selects PZ or PBE in both codes. Its small cutoff and
cell are for checking equations, not predicting a physical gap; the LDA
empty-state curvature can be negative and its Ewald alpha can exceed one.

Both XC choices were tested with a neutralizing background and with Ewald
correction at epsilon=2, on CPU and GPU. For the CPU comparison, maxima over
both orbitals and both electrostatic choices are:

| XC | Curvature difference (Ha) | Alpha difference | Hamiltonian-shift difference (Ha) |
| --- | ---: | ---: | ---: |
| LDA-PZ | 8.80e-7 | 2.52e-5 | 1.99e-6 |
| PBE | 5.30e-5 | 2.91e-5 | 2.68e-5 |

The largest Hamiltonian-shift difference is about **0.00073 eV**. The
comparisons allow finite numerical differences in the two XC/SCF/response
implementations and the four-decimal QE band printout; they do not assert
bitwise equality. PBE curvature tolerance is 8e-5 Ha. The independent native
PBE potential/SCF finite differences provide an additional check of its kernel.
QE's [GGA derivative driver](https://github.com/QEF/q-e/blob/qe-7.5/XClib/qe_drivers_d_gga.f90)
uses density/gradient finite differences capped at a 1% relative step and
special handling below 1e-6; the native kernel uses different numerical
steps and regularization. These choices can contribute to the residual
PBE differences; their individual contributions were not isolated here.

The observed Gamma image curvature is 0.354662184935 Ha, matching
2.837297479480619/8 for a unit charge in this cubic cell. The new reduced-mesh
tests verify independent Ewald sums for 3x3x1, 1x3x1 and 1x1x1 meshes, retain
the same full-mesh Hamiltonian ingredients, and pass symmetry and two-rank
MPI comparisons on CPU and GPU. The existing `koopmansKI`, `koopmansSpinKI`,
and `koopmansSpinKIPBE` CTest regressions pass after the change.

Reproduce the direct comparison from the repository root:

```bash
python jdftx/test/koopmans/checkQEKI.py build /home/tyler/apps/qe-7.5/bin \
  /home/tyler/apps/build_9-13-2026/pseudopotentials/SG15/H_ONCV_PBE.upf
# To retain inputs, logs and comparison.json, add --work-dir NEW_EMPTY_DIRECTORY.
# To use native GPU executables, set JDFTX_TEST_EXECUTABLE_SUFFIX=_gpu.
```

Audit artifacts and source checksums are in
[`archive/implementation-audit-20261012/qe-comparison/`](archive/implementation-audit-20261012/qe-comparison/).

## Why the tutorial remains a different calculation

The pinned KCP source
[nksic_corrections.f90](https://github.com/epfl-theos/koopmans-kcp/blob/dd88e3209ce0e97d58591f5689cb16c05ac574fd/CPV/nksic_corrections.f90)
evaluates XC energies and potentials at the reference density, at that
density minus the occupied orbital density, and at the density with the
orbital fully occupied. It retains nonlinear XC contributions that a
second-order Hessian expansion omits. Its DSCF workflow obtains alpha from
N±1 energy differences. These are method differences, not a missing scalar
spin factor that can be corrected by rescaling native alpha.

The [official tutorial](https://koopmans-functionals.org/en/latest/tutorials/tutorial_2.html)
also uses four occupied/four empty sp3 Wanniers, whereas the saved native
calculation has six p-like empty Wanniers. The downloaded latest JSON uses
60 Ry, a 2x2x2 teaching mesh, `alpha_guess=0.077`,
`mp_correction=false`, and the PseudoDojo LDA library. The prose calls its
baseline PBE, while the JSON omits explicit `base_functional`/`input_dft`.
Check the actual generated QE/KCP XC settings before calling it KI@PBE.
The tutorial itself says the teaching mesh gives poor interpolation.

The exact unchanged download is now bundled as
[`literature/si-official-tutorial.json`](literature/si-official-tutorial.json).
[`literature/run_official_tutorial.py`](literature/run_official_tutorial.py)
switches its task to `singlepoint`, as directed by the tutorial, and can
override mesh/pool count explicitly. The earlier full-KI PBE preset remains
a separate literature-oriented variant. See
[the launcher instructions](literature/README.md#exact-official-tutorial-input).

**Limits:** the direct QE/native numerical comparison here is Gamma-only.
Finite-q normalization/wrapping and multi-orbital assembly were reviewed in
source and tested against independent native supercell constructions, but a
matched finite-q, multi-Wannier QE/JDFTx silicon run has not been performed.
The official full-KI Python/KCP environment is absent, so its silicon DSCF
calculation has not been run here. Production silicon outputs were preserved;
the cause and corrected value of its full-KI gap are not established by this
audit.

## Comparison with the official coarse-grid workflow (13 October 2026)

The native q-only approximation and its adaptive refinement are **not** the
same method as `dfpt_coarse_grid` in the official `koopmans` package. Source
inspection uses repository commit `023fe26857466c5fae2520f5794dd5a86c416a84`.

The [official DFPT workflow](https://github.com/epfl-theos/koopmans/blob/023fe26857466c5fae2520f5794dd5a86c416a84/src/koopmans/workflows/_koopmans_dfpt.py#L143)
creates a separate coarse subworkflow, preserves the original SCF grid via
`scf_kgrid`, and changes `kpoints.grid` to `dfpt_coarse_grid`. It performs
coarse NSCF and Wannierization, then computes the screening coefficients on
that coarse electronic/Wannier grid. The parent performs fine-grid
Wannierization and constructs its Hamiltonian with the resulting alphas.
The [Wannierization workflow](https://github.com/epfl-theos/koopmans/blob/023fe26857466c5fae2520f5794dd5a86c416a84/src/koopmans/workflows/_wannierize.py#L134)
uses the explicit SCF grid for SCF, while NSCF and Wannierization use the
subworkflow grid.

Native screening instead retains the fine-grid orbital densities and
electronic response k integration while sampling fewer q points. Adaptive
refinement checks convergence of this q-only estimate and can recover the
native full-mesh result. It does not reproduce the separate coarse
Wannier/response model. The retained-q tests establish consistency with the
native full calculation, not equivalence to the official coarse workflow.

Matching the official approximation requires a separate coarse
NSCF/Wannier/DFPT calculation and an explicit, validated transfer of its
orbital screening coefficients to the fine-grid Hamiltonian. Merely
changing q weights, adding Ewald terms, or enabling adaptive refinement
does not establish that equivalence.
