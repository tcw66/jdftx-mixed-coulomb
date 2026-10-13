# Adaptive KI screening

The screening-mesh command now selects an initial mesh, followed by refinement
with reuse of every completed orbital/q response. CPU `wannier` and CUDA
`wannier_gpu` were rebuilt with this change on October 12, 2026.

```text
wannier-koopmans-screening-mesh 2 2 2 0.005
```

On the current 8×8×8 electronic mesh, this evaluates 2×2×2, then 4×4×4,
then 8×8×8 as required. Leave the optional line in `ki-screening-mesh.in`
commented for a direct full-mesh calculation. Existing running processes
continue with the executable and input they originally loaded.

The default tolerance is 0.005 eV per manifold. Accepting a coarse result
requires **two successive** refinements below this tolerance. Consequently,
starting at 2×2×2 on an 8×8×8 target reaches the full mesh: there are only
three levels. Starting at 1×1×1 permits earlier acceptance at 4×4×4 if both
preceding changes pass. A larger electronic target can permit earlier
acceptance too. No speedup is guaranteed for silicon.

For an explicit fixed-mesh diagnostic, append a zero tolerance:

```text
wannier-koopmans-screening-mesh 2 2 2 0
```

This reproduces the old quadrature approximation. Source-only symmetry audits
also retain the requested mesh. Solver benchmarks explicitly select fixed
subsets so the solvers are compared on the same problem.

## Why change the workflow?

The retained-q tests found that the coarse calculations solve the same response
equations as the full calculation. In the production valence calculation,
identical orbital checkpoints gave identical retained response values, yet
2×2×2 quadrature yielded alpha ≈0.268240 while full 8×8×8 gave ≈0.181382.
The remaining issue is sparse integration of signed bare and relaxed
curvatures, especially where their cancellation leaves a small screened
curvature. A negative coefficient by itself cannot identify a code defect.

This replacement controls refinement and acceptance of the approximation. It
does not introduce dielectric interpolation, change the underlying response
kernel, fit epsilon, or reproduce the tutorial's full KI functional. The
native Hamiltonian remains KI(2). The [official QE KCW documentation](https://www.quantum-espresso.org/Doc/INPUT_kcw.html)
describes the separately supplied macroscopic dielectric constant for its
periodic Coulomb correction.

## Implementation

Electronic k integration, orbital sources, the bare Hamiltonian, and the
Hamiltonian's Ewald term retain the full electronic mesh. At each screening
level, both curvatures in the screening ratio use that level's q quadrature
and its independently constructed supercell Ewald term. Epsilon=0 retains
the neutralizing-background convention at every level.

Meshes grow through nested divisors of the electronic folding. Completed
responses are retained in memory and seed the verified symmetry cache at
each level; no already computed orbital/q response is solved again. MPI
response groups keep their ownership throughout refinement. Negative or
nonfinite coarse alpha prevents early acceptance. At the full mesh, negative
alpha is reported without clipping; invalid or vanishing final curvatures
still fail the calculation.

For each pair of successive levels, the code forms the difference of the
Hermitian KI corrections, using the same upper-triangle screening convention
as the final Hamiltonian. Its maximum Frobenius norm over native k points
bounds each eigenvalue's change **between those two meshes**. Two consecutive
changes must pass the tolerance. This is a convergence estimate; it is not a
rigorous bound on the remaining mesh error, the combined valence/conduction
gap error, or arbitrary-k interpolation errors.

`koopmansScreening` records the requested and final meshes, screening mode,
tolerance, every stage's bare/relaxed curvatures and alpha, and Hamiltonian
change estimates. `koopmansResponse` retains its original column layout and
reports the final selection and Ewald term. Full-mesh screening remains the
default when the command is omitted.

## Validation

The independent H2 regression passed on both CPU and CUDA, including serial
and MPI runs. It covers fixed and adaptive screening, occupied
and empty orbitals, PBE charge/magnetic sectors, Ewald and background
conventions, serial execution and two independent MPI q groups. It checks
that refinement reaches the same full-mesh Hamiltonian as a direct run,
that completed responses are not solved twice, and that the reported
Hamiltonian-change bound agrees with native eigenvalue changes for a
one-orbital model. An 8×1×1 fixture with a deliberately loose tolerance
exercises early acceptance after two small changes and verifies equality
with a direct fixed 4×1×1 run. That loose tolerance is only a control-flow test.

The frozen-orbital silicon regression uses PBE/SG15, a 4×4×4 electronic mesh,
12 Ha cutoff, epsilon=0, and two CPU MPI q groups. It is a numerical fixture,
not a converged band-gap benchmark. Both manifolds refine from 2×2×2 to
4×4×4 and match independently saved direct full-mesh results:

| Manifold | Maximum response difference (Ha) | Maximum alpha difference |
|---|---:|---:|
| Valence | 1.40e-14 | 6.70e-14 |
| Conduction | 5.00e-15 | 1.97e-13 |

Its conduction coefficients change from approximately -0.475260 to
-0.0484765. The direct full-mesh reference also has the latter negative
values. This verifies full-mesh agreement without enforcing positivity;
further electrostatic/model convergence remains necessary for that fixture.

Reproducible logs, inputs, checkpoints and comparison JSON are under
`archive/coarse-mesh-audit-20261012/adaptive-tests/`. The silicon driver is
`check-silicon-adaptive.py` in that directory. The tracked H2 test is
`jdftx/test/koopmans/checkScreeningMesh.py` in the JDFTx source tree. Use
`/usr/local/bin/mpiexec` on this machine; `/usr/bin/mpirun` loads an incompatible
system PMIx library.

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

The official-style separate electronic-grid workflow is now implemented: see [official-coarse-workflow.md](official-coarse-workflow.md). The q-only/adaptive method described above remains a distinct option.
