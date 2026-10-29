Truncated density-convolution Wannier localization
================================================

Select `wannier localizationMeasure TruncatedDC`. This implements Eq. (7)
of Li, Ko, DiStasio and Damle, PRB 110, 085127 (2024), rather than the full
real-space density-convolution integral. It uses the existing finite-difference
neighbor shells and overlap files. The default remains FiniteDifference.

For entangled bands, specify `outerWindow <min> <max>` and, optionally,
`innerWindow <min> <max>` (energies in Hartree). Include enough DFT bands
and Wannier centers to contain the inner window at every k point. For example:

```
wannier localizationMeasure TruncatedDC outerWindow -1 1 innerWindow -1 0
```

JDFTx jointly optimizes its rectangular subspace and the gauge, retaining
inner-window constraints, as in the variational approach of Damle, Levitt and
Lin, MMS 17, 167–191 (2019). No separate MV disentanglement pass is used.
This change does not implement the paper's SCDM initialization: existing trial
orbitals and compatible rotation checkpoints are used.

The objective is `2 sum_(b in half mesh) wb sum_n (1-|rho_n(b)|)`, with
`rho_n(b) = sum_k weight_k M_nn(k,b)`. Averaging precedes the modulus;
MPI ranks reduce these complex averages before evaluating the objective and
gradient. At an exactly zero average, the modulus is nonsmooth and the code
selects the zero subgradient. Existing finite-difference weights are reused;
the lower-bound interpretation assumes nonnegative weights.

Reported per-orbital spreads are TDC spreads, not MV variances or exact DC
spreads. Centers are phase-based estimates on branches near the trial centers;
for diffuse orbitals they can be ambiguous and are not variational DC centers.
`OmegaI` remains an MV subspace diagnostic and is not a TDC decomposition.
Soft-pinned centers are rejected. Frozen Wannier orbitals and inner-window bands
continue to use the existing constraints. Position and position-momentum
matrix exports retain their existing finite-difference definitions.

Run `python3 jdftx/test/wannier/checkTruncatedDC.py build` from the repository
root for isolated- and entangled-band Si minimizations (with and without an inner
window), pinned-center rejection, and numerical gradient
checks. It uses a temporary directory and the build's SG15 silicon potential.
Set `WANNIER_MPI='mpirun -np 2'` to exercise distributed Wannier gradients.

Hard center equality constraints
--------------------------------

`wannier fixCenters yes` (as part of the existing wannier command) constrains
centers at the input trial positions with either reciprocal-space measure.
The default `wannier-center-constraint-solver MatrixFree` uses analytical
Jacobian/adjoint products and PCG on the positive-semidefinite normal system.
The preconditioner is positive overlap-based row scaling. The solver checks
the true residual and rejects unsuccessful linear solves; it never substitutes
a penalty or relaxes the inner window. `Explicit` retains the stored-Jacobian
pseudoinverse reference, with one batched MPI reduction for its Gram matrix.

`checkFixedCenters.py BUILD` runs both solvers with full and partial targets,
including Cartesian coordinates. It checks central differences, the adjoint
identity and (for up to 24 scalar constraints) matrix-free products against
the explicit rows. It independently inspects exported rectangular rotations
to verify orthonormality and frozen-band inclusion. `--solver MatrixFree` or
`--solver Explicit` selects one path. Add `--suffix _gpu --np 2` for CUDA/MPI.
`checkFixedAtomicCenters.py` covers explicit and automatically added atomic
projections, and accepts the same suffix/rank options and either solver.

`benchmarkFixedCenters.py BUILD --work /new/directory` compares both solvers
for 32 vacuum orbitals and 96 source bands with an inner window. It checks
matching feasible objectives and center errors, reports wall/solver times and
persistent derivative storage, and writes `summary.json`. This is a zero-step
localization benchmark of restoration and tangent projection, not convergence
of a full minimization. Reported storage excludes source states, full-band
overlaps and transient work. GPU timings require an idle allocation for a
meaningful comparison; no speedup is asserted by the script.

`checkStructuredUpdates` and `checkStructuredUpdates_gpu` independently validate
structured outer-window exponentials and complex column contractions against
dense matrix references, including rank-deficient cases. See
[performance validation](PERFORMANCE_VALIDATION.md) for retained optimizations,
timings, accuracy checks, and the rejected projected L-BFGS candidate.

Projected limited-memory BFGS
-----------------------------

Select `wannier-center-constraint-method LBFGS` to reuse objective curvature
without numerical Hessian products. For example, with existing hard centers:

```text
wannier-center-constraint-method LBFGS
wannier-minimize nIterations 2000 history 100 energyDiffThreshold 0 knormThreshold 1e-6
```

The default method remains CG. LBFGS uses the existing center normal solver,
Newton restoration, and inner-window-preserving orbital rotations. New secants
use projection transport of the previous gradient and accepted direction.
Positive-curvature pairs define an inverse metric in the rotation-generator
coordinates; applying `P H P` gives a tangent search direction. Stored pairs
are not individually projected at each iteration, avoiding two normal solves
per history entry. This is a safeguarded projected quasi-Newton method, not an
implementation of exact parallel-transport Riemannian BFGS.

The initial inverse scale is `s.y / y.y`, bounded to [1e-6, 1e8]. Pairs with
nonpositive or numerically weak curvature are skipped; failed transport or
loss of descent clears the history. Armijo backtracking checks the actual
restored objective. The internal center restoration tolerance is the smaller
of the requested tolerance and 1e-11 bohr, to reduce noise in curvature pairs.
Each orbital's invariant global phase is removed as in SQP; `fdTest yes`
independently checks phase invariance, objective/center derivatives, and the
restored feasible-curve derivative.

`wannier-minimize history` controls the number of pairs (default 15, must be
positive). A history of 100 helped the small silicon problem; it is not a
universal choice. Storage grows as two distributed WannierGradient vectors
per pair, and the two-loop recursion requires MPI reductions. The log reports
peak history storage and objective/gradient evaluation counts. `alphaTstart`
controls the first steepest-descent trial; subsequent quasi-Newton trials start
at unit step length. Generic direction and line-search choices do not apply.

The completed matched silicon test took 6909 CG steps, 296 LBFGS steps with
history 100, or 51 SQP steps. Corresponding iteration times were 34.82, 3.39,
and 7.86 seconds on this machine. All met the same 1e-6 projected-gradient and
1e-8-bohr center tolerances. These are small-system measurements; the bounded
Nb tests do not establish a completed metallic speedup. See
[iteration-speedup validation](ITERATION_SPEEDUP_VALIDATION.md).

Reproduce a strict comparison from one shared silicon SCF state:

```bash
python3 jdftx/test/wannier/benchmarkFixedCenterOptimizers.py build \
    --work /tmp/new-fixed-center-comparison --history 100
```

The script disables energy-change stopping, records every trajectory in
`summary.json`, checks exported centers/orthonormality and inner-window
inclusion, and fails if any method misses the gradient threshold. Use
`--scf-prefix /path/to/scf` to read a compatible saved state, `--np 2` for MPI,
or `--methods LBFGS` for the shorter strict silicon regression.

Matrix-free trust-region SQP
---------------------------

Select `wannier-center-constraint-method SQP` with hard center targets to use
trust-region sequential quadratic programming. The default method remains `CG`. SQP is experimental: the small regressions pass,
but the bounded bulk-Nb tests have not demonstrated convergence to the strict
stationarity threshold or a runtime speedup; see [SQP validation](SQP_VALIDATION.md).
`wannier-center-constraint-solver MatrixFree|Explicit` independently selects the
normal/tangent solver; neither choice stores the localization Hessian.

SQP estimates the center multipliers at each feasible iterate and applies the
Lagrangian Hessian by central differences of analytical objective and center
adjoint gradients, with frozen multipliers. Orbital rotation chart corrections
are included: differencing the moving projected objective gradient alone would
miss curvature. Probe displacements have norm 1e-4 in the WannierGradient metric.
The tangent projector also removes each orbital's k-independent global phase
using its small exact Gram matrix. These phases leave the centers and objective
invariant and otherwise introduce redundant Newton directions. `fdTest yes`
checks phase removal, the center Jacobian, and scalar objective invariance.
Steihaug CG approximately solves the tangent quadratic subproblem, truncating at
the trust boundary or nonpositive curvature. The initial radius is 0.5, the
maximum radius is max(10, sqrt(Nk*Ncenters)) in the summed gradient metric,
and the Krylov limit is 200. Set `wannier-center-constraint-sqp-krylov <iterations>`
to tune the subproblem solve limit; larger metallic meshes may require substantially
more iterations to meet the inner residual target. Set
`wannier-center-constraint-sqp-radius <radius>` to choose the initial trust radius
(for example when resuming a checkpoint). The maximum radius also includes the
requested initial radius. Actual/predicted objective
reduction controls acceptance and the radius. Every accepted step is restored
to the original center targets (internally to min(requested tolerance, 1e-11)
bohr); orthonormality and inner-window inclusion are preserved by the rotations.
Generic `wannier-minimize` direction, history, line search, and `alphaTstart`
settings do not control SQP. Iteration and convergence thresholds still apply.

This is a numerical-Hessian Newton implementation, rather than an analytical
Hessian implementation. Each Hessian product requires two gradient evaluations,
center adjoints, and rebuilding the base Jacobian cache. Smaller iteration
counts need not reduce runtime. `fdTest yes` additionally checks Hessian symmetry
and its quadratic form against independent scalar Lagrangian second differences.
The output reports products, rejections, and negative-curvature truncations.
Hessian timings include orbital updates, gradients and cache preparation; these
timing categories overlap.

```bash
python3 jdftx/test/wannier/checkFixedCenters.py build --method SQP
python3 jdftx/test/wannier/checkFixedCenters.py build --method SQP --np 2
python3 jdftx/test/wannier/checkFixedAtomicCenters.py build --method SQP
ctest --test-dir build -R '^wannierFixedCentersSQP' --output-on-failure
```

`checkFixedCentersNb.py` compares CG and SQP on an existing bulk BCC Nb
12x12x12, 20-band checkpoint, without running a new SCF calculation or changing
source files. Six s/d orbitals are fixed at the Nb atom. The inner window is
[EF-0.15, EF+0.05] Hartree and the outer window [EF-0.30, EF+0.45] Hartree;
the semicore bands are excluded. The default measure is TruncatedDC; use
`--measure FiniteDifference` for MV localization. Both methods start from the
same atomic trials. An optional cache reuses only overlap matrices and their
identity metadata, since rotations for different windows may be incompatible.
The script independently checks exported centers, orthonormality, and
inner-window inclusion, and saves timings and convergence to `summary.json`.

```bash
python3 jdftx/test/wannier/checkFixedCentersNb.py build \
  --data /home/tyler/Nb_DFPT --cache build/me-niobium-input/wannier \
  --work /tmp/new-nb-sqp-comparison --method SQP
```

To run both methods, omit `--method SQP`. Add `--allow-unconverged-cg` to
record a CG reference at its iteration limit while still requiring SQP convergence.

`--restart /path/to/wannier` resumes compatible rotations and overlaps for the
same source state and windows; outputs are still written only under `--work`.
