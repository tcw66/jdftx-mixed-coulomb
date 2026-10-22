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
