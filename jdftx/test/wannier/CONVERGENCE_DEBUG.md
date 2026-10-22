# Small-system fixed-center convergence investigation

Investigated on 2026-10-22 with CPU, MPI and CUDA builds. Tests disable
energy-change stopping when measuring gradient convergence. A small center
error is recorded separately from localization convergence.

## Findings and implementation changes

- Raw objective derivatives, center derivatives and the adjoint identity pass.
  The added central-difference test follows a **restored feasible curve**;
  its error on the 14-orbital graphene example is approximately 1e-9.
- A normal-system residual small relative to `J*rawGradient` can still leave
  an appreciable center-changing component in the projected result. The
  optimizer now checks `J*tangentGradient`, refines the projection, and
  rejects a result whose tangency cannot be resolved accurately.
- A failed projection of the previous CG direction previously aborted a run
  even when the current projected gradient was valid. It now restarts from
  projected steepest descent.
- Periodic PCG restarts every 32 iterations discarded useful Krylov information
  in weak constraint directions. Restarts now occur when a candidate solution
  fails its independently recomputed true-residual check.
- The explicit reference equilibrates constraint rows before eigenvalue
  truncation. This reduces scale-induced loss of independent constraints.
- The arbitrary localization step-size cap of three was removed. Armijo
  backtracking still controls acceptance. This helps some soft modes, but is
  not a guarantee of fast convergence for poorly conditioned problems.

With `fdTest yes`, each optimization iteration reports unnormalized raw and
projected norms, the norm of `J*tangentGradient`, phase-cut margin (radians),
and minimum diagonal overlap magnitude. These extra diagnostics cost work;
they are disabled in normal runs. A separate warning flags overlaps near zero
or phases near their cut. The feasible-curve derivative check temporarily
uses center restoration at 1e-12 bohr or tighter if requested. Production
optimization retains the configured center tolerance.

## Smooth convergence tests

The strengthened four-function vacuum regression uses 24 source bands, a
2x2x2 mesh and an inner window. It now requires projected-gradient convergence
below 1e-6 with `energyDiffThreshold 0`, rather than merely a decreasing spread.
Both Explicit and MatrixFree pass, including a partial Cartesian target.
Representative CPU MatrixFree results:

| Case | Iterations | Final projected gradient | Maximum center error (bohr) |
|---|---:|---:|---:|
| MV, four fixed trial positions | 34 | 6.78e-7 | 7.50e-10 |
| TruncatedDC, four fixed trial positions | 33 | 8.39e-7 | 3.50e-9 |
| TruncatedDC, one Cartesian target | 79 | 9.67e-7 | 6.15e-9 |

CPU and two-rank GPU checks pass these same convergence assertions.

Si tests use eight atomic s/p projections, 12 source bands, a 2x2x2 mesh and
an inner window. Both explicit and automatically added projections are
checked. MV reached a projected gradient below 1e-6 within 600 iterations in
exploratory CPU runs. TruncatedDC was slow, rather than stationary at its
1000-iteration value. A longer MatrixFree run with explicit atomic trials and
the final adaptive line search converged in 4827 iterations:

- Raw gradient norm: 0.1548995.
- Projected gradient norm: 9.7338e-7.
- Tangent residual: 1.8673e-11.
- Maximum center error: 7.962e-9 bohr.
- Orthonormality error: 1.539e-12; inner-window inclusion error: 4.320e-13.

Thus a nonzero raw gradient is compatible with the measured constrained
stationarity. This long run is not part of the short default Si regression.
For comparison, the version retaining the step cap converged after 5026
iterations on the same saved Si state. Iteration counts depend on initialization
and floating-point trajectories.

## Remaining difficult case: coarse graphene with vacuum projections

The diagnostic reproducer uses two C atoms, eight atomic s/p projections and
six AB-stacked Gaussian guesses, a 4.651-bohr basal cell, 24-bohr height,
40 source bands, 8-Ha cutoff, inner window [-1,0.1] Ha and outer window
[-1,1.2] Ha. Its SCF calculation is converged before Wannier optimization.

On a 2x2x1 mesh, some diagonal overlaps decrease toward zero. Near failure,
one MatrixFree run had minimum overlap 4.74e-5, raw norm about 1487, and tangent
norm about 0.383. The phase margin remained about 2.85 radians: this observed
failure was **not** a phase-cut crossing. Normal solves eventually become
unreliable. The new implementation rejects the projection; it does not label
that state converged or suppress its gradient.

Before the accuracy check, the explicit solver could instead continue with a
large `J*tangentGradient` residual and eventually fail its line search. Making
all center restoration tolerances 1e-12 did not fix the coarse-mesh plateau,
so that blanket production change was not retained.

On a 4x4x1 mesh, a 150-step MatrixFree check remains well conditioned: minimum
overlap about 0.828, tangent residual 2.60e-10, and projected gradient reduced
from 5.72 to 0.0285. It is still not gradient-converged after 150 iterations.
These meshes represent different discrete problems; compare conditioning and
convergence behavior, not their absolute objectives as if they were identical.
There is no demonstrated universal cure for the coarse ill-conditioned case.

## Reproduction

From the repository root:

```bash
ctest --test-dir build -R '^wannier' --output-on-failure
python3 jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2
python3 jdftx/test/wannier/debugFixedCenterConvergence.py build \
    --work /tmp/graphene-center-debug --mesh 2 --iterations 150
# Use a separate work directory to compare --mesh 4.
```

The diagnostic graphene driver writes inputs, SCF states, logs and a JSON
summary for both measures and the explicit MV reference. Solver failures are
recorded as failures, not treated as successful convergence. Its purpose is
to reproduce difficult cases; it is not an assertion that every case converges.
