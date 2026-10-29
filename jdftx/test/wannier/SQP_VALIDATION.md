# Matrix-free trust-region SQP validation

Enable with `wannier-center-constraint-method SQP`. Projected CG remains the
default. The Hessian is applied by central differences of analytical Lagrangian
gradients, including first-order exponential-chart pullbacks. No localization
Hessian or KKT matrix is assembled. The existing normal-system solver and
nonlinear restoration enforce the original center targets.

## Regression results

CPU and CUDA `wannier` targets built successfully. CUDA execution was not tested
because the GPUs were occupied by an existing calculation. Seven CPU CTest
regressions cover CG, SQP, two-rank SQP, unconstrained TruncatedDC, ordinary and
SQP atomic projections, and selectively fixed centers. SQP convergence tests
exercise both the matrix-free and explicit normal solvers, all/partial targets,
Cartesian coordinates, independent center derivatives and adjoint identities,
Hessian symmetry and scalar Lagrangian curvature, exported orthonormality, and
inner-window inclusion.

For the small vacuum test with a frozen inner window, matrix-free results were:

| Localization / targets | CG steps | SQP steps |
|---|---:|---:|
| FiniteDifference / all | 34 | 6 |
| TruncatedDC / all | 33 | 6 |
| TruncatedDC / partial | 79 | 7 |

Every case reached a projected-gradient norm below 1e-6 and retained center
errors below 1e-8 bohr. Across the six serial SQP cases, the largest Hessian
symmetry error was below 4e-12 and the largest scalar Lagrangian curvature
check error was below 3e-8. These counts do not establish a runtime speedup:
each numerical Hessian product requires two gradient evaluations.

Reproduce the registered tests with:

```bash
cmake --build build --target wannier wannier_gpu -j2
ctest --test-dir build \
  -R '^wannier(FixedCenters(SQP(MPI)?)?|TruncatedDC|FixedAtomicCenters(SQP)?|SelectiveCenters)$' \
  --output-on-failure
```

## Bulk Nb near the Fermi level

The Nb test uses the existing nonmagnetic BCC Nb checkpoint: 12x12x12 k points,
20 source bands, 30 Hartree cutoff, SG15 Nb, and six s/d Wannier orbitals fixed
at the atom. The chemical potential is 0.651769107 Hartree; the inner window is
[EF-0.15, EF+0.05] Hartree and the outer window is [EF-0.30, EF+0.45] Hartree.
There are at most five inner-window bands and at least six outer-window bands
at every k point. Semicore bands are excluded. Initial rotations are built from
atomic trials; only source overlap matrices and their identity metadata are
copied into the isolated output directory.

The supplementary FiniteDifference run approached an individual overlap of
order 1e-6 and failed the matrix-free normal solve near iteration 47.
The CG probe also approached small individual overlaps and was stopped without
claiming convergence. This is a remaining limitation of the phase-based MV
center constraints for this initialization/window, not a passing Nb validation.
TruncatedDC uses k-averaged overlaps and avoids that individual-overlap
singularity, but the tested Nb quadratic subproblems remain poorly conditioned.

**The strict Nb stationarity test did not pass.** Bounded TruncatedDC tests,
including four-rank restarts and inner budgets up to 2,000 Krylov products,
continued to lower the objective and preserve the centers, but did not reach
the requested projected-gradient norm of 1e-6. They were stopped cleanly;
the regression's final convergence assertion correctly failed. These are
partial trajectories, not completed convergence or runtime benchmarks.

The final implementation also removes each orbital's k-independent global
phase from Newton directions, using its exact small Gram matrix. Independent
checks confirm that these phases leave the objective and centers invariant.
This removes a redundant direction but did not resolve Nb's remaining
conditioning/convergence problem in the bounded test.

The last run with the final phase projection ended at gradient norm
**0.0002939**, objective **63.3907282440688 bohr^2**, and maximum
exported center error **3.55965e-14 bohr**. Exported rotations
independently gave orthonormality error 9.72438e-14 and
inner-window inclusion error 5.43445e-14. Its starting point
was a compatible same-window checkpoint from the earlier test stages.

The earlier CG reference was interrupted at step 274, with objective
63.5836422211652 and gradient 0.02424; it did not converge either.
No end-to-end speedup over CG is claimed. SQP remains opt-in and experimental;
CG remains the default. A stronger tangent-Hessian preconditioner or another
strategy for the soft metallic modes is needed before recommending this SQP
implementation as a faster Nb minimizer.

[Retained results](SQP_RESULTS.json) record all test stages and their status.
Full final Nb logs and exported rotations are in `/tmp/sqp-nb-horizontal/sqp/`;
its summary/trajectory are in `/tmp/sqp-nb-horizontal/summary.json`. Source files
under `/home/tyler/Nb_DFPT` were only read. The final Nb run passed independent
center/Jacobian/adjoint, feasible-curve, global-phase and Lagrangian Hessian
checks; passing these checks does not imply stationarity.

To reproduce a fresh strict test from atomic trials:

```bash
python3 jdftx/test/wannier/checkFixedCentersNb.py build \
  --data /home/tyler/Nb_DFPT --cache build/me-niobium-input/wannier \
  --work /tmp/new-nb-sqp-validation --method SQP --np 4 --krylov 2000
```

For a compatible same-window restart, replace `--cache` with
`--restart /path/to/wannier`; `--radius` selects the initial trust radius.
Use a new output directory. The script fails if SQP does not reach the requested
gradient tolerance; `--allow-unconverged-cg` relaxes only the reference CG check.
