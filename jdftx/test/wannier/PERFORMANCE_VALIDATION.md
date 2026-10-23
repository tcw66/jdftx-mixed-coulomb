# Fixed-center performance changes

The retained changes preserve the center tolerance, objective, tangent-residual
checks, inner window, and orthonormality requirements:

- Complex CPU/CUDA column contractions replace full products whose off-diagonal
  elements were discarded by the matrix-free center Jacobian.
- Off-diagonal orbital rotations use the smaller Gram eigensystem. Analytic
  sinc/cosine limits handle zero and rank-deficient generators without division
  by small singular values; either rectangular orientation is supported.
  Very large rotation angles fall back to the dense exponential, preventing
  amplification of Gram roundoff in nearly-null modes. A rank-deficient
  alpha=1000 stress test motivated this safeguard.
- The matrix-free normal solver recycles eight positive-curvature secant pairs
  as an inverse preconditioner. Its preconditioner stays fixed within each PCG
  solve. The original diagonal solve is retried if recycling fails, and true
  residuals and final tangency remain checked.
- Restoring the old local state immediately before an update skips a redundant
  broadcast: the update communicates the new rotations itself. Restores that
  are not immediately followed by an update still communicate.
- Phase timings and line-search counts are printed for diagnosis.

A projected L-BFGS candidate was implemented, tested and **reverted**. On the
matrix-free CPU regression its extra tangent transports increased runtime
(about 0.12 to 0.21 seconds in the MV case) despite reducing iterations. The
retained optimizer is the original projected Polak–Ribiere CG. Generic
`dirUpdateScheme`/`history` settings remain ignored for hard-center optimization.

## Numerical checks

`checkStructuredUpdates` and `checkStructuredUpdates_gpu` compare the new
rotation against the full Hermitian exponential, including zero generators,
rank-deficient generators, both rectangular orientations, tiny/negative/large
steps, and a 24+384-dimensional outer-window example. They independently compare
complex column contractions with the diagonals of a full matrix product.
Required relative update agreement is 1e-10. CPU and GPU passed.

`checkFixedCenters.py` checks MV and TruncatedDC, all and partial targets,
Cartesian and lattice coordinates, explicit and matrix-free solvers, finite
differences, the adjoint identity, projected-gradient convergence, exported
center positions, orthonormality, and frozen-band inclusion. CPU and two-rank,
two-GPU runs passed. The final CPU maximum center errors were below 6.2e-9 bohr;
the two-GPU CG runs were below 6.3e-9 bohr, for a requested 1e-8 bohr tolerance.
The independent atomic-trial center regression and unconstrained TruncatedDC
regressions also passed. The discarded L-BFGS candidate also passed accuracy
checks; its rejection was based on overhead, not center accuracy.

## Timings (machine-specific, not universal speedups)

The isolated 600-row, 24+384 orbital update measured:

| Backend | Original full exponential | Structured update |
|---|---:|---:|
| CPU | 0.12049 s | 0.01239 s |
| GPU | 0.03494 s | 0.00864 s |

The 32-orbital/96-band benchmark (`benchmarkFixedCenters.py`) measured:

| Matrix-free normal solves | Original | Optimized |
|---|---:|---:|
| MV | 0.11329 s | 0.05473 s |
| TruncatedDC | 0.11264 s | 0.05517 s |

This benchmark measures restoration/projection with zero localization iterations;
it is **not** evidence for convergence to a minimum. Its objectives agreed to
about 1e-12 and maximum center errors were about 5e-13 bohr. Complete small-system
minimizations are covered by the separate integration checks above.

GPU timings were taken while another user calculation was active; they should
not be extrapolated to end-to-end graphene throughput. The ongoing user job was
not modified. Large restart comparisons use separate temporary directories,
with independent copies of writable Wannier checkpoints.

## Graphene restart comparison

The supplied 60-orbital, 600-band, 12×12×1 graphene checkpoint was restarted
for two CPU optimization steps on two MPI ranks, with identical inputs and
`alphaTstart 0.001`. This deliberately short local test checks trajectory and
cost; it does not establish graphene convergence.

| Quantity | Original | Retained implementation |
|---|---:|---:|
| Time from iteration 0 to iteration 2 | 118.72 s | 82.90 s |
| Normal-solver time | 66.28 s | 27.93 s |
| Normal-operator products | 254 | 288 |
| Final objective | 293.2865746247031 | 293.2865746247165 |
| Maximum center residual | 1.48144e-9 bohr | 1.48145e-9 bohr |

The exported centers differ by at most 6.9e-14 bohr. Frozen-band inclusion and
orthonormality errors remain below 3e-12. Startup and final dumping are excluded
from the iteration-time comparison; the complete process walls were 244.81 s
and 195.66 s. Product counts show that the recycled preconditioner does not
universally reduce iterations: this case used more products, but each product
was cheaper. Aggregate timings are not an ablation of individual changes.

[Machine-readable measurements](PERFORMANCE_RESULTS.json) include both the
32-orbital benchmark and the graphene restart trajectories. Small-system full
convergence tests, rather than this two-step restart, establish that accepted
changes preserve minimization to the specified projected-gradient tolerance.

## Reproduce

```bash
cmake --build build --target wannier wannier_gpu checkStructuredUpdates checkStructuredUpdates_gpu -j6
ctest --test-dir build -R '^structuredWannierUpdates' --output-on-failure
python jdftx/test/wannier/checkFixedCenters.py build
python jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2
python jdftx/test/wannier/checkFixedAtomicCenters.py build
python jdftx/test/wannier/checkTruncatedDC.py build
python jdftx/test/wannier/benchmarkFixedCenters.py build --work /tmp/new-center-benchmark
```

Do not assume that a smaller iteration count is faster: compare runtime to the
same center and projected-gradient tolerances, and inspect rejected trials.
