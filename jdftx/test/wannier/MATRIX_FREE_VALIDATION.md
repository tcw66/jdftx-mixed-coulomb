# Matrix-free hard-center validation

Validated on 2026-10-22 with the local CPU build and two RTX 4090 GPUs.

The matched vacuum benchmark uses 32 Wannier orbitals, 96 source bands, a
2x2x2 mesh, a frozen inner window and slightly displaced FCC trial positions.
It restores the hard centers and evaluates the projected gradient at iteration
zero. These measurements do not establish full-minimization convergence.

| Measure | Solver | Persistent derivative MiB/rank | Total wall seconds | Maximum center error (bohr) |
|---|---|---:|---:|---:|
| FiniteDifference | Explicit | 35.484 | 1.343 | 4.67e-13 |
| FiniteDifference | MatrixFree | 2.262 | 0.727 | 4.68e-13 |
| TruncatedDC | Explicit | 35.484 | 1.555 | 4.66e-13 |
| TruncatedDC | MatrixFree | 2.262 | 0.837 | 4.66e-13 |

CPU measurements use one MPI process, two operator threads and one BLAS thread.
Persistent derivative storage excludes wavefunctions, full-band overlaps,
transient work and allocator caches. The explicit reference includes the
batched Gram-matrix reduction implemented in this change.

Matching objectives agree to better than 2e-12 bohr^2. Persistent derivative
storage is about 15.7 times smaller; measured total wall time is about 1.8–1.9
times shorter. CPU timings are illustrative and hardware/workload dependent.

The two-rank GPU benchmark also passed objective, center, orthonormality and
inner-window checks. Its persistent derivative storage was 18.141 MiB/rank
with Explicit and 1.132 MiB/rank with MatrixFree. GPU timing is omitted because
another production job was using both GPUs.

Additional regressions passed for both MV and TruncatedDC: finite differences,
explicit/matrix-free Jacobian products, the Jacobian/adjoint identity, partial
Cartesian targets, exported-rotation inner-window checks, and explicit/automatic
Si atomic projections. CPU MPI and two-GPU runs exercise distributed products.

Reproduce from the repository root:

```bash
ctest --test-dir build -R '^wannier' --output-on-failure
python3 jdftx/test/wannier/benchmarkFixedCenters.py build --work /tmp/new-center-benchmark
python3 jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2
python3 jdftx/test/wannier/checkFixedAtomicCenters.py build --np 2
```

The matrix-free solver is the default. Set
`wannier-center-constraint-solver Explicit` to use the reference pseudoinverse.
Failed linear solves are rejected; constraints are never converted to penalties.

## Graphene HCP feasibility check

The local graphene input was also checked with 600 source bands, a 12x12x1
mesh, a 4.651-bohr hexagonal basal cell and a 200-bohr height. Its 60 trial
orbitals comprise eight C s/p projections and 52 AB-stacked vacuum Gaussians.
The copied input uses [-1, 0.1] Ha (inner) and [-1, 1] Ha (outer). The test
uses two CPU MPI ranks, two operator threads per rank, MatrixFree, compatible
source overlaps, and zero localization iterations. It writes to a separate
`matrixfree-hcp-check/` directory; the production input/job is unchanged.

- Four Newton corrections reduced the maximum center residual from
  0.3165673 to 6.40e-12 bohr. Exported centers were independently checked
  against Gaussian input coordinates and reconstructed symmetry-adjusted
  atomic positions.
- The six-decimal atomic coordinates receive JDFTx's usual symmetry adjustment
  (1.34e-6 bohr RMS) before atomic trial centers are defined. They are not the
  literal unsymmetrized coordinates. The Gaussian targets are unmodified.
- Persistent derivative storage: 108.571 MiB per rank (maximum across ranks).
- Orthonormality error: 4.07e-12; inner-window inclusion error: 2.12e-12.
- Five normal solves used 365 operator products, including residual checks.
- Constraint preparation took 44.47 s and normal solves 93.93 s; total run
  time was about 330 s including initialization and exports. These are CPU
  measurements during another active GPU job and are not a CPU/GPU comparison.

This verifies feasibility and tangent projection for the large input. The
spread (1228.57084 bohr^2) is the iteration-zero value, not a converged
localization result. No explicit-solver timing comparison was run on this
large case.
