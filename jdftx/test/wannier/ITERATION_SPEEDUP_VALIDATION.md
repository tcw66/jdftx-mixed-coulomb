# Fixed-center iteration reduction

Validated on 2026-10-28. The new opt-in method is
`wannier-center-constraint-method LBFGS`; CG remains the default. This work
extends the pre-existing experimental SQP implementation without replacing it.

## Completed comparisons

All comparisons below disable energy-change stopping and require a projected
gradient below 1e-6 and maximum center displacement below 1e-8 bohr. Every
accepted state satisfies the requested center tolerance. Orthonormality and
inner-window inclusion are checked independently of the localization gradient.

The small vacuum tests use 24 source bands, four Gaussian trials, a 2x2x2
mesh, and a frozen inner window. MatrixFree results with history 15 are:

| Objective / fixed targets | CG steps | LBFGS steps |
| --- | ---: | ---: |
| FiniteDifference / all four | 34 | 24 |
| TruncatedDC / all four | 33 | 20 |
| TruncatedDC / one Cartesian target | 79 | 33 |

Explicit reference solves and two-rank LBFGS runs also pass the complete
gradient-convergence and exported-center checks. Tests include the center
Jacobian/adjoint identities, restored feasible-curve derivative, phase
invariance, exported orthonormality and independent frozen-band inclusion.

The harder silicon test uses eight s/p atomic trials, 12 source bands, a
2x2x2 mesh, 5 Ha cutoff, inner window [-100,-0.15] Ha and outer window
[-100,100] Ha. All methods start from identical atomic trials and one saved
SCF state. `fdTest no` avoids derivative-check overhead in the timing runs.

| Method | History | Steps | Gradient evaluations | Iteration time |
| --- | ---: | ---: | ---: | ---: |
| CG | ignored | 6909 | 6910 | 34.82 s |
| LBFGS | 100 | 296 | 297 | 3.39 s |
| Existing SQP | ignored | 51 | 4036 | 7.86 s |

Thus LBFGS uses about 23 times fewer outer steps and about 10 times less
iteration time than CG in this test. SQP uses fewer outer steps, but its
1,992 central-difference Hessian products cost more gradient evaluations.
Whole-process times, including startup and dumps, were 35.01, 3.61 and
8.06 seconds respectively. Timings are specific to this machine (i9-12900K,
two OpenMP threads, one BLAS thread); they are not universal speedups.

Final spreads were 42.14641399366801 (CG), 42.14641398910421 (LBFGS), and
42.14641398910429 (SQP) bohr squared. Final projected-gradient norms were
9.733e-7, 5.722e-7 and 5.128e-8. Maximum exported center errors were
6.50e-10, 4.04e-12 and 2.84e-12 bohr. Exported orthonormality errors were
below 2.5e-12 and inner-window inclusion errors below 1.4e-12.
These checks establish convergence to the requested first-order tolerance,
not a global-minimum guarantee.

History length matters: in separate exploratory silicon runs, history 15
required 1,031 steps and history 50 required 491 steps. Projecting every stored
pair individually was tested and discarded: that variant took 1,246 steps
and about 59 seconds. The retained method projects the inverse metric's
output and only transports the new secant, avoiding history-length-dependent
normal solves. Larger histories still cost vector storage and MPI reductions.

## Implementation and checks

Positive-curvature secants define an SPD limited-memory inverse metric in
rotation-generator coordinates. The direction is obtained with `P H P`,
using the original center tangent projector and global-phase removal. A
curvature safeguard skips nonpositive/weak secants. Direction failures clear
history; restored-objective Armijo checks still govern acceptance. Internal
restoration uses min(requested tolerance, 1e-11 bohr), reducing noise in
gradient differences. This is a safeguarded projected quasi-Newton method,
not an exact parallel-transport Riemannian BFGS implementation.

CPU and CUDA binaries build successfully. Ten CPU CTests pass, covering CG,
SQP, LBFGS, two-rank SQP/LBFGS, a complete strict silicon LBFGS solve,
unconstrained TruncatedDC, atomic trials, and selective hard centers.
CUDA execution was not tested because the GPUs were occupied by an existing
user calculation. That calculation and source SCF/Wannier files were not modified.

## Niobium model update

The initial six-s/d-orbital tests remain unconverged. A 500-step LBFGS run
with history 15, resumed from the earlier SQP checkpoint, ended at projected
gradient 1.894e-4 while retaining center, orthonormality and inner-window
accuracy. A history-100 six-orbital attempt was stopped cleanly when the
user supplied the new p-orbital context. Neither is a completed speedup.

The user's `/home/tyler/Nb_DFPT/pseudopotentials/Nb-SG15-5p/Nb_SG15_PBE_5p.upf`
adds zero-occupation 5p trial orbitals. Its local potential, density, and
nonlocal Hamiltonian blocks match the original SG15 numerical data. Existing
Bloch states can therefore be reused. The added JDFTx trials are `2px`, `2py`
and `2pz`; `px`, `py`, `pz` select the first, semicore p shell instead.

Adding three Wannier functions changes the variational model. It is not an
optimizer-only modification, and absolute six/nine-orbital spreads should
not be compared as an optimizer speedup. The user's saved interpolation
comparison provides a separate reason for this basis: nearest-band RMS
error for the same DFT states within 1 eV of EF decreases from 88.7 to
4.84 meV for the nine-orbital model. This is evidence about that saved
interpolation, not a validation of newly constrained output.

The old narrow outer window contains as few as six bands at some k points
and cannot support nine Wannier functions. The new matched nine-orbital
comparison uses the user's windows: inner [0,0.70] and outer [0,1.7] Ha.
The outer window contains at least twelve bands and the inner window at
most five. Each optimizer reads an independent writable copy of the same
nine-orbital checkpoint. The benchmark checks exported rotations, center
accuracy and actual projected-gradient convergence; partial trajectories
are recorded as unconverged.

Both nine-orbital restart runs were stopped cleanly after about 6.5 minutes.
At iteration 100, CG had spread 74.68242037081961 and projected gradient
6.093e-2; LBFGS had spread 72.26043739778537 and gradient 5.768e-2.
Neither approached the strict 1e-6 threshold, and LBFGS took longer per
iteration in this case. This is not evidence of an end-to-end Nb speedup.
Exported center errors stayed below 4.2e-14 bohr, and orthonormality and
inner-window errors below 1e-12. Full partial trajectories are retained in
`/tmp/wannier-lbfgs-spd-nb/summary.json` and
`/tmp/wannier-cg-spd-nb/summary.json`. For cleanly interrupted runs the JSON
gradient is the last printed stationarity measurement; exported-state
invariants are measured after final dumping.

Initialization is another significant issue: the unconstrained checkpoint's
centers move by up to about four bohr during initial feasibility restoration,
raising its feasible starting spread to 539.355 bohr squared. A separate
one-step probe from atomic s/p/d trials starts feasible at 99.704 bohr squared
and decreases to 95.504 after one step. That probe is not gradient-converged.
Use direct atomic trials for further full comparisons, or resume an already
constrained checkpoint with matching windows. Adding p orbitals is useful
for the model, but has not by itself resolved the hard-center optimizer's
Nb convergence difficulty.

## Reproduce

From the repository root:

```bash
cmake --build build --target wannier wannier_gpu -j2
ctest --test-dir build \
  -R '^wannier(FixedCenters(SQP(MPI)?|LBFGS(MPI|Silicon)?)?|TruncatedDC|FixedAtomicCenters(SQP)?|SelectiveCenters)$' \
  --output-on-failure -j2
python3 jdftx/test/wannier/benchmarkFixedCenterOptimizers.py build \
  --work /tmp/new-fixed-center-optimizer-comparison --history 100
```

For a saved compatible silicon state, add
`--scf-prefix /path/to/scf`. Full matched silicon logs/trajectories from this
validation are in `/tmp/wannier-iteration-final/`; its `summary.json` retains
the trajectory for every method. The small vacuum comparison logs are in
`/tmp/wannier-iteration-cg-vacuum/` and `/tmp/wannier-lbfgs-metric-vacuum/`.

For the augmented nine-orbital Nb model, use a new work directory for each method:

```bash
python3 jdftx/test/wannier/checkFixedCentersNb.py build \
  --data /home/tyler/Nb_DFPT --cache /home/tyler/Nb_DFPT/wannier \
  --work /tmp/new-nb-spd-lbfgs --method LBFGS --np 4 --history 100 \
  --orbitals spd --inner-min 0 --inner-max 0.70 --outer-min 0 --outer-max 1.7
```

This initializes from atomic trials while reusing compatible overlaps.
Use `--restart` instead to resume compatible constrained rotations.
Source files are read only. The script fails
if the selected method does not reach the requested projected-gradient
threshold, retaining partial results in `summary.json`.

[Retained machine-readable results](ITERATION_SPEEDUP_RESULTS.json) distinguish
completed comparisons from bounded, unconverged tests.
