# Wannier GPU optimization

The finite-difference MV and TruncatedDC paths now pack diagonal overlaps on
the device and read the packed result once per local k-mesh sweep. Full overlap
matrices stay on the GPU for gradient propagation. Objective-only evaluations
form only the Wannier overlap block and discard each overlap after packing.

A fused kernel accumulates both endpoint rotation derivatives without sliced,
scaled and transposed matrix temporaries. The same operation serves the center
Jacobian adjoint and explicit center derivatives. Matrix-free Jacobian products
contract their two rectangular factors directly into a packed device array,
with a single host read for the sweep. CPU implementations retain the same
formulas.

Band-contribution export also reads each rotation matrix once. Previously,
non-root MPI ranks could issue one device-to-host copy per matrix element.

## Measurement, 2026-10-29

A restart from the user's saved 27-Cu checkpoint was copied into independent
benchmark directories. All 243 centers were made free; the 648 source bands,
3x3x3 k mesh, energy windows and initial rotations were identical. Each run
performed ten L-BFGS iterations with both stopping thresholds disabled, using
two MPI ranks, two CPU threads per rank, and two RTX 4090 GPUs for CUDA runs.
The baseline binaries and shared libraries were saved before rebuilding commit
c3d3e8a1 plus these changes. No other GPU calculation was running.

| Build | Time for ten iterations | Whole process |
|---|---:|---:|
| Original GPU | 8.36 s | 55.54 s |
| Optimized GPU | 5.57 s | 36.32 s |
| Optimized CPU (same two ranks and two threads/rank) | 58.68 s | 129.25 s |

The GPU iteration speedup is 1.50x in this single comparison. Both runs report
initial spread 708.4939921073411 and final spread 706.6739385454717 bohr².
The process timing includes startup, checkpoint reading, diagnostics and exports.
The CPU run reached final spread 706.6739385454790 bohr². Its CPU thread
count matches the CUDA runs; this is not a search for the fastest CPU threading
configuration. These timings do not establish a universal GPU/CPU crossover, particularly for
small matrices where launch, eigensolver and MPI overhead remain significant.

The inputs, saved baseline binaries, logs, timing script and JSON results for
this measurement are in `/tmp/wannier-gpu-opt/` on the development machine.

## Validation

`checkFDOperators.cpp` compares the fused derivatives with the original matrix
expressions for full and rectangular overlaps, unequal endpoint dimensions,
self edges, several packing offsets and nonzero destination accumulators. It
also checks the center directional derivatives and device-resident diagonal
packing. Both CPU and CUDA variants are registered with CTest.

Validation passed: all 13 selected CTest checks; GPU isolated/entangled/free
subspace gradient checks for MV and TruncatedDC (MV on two MPI ranks); and
two-rank GPU hard-center checks with CG, L-BFGS and SQP, each exercising both
explicit and matrix-free center solvers.

Run the Wannier regression suite and GPU integrations with:

```bash
ctest --test-dir build -R '^wannier' --output-on-failure
python3 jdftx/test/wannier/checkTruncatedDC.py build --suffix _gpu
python3 jdftx/test/wannier/checkTruncatedDC.py build --suffix _gpu --measure FiniteDifference
python3 jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2
python3 jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2 --method LBFGS
python3 jdftx/test/wannier/checkFixedCenters.py build --suffix _gpu --np 2 --method SQP
```

The hard-center integration checks cover MV and TruncatedDC, explicit and
matrix-free solvers, finite-difference derivatives, target feasibility,
orthonormality and inner-window inclusion.
