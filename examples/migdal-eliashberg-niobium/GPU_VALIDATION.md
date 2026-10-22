# CUDA validation — 2026-10-15

Backend: CuPy 13.6.0, device 0, NVIDIA GeForce RTX 4090. Device 1 selection was
also checked. CPU-only operation and explicit CUDA failure/automatic fallback
behavior have regression coverage. The implementation uses float64/complex128;
native input files and exported NPZ schemas retain the same physical conventions.

## Matched CPU/GPU comparison

The saved 4³ q-grid reference configuration was used: k=12³, q=4³, sigma=0.10 eV,
μ*=0.10, Matsubara cutoff 0.50 eV, Coulomb cutoff 0.20 eV. Both backends used
batch size 256, with `OPENBLAS_NUM_THREADS=1`. Results are saved separately under
`results-cpu-q4/` and `results-gpu-q4/`. Existing Nb outputs were preserved.

| Quantity | CPU | CUDA |
| --- | ---: | ---: |
| Spectral stage elapsed time | 62.4251 s | 4.92298 s |
| ME stage elapsed time | 0.0526 s | 0.1665 s |
| λ | 0.7082113229481 | 0.7082113229481 |
| Linearized ME Tc | 8.921986911985 K | 8.921986911984 K |

This single spectral-stage measurement gives a 12.7× speedup. It includes backend
initialization and file work and is not a repeated throughput benchmark. The
ME workload is small enough that GPU overhead exceeds its compute savings.

The largest per-q difference in the sum of mode λ values was 8.66e-13. Individual
couplings within degenerate phonon subspaces differ because of arbitrary basis
rotations. Spectral moments and Tc agree; these degeneracy rotations do not
change the isotropic spectrum. At 2, 4 and 6 K, Z, Δ and Φ agreed within the
comparison tolerances (absolute 1e-11, relative 1e-9). The largest difference
between the CPU and GPU BdG matrices at those temperatures was 2.1e-13 eV.
Real-space hoppings, pairings and eigenvalues also passed numerical comparisons.

## Current 8³ q-grid configuration

The current `nb.json` uses k=12³, q=8³, sigma=0.10 eV and μ*=0.13. A complete GPU
run using batch size 256 is saved in `results-gpu/`:

| Quantity | Value |
| --- | ---: |
| Spectral stage elapsed time | 19.3813 s |
| λ | 1.077750918 |
| ωlog | 14.9138 meV |
| Allen–Dynes Tc | 12.4791 K |
| Linearized ME Tc | 14.1718 K |
| Δ(iω0) at 2 K | 2.42187 meV |

All three temperature solutions and 24×24 BdG exports completed. Their BdG
symmetry residuals were zero at reported precision. This is a denser integration
test, not a converged prediction: native DFPT data still use a 4³ supercell,
and k-mesh, smearing and native-input convergence remain to be established.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 ME_TEST_GPU=1 python jdftx/test/dfpt/testMigdalEliashberg.py
ctest --test-dir build -R '^migdalEliashberg$' --output-on-failure
```

CUDA tests cover native weighted-cell vertex interpolation and zero-point
normalization, ME kernels, pairing eigenvalues and Tc, nonlinear solutions,
general coherent Nambu reduction, FFT coefficients, poles and device selection.
The ordinary CTest run remains CPU-compatible and skips opt-in GPU cases.
