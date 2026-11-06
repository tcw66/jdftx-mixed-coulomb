# PAW validation record — 2026-11-05

The tests below were run in `jdftx_defect_coulomb` using the supplied JTH v2.0 PBE XML archive. CPU and CUDA binaries were rebuilt, including the native Wannier executables. CUDA execution used an NVIDIA RTX 4090. These results establish initial implementation coverage; further material-specific convergence and cross-code benchmarks remain necessary.

## Automated checks

```bash
cmake --build build -j 8
ctest --test-dir build -R '^(paw|berryPhase$|kMeshCorrelation$|batchedProjectors$)' --output-on-failure
CUDA_VISIBLE_DEVICES=0 python3 jdftx/test/paw/checkPAWWorkflow.py \
  build/jdftx_gpu build/wannier_gpu \
  pseudopotentials/JTH-v2.0-GGA-PBE-atomicdata-xml/Si.GGA_PBE-JTH.xml
```

All ten selected CTest checks passed, along with the independent CUDA workflow. The checks include:

- On-site PBE energy derivatives for H, Si, and Zr, plus multipole normalization and analytic Gaussian electrostatics.
- Periodic PAW energy versus orbital derivatives, metric/overlap consistency, and momentum Hermiticity, on one and two MPI ranks. Representative full-functional derivative errors were below `1e-7` Hartree per test direction.
- Rejection of truncated or nonfinite arrays, unknown grids, duplicate state IDs, DTDs, and unsupported functionals.
- SCF, AE wavefunction/matrix export, real-space inverse FFT and Parseval identity, independent reconstruction-cutoff convergence, fixed-density and fixed-potential restart, and native Wannier Hamiltonian/momentum output.
- Rejection of unsupported spin/force requests and missing PAW restart data.
- Existing norm-conserving Berry-phase, k-mesh, and batched-projector regressions.

An archive-wide reader scan accepted 185 of 186 datasets. The full-resolution Ba dataset passes after radial-mesh biorthogonalization. `Ba.GGA_PBE-JTH_light.xml` exceeds the 0.05 input projector-duality tolerance and is rejected. Dataset acceptance alone does not establish physical accuracy for all elements.

## QE comparisons

QE 7.5 was run using the matching JTH v2.0 UPF files from the ABINIT JTH repository, with PBE, identical cells, and unshifted `2 x 2 x 2` k meshes. Comparisons use eigenvalues relative to the lowest Gamma-point band, which removes the arbitrary common energy reference.

| System | Geometry | Wavefunction cutoff | Density cutoff | Largest band-splitting difference |
|---|---|---:|---:|---:|
| Si, 2 atoms | Diamond, cubic parameter 10.26 bohr | 12 Ha / 24 Ry | 48 Ha / 96 Ry | 3.1 meV |
| Zr, 1 atom | BCC, cubic parameter 6.8 bohr | 15 Ha / 30 Ry | 60 Ha / 120 Ry | 19 meV |

The Zr comparison includes semicore bands and eight total bands, with Fermi–Dirac smearing of 0.01 Ha. These low-cutoff examples have finite basis, radial quadrature, and generator/interpolation errors. Broad accuracy and convergence studies, especially for semicore levels and heavier elements, remain outstanding.

A silicon CUDA warm restart agreed with the CPU result within approximately `0.16 meV` in eigenvalues and `8e-9 Ha` in total energy, at the SCF tolerances used for that check.

## Reconstruction convergence

For the Si state obtained at a 12 Ha smooth cutoff, reconstructed coefficients were exported at two independent AE cutoffs. Momentum calculated directly from those coefficients was compared with the analytic on-site PAW momentum correction, in the same Bloch gauge.

| AE export cutoff | Largest norm error | Largest momentum-entry difference, atomic units |
|---:|---:|---:|
| 60 Ha | 0.00440 | 0.0128 |
| 200 Ha | 0.000763 | 0.00157 |

No normalization correction was applied. The analytic metric differed from identity by less than `1e-13`, and analytic momentum matrices were Hermitian to approximately `1e-16`. The automated CPU/CUDA workflows also verify that both norm and momentum errors decrease when the reconstruction cutoff rises from 30 to 80 Ha.

The local inputs, QE outputs, and exported example states from this session are under `build/paw-validation/`. The reproducible source tests are in `jdftx/test/paw/`.
