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

The selected CTest checks passed, along with the independent CUDA workflow. The checks include:

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

## MPI reconstruction and CUDA quadrature follow-up

An MPI export failure was traced to collective radial transforms inside the loop over rank-local k points. Unequal ownership caused incompatible collective calls. Reconstruction now caches transforms computed locally, once per partial-wave state and required reciprocal range. The original four-rank export timed out; the corrected CPU and CUDA exports completed. `pawExportMPI` compares serial, two-rank, and four-rank exports of three irreducible k points, covering both unequal work and an idle rank. Coefficients, reciprocal vectors, eigenvalues, and all five matrices agree within the regression tolerances (2e-11 for coefficients and 2e-10 for matrix entries).

All 12 PAW CTest checks passed against the rebuilt standard binaries. The user's two-rank silicon run also completed reconstruction of all 29 irreducible k points and eight bands at a 200 Ha AE cutoff. All coefficient files were finite and had the expected sizes. The largest analytic overlap error was 1.79e-13, matrix Hermiticity error 8.84e-16, and finite-cutoff reconstructed norm error 0.00173.

The on-site angular PBE calculation now runs on CUDA, while the CPU path batches radial shells. For the Si dataset with the same density matrix, four CPU threads, and three calls including workspace initialization:

| Backend | Original seconds per atom/call | Updated seconds per atom/call |
|---|---:|---:|
| CPU | 0.491 | 0.232 |
| RTX 4090 CUDA | 0.484 | 0.00993 |

The updated CPU result was identical to the original energy and gradient. The CUDA energy differed by 1.14e-13 Ha and the largest gradient-entry difference was 2.85e-14 Ha. These are timings of the on-site functional, not whole-job speedups. GPU finite-difference checks passed for H, Si, Zr, and Ce (including f channels), along with periodic Hamiltonian checks and the complete SCF/export/restart/Wannier workflow. Local diagnostic outputs are in `build/paw-performance/`.

## Independent AE dumps and native wavefunction layout

`JDFTx-PAW-2` replaces the q-indexed coefficient files with one `paw.wfns` in native ColumnBundle layout, together with a native text-block `paw.Gvectors` and per-state sizes/byte offsets in the JSON. The writer streams one reconstructed bundle at a time. `AERealSpaceWfns` is a separate selection. `AEOverlap`, `AEMomenta`, and `AEBandEigs` select separate operator files; the Hamiltonian is represented only by its eigenvalues. `AEMatrices` is a shorthand for these three operator dumps.

The CPU and CUDA workflows verify every selection independently, including absence of unrequested files, real-space agreement with an independent NumPy inverse FFT, Parseval normalization, and identical `AEBandEigs`/`BandEigs` values. Matrix-only exports work without initializing a reconstruction grid. Serial/two-rank/four-rank tests compare the combined coefficient file, G-vector blocks, all real-space band files, eigenvalues, overlaps, and momenta, including an idle head rank. Both MPI-I/O and `MPISafeWrite=ON` paths are exercised.

Against the saved version-1 Si example at a 60 Ha reconstruction cutoff, the maximum coefficient difference was 2.57e-14, overlap difference 6.22e-15, momentum difference 1.25e-15, and eigenvalue difference 1.56e-15 Ha. The reader retains version-1 support for existing exports. The native-layout regression additionally reads and rewrites the new coefficient file with `ElecInfo::read`/`ElecInfo::write` and requires byte-for-byte equality.

## AE Wannier orbitals and operator outputs

The standard CPU/CUDA `jdftx`, `wannier`, and `phonon` executables were rebuilt after adding independent `wannier saveAEWfns` and `saveAEWfnsRealSpace` selections. All 13 PAW CTest checks passed, including `pawWannierAE`; the separate CUDA Wannier regression also passed with two- and four-rank exports. The checks cover:

- Independent NumPy assembly of the AE Wannier coefficient bundle from exported reduced AE Bloch coefficients and the final `mlwfU` matrices, including nonsymmorphic translations and antiunitary transformations.
- Inverse FFT agreement with every complex real-space Wannier file, preservation of the shifted-supercell Bloch phase, and finite-cutoff normalization without correction.
- Serial/two-rank/four-rank output agreement, including an idle head rank; independent real/reciprocal selectors; fixed-gauge exports without relocalization; and TruncatedDC output.
- Independent matrix checks for the rotated/Fourier-transformed Hamiltonian and canonical `i*p`, finite-difference `R*i*p` using explicit AE Fourier coefficients, and a constant scalar-weight integral. Differences for H/P/RP/W were below 1e-14 in the tested Si example.

For four Wannier orbitals from the 8 Ha Si SCF example on a `2 x 2 x 2` mesh:

| AE reconstruction cutoff | Representative Wannier norm | Largest difference between analytic PAW R and finite-cutoff Fourier R, bohr |
|---:|---:|---:|
| 30 Ha | 0.993179 | 0.00481 |
| 80 Ha | 0.998179 | 0.00186 |

The analytic H/P/R outputs were invariant when only the reconstruction cutoff changed. RP and scalar-weight integrals use reconstructed AE orbitals and require cutoff convergence; R/RP also retain the existing finite-difference mesh and short-range interpolation conventions. These checks establish implementation consistency, rather than physical convergence of this coarse example.

A separate fixed-density graphene band run exposed an input-geometry mismatch: SCF symmetry moved rounded carbon positions by approximately 1e-6 bohr, while the symmetry-disabled band input retained the unsymmetrized positions. Matching the exact saved SCF coordinates allowed all 164 path points and 20 bands to converge, with identical BandEigs/AEBandEigs values. The geometry check remains strict; its error now reports the saved/current coordinate and recommends using the exact SCF geometry.

The shipped five-point Si band example now disables k-point reordering/folding and assigns weights of 0.2 so they sum to one. Its fixed-density calculation and AE exports also passed a direct GPU smoke check using a saved Si state.
