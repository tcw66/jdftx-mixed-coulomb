# Bulk aluminum with TruncatedDC Wannier localization

Run both steps from this directory:

```bash
./run.sh
```

The runner uses this repository's `build/jdftx`, `build/wannier`, and SG15
`Al_ONCV_PBE.upf`. It runs the SCF calculation, reads its Fermi energy,
writes `results/windows.in`, and localizes five Wannier functions using
`localizationMeasure TruncatedDC`. Python 3 is required for window generation.
Results are written under `results/`; rerunning replaces them and starts the
Wannier optimization from the trial orbitals again.

Optional MPI execution:

```bash
NP=2 ./run.sh
```

Override the build directory with an absolute path if needed:

```bash
JDFTX_BUILD=/absolute/path/to/build ./run.sh
```

Use the CUDA executables with:

```bash
JDFTX_BUILD=/home/tyler/apps/jdftx_defect_coulomb/build-gpu JDFTX_SUFFIX=_gpu ./run.sh
```

`OMP_NUM_THREADS` defaults to 2 per process; `OPENBLAS_NUM_THREADS` defaults
to 1. `AL_PSEUDO` can select an absolute pseudopotential path, but changing
the pseudopotential requires checking the energy windows and band count.

## Model and windows

The primitive fcc cell contains one Al atom, with conventional lattice constant
7.65 bohr (about 4.05 angstrom). The demonstration uses PBE, a 20-Hartree
wavefunction cutoff, a 4x4x4 k mesh, 16 DFT bands and Fermi smearing of
0.01 Hartree. These are quick demonstration settings; converge the cutoff,
k mesh and smearing for quantitative work.

The default SG15 Al pseudopotential has 11 valence electrons, including eight
semicore electrons. The five Wannier functions describe the metallic valence
manifold, excluding the four deep semicore bands. The generated windows are:

| Window | Relative to SCF Fermi energy |
| --- | --- |
| Inner: bands retained exactly | -12 to +5 eV |
| Outer: bands available for optimization | -12 to +25 eV |

The input uses **absolute Hartree energies**, converted automatically by
`run.sh`. Five Gaussian trial centers follow JDFTx's entangled-Al tutorial.
They are unpinned, allowing joint optimization of the subspace and gauge.
The inner window must contain at most five bands at each k point, and the
outer window must contain at least five. Recheck this when changing settings.

## Inspecting results

```bash
rg 'SCF: Converged' results/scf.out
rg 'WannierMinimize: Converged|Omega:' results/wannier.out | tail -5
cat results/al.mlwfCenters
```

`results/wannier.out` reports the optimized centers and individual TDC spreads.
`results/al.mlwfH` contains the Wannier Hamiltonian, and `results/al.mlwfU`
contains the rotations. The real-space orbitals are not exported by default
(`saveWfns no`). To change Wannier settings alone, edit `results/windows.in`
and rerun:

```bash
../../build/wannier -i 02-wannier.in -o results/wannier.out
```

Running `run.sh` again regenerates the windows. The total TDC spread in a
test run was approximately 22.1979 bohr^2, down from 35.0877 bohr^2, with
both SCF and Wannier minimization converged. This value is a smoke-test
reference for these coarse settings, not a converged material property.
TDC spreads differ from MV spreads, and the printed centers are phase-based
estimates. See [the implementation notes](../../jdftx/test/wannier/README.md).

## Compare DFT and Wannier band structures

After SCF and Wannierization, run:

```bash
./run_bands.sh
```

For the GPU build:

```bash
JDFTX_BUILD=/home/tyler/apps/jdftx_defect_coulomb/build-gpu \
JDFTX_SUFFIX=_gpu ./run_bands.sh
```

This reuses the existing `results/scf.wfns` and fillings to export the fixed
SCF potential, then calculates DFT bands on Γ–X–W–K–Γ–L–U–W–L–K. It does
not rerun SCF or change the Wannier localization. Both methods use identical
primitive reciprocal coordinates and the SCF Fermi energy as energy zero.
`model.in` contains the common material settings; `common.in` adds the SCF
mesh. Thus the band calculation uses the same lattice, potential, cutoff and
DFT band count, but no mesh folding or symmetry reduction of the band path.

The comparison reads the actual number of Wannier functions from
`al.mlwfCenters`, so it supports either explicit Gaussian centers or your
`addAtomicOrbitals yes` configuration. It expands the native real-space
Hamiltonian using `mlwfCellMap` and the pair-dependent `mlwfCellWeights`.
Keep the SCF, Wannier files, windows, and mesh settings from the same run.
After changing `model.in` or `common.in`, regenerate the corresponding SCF
and Wannier results before comparing.

Outputs:

- `results/band-comparison.png` and `.pdf`: gray DFT bands, dashed red Wannier
  bands, Fermi level, and shaded inner window.
- `results/dft-bands.dat` and `results/wannier-bands.dat`: path distance in
  inverse bohr followed by band energies in eV relative to the SCF Fermi level.
- `results/band-errors.json`: RMS and maximum distance from inner-window DFT
  eigenvalues to the nearest interpolated eigenvalue. This is a diagnostic,
  not a band-index assignment or a guarantee against missing/extra bands.
- `results/bandstruct.out`: fixed-potential DFT calculation log.

The default path has 20 intervals per segment (181 points). Use
`BAND_STEPS=40 ./run_bands.sh` for a denser path, or `NP=2` for MPI. Python
requires NumPy and Matplotlib. To replot without recalculating DFT:

```bash
python3 compare_bands.py
```

Inner-window constraints hold at the original Wannier mesh points; interpolation
between those points still has finite-mesh error. Outside the inner window,
the selected Wannier subspace need not reproduce all DFT bands. The plot
shows -13 to +15 eV relative to the Fermi level, excluding deep semicore bands.

To regenerate only the Wannier functions and interpolated bands with your
current `results/windows.in` and centers (preserving SCF and DFT band-path
results), use:

```bash
JDFTX_BUILD=/home/tyler/apps/jdftx_defect_coulomb/build-gpu \
JDFTX_SUFFIX=_gpu ./run_wannier.sh
```

The same script works with the CPU build when the environment overrides are
omitted. If the DFT band-path files exist, it also refreshes the comparison
plot and error table. Unlike `run.sh`, it does not regenerate the windows.

## Matched MV versus TruncatedDC comparison

```bash
python3 compare_measures.py
```

This snapshots the current SCF orbitals, fillings and eigenvalues and runs
fresh localizations with identical atomic/trial orbitals, windows, mesh and
convergence settings. Only `localizationMeasure` changes: MV uses
`FiniteDifference`. The default is two MPI ranks with the GPU build; use
`--np 1` for a single GPU, or `--build /path/to/build --suffix ''` for CPU.
Avoid concurrent GPU jobs when benchmarking runtime.

All files go under `results/measure-comparison/`, preserving the main example
results. The script refuses to overwrite an existing comparison directory;
move it aside to repeat. Each measure gets a separate log, interpolated band
table, error report and plot. A new direct DFT path calculation from the
snapshot supplies a shared reference. `summary.json` records iterations,
wall time, convergence status and nearest-band interpolation errors.
`comparison.png`/`.pdf` overlay the bands and show each measure's objective
relative to its own final value. Absolute MV and TDC spreads have different
definitions and should not be interpreted as the same observable.
`cross-objectives.json` evaluates each final gauge with the other measure,
without further optimization, for a like-for-like assessment under each
objective. A single starting point does not establish general robustness.
