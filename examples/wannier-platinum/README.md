# Bulk fcc Pt: MV versus TruncatedDC

Run the SCF calculation, matched Wannier localizations and shared DFT band
comparison:

```bash
cd /home/tyler/apps/jdftx_defect_coulomb/examples/wannier-platinum
./run.sh
```

Defaults: two MPI ranks, one per GPU, using this repository's `build-gpu`.
Use `NP=1` for a single GPU, or override `JDFTX_BUILD` and `JDFTX_SUFFIX`.
For CPU, set `JDFTX_SUFFIX=''` and select a build containing `jdftx` and
`wannier`. `OMP_NUM_THREADS` defaults to 2 and `OPENBLAS_NUM_THREADS` to 1.
Python requires NumPy and Matplotlib. `PT_PSEUDO` may specify an absolute
pseudopotential path; changing the potential requires rechecking windows.

## Model

- Nonmagnetic primitive fcc Pt, conventional lattice constant 7.41 bohr
  (3.921 angstrom); **8x8x8 SCF mesh**.
- PBE, SG15 Pt norm-conserving potential, 40-Hartree wavefunction cutoff,
  40 DFT bands, Fermi smearing 0.01 Ha.
- The installed Pt potential is **scalar relativistic, without spin-orbit
  coupling**. This is a localization-method comparison; a quantitative Pt
  band calculation should separately assess SOC and numerical convergence.
- The potential has 18 explicit electrons. Nine atom-derived valence s/p/d
  trial functions are generated with `ignoreSemiCore yes`; deep semicore
  orbitals are excluded from the Wannier model.
- Inner window EF-12 to EF+3 eV; outer window EF-20 to EF+40 eV, converted
  to absolute Hartree energies using the SCF Fermi energy. The wider outer
  window includes at least nine bands everywhere and supports the p trials.
- MV (`FiniteDifference`) and TruncatedDC both start fresh from identical
  projections, with identical windows and minimization thresholds.
- Shared fixed-potential DFT reference on Gamma-X-W-K-Gamma-L-U-W-L-K,
  20 intervals per segment (181 points).

The cutoff, lattice constant, smearing and band count are test settings;
converge them before making quantitative material predictions.

## Outputs and reruns

All comparison outputs are under `results/measure-comparison/`:

- `comparison.png` / `.pdf`: overlaid bands and objective convergence.
- `summary.json`: iterations, wall times and nearest-band errors.
- `cross-objectives.json`: each final gauge evaluated under the other spread
  measure, without further optimization.
- `MV/results/` and `TruncatedDC/results/`: native Hamiltonians, centers,
  rotations, individual plots, eigenvalue tables and logs.
- `REPORT.md`: results from the completed test.

The driver snapshots SCF files and model inputs and never overwrites an
existing comparison directory. To repeat with altered windows, move
`results/measure-comparison/` aside, edit `results/windows.in`, then run:

```bash
python3 compare_measures.py
```

Use `--np`, `--build`, and `--suffix` to select execution settings. Running
`run.sh` also regenerates SCF and the energy windows. If you change the model
or SCF mesh, regenerate the corresponding states before comparing.

Band errors refer to the nearest interpolated eigenvalue for each DFT
inner-window state, not an assigned band index. Constraints apply at the
Wannier mesh, with interpolation error between mesh points. Absolute MV and
TDC spreads have different definitions; use the cross-objective evaluations
for comparisons under one definition. One atomic initial guess does not
measure general robustness across different starting guesses.
