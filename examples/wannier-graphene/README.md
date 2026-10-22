# Graphene: MV versus TruncatedDC

Run the SCF calculation and both Wannier localizations, then calculate and
plot a shared DFT band reference:

```bash
cd /home/tyler/apps/jdftx_defect_coulomb/examples/wannier-graphene
./run.sh
```

Defaults: two MPI ranks, one per GPU, using the repository's `build-gpu`
executables. Set `NP=1` for a single GPU. CPU execution is available with
`JDFTX_BUILD=/absolute/path/to/build JDFTX_SUFFIX='' NP=2 ./run.sh`.
`OMP_NUM_THREADS` defaults to 2 and `OPENBLAS_NUM_THREADS` to 1.
Python requires NumPy and Matplotlib. `C_PSEUDO` can override the absolute
carbon pseudopotential path.

## Model

- Two C atoms in ideal free-standing graphene; lattice constant 4.651 bohr
  (2.461 angstrom), cell height 25 bohr (13.23 angstrom).
- **12x12x1 SCF mesh**, PBE, SG15 norm-conserving carbon potential,
  30-Hartree wavefunction cutoff, 16 DFT bands.
- Slab Coulomb truncation normal to z; nonmagnetic, Fermi smearing 0.002 Ha.
- Five Wannier functions: three bond-centered sigma trial functions and one
  atomic pz orbital on each carbon. The pi and pi* branches are retained
  together, including their crossing at K.
- Inner window EF-25 to EF+2 eV; outer window EF-25 to EF+25 eV. The runner
  converts these to absolute Hartree energies using the calculated SCF EF.
- Both measures start fresh from identical trial functions and use the same
  windows and minimization thresholds. Only the localization measure changes.
- Direct DFT reference on Gamma-M-K-Gamma, with 40 intervals per segment
  (121 points), evaluated at the same fixed SCF potential.

The cutoff, vacuum height, smearing and band count are demonstration choices;
converge them for quantitative material properties. The comparison snapshot
preserves all SCF data and inputs so later edits do not change either run.

## Outputs and reruns

The main outputs are under `results/measure-comparison/`:

- `comparison.png` / `.pdf`: band overlay and objective convergence.
- `summary.json`: iterations, times and nearest-band errors in the inner window.
- `cross-objectives.json`: each final set of orbitals evaluated under the other
  measure, without further optimization.
- `MV/results/` and `TruncatedDC/results/`: native Wannier Hamiltonians,
  centers, rotations, individual comparison plots, band tables and run logs.
- `REPORT.md`: results from the initial completed test.

A comparison directory is never overwritten automatically. To repeat with
modified windows or trial functions, move `results/measure-comparison/` aside,
edit `results/windows.in` and/or `02-wannier.in`, and run:

```bash
python3 compare_measures.py
```

That reuses SCF, runs fresh localizations and regenerates the DFT reference.
Use `--np 1` or `--np 2`, `--build`, and `--suffix` to select execution settings.
`run.sh` also repeats SCF and regenerates the energy windows. SCF and Wannier
states must be regenerated if the model or mesh changes.

Band errors use nearest eigenvalues, rather than assigned band indices.
Constraints apply on the Wannier mesh; interpolation between mesh points
has finite-mesh error. Absolute MV and TDC spreads have different definitions.
A single physically informed starting guess does not test general robustness.
