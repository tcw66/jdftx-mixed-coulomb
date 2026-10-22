# Coarse electronic-grid KI screening in JDFTx

These scripts run the existing `../koopmans/run_coarse_screening.py` workflow
on the completed bulk-Si calculation in `/home/tyler/koopmans`. They use its
fine SCF density, model, occupied/empty localization settings, and checkpoints.
They do not recompute or overwrite that source calculation.

1. Retain the fine-grid SCF density and solve coarse-grid NSCF states.
2. Independently localize occupied and empty Wannier orbitals on the coarse grid.
3. Solve single-spin KI(2)/DFPT screening on the full k/q meshes of that coarse model.
4. Match coarse orbitals to fine orbitals using periodic density overlaps.
5. Import the matched alphas into the fine-grid KI Hamiltonians without new screening solves.
6. Plot the fine-grid bands and verify the imported alphas and recorded product hashes.

This implements separate electronic-grid screening, corresponding to the
official package's DFPT coarse workflow. It is not fine-k/coarse-q subsampling,
and it is not full finite-occupation KI/Delta-SCF.

## Files and settings

- `config.sh`: source/output paths, coarse grid, CPU/GPU, MPI ranks, threads,
  optional dielectric constant and orbital map.
- `workflow.sh`: prepare, execute individual stages, plot, or check results.
- `run_all.sh`: complete execution followed by result checks.
- `check_results.py`: alpha-transfer validation and per-orbital CSV/JSON summary.
- `../koopmans/run_coarse_screening.py`: core implementation, orbital assignment,
  localization guards, native input generation, and verified stage resume.
- `../koopmans/plot_separate_bands.py` and `interpolate.py`: band plotting/interpolation.

Default settings: GPU, one MPI rank, four threads, 4x4x4 coarse grid, existing
8x8x8 fine Hamiltonian, output under
`/home/tyler/koopmans/archive/coarse-screening-4x4x4-20261014`.
Environment settings override `config.sh`. CPU/GPU binaries default to this
repository's `build/`; `JDFTX_BINARY` and `WANNIER_BINARY` can override them.
For multiple ranks, the core runner uses `$MPIEXEC` (default `/usr/local/bin/mpiexec`).
Assign ranks to GPUs according to your cluster configuration; the launcher does
not allocate GPUs. NumPy and SciPy are required; plotting also needs Matplotlib.

## Prepare and run

```bash
cd /home/tyler/apps/jdftx_defect_coulomb/examples/coarse-screening-silicon
./workflow.sh prepare
./run_all.sh
```

Preparation copies the fine density/checkpoints and writes native `.in` files
into the isolated output directory. It executes no electronic calculations.
Review `inputs/` and `manifest.json` before launching a long calculation.

CPU example with a distinct output directory:

```bash
BACKEND=cpu RANKS=2 THREADS=4 \
OUTPUT_DIR=/home/tyler/koopmans/archive/coarse-screening-cpu-4 \
./run_all.sh
```

Run stages separately, using the same settings for every invocation:

```bash
./workflow.sh nscf
./workflow.sh wannier
./workflow.sh screening
./workflow.sh hamiltonian
./workflow.sh plot
./workflow.sh check
```

Each execution checks preceding stages and reuses matching completed products.
`plot` ensures the Hamiltonian stages exist before plotting; it can perform
missing calculation stages. Changed source state, model, backend, grid or
epsilon requires a new output directory. Do not run two processes into one
directory. Locks and recorded hashes protect stage integrity.

## Dielectric constant and convergence

By default the source KI epsilon is inherited; the current Si source uses zero,
which disables image correction. No dielectric constant is guessed or fitted.
To supply a separately calculated fixed-ion electronic dielectric constant:

```bash
EPSILON_INFINITY=YOUR_COMPUTED_VALUE \
OUTPUT_DIR=/home/tyler/koopmans/archive/coarse-screening-with-epsilon \
./run_all.sh
```

Use the same setting for both stages and all subsequent invocations. Positive
native epsilon must be >=1. Each model uses its own mesh-dependent Ewald term.
Consult `../dielectric-silicon/README.md` for the dielectric DFPT calculation.

Coarse and fine electronic grids need not divide one another. Increase the
coarse grid in separate output directories to test convergence. A 2x2x2 Si
empty-state localization previously failed the gradient checks; the runner
rejects such results rather than importing their alphas. A 4x4x4 result is not
automatically converged. Orbitals must pass the minimum density-overlap and
ambiguity checks; alphas are never clipped or adjusted to target a gap.

## Outputs

- `inputs/`: generated native JDFTx and Wannier inputs.
- `logs/`: native and launcher logs for every stage.
- `reference/scf.n`: retained fine-grid density.
- `coarse/{valence,conduction}.koopmansScreening`: coarse curvatures and alphas.
- `fine/{valence,conduction}.alpha` and `.alpha.mapping.json`: matched alphas and mapping evidence.
- `fine/*.mlwfKI`, `*.mlwfHki`, `*.koopmansBands`: corrected fine-grid matrices/bands.
- `fine/koopmans-bands-comparison.{png,pdf,dat,json}`: interpolated bands and gap.
- `coarse-screening-summary.{csv,json}`: checked per-orbital alpha-transfer summary.
- `manifest.json`, `stages.json`: source/model settings and stage/product hashes.

The new wrappers do not require copies of the core scripts into the production
directory. Keep the repository's example directories together. The existing
core implementation also has a native H2 regression in
`jdftx/test/koopmans/checkCoarseWorkflow.py`.

## Validation of this package

On 14 October 2026, the shell scripts passed syntax checks and the Python checker
compiled successfully. Preparation against the actual `/home/tyler/koopmans`
source wrote the complete 4x4x4 coarse / 8x8x8 fine input set in an isolated
temporary directory. The checker passed on the existing completed production-4
Si archive, with summaries written separately. The native CPU
`koopmansCoarseWorkflow` regression passed in 18.17 seconds. No new production
screening calculation or GPU regression was launched for this packaging task.
