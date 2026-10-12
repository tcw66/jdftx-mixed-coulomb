# Silicon Koopmans calculations

The inputs use a spin-unpolarized PBE ground state (`spintype no-spin`), an
8×8×8 mesh, 60 stored bands and a 30 Ha cutoff. KI treats each orbital source
as one electron and includes charge and magnetic response about that
nonmagnetic ground state (`wannier-koopmans-spin yes`).

KI output is concise by default: each independent response reports its q
group, symmetry q-star number, actual q-point index/vector, orbital and
charge/magnetic sector before the solve starts. q vectors use fractional
reciprocal-lattice coordinates; star numbers count the selected screening
mesh's symmetry classes. Equivalent pairs that reuse a result stay quiet.
Final residuals, screening summaries and warnings/errors are retained.
To restore k-point and
basis listings, symmetry-reuse messages and every response iteration, set
`wannier-koopmans-verbose yes` in `ki-response.in`. The flag controls both
terminal and native log output and takes effect on the next native launch.

Run from this directory:

```sh
./run.sh --dry-run
./run.sh
```

The default run reuses the supplied matching SCF checkpoint and numerical
conduction seeds, calculates separate valence and conduction KI Hamiltonians,
calculates direct PBE bands on the same path, and writes the comparison plots
to `results/`. The plotter recovers the original projected PBE Hamiltonians
from the KI runs; the direct path calculation remains available as a separate
cross-check. All plotted bands use the same energy zero.

After both valence and conduction runs finish, regenerate the comparison
without running another electronic calculation:

```sh
python3 plot_separate_bands.py
```

The script reads `results/valence.*` and `results/conduction.*`, preserves
complex hoppings, and interpolates both KI and original PBE on the path in
`paths/`. Original PBE is recovered as `mlwfHki - mlwfKI`, rather than using
the legacy real-only `mlwfH` dump. These are PBE bands projected into the
chosen valence and conduction Wannier spaces, so their accuracy depends on
localization and disentanglement.

It reports the sampled path gap and an interpolated Brillouin-zone gap
estimated from a 16×16×16 mesh plus the path, followed by local refinement
from multiple starting points. The latter can find band edges away from the
plotted path. Increase the search mesh to check convergence, for example:

```sh
python3 plot_separate_bands.py --gap-mesh 24 24 24
```

`--no-refine` reports mesh/path extrema only. `--fold NX NY NZ` selects the
electronic mesh (otherwise read from `common.in`). `--valence PREFIX`,
`--conduction PREFIX`, and `--output-dir DIRECTORY` select other completed
runs. Inputs and checkpoints are read only. Missing final KI files or an
unfinished native log produce an explanatory error.

Outputs are `results/koopmans-bands-comparison.{png,pdf,dat,json}`. The plot
labels KI and PBE gaps and uses the interpolated PBE VBM as a common energy
zero. The JSON includes absolute VBM/CBM energies, fractional k coordinates,
path and BZ-search gaps, refinement status, native-band agreement, curvature
diagnostics and input hashes. Dependencies are NumPy, SciPy and Matplotlib.

Use `./run.sh --from-scf` to start again from SCF; the launcher archives the
current state, outputs and logs first. Use `./run_wannier.sh` for only the
separate KI stages, or `./run_wannier.sh --dry-run` to check their inputs.
`--cpu` selects the CPU executables. Dry runs redirect native dump prefixes
to temporary storage, protecting the saved checkpoints. Earlier native logs
are moved to `archive/logs/` before each stage runs.

The launchers use the rebuilt `../../build` executables (`*_gpu` by default), two MPI ranks,
four CPU threads per rank, and one OpenBLAS thread. Override these with
`JDFTX_BINARY`, `WANNIER_BINARY`, `MPIEXEC`, `MPI_RANKS` and `WANNIER_THREADS`.
Two q groups require at least two ranks. The pseudopotential search uses the
native build library or `JDFTX_PSEUDO_DIR` (the launcher defaults to
`../../build/pseudopotentials`).

Wannier restarts now record a content fingerprint in `*.mlwfState`. Changing
the saved SCF wavefunctions invalidates both rotations and overlaps, even if
the model and mesh are unchanged. The native code then starts from the trial
orbitals and recomputes overlaps. A frozen `nIterations 0` calculation rejects
a changed SCF state rather than silently using stale rotations. For legacy
checkpoints, a newer SCF wavefunction file than either `mlwfU` or `mlwfM0`
also invalidates the restart. Numerical conduction seeds remain physical
orbital guesses; their rotations and overlaps receive the same restart check.
The launcher also detects direct native jobs running in this directory.

Full 8×8×8 screening remains the default. To compare a coarser screening
quadrature, uncomment the 4×4×4 command in `ki-screening-mesh.in`. This reduces
only the q mesh used in the bare/relaxed curvature ratio for alpha. The
occupied-state k integration, orbital sources, bare Hamiltonian and output
bands retain the 8×8×8 mesh. Each coarse dimension must divide the corresponding
full-mesh dimension. Converge alpha and the band gap before adopting this
approximation. `*.koopmansResponse` records unweighted contributions for every
q point, including a flag for whether its response was calculated; a zero
response with flag 0 is an uncomputed value.

Compare coupled MINRES, nested CG and nested MINRES at identical localized
rotations with an isolated benchmark:

```sh
python3 benchmark_valence.py --screening-mesh 2 2 2
```

This still uses the full saved k mesh, but samples fewer q points for timing.
Omit `--screening-mesh` to benchmark the full production problem. The benchmark
relocalizes when necessary, checks agreement between solver results, and
stores its inputs, checkpoints, logs and `timings.json` in a new archive
directory. `--cpu`, `--ranks` and `--threads` select benchmark resources.
Coupled MINRES remains the default because the tested nested solvers have
been slower for this valence problem.

For a quick symmetry audit without response solves, set
`JDFTX_KI_SYMMETRY_AUDIT=1` when running Wannier with an isolated dump prefix.
The audit checks actual source densities and curvatures and reports the
number of independent pairs. It produces no KI spectra. The 8×8×8 audit
recovered 65 independent pairs after relocalizing against the current SCF;
stale rotations had required 1,040 independent pairs.

The files and directories are:

- `model.in`: geometry, pseudopotential, functional, spin treatment and cutoff.
- `common.in`: uniform SCF/KI mesh and SCF convergence settings.
- `ki-response.in`: common KI screening and solver settings.
- `ki-screening-mesh.in`: optional coarse screening q mesh; disabled by default.
- `01-scf.in`: ground-state calculation.
- `02-valence.in`: four occupied Wannier functions and KI correction.
- `02a-conduction-seed.in`: export six numerical empty-orbital seeds when needed.
- `03-conduction.in`: six empty Wannier functions and KI correction.
- `bandstruct.in`: direct PBE path calculation, with folding explicitly 1×1×1.
- `paths/`: matching k points, distances and symmetry labels.
- `state/`: SCF and numerical-orbital checkpoints, with an input fingerprint.
- `results/`: Wannier rotation checkpoints, new KI/PBE data and plots.
- `logs/`: native calculation and dry-run logs.
- `archive/20261011-before-cleanup/`: all earlier runs, plots, audits and scripts.

`results/` initially contains only the saved rotation/overlap checkpoints;
the earlier charge-only spectra and plots are in the archive. They are not
presented as new spin-response results. The archive includes the original
implementation notes, the factor audit, the spin-response validation reports,
and a cleanup manifest. Internal diagnostic restart symlinks were relocated
with their source data. Archived run inputs retain their historical paths.

Changing the model or mesh requires a fresh SCF run with `--from-scf`.
The conduction-window lower bound, 0.239616 Ha, is inside the current 8×8×8
SCF gap; recheck it after changing the ground-state model or mesh.
Conduction localization and coarse-mesh interpolation still need convergence
checks before interpreting a KI band gap. In particular, the archived 2×2×2
spin diagnostics had unconverged conduction orbitals and negative curvatures.
