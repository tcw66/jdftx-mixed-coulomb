# Silicon SCF → DFPT phonons → interpolated dispersion

This example uses the new `phonon-dfpt` command in the **`jdftx` executable**.
It contains all inputs and scripts for a diamond-silicon primitive cell:

| File | Purpose |
| --- | --- |
| `system.in` | Silicon lattice, atoms and SG15 pseudopotential |
| `common.in` | Shared cutoff, electronic k mesh, PBE and occupied bands |
| `01-scf.in` | Ground-state SCF and restart files |
| `02-dfpt.in` | Restart, orbital polish and coarse phonon q mesh |
| `run.sh` / `run.py` | Configure inputs and run stages in order |
| `make_qpath.py` | Generate Γ–X–W–K–Γ–L–U–W–L–K \| U–X |
| `analyze.py` | Check the Fourier transform and acoustic sum rule; plot dispersion |
| `arbitrary-q.dat` | Example points for interpolation away from the coarse mesh |

The `.in` files in this directory are templates. The runner expands their
`${...}` placeholders and writes ordinary runnable JDFTx inputs into the
calculation directory. It preserves literal `$VAR` filename patterns.

## Run the complete calculation

From the repository root:

```bash
./examples/dfpt-silicon/run.sh
```

Defaults are a 30 Hartree wavefunction cutoff, a **6×6×6 electronic k mesh**, a
**4×4×4 phonon q mesh**, and a conventional cubic lattice parameter of 10.26 bohr.
The primitive cell has two atoms and four fully occupied scalar bands. There is
no smearing. The meshes are Gamma centered; electronic symmetries are disabled as required
by the current finite-q implementation.

Phonon q points are reduced automatically using compatible space-group
operations and time reversal. The electronic response k mesh remains unreduced. The
DFPT log reports how many representative q points are actually solved; all
coarse-grid matrices are reconstructed before interpolation. To compare with
an explicit calculation at every q, set `phonon-dfpt-symmetries no` in
`02-dfpt.in` and use a separate calculation directory.

To parallelize over q and k together, use the runner's `--q-groups` option:

```bash
./examples/dfpt-silicon/run.sh --launcher 'mpirun -n 4' --q-groups 2
```

This gives two q groups with two ranks per group. Each group initializes the
complete ground state and computes its assigned representative q points;
electronic k points are divided among ranks within the group. For a direct
JDFTx invocation, add `phonon-dfpt-q-groups 2` to `02-dfpt.in`. The default is
one group. For the two-GPU machine, `mpirun -n 2` with two q groups assigns one
rank per GPU and lets the GPUs solve different q points. More groups replicate
more ground-state storage and initialization work. Choose no more groups than
ranks or useful q representatives. Atomic displacements remain sequential per
group. The final phonon files and any requested state dumps are written once.

These are starting settings for convergence studies, not certified converged
silicon frequencies. The lattice parameter is a reference value, not the
calculated PBE equilibrium lattice. Adjust it or determine the equilibrium
lattice separately before a production phonon calculation. The workflow does
not relax the structure.

Displacement symmetry reduction and reuse of commensurate k±q occupied states
are enabled automatically. For silicon on a 2×2×2 phonon mesh, displacement
reduction needs four response solves instead of eighteen after q reduction.
To validate against explicit solves, add these commands to the DFPT input:

```text
phonon-dfpt-displacement-symmetries no
phonon-dfpt-reuse-offset-states no
```

To run the ground-state SCF on its irreducible k mesh, use `--reduced-scf` with
`run.sh`. This writes a separate symmetry-enabled `common-scf.in` and initializes
DFPT with `phonon-dfpt-initial-state 01-scf.in scf.$VAR`. For a manual run,
save `State` and `Fillings` from the symmetry-enabled SCF; replace `initial-state`
in the DFPT input with that command and retain `symmetries none` for DFPT.
The original SCF input specifies the checkpoint's geometry, cutoff, pseudopotentials,
bands and symmetry mapping, which must match the DFPT calculation. All paths
resolve from the calculation working directory. The full electronic response
mesh remains allocated, but the expensive initial SCF can use symmetry.

The runner defaults to this repository's `build/jdftx` and finds an installed
`SG15/Si_ONCV_PBE.upf` in local build libraries or `$JDFTX_PSEUDO_DIR`. Override
paths explicitly if needed:

```bash
./examples/dfpt-silicon/run.sh \
  --jdftx-bin /path/to/rebuilt/jdftx \
  --pseudo /path/to/SG15/Si_ONCV_PBE.upf \
  --out examples/dfpt-silicon/results-production
```

Use the rebuilt executable containing `phonon-dfpt`. No k±q wavefunction files
need to be supplied. Python needs NumPy; Matplotlib is needed for PNG/PDF plots.
`PYTHON=/path/to/python ./examples/dfpt-silicon/run.sh` selects a Python environment.
Use `--no-plot` if only numerical output is wanted.

## Check the setup or run a fast example

```bash
# Initialize both default inputs without solving SCF or DFPT:
./examples/dfpt-silicon/run.sh --stage check

# Run all stages on small meshes to verify the workflow:
./examples/dfpt-silicon/run.sh --quick
```

`--quick` uses 8 Hartree, a 2×2×2 electronic mesh and a 2×1×1 phonon mesh. Its
outputs go to `results-quick/`, separate from the default `results/`. The quick
results are **workflow validation only**; that anisotropic q mesh is unsuitable
for a converged dispersion throughout the Brillouin zone.

## Run individual stages

```bash
./examples/dfpt-silicon/run.sh --stage scf
./examples/dfpt-silicon/run.sh --stage dfpt
./examples/dfpt-silicon/run.sh --stage interpolate
```

Use the same `--out`, cutoff, k mesh, lattice and pseudopotential options for
these commands. For the small-mesh example, add `--quick` to each command.
The runner checks SCF convergence and the DFPT ground-state polish, and records
checksums of successful stage outputs. It rejects a restart if the ground-state
inputs or checkpoint files have changed. Changing only the q mesh lets you
reuse the same SCF state:

```bash
./examples/dfpt-silicon/run.sh --stage dfpt --qgrid 6 6 6
./examples/dfpt-silicon/run.sh --stage interpolate --qgrid 6 6 6
```

SCF produces `scf.wfns`, `scf.fillings` and `scf.eigenvals`. DFPT reads these,
polishes the ground state variationally, solves the offset occupied states and
responses, computes the q-space force matrices, and Fourier transforms them
to lattice space. `--stage all` (the default) reruns every stage; it does not
silently skip existing calculations. Logs and generated results in the chosen
directory are overwritten by the stages you rerun.

## MPI and sampling settings

```bash
./examples/dfpt-silicon/run.sh \
  --launcher 'mpirun -n 4' --threads 1 \
  --ecut 40 --kgrid 8 8 8 --qgrid 4 4 4 \
  --out examples/dfpt-silicon/results-e40-k8-q4
```

`--launcher 'srun -n 4'` also works inside a suitable Slurm allocation. Match
Slurm's CPUs per task to the desired thread count. The electronic k mesh and
phonon q mesh are independent; no electronic supercell is built. The current
driver solves every q point and displacement explicitly, so denser meshes can
be expensive. Increase cutoff, k sampling and q sampling independently to
assess convergence, using separate calculation directories for comparisons.

## Results and plotting

All outputs are inside the chosen calculation directory, normally `results/`:

| Output | Contents |
| --- | --- |
| `scf.out`, `dfpt.out` | Ground-state and response logs |
| `phonon.phononForceMatrixQ` | Complex coarse-q force matrices, Hartree/bohr² |
| `phonon.phononCellMap`, `phonon.phononCellWeights` | Lattice vectors and atom-pair weights |
| `phonon.phononOmegaSq`, `phonon.phononBasis` | Mass-weighted lattice-space matrices and mode definitions |
| `qpath.dat`, `qpath-distance.dat`, `qpath.json` | Fractional q points, Cartesian path distances and labels |
| `frequencies.dat` | q coordinates followed by six frequencies in cm⁻¹ |
| `phonons.npz` | q points, dynamical matrices, squared frequencies and eigenvectors |
| `dispersion.dat` | Path distance, q coordinates and frequencies |
| `dispersion.png`, `dispersion.pdf` | Plotted phonon dispersion |
| `checks.json` | Fourier round-trip error, acoustic sum rule and response residuals |
| `settings.json` | Paths, sampling, units and input signatures |

Negative frequencies denote imaginary modes; they are retained rather than
discarded. Tiny signed values at Γ can result from numerical diagonalization
of the three zero-frequency acoustic translations. Path coordinates use the
primitive reciprocal basis of `system.in`: for example, X is `(0.5,0,0.5)`
and L is `(0.5,0.5,0.5)`. The plot does not connect K to U across the path break.

Change path resolution without recomputing SCF or DFPT:

```bash
./examples/dfpt-silicon/run.sh --stage interpolate --points-per-segment 100
```

For arbitrary q points, use the interpolation utility directly. From the repo
root, after the default calculation:

```bash
python3 jdftx/scripts/phononDFPT \
  examples/dfpt-silicon/results/phonon. \
  examples/dfpt-silicon/arbitrary-q.dat \
  -o examples/dfpt-silicon/results/arbitrary-frequencies.dat \
  --matrices examples/dfpt-silicon/results/arbitrary-phonons.npz
```

Replace the q file with any text file containing three fractional reciprocal
coordinates per row. Interpolation Fourier-transforms the dynamical matrix
and then diagonalizes it; it does not interpolate individual eigenvalues.

## Scope

The current primitive-cell DFPT path supports periodic, nonmagnetic
LDA/GGA calculations with norm-conserving pseudopotentials without nonlinear
core corrections. Keep all bands fully occupied in this silicon example.
Metals and gapless systems can instead use Fermi or Gauss smearing at fixed
electron number with enough empty bands; see the
[aluminum example](../dfpt-aluminum/README.md).
Ultrasoft potentials,
nonlinear core corrections and polar LO–TO interpolation are not supported.
Silicon is nonpolar, so no Born-charge correction is needed for this example.
Further implementation details are in
[the DFPT guide](../../jdftx/doc/tutorials/phonon/DFPT.dox).
