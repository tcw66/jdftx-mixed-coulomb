# Graphene DFPT phonons and Wannier electron–phonon matrices

This example uses slab Coulomb truncation with embedding, a symmetry-reduced
SCF checkpoint, a 12×12×1 electronic mesh and a 6×6×1 phonon q grid. The provided
`latMin.lattice` and `latMin.ionpos` supply the starting geometry. Set the
pseudopotential path in `common.in` for your installation.

From this directory, run SCF before DFPT:

```bash
../../build/jdftx -i totalE.in -o totalE.out
mpirun -n 4 ../../build/jdftx -i dfpt.in -o dfpt.out
```

For GPU runs, replace the executable with `../../build/jdftx_gpu`.
`dfpt.in` uses two MPI q groups, so launch with at least two ranks; set
`phonon-dfpt-q-groups 1` for a serial run. The nested solver and offset reuse
are enabled. Electron–phonon export retains q and Cartesian displacement
symmetry reductions and reconstructs equivalent response potentials. Disable
both DFPT symmetry switches to compare against explicit responses.

## Slab and restart settings

`common.in` contains:

```text
coulomb-interaction Slab 001
coulomb-truncation-embed 0 0 0
kpoint-folding 12 12 1
```

The atoms lie at z=0. Embedding centers use the active `coords-type` coordinate
system. The normal q-grid dimension is 1, while all three Cartesian displacement
directions, including out-of-plane phonons, are computed. Keep sufficient vacuum
for the density and Coulomb ion margin.

SCF and DFPT must use the same geometry, pseudopotentials, cutoff, band count,
smearing, Coulomb geometry, embedding center, ion margin and FFT grid. Include
`kpoint-folding` in both inputs through `common.in`; the restart command does
not inherit it from `totalE.in`. Regenerate SCF after changing these settings.
Graphene uses Fermi smearing and empty bands. Increase `elec-n-bands` and rerun
SCF if the highest-band occupation exceeds 1e-6. Converge the cutoff, meshes,
smearing width, band count and vacuum for your calculation.

## Native Wannier output

`phonon-dfpt-electron-phonon yes` is enabled. DFPT writes:

- `phonon.phononHsub`, with native mass-weighted, supercell-normalized complex
  matrices for the 36 commensurate electronic k points and every stored band.
- `phonon.phononKpts` and `phonon.phononBasis`, defining k order and Cartesian modes.
- `phonon.wfns`, `phonon.eigenvals` and `phonon.fillings`, with matching electronic
  states for the full 144-point mesh, even with `dump End None`.
- The usual force matrices and lattice-space phonon interpolation files.

For 10 bands and six Cartesian modes, `phononHsub` occupies 12,441,600 bytes.
Its ordering and normalization match the native supercell phonon executable.
MPI q groups share one electronic band gauge and write disjoint file blocks.

Prepare a Wannier input with the same geometry and electronic settings,
appropriate trial orbitals and band windows, and:

```text
include common.in
symmetries none
wannier phononSupercell 6 6 1
wannier-initial-state phonon.$VAR
wannier-dump-name wannier.$VAR
# Add wannier-center and band-window settings for the desired orbitals.
```

Run `../../build/wannier -i wannier.in -o wannier.out` (or `wannier_gpu`).
Use the DFPT prefix as shown, so the band rotations match the matrix-element
gauge. Wannier writes `wannier.mlwfHePh` and phonon cell maps/weights for
interpolation. No conversion from per-q files is needed. See the
[silicon example](../dfpt-silicon-electron-phonon/README.md) for a complete
Wannier input and the [DFPT guide](../../jdftx/doc/tutorials/phonon/DFPT.dox)
for normalization, phonon interpolation and supported features.

## Plot the phonon dispersion

After DFPT finishes, run from this directory:

```bash
python3 plot_dispersion.py
```

The script interpolates the native force matrices along Γ–M–K–Γ with distances
in Cartesian reciprocal space. It writes PNG/PDF plots, a q-path, a frequency
table and eigenvectors to `results_dispersion/`. The plot shows meV and cm⁻¹,
with in-plane and out-of-plane motion classified from the phonon eigenvectors.
Negative values denote imaginary frequencies. Python, NumPy and Matplotlib
are required. Use `--output` or `--points-per-segment` to customize output.
