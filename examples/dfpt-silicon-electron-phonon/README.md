# Silicon DFPT electron-phonon matrix elements

Run from this directory:

```bash
../../build/jdftx -i 01-scf.in -o scf.out
../../build/jdftx -i 02-dfpt.in -o dfpt.out
../../build/wannier -i 03-wannier.in -o wannier.out
```

Use `../../build-gpu/jdftx_gpu` and `wannier_gpu` for GPU runs.
The example creates a fresh symmetry-reduced SCF checkpoint with 12 bands,
including conduction states. Existing four-band silicon checkpoints cannot
supply these extra bands. DFPT unfolds the checkpoint and integrates the
response on the full 4×4×4 electronic mesh with a 2×2×2 phonon q grid.
Converge the cutoff, meshes, smearing width and band count for your calculation.

`phonon-dfpt-electron-phonon yes` writes the native `phononHsub` format,
`phononKpts`, `phononBasis`, and matching `wfns`, `eigenvals` and `fillings`.
The matrix file contains all 12 bands on the eight electronic k points
commensurate with the phonon grid, using the native mass and supercell factors.
The Wannier example constructs four valence bond orbitals and outputs
`wannier.mlwfHePh` for interpolation. Adjust trial orbitals and band windows
to Wannierize conduction states or a different material.

Export computes every q and Cartesian displacement explicitly. To distribute
q work, add `phonon-dfpt-q-groups 2` and launch with at least two MPI ranks.
All groups use a common band gauge. If the highest-band occupation check fails,
increase `elec-n-bands` in both inputs and rerun SCF.

Use the DFPT output prefix, `phonon.$VAR`, for `wannier-initial-state` so Wannier
reads precisely the electronic states used to construct the matrix elements.
Set `wannier phononSupercell` to the DFPT q-grid dimensions, and use the same
full electronic mesh with `symmetries none`.

See [the DFPT tutorial](../../jdftx/doc/tutorials/phonon/DFPT.dox) for the file
layout, normalization and supported systems.
