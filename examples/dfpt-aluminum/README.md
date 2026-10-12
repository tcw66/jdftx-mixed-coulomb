# Metallic aluminum DFPT example

Run from this directory with the rebuilt CPU or GPU executable:

```bash
../../build/jdftx_gpu -i 01-scf.in -o scf.out
../../build/jdftx_gpu -i 02-dfpt.in -o dfpt.out
```

The SCF uses symmetry; DFPT unfolds its checkpoint onto the full electronic k
mesh. The calculation uses electronic smearing at fixed electron number and
includes the occupation response and the Gamma-point Fermi-level shift.
The force matrices are second derivatives of the smeared electronic free energy
at the specified width. Fermi and Gauss smearing are supported. MP1, Cold,
fixed chemical potential, spin polarization, ultrasoft pseudopotentials and
nonlinear core corrections are not supported by this phonon path.

The settings (30 Ha cutoff, 8×8×8 k mesh, 4×4×4 q mesh, Fermi width 0.01 Ha)
are starting values, not a convergence study. Converge the k mesh and smearing
width together. Increase `elec-n-bands` if the highest-band occupation exceeds
1e-6 at any source or offset k point. Keep geometry, bands and smearing identical
between the SCF and DFPT inputs; rerun SCF when changing them. Update the
pseudopotential path in `common.in` if using a different installation.

To distribute independent q points, add `phonon-dfpt-q-groups 2` to `02-dfpt.in`
and launch with at least two MPI ranks, for example
`mpirun -n 2 ../../build/jdftx_gpu -i 02-dfpt.in -o dfpt.out`.

Interpolate the resulting matrices with `../../jdftx/scripts/phononDFPT`, using
a q-path file in fractional reciprocal coordinates:

```bash
python3 ../../jdftx/scripts/phononDFPT phonon. qpath.dat -o dispersion.dat
```

See [the DFPT tutorial](../../jdftx/doc/tutorials/phonon/DFPT.dox) for scope and
output definitions. The implementation follows Sec. II.C.4 of
[Baroni et al., Rev. Mod. Phys. 73, 515 (2001)](https://doi.org/10.1103/RevModPhys.73.515).

To export screened electron-phonon matrix elements, add
`phonon-dfpt-electron-phonon yes` to `02-dfpt.in`. Symmetry-equivalent q points
and displacement directions reuse transformed response potentials; offset
reuse and q groups remain available. Disable both DFPT symmetry options to
compare against explicit responses.
This writes native `phononHsub`, `phononKpts` and matching electronic states.
Use `wannier-initial-state phonon.$VAR` and `wannier phononSupercell` with the
DFPT q-grid dimensions to output interpolatable `mlwfHePh` matrix elements.
The electronic mesh must be Gamma centered and contain the phonon grid.
