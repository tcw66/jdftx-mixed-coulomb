# Silicon DFPT electron-phonon matrix elements

Run from this directory with the rebuilt GPU executable:

```bash
../../build-gpu/jdftx_gpu -i 01-scf.in -o scf.out
../../build-gpu/jdftx_gpu -i 02-dfpt.in -o dfpt.out
python3 ../../jdftx/scripts/electronPhononDFPT phonon. -o electron-phonon
```

Use `../../build/jdftx` for CPU runs. The example starts from a fresh
symmetry-reduced SCF checkpoint with 12 bands, including conduction states.
The existing four-band silicon checkpoints cannot supply these extra bands.
DFPT unfolds the checkpoint and computes screened derivatives on the full
4×4×4 electronic mesh and 2×2×2 q mesh, using the primitive-cell FFT grid.
The cutoff, meshes, Fermi width and band count are starting values to converge.

Electron-phonon export computes all q points and Cartesian displacements
explicitly. To parallelize over q, add `phonon-dfpt-q-groups 2` to `02-dfpt.in`
and launch with two or more MPI ranks. Offset states are reused on commensurate
meshes. Increase `elec-n-bands` in both inputs if the highest-band occupation
check fails, then rerun SCF.

Read a result with NumPy:

```python
import numpy as np
data = np.load('electron-phonon/q000001.npz')
g = data['g']  # complex matrix elements in meV: [k, phonon mode, final band, initial band]
probability = abs(g)**2
valid_modes = data['valid_modes']
```

Band energies, signed phonon energies, q, k, k+q, k weights, phonon eigenvectors
and raw Cartesian derivatives are also stored. Zero/imaginary modes and positive
frequencies below 0.01 meV have NaN couplings; use `valid_modes` when summing.
Degenerate band/mode subspaces have basis-dependent individual matrix elements.
Use `--q-index 1 --bands 0 8` to process selected q/bands, or `--units eV` for eV.

These are coarse-mesh Bloch matrix elements. Phonon dispersions can be
interpolated with `phononDFPT`; electron-phonon interpolation requires electronic
gauge alignment and is not implemented by this converter. The output is
distinct from the existing supercell/Wannier `phononHsub` format.

See [the DFPT tutorial](../../jdftx/doc/tutorials/phonon/DFPT.dox) for
normalization, file formats and supported systems.
