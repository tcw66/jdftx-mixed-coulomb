# Native PAW-XML support

This implementation adds frozen-core, unpolarized scalar PAW with the internal PBE functional. It reads the supplied JTH PAW-XML atomic datasets directly; UPF conversion is unnecessary. Both CPU and CUDA executables use the same on-site functional and quadrature. CUDA builds evaluate the angular density, PBE, and potential projections on GPU with cached workspaces; CPU builds batch radial shells to reduce thread-launch overhead. Radial electrostatics and density-matrix assembly run on CPU.

This is an initial implementation, with limited numerical validation, rather than a fully validated replacement for established PAW codes. Spin polarization, spin-orbit coupling, ionic forces, stress, relaxation, molecular dynamics, DFPT/phonons, DFT+U, hybrid/meta-GGA functionals, and solvation are rejected. Core orbital eigenstates in the separate `corewf` archive are not calculated. The reconstructed states are the **valence Kohn–Sham states in the frozen-core PAW approximation**.

## Build and run

The new build dependency is the libxml2 development library. Download the PBE atomic XML archive from the [JTH dataset page](https://www.abinit.org/atomic_data/paw/) and place it in the local `pseudopotentials/` directory. Datasets and local calculation outputs are supplied separately from the source code. Reconfigure before building so CMake discovers the new source files:

```bash
# From the repository root:
tar --skip-old-files -xzf pseudopotentials/JTH-v2.0-GGA-PBE-atomicdata-xml.tar.gz -C pseudopotentials
cmake -S jdftx -B build
cmake --build build --target jdftx wannier -j 8
# CUDA, if already configured:
cmake --build build --target jdftx_gpu wannier_gpu -j 8

cd examples/paw
mkdir -p results
../../build/jdftx -i 01-scf.in -o results/scf.out
../../build/wannier -i 02-wannier.in -o results/wannier.out
../../build/jdftx -i 03-bands.in -o results/bands.out
```

The examples use a 20 Ha wavefunction cutoff, a 120 Ha density cutoff, 20 bands, and an 8×8×8 SCF k mesh. The Wannier input reads that same band count and mesh. Converge the k mesh, smooth wavefunction cutoff, density cutoff, and reconstruction cutoff for the material and quantity being studied. PAW augmentation spheres should not overlap.

Start CUDA runs with one MPI rank per GPU, then benchmark alternatives for the actual system. For example, `mpirun -n 2 ../../build/jdftx_gpu -i 01-scf.in` uses two ranks on this two-GPU workstation. Small cells and few bands can remain dominated by CPU work and MPI overhead; the on-site kernel speedup does not translate directly into the same whole-job speedup.

Use the same wildcard input as the existing pseudopotential libraries:

```text
ion-species PAW/$ID_PBE.xml
```

For example, `ion Si ...` selects `PAW/Si_PBE.xml`, and `ion Zr ...` selects `PAW/Zr_PBE.xml`. CMake copies the full-resolution JTH datasets into `build/pseudopotentials/PAW` under these names; the normal build/install search paths find them from any working directory. The XML contents are unchanged. Dataset variants retain a suffix, for example `PAW/As_PBE_sp.xml`. Set `-DPAW_JTH_DIR=/path/to/extracted/JTH-datasets` when configuring to use another dataset directory. Direct XML filenames remain supported.

## Choosing cutoffs

For the supplied Zr dataset, start with a smooth-wavefunction cutoff of **20 Ha** and compare against 15, 25, and 30 Ha. A 30 Ha calculation provides a higher-cutoff reference for this convergence sweep. During this sweep, keep the density cutoff fixed:

```text
elec-cutoff 20 120
```

Both values are in Hartree. The first controls the smooth wavefunction basis; the second controls the density grid. JDFTx requires the second value to be at least four times the first and defaults to that minimum if it is omitted. A fixed 120 Ha density cutoff permits the entire 15–30 Ha wavefunction sweep; converge the density cutoff separately afterward. Compare the target band energies, momentum matrix elements, and total-energy differences to the accuracy required for the calculation.

The `pw_ecut` entries in the supplied full-resolution XML files give these high-accuracy starting recommendations:

| Element | Dataset wavefunction cutoff recommendation |
|---|---:|
| Si | 10 Ha |
| Zr | 12 Ha |
| O | 17.5 Ha |
| H | 25 Ha |
| Na | 40 Ha |

For compounds, account for the most demanding constituent. These are dataset recommendations; they require verification for the material and this implementation. The published [JTH library validation](https://www.abinit.org/atomic_data/paw/) used 20 Ha.

For exported all-electron wavefunctions, independently test a higher resolution, for example:

```text
paw-reconstruction-cutoff 200
```

This sets the Fourier projection used by `AEWfns`. It does not enlarge the smooth SCF basis. The analytic PAW matrix corrections in `AEOverlap` and `AEMomenta` do not depend on this export cutoff. The [validation record](VALIDATION.md) shows how increasing it changes reconstructed norms and momentum matrix elements.

## Reconstruction and eigenvalues

The implementation applies

```
|psi_AE> = T |psi_tilde>
T = 1 + sum_ai (|phi_ai> - |phi_tilde_ai>) <p_tilde_ai|
S = T^dagger T = 1 + sum_aij |p_tilde_ai> DeltaS_ij <p_tilde_aj|
H_tilde |psi_tilde_nk> = epsilon_nk S |psi_tilde_nk>.
```

PAW supplies the same valence eigenvalues `epsilon_nk` before and after the transformation. There is no additional set of shifted “all-electron eigenvalues.” `dump End AEBandEigs` writes `paw.eigenvals`, with exactly the same native float64 layout and values as `BandEigs`, in Hartree. No diagonal Hamiltonian matrix is stored.

`dump End AEWfns` writes the Fourier projection of `T|psi_tilde>` onto a separate output basis, in **one binary file**. Set its cutoff with `paw-reconstruction-cutoff <Hartree>`; the default is four times `elec-cutoff`. This does not increase the variational basis used by SCF. Reconstruction near a nucleus needs much higher resolution than the smooth wavefunctions. No renormalization hides missing high-G weight. Analytic PAW overlap and operator corrections remain preferable to matrix elements calculated from a finite reconstruction grid.

Choose the outputs independently:

| Dump variable | Files with `dump-name results/Si.$VAR` |
|---|---|
| `AEWfns` | `Si.paw.wfns`, `Si.paw.Gvectors` |
| `AERealSpaceWfns` | `Si.paw.q<state>.b<band>.rs` |
| `AEOverlap` | `Si.paw.overlap` |
| `AEMomenta` | `Si.paw.momenta` |
| `AEBandEigs` | `Si.paw.eigenvals` |

Each selection also writes `Si.paw.json`, which records the requested outputs, lattice, k points, units, basis sizes, byte offsets, and FFT grid. Nothing else is written implicitly: for example, `AEWfns` does not write eigenvalues or real-space orbitals. To request the reciprocal wavefunctions and all operators:

```text
dump End AEWfns AEOverlap AEMomenta AEBandEigs
paw-reconstruction-cutoff 200
```

`paw.wfns` uses exactly the native `wfns` ColumnBundle binary layout: little-endian complex128, state outermost, band next, and G innermost, with no header or padding between bundles. The plane-wave count can differ between k points. `basis_sizes` and `wavefunction_offsets_bytes` in the JSON locate each bundle. `paw.Gvectors` uses the native text-block convention, with a commented k-point header and reciprocal-lattice integer G vectors for each state. Native code can read the file using `ElecInfo::read` with bases at the **AE reconstruction cutoff**, rather than the smooth SCF cutoff. The reconstructed orbitals are outputs of the PAW transformation; SCF restarts continue to use the smooth `wfns` file.

The full Bloch orbital is `psi_nk(r) = exp(i k.r) sum_G c_nkG exp(i G.r)`. Coefficients use `integral_cell |psi|^2 = volume * sum_G |c_G|^2`. All lengths and momenta are in atomic units.

`dump End AERealSpaceWfns` independently writes the **periodic part `u_nk`**, following native `RealSpaceWfns` conventions, one little-endian complex128 file per state/band. Arrays have shape `fft_grid` from the JSON, with x outermost and z fastest. Grid points are `r = R*(ix/Sx, iy/Sy, iz/Sz)`. The grid is chosen for the AE reconstruction cutoff. Requesting both wavefunction dumps reuses each reconstructed bundle; the writer streams one bundle at a time to bound memory.

The Python reader handles the new `JDFTx-PAW-2` format and existing `JDFTx-PAW-1` q-indexed exports:

```bash
python ../../jdftx/scripts/read_paw.py results/Si.paw
python ../../jdftx/scripts/read_paw.py results/Si.paw --state 0 --real-space results/Si.q0.npz
```

The saved `u` array has axes `(band, x, y, z)`. Add `--bloch` to include the Bloch phase. For reciprocal exports, the NPZ also includes G vectors and coefficients. In Python, `PAWExport.wavefunction(q)` maps just the requested bundle from the single file; `real_space(q)` applies an inverse FFT, and `real_space_dump(q)` reads the separately requested real-space files. `eigenvalues`, `overlap`, and `momenta` are `None` when their corresponding output was not requested.

## Matrix elements and Wannierization

`dump End AEOverlap` writes the all-electron overlap in the generalized PAW metric. `dump End AEMomenta` writes the three Cartesian canonical momentum matrices `p_x`, `p_y`, `p_z`. Both use the eigenstate gauge of the exported wavefunctions. Use `AEBandEigs` for the diagonal Hamiltonian eigenvalues. Momentum is

```
<p>_AE = <psi_tilde|p|psi_tilde>
       + sum_aij <psi_tilde|p_tilde_ai>
         (<phi_ai|p|phi_aj> - <phi_tilde_ai|p|phi_tilde_aj>)
         <p_tilde_aj|psi_tilde>.
```

Overlap uses little-endian complex128 in state, ket band, bra band order, with bra fastest. Momenta use state, Cartesian direction `[x,y,z]`, ket band, bra band order. The reader returns conventional `(state, bra, ket)` and `(state, direction, bra, ket)` arrays. Canonical momentum and velocity differ for scalar-relativistic Hamiltonians; `AEMomenta` exports canonical momentum. Existing `Momenta`/`Velocities` dumps retain the `[r,H]` velocity convention.

`AEMatrices` remains a shorthand for `AEOverlap AEMomenta AEBandEigs`. It now produces these separate files and no `paw.matrices` file or redundant Hamiltonian matrix. Older output files are not deleted; the versioned JSON identifies the files requested by the current export.

The `FiniteDifference` and `TruncatedDC` localization measures support PAW; `RealSpace` is rejected because its augmentation has not been implemented. Native Wannierization uses the true `phi_i phi_j - phi_tilde_i phi_tilde_j` augmentation at finite k separation, including the phase of each atom. Electrostatic compensation functions are not used as a substitute for these overlaps. With PAW, `wannier saveMomenta yes` exports `i*p_AE` in the existing `mlwfP` layout. This has the usual real, antisymmetric representation when real Wannier functions are chosen. `mlwfH` uses the PAW valence eigenvalues. Standard Wannier wavefunction dumps retain the smooth orbitals. The separate `saveAEWfns` and `saveAEWfnsRealSpace` options reconstruct all-electron Wannier orbitals, as described below. This change does not add a Wannier90 `.amn`/`.mmn` exporter or arbitrary-operator matrix element support.

## AE Wannier orbitals and operators

Extend the existing `wannier` command with the requested outputs (the input accepts one `wannier` command):

```text
wannier saveAEWfns yes saveAEWfnsRealSpace yes saveMomenta yes saveR yes saveRP yes
paw-reconstruction-cutoff 200
```

Both AE wavefunction options default to `no` and are independent of the smooth `saveWfns`/`saveWfnsRealSpace` options. [04-wannier-ae.in](04-wannier-ae.in) is a complete example using the same SCF state as the other inputs. Match its `kpoint-folding` and the common `elec-n-bands` to the saved SCF state if you change the example settings. With `wannier-dump-name results/Si-AE.$VAR`, the outputs are:

- `Si-AE.mlwfCAE`: all AE Wannier orbitals in one native ColumnBundle file, little-endian complex128, orbital outermost and supercell G innermost.
- `Si-AE.mlwfCAE.header`: native `mlwfC` basis-header layout, containing orbital/G counts, reciprocal lattice, and supercell integer G vectors.
- `Si-AE.0.mlwfAE`, etc.: complex128 real-space supercell orbitals, x/y/z order with z fastest. These preserve their complex phases; no real-part projection or phase removal is applied.
- `Si-AE.mlwfCAE.json`: supercell lattice, FFT shape, cutoff, supercell Bloch vector, the half-supercell centering translation, and the full k-mesh/symmetry mapping associated with the final `mlwfU` rotations.
- `Si-AE.mlwfPAW.json`: operator definitions, requested matrix outputs, reconstruction cutoff, and the finite-difference stencil used for position-related outputs.

The writer reconstructs each reduced Bloch state once, applies the final Wannier rotations, and expands the complete k mesh using the existing space-group and time-reversal transformations. The weighted sum is centered at half the supercell, matching native Wannier output. Real-space exports include the supercell Bloch phase for shifted k meshes. No normalization correction is applied. Both `FiniteDifference` and `TruncatedDC` localization are supported; PAW `RealSpace` localization remains unsupported. Supercell AE wavefunctions can be much larger than smooth orbitals; the real-space selector can be omitted when only reciprocal coefficients are needed.

For an existing compatible Wannier checkpoint, retain its SCF state, centers, mesh, and dump prefix; set `loadRotations yes` in the same `wannier` command and `wannier-minimize nIterations 0` to export without relocalizing.

Typical Wannier matrices automatically use the AE representation for PAW calculations:

| Selection | Output and PAW treatment |
|---|---|
| Always | `mlwfH`: PAW valence eigenvalues rotated and Fourier transformed |
| `saveMomenta yes` | `mlwfP`: analytic PAW canonical `i*p`, including AE-minus-PS corrections |
| `saveR yes` | `mlwfR`: position contribution from true finite-k PAW overlaps |
| `saveRP yes` | `mlwfRP`: position times canonical `i*p`, integrated using reconstructed AE orbitals |
| `saveZ` / `slabWeight` | `mlwfZ` / `mlwfW`: scalar-weight integrals using reconstructed AE orbitals |

Matrix filenames and native storage layouts are retained. `R` and `RP` retain the existing short-range finite-difference convention; derivatives of Wannier rotations required by interpolation remain part of the existing postprocessing. Converge the k mesh for these operators. `RP`, `Z`, and `W` additionally depend on `paw-reconstruction-cutoff`; analytic `H`, `P`, and PAW-overlap `R` do not. Canonical momentum differs from velocity for scalar-relativistic datasets. Spin/SOC and electron-phonon outputs remain outside the current PAW scope.

The reader for the standard `<prefix>.$VAR` filename pattern maps one orbital at a time and can export it to NPZ:

```bash
python ../../jdftx/scripts/read_paw_wannier.py results/Si-AE --orbital 0 \
  --real-space results/Si-AE.w0.npz
```

The NPZ contains `psi[x,y,z]` in the supercell, including any supercell Bloch phase, together with the lattice, orbital index, and cutoff. Frozen-core orbitals are not included.

## Restart and fixed-H calculations

PAW has independent on-site density matrices in addition to the smooth grid density. The SCF mixer includes both. `State`, `ElecDensity`, and `Vscloc` dumps also save a versioned `pawState` text file containing on-site densities and potentials. It validates the lattice, positions, species/channel counts, and the dataset fingerprint when read.

`fix-electron-density results/Si.$VAR` reads `Si.n` and `Si.pawState`, then reconstructs the on-site potential from the saved densities. `fix-electron-potential` reads `Si.Vscloc` and `Si.pawState`. This allows new bands and k points without rerunning SCF. A missing or mismatched PAW state fails explicitly. Ordinary SCF restarts can reconstruct on-site occupations from saved wavefunctions.

Fixed-H PAW restarts require the exact SCF geometry. Automatic symmetry can slightly adjust rounded input positions, while a band path usually disables symmetry. Save `dump End IonicPositions` during SCF and use that saved geometry for the band calculation. Keeping ion declarations in a separate geometry include makes it possible to replace the original geometry include with the saved `ionpos` file, while sharing lattice, species, and cutoff settings.

To export AE states after SCF has already saved `State`, use a separate input with the same cell, ions, cutoffs, band count, and k mesh:

```text
include common.in
kpoint-folding 8 8 8
initial-state results/Si.$VAR
dump-only
dump-name results/Si-AE.$VAR
dump End AEWfns AEOverlap AEMomenta AEBandEigs
paw-reconstruction-cutoff 200
```

Set the folding to the mesh used by the saved state. `dump-only` evaluates the saved state and writes the exports without another SCF cycle. AE reconstruction uses local, cached radial transforms and supports unequal k-point ownership, including MPI ranks with no local k points.

## Implementation and validation

`PAW.cpp` implements all allowed density multipoles, spherical electrostatics including nuclear/core terms, and nonlinear PBE on a tensor Gauss–Legendre/angular quadrature. Smooth compensation charges enter the existing ultrasoft density machinery. The on-site energy removes the corresponding smooth electrostatic/XC/zero-potential terms and supplies their analytic density-matrix derivatives. Kinetic-energy differences come from the XML dataset. Projectors are biorthogonalized on the numerical radial mesh; an input duality error greater than 0.05 is rejected. Sinc and Bessel compensation shapes and partial waves through f are supported. The supplied archive contains 186 datasets: 185 pass the reader checks. `Ba.GGA_PBE-JTH_light.xml` exceeds the input duality tolerance and is rejected; use the full-resolution Ba dataset. Frozen-core total energies include core constants, so absolute energies need not match another code's pseudopotential reference convention.

See the [validation record](VALIDATION.md) for measured errors and cross-code comparisons. Regression checks cover XML rejection, radial duality and multipoles, Gaussian electrostatics, on-site and periodic energy derivatives, overlap identities, momentum Hermiticity, MPI, SCF, reconstruction, both fixed-H paths, and native Wannierization. Tests use `PAW_JTH_DIR` (a CMake cache path) to find the datasets:

```bash
cmake --build build --target checkPAWData checkPAW checkPAWNative jdftx wannier -j 8
ctest --test-dir build -R '^paw' --output-on-failure
```

For a CUDA build, also build `checkPAWData_gpu` and `checkPAW_gpu` before running the suite. `pawExportMPI` compares serial exports with two- and four-rank exports of three k points, checking the single-file coefficients, reciprocal vectors, real-space orbitals, eigenvalues, and separate matrices. The workflow also tests each dump independently, compares the real-space dump with an inverse FFT, and round-trips the single coefficient file through the native C++ ColumnBundle reader/writer without changing any bytes.

The implementation was guided by the PAW-XML specification and QE's `PW/src/paw_onecenter.f90` and projector/augmentation interfaces. References:

- [PAW-XML specification](https://wiki.fysik.dtu.dk/gpaw-files/pawxml/pawxml.pdf).
- [JTH datasets and their validation](https://www.abinit.org/atomic_data/paw/).
- [QE one-center implementation](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/paw_onecenter.f90).
- [Blöchl, Phys. Rev. B 50, 17953 (1994)](https://doi.org/10.1103/PhysRevB.50.17953).
- [Kresse and Joubert, Phys. Rev. B 59, 1758 (1999)](https://doi.org/10.1103/PhysRevB.59.1758).
