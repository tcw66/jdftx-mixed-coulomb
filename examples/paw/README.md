# Native PAW-XML support

This implementation adds frozen-core, unpolarized scalar PAW with the internal PBE functional. It reads the supplied JTH PAW-XML atomic datasets directly; UPF conversion is unnecessary. Both CPU and CUDA executables use the same on-site functional, evaluated on CPU radial/angular quadrature.

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

This sets the Fourier projection used by `AEWfns`. It does not enlarge the smooth SCF basis. The analytic PAW matrix corrections in `AEMatrices` do not depend on this export cutoff. The [validation record](VALIDATION.md) shows how increasing it changes reconstructed norms and momentum matrix elements.

## Reconstruction and eigenvalues

The implementation applies

```
|psi_AE> = T |psi_tilde>
T = 1 + sum_ai (|phi_ai> - |phi_tilde_ai>) <p_tilde_ai|
S = T^dagger T = 1 + sum_aij |p_tilde_ai> DeltaS_ij <p_tilde_aj|
H_tilde |psi_tilde_nk> = epsilon_nk S |psi_tilde_nk>.
```

PAW supplies the same valence eigenvalues `epsilon_nk` before and after the transformation. There is no additional set of shifted “all-electron eigenvalues.” `BandEigs` and `paw.eigenvals` therefore contain identical values, in Hartree.

`dump End AEWfns` writes the Fourier projection of `T|psi_tilde>` onto a separate output basis. Set its cutoff with `paw-reconstruction-cutoff <Hartree>`; the default is four times `elec-cutoff`. This does not increase the variational basis used by SCF. Reconstruction near a nucleus needs much higher resolution than the smooth wavefunctions. No renormalization hides missing high-G weight. Analytic PAW overlap and operator corrections remain preferable to matrix elements calculated from a finite reconstruction grid.

With `dump-name results/Si.$VAR`, output files are:

- `Si.paw.json`: lattice, k points, units, binary layouts, and reconstruction cutoff.
- `Si.paw.q0.wfns`, etc.: native-endian complex128 coefficients of the periodic part `u_nk`, with band outermost and G innermost.
- `Si.paw.q0.Gvectors`, etc.: matching reciprocal-lattice integer G vectors in text.
- `Si.paw.eigenvals`: native-endian float64, state outermost and band innermost.
- `Si.paw.matrices`: produced by `AEMatrices`, as described below.

The full Bloch orbital is `psi_nk(r) = exp(i k.r) sum_G c_nkG exp(i G.r)`. Coefficients use `integral_cell |psi|^2 = volume * sum_G |c_G|^2`. All lengths and momenta are in atomic units.

The Python reader can inspect these files or transform a state onto a real-space grid:

```bash
python ../../jdftx/scripts/read_paw.py results/Si.paw
python ../../jdftx/scripts/read_paw.py results/Si.paw --state 0 --real-space results/Si.q0.npz
```

The saved `u` array has axes `(band, x, y, z)`. Add `--bloch` to include the Bloch phase. The NPZ also includes G vectors, reciprocal coefficients, and the k point.

## Matrix elements and Wannierization

`dump End AEMatrices` writes five matrices per k point: overlap, Hamiltonian, and Cartesian canonical momenta `p_x`, `p_y`, `p_z`. Matrices use the eigenstate gauge of the exported wavefunctions. The Hamiltonian is diagonal, the overlap is the generalized PAW metric, and momentum is

```
<p>_AE = <psi_tilde|p|psi_tilde>
       + sum_aij <psi_tilde|p_tilde_ai>
         (<phi_ai|p|phi_aj> - <phi_tilde_ai|p|phi_tilde_aj>)
         <p_tilde_aj|psi_tilde>.
```

The binary order is state, operator, ket band, bra band, with the last index fastest. The reader returns conventional `(state, operator, bra, ket)` arrays. Canonical momentum and velocity differ for scalar-relativistic Hamiltonians; `AEMatrices` exports canonical momentum. Existing `Momenta`/`Velocities` dumps retain the `[r,H]` velocity convention.

The `FiniteDifference` and `TruncatedDC` localization measures support PAW; `RealSpace` is rejected because its augmentation has not been implemented. Native Wannierization uses the true `phi_i phi_j - phi_tilde_i phi_tilde_j` augmentation at finite k separation, including the phase of each atom. Electrostatic compensation functions are not used as a substitute for these overlaps. With PAW, `wannier saveMomenta yes` exports `i*p_AE` in the existing `mlwfP` layout. This has the usual real, antisymmetric representation when real Wannier functions are chosen. `mlwfH` uses the PAW valence eigenvalues. Standard Wannier wavefunction dumps still contain the smooth orbitals; use `AEWfns` for reconstructed Bloch states. This change does not add a Wannier90 `.amn`/`.mmn` exporter or arbitrary-operator matrix element support.

## Restart and fixed-H calculations

PAW has independent on-site density matrices in addition to the smooth grid density. The SCF mixer includes both. `State`, `ElecDensity`, and `Vscloc` dumps also save a versioned `pawState` text file containing on-site densities and potentials. It validates the lattice, positions, species/channel counts, and the dataset fingerprint when read.

`fix-electron-density results/Si.$VAR` reads `Si.n` and `Si.pawState`, then reconstructs the on-site potential from the saved densities. `fix-electron-potential` reads `Si.Vscloc` and `Si.pawState`. This allows new bands and k points without rerunning SCF. A missing or mismatched PAW state fails explicitly. Ordinary SCF restarts can reconstruct on-site occupations from saved wavefunctions.

## Implementation and validation

`PAW.cpp` implements all allowed density multipoles, spherical electrostatics including nuclear/core terms, and nonlinear PBE on a tensor Gauss–Legendre/angular quadrature. Smooth compensation charges enter the existing ultrasoft density machinery. The on-site energy removes the corresponding smooth electrostatic/XC/zero-potential terms and supplies their analytic density-matrix derivatives. Kinetic-energy differences come from the XML dataset. Projectors are biorthogonalized on the numerical radial mesh; an input duality error greater than 0.05 is rejected. Sinc and Bessel compensation shapes and partial waves through f are supported. The supplied archive contains 186 datasets: 185 pass the reader checks. `Ba.GGA_PBE-JTH_light.xml` exceeds the input duality tolerance and is rejected; use the full-resolution Ba dataset. Frozen-core total energies include core constants, so absolute energies need not match another code's pseudopotential reference convention.

See the [validation record](VALIDATION.md) for measured errors and cross-code comparisons. Regression checks cover XML rejection, radial duality and multipoles, Gaussian electrostatics, on-site and periodic energy derivatives, overlap identities, momentum Hermiticity, MPI, SCF, reconstruction, both fixed-H paths, and native Wannierization. Tests use `PAW_JTH_DIR` (a CMake cache path) to find the datasets:

```bash
cmake --build build --target checkPAWData checkPAW jdftx wannier -j 8
ctest --test-dir build -R '^paw' --output-on-failure
```

The implementation was guided by the PAW-XML specification and QE's `PW/src/paw_onecenter.f90` and projector/augmentation interfaces. References:

- [PAW-XML specification](https://wiki.fysik.dtu.dk/gpaw-files/pawxml/pawxml.pdf).
- [JTH datasets and their validation](https://www.abinit.org/atomic_data/paw/).
- [QE one-center implementation](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/paw_onecenter.f90).
- [Blöchl, Phys. Rev. B 50, 17953 (1994)](https://doi.org/10.1103/PhysRevB.50.17953).
- [Kresse and Joubert, Phys. Rev. B 59, 1758 (1999)](https://doi.org/10.1103/PhysRevB.59.1758).
