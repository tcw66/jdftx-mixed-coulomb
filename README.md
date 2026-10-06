# JDFTx with mixed Coulomb truncation, DFPT phonons and electron–phonon coupling

This version of [JDFTx](https://github.com/shankar1729/jdftx) adds defect-only
mixed Coulomb truncation for localized, neutral defects on periodic slabs.
The clean host retains slab electrostatics, while the defect-induced total
charge is treated with an isolated Wigner–Seitz Coulomb kernel.

This build also adds primitive-cell density-functional perturbation theory
(DFPT) phonons for insulators and metals, Fourier interpolation of phonon
dispersions, and screened electron–phonon matrix elements. These capabilities
run through `jdftx` or `jdftx_gpu` and use the primitive-cell electronic FFT grid.

The mixed Coulomb implementation includes:

- A variational correction to the electronic SCF potential and energy, using
  a fixed clean-system reference.
- Support for Coulomb embedding and a separate isolated-defect embedding center.
- `defectDensityToVscloc`, a one-shot utility that constructs a native `Vscloc`
  and an NSCF input from clean and defect electronic densities and geometries.
- CPU/CUDA implementations and regression tests for SCF, potential construction,
  embedding, and finite-difference derivatives.

See [the method and usage guide](README-defect-coulomb.md) for input commands,
reference generation, charge conventions, embedding settings, limitations, and
validation. The original [method outline](Defect_Only_Coulomb_Truncation_O_Ag111.pdf)
is also included.

The difference charge includes both electrons and the ionic Coulomb source.
This matters when adding or removing atoms: the electronic density difference
alone need not be neutral. The current SCF method supports fixed ions and cell,
neutral vacuum calculations, and disabled spatial symmetries. Localization of
the difference charge must be checked for the chosen cell and embedding center.
A one-shot potential does not make its supplied density self-consistent.

## Primitive-cell DFPT phonons

`phonon-dfpt Nq0 Nq1 Nq2` computes the Cartesian force matrices on a
Gamma-centered coarse q mesh, Fourier transforms them to lattice space, and
writes matrices for interpolation at arbitrary q. No electronic supercell is
required. The electronic k mesh and phonon q mesh can be chosen independently;
states at k+q and k−q are generated automatically in the ground-state potential.
The existing supercell `phonon` executable remains available for finite-difference
calculations and comparison.

The phonon implementation follows
[Baroni et al., Rev. Mod. Phys. 73, 515 (2001)](https://doi.org/10.1103/RevModPhys.73.515),
including the smeared metallic response of Sec. II.C.4. JDFTx reports the review
in its usual citation list, and also reports de Gironcoli's metallic-response
reference when smearing is used. The review is available locally as
`RevModPhys.73.515.pdf`.

### SCF and phonon inputs

Converge and relax the primitive structure separately first. Put the geometry,
pseudopotentials, functional, cutoffs, electronic k mesh, band count and any
smearing in a shared `common.in`. A symmetry-reduced SCF input can contain:

```text
include common.in
symmetries automatic
electronic-scf nIterations 200 nEigSteps 30 \
    energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10
dump-name scf.$VAR
dump End State Fillings BandEigs Forces
```

Then use a separate DFPT input with the same electronic settings:

```text
include common.in
symmetries none
coulomb-interaction periodic
phonon-dfpt-initial-state 01-scf.in scf.$VAR
electronic-minimize nIterations 500 energyDiffThreshold 1e-13
perturb-minimize nIterations 500 algorithm MINRES \
    residualTol 1e-7 residualDiffThreshold 0
phonon-dfpt 4 4 4
dump-name phonon.$VAR
dump End None
```

Run SCF before DFPT, from the directory containing the inputs. The
`phonon-dfpt-initial-state` command unfolds symmetry-reduced SCF wavefunctions
onto the full electronic mesh. The original SCF input and checkpoint must match
the DFPT geometry, pseudopotentials, cutoff, band count, smearing and compatible
weighted k mesh. Paths resolve from the calculation working directory. For an
existing full-mesh checkpoint, use `initial-state scf.$VAR` instead. DFPT still
polishes the ground state before solving the responses.

Complete examples are provided for:

| Example | Workflow |
| --- | --- |
| [Silicon phonons](examples/dfpt-silicon/README.md) | Insulating SCF, coarse q mesh, interpolated dispersion, checks and plots |
| [Aluminum phonons](examples/dfpt-aluminum/README.md) | Symmetry-reduced SCF and metallic DFPT with empty bands |
| [BCC niobium input](examples/dfpt-niobium/results_dfpt/02-dfpt.in) | Metallic DFPT with response k-point reduction, nested solves and a standalone BCC dispersion plotter |
| [Silicon electron–phonon coupling](examples/dfpt-silicon-electron-phonon/README.md) | Fresh SCF including conduction bands, screened derivatives and mode couplings |

For example, from the repository root:

```bash
./examples/dfpt-silicon/run.sh --reduced-scf --out examples/dfpt-silicon/results-new
```

The silicon runner accepts `--jdftx-bin ./build-gpu/jdftx_gpu`, `--pseudo`,
`--kgrid`, `--qgrid`, `--launcher`, and `--q-groups` overrides. Python 3 and
NumPy are required for interpolation; Matplotlib is needed for plots. Example
parameters are starting values for convergence studies.

### Symmetry, response solvers and parallelism

These optimizations are enabled by default for phonon-only calculations:

```text
phonon-dfpt-symmetries yes
phonon-dfpt-displacement-symmetries yes
phonon-dfpt-reuse-offset-states yes
phonon-dfpt-cache yes
phonon-dfpt-source-cache-size 128
phonon-dfpt-balance-q-groups yes
```

Compatible space-group operations and time reversal reduce the q mesh. The
little group of each representative q reduces independent displacement solves,
including Cartesian rotations, atom permutations and cell phases. Offset states
are reused when k±q lies on the electronic mesh; otherwise a frozen-potential
eigensolver generates them. Offset Hamiltonian residuals and response convergence
are checked. Set the symmetry or offset-reuse option to `no` for explicit
comparisons.

Electronic symmetries in the DFPT input must remain `none`. Ground-state and
offset-state setup use the full electronic mesh even when the initial SCF was
symmetry reduced. Enable `phonon-dfpt-k-symmetries yes` to reduce the electronic
**response** mesh for each displacement (default `no`). This uses operations
without translations that preserve q exactly, fix the displaced atom without a
cell shift, and map its direction to itself or its negative. Response densities
are symmetry averaged; full-mesh responses are reconstructed with band sewing
matrices before force contractions, including degenerate electronic subspaces.
Other operations retain the full response mesh. Setting
`phonon-dfpt-symmetries no` also disables electronic response reduction.

The QE-style nested solver is selected separately:

```text
phonon-dfpt-k-symmetries yes
phonon-dfpt-solver nested
phonon-dfpt-mixing 0.4 8 80
```

Nested iteration fixes the response potential during inner Sternheimer solves
and mixes the induced potential using a regularized DIIS history. Independent
band CG uses kinetic preconditioning and removes converged bands from subsequent
Hamiltonian calls. Inner accuracy tightens with the coupled residual, and each
outer iteration reuses the previous response through a correction solve.
Nonpositive curvature or an inner iteration limit triggers MINRES. The three
mixing arguments are the fraction, history length and outer iteration limit;
`0.4 8 80` are their defaults. Both solver paths verify the coupled response
residual and fixed-electron-number Gamma constraint. The default solver remains
`coupled`: nested iteration can be slower, so compare timings for the intended
material and mesh. The [Nb input](examples/dfpt-niobium/results_dfpt/02-dfpt.in)
enables the nested path and k-point reduction explicitly.

Frozen-ground-state XC derivatives and bare displacement potentials are reused
within each representative q. Source vectors are retained for force contractions
where applicable, with at most 128 MiB of additional retained storage per MPI
rank by default. `phonon-dfpt-source-cache-size 0` disables source-vector storage
while keeping the potential and XC caches; `phonon-dfpt-cache no` disables these
caches for comparison. Cache entries are released after each q.

Normally MPI distributes k points while q and displacement solves are sequential.
Add `phonon-dfpt-q-groups 2` to divide ranks between two independent q groups:

```bash
./examples/dfpt-silicon/run.sh --reduced-scf \
    --launcher 'mpirun -n 4' --q-groups 2 \
    --out examples/dfpt-silicon/results-q-parallel
```

The example command runs from the repository root. Four ranks and two groups
give two ranks per q group; each group distributes its k points internally.
Each group initializes its own ground state and stores a
full electronic mesh distributed over its ranks. Use no more groups than MPI
ranks, and preferably no more than the number of independent q points. CPU
threads and CUDA accelerate local operations. Displacements remain sequential
within each group. Completed q responses are not currently saved for restarting
an interrupted grid calculation.

Q groups receive representative q points by estimated work, using displacement
counts, off-mesh offset-state work and group sizes. Unequal group sizes are
supported, and all displacements at a q stay together to retain its caches.
`phonon-dfpt-balance-q-groups no` restores round-robin assignment. The default
is one group; balancing takes effect when multiple groups are requested.

Measured Nb regression timings for the coupled solver show the effect of
electronic response reduction alone:

| Build | Electronic k mesh | Full response mesh | Reduced response mesh | Force-matrix relative difference |
| --- | --- | --- | --- | --- |
| CPU, one rank/thread | 4 × 4 × 4 | 116.6 s | 54.2 s | 7.1 × 10⁻⁹ |
| CUDA, one RTX 4090 | 2 × 2 × 2 | 29.4 s | 19.1 s | 6.3 × 10⁻⁹ |

These are complete DFPT run times using a 2 × 2 × 2 q mesh, 14 bands, an 8-Hartree
cutoff and Fermi smearing of 0.01 Hartree. Displacement reduction was disabled
to isolate electronic k reduction. These small regression settings establish
numerical equivalence; they are not converged production parameters. Timings
depend on the mesh and hardware, and CPU/CUDA rows use different electronic
meshes. [The validation script](jdftx/test/dfpt/checkFrozenCache.py) compares
full-mesh, k-reduced, nested and combined calculations and records their timings.

### Metals and supported scope

Insulating inputs can use fully occupied bands without smearing. Metals and
gapless systems require `elec-smearing Fermi <width>` or
`elec-smearing Gauss <width>` at fixed electron number, with enough empty states
set by `elec-n-bands`. Widths are in Hartree. The highest stored band must have
occupation below `1e-6` at every source and offset k point. Increase the band
count in both SCF and DFPT and regenerate the checkpoint if this check fails.
The response includes changing occupations, finite limits for degenerate
energies, and the Gamma-point Fermi-level shift enforcing fixed electron number.
Metallic force matrices are derivatives of the smeared free energy at fixed
width; converge the k mesh, width and band count together.

The primitive-cell path currently supports periodic, nonmagnetic scalar states,
analytic LDA/GGA kernels, and nonrelativistic norm-conserving pseudopotentials
without nonlinear core corrections, such as suitable SG15 potentials. Ultrasoft
potentials, nonlinear core corrections, spin polarization/spinors, exact exchange,
DFT+U, meta-GGAs, fluids, vdW pair corrections, fixed chemical potential, MP1 and
Cold smearing are unsupported. Ionic/lattice optimization must be performed in
a separate calculation.

**DFPT phonons currently require periodic Coulomb and cannot be combined with
the defect-only mixed slab truncation described above.** Born effective charges
and long-range polar corrections are not implemented, so ordinary Fourier
interpolation does not recover the directional LO–TO limit at Gamma in polar
crystals.

### Outputs and interpolation

With `dump-name phonon.$VAR`, the driver writes:

| Output | Contents |
| --- | --- |
| `phonon.phononForceMatrixQ` | Coarse q points and complex Cartesian force matrices, Hartree/bohr² |
| `phonon.phononCellMap` | Integer lattice vectors and Cartesian cell offsets |
| `phonon.phononCellWeights` | Binary atom-pair Wigner–Seitz weights |
| `phonon.phononOmegaSq` | Binary column-major lattice-space dynamical matrices, with cell weights and masses included |
| `phonon.phononBasis` | Atom identities, mass-weighted Cartesian directions and masses |

Force matrices are checked for Hermiticity, reciprocal reality is enforced,
and the Gamma acoustic sum rule is applied before interpolation. Create
`qpath.dat` with three fractional reciprocal coordinates per row, then run from
the repository root, substituting the paths to your results:

```bash
python3 jdftx/scripts/phononDFPT /path/to/results/phonon. /path/to/qpath.dat \
    -o dispersion.dat --matrices dispersion.npz
```

The text output gives frequencies in cm⁻¹; negative values denote imaginary
modes. The optional NPZ stores dynamical matrices, squared frequencies in atomic
units and eigenvectors. Phonon energies in meV are
`sign(omega_squared) * sqrt(abs(omega_squared)) * 27211.386245988`.
Interpolation operates on the lattice-space matrices before diagonalization.

For BCC Nb, the standalone plotter reads the same `phonon.*` matrix inputs and
the adjacent `Nb.lattice`, generates the Γ–H–N–Γ–P–H | P–N path, and writes PNG/PDF
plots and dispersion tables in meV. Run it from the repository root after DFPT:

```bash
python3 examples/dfpt-niobium/results_dfpt/plot_dispersion.py
```

Output goes to `results_dispersion` beside the plotter. Use `--dfpt` and `--fd`
to compare calculation directories, or `--metadata` to reuse existing q-path
files. NumPy and Matplotlib are required.

## Electron–phonon matrix elements

Add the following to a DFPT phonon input:

```text
phonon-dfpt-electron-phonon yes
```

The export contains the screened Cartesian Hamiltonian derivatives
`<psi_m,k+q | dH_SCF/du_a(q) | psi_n,k>`, including ionic local and nonlocal
pseudopotential derivatives and the induced Hartree/XC potential. Raw matrices
are in Hartree/bohr, without occupation, k-weight, mass or zero-point factors.
All stored bands are included. To include conduction bands, prepare a fresh SCF
checkpoint with Fermi/Gauss smearing and additional bands, keeping those settings
identical in DFPT; the existing four-band silicon checkpoints cannot supply
extra conduction states.

Electron–phonon export currently solves every q and Cartesian displacement
on the full electronic response mesh, overriding q, displacement and electronic
response symmetry reductions. Offset reuse,
symmetry-reduced SCF unfolding and MPI q groups remain available. Storage for
the raw matrices is `16 * Nq * Nk * (3 * Natoms) * Nbands²` bytes plus metadata.
This option can substantially increase work compared with phonon-only runs.

The additional files are `phonon.phononEph` (manifest),
`phonon.phononEphQ000000`, `phonon.phononEphQ000001`, etc. (complex128 Cartesian
derivatives), and matching `.meta` files (q, k, k+q, integration weights and band
energies). Convert them to phonon-mode couplings from the repository root:

```bash
python3 jdftx/scripts/electronPhononDFPT /path/to/results/phonon. -o electron-phonon
```

Each q produces an NPZ with complex `g[k, mode, final_band, initial_band]`, band
energies and signed phonon energies in **meV** by default. Mass weighting and
zero-point normalization are included in `g`. `--units eV` or `--units Hartree`
changes the energy unit; `--q-index 1` selects a computed q, and `--bands 0 8`
selects zero-based bands 0 through 7. Raw Cartesian derivatives and phonon
eigenvectors are also stored. Imaginary, zero and positive frequencies below
the default 0.01 meV threshold have NaN couplings; use `valid_modes` when summing.
The threshold is adjustable with `--min-frequency-meV`.

These are coarse-mesh Bloch matrix elements. Arbitrary-q phonon interpolation
remains available, but electron–phonon interpolation needs electronic gauge
alignment and is not implemented by this converter. Individual matrix entries
depend on electronic phases and degenerate-subspace rotations; include complete
degenerate multiplets when comparing subspace sums across runs. The export is
distinct from the supercell `phononHsub` format consumed by the existing Wannier
driver.

See the [DFPT method and format guide](jdftx/doc/tutorials/phonon/DFPT.dox)
for response equations, phase conventions, normalization and full output details.

## Build

Install the normal JDFTx dependencies (C++ compiler, CMake, GSL, FFTW,
BLAS/LAPACK, and MPI for the default configuration), then build from this root:

```bash
cmake -S jdftx -B build
cmake --build build --parallel --target jdftx ConstructDefectPotential TestDefectCoulomb
```

Use `build/jdftx` for the modified SCF calculation and
`jdftx/scripts/defectDensityToVscloc --help` for the one-shot interface.
Provide the required pseudopotentials through JDFTx's normal search paths or
`JDFTX_PSEUDO_DIR`. Python 3 is required for the one-shot wrapper and regression
scripts. CUDA build details and input examples are in the usage guide.

For the DFPT auxiliary checks, also build `TestDFPT`:

```bash
cmake --build build --parallel --target TestDFPT
```

To configure and build the GPU versions, install CUDA and use:

```bash
cmake -S jdftx -B build-gpu -DEnableCUDA=ON -DEnableMPI=ON
cmake --build build-gpu --parallel --target jdftx_gpu ConstructDefectPotential_gpu TestDefectCoulomb_gpu
```

The existing `build-gpu` in this workspace uses CUDA architecture 89 for the
RTX 4090 GPUs. Set `CMAKE_CUDA_ARCHITECTURES` for the target hardware when
configuring a different machine. The DFPT phonon and electron–phonon commands
are part of the main `jdftx`/`jdftx_gpu` executable; no separate phonon executable
is needed for those commands.

## Tests

With the targets above built and the test pseudopotentials available:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir build -R '^defect(Coulomb|Potential|Embedding)$' --output-on-failure
```

Build products and calculation output are not part of the source repository.

Run the available DFPT regressions with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir build -R '^dfpt' --output-on-failure
```

Physical tests are registered when Python/NumPy and the required SG15
pseudopotentials are available; MPI checks additionally require an MPI-enabled
build and launcher. Coverage includes analytic ionic Ewald derivatives,
finite-q response against independently relaxed supercell forces, Fourier
interpolation, symmetry reconstruction, offset reuse, reduced-SCF restarts,
metallic Fermi/Gauss response, and MPI q groups. `dfptMainSpeedups` and
`dfptMainSpeedupsNb` compare full-mesh coupled, k-reduced, nested and combined
response calculations; `dfptMainSpeedupsNbMPI` also checks unequal MPI groups
and Gamma charge conservation. `dfptKSymmetryNbDense` tests a 64-point Nb mesh
with truncated degenerate empty-band subspaces. `dfptFrozenCache` and its Nb/MPI
variants check cache budgets and cached/uncached force matrices. Performance
tests record timings without requiring a speedup on every machine.
Electron–phonon tests compare
screened diagonal band derivatives with independent SCF finite differences,
check q/−q reciprocity and mass/frequency normalization, exercise off-mesh target
states, and compare complete-band-subspace coupling norms across MPI q groups.

CPU and CUDA response checks and MPI q-group checks have passed in this
workspace. The example inputs still require physical convergence studies for
production use. Detailed test definitions are in
[the test configuration](jdftx/test/CMakeLists.txt).

## Upstream and license

This repository preserves upstream history, based on JDFTx commit `52548a96`.
JDFTx and these modifications are distributed under the GNU General Public
License, version 3 or later; see [COPYING](jdftx/COPYING) and the source notices.
Retain JDFTx attribution and cite the software and methods used in calculations.
