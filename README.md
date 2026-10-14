# JDFTx with mixed Coulomb electrostatics, Koopmans bands and DFPT

This version of [JDFTx](https://github.com/shankar1729/jdftx) provides:

- Defect-only mixed Coulomb truncation for localized, neutral defects on
  periodic slabs, with self-consistent and one-shot potential workflows.
- Single-spin Koopmans (KI) corrections around a spin-unpolarized PBE ground
  state, symmetry reuse, parallel response groups and KI/PBE band comparison.
- Primitive-cell DFPT phonons and screened electron–phonon matrix elements,
  with periodic/slab electrostatics, Fourier interpolation and response
  solver, symmetry and cache optimizations.
- Occupied-state Berry-phase bulk polarization and self-consistent finite-field
  dielectric tensors, with fixed-strain internal-ion relaxation.
- CPU and CUDA executables in one shared build directory.

This is the main project guide, including the complete mixed Coulomb method
and usage instructions. Material-specific inputs and walkthroughs are linked
in the relevant sections.

## Contents

- [Build](#build)
- [KI and original PBE band comparison](#ki-and-original-pbe-band-comparison)
  - [Coarse-grid screening workflow](#coarse-grid-screening-workflow)
  - [KI input reference](#ki-input-reference)
- [Mixed Coulomb features](#mixed-coulomb-features)
  - [Mixed Coulomb input reference](#mixed-coulomb-input-reference)
  - [Clean reference](#clean-reference)
  - [Defect SCF](#defect-scf)
  - [Outputs and supported scope](#outputs-and-supported-scope)
  - [Mixed Coulomb validation](#mixed-coulomb-validation)
  - [One-shot potential construction for NSCF](#one-shot-potential-construction-for-nscf)
  - [Embedded mixed Coulomb truncation](#embedded-mixed-coulomb-truncation)
- [Primitive-cell DFPT phonons](#primitive-cell-dfpt-phonons)
  - [DFPT input reference](#dfpt-input-reference)
- [Electron–phonon matrix elements](#electronphonon-matrix-elements)
- [Berry-phase bulk polarization and static dielectric response](#berry-phase-bulk-polarization-and-static-dielectric-response)
- [Tests](#tests)
- [Upstream and license](#upstream-and-license)

### Added command index

| Command | Executable |
|---|---|
| [`berry-phase-field`](#berry-phase-bulk-polarization-and-static-dielectric-response) | `jdftx` / `jdftx_gpu` |
| [`defect-coulomb`](#defect-coulomb) | `jdftx` / `jdftx_gpu` |
| [`defect-coulomb-center`](#defect-coulomb-center) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt`](#phonon-dfpt) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-balance-q-groups`](#phonon-dfpt-balance-q-groups) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-cache`](#phonon-dfpt-cache) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-displacement-symmetries`](#phonon-dfpt-displacement-symmetries) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-electron-phonon`](#phonon-dfpt-electron-phonon) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-eph-cache-size`](#phonon-dfpt-eph-cache-size) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-eph-reciprocity`](#phonon-dfpt-eph-reciprocity) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-initial-state`](#phonon-dfpt-initial-state) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-k-symmetries`](#phonon-dfpt-k-symmetries) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-mixing`](#phonon-dfpt-mixing) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-optimize`](#phonon-dfpt-optimize) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-q-groups`](#phonon-dfpt-q-groups) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-reuse-offset-states`](#phonon-dfpt-reuse-offset-states) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-solver`](#phonon-dfpt-solver) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-source-cache-size`](#phonon-dfpt-source-cache-size) | `jdftx` / `jdftx_gpu` |
| [`phonon-dfpt-symmetries`](#phonon-dfpt-symmetries) | `jdftx` / `jdftx_gpu` |
| [`wannier-koopmans`](#wannier-koopmans) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-response`](#wannier-koopmans-response) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-screening-mesh`](#wannier-koopmans-screening-mesh) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-reference-density`](#wannier-koopmans-reference-density) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-screening-only`](#wannier-koopmans-screening-only) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-alpha-file`](#wannier-koopmans-alpha-file) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-save-densities`](#wannier-koopmans-save-densities) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-spin`](#wannier-koopmans-spin) | `wannier` / `wannier_gpu` |
| [`wannier-koopmans-verbose`](#wannier-koopmans-verbose) | `wannier` / `wannier_gpu` |

The references also cover the added `dump` variables and the existing
`perturb-minimize` settings required by DFPT. These are native input commands;
launching a program or choosing MPI ranks remains a separate shell operation.

## Build

Install the normal JDFTx dependencies: C++ compiler, CMake, GSL, FFTW,
BLAS/LAPACK and MPI. The supplied shared-build presets also require CUDA 12
at `/usr/local/cuda-12` and select CUDA architecture 89 for the RTX 4090s in
this workspace. Adjust the CUDA toolkit path and architecture for other
machines.

From the repository root, configure and build all six CPU/GPU executables:

```bash
cmake --preset unified -S jdftx
cmake --build build --parallel 4 --target \
    jdftx wannier phonon jdftx_gpu wannier_gpu phonon_gpu
```

Alternatively, run `cmake --preset unified` and
`cmake --build --preset unified` from `jdftx/`. The `cpu` and `gpu` presets
select subsets of targets in the same CUDA-enabled configuration. Release
mode and MPI are enabled; profiling is disabled.

The shared `build/` contains `jdftx`, `wannier`, `phonon` and their `_gpu`
versions. Source `default-env.sh` from the repository root to select them.
`build-gpu` is a local compatibility symlink to `build/`; previous configurations
and diagnostic results are archived in `archive/build-cleanup-20261012/`.
The DFPT phonon and electron–phonon commands run through `jdftx` or
`jdftx_gpu`; the separate `phonon` executable provides the existing
supercell finite-difference workflow.

Build the mixed Coulomb utilities and native validation tools as needed:

```bash
cmake --build build --parallel 4 --target \
    ConstructDefectPotential ConstructDefectPotential_gpu \
    TestDefectCoulomb TestDefectCoulomb_gpu TestDFPT
```

For a CPU-only installation without CUDA, configure a separate directory:

```bash
cmake -S jdftx -B build-cpu -DEnableCUDA=OFF -DEnableMPI=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 4 --target \
    jdftx wannier phonon ConstructDefectPotential TestDefectCoulomb TestDFPT
```

Use that directory instead of `build` in the commands below. Provide
pseudopotentials through JDFTx's normal search paths or `JDFTX_PSEUDO_DIR`.

Executable targets and the CPU/GPU/unified presets automatically prepare the
bundled SG15/GBRV library in `build/pseudopotentials`. Preparation is incremental;
unchanged archives are not extracted again. Native inputs can therefore use:

```text
ion-species SG15/$ID_ONCV_PBE.upf
```

JDFTx replaces literal `$ID` with each ion's species name and resolves the
unversioned SG15 alias to the latest bundled version, with a matching Pulay
file when available. The compiled build-library search path works from any
calculation directory without an environment override. Keep `$ID` literal in
input files; quote or use a quoted here-document when generating them in a shell.
`JDFTX_PSEUDO_DIR` supplies an additional library search directory, for example
`/path/to/pseudopotentials` containing `SG15/`. It does not replace the native
current-directory/build/install search paths. Installed builds include these
libraries under `share/jdftx/pseudopotentials`.

Python 3 is needed for utility wrappers and regression scripts; band and
phonon plotting dependencies are listed with their workflows.

## KI and original PBE band comparison

The [silicon Koopmans example](examples/koopmans/README.md) uses separate
valence and conduction KI runs with a spin-unpolarized PBE ground state.
Once both runs finish, regenerate their band comparison from the repository
root:

```bash
python3 examples/koopmans/plot_separate_bands.py
```

The plotter reads `examples/koopmans/results/{valence,conduction}.*` and
recovers the original projected PBE Hamiltonians as `mlwfHki - mlwfKI`,
preserving complex hoppings. It overlays KI and PBE on the same axes with a
common energy zero; no additional PBE calculation is needed. PNG/PDF plots,
a band table, and a JSON gap report are saved in the example's `results/`.

Each plot labels the sampled path gaps and interpolated Brillouin-zone gaps.
The latter use a 16×16×16 search mesh plus the path, followed by local
refinement from multiple starting points. Use `--gap-mesh 24 24 24` to check
search convergence; `--fold NX NY NZ`, `--valence PREFIX`, `--conduction PREFIX`
and `--output-dir DIRECTORY` select other completed runs. The underlying
Wannier localization and disentanglement also need convergence. NumPy,
SciPy and Matplotlib are required. Incomplete runs are rejected before plots
are written.

The native `wannier-koopmans` implementation screens one spin orbital through
charge and magnetic response around the nonmagnetic ground state. It retains
the original DFT total energy and writes the corrected spectral Hamiltonian.
`wannier-koopmans-spin yes` is the default; `no` selects the legacy charge-only
response. Verified orbital/q symmetry reuse, occupied-state response columns,
MPI q groups and bounded caches reduce repeated work. The optional
`wannier-koopmans-screening-mesh` changes only the q quadrature for screening;
the electronic mesh and bare Hamiltonian keep the full resolution.
The response planner reports the number of independent q points before solves
start. For the current 8×8×8 conduction functions at symmetry tolerance `1e-4`,
this is **65 q points**, 300 orbital–q pairs and 600 charge/magnetic sector
solves. Progress uses each MPI group's assigned q total, and each completed
sector prints its elapsed solve time. These counts depend on the Wannier
functions and settings. Detailed iterations are enabled with
`wannier-koopmans-verbose yes`.

The [KI input reference below](#ki-input-reference) documents the command syntax,
native defaults, spin treatment, logging, solver controls, caches and coarse
screening. The [silicon example guide](examples/koopmans/README.md) also documents
localization settings, workflow/script options, environment variables and
output formats.

Wannier restart fingerprints invalidate rotations and overlap caches when
the saved SCF wavefunctions change. KI regressions cover spin kernels,
projector caches, mesh correlations, occupied-state response, symmetry reuse,
solver agreement, coarse screening and MPI q groups. Run them with
`ctest --test-dir build -R '^(koopmans|projectorCache|kMeshCorrelation|batchedProjectors)' --output-on-failure`.

### Coarse-grid screening workflow

The separate-grid launcher follows the official Koopmans KI(2)/DFPT workflow:
retain the fine-grid SCF density, solve NSCF states on a coarse electronic
mesh, localize coarse Wannier orbitals, calculate screening on the full coarse
k/q mesh, and transfer matched orbital coefficients to the fine-grid
Hamiltonian. The fine Hamiltonian stage performs no DFPT screening solves.

First complete the fine SCF and both valence/conduction localizations using
the [silicon inputs](examples/koopmans/README.md). The source directory must
contain the matching SCF `wfns`, `eigenvals` and `n` files, each manifold's
`mlwfU`, `mlwfM0` and `mlwfState`, its input files, and any numerical trials.
Then run from the repository root:

```bash
python3 examples/koopmans/run_coarse_screening.py \
  --example-dir /path/to/completed-silicon \
  --coarse-grid 4 4 4 --cpu --ranks 2 --threads 2 \
  --directory /path/to/coarse-4-run --plot
```

Use a new output directory. Omit `--cpu` to select CUDA. NumPy and SciPy are
required; plotting also requires Matplotlib. The launcher defaults to the
executables in `build/`; `JDFTX_BINARY`, `WANNIER_BINARY` and `MPIEXEC` override
the executables and MPI launcher. The fine grid is read from `common.in`;
coarse dimensions need not divide it.

`--prepare-only` writes inputs without native calculations. To continue a
prepared or interrupted run, repeat the command with `--resume`; reuse is
validated against input, checkpoint, executable, library and product hashes.
`--stop-after nscf|wannier|screening|hamiltonian` selects the final stage.
Epsilon is inherited from the source KI input, or overridden with
`--epsilon-infinity VALUE` at both grids. A nonzero value must come from the
same functional/pseudopotential model. With epsilon=0, check mesh convergence
of charged-orbital screening carefully.

Outputs are separated into `coarse/` screening and `fine/` Hamiltonians.
`fine/*.alpha.mapping.json` records the orbital correspondence; `logs/`
contains stage logs and `stages.json` records validated stages. With `--plot`,
`fine/koopmans-bands-comparison.{png,pdf,dat,json}` contains the band comparison
and gap search. Source checkpoints are preserved. Poor localization or
ambiguous orbital matching stops the workflow instead of transferring
unverified coefficients.

This differs from `wannier-koopmans-screening-mesh`, which refines a q subset
while retaining fine electronic states and orbitals. It also remains KI(2),
whereas the official silicon supercell tutorial uses full KI/ΔSCF. The
completed PBE/SG15 4³→8³ run gives 1.652170 eV at epsilon=0 with positive empty
alphas; that is not a converged literature-gap claim. An additional PBE
empty-state comparison against QE remains unresolved. See the
[workflow report](examples/koopmans/official-coarse-workflow.md) for source
references, validation and limitations.

### KI input reference

All commands below belong in a Wannier input. `yes` and `no` are literal
boolean values. Arguments to `wannier-koopmans` and
`wannier-koopmans-response` are **positional**, in the order shown. Omitting
trailing arguments uses native defaults; to change a later argument, supply
all earlier arguments. The angle brackets below name arguments and are not
part of a runnable input. The ancillary `wannier-koopmans-*` commands require
`wannier-koopmans` to be present, except `wannier-koopmans-save-densities`,
which can export densities from localization alone.

The supported ground state is a nonmagnetic insulator (`spintype no-spin`)
with fixed integer occupations, nonrelativistic norm-conserving pseudopotentials,
an analytic semilocal LDA/GGA XC kernel and a Gamma-centered uniform electronic
mesh. KI currently requires **3D periodic Coulomb without embedding or electric
fields**; slab/mixed Coulomb KI, metals, spin-polarized ground states, hybrids,
meta-GGAs and DFT+U are unsupported. Store at least one empty band so the
implementation can check the DFT gap. SCF symmetry reduction
is supported: KI reconstructs the full electronic mesh. Occupied and empty
Wannier manifolds are localized separately in this example. KI changes the
spectral Hamiltonian; the underlying DFT total energy is retained.

#### `wannier-koopmans`

```text
wannier-koopmans <epsilonInfinity> <nIterations> <residualTol> <reduceSymmetry> <symmetryTol> <occupiedResponse> <qGroups> <optimizeResponse>
```

The current silicon example line in
`examples/koopmans/ki-response.in` is:

```text
wannier-koopmans 0 400 1e-7 yes 1e-4 yes 2 yes
```

This example requests two q groups and therefore needs at least two MPI ranks.
For a single-rank launch, set `qGroups` to `1`. Simply adding
`wannier-koopmans` with no arguments enables KI with all native defaults:

```text
wannier-koopmans 0 300 1e-7 yes 1e-6 yes 1 yes
```

The command runs after Wannier localization and writes `mlwfHki` (corrected
Hamiltonian), `mlwfKI` (correction), `koopmansScreening` (orbital curvatures and
screening coefficients), `koopmansBands` and interpolation/format sidecars.
The original projected PBE Hamiltonian is `mlwfHki - mlwfKI`. No extra spin
factor should be applied when reading these outputs.

| Argument | Native default | Example | Meaning and restrictions |
|---|---:|---:|---|
| `epsilonInfinity` | `0` | `0` | Dimensionless electronic dielectric constant for the point-charge Ewald image correction. `0` selects neutralizing-background electrostatics without that correction; converge the screening mesh. A finite value **≥1** enables the isotropic correction for cubic materials: the Gamma image term enters bare curvature as `D` and relaxed curvature as `D/epsilonInfinity`. Values between 0 and 1 and negative values are rejected. |
| `nIterations` | `300` | `400` | Positive maximum iteration count for a response solve. This is distinct from the localization iteration count in `wannier-minimize`. In nested mode it limits the inner solves; the outer cap is `mixIterations` below. |
| `residualTol` | `1e-7` | `1e-7` | Positive, finite relative response tolerance. A smaller value requests a more accurate solve. The final reconstructed response residual is also checked; a nonfinite residual or one exceeding ten times this tolerance aborts KI. |
| `reduceSymmetry` | `yes` | `yes` | Reuse a scalar screening response only after the transformed orbital source passes the numerical equivalence checks. `no` solves every selected orbital–q pair and provides an exhaustive reference. This does not discard terms from the full-mesh bare Hamiltonian. |
| `symmetryTol` | `1e-6` | `1e-4` | Dimensionless relative tolerance for source norm, phase-aligned source-field mismatch and bare-curvature agreement. Must be finite and satisfy **0 < tolerance ≤ 0.01**. Smaller values reject more approximate mappings. This is a symmetry acceptance tolerance, independent of `residualTol`. The example currently uses `1e-4`; the native default remains `1e-6`. Check audit errors and reduced/exhaustive agreement when changing it. |
| `occupiedResponse` | `yes` | `yes` | Keep only occupied response columns for both valence and conduction perturbations. The full plane-wave response space remains available, including empty-state response components. Here that gives 4 columns rather than 60 stored-band columns. `no` uses all stored columns as a reference; it does not select a different Wannier manifold. |
| `qGroups` | `1` | `2` | Integer number of independent MPI response groups. Must be between 1 and the launched MPI rank count. Each group retains k-point parallelism; symmetry-connected q stars stay together for reuse. The active count can be reduced to the number of available screening q stars. This does not set the MPI rank count; set that in the launcher separately. |
| `optimizeResponse` | `yes` | `yes` | Enable frozen XC-kernel reuse, complex-potential application, reused GGA gradients, frozen-orbital caches, batched GPU operations and q-star scheduling. `no` selects reference operators and the original q traversal for comparison. These optimizations retain the response equations and requested tolerance. |

Orbital symmetry reuse concerns the actual Wannier sources, rather than
crystal geometry alone. A rotation may map an orbital to another orbital at
a different q. A symmetry that mixes orbitals does not automatically permit
copying one orbital's response. Equal centers or spreads alone are insufficient.
The cached source is transformed with crystal rotations/translations,
reciprocal wrapping and optional time reversal; an overall source phase is
removed before comparing fields. Accepted source and curvature errors must
both pass the configured checks.

#### `wannier-koopmans-spin`

```text
wannier-koopmans-spin <spinResolved>
```

`spinResolved` is `yes` by default and in this example. A single spin orbital
has charge source `n_i = rho_i` and magnetization source `m_i = rho_i`.
The implementation solves separate charge and magnetic responses and combines
their contractions; the Hartree kernel appears only in the charge sector.
The PBE ground state remains spin unpolarized. Do not apply an additional
spin-degeneracy factor to the emitted KI Hamiltonian or screening coefficients.
`no` selects the legacy charge-only calculation for diagnostics. The same
source-symmetry checks apply to both modes; spin treatment does not switch off
orbital symmetry reuse.

#### `wannier-koopmans-verbose`

```text
wannier-koopmans-verbose <verbose>
```

`verbose` defaults to `no`, as in the example. `no` retains progress for
independent orbital–q responses, final residuals, screening summaries and
warnings/errors. `yes` additionally prints individual response iterations,
k-point/basis listings and symmetry-reuse messages. It controls native log
files as well as terminal output. Before response solves, a source-only symmetry
planning pass reports the global number of independent q points, orbital–q
pairs and charge/magnetic sector solves. It uses the actual localized orbitals,
symmetry tolerance, selected screening mesh and cache budget; the current
8×8×8 conduction setup at tolerance `1e-4` needs **65 independent q points**,
300 orbital–q pairs and 600 sector solves. This is not a universal count for
every 8×8×8 calculation. Each group reports its assigned independent-q total.
The group-local `computed q i/N` counter advances
only when a new q needs an independent response; it is not a q-star number
or a fraction of the full mesh. Other orbitals and spin sectors at that q
retain its counter value. After each completed charge or magnetic response,
a matching progress line reports the wall-clock solve duration in seconds
(`charge response completed in 123.45 s.`). These timings remain visible with
`verbose no` and are written to the main or corresponding q-group log.
Symmetry-reused responses and source-only audits do not print solve timings.

#### `wannier-koopmans-screening-mesh`

```text
wannier-koopmans-screening-mesh <nq0> <nq1> <nq2> [<energyTolerance_eV>=0.005]
```

All three dimensions are required positive integers. Each must divide the
corresponding electronic `kpoint-folding` dimension. Omit the command to use
the full electronic mesh for screening. For an 8×8×8 electronic mesh,
`4 4 4`, `2 2 2` and `1 1 1` are valid; `3 3 3` is not.

The implementation samples a Gamma-centered subset of q points and uses
properly normalized quadrature weights for the bare/relaxed curvature ratio
that determines `alpha`. Electronic k integration, orbital sources, the bare
spectral Hamiltonian and band interpolation retain the full 8×8×8 mesh.
The requested q mesh is the initial adaptive mesh. Responses are reused
as it grows; refinement stops after two successive KI Hamiltonian changes
below `energyTolerance_eV` (default 0.005 eV), or at the full mesh. This is
a change-between-mesh criterion, not a rigorous error bound. A fourth
argument of `0` requests a fixed q-mesh diagnostic. This q-only approximation
is distinct from the separate coarse electronic-grid workflow above.

In `examples/koopmans/ki-screening-mesh.in`, the line below is **commented out**, so coarse
screening is currently disabled:

```text
# wannier-koopmans-screening-mesh 4 4 4
```

Remove the leading `#` to enable it for the next native launch. Editing an
input does not change a running process. Gamma-only screening (`1 1 1`) is
useful for a diagnostic comparison; a source-symmetry check at Gamma does not validate
it as a production gap approximation.

#### `wannier-koopmans-reference-density`

```text
wannier-koopmans-reference-density <filename>
```

Read the raw fine SCF `.n` density for KI potentials and the XC kernel instead
of reconstructing it from loaded NSCF wavefunctions. The reference must match
the model, electron count and FFT grid. This preserves the fine SCF reference
when screening uses coarse NSCF states.

#### `wannier-koopmans-screening-only`

```text
wannier-koopmans-screening-only [yes]
```

Calculate coefficients and response diagnostics without the corrected
spectral Hamiltonian. Requires KI; incompatible with an alpha file or a
screening-mesh override. Use the full k/q mesh of the coarse electronic model.

#### `wannier-koopmans-alpha-file`

```text
wannier-koopmans-alpha-file <filename>
```

Read a complete `koopmansScreening` table in target orbital order and build
the Hamiltonian without screening response solves. KI(2), spin, epsilon and
occupations must match. The donor must have equal electronic k and screening
q meshes; old q-only tables are rejected. The launcher validates model and
orbital correspondence before import. Incompatible with screening-only or a
screening-mesh override. Coefficients are not clipped or fitted.

#### `wannier-koopmans-save-densities`

```text
wannier-koopmans-save-densities [yes]
```

Export Gamma-periodic orbital densities as float64
`koopmansOrbitalDensities` with grid/lattice metadata in
`koopmansDensityFormat`. Requires Wannier localization and a nonmagnetic
ground state; does not require KI. The launcher uses these densities to match
fine and coarse orbitals.

#### `wannier-koopmans-response`

```text
wannier-koopmans-response <nested> <independentBandCG> <mixFraction> <mixHistory> <mixIterations> <orbitalCacheMiB> <symmetryCacheMiB> <contractionCacheMiB>
```

The example supplies all eight native defaults explicitly:

```text
wannier-koopmans-response no yes 0.4 8 80 2048 512 4096
```

| Argument | Default | Meaning and restrictions |
|---|---:|---|
| `nested` | `no` | `no` uses coupled MINRES for screening. `yes` uses fixed-potential inner Sternheimer solves with outer induced-potential mixing. The response tolerance remains the `wannier-koopmans` value. |
| `independentBandCG` | `yes` | In nested mode, solve bands with independently converged CG, falling back to MINRES for nonpositive curvature or a nondiagonal occupied Hamiltonian. `no` uses MINRES inner solves. This setting does not change the default coupled solver when `nested=no`. |
| `mixFraction` | `0.4` | Finite outer mixing fraction, **0 < fraction ≤ 1**. Active in nested mode; smaller values mix a smaller potential update. |
| `mixHistory` | `8` | Positive integer number of previous outer mixing entries retained. Active in nested mode. |
| `mixIterations` | `80` | Positive maximum number of outer mixing iterations. Active in nested mode; separate from the inner response and localization iteration caps. |
| `orbitalCacheMiB` | `2048` | Finite nonnegative per-rank budget in MiB for frozen real-space wavefunctions. `0` disables this cache. Used with response optimization enabled. |
| `symmetryCacheMiB` | `512` | Finite nonnegative budget in MiB for cached source densities on each q-group head. `0` disables retained sources, reducing reuse opportunities. Optimized scheduling releases sources after each q star. The reference path currently uses its fixed 256 MiB source-cache budget. |
| `contractionCacheMiB` | `4096` | Finite nonnegative per-rank budget in MiB for optimized mesh-correlation/conduction-contraction storage, including FFT source/workspace caches or packed-orbital fallback storage. `0` disables the corresponding retained storage. `JDFTX_KI_CORRELATION_CACHE_MIB` can override the FFT source/workspace budget separately. |

One MiB is 1,048,576 bytes. Cache budgets are per rank/group head, not a
single total allocation limit; other wavefunctions, solver vectors and CUDA
storage also consume memory. Insufficient cache space selects fallback
paths or reduces reuse rather than changing the physical equations. Nested
mode can be used without operator optimization. The coupled default was faster than the tested nested alternatives on the
small fixed-orbital valence benchmark; compare the solvers for the intended
material and mesh. `wannier-koopmans-response` does not enable coarse screening: use
the separate-grid launcher for coarse electronic screening, or
`wannier-koopmans-screening-mesh` for adaptive q-only integration.

## Mixed Coulomb features

This source tree implements the fixed-ion SCF method in
`Defect_Only_Coulomb_Truncation_O_Ag111.pdf`. At every electronic density/potential
evaluation it adds (for unembedded truncation)

```
deltaRho = (n + rhoIon) - (nClean + rhoIonClean)
Vcorr = (Kisolated - Kslab) deltaRho
EdefectCoulomb = 0.5 * integral(deltaRho * Vcorr)
```

The clean reference remains fixed. The ordinary slab calculation still treats the
periodic host; the isolated 3D Wigner–Seitz kernel acts on the defect-induced
charge. Exchange-correlation and short-range/nonlocal pseudopotentials keep their
ordinary dependence on the full electronic density. The correction enters the
SCF potential before orbital solution and mixing, and its energy enters the
reported total energy. Both density and potential mixing are supported.

JDFTx's charge convention is positive electronic number density and negative
ionic charge. Both parts must be included: subtracting only the electronic
densities after adding O leaves a charged difference. Core densities used only
for exchange-correlation are excluded. The ionic source is the **bare** charge
that generates the long-range part of `Vlocps`, independently of `ion-width`.
For unembedded truncation, both source and correction potential are projected onto the real FFT grid
with `I`/`J`; this removes unrepresentable point-ion Nyquist components and makes
the energy, real-space potential, and ionic derivative consistent on that grid.

### Mixed Coulomb input reference

These commands run through `jdftx` or `jdftx_gpu`. Required arguments are
shown in angle brackets; square brackets mark optional trailing arguments.
Omitting both commands gives ordinary JDFTx electrostatics.

#### `defect-coulomb`

```text
defect-coulomb <referenceFile> [<chargeTolerance>=1e-5]
```

Enable the variational mixed Coulomb correction during electronic SCF.

| Argument | Default | Meaning and restrictions |
|---|---|---|
| `referenceFile` | Required | Native clean-reference file generated by `dump End DefectReference`. A raw electronic density file is insufficient. Cell, FFT grid, cutoffs, spin, smearing, host pseudopotentials and embedding metadata must be compatible with the current calculation. |
| `chargeTolerance` | `1e-5` | Finite positive absolute tolerance, in electrons, on the integral of the defect-induced total charge. A non-neutral difference aborts; no compensating background is added. |

Requires `coulomb-interaction Slab ...`, `electronic-scf`, `symmetries none`,
a neutral fixed-electron-number vacuum calculation, and fixed ions/cell.
Embedding is optional, with compatible reference settings. The correction
includes both electronic and ionic charge differences. It changes the local
SCF potential and total energy; it is not a postprocessing-only correction.
The [supported scope](#outputs-and-supported-scope) lists incompatible modes.

#### `defect-coulomb-center`

```text
defect-coulomb-center <c0> <c1> <c2>
```

All three coordinates are required and finite. They follow `coords-type`:
Cartesian values are in bohr, and lattice values are fractional direct-lattice
coordinates. Requires both `defect-coulomb` and `coulomb-truncation-embed`.
Omitting this command inherits all three coordinates of the slab embedding
center. An explicit center moves only the isolated defect operator; the slab
operator keeps its own center. Choose the lateral center near the localized
charge difference so it remains away from the truncation cuts.

#### New `dump` variables

| Input | Effect |
|---|---|
| `dump End DefectReference` | Export the clean spin-summed electronic density, bare ionic Coulomb source and compatibility metadata as `PREFIX.defectReference`, using `dump-name PREFIX.$VAR`. Generate it from the converged neutral clean slab. |
| `dump End DefectCoulomb` | With the mixed correction active, export `PREFIX.deltaRho` and `PREFIX.Vcorr`; embedded calculations also export `PREFIX.deltaRhoEffective`. These are diagnostic grids, not replacement SCF checkpoints. |

These are additional variables of the existing `dump` command. Neither is
written unless requested. Their interpretation and embedded energy checks
are explained in the workflow below.

### Clean reference

First relax and converge the clean slab with the lattice, pseudopotentials,
cutoffs, spin treatment, smearing, and FFT grid intended for the defect run.
Use slab truncation, with or without embedding. Preserve the Coulomb settings
that generated the density. In the final clean calculation add:

```text
coulomb-interaction Slab 001
symmetries none
# Keep your lattice, species, ions, k points, cutoffs and smearing here.
dump-name clean.$VAR
dump End State ElecDensity DefectReference
```

`clean.defectReference` contains the converged spin-summed electronic density,
the ionic Coulomb source at the clean geometry, and metadata. There is no need to
reconstruct clean ions in the defect calculation, or to supply an atomic-density
approximation. The stored reference can represent either the relaxed pristine
surface or a separately converged frozen-host geometry.

To export an already converged calculation with saved wavefunctions, use its
original input/settings and ionic positions, plus:

```text
initial-state clean.$VAR
dump-only
dump-name clean.$VAR
dump End DefectReference
```

Remove an explicit `wavefunction` command when using `initial-state`. A raw
`clean.n` file alone is insufficient: it lacks the reference ionic contribution
and the cell/grid metadata. The new reference dump packages the needed inputs.

### Defect SCF

Use the same cell and grid, and add the defect to your ionic input. Keep the
reference and current geometries in the same coordinate origin. For O/Ag(111),
keep Ag pseudopotentials identical to the clean run and add the O species/ion.

```text
coulomb-interaction Slab 001
symmetries none
electronic-scf
defect-coulomb clean.defectReference

dump-name defect.$VAR
dump End State ElecDensity Vscloc Ecomponents Dvac DefectCoulomb
```

The optional second argument of `defect-coulomb` is an absolute neutrality
tolerance in electrons, default `1e-5`. Each density evaluation checks the
integral of `deltaRho`; a failure aborts without silently inserting a charge
background. Each SCF report includes `Qdefect` and `Ecorr`.

At startup the reader checks the file version and payload length, cell, grid,
cutoffs, slab direction, spin treatment, smearing, finite density values,
reference neutrality, and the valence charge and content fingerprint of each
host pseudopotential. Host atoms may move between the separately prepared clean
and defect geometries; the difference includes their charge displacement.
Added defect species are permitted. Use the same k-point sampling and functional
in both runs; these are workflow requirements, not checked by the reference
header. Verify electronic convergence before exporting the reference.

### Outputs and supported scope

`dump End DefectCoulomb` writes `defect.deltaRho` and `defect.Vcorr` as standard
JDFTx real-space binary grids. `Dvac` and `Dtot` include the correction as well.
Use `potential-subtraction no` in both clean and defect runs when comparing the
unsmoothed electrostatic potentials directly. The energy component is named
`EdefectCoulomb`.

This implementation supports neutral, fixed-electron-number, vacuum SCF
calculations with fixed ions and lattice, slab truncation with optional embedding, and
`symmetries none`. It rejects ionic/lattice optimization, molecular dynamics,
stress, vibrations, variational perturbation calculations, fluids, external
charges/fields, and fixed chemical potential. The explicit ionic correction is
included in force evaluation and has a finite-difference regression check;
geometry optimization remains disabled pending surface-system force validation.
Spin-unpolarized, collinear and noncollinear densities are summed over their
diagonal charge components; regression checks cover unpolarized and collinear
cases. The existing exchange kernel is unchanged.

The density difference must be localized within the support allowed by isolated
WS truncation. In the unembedded scheme, charge separations must fit inside the
truncation cell (the usual half-cell confinement requirement). Plot
`deltaRho` and `Vcorr` and check their lateral behavior. The code checks
neutrality, but does not certify localization. See the
[JDFTx Coulomb-interaction documentation](https://jdftx.org/CommandCoulombInteraction.html)
for the truncation confinement requirement.

The regression system is a small neutral He host with a localized He addition;
it validates the implementation, rather than the physical approximation for
O/Ag(111). The PDF's
O/Ag force checks, lateral-size comparisons, and local Wannier-Hamiltonian
localization study remain necessary before interpreting production results.

### Mixed Coulomb validation

Build the utility and validation targets listed in [Build](#build), then run:

```bash
export JDFTX_PSEUDO_DIR=/path/to/jdftx/pseudopotentials
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
ctest --test-dir build -R '^defectCoulomb$' --output-on-failure
```

The integration test has 23 checks: clean recovery, a nonzero converged defect
correction, charge neutrality, real-space energy/potential consistency,
collinear spin recovery, a hexagonal cell with potential mixing and smearing, independence from ionic plotting width, and rejection
of missing, truncated, incompatible, or charged reference files. Like other
JDFTx tests, successful `.out` files are reused; clear the test run's outputs
when rerunning calculations after source changes.

The additional native test checks density energy derivatives, insertion into
the KS potential and total energy, the frozen reference, preservation of the
ionic source, and the explicit ionic derivative through the pseudopotential
force path:

```bash
cd build/test/defectCoulomb
export SRCDIR=../../../jdftx/test/defectCoulomb
../../aux/TestDefectCoulomb -i "$SRCDIR/recovery.in" -o derivatives.out
```

For MPI, set `JDFTX_LAUNCH='mpirun -n 2'` before rerunning the integration test.
GPU builds use the same field operators; enable CUDA and build `jdftx_gpu` and
`TestDefectCoulomb_gpu`. For the RTX 4090s in this workspace, the GPU build is
configured in the shared `build/` with CUDA architecture 89.

The main implementation is in `jdftx/electronic/DefectCoulomb.{h,cpp}`, with input
handling in `jdftx/commands/defect_coulomb.cpp` and integration into `ElecVars`,
`IonInfo`, `SCF`, and `Dump`.

Validation completed in this workspace: all 23 integration checks passed on
CPU (including two MPI processes) and CUDA. Native density, KS-potential, energy,
and ionic finite-difference checks passed on CPU/MPI and CUDA; orthorhombic and
hexagonal cells were checked. Existing `openShell`, `graphene`, and `checkpoint`
regressions passed. These checks preceded the embedded tests described below.

### One-shot potential construction for NSCF

`jdftx/scripts/defectDensityToVscloc` takes the saved clean and defect electronic
densities and constructs the corrected **full local KS potential** once. It uses
JDFTx itself to evaluate slab Hartree, local pseudopotentials, XC (including
partial core corrections), and the defect correction. It preserves JDFTx's
internal `Vscloc` normalization and spin channel naming, so the result can be
read directly with `fix-electron-potential`.

Run from the source-tree root:

```bash
./jdftx/scripts/defectDensityToVscloc \
  --clean-input /path/to/clean.in \
  --defect-input /path/to/defect.in \
  --clean-density /path/to/clean.n \
  --defect-density /path/to/defect.n \
  --output-prefix /path/to/nscf/defect-mixed
```

The two input files supply the lattice, grid, ionic geometries,
pseudopotentials, spin treatment and functional. Their ionic positions must
match the saved densities, including any prior relaxation. The script expands
includes and environment variables through JDFTx's input reader, and runs each
stage from that original input file's directory. It ignores the original
SCF/minimization, wavefunction/restart and dump controls; no orbitals are
allocated or solved, and no density iteration occurs. It retains the original
FFT grid, then disables symmetry averaging of the corrected potential.

For wildcard species such as `SG15/$ID_ONCV_PBE.upf`, the script uses the usual
JDFTx search paths. If `JDFTX_PSEUDO_DIR` is unset, it also looks for the library
beside `jdftx` or `jdftx_gpu` on your `PATH` (including the installed library
under `share/jdftx/pseudopotentials`). Use `--pseudo-dir /path/to/pseudopotentials`
to set the additional search directory explicitly. The selected directory is
printed, and the generated NSCF input records absolute paths to the actual
pseudopotential files used for potential construction.

For collinear spin, supply quoted filename patterns instead:

```bash
--clean-density '/path/to/clean.$VAR' \
--defect-density '/path/to/defect.$VAR'
```

JDFTx reads `n_up` and `n_dn`; noncollinear calculations additionally need `n_re`
and `n_im`. A raw `.n_up` filename is also accepted and converted to the pattern.
The raw files must be standard JDFTx `ElecDensity` dumps on matching grids.
Their byte counts are checked, but raw densities contain no cell metadata:
the supplied input files must actually describe the calculations that produced
them. The clean ionic charge is reconstructed at the clean geometry and stored
in the generated reference, rather than inferred from the defect geometry.

Outputs with this prefix are:

- `.Vscloc` (or `.Vscloc_up`, `.Vscloc_dn`, and noncollinear off-diagonal channels)
- `.Vcorr` and `.deltaRho` for diagnostics (also `.deltaRhoEffective` with embedding)
- `.defectReference`, containing the clean density and reconstructed clean ions
- `.clean.in`/`.defect.in` recipes and `.clean.out`/`.defect.out` logs
- `.nscf.in`, a fixed-potential NSCF input with the exact FFT grid

Run the generated NSCF input from the original defect-input directory. It
already contains:

```text
fix-electron-potential /path/to/nscf/defect-mixed.$VAR
```

Adjust k points and `elec-n-bands` there for the desired NSCF calculation. Keep
the cell, density grid, spin treatment, ionic geometry and pseudopotentials
consistent. Do not add `defect-coulomb` or `electronic-scf` to this NSCF input;
the saved potential already includes the correction. The ordinary nonlocal
pseudopotential operator is provided by the retained species settings.

This path accepts internal LDA/GGA functionals, neutral vacuum slabs
and slab truncation with optional embedding. It rejects hybrids, meta-GGAs, orbital-dependent
functionals and DFT+U, which require data beyond two electronic densities.
The same neutrality and reference compatibility checks apply as in SCF mode.
Use `--charge-tolerance` to change the default `1e-5` electron tolerance, and
`--overwrite` to reuse an output prefix. Source densities are preserved.

Add `--gpu` to use CUDA, or `--launcher 'mpirun -n 2'` to launch the backend with
MPI. `--engine` selects a specific `ConstructDefectPotential` executable. Both
CPU and GPU backends have been built in this workspace; to rebuild:

```bash
cmake --build build --parallel 4 --target \
    ConstructDefectPotential ConstructDefectPotential_gpu
```

The potential is evaluated at the supplied defect density. If that density came
from a conventional periodic calculation, the one-shot correction does **not**
make it self-consistent under mixed electrostatics; use the corrected SCF path
when that electronic response is required.

`ctest --test-dir build -R '^defectPotential$' --output-on-failure` runs the
required density fixture and the end-to-end one-shot regression. It compares
the complete potential with an ordinary fixed-density JDFTx calculation having
the identical `Vcorr` added externally, covers spin and hexagonal/smearing cases,
checks source-density preservation and overwrite protection, and performs an
actual NSCF solve using the generated potential.

The one-shot regression passed on CPU and CUDA in this workspace, including
actual fixed-potential NSCF solves. The two-process MPI script launch was also
validated.

### Embedded mixed Coulomb truncation

Keep `coulomb-truncation-embed` in the clean and defect inputs. Both the normal
embedded slab Hamiltonian and the mixed correction are supported in SCF and in
the one-shot utility. The isolated operator doubles its auxiliary Coulomb grid
in all three directions; the slab operator doubles only its truncated direction.
The wavefunction cell and density grid stay the same.

Version-2 clean references record the embedding mode, center in lattice
coordinates, and ionic margin, and preserve the complete bare ionic Fourier
source. These settings must agree between reference generation and the defect
run. Centers differing by integer lattice vectors are equivalent. Version-1
references remain readable for unembedded calculations; regenerate them for
embedded calculations. Raw density files contain no record of the Coulomb
settings: supply the physical settings that actually generated the densities.
Removing embedding from a reconstruction input changes its underlying potential.

With embedding, let `S` be JDFTx's Gaussian range-separation filter (its width is
taken from the slab Coulomb operator), and `D=Kisolated-Kslab` on smooth sources.
The common ionic short-range kernel cancels between the two operators. The
numerical correction is

```text
deltaN = n - nClean
deltaIon = rhoIonBare - rhoIonBareClean
deltaRhoEffective = deltaN + S deltaIon
Vcorr = D deltaRhoEffective
EdefectCoulomb = 0.5 * integral(deltaRhoEffective * Vcorr)
ionicGradientPotential = S Vcorr
```

This is equivalent to applying the difference of JDFTx's `PointChargeRight`
operators to the ionic difference, and smooth operators to the electron
difference. The ionic derivative uses the adjoint Gaussian filter so the energy
and forces remain consistent. It does not change the local/nonlocal
pseudopotentials or add another ionic short-range contribution. Range separation
is independent of the `ion-width` used for fluid/plotting charges.

`dump End DefectCoulomb` additionally writes `deltaRhoEffective` when embedding
is enabled. Use this field with `Vcorr` for the real-space energy check;
`deltaRho` retains the unsmoothed total difference for charge diagnostics.

The isolated operator inherits all three components of the slab embedding
center. Its lateral coordinates matter: place the isolated center near the
localized defect so the difference does not straddle a cut. You can set a
separate isolated center without changing the slab center:

```text
coords-type Cartesian
coulomb-interaction Slab 001
coulomb-truncation-embed 0 0 18.105968516010567
defect-coulomb clean.defectReference
defect-coulomb-center 2.771410394108804 8.001439334819281 18.105968516010567
```

The example uses Cartesian coordinates in bohr. Lattice coordinates are also
supported. `defect-coulomb-center` requires embedding. The one-shot utility
preserves this center if it is present in the defect input. Use
`--defect-center C0 C1 C2` to override it explicitly; its
coordinates follow the defect input's `coords-type`. The generated NSCF input
retains the slab embedding command; the saved potential already includes the
correction.

The confinement condition now refers to the doubled Coulomb box. Charge must
remain localized away from the cuts of the original cell centered on the
specified embedding center. Check lateral localization of the defect-induced
charge, particularly for a small metallic surface supercell. Embedding does not
establish localization by itself.

The `defectEmbedding` regression covers orthorhombic and hexagonal embedded SCF,
clean reference recovery, nonzero corrections, one-shot reconstruction against
ordinary fixed-density JDFTx, finite-difference density and ionic derivatives,
equivalence to `PointChargeRight`, and rejection of inconsistent embedding
modes, centers, and ionic margins. Build `TestDefectCoulomb` (or its GPU target)
before running:

```bash
ctest --test-dir build -R '^defectEmbedding$' --output-on-failure
```

## Primitive-cell DFPT phonons

`phonon-dfpt Nq0 Nq1 Nq2` computes the Cartesian force matrices on a
Gamma-centered coarse q mesh, Fourier transforms them to lattice space, and
writes matrices for interpolation at arbitrary q. No electronic supercell is
required. For phonon-only runs, the electronic k mesh and phonon q mesh can be chosen independently;
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
weighted k mesh, Coulomb geometry, embedding settings and FFT grid.
Include the same `kpoint-folding` in both inputs: the restart command does not
inherit the electronic mesh. Paths resolve from the calculation working directory. For an
existing full-mesh checkpoint, use `initial-state scf.$VAR` instead. DFPT still
polishes the ground state before solving the responses.

Complete examples are provided for:

| Example | Workflow |
| --- | --- |
| [Silicon phonons](examples/dfpt-silicon/README.md) | Insulating SCF, coarse q mesh, interpolated dispersion, checks and plots |
| [Aluminum phonons](examples/dfpt-aluminum/README.md) | Symmetry-reduced SCF and metallic DFPT with empty bands |
| [BCC niobium input](examples/dfpt-niobium/results_dfpt/02-dfpt.in) | Metallic DFPT with response k-point reduction, nested solves and a standalone BCC dispersion plotter |
| [Graphene phonons and coupling](examples/dfpt-graphene/README.md) | Slab truncation with embedding, metallic DFPT and native Wannier matrix output |
| [Silicon electron–phonon coupling](examples/dfpt-silicon-electron-phonon/README.md) | Fresh SCF including conduction bands, native matrix output and Wannier interpolation |

For example, from the repository root:

```bash
./examples/dfpt-silicon/run.sh --reduced-scf --out examples/dfpt-silicon/results-new
```

The silicon runner accepts `--jdftx-bin ./build/jdftx_gpu`, `--pseudo`,
`--kgrid`, `--qgrid`, `--launcher`, and `--q-groups` overrides. Python 3 and
NumPy are required for interpolation; Matplotlib is needed for plots. Example
parameters are starting values for convergence studies.

### Slab truncation and embedding for 2D materials

Put the Coulomb settings in the shared `common.in` so SCF and DFPT agree:

```text
coulomb-interaction Slab 001
kpoint-folding 12 12 1
# Optional embedding, for a slab centered at z=0:
coulomb-truncation-embed 0 0 0
```

Use `phonon-dfpt 6 6 1` for this example. The q-grid dimension along the
truncated direction must be 1; phonon wavevectors lie in the periodic plane.
All three Cartesian displacement directions are included, so out-of-plane
phonons are computed too. Phonon-only runs can use independent in-plane k/q
meshes; native electron–phonon export requires the electronic mesh to contain
the phonon grid.

Embedding centers use the coordinate system selected by `coords-type`.
Keep the same center, `coulomb-truncation-ion-margin` and FFT grid in SCF and
DFPT, and provide sufficient vacuum for the density and ion margin. The response
uses the slab Bloch Hartree kernel, an analytic 2D Ewald ionic Hessian, and,
with embedding, the doubled slab grid and Gaussian ionic range separation.
External electric fields and fluid embedding are unsupported.

The [graphene example](examples/dfpt-graphene/README.md) provides shared slab
settings, a symmetry-reduced SCF restart and a 6×6×1 phonon/Wannier grid.

### Symmetry, response solvers and parallelism

These response optimizations are enabled by default for phonon calculations,
including electron–phonon export:

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

### DFPT input reference

The following 16 commands are additions to the primitive-cell `jdftx`/
`jdftx_gpu` input interface. `phonon-dfpt` enables the driver; all other
`phonon-dfpt-*` commands require it. Arguments are positional. **Every argument
shown for these DFPT commands is required when that command is present**,
including explicit `yes`/`no` values. Defaults below apply when the optional
command is omitted. Grid dimensions and iteration/history/group counts are
integers. Cache limits are in MiB (1 MiB = 1,048,576 bytes).

#### `phonon-dfpt`

```text
phonon-dfpt <Nq0> <Nq1> <Nq2>
```

All three dimensions are positive integers; their product must fit in a
signed 32-bit integer. They define a Gamma-centered uniform phonon q mesh.
Without this command, the phonon driver is disabled. For slab Coulomb the
dimension along the truncated direction must be `1`; all three Cartesian
displacement directions are still included. The phonon-only electronic k
mesh need not contain the q mesh. Electron–phonon export has the additional
commensurability requirement documented below.

Requires `symmetries none`, nonmagnetic scalar states, periodic or slab
Coulomb (including slab embedding), analytic LDA/GGA and nonrelativistic
norm-conserving pseudopotentials without nonlinear core corrections. Set
`perturb-minimize nIterations` to a positive value and choose its response
tolerance explicitly. Insulators use fully occupied stored bands; metals
use fixed-electron-number Fermi/Gauss smearing and enough empty bands.
See [Metals and supported scope](#metals-and-supported-scope) for restrictions.

Do not combine this driver with `perturb-ion`, `perturb-Vexternal`,
`perturb-rhoExternal`, `perturb-electric-field`, `perturb-wavevector` or
`perturb-incommensurate-wavefunctions`: the driver constructs its own
perturbations and offset states. It writes the force/interpolation files in
[Outputs and interpolation](#outputs-and-interpolation), using `dump-name`.

#### `phonon-dfpt-initial-state`

```text
phonon-dfpt-initial-state <scf-input> <state-pattern>
```

| Argument | Default | Meaning and restrictions |
|---|---|---|
| `scf-input` | Required | Original SCF input describing the geometry, species, pseudopotentials, basis, electronic mesh and symmetries used to generate the checkpoint. |
| `state-pattern` | Required | Saved-state filename pattern containing literal `$VAR`, such as `scf.$VAR`. Reads and unfolds saved wavefunctions and fillings onto the full DFPT mesh. |

Use this command for a symmetry-reduced SCF restart; otherwise the driver
initializes normally or reads a full-mesh checkpoint through `initial-state`.
The target still uses `symmetries none`. Geometry, species, cutoff, bands,
smearing, compatible weighted k mesh, Coulomb/embedding settings and FFT grid
must match. Paths resolve from the calculation working directory. This
command cannot coexist with `initial-state`, `wavefunction` or
`elec-initial-fillings`. The ground state is polished before responses.

#### `phonon-dfpt-symmetries`

```text
phonon-dfpt-symmetries <yes|no>
```

Default `yes`: reduce phonon q points with compatible space-group operations
and time reversal, then reconstruct the full mesh. Compatibility checks cover
the q grid, electronic mesh/weights, FFT sampling and frozen Hamiltonian.
`no` computes all q points and also removes the symmetry operations used for
displacement and response k reduction. The electronic input must still use
`symmetries none`; these are driver-specific response symmetries.

#### `phonon-dfpt-displacement-symmetries`

```text
phonon-dfpt-displacement-symmetries <yes|no>
```

Default `yes`: use the little group of each representative q to reduce atomic
Cartesian displacement solves, including rotations, atom permutations,
cell phases and time reversal. Reconstruct unsolved force-matrix columns and,
when exporting couplings, screened potentials. `no` solves every Cartesian
displacement at each representative q; q reduction remains independently
controlled by `phonon-dfpt-symmetries`.

#### `phonon-dfpt-k-symmetries`

```text
phonon-dfpt-k-symmetries <yes|no>
```

Default `no`. `yes` reduces electronic **response** k points using operations
that preserve q exactly, have no translation, fix the displaced atom without
a cell shift and map its direction to itself or its negative. Response
symmetry averaging and band sewing reconstruct the full response before
contractions. Unsupported operations retain the full response mesh.
Ground-state and offset-state setup still use the full mesh, as do
exported electron–phonon band contractions. Requires available driver
symmetries; `phonon-dfpt-symmetries no` disables this reduction too.

#### `phonon-dfpt-reuse-offset-states`

```text
phonon-dfpt-reuse-offset-states <yes|no>
```

Default `yes`: reuse ground-state wavefunctions at on-mesh k±q points,
including reciprocal wrapping; check Hamiltonian residuals and refine when
needed. Off-mesh states are solved in the frozen potential. `no` disables
on-mesh reuse for a numerical/timing comparison. It does not change the
chosen electronic or phonon mesh.

#### `phonon-dfpt-q-groups`

```text
phonon-dfpt-q-groups <nGroups>
```

Default `1`. `nGroups` is a positive integer no larger than the launched MPI
rank count. Groups solve different representative q points; each group's
ranks distribute electronic k points. Unequal group sizes are supported.
The launcher sets the rank count separately. Each group initializes and
polishes its own ground state, increasing replicated memory/setup work.
Use preferably no more groups than representative q points. All displacements
at a q stay together, and final phonon files are written once.

#### `phonon-dfpt-balance-q-groups`

```text
phonon-dfpt-balance-q-groups <yes|no>
```

Default `yes`: assign representative q points by estimated displacement work,
off-mesh offset-state work, export-orbit size and group capacity. Larger
groups can receive more work. `no` restores round-robin assignment. It has
no distribution benefit for one q group and does not set the group count.

#### `phonon-dfpt-solver`

```text
phonon-dfpt-solver <coupled|nested>
```

Default `coupled`: solve the coupled response using the algorithm selected
by `perturb-minimize` (MINRES by default). `nested` fixes the response
potential during inner Sternheimer solves and mixes the induced potential
in outer iterations. Inner accuracy adapts to the coupled residual, reusing
the prior response through a correction solve. Both paths check the same
final coupled residual and fixed-electron-number Gamma constraint.
Nested-mode speed depends on the material and mesh.

#### `phonon-dfpt-mixing`

```text
phonon-dfpt-mixing <fraction> <history> <iterations>
```

| Argument | Default | Meaning and restrictions |
|---|---:|---|
| `fraction` | `0.4` | Finite outer potential mixing fraction in **(0, 1]**. Smaller fractions take a smaller potential update. |
| `history` | `8` | Positive integer number of previous outer mixing entries retained for the regularized DIIS history. |
| `iterations` | `80` | Positive maximum outer iteration count. Separate from the inner `perturb-minimize nIterations` cap. |

Only active with `phonon-dfpt-solver nested`. Supply all three arguments
when writing this command, even to change only the last one.

#### `phonon-dfpt-cache`

```text
phonon-dfpt-cache <yes|no>
```

Default `yes`: reuse frozen XC data and bare displacement potentials within
one representative q; also retain source vectors within the source-cache
budget. `no` recomputes them for reference comparisons. These caches are
released after each q and do not alter the response equations or tolerance.
This setting is separate from the electron–phonon export cache below.

#### `phonon-dfpt-source-cache-size`

```text
phonon-dfpt-source-cache-size <MiB>
```

Default `128`. Supply a finite nonnegative additional source-vector storage
budget **per MPI rank**. `0` disables retained source vectors while preserving
the potential/XC caches when `phonon-dfpt-cache yes`. `phonon-dfpt-cache no`
disables source caching regardless of this budget. Source cache entries
are released after each representative q; this is not a total memory limit.

#### `phonon-dfpt-optimize`

```text
phonon-dfpt-optimize <yes|no>
```

Default `yes`: use direct complex response operators and fused
bare-plus-induced electron–phonon contractions. It also enables bounded
export-wavefunction caching and the conditional reciprocity shortcut.
`no` selects reference operator/contraction paths, including separate
bare/induced export contractions. Symmetry, the response cache, solver
selection and MPI groups have their own controls; this flag does not
switch them all off or enable electron–phonon export by itself.

#### `phonon-dfpt-electron-phonon`

```text
phonon-dfpt-electron-phonon <yes|no>
```

Default `no`. `yes` exports screened `phononHsub` matrices and `phononKpts`,
plus matching full-electronic-mesh wavefunctions, eigenvalues and fillings,
even with `dump End None`. The electronic mesh must be Gamma centered and
each `kpoint-folding` dimension divisible by the phonon q-grid dimension.
All stored bands enter the exported matrices. To include conduction bands,
use matching Fermi/Gauss smearing and enough bands in SCF and DFPT.

Q/displacement symmetries reconstruct screened potentials; optional response
k reduction remains available. Matrix contractions use full saved electronic
states. Offset reuse, symmetry-reduced SCF restarts and q groups are supported.
File ordering, mass factors and normalization are documented in
[Electron–phonon matrix elements](#electronphonon-matrix-elements).

#### `phonon-dfpt-eph-cache-size`

```text
phonon-dfpt-eph-cache-size <MiB>
```

Default `512`. Supply a finite nonnegative budget **per MPI rank** for
real-space electronic wavefunctions retained during optimized coupling
export. The byte budget must fit in the address space. `0` disables this
cache; it is also inactive with `phonon-dfpt-optimize no` or export disabled.
Storage is released before the next response solve. This is independent of
`phonon-dfpt-source-cache-size` and does not cap all export/solver memory.

#### `phonon-dfpt-eph-reciprocity`

```text
phonon-dfpt-eph-reciprocity <yes|no>
```

Default `yes`: during optimized export, contract one Hermitian k pair and
write its conjugate-transpose partner when q and −q belong to the same
symmetry orbit. Otherwise the code falls back to full contractions.
`no` explicitly contracts both partners for reference comparisons.
Requires export and `phonon-dfpt-optimize yes` to have an effect.

#### Response convergence: `perturb-minimize`

This existing command supplies essential convergence settings for the new
DFPT driver. Its named key/value pairs may appear in any order:

```text
perturb-minimize nIterations 500 algorithm MINRES residualTol 1e-7 residualDiffThreshold 0
```

| Key | Native default | Meaning |
|---|---|---|
| `nIterations` | `0` | Nonnegative maximum solve iterations; **must be positive for DFPT**. In nested mode, caps the inner solves. |
| `algorithm` | `MINRES` | `MINRES` or `CG` for the coupled linear solver. Nested inner band CG can fall back to MINRES. |
| `residualTol` | `1e-4` | Positive relative residual tolerance; the example requests `1e-7`. |
| `residualDiffThreshold` | `1e-4` | Nonnegative successive-residual stopping threshold for MINRES. Set `0` for response calculations that should converge by residual magnitude rather than stagnation. |
| `CGBypass` | `no` | Bypass CG's stop after three consecutive residual increases. Applies to CG. |
| `recomputeResidual` | `no` | Recompute the residual every ten CG iterations. Applies to CG. |

KI sets its solver caps/tolerance through `wannier-koopmans` instead.
For generic finite-q perturbations outside `phonon-dfpt`, the existing
`perturb-incommensurate-wavefunctions <filename> [<EcutOld>]` is now optional:
offset states are generated automatically when the command is absent.
`filename` supplies native k+q/k−q occupied wavefunctions; positive `EcutOld`
(in Hartree) enables old-cutoff conversion, while omission/`0` disables it.
The phonon driver forbids this manual-offset command.

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

The primitive-cell path supports periodic or slab Coulomb, including slab
embedding, with nonmagnetic scalar states, analytic LDA/GGA kernels, and nonrelativistic norm-conserving pseudopotentials
without nonlinear core corrections, such as suitable SG15 potentials. Ultrasoft
potentials, nonlinear core corrections, spin polarization/spinors, exact exchange,
DFT+U, meta-GGAs, fluids, vdW pair corrections, fixed chemical potential, MP1 and
Cold smearing are unsupported. Ionic/lattice optimization must be performed in
a separate calculation.

DFPT cannot be combined with the defect-only mixed slab correction described
above. Ordinary slab truncation and its Coulomb embedding are supported. Born
effective charges and long-range polar corrections are not implemented, so ordinary Fourier
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

The export uses exactly the `phononHsub` format consumed by the existing
JDFTx Wannier executable. It includes ionic local and nonlocal pseudopotential
derivatives and the screened Hartree/XC potential. Periodic and slab Coulomb,
including slab embedding, are supported.

The electronic mesh must be Gamma centered and contain the phonon q grid:
each `kpoint-folding` dimension must be divisible by the corresponding
`phonon-dfpt` dimension. For graphene, `kpoint-folding 12 12 1` and
`phonon-dfpt 6 6 1` satisfy this requirement. The file stores only the 36
electronic k points commensurate with that phonon supercell; the response still
integrates over the full 144-point electronic mesh.

All stored electronic bands are included. To include conduction states, prepare
a fresh SCF checkpoint with Fermi/Gauss smearing and additional bands, keeping
those settings identical in DFPT. Electron–phonon export uses q and displacement
symmetry reduction to reconstruct the full set of screened potentials. Optional
response k-point reduction also remains available; band contractions use the
full saved electronic states. Offset reuse, symmetry-reduced SCF unfolding and
MPI q groups remain available. Q groups synchronize electronic phases and
degenerate-band rotations, then write disjoint blocks of one matrix file.
Optimized export can cache real-space wavefunctions and use Hermitian-pair
reciprocity where compatible; see the [DFPT input reference](#dfpt-input-reference)
for the separate optimization, cache and reciprocity controls.

With `dump-name phonon.$VAR`, the additional output is:

| Output | Contents |
| --- | --- |
| `phonon.phononHsub` | Native complex128 matrices: Cartesian mode, target k, source k, then column-major band matrix; rows are final bands, columns initial bands |
| `phonon.phononKpts` | Ordered electronic k subset commensurate with the phonon grid, in centered reciprocal coordinates |
| `phonon.wfns`, `phonon.eigenvals`, `phonon.fillings` | Matching eigenvectors, band energies and occupations for the full electronic mesh, saved even with `dump End None` |

The existing `phonon.phononBasis` provides mass-weighted Cartesian directions
and atomic masses. For mode a, the matrix file contains
`<psi_m,k_target | dH_SCF/du_a(q) | psi_n,k_source> / (Nq * sqrt(M_a))`,
where q = k_target − k_source, the unit displacement is in bohrs and M is in
electron masses. This includes the native mass factor and the 1/Nq supercell
normalization of a displacement localized in one cell. Occupations, electronic
integration weights and phonon zero-point factors are not included. Matrix
storage is `16 * (3 * Natoms) * Nq² * Nbands²` bytes. The former `phononEph`
manifest, per-q files and their converter have been retired.

### Wannier interpolation

Use the same cell, ions, pseudopotentials, cutoff, electronic mesh and band count
in the Wannier input, together with the trial orbitals and band windows for your
material. For the graphene grid above, include:

```text
symmetries none
wannier phononSupercell 6 6 1
wannier-initial-state phonon.$VAR
wannier-dump-name wannier.$VAR
```

`phononSupercell` must match `phonon-dfpt`. Use the **DFPT output prefix** for
`wannier-initial-state`: its saved eigenvectors have exactly the electronic gauge
used in `phononHsub`. The input SCF checkpoint may have a different gauge.
Run the existing Wannier executable from the calculation directory:

```bash
/path/to/build/wannier -i wannier.in -o wannier.out
# Or /path/to/build/wannier_gpu for CUDA.
```

Wannier reads `phononHsub` and `phononBasis`, applies its usual translational
correction and band rotations, and writes `wannier.mlwfHePh` with the phonon
cell maps and weights needed for real-space interpolation. No file-format
conversion is required. Mode-resolved scattering couplings require contraction
with phonon eigenvectors and the zero-point factor after interpolation.

The [silicon coupling example](examples/dfpt-silicon-electron-phonon/README.md)
includes a complete valence-bond Wannier input. See the
[DFPT method and format guide](jdftx/doc/tutorials/phonon/DFPT.dox) for phase
conventions, normalization and supported features.

## Berry-phase bulk polarization and static dielectric response

`berry-phase-field Ex Ey Ez` minimizes the electric enthalpy
`F = E_DFT - Omega E . (P_electronic + P_ionic)` self-consistently. The Cartesian
field is the total macroscopic internal field in Hartree atomic units, with
the physical positive-ion sign convention. For example, add the following to
an insulating bulk input containing only fully occupied bands:

```text
berry-phase-field 0 0 0.0001
electronic-minimize nIterations 1000 energyDiffThreshold 0 knormThreshold 1e-9
```

The occupied-state Berry overlaps couple neighboring k points while every
minimization step updates the density, Hartree potential and XC potential.
No empty-band sum or Wannier construction is required. Wannier functions can
be computed independently from the converged occupied wavefunctions.
The minimized energy includes `EfieldBerry`, and each run writes a
`.polarization` file with electronic, ionic and total polarization, internal
energy, enthalpy and individual string phases.

Use a neutral insulating cell, norm-conserving pseudopotentials, periodic
Coulomb interaction, a uniform `kpoint-folding` mesh and constant full
occupations. The command disables spatial and k-inversion reduction.
At nonzero field use `electronic-minimize`; `electronic-scf`, fixed-Hamiltonian
runs, `converge-empty-states`, finite-field DFPT, fluid, stress and lattice
relaxation are unsupported. Internal-ion relaxation through `ionic-minimize`
is supported at fixed lattice vectors. Verify the insulating gap separately
and converge field amplitude and k mesh together to follow a stable branch.

Enable `berry-phase-field 0 0 0` for the full-mesh zero-field reference, then
restart both field signs from its occupied wavefunctions. Centered polarization
differences give `epsilon_ab = delta_ab + 4*pi*dP_a/dE_b`; relaxing internal
ions at each field includes the lattice contribution to the static response.
The [silicon workflow](examples/berry-dielectric-silicon/README.md) prepares,
runs and analyzes clamped-ion and relaxed-ion dielectric tensors.
The [validation report](examples/berry-dielectric-silicon/VALIDATION.md) shows
silicon's response approaching the experimental value near 12 as the mesh
is refined, with remaining mesh error. The
[PRL audit](examples/berry-dielectric-silicon/PRL_AUDIT.md) documents the
variational derivatives, polarization branches and an independent Quantum
ESPRESSO comparison agreeing to about 5 parts per million on the same mesh.

Every Berry-phase run, including zero-field references and dry runs, prints
the method citation in JDFTx's used-feature citation report:
I. Souza, J. Iniguez and D. Vanderbilt,
*First-Principles Approach to Insulators in Finite Electric Fields*,
Phys. Rev. Lett. **89**, 117602 (2002),
[doi:10.1103/PhysRevLett.89.117602](https://doi.org/10.1103/PhysRevLett.89.117602).

## Tests

With the targets above built and the test pseudopotentials available:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir build -R '^defect(Coulomb|Potential|Embedding)$' --output-on-failure
```

Build products and calculation output are not part of the source repository.

Run the Berry-phase regressions with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir build -R berryPhase --output-on-failure
```

Coverage includes variational orbital derivatives, occupied-gauge invariance,
reciprocal boundary mapping, polarization branches, ionic forces, spin factors,
shifted and noncubic meshes, CPU/CUDA and MPI, and the silicon finite-field
workflow with polarization/enthalpy consistency and method-citation checks.

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
`dfptResponseSlab` / `dfptResponseSlabLDA` and
`dfptResponseEmbeddedSlab` / `dfptResponseEmbeddedSlabLDA` compare in-plane and
out-of-plane response forces against independent supercell finite differences,
including fractional embedding centers and symmetry-reduced SCF restarts.
Electron–phonon tests check native file dimensions and centered k ordering,
mass/supercell normalization, screened band derivatives against independent
SCF finite differences, full complex matrices against native frozen phonons,
and an actual Wannier export. `dfptElectronPhononEmbeddedSlab` covers embedding;
`dfptElectronPhononMPI` checks complex reciprocity across q groups and compares
the resulting Wannier couplings. Incompatible native-export meshes are rejected.
Phonon-only off-mesh q calculations remain supported. Set `JDFTX_BINARY` and
`WANNIER_BINARY` to the CUDA executable paths when running
`jdftx/test/dfpt/checkElectronPhonon.py` against the CPU build/pseudopotential
arguments to check GPU DFPT and Wannier output too.

CPU and CUDA response checks and MPI q-group checks have passed in this
workspace. The example inputs still require physical convergence studies for
production use. Detailed test definitions are in
[the test configuration](jdftx/test/CMakeLists.txt).

## Upstream and license

This repository preserves upstream history, based on JDFTx commit `52548a96`.
JDFTx and these modifications are distributed under the GNU General Public
License, version 3 or later; see [COPYING](jdftx/COPYING) and the source notices.
Retain JDFTx attribution and cite the software and methods used in calculations.
