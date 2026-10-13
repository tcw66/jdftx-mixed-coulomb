# Silicon Koopmans calculations

For the complete syntax, defaults, units and restrictions, see the
[native KI input reference](#native-ki-input-reference),
[Wannier localization settings](#wannier-and-localization-settings),
[script options](#script-options) and
[environment variables](#environment-variables) below. Shell examples assume
this directory is the working directory; the Python scripts' default paths
are anchored to their own location.

The inputs use a spin-unpolarized PBE ground state (`spintype no-spin`), an
8×8×8 mesh, 60 stored bands and a 30 Ha cutoff. KI treats each orbital source
as one electron and includes charge and magnetic response about that
nonmagnetic ground state (`wannier-koopmans-spin yes`).

The native spectral correction is one-shot **KI(2)**, with a second-order
energy expansion and infinitesimal-occupation DFPT screening. Literature
comparisons must match that approximation, the base functional, electrostatic
image correction and Wannier manifolds. Full KI calculations use finite
occupation changes and the full nonlinear correction. The published silicon
validation in [Colonna et al. (2022)](https://doi.org/10.1021/acs.jctc.2c00161)
uses four antibonding conduction Wanniers and LDA; these inputs use six
conduction Wanniers and PBE. See the [implementation audit](implementation-audit.md)
for the checked equations, kernel correction and comparison with saved results.
The [direct QE comparison](qe-implementation-comparison.md) documents the
official KCW/KCP source review, numerical tests and reduced-mesh Ewald fix.

For fresh reruns using literature base functionals, matching pseudopotentials,
four conduction Wanniers and an ab initio dielectric correction, use the
[LDA/PBE literature launchers](literature/README.md). That directory also
contains both a literature-oriented full-KI@PBE preset and the exact downloaded
official silicon tutorial input with its own runner.

## Coarse electronic-grid screening

After completing the fine SCF and both Wannier localizations, run:

```bash
python3 run_coarse_screening.py \
  --example-dir /path/to/completed-silicon \
  --coarse-grid 4 4 4 --cpu --ranks 2 --threads 2 \
  --directory /path/to/coarse-4-run --plot
```

This retains the fine SCF density, computes coarse NSCF states and localized
orbitals, solves screening on the full coarse k/q mesh, then imports matched
alphas into the fine-grid Hamiltonian. Source states are preserved. The fine
mesh comes from `common.in`; no divisibility relation is required.

Use a new output directory, or add `--resume` to continue an existing run
with hash-validated reuse. `--prepare-only` writes inputs for review. Omit
`--cpu` for CUDA. NumPy/SciPy are required; plots also require Matplotlib.
Epsilon is inherited from the KI input; `--epsilon-infinity VALUE` overrides
it consistently at both grids. A nonzero value must be computed for the same
functional and pseudopotentials.

Screening data are in `coarse/`; final Hamiltonians, alpha mappings and plots
are in `fine/`; stage logs are in `logs/`. Failed localization or ambiguous
orbital correspondence stops the transfer. See the
[complete separate-grid workflow](official-coarse-workflow.md) for prerequisites,
options, native commands and validation. The native q-only
`wannier-koopmans-screening-mesh` command is a separate approximation.

KI output is concise by default. A source-only symmetry planning pass first
reports the global number of independent q points, orbital–q pairs and
charge/magnetic sector solves. It uses the actual Wannier functions, screening
mesh, tolerance and cache budget. The current 8×8×8 conduction setup at tolerance
`1e-4` needs **65 independent q points**, 300 orbital–q pairs and 600 sector
solves; this count can change with the orbitals and settings.

Each independent response reports its q group, `computed q i/N` counter,
actual mesh index/vector, orbital and charge/magnetic sector before the solve
starts. The denominator is the group's assigned independent-q total, and the
counter advances only when a new q needs an independent response. Other
orbitals and spin sectors at the same q retain its counter. After each completed
charge or magnetic response, a matching line prints its wall-clock solve
duration in seconds, for example `charge response completed in 123.45 s.`.
These messages remain visible with `verbose no`, in the main or corresponding
q-group log. q vectors use fractional reciprocal-lattice coordinates.
Equivalent pairs that reuse a result stay quiet, and source-only audits do not
print solve timings. Each group's final computed-q count is also printed.
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

Full 8×8×8 screening remains the default. A screening-mesh command now
selects the **initial** q mesh for adaptive refinement. Completed responses
are reused as the mesh grows. A coarse result needs two successive changes
in the KI Hamiltonian below 0.005 eV; otherwise refinement reaches the full
mesh. The electronic k integration, orbital sources and bare Hamiltonian
retain the full mesh throughout. The tolerance controls changes between
meshes, not a rigorous bound on the remaining integration error. The output
records the requested mesh, final mesh and every refinement stage. A fourth
argument of `0` explicitly requests the legacy fixed-mesh diagnostic.

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

The six conduction functions form the triplets `{1,4,5}` and `{2,3,6}`
(one-based labels). Their regenerated sources were approximately symmetric,
but errors of 2×10⁻⁶ to 1.2×10⁻⁵ at Gamma exceeded the original 10⁻⁶ reuse
cutoff. An overlap extremely close to one can still correspond to an
amplitude error larger than that cutoff. The previous localization stopped
on spread-energy convergence at a gradient of 1.8×10⁻⁵. The conduction and
seed inputs now require both gradient and spread convergence, with a tighter
spread-energy threshold. Native minimization can still stop at its numerical
roundoff limit; check the subsequent symmetry audit rather than treating
that stop as proof of exact symmetry.

Audit the saved functions, or refine a copy before auditing:

```sh
python3 analyze_symmetry.py
python3 analyze_symmetry.py --refine --symmetry-tolerance 1e-6
```

The analyzer copies rotations, overlaps and restart metadata into a new
archive directory and preserves production checkpoints. It uses the native
finite-q source and curvature checks, performs zero response solves, and
writes `symmetry-report.json` with Gamma source groups, all finite-q reuse
mappings, accepted errors and the predicted independent orbital–q workload.
Gamma groups alone do not establish equivalence at every q; each finite-q
mapping is independently tested. `--screening-mesh 1 1 1` limits the audit
to Gamma, and `--diagnostics` prints nearest rejected candidates. The default
backend is CPU; `--gpu` selects CUDA. `--input`, `--prefix`, `--threads` and
`--output` select the input, saved run and resources. The report records the
actual symmetry tolerance, including overrides supplied for that audit.

On an isolated copy of the current six-orbital state, tighter localization
restored both Gamma triplets at 10⁻⁶ and reduced the full 8³ audit to 438
independent pairs out of 3,072, versus approximately 1,560 before refinement.
The original checkpoints audited at 10⁻⁴ required 300 independent pairs.
These are workload counts; validate spectra and screening coefficients when
choosing a looser symmetry tolerance.

The refined-state reduced/exhaustive comparison on the full 8³ electronic
mesh with Gamma-only screening required two versus six independent pairs.
The maximum alpha difference was 1.1×10⁻⁷ and the maximum corrected mesh
eigenvalue difference was 3×10⁻⁷ eV. This validates the strict reuse test for
that diagnostic; Gamma-only screening does not establish a converged gap.

## Native KI input reference

All commands below belong in a Wannier input. `yes` and `no` are literal
boolean values. Arguments to `wannier-koopmans` and
`wannier-koopmans-response` are **positional**, in the order shown. Omitting
trailing arguments uses native defaults; to change a later argument, supply
all earlier arguments. The angle brackets below name arguments and are not
part of a runnable input. The ancillary `wannier-koopmans-*` commands require
`wannier-koopmans` to be present.

The supported ground state is a nonmagnetic insulator (`spintype no-spin`),
with norm-conserving, nonrelativistic pseudopotentials, an analytic LDA/GGA
XC kernel and a Gamma-centered uniform electronic mesh. SCF symmetry reduction
is supported: KI reconstructs the full electronic mesh. Occupied and empty
Wannier manifolds are localized separately in this example. KI changes the
spectral Hamiltonian; the underlying DFT total energy is retained.

### `wannier-koopmans`

```text
wannier-koopmans <epsilonInfinity> <nIterations> <residualTol> <reduceSymmetry> <symmetryTol> <occupiedResponse> <qGroups> <optimizeResponse>
```

The current example line in `ki-response.in` is:

```text
wannier-koopmans 0 400 1e-7 yes 1e-4 yes 2 yes
```

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

### `wannier-koopmans-spin`

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

### `wannier-koopmans-verbose`

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

### `wannier-koopmans-screening-mesh`

```text
wannier-koopmans-screening-mesh <nq0> <nq1> <nq2> [<energyTolerance_eV>=0.005]
```

Each positive dimension must divide the corresponding electronic
`kpoint-folding` dimension. Omit this command for full-mesh screening.
For example, on an 8×8×8 electronic mesh:

```text
wannier-koopmans-screening-mesh 2 2 2 0.005
```

This starts with 2×2×2, refines to 4×4×4 and, if needed, 8×8×8. Previously
computed responses are retained; sources and response equations stay the
same. Other mesh sizes grow along nested divisors of the electronic mesh.
At least two successive Hamiltonian changes must pass the tolerance before
accepting a coarse result. Negative or nonfinite coarse coefficients cause
continued refinement. A full-mesh negative coefficient is reported without
clipping and still needs electrostatic/model convergence checks.

The change is measured with the maximum Frobenius norm of the difference in
the Hermitian KI correction over all native k points. This bounds the change
in each native-k eigenvalue between the two meshes. It does not bound the
remaining mesh error or errors at arbitrary interpolated k points. Separate
valence/conduction calculations each use the specified tolerance. No speedup
is guaranteed if screening requires the full mesh.

With Ewald correction enabled, each screening stage uses its own supercell
image term; the spectral Hamiltonian retains the full-mesh image term.
No dielectric constant is inferred or fitted by refinement. `epsilon=0`
keeps the neutralizing-background convention at every stage.

For an explicit fixed-mesh diagnostic, use a zero tolerance:

```text
wannier-koopmans-screening-mesh 2 2 2 0
```

Source-only symmetry audits also retain the requested mesh. The solver
benchmark's `--screening-mesh` option selects a fixed diagnostic subset so
that the compared solvers run the same problem. The optional command in
`ki-screening-mesh.in` remains commented out; editing an input affects the
next native launch. See [adaptive-screening.md](adaptive-screening.md) for
validation results and limitations.

### `wannier-koopmans-response`

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
| `contractionCacheMiB` | `4096` | Finite nonnegative per-rank budget in MiB for optimized mesh-correlation/conduction-contraction storage, including FFT source/workspace caches or packed-orbital fallback storage. `0` disables the corresponding retained storage. A separate FFT-correlation environment override is documented below. |

One MiB is 1,048,576 bytes. Cache budgets are per rank/group head, not a
single total allocation limit; other wavefunctions, solver vectors and CUDA
storage also consume memory. Insufficient cache space selects fallback
paths or reduces reuse rather than changing the physical equations. Nested
mode can be used without operator optimization. The present coupled default
was faster than the tested nested alternatives on the small fixed-orbital
valence benchmark; that timing is not a universal solver ranking.

## Wannier and localization settings

These existing JDFTx controls are important to the revised KI workflow:

| Setting | Meaning in these inputs |
|---|---|
| `spintype no-spin` | Spin-unpolarized PBE ground state. Single-spin KI screening is configured independently with `wannier-koopmans-spin`. |
| `kpoint-folding 8 8 8` | Full electronic mesh in `common.in`. A screening-mesh override does not change it. Changing the electronic model/mesh requires a matching SCF checkpoint. |
| `wannier-initial-state state/scf.$VAR` | Read the saved SCF data. `$VAR` is JDFTx's filename substitution, not a shell variable to expand manually. |
| `wannier-dump-name PREFIX.$VAR` | Write Wannier data with the chosen prefix. Valence/conduction output prefixes are separate to prevent mixing occupied and empty manifolds. |
| `bStart 0` | Zero-based first band of the fixed valence manifold. Four `wannier-center` lines select four bands. Native default is 0; an `outerWindow` selection supersedes it. |
| `innerWindow eMin eMax` | Frozen energy window: bands inside it are retained exactly in the disentangled subspace. Requires `outerWindow` and must lie inside it. Conduction uses `0.239616 0.5`. |
| `outerWindow eMin eMax` | Candidate bands allowed to contribute to the disentangled Wannier space. Conduction uses `0.239616 1.5`. Both windows use **absolute native eigenvalue energies in Hartree**, not energies shifted to the plotted VBM. The leading `+` in the seed input simply denotes a positive number. |
| `loadRotations yes` | Resume saved rotations when compatible. Default is `no`; the seed input explicitly uses `no`, while the separate KI stages use `yes`. When resuming, saved rotations take precedence over new trial projections. SCF-content fingerprints invalidate stale rotations and overlaps. |
| `saveWfns yes/no` | Export numerical supercell plane-wave functions and their header as `mlwfC*` when `yes`. Default is `no`. The seed stage needs `yes`; KI spectra do not require this export. |
| `localizationMeasure FiniteDifference` | Default reciprocal-space finite-difference spread functional, used here. `RealSpace` is the alternative existing JDFTx measure; its spread estimator and numerical behavior differ. |
| `numericalOrbitals state/conduction-seed.mlwfC` | Load numerical trial functions together with the adjacent `.header` file. The conduction stage uses the six exported empty-orbital seeds. |
| `wannier-center Gaussian x y z` | Supply a Gaussian trial projection at the specified coordinates, interpreted using `coords-type` (`lattice` here). It seeds localization; it does not force the final Wannier center to remain at that position. |
| `wannier-center Numerical i` | Choose zero-based numerical trial column `i`. The six lines use indices 0–5; symmetry reports label functions 1–6. |

The revised conduction and seed localization controls are:

```text
wannier-minimize nIterations 600 knormThreshold 1e-8 \
    energyDiffThreshold 1e-14 convergeAll yes
```

| Flag | Effect |
|---|---|
| `nIterations 600` | Cap the localization steps at 600. This does not set the KI response iteration count. `0` freezes compatible saved rotations; a stale SCF restart is rejected rather than silently frozen. |
| `knormThreshold 1e-8` | Requested threshold on the native preconditioned localization gradient norm, printed as `\|grad\|_K`. |
| `energyDiffThreshold 1e-14` | Requested change threshold on the spread functional, not on electronic energy. For this finite-difference calculation the functional is expressed in bohr². |
| `convergeAll yes` | Require both gradient and successive spread-change criteria. The former default `no` allowed the spread criterion alone to stop the run while source asymmetry remained too large for reuse. |

Line-search roundoff or the iteration cap can stop minimization before both
criteria pass. Inspect the gradient, stop reason and source audit. The
analyzer reports the final gradient and whether its localization stage hit
roundoff. The valence input retains its existing 300-step, `1e-8` gradient
and `1e-8` spread-change settings; the stricter changes apply to conduction
and conduction-seed localization.

## Script options

### Symmetry analyzer: `analyze_symmetry.py`

The analyzer runs one native process, forces one q group and enables native
symmetry reduction and verbose output in its copied input. It forces
`saveWfns no`, resumes copied rotations and enables response-free audit mode.
It keeps the input's spin treatment and solver settings. Source checkpoints
are hashed before and after the audit. NumPy is not required by this wrapper;
the native executable and pseudopotentials must be available.

| Flag | Default | Effect |
|---|---|---|
| `--input FILE` | `03-conduction.in` beside the script | Wannier input to expand, including nested `include` files. Must contain one `wannier-minimize` and one `wannier-koopmans` command. |
| `--prefix PREFIX` | `results/conduction` beside the script | Saved Wannier dump prefix to copy. Requires `.mlwfU` and `.mlwfM0`; copies `.mlwfState` if present. Must match the input model and SCF state. |
| `--refine` | Off | Refine only the copied rotations with 600 iterations, gradient `1e-8`, spread change `1e-14` and `convergeAll yes`, then audit. Without it, use zero localization iterations. Refined copies are not installed into production. |
| `--symmetry-tolerance T` | Inherit the input | Override source/curvature acceptance tolerance for this audit only; finite **0 < T ≤ 0.01**. Does not edit `ki-response.in` or change response convergence tolerance. |
| `--screening-mesh NX NY NZ` | Inherit the input; full electronic mesh if omitted there | Override the screening subset for this audit only. Positive dimensions must divide the electronic mesh. `1 1 1` checks Gamma only. |
| `--diagnostics` | Off | Also evaluate and print nearest rejected cached-source candidates, including source, norm and curvature mismatches. Can increase audit cost; acceptance checks remain in force. |
| `--gpu` | Off (CPU) | Select `build/wannier_gpu` instead of `build/wannier`. `WANNIER_BINARY` overrides this default selection. |
| `--threads N` | `8` | Positive native thread count, also exported as `OMP_NUM_THREADS`. The wrapper sets OpenBLAS threads to 1. |
| `--output DIRECTORY` | New timestamped `archive/symmetry-audit-*` directory | Destination for copied checkpoints, `audit.in`, `audit.out`, `launcher.log` and `symmetry-report.json`. The directory must not already exist. |
| `--help` / `-h` | — | Print usage and exit without auditing. |

`Gamma_screening_source_classes` certifies source equivalence at Gamma at the
reported tolerance. `finite_q_mappings` separately records the decision for
every selected orbital–q pair; a Gamma group does not force reuse elsewhere.
`independent_orbital_q_pairs` predicts combined charge/magnetic response pairs,
while `computed_q_points` counts distinct q points needing at least one
independent orbital response. With single-spin screening, each independent
pair normally requires two sector solves. `response_solves_performed` is zero
for the analyzer. Native `KI symmetry audit pairs` counts planned work, not
completed response solves.

For a valence audit, select matching input and checkpoint prefixes together:

```sh
python3 analyze_symmetry.py --input 02-valence.in --prefix results/valence
python3 analyze_symmetry.py --refine --symmetry-tolerance 1e-6 --screening-mesh 4 4 4
```

### Workflow launchers: `run.sh` and `run_wannier.sh`

| Flag | Effect |
|---|---|
| No flags | Reuse matching SCF/seeds when available; run separate KI stages, direct PBE path cross-check, and the KI/PBE plotter. GPU backend is the default. |
| `--dry-run` | Validate selected native inputs. Redirect Wannier dump prefixes to temporary storage so checkpoints remain unchanged; write `logs/*.dry-run.out`. |
| `--from-scf` | Archive current state/results/logs and begin a fresh SCF calculation, followed by the complete workflow. |
| `--ki-only` | Require a matching saved SCF and run only the KI Wannier stages and any needed seed stage. Skip SCF, direct PBE path calculation and plotting. |
| `--cpu` | Select CPU executables; explicit binary environment overrides still take precedence. |
| `--help` / `-h` | Print launcher usage and exit. |

`--from-scf` and `--ki-only` are mutually exclusive. `run_wannier.sh` invokes
`run.sh --ki-only` and forwards its flags. The launcher refuses to start over
another native calculation in the same working directory.

### Band plotter: `plot_separate_bands.py`

| Flag | Default | Effect |
|---|---|---|
| `--valence PREFIX` | `results/valence` | Completed valence KI dump prefix. |
| `--conduction PREFIX` | `results/conduction` | Completed conduction KI dump prefix; must use the same spin treatment as valence. |
| `--fold NX NY NZ` | Read `common.in` | Electronic mesh used to decode the Hamiltonian. Must match the number of cells in `koopmansFormat`; this is not the screening mesh. |
| `--gap-mesh NX NY NZ` | `16 16 16` | Positive dimensions of the full reciprocal-cell search grid for band extrema. The plotted path is also included as a set of starting samples. |
| `--no-refine` | Off | Skip local optimization of extrema; report sampled mesh/path extrema only. Otherwise use multiple mesh/path starting points for local refinement. |
| `--path FILE` | `paths/path.kpoints` | Three fractional reciprocal coordinates per plotted point. |
| `--distance FILE` | `paths/path-distance.dat` | Matching nondecreasing scalar path distances. |
| `--ticks FILE` | `paths/path-ticks.dat` | Matching symmetry labels, point indices, distances and coordinates. |
| `--output-dir DIRECTORY` | `results/` | Write comparison PNG/PDF, energy table and JSON report here; existing comparison files can be overwritten. |
| `--help` / `-h` | — | Print usage and exit. |

Relative paths explicitly supplied on the command line resolve from the
shell's working directory. Default paths above are relative to the example.
The gap-search grid samples the interpolated Hamiltonian; increasing it
does not improve the underlying Wannier model or screening quadrature.

### Solver benchmark: `benchmark_valence.py`

| Flag | Default | Effect |
|---|---|---|
| `--example-dir DIRECTORY` | This example | Source inputs and matching saved SCF/valence checkpoints. |
| `--cpu` | Off (GPU) | Use CPU Wannier; `WANNIER_BINARY` overrides backend defaults. |
| `--ranks N` | `2` | Positive MPI rank count; one rank runs directly. The benchmark uses up to two q groups. |
| `--threads N` | `4` | Positive native threads per rank; OpenBLAS threads are fixed to 1. |
| `--screening-mesh NX NY NZ` | Full electronic mesh | Fixed diagnostic screening subset for each compared solver (native energy tolerance 0). |
| `--output DIRECTORY` | New timestamped archive directory | New benchmark directory; must not already exist. Holds copied/refined checkpoints, inputs, logs and `timings.json`. |
| `--help` / `-h` | — | Print usage and exit. |

The benchmark compares coupled MINRES, nested independent-band CG and nested
MINRES at identical frozen rotations, checks agreement, and records timings.

## Environment variables

### Launcher and tool configuration

| Variable | Default / scope | Effect |
|---|---|---|
| `JDFTX_BINARY` | `build/jdftx_gpu`, or `build/jdftx` with `--cpu` | Override the workflow's electronic executable. |
| `WANNIER_BINARY` | Backend-selected executable in `build/` | Override Wannier for the workflow, symmetry analyzer and benchmark. |
| `MPIEXEC` | `mpirun` | MPI launcher executable for the workflow/benchmark. Use an executable name/path, not a shell string containing extra flags. The symmetry analyzer uses one process directly. |
| `MPI_RANKS` | `2`, workflow only | Number of launched ranks. The workflow currently requires at least 2 for the configured two q groups. Benchmark ranks use `--ranks`. |
| `WANNIER_THREADS` | `4`, workflow only | Threads per rank for its electronic and Wannier stages. Sets native `-c` and `OMP_NUM_THREADS`; other tools have `--threads`. |
| `JDFTX_PSEUDO_DIR` | `build/pseudopotentials` in workflow/analyzer | Pseudopotential-library search directory. An explicit setting takes precedence. |
| `PYTHON` | `python3`, workflow only | Python executable used for the final plotter. |
| `OPENBLAS_NUM_THREADS` | Forced to `1` by these launchers/tools | Keep small BLAS work single threaded and avoid multiplying BLAS threads by native threads/MPI ranks. |

### Native audit and optimization controls

These are environment variables, not lines to put in `.in` files. They act
when a new process starts. Diagnostic/reference switches preserve the intended
physical equations; approximate symmetry acceptance and coarse screening
remain controlled by the native input arguments documented above.

| Variable | Default | Effect / conditions |
|---|---|---|
| `JDFTX_KI_SYMMETRY_AUDIT` | Unset / `0` | `1` runs source/curvature symmetry checks and reports planned independent pairs without response solves or new KI spectra. Only `0` and `1` are accepted. Use an isolated dump prefix: native Wannier setup can still write ordinary Wannier sidecars. The wrapper handles this isolation. |
| `JDFTX_KI_SYMMETRY_DIAGNOSTICS` | Unset / `0` | `1` prints nearest rejected source candidates; `0` disables it. Only `0` and `1` are accepted. It evaluates additional candidates but does not relax acceptance. The wrapper's `--diagnostics` sets this to 1. |
| `JDFTX_FFT_BATCH_SIZE` | `32` | GPU maximum scalar-wavefunction FFT batch size, integer **0–256**. `0` selects the original per-column path. Larger values use more scratch memory; CPU operations do not use this CUDA batching path. |
| `JDFTX_KI_FFT_CORRELATION` | Enabled when applicable | Set `0` to force the direct orbital-source/conduction-contraction path. Otherwise compatible meshes and sufficient cache space enable exact FFT mesh correlations. Requires response optimization. |
| `JDFTX_KI_CORRELATION_CACHE_MIB` | `contractionCacheMiB` | Finite nonnegative MiB override for FFT mesh-correlation source and workspace storage per rank. An insufficient/zero budget selects the direct path. Does not replace every other cache budget. |
| `JDFTX_KI_TIME_REVERSAL` | Unset / `1` | Allow the verified one-branch Sternheimer optimization. `0` forces both branches; only `0`/`1` are accepted. `1` still falls back to two branches if the occupied projector/Hamiltonian test fails. This is separate from symmetry reuse between q and −q. |
| `JDFTX_KI_EMPTY_PRECOND_BANDS` | `8` | Maximum low empty-state columns in the response preconditioner, integer **0–64**; capped by available bands and memory. `0` disables it. Active only with optimized occupied-only nested independent-band CG. |
| `JDFTX_KI_EMPTY_PRECOND_MIB` | `256` | Per-rank budget for that preconditioner's retained source/offset spaces, finite **0–65536 MiB**. `0` disables it. |
| `JDFTX_KI_CANONICAL_RESPONSE` | Canonicalization enabled | Set `0` to retain the original occupied response gauge. Otherwise optimized occupied-only nested independent-band CG rotates that response basis to Hamiltonian eigenvectors. The output Wannier gauge is retained. |
| `JDFTX_KI_DIELECTRIC_RECYCLE_MIB` | `16` | Per-rank induced-potential recycling budget, finite **0–65536 MiB**, for optimized nested KI at fixed q. Charge and magnetic caches are separate; q/basis changes clear retained entries. `0` disables recycling. |
| `JDFTX_KI_REUSE_BARE_HESSIAN` | Unset / `1` | Reuse frozen bare-Hessian products in optimized nested insulating KI. `0` selects recomputation for reference comparisons; only `0`/`1` are accepted. |

For example, audit copied conduction functions using the CUDA backend and
verbose rejected-candidate diagnostics:

```sh
python3 analyze_symmetry.py --gpu --diagnostics --screening-mesh 4 4 4
```

## Output reference

The comparison plotter checks known valence/conduction screening-mesh and
epsilonInfinity metadata and rejects mismatches. During a rerun, select a
matching completed pair before generating a combined gap. Older dumps without
these metadata fields retain their previous support.

| Output suffix | Contents |
|---|---|
| `mlwfHki` | Corrected KI real-space Hamiltonian. |
| `mlwfKI` | Real-space KI correction; recover projected PBE as `mlwfHki - mlwfKI`. |
| `koopmansFormat` | `real64` or `complex128`, followed by electronic-mesh cell count and Wannier band count. Use it to decode both KI binaries. |
| `mlwfCellMap`, `mlwfCellWeights` | Real-space cells and interpolation weights. Keep them with the Hamiltonians from the same run. |
| `koopmansBands` | Full electronic-mesh k coordinates followed by corrected eigenvalues in Hartree. |
| `koopmansScreening` | Spin, requested/final screening meshes, refinement-stage metadata and columns: zero-based orbital, occupation (0/1), full-mesh bare curvature (Ha), alpha-scaled relaxed curvature (Ha), dimensionless alpha, and image correction (Ha). Coarse runs also list the curvatures used in the screening ratio. Alpha is not silently clamped. |
| `koopmansResponse` | Zero-based q/orbital indices, q coordinates, unweighted bare/response/full-mesh image contributions in Ha and screening-selection flag, followed by the screening-mesh image contribution in Ha. Flag `1` includes calculated or symmetry-reused screening response; flag `0` is outside the selected screening mesh and its response entry is uncomputed. The appended image column can differ from the full-mesh image when screening uses fewer q points. |
| `koopmansQGroup1.out`, etc. | Additional MPI q-group logs; suffix numbering is zero based, so group 2 uses suffix `1`. The first group's log is the main native log. |
| `mlwfState` | Native SCF-content/basis/mesh fingerprint for validating rotations and overlaps before restart. |

## Files and directories

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

For screening on a separate coarse electronic/Wannier grid following the official package, use [the separate-grid workflow](official-coarse-workflow.md). Its `--coarse-grid` option differs from the native q-only screening-mesh override.
