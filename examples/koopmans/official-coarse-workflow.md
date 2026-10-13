# Separate coarse electronic-grid KI screening

The new workflow follows the DFPT/KCW coarse-grid structure of the official
`epfl-theos/koopmans` package. It replaces q-only subsampling as the recommended
way to perform an official-style coarse screening calculation.

The reference is package commit `023fe26857466c5fae2520f5794dd5a86c416a84`,
whose QE submodule is `48fcd06f3b2c77fc8cd64078d5a0e4b79457a01f`.
The six audited KCW routines (`rho_of_q`, `bare_pot`, `screen_coeff`,
`solve_linter_koop_mod`, `koopmans_ham`, `coulomb`) match the local QE 7.5
sources byte-for-byte. The package's [DFPT workflow](https://github.com/epfl-theos/koopmans/blob/023fe26857466c5fae2520f5794dd5a86c416a84/src/koopmans/workflows/_koopmans_dfpt.py#L154)
creates a coarse subworkflow retaining the original SCF grid. Its
[Wannierization workflow](https://github.com/epfl-theos/koopmans/blob/023fe26857466c5fae2520f5794dd5a86c416a84/src/koopmans/workflows/_wannierize.py)
then uses coarse NSCF states, coarse Wannier orbitals, and coarse DFPT
screening. The parent uses those alphas in its fine-grid Hamiltonian.

## What changed

| Stage | Previous q-only method | New separate-grid workflow |
|---|---|---|
| SCF density | Fine grid | Fine grid, retained explicitly |
| Screening NSCF states | Fine grid | Independently solved on the coarse grid |
| Screening Wannier orbitals | Fine grid | Independently localized on the coarse grid |
| Screening k integration | Fine grid | Coarse grid |
| Screening q integration | Coarse subset | Full q mesh of the coarse electronic model |
| Final bare Hamiltonian | Fine grid | Fine grid |
| Screening transfer | Ratio from sparse fine-orbital sampling | Imported, matched coarse-orbital alphas |

An important native correction was needed: KI postprocessing previously
reconstructed the density from its loaded wavefunctions. For NSCF orbitals,
that would overwrite the fine SCF reference with a coarse wavefunction
sum. The new `wannier-koopmans-reference-density` path reads the reference
density, builds its potential and XC kernel, and retains them while the
response uses the coarse electronic states. Both stages keep the same XC,
pseudopotentials, lattice, FFT grid, and electrostatic convention.

## Run on the existing silicon calculation

Use an isolated output directory; production inputs and checkpoints are
read but not overwritten. For the current directory, the wrapper is:

```bash
~/koopmans/run_official_coarse.sh \
  --coarse-grid 4 4 4 --cpu --ranks 2 --threads 2 \
  --directory ~/koopmans/archive/my-coarse-4 --plot
```

The portable source launcher is `run_coarse_screening.py` beside this file.
Use `--example-dir /path/to/calculation` when launching it directly.
`--dfpt-coarse-grid` is an alias of `--coarse-grid`. The screening and
Hamiltonian grids are independent; divisibility is not required. The launcher requires a
completed fine SCF density, wavefunctions, eigenvalues, and localized
occupied/empty checkpoints. It inherits the localization projections and
settings from the source calculation. Numerical trial functions are shared
as physical projections and resampled by JDFTx; the coarse localization is
still independent.

`--prepare-only` writes reviewable inputs. `--resume` verifies state/input,
executable, library, checkpoint, and product hashes before reusing stages.
`--stop-after nscf|wannier|screening|hamiltonian` permits staged execution.
Omit `--cpu` for CUDA; `--ranks` and `--threads` control native execution.
The optional `--plot` writes the completed fine-grid comparison and gap search.

The launcher inherits epsilon from the source KI input unless explicitly
overridden with `--epsilon-infinity`. A nonzero value must be a separately
computed electronic dielectric constant for the same XC and pseudopotential
model; this workflow neither computes nor fits it. Epsilon=0 retains the
neutralizing-background convention and may require substantially denser
coarse meshes. Each screening stage uses the coarse supercell's Ewald term;
the final bare Hamiltonian uses the fine supercell's term.

## Orbital correspondence and localization checks

The official workflow transfers alphas through its shared band ordering.
Here, Gamma-periodic orbital densities establish a fine-to-coarse permutation.
The launcher checks density normalization, FFT grid and lattice, solves a
maximum-overlap assignment, and records its score matrix and permutation in
`fine/*.alpha.mapping.json`. The default minimum cosine overlap is 0.9.
Near-tied assignments are accepted only if they transfer alphas equivalent
within 1e-4; otherwise the workflow stops. An explicit JSON `--orbital-map`
can specify `valence` and `conduction` permutations, still subject to the
minimum-overlap check. These checks support correspondence; they do not
prove convergence of the coarse approximation.

A large-gradient Wannier stop is rejected even if native output says `Done!`.
A numerical-floor line-search stop is accepted only when the final gradient
is at most 1e-6, and its measured residual is recorded in `stages.json`.
The 2³ silicon regression exposed a failed empty-state localization with
gradient about 5.8; that result is not a validated gap calculation.
The current production 4³ localizations reached gradients of 2.67e-7 and
1.70e-7. No source rotations are silently reassigned to different orbital
spaces, and screening coefficients are not clipped or fitted.

## Native interface

Coarse screening:

```text
wannier-koopmans <epsilonInfinity> ...
wannier-koopmans-reference-density /path/to/fine-scf.n
wannier-koopmans-screening-only yes
```

Fine Hamiltonian:

```text
wannier-koopmans <same-epsilonInfinity> ...
wannier-koopmans-reference-density /path/to/fine-scf.n
wannier-koopmans-alpha-file /path/to/matched-coarse.alpha
```

The importer requires a complete `koopmansScreening`-format table in target
orbital order, matching occupations, KI(2), spin and epsilon metadata, and a
source with equal electronic k and screening q meshes. This rejects use of
an old fine-k/coarse-q table as an official-style coarse-grid source. Model
and orbital correspondence are established by the launcher. The fine stage
solves no screening responses. Its response dump therefore has no selected
responses; its screening table records the imported source mesh and filename,
while its bare curvature and Hamiltonian use the full fine mesh.

`wannier-koopmans-save-densities yes` exports the periodic orbital densities
and their format/lattice metadata from a nonmagnetic Wannier calculation.
The old `wannier-koopmans-screening-mesh` remains a separate q-only diagnostic
or adaptive integration option; it is not the equivalent of
`dfpt_coarse_grid`.

## Validation and limits

CPU and CUDA H2 tests passed, including serial and two-rank MPI workflows,
reference-density retention, same-grid import recovery, zero/negative alpha
handling, metadata rejection, orbital permutations, ambiguous matching,
localization guards, and verified resume. CTest registers
`koopmansCoarseWorkflow` and `koopmansCoarseWorkflowMPI`.

A further QE comparison uses a 2×2×1 SCF mesh and Gamma NSCF/screening states
at the retained fine density. LDA-PZ agrees at the existing strict tolerances.
The PBE occupied state also passes, but the empty state differs: its bare
curvature is approximately -0.0689667 Ha in JDFTx and -0.0736120 Ha in QE.
The orbital and reference densities match closely, excluding a different
empty-state density as the explanation. Independent full-PBE potential
finite differences validate the native charge and magnetic kernels for this
actual reference/source pair (relative errors below 8.1e-7; curvature errors
below 1e-8 Ha). QE uses different GGA derivative steps and gradient/density
thresholds. Their exact contribution to this discrepancy has not been fully
isolated. Strict numerical identity with QE for this extra PBE fixture is
**not claimed**, and its comparison tolerance has not been loosened.

This change implements the official-style **KI(2)/DFPT coarse workflow**.
The linked silicon tutorial uses full KI/finite-occupation DSCF through KCP,
which remains a different functional and screening method. Coarse mesh,
dielectric, orbital-space, and functional convergence still determine the
physical gap; reproducing the workflow alone does not fix a target gap.

## Current production result (13 October 2026)

The isolated calculation in
`archive/official-coarse-workflow-20261013/production-4/` completed 4³
coarse NSCF/Wannier/screening and 8³ fine Hamiltonians for the existing
PBE/SG15 silicon model, with epsilon=0 and four occupied/six empty orbitals.

- Occupied alpha: approximately 0.209939697.
- Empty alpha: 0.055503571–0.055640119; all six are positive.
- Fine-grid alpha import matches the mapped source table exactly.
- Both fine Hamiltonian stages report zero screening response solves.
- Interpolated KI(2) gap: **1.652170 eV**; PBE gap: **0.545074 eV**.

The output plot/data/JSON are under `production-4/fine/`. Original production
inputs, SCF state, and running full-mesh calculation were left intact.
This is a completed workflow validation, not agreement with a reported full-KI
gap. In particular, zero-epsilon coarse-grid charged-orbital curvatures and
the occupied alphas remain mesh dependent; a correctly structured coarse
workflow still needs electrostatic and mesh convergence.
