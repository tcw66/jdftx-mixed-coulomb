# Silicon reruns with literature functional choices

Run from this directory. The native launcher uses the rebuilt JDFTx binaries
and the installed Quantum ESPRESSO `pw.x`/`ph.x`. Both native presets are
**one-shot KI(2)**. A separate official-backend input below provides **full
KI@PBE with finite-occupation screening**, the method behind commonly cited
silicon gaps around 1.1–1.2 eV.

| Launcher/preset | Base XC | Spectral correction | Pseudopotential |
| --- | --- | --- | --- |
| `run.sh --functional lda-pw` (default) | Slater exchange + Perdew–Wang LDA | Native KI(2), DFPT screening | PseudoDojo NC scalar-relativistic LDA/PW, v0.4 standard |
| `run.sh --functional pbe` | PBE | Native KI(2), DFPT screening | PseudoDojo NC scalar-relativistic PBE, v0.4 standard |
| `run_full_ki_pbe.py` | PBE | Official full KI, DSCF screening | SG15 v1.2 PBE scalar-relativistic |
| `run_official_tutorial.py` | As specified by the downloaded official input | Official full KI, DSCF screening | PseudoDojo v0.4 LDA scalar-relativistic standard |

The PseudoDojo LDA file explicitly specifies `SLA PW NOGX NOGC`, so its
matching native command is `elec-ex-corr lda-PW`. It is not LDA-PZ.
Pseudopotentials are downloaded from [PseudoDojo](https://www.pseudo-dojo.org/),
checked against pinned SHA-256 values, and checked for the correct XC and
norm-conserving format. `--pseudo FILE` accepts another matching NC UPF
and records its checksum. No PBE checkpoint is reused for an LDA calculation.

## Native LDA/PBE runs

```bash
./run.sh --functional lda-pw --dry-run
./run.sh --functional lda-pw --resume

./run.sh --functional pbe --dry-run
./run.sh --functional pbe --resume
```

Each run has its own directory:
`runs/lda-pw-ki2-4x4x4/` or `runs/pbe-ki2-4x4x4/`. The launchers do not use
the existing `../state` or `../results` directories. A directory whose settings
do not match is rejected; use `--directory` to start another calculation.
`--resume` skips matching completed stages. It checks included input files,
native SCF/rotation dependencies, executable/library hashes and required
output files before reusing a stage.

Native defaults are:

* Experimental silicon lattice parameter 10.2622 bohr; no geometry relaxation.
* 40 Ha wavefunction cutoff, equivalent to 80 Ry. The QE density cutoff is
  320 Ry; the native norm-conserving density grid follows its product cutoff.
* 4x4x4 Gamma-centered electronic and full screening meshes, 20 stored bands.
* Four bonding occupied Wanniers and four empty Wanniers initialized from
  atom-centered tetrahedral sp3 combinations. The empty manifold is
  disentangled separately, excluding occupied bands.
* A separate 12x12x12 QE mesh for the electronic dielectric tensor.
  `ph.x` computes electronic response with `epsil=.true., trans=.false.`.
  Its cubic tensor average supplies `epsilonInfinity` to native KI, enabling
  the point-charge Ewald correction. No experimental or fitted epsilon is used.
* Single-spin charge and magnetic KI response about a no-spin ground state.
* Two MPI ranks/q groups, four threads per rank, GPU native executables.

The launcher first runs a fresh native SCF, then matching QE SCF/DFPT
dielectric stages. It builds conduction energy windows from the new SCF
eigenvalues; absolute energy thresholds from the old PBE checkpoint are not
transferred to LDA. Next it localizes both manifolds, freezes the rotations,
computes both KI Hamiltonians, and calls the existing comparison plotter.
Original DFT is recovered from the same frozen Hamiltonians as
`mlwfHki - mlwfKI`. Plots and JSON label the DFT bands `LDA-PW` or `PBE`
according to the selected preset.
`results/quality-checks.json` records band overlap and negative relaxed
curvature checks. Resolve flagged model/screening problems before comparing
the KI gap with a reference value.

Inspect orbitals before spending time on screening:

```bash
./run.sh --functional lda-pw --resume --stop-after wannier
# Inspect logs/*localize.out, the centers/spreads and wannier-windows.json.
./run.sh --functional lda-pw --resume
```

For convergence, select new directories and change the meshes:

```bash
./run.sh --functional lda-pw --mesh 8 8 8 --dielectric-mesh 16 16 16
./run.sh --functional pbe --mesh 8 8 8 --dielectric-mesh 16 16 16
```

`--epsilon-infinity VALUE` supplies a computed dielectric constant and skips
QE. `--epsilon-infinity 0` explicitly selects background electrostatics for
a convergence diagnostic. `--prepare-only` writes the static model/QE/SCF
inputs without launching calculations. `--dry-run` checks the native SCF
input; once a checkpoint and dielectric constant exist it also checks the
Wannier inputs with isolated dump prefixes. `--stop-after` accepts `scf`,
`dielectric`, `wannier` or `ki`.

Resource/path overrides:

```bash
MPI_RANKS=2 WANNIER_THREADS=4 ./run.sh --functional lda-pw --resume
./run.sh --functional lda-pw --cpu --directory runs/lda-pw-cpu
QE_BIN=/path/to/quantum-espresso/bin ./run.sh --functional lda-pw --directory runs/lda-pw-qe
```

Other overrides are `PYTHON`, `MPIEXEC`, `JDFTX_BINARY`, `WANNIER_BINARY`,
`QE_PW_BINARY`, and `QE_PH_BINARY`. Native model energies and cutoffs use Ha;
QE and the official koopmans JSON use Ry for cutoffs. Scripts require NumPy;
final plots additionally require SciPy and Matplotlib.

## Full KI@PBE reference workflow

Use this when the comparison target is the full nonlinear KI functional,
rather than its second-order approximation:

```bash
python3 run_full_ki_pbe.py --prepare-only
# Activate an official koopmans environment with its compiled backend.
python3 run_full_ki_pbe.py --resume
```

The JSON uses `functional="ki", method="dscf"`, MLWF initialization,
four occupied/four empty sp3 Wanniers, a 4x4x4 initial mesh, SG15 v1.2 PBE,
and a Makov–Payne correction with an ab initio electronic dielectric constant.
Its 80/320 Ry cutoffs equal the native 40/160 Ha wavefunction/density cutoffs.
The frozen/outer Wannier90 windows (10.6/16.9 eV) come from the official
silicon tutorial and should be checked against the generated DFT bands.
This is a literature-method preset, not an exact copy of archived paper inputs.

This separate workflow requires the
[official koopmans installation and compiled backend](https://koopmans-functionals.org/en/stable/installation.html),
including KCP and Wannier-to-supercell tools. Ordinary QE `pw.x`/`ph.x` and
the native JDFTx KI(2) executable do not implement this full-KI DSCF workflow.
The launcher prepares its input without those dependencies and explains
what is missing if run outside that environment. No package installation
or full-KI production calculation was performed here.

## Literature scope and validation

[Colonna et al., JCTC 18, 5435 (2022)](https://doi.org/10.1021/acs.jctc.2c00161)
uses LDA for its main KI(2) calculations and four bonding/four antibonding
silicon Wanniers. Its silicon 4x4x4 validation gap of 1.41 eV is explicitly
not converged in sampling. The methods give an 80 Ry cutoff; the archive
does not contain a complete silicon input set. The native lattice, window
selection and separate dielectric mesh here are documented rerun choices,
not claims of exact paper settings.

[De Gennaro et al., PRB 106, 035106 (2022)](https://doi.org/10.1103/PhysRevB.106.035106)
and [Linscott et al., JCTC 19, 7097 (2023)](https://doi.org/10.1021/acs.jctc.3c00652)
report full KI@PBE results with MLWFs and finite-occupation screening.
The full-KI PBE preset is a literature-oriented variant; it is not an exact
copy of the latest tutorial input. The exact downloaded tutorial is provided
separately below.
Changing the base XC in the native code does not change KI(2) into full KI.

## Exact official tutorial input

`si-official-tutorial.json` is the unchanged input downloaded from the
[latest official tutorial](https://koopmans-functionals.org/en/latest/tutorials/tutorial_2.html)
on 12 October 2026, SHA-256
`4216e7c41f70620a4ded59c2277accee370ae870aa34a812e8df09ef917c0e88`.
The launcher changes `task` from `wannierize` to `singlepoint`, as instructed
by the tutorial. Other changes require explicit command-line options.

```bash
python run_official_tutorial.py --prepare-only
# Activate the official koopmans environment, including KCP and wan2odd:
python run_official_tutorial.py --resume
# The tutorial recommends checking better sampling for production:
python run_official_tutorial.py --mesh 4 4 4
```

This input uses 60 Ry, 20 bands, four occupied/four empty sp3 Wanniers,
`alpha_guess=0.077`, and **mp_correction=false**. The downloaded JSON chooses
the **LDA PseudoDojo library** and omits explicit `base_functional`/`input_dft`,
although the tutorial prose labels its baseline PBE. We preserve this
discrepancy instead of silently replacing its pseudopotentials or XC settings;
inspect the generated QE/KCP input and reported XC when comparing results.
The initial 2x2x2 mesh is explicitly an unconverged teaching example.

The original `pw.x` pool count is four; configure at least four MPI ranks in
the official environment, or use `--npool 1`. `--resume` permits an existing
matching JSON; the official engine controls recalculation, and the downloaded
`from_scratch=true` remains unchanged. This runner was checked in preparation
mode. The official Python/KCP environment is absent here, so a full tutorial
calculation has not been executed.

Native LDA and PBE input dry runs and small SCF/dielectric/localization
and complete KI/plotting pipelines were checked, including stage reuse on
resume. The 2x2x2 smoke models have spurious interpolated DFT band overlap
and are unsuitable for physical gap comparison. They establish executable
inputs and stage handoffs only.

The default LDA setup (4x4x4 native mesh, 40 Ha cutoff, 12x12x12 dielectric
mesh) was also checked through Wannier localization: its computed electronic
dielectric constant was 13.445485177, all four conduction spreads agreed at
16.6548 bohr^2, and a 16x16x16 search of the frozen DFT interpolation gave a
positive 0.481846 eV gap. Its full KI screening was left for the user rerun.
The separate full-KI JSON was checked structurally; its external production
workflow was not executed because the official backend is not installed.
