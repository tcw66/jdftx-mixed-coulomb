# Vacuum Wannier functions on four FCC sites

This example follows the zero-electron, zero-potential vacuum calculation
in `/home/tyler/vac/hcp`, with four unpinned Gaussian trial functions at
close-packed FCC positions instead of the HCP trial array. The reference
files are not modified. No dummy atom or pseudopotential is used.

```bash
cd /home/tyler/apps/jdftx_defect_coulomb/examples/wannier-vacuum-fcc
./run.sh
python3 inspect_centers.py
```

The runner defaults to two MPI ranks and the GPU executables in `build-gpu`.
Use `NP=1` for one GPU, or select `JDFTX_BUILD` and `JDFTX_SUFFIX` explicitly.
CPU execution uses `JDFTX_SUFFIX=''`. NumPy and Matplotlib are required.
The atom-free `wannier-center-vacuum` command is included in the rebuilt
`build/wannier` and `build/wannier_gpu` executables.

## Definition

- A conventional cubic cell of side **6 bohr**, containing four Gaussian
  trial centers at **(0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0)**.
  Positions are fractional; Gaussian sigma is **1 bohr**, as in the HCP input.
- These sites and their periodic copies form an FCC array, with nearest
  neighbors separated by 6/sqrt(2) = 4.24264069 bohr. The electronic bands
  are represented in the conventional cubic cell and are therefore folded.
- Zero atoms, zero electrons, zero local potential: **H = -Laplacian/2**.
  There is no SCF density optimization and no Fermi level. Energies are
  referenced directly to the zero vacuum potential.
- **8x8x8 k mesh**, 40 empty bands, 30-Hartree plane-wave cutoff, and an
  explicitly fixed 32x32x32 potential grid. `results/zero.Vscloc` is generated
  as exactly zero float64 values on that grid.
- Four entangled-band Wannier functions; inner window [-1e-6, 0.3] Ha and
  outer window [-1e-6, 1.5] Ha. The tiny negative lower bound includes E=0.
- Both MV and TruncatedDC start fresh from exactly the same Gaussian trials.
- The reference bands are solved at zero potential along the conventional
  cubic Gamma-X-M-Gamma-R-X path, and independently checked against the
  sorted analytic energies |k+G|²/2.

The parameter choices keep the first demonstration small. The free-electron
bands cross and are not an isolated four-band manifold. The inner window
imposes a target subspace; a finite Wannier mesh still has interpolation
error away from its sample points. Do not interpret the four functions as
an exact representation of every free-electron state.

## Inspect the result

Outputs are under `results/measure-comparison/`:

- `CENTERS.md`, `center-analysis.json`, and `centers.png`: optimized center
  coordinates, periodic displacements from the FCC trials, site spreads,
  and checks of exported orbital orthonormality.
- `comparison.png` / `.pdf`: folded band overlay and objective convergence.
- `summary.json`, `cross-objectives.json`, `REPORT.md`: numerical results.
- `MV/results/vac.mlwfCenters` and `TruncatedDC/results/vac.mlwfCenters`:
  Cartesian centers in bohr, fractional centers, and measure-specific spreads.
- `MV/results/vac.0.mlwf` through `vac.3.mlwf`, and the corresponding
  TruncatedDC files: **real-space Wannier amplitudes**, native float64 binary
  arrays in JDFTx grid order. For the tested settings the array is 120x120x120
  over a periodic cubic supercell of side 48 bohr; confirm dimensions in
  `wannier.out` if changing the input. The first point is at the origin.
- Each measure's `analytic-errors.json` checks direct eigenvalues against
  analytic free electrons; `band-errors.json` measures interpolation errors.

`wannier-center-vacuum Gaussian ...` accepts the same Gaussian contribution
syntax as `wannier-center`, without requiring any ions. It does not pin the
centers. TruncatedDC center outputs are phase-based estimates, rather than
variational full-DC centers. A homogeneous vacuum has no preferred origin;
remaining at the FCC trials does not establish a unique optimal arrangement.

To repeat, move the existing comparison directory aside and rerun `run.sh`.
To change only localization, edit `02-wannier.in` and `results/windows.in`,
move the comparison directory aside, and run `python3 compare_measures.py`.
The driver snapshots the empty-band states and potential. It refuses to
replace an existing comparison automatically.

The previous 8-Hartree run is preserved under `results-8Ha/`, including its
SCF snapshots, Wannier files, centers and comparison plots. The 30-Hartree
rerun uses the same Gaussian trials, k mesh, windows and band count.

## Perturbing Gaussian trial positions

```bash
python3 perturb_centers.py
```

This uses the saved 30-Hartree TruncatedDC SCF snapshot and runs fresh
localizations, preserving all baseline outputs. The nine random tests use
RMS displacement lengths of 0.1, 0.5 and 1.0 bohr per center, each with seeds
23, 71 and 181. Random Cartesian displacements are drawn from a normal
distribution, their common translation is removed, and their RMS vector
length is normalized to the specified value. Gaussian sigma remains 1 bohr.
An additional control shifts all four trials by (0.3,-0.2,0.1) bohr.

The lattice, potential, band set, energy windows, widths, k mesh and stopping
criteria are unchanged. No optimized rotations are loaded. The default is
two MPI ranks with the GPU build; `--np`, `--build`, `--suffix`,
`--amplitudes` and `--seeds` select alternatives.

Outputs are under `results/truncateddc-perturbations/`: `summary.json`,
`REPORT.md`, `perturbation-summary.png`/`.pdf`, and one subdirectory per
initial guess with its coordinates, input, final centers, native Hamiltonian,
rotations and minimization log. Existing perturbation results are not
automatically overwritten. Raw real-space functions are not exported for
this sweep, to avoid duplicating large files.

Final centers are registered to an FCC array allowing a common translation,
site permutations and periodic images. A vacuum Hamiltonian has no preferred
origin, so a collective shift does not count as symmetry breaking. The final
FCC RMS residual measures distortion of the arrangement after that fit.
Center outputs are phase-based estimates. Objective differences and center
geometry are both reported; neither a few successful seeds nor a lower value
from one start establishes the global optimum.

Repeat exactly the same perturbations with MV:

```bash
python3 perturb_centers.py --measure MV
python3 compare_perturbations.py
```

MV results are written separately to `results/mv-perturbations/`; the original
TruncatedDC sweep remains in `results/truncateddc-perturbations/`. The matched
comparison is saved to `results/perturbation-method-comparison/`. Its driver
verifies identical saved model, potential, states and initial Gaussian positions
before comparing iteration counts, final center distortions and each measure's
objective relative to its own symmetric baseline.

## Uniform random positions throughout the cell

```bash
python3 random_centers.py
```

Unlike the perturbation sweep, this draws **four independent uniform positions
in the entire [0,1)^3 cell**, without FCC offsets, zero-mean constraints or a
minimum separation. Seeds 23, 71 and 181 supply three identical paired starts
for MV and TruncatedDC. Widths, windows, 30-Hartree states, 8x8x8 mesh and
stopping criteria are held fixed; no restart rotations are loaded. The
300-iteration limit matches the earlier tests. `--seeds`, `--np`, `--build`
and `--suffix` allow other test sets and execution configurations.

Results are separate under `results/random-cell-starts/`: `REPORT.md`,
`summary.json`, comparison plots, center plots, and one directory per seed
and measure. Both methods' inputs are built from the same shared SCF snapshot
and initial-coordinate array. FCC fitting removes a common translation and
accounts for periodic images and permutations, because the homogeneous vacuum
has no preferred origin. Final orbital-label centers, fitted translations,
spreads, residuals and convergence status are preserved. The script refuses
to overwrite an existing random-position sweep.

## Exact center constraints with an inner window

The hard-center implementation supports both MV (`FiniteDifference`) and
`TruncatedDC`, including entangled bands and the existing inner/outer windows.
To keep each Wannier center at the position of its own original Gaussian
initial guess, include `fixCenters yes` in the existing `wannier` command:

```text
wannier localizationMeasure TruncatedDC fixCenters yes \
    innerWindow -0.000001 0.3 outerWindow -0.000001 1.5
wannier-center-constraint-tolerance 1e-8
```

This uses the input Gaussian coordinates as the hard targets. Random initial
positions therefore remain at those same random positions throughout the
accepted localization steps. It does not move them onto FCC sites. For composite
trial orbitals, the target is the coefficient-squared weighted position used
by JDFTx for the trial-orbital center. Frozen Wannier orbitals remain untouched.
The tolerance is the maximum Euclidean displacement in bohr.

The same flag applies to orbital projections. For example,
`wannier-center Si 1 px` fixes that Wannier center at the input position of
the first Si atom. All s/p/d projections on an atom receive that atom's
position, while remaining separate orthogonal Wannier functions. With
`addAtomicOrbitals yes fixCenters yes` in the `wannier` command, automatically
added atomic projections are fixed in the same way. These are hard center
constraints; leave `pinAtomicOrbitals` at its default `no`.
For `wannier-center Numerical <b> <x> <y> <z>`, the specified position is the
target. No center is inferred from the projected wavefunction. Composite
linear combinations use JDFTx's coefficient-squared weighted trial position.

The Si regression `wannierFixedAtomicCenters` exercises explicit s/px/py/pz
and automatically added atomic projections with an inner window for both
MV and TruncatedDC.

After the free-electron SCF calculation from `./run.sh`, run the matched test:

```bash
python3 fixed_centers.py                  # two MPI ranks / two GPUs, seed 23
# CPU alternative, with a separate output directory:
python3 fixed_centers.py --suffix= --np 2 --out results/fixed-initial-centers-cpu
```

The driver reuses the saved 30 Ha, 8x8x8 SCF states, starts four Gaussians at
independent uniform random positions, and keeps each orbital centered at its
original Gaussian position. Both methods use inner window [-0.000001, 0.3] Ha and outer
window [-0.000001, 1.5] Ha. Inputs, logs, exported rotations, centers and a JSON
summary are saved under `results/fixed-initial-centers/`. The exported center
coordinates are independently compared with the input Gaussian coordinates.
Existing results are preserved;
choose a new `--out` to repeat. The random seed can be changed with `--seed`.
The driver allows 300 localization iterations. Check `converged` in the JSON
summary separately from the center error: a run can satisfy the hard centers
and still require more iterations to minimize the spread. Use, for example,
`--iterations 1000 --out results/fixed-initial-centers-long` for a longer run.

For a subset of orbitals or separately prescribed targets, the lower-level
`wannier-center-constraint <index> <x> <y> <z>` command remains available instead
of `fixCenters yes`. Its index is zero-based, including preceding frozen
orbitals, and coordinates follow `coords-type`. Do not combine this command
with `fixCenters yes`.

The solver projects localization gradients onto the null space of the center
Jacobian and restores the nonlinear equalities after trial steps. It uses the
existing inner-window-preserving rotations throughout. Thus the objective is
the original localization functional, with no added center penalty. Every
accepted optimization state satisfies the center tolerance. The final log
also checks orthonormality and inclusion of every inner-window state.

Hard-center runs use projected Polak–Ribiere conjugate gradients, an Armijo
search and damped Newton restoration. `nIterations`, `energyDiffThreshold`,
`nEnergyDiff`, `knormThreshold`, `convergeAll`, `alphaTstart` and `fdTest` are
supported. Generic direction and line-search choices do not select the
hard-center algorithm; `maxThreshold` is rejected. `fdTest yes` checks both the
objective derivative and the center Jacobian before feasibility restoration.

MV constrains its usual discrete Berry-phase centers. TruncatedDC constrains
the phase-based centers reported by JDFTx, obtained from phases of the
k-averaged diagonal overlaps; these are not full density-convolution centers.
Zero relevant overlaps make these centers undefined. Infeasible targets or a
locally singular center map cause an explicit failure, without relaxing the
centers or the inner window. A very coarse mesh or poor initial orbitals can
make Newton restoration fail at phase branch changes even when a solution
exists; better initial orbitals or a denser mesh can help. RealSpace measures,
targets on frozen Wannier orbitals, and simultaneous hard/soft pinning of the
same orbital are currently unsupported.

Earlier validation with **separate FCC targets**, rather than fixed initial
positions, is retained in `results/fixed-center-check/`. This is a different
test from the driver above. MV converged in 114 iterations with objective
26.5503269637 bohr² and maximum center error 2.86e-9 bohr. TruncatedDC converged
in 120 iterations with objective 26.9063194528 bohr² and error 1.78e-9 bohr.
These agree with the respective unconstrained FCC localization values within
7e-8 bohr². The final inner-window inclusion errors were 4.36e-13 and 2.71e-13;
orthonormality errors were 1.09e-12 and 4.56e-13. Convergence here uses the
energy-change criterion; the final projected gradient norms were about 1e-4
and 3e-4, respectively.

The corrected **fixed initial random positions** run is retained in
`results/fixed-initial-centers-cpu/`. Both methods reached the 300-iteration
limit without localization convergence. MV reduced its objective from
85.9416640 to 42.5524865 bohr² and retained the input centers within
3.53e-12 bohr. TruncatedDC finished at 39.6262302 bohr² with center error
3.81e-12 bohr. These constrained objectives must be compared separately
from the freely translating FCC tests above.

The smaller automated regression also differentiates the analytic Jacobians,
checks every feasible iterate, and independently reads the exported rotations
to verify orthonormality and inner-window inclusion. It includes a partial
constraint with a shifted Cartesian target and rejection of duplicate targets:

```bash
ctest --test-dir ../../build -R 'wannier(FixedCenters|TruncatedDC)' --output-on-failure
python3 ../../jdftx/test/wannier/checkFixedCenters.py ../../build --suffix _gpu --np 2
```
