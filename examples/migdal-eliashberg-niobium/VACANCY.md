# Self-consistent Nb vacancy BdG at 2 K

`vacancy.py` follows §14 of the updated
[workflow PDF](../../references/migdal_eliashberg_wannier_bdg_updated.pdf).
It removes every orbital on one Nb site and solves a **specified static
pairing model** self-consistently. It does not hold the bulk pairing field
fixed, and it is not a defect-resolved dynamical Eliashberg calculation.

The [completed calculation and validation](VACANCY_VALIDATION.md) reports the
final 24³-twist result with a 128³ bulk calibration, including the remaining
sampling and finite-supercell uncertainties. The early 32³-calibrated pilot
results below are retained to document how the numerical settings were refined.

## Geometry and charge prescription

The first model uses a periodic 4×4×4 primitive-cell supercell. All six spatial
Wannier orbitals at cell (0,0,0), including both spin copies, are deleted before
diagonalizing the normal Hamiltonian or selecting the pairing subspace. This
leaves 63 Nb sites, 378 spatial orbitals, and a spin-reduced BdG dimension of
756 before the low-energy projection. The equivalent complete spinful Nambu
matrix would have dimension 1512.

This is a periodic vacancy array at concentration 1/64, not an isolated vacancy.
Periodic boundaries represent bulk material without introducing surfaces.
The chemical potential remains at its bulk value, representing a bulk reservoir.
The model's occupations can redistribute; no fixed particle-number constraint
is imposed. A neutral closed sample would need an explicit electron-number
prescription and a chemical-potential update. Orbital deletion alone does not
define an ab initio neutral-vacancy charge state.

No ionic relaxation, changed phonon force constants, defect screening/Hartree
potential, spin polarization or vacancy-induced change in Z is included.

## Pairing interaction and cutoff

The interaction channels are independent onsite intraorbital singlets,
`B_ia = c_ia,down c_ia,up`, with `H_int = -sum_ia U_a B_ia† B_ia`.
There is one bulk-calibrated attractive coefficient for each of the six orbital
indices, repeated identically on all surviving Nb sites. The update is
`Delta_ia = U_a F_ia`. Every site and orbital has an independent pairing field.

The normal Hamiltonian is `(h_W-mu)/Z0`, using Z0 from the saved 2 K ME solution.
The reference pairing field is `Phi0/Z0`. The pairing window is a **normal-state
spectral projector** `P = theta(Ecut - abs(h_qp))`, with Ecut chosen as the largest
phonon energy in the saved bulk spectrum (27.4472 meV for this snapshot).
The projected pairing block is `P Delta P` in the compatible time-reversal gauge.
This defines local interaction channels before projection, not strictly local
matrix elements after applying the energy window.

At a given supercell Bloch point q, let U_q contain the retained normal
eigenvectors. In the electron basis U_q and its consistently time-reversed hole
basis, the pairing block is `U_q† diag(Delta_ia) U_q`. **All matrix elements of
this block are retained**, including interband pairing. The code diagonalizes
the projected BdG Hamiltonian, transforms its u and v components back to the
Wannier orbitals, and evaluates

```
F_ia = sum_q w_q sum_(E_n>0) u_n(ia) v_n(ia)* tanh(E_n/(2 kB T)).
```

All states of the projected BdG matrix participate in the update. A second
cutoff on its eigenvalues is not applied. Normal occupations outside the
pairing window are retained when reporting model particle numbers.

The six U_a are fitted so the homogeneous solution has the source ME gap at
2 K: `U_a = Delta_bulk / F_bulk,a`. The channel choice and window make this a
particular model reduction; the bulk gap alone does not uniquely specify it.
The clean self-consistency problem is then solved independently. Its gaps at
4 and 6 K and its linearized Tc are checked against the saved ME solution.
The pilot fit predicts static Tc≈11.96 K versus ME Tc≈11.22 K; this discrepancy
is retained and documented rather than claiming an exact ME mapping.

## Brillouin-zone sampling and hoppings

The supercell Hamiltonian is obtained by exact Fourier folding of the original
Wannier H(k). All native hopping ranges contribute, including hops that cross
the supercell boundaries multiple times. It does not use a truncated finite
8³ bulk BdG hopping export. A pristine-supercell folding test compares its
eigenvalues with the corresponding folded primitive-cell bands.

Supercell k-point integration is essential for resolving meV gaps. A Γ-only
finite-cell calculation can be dominated by discrete normal-state levels.
`--twist-mesh 8 8 8` samples 512 supercell Bloch points; with no defect this
is equivalent to a primitive-cell 32³ mesh. Higher sampling is checked with
the **same fitted interaction**, not by refitting U at every twist mesh.

## Reproduce the calculations

The completed 20³-q bulk input was frozen in `results-vacancy-T2/input/` while
the independent 40³-q bulk job continued. Hashes identify the source ME/BdG
files and the updated PDF. That frozen input, not a subsequently overwritten
bulk output directory, is used for the vacancy runs.

```bash
# Pilot calculation and reuse of its saved normal basis:
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy.py --resume

# Denser supercell sampling with the same bulk interaction:
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy.py \
  --twist-mesh 12 12 12 --calibration-mesh 32 32 32 \
  --fixed-calibration examples/migdal-eliashberg-niobium/results-vacancy-T2/pairing-calibration.json \
  --output examples/migdal-eliashberg-niobium/results-vacancy-T2-tw12 --resume
```

CUDA device 1 is the default for this example, leaving device 0 available for
the separate bulk run. Use `--gpu-device 0` on a one-GPU machine, or
`--backend cpu` for CPU execution. The static interaction can be calibrated on
an independent primitive-cell mesh with `--calibration-mesh`. `--supercell`
changes the vacancy geometry; keep `--fixed-calibration` when testing it.
`--normal-batch-size` controls memory use in normal-state diagonalization.

To start from a newly completed bulk result, use `--snapshot-from` and a new
`--input` directory. Wait for its ME and BdG stages to finish first:

```bash
python examples/migdal-eliashberg-niobium/vacancy.py \
  --snapshot-from examples/migdal-eliashberg-niobium/results-gpu \
  --input /tmp/new-nb-vacancy/input --output /tmp/new-nb-vacancy --gpu-device 1
```

## Convergence and outputs

Three positive initial gaps, 0.5, 1 and 1.5 times the bulk reference, are iterated
with mixing. The maximum gap-equation residual is required below 1e-9 eV.
Their solutions and grand potentials relative to the corresponding normal state
are compared. The lowest grand-potential solution among those starts is retained.
This checks those branches; it does not prove a global minimum over magnetic,
complex-phase or other pairing-channel states excluded from the model.

Outputs include:

* `pairing-calibration.json`: U_a, cutoff, source parameters and bulk validation.
* `vacancy-basis.npz`: normal-state active bases, their energies and twists,
  full normal eigenvalue lists, and normal occupations. It permits reconstruction
  of the projected BdG matrices with the saved onsite fields.
* `vacancy-T2.npz`: surviving site/orbital maps, onsite gaps, anomalous densities,
  model occupations, active BdG spectra and local DOS.
* `site-profile.dat`: six orbital fields and occupation changes by site.
* `vacancy-T2.png` / `.pdf`: radial pairing fields, occupation changes and LDOS.
* `solution-start-*.npz`, `convergence.png`: initial-condition and residual checks.
* `summary.json`: numerical residuals, geometry, grand potential and limitations.

The onsite pairing fields are not generally equal to excitation-gap edges.
The minimum positive energy is a separate quantity. The LDOS is calculated in
the active pairing window with 0.05 meV Lorentzian broadening. Multiplication
by 1/Z0 gives the coherent electron spectral weight per spin; incoherent weight
and phonon satellites are absent. Sharp LDOS structure must be checked against
k sampling and broadening before interpreting it as an impurity resonance.

Self-consistency convergence, Brillouin-zone integration convergence and the
isolated-vacancy limit are separate tests. In the pilot, increasing the twist
mesh from 8³ to 12³ changed the mean onsite field from 1.234 to 1.426 meV. Thus
the 8³ sampling is insufficient for a quantitative vacancy prediction. Finite
supercell/image effects, the fitted-channel choice and cutoff also remain model
uncertainties even after the iterative residual is small.

## Numerical tests

For a single vacancy embedded in the infinite superconducting host and
continuum reconstruction with `mlwfC`, see [EMBEDDED_VACANCY.md](EMBEDDED_VACANCY.md).

For radial pairing/correlation profiles, local spectra, and spatial site maps
from the final 2 K solution, see [LOCAL_PROFILES.md](LOCAL_PROFILES.md).

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/test/dfpt/testSelfConsistentBdG.py
OPENBLAS_NUM_THREADS=1 ME_TEST_GPU=1 ME_TEST_GPU_DEVICE=1 \
  python jdftx/test/dfpt/testSelfConsistentBdG.py
ctest --test-dir build -R '^(selfConsistentBdG|migdalEliashberg)$' --output-on-failure
```

These check a one-level analytic gap equation and particle number, convergence
from different initial fields, the variational grand-potential gradient in a
complex projected basis, save/restart, exact pristine folding, and deletion of
a site in an analytic tight-binding chain, plus anomalous-density and particle-
number normalization between pristine supercell and primitive-cell bases.
The same tests run on CPU and CUDA.
