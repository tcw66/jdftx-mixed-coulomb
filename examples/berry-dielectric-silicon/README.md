# Occupied-state Berry-phase static dielectric response

`berry-phase-field Ex Ey Ez` minimizes the bulk electric enthalpy

```
F = E_DFT - Omega E . (P_electronic + P_ionic)
epsilon_ab = delta_ab + 4*pi*dP_a/dE_b
```

The field is the prescribed **total macroscopic internal field**, in atomic
units (Hartree / electron / bohr), with the physical positive-ion convention.
This sign convention differs from JDFTx's existing `electric-field`, which
uses an electron-positive convention and a sinusoidal periodic potential.
The two commands cannot be combined.

Each electronic minimization step recomputes the density, Hartree potential,
and XC potential. The field gradient couples neighboring occupied Bloch
subspaces through the inverse overlap matrices. No sum over empty bands or
Wannier construction is used in the minimization. Saved occupied `.wfns`
files can be Wannierized independently after convergence.

Runs enabling `berry-phase-field`, including zero-field reference runs and
dry runs, automatically list the method citation in JDFTx's used-feature
citation report: I. Souza, J. Iniguez and D. Vanderbilt,
*First-Principles Approach to Insulators in Finite Electric Fields*,
Phys. Rev. Lett. **89**, 117602 (2002),
[doi:10.1103/PhysRevLett.89.117602](https://doi.org/10.1103/PhysRevLett.89.117602).

The electronic polarization is computed as
`spinWeight * sum_i a_i * Im log product_j det M_j / (2*pi*Omega*Ntransverse)`.
Overlap matrices include the cell-volume normalization of JDFTx coefficients
and the reciprocal boundary shift of the periodic Bloch functions. SVD gives
the overlap phase and inverse without multiplying small determinants. The
ionic term is `sum_I Z_I R_I / Omega`, using pseudopotential valence charges.
String phases are unwrapped transversely at initialization and then advanced
only at accepted electronic iterates. References remain fixed throughout each
line search. Trials changing a coupled string phase by more than pi/2, or
making an adjacent overlap singular, return an invalid energy so the line
search can backtrack. This preserves a continuous enthalpy branch through
cumulative polarization changes beyond a quantum.
Initialize both field signs from the same zero-field occupied wavefunctions.

## Supported calculations

Use a neutral insulating bulk crystal, norm-conserving pseudopotentials,
constant fully occupied bands, periodic Coulomb interaction, and a single
uniform `kpoint-folding` mesh. All occupied spin states are included for
unpolarized, collinear, or spinor calculations. Space-group and k-inversion
reduction are disabled automatically, including at zero field, so restart
files use the same full mesh. Collinear calculations currently require equal
occupied band counts in both spin channels because every stored column must
be filled. Choose the band count to contain exactly the
occupied manifold; do not enable smearing.

At nonzero field use `electronic-minimize`, without `electronic-scf` or a
fixed Hamiltonian. Variational minimization is self-consistent: Hartree and
XC screening are included. Conventional density mixing remains available
for zero-field runs with `berry-phase-field 0 0 0`.
Do not enable `converge-empty-states`: that postprocessing diagonalizes the
ordinary independent-k DFT Hamiltonian and would erase the polarized occupied
subspace. Combining this command with `perturb-minimize` or `phonon-dfpt` is
also rejected because their Hessians do not include finite-field Berry terms.

Internal-ion relaxation through `ionic-minimize` is supported at fixed
lattice vectors. For norm-conserving orbitals the Berry electronic term has
no explicit ionic-coordinate derivative at fixed plane-wave coefficients.
The DFT forces evaluated with field-polarized states plus the explicit ionic
`Z_I E` force are therefore the variational electric-enthalpy forces. The
ionic response includes the electronic polarization change as ions move.
No Born-charge approximation is added to the finite-field relaxation.

Ultrasoft/PAW overlaps and their force augmentation, fluid, stress, and
lattice relaxation are not supported by this command and are rejected.
Filled occupations alone do not establish that a crystal is insulating:
verify the zero-field gap independently. Keep fields weak enough to follow
the local insulating branch. Singular initial/final overlaps are rejected;
singular line-search trials can backtrack. Overlap nonsingularity alone does
not certify that a metastable finite-field minimum exists.
For larger polarization changes, follow a sequence of small fields and
explicitly track polarization branches; the supplied workflow targets the
weak-field derivative at zero.

## Run silicon

Build the modified executables from the repository root:

```bash
cmake -S jdftx -B build
cmake --build build -j 6
```

Prepare all three Cartesian directions, two field amplitudes, and both
clamped-ion and relaxed-ion calculations:

```bash
cd examples/berry-dielectric-silicon
python3 run.py --mesh 4 6 8 --fields 0.0002 0.0001 --relax-ions
```

Add `--run` to execute. Use `--threads N` and, optionally, `--ranks N` for
OpenMP/MPI, or `--gpu` to select `jdftx_gpu`. Use `--resume` to reuse converged
outputs with identical inputs. The default is PBE/SG15, a 30-Hartree cutoff, a two-atom diamond
cell with conventional lattice constant 10.26 bohr, and four occupied bands.
The independent zero-field density gap check stores eight bands; these
empty bands are used only for the check and never in electric enthalpy.
The runner checks a positive sampled gap and minimization convergence. It
also reports the PRL estimate `N_i * abs(E.a_i) / gap` and rejects fields for
which any direction reaches one. This is a heuristic, conservative check
using the sampled indirect gap, not a proof of stability; converge field size
and mesh together.

For an inexpensive initial calculation:

```bash
python3 run.py --work smoke --mesh 4 --cutoff 12 --axes z --relax-ions --run
```

`--axes z` computes the z column, without assuming cubic symmetry. Omit it
for the complete tensor. The default electronic gradient threshold is
`1e-9`; `--electronic-threshold` and `--force-threshold` allow explicit
convergence checks. Converge the cutoff and mesh as well as field amplitude;
a coarse Berry mesh substantially underestimates the silicon response.
The lattice-direction finite differences also have finite-mesh tensor
anisotropy, which should decrease with increasing mesh density.

`analyze.py runs/k4 runs/k6 runs/k8` reads the `.polarization` files, aligns
each electronic string phase to the zero-field branch before averaging, and
writes `dielectric.json`. A shift of one string contributes a fractional
polarization quantum, which cannot always be corrected by rounding the total
polarization alone. Older files without individual phases remain readable.
It reports the central-difference tensor and its diagonal independently
from enthalpy curvature, with the same branch correction applied to both.
When ion relaxation is enabled it also reports `epsilon_static - epsilon_electronic`.
Uncomputed tensor columns are `null` in JSON.

The `.polarization` output contains Cartesian electronic, ionic and total
polarization in electrons/bohr squared, the applied field, the field coupling,
internal DFT energy, electric enthalpy, minimum overlap singular value, occupied
spin weight, and individual unwrapped string phases and lattice directions.
The `EfieldBerry` energy component is included in JDFTx's minimized energy.
Ordinary dumped subspace eigenvalues are DFT Hamiltonian eigenvalues in the
occupied field-polarized subspace, not independently diagonalized finite-field
bands.

Ideal diamond silicon's linear lattice dielectric contribution vanishes by
symmetry, so its relaxed-ion static and clamped-ion electronic responses
should agree in the weak-field limit. The experimental value near 12 is a
comparison point, not a fitted target; PBE and discretization errors remain.
Measured results, convergence plots, and backend comparisons are in the
[validation report](VALIDATION.md).
The equation-by-equation PRL audit and independent Quantum ESPRESSO comparison
are in [PRL_AUDIT.md](PRL_AUDIT.md).

## Verification

```bash
ctest --test-dir ../../build -R berryPhase --output-on-failure
```

The regressions check Berry and full DFT orbital derivatives, independent
occupied-gauge rotations, electron translation across reciprocal boundaries,
neutral joint translations, finite-field ionic forces, spin occupation
factors, shifted meshes, a common plane-wave basis, and MPI distribution.
The silicon workflow regression checks symmetric field amplitudes,
polarization/enthalpy consistency, relaxed-ion response, input guards,
and citation output in Berry runs and its absence in ordinary calculations.

The implementation follows the supplied Berry-phase workflow and the
occupied-state electric-enthalpy method of
[Souza, Iniguez and Vanderbilt](https://doi.org/10.1103/PhysRevLett.89.117602)
([preprint](https://arxiv.org/abs/cond-mat/0205633)).
Occupied wavefunctions are currently broadcast to all MPI ranks for Berry
links, so memory per rank scales with the complete occupied mesh. This is
suitable for the silicon validation; large-cell scaling can be improved with
neighbor-only communication.
