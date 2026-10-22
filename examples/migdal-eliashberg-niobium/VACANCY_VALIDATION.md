# Self-consistent vacancy result at 2 K — 2026-10-15

The reported result is `results-vacancy-T2-final/`: one Nb site removed from a
periodic 4³ primitive-cell array, with 63 surviving sites and 378 spatial
orbitals. All six orbitals at the vacancy were deleted in the normal-state
Wannier Hamiltonian before selecting the active pairing subspace. The bulk
chemical potential is fixed. See [the model definition](VACANCY.md).

## Bulk reference and fitted interaction

The completed 20³-q ME result was snapshotted while a separate 40³-q bulk run
continued. Its 2 K values are Z0=1.935969913, Δbulk=1.873860883 meV,
μ=17.735540971 eV. The normal Hamiltonian is `(h_W-mu)/Z0` and the normal-state
pairing window is ±27.447220802 meV.

A 128³ primitive-cell electronic mesh was used to fit the independent onsite
singlet attractions. In the original Wannier ordering:

| Orbital | U (eV) |
| --- | ---: |
| s | 3.196420420 |
| dz2 | 2.522380980 |
| dxy | 0.959807420 |
| dxz | 0.960996359 |
| dx2-y2 | 2.553213613 |
| dyz | 0.960996359 |

These are **effective model parameters fitted to the ME gap**, including the
chosen window and residue; they are not independently calculated screened
Coulomb or bare electron–phonon matrix elements. The homogeneous 2 K solution
reproduces the reference gap. Independent iteration from 1.2 times that gap
recovers it within 3.2e-9 eV.

Without further fitting, the homogeneous static model predicts Tc=12.3181 K,
versus the source ME Tc=11.2212 K. The average orbital gaps at 4 and 6 K are
approximately 1.86494 and 1.80154 meV, compared with ME values 1.85390 and
1.76113 meV. A static onsite model does not reproduce the full retarded ME
temperature dependence exactly.

## Vacancy solution

The final supercell Brillouin-zone mesh is 24³ (13,824 twists). Normal-state
diagonalization and orbital deletion use all native Wannier hopping ranges.
The largest projected spin-reduced BdG matrix is 26×26 at an individual twist;
the unprojected dimension is 756. Projection is an explicit part of the selected
low-energy model, with all interband pairing entries inside that subspace kept.

| Quantity | Value |
| --- | ---: |
| Mean onsite pairing field | 1.525536 meV |
| Nearest-neighbor-site mean field | 1.605578 meV |
| Minimum onsite field | 0.740016 meV |
| Maximum onsite field | 2.135457 meV |
| Minimum positive BdG excitation | 1.539258 meV |
| Mean field reduction from bulk reference | 18.59% |
| Grand potential relative to normal state, per supercell | −8.60735e-5 eV |

Some orbital fields exceed the bulk reference; the reduction above refers to
the mean, not uniform suppression of every orbital. Local pairing fields and
spectral gap edges are distinct quantities.

The starts 0.5, 1 and 1.5 times the bulk gap converged in 106, 97 and 107
iterations. Their largest mutual field difference was 6.7e-9 eV. The retained
solution has a maximum gap fixed-point residual of 8.9e-10 eV and a particle–hole
eigenvalue residual of 2.1e-16 eV. The model number changes by only 5.9e-9
electrons on a final gap update. The pristine folding check is accurate to
5e-12 eV. The normal reference has the same projected subspace and Hamiltonian,
so the condensation grand-potential comparison uses consistent counting.

At fixed μ the occupations are allowed to change. Their total change relative
to 63 copies of the homogeneous reference is −3.2594 canonical model electrons.
This is not an ab initio vacancy charge: missing ionic potential/relaxation,
Hartree screening, and incoherent spectral weight prevent that interpretation.

## Numerical and physical limits

Holding the final interaction fixed, the 16³ and 24³ twist meshes give mean
fields 1.513823 and 1.525536 meV (0.77% change), and minimum positive excitations
1.530199 and 1.539258 meV (0.59% change). The maximum change of an individual
onsite orbital field is 0.1169 meV. Thus the mean and lowest excitation are
more stable than every detail of the spatial/orbital profile.

The early 8³, 12³, 16³ and 24³ pilot runs used a coarser 32³ bulk calibration
and are retained separately. Their values must not be mixed with the final
128³-calibrated interaction when assessing twist convergence. Fitted U_a also
change between 64³ and 128³ bulk calibration meshes (up to roughly 9% for the s
channel), so the interaction fit itself should be refined for quantitative
material predictions.

No larger vacancy supercell has been tested. This remains a 1.5625% periodic
vacancy array; it is not an established isolated-vacancy limit. The physical
uncertainties also include the onsite-channel restriction, cutoff choice,
unrelaxed orbital-deletion approximation, fixed bulk residue and omitted Hartree
and defect phonon response. LDOS fine structure depends on the chosen 0.05 meV
broadening and k sampling; it has not been established as isolated bound states.

## Reproduce and inspect

```bash
# Create the bulk fit on the finer electronic mesh:
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy.py \
  --calibration-only --calibration-mesh 128 128 128 \
  --output examples/migdal-eliashberg-niobium/results-vacancy-T2-cal128 --resume

# Solve the vacancy with that fixed interaction:
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy.py \
  --twist-mesh 24 24 24 --calibration-mesh 128 128 128 \
  --fixed-calibration examples/migdal-eliashberg-niobium/results-vacancy-T2-cal128/pairing-calibration.json \
  --output examples/migdal-eliashberg-niobium/results-vacancy-T2-final --resume
```

Use GPU 1 by default, or pass `--gpu-device 0`. The final directory contains
`vacancy-T2.png`, `vacancy-T2.pdf`, `convergence.png`, `summary.json`, orbital/site
tables, and `vacancy-T2.npz`. The NPZ includes the converged projected BdG
matrices, eigenvalues and twists grouped by active dimension, plus gap fields,
anomalous densities, occupations and coherent LDOS. The companion
`vacancy-basis.npz` supplies their normal-to-Wannier transformations.

The analytic single-level, variational-gradient, pristine-folding, exact site
deletion and clean supercell/primitive-cell normalization checks pass on both
CPU and CUDA. CTest passes `selfConsistentBdG` and the existing
`migdalEliashberg` regressions.
