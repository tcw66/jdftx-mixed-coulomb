# Bulk Nb post-processing validation — 2026-10-15

Source inputs: `/home/tyler/Nb_DFPT`. The original Wannier export predated the
latest `phonon.phononHsub`. A new export was generated with the source
`03-wannier.in` and matching `phonon.$VAR` checkpoint, writing only to
`build/me-niobium-input/`. The Wannier run completed successfully. The six
spatial orbitals retain the source order: s, dz2, dxy, dxz, dx2-y2, dyz.
Native electronic mesh: 12³; native phonon supercell: 4³.

The post-processing ran with `nb.json`: k mesh 12³, q mesh 4³, Gaussian sigma
0.10 eV, μ*=0.13, Coulomb cutoff 0.20 eV, Matsubara cutoff 0.50 eV.
Three Gamma acoustic modes were excluded at a floor of 0.01 meV. No phonon
instabilities exceeded the 0.10 meV tolerance on the sampled mesh.

| Quantity | Value |
| --- | ---: |
| Per-spin DOS / (eV cell) | 0.496421 |
| λ | 0.708211 |
| ωlog | 18.4036 meV |
| ω2 | 19.2245 meV |
| Allen–Dynes Tc | 6.05850 K |
| Linearized ME Tc | 7.60868 K |
| Z0 estimate at 2 K | 1.70146 |
| Φ0 estimate at 2 K | 2.03202 meV |
| Static Δqp estimate at 2 K | 1.19428 meV |
| Δ(iω0) at 2 K | 1.19367 meV |

Nonlinear solutions converged at 2, 4 and 6 K, with gap fixed-point residuals
below 1e-10 eV. The 2 K solve retained 463 positive Matsubara frequencies.
The first four frequencies used in its low-frequency fit span 0.541–3.790 meV.
The corresponding finite-temperature fitted ratio `2 Δqp / (kB Tc)` is 3.64295.
This is not a real-axis or zero-temperature gap determination.

The 8³ BdG export has shape `(512,24,24)`. Momentum-space Hermiticity,
particle–hole symmetry, pairing antisymmetry, and both real-space conjugation
checks all had zero reported residual at double precision. Exports at all three
temperatures are under `results-nb/`, with JSON and NPZ files recording parameters
and provenance. Generated results and the fresh Wannier export are local artifacts
excluded from version control by the repository's existing ignore rules.

An independent comparison with the existing `/home/tyler/Nb_DFPT/WannierEph.py`
evaluated two non-grid k points at q=(0.21,0.07,0.13), using the fresh export.
The largest difference in the magnitudes of the band-resolved, zero-point
normalized vertices was 2.0e-13 eV.

A preliminary 8³ k / 2³ q calculation with the same sigma gave λ=0.338807 and
Allen–Dynes Tc=0.112839 K. Both meshes changed, so this comparison does not
isolate either convergence parameter; it shows why the reported 12³ / 4³
calculation must **not** be treated as a converged prediction for bulk Nb.
Separate k/q/smearing and Matsubara-cutoff convergence studies are still required
for quantitative material predictions.

The analytic/native-format regression suite passes, both directly and through
CTest (`migdalEliashberg`). It covers the numerical equations and general matrix
reduction independently of the physical Nb agreement.
