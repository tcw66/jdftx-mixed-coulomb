# Occupied-Wannier polarization response on the 16³ silicon mesh

The four occupied, spin-degenerate Wannier centers give
**εzz = 12.655843** for the completed calculations with both ions and unit cell
relaxed at each field. The corresponding Berry-string result is **12.668898**.
Only the z column was calculated; these numbers are finite-mesh, finite-field
estimates rather than a fully converged dielectric tensor.

| Calculation | Wannier εzz | Berry εzz |
| --- | ---: | ---: |
| Clamped cell and ions | 12.655829 | 12.668863 |
| Fixed cell, relaxed ions | 12.655829 | 12.668863 |
| Relaxed cell and ions | 12.655843 | 12.668898 |

The fixed-cell comparison holds the zero-field lattice at **10.350505 bohr**
while allowing both Si ions to relax independently at each field sign. Both
SCF runs met the electronic gradient threshold `1e-10` and ionic threshold
`1e-7`; the largest final Cartesian force was `1.99e-8` Hartree/bohr.
The ions stayed at `(0,0,0)` and `(1/4,1/4,1/4)` to the dumped precision.
Independent Wannierization of these states met `|grad|_K < 5e-7` and gave
`epsilon_zz = 12.655829075`. Its difference from the clamped-ion result is
`-1.12e-7`, below the established localization precision. There is no resolved
ionic contribution in this ideal silicon cell. Allowing the cell to relax
changes the Berry response by only `3.51e-5` and the Wannier response by
`1.36e-5`. All three choices use the same zero-field reference geometry,
so this is a direct comparison of the structural constraints at 16³.

## States and polarization

The source SCF states are the converged PBE/SG15 calculations described in
[the cell-relaxation report](VALIDATION_CELL_16.md): a 30-Hartree cutoff,
4096 k points, four occupied bands, and Cartesian internal fields
Ez = ±0.0001 Hartree/(e bohr). Wannierization rotates only this filled
manifold and does not update the SCF state. Four unpinned Gaussian trial
orbitals start at the Si bond centers `(1/8,1/8,1/8)` and its three images
with one coordinate changed to `-3/8`.

For each final cell, polarization is evaluated independently from the centers:

```
P_el  = -2 sum_n r_n / Omega
P_ion =  4 sum_I R_I / Omega
P     = P_el + P_ion + R @ (-1,-1,-1) / Omega
epsilon_az = delta_az + 4*pi*[P_a(+h)-P_a(-h)]/(2*h)
```

Here `R` contains the actual final lattice vectors as columns. Center labels
and primitive-cell images are matched to the appropriate zero-field Wannier
reference before taking differences. Ionic charges are the Si pseudopotential
valence charges; the factor two counts the two electrons in each occupied
spatial Wannier function. The polarization branch follows the strained cell.
Every initial/final plane-wave G-vector membership was checked and found
unchanged, so the saved cell-relaxed wavefunctions can be read without a
cutoff-basis conversion.

The cell-relaxed polarization components in e/bohr² are:

| Field Ez | Px | Py | Pz |
| --- | ---: | ---: | ---: |
| −0.0001 | −1.59e−13 | −7.35e−13 | −9.27542889095e−5 |
| +0.0001 | −1.70e−13 | −7.34e−13 | +9.27542089982e−5 |

The Wannier transverse dielectric components have magnitude below 2e−9.
The zero-field residual polarization has magnitude below 9e−10 e/bohr².
Individual Wannier spreads are about 7.978 bohr². Full-precision centers,
spreads, ionic coordinates, final lattices, dipole terms and response columns
are stored in [validation-wannier-k16.json](validation-wannier-k16.json).

## Numerical checks and interpretation

All eight reference/field Wannier calculations met the native gradient
criterion `|grad|_K < 5e-7`; spread-based stopping was disabled. Fresh
calculations at `1e-6` changed εzz by 3.6e−7 for the clamped states and
5.1e−6 for the cell-relaxed states. The small cell-relaxed minus clamped
Wannier response, about 1.35e−5, should therefore not be interpreted beyond
the localization and SCF precision established here.

The cell-relaxed Wannier/Berry difference is 0.013056, or approximately
0.103% of εzz. It is consistent with the different finite-mesh estimators:
Wannier centers use diagonal overlap phases in the optimized gauge and
the Marzari–Vanderbilt Cartesian neighbor stencil, whereas the SCF enthalpy
uses determinant phases along primitive reciprocal strings. Exact equality
on a finite mesh is not required. The Berry z column has transverse
components approximately 0.004757, while the Wannier stencil retains cubic
symmetry to the precision above. Mesh refinement is needed to establish the
continuum limit. JDFTx documents the finite-difference localization measure's
quadratic error in mesh spacing and its alternative real-space measure in the
[official Wannier command reference](https://jdftx.org/CommandWannier.html).
The occupied-band Wannier construction follows
[Marzari and Vanderbilt, Phys. Rev. B 56, 12847 (1997)](https://arxiv.org/abs/cond-mat/9707145).

Four physical regression checks passed: neutral-cell translation invariance,
permuted/image-shifted centers and their polarization quantum, a known
dielectric response with different strained volumes at the two field signs,
and rejection of an incompatible cutoff basis. Resume checks reused all six
original outputs only after matching inputs and source-file SHA-256 identities;
the two fixed-cell ion-relaxed states were Wannierized independently.
A full-mesh GPU evaluation of the same converged Wannier rotations agreed
with the CPU centers within 2.3e−16 bohr and spreads within 4.7e−14 bohr².

## Reproduce

Build the updated `wannier` executable and retain the source states under
`results-cell-k16/k16`. From this example directory:

```bash
python3 wannier_response.py results-cell-k16/k16 --run --resume
python3 wannier_response.py results-cell-k16/k16 --analyze-only
python3 -m unittest -v test_wannier_response.py
```

The raw states and Wannier checkpoints remain in the ignored results directory;
the full-precision numerical report is committed. The new `.mlwfCenters`
dump avoids differencing the rounded center display in the text log.
