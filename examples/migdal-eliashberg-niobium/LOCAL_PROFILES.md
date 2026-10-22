# Local superconducting profiles at 2 K

`plot_local_profiles.py` implements the site and bond diagnostics in §16 of
[`migdal_eliashberg_wannier_bdg_local_profiles.pdf`](../../references/migdal_eliashberg_wannier_bdg_local_profiles.pdf)
using the saved self-consistent vacancy solution. Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/plot_local_profiles.py --gpu-device 1
```

Use `--backend cpu` without CUDA, or `--reuse-correlations` to reuse the cached
correlations when adjusting plots. The default source is `results-vacancy-T2-final`;
`--results` selects another compatible saved calculation. No new SCF run is needed.

Outputs are in `results-vacancy-T2-final/local-profiles/`:

* `radial_profiles.png` and `.pdf`: pairing-field norm, onsite and nearest-neighbor
  pair-amplitude norms, and threshold-based apparent LDOS half-gaps.
* `spectral_profiles.png` and `.pdf`: shell-averaged LDOS heatmap and selected spectra.
* `spatial_profiles.png` and `.pdf`: three-dimensional site maps of pairing and
  anomalous-amplitude norms, with the missing site marked by a cross.
* `site-profiles.dat`, `bond-profiles.dat`, and `local-profiles.npz`: numerical profiles.
* `correlations.npz`: reconstructed full orbital correlation blocks and bulk references.
* `profiles.json`: conventions, source hashes, broadening, thresholds, and checks.

The site pairing diagnostic is sqrt(mean_a |Delta_ia|^2), divided by the bulk
gap. The anomalous amplitude is F_ia,jb = <c_jb,down c_ia,up>, reconstructed
from positive-energy BdG eigenvectors with tanh(E/2kT) and the supercell twist
weights. Periodic bonds include their Bloch image phases. The plots use full
6-by-6 onsite and nearest-neighbor Frobenius norms, divided by corresponding
bulk norms. There are 63 surviving sites and 248 undirected nearest-neighbor
bonds. These are Wannier-basis diagnostics; a continuum F(r,r') reconstruction
would additionally require spatial Wannier functions. In this onsite model,
diagonal F_ia,ia = Delta_ia/U_a; the off-diagonal and bond blocks provide
additional correlation information.

F is reported in the canonical quasiparticle basis. Its coherent electron
contribution F/Z0 is also exported. The saved LDOS is the coherent electron LDOS
per spin per surviving Nb site, with Lorentzian width eta = 0.05 meV.
The apparent half-gap averages the first positive and negative threshold
crossings at 5%, 10%, or 20% of each side's peak within 3.5 meV of zero.
Lorentzian tails move these crossings below the actual spectral edge;
they are neither the local pairing field nor new excitation energies.
At the 10% threshold, site half-gaps range from 1.397 to 1.405 meV;
the model's minimum positive excitation energy is 1.539 meV.

Radial shell bars show the spread among sites or bond midpoints, not statistical
uncertainty. The nonmonotonic profiles belong to a periodic 4-by-4-by-4 vacancy
array and inherit its finite-size and momentum-integration limitations; no
isolated-vacancy decay length is fitted. See [VACANCY_VALIDATION.md](VACANCY_VALIDATION.md).

Reconstruction reproduces saved onsite anomalous densities to 1.4e-18 and
singlet transpose symmetry to 3e-11. Analytic CPU and CUDA tests cover
normalization, periodic bond phases, bulk reference correlations, shell
grouping, and threshold extraction:

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/test/dfpt/testLocalBdGProfiles.py
OPENBLAS_NUM_THREADS=1 ME_TEST_GPU=1 ME_TEST_GPU_DEVICE=1 python jdftx/test/dfpt/testLocalBdGProfiles.py
```
