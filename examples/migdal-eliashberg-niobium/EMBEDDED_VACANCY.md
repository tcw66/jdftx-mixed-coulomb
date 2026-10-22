# Isolated vacancy embedded in superconducting Nb

`vacancy_embedded.py` constructs the bulk Nambu Green function by Fourier
integration, deletes all six orbitals of the origin in both Nambu sectors,
and iterates local onsite singlet pairing changes inside a sphere. The bulk
environment remains coupled to this sphere through its Green function;
no hopping bonds are cut at the edge and there is no periodic vacancy array.
The finite k mesh still approximates a bulk Brillouin-zone integral.

The implementation follows the Schur-complement and T-matrix machinery in
`/home/tyler/defect_scattering/0D/wannier_scattering/{system,cavity,continuum}.py`.
It is self-contained in `me_postprocess/embedded.py`, so it does not require
changing or installing the TBG repository. The reduced singlet Nambu basis is
all up-spin electrons followed by all down-spin holes. A real, nonmagnetic
Wannier Hamiltonian is required. This path does not implement spin-orbit or
magnetic pairing.

## Run

```bash
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy_embedded.py \
  --mesh 128 --radius 6 --observation-radius 9 --gpu-device 1
```

The default prefix is `/home/tyler/Nb_DFPT/wannier.`; all Hamiltonian, rotation,
center and spatial-orbital exports should come from the same Wannier run.
The earlier 2 K bulk calibration supplies only the bulk chemical potential,
Z0 and target gap. Each run records hashes of the new Hamiltonian inputs,
rotation file, lattice, calibration, and spatial Wannier files.
Use `--backend cpu` without CUDA, `--skip-continuum` for pairing fields only,
and `--reuse` to reconstruct maps from a saved compatible solution.
Use a separate `--output` for each mesh, pairing radius or quadrature order.

## Pairing regularization

This embedded path uses the full-bandwidth static local attractive model,
with six positive interactions fitted independently to the same homogeneous
2 K ME gap: U_a = Delta_bulk / F_bulk,aa. Both calibration and embedding use
the same bulk integration mesh. It does **not** reuse the old U_a values:
the periodic solver's vacancy-dependent normal-state energy projector is a
different regularization. Simply transplanting those values would change the
bulk gap and would not reproduce that projected model. Compare qualitative
profiles across the two models with this difference in mind.

The bulk anomalous amplitude is integrated analytically over the band
eigenstates. Only the defect correction is summed over Matsubara frequencies:
F = F_bulk - 2 kBT sum_(n>=0) Re(G_eh - G_bulk,eh).
The positive sum uses exact low frequencies and Gaussian quadrature for the
discrete Matsubara measure in dyadic blocks, followed by the analytic leading
high-frequency tail. This is a resummed Dyson solution at every iteration,
not a first-order response calculation. The residual is max|U_a F_ia - Delta_ia|.
Bulk mu and Z0 remain fixed, and the present model has no Hartree update,
atomic relaxation or vacancy-induced phonon changes.

## Continuum fields

The new `mlwfC` and `mlwfC.header` reconstruct complex, volume-normalized
Wannier amplitudes on a three-dimensional FFT grid. The header's skew lattice
and the centered-supercell origin are retained. We check
V sum_G |C_aG|^2 = 1 and compare interpolation with direct Fourier sums.
The default grid doubles the minimal alias-free FFT size; increase
`--wannier-grid-factor` to check interpolation convergence.

Within a larger observation sphere, the fully perturbed Green function and
the full anomalous matrix include every orbital cross term. Projection gives
rho(r,E) = W(r) A_ee(E) W(r)^dagger / Z0 and
F_coh(r,r) = W(r) F W(r)^T / Z0. Pair amplitudes are not pairing potentials:
a continuum Delta(r,r') would require an interaction-kernel prescription.
The fields use bulk Wannier shapes, not defect-relaxed orbitals, and are
pseudowavefunction observables of the supplied Wannier subspace.

The Cartesian z=0 slice spans +/-2 angstrom by default. A cosine taper removes
the last 2 bohr of the localized Wannier cutoff (default 10 bohr). The
observation sphere must contain the whole slice plus that cutoff. Increase
both radii/cutoffs together to check spatial truncation. Orbital values at
the missing site are excluded exactly, but tails of neighboring Wannier
functions can still give nonzero continuum fields there.

## Outputs and convergence

* `solution.npz`: surviving cells, six onsite fields, full support F matrix,
  fitted interactions, residual history and Matsubara quadrature.
* `configuration.json`, `summary.json`: input hashes, model conventions and
  numerical checks. `site-profile.dat` gives orbital fields versus distance.
* `pairing-profile.png/.pdf`: radial pairing fields compared with bulk.
* `continuum.npz`: coherent local pair amplitudes and LDOS at 0, 1.5 and
  2.5 meV, plus pristine references on the same observation basis.
* `continuum-profiles.png/.pdf`: Cartesian pair-amplitude and LDOS-change maps.

The vacancy and pristine pair-amplitude panels share a color scale. Difference
maps use symmetric limits about zero. `continuum_figures` can replot the saved
`continuum.npz` arrays without any new Green-function calculation.

For a wider 9-by-9 angstrom slice that includes eight neighboring Nb sites in
the z=0 plane, reuse the same self-consistent solution and enlarge the
observation sphere to retain the Wannier tails at the viewing edges:

```bash
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/vacancy_embedded.py \
  --reuse --extent 4.5 --pixels 181 --observation-radius 12 \
  --continuum-output examples/migdal-eliashberg-niobium/results-vacancy-embedded-T2/wide-view
```

This preserves the original close-up and writes the new plots, arrays, and
summary under `wide-view/`. Hollow circles identify surviving atomic sites
in the slice; the cross identifies the vacancy. The pixel spacing remains
0.05 angstrom. The pairing-update radius stays at 6 angstrom; increasing the
viewing and observation regions does not change the solved onsite fields.

LDOS uses 0.05 meV Lorentzian broadening and is per spin, in eV^-1 bohr^-3.
F_coh is in bohr^-3. The saved maps include no incoherent spectral weight.
Finite k-mesh integration, the pairing-update radius, quadrature order,
Wannier interpolation, spatial cutoff and broadening are separate convergence
parameters. A small iterative residual alone does not establish an
isolated-vacancy profile converged with respect to all of them.

The checked 2 K run uses 128^3 bulk integration, a 6 angstrom pairing sphere
(58 surviving sites), and a 9 angstrom observation sphere (169 cells,
including the deleted origin). Its gap-equation residual is 7.4e-9 eV.
The orbital pairing fields span 1.633--2.225 meV, compared with the fitted
bulk gap of 1.874 meV. At fixed 5 angstrom pairing radius, refining the host
mesh from 64^3 to 128^3 changes the largest orbital field by 0.0259 meV.
At fixed 128^3 mesh, expanding the pairing radius from 5 to 6 angstrom
changes the largest field on shared inner sites by 0.0086 meV. The mean over
all sites changes its sampling population when this sphere grows and should
not be used directly as a radius-convergence measure.

Tests compare the cavity and Dyson operations to direct complex resolvents,
and the complete embedded self-consistent solution to direct thermal BdG
diagonalization of a reference chain. They also check the discrete Matsubara
sum, orbital interference, skew-lattice Fourier phases and physical Wannier
normalization. They run on CPU or CUDA:

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/test/dfpt/testEmbeddedBdG.py
OPENBLAS_NUM_THREADS=1 ME_TEST_GPU=1 ME_TEST_GPU_DEVICE=1 \
  python jdftx/test/dfpt/testEmbeddedBdG.py
```
