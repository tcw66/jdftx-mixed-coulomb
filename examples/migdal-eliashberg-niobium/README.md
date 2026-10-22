# Wannier electron–phonon → isotropic ME → static Wannier BdG

The [embedded vacancy workflow](EMBEDDED_VACANCY.md) uses an infinite bulk
Green function and the spatial Nb Wannier orbitals to compute self-consistent
2 K vacancy fields and continuum LDOS/pair-amplitude maps.

This example implements the basic workflow in
[`migdal_eliashberg_wannier_bdg.pdf`](../../references/migdal_eliashberg_wannier_bdg.pdf),
with native JDFTx Wannier and phonon inputs. It calculates the Fermi-level DOS,
mode linewidths and couplings, α²F, cumulative λ, ωlog, ω2, Allen–Dynes Tc,
linearized ME Tc, nonlinear imaginary-axis Z, Δ and Φ, and the condensation
free energy. It exports both momentum-space and real-space **spinful Wannier
BdG matrices**. Dependencies: Python 3, NumPy and SciPy; no new DFT run is needed
when a matching Wannier electron–phonon export is available.

There are two distinct BdG paths:

* The isotropic path **assumes** local, orbital-independent s-wave singlet pairing
  in a real, nonmagnetic Wannier basis. A scalar α²F does not determine the orbital
  pairing texture. This path doubles the spatial orbitals for spin explicitly.
* `matrix-bdg` accepts a supplied matrix Nambu self-energy at zero frequency and
  its frequency metric. It implements the general coherent reduction in §13.4,
  including off-diagonal frequency derivatives and spin–orbit coupling when
  present in the supplied matrices. It does not solve matrix-valued Eliashberg
  equations or infer missing self-energies from the isotropic solution.

## Run bulk Nb

From the repository root, use the checked local configuration:

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/scripts/migdalEliashberg run \
  examples/migdal-eliashberg-niobium/nb.json
```

The checked configuration records the local Nb input location and the isolated
Wannier export used for validation. For another checkout, prepare a configuration
with the runner. It uses `03-wannier.in` from the input directory; refreshing
writes exports under the requested output directory and leaves the original
Nb files intact:

```bash
python examples/migdal-eliashberg-niobium/run.py \
  --data /home/tyler/Nb_DFPT --output /tmp/nb-me --refresh-wannier
```

Without `--refresh-wannier`, the runner uses the existing `wannier.*` files.
Couplings older than `phononHsub` are rejected: those exports can correspond to
a different electronic gauge or checkpoint. Do not bypass this by changing file
timestamps. `wannier-initial-state` must identify the checkpoint matching the
native phononHsub export. The refreshed Wannier input keeps the original trial
orbitals, band windows, full electronic mesh and phonon supercell.

Each stage can also be run separately, retaining previous expensive work:

```bash
python jdftx/scripts/migdalEliashberg spectrum examples/migdal-eliashberg-niobium/nb.json
python jdftx/scripts/migdalEliashberg solve examples/migdal-eliashberg-niobium/nb.json
python jdftx/scripts/migdalEliashberg bdg examples/migdal-eliashberg-niobium/nb.json
```

Edit `nb.json` to choose the k/q meshes, Gaussian electronic sigma in eV,
phonon floor, μ*, Coulomb and Matsubara cutoffs, Tc bracket and temperatures.
Paths are relative to the configuration file unless absolute. The chemical
potential is read from the DFPT log, in Hartree, and converted to eV; `mu_ev`
explicitly overrides it. `metadata_log` must describe the matching **no-spin**
calculation. The electronic and phonon supercells here are the native export
meshes; `kmesh` and `qmesh` are the independent post-processing integration meshes.

The meshes and smearings in this example are **validation settings, not a
converged prediction for niobium**. Compare k meshes, q meshes and sigma, and
then vary μ*, the two frequency cutoffs, low-temperature Matsubara sampling and
the number of fit points. A failed Tc bracket is reported explicitly; extend
the bracket and, if necessary, `max_frequencies`, instead of treating the lower
endpoint as Tc. Matsubara matrices currently require O(Nfreq²) memory.

## GPU execution

An optional [CuPy](https://docs.cupy.dev/en/stable/install.html) backend accelerates
Wannier and phonon interpolation, eigensolves, weighted electron–phonon vertex
contractions, Fermi-surface sums, ME kernels and nonlinear iterations, coherent
BdG reduction, BdG eigenvalues and real-space FFTs. Computation remains in
float64/complex128. NumPy/SciPy remain required; CuPy is imported only when CUDA
or automatic selection is requested. CuPy 13.6.0 is already installed in the
local Python environment used for the Nb validation.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/scripts/migdalEliashberg run \
  examples/migdal-eliashberg-niobium/nb.json \
  --backend cuda --gpu-device 0 --chunk-size 256 \
  --output examples/migdal-eliashberg-niobium/results-gpu
```

The same flags work for separate `spectrum`, `solve` and `bdg` stages, and the
example `run.py`. For `matrix-bdg`, use `--backend cuda --gpu-device 0` after the
existing arguments. Configuration keys `backend`, `gpu_device` and `chunk_size`
provide the same controls; command-line values override them. CPU remains the
default. `--backend cuda` fails explicitly when CUDA is unavailable;
`--backend auto` may select CPU and records the fallback reason. CPU and GPU
stages can reuse each other's NPZ outputs with matching physical parameters.

One GPU is used per process. Static interpolation data are copied to that GPU
once; k-point batches bound temporary memory. Reduce `--chunk-size` if memory
is insufficient. The static vertex itself must fit in device memory. There is
no automatic multi-GPU distribution. The nonsymmetric linearized ME eigenproblem
uses CPU SciPy/ARPACK control with GPU matrix-vector products; very small
eigenproblems, scalar root finding, symmetry validation, low-frequency fitting,
and file I/O remain on CPU. Small ME solves may be faster with `--backend cpu`.

Backend, GPU identity and spectrum/ME elapsed times are recorded in the JSON
summaries. Timing covers backend/model initialization and the stage calculation,
including most output work before the summary is written. Device execution is
synchronized by the NumPy result transfer before recording its elapsed time.

For Nb at k=12³, q=4³ and batch size 256, a single matched test took **62.4 s on
CPU and 4.92 s on one RTX 4090** for the spectral stage, about 12.7× faster.
Individual mode couplings inside exactly degenerate phonon subspaces can rotate
between CPU and GPU eigensolvers. Compare subspace sums and spectral moments;
their individual sorted indices do not define a unique basis. See
[GPU validation](GPU_VALIDATION.md) for parity checks and the completed 8³ q run.

## Conventions and approximations

All internal and exported energies are **eV**; temperatures are K. The native
binary reader supports real or complex Hamiltonians/vertices and uses JDFTx's
column-major matrix entries, cell maps, and both sets of cell weights. Phonon
OmegaSq is already mass weighted. `mlwfHePh` is a mass-weighted Cartesian
derivative, so the phonon eigenvectors and `sqrt(1/(2*omega_Hartree))` are applied
once after interpolation. It must not be divided by the ionic mass again.

BZ weights are normalized to one; the DOS is per spin, per primitive cell, per
eV. The normalized Gaussian has standard deviation `sigma_ev`. No electronic
spin factor is inserted in the coupling numerator. `gamma_ev` follows Eq. (2)
of the note, the EPW half-width convention. Tiny acoustic modes below the
configured floor are excluded and counted. Imaginary modes exceeding the
configured tolerance stop the calculation.

The solver uses positive fermionic Matsubara frequencies, including both
λ(ωn−ωm) and λ(ωn+ωm), and a Coulomb cutoff on the **summed** frequency. An exact
constant-DOS normal-state Matsubara tail is added to Z; the nonlinear correction
and pairing sums retain the finite user cutoff. The leading linearized pairing
eigenvalue determines Tc. This is the isotropic Fermi-surface approximation with
particle–hole symmetric constant DOS, scalar μ*, and fixed harmonic phonons.

For the static isotropic model, a linear fit against ωn² of the first few
Matsubara Z and Φ values estimates Z0 and Φ0. Thus Δqp=Φ0/Z0, and the normal
Hamiltonian is also divided by Z0. This is an explicit **low-frequency estimate**,
not analytic continuation or an exact real-axis spectral gap. Use temperatures
where the fit window is small relative to relevant phonon scales; inspect fit
point and temperature sensitivity. `two_delta_fit_over_kbtc` uses that finite-T
estimate, not an asserted zero-temperature observable. Real-axis spectroscopy,
specific-heat differentiation, transport, and anisotropic/matrix ME iteration
are not implemented in this basic workflow.

The isotropic electron order is `(orb0 up, orb0 down, orb1 up, orb1 down, ...)`,
followed by the matching holes at −k. For Norb spatial Wannier orbitals, the full
Nambu dimension is 4*Norb. The pairing spin matrix is `i sigma_y`, and the hole
block is `-h_qp(-k).T`. A six-orbital Nb model therefore produces 24×24 matrices.
The real Wannier time-reversal condition is checked before assuming local pairing.

The global division of normal-state energies by Z0 defines a low-energy coherent
model. It is intended near the Fermi level, not to reproduce bands far from it.
The particle–hole symmetric isotropic approximation takes the static normal
self-energy shift to be zero; a supplied matrix self-energy can include it.

## Outputs and use in a BdG solver

* `spectrum.npz`: k/q points, frequencies, positive spectral-line weights,
  λqν, γqν, electronic sigma, μ, per-spin DOS, excluded-mode count.
* `a2f.dat`: binned α²F for plotting and exact cumulative spectral-line λ.
  The ME solver uses the discrete lines, avoiding low-energy histogram moment errors.
* `spectral-summary.json`: moments, Allen–Dynes estimate, sorted-mode contributions,
  configuration and input file provenance. Sorted eigenvalue indices are not
  phonon branches tracked continuously through crossings.
* `me-summary.json`, `me-T*.npz`: Tc, positive Matsubara frequencies, Z, normal Z,
  Δ, Φ, residuals, free energy and low-frequency estimates.
* `bdg-T*.npz`: `kpoints`, `bdg_ev`, `eigenvalues_ev`, `h_qp_ev`, `delta_qp_ev`,
  `cells`, `hopping_ev`, `pairing_ev`, residue square root, basis order, μ and T.
* `bdg-T*-checks.json`: Hermiticity, particle–hole, pairing antisymmetry and
  real-space conjugation residuals.

For the real-space export, `h_qp(k) = sum_R exp(+2*pi*i*k.R)*hopping_ev[R]`,
and the same convention holds for `pairing_ev`. Both arrays are in the original
orbital order, with explicit spin indices. R is a signed integer lattice vector.
The discrete Fourier export uses a complete unshifted grid and represents a
periodic supercell; −R at Nyquist boundaries is interpreted modulo that supercell.
It does not assume the real-space pairing range beyond the selected mesh.

```python
import numpy as np
model = np.load('results-nb/bdg-T2.npz')
R = model['cells']
t = model['hopping_ev']
pairing = model['pairing_ev']
Hk = model['bdg_ev']
```

## BdG band structure

Compute a saved temperature's bands along the BCC path Γ–H–N–Γ–P–H | P–N:

```bash
OPENBLAS_NUM_THREADS=1 python examples/migdal-eliashberg-niobium/plot_bands.py \
  --results examples/migdal-eliashberg-niobium/results-gpu \
  --temperature 2 --backend cuda --gpu-device 0
```

The plotter reads the configuration and coherent parameters from the saved
result directory, so later edits to `nb.json` do not change the input model.
It evaluates the original Wannier Hamiltonian at the path points, using the
saved Z0, Φ0 and μ. It checks that the source export and saved ME/BdG solution
match. It does not interpolate the finite 8³ BdG export mesh. This preserves
the longer-range normal-state hoppings from the original Wannier export.

Sampling starts with 300 intervals per segment, then finds normal-state Fermi
crossings by root finding and adds dense samples around each crossing. The
outputs under `results-gpu/bands-T2/` are PNG/PDF plots, `bands.dat` (distance,
fractional k and all 24 BdG energies in eV), `bands.npz` (also electron-sector
weights and renormalized normal-state energies), and `bands.json` (path,
crossings, source parameters and validation). The figures include a band overview,
a meV-scale view and a resolved avoided crossing. Coincident spin copies are
drawn once, but all 24 bands are exported.

The computed eigenvalues are checked against
`E = ±sqrt(((epsilon_n(k)-mu)/Z0)**2 + (Phi0/Z0)**2)`, in addition to Nambu
symmetries. They are bands of the static coherent singlet model; they do not
include real-axis damping, phonon satellites or energy-dependent residues.

## General matrix self-energy reduction

For a spatially varying, genuinely self-consistent static pairing calculation,
see [the Nb vacancy workflow](VACANCY.md). It fits an explicit onsite interaction
to the bulk ME solution and iterates the local anomalous densities after deleting
all orbitals at one Nb site. This is separate from the fixed-gap bulk exports
and the supplied-matrix coherent reduction below.

Create an NPZ file with these arrays in a single, consistently ordered spinful
Wannier gauge:

| Array | Shape | Meaning |
| --- | --- | --- |
| `kpoints` | `(Nk,3)` | Complete uniform Gamma-centered fractional mesh |
| `h_ev` | `(Nk,Nw,Nw)` | Spinful normal Hamiltonian, before subtracting μ |
| `mu_ev` | scalar | Chemical potential |
| `sigma0_ev` | `(Nk,2*Nw,2*Nw)` | Hermitian part of retarded Nambu self-energy at zero |
| `metric` | `(Nk,2*Nw,2*Nw)` | `I - d_omega Hermitian(Sigma^R)(0)` |

```bash
python jdftx/scripts/migdalEliashberg matrix-bdg self-energy.npz bdg.npz --mesh 8 8 8
```

The script requires a positive definite metric and consistent Nambu symmetries,
then evaluates `S (H0 + Sigma0) S` with `S=metric**(-1/2)` via Hermitian
eigendecomposition. Large symmetry violations are errors; only roundoff is
projected out. Hermitian part means `(Sigma + Sigma.dagger)/2`, preserving
complex orbital phases; it is not an elementwise real part. This path requires
weak damping for the physical static interpretation. It cannot test damping
because the supplied matrices contain only the Hermitian low-energy data.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 python jdftx/test/dfpt/testMigdalEliashberg.py
```

To require the additional CUDA checks (they fail if CUDA is unavailable):

```bash
OPENBLAS_NUM_THREADS=1 ME_TEST_GPU=1 python jdftx/test/dfpt/testMigdalEliashberg.py
```

The tests check an Einstein spectrum, signed-frequency vs positive-frequency ME
equations and eigenvalues, nonlinear residuals, scale covariance, native binary
layout and weighted-cell interpolation, zero-point normalization, full Nambu
gauge covariance, singlet poles, Fourier round trips and rejection of invalid
metrics or pairing matrices. CMake registers these as `migdalEliashberg` when
NumPy and SciPy are available. The conventions also follow the primary
[EPW theory documentation](https://docs.epw-code.org/Theory.html).
