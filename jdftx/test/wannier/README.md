Truncated density-convolution Wannier localization
================================================

Select `wannier localizationMeasure TruncatedDC`. This implements Eq. (7)
of Li, Ko, DiStasio and Damle, PRB 110, 085127 (2024), rather than the full
real-space density-convolution integral. It uses the existing finite-difference
neighbor shells and overlap files. The default remains FiniteDifference.

For entangled bands, specify `outerWindow <min> <max>` and, optionally,
`innerWindow <min> <max>` (energies in Hartree). Include enough DFT bands
and Wannier centers to contain the inner window at every k point. For example:

```
wannier localizationMeasure TruncatedDC outerWindow -1 1 innerWindow -1 0
```

JDFTx jointly optimizes its rectangular subspace and the gauge, retaining
inner-window constraints, as in the variational approach of Damle, Levitt and
Lin, MMS 17, 167–191 (2019). No separate MV disentanglement pass is used.
This change does not implement the paper's SCDM initialization: existing trial
orbitals and compatible rotation checkpoints are used.

The objective is `2 sum_(b in half mesh) wb sum_n (1-|rho_n(b)|)`, with
`rho_n(b) = sum_k weight_k M_nn(k,b)`. Averaging precedes the modulus;
MPI ranks reduce these complex averages before evaluating the objective and
gradient. At an exactly zero average, the modulus is nonsmooth and the code
selects the zero subgradient. Existing finite-difference weights are reused;
the lower-bound interpretation assumes nonnegative weights.

Reported per-orbital spreads are TDC spreads, not MV variances or exact DC
spreads. Centers are phase-based estimates on branches near the trial centers;
for diffuse orbitals they can be ambiguous and are not variational DC centers.
`OmegaI` remains an MV subspace diagnostic and is not a TDC decomposition.
Pinned centers are rejected. Frozen Wannier orbitals and inner-window bands
continue to use the existing constraints. Position and position-momentum
matrix exports retain their existing finite-difference definitions.

Run `python3 jdftx/test/wannier/checkTruncatedDC.py build` from the repository
root for isolated- and entangled-band Si minimizations (with and without an inner
window), pinned-center rejection, and numerical gradient
checks. It uses a temporary directory and the build's SG15 silicon potential.
Set `WANNIER_MPI='mpirun -np 2'` to exercise distributed Wannier gradients.
