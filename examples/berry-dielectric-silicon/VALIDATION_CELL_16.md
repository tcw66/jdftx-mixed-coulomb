# Silicon 16×16×16 response with unit-cell relaxation

Completed October 14, 2026. PBE/SG15, 30-Hartree cutoff, two-atom diamond
silicon, four occupied spatial bands, all 4096 k points, and physical
Cartesian fields `E_z = ±0.0001` Hartree atomic units. The density, Hartree
potential and XC potential were minimized self-consistently at every
structural step. Cell shape, volume and internal positions were allowed to
relax at fixed Cartesian field and zero external mechanical stress.

| Calculation | εzz from polarization | εzz from enthalpy curvature |
|---|---:|---:|
| Clamped ions and cell | 12.668863 | 12.668674 |
| Relaxed ions, fixed cell | 12.668863 | 12.668674 |
| Relaxed ions and cell | **12.668898** | 12.668705 |

The fully relaxed response is **εzz ≈ 12.669**. The cell-relaxation change at
this finite field is `3.51e-5`; internal-ion relaxation produces no resolved
change. This agrees with the absence of a linear lattice/piezoelectric
dielectric contribution in ideal diamond silicon. Polarization and enthalpy
estimates agree within `1.94e-4`.

The zero-field conventional lattice constant is **10.350505 bohr = 5.477251 Å**,
with primitive-cell volume **277.220024 bohr³**. Both response references
have this same lattice. The initial cell was equilibrated, its cutoff basis
was regenerated, and `cell-zero` and both cell-relaxed field signs used the
same initial G-vector lists. The nonpolar branch was selected with
`berry-phase-branch -1 -1 -1` throughout.

Convergence and stability checks:

- Every reference and field calculation satisfied the electronic gradient
  threshold `1e-10`. Joint cell/ionic minimization satisfied the maximum
  preconditioned-gradient threshold `1e-6`; the fixed-cell ionic threshold
  was `1e-7`.
- Maximum zero-field residual stress was approximately **0.131 bar**. The
  maximum over either cell-relaxed field state was **0.439 bar**. Final
  Cartesian forces were checked independently against `1e-7` Hartree/bohr.
- The independent eight-band, fixed-density gap calculation converged at
  every k point, giving a sampled indirect gap of **0.608868 eV**. Empty
  bands were used only for this check. The maximum PRL mesh/gap estimate
  `N_i |E.a_i|/gap` was **0.370066**, below the heuristic threshold of one.
- Nine Berry regressions passed, including orbital and all six strain
  derivatives, branch/gauge invariance, CPU, CUDA, MPI, combined GPU/MPI,
  and the full cell-relaxation workflow. A separate fixed-state comparison
  on this 16³ mesh verified two-GPU/two-rank energy and polarization against
  the single-GPU reference to roundoff.
- Each Berry run printed the Souza–Iniguez–Vanderbilt PRL method citation.

The production field signs ran independently on the two RTX 4090 GPUs.
Electronic convergence was tightened after the initial very strict cell
criterion reached energy-roundoff limitations; the final calculations all
passed the stated criteria. Saved occupied seeds were converted using their
known initial-cell cutoff and then reconverged. No unconverged calculation
was included in the response analysis.

This is the **z column**, rather than a full tensor calculation. Finite-mesh
off-diagonal elements are approximately `εxz = εyz = 0.004757`. The result
is a finite-mesh, finite-field PBE value, not an extrapolated dielectric
constant. The earlier [mesh-convergence report](VALIDATION.md) used a fixed
10.26-bohr conventional lattice constant, so its values are not a controlled
mesh-only comparison with this relaxed-cell result. The experimental value
near 12 was not used to tune any parameter.

The compact [numerical record](validation-cell-k16.json) contains all response
estimates, final lattice matrices, polarizations, energies, forces, stresses,
electronic convergence values and the GPU/MPI comparison. Raw inputs and
outputs remain in `results-cell-k16/k16/` locally.

Reproduce from fresh initialization with:

```bash
JDFTX_CACHE_SIZE=4096 python3 run.py --work results-cell-k16 --mesh 16 \
  --fields 0.0001 --axes z --relax-cell --lattice-threshold 1e-6 \
  --electronic-threshold 1e-10 --gpu --threads 4 --run --resume
```

The runner executes field signs sequentially by default. Independent signs
can run on separate GPUs after the common references and gap check converge.
