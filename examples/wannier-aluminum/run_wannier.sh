#!/usr/bin/env bash
# Rerun localization with the current windows and centers, preserving SCF.
set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
build_dir=${JDFTX_BUILD:-"$repo_dir/build"}
exe_suffix=${JDFTX_SUFFIX:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
cd "$example_dir"
launcher=()
if [[ ${NP:-1} -gt 1 ]]; then launcher=(mpirun -np "$NP"); fi
: > results/wannier.out
"${launcher[@]}" "$build_dir/wannier$exe_suffix" -i 02-wannier.in -o results/wannier.out
if [[ -s results/bandstruct.eigenvals && -s results/bandpath.dat ]]; then
    python3 compare_bands.py
fi
