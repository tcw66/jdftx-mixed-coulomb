#!/usr/bin/env bash
set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
build_dir=${JDFTX_BUILD:-"$repo_dir/build"}
exe_suffix=${JDFTX_SUFFIX:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
cd "$example_dir"
for file in scf.wfns scf.fillings al.mlwfH al.mlwfCenters al.mlwfCellMap al.mlwfCellWeights pseudo.in windows.in; do
    [[ -s "results/$file" ]] || { echo "Missing results/$file; run SCF and Wannierization first." >&2; exit 1; }
done
launcher=()
if [[ ${NP:-1} -gt 1 ]]; then launcher=(mpirun -np "$NP"); fi
python3 compare_bands.py --prepare --steps "${BAND_STEPS:-20}"
: > results/export-potential.out
"${launcher[@]}" "$build_dir/jdftx$exe_suffix" -i export-potential.in -o results/export-potential.out
: > results/bandstruct.out
"${launcher[@]}" "$build_dir/jdftx$exe_suffix" -i 03-bandstruct.in -o results/bandstruct.out
python3 compare_bands.py
