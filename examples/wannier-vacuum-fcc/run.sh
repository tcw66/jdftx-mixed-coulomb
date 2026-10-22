#!/usr/bin/env bash
set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
build_dir=${JDFTX_BUILD:-"$repo_dir/build-gpu"}
exe_suffix=${JDFTX_SUFFIX-_gpu}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
cd "$example_dir"
mkdir -p results
[[ ! -e results/measure-comparison ]] || { echo 'Move the existing comparison aside before repeating.' >&2; exit 1; }
python3 - <<'PY'
from pathlib import Path
import numpy as np
import re
# Exactly matches model.in's explicitly fixed fftbox.
shape = tuple(map(int, re.search(r'^fftbox\s+(\d+)\s+(\d+)\s+(\d+)', Path('model.in').read_text(), re.M).groups()))
np.zeros(shape,dtype=np.float64).tofile('results/zero.Vscloc')
Path('results/pseudo.in').write_text('# Empty cell: no pseudopotentials.\n')
Path('results/windows.in').write_text(
    'wannier localizationMeasure TruncatedDC '
    'innerWindow -0.000001 0.3 outerWindow -0.000001 1.5 '
    'saveWfns no saveWfnsRealSpace yes loadRotations no\n')
PY
launcher=()
if [[ ${NP:-2} -gt 1 ]]; then launcher=(mpirun -np "${NP:-2}"); fi
: > results/scf.out
"${launcher[@]}" "$build_dir/jdftx$exe_suffix" -i 01-free-electron.in -o results/scf.out
python3 compare_measures.py --build "$build_dir" --suffix="$exe_suffix" --np "${NP:-2}"
python3 inspect_centers.py
