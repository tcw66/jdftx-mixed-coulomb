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
[[ ! -e results/measure-comparison ]] || { echo 'Existing comparison found; move it aside before repeating.' >&2; exit 1; }
pseudo=${C_PSEUDO:-"$build_dir/pseudopotentials/SG15/C_ONCV_PBE.upf"}
[[ -f "$pseudo" ]] || { echo "Missing C pseudopotential: $pseudo" >&2; exit 1; }
printf 'ion-species %s\n' "$pseudo" > results/pseudo.in
launcher=()
if [[ ${NP:-2} -gt 1 ]]; then launcher=(mpirun -np "${NP:-2}"); fi
: > results/scf.out
"${launcher[@]}" "$build_dir/jdftx$exe_suffix" -i 01-scf.in -o results/scf.out
python3 - <<'PY'
from pathlib import Path
import re
out = Path('results/scf.out').read_text()
if 'SCF: Converged' not in out:
    raise SystemExit('SCF did not converge; inspect results/scf.out.')
mu = float(re.findall(r'FillingsUpdate:\s+mu:\s+(\S+)',out)[-1])
ev = 1/27.211386245988
# All occupied sigma/pi bands plus the low-lying pi* branch are frozen.
Path('results/windows.in').write_text(
    f'# SCF Fermi energy {mu:.12f} Ha\n'
    f'wannier localizationMeasure TruncatedDC '
    f'outerWindow {mu-25*ev:.12f} {mu+25*ev:.12f} '
    f'innerWindow {mu-25*ev:.12f} {mu+2*ev:.12f} '
    'saveWfns no loadRotations no\n')
print(f'Graphene SCF Fermi energy: {mu:.9f} Ha')
PY
python3 compare_measures.py --build "$build_dir" --suffix="$exe_suffix" --np "${NP:-2}"
