#!/usr/bin/env bash
set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
build_dir=${JDFTX_BUILD:-"$repo_dir/build"}
pseudo=${AL_PSEUDO:-"$build_dir/pseudopotentials/SG15/Al_ONCV_PBE.upf"}
exe_suffix=${JDFTX_SUFFIX:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
for exe in jdftx wannier; do
    [[ -x "$build_dir/$exe$exe_suffix" ]] || { echo "Missing executable: $build_dir/$exe$exe_suffix" >&2; exit 1; }
done
[[ -f "$pseudo" ]] || { echo "Missing Al pseudopotential: $pseudo" >&2; exit 1; }
cd "$example_dir"
mkdir -p results
printf 'ion-species %s\n' "$pseudo" > results/pseudo.in
launcher=()
if [[ ${NP:-1} -gt 1 ]]; then launcher=(mpirun -np "$NP"); fi
: > results/scf.out
"${launcher[@]}" "$build_dir/jdftx$exe_suffix" -i 01-scf.in -o results/scf.out
python3 - <<'PY'
from pathlib import Path
import re
out = Path('results/scf.out').read_text()
if 'SCF: Converged' not in out:
    raise SystemExit('SCF did not converge; inspect results/scf.out before Wannierization.')
mu = float(re.findall(r'FillingsUpdate:\s+mu:\s+(\S+)', out)[-1])
ev = 1 / 27.211386245988
# Exclude the four semicore bands of the default 11-electron SG15 potential.
# Keep valence states through EF+5 eV exactly; optimize in a larger space.
inner = (mu-12*ev, mu+5*ev)
outer = (mu-12*ev, mu+25*ev)
Path('results/windows.in').write_text(
    f'# SCF Fermi energy: {mu:.12f} Hartree\n'
    f'wannier localizationMeasure TruncatedDC '
    f'outerWindow {outer[0]:.12f} {outer[1]:.12f} '
    f'innerWindow {inner[0]:.12f} {inner[1]:.12f} '
    'saveWfns no loadRotations no\n')
print(f'Fermi energy: {mu:.9f} Ha; wrote results/windows.in')
PY
: > results/wannier.out
"${launcher[@]}" "$build_dir/wannier$exe_suffix" -i 02-wannier.in -o results/wannier.out
printf '\nDone. See %s/results/wannier.out for centers and TDC spreads.\n' "$example_dir"
