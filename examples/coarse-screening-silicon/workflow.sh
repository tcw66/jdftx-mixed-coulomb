#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.sh"
stage="${1:-prepare}"
if (( $# )); then shift; fi
read -r -a coarse_grid <<< "$COARSE_GRID"
if [[ ${#coarse_grid[@]} -ne 3 ]]; then
  echo 'COARSE_GRID must contain three positive integers.' >&2
  exit 2
fi
for size in "${coarse_grid[@]}"; do
  if [[ ! "$size" =~ ^[1-9][0-9]*$ ]]; then
    echo 'COARSE_GRID must contain three positive integers.' >&2
    exit 2
  fi
done
args=(--example-dir "$SOURCE_DIR" --directory "$OUTPUT_DIR"
      --coarse-grid "${coarse_grid[@]}" --ranks "$RANKS" --threads "$THREADS"
      --minimum-density-overlap "$MINIMUM_DENSITY_OVERLAP" --scf-log "$SCF_LOG")
case "$BACKEND" in
  cpu) args+=(--cpu) ;;
  gpu) ;;
  *) echo 'BACKEND must be cpu or gpu.' >&2; exit 2 ;;
esac
[[ -z "$EPSILON_INFINITY" ]] || args+=(--epsilon-infinity "$EPSILON_INFINITY")
[[ -z "$ORBITAL_MAP" ]] || args+=(--orbital-map "$ORBITAL_MAP")
case "$stage" in
  prepare) args+=(--prepare-only) ;;
  nscf|wannier|screening|hamiltonian) args+=(--resume --stop-after "$stage") ;;
  all|plot) args+=(--resume --plot) ;;
  check) exec python3 "$script_dir/check_results.py" "$OUTPUT_DIR" "$@" ;;
  *) echo 'Usage: workflow.sh prepare|nscf|wannier|screening|hamiltonian|plot|all|check [runner options]' >&2; exit 2 ;;
esac
exec python3 "$script_dir/../koopmans/run_coarse_screening.py" "${args[@]}" "$@"
