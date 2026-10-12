#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
repo_dir=$(readlink -f -- ../..)
dry_run=false
from_scf=false
ki_only=false
backend=gpu
for option in "$@"; do
  case "$option" in
    --dry-run) dry_run=true ;;
    --from-scf) from_scf=true ;;
    --ki-only) ki_only=true ;;
    --cpu) backend=cpu ;;
    --help|-h)
      cat <<'HELP'
Usage: ./run.sh [--dry-run] [--from-scf] [--ki-only] [--cpu]
  Run separate KI valence/conduction calculations with a no-spin ground state,
  then calculate direct PBE bands and plot both on the same axes.
  Reuse the matching SCF state and conduction seeds when present.
  --from-scf  Archive the current state/results/logs and start a fresh SCF run.
  --ki-only   Run only the valence/conduction KI stages, using the saved SCF.
  --dry-run   Validate selected native inputs; keep checkpoints unchanged.
  --cpu       Select build/jdftx and build/wannier instead of their _gpu variants.
Environment: JDFTX_BINARY, WANNIER_BINARY, MPIEXEC, MPI_RANKS (default 2),
             WANNIER_THREADS (default 4), JDFTX_PSEUDO_DIR, PYTHON (default python3).
HELP
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$option" >&2; exit 2 ;;
  esac
done
if $from_scf && $ki_only; then
  printf -- '--from-scf and --ki-only select different workflows.\n' >&2
  exit 2
fi
if [[ $backend == cpu ]]; then
  jdftx_binary=${JDFTX_BINARY:-$repo_dir/build/jdftx}
  wannier_binary=${WANNIER_BINARY:-$repo_dir/build/wannier}
else
  jdftx_binary=${JDFTX_BINARY:-$repo_dir/build/jdftx_gpu}
  wannier_binary=${WANNIER_BINARY:-$repo_dir/build/wannier_gpu}
fi
ranks=${MPI_RANKS:-2}
threads=${WANNIER_THREADS:-4}
[[ $ranks =~ ^[1-9][0-9]*$ && $threads =~ ^[1-9][0-9]*$ ]] || {
  printf 'MPI_RANKS and WANNIER_THREADS must be positive integers.\n' >&2; exit 2;
}
(( ranks >= 2 )) || {
  printf 'ki-response.in requests two q groups; use MPI_RANKS>=2.\n' >&2; exit 2;
}
for binary in "$jdftx_binary" "$wannier_binary"; do
  [[ -x $binary ]] || { printf 'Missing executable: %s\n' "$binary" >&2; exit 1; }
done
# Direct native launches do not take the workflow lock. Catch them before
# moving logs or writing checkpoints in the same working directory.
while read -r native_pid; do
  [[ $(readlink -- "/proc/$native_pid/cwd" 2>/dev/null || true) != "$PWD" ]] || {
    printf 'A native calculation is already running here (PID %s); finish or stop it before launching another run.\n' "$native_pid" >&2
    exit 1
  }
done < <(pgrep -x 'wannier(_gpu)?|jdftx(_gpu)?' || true)
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=1
export JDFTX_FFT_BATCH_SIZE=${JDFTX_FFT_BATCH_SIZE:-32}
export JDFTX_PSEUDO_DIR=${JDFTX_PSEUDO_DIR:-$repo_dir/build/pseudopotentials}
launch=("${MPIEXEC:-mpirun}" -n "$ranks" --map-by "slot:PE=$threads" --bind-to core)
mkdir -p state results logs
exec 9>.run.lock
flock -n 9 || { printf 'Another calculation is using this directory.\n' >&2; exit 1; }
scratch=$(mktemp -d "${TMPDIR:-/tmp}/koopmans-input-check.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

run_native() {
  local binary=$1 input=$2 label=$3 native_input=$2
  local options=() log="logs/$label.out"
  if $dry_run; then
    options=(--dry-run)
    log="logs/$label.dry-run.out"
    native_input="$scratch/$label.in"
    # Wannier setup writes sidecars even in a native dry run.
    awk -v prefix="$scratch/$label." '
      $1 == "dump-name" || $1 == "wannier-dump-name" { print $1 " " prefix "$VAR"; next }
      { print }
    ' "$input" > "$native_input"
  fi
  printf '%s: %s\n' "$label" "$input"
  # Native output files are opened in append mode. Preserve the previous log
  # so the active log describes only this invocation, including failed runs.
  if [[ -e $log ]]; then
    mkdir -p archive/logs
    mv -- "$log" "archive/logs/$label-$(date +%Y%m%d-%H%M%S-%N).out"
  fi
  "${launch[@]}" "$binary" -c "$threads" "${options[@]}" -i "$native_input" -o "$log"
}

have_scf=true
for variable in wfns eigenvals n Vscloc; do
  [[ -s state/scf.$variable ]] || have_scf=false
done
if ! $ki_only && ! $dry_run && { $from_scf || ! $have_scf; }; then
  archive="archive/run-$(date +%Y%m%d-%H%M%S-%N)"
  mkdir -p -- "$archive"
  cp -- model.in common.in ki-response.in ki-screening-mesh.in 01-scf.in 02-valence.in 02a-conduction-seed.in 03-conduction.in bandstruct.in "$archive/"
  for directory in state results logs; do
    mv -- "$directory" "$archive/"
    mkdir -- "$directory"
  done
  printf 'Previous run archived in %s\n' "$archive"
fi
if $have_scf && ! $from_scf; then
  [[ -s state/model.sha256 ]] && sha256sum --check --status state/model.sha256 || {
    printf 'Saved SCF model does not match model.in/common.in; use --from-scf.\n' >&2
    exit 1
  }
fi
if $ki_only && ! $have_scf; then
  printf 'Missing saved SCF state; run ./run.sh --from-scf first.\n' >&2
  exit 1
fi
if ! $ki_only; then
  if $from_scf || ! $have_scf || $dry_run; then
    run_native "$jdftx_binary" 01-scf.in scf
    if ! $dry_run; then
      sha256sum model.in common.in > state/model.sha256
      cp -- model.in common.in state/
    fi
  else
    printf 'Reusing the matching no-spin SCF checkpoint.\n'
  fi
fi
run_native "$wannier_binary" 02-valence.in valence
if $from_scf || [[ ! -s state/conduction-seed.mlwfC || ! -s state/conduction-seed.mlwfC.header ]] || $dry_run; then
  run_native "$wannier_binary" 02a-conduction-seed.in conduction-seed
fi
run_native "$wannier_binary" 03-conduction.in conduction
if ! $ki_only; then
  run_native "$jdftx_binary" bandstruct.in bandstruct
  if ! $dry_run; then
    "${PYTHON:-python3}" plot_separate_bands.py
  fi
fi
if $dry_run; then
  printf 'Input checks passed. Logs: %s/logs/*.dry-run.out\n' "$PWD"
else
  printf 'Calculation outputs: %s/results/\n' "$PWD"
fi
