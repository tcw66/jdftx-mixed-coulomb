#!/usr/bin/env bash
set -euo pipefail
if (( $# )); then
  echo 'Set options in config.sh or environment variables; run_all.sh accepts no arguments.' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/workflow.sh" all
"$script_dir/workflow.sh" check
