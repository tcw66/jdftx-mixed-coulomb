# Source this file from Bash to select the CPU and GPU executables.
_jdftx_build_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build"
case "${PATH-}" in
  "$_jdftx_build_path":*) ;;
  *) export PATH="$_jdftx_build_path${PATH:+:$PATH}" ;;
esac
unset _jdftx_build_path
