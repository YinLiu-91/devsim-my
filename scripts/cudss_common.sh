#!/usr/bin/env bash

cudss_root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cudss_default_devsim_so() {
  printf '%s\n' "${DEVSIM_SO:-${cudss_root_dir}/linux_x86_64_release/src/main/devsim_py3.so}"
}

cudss_require_devsim_so() {
  local so_path="${1:-$(cudss_default_devsim_so)}"
  if [[ ! -f "${so_path}" ]]; then
    echo "DEVSIM_SO not found: ${so_path}" >&2
    return 1
  fi
  printf '%s\n' "${so_path}"
}

cudss_export_native_mt_defaults() {
  export DEVSIM_CUDSS_BACKEND_POLICY="${DEVSIM_CUDSS_BACKEND_POLICY:-native}"
  export DEVSIM_CUDSS_MT_MODE="${DEVSIM_CUDSS_MT_MODE:-1}"
  export DEVSIM_CUDSS_USE_STREAM="${DEVSIM_CUDSS_USE_STREAM:-0}"
  export DEVSIM_CUDSS_RESULT_MODE="${DEVSIM_CUDSS_RESULT_MODE:-device_experimental}"
  export DEVSIM_CUDSS_DIRECT_SOLVER="${DEVSIM_CUDSS_DIRECT_SOLVER:-cudss}"
}
