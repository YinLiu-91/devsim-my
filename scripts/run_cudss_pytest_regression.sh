#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cudss_common.sh"
ROOT_DIR="${cudss_root_dir}"
cd "${ROOT_DIR}"

DEVSIM_SO="$(cudss_require_devsim_so)"
SOLVER_MODE="${SOLVER_MODE:-both}"

cudss_export_native_mt_defaults

python3 -m pytest testing/pytest \
  --solver-mode="${SOLVER_MODE}" \
  --devsim-so="${DEVSIM_SO}" \
  -q \
  "$@"
