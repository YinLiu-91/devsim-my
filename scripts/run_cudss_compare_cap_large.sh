#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cudss_common.sh"
ROOT_DIR="${cudss_root_dir}"
cd "${ROOT_DIR}"

DEVSIM_SO="$(cudss_require_devsim_so)"
CAP2D_LARGE_MESH_SCALE="${CAP2D_LARGE_MESH_SCALE:-0.1}"

cudss_export_native_mt_defaults

export CAP2D_LARGE_MESH_SCALE

python3 -m pytest testing/pytest/test_cudss_compare.py \
  --case-filter='examples/capacitance/cap2d_large' \
  --solver-mode=both \
  --strict-cudss \
  --devsim-so="${DEVSIM_SO}" \
  -q
