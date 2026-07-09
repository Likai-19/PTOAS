#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# User-facing entrypoint: generic msprof transport for one Python entry.

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
SCRIPT_DIR="$(cd -- "$(dirname -- "$(realpath -- "${SCRIPT_PATH}")")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

kt_resolve_paths "${SCRIPT_PATH}"

OUTPUT_DIR=""
SCRIPT_FILE=""
declare -a SCRIPT_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --output)
      [ $# -lt 2 ] && { echo "--output requires a value" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: run_msprof.sh [options] <script.py> [-- <script args...>]

Generic msprof transport for one Python entrypoint.
Kernel-aware case selection should stay in the Python CLI or run_cycle.sh.

Options:
  --output <dir>        Final output directory for this msprof run.
EOF
      exit 0
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
      ;;
    *)
      if [ -z "${SCRIPT_FILE}" ]; then
        SCRIPT_FILE="$1"
        shift
      else
        SCRIPT_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

[ -n "${SCRIPT_FILE}" ] || { echo "<script.py> is required" >&2; exit 1; }

SCRIPT_FILE="$(realpath -m -- "${SCRIPT_FILE}")"
[ -f "${SCRIPT_FILE}" ] || { echo "script not found: ${SCRIPT_FILE}" >&2; exit 1; }

if [ -z "${OUTPUT_DIR}" ]; then
  SCRIPT_STEM="$(basename -- "${SCRIPT_FILE}")"
  SCRIPT_STEM="${SCRIPT_STEM%.py}"
  OUTPUT_DIR="${KT_ROOT}/sim_outputs/${SCRIPT_STEM}/msprof"
fi
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
MSPROF_WRAPPER="${KT_REPO_ROOT}/scripts/run_sim.sh"

mkdir -p "${OUTPUT_DIR}"

kt_source_ascend_env

echo "============================================"
echo "msprof run: script=${SCRIPT_FILE}"
echo "output: ${OUTPUT_DIR}"
echo "============================================"

"${MSPROF_WRAPPER}" \
  --output "${OUTPUT_DIR}" \
  --repo-root "${KT_REPO_ROOT}" \
  "${SCRIPT_FILE}" \
  -- \
  "${SCRIPT_ARGS[@]}"
