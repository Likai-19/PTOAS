#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# User-facing entrypoint: generic cannsim transport for one Python entry.

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
SCRIPT_DIR="$(cd -- "$(dirname -- "$(realpath -- "${SCRIPT_PATH}")")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

kt_resolve_paths "${SCRIPT_PATH}"

OUTPUT_DIR=""
WORK_DIR=""
PYTHON_CMD="$(kt_default_python_cmd)"
SOC="Ascend950"
SCRIPT_FILE=""
declare -a SCRIPT_ARGS=()

normalize_kernel_dir_args() {
  local -n normalized_ref="$1"
  shift

  local arg=""
  local value=""
  normalized_ref=()

  while [ $# -gt 0 ]; do
    arg="$1"
    case "${arg}" in
      --kernel-dir)
        [ $# -lt 2 ] && { echo "--kernel-dir requires a value" >&2; exit 1; }
        value="$(realpath -m -- "$2")"
        normalized_ref+=("${arg}" "${value}")
        shift 2
        ;;
      --kernel-dir=*)
        value="${arg#--kernel-dir=}"
        value="$(realpath -m -- "${value}")"
        normalized_ref+=("--kernel-dir=${value}")
        shift
        ;;
      *)
        normalized_ref+=("${arg}")
        shift
        ;;
    esac
  done
}

while [ $# -gt 0 ]; do
  case "$1" in
    --output)
      [ $# -lt 2 ] && { echo "--output requires a value" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --work-dir)
      [ $# -lt 2 ] && { echo "--work-dir requires a value" >&2; exit 1; }
      WORK_DIR="$2"
      shift 2
      ;;
    --soc)
      [ $# -lt 2 ] && { echo "--soc requires a value" >&2; exit 1; }
      SOC="$2"
      shift 2
      ;;
    --python-cmd)
      [ $# -lt 2 ] && { echo "--python-cmd requires a value" >&2; exit 1; }
      PYTHON_CMD="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: run_sim.sh [options] <script.py> [-- <script args...>]

Generic cannsim transport for one Python entrypoint.
Kernel-aware case selection belongs in the Python CLI or run_cycle.sh.

Options:
  --output <dir>        Final output directory for this simulator run.
  --work-dir <dir>      Workspace directory. Default: <output>/work
  --soc <name>          cannsim SoC name. Default: Ascend950
  --python-cmd <cmd>    Python launcher command. Default: $(kt_default_python_cmd)
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
  OUTPUT_DIR="${KT_ROOT}/sim_outputs/${SCRIPT_STEM}"
fi
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"

if [ -z "${WORK_DIR}" ]; then
  WORK_DIR="${OUTPUT_DIR}/work"
fi
WORK_DIR="$(realpath -m -- "${WORK_DIR}")"

normalize_kernel_dir_args SCRIPT_ARGS "${SCRIPT_ARGS[@]}"
if [ -n "${KERNEL_TEST_KERNEL_DIR:-}" ]; then
  export KERNEL_TEST_KERNEL_DIR
  KERNEL_TEST_KERNEL_DIR="$(realpath -m -- "${KERNEL_TEST_KERNEL_DIR}")"
fi

ENTRY_SCRIPT="${WORK_DIR}/run_sim_entry.sh"
ENTRY_COMMON="${WORK_DIR}/common.sh"
ACTIVE_LOG="${WORK_DIR}/cannsim.log"
EXIT_GRACE_S="${KERNEL_TEST_CYCLE_EXIT_GRACE_S:-5}"
SCRIPT_BASENAME="$(basename -- "${SCRIPT_FILE}")"
SCRIPT_PATTERN="${SCRIPT_BASENAME//./\\.}"

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"
cp -f "${KT_SCRIPT_DIR}/helpers/run_sim_entry.sh" "${ENTRY_SCRIPT}"
cp -f "${KT_SCRIPT_DIR}/helpers/common.sh" "${ENTRY_COMMON}"
chmod +x "${ENTRY_SCRIPT}"

ARGS_FILE="$(kt_write_args_file "${WORK_DIR}" "${SCRIPT_ARGS[@]}")"

cleanup() {
  if [ -n "${ARGS_FILE:-}" ] && [ -f "${ARGS_FILE}" ]; then
    rm -f "${ARGS_FILE}"
  fi
  rm -f "${ENTRY_SCRIPT}"
  rm -f "${ENTRY_COMMON}"
}
trap cleanup EXIT INT TERM

export KERNEL_TEST_PYTHON_CMD="${PYTHON_CMD}"
export KERNEL_TEST_ARGS_FILE="${ARGS_FILE}"
export KERNEL_TEST_WORK_DIR="${WORK_DIR}"

kt_source_ascend_env

echo "============================================"
echo "cannsim run: script=${SCRIPT_FILE}"
echo "output: ${OUTPUT_DIR}"
echo "============================================"

set +e
cannsim record \
  -o "${OUTPUT_DIR}" \
  -s "${SOC}" \
  --gen-report \
  "${ENTRY_SCRIPT}" \
  -u "${SCRIPT_FILE}" &
RECORD_PID=$!

MARKER_TS=""
USER_APP_TERM_SENT=0

while kill -0 "${RECORD_PID}" 2>/dev/null; do
  if [ -z "${MARKER_TS}" ] && kt_log_has_cycle_completion "${ACTIVE_LOG}"; then
    MARKER_TS="$(date +%s)"
    echo "==> detected framework cycle completion marker"
  fi

  if [ -n "${MARKER_TS}" ] && [ "${USER_APP_TERM_SENT}" -eq 0 ]; then
    NOW_TS="$(date +%s)"
    if [ $((NOW_TS - MARKER_TS)) -ge "${EXIT_GRACE_S}" ]; then
      echo "==> user app did not exit after ${EXIT_GRACE_S}s; terminating post-cycle process"
      kt_terminate_descendants_matching "${RECORD_PID}" "run_sim_entry\\.sh|${SCRIPT_PATTERN}"
      USER_APP_TERM_SENT=1
    fi
  fi

  sleep 1
done

wait "${RECORD_PID}"
RC=$?
set -e

kt_collect_cannsim_artifacts "${WORK_DIR}" "${OUTPUT_DIR}"
kt_handle_cannsim_exit "${RC}" "${OUTPUT_DIR}"
