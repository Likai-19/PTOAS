#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# User-facing entrypoint: expand selected cases and collect cycle metrics.

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
SCRIPT_DIR="$(cd -- "$(dirname -- "$(realpath -- "${SCRIPT_PATH}")")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

kt_resolve_paths "${SCRIPT_PATH}"

OP=""
BACKEND=""
OUTPUT_ROOT="${KT_ROOT}/sim_outputs"
PYTHON_CMD="$(kt_default_python_cmd)"
CASE_FILTER=""
PARALLEL_SIM=0
JOBS=""
ENGINE="msprof"
declare -a REQUESTED_CASES=()
declare -a SUCCESS_CASE_DIRS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --op)
      [ $# -lt 2 ] && { echo "--op requires a value" >&2; exit 1; }
      OP="$2"
      shift 2
      ;;
    --backend)
      [ $# -lt 2 ] && { echo "--backend requires a value" >&2; exit 1; }
      BACKEND="$2"
      shift 2
      ;;
    --case)
      [ $# -lt 2 ] && { echo "--case requires a value" >&2; exit 1; }
      REQUESTED_CASES+=("$2")
      shift 2
      ;;
    --case-filter)
      [ $# -lt 2 ] && { echo "--case-filter requires a value" >&2; exit 1; }
      CASE_FILTER="$2"
      shift 2
      ;;
    --parallel-sim)
      [ $# -lt 2 ] && { echo "--parallel-sim requires 0 or 1" >&2; exit 1; }
      PARALLEL_SIM="$2"
      shift 2
      ;;
    --engine)
      [ $# -lt 2 ] && { echo "--engine requires a value" >&2; exit 1; }
      ENGINE="$2"
      shift 2
      ;;
    --jobs)
      [ $# -lt 2 ] && { echo "--jobs requires a value" >&2; exit 1; }
      JOBS="$2"
      shift 2
      ;;
    --output-root)
      [ $# -lt 2 ] && { echo "--output-root requires a value" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python-cmd)
      [ $# -lt 2 ] && { echo "--python-cmd requires a value" >&2; exit 1; }
      PYTHON_CMD="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: run_cycle.sh --op <kernel> --backend <backend> [options]

Recommended user-facing command for cycle measurement.

Options:
  --case <id>           Select one case. Repeatable.
  --case-filter <text>  Filter case ids by substring.
  --engine <name>       Cycle runner: msprof or cannsim. Default: msprof.
  --parallel-sim <0|1>  Run cannsim jobs sequentially or in parallel.
  --jobs <n>            Max parallel jobs when --parallel-sim=1.
  --output-root <dir>   Root directory for sim outputs.
  --python-cmd <cmd>    Python launcher for cannsim runs. Default: $(kt_default_python_cmd)
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[ -n "${OP}" ] || { echo "--op is required" >&2; exit 1; }
[ -n "${BACKEND}" ] || { echo "--backend is required" >&2; exit 1; }
[[ "${PARALLEL_SIM}" =~ ^[01]$ ]] || { echo "--parallel-sim must be 0 or 1" >&2; exit 1; }
[[ "${ENGINE}" =~ ^(msprof|cannsim)$ ]] || { echo "--engine must be msprof or cannsim" >&2; exit 1; }

OUTPUT_ROOT="$(realpath -m -- "${OUTPUT_ROOT}")"
mkdir -p "${OUTPUT_ROOT}/${OP}/${BACKEND}"

list_cases() {
  kt_run_python_cmd "${PYTHON_CMD}" "${KT_ROOT}/run.py" --op "${OP}" --workflow cycle --list-cases
}

declare -a CASES=()
if [ "${#REQUESTED_CASES[@]}" -gt 0 ]; then
  CASES=("${REQUESTED_CASES[@]}")
else
  while IFS= read -r case_id; do
    [ -n "${case_id}" ] || continue
    CASES+=("${case_id}")
  done < <(list_cases)
fi

if [ -n "${CASE_FILTER}" ]; then
  declare -a FILTERED_CASES=()
  for case_id in "${CASES[@]}"; do
    if [[ "${case_id}" == *"${CASE_FILTER}"* ]]; then
      FILTERED_CASES+=("${case_id}")
    fi
  done
  CASES=("${FILTERED_CASES[@]}")
fi

[ "${#CASES[@]}" -gt 0 ] || { echo "no cases matched the requested selection" >&2; exit 1; }

if [ "${PARALLEL_SIM}" -eq 1 ] && [ -z "${JOBS}" ]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi
if [ -n "${JOBS}" ] && ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 1
fi

run_one_case() {
  local case_id="$1"
  local case_dir="${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}"
  local log_file="${case_dir}/driver.log"
  local run_py="${KT_ROOT}/run.py"

  mkdir -p "${case_dir}"
  echo "==> ${ENGINE} case ${case_id}"
  if [ "${ENGINE}" = "msprof" ]; then
    "${KT_SCRIPT_DIR}/run_msprof.sh" \
      --output "${case_dir}/msprof" \
      "${run_py}" \
      -- \
      --op "${OP}" \
      --workflow cycle \
      --backend "${BACKEND}" \
      --case "${case_id}" \
      > "${log_file}" 2>&1
  else
    "${KT_SCRIPT_DIR}/run_sim.sh" \
      --output "${case_dir}" \
      --python-cmd "${PYTHON_CMD}" \
      "${run_py}" \
      -- \
      --op "${OP}" \
      --workflow cycle \
      --backend "${BACKEND}" \
      --case "${case_id}" \
      > "${log_file}" 2>&1
  fi
}

report_cycles() {
  if [ "${#SUCCESS_CASE_DIRS[@]}" -eq 0 ]; then
    return 0
  fi
  if [ ! -f "${KT_SCRIPT_DIR}/helpers/report_cycles.py" ]; then
    return 0
  fi

  report_args=(--op "${OP}")
  if [ "${#SUCCESS_CASE_DIRS[@]}" -gt 1 ]; then
    report_args+=(--table)
  fi
  report_args+=("${SUCCESS_CASE_DIRS[@]}")

  echo "==> analyze cycles"
  kt_run_python_cmd "${PYTHON_CMD}" "${KT_SCRIPT_DIR}/helpers/report_cycles.py" "${report_args[@]}"
}

FAILED=0
declare -a FAILED_CASES=()

if [ "${PARALLEL_SIM}" -eq 0 ]; then
  for case_id in "${CASES[@]}"; do
    if ! run_one_case "${case_id}"; then
      FAILED=1
      FAILED_CASES+=("${case_id}")
      echo "FAIL case=${case_id} log=${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}/driver.log" >&2
    else
      SUCCESS_CASE_DIRS+=("${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}")
      echo "PASS case=${case_id} log=${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}/driver.log"
    fi
  done
else
  declare -A PID_TO_CASE=()
  active_jobs=0

  handle_finished_job() {
    local finished_pid="$1"
    local job_rc="$2"
    local case_id="${PID_TO_CASE[${finished_pid}]}"

    unset 'PID_TO_CASE[$finished_pid]'
    active_jobs=$((active_jobs - 1))

    if [ "${job_rc}" -ne 0 ]; then
      FAILED=1
      FAILED_CASES+=("${case_id}")
      echo "FAIL case=${case_id} log=${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}/driver.log" >&2
    else
      SUCCESS_CASE_DIRS+=("${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}")
      echo "PASS case=${case_id} log=${OUTPUT_ROOT}/${OP}/${BACKEND}/${case_id}/driver.log"
    fi
  }

  for case_id in "${CASES[@]}"; do
    while [ "${active_jobs}" -ge "${JOBS}" ]; do
      finished_pid=""
      if wait -n -p finished_pid; then
        job_rc=0
      else
        job_rc=$?
      fi
      handle_finished_job "${finished_pid}" "${job_rc}"
    done

    run_one_case "${case_id}" &
    pid=$!
    PID_TO_CASE["${pid}"]="${case_id}"
    active_jobs=$((active_jobs + 1))
  done

  while [ "${#PID_TO_CASE[@]}" -gt 0 ]; do
    finished_pid=""
    if wait -n -p finished_pid; then
      job_rc=0
    else
      job_rc=$?
    fi
    handle_finished_job "${finished_pid}" "${job_rc}"
  done
fi

echo "SUMMARY op=${OP} backend=${BACKEND} total=${#CASES[@]} failed=${#FAILED_CASES[@]}"
report_cycles
if [ "${FAILED}" -ne 0 ]; then
  printf 'FAILED_CASES %s\n' "${FAILED_CASES[*]}" >&2
  exit 1
fi
