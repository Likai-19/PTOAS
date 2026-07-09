#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# Internal helper library for kernel-test shell entrypoints.

kt_resolve_paths() {
  local invoked_path="${1:-${BASH_SOURCE[0]}}"

  KT_SCRIPT_PATH="$(realpath -- "${invoked_path}")"
  KT_SCRIPT_DIR="$(cd -- "$(dirname -- "${KT_SCRIPT_PATH}")" && pwd)"
  KT_ROOT="$(cd -- "${KT_SCRIPT_DIR}/.." && pwd)"
  KT_REPO_ROOT="$(cd -- "${KT_ROOT}/.." && pwd)"
}

kt_source_ascend_env() {
  local ascend_home_path
  ascend_home_path="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
  # shellcheck disable=SC1090
  source "${ascend_home_path}/bin/setenv.bash"
}

kt_default_python_cmd() {
  printf '%s\n' "${KERNEL_TEST_PYTHON_CMD:-python}"
}

kt_quote_words() {
  local word=""

  for word in "$@"; do
    printf '%q ' "${word}"
  done
}

kt_run_python_cmd() {
  local python_cmd="$1"
  shift

  local quoted_args=""
  quoted_args="$(kt_quote_words "$@")"
  eval "${python_cmd} ${quoted_args}"
}

kt_exec_python_cmd() {
  local python_cmd="$1"
  shift

  local quoted_args=""
  quoted_args="$(kt_quote_words "$@")"
  eval "exec ${python_cmd} ${quoted_args}"
}

kt_write_args_file() {
  local work_dir="$1"
  shift

  local args_file=""
  args_file="$(mktemp "${work_dir}/.kernel_test_args.XXXXXX")"
  printf '%s\0' "$@" > "${args_file}"
  printf '%s\n' "${args_file}"
}

kt_load_args_file() {
  local args_file="${1:-${KERNEL_TEST_ARGS_FILE:-}}"
  local arg=""

  KT_LOADED_ARGS=()
  if [ -z "${args_file}" ]; then
    return 0
  fi
  if [ ! -f "${args_file}" ]; then
    echo "args file not found: ${args_file}" >&2
    return 1
  fi

  while IFS= read -r -d '' arg; do
    KT_LOADED_ARGS+=("${arg}")
  done < "${args_file}"
}

kt_prepare_cannsim_workspace() {
  local work_dir="$1"

  mkdir -p "${work_dir}/log_ca"
  rm -f "${work_dir}/instr.bin"
  find "${work_dir}/log_ca" -mindepth 1 -delete 2>/dev/null || rm -rf "${work_dir}/log_ca"/* 2>/dev/null || true
  cd "${work_dir}"
}

kt_find_latest_log() {
  local output_dir="$1"

  find "${output_dir}" -name cannsim.log -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

kt_log_has_cycle_completion() {
  local log_path="$1"

  [ -f "${log_path}" ] || return 1
  grep -Eq '^CYCLE_(DONE|SKIP) ' "${log_path}"
}

kt_descendant_pids() {
  local root_pid="$1"
  local child_pid=""

  pgrep -P "${root_pid}" 2>/dev/null || true
  for child_pid in $(pgrep -P "${root_pid}" 2>/dev/null || true); do
    kt_descendant_pids "${child_pid}"
  done
}

kt_terminate_descendants_matching() {
  local root_pid="$1"
  local pattern="$2"
  local descendant=""
  local cmdline=""

  while IFS= read -r descendant; do
    [ -n "${descendant}" ] || continue
    cmdline="$(ps -p "${descendant}" -o cmd= 2>/dev/null || true)"
    if [[ "${cmdline}" =~ ${pattern} ]]; then
      kill -TERM "${descendant}" 2>/dev/null || true
    fi
  done < <(kt_descendant_pids "${root_pid}" | sort -nr | uniq)
}

kt_handle_cannsim_exit() {
  local rc="$1"
  local output_dir="$2"
  local latest_log=""

  if [ "${rc}" -eq 0 ]; then
    return 0
  fi

  latest_log="$(kt_find_latest_log "${output_dir}")"
  if [ -n "${latest_log}" ]; then
    if grep -q "All tests PASSED!" "${latest_log}"; then
      echo "==> cannsim exited ${rc} but ${latest_log} reports success; treating as success"
      return 0
    fi
    if grep -Eq '^CYCLE_(DONE|SKIP) ' "${latest_log}"; then
      echo "==> cannsim exited ${rc} after framework cycle completion markers in ${latest_log}; treating as success"
      return 0
    fi
  fi

  echo "==> cannsim exited ${rc}; inspect ${output_dir}" >&2
  return "${rc}"
}

kt_collect_cannsim_artifacts() {
  local work_dir="$1"
  local out_dir="$2"
  local cannsim_run=""
  local dest=""
  local tmp=""

  cannsim_run="$(find "${out_dir}" -maxdepth 1 -type d -name 'cannsim_*' 2>/dev/null | sort | tail -1 || true)"
  if [ -z "${cannsim_run}" ]; then
    return 0
  fi

  if [ -d "${work_dir}/log_ca" ] && [ ! -d "${cannsim_run}/log_ca" ]; then
    cp -a "${work_dir}/log_ca" "${cannsim_run}/log_ca"
  fi

  if [ -f "${work_dir}/instr.bin" ]; then
    dest="${cannsim_run}/instr.bin"
    if [ ! -f "${dest}" ] || [ -L "${dest}" ] || [ "${dest}" -ef "${work_dir}/instr.bin" ]; then
      tmp="${cannsim_run}/.instr.bin.tmp"
      cp -f "${work_dir}/instr.bin" "${tmp}"
      mv -f "${tmp}" "${dest}"
    fi
  fi

  if [ -f "${cannsim_run}/instr.bin" ]; then
    mkdir -p "${cannsim_run}/report"
    set +e
    cannsim report -e "${cannsim_run}" -o "${cannsim_run}/report" -n 0
    set -e
  fi
}
