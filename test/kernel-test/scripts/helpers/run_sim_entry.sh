#!/usr/bin/env bash
set -euo pipefail

# Internal cannsim entry shim. Not intended for direct user invocation.

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "$(realpath -- "${SCRIPT_PATH}")")" && pwd)/common.sh"

kt_resolve_paths "${SCRIPT_PATH}"
kt_load_args_file "${KERNEL_TEST_ARGS_FILE:-}"
kt_prepare_cannsim_workspace "${KERNEL_TEST_WORK_DIR:?KERNEL_TEST_WORK_DIR is required}"

if [ "$#" -lt 1 ]; then
  echo "run_sim_entry.sh requires the python script path from cannsim -u" >&2
  exit 2
fi

kt_exec_python_cmd "$(kt_default_python_cmd)" "$1" "${KT_LOADED_ARGS[@]}"
