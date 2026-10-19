#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PTOAS_SIM_MODEL="${PTOAS_SIM_MODEL:-ca}"

usage() {
  cat <<'EOF'
Run a PTODSL JIT example under `msprof op simulator`.

Usage:
  scripts/sim_dsl.sh [options] <example.py|dsl-st-dir> [-- <example args...>]

Options:
  --output <dir>        Final directory to sync results into after the run.
  --soc-version <soc>   Override simulator soc version. Default: Ascend950PR_9599
  --verbose-msprof      Show the full `msprof` simulator log stream.
  --quiet-msprof        Suppress noisy `msprof` INFO/WARN/ERROR lines. Default.
  -h, --help            Show this help.

Environment:
  PTOAS_SIM_MODEL        Simulator model: ca (default) or pv. pv builds a
                         private CANN symlink tree that shadows the four
                         CA simulator libraries with their PV counterparts.
                         An explicit pv never falls back to ca: when the tree
                         cannot be built or a PV library is missing, the run
                         fails and prints the reason.
  PTOAS_SIM_MODEL_ROOT   Root of that private tree (default:
                         $HOME/.cache/ptoas/sim-pv-<soc>).
  PTOAS_MSPROF_PRIVATE_ROOT
                        Private root used for the actual `msprof --output`.
                        Defaults to `$XDG_RUNTIME_DIR/ptoas-msprof` when available,
                        otherwise `$HOME/.cache/ptoas/msprof`.
  PTOAS_KEEP_MSPROF_STAGING=1
                        Keep the private staging directory after a successful sync.
  PTOAS_MSPROF_LOG_MODE=quiet|verbose
                        Override the default simulator log rendering mode.
  PYTHON_BIN             Python executable used for the PTODSL example.
                        Defaults to python3.

Examples:
  scripts/sim_dsl.sh ptodsl/examples/jit/tadd_launch.py
  scripts/sim_dsl.sh test/dsl-st
  scripts/sim_dsl.sh \
    --output "$PWD/build/msprof_res/flash_softmax" \
    ptodsl/examples/jit/flash_attention_softmax_launch.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[sim_dsl] $*"
}

resolve_executable() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    command -v -- "${candidate}"
    return 0
  fi
  command -v -- "${candidate}"
}

prepend_path() {
  local dir="$1"
  if [[ -z "${dir}" || ! -d "${dir}" ]]; then
    return 0
  fi
  if [[ ":${PATH}:" == *":${dir}:"* ]]; then
    return 0
  fi
  PATH="${dir}:${PATH}"
  export PATH
}

ensure_private_dir() {
  local dir="$1"
  umask 077
  mkdir -p "${dir}"
  chmod 700 "${dir}"
}

# --- PV model shadow tree -------------------------------------------------
#
# `msprof op simulator` hard codes the CA simulator libraries, so the only way
# to run the PV model is a private CANN tree whose simulator lib directory
# shadows the four CA names with their PV counterparts:
#   libruntime_camodel.so -> libruntime_cmodel.so
#   libnpu_drv_camodel.so -> libnpu_drv_pvmodel.so
#   libmodel_top.so       -> libmodel_top_pv.so
#   libstars.so           -> libstars_pv.so
# libascend_hal.so must NOT be shadowed: msprof itself links it, and the PV
# build leaves drvDeviceStatus undefined there. Everything else is symlinked
# back to the real CANN so torch_npu and the ptoas tools keep working, and
# bin/setenv.bash is a shim that sources the real setenv and then points
# ASCEND_HOME_PATH back at the private root.
pv_shadow_lib() {
  case "$1" in
    libmodel_top.so) printf "%s\n" "libmodel_top_pv.so" ;;
    libstars.so) printf "%s\n" "libstars_pv.so" ;;
    libnpu_drv_camodel.so) printf "%s\n" "libnpu_drv_pvmodel.so" ;;
    libruntime_camodel.so) printf "%s\n" "libruntime_cmodel.so" ;;
    *) printf "%s\n" "" ;;
  esac
}

pv_tree_valid() {
  local root="$1" soc="$2" real="$3" name
  [[ -f "${root}/.ptoas-sim-pv" ]] || return 1
  [[ "$(< "${root}/.ptoas-sim-pv")" == "${real}" ]] || return 1
  for name in libmodel_top.so libstars.so libnpu_drv_camodel.so libruntime_camodel.so; do
    [[ -e "${root}/tools/simulator/${soc}/lib/${name}" ]] || return 1
  done
  return 0
}

pv_tree_fail() {
  PV_TREE_ERROR="$1"
  return 1
}

build_pv_tree() {
  local root="$1" soc="$2" real="$3"
  local real_sim="${real}/x86_64-linux/simulator"
  local real_soc
  real_soc="$(readlink -f "${real_sim}/${soc}" 2>/dev/null || true)"
  if [[ -z "${real_soc}" || ! -d "${real_soc}/lib" ]]; then
    pv_tree_fail "no simulator lib directory at ${real_sim}/${soc}/lib"
    return 1
  fi
  local real_lib="${real_soc}/lib" name missing=()
  for name in libmodel_top_pv.so libstars_pv.so libnpu_drv_pvmodel.so libruntime_cmodel.so; do
    [[ -f "${real_lib}/${name}" ]] || missing+=("${name}")
  done
  if (( ${#missing[@]} > 0 )); then
    pv_tree_fail "the CANN simulator at ${real_lib} does not ship the PV libraries: ${missing[*]}"
    return 1
  fi

  # The mutating part runs in a subshell. errexit is deliberately NOT trusted:
  # the caller invokes this function as an if-condition, a context where bash
  # suppresses errexit even inside the subshell. Every fallible command is
  # therefore checked explicitly, the lock failure stops immediately, and the
  # staging tree is verified and only then published. The caller reports the
  # failure and exits instead of degrading to ca. The flock is released when the
  # subshell exits.
  if ! (
    set -e
    mkdir -p -- "$(dirname -- "${root}")" || exit 1
    exec 9>"${root}.lock" || exit 1
    flock -w 600 9 || exit 1
    if pv_tree_valid "${root}" "${soc}" "${real}"; then
      exit 0
    fi

    staging="${root}.build.$$"
    # Any failure below leaves only this private staging tree; remove it.
    trap 'rm -rf -- "${staging:-}"' EXIT
    rm -rf -- "${staging}" || exit 1
    mkdir -p -- "${staging}/bin" "${staging}/tools" "${staging}/x86_64-linux/simulator" || exit 1

    for entry in "${real}"/*; do
      entry_name="$(basename -- "${entry}")"
      case "${entry_name}" in bin | tools | x86_64-linux) continue ;; esac
      ln -s -- "${entry}" "${staging}/${entry_name}" || exit 1
    done
    for entry in "${real}/bin"/*; do
      entry_name="$(basename -- "${entry}")"
      [[ "${entry_name}" == "setenv.bash" ]] && continue
      ln -s -- "${entry}" "${staging}/bin/${entry_name}" || exit 1
    done
    {
      printf "source %s/bin/setenv.bash\n" "${real}"
      printf "export ASCEND_HOME_PATH=%s\n" "${root}"
    } > "${staging}/bin/setenv.bash" || exit 1
    for entry in "${real}/tools"/*; do
      entry_name="$(basename -- "${entry}")"
      [[ "${entry_name}" == "simulator" ]] && continue
      ln -s -- "${entry}" "${staging}/tools/${entry_name}" || exit 1
    done
    ln -s -- ../x86_64-linux/simulator "${staging}/tools/simulator" || exit 1
    for entry in "${real}/x86_64-linux"/*; do
      entry_name="$(basename -- "${entry}")"
      [[ "${entry_name}" == "simulator" ]] && continue
      ln -s -- "${entry}" "${staging}/x86_64-linux/${entry_name}" || exit 1
    done

    sim="${staging}/x86_64-linux/simulator"
    for entry in "${real_sim}"/*; do
      entry_name="$(basename -- "${entry}")"
      if [[ "$(readlink -f "${entry}" 2>/dev/null || true)" != "${real_soc}" ]]; then
        ln -s -- "${entry}" "${sim}/${entry_name}" || exit 1
        continue
      fi
      if [[ "${entry_name}" != "${soc}" ]]; then
        ln -s -- "${soc}" "${sim}/${entry_name}" || exit 1
        continue
      fi
      soc_dir="${sim}/${entry_name}"
      mkdir -p -- "${soc_dir}/lib" || exit 1
      for sub in "${real_soc}"/*; do
        sub_name="$(basename -- "${sub}")"
        [[ "${sub_name}" == "lib" ]] && continue
        ln -s -- "${sub}" "${soc_dir}/${sub_name}" || exit 1
      done
      for libf in "${real_lib}"/*; do
        lib_name="$(basename -- "${libf}")"
        shadow="$(pv_shadow_lib "${lib_name}")"
        if [[ -n "${shadow}" ]]; then
          ln -s -- "${real_lib}/${shadow}" "${soc_dir}/lib/${lib_name}" || exit 1
        else
          ln -s -- "${libf}" "${soc_dir}/lib/${lib_name}" || exit 1
        fi
      done
    done

    # Write the marker into staging and verify the complete tree there before
    # it replaces the shared root, so an incomplete build can never be
    # published. The EXIT trap removes the private staging tree on any failure.
    printf "%s\n" "${real}" > "${staging}/.ptoas-sim-pv" || exit 1
    pv_tree_valid "${staging}" "${soc}" "${real}" || exit 1
    rm -rf -- "${root}" || exit 1
    mv -- "${staging}" "${root}" || exit 1
  ); then
    pv_tree_fail "failed to build the private PV shadow tree at ${root}"
    return 1
  fi
  if ! pv_tree_valid "${root}" "${soc}" "${real}"; then
    pv_tree_fail "the private PV shadow tree at ${root} is incomplete after the build"
    return 1
  fi
  return 0
}

sync_msprof_output() {
  local src_dir="$1"
  local dst_dir="$2"

  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"
}

print_msprof_log() {
  local log_file="$1"
  local mode="$2"
  local status="$3"

  if [[ "${mode}" == "verbose" ]]; then
    cat "${log_file}"
    return
  fi

  grep -Ev '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
    "${log_file}" || true

  if [[ ${status} -ne 0 ]]; then
    local suppressed_count
    suppressed_count=$(wc -l < "${log_file}")
    local filtered_count
    filtered_count=$(
      grep -Evc '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
        "${log_file}" || true
    )
    suppressed_count=$((suppressed_count - filtered_count))
    if [[ ${suppressed_count} -gt 0 ]]; then
      log "msprof failure tail:"
      tail -n 20 "${log_file}"
    fi
    log "full msprof log saved at ${log_file}"
  fi
}

python_can_import_ptodsl() {
  local python_bin="$1"
  "${python_bin}" - <<'PY' >/dev/null 2>&1
import ptoas.mlir.ir  # noqa: F401
from ptodsl import pto  # noqa: F401
PY
}

SOC_VERSION="Ascend950PR_9599"
OUTPUT_DIR=""
EXAMPLE_PATH=""
EXAMPLE_ARGS=()
MSPROF_LOG_MODE="${PTOAS_MSPROF_LOG_MODE:-quiet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --soc-version)
      [[ $# -ge 2 ]] || die "--soc-version requires a value"
      SOC_VERSION="$2"
      shift 2
      ;;
    --verbose-msprof)
      MSPROF_LOG_MODE="verbose"
      shift
      ;;
    --quiet-msprof)
      MSPROF_LOG_MODE="quiet"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXAMPLE_ARGS=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${EXAMPLE_PATH}" ]]; then
        EXAMPLE_PATH="$1"
      else
        EXAMPLE_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

[[ -n "${EXAMPLE_PATH}" ]] || die "missing <example.py|dsl-st-dir>"

if [[ "${EXAMPLE_PATH}" != /* ]]; then
  EXAMPLE_PATH="${REPO_ROOT}/${EXAMPLE_PATH}"
fi
if [[ -d "${EXAMPLE_PATH}" ]]; then
  EXAMPLE_PATH="${EXAMPLE_PATH}/__main__.py"
fi
[[ -f "${EXAMPLE_PATH}" ]] || die "example script not found: ${EXAMPLE_PATH}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  die "ASCEND_HOME_PATH is not set; source CANN setenv or export it first"
fi

# Model selection. ca leaves the real CANN in place. pv swaps in the private
# shadow tree; an explicit pv never degrades to ca, so a missing _pv library
# or a broken tree fails the run with the concrete reason.
case "${PTOAS_SIM_MODEL}" in
  ca)
    log "model: ca"
    ;;
  pv)
    if [[ -f "${ASCEND_HOME_PATH}/.ptoas-sim-pv" ]]; then
      log "model: pv (root=${ASCEND_HOME_PATH})"
    else
      PV_MODEL_ROOT="${PTOAS_SIM_MODEL_ROOT:-${HOME}/.cache/ptoas/sim-pv-${SOC_VERSION}}"
      REAL_ASCEND_HOME="${ASCEND_HOME_PATH}"
      PV_TREE_ERROR=""
      if ! build_pv_tree "${PV_MODEL_ROOT}" "${SOC_VERSION}" "${REAL_ASCEND_HOME}"; then
        die "PTOAS_SIM_MODEL=pv is not available: ${PV_TREE_ERROR:-unknown reason};" \
          "private tree root ${PV_MODEL_ROOT}, CANN ${REAL_ASCEND_HOME}." \
          "Provide the PV libraries, fix that path/permissions, or run the" \
          "CA model explicitly with PTOAS_SIM_MODEL=ca"
      fi
      ASCEND_HOME_PATH="${PV_MODEL_ROOT}"
      export ASCEND_HOME_PATH
      log "model: pv (root=${ASCEND_HOME_PATH})"
    fi
    ;;
  *)
    die "PTOAS_SIM_MODEL must be ca or pv, got: ${PTOAS_SIM_MODEL}"
    ;;
esac

if [[ -z "${OUTPUT_DIR}" ]]; then
  EXAMPLE_STEM="$(basename -- "${EXAMPLE_PATH}" .py)"
  OUTPUT_DIR="${REPO_ROOT}/build/msprof_res/${EXAMPLE_STEM}"
else
  EXAMPLE_STEM="$(basename -- "${EXAMPLE_PATH}" .py)"
fi

SIM_LIB_DIR="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
[[ -d "${SIM_LIB_DIR}" ]] || die "simulator library directory not found: ${SIM_LIB_DIR}"

PRIVATE_ROOT="${PTOAS_MSPROF_PRIVATE_ROOT:-${XDG_RUNTIME_DIR:-${HOME}/.cache}/ptoas/msprof}"
ensure_private_dir "${PRIVATE_ROOT}"
# A unique staging directory under PRIVATE_ROOT. It is not a bare mktemp:
# the run exits through the cleanup below, which removes this tree with
# rm -rf "${RUNTIME_OUTPUT_DIR}" unless --keep-output was requested.
RUNTIME_OUTPUT_DIR="${PRIVATE_ROOT}/${EXAMPLE_STEM}.$$.${RANDOM}"
mkdir -p "${RUNTIME_OUTPUT_DIR}"
chmod 700 "${RUNTIME_OUTPUT_DIR}"
MSPROF_STDIO_LOG="${RUNTIME_OUTPUT_DIR}/msprof.stdout.log"
EXAMPLE_EXIT_CODE_FILE="${RUNTIME_OUTPUT_DIR}/example.exitcode"
EXAMPLE_LAUNCHER="${RUNTIME_OUTPUT_DIR}/run_example.sh"
REQUESTED_PYTHON_BIN="${PTO_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
REQUESTED_PTOAS_BIN="${PTOAS_BIN:-}"

if ! RESOLVED_PYTHON_BIN="$(resolve_executable "${REQUESTED_PYTHON_BIN}")"; then
  die "PYTHON_BIN is not executable or not found on PATH: ${REQUESTED_PYTHON_BIN}"
fi
if [[ -n "${REQUESTED_PTOAS_BIN}" ]]; then
  if ! RESOLVED_PTOAS_BIN="$(resolve_executable "${REQUESTED_PTOAS_BIN}")"; then
    die "PTOAS_BIN is not executable or not found on PATH: ${REQUESTED_PTOAS_BIN}"
  fi
fi

source "${ASCEND_HOME_PATH}/bin/setenv.bash"
if [[ -n "${RESOLVED_PTOAS_BIN:-}" ]]; then
  prepend_path "$(dirname -- "${RESOLVED_PTOAS_BIN}")"
  export PTOAS_BIN="${RESOLVED_PTOAS_BIN}"
fi

# Keep this interpreter's packages ahead of the CANN installation: CANN's
# setenv.bash prepends its python/site-packages to PYTHONPATH, which would
# shadow identically named packages installed in this environment (pypto now,
# ptoas next). Re-prepend ours after sourcing it so this environment wins.
PURELIB="$("${RESOLVED_PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths().get("purelib") or "")')"
if [[ -n "${PURELIB}" ]]; then
  export PYTHONPATH="${PURELIB}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if ! python_can_import_ptodsl "${RESOLVED_PYTHON_BIN}"; then
  die "active Python environment cannot import ptodsl/ptoas.mlir.ir;" \
      "install the PTOAS wheel or use a preconfigured environment"
fi
if ! command -v ptoas >/dev/null 2>&1; then
  die "ptoas is not available on PATH; install the PTOAS wheel or export PTOAS_BIN"
fi
log "using installed Python environment from ${RESOLVED_PYTHON_BIN}"
log "using ptoas from $(command -v ptoas)"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

cat > "${EXAMPLE_LAUNCHER}" <<'EOF'
#!/usr/bin/env bash
set +e
"${PTOAS_SIM_DSL_PYTHON_BIN}" "${PTOAS_SIM_DSL_EXAMPLE_PATH}" "$@"
status=$?
printf '%s\n' "${status}" > "${PTOAS_SIM_DSL_EXIT_CODE_FILE}"
exit "${status}"
EOF
chmod 700 "${EXAMPLE_LAUNCHER}"
export PTOAS_SIM_DSL_PYTHON_BIN="${RESOLVED_PYTHON_BIN}"
export PTOAS_SIM_DSL_EXAMPLE_PATH="${EXAMPLE_PATH}"
export PTOAS_SIM_DSL_EXIT_CODE_FILE="${EXAMPLE_EXIT_CODE_FILE}"

# msprof rejects group/other-writable working directories, so always launch
# from a private directory and use an absolute path for the example script.
cd "${HOME}"

log "staging msprof output in ${RUNTIME_OUTPUT_DIR}"
if [[ "${OUTPUT_DIR}" != "${RUNTIME_OUTPUT_DIR}" ]]; then
  log "final results will be synced to ${OUTPUT_DIR}"
fi

set +e
msprof op simulator \
  --soc-version="${SOC_VERSION}" \
  --output="${RUNTIME_OUTPUT_DIR}" \
  "${EXAMPLE_LAUNCHER}" "${EXAMPLE_ARGS[@]}" \
  > "${MSPROF_STDIO_LOG}" 2>&1
MSPROF_STATUS=$?
set -e

EXAMPLE_STATUS=0
if [[ -f "${EXAMPLE_EXIT_CODE_FILE}" ]]; then
  EXAMPLE_STATUS="$(< "${EXAMPLE_EXIT_CODE_FILE}")"
  if [[ ! "${EXAMPLE_STATUS}" =~ ^[0-9]+$ ]]; then
    log "invalid example exit code recorded in ${EXAMPLE_EXIT_CODE_FILE}: ${EXAMPLE_STATUS}"
    EXAMPLE_STATUS=1
  fi
else
  log "example exit code file was not produced: ${EXAMPLE_EXIT_CODE_FILE}"
  EXAMPLE_STATUS=1
fi

STATUS=0
if [[ ${MSPROF_STATUS} -ne 0 ]]; then
  STATUS=${MSPROF_STATUS}
elif [[ ${EXAMPLE_STATUS} -ne 0 ]]; then
  STATUS=${EXAMPLE_STATUS}
fi

print_msprof_log "${MSPROF_STDIO_LOG}" "${MSPROF_LOG_MODE}" "${STATUS}"

SYNC_STATUS=0
if [[ -d "${RUNTIME_OUTPUT_DIR}" ]]; then
  if sync_msprof_output "${RUNTIME_OUTPUT_DIR}" "${OUTPUT_DIR}"; then
    log "synced msprof results to ${OUTPUT_DIR}"
  else
    SYNC_STATUS=$?
    log "failed to sync msprof results to ${OUTPUT_DIR}"
  fi
fi

if [[ "${PTOAS_KEEP_MSPROF_STAGING:-0}" != "1" && ${SYNC_STATUS} -eq 0 ]]; then
  rm -rf "${RUNTIME_OUTPUT_DIR}"
else
  log "kept staging directory at ${RUNTIME_OUTPUT_DIR}"
fi

if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi

exit ${SYNC_STATUS}
