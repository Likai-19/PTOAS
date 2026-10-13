#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Regression for the PV shadow-tree failure paths in scripts/sim_dsl.sh (#71).
# build_pv_tree runs as an if-condition, so bash suppresses errexit inside it;
# every fallible step is checked explicitly and a failure must leave the shared
# root untouched and clean up the private staging tree.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SIM_DSL="${REPO_ROOT}/scripts/sim_dsl.sh"
SOC="Ascend950PR_9599"
PV_LIBS=(libmodel_top_pv.so libstars_pv.so libnpu_drv_pvmodel.so libruntime_cmodel.so)

WORK="${TMPDIR:-/tmp}/ptoas-pv-lock.$$"
rm -rf -- "${WORK}"
mkdir -p -- "${WORK}" || exit 1
trap 'rm -rf -- "${WORK}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# One fake CANN tree plus a fake failing tool on PATH, then run the real script.
run_case() {
  local tool="$1"
  local root="${WORK}/${tool}"
  local lib
  mkdir -p -- "${root}/cann/x86_64-linux/simulator/${SOC}/lib" "${root}/bin" || exit 1
  for lib in "${PV_LIBS[@]}"; do
    : > "${root}/cann/x86_64-linux/simulator/${SOC}/lib/${lib}" || exit 1
  done
  printf '#!/usr/bin/env bash\nexit 1\n' > "${root}/bin/${tool}" || exit 1
  chmod 755 "${root}/bin/${tool}" || exit 1
  printf 'pass\n' > "${root}/example.py" || exit 1

  local shared="${root}/pv-root"
  local out="${root}/run.log"
  PATH="${root}/bin:${PATH}" HOME="${root}" ASCEND_HOME_PATH="${root}/cann" \
    PTOAS_SIM_MODEL=pv PTOAS_SIM_MODEL_ROOT="${shared}" \
    bash "${SIM_DSL}" --output "${root}/out" "${root}/example.py" > "${out}" 2>&1
  local rc=$?
  [ "${rc}" -ne 0 ] || fail "${tool}: the run reported success after a ${tool} failure"
  grep -q "PTOAS_SIM_MODEL=pv is not available" "${out}" \
    || fail "${tool}: the run did not report the pv failure reason"
  [ ! -e "${shared}" ] || fail "${tool}: the shared root was created or replaced"
  local leftovers
  leftovers="$(find "${root}" -maxdepth 1 -name "$(basename -- "${shared}").build.*" | wc -l)"
  [ "${leftovers}" -eq 0 ] || fail "${tool}: a staging tree was left behind"
  echo "PASS pv tree ${tool} failure"
}

[ -f "${SIM_DSL}" ] || fail "missing ${SIM_DSL}"
run_case flock
run_case ln
echo "all sim_dsl PV failure-path checks passed"
