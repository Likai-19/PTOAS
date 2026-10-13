#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Regression for the PTODSL parallel-case coverage gate (#72). Comparing the
# PASS line count with the expected count lets one case report PASS twice while
# another never runs; the driver must compare the reported PASS names and the
# global discovery must enforce suite-wide case-name uniqueness.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DRIVER="${REPO_ROOT}/.gitcode/scripts/run_ptodsl_cases_parallel.sh"

WORK="${TMPDIR:-/tmp}/ptoas-coverage.$$"
rm -rf -- "${WORK}"
mkdir -p -- "${WORK}" || exit 1
trap 'rm -rf -- "${WORK}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# A stub interpreter that prints the discovered names and a stub simulator
# runner that prints the PASS lines listed in FAKE_PASS_FILE.
make_stubs() {
  local root="$1"
  mkdir -p -- "${root}/ws/test/dsl-st" "${root}/bin" || exit 1
  printf '# stub case module\n' > "${root}/ws/test/dsl-st/fake_mod.py" || exit 1
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" case_a case_b\n' > "${root}/bin/fake-python" || exit 1
  {
    echo '#!/usr/bin/env bash'
    echo 'printf "%s\\n" "model: ca"'
    echo 'if [ -n "${FAKE_PASS_FILE:-}" ] && [ -f "${FAKE_PASS_FILE}" ]; then cat "${FAKE_PASS_FILE}"; fi'
    echo 'exit 0'
  } > "${root}/bin/fake-sim-dsl" || exit 1
  chmod 755 "${root}/bin/fake-python" "${root}/bin/fake-sim-dsl" || exit 1
}

run_case() {
  local name="$1" pass_lines="$2" expect_rc="$3" expect_grep="$4"
  local root="${WORK}/${name}"
  make_stubs "${root}"
  printf '%s\n' "${pass_lines}" > "${root}/pass.txt" || exit 1
  local out="${root}/driver.log"
  WORKSPACE="${root}/ws" BUILD_ROOT="${root}/build" \
    PTODSL_DSL_ROOT="${root}/ws/test/dsl-st" \
    SIM_DSL_SCRIPT="${root}/bin/fake-sim-dsl" \
    PYTHON_BIN="${root}/bin/fake-python" \
    PTODSL_CASE_JOBS=2 FAKE_PASS_FILE="${root}/pass.txt" \
    bash "${DRIVER}" > "${out}" 2>&1
  local rc=$?
  if [ "${expect_rc}" = "0" ]; then
    [ "${rc}" -eq 0 ] || fail "${name}: expected rc=0, got ${rc}"
  else
    [ "${rc}" -ne 0 ] || fail "${name}: expected a non-zero rc, got 0"
  fi
  grep -q "${expect_grep}" "${out}" || fail "${name}: missing ${expect_grep}"
  echo "PASS coverage ${name}"
}

[ -f "${DRIVER}" ] || fail "missing ${DRIVER}"
run_case duplicate "$(printf 'PASS case_a\nPASS case_a')" 1 "COVERAGE-DUPLICATE"
run_case missing "$(printf 'PASS case_a')" 1 "COVERAGE-MISMATCH"
run_case exact "$(printf 'PASS case_a\nPASS case_b')" 0 "all 2 case(s) passed"
echo "all parallel coverage checks passed"
