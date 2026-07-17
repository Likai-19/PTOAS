#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Test wheel installation by verifying the installed Python contract works.
#
# Usage: ./test_wheel_imports.sh
#
# This script tests that the installed wheel can import:
#   - mlir.ir
#   - mlir.dialects.pto
#   - ptodsl
#   - from ptodsl import pto, scalar
#   - ptoas CLI entry
# and that a minimal PTODSL compile-only probe succeeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

WHEEL_GLOB="${PTO_TEST_WHEEL_PATH:-}"
if [[ -z "${WHEEL_GLOB}" && -n "${PTO_WHEEL_DIST_DIR:-}" ]]; then
  WHEEL_GLOB="${PTO_WHEEL_DIST_DIR}/ptoas-*.whl"
fi
if [[ -z "${WHEEL_GLOB}" && -d "${REPO_ROOT}/build/wheel-dist" ]]; then
  WHEEL_GLOB="${REPO_ROOT}/build/wheel-dist/ptoas-*.whl"
fi

TEST_TMPDIR=""
cleanup() {
  if [[ -n "${TEST_TMPDIR}" && -d "${TEST_TMPDIR}" ]]; then
    rm -rf "${TEST_TMPDIR}"
  fi
}
trap cleanup EXIT

echo "Testing wheel imports..."

if [[ -n "${WHEEL_GLOB}" ]] && compgen -G "${WHEEL_GLOB}" >/dev/null 2>&1; then
  TEST_WHEEL="$(compgen -G "${WHEEL_GLOB}" | sort | tail -n 1)"
  TEST_TMPDIR="$(mktemp -d /tmp/ptoas-wheel-test.XXXXXX)"
  echo "Installing wheel into isolated venv: ${TEST_WHEEL}"
  echo "Checking wheel payload before installation..."
  TEST_WHEEL="${TEST_WHEEL}" "${PYTHON_BIN}" - <<'PY'
import os
import zipfile
from pathlib import Path

wheel = Path(os.environ["TEST_WHEEL"])
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
    required = {"ptoas/__init__.py", "ptoas/_launcher.py", "pto/ptoas.so"}
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"wheel is missing required payload files: {missing}")
    if "ptoas/_runtime/bin/ptoas" in names:
        raise SystemExit("wheel unexpectedly contains ptoas/_runtime/bin/ptoas")
    entry_points_name = next(
        name for name in names
        if name.startswith("ptoas-") and name.endswith(".dist-info/entry_points.txt")
    )
    entry_points = zf.read(entry_points_name).decode("utf-8")
    if "ptoas=ptoas._launcher:main" not in entry_points:
        raise SystemExit("wheel entry points do not route ptoas through ptoas._launcher:main")
print(f"Wheel payload check passed: {wheel.name}")
PY
  "${PYTHON_BIN}" -m venv "${TEST_TMPDIR}/venv"
  source "${TEST_TMPDIR}/venv/bin/activate"
  python -m pip install --no-deps --force-reinstall "${TEST_WHEEL}"
  PYTHON_BIN="python"
else
  echo "No wheel artifact supplied; testing the currently installed environment."
fi

unset PYTHONPATH

# Test in a clean directory to avoid local imports
cd /tmp

echo "Testing mlir.ir import..."
"$PYTHON_BIN" -c "import mlir.ir; print('mlir.ir imported successfully')"

echo "Testing pto dialect import..."
"$PYTHON_BIN" -c "from mlir.dialects import pto; print('pto dialect imported successfully')"

echo "Testing ptodsl import..."
"$PYTHON_BIN" -c "import ptodsl; print(f'ptodsl imported successfully from {ptodsl.__file__}')"

echo "Testing ptodsl public imports..."
"$PYTHON_BIN" -c "from ptodsl import pto, scalar; print('ptodsl public imports imported successfully')"

echo "Testing installed ptoas console entry..."
PTOAS_VERSION_OUTPUT="$(ptoas --version | tr -d '\r')"
echo "${PTOAS_VERSION_OUTPUT}"
if [[ -n "${PTOAS_VERSION:-}" ]]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${PTOAS_VERSION}"
  if [[ "${PTOAS_VERSION_OUTPUT}" != "${EXPECTED_VERSION_OUTPUT}" ]]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${PTOAS_VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  echo "${PTOAS_VERSION_OUTPUT}" | grep -Eq '^ptoas [0-9]+\.[0-9]+$'
fi

echo "Testing PTODSL compile-only probe..."
"$PYTHON_BIN" - <<'PY'
from ptodsl import pto, scalar


@pto.jit(target="a5")
def wheel_compile_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    src = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    dst = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(src, tile)
    pto.tile.store(tile, dst)


mlir_text = wheel_compile_probe.compile().mlir_text()
if "func.func @wheel_compile_probe" not in mlir_text:
    raise SystemExit("PTODSL compile probe did not preserve the kernel symbol")
if "pto.tload" not in mlir_text or "pto.tstore" not in mlir_text:
    raise SystemExit("PTODSL compile probe did not emit the expected tile ops")
print("PTODSL compile probe succeeded")
PY

echo "All wheel import tests passed!"
