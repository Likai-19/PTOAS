#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Collect ptoas binary and its dependencies into a self-contained distribution.
#
# Usage: ./collect_ptoas_dist.sh <output_directory>
#
# Required environment variables:
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_BUILD_DIR   - Path to PTO build directory (optional, defaults to PTO_SOURCE_DIR/build)
#   PTO_INSTALL_DIR - Path to PTO install directory
#   PTO_SOURCE_DIR  - Path to PTO source directory
#
# Output structure:
#   <output_directory>/
#     ptoas           - Wrapper script that sets up LD_LIBRARY_PATH
#     bin/ptoas       - Python wrapper entrypoint
#     python/ptoas/    - Launcher package used by the wrapper
#     lib/ptoas.so    - Shared runtime loaded by the launcher
#     lib/*.so*       - Required shared library dependencies
#     share/ptoas/TileOps - TileLang template library
#     tilelang_dsl/   - TileLang DSL Python package
#
# This compiler-oriented binary artifact intentionally does not bundle the
# PTODSL Python package. PTODSL is provided only by the `ptoas` wheel.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_directory>" >&2
  exit 1
fi

PTOAS_DIST_DIR="$1"

# Validate required environment variables
for var in LLVM_BUILD_DIR PTO_INSTALL_DIR PTO_SOURCE_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

PTO_BUILD_DIR="${PTO_BUILD_DIR:-${PTO_SOURCE_DIR}/build}"
PTOAS_BIN="${PTO_BUILD_DIR}/tools/ptoas/ptoas"
PTOAS_SHARED_MODULE="${PTO_INSTALL_DIR}/lib/ptoas.so"
PTOAS_DEPS_DIR="${PTOAS_DIST_DIR}/lib"
PTOAS_SHARED_MODULE_DIST_PATH="${PTOAS_DEPS_DIR}/ptoas.so"
PTOAS_TILEOPS_SRC_DIR="${PTO_INSTALL_DIR}/share/ptoas/TileOps"
PTOAS_TILEOPS_DIST_DIR="${PTOAS_DIST_DIR}/share/ptoas/TileOps"
PTOAS_TILELANG_DSL_SRC_DIR="${PTO_INSTALL_DIR}/tilelang_dsl"
PTOAS_TILELANG_DSL_DIST_DIR="${PTOAS_DIST_DIR}/tilelang_dsl"
PTOAS_WRAPPER_PKG_SRC_DIR="${PTO_INSTALL_DIR}/ptoas"
PTOAS_PYTHON_ROOT_DIST_DIR="${PTOAS_DIST_DIR}/python"
PTOAS_WRAPPER_PKG_DIST_DIR="${PTOAS_PYTHON_ROOT_DIST_DIR}/ptoas"

if [ ! -f "$PTOAS_BIN" ]; then
  echo "Error: ptoas wrapper not found at $PTOAS_BIN" >&2
  exit 1
fi
if [ ! -f "$PTOAS_SHARED_MODULE" ]; then
  echo "Error: shared launcher module not found at $PTOAS_SHARED_MODULE" >&2
  exit 1
fi

remove_rpath() {
  local path="$1"
  if ! has_rpath "$path"; then
    return
  fi
  if ! can_scrub_rpath; then
    echo "WARN: skipping RPATH/RUNPATH scrub for ${path}; install patchelf or chrpath to harden local dist artifacts" >&2
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path"
  fi
  if has_rpath "$path" && command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path"
  fi
  if has_rpath "$path"; then
    echo "Error: failed to scrub RPATH/RUNPATH from ${path}" >&2
    exit 1
  fi
}

strip_symbols() {
  local path="$1"
  strip --strip-unneeded "$path"
}

has_rpath() {
  local path="$1"
  if command -v patchelf >/dev/null 2>&1; then
    local rpath_value
    rpath_value="$(patchelf --print-rpath "$path" 2>/dev/null || true)"
    [[ -n "$rpath_value" ]]
    return
  fi
  readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'
}

can_scrub_rpath() {
  command -v patchelf >/dev/null 2>&1 || command -v chrpath >/dev/null 2>&1
}

assert_relro() {
  local path="$1"
  if ! readelf -l "$path" 2>/dev/null | grep -q 'GNU_RELRO'; then
    echo "WARN: RELRO segment missing in ${path}" >&2
    return
  fi
  if ! readelf -d "$path" 2>/dev/null | grep -Eq '(BIND_NOW|FLAGS.*NOW|FLAGS_1.*NOW)'; then
    echo "WARN: NOW binding missing in ${path}" >&2
  fi
}

assert_no_symtab() {
  local path="$1"
  if readelf -S "$path" 2>/dev/null | grep -Eq '[[:space:]]\\.symtab[[:space:]]'; then
    echo "Error: symbol table still present in ${path}" >&2
    exit 1
  fi
}

assert_no_rpath() {
  local path="$1"
  if ! can_scrub_rpath; then
    return
  fi
  if has_rpath "$path"; then
    echo "Error: runtime search path still present in ${path}" >&2
    exit 1
  fi
}

harden_elf() {
  local path="$1"
  remove_rpath "$path"
  strip_symbols "$path"
  assert_relro "$path"
  assert_no_symtab "$path"
  assert_no_rpath "$path"
}

# Create output directories
mkdir -p \
  "${PTOAS_DIST_DIR}/bin" \
  "${PTOAS_DEPS_DIR}" \
  "${PTOAS_PYTHON_ROOT_DIST_DIR}" \
  "$(dirname "${PTOAS_TILEOPS_DIST_DIR}")"
rm -rf "${PTOAS_WRAPPER_PKG_DIST_DIR}"
cp -R "${PTOAS_WRAPPER_PKG_SRC_DIR}" "${PTOAS_WRAPPER_PKG_DIST_DIR}"

# Copy ptoas binary
echo "Copying ptoas wrapper..."
cp "$PTOAS_BIN" "${PTOAS_DIST_DIR}/bin/"
chmod +x "${PTOAS_DIST_DIR}/bin/ptoas"

echo "Copying ptoas shared runtime..."
cp -fL "$PTOAS_SHARED_MODULE" "${PTOAS_SHARED_MODULE_DIST_PATH}"

# Collect non-system *.so dependencies needed by the packaged shared runtime.
echo "Collecting shared library dependencies..."
linux_runtime_dep_paths() {
  local path="$1"
  ldd "$path" 2>/dev/null | awk '
    /=> \// { print $3 }
    /^\// { print $1 }
  '
}

should_bundle_linux_dep() {
  local path="$1"
  case "$path" in
    /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*)
      return 1
      ;;
  esac
  return 0
}

copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${PTOAS_DEPS_DIR}/${name}" ]] && return 0
  cp -L -n "$f" "${PTOAS_DEPS_DIR}/" 2>/dev/null || true
  harden_elf "${PTOAS_DEPS_DIR}/${name}"
  while read -r res; do
    [[ -n "$res" ]] || continue
    should_bundle_linux_dep "$res" || continue
    copy_so "$res"
  done < <(linux_runtime_dep_paths "$f")
}

while read -r res; do
  [[ -n "$res" ]] || continue
  should_bundle_linux_dep "$res" || continue
  copy_so "$res"
done < <(linux_runtime_dep_paths "$PTOAS_BIN")
while read -r res; do
  [[ -n "$res" ]] || continue
  should_bundle_linux_dep "$res" || continue
  copy_so "$res"
done < <(linux_runtime_dep_paths "$PTOAS_SHARED_MODULE")

while read -r packaged; do
  harden_elf "$packaged"
done < <(find "${PTOAS_DEPS_DIR}" -type f | sort)

echo "Copying TileLang runtime resources..."
if [ ! -d "${PTOAS_TILEOPS_SRC_DIR}" ]; then
  echo "Error: TileOps resource directory not found at ${PTOAS_TILEOPS_SRC_DIR}" >&2
  exit 1
fi
if [ ! -d "${PTOAS_TILELANG_DSL_SRC_DIR}" ]; then
  echo "Error: tilelang_dsl package directory not found at ${PTOAS_TILELANG_DSL_SRC_DIR}" >&2
  exit 1
fi
rm -rf "${PTOAS_TILEOPS_DIST_DIR}" "${PTOAS_TILELANG_DSL_DIST_DIR}"
cp -R "${PTOAS_TILEOPS_SRC_DIR}" "${PTOAS_TILEOPS_DIST_DIR}"
cp -R "${PTOAS_TILELANG_DSL_SRC_DIR}" "${PTOAS_TILELANG_DSL_DIST_DIR}"

# Create wrapper script
echo "Creating wrapper script..."
cat > "${PTOAS_DIST_DIR}/ptoas" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTO_INSTALL_DIR="${SCRIPT_DIR}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PTOAS_PYTHON_ROOT="${SCRIPT_DIR}/python"
exec "${SCRIPT_DIR}/bin/ptoas" "$@"
WRAPPER_EOF
chmod +x "${PTOAS_DIST_DIR}/ptoas"

echo "Smoke testing packaged ptoas dist..."
VERSION_OUTPUT="$(env -u PYTHONPATH -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  "${PTOAS_DIST_DIR}/ptoas" --version | tr -d '\r')"
echo "$VERSION_OUTPUT"
EXPECTED_PTOAS_CLI_VERSION="${PTOAS_CLI_VERSION:-${PTOAS_VERSION:-}}"
if [ -n "${EXPECTED_PTOAS_CLI_VERSION}" ]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${EXPECTED_PTOAS_CLI_VERSION}"
  if [ "${VERSION_OUTPUT}" != "${EXPECTED_VERSION_OUTPUT}" ]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  echo "$VERSION_OUTPUT" | grep -Eq '^ptoas [0-9]+\.[0-9]+$'
fi

test -d "${PTOAS_TILEOPS_DIST_DIR}"
test -f "${PTOAS_TILELANG_DSL_DIST_DIR}/__init__.py"
test -f "${PTOAS_SHARED_MODULE_DIST_PATH}"
if [ -e "${PTOAS_DIST_DIR}/ptodsl" ]; then
  echo "Error: compiler-oriented ptoas dist must not bundle PTODSL" >&2
  exit 1
fi

# Show collected files
echo ""
echo "=== ptoas distribution contents ==="
ls -la "${PTOAS_DIST_DIR}/"
ls -la "${PTOAS_DIST_DIR}/bin/"
ls -la "${PTOAS_DIST_DIR}/python/"
ls -la "${PTOAS_DIST_DIR}/share/ptoas/"
ls -la "${PTOAS_TILELANG_DSL_DIST_DIR}"
SO_COUNT=$(find "${PTOAS_DEPS_DIR}" -name "*.so*" 2>/dev/null | wc -l)
echo "=== Collected .so dependencies (${SO_COUNT} files) ==="
du -sh "${PTOAS_DEPS_DIR}/"
echo ""
echo "Distribution created at: ${PTOAS_DIST_DIR}"
