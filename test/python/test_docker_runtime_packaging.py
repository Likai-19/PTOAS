#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COPY_DEPS_SCRIPT = REPO_ROOT / "docker" / "copy_ptoas_deps.sh"
COLLECT_DIST_SCRIPT = REPO_ROOT / "docker" / "collect_ptoas_dist.sh"
COLLECT_DIST_MAC_SCRIPT = REPO_ROOT / "docker" / "collect_ptoas_dist_mac.sh"
BUILD_WHEEL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel.yml"
BUILD_WHEEL_MAC_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel_mac.yml"


class DockerRuntimePackagingTests(unittest.TestCase):
    def test_runtime_image_uses_wheel_entrypoint_instead_of_copied_wrapper(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("COPY --from=builder /wheelhouse/ptoas*.whl /tmp/", dockerfile)
        self.assertIn(
            "RUN pip install --no-cache-dir /tmp/ptoas*.whl && rm /tmp/ptoas*.whl",
            dockerfile,
        )
        self.assertNotIn(
            "COPY --from=builder /llvm-workspace/PTOAS/build-release/tools/ptoas/ptoas /usr/local/bin/ptoas",
            dockerfile,
        )
        self.assertNotIn("/usr/local/lib/ptoas", dockerfile)

    def test_dependency_collection_roots_at_shared_module(self):
        script = COPY_DEPS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PTOAS_SHARED_MODULE="${PTO_INSTALL_DIR}/lib/ptoas.so"', script)
        self.assertIn('done < <(linux_runtime_dep_paths "$PTOAS_SHARED_MODULE")', script)
        self.assertNotIn('PTOAS_BIN="${PTO_BUILD_DIR}/tools/ptoas/ptoas"', script)
        self.assertNotIn('done < <(linux_runtime_dep_paths "$PTOAS_BIN")', script)

    def test_linux_dist_stages_install_tree_python_root(self):
        script = COLLECT_DIST_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PTOAS_WRAPPER_PKG_SRC_DIR="${PTO_INSTALL_DIR}/ptoas"', script)
        self.assertIn('PTOAS_WRAPPER_PKG_DIST_DIR="${PTOAS_DIST_DIR}/ptoas"', script)
        self.assertIn('PTOAS_WHEEL_BOOTSTRAP_SRC="${PTO_INSTALL_DIR}/ptoas_wheel_bootstrap.py"', script)
        self.assertIn('cp "${PTOAS_WHEEL_BOOTSTRAP_SRC}" "${PTOAS_WHEEL_BOOTSTRAP_DIST_PATH}"', script)
        self.assertIn('test -f "${PTOAS_WRAPPER_PKG_DIST_DIR}/_runtime_entry.py"', script)
        self.assertIn('test -f "${PTOAS_WHEEL_BOOTSTRAP_DIST_PATH}"', script)
        self.assertIn('"${PTOAS_DIST_DIR}/bin/ptoas" --version', script)
        self.assertNotIn("PTOAS_PYTHON_ROOT_DIST_DIR", script)
        self.assertNotIn("export PTOAS_PYTHON_ROOT", script)
        self.assertNotIn('cat > "${PTOAS_DIST_DIR}/ptoas"', script)

    def test_macos_dist_stages_install_tree_python_root(self):
        script = COLLECT_DIST_MAC_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PTOAS_SHARED_MODULE="${PTO_INSTALL_DIR}/lib/ptoas.so"', script)
        self.assertIn('PTOAS_WRAPPER_PKG_SRC_DIR="${PTO_INSTALL_DIR}/ptoas"', script)
        self.assertIn('PTOAS_WRAPPER_PKG_DIST_DIR="${PTOAS_DIST_DIR}/ptoas"', script)
        self.assertIn('PTOAS_WHEEL_BOOTSTRAP_SRC="${PTO_INSTALL_DIR}/ptoas_wheel_bootstrap.py"', script)
        self.assertIn('cp "${PTOAS_WHEEL_BOOTSTRAP_SRC}" "${PTOAS_WHEEL_BOOTSTRAP_DIST_PATH}"', script)
        self.assertIn('collect_dylibs "${PTOAS_SHARED_MODULE_DIST_PATH}"', script)
        self.assertIn('test -f "${PTOAS_WRAPPER_PKG_DIST_DIR}/_runtime_entry.py"', script)
        self.assertIn('test -f "${PTOAS_WHEEL_BOOTSTRAP_DIST_PATH}"', script)
        self.assertIn('"${PTOAS_DIST_DIR}/bin/ptoas" --version', script)
        self.assertNotIn('cat > "${PTOAS_DIST_DIR}/ptoas"', script)

    def test_dist_archives_use_bin_entrypoint(self):
        linux_workflow = BUILD_WHEEL_WORKFLOW.read_text(encoding="utf-8")
        mac_workflow = BUILD_WHEEL_MAC_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('chmod +x "$GITHUB_WORKSPACE/ptoas-dist/bin/ptoas"', linux_workflow)
        self.assertNotIn('chmod +x "$GITHUB_WORKSPACE/ptoas-dist/ptoas"', linux_workflow)
        self.assertIn('chmod +x "$GITHUB_WORKSPACE/ptoas-dist/bin/ptoas"', mac_workflow)
        self.assertNotIn('chmod +x "$GITHUB_WORKSPACE/ptoas-dist/ptoas"', mac_workflow)
        self.assertIn('"$TEST_DIR/extracted/bin/ptoas" --version', mac_workflow)
        self.assertNotIn('"$TEST_DIR/extracted/ptoas" --version', mac_workflow)

    def test_wheel_import_smoke_covers_polluted_ptoas_version(self):
        script = (REPO_ROOT / "docker" / "test_wheel_imports.sh").read_text(encoding="utf-8")

        self.assertIn("Testing installed ptoas console entry under polluted Python and LLVM paths...", script)
        self.assertIn("POLLUTED_ENV_DIR=\"${TEST_TMPDIR}/polluted-env\"", script)
        self.assertIn('PYTHONPATH="${POLLUTED_ENV_DIR}"', script)
        self.assertIn("POLLUTED_PTOAS_VERSION_OUTPUT", script)
        self.assertIn('"/tmp/polluted-llvm"', script)
        self.assertIn('"/tmp/polluted-dylib"', script)


if __name__ == "__main__":
    unittest.main()
