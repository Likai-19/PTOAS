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


if __name__ == "__main__":
    unittest.main()
