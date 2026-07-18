#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _ptoas_build_backend as build_backend


class PtoasBuildBackendTests(unittest.TestCase):
    def test_prepare_metadata_honors_package_name_override(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            build_backend.os.environ,
            {
                "PTOAS_PYTHON_PACKAGE_NAME": "ptoas-vmi",
                "PTOAS_PYTHON_PACKAGE_VERSION": "0.1.0",
            },
            clear=False,
        ):
            dist_info_name = build_backend.prepare_metadata_for_build_wheel(temp_dir)

            dist_info = Path(temp_dir) / dist_info_name
            metadata = (dist_info / "METADATA").read_text(encoding="utf-8")

        self.assertEqual(dist_info_name, "ptoas_vmi-0.1.0.dist-info")
        self.assertIn("Name: ptoas-vmi", metadata)
        self.assertIn("Version: 0.1.0", metadata)

    def test_cmake_build_driver_is_used_for_build_and_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo = temp_root / "repo"
            build_dir = temp_root / "build"
            llvm_build = temp_root / "llvm-build"
            install_dir = temp_root / "install"
            (repo / "cmake").mkdir(parents=True)
            (install_dir / "lib").mkdir(parents=True)
            (install_dir / "lib" / "ptoas.so").write_bytes(b"fake shared module")

            check_call_args = []

            def fake_check_call(args):
                check_call_args.append(args)

            with mock.patch.object(build_backend, "_REPO", repo), mock.patch.object(
                build_backend, "_BUILD_DIR", build_dir
            ), mock.patch.object(
                build_backend, "_LLVM_BUILD_DIR", llvm_build
            ), mock.patch.object(
                build_backend, "_PTO_INSTALL_DIR", install_dir
            ), mock.patch.object(
                build_backend,
                "_should_use_linux_hardening_cache",
                return_value=False,
            ), mock.patch.object(
                build_backend.subprocess,
                "check_output",
                return_value="/tmp/pybind11-cmake",
            ), mock.patch.object(
                build_backend.subprocess,
                "check_call",
                side_effect=fake_check_call,
            ), mock.patch.object(
                build_backend,
                "_assert_installed_ptodsl_payload",
            ):
                build_backend._cmake_configure_and_build()

        self.assertEqual(
            check_call_args[-2],
            ["cmake", "--build", str(build_dir)],
        )
        self.assertEqual(
            check_call_args[-1],
            ["cmake", "--build", str(build_dir), "--target", "install"],
        )

    def test_installed_shared_module_must_be_non_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            install_dir = Path(temp_dir) / "install"
            shared_module = install_dir / "lib" / "ptoas.so"
            shared_module.parent.mkdir(parents=True)
            shared_module.write_bytes(b"")

            with mock.patch.object(build_backend, "_PTO_INSTALL_DIR", install_dir):
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    build_backend._assert_installed_ptoas_shared_module()


if __name__ == "__main__":
    unittest.main()
