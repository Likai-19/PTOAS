#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptoas import _launcher


class WheelLauncherTests(unittest.TestCase):
    def _make_runtime_tree(self, temp_root: Path) -> tuple[Path, Path, Path]:
        package_root = temp_root / "site-packages" / "ptoas"
        runtime_root = package_root / "_runtime"
        (runtime_root / "bin").mkdir(parents=True, exist_ok=True)
        (runtime_root / "lib").mkdir(parents=True, exist_ok=True)
        (runtime_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        wrapper = temp_root / "bin" / "ptoas"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("", encoding="utf-8")
        (temp_root / "site-packages" / "ptodsl").mkdir(parents=True, exist_ok=True)
        (temp_root / "site-packages" / "tilelang_dsl").mkdir(parents=True, exist_ok=True)
        (temp_root / "site-packages" / "mlir").mkdir(parents=True, exist_ok=True)
        shared_module = runtime_root / "lib" / "ptoas.so"
        shared_module.write_text("fake shared module", encoding="utf-8")
        return package_root, wrapper, shared_module

    def _make_editable_tree(self, temp_root: Path) -> tuple[Path, Path, Path]:
        repo_root = temp_root / "repo"
        package_root = repo_root / "ptodsl" / "ptoas"
        install_root = repo_root / "install"
        (package_root).mkdir(parents=True, exist_ok=True)
        (install_root / "bin").mkdir(parents=True, exist_ok=True)
        (install_root / "lib").mkdir(parents=True, exist_ok=True)
        (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        (install_root / "tilelang_dsl").mkdir(parents=True, exist_ok=True)
        (install_root / "mlir").mkdir(parents=True, exist_ok=True)
        wrapper = install_root / "bin" / "ptoas"
        wrapper.write_text("", encoding="utf-8")
        shared_module = install_root / "lib" / "ptoas.so"
        shared_module.write_text("fake shared module", encoding="utf-8")
        return package_root, install_root, wrapper

    def test_launcher_exports_runtime_contract_and_injects_default_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root, wrapper, shared_module = self._make_runtime_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.dict(_launcher.os.environ, {}, clear=True), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", [str(wrapper), "--version"]
            ), mock.patch.object(_launcher, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

                self.assertEqual(exc.exception.code, 0)
                load_entrypoint.assert_called_once_with(shared_module, package_root / "_runtime" / "lib")
                entrypoint = load_entrypoint.return_value
                self.assertEqual(entrypoint.call_count, 1)
                argc, c_argv = entrypoint.call_args.args
                self.assertEqual(argc, 6)
                self.assertEqual([
                    c_argv[i].decode("utf-8") for i in range(argc)
                ], [
                    str(wrapper),
                    "--tilelang-path",
                    str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
                    "--tilelang-pkg-path",
                    str(package_root.parent),
                    "--version",
                ])
                env = _launcher.os.environ
                self.assertEqual(env["PTOAS_HOME"], str(package_root / "_runtime"))
                self.assertEqual(env["PTOAS_BIN"], str(wrapper))
                self.assertEqual(
                    env["PTOAS_TILEOPS_DIR"],
                    str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
                )
                self.assertEqual(env["PATH"].split(os.pathsep)[0], str(wrapper.parent))
                self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(package_root.parent))

    def test_resolve_runtime_root_defaults_to_repo_install_tree(self):
        package_root = Path("/tmp/repo/ptodsl/ptoas")
        with mock.patch.dict(_launcher.os.environ, {}, clear=True):
            runtime_root = _launcher._resolve_runtime_root(package_root)
        self.assertEqual(runtime_root, Path("/tmp/repo/install"))

    def test_resolve_wrapper_path_falls_back_to_shutil_lookup(self):
        with mock.patch.object(_launcher.sys, "argv", ["ptoas"]), mock.patch.object(
            _launcher.shutil, "which", return_value="/tmp/bin/ptoas"
        ):
            wrapper = _launcher._resolve_wrapper_path()

        self.assertEqual(wrapper, Path("/tmp/bin/ptoas"))

    def test_load_shared_entrypoint_configures_in_process_ctypes_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_lib_dir = temp_root / "runtime" / "lib"
            runtime_lib_dir.mkdir(parents=True, exist_ok=True)
            dep = runtime_lib_dir / "libMLIRMlirOptMain.so.21.1"
            dep.write_text("fake dep", encoding="utf-8")
            shared_module = temp_root / "runtime" / "pto" / "ptoas.so"
            shared_module.parent.mkdir(parents=True, exist_ok=True)
            shared_module.write_text("fake shared module", encoding="utf-8")

            def fake_ldd(cmd, *, text, stderr, env):
                target = Path(cmd[1]).resolve()
                if target == shared_module.resolve():
                    return f"\tlibMLIRMlirOptMain.so.21.1 => {dep} (0x0)\n"
                return ""

            dep_library = mock.Mock()
            shared_library = mock.Mock()
            with mock.patch.object(_launcher.subprocess, "check_output", side_effect=fake_ldd), mock.patch.object(
                _launcher.ctypes, "CDLL", side_effect=[dep_library, shared_library]
            ) as load_library:
                entrypoint = _launcher._load_shared_entrypoint(shared_module, runtime_lib_dir)

        self.assertEqual(
            [call.args[0] for call in load_library.call_args_list],
            [str(dep.resolve()), str(shared_module.resolve())],
        )
        self.assertTrue(
            all(
                call.kwargs["mode"] == getattr(_launcher.ctypes, "RTLD_GLOBAL", 0)
                for call in load_library.call_args_list
            )
        )
        self.assertIs(entrypoint, shared_library.ptoas_entrypoint)
        self.assertEqual(entrypoint.argtypes, [_launcher.ctypes.c_int, _launcher.ctypes.POINTER(_launcher.ctypes.c_char_p)])
        self.assertIs(entrypoint.restype, _launcher.ctypes.c_int)

    def test_launcher_falls_back_to_env_install_tree_for_editable_installs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root, install_root, wrapper = self._make_editable_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.dict(
                _launcher.os.environ,
                {"PTO_INSTALL_DIR": str(install_root)},
                clear=True,
            ), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", [str(wrapper), "--version"]
            ), mock.patch.object(_launcher, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

                self.assertEqual(exc.exception.code, 0)
                load_entrypoint.assert_called_once_with(install_root / "lib" / "ptoas.so", install_root / "lib")
                entrypoint = load_entrypoint.return_value
                argc, c_argv = entrypoint.call_args.args
                self.assertEqual([
                    c_argv[i].decode("utf-8") for i in range(argc)
                ], [
                    str(wrapper),
                    "--tilelang-path",
                    str(install_root / "share" / "ptoas" / "TileOps"),
                    "--tilelang-pkg-path",
                    str(install_root),
                    "--version",
                ])
                env = _launcher.os.environ
                self.assertEqual(env["PTOAS_HOME"], str(install_root))
                self.assertEqual(env["PTOAS_BIN"], str(wrapper))
                self.assertEqual(
                    env["PTOAS_TILEOPS_DIR"],
                    str(install_root / "share" / "ptoas" / "TileOps"),
                )
                self.assertEqual(env["PATH"].split(os.pathsep)[0], str(wrapper.parent))
                self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(install_root))

    def test_launcher_respects_explicit_tilelang_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root, wrapper, shared_module = self._make_runtime_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.object(_launcher, "__file__", str(fake_launcher)), mock.patch.object(
                _launcher.sys,
                "argv",
                [
                    str(wrapper),
                    "--tilelang-path=/tmp/custom-tileops",
                    "--tilelang-pkg-path",
                    "/tmp/custom-python",
                    "--help",
                ],
            ), mock.patch.object(_launcher, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

            self.assertEqual(exc.exception.code, 0)
            entrypoint = load_entrypoint.return_value
            argc, c_argv = entrypoint.call_args.args
            self.assertEqual([
                c_argv[i].decode("utf-8") for i in range(argc)
            ], [
                str(wrapper),
                "--tilelang-path=/tmp/custom-tileops",
                "--tilelang-pkg-path",
                "/tmp/custom-python",
                "--help",
            ])

    def test_launcher_prefers_local_shared_module_next_to_build_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = temp_root / "build" / "python" / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")
            wrapper = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text("", encoding="utf-8")
            shared_module = wrapper.parent / "ptoas.so"
            shared_module.write_text("fake shared module", encoding="utf-8")

            with mock.patch.dict(
                _launcher.os.environ,
                {"PTO_INSTALL_DIR": str(temp_root / "install")},
                clear=True,
            ), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", [str(wrapper), "--version"]
            ), mock.patch.object(
                _launcher, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)
            ) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

            self.assertEqual(exc.exception.code, 0)
            load_entrypoint.assert_called_once_with(shared_module, temp_root / "install" / "lib")

    def test_launcher_skips_empty_placeholder_shared_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = temp_root / "build" / "python" / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")
            wrapper = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text("", encoding="utf-8")
            (temp_root / "build" / "python" / "pto").mkdir(parents=True, exist_ok=True)
            (temp_root / "build" / "python" / "pto" / "ptoas.so").write_text("", encoding="utf-8")
            install_shared_module = temp_root / "install" / "lib" / "ptoas.so"
            install_shared_module.parent.mkdir(parents=True, exist_ok=True)
            install_shared_module.write_text("fake shared module", encoding="utf-8")

            with mock.patch.dict(
                _launcher.os.environ,
                {"PTO_INSTALL_DIR": str(temp_root / "install")},
                clear=True,
            ), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", [str(wrapper), "--version"]
            ), mock.patch.object(
                _launcher, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)
            ) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

            self.assertEqual(exc.exception.code, 0)
            load_entrypoint.assert_called_once_with(install_shared_module, temp_root / "install" / "lib")


if __name__ == "__main__":
    unittest.main()
