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

from ptoas import _launcher, _runtime_entry


class RuntimeEntryTests(unittest.TestCase):
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

    def _make_install_tree(self, temp_root: Path) -> tuple[Path, Path]:
        install_root = temp_root / "install"
        (install_root / "bin").mkdir(parents=True, exist_ok=True)
        (install_root / "lib").mkdir(parents=True, exist_ok=True)
        (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        (install_root / "ptoas").mkdir(parents=True, exist_ok=True)
        (install_root / "ptodsl").mkdir(parents=True, exist_ok=True)
        (install_root / "tilelang_dsl").mkdir(parents=True, exist_ok=True)
        wrapper = install_root / "bin" / "ptoas"
        wrapper.write_text("", encoding="utf-8")
        (install_root / "lib" / "ptoas.so").write_text("fake shared module", encoding="utf-8")
        (install_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
        return install_root, wrapper

    def test_wheel_launcher_exports_runtime_contract_and_injects_default_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root, wrapper, shared_module = self._make_runtime_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.dict(_runtime_entry.os.environ, {}, clear=True), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", [str(wrapper), "--version"]
            ), mock.patch.object(
                _runtime_entry, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)
            ) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

                env = dict(_runtime_entry.os.environ)
                self.assertEqual(exc.exception.code, 0)
                load_entrypoint.assert_called_once_with(shared_module, package_root / "_runtime" / "lib")
                entrypoint = load_entrypoint.return_value
                argc, c_argv = entrypoint.call_args.args
                self.assertEqual(
                    [c_argv[i].decode("utf-8") for i in range(argc)],
                    [
                        str(wrapper),
                        "--tilelang-path",
                        str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
                        "--tilelang-pkg-path",
                        str(package_root.parent),
                        "--version",
                    ],
                )
                self.assertEqual(env["PTOAS_HOME"], str(package_root / "_runtime"))
                self.assertEqual(env["PTOAS_BIN"], str(wrapper))
                self.assertEqual(
                    env["PTOAS_TILEOPS_DIR"],
                    str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
                )
                self.assertEqual(env["PATH"].split(os.pathsep)[0], str(wrapper.parent))
                self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(package_root.parent))

    def test_wheel_launcher_respects_explicit_tilelang_flags(self):
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
            ), mock.patch.object(
                _runtime_entry, "_load_shared_entrypoint", return_value=mock.Mock(return_value=0)
            ) as load_entrypoint:
                with self.assertRaises(SystemExit) as exc:
                    _launcher.main()

            self.assertEqual(exc.exception.code, 0)
            entrypoint = load_entrypoint.return_value
            argc, c_argv = entrypoint.call_args.args
            self.assertEqual(
                [c_argv[i].decode("utf-8") for i in range(argc)],
                [
                    str(wrapper),
                    "--tilelang-path=/tmp/custom-tileops",
                    "--tilelang-pkg-path",
                    "/tmp/custom-python",
                    "--help",
                ],
            )

    def test_resolve_wrapper_path_falls_back_to_shutil_lookup(self):
        with mock.patch.object(_runtime_entry.sys, "argv", ["ptoas"]), mock.patch.object(
            _runtime_entry.shutil, "which", return_value="/tmp/bin/ptoas"
        ):
            wrapper = _runtime_entry.resolve_wrapper_path()

        self.assertEqual(wrapper, Path("/tmp/bin/ptoas"))

    def test_resolve_tree_layout_prefers_local_install_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            install_root, wrapper = self._make_install_tree(temp_root)

            layout = _runtime_entry.resolve_tree_layout(
                wrapper,
                build_python_root=temp_root / "build" / "python",
                install_root=temp_root / "unused-install",
                env={},
            )

        self.assertEqual(layout.runtime_root, install_root)
        self.assertEqual(layout.python_root, install_root)
        self.assertEqual(layout.shared_module, install_root / "lib" / "ptoas.so")
        self.assertEqual(layout.tileops_dir, install_root / "share" / "ptoas" / "TileOps")

    def test_resolve_tree_layout_uses_env_install_root_for_build_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_wrapper = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            build_wrapper.parent.mkdir(parents=True, exist_ok=True)
            build_wrapper.write_text("", encoding="utf-8")
            build_python_root = temp_root / "build" / "python"
            (build_python_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (build_python_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            install_root, _ = self._make_install_tree(temp_root)

            layout = _runtime_entry.resolve_tree_layout(
                build_wrapper,
                build_python_root=build_python_root,
                install_root=temp_root / "unused-install",
                env={"PTO_INSTALL_DIR": str(install_root)},
            )

        self.assertEqual(layout.runtime_root, install_root)
        self.assertEqual(layout.python_root, build_python_root)
        self.assertEqual(layout.shared_module, install_root / "lib" / "ptoas.so")

    def test_resolve_tree_layout_prefers_install_shared_module_over_build_python_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_wrapper = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            build_wrapper.parent.mkdir(parents=True, exist_ok=True)
            build_wrapper.write_text("", encoding="utf-8")
            build_python_root = temp_root / "build" / "python"
            (build_python_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (build_python_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            (build_python_root / "pto").mkdir(parents=True, exist_ok=True)
            (build_python_root / "pto" / "ptoas.so").write_text("build python shared module", encoding="utf-8")
            install_root, _ = self._make_install_tree(temp_root)

            layout = _runtime_entry.resolve_tree_layout(
                build_wrapper,
                build_python_root=build_python_root,
                install_root=install_root,
                env={"PTO_INSTALL_DIR": str(install_root)},
            )

        self.assertEqual(layout.shared_module, install_root / "lib" / "ptoas.so")

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

            dep_library = mock.Mock()
            shared_library = mock.Mock()
            with mock.patch.object(
                _runtime_entry.ctypes, "CDLL", side_effect=[dep_library, shared_library]
            ) as load_library:
                entrypoint = _runtime_entry._load_shared_entrypoint(shared_module, runtime_lib_dir)

        self.assertEqual(
            [call.args[0] for call in load_library.call_args_list],
            [str(dep.resolve()), str(shared_module.resolve())],
        )
        self.assertTrue(
            all(
                call.kwargs["mode"] == getattr(_runtime_entry.ctypes, "RTLD_GLOBAL", 0)
                for call in load_library.call_args_list
            )
        )
        self.assertIs(entrypoint, shared_library.ptoas_entrypoint)
        self.assertEqual(
            entrypoint.argtypes,
            [_runtime_entry.ctypes.c_int, _runtime_entry.ctypes.POINTER(_runtime_entry.ctypes.c_char_p)],
        )
        self.assertIs(entrypoint.restype, _runtime_entry.ctypes.c_int)

    def test_preload_runtime_libraries_retries_until_dependencies_resolve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_lib_dir = temp_root / "runtime" / "lib"
            runtime_lib_dir.mkdir(parents=True, exist_ok=True)
            dep_a = runtime_lib_dir / "libA.so.1"
            dep_b = runtime_lib_dir / "libB.so.1"
            dep_a.write_text("fake dep a", encoding="utf-8")
            dep_b.write_text("fake dep b", encoding="utf-8")
            shared_module = temp_root / "runtime" / "pto" / "ptoas.so"
            shared_module.parent.mkdir(parents=True, exist_ok=True)
            shared_module.write_text("fake shared module", encoding="utf-8")

            loaded: list[str] = []
            dep_b_attempts = 0

            def fake_cdll(path: str, *, mode: int):
                nonlocal dep_b_attempts
                loaded.append(path)
                if Path(path).resolve() == dep_b.resolve() and dep_b_attempts == 0:
                    dep_b_attempts += 1
                    raise OSError("missing libA dependency")
                return mock.Mock()

            with mock.patch.object(_runtime_entry.ctypes, "CDLL", side_effect=fake_cdll):
                _runtime_entry._preload_runtime_libraries(shared_module, runtime_lib_dir)

        self.assertEqual(
            loaded,
            [
                str(dep_a.resolve()),
                str(dep_b.resolve()),
                str(dep_b.resolve()),
            ],
        )


if __name__ == "__main__":
    unittest.main()
