#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_SOURCE = REPO_ROOT / "tools" / "ptoas" / "ptoas_wrapper.py"


class TreeWrapperTests(unittest.TestCase):
    def test_build_tree_wrapper_bootstraps_runtime_entry_and_delegates_shared_launcher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            build_python_root = build_root / "python"
            package_root = build_python_root / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "_runtime_entry.py").write_text(
                """
from dataclasses import dataclass

calls = []


@dataclass(frozen=True)
class PTOASRuntimeLayout:
    wrapper: object
    runtime_root: object
    python_root: object
    shared_module: object
    tileops_dir: object
    isolated_env: bool


def launch(layout, user_args):
    calls.append(("launch", layout, list(user_args)))
    return 17
""".strip()
                + "\n",
                encoding="utf-8",
            )

            runtime_root = build_root / "runtime-staging"
            (runtime_root / "lib").mkdir(parents=True, exist_ok=True)
            (runtime_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (runtime_root / "lib" / "ptoas.so").write_text("fake shared module", encoding="utf-8")
            install_root = temp_root / "install"
            (install_root / "lib").mkdir(parents=True, exist_ok=True)
            (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (install_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (install_root / "lib" / "ptoas.so").write_text("install shared module", encoding="utf-8")
            (install_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            wrapper_path = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)

            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            saved_ptoas = sys.modules.pop("ptoas", None)
            saved_runtime_entry = sys.modules.pop("ptoas._runtime_entry", None)
            try:
                sys.path = [entry for entry in sys.path if entry != str(build_python_root)]
                sys.argv = [str(wrapper_path), "--version"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertEqual(exc.exception.code, 17)
                self.assertEqual(sys.path[0], str(build_python_root))

                runtime_entry = sys.modules["ptoas._runtime_entry"]
                self.assertEqual(
                    runtime_entry.calls,
                    [
                        (
                            "launch",
                            runtime_entry.PTOASRuntimeLayout(
                                wrapper=wrapper_path.resolve(),
                                runtime_root=runtime_root.resolve(),
                                python_root=build_python_root.resolve(),
                                shared_module=(runtime_root / "lib" / "ptoas.so").resolve(),
                                tileops_dir=(runtime_root / "share" / "ptoas" / "TileOps").resolve(),
                                isolated_env=False,
                            ),
                            ["--version"],
                        ),
                    ],
                )
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv
                sys.modules.pop("ptoas", None)
                sys.modules.pop("ptoas._runtime_entry", None)
                if saved_ptoas is not None:
                    sys.modules["ptoas"] = saved_ptoas
                if saved_runtime_entry is not None:
                    sys.modules["ptoas._runtime_entry"] = saved_runtime_entry

    def test_install_tree_wrapper_bootstraps_install_prefix_and_delegates_shared_launcher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            build_python_root = build_root / "python"
            (build_python_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (build_python_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            (build_root / "runtime-staging" / "lib").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "lib" / "ptoas.so").write_text("build shared module", encoding="utf-8")

            install_root = temp_root / "install"
            install_python_root = install_root
            package_root = install_python_root / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "_runtime_entry.py").write_text(
                """
from dataclasses import dataclass

calls = []


@dataclass(frozen=True)
class PTOASRuntimeLayout:
    wrapper: object
    runtime_root: object
    python_root: object
    shared_module: object
    tileops_dir: object
    isolated_env: bool


def launch(layout, user_args):
    calls.append(("launch", layout, list(user_args)))
    return 23
""".strip()
                + "\n",
                encoding="utf-8",
            )
            (install_root / "lib").mkdir(parents=True, exist_ok=True)
            (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (install_root / "lib" / "ptoas.so").write_text("install shared module", encoding="utf-8")
            wrapper_path = install_root / "bin" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper_install", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)

            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            saved_ptoas = sys.modules.pop("ptoas", None)
            saved_runtime_entry = sys.modules.pop("ptoas._runtime_entry", None)
            try:
                sys.path = [entry for entry in sys.path if entry != str(install_python_root)]
                sys.argv = [str(wrapper_path), "--help"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertEqual(exc.exception.code, 23)
                self.assertEqual(sys.path[0], str(install_python_root))

                runtime_entry = sys.modules["ptoas._runtime_entry"]
                self.assertEqual(
                    runtime_entry.calls,
                    [
                        (
                            "launch",
                            runtime_entry.PTOASRuntimeLayout(
                                wrapper=wrapper_path.resolve(),
                                runtime_root=install_root.resolve(),
                                python_root=install_python_root.resolve(),
                                shared_module=(install_root / "lib" / "ptoas.so").resolve(),
                                tileops_dir=(install_root / "share" / "ptoas" / "TileOps").resolve(),
                                isolated_env=False,
                            ),
                            ["--help"],
                        ),
                    ],
                )
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv
                sys.modules.pop("ptoas", None)
                sys.modules.pop("ptoas._runtime_entry", None)
                if saved_ptoas is not None:
                    sys.modules["ptoas"] = saved_ptoas
                if saved_runtime_entry is not None:
                    sys.modules["ptoas._runtime_entry"] = saved_runtime_entry

    def test_build_tree_wrapper_does_not_fall_back_to_install_prefix_python_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            (build_root / "runtime-staging" / "lib").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "lib" / "ptoas.so").write_text("fake shared module", encoding="utf-8")

            install_root = temp_root / "install"
            (install_root / "lib").mkdir(parents=True, exist_ok=True)
            (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (install_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (install_root / "lib" / "ptoas.so").write_text("install shared module", encoding="utf-8")
            (install_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            wrapper_path = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper_build_fallback", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)
            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            try:
                sys.path = [entry for entry in sys.path if entry != str(install_root)]
                sys.argv = [str(wrapper_path), "--version"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertIn("build-tree ptoas Python package root", str(exc.exception))
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv

    def test_build_tree_wrapper_fails_when_build_runtime_layout_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            build_python_root = build_root / "python"
            package_root = build_python_root / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "_runtime_entry.py").write_text(
                """
from dataclasses import dataclass


@dataclass(frozen=True)
class PTOASRuntimeLayout:
    wrapper: object
    runtime_root: object
    python_root: object
    shared_module: object
    tileops_dir: object
    isolated_env: bool


def launch(layout, user_args):
    raise AssertionError("missing runtime layout must fail before launch")
""".strip()
                + "\n",
                encoding="utf-8",
            )
            install_root = temp_root / "install"
            (install_root / "lib").mkdir(parents=True, exist_ok=True)
            (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (install_root / "lib" / "ptoas.so").write_text("install shared module", encoding="utf-8")
            wrapper_path = build_root / "tools" / "ptoas" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper_build_missing_runtime", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)
            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            saved_ptoas = sys.modules.pop("ptoas", None)
            saved_runtime_entry = sys.modules.pop("ptoas._runtime_entry", None)
            try:
                sys.path = [entry for entry in sys.path if entry != str(build_python_root)]
                sys.argv = [str(wrapper_path), "--version"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertIn("ptoas runtime tree", str(exc.exception))
                self.assertIn(str(build_root / "runtime-staging"), str(exc.exception))
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv
                sys.modules.pop("ptoas", None)
                sys.modules.pop("ptoas._runtime_entry", None)
                if saved_ptoas is not None:
                    sys.modules["ptoas"] = saved_ptoas
                if saved_runtime_entry is not None:
                    sys.modules["ptoas._runtime_entry"] = saved_runtime_entry

    def test_install_tree_wrapper_does_not_fall_back_to_build_tree_python_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            build_python_root = build_root / "python"
            (build_python_root / "ptoas").mkdir(parents=True, exist_ok=True)
            (build_python_root / "ptoas" / "_runtime_entry.py").write_text("", encoding="utf-8")
            (build_root / "runtime-staging" / "lib").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "lib" / "ptoas.so").write_text("build shared module", encoding="utf-8")

            install_root = temp_root / "install"
            wrapper_path = install_root / "bin" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper_install_fallback", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)
            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            try:
                sys.path = [entry for entry in sys.path if entry != str(build_python_root)]
                sys.argv = [str(wrapper_path), "--help"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertIn("install-tree ptoas Python package root", str(exc.exception))
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv

    def test_install_tree_wrapper_fails_when_install_runtime_layout_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            build_root = temp_root / "build"
            (build_root / "runtime-staging" / "lib").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
            (build_root / "runtime-staging" / "lib" / "ptoas.so").write_text("build shared module", encoding="utf-8")

            install_root = temp_root / "install"
            package_root = install_root / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "_runtime_entry.py").write_text(
                """
from dataclasses import dataclass


@dataclass(frozen=True)
class PTOASRuntimeLayout:
    wrapper: object
    runtime_root: object
    python_root: object
    shared_module: object
    tileops_dir: object
    isolated_env: bool


def launch(layout, user_args):
    raise AssertionError("missing runtime layout must fail before launch")
""".strip()
                + "\n",
                encoding="utf-8",
            )
            wrapper_path = install_root / "bin" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper_install_missing_runtime", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)
            wrapper_module.__file__ = str(wrapper_path)

            saved_sys_path = list(sys.path)
            saved_argv = list(sys.argv)
            saved_ptoas = sys.modules.pop("ptoas", None)
            saved_runtime_entry = sys.modules.pop("ptoas._runtime_entry", None)
            try:
                sys.path = [entry for entry in sys.path if entry != str(install_root)]
                sys.argv = [str(wrapper_path), "--help"]

                with self.assertRaises(SystemExit) as exc:
                    wrapper_module.main()

                self.assertIn("ptoas runtime tree", str(exc.exception))
                self.assertIn(str(install_root), str(exc.exception))
            finally:
                sys.path = saved_sys_path
                sys.argv = saved_argv
                sys.modules.pop("ptoas", None)
                sys.modules.pop("ptoas._runtime_entry", None)
                if saved_ptoas is not None:
                    sys.modules["ptoas"] = saved_ptoas
                if saved_runtime_entry is not None:
                    sys.modules["ptoas._runtime_entry"] = saved_runtime_entry


if __name__ == "__main__":
    unittest.main()
