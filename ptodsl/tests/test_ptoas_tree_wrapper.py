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
            build_python_root = temp_root / "build-python"
            package_root = build_python_root / "ptoas"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "_runtime_entry.py").write_text(
                """
calls = []


def resolve_tree_layout(wrapper, *, build_python_root=None, install_root=None, env=None):
    calls.append(
        ("resolve_tree_layout", str(wrapper), str(build_python_root), str(install_root))
    )
    return {
        "wrapper": str(wrapper),
        "build_python_root": str(build_python_root),
        "install_root": str(install_root),
    }


def launch(layout, user_args):
    calls.append(("launch", layout, list(user_args)))
    return 17
""".strip()
                + "\n",
                encoding="utf-8",
            )

            install_root = temp_root / "install"
            wrapper_path = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper_path.write_text("", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("test_ptoas_wrapper", WRAPPER_SOURCE)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)

            wrapper_module._DEFAULT_BUILD_PYTHON_ROOT = build_python_root
            wrapper_module._DEFAULT_INSTALL_ROOT = install_root
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
                            "resolve_tree_layout",
                            str(wrapper_path.resolve()),
                            str(build_python_root),
                            str(install_root),
                        ),
                        (
                            "launch",
                            {
                                "wrapper": str(wrapper_path.resolve()),
                                "build_python_root": str(build_python_root),
                                "install_root": str(install_root),
                            },
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


if __name__ == "__main__":
    unittest.main()
