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

from ptoas import _runtime_entry


class RuntimeEntryTests(unittest.TestCase):
    def test_launch_injects_runtime_environment_and_calls_shared_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            wrapper = temp_root / "bin" / "ptoas"
            runtime_root = temp_root / "runtime"
            python_root = temp_root / "python"
            tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
            shared_module = runtime_root / "lib" / "ptoas.so"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            runtime_root.mkdir(parents=True, exist_ok=True)
            python_root.mkdir(parents=True, exist_ok=True)
            tileops_dir.mkdir(parents=True, exist_ok=True)
            shared_module.parent.mkdir(parents=True, exist_ok=True)
            shared_module.write_text("fake shared module", encoding="utf-8")

            layout = _runtime_entry.PTOASRuntimeLayout(
                wrapper=wrapper,
                runtime_root=runtime_root,
                python_root=python_root,
                shared_module=shared_module,
                tileops_dir=tileops_dir,
                isolated_env=False,
            )
            entrypoint = mock.Mock(return_value=0)

            with mock.patch.dict(
                _runtime_entry.os.environ,
                {
                    "PYTHONPATH": "/tmp/external-python",
                    "LD_LIBRARY_PATH": "/tmp/external-lib",
                    "DYLD_LIBRARY_PATH": "/tmp/external-dylib",
                },
                clear=True,
            ), mock.patch.object(
                _runtime_entry,
                "_load_shared_entrypoint",
                return_value=entrypoint,
            ) as load_shared_entrypoint:
                exit_code = _runtime_entry.launch(layout, ["--version"])
                env = dict(_runtime_entry.os.environ)

        self.assertEqual(exit_code, 0)
        load_shared_entrypoint.assert_called_once_with(layout.shared_module, layout.runtime_root / "lib")
        argc, c_argv = entrypoint.call_args.args
        self.assertEqual(
            [c_argv[i].decode("utf-8") for i in range(argc)],
            [
                str(layout.wrapper),
                "--tilelang-path",
                str(layout.tileops_dir),
                "--tilelang-pkg-path",
                str(layout.python_root),
                "--ptodsl-pkg-path",
                str(layout.python_root),
                "--version",
            ],
        )
        self.assertEqual(env["PTOAS_HOME"], str(layout.runtime_root))
        self.assertEqual(env["PTOAS_BIN"], str(layout.wrapper))
        self.assertEqual(env["PTOAS_TILEOPS_DIR"], str(layout.tileops_dir))
        self.assertEqual(env["PYTHONPATH"], str(layout.python_root) + os.pathsep + "/tmp/external-python")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(layout.runtime_root / "lib") + os.pathsep + "/tmp/external-lib")
        self.assertEqual(env["DYLD_LIBRARY_PATH"], str(layout.runtime_root / "lib") + os.pathsep + "/tmp/external-dylib")

    def test_launch_isolates_runtime_environment_for_wheel_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            wrapper = temp_root / "bin" / "ptoas"
            runtime_root = temp_root / "runtime"
            python_root = temp_root / "python"
            tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
            shared_module = runtime_root / "lib" / "ptoas.so"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            runtime_root.mkdir(parents=True, exist_ok=True)
            python_root.mkdir(parents=True, exist_ok=True)
            tileops_dir.mkdir(parents=True, exist_ok=True)
            shared_module.parent.mkdir(parents=True, exist_ok=True)
            shared_module.write_text("fake shared module", encoding="utf-8")

            layout = _runtime_entry.PTOASRuntimeLayout(
                wrapper=wrapper,
                runtime_root=runtime_root,
                python_root=python_root,
                shared_module=shared_module,
                tileops_dir=tileops_dir,
                isolated_env=True,
            )
            entrypoint = mock.Mock(return_value=0)

            with mock.patch.dict(
                _runtime_entry.os.environ,
                {
                    "PYTHONPATH": "/tmp/external-python",
                    "LD_LIBRARY_PATH": "/tmp/external-lib",
                    "DYLD_LIBRARY_PATH": "/tmp/external-dylib",
                },
                clear=True,
            ), mock.patch.object(
                _runtime_entry,
                "_load_shared_entrypoint",
                return_value=entrypoint,
            ):
                exit_code = _runtime_entry.launch(layout, ["--help"])
                env = dict(_runtime_entry.os.environ)

        self.assertEqual(exit_code, 0)
        self.assertEqual(env["PYTHONPATH"], str(layout.python_root))
        self.assertEqual(env["LD_LIBRARY_PATH"], str(layout.runtime_root / "lib"))
        self.assertEqual(env["DYLD_LIBRARY_PATH"], str(layout.runtime_root / "lib"))

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
