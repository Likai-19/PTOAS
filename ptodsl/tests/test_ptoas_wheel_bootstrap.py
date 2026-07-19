#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ptoas_wheel_bootstrap as wheel_bootstrap


class WheelBootstrapTests(unittest.TestCase):
    def _make_runtime_tree(self, temp_root: Path) -> tuple[Path, Path]:
        python_root = temp_root / "site-packages"
        package_root = python_root / "ptoas"
        runtime_root = package_root / "_runtime"
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "_runtime_entry.py").write_text("", encoding="utf-8")
        (runtime_root / "lib").mkdir(parents=True, exist_ok=True)
        (runtime_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        (runtime_root / "lib" / "ptoas.so").write_text("fake shared module", encoding="utf-8")
        bootstrap = python_root / "ptoas_wheel_bootstrap.py"
        bootstrap.write_text("", encoding="utf-8")
        wrapper = temp_root / "bin" / "ptoas"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("", encoding="utf-8")
        return bootstrap, wrapper

    def test_stage1_reexecs_with_isolated_python_and_library_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bootstrap, wrapper = self._make_runtime_tree(temp_root)

            with mock.patch.dict(
                wheel_bootstrap.os.environ,
                {
                    "PYTHONPATH": "/tmp/external-python",
                    "LD_LIBRARY_PATH": "/tmp/external-lib",
                },
                clear=True,
            ), mock.patch.object(
                wheel_bootstrap,
                "__file__",
                str(bootstrap),
            ), mock.patch.object(
                wheel_bootstrap.sys,
                "argv",
                [str(wrapper), "--version"],
            ), mock.patch.object(
                wheel_bootstrap.os,
                "execve",
                side_effect=SystemExit(19),
            ) as execve:
                with self.assertRaises(SystemExit) as exc:
                    wheel_bootstrap.main()

        self.assertEqual(exc.exception.code, 19)
        execve.assert_called_once()
        invoked_path, invoked_argv, invoked_env = execve.call_args.args
        self.assertEqual(invoked_path, str(wrapper.resolve()))
        self.assertEqual(invoked_argv, [str(wrapper.resolve()), "--version"])
        self.assertEqual(invoked_env["PTOAS_WHEEL_STAGE2"], "1")
        self.assertEqual(invoked_env["PYTHONPATH"], str(bootstrap.parent.resolve()))
        self.assertEqual(
            invoked_env["LD_LIBRARY_PATH"],
            str((bootstrap.parent / "ptoas" / "_runtime" / "lib").resolve()),
        )
        self.assertEqual(
            invoked_env["DYLD_LIBRARY_PATH"],
            str((bootstrap.parent / "ptoas" / "_runtime" / "lib").resolve()),
        )

    def test_stage2_loads_runtime_entry_without_reexec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bootstrap, wrapper = self._make_runtime_tree(temp_root)
            @dataclass(frozen=True)
            class FakeLayout:
                wrapper: Path
                runtime_root: Path
                python_root: Path
                shared_module: Path
                tileops_dir: Path
                isolated_env: bool

            runtime_entry = SimpleNamespace(
                PTOASRuntimeLayout=FakeLayout,
                launch=mock.Mock(return_value=7),
            )

            with mock.patch.dict(
                wheel_bootstrap.os.environ,
                {"PTOAS_WHEEL_STAGE2": "1"},
                clear=True,
            ), mock.patch.object(
                wheel_bootstrap,
                "__file__",
                str(bootstrap),
            ), mock.patch.object(
                wheel_bootstrap.sys,
                "argv",
                [str(wrapper), "--help"],
            ), mock.patch.object(
                wheel_bootstrap,
                "_load_runtime_entry",
                return_value=runtime_entry,
            ) as load_runtime_entry, mock.patch.object(
                wheel_bootstrap.os,
                "execve",
            ) as execve:
                with self.assertRaises(SystemExit) as exc:
                    wheel_bootstrap.main()

        self.assertEqual(exc.exception.code, 7)
        execve.assert_not_called()
        load_runtime_entry.assert_called_once_with(bootstrap.parent.resolve(), (bootstrap.parent / "ptoas").resolve())
        runtime_entry.launch.assert_called_once_with(
            FakeLayout(
                wrapper=wrapper.resolve(),
                runtime_root=(bootstrap.parent / "ptoas" / "_runtime").resolve(),
                python_root=bootstrap.parent.resolve(),
                shared_module=(bootstrap.parent / "ptoas" / "_runtime" / "lib" / "ptoas.so").resolve(),
                tileops_dir=(bootstrap.parent / "ptoas" / "_runtime" / "share" / "ptoas" / "TileOps").resolve(),
                isolated_env=True,
            ),
            ["--help"],
        )

    def test_resolve_wheel_python_root_prefers_installed_package_when_shadowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            bootstrap, wrapper = self._make_runtime_tree(temp_root)
            package_root = bootstrap.parent / "ptoas"
            shadow_launcher = temp_root / "shadow" / "ptoas" / "_launcher.py"
            shadow_launcher.parent.mkdir(parents=True, exist_ok=True)
            shadow_launcher.write_text("", encoding="utf-8")

            def fake_get_path(name: str) -> str:
                if name in ("purelib", "platlib"):
                    return str(package_root.parent)
                return ""

            with mock.patch.object(
                wheel_bootstrap.sysconfig,
                "get_path",
                side_effect=fake_get_path,
            ):
                python_root, resolved_package_root = wheel_bootstrap._resolve_wheel_python_root(
                    wrapper,
                    str(shadow_launcher),
                )

        self.assertEqual(python_root, package_root.parent.resolve())
        self.assertEqual(resolved_package_root, package_root.resolve())

    def test_console_entry_reexec_ignores_shadowed_ptoas_and_polluted_library_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            wheel_root = temp_root / "wheel-site"
            package_root = wheel_root / "ptoas"
            runtime_root = package_root / "_runtime"
            runtime_lib_dir = runtime_root / "lib"
            tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
            result_path = temp_root / "launch-result.json"
            wrapper = temp_root / "venv" / "bin" / "ptoas"

            package_root.mkdir(parents=True, exist_ok=True)
            runtime_lib_dir.mkdir(parents=True, exist_ok=True)
            tileops_dir.mkdir(parents=True, exist_ok=True)
            (runtime_lib_dir / "ptoas.so").write_text("fake shared module", encoding="utf-8")
            (wheel_root / "ptoas_wheel_bootstrap.py").write_text(
                Path(wheel_bootstrap.__file__).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (package_root / "_runtime_entry.py").write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    from dataclasses import dataclass
                    from pathlib import Path

                    @dataclass(frozen=True)
                    class PTOASRuntimeLayout:
                        wrapper: Path
                        runtime_root: Path
                        python_root: Path
                        shared_module: Path
                        tileops_dir: Path
                        isolated_env: bool

                    def launch(layout, user_args):
                        result = {
                            "user_args": list(user_args),
                            "env": {
                                "PTOAS_WHEEL_STAGE2": os.environ.get("PTOAS_WHEEL_STAGE2"),
                                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
                            },
                            "layout": {
                                "wrapper": str(layout.wrapper),
                                "runtime_root": str(layout.runtime_root),
                                "python_root": str(layout.python_root),
                                "shared_module": str(layout.shared_module),
                                "tileops_dir": str(layout.tileops_dir),
                                "isolated_env": layout.isolated_env,
                            },
                        }
                        Path(os.environ["PTOAS_TEST_RESULT"]).write_text(
                            json.dumps(result, sort_keys=True),
                            encoding="utf-8",
                        )
                        return 23
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            shadow_root = temp_root / "shadow-site"
            shadow_package = shadow_root / "ptoas"
            shadow_package.mkdir(parents=True, exist_ok=True)
            (shadow_package / "__init__.py").write_text(
                "raise RuntimeError('shadow ptoas package must not be imported')\n",
                encoding="utf-8",
            )
            (shadow_package / "_runtime_entry.py").write_text(
                "raise RuntimeError('shadow runtime entry must not be imported')\n",
                encoding="utf-8",
            )

            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text(
                textwrap.dedent(
                    f"""
                    #!{sys.executable}
                    import ptoas_wheel_bootstrap
                    ptoas_wheel_bootstrap.main()
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

            env = os.environ.copy()
            env.pop("PTOAS_WHEEL_STAGE2", None)
            env["PYTHONPATH"] = os.pathsep.join([str(shadow_root), str(wheel_root)])
            env["LD_LIBRARY_PATH"] = "/tmp/polluted-llvm"
            env["DYLD_LIBRARY_PATH"] = "/tmp/polluted-dylib"
            env["PTOAS_TEST_RESULT"] = str(result_path)

            completed = subprocess.run(
                [str(wrapper), "--version"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                23,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(result["user_args"], ["--version"])
        self.assertEqual(result["env"]["PTOAS_WHEEL_STAGE2"], "1")
        self.assertEqual(result["env"]["PYTHONPATH"], str(wheel_root.resolve()))
        self.assertEqual(result["env"]["LD_LIBRARY_PATH"], str(runtime_lib_dir.resolve()))
        self.assertEqual(result["env"]["DYLD_LIBRARY_PATH"], str(runtime_lib_dir.resolve()))
        self.assertNotIn(str(shadow_root), result["env"]["PYTHONPATH"])
        self.assertNotIn("/tmp/polluted-llvm", result["env"]["LD_LIBRARY_PATH"])
        self.assertNotIn("/tmp/polluted-dylib", result["env"]["DYLD_LIBRARY_PATH"])
        self.assertEqual(
            result["layout"],
            {
                "wrapper": str(wrapper.resolve()),
                "runtime_root": str(runtime_root.resolve()),
                "python_root": str(wheel_root.resolve()),
                "shared_module": str((runtime_lib_dir / "ptoas.so").resolve()),
                "tileops_dir": str(tileops_dir.resolve()),
                "isolated_env": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
