# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend-aware kernel object production in the ptodsl native build (no device needed)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ptodsl._runtime import cache, native_build
from ptodsl._runtime.native_build import (
    _compile_native_artifacts,
    _is_elf_object,
    _resolve_ptoas_backend,
)

PTOAS_BIN = "/fake/bin/ptoas"
BISHENG_BIN = "/fake/bin/bisheng"
ELF_BYTES = b"\x7fELF fake fatobj"
CPP_BYTES = b'#include "pto/pto-inst.hpp"\n'


class ResolvePtoasBackendTest(unittest.TestCase):
    @staticmethod
    def _spec(backend):
        return SimpleNamespace(backend=backend)

    def test_override_wins_over_module_attribute(self):
        self.assertEqual(
            _resolve_ptoas_backend(self._spec("emitc"), {"backend": "vpto"}),
            "vpto",
        )

    def test_none_override_falls_back_to_module_attribute(self):
        self.assertEqual(
            _resolve_ptoas_backend(self._spec("emitc"), {"backend": None}),
            "emitc",
        )

    def test_module_attribute_used_without_override(self):
        self.assertEqual(
            _resolve_ptoas_backend(self._spec("vpto"), {}),
            "vpto",
        )

    def test_missing_module_attribute_resolves_to_none(self):
        self.assertIsNone(_resolve_ptoas_backend(SimpleNamespace(), {}))


class IsElfObjectTest(unittest.TestCase):
    def test_elf_magic_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kernel.o"
            path.write_bytes(ELF_BYTES)
            self.assertTrue(_is_elf_object(path))

    def test_text_file_is_not_elf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kernel.cpp"
            path.write_bytes(CPP_BYTES)
            self.assertFalse(_is_elf_object(path))

    def test_missing_file_is_not_elf(self):
        self.assertFalse(_is_elf_object(Path("/nonexistent/kernel.o")))


class ProduceKernelObjectTest(unittest.TestCase):
    """EmitC ptoas output is C++ source and must be compiled before the fat link (#1605)."""

    @staticmethod
    def _module_spec(backend):
        return SimpleNamespace(
            function_name="fn",
            target_arch="a5",
            kernel_kind="vector",
            mode="auto",
            insert_sync=None,
            backend=backend,
            jit_source=None,
        )

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ptodsl-native-")
        self.addCleanup(temp.cleanup)
        self.artifacts = cache.artifact_paths(
            "foo", "fn", "key", cache_root=Path(temp.name)
        )
        self.artifacts.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.mlir_path.write_text("module {}", encoding="utf-8")
        self.artifacts.launch_cpp.write_text("// launch", encoding="utf-8")
        self.kernel_cpp = self.artifacts.kernel_object.with_suffix(".cpp")
        self.launch_object = self.artifacts.cache_dir / "launch.o"
        self.commands = []

    def test_emitc_routes_ptoas_to_cpp_and_compiles_the_source(self):
        self._compile(self._module_spec("emitc"))

        self.assertEqual(
            self.commands[0],
            [
                PTOAS_BIN,
                "--pto-arch=a5",
                "--enable-insert-sync",
                "--enable-tile-op-expand",
                str(self.artifacts.mlir_path),
                "-o",
                str(self.kernel_cpp),
            ],
        )
        self.assertNotIn("--pto-backend=emitc", self.commands[0])

        kernel_compile = self.commands[1]
        self.assertEqual(kernel_compile[0], BISHENG_BIN)
        self.assertIn("-c", kernel_compile)
        self.assertIn(str(self.kernel_cpp), kernel_compile)
        self.assertEqual(kernel_compile[kernel_compile.index("-o") + 1], str(self.artifacts.kernel_object))
        self.assertNotIn("-Dfn_EXPORTS", kernel_compile)
        self.assertIn("--cce-aicore-arch=dav-c310-vec", kernel_compile)

        launch_compile = self.commands[2]
        self.assertEqual(launch_compile[0], BISHENG_BIN)
        self.assertIn("-Dfn_EXPORTS", launch_compile)
        self.assertIn(str(self.artifacts.launch_cpp), launch_compile)
        self.assertEqual(launch_compile[launch_compile.index("-o") + 1], str(self.launch_object))

        link = self.commands[3]
        self.assertIn("--cce-fatobj-link", link)
        self.assertEqual(link[-2:], [str(self.launch_object), str(self.artifacts.kernel_object)])
        self.assertEqual(len(self.commands), 4)

        self.assertTrue(self.kernel_cpp.is_file())
        self.assertEqual(self.artifacts.kernel_object.read_bytes(), ELF_BYTES)

    def test_vpto_links_the_ptoas_object_directly(self):
        self._compile(self._module_spec("vpto"), ptoas_writes=ELF_BYTES)

        ptoas_cmd = self.commands[0]
        self.assertEqual(ptoas_cmd[-1], str(self.artifacts.kernel_object))
        self.assertNotIn("--pto-backend", ptoas_cmd)
        self.assertEqual(len(self.commands), 3)
        self.assertFalse(any(str(self.kernel_cpp) in cmd for cmd in self.commands))

        link = self.commands[2]
        self.assertIn("--cce-fatobj-link", link)
        self.assertEqual(link[-2:], [str(self.launch_object), str(self.artifacts.kernel_object)])
        self.assertEqual(self.artifacts.kernel_object.read_bytes(), ELF_BYTES)

    def test_backend_override_routes_ptoas_to_cpp(self):
        self._compile(
            self._module_spec("vpto"),
            ptoas_overrides={"backend": "emitc"},
        )

        self.assertIn("--pto-backend=emitc", self.commands[0])
        self.assertEqual(self.commands[0][-1], str(self.kernel_cpp))
        self.assertEqual(len(self.commands), 4)

    def test_ptoas_elf_at_cpp_output_is_adopted_as_kernel_object(self):
        """A mixed-backend container makes ptoas write its own fatobj ELF."""
        self._compile(self._module_spec("emitc"), ptoas_writes=ELF_BYTES)

        self.assertEqual(self.commands[0][-1], str(self.kernel_cpp))
        self.assertEqual(len(self.commands), 3)
        self.assertFalse(any(str(self.kernel_cpp) in cmd for cmd in self.commands[1:]))
        self.assertFalse(self.kernel_cpp.exists())
        self.assertEqual(self.artifacts.kernel_object.read_bytes(), ELF_BYTES)

    def test_source_output_for_non_emitc_backend_fails_before_any_bisheng_call(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._compile(self._module_spec("vpto"), ptoas_writes=CPP_BYTES)

        self.assertIn("C++ source", str(ctx.exception))
        self.assertIn("vpto", str(ctx.exception))
        self.assertEqual(
            len(self.commands),
            1,
            "the mismatch must surface right after ptoas, before launch compilation or linking",
        )

    def test_missing_ptoas_output_fails_with_the_output_path(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._compile(self._module_spec("vpto"), ptoas_writes=None)

        self.assertIn("produced no output", str(ctx.exception))
        self.assertIn(str(self.artifacts.kernel_object), str(ctx.exception))

    def _compile(self, module_spec, *, ptoas_overrides=None, ptoas_writes=CPP_BYTES):
        def fake_run(cmd, *, cwd=None):
            self.commands.append(list(cmd))
            output = Path(cmd[cmd.index("-o") + 1])
            if cmd[0] == PTOAS_BIN:
                if ptoas_writes is not None:
                    output.write_bytes(ptoas_writes)
                return
            if "--cce-fatobj-link" in cmd:
                output.write_bytes(b"fake shared library")
                return
            output.write_bytes(ELF_BYTES)

        with mock.patch.object(
            native_build, "resolve_ptoas_binary", return_value=Path(PTOAS_BIN)
        ), mock.patch.object(
            native_build, "resolve_bisheng", return_value=BISHENG_BIN
        ), mock.patch.object(
            native_build, "common_include_flags", return_value=[]
        ), mock.patch.object(
            native_build, "runtime_library_flags", return_value=[]
        ), mock.patch.object(
            native_build, "_run", side_effect=fake_run
        ):
            _compile_native_artifacts(
                self.artifacts,
                module_spec,
                effective_insert_sync=True,
                effective_pto_level=None,
                ptoas_overrides=ptoas_overrides or {},
            )


if __name__ == "__main__":
    unittest.main()
