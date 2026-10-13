# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Cache policy checks for the ptodsl JIT artifact build (no device needed)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ptodsl._kernel_compilation import CompiledKernelHandle
from ptodsl._runtime import cache, native_build


class CachePolicyTest(unittest.TestCase):
    @staticmethod
    def _module_spec():
        return SimpleNamespace(
            function_name="fn",
            target_arch="a5",
            kernel_kind=None,
            mode="explicit",
            insert_sync=None,
            backend="vpto",
            jit_source=None,
        )

    @staticmethod
    def _optin_dir(case):
        optin_td = tempfile.TemporaryDirectory(prefix="ptodsl-optin-")
        case.addCleanup(optin_td.cleanup)
        return optin_td.name

    def setUp(self):
        self._saved = dict(os.environ)
        os.environ.pop("PTODSL_CACHE_DIR", None)
        os.environ.pop("PTODSL_NO_CACHE", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_default_cache_is_disabled(self):
        self.assertFalse(cache.cache_enabled())

    def test_cache_dir_opts_in(self):
        optin = self._optin_dir(self)
        os.environ["PTODSL_CACHE_DIR"] = optin
        self.assertTrue(cache.cache_enabled())
        artifacts = cache.artifact_paths("foo", "fn", "key")
        self.assertTrue(str(artifacts.cache_dir).startswith(optin))

    def test_no_cache_wins_over_cache_dir(self):
        os.environ["PTODSL_CACHE_DIR"] = self._optin_dir(self)
        os.environ["PTODSL_NO_CACHE"] = "1"
        self.assertFalse(cache.cache_enabled())

    def test_build_temp_dir_is_removed_after_load(self):
        captured = {}

        def fake_materialize(artifacts, **_kwargs):
            captured["dir"] = Path(artifacts.cache_dir)
            artifacts.cache_dir.mkdir(parents=True, exist_ok=True)
            artifacts.shared_library.write_text("fake library\n", encoding="utf-8")
            return artifacts.shared_library

        sentinel = object()
        with mock.patch.object(native_build, "_materialize_native_library", side_effect=fake_materialize), \
                mock.patch.object(native_build.ctypes, "CDLL", return_value=sentinel):
            library, launch_symbol = native_build.build_native_library(
                py_name="foo",
                module_spec=self._module_spec(),
                kernel_signature=SimpleNamespace(positional_parameters=[]),
                mlir_text="",
                specialization_key="key",
            )
        self.assertIs(library, sentinel)
        self.assertTrue(launch_symbol.startswith("ptodsl_launch_"))
        self.assertFalse(
            captured["dir"].exists(),
            "the standard TemporaryDirectory must remove the build tree on exit",
        )


class NativeLibraryReuseTest(unittest.TestCase):
    """A specialization builds the native library once per process (#69)."""

    def test_each_launch_handle_reuses_the_compiled_build(self):
        handle = self._handle()
        calls = []

        def fake_launch(*_args, **_kwargs):
            return None

        def fake_build(**kwargs):
            calls.append(kwargs)
            return (SimpleNamespace(ptodsl_launch_fn=fake_launch), "ptodsl_launch_fn")

        with mock.patch.object(native_build, "build_native_library", side_effect=fake_build):
            for _ in range(3):
                handle[1, 0]()
        self.assertEqual(len(calls), 1)

    def test_native_library_builds_once(self):
        handle = self._handle()
        calls = []

        def fake_build(**kwargs):
            calls.append(kwargs)
            return (object(), "launch_symbol")

        with mock.patch.object(native_build, "build_native_library", side_effect=fake_build):
            first = handle.native_library()
            second = handle.native_library()
        self.assertEqual(len(calls), 1)
        self.assertIs(first, second)

    def _handle(self):
        def module_factory():
            return object(), {"mlir_text": ""}

        return CompiledKernelHandle(
            "m",
            specialization_key="key",
            constexpr_bindings={},
            module_factory=module_factory,
            module_spec=SimpleNamespace(function_name="kernel", entry=True),
            kernel_signature=SimpleNamespace(positional_parameters=[]),
        )


if __name__ == "__main__":
    unittest.main()
