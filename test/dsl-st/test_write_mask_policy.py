#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression for the write-mask policy of the migrated DSL ST modules.

The default comparison keeps the original golden/compare semantics: a buffer the
golden creates with zeros/full/ones has every element defined, so the whole
buffer is compared even when the golden later writes only part of it again.
Only a buffer created with np.empty has genuinely undefined elements; that one
keeps the partial mask recorded from the golden's in-place writes, and the audit
in README.md lists the modules that rely on it.

These tests load the real _materialize/_apply_write_masks/_run_compare helpers
from a generated module and drive them with in-memory fixtures, without a
simulator or an NPU. The final test is the review counterexample: a zeros
golden written in part must reject a wrong output instead of masking it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parent
HOST_MODULE = "legacy_micro_op_compare_select_vcmps_f32"


def _load_host():
    """Import one generated module to reuse its masking helpers."""
    if str(CASE_ROOT) not in sys.path:
        sys.path.insert(0, str(CASE_ROOT))
    path = CASE_ROOT / (HOST_MODULE + ".py")
    spec = importlib.util.spec_from_file_location("_write_mask_host", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the write-mask host module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_scope(module, golden_scope, compare_scope, pairs):
    """Point the generated module at an in-memory fixture golden/compare."""
    module._golden_scope = golden_scope
    module._compare_scope = compare_scope
    module.SEED_EXPR = None
    module.HELPERS = {}
    module.COMPARE_PAIRS = pairs
    return module


def _golden_scope_policy():
    _files = {}

    def _generate(seed):
        defined = np.zeros(8, dtype=np.int32)
        defined[1] = 11
        undefined = np.empty(8, dtype=np.int32)
        undefined[5] = 13
        fully = np.zeros(8, dtype=np.int32)
        fully[:] = np.arange(8, dtype=np.int32)
        _files["defined.bin"] = defined
        _files["golden_defined.bin"] = defined
        _files["undefined.bin"] = undefined
        _files["golden_undefined.bin"] = undefined
        _files["fully.bin"] = fully
        _files["golden_fully.bin"] = fully

    return locals()


def _compare_scope_policy():
    def main():
        for name in ("defined", "undefined", "fully"):
            golden = np.fromfile("golden_%s.bin" % name, dtype=np.int32)
            out = np.fromfile("%s.bin" % name, dtype=np.int32)
            if not np.array_equal(golden, out):
                print("[ERROR] %s mismatch" % name)
                sys.exit(2)

    return locals()


def _golden_scope_review_counterexample():
    _files = {}

    def _generate(seed):
        golden = np.zeros(4, dtype=np.float32)
        golden[0] = 1
        _files["out.bin"] = golden
        _files["golden_out.bin"] = golden

    return locals()


def _compare_scope_exact():
    def main():
        golden = np.fromfile("golden_out.bin", dtype=np.float32)
        out = np.fromfile("out.bin", dtype=np.float32)
        if not np.array_equal(golden, out):
            print("[ERROR] mismatch")
            sys.exit(2)

    return locals()


def _policy_module():
    module = _load_host()
    return _install_scope(
        module,
        _golden_scope_policy,
        _compare_scope_policy,
        [
            ("golden_defined.bin", "defined.bin"),
            ("golden_undefined.bin", "undefined.bin"),
            ("golden_fully.bin", "fully.bin"),
        ],
    )


class WriteMaskPolicyTest(unittest.TestCase):
    """Defined buffers compare whole; only np.empty keeps a partial mask."""

    @classmethod
    def setUpClass(cls):
        cls.module = _policy_module()
        cls.produced = cls.module._materialize()
        cls.masks = cls.produced["@write_mask"]

    def _files(self):
        return {
            name: np.asarray(value).copy()
            for name, value in self.produced.items()
            if not name.startswith("@")
        }

    def test_partially_written_zeros_buffer_is_compared_whole(self):
        mask = np.asarray(self.masks["golden_defined.bin"])
        self.assertTrue(
            bool(mask.all()),
            "a zeros buffer defines every element, so it must compare whole",
        )

    def test_partially_written_empty_buffer_keeps_a_partial_mask(self):
        mask = np.asarray(self.masks["golden_undefined.bin"])
        self.assertFalse(bool(mask.all()))
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(bool(mask[5]))

    def test_fully_written_buffer_mask_covers_every_element(self):
        mask = np.asarray(self.masks["golden_fully.bin"])
        self.assertTrue(bool(mask.all()))

    def test_wrong_defined_dont_care_fails_the_compare(self):
        files = self._files()
        files["defined.bin"][3] = 999  # never written by the golden, but defined
        self.module._apply_write_masks(files, self.produced)
        with self.assertRaises(AssertionError):
            self.module._run_compare(files)

    def test_wrong_empty_dont_care_is_ignored(self):
        files = self._files()
        files["undefined.bin"][0] = 999  # never written and genuinely undefined
        self.module._apply_write_masks(files, self.produced)
        self.module._run_compare(files)

    def test_wrong_written_element_fails_the_compare(self):
        files = self._files()
        files["defined.bin"][1] = 999  # the element the golden wrote
        self.module._apply_write_masks(files, self.produced)
        with self.assertRaises(AssertionError):
            self.module._run_compare(files)

    def test_wrong_fully_written_element_fails_the_compare(self):
        files = self._files()
        files["fully.bin"][4] = 999
        self.module._apply_write_masks(files, self.produced)
        with self.assertRaises(AssertionError):
            self.module._run_compare(files)

    def test_review_zeros_counterexample_fails_the_compare(self):
        # golden = zeros(4, float32); golden[0] = 1; the wrong output
        # [1, 999, 999, 999] used to be masked to equal the golden.
        module = _load_host()
        _install_scope(
            module,
            _golden_scope_review_counterexample,
            _compare_scope_exact,
            [("golden_out.bin", "out.bin")],
        )
        produced = module._materialize()
        mask = np.asarray(produced["@write_mask"]["golden_out.bin"])
        self.assertTrue(bool(mask.all()), "the zeros golden is fully defined")
        files = {
            "golden_out.bin": np.asarray(produced["golden_out.bin"]).copy(),
            "out.bin": np.array([1, 999, 999, 999], dtype=np.float32),
        }
        module._apply_write_masks(files, produced)
        self.assertFalse(
            np.array_equal(files["golden_out.bin"], files["out.bin"]),
            "the wrong output must not be rewritten into the golden",
        )
        with self.assertRaises(AssertionError):
            module._run_compare(files)


if __name__ == "__main__":
    unittest.main()
