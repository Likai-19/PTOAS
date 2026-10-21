#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest

from ptoas.mlir.ir import Module

from ptodsl import pto
from ptodsl._context import make_context
from ptodsl._tracing.runtime import _validate_emitc_operations


def cv_entry(backend, direction):
    @pto.jit(target="a5", backend=backend, entry=False, kernel_kind="cube", mode="auto")
    def cube(slot: pto.ptr(pto.f32, "gm")):
        view = pto.make_tensor_view(slot, shape=[16, 16], strides=[16, 1])
        pipe = getattr(pto.pipe, direction)(slot_size=1024, gm_slot_tensor=view, id=0)
        pipe.init_cube()

    @pto.jit(target="a5", backend=backend, entry=False, kernel_kind="vector", mode="auto")
    def vector(slot: pto.ptr(pto.f32, "gm")):
        view = pto.make_tensor_view(slot, shape=[16, 16], strides=[16, 1])
        pipe = getattr(pto.pipe, direction)(slot_size=1024, gm_slot_tensor=view, id=0)
        pipe.init_simd()

    @pto.jit(target="a5", backend=backend, mode="auto")
    def entry(slot: pto.ptr(pto.f32, "gm")):
        cube(slot)
        vector(slot)

    return entry


def vmi_entry(backend, child_backend=None):
    @pto.jit(target="a5", backend=child_backend or backend, entry=False,
             kernel_kind="vector", mode="auto")
    def vector(src: pto.ptr(pto.f32, "gm")):
        tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
        pto.vmi.vload(tile.as_ptr(), pto.const(0), size=64)

    @pto.jit(target="a5", backend=backend, mode="auto")
    def entry(src: pto.ptr(pto.f32, "gm")):
        vector(src)

    return entry


class CVBackendContractTest(unittest.TestCase):
    def test_imported_pipe_kernels_verify(self):
        for backend in ("vpto", "emitc"):
            for direction in ("c2v", "v2c"):
                with self.subTest(backend=backend, direction=direction):
                    text = cv_entry(backend, direction).compile().mlir_text()
                    self.assertIn("pto.aic_initialize_pipe", text)
                    self.assertIn("pto.aiv_initialize_pipe", text)
                    self.assertIn("#pto.kernel_kind<cube>", text)
                    self.assertIn("#pto.kernel_kind<vector>", text)

    def test_emitc_vmi_rejected_during_compile(self):
        with self.assertRaisesRegex(RuntimeError, "backend='emitc'.*pto.vmi.vload"):
            vmi_entry("emitc").compile()

    def test_vpto_child_under_emitc_entry_remains_valid(self):
        text = vmi_entry("emitc", child_backend="vpto").compile().mlir_text()
        self.assertIn("pto.vmi.vload", text)

    def test_emitc_mad_rejected_during_compile(self):
        @pto.jit(target="a5", backend="emitc", mode="auto", kernel_kind="cube")
        def kernel():
            lhs = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, memory_space="left")
            rhs = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, memory_space="right")
            acc = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, memory_space="acc")
            pto.mad(lhs.as_ptr(), rhs.as_ptr(), acc.as_ptr(), 16, 16, 16)

        with self.assertRaisesRegex(RuntimeError, "backend='emitc'.*pto.mad"):
            kernel.compile()

    def test_emitc_vlds_rejected_during_compile(self):
        @pto.jit(target="a5", backend="emitc", mode="auto", kernel_kind="vector")
        def kernel():
            tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
            pto.vlds(tile.as_ptr(), pto.const(0))

        with self.assertRaisesRegex(RuntimeError, "backend='emitc'.*pto.vlds"):
            kernel.compile()

    def test_emitc_vadd_rejected_during_verification(self):
        # Use valid IR arguments so an earlier unsupported load/mask cannot
        # hide a missing vadd diagnostic. VRegs are not kernel-module ABI types.
        with make_context():
            module = Module.parse('''
                module attributes {pto.backend = "emitc"} {
                  func.func @add(%lhs: !pto.vreg<64xf32>,
                                 %rhs: !pto.vreg<64xf32>, %mask: !pto.mask<b32>) {
                    %result = pto.vadd %lhs, %rhs, %mask
                      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
                        -> !pto.vreg<64xf32>
                    return
                  }
                }
            ''')
            self.assertTrue(module.operation.verify())
            with self.assertRaisesRegex(RuntimeError, "backend='emitc'.*pto.vadd"):
                _validate_emitc_operations(module.operation)

    def test_emitc_mte_rejected_during_compile(self):
        @pto.jit(target="a5", backend="emitc", mode="explicit", kernel_kind="vector")
        def kernel(src: pto.ptr(pto.f32, "gm")):
            tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
            pto.mte_gm_ub(src, tile.as_ptr(), 0, 64, nburst=(1, 64, 64))

        with self.assertRaisesRegex(RuntimeError, "backend='emitc'.*pto.mte_gm_ub"):
            kernel.compile()

    def test_emitc_shared_tile_scalar_pointer_and_view_operations_remain_valid(self):
        @pto.jit(target="a5", backend="emitc", mode="auto", kernel_kind="vector")
        def kernel(src: pto.ptr(pto.f32, "gm"), row: pto.i32):
            view = pto.make_tensor_view(pto.addptr(src, row * 64),
                                        shape=[1, 64], strides=[64, 1])
            part = pto.partition_view(view, offsets=[0, 0], sizes=[1, 64])
            tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
            pto.tile.load(part, tile)
            pto.tile.add(tile, tile, tile)
            pto.tile.store(tile, part)
            pto.pipe_barrier(pto.Pipe.ALL)

        text = kernel.compile().mlir_text()
        for name in ("pto.muli", "pto.addptr", "pto.make_tensor_view",
                     "pto.tload", "pto.tadd", "pto.tstore", "pto.barrier"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
