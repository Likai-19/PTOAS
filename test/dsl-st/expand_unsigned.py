#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 unsigned scalar, row, and column expansion runtime checks."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="expands_ui8_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def expands_ui8(out: pto.ptr(pto.ui8, "gm")):
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.ui8, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.expands(pto.ui8(213), dst)
    pto.tile.store(dst, out_view)


@pto.jit(name="rowexpandadd_ui16_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def rowexpandadd_ui16(a: pto.ptr(pto.ui16, "gm"), b: pto.ptr(pto.ui16, "gm"),
                      out: pto.ptr(pto.ui16, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[8, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 16], dtype=pto.ui16, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[16, 1], dtype=pto.ui16, valid_shape=[8, 1], blayout="ColMajor")
    dst = pto.alloc_tile(shape=[8, 16], dtype=pto.ui16, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.rowexpandadd(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


@pto.jit(name="colexpanddiv_ui32_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def colexpanddiv_ui32(a: pto.ptr(pto.ui32, "gm"), b: pto.ptr(pto.ui32, "gm"),
                      out: pto.ptr(pto.ui32, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[1, 8], strides=[8, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 8], dtype=pto.ui32, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[8, 8], dtype=pto.ui32, valid_shape=[1, 8], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 8], dtype=pto.ui32, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.colexpanddiv(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


def _row_inputs():
    lhs = (np.arange(64, dtype=np.uint32) * 1013 + 65000).astype(np.uint16).reshape(8, 8)
    rhs = np.array([1, 7, 255, 65535, 2, 17, 4096, 32768], dtype=np.uint16).reshape(8, 1)
    return [lhs, rhs]


def _col_inputs():
    lhs = (np.arange(64, dtype=np.uint64) * 676717 + 0xF0000000).astype(np.uint32).reshape(8, 8)
    rhs = np.array([1, 2, 3, 7, 65535, 0x80000000, 0xffffffff, 17], dtype=np.uint32).reshape(1, 8)
    return [lhs, rhs]


CASES = [
    golden_output_case("expands_ui8_8x8", expands_ui8, inputs=lambda: [],
                       expected=lambda: np.full((8, 8), 213, dtype=np.uint8), rtol=0, atol=0),
    golden_output_case("rowexpandadd_ui16_8x8", rowexpandadd_ui16, inputs=_row_inputs,
                       expected=lambda a, b: (a.astype(np.uint32) + b.astype(np.uint32)).astype(np.uint16),
                       rtol=0, atol=0),
    golden_output_case("colexpanddiv_ui32_8x8", colexpanddiv_ui32, inputs=_col_inputs,
                       expected=lambda a, b: a // b, rtol=0, atol=0),
]


auto_main(globals())
