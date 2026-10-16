#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 i8 row/column expand multiplication and compact ColMajor row scalars."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="rowexpandmul_i8_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def rowexpandmul_i8(a: pto.ptr(pto.i8, "gm"), b: pto.ptr(pto.i8, "gm"), out: pto.ptr(pto.i8, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[8, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[32, 1], dtype=pto.i8, valid_shape=[8, 1], blayout="ColMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.rowexpandmul(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


@pto.jit(name="colexpandmul_i8_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def colexpandmul_i8(a: pto.ptr(pto.i8, "gm"), b: pto.ptr(pto.i8, "gm"), out: pto.ptr(pto.i8, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[1, 8], strides=[8, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[1, 8], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.colexpandmul(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


def _lhs():
    return np.arange(-32, 32, dtype=np.int8).reshape(8, 8)


def _row_inputs():
    return [_lhs(), np.array([-128, -17, -3, -1, 0, 3, 17, 127], dtype=np.int8).reshape(8, 1)]


def _col_inputs():
    return [_lhs(), np.array([-128, -17, -3, -1, 0, 3, 17, 127], dtype=np.int8).reshape(1, 8)]


def _expected(lhs, rhs):
    return (lhs.astype(np.int16) * rhs.astype(np.int16)).astype(np.int8)


CASES = [
    golden_output_case("rowexpandmul_i8_8x8", rowexpandmul_i8,
                       inputs=_row_inputs, expected=_expected, rtol=0, atol=0),
    golden_output_case("colexpandmul_i8_8x8", colexpandmul_i8,
                       inputs=_col_inputs, expected=_expected, rtol=0, atol=0),
]


auto_main(globals())
