#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 bf16 row expand division runtime check."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="rowexpanddiv_bf16_8x8", kernel_kind="vector", target="a5", backend="vpto", mode="auto")
def rowexpanddiv_bf16(a: pto.ptr(pto.bf16, "gm"), b: pto.ptr(pto.bf16, "gm"),
                      out: pto.ptr(pto.bf16, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[8, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 16], dtype=pto.bf16, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[16, 1], dtype=pto.bf16, valid_shape=[8, 1], blayout="ColMajor")
    dst = pto.alloc_tile(shape=[8, 16], dtype=pto.bf16, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.rowexpanddiv(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


def _inputs():
    lhs = (np.arange(1, 65, dtype=np.float32).reshape(8, 8).view(np.uint32) >> 16).astype(np.uint16)
    rhs = (np.array([3, 2, 4, 8, 1, 2, 4, 8], dtype=np.float32).reshape(8, 1).view(np.uint32) >> 16)
    rhs = rhs.astype(np.uint16)
    return [lhs, rhs]


def _expected(lhs, rhs):
    lhs_f32 = (lhs.astype(np.uint32) << 16).view(np.float32)
    rhs_f32 = (rhs.astype(np.uint32) << 16).view(np.float32)
    bits = (lhs_f32 / rhs_f32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


CASES = [
    golden_output_case("rowexpanddiv_bf16_8x8", rowexpanddiv_bf16, inputs=_inputs,
                       expected=_expected, rtol=0, atol=0),
]


auto_main(globals())
