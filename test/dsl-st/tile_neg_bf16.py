#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for A5 BF16 tile negation through PTODSL and VPTO."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

ROWS = 8
COLS = 32


@pto.jit(
    name="tile_neg_bf16_kernel",
    target="a5",
    kernel_kind="vector",
    backend="vpto",
    mode="explicit",
    insert_sync=True,
)
def tile_neg_bf16_kernel(
    src_ptr: pto.ptr(pto.bf16, "gm"),
    dst_ptr: pto.ptr(pto.bf16, "gm"),
):
    src_view = pto.make_tensor_view(
        src_ptr,
        shape=[ROWS, COLS],
        strides=[COLS, 1],
    )
    dst_view = pto.make_tensor_view(
        dst_ptr,
        shape=[ROWS, COLS],
        strides=[COLS, 1],
    )
    src = pto.alloc_tile(
        shape=[ROWS, COLS],
        dtype=pto.bf16,
        addr=0,
        valid_shape=[ROWS, COLS],
        blayout="RowMajor",
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS],
        dtype=pto.bf16,
        addr=1024,
        valid_shape=[ROWS, COLS],
        blayout="RowMajor",
    )

    pto.tile.load(src_view, src)
    pto.tile.neg(src, dst)
    pto.tile.store(dst, dst_view)


def make_inputs():
    bf16_values = np.array(
        [
            0x0000,  # +0
            0x8000,  # -0
            0x3F80,  # +1
            0xBF80,  # -1
            0x4000,  # +2
            0xC000,  # -2
            0x4050,  # +3.25
            0xC050,  # -3.25
            0x3F00,  # +0.5
            0xBF00,  # -0.5
            0x4120,  # +10
            0xC120,  # -10
        ],
        dtype=np.uint16,
    )
    return [np.resize(bf16_values, (ROWS, COLS))]


def make_expected(src):
    return np.bitwise_xor(src, np.uint16(0x8000))


CASES = [
    golden_output_case(
        "tile_neg_bf16_8x32",
        tile_neg_bf16_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
