#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""tcvt low-precision simulator ST (issue #1612).

Cases:

* i32/ui32 -> ui8 over a 4x128 tile whose valid region is 4x65: each row is one
  full 64-element chunk plus a one-element tail, so the b8 store mask has to be
  rebuilt per chunk.  The old code pbitcast the b32 element mask instead, which
  enabled only every fourth byte and left the rest of the row untouched.
* bf16 -> f4e2m1x2 over 8x64 -> 8x32 packed pairs: a whole-row b8 mask is 256
  lanes wide while a chunk produces 64 pairs, so the same accounting bug left
  every chunk after the first unwritten.
* f8e4m3 -> f32: the widened pair added for the issue, exercised end to end.
"""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import auto_main, golden_output_case
from ptodsl import pto

# 4:1 narrowing: 65 valid columns = one 64-element chunk plus a 1-element tail.
NARROW_SHAPE = (4, 128)
NARROW_VALID = (4, 65)
NARROW_TOTAL = NARROW_SHAPE[0] * NARROW_SHAPE[1]
NARROW_PROBE = np.array([0, 1, 2, 3, 4, 7, 8, 63, 64, 65, 127, 128, 255, 256, 1000, -1], dtype=np.int32)

# bf16 -> f4e2m1x2: two source values pack into one destination byte.
FP4_ROWS, FP4_SRC_COLS = 8, 64
FP4_DST_COLS = FP4_SRC_COLS // 2
# Exactly representable e2m1 values, in nibble order (0x0..0xF).
FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
# Keyed by the bf16 bit pattern so that +0.0 and -0.0 stay distinct.
FP4_CODES = {int(np.float32(v).view(np.uint32) >> 16): code for code, v in enumerate(FP4_VALUES)}

# Exactly representable e4m3 values and their bit patterns.
E4M3_VALUES = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 448.0, -1.0, -4.0], dtype=np.float32)
E4M3_BITS = np.array([0x00, 0x28, 0x30, 0x38, 0x3C, 0x40, 0x48, 0x7E, 0xB8, 0xC8], dtype=np.uint8)
# Exactly representable hif8 values and their byte codes (reference table:
# test/vpto/cases/micro-op/conversion/vcvt-low-precision-special/golden.py).
HIF8_VALUES = np.array(
    [
        0.0,
        2.384185791015625e-07,
        1.52587890625e-05,
        1.0,
        2.0,
        0.5,
        4.0,
        16.0,
        0.0625,
        256.0,
        0.00390625,
        -2.384185791015625e-07,
        -1.52587890625e-05,
        -1.0,
        -2.0,
        -0.5,
        -4.0,
        -16.0,
        -0.0625,
        -256.0,
        -0.00390625,
    ],
    dtype=np.float32,
)
HIF8_BITS = np.array(
    [
        0x00, 0x01, 0x07, 0x08, 0x10, 0x18, 0x20, 0x40, 0x50, 0x60, 0x70,
        0x81, 0x87, 0x88, 0x90, 0x98, 0xA0, 0xC0, 0xD0, 0xE0, 0xF0,
    ],
    dtype=np.uint8,
)

SI8_PROBE = np.array([0, 1, -1, 127, -128, 2, -2, 64, -64, 3, -3, 100, -100, 42, -42, 7], dtype=np.int8)


def _tile(values, dtype, rows, cols):
    total = rows * cols
    tiled = np.tile(values, (total + len(values) - 1) // len(values))[:total]
    return tiled.reshape(rows, cols).astype(dtype)


def _convert_kernel(name, src_dtype, dst_dtype, shape, valid_shape, dst_shape=None, dst_valid_shape=None):
    rows, cols = shape
    v_rows, v_cols = valid_shape
    d_rows, d_cols = dst_shape or shape
    dv_rows, dv_cols = dst_valid_shape or (v_rows, v_cols)

    @pto.jit(name=name, target="a5")
    def _body(src_ptr: pto.ptr(src_dtype, "gm"), out_ptr: pto.ptr(dst_dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[d_rows, d_cols], strides=[d_cols, 1])
        src = pto.alloc_tile(shape=[rows, cols], dtype=src_dtype, valid_shape=[v_rows, v_cols])
        dst = pto.alloc_tile(shape=[d_rows, d_cols], dtype=dst_dtype, valid_shape=[dv_rows, dv_cols])
        pto.tile.load(src_view, src)
        pto.tile.cvt(src, dst, rmode=pto.RoundMode.NONE)
        pto.tile.store(dst, out_view)

    return _body


_i32_ui8 = _convert_kernel("cvt_i32_to_ui8_s4x65", pto.i32, pto.ui8, NARROW_SHAPE, NARROW_VALID)
_ui32_ui8 = _convert_kernel("cvt_ui32_to_ui8_s4x65", pto.ui32, pto.ui8, NARROW_SHAPE, NARROW_VALID)
_bf16_fp4 = _convert_kernel(
    "cvt_bf16_to_f4e2m1x2_s8x64",
    pto.bf16,
    pto.f4e2m1x2,
    (FP4_ROWS, FP4_SRC_COLS),
    (FP4_ROWS, FP4_SRC_COLS),
    (FP4_ROWS, FP4_DST_COLS),
    (FP4_ROWS, FP4_DST_COLS),
)
_f8e4m3_f32 = _convert_kernel("cvt_f8e4m3_to_f32_s8x32", pto.f8e4m3, pto.f32, (8, 32), (8, 32))
_f8e4m3_f32_2d = _convert_kernel("cvt_f8e4m3_to_f32_s8x48", pto.f8e4m3, pto.f32, (8, 64), (8, 48))
_f32_f8e4m3_2d = _convert_kernel("cvt_f32_to_f8e4m3_s8x48", pto.f32, pto.f8e4m3, (8, 64), (8, 48))
_bf16_fp4_2d = _convert_kernel(
    "cvt_bf16_to_f4e2m1x2_s8x48",
    pto.bf16,
    pto.f4e2m1x2,
    (FP4_ROWS, FP4_SRC_COLS),
    (FP4_ROWS, 48),
    (FP4_ROWS, FP4_DST_COLS),
    (FP4_ROWS, 24),
)
# si8 -> i32 and f32 -> hif8 use the renderers shared/reworked in this issue.
_si8_i32 = _convert_kernel("cvt_si8_to_i32_s8x32", pto.si8, pto.i32, (8, 32), (8, 32))
_si8_i32_2d = _convert_kernel("cvt_si8_to_i32_s8x48", pto.si8, pto.i32, (8, 64), (8, 48))
_f32_hif8 = _convert_kernel("cvt_f32_to_hif8_s8x32", pto.f32, pto.hif8, (8, 32), (8, 32))
_f32_hif8_2d = _convert_kernel("cvt_f32_to_hif8_s8x48", pto.f32, pto.hif8, (8, 64), (8, 48))


def _narrow_inputs(dtype) -> list:
    return [_tile(np.abs(NARROW_PROBE), dtype, *NARROW_SHAPE)]


def _narrow_expected(src):
    """Saturating narrowing, with the bytes outside the valid region untouched."""

    golden = np.zeros(NARROW_SHAPE, dtype=np.uint8)
    v_rows, v_cols = NARROW_VALID
    golden[:v_rows, :v_cols] = np.clip(src[:v_rows, :v_cols], 0, 255).astype(np.uint8)
    return golden


def _partial(values, valid_rows, valid_cols):
    """keep the valid region, zero the rest (the store leaves it untouched)"""

    masked = np.zeros_like(values)
    masked[:valid_rows, :valid_cols] = values[:valid_rows, :valid_cols]
    return masked


def _bf16_bits(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _fp4_inputs() -> list:
    values = _tile(FP4_VALUES, np.float32, FP4_ROWS, FP4_SRC_COLS)
    return [_bf16_bits(values)]


def _fp4_expected(src_bits):
    codes = np.vectorize(lambda bits: FP4_CODES[int(bits)], otypes=[np.uint8])(src_bits)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def _f8bits_inputs_full() -> list:
    return [_tile(E4M3_BITS, np.uint8, 8, 64)]


def _f8bits_inputs() -> list:
    return [_tile(E4M3_BITS, np.uint8, 8, 32)]


def _f8values_inputs_full() -> list:
    return [_tile(E4M3_VALUES, np.float32, 8, 64)]


def _si8_inputs(rows, cols) -> list:
    return [_tile(SI8_PROBE, np.int8, rows, cols)]


def _si8_expected(src):
    return src.astype(np.int32)


def _si8_expected_partial(src):
    return _partial(src.astype(np.int32), 8, 48)


def _hif8_inputs(rows, cols) -> list:
    return [_tile(HIF8_VALUES, np.float32, rows, cols)]


def _hif8_expected(rows, cols):
    return _tile(HIF8_BITS, np.uint8, rows, cols)


def _hif8_expected_partial():
    return _partial(_tile(HIF8_BITS, np.uint8, 8, 64), 8, 48)


CASES = [
    golden_output_case(
        "cvt_i32_to_ui8_s4x65",
        _i32_ui8,
        inputs=lambda: _narrow_inputs(np.int32),
        expected=_narrow_expected,
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_ui32_to_ui8_s4x65",
        _ui32_ui8,
        inputs=lambda: _narrow_inputs(np.uint32),
        expected=_narrow_expected,
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_bf16_to_f4e2m1x2_s8x64",
        _bf16_fp4,
        inputs=_fp4_inputs,
        expected=_fp4_expected,
        output_shape=(FP4_ROWS, FP4_DST_COLS),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f8e4m3_to_f32_s8x32",
        _f8e4m3_f32,
        inputs=_f8bits_inputs,
        expected=lambda src: _tile(E4M3_VALUES, np.float32, 8, 32),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f8e4m3_to_f32_s8x48",
        _f8e4m3_f32_2d,
        inputs=_f8bits_inputs_full,
        expected=lambda src: _partial(_tile(E4M3_VALUES, np.float32, 8, 64), 8, 48),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_f8e4m3_s8x48",
        _f32_f8e4m3_2d,
        inputs=_f8values_inputs_full,
        expected=lambda src: _partial(_tile(E4M3_BITS, np.uint8, 8, 64), 8, 48),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_bf16_to_f4e2m1x2_s8x48",
        _bf16_fp4_2d,
        inputs=_fp4_inputs,
        expected=lambda src: _partial(_fp4_expected(src), FP4_ROWS, 24),
        output_shape=(FP4_ROWS, FP4_DST_COLS),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_si8_to_i32_s8x32",
        _si8_i32,
        inputs=lambda: _si8_inputs(8, 32),
        expected=_si8_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_si8_to_i32_s8x48",
        _si8_i32_2d,
        inputs=lambda: _si8_inputs(8, 64),
        expected=_si8_expected_partial,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_hif8_s8x32",
        _f32_hif8,
        inputs=lambda: _hif8_inputs(8, 32),
        expected=lambda src: _hif8_expected(8, 32),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_hif8_s8x48",
        _f32_hif8_2d,
        inputs=lambda: _hif8_inputs(8, 64),
        expected=lambda src: _hif8_expected_partial(),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
]

auto_main(globals())
