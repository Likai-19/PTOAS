# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared constants/helpers for A5 TileLib template ports."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


FLOAT_DTYPES = ("f16", "bf16", "f32")
SIGNED_DTYPES = ("i8", "i16", "i32")
UNSIGNED_DTYPES = ("ui8", "ui16", "ui32")
INT_DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES
NUMERIC_DTYPES = ("i8", "i16", "i32", "ui8", "ui16", "ui32", "f16", "bf16", "f32")
ROW_REDUCTION_DTYPES = ("i16", "i32", "ui16", "ui32", "f16", "f32")


def same_dtype_signatures(arity, dtypes=NUMERIC_DTYPES):
    return [tuple([dtype] * arity) for dtype in dtypes]


def ub_row_major_constraints(*operand_names, require_same_valid_shape=True):
    constraints = [
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
    ]
    if require_same_valid_shape and operand_names:
        constraints.append(tilelib.require_same_valid_shape(*operand_names))
    return constraints


def dtype_name(dtype) -> str:
    return str(dtype)


def is_f32(dtype) -> bool:
    return dtype_name(dtype) == "f32"


def is_f16(dtype) -> bool:
    return dtype_name(dtype) == "f16"


def is_bf16(dtype) -> bool:
    return dtype_name(dtype) == "bf16"


def is_i16(dtype) -> bool:
    return dtype_name(dtype) == "i16"


def is_i32(dtype) -> bool:
    return dtype_name(dtype) == "i32"


def element_store_dist(dtype):
    bytewidth = 4
    name = dtype_name(dtype)
    if name in {"f16", "bf16", "i16", "ui16", "si16"}:
        bytewidth = 2
    elif name in {"i8", "ui8", "si8"}:
        bytewidth = 1
    if bytewidth == 4:
        return "1PT_B32"
    if bytewidth == 2:
        return "1PT_B16"
    return "1PT_B8"


def scalar_literal(dtype, value):
    name = dtype_name(dtype)
    if name == "f32":
        return pto.f32(float(value))
    if name == "f16":
        return pto.f16(float(value))
    if name == "bf16":
        return pto.bf16(float(value))
    if name == "ui32":
        return pto.ui32(value)
    if name == "ui16":
        return pto.ui16(value)
    if name == "ui8":
        return pto.ui8(value)
    if name in {"i32", "si32"}:
        return pto.i32(value)
    if name in {"i16", "si16"}:
        return pto.i16(value)
    return pto.i8(value)
