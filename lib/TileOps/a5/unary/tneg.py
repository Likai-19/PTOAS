# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tneg."""

from ptodsl import pto

from .._elementwise import register_unary


_DTYPES = [
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
    ("ui16", "ui16"),
    ("ui32", "ui32"),
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
]


def _neg(value, mask):
    # Unsigned negation is multiplication modulo the element width, as in TMULS.
    if str(value.type).endswith("xui16>"):
        return pto.vmuls(value, pto.ui16(0xFFFF), mask)
    if str(value.type).endswith("xui32>"):
        return pto.vmuls(value, pto.ui32(0xFFFFFFFF), mask)
    return pto.vneg(value, mask)


template_tneg = register_unary(
    op="pto.tneg",
    name="template_tneg",
    vector_op=_neg,
    dtypes=_DTYPES,
)


template_tneg_1d = register_unary(
    op="pto.tneg",
    name="template_tneg_1d",
    vector_op=_neg,
    dtypes=_DTYPES,
    traversal="1d",
)
