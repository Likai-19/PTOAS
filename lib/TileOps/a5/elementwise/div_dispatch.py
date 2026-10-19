# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Element type table and division dispatch shared by pto.tdiv and pto.tdivs.

Both templates accept the same signatures and pick between the same three
implementations, so the table and the dispatch live here instead of being
duplicated in each template module.
"""

from SoftOps import div_i32_soft

from ptodsl import pto

from .div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


DIV_DTYPES = [(dtype,) * 3 for dtype in ("f16", "f32", "i32", "i16", "ui16", "ui32")]

_HIGH_PRECISION_DTYPES = {"f16", "f32"}


def divide(lhs, rhs, dtype, mask, precision_type):
    """Select the division implementation for one element type.

    ``precisionType`` only selects an IEEE-754 path for f16/f32. Integer
    division is already exact, so high_precision is accepted and ignored on
    integer tiles rather than rejected; the dtype guard is also what keeps
    integers out of the f16 fallthrough.
    """
    name = str(dtype)
    if name == "i32":
        # i32 stays inlined in the template rather than deferring to the late
        # SoftLib expansion of pto.vdiv, which would also accept it. The
        # expanded form is pinned by test/lit/vpto/issue_1181_tdiv_i32_tilelib_
        # expand.pto, which requires the signed helper's ops to appear directly
        # in the tile-op-expand output with no integer pto.vdiv left behind.
        return div_i32_soft(lhs, rhs, mask)
    if precision_type == "high_precision" and name in _HIGH_PRECISION_DTYPES:
        if name == "f32":
            return _div_ieee754_f32_impl(lhs, rhs, mask)
        return _div_ieee754_f16_impl(lhs, rhs, mask)
    return pto.vdiv(lhs, rhs, mask)
