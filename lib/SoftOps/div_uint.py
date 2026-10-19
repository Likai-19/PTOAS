# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact unsigned vector division for A5 integer TileOps."""

from ptodsl import pto


def div_uint_soft(lhs, rhs, mask, dtype):
    """Restoring division, with one quotient bit resolved per iteration.

    A zero divisor is undefined; this loop sets every quotient bit in that lane.
    """
    bits = {"ui8": 8, "ui16": 16, "ui32": 32}[str(dtype)]
    scalar_dtype = {8: pto.ui8, 16: pto.ui16, 32: pto.ui32}[bits]
    zero = pto.vbr(scalar_dtype(0))
    one = pto.vbr(scalar_dtype(1))
    remainder = zero
    quotient = zero

    for bit in range(bits - 1, -1, -1):
        carry = pto.vcmps(remainder, scalar_dtype(1 << (bits - 1)), mask, pto.CmpMode.GE)
        next_bit = pto.vands(pto.vshrs(lhs, bit, mask), scalar_dtype(1), mask)
        remainder = pto.vor(pto.vshls(remainder, 1, mask), next_bit, mask)
        ge = pto.por(carry, pto.vcmp(remainder, rhs, mask, pto.CmpMode.GE), mask)
        remainder = pto.vsel(pto.vsub(remainder, rhs, mask), remainder, ge)
        quotient_bit = pto.vshls(one, bit, mask)
        quotient = pto.vsel(pto.vor(quotient, quotient_bit, mask), quotient, ge)
    return quotient


# The SoftLib vdiv ABI traces a three-argument helper (lhs, rhs, mask), so the
# element type is bound here rather than passed through. Both widths share the
# restoring loop above instead of carrying a second unsigned algorithm: only
# that loop reaches ui8, which the expand templates need, and the alternative
# borrowed the signed primitive, which has no i8 form to extend.
#
# The loop reads its operand width from ``str(dtype)``; the expand templates
# hand it a tile dtype, and these entry points have only vregs, so they pass
# the element type name directly.
def div_ui16_soft(vec, divisor, mask):
    """Divide unsigned 16-bit vectors, including values with the top bit set."""
    return div_uint_soft(vec, divisor, mask, "ui16")


def div_ui32_soft(vec, divisor, mask):
    """Divide unsigned 32-bit vectors, including values with the top bit set."""
    return div_uint_soft(vec, divisor, mask, "ui32")
