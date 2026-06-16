# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowexpanddiv with IEEE 754 high-precision support

Divide each row of src0 by a per-row scalar from src1[row, 0].
Semantics: dst[row, col] = src0[row, col] / src1[row, 0]
"""

import sys
from pathlib import Path
import tilelang_dsl as pto

# Import shared high-precision division algorithms
from div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


def _config_value(config, name):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _matches_layout(value, expected, expected_name):
    if value is None:
        return False
    return value == expected or value == expected_name or str(value).lower().endswith(expected_name)


def _is_row_major(tile: pto.Tile) -> bool:
    return _matches_layout(_config_value(tile.config, "b_layout"), pto.BLayout.ROW_MAJOR, "row_major")


def _is_col_major(tile: pto.Tile) -> bool:
    return _matches_layout(_config_value(tile.config, "b_layout"), pto.BLayout.COL_MAJOR, "col_major")


def _is_col_major_row_scalar(tile: pto.Tile) -> bool:
    shape = tuple(tile.shape)
    return len(shape) == 2 and _is_col_major(tile) and shape[1] == 1


def _constraint_trowexpanddiv_row_major(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint for RowMajor layout trowexpanddiv template."""
    src1_supported = _is_row_major(src1) or _is_col_major_row_scalar(src1)
    return _is_row_major(src0) and src1_supported and _is_row_major(dst)


@pto.vkernel(
    target="a5",
    op="pto.trowexpanddiv",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    constraints=[_constraint_trowexpanddiv_row_major],
)
def template_trowexpanddiv_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpanddiv with f32 dtype and optional high-precision mode.

    When src1 is col_major with shape [M, 1] (a per-row scalar column),
    vlds on the col_major tile slice would access UB at non-512B-aligned
    addresses (error 340 on A5).  Use vldas+vldus (unaligned load pipeline)
    for src1 in that case; keep the aligned vlds path for row_major src1.
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        if pto.constexpr(src1.config.b_layout == pto.BLayout.COL_MAJOR):
            # col_major / high_precision
            for row in range(0, valid_rows, 1):
                align_src1 = pto.vldas(src1[row, :])
                scalar_vec, _ = pto.vldus(src1[row, :], align_src1)
                broadcasted = pto.vdup(scalar_vec, pto.make_mask(dtype, pto.PAT.ALL))
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    result = _div_ieee754_f32_impl(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
        else:
            # row_major / high_precision
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    scalar_vec = pto.vlds(src1[row, :])
                    broadcasted = pto.vdup(scalar_vec, mask)
                    lhs = pto.vlds(src0[row, col:])
                    result = _div_ieee754_f32_impl(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
    else:
        if pto.constexpr(src1.config.b_layout == pto.BLayout.COL_MAJOR):
            # col_major / default precision
            for row in range(0, valid_rows, 1):
                align_src1 = pto.vldas(src1[row, :])
                scalar_vec, _ = pto.vldus(src1[row, :], align_src1)
                broadcasted = pto.vdup(scalar_vec, pto.make_mask(dtype, pto.PAT.ALL))
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    result = pto.vdiv(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
        else:
            # row_major / default precision (existing behaviour)
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    scalar_vec = pto.vlds(src1[row, :])
                    broadcasted = pto.vdup(scalar_vec, mask)
                    lhs = pto.vlds(src0[row, col:])
                    result = pto.vdiv(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.trowexpanddiv",
    dtypes=[(pto.f16, pto.f16, pto.f16)],
    constraints=[_constraint_trowexpanddiv_row_major],
)
def template_trowexpanddiv_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpanddiv with f16 dtype and optional high-precision mode.

    When src1 is col_major with shape [M, 1] (a per-row scalar column),
    vlds on the col_major tile slice would access UB at non-512B-aligned
    addresses (error 340 on A5).  Use vldas+vldus (unaligned load pipeline)
    for src1 in that case; keep the aligned vlds path for row_major src1.
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    precision_type = pto.get_op_attr("precisionType", "default")
    if pto.constexpr(precision_type == "high_precision"):
        if pto.constexpr(src1.config.b_layout == pto.BLayout.COL_MAJOR):
            # col_major / high_precision
            for row in range(0, valid_rows, 1):
                align_src1 = pto.vldas(src1[row, :])
                scalar_vec, _ = pto.vldus(src1[row, :], align_src1)
                broadcasted = pto.vdup(scalar_vec, pto.make_mask(dtype, pto.PAT.ALL))
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    result = _div_ieee754_f16_impl(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
        else:
            # row_major / high_precision
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    scalar_vec = pto.vlds(src1[row, :])
                    broadcasted = pto.vdup(scalar_vec, mask)
                    lhs = pto.vlds(src0[row, col:])
                    result = _div_ieee754_f16_impl(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
    else:
        if pto.constexpr(src1.config.b_layout == pto.BLayout.COL_MAJOR):
            # col_major / default precision
            for row in range(0, valid_rows, 1):
                align_src1 = pto.vldas(src1[row, :])
                scalar_vec, _ = pto.vldus(src1[row, :], align_src1)
                broadcasted = pto.vdup(scalar_vec, pto.make_mask(dtype, pto.PAT.ALL))
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    result = pto.vdiv(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
        else:
            # row_major / default precision (existing behaviour)
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    scalar_vec = pto.vlds(src1[row, :])
                    broadcasted = pto.vdup(scalar_vec, mask)
                    lhs = pto.vlds(src0[row, col:])
                    result = pto.vdiv(lhs, broadcasted, mask)
                    pto.vsts(result, dst[row, col:], mask)
    return
