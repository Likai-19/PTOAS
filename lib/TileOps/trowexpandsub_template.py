# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowexpandsub"""

import sys
from pathlib import Path
import tilelang_dsl as pto


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


def _constraint_trowexpandsub_row_major(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint for RowMajor layout trowexpandsub template."""
    src1_supported = _is_row_major(src1) or _is_col_major_row_scalar(src1)
    return _is_row_major(src0) and src1_supported and _is_row_major(dst)


@pto.vkernel(
    target="a5",
    op="pto.trowexpandsub",
    dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat), (pto.AnyInt, pto.AnyInt, pto.AnyInt)],
    constraints=[_constraint_trowexpandsub_row_major],
)
def template_trowexpandsub(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpandsub.

    Subtract a per-row scalar from src1[row, 0] from each row of src0.
    Semantics: dst[row, col] = src0[row, col] - src1[row, 0]

    When src1 is col_major with shape [M, 1] (a per-row scalar column),
    vlds on the col_major tile slice would access UB at non-512B-aligned
    addresses (error 340 on A5).  Use vldas+vldus (unaligned load pipeline)
    for src1 in that case; keep the aligned vlds path for row_major src1.
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    if pto.constexpr(src1.config.b_layout == pto.BLayout.COL_MAJOR):
        # ---- col_major [M, 1] path: unaligned load for src1 ----
        for row in range(0, valid_rows, 1):
            # vldas+vldus once per row, broadcast across all col iterations
            align_src1 = pto.vldas(src1[row, :])
            scalar_vec, _ = pto.vldus(src1[row, :], align_src1)
            broadcasted = pto.vdup(scalar_vec, pto.make_mask(dtype, pto.PAT.ALL))
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                result = pto.vsub(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    else:
        # ---- row_major path: aligned vlds (existing behaviour) ----
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                # Load the scalar vector from src1[row, :]
                # For row-major src1, valid_shape[1] is 32/sizeof(dtype)
                # (e.g., 8 for f32).  vdup broadcasts the first element
                # to the full vector width.
                scalar_vec = pto.vlds(src1[row, :])
                broadcasted = pto.vdup(scalar_vec, mask)
                lhs = pto.vlds(src0[row, col:])
                result = pto.vsub(lhs, broadcasted, mask)
                pto.vsts(result, dst[row, col:], mask)
    return
