# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tfillpad

Semantic (based on C++ TFillPad.hpp reference):
  - TFILLPAD: copies src.valid data to dst, then fills dst expansion with FillPadVal
  - TFILLPAD_INPLACE: same physical buffer (src == dst), skips copy phase, only fills expansion

Key logic from C++:
  if constexpr (!inplace) {
      CopyValidElementsVec(dst, src, ...);  // Phase 1+3: copy valid data
  }
  // Phase 2+4: fill expansion (always executed if dst has larger valid region)
  FillExpansion(dst, padCols, padRows, padValue);

Strategy:
  - Phase 1: Copy aligned valid blocks (cols 0 to aligned_col-1) [only if !inplace]
  - Phase 2: Fill cols aligned_col to dst_valid_cols-1 with FillPadVal
  - Phase 3: Copy tail valid lanes (cols aligned_col to src_valid_cols-1) [only if !inplace]
  - Phase 4: Fill row expansion

Address alignment and unaligned handling:
  - vlds/vsts require 32-byte aligned base addresses
  - Phase 1: col=0 is always aligned (tile base address is aligned)
  - Phase 2/4: handle non-aligned lengths using make_mask() to control active lanes

Note: There is no separate pto.tfillpad_inplace operation in PTO IR.
      In-place mode is expressed via tfillpad with src and dst being the same SSA value.
      This template handles both cases by detecting if copy phase is needed.
"""

import tilelang_dsl as pto

_NEG1_F32 = -1.0

# All supported dtype pairs
_DTYPE_SIGNATURES = [
    (pto.f32, pto.f32),
    (pto.i16, pto.i16),
    (pto.si16, pto.si16),
    (pto.ui16, pto.ui16),
    (pto.i32, pto.i32),
    (pto.si32, pto.si32),
    (pto.ui32, pto.ui32),
    (pto.i8, pto.i8),
    (pto.si8, pto.si8),
    (pto.ui8, pto.ui8),
]


@pto.vkernel(
    target="a5",
    op="pto.tfillpad",
    dtypes=_DTYPE_SIGNATURES,
)
def template_tfillpad(src: pto.Tile, dst: pto.Tile):
    """tfillpad: copy src.valid to dst and fill expansion regions.

    Based on C++ TFillPad.hpp reference:
      - TFILLPAD (non-inplace): CopyValidElementsVec + FillExpansion

    tfillpad requires src.shape == dst.shape (same physical size).
    If dst.valid > src.valid, fill the expansion regions.
    """
    dtype = dst.element_type
    src_rows, _ = src.shape
    src_valid_rows, src_valid_cols = src.valid_shape
    dst_rows, dst_cols = dst.shape
    dst_valid_rows, dst_valid_cols = dst.valid_shape

    lanes = pto.get_lanes(dtype)
    aligned_col = (src_valid_cols // lanes) * lanes
    has_tail = src_valid_cols > aligned_col
    has_valid_expansion = (src_valid_cols < dst_valid_cols) or (src_valid_rows < dst_valid_rows)

    # PadValue handling - dtype-specific (inline to avoid external call)
    if pto.constexpr(dtype == pto.f32):
        if pto.constexpr(dst.pad_value == pto.PadValue.ZERO):
            if has_valid_expansion:
                fill_scalar = pto.f32(_NEG1_F32)
            else:
                fill_scalar = pto.f32(0.0)
        elif pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.f32(0.0)
    elif pto.constexpr(dtype == pto.ui16):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.ui16(0)
    elif pto.constexpr(dtype == pto.si16):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.si16(0)
    elif pto.constexpr(dtype == pto.i16):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.i16(0)
    elif pto.constexpr(dtype == pto.ui32):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.ui32(0)
    elif pto.constexpr(dtype == pto.si32):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.si32(0)
    elif pto.constexpr(dtype == pto.i32):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.i32(0)
    elif pto.constexpr(dtype == pto.ui8):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.ui8(0)
    elif pto.constexpr(dtype == pto.si8):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.si8(0)
    elif pto.constexpr(dtype == pto.i8):
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            fill_scalar = dst.pad_value.eval()
        else:
            fill_scalar = pto.i8(0)

    # Phase 1: Copy aligned valid blocks
    if aligned_col > 0:
        for row in range(0, src_valid_rows, 1):
            remained = aligned_col
            for col in range(0, aligned_col, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                data = pto.vlds(src[row, col:])
                pto.vsts(data, dst[row, col:], mask)

    # Phase 2: Fill cols from aligned_col to dst_cols-1
    if aligned_col < dst_cols:
        for row in range(0, src_valid_rows, 1):
            remained = dst_cols - aligned_col
            for col in range(aligned_col, dst_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vdup(fill_scalar, mask)
                pto.vsts(vec, dst[row, col:], mask)

    # Phase 3: Copy tail valid lanes
    if has_tail:
        for row in range(0, src_valid_rows, 1):
            remained = src_valid_cols - aligned_col
            mask_copy, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, aligned_col:])
            pto.vsts(data, dst[row, aligned_col:], mask_copy)

    # Phase 4: Fill row expansion
    if pto.constexpr(src_rows < dst_rows):
        for row in range(src_rows, dst_rows, 1):
            remained = dst_cols
            for col in range(0, dst_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vdup(fill_scalar, mask)
                pto.vsts(vec, dst[row, col:], mask)

    return
