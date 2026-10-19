# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tcmp."""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._elementwise import _ub_or_vec_row_major, traversal_metadata


_DTYPES = [
    (dtype, dtype, "i8")
    for dtype in ("f32", "i32", "ui32", "f16", "bf16", "i16", "ui16", "i8", "ui8")
]


@rewrite_jit_function
def _emit_tcmp_1d(src0, src1, dst):
    dtype = src0.dtype
    valid_rows, valid_cols = src0.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    total_elements = valid_rows * valid_cols
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    dst_ptr = dst.as_ptr()

    if str(dtype) in {"f32", "i32", "ui32"}:
        remained = total_elements
        for offset in range(0, total_elements, lanes * 2):
            first_mask, remained = pto.make_mask(dtype, remained)
            first_lhs = pto.vlds(src0_ptr, offset)
            first_rhs = pto.vlds(src1_ptr, offset)
            first_cmp = pto.vcmp(first_lhs, first_rhs, first_mask, cmp_mode)
            first_cmp_b8 = pto.pbitcast(first_cmp, pto.mask_b8)

            second_mask, remained = pto.make_mask(dtype, remained)
            second_lhs = pto.vlds(src0_ptr, offset + lanes)
            second_rhs = pto.vlds(src1_ptr, offset + lanes)
            second_cmp = pto.vcmp(second_lhs, second_rhs, second_mask, cmp_mode)
            second_cmp_b8 = pto.pbitcast(second_cmp, pto.mask_b8)

            packed_low, _ = pto.pdintlv_b8(first_cmp_b8, second_cmp_b8)
            pto.psts(packed_low, dst_ptr, offset // 8, dist=pto.PredicateDist.PK)
    elif str(dtype) in {"f16", "bf16", "i16", "ui16"}:
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0_ptr, offset)
            rhs = pto.vlds(src1_ptr, offset)
            cmp = pto.vcmp(lhs, rhs, mask, cmp_mode)
            cmp_b8 = pto.pbitcast(cmp, pto.mask_b8)
            pto.psts(
                cmp_b8,
                dst_ptr,
                offset // 8,
                dist=pto.PredicateDist.PK,
            )
    else:
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0_ptr, offset)
            rhs = pto.vlds(src1_ptr, offset)
            cmp = pto.vcmp(lhs, rhs, mask, cmp_mode)
            pto.psts(
                cmp,
                dst_ptr,
                offset // 8,
                dist=pto.PredicateDist.NORM,
            )


def _static_valid_cols(tile):
    """Return the tile's static valid column count, or None when dynamic."""
    static_valid_shape = getattr(tile, "_template_static_valid_shape", None)
    if static_valid_shape is None:
        static_valid_shape = getattr(tile, "static_valid_shape", None)
    if static_valid_shape is not None and static_valid_shape[1] is not None:
        return static_valid_shape[1]
    return None


def _needs_row_tail_fill(static_cols, elements_per_store, bytes_per_store):
    """Statically decide whether a dst valid row extends past the store span.

    Comparing traced surface values would emit runtime compares, so the gate
    only uses plain Python ints. A dynamic valid shape returns True and lets
    the fill loop take zero trips at runtime when the stores already cover
    the whole valid row.
    """
    if static_cols is None:
        return True
    store_span = (
        (static_cols + elements_per_store - 1) // elements_per_store
    ) * bytes_per_store
    return store_span < static_cols


@rewrite_jit_function
def _emit_tcmp_2d_tail_fill(dst_ptr, dst_stride, valid_rows, store_span, valid_cols):
    """Zero the dst valid-row bytes that the predicate stores do not cover.

    A predicate store writes a fixed span selected by DIST (PK 16B / NORM 32B
    on A5) regardless of how many predicate bits are valid. When a valid row
    is wider than the stores' total span, the tail bytes would otherwise keep
    whatever the UB held before the compare, so they are zeroed here to honor
    the packed-mask contract (packed bits, LSB-first, then zero padding).

    The fill reuses PK predicate stores of an all-inactive mask: one 16-byte
    zero payload per store, at the same 16-byte address granularity the
    packed PK stores already use (a vector store would need 32-byte-aligned
    addresses and cannot express the 16-byte offset). The final chunk ends at
    round_up_16(valid_cols), which never crosses the row end because dst rows
    are 32-byte aligned and at least valid_cols wide. Bounds may be static
    ints or traced runtime values; the loop is zero-trip when the predicate
    stores already cover the whole valid row.
    """
    zero_mask = pto.pset_b8(pto.PAT.ALLF)
    for row in range(0, valid_rows, 1):
        for offset in range(store_span, valid_cols, 16):
            pto.psts(
                zero_mask,
                dst_ptr,
                row * dst_stride + offset,
                dist=pto.PredicateDist.PK,
            )


@rewrite_jit_function
def _emit_tcmp_2d_f32(src0, src1, dst, cmp_mode):
    dtype = src0.dtype
    valid_rows, valid_cols = src0.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    dst_ptr = dst.as_ptr()
    dst_stride = dst.shape[1]
    repeat_times = (valid_cols + lanes - 1) // lanes + 1
    iterations = repeat_times // 2
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, iterations, 1):
            first_offset = col * lanes * 2
            second_offset = (col * 2 + 1) * lanes

            first_mask, remained = pto.make_mask(dtype, remained)
            first_lhs = pto.vlds(src0[row, first_offset:])
            first_rhs = pto.vlds(src1[row, first_offset:])
            first_cmp = pto.vcmp(
                first_lhs,
                first_rhs,
                first_mask,
                cmp_mode,
            )
            first_cmp_b8 = pto.pbitcast(first_cmp, pto.mask_b8)

            second_mask, remained = pto.make_mask(dtype, remained)
            second_lhs = pto.vlds(src0[row, second_offset:])
            second_rhs = pto.vlds(src1[row, second_offset:])
            second_cmp = pto.vcmp(
                second_lhs,
                second_rhs,
                second_mask,
                cmp_mode,
            )
            second_cmp_b8 = pto.pbitcast(second_cmp, pto.mask_b8)

            packed_low, _ = pto.pdintlv_b8(
                first_cmp_b8,
                second_cmp_b8,
            )
            store_offset = row * dst_stride + col * 16
            pto.psts(
                packed_low,
                dst_ptr,
                store_offset,
                dist=pto.PredicateDist.PK,
            )
    if _needs_row_tail_fill(_static_valid_cols(src0), lanes * 2, 16):
        _emit_tcmp_2d_tail_fill(
            dst_ptr, dst_stride, valid_rows, iterations * 16, valid_cols
        )


@rewrite_jit_function
def _emit_tcmp_2d_narrow(src0, src1, dst, cmp_mode, packed):
    """Emit row-wise compares for packed f16/i16 or per-byte i8 predicates."""
    dtype = src0.dtype
    valid_rows, valid_cols = src0.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    dst_ptr = dst.as_ptr()
    dst_stride = dst.shape[1]
    iterations = (valid_cols + lanes - 1) // lanes
    bytes_per_iter = 16 if packed else 32
    dist = pto.PredicateDist.PK if packed else pto.PredicateDist.NORM
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, iterations, 1):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col * lanes:])
            rhs = pto.vlds(src1[row, col * lanes:])
            cmp = pto.vcmp(lhs, rhs, mask, cmp_mode)
            if packed:
                cmp = pto.pbitcast(cmp, pto.mask_b8)
            store_offset = row * dst_stride + col * bytes_per_iter
            pto.psts(
                cmp,
                dst_ptr,
                store_offset,
                dist=dist,
            )
    if _needs_row_tail_fill(_static_valid_cols(src0), lanes, bytes_per_iter):
        _emit_tcmp_2d_tail_fill(
            dst_ptr, dst_stride, valid_rows, iterations * bytes_per_iter, valid_cols
        )


@rewrite_jit_function
def _emit_tcmp_2d(src0, src1, dst):
    dtype = src0.dtype
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    if str(dtype) in {"f32", "i32", "ui32"}:
        _emit_tcmp_2d_f32(src0, src1, dst, cmp_mode)
    else:
        _emit_tcmp_2d_narrow(src0, src1, dst, cmp_mode, str(dtype) in {"f16", "bf16", "i16", "ui16"})


def _register_tcmp(*, name, traversal):
    constraints = [
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape("src0", "src1", "dst"),
    ]
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(
            tilelib.require_predicate_compare_1d(
                "src0",
                "src1",
                predicate_operand="dst",
            )
        )

    @tilelib.tile_template(
        op="pto.tcmp",
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("compare", "predicate-store"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        if traversal == "1d":
            _emit_tcmp_1d(src0, src1, dst)
        else:
            _emit_tcmp_2d(src0, src1, dst)

    return template


template_tcmp = _register_tcmp(
    name="template_tcmp",
    traversal="2d",
)

template_tcmp_1d = _register_tcmp(
    name="template_tcmp_1d",
    traversal="1d",
)
