#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime semantics of the ``pto.vmi.vci`` index ramp (#1482).

Lane ``i`` of the ramp is ``base + i`` (ASC) or ``base - i`` (DESC); with
``group=C`` each of the ``C`` groups restarts the ramp.  The hardware
``pto.vci`` DESC mode ramps as ``(index + VL - 1) - lane`` instead, so the
descending result is built from ``vci(0) + negate + add``.  A wrongly chosen
DESC source keeps the same IR shape and only shows up in the stored values,
which is what these cases compare.

The shapes cover the full physical VL, sub-VL (compact) ramps, the 1/2/4-lane
compact carriers, the group-periodic forms and the multi-chunk ramps (two
physical chunks, where each chunk gets its own base), across the element types
``vci`` accepts (the hardware VCI types ``i8``/``i16``/``i32``/``f16``/``f32``;
``bf16`` is rejected by the ``vmi`` and ``vpto`` verifiers alike).

The group shapes include both the compact carrier (``group = section``, one lane
per group, which only pins the base) and group sizes greater than one (e.g.
``L = 128, group = 2``), which pin the per-group ``base ± j`` ramp and its
restart at a group boundary.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

ORDERS = ("ASC", "DESC")

# (label, pto dtype, numpy dtype, element bytes, base, lane count, group,
#  orders).
#
# group = lane count is the compact group_slots carrier (one lane per group);
# the group = 2 shapes are the ones that exercise a group size greater than one.
SHAPES = (
    ("f16", pto.f16, np.float16, 2, 7.0, 128, None, ORDERS),
    ("f16", pto.f16, np.float16, 2, 7.0, 64, None, ORDERS),
    ("f16", pto.f16, np.float16, 2, 7.0, 8, 8, ORDERS),
    ("f16", pto.f16, np.float16, 2, 7.0, 1, None, ORDERS),
    # 256 f16 lanes span two physical chunks, so this checks the per-chunk base.
    ("f16", pto.f16, np.float16, 2, 7.0, 256, None, ORDERS),
    ("f16", pto.f16, np.float16, 2, 7.0, 128, 2, ORDERS),
    ("f16", pto.f16, np.float16, 2, 7.0, 64, 2, ORDERS),
    ("f32", pto.f32, np.float32, 4, 10.0, 64, None, ORDERS),
    ("f32", pto.f32, np.float32, 4, 10.0, 8, None, ORDERS),
    ("f32", pto.f32, np.float32, 4, 10.0, 1, None, ORDERS),
    ("f32", pto.f32, np.float32, 4, 10.0, 64, 2, ORDERS),
    ("i32", pto.i32, np.int32, 4, 11, 64, None, ORDERS),
    # 128 i32 lanes span two physical chunks, so this checks the per-chunk base.
    ("i32", pto.i32, np.int32, 4, 11, 128, None, ORDERS),
    ("i32", pto.i32, np.int32, 4, 11, 64, 2, ORDERS),
    ("i32", pto.i32, np.int32, 4, 11, 2, 2, ORDERS),
    ("i32", pto.i32, np.int32, 4, 11, 8, 8, ORDERS),
    ("i16", pto.i16, np.int16, 2, 300, 128, None, ORDERS),
    ("i16", pto.i16, np.int16, 2, 300, 128, 2, ORDERS),
    ("i16", pto.i16, np.int16, 2, 300, 8, None, ORDERS),
    ("i16", pto.i16, np.int16, 2, 300, 1, None, ORDERS),
    ("i8", pto.i8, np.int8, 1, 9, 256, None, ORDERS),
    ("i8", pto.i8, np.int8, 1, 9, 256, 2, ORDERS),
    ("i8", pto.i8, np.int8, 1, 9, 4, None, ORDERS),
    ("i8", pto.i8, np.int8, 1, 9, 1, None, ORDERS),
)


def ramp_golden(base, size, order, group, np_dtype):
    """Golden ramp: per-lane ``base ± i``, restarted inside every group."""
    lane = np.arange(size, dtype=np.int64)
    span = size if group is None else size // group
    local = lane % span
    values = base + local if order == "ASC" else base - local
    return values.astype(np_dtype)


def build_vci_case(label, dtype, np_dtype, itemsize, base, size, group, order):
    nbytes = size * itemsize
    name = f"vci_{label}_vl{size}_g{group or 1}_{order.lower()}"

    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(out_ptr: pto.ptr(dtype, "gm")) -> None:
        ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        ramp = pto.vmi.vci(dtype(base), size=size, order=order, group=group)
        pto.vmi.vstore(ramp, ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub, out_ptr, nbytes, nburst=(1, nbytes, nbytes))
        pto.pipe_barrier(pto.Pipe.ALL)

    golden = ramp_golden(base, size, order, group, np_dtype)
    return golden_output_case(
        name,
        kernel,
        inputs=lambda: [],
        expected=golden,
        output_shape=(size,),
        output_dtype=golden.dtype,
        rtol=0.0,
        atol=0.0,
    )


CASES = [
    build_vci_case(label, dtype, np_dtype, itemsize, base, size, group, order)
    for label, dtype, np_dtype, itemsize, base, size, group, orders in SHAPES
    for order in orders
]


auto_main(globals())
