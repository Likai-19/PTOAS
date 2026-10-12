#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Descending index vectors feeding ``pto.vmi.vselr`` (#1482).

A lane permutation is expressed as a select by index vector, and the index
vector is a `pto.vmi.vci` ramp: ``vci(63, DESC)`` reverses a 64-lane chunk while
``vci(0, ASC)`` is the identity.  #1482 showed the descending ramp must be
``base - lane``; with the hardware ``pto.vci`` DESC mode (``(index + VL - 1) -
lane``) the index wraps modulo the lane count and vselr rotates the chunk
instead of reversing it, so these cases compare the stored lanes rather than the
IR.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 64
ITEMSIZE = 4
NBYTES = LANES * ITEMSIZE
# Distinctive values so a wrong index is unmistakable in the comparison.
SOURCE = (100 + np.arange(LANES)).astype(np.float32)


def build_vselr_case(descending):
    order = "DESC" if descending else "ASC"
    base = LANES - 1 if descending else 0
    name = f"vselr_vl{LANES}_{order.lower()}_index"

    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(src_ptr: pto.ptr(pto.f32, "gm"),
               out_ptr: pto.ptr(pto.f32, "gm")) -> None:
        ub_src = pto.castptr(pto.i64(0), pto.ptr(pto.f32, "ub"))
        ub_out = pto.castptr(pto.i64(4096), pto.ptr(pto.f32, "ub"))
        pto.mte_gm_ub(src_ptr, ub_src, 0, NBYTES,
                      nburst=(1, NBYTES, NBYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        source = pto.vmi.vload(ub_src, 0, size=LANES)
        index = pto.vmi.vci(pto.i32(base), size=LANES, order=order)
        permuted = pto.vmi.vselr(source, index)
        pto.vmi.vstore(permuted, ub_out, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_out, out_ptr, NBYTES, nburst=(1, NBYTES, NBYTES))
        pto.pipe_barrier(pto.Pipe.ALL)

    def make_expected(source):
        return source[::-1].copy() if descending else source.copy()

    return golden_output_case(
        name,
        kernel,
        inputs=lambda: [SOURCE.copy()],
        expected=make_expected,
        output_index=1,
        rtol=0.0,
        atol=0.0,
    )


CASES = [build_vselr_case(descending) for descending in (True, False)]


auto_main(globals())
