#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Vecscope auxiliary-scalar reload ST (issue #1593, case e150-l128-max-consume).

The kernel doubles 150 f32 in UB, reduces their maximum into UB[152] with a
vector store, then reloads that slot and rescales every lane by it. This retains
the kernel operations from vector_sclaer.py, without its unused TileLang imports.
The scalar load followed by broadcast can fold into a vector broadcast load;
the required barrier family depends on the lowered accesses.

Observable: out[i] == 2*src[i] * max(2*src).
Real-device commands, including the all-only membar configuration, are in
../../vecscope_scalar_reload.md (repository root).
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value


ELEMS = 150
BYTES = ELEMS * 4  # 600 bytes, matches the kernel's DMA extents


@pto.jit(
    name="vecscope_scalar_reload_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def vecscope_scalar_reload_kernel(input_data: pto.ptr(pto.f32, "gm"), output_data: pto.ptr(pto.f32, "gm")):
    buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    pto.mte_gm_ub(input_data, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 0, 600, nburst=(1, 600, 600))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    with pto.vecscope():
        remaining = pto.const(0, dtype=pto.int64)
        remaining = _tl_wrap_surface_value(_tl_coerce_i64(150, context="PTO local.var store"))
        for q in range(0, 2, 1):
            mask = pto.vmi.create_mask(remaining, size=128)
            vload_0 = pto.vmi.vload(
                pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), q * 128), 0, size=128
            )
            brc_1 = pto.vmi.vbrc(pto.f32(float.fromhex("0x1p+1")), size=128)
            vmul_2 = pto.vmi.vmul(vload_0, brc_1, mask)
            pto.vmi.vstore(
                vmul_2, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), q * 128), 0, mask
            )
            remaining = _tl_wrap_surface_value(
                _tl_coerce_i64(remaining - 128, context="PTO local.var store")
            )
        mask_3 = pto.vmi.create_mask(128, size=128)
        vld_red_4 = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 0, size=128)
        vld_red_5 = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 128, size=128)
        mask_6 = pto.vmi.create_mask(22, size=128)
        vmax_7 = pto.vmi.vmax(vld_red_4, vld_red_5, mask_6)
        vsel_8 = pto.vmi.vsel(mask_6, vmax_7, vld_red_4)
        vcmax_9 = pto.vmi.vcmax(vsel_8, mask_3)
        vbrc_red_10 = pto.vmi.vbrc(vcmax_9, size=128)
        mask_11 = pto.vmi.create_mask(1, size=128)
        pto.vmi.vstore(
            vbrc_red_10, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 152), 0, mask_11
        )
        remaining_1 = pto.const(0, dtype=pto.int64)
        remaining_1 = _tl_wrap_surface_value(_tl_coerce_i64(150, context="PTO local.var store"))
        for q_1 in range(0, 2, 1):
            mask_1 = pto.vmi.create_mask(remaining_1, size=128)
            vload_0_1 = pto.vmi.vload(
                pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), q_1 * 128), 0, size=128
            )
            sload_1 = pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 152)
            brc_2 = pto.vmi.vbrc(sload_1, size=128)
            vmul_3 = pto.vmi.vmul(vload_0_1, brc_2, mask_1)
            pto.vmi.vstore(
                vmul_3, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), q_1 * 128), 0, mask_1
            )
            remaining_1 = _tl_wrap_surface_value(
                _tl_coerce_i64(remaining_1 - 128, context="PTO local.var store")
            )
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(
        pto.castptr(pto.const(0, dtype=pto.int64), pto.ptr(pto.f32, "ub")),
        output_data,
        600,
        nburst=(1, 600, 600),
        l2_cache="naci",
    )


def make_inputs():
    rng = np.random.default_rng(1593)
    return [rng.uniform(-4.0, 4.0, ELEMS).astype(np.float32)]


def make_expected(src):
    scaled = (src * np.float32(2.0)).astype(np.float32)
    peak = np.float32(scaled.max())
    return (scaled * peak).astype(np.float32)


CASES = [
    golden_output_case(
        "vecscope_reduce_scalar_reload_e150_l128_max_consume",
        vecscope_scalar_reload_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
