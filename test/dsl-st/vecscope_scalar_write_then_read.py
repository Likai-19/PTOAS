#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Vector/scalar UB ordering with a scalar overwrite of the reduction slot.

Double 256 f32 values, store their maximum at UB[256], overwrite UB[256]
with doubled input[0] via scalar load/store, then broadcast and add it.
Expected: output[i] = 2 * input[i] + 2 * input[0]. The intermediate reduction
is intentionally retained to exercise vector-store versus scalar-store order.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(
    name="vecscope_scalar_write_then_read_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def vecscope_scalar_write_then_read_kernel(input_data: pto.ptr(pto.f32, 'gm'), output_data: pto.ptr(pto.f32, 'gm')):
    buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, 'ub'))
    pto.mte_gm_ub(input_data, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 0, 1024, nburst=(1, 1024, 1024))
    pto.set_flag('MTE2', 'V', event_id=0)
    pto.wait_flag('MTE2', 'V', event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        for q in range(0, 2, 1):
            vload_0 = pto.vmi.vload(
                pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), q * 128),
                0, size=128,
            )
            brc_1 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1p+1')), size=128)
            vmul_2 = pto.vmi.vmul(vload_0, brc_1, mask)
            pto.vmi.vstore(vmul_2, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), q * 128), 0, mask)
        mask_3 = pto.vmi.create_mask(128, size=128)
        vld_red_4 = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 0, size=128)
        vld_red_5 = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 128, size=128)
        mask_6 = pto.vmi.create_mask(128, size=128)
        vmax_7 = pto.vmi.vmax(vld_red_4, vld_red_5, mask_6)
        vsel_8 = pto.vmi.vsel(mask_6, vmax_7, vld_red_4)
        vcmax_9 = pto.vmi.vcmax(vsel_8, mask_3)
        vbrc_red_10 = pto.vmi.vbrc(vcmax_9, size=128)
        mask_11 = pto.vmi.create_mask(1, size=128)
        pto.vmi.vstore(vbrc_red_10, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 256), 0, mask_11)
        pto.store(
            pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 0),
            pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 256,
        )
        for q_1 in range(0, 2, 1):
            vload_0_1 = pto.vmi.vload(
                pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), q_1 * 128),
                0, size=128,
            )
            sload_1 = pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), 256)
            brc_2 = pto.vmi.vbrc(sload_1, size=128)
            vadd_3 = pto.vmi.vadd(vload_0_1, brc_2, mask)
            pto.vmi.vstore(vadd_3, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, 'ub')), q_1 * 128), 0, mask)
    pto.set_flag('V', 'MTE3', event_id=0)
    pto.wait_flag('V', 'MTE3', event_id=0)
    pto.mte_ub_gm(
        pto.castptr(pto.const(0, dtype=pto.int64), pto.ptr(pto.f32, 'ub')),
        output_data, 1024, nburst=(1, 1024, 1024), l2_cache='naci',
    )


ELEMS = 256


def make_inputs():
    rng = np.random.default_rng(256)
    return [rng.uniform(-4.0, 4.0, ELEMS).astype(np.float32)]


def make_expected(src):
    scaled = (src * np.float32(2.0)).astype(np.float32)
    # The scalar store overwrites the reduced maximum with doubled input[0].
    return (scaled + scaled[0]).astype(np.float32)


CASES = [
    golden_output_case(
        "vecscope_scalar_write_then_read_e256_l128",
        vecscope_scalar_write_then_read_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
