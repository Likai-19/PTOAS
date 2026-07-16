# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL VMI implementation of the group_count dhist benchmark."""

from ptodsl import pto
from ptodsl._control_flow import vecscope as _vecscope
from ptodsl._runtime import native_build as _native_build

_native_build._source_ptoas_overrides = (  # noqa: SLF001
    lambda module_spec: {
        "backend": getattr(module_spec, "backend", None) or "vpto",
        "pto_level": "level3",
    }
)

SRC_ELEMS = 24064
NBINS = 256
NUM_GROUPS = 9
VMI_TILE_ELEMS = 256
NUM_VMI_TILES = SRC_ELEMS // VMI_TILE_ELEMS

_SRC_UB = 0
_OUT_UB = SRC_ELEMS * 4


@pto.jit(
    name="group_count_dhist_vmi",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def group_count_dhist_vmi(
    src_gm: pto.ptr(pto.si64, "gm"),
    out_gm: pto.ptr(pto.ui32, "gm"),
):
    src_ub = pto.castptr(pto.const(_SRC_UB, dtype=pto.ui64), pto.ptr(pto.si64, "ub"))
    src32_ub = pto.castptr(pto.const(_SRC_UB, dtype=pto.ui64), pto.ptr(pto.ui32, "ub"))
    out_ub = pto.castptr(pto.const(_OUT_UB, dtype=pto.ui64), pto.ptr(pto.ui32, "ub"))

    pto.mte_gm_ub(src_gm, src_ub, 0, SRC_ELEMS * 8, nburst=(1, SRC_ELEMS * 8, SRC_ELEMS * 8))
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    with _vecscope():
        hist_source_mask = pto.vmi.create_mask(VMI_TILE_ELEMS, size=VMI_TILE_ELEMS)
        hist_mask = pto.vmi.create_mask(NBINS, size=NBINS)

        hist16_zero = pto.vmi.vbrc(pto.ui16(0), result_type=pto.vmi.vreg(256, pto.ui16))
        hist32_init = pto.vmi.vbrc(pto.ui32(0), result_type=pto.vmi.vreg(256, pto.ui32))

        tile_loop = pto.for_(0, NUM_VMI_TILES, step=1).carry(hist32=hist32_init)
        with tile_loop:
            idx32, _ = pto.vmi.vload(
                src32_ub,
                tile_loop.iv * VMI_TILE_ELEMS * 2,
                size=VMI_TILE_ELEMS,
                dist_mode="dintlv",
            )
            valid = pto.vmi.vcmps(idx32, pto.ui32(NUM_GROUPS), hist_source_mask, "lt", result_type=pto.vmi.mask(256))
            source = pto.vmi.vcvt(idx32, pto.ui8, result_type=pto.vmi.vreg(256, pto.ui8))
            hist = pto.vmi.vdhist(
                hist16_zero,
                source,
                valid,
                result_type=pto.vmi.vreg(256, pto.ui16),
            )
            hist32 = pto.vmi.vcvt(hist, pto.ui32, result_type=pto.vmi.vreg(256, pto.ui32))
            hist32 = pto.vmi.vadd(tile_loop.hist32, hist32, hist_mask, result_type=pto.vmi.vreg(256, pto.ui32))
            tile_loop.update(hist32=hist32)

        pto.vmi.vstore(tile_loop.final("hist32"), out_ub, 0, hist_mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
    pto.mte_ub_gm(out_ub, out_gm, NBINS * 4, nburst=(1, NBINS * 4, NBINS * 4))
    pto.mem_bar(pto.BarrierType.SS_ALL)
