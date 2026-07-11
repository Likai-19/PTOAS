# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VMI rewrite of AntiMxQuantTailAxis::ComputeData for the FP8 + FP32-scale path.

Source region:
  test/kernel-test/kernels/dequant/anti_mx_quant_tail_axis.h:380

Current modeling notes:
  1. The source does ``DIST_DINTLV_B8`` and then immediately interleaves the
     two FP8 registers before FP8-to-FP32 part casts. Logically that is
     equivalent to rebuilding two contiguous 256-lane FP8 chunks, so the VMI
     rewrite models this as two dense FP8 loads and lets ``vcvt`` produce the
     natural deinterleaved=4 FP32 layout directly.
  2. ``scaleBufAddr`` is still modeled as the logical FP32 scale expansion
     produced by ``ComputeScale(..., float*)``. Each 64-lane scale chunk is
     expressed as a grouped zero-stride VMI load that expands to the matching
     256-lane FP32 half consumed by the FP8 payload.
"""

from ptodsl import pto

_FP8_HALF_LANES = 256
_CHUNK_LANES = 64
_SCALE_REPEAT_GROUPS = _FP8_HALF_LANES // _CHUNK_LANES
_SCALE_LANES_PER_LOOP = 128
_ELEMS_PER_LOOP = 512
_X_BYTES_PER_LOOP = _ELEMS_PER_LOOP
_SCALE_BYTES_PER_LOOP = _SCALE_LANES_PER_LOOP * 4
_F32_Y_BYTES_PER_LOOP = _ELEMS_PER_LOOP * 4
_B16_Y_BYTES_PER_LOOP = _ELEMS_PER_LOOP * 2

_X_BASE_ADDR = 0
_SCALE_BASE_ADDR = 0x4000
_Y_BASE_ADDR = 0x8000
def _compute_data_ub(
    x_local_addr: pto.i64,
    scale_buf_addr: pto.i64,
    y_local_addr: pto.i64,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    if pto.const_expr(SRC_FMT == "e4m3"):
        x_ptr = pto.castptr(x_local_addr, pto.ptr(pto.f8e4m3, "ub"))
    elif pto.const_expr(SRC_FMT == "e5m2"):
        x_ptr = pto.castptr(x_local_addr, pto.ptr(pto.f8e5m2, "ub"))
    else:
        raise ValueError(f"unsupported SRC_FMT specialization: {SRC_FMT}")

    scale_ptr = pto.castptr(scale_buf_addr, pto.ptr(pto.f32, "ub"))

    if pto.const_expr(DST_FMT == "f32"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.f32, "ub"))
        dst_dtype = pto.f32
    elif pto.const_expr(DST_FMT == "bf16"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.bf16, "ub"))
        dst_dtype = pto.bf16
    elif pto.const_expr(DST_FMT == "f16"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.f16, "ub"))
        dst_dtype = pto.f16
    else:
        raise ValueError(f"unsupported DST_FMT specialization: {DST_FMT}")

    mask256 = pto.vmi.create_mask(_FP8_HALF_LANES, size=_FP8_HALF_LANES)

    for i in range(LOOP_NUM2VF):
        x_off = i * _ELEMS_PER_LOOP
        scale_off = i * _SCALE_LANES_PER_LOOP
        y_off = i * _ELEMS_PER_LOOP

        x_lo_f8 = pto.vmi.vload(x_ptr, x_off, size=_FP8_HALF_LANES)
        x_hi_f8 = pto.vmi.vload(x_ptr, x_off + _FP8_HALF_LANES, size=_FP8_HALF_LANES)
        x_lo_f32 = pto.vmi.vcvt(x_lo_f8, pto.f32)
        x_hi_f32 = pto.vmi.vcvt(x_hi_f8, pto.f32)
        # `size` is the logical result width. With `stride=0` and `group=4`,
        # VMI reuses the same 64-lane scale chunk across 4 groups, so the
        # source footprint is still one VL chunk rather than 256 distinct loads.
        scale_lo = pto.vmi.vload(
            scale_ptr,
            scale_off,
            size=_FP8_HALF_LANES,
            stride=0,
            group=_SCALE_REPEAT_GROUPS,
        )
        scale_hi = pto.vmi.vload(
            scale_ptr,
            scale_off + _CHUNK_LANES,
            size=_FP8_HALF_LANES,
            stride=0,
            group=_SCALE_REPEAT_GROUPS,
        )
        y_lo = pto.vmi.vmul(x_lo_f32, scale_lo, mask256)
        y_hi = pto.vmi.vmul(x_hi_f32, scale_hi, mask256)

        if pto.const_expr(DST_FMT == "f32"):
            pto.vmi.vstore(y_lo, y_ptr, y_off, mask256)
            pto.vmi.vstore(y_hi, y_ptr, y_off + _FP8_HALF_LANES, mask256)
        else:
            pto.vmi.vstore(
                pto.vmi.vcvt(y_lo, dst_dtype),
                y_ptr,
                y_off,
                mask256,
            )
            pto.vmi.vstore(
                pto.vmi.vcvt(y_hi, dst_dtype),
                y_ptr,
                y_off + _FP8_HALF_LANES,
                mask256,
            )


def _runtime_entry(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.f32, "gm"),
    y_gm,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    x_ub_ptr = pto.castptr(pto.const(_X_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.ui8, "ub"))
    scale_ub_ptr = pto.castptr(pto.const(_SCALE_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    x_bytes = LOOP_NUM2VF * _X_BYTES_PER_LOOP
    scale_bytes = LOOP_NUM2VF * _SCALE_BYTES_PER_LOOP

    if pto.const_expr(DST_FMT == "f32"):
        y_ub_ptr = pto.castptr(pto.const(_Y_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
        y_bytes = LOOP_NUM2VF * _F32_Y_BYTES_PER_LOOP
    else:
        y_ub_ptr = pto.castptr(
            pto.const(_Y_BASE_ADDR, dtype=pto.ui64),
            pto.ptr(pto.bf16 if pto.const_expr(DST_FMT == "bf16") else pto.f16, "ub"),
        )
        y_bytes = LOOP_NUM2VF * _B16_Y_BYTES_PER_LOOP

    pto.mte_gm_ub(x_gm, x_ub_ptr, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
    pto.mte_gm_ub(scale_gm, scale_ub_ptr, 0, scale_bytes, nburst=(1, scale_bytes, scale_bytes))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    _compute_data_ub(
        pto.const(_X_BASE_ADDR, dtype=pto.i64),
        pto.const(_SCALE_BASE_ADDR, dtype=pto.i64),
        pto.const(_Y_BASE_ADDR, dtype=pto.i64),
        SRC_FMT=SRC_FMT,
        DST_FMT=DST_FMT,
        LOOP_NUM2VF=LOOP_NUM2VF,
    )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(y_ub_ptr, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="anti_mx_quant_tail_axis_compute_data_probe",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def anti_mx_quant_tail_axis_compute_data_probe(
    x_local_addr: pto.i64 = _X_BASE_ADDR,
    scale_buf_addr: pto.i64 = _SCALE_BASE_ADDR,
    y_local_addr: pto.i64 = _Y_BASE_ADDR,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _compute_data_ub(
        x_local_addr,
        scale_buf_addr,
        y_local_addr,
        SRC_FMT=SRC_FMT,
        DST_FMT=DST_FMT,
        LOOP_NUM2VF=LOOP_NUM2VF,
    )


@pto.jit(
    name="dequant_vmi_f32",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_f32(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.f32, "gm"),
    y_gm: pto.ptr(pto.f32, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="f32",
        LOOP_NUM2VF=LOOP_NUM2VF,
    )


@pto.jit(
    name="dequant_vmi_bf16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_bf16(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.f32, "gm"),
    y_gm: pto.ptr(pto.bf16, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="bf16",
        LOOP_NUM2VF=LOOP_NUM2VF,
    )


@pto.jit(
    name="dequant_vmi_f16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_f16(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.f32, "gm"),
    y_gm: pto.ptr(pto.f16, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="f16",
        LOOP_NUM2VF=LOOP_NUM2VF,
    )
