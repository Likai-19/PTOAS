#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vcvt-f32-to-f8-rahz.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 256

F8E4M3_VALUES = np.array(
    [1.0625, 1.1875, -1.0625, -1.1875, 1000.0, -1000.0, np.nan, 0.0],
    dtype=np.float32,
)
F8E5M2_VALUES = np.array(
    [1.125, 1.375, -1.125, -1.375, 1.0e10, -1.0e10, np.nan, 0.0],
    dtype=np.float32,
)

CASE_SPECS = [
    {
        "name": "vmi_vcvt_f32_to_f8e4m3_r",
        "dst_dtype": pto.f8e4m3,
        "rounding": pto.VcvtRoundMode.R,
        "values": F8E4M3_VALUES,
        "golden": np.array([0x38, 0x3A, 0xB8, 0xBA, 0x7E, 0xFE, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e4m3_a",
        "dst_dtype": pto.f8e4m3,
        "rounding": pto.VcvtRoundMode.A,
        "values": F8E4M3_VALUES,
        "golden": np.array([0x39, 0x3A, 0xB9, 0xBA, 0x78, 0xF8, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e4m3_h",
        "dst_dtype": pto.f8e4m3,
        "rounding": pto.VcvtRoundMode.H,
        "values": F8E4M3_VALUES,
        "golden": np.array([0x38, 0x39, 0xB8, 0xB9, 0x7E, 0xFE, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e4m3_z",
        "dst_dtype": pto.f8e4m3,
        "rounding": pto.VcvtRoundMode.Z,
        "values": F8E4M3_VALUES,
        "golden": np.array([0x38, 0x39, 0xB8, 0xB9, 0x7E, 0xFE, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e5m2_r",
        "dst_dtype": pto.f8e5m2,
        "rounding": pto.VcvtRoundMode.R,
        "values": F8E5M2_VALUES,
        "golden": np.array([0x3C, 0x3E, 0xBC, 0xBE, 0x7B, 0xFB, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e5m2_a",
        "dst_dtype": pto.f8e5m2,
        "rounding": pto.VcvtRoundMode.A,
        "values": F8E5M2_VALUES,
        "golden": np.array([0x3D, 0x3E, 0xBD, 0xBE, 0x7D, 0xFD, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e5m2_h",
        "dst_dtype": pto.f8e5m2,
        "rounding": pto.VcvtRoundMode.H,
        "values": F8E5M2_VALUES,
        "golden": np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7C, 0xFC, 0x00, 0x00], dtype=np.uint8),
    },
    {
        "name": "vmi_vcvt_f32_to_f8e5m2_z",
        "dst_dtype": pto.f8e5m2,
        "rounding": pto.VcvtRoundMode.Z,
        "values": F8E5M2_VALUES,
        "golden": np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7C, 0xFC, 0x00, 0x00], dtype=np.uint8),
    },
]


def _repeat_pattern(pattern: np.ndarray) -> np.ndarray:
    repeats = (ELEMS + len(pattern) - 1) // len(pattern)
    return np.tile(pattern, repeats)[:ELEMS].copy()


def _build_kernel(case):
    dst_dtype = case["dst_dtype"]
    rounding = case["rounding"]
    kernel_name = f"{case['name']}_kernel"

    @pto.jit(
        name=kernel_name,
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.ui8, "gm")):
        ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        ub_dst_u8 = pto.castptr(pto.const(2048, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
        ub_dst_f8 = pto.castptr(pto.const(2048, dtype=pto.i64), pto.ptr(dst_dtype, "ub"))

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(src, ub_src, 0, 256, nburst=(4, 256, 256))
        pto.rls_buf(pto.Pipe.MTE2, 0)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

        offset = pto.const(0, dtype=pto.index)
        mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
        wide = pto.vmi.vload(ub_src, offset, size=256)
        packed = pto.vmi.vcvt(
            wide,
            dst_dtype,
            rounding=rounding,
            saturate=pto.VcvtSatMode.SAT,
        )
        pto.vmi.vstore(packed, ub_dst_f8, offset, mask)
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_dst_u8, dst, 256, nburst=(1, 256, 256))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _make_inputs(case):
    src = _repeat_pattern(case["values"]).astype(np.float32)
    return [src]


CASES = []
KERNELS = []
for case in CASE_SPECS:
    kernel = _build_kernel(case)
    KERNELS.append(kernel)
    CASES.append(
        golden_output_case(
            case["name"],
            kernel,
            inputs=lambda case=case: _make_inputs(case),
            expected=_repeat_pattern(case["golden"]),
            output_shape=(ELEMS,),
            output_dtype=np.uint8,
            rtol=0.0,
            atol=0.0,
        )
    )


auto_main(globals())
