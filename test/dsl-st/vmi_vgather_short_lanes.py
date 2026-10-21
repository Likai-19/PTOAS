#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""B32 vgather at lane counts that partially fill the single physical chunk.

Issue #1638: 32-bit gathers with VL in {1, 2, 4, 8} were rejected by the
VMIToVPTO full-chunk gate even though a 32-bit physical chunk is 64 lanes and
padding lanes stay masked off. They now lower through one pto.vgather2_bc over
the single physical chunk, matching the 16-bit surface. VL=64 is the full
single chunk and acts as the boundary. Prefix masks exercise the passthrough
vsel path; full masks take the static all-active path without a vsel. The
output buffer is preloaded with a canary, so a gather that reads the wrong
lanes or a store that writes padding lanes shows up as a canary mismatch.
"""

import numpy as np
from common import assert_close, auto_main
from ptodsl import pto

SRC_ELEMS = 256
IDX_ELEMS = 256
OUT_ELEMS = 256
CANARY = 211

LANES = (1, 2, 4, 8, 64)
DTYPES = (
    (pto.f32, np.float32),
    (pto.i32, np.int32),
    (pto.ui32, np.uint32),
)
MODES = ("full", "prefix")


def build(n, dtype, runtime_active, name):
    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(dtype, "gm"), indices: pto.ptr(pto.i32, "gm"),
               output: pto.ptr(dtype, "gm"), active: pto.i32):
        ub_src = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        ub_idx = pto.castptr(pto.i64(2048), pto.ptr(pto.i32, "ub"))
        ub_out = pto.castptr(pto.i64(4096), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(src, ub_src, 0, SRC_ELEMS * 4,
                      nburst=(1, SRC_ELEMS * 4, SRC_ELEMS * 4))
        pto.mte_gm_ub(indices, ub_idx, 0, IDX_ELEMS * 4,
                      nburst=(1, IDX_ELEMS * 4, IDX_ELEMS * 4))
        pto.mte_gm_ub(output, ub_out, 0, OUT_ELEMS * 4,
                      nburst=(1, OUT_ELEMS * 4, OUT_ELEMS * 4))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        offsets = pto.vmi.vload(ub_idx, 0, size=n)
        if runtime_active:
            mask = pto.vmi.create_mask(active, size=n)
        else:
            mask = pto.vmi.create_mask(n, size=n)
        gathered = pto.vmi.vgather(ub_src, offsets, mask, pmode="zero")
        pto.vmi.vstore(gathered, ub_out, 0)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_out, output, OUT_ELEMS * 4,
                      nburst=(1, OUT_ELEMS * 4, OUT_ELEMS * 4))
        pto.pipe_barrier(pto.Pipe.ALL)
    return kernel


def check_case(inputs, expected):
    assert_close(inputs[-1].cpu().numpy(), expected, rtol=0.0, atol=0.0)


CASES = []
for n in LANES:
    for dtype, np_dtype in DTYPES:
        for mode in MODES:
            # A one-lane prefix keeps at least one active lane for VL=1.
            active = n - 1 if n > 1 else 1
            runtime_active = mode == "prefix"
            name = f"vmi_vgather_{np.dtype(np_dtype).name}_vl{n}_{mode}"
            source = ((np.arange(SRC_ELEMS, dtype=np.int64) * 53 + 137)
                      % (2 ** 31 - 1)).astype(np_dtype)
            index_pool = (np.arange(IDX_ELEMS, dtype=np.int64) * 37 + 11) % SRC_ELEMS
            indices = index_pool.astype(np.int32)
            output = np.full(OUT_ELEMS, CANARY, dtype=np_dtype)
            expected = output.copy()
            active_count = n if mode == "full" else active
            selected = np.arange(n) < active_count
            if active_count < n:
                # Masked-off lanes carry an offset far outside the 1 KB source
                # window; the micro-ISA exempts masked-off lanes from address
                # overflow, so the run must not fault and the expected result
                # stays the pmode=zero passthru.
                indices = indices.copy()
                indices[active_count:n] = 0x3FFF0000
            gathered = np.zeros(n, dtype=np_dtype)
            gathered[selected] = source[indices[:n][selected]]
            expected[:n] = gathered

            def make_case(source=source, indices=indices, output=output,
                          expected=expected, active=active):
                return ([source.copy(), indices.copy(), output.copy()],
                        expected.copy(), [active])

            CASES.append(dict(name=name,
                              kernel=build(n, dtype, runtime_active, name),
                              make_case=make_case, check=check_case))


auto_main(globals())
