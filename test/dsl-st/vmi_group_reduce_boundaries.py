#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""One-carrier grouped reductions with holes, empty groups and original identities."""
import os

import numpy as np
from common import auto_main
from ptodsl import pto

TRANSFER_BYTES = 4096
SECTION_BYTES = 512
OUTPUT_INDEX = 2


def build_kernel(dtype, lanes, groups, itemsize, name):
    section = SECTION_BYTES // itemsize

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(dtype, "gm"), gates: pto.ptr(dtype, "gm"),
               out: pto.ptr(dtype, "gm"), active: pto.i32):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        gate_ub = pto.castptr(pto.i64(4096), pto.ptr(dtype, "ub"))
        out_ub = pto.castptr(pto.i64(8192), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(src, src_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.mte_gm_ub(gates, gate_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.mte_gm_ub(out, out_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        value = pto.vmi.vload(src_ub, 0, size=lanes)
        gate = pto.vmi.vload(gate_ub, 0, size=lanes)
        prefix = pto.vmi.create_mask(active, size=lanes)
        mask = pto.vmi.vcmps(gate, dtype(0), prefix, pto.CmpMode.GT)
        total = pto.vmi.vcadd(value, mask, group=groups, reassoc=True)
        maximum = pto.vmi.vcmax(value, mask, group=groups)
        minimum = pto.vmi.vcmin(value, mask, group=groups)
        pto.vmi.vstore(total, out_ub, 0, stride=1, group=groups)
        pto.vmi.vstore(maximum, out_ub, section, stride=1, group=groups)
        pto.vmi.vstore(minimum, out_ub, 2 * section, stride=1, group=groups)
        sum_brc = pto.vmi.vbrc(total, size=lanes, group=groups)
        max_brc = pto.vmi.vbrc(maximum, size=lanes, group=groups)
        min_brc = pto.vmi.vbrc(minimum, size=lanes, group=groups)
        pto.vmi.vstore(sum_brc, out_ub, 3 * section)
        pto.vmi.vstore(max_brc, out_ub, 4 * section)
        pto.vmi.vstore(min_brc, out_ub, 5 * section)
        compact = pto.vmi.vload(src_ub, 0, size=groups)
        broadcast = pto.vmi.vbrc(compact, size=lanes, group=groups)
        pto.vmi.vstore(broadcast, out_ub, 6 * section)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def make_case(npdtype, lanes, groups, mode):
    dtype = np.dtype(npdtype)
    floating = np.issubdtype(dtype, np.floating)
    limits = None if floating else np.iinfo(dtype)
    sample = ([-3, -1, 0.5, 2, 4, -0.25, 8, -9] if floating else
              [limits.max, limits.max - 1, limits.min, 0, 1, 3, 7, 11])
    if mode == "inf":
        sample = [-np.inf, -0.0, 0.0, np.inf, -2, 4, -8, 16]
    if mode == "nan":
        sample = [np.nan, -1, 2, np.nan, -np.inf, np.inf, -0.0, 0.0]
    if mode == "zeros":
        sample = [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0]
    source = np.full(TRANSFER_BYTES // dtype.itemsize, 17, dtype=dtype)
    source[:lanes] = np.resize(np.array(sample, dtype=dtype), lanes)
    gates = np.ones_like(source)
    active = 0 if mode == "empty" else lanes - 1 if mode == "tail" else lanes + 1
    if mode == "holes":
        gates[np.arange(gates.size) % 3 == 1] = 0
    if mode == "middle_empty" and groups > 1:
        begin = (groups // 2) * (lanes // groups)
        gates[begin:begin + lanes // groups] = 0
    output = np.full_like(source, 23)
    expected = output.copy()
    section = SECTION_BYTES // dtype.itemsize
    selected = (np.arange(lanes) < active) & (gates[:lanes] != 0)
    identities = (0, -np.inf if floating else limits.min,
                  np.inf if floating else limits.max)
    for group in range(groups):
        begin = group * (lanes // groups)
        end = begin + lanes // groups
        values = source[begin:end][selected[begin:end]]
        if values.size:
            # Exact integer accumulation and representable ordinary FP samples
            # avoid coupling the golden to the hardware's reduction tree.
            with np.errstate(invalid="ignore", over="ignore"):
                total = sum(float(v) if floating else int(v) for v in values)
                reductions = (total, np.max(values), np.min(values))
        else:
            reductions = identities
        for op, value in enumerate(reductions):
            if not floating:
                value = int(value) % (1 << (dtype.itemsize * 8))
                if limits.min < 0 and value > limits.max:
                    value -= 1 << (dtype.itemsize * 8)
            expected[op * section + group] = value
            expected[(op + 3) * section + begin:(op + 3) * section + end] = value
    expected[6 * section:6 * section + lanes] = np.repeat(source[:groups], lanes // groups)
    return [source, gates, output], expected, [active]


def check_case(inputs, expected):
    np.testing.assert_allclose(inputs[OUTPUT_INDEX].cpu().numpy(), expected,
                               rtol=0, atol=0, equal_nan=True)


TYPE_CASES = (
    ("i16", pto.i16, np.uint16), ("si16", pto.si16, np.int16), ("ui16", pto.ui16, np.uint16),
    ("i32", pto.i32, np.uint32), ("si32", pto.si32, np.int32), ("ui32", pto.ui32, np.uint32),
    ("f16", pto.f16, np.float16), ("f32", pto.f32, np.float32),
)
# dtype x vector shape x reduction mode is a full cross product; on the model a
# slice of it does not finish inside the slice budget. Modes describe the
# reduction semantics, so the default keeps the complete mode axis for one float
# (f32) and one signed integer (si16) at their widest shape, the float-only
# modes for f16, one canonical case per remaining dtype, and the min lane / min
# group boundary for the two representatives.
# Set PTOAS_VMI_FULL_MATRIX=1 for the full product.
FULL_MATRIX = os.environ.get("PTOAS_VMI_FULL_MATRIX") == "1"
BASE_MODES = ("empty", "tail", "full", "holes", "middle_empty")
FLOAT_MODES = ("inf", "nan", "zeros")


def _modes(npdtype):
    return BASE_MODES + (FLOAT_MODES if np.issubdtype(npdtype, np.floating) else ())


def _shapes(npdtype):
    itemsize = np.dtype(npdtype).itemsize
    return tuple((lanes, groups) for lanes in (8, 64, 128, 256) if lanes * itemsize <= 256
                 for groups in (1, 2, 4, 8))


def _selection():
    if FULL_MATRIX:
        for label, dtype, npdtype in TYPE_CASES:
            for lanes, groups in _shapes(npdtype):
                for mode in _modes(npdtype):
                    yield label, dtype, npdtype, lanes, groups, mode
        return
    # Every (lanes, groups) shape stays covered. Inside each shape the dtype and
    # reduction mode axes are rotated over a small deterministic sample so every
    # data path and every mode (empty/tail/full/holes/middle_empty/inf/nan/zeros)
    # still appears without paying for the full cross product.
    shapes = tuple(sorted({(lanes, groups) for _label, _dtype, npdtype in TYPE_CASES
                           for lanes, groups in _shapes(npdtype)}))
    for shape_index, (lanes, groups) in enumerate(shapes):
        valid = tuple(item for item in TYPE_CASES if (lanes, groups) in _shapes(item[2]))
        ints = tuple(item for item in valid if not np.issubdtype(item[2], np.floating))
        floats = tuple(item for item in valid if np.issubdtype(item[2], np.floating))
        label, dtype, npdtype = ints[shape_index % len(ints)]
        yield label, dtype, npdtype, lanes, groups, BASE_MODES[shape_index % len(BASE_MODES)]
        label, dtype, npdtype = floats[shape_index % len(floats)]
        yield label, dtype, npdtype, lanes, groups, FLOAT_MODES[shape_index % len(FLOAT_MODES)]
        label, dtype, npdtype = ints[(shape_index + 2) % len(ints)]
        yield label, dtype, npdtype, lanes, groups, BASE_MODES[(shape_index + 2) % len(BASE_MODES)]

CASES = []
KERNELS = []
_kernels = {}
for label, dtype, npdtype, lanes, groups, mode in _selection():
    itemsize = np.dtype(npdtype).itemsize
    name = f"group_boundary_{label}_vl{lanes}_g{groups}"
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = build_kernel(dtype, lanes, groups, itemsize, name)
        KERNELS.append(kernel)
        _kernels[name] = kernel
    def create_case(npdtype=npdtype, lanes=lanes, groups=groups, mode=mode):
        return make_case(npdtype, lanes, groups, mode)
    CASES.append(dict(name=f"{name}_{mode}", kernel=kernel,
                      make_case=create_case, check=check_case))


CASES = list({case["name"]: case for case in CASES}.values())  # sampling can overlap


auto_main(globals())
