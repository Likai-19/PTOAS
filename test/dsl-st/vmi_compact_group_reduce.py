#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small grouped reductions/broadcasts with dynamic masks and output canaries."""

import os

import numpy as np
from common import auto_main, assert_close
from ptodsl import pto

TRANSFER_BYTES = 256
BLOCK_BYTES = 32


def build_kernel(dtype, lanes, groups, itemsize, name):
    block = BLOCK_BYTES // itemsize

    def kernel(src: pto.ptr(dtype, "gm"), out: pto.ptr(dtype, "gm"),
               active: pto.i32):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        out_ub = pto.castptr(pto.i64(4096), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(src, src_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.mte_gm_ub(out, out_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        value = pto.vmi.vload(src_ub, 0, size=lanes)
        mask = pto.vmi.create_mask(active, size=lanes)
        total = pto.vmi.vcadd(value, mask, group=groups, reassoc=True)
        maximum = pto.vmi.vcmax(value, mask, group=groups)
        minimum = pto.vmi.vcmin(value, mask, group=groups)
        sum_brc = pto.vmi.vbrc(total, size=lanes, group=groups)
        max_brc = pto.vmi.vbrc(maximum, size=lanes, group=groups)
        min_brc = pto.vmi.vbrc(minimum, size=lanes, group=groups)
        pto.vmi.vstore(sum_brc, out_ub, 0)
        pto.vmi.vstore(max_brc, out_ub, block)
        pto.vmi.vstore(min_brc, out_ub, 2 * block)
        compact = pto.vmi.vload(src_ub, 0, size=groups)
        broadcast = pto.vmi.vbrc(compact, size=lanes, group=groups)
        pto.vmi.vstore(broadcast, out_ub, 3 * block)
        pto.vmi.vstore(total, out_ub, 4 * block, stride=1, group=groups)
        pto.vmi.vstore(maximum, out_ub, 5 * block, stride=1, group=groups)
        pto.vmi.vstore(minimum, out_ub, 6 * block, stride=1, group=groups)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.pipe_barrier(pto.Pipe.ALL)

    kernel.__name__ = name
    return pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
                   kernel_kind="vector", insert_sync=False)(kernel)


def source_values(dtype, edges):
    if np.issubdtype(dtype, np.floating):
        values = ([-np.inf, -0.0, 0.0, np.inf, -2.0, 4.0, -8.0, 16.0]
                  if edges else [-3.0, -1.0, 0.5, 2.0, 4.0, -0.25, 8.0, -9.0])
    else:
        limits = np.iinfo(dtype)
        values = [limits.max, limits.max - 1, limits.min, 0, 1, 3, 7, 11]
    source = np.full(TRANSFER_BYTES // np.dtype(dtype).itemsize, 17, dtype=dtype)
    source[:8] = values
    return source


def reference(source, output, lanes, groups, active):
    result = output.copy()
    block = BLOCK_BYTES // source.dtype.itemsize
    group_size = lanes // groups
    floating = np.issubdtype(source.dtype, np.floating)
    limits = None if floating else np.iinfo(source.dtype)
    identities = (0, -np.inf if floating else limits.min,
                  np.inf if floating else limits.max)
    for group in range(groups):
        begin = group * group_size
        end = min((group + 1) * group_size, active)
        values = source[begin:max(begin, end)]
        if values.size:
            total = sum(float(x) if floating else int(x) for x in values)
            reductions = (total, max(values), min(values))
        else:
            reductions = identities
        for op, value in enumerate(reductions):
            if not floating:
                value = int(value) % (1 << (source.dtype.itemsize * 8))
                if limits.min < 0 and value > limits.max:
                    value -= 1 << (source.dtype.itemsize * 8)
            result[op * block + begin:op * block + begin + group_size] = value
            result[(4 + op) * block + group] = value
    result[3 * block:3 * block + lanes] = np.repeat(source[:groups], group_size)
    return result


def make_case(source, lanes, groups, active):
    output = np.full_like(source, 23)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = reference(source, output, lanes, groups, active)
    return [source.copy(), output], expected, [active]


def check_case(inputs, expected):
    assert_close(inputs[1].cpu().numpy(), expected, rtol=0.0, atol=0.0)


TYPE_CASES = (
    ("i16", pto.i16, np.uint16), ("si16", pto.si16, np.int16), ("ui16", pto.ui16, np.uint16),
    ("i32", pto.i32, np.uint32), ("si32", pto.si32, np.int32), ("ui32", pto.ui32, np.uint32),
    ("f16", pto.f16, np.float16), ("f32", pto.f32, np.float32),
)
# The full cross product is dtype x shape x edges x active. Dtype selects the
# data path, shape selects the layout and active selects the mask boundary. The
# default suite is a stratified sample: the complete mask boundary axis
# (0/1/tail/full/over) and both edge states for one float (f32) and one signed
# integer (si16) at the canonical shape, one canonical case per remaining dtype,
# and the min/max lane and group boundaries for those two representatives.
# Set PTOAS_VMI_FULL_MATRIX=1 for the full product.
FULL_MATRIX = os.environ.get("PTOAS_VMI_FULL_MATRIX") == "1"
SHAPE_AXIS = tuple((lanes, groups) for lanes in (1, 2, 4, 8)
                   for groups in (1, 2, 4, 8) if groups <= lanes)
CANONICAL_SHAPE = (8, 8)


def _selection():
    if FULL_MATRIX:
        for label, dtype, npdtype in TYPE_CASES:
            edges = (False, True) if np.issubdtype(npdtype, np.floating) else (False,)
            for lanes, groups in SHAPE_AXIS:
                for edge in edges:
                    for active in sorted({0, 1, lanes - 1, lanes, lanes + 1}):
                        yield label, dtype, npdtype, lanes, groups, edge, active
        return
    # Every (lanes, groups) shape stays covered. Inside each shape the dtype,
    # edge and mask axes are rotated over a small deterministic sample so every
    # data path, both edge states and every mask boundary (empty 0 / one / tail
    # / full / over) still appears without paying for the full cross product.
    int_dtypes = tuple(item for item in TYPE_CASES if not np.issubdtype(item[2], np.floating))
    float_dtypes = tuple(item for item in TYPE_CASES if np.issubdtype(item[2], np.floating))
    for shape_index, (lanes, groups) in enumerate(SHAPE_AXIS):
        # empty (0) and full (lanes) stay on every shape; the third slot cycles
        # the remaining mask boundaries (one / over / tail) across the shapes.
        label, dtype, npdtype = int_dtypes[shape_index % len(int_dtypes)]
        yield label, dtype, npdtype, lanes, groups, False, 0
        label, dtype, npdtype = float_dtypes[shape_index % len(float_dtypes)]
        yield label, dtype, npdtype, lanes, groups, True, lanes
        label, dtype, npdtype = int_dtypes[(shape_index + 3) % len(int_dtypes)]
        third = (1, lanes + 1, max(1, lanes - 1))[shape_index % 3]
        yield label, dtype, npdtype, lanes, groups, False, third

CASES = []
KERNELS = []
_kernels = {}
for label, dtype, npdtype, lanes, groups, edge, active in _selection():
    name = f"compact_group_{label}_vl{lanes}_g{groups}"
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = build_kernel(dtype, lanes, groups, np.dtype(npdtype).itemsize, name)
        KERNELS.append(kernel)
        _kernels[name] = kernel
    source = source_values(npdtype, edge)
    def create_case(source=source, lanes=lanes, groups=groups, active=active):
        return make_case(source, lanes, groups, active)
    CASES.append(dict(name=f"{name}_active{active}_edges{int(edge)}",
                      kernel=kernel, make_case=create_case, check=check_case))


CASES = list({case["name"]: case for case in CASES}.values())  # sampling can overlap


auto_main(globals())
