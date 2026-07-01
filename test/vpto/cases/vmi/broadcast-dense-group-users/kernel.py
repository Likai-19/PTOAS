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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from broadcast-dense-group-users kernel.py")


_bootstrap_dsl_st_common()

from common import auto_main
from ptodsl import pto


ROWS = 8
COLS = 32
SCALE = np.float32(0.5)


@pto.jit(
    name="vmi_broadcast_dense_group_users_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source="kernel.pto",
)
def vmi_broadcast_dense_group_users_kernel(
    src_gm: pto.ptr(pto.f32, "gm"),
    copy_gm: pto.ptr(pto.f32, "gm"),
    sum_gm: pto.ptr(pto.f32, "gm"),
):
    pass


def make_inputs():
    base = np.linspace(-0.875, 0.625, COLS, dtype=np.float32)
    src = np.empty((ROWS, COLS), dtype=np.float32)
    for row in range(ROWS):
        src[row, :] = base + np.float32(row) * np.float32(0.03125)
    copy = np.zeros((ROWS, COLS), dtype=np.float32)
    sums = np.zeros(ROWS, dtype=np.float32)
    return [src.reshape(-1), copy.reshape(-1), sums]


def make_case():
    host_inputs = make_inputs()
    src = host_inputs[0].reshape(ROWS, COLS)
    golden_copy = (src + SCALE).astype(np.float32).reshape(-1)
    golden_sum = np.sum(src * SCALE, axis=1, dtype=np.float32).astype(np.float32)
    return host_inputs, (golden_copy, golden_sum)


def check_case(device_inputs, golden):
    golden_copy, golden_sum = golden
    actual_copy = device_inputs[1].cpu().numpy()
    actual_sum = device_inputs[2].cpu().numpy()
    np.testing.assert_allclose(actual_copy, golden_copy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_sum, golden_sum, rtol=1e-4, atol=1e-4)


CASES = [
    {
        "name": "vmi_broadcast_dense_group_users",
        "kernel": vmi_broadcast_dense_group_users_kernel,
        "make_case": make_case,
        "check": check_case,
    },
]


auto_main(globals())
