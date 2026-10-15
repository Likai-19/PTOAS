#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowexpanddiv ST test cases.

trowexpanddiv: dst = src0 / broadcast(src1) across columns.
- src1Col determines how src1 is broadcast:
  - src1Col=32/sizeof(dtype): the 32-byte block repeats across the row (pto-isa TROWEXPAND Mode 2)
  - src1Col>1: RowMajor 32-byte block per row (src1Col == 32/sizeof(dtype)) *repeats* across
    the row (dst[:, c] = src0[:, c] / src1[:, c % src1Col])
- src1 physical cols = 32/sizeof(dtype) for NPU alignment
- highPrecision: use high precision mode for computation
"""

import numpy as np

CASES = [
    # launchTRowExpandDiv<aclFloat16, 16, 32, 16, 1, true, false>
    {
        "name": "f16_16x32",
        "dtype": np.float16,
        "src0_shape": (16, 32),
        "src0_valid_shape": (16, 32),
        "src1_shape": (16, 16),       # physical: 32/sizeof(f16)=16
        "src1_valid_shape": (16, 16),
        "dst_shape": (16, 32),
        "dst_valid_shape": (16, 32),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 40, 64, 40, 1, true, false>
    {
        "name": "f32_40x64",
        "dtype": np.float32,
        "src0_shape": (40, 64),       # src0eqdst=true
        "src0_valid_shape": (40, 64),
        "src1_shape": (40, 8),        # physical: 32/sizeof(f32)=8
        "src1_valid_shape": (40, 8),  # src1Col = 8
        "dst_shape": (40, 64),
        "dst_valid_shape": (40, 64),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 16, 256, 16, 1, true, false>
    {
        "name": "f32_16x256",
        "dtype": np.float32,
        "src0_shape": (16, 256),
        "src0_valid_shape": (16, 256),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 8),
        "dst_shape": (16, 256),
        "dst_valid_shape": (16, 256),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<aclFloat16, 32, 512, 32, 1, true, false>
    {
        "name": "f16_32x512",
        "dtype": np.float16,
        "src0_shape": (32, 512),
        "src0_valid_shape": (32, 512),
        "src1_shape": (32, 16),
        "src1_valid_shape": (32, 16),
        "dst_shape": (32, 512),
        "dst_valid_shape": (32, 512),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 16, 128, 16, 1, false, false>
    {
        "name": "f32_16x128_noeq",
        "dtype": np.float32,
        "src0_shape": (16, 128),      # src0eqdst=false
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 8),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<aclFloat16, 32, 64, 32, 1, false, false>
    {
        "name": "f16_32x64_noeq",
        "dtype": np.float16,
        "src0_shape": (32, 64),
        "src0_valid_shape": (32, 64),
        "src1_shape": (32, 16),
        "src1_valid_shape": (32, 16),
        "dst_shape": (32, 64),
        "dst_valid_shape": (32, 64),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 40, 32, 40, 1, true, true>
    {
        "name": "f32_40x32_hp",
        "dtype": np.float32,
        "src0_shape": (40, 32),
        "src0_valid_shape": (40, 32),
        "src1_shape": (40, 8),
        "src1_valid_shape": (40, 8),
        "dst_shape": (40, 32),
        "dst_valid_shape": (40, 32),
        "eps": 1e-6,
        "high_precision": True,
    },
    # launchTRowExpandDiv<aclFloat16, 16, 128, 16, 1, true, true>
    {
        "name": "f16_16x128_hp",
        "dtype": np.float16,
        "src0_shape": (16, 128),
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 16),
        "src1_valid_shape": (16, 16),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-3,
        "high_precision": True,
    },
]

_SMOKE_CASE_NAMES = ['f16_16x32', 'f32_40x32_hp']
_SMOKE_CASE_NAME_SET = set(_SMOKE_CASE_NAMES)
_missing = [name for name in _SMOKE_CASE_NAMES if name not in {case["name"] for case in CASES}]
if _missing:
    raise RuntimeError("unknown smoke case(s): " + ", ".join(_missing))
CASES = [case for case in CASES if case["name"] in _SMOKE_CASE_NAME_SET]
