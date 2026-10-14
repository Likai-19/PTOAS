#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolargmax ST test cases.

Each case now also carries explicit src/tmp fields so A5 tmp placeholders are
not conflated with src shape.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_1x256",
        "dtype": np.float32,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "f32_16x128",
        "dtype": np.float32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "f32_16x256",
        "dtype": np.float32,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "f16_1x256",
        "dtype": np.float16,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "f16_16x128",
        "dtype": np.float16,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "f16_16x256",
        "dtype": np.float16,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui32_1x256",
        "dtype": np.uint32,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui32_16x128",
        "dtype": np.uint32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui32_16x256",
        "dtype": np.uint32,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui16_1x256",
        "dtype": np.uint16,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui16_16x128",
        "dtype": np.uint16,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui16_16x256",
        "dtype": np.uint16,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui8_1x256",
        "dtype": np.uint8,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui8_16x128",
        "dtype": np.uint8,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "ui8_16x256",
        "dtype": np.uint8,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "i8_1x256",
        "dtype": np.int8,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "i8_16x128",
        "dtype": np.int8,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "i8_16x256",
        "dtype": np.int8,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    # Issue #1600: i16/i32 sources and ui32 index outputs.
    {
        "name": "i16_16x128",
        "dtype": np.int16,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "i32_16x128",
        "dtype": np.int32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.int32,
        "eps": 0,
    },
    {
        "name": "i32_16x128_u32",
        "dtype": np.int32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.uint32,
        "eps": 0,
    },
    {
        "name": "f32_16x128_u32",
        "dtype": np.float32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.uint32,
        "eps": 0,
    },
    {
        "name": "i16_16x128_u32",
        "dtype": np.int16,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.uint32,
        "eps": 0,
    },
    {
        "name": "i8_16x128_u32",
        "dtype": np.int8,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "dst_dtype": np.uint32,
        "eps": 0,
    },
]


def _a5_tmp_placeholder_shape(dtype):
    return (1, max(1, 32 // np.dtype(dtype).itemsize))


def _augment_case(case):
    case = dict(case)
    case.setdefault("src_shape", case["shape"])
    case.setdefault("src_valid_shape", case["valid_shape"])
    case.setdefault("tmp_shape", _a5_tmp_placeholder_shape(case["dtype"]))
    case.setdefault("tmp_valid_shape", (1, 1))
    return case


CASES = [_augment_case(case) for case in CASES]
