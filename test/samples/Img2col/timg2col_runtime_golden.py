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

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, write_buffers, write_golden


def _src_offset(n: int, c1: int, h: int, w: int, c0: int, *, c1_size: int, h_size: int, w_size: int, c0_size: int) -> int:
    return (((n * c1_size + c1) * h_size + h) * w_size + w) * c0_size + c0


def main():
    meta = load_case_meta()
    src_name, dst_name = meta.inputs
    generator = rng()
    src = np.asarray(float_values(generator, meta.elem_counts[src_name], style="signed"), dtype=np.float32)
    src = src.reshape(1, 1, 3, 4, 8)

    fmap_h = 3
    fmap_w = 4
    c0_size = 8
    filter_h = 2
    filter_w = 2
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    pad_left = 1
    pad_right = 0
    pad_top = 1
    pad_bottom = 0
    pos_m = 1
    pos_k = 8
    valid_rows = 9
    valid_cols = 32
    pad_value = np.float32(0.0)

    out_h = (fmap_h + pad_top + pad_bottom - dilation_h * (filter_h - 1) - 1) // stride_h + 1
    out_w = (fmap_w + pad_left + pad_right - dilation_w * (filter_w - 1) - 1) // stride_w + 1
    assert pos_m + valid_rows <= out_h * out_w
    golden = np.zeros((16, 32), dtype=np.float32)

    for row in range(valid_rows):
        m = pos_m + row
        out_row = m // out_w
        out_col = m % out_w
        for col in range(valid_cols):
            k = pos_k + col
            c1 = k // (c0_size * filter_h * filter_w)
            kernel_offset = (k % (c0_size * filter_h * filter_w)) // c0_size
            c0 = k % c0_size
            kernel_h = kernel_offset // filter_w
            kernel_w = kernel_offset % filter_w
            input_h = out_row * stride_h + kernel_h * dilation_h - pad_top
            input_w = out_col * stride_w + kernel_w * dilation_w - pad_left
            value = pad_value
            if 0 <= input_h < fmap_h and 0 <= input_w < fmap_w and c1 < 1:
                value = src[
                    _src_offset(
                        0,
                        c1,
                        input_h,
                        input_w,
                        c0,
                        c1_size=1,
                        h_size=fmap_h,
                        w_size=fmap_w,
                        c0_size=c0_size,
                    )
                ]
            golden[row, col] = value

    buffers = default_buffers(meta)
    buffers[src_name] = src.reshape(-1)
    write_buffers(meta, buffers)
    write_golden(meta, {dst_name: golden.reshape(-1)})


if __name__ == "__main__":
    main()
