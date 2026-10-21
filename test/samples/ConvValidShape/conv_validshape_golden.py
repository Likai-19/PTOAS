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

from validation_runtime import load_case_meta, rng, write_buffers, write_golden

# Mirrors the geometry encoded in conv_validshape-pto.pto: identical to
# ConvPipeline except the weight FRACTAL_Z tile declares N capacity 32
# (N1=2) with valid_shape [2,1,16,16] -- only the first 16 output
# channels hold real weights, the tail block is zeros in GM.
FMAP_H = 3
FMAP_W = 4
C1 = 1
C0 = 16
FILTER_H = 2
FILTER_W = 2
STRIDE = 1
DILATION = 1
PAD = 0
VALID_M = 6
VALID_K = 32
DECLARED_N = 32
VALID_N = 16
POS_M = 0
POS_K = 0


def im2col(fmap):
    """Expand NC1HWC0 fmap into the [M, K] LEFT operand timg2col produces."""
    out_w = (FMAP_W + 2 * PAD - DILATION * (FILTER_W - 1) - 1) // STRIDE + 1
    left = np.zeros((VALID_M, VALID_K), dtype=np.float32)
    for row in range(VALID_M):
        out_row, out_col = divmod(POS_M + row, out_w)
        for col in range(VALID_K):
            k = POS_K + col
            c1, rest = divmod(k, C0 * FILTER_H * FILTER_W)
            kernel_offset, c0 = divmod(rest, C0)
            kh, kw = divmod(kernel_offset, FILTER_W)
            ih = out_row * STRIDE + kh * DILATION - PAD
            iw = out_col * STRIDE + kw * DILATION - PAD
            if c1 < C1 and 0 <= ih < FMAP_H and 0 <= iw < FMAP_W:
                left[row, col] = fmap[0, c1, ih, iw, c0]
    return left


def fractal_z_to_kn(weight):
    """Flatten FRACTAL_Z [C1HW, N1, 16, C0] into the [K, N] RIGHT operand."""
    k1, n1, n0, c0 = weight.shape
    right = np.zeros((k1 * c0, n1 * n0), dtype=np.float32)
    for kk in range(k1):
        for nn in range(n1):
            for row in range(n0):
                for col in range(c0):
                    right[kk * c0 + col, nn * n0 + row] = weight[kk, nn, row, col]
    return right


def main():
    meta = load_case_meta()
    fmap_name, weight_name = meta.inputs
    out_name = meta.outputs[0]
    generator = rng()

    # Small signed integers keep the f16 inputs and the f32 accumulator exact,
    # so the comparison below is bit-for-bit rather than tolerance-based.
    fmap = generator.integers(-4, 5, size=meta.elem_counts[fmap_name])
    fmap = fmap.astype(np.float16).reshape(1, C1, FMAP_H, FMAP_W, C0)
    # Random weights only in the valid N1 block; the declared tail block
    # (output channels VALID_N..DECLARED_N-1) stays zero, matching the
    # tile's valid_shape declaration.
    weight = np.zeros((VALID_K // C0, DECLARED_N // VALID_N, VALID_N, C0), dtype=np.float16)
    weight[:, 0, :, :] = generator.integers(-4, 5, size=(VALID_K // C0, VALID_N, C0)).astype(np.float16)

    write_buffers(
        meta,
        {
            fmap_name: fmap,
            weight_name: weight.reshape(-1),
            out_name: np.zeros(meta.elem_counts[out_name], dtype=np.float32),
        },
    )

    left = im2col(fmap.astype(np.float32))
    right = fractal_z_to_kn(weight.astype(np.float32))
    assert right.shape == (VALID_K, DECLARED_N)
    assert np.count_nonzero(right[:, VALID_N:]) == 0
    write_golden(meta, {out_name: (left @ right).astype(np.float32).reshape(-1)})
    print(f"[INFO] conv valid-shape golden {left.shape} @ {right.shape} -> ({VALID_M}, {DECLARED_N})")


if __name__ == "__main__":
    main()
