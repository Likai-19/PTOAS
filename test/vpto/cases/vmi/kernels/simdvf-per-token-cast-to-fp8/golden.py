#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
GROUPS = 2
GROUP_SIZE = ELEMS // GROUPS
FP8_MAX = np.float32(448.0)
SCALES = np.array([0.25, 0.5], dtype=np.float32)
SENTINEL_F32 = np.float32(-777.0)
SENTINEL_U8 = np.uint8(0xA5)

Q_VALUES = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0, 448.0], dtype=np.float32)
F8E4M3FN_BYTES = np.array([0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8, 0x7E], dtype=np.uint8)


def generate(output_dir: Path) -> None:
    repeats = (GROUP_SIZE + len(Q_VALUES) - 1) // len(Q_VALUES)
    q_group = np.tile(Q_VALUES, repeats)[:GROUP_SIZE].astype(np.float32)
    q = np.concatenate([q_group, q_group]).astype(np.float32)
    src = np.empty(ELEMS, dtype=np.float32)
    golden_scale = np.full(ELEMS, SENTINEL_F32, dtype=np.float32)
    for group in range(GROUPS):
        begin = group * GROUP_SIZE
        end = begin + GROUP_SIZE
        src[begin:end] = (q_group * SCALES[group]).astype(np.float32)
        amax = np.max(np.abs(src[begin:end])).astype(np.float32)
        scale = np.maximum(amax, np.float32(1.0e-4)) / FP8_MAX
        golden_scale[group * 8] = scale
    golden_out8_group = np.tile(F8E4M3FN_BYTES, repeats)[:GROUP_SIZE].astype(np.uint8)
    golden_out8 = np.concatenate([golden_out8_group, golden_out8_group]).astype(np.uint8)

    scale_out = np.full(ELEMS, SENTINEL_F32, dtype=np.float32)
    out8 = np.full(ELEMS, SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    scale_out.tofile(output_dir / "v2.bin")
    out8.tofile(output_dir / "v3.bin")
    golden_scale.tofile(output_dir / "golden_v2.bin")
    golden_out8.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
