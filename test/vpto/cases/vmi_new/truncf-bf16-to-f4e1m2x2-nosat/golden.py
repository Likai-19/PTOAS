#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# bf16 -> f4E1M2x2 golden for the "NOSAT" case.
#
# FP4E1M2 nibble -> bf16 pattern (mirror of tcvt/gen_data.py::FP4E1M2_TO_BF16):
#   nibble 0x0..0x7 =
#     { +0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5, +1.75 }
#   nibble 0x8..0xF =
#     { -0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75 }
#
# Every input bf16 pattern we feed is one of these 16 exact reprs, so the
# NOSAT vs. SAT semantics do NOT matter for correctness -- no input can
# overflow.  This isolates the effect being tested: the "saturate=NOSAT"
# attribute must reach the physical vcvt intact, and produce the same
# per-lane quantization as the SAT/default baseline.
#
# 256 bf16 lanes -> 128 packed FP4 bytes (two lanes per byte, low nibble
# is the even lane).

import argparse
from pathlib import Path

import numpy as np

BF16_ELEMS = 256
DST_BYTES = 128

# FP4E1M2 nibble -> bf16 bit pattern (identical to FP4E1M2_TO_BF16 in the
# TileLang reference generator).
FP4E1M2_TO_BF16 = np.array(
    [
        0x0000, 0x3E80, 0x3F00, 0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0,
        0x8000, 0xBE80, 0xBF00, 0xBF40, 0xBF80, 0xBFA0, 0xBFC0, 0xBFE0,
    ],
    dtype=np.uint16,
)


def generate(output_dir: Path) -> None:
    # Walk all 16 nibbles in order and tile them to fill 256 bf16 lanes.
    nib_seq = np.arange(16, dtype=np.uint8)
    bf16_seq = FP4E1M2_TO_BF16.copy()

    src = np.resize(bf16_seq, BF16_ELEMS).astype(np.uint16)
    nibbles = np.resize(nib_seq, BF16_ELEMS).astype(np.uint8)

    # Pack two consecutive nibbles into one byte, low nibble = even lane.
    golden = np.empty(DST_BYTES, dtype=np.uint8)
    for i in range(DST_BYTES):
        golden[i] = (nibbles[2 * i] & 0x0F) | ((nibbles[2 * i + 1] & 0x0F) << 4)

    dst = np.full(DST_BYTES, 0xA5, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()