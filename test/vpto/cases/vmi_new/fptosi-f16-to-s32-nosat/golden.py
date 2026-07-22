#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# f16 -> si32 (FpToSi) WITHOUT saturate golden.
#
# requiresSat=false for f16->s32: f16 max (65504) fits comfortably in si32
# (+-2.1e9), so no overflow is possible and the kernel omits the saturate attr.
# The golden therefore is plain round-to-nearest-even -- NO clamping.
#
# IMPORTANT: without SAT, out-of-range/NaN inputs are architecturally
# undefined, so the probe deliberately contains ONLY finite f16-representable
# values. Golden is computed on the exact f16 value the kernel consumes.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256


def f16(x):
    return np.float16(x)


# A 32-value probe of finite f16 values only (no inf/nan -- undefined without
# saturate). ELEMS = 256 = 8 * 32, giving a clean 8x-repeat schedule.
PROBE = [
    f16(0.0),          # 0
    f16(0.5),          # RN -> 0
    f16(1.5),          # RN -> 2
    f16(2.5),          # RN -> 2
    f16(-0.5),         # RN -> 0
    f16(-1.5),         # RN -> -2
    f16(-2.5),         # RN -> -2
    f16(1.0),          # 1
    f16(-1.0),         # -1
    f16(127.0),        # 127
    f16(-128.0),       # -128
    f16(32767.0),      # f16 stores 32768 (rounds up), in range
    f16(-32768.0),     # -32768
    f16(65504.0),      # f16 max, fits si32
    f16(-65504.0),     # -65504
    f16(60000.0),      # 60000
    f16(-60000.0),     # -60000
    f16(0.25),         # RN -> 0
    f16(0.75),         # RN -> 1
    f16(-0.25),        # RN -> 0
    f16(-0.75),        # RN -> -1
    f16(1234.5),       # RN -> 1234 (banker's: 1234.5 -> 1234)
    f16(-1234.5),      # RN -> -1234
    f16(2.0),          # 2
    f16(-2.0),         # -2
    f16(100.5),        # RN -> 100 (banker's)
    f16(-100.5),       # RN -> -100
    f16(16384.0),      # 16384
    f16(-16384.0),     # -16384
    f16(3.0),          # 3
    f16(-3.0),         # -3
    f16(5.5),          # RN -> 6
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def rn_to_int(x):
    # np.rint uses banker's rounding, matching RN on integer boundaries.
    return int(np.rint(x))


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float16)
    src = np.tile(probe.view(np.uint16), ELEMS // len(PROBE))[:ELEMS].astype(np.uint16)
    src_f = src.view(np.float16).astype(np.float32)
    # No clamp: f16 range always fits si32.
    golden = np.array([rn_to_int(v) for v in src_f], dtype=np.int32)

    dst = np.full(ELEMS, -559038737, dtype=np.int32)  # 0xDEADBEEF sentinel

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
