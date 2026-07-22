#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# f16 -> si8 (FpToSi) with saturate = "SAT" golden.
#
# V300 SAT policy for FpToSi (rounding = "R", i.e. RN):
#   * finite in-range values        -> round-to-nearest-even to int8
#   * finite out-of-range positive  -> INT8_MAX (127)
#   * finite out-of-range negative  -> INT8_MIN (-128)
#   * +inf / -inf                   -> INT8_MAX / INT8_MIN
#   * NaN                           -> 0
#
# f16 range is [-65504, 65504], so almost everything above 127 clamps. Golden
# is computed on the exact f16 value the kernel consumes.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256
INT8_MAX = 127
INT8_MIN = -128


def f16(x):
    return np.float16(x)


# A 32-value probe. ELEMS = 256 = 8 * 32, giving a clean 8x-repeat schedule.
PROBE = [
    # --- in-range RN edges + typical values ---
    f16(0.0),          # 0
    f16(0.5),          # RN -> 0
    f16(1.5),          # RN -> 2
    f16(2.5),          # RN -> 2
    f16(-0.5),         # RN -> 0
    f16(-1.5),         # RN -> -2
    f16(-2.5),         # RN -> -2
    f16(1.0),          # 1
    f16(-1.0),         # -1
    f16(100.0),        # 100
    f16(-100.0),       # -100

    # --- boundary clamping ---
    f16(127.0),        # exactly INT8_MAX
    f16(128.0),        # clamp 127
    f16(130.0),        # clamp 127
    f16(-128.0),       # exactly INT8_MIN
    f16(-129.0),       # clamp -128
    f16(-130.0),       # clamp -128

    # --- well out of range -> clamp ---
    f16(200.0),        # clamp 127
    f16(-200.0),       # clamp -128
    f16(1000.0),       # clamp 127
    f16(-1000.0),      # clamp -128
    f16(60000.0),      # clamp 127
    f16(-60000.0),     # clamp -128
    f16(65504.0),      # f16 max -> clamp 127
    f16(-65504.0),     # clamp -128
    f16(float("inf")), # clamp 127
    f16(float("-inf")),# clamp -128

    # --- NaN saturates to 0 under V300 SAT ---
    f16(float("nan")),

    # --- extra RN edges ---
    f16(0.25),         # RN -> 0
    f16(0.75),         # RN -> 1
    f16(-0.25),        # RN -> 0
    f16(3.5),          # RN -> 4
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def rn_to_int(x):
    return int(np.rint(x))


def saturating_fptosi(v):
    if np.isnan(v):
        return 0
    if np.isposinf(v):
        return INT8_MAX
    if np.isneginf(v):
        return INT8_MIN
    return max(INT8_MIN, min(INT8_MAX, rn_to_int(v)))


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float16)
    src = np.tile(probe.view(np.uint16), ELEMS // len(PROBE))[:ELEMS].astype(np.uint16)
    src_f = src.view(np.float16).astype(np.float32)
    golden = np.array([saturating_fptosi(v) for v in src_f], dtype=np.int8)

    dst = np.full(ELEMS, -86, dtype=np.int8)  # 0xAA sentinel

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
