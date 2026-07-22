#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# f32 -> si16 (FpToSi) with saturate = "SAT" golden.
#
# V300 SAT policy for FpToSi (rounding = "R", i.e. RN):
#   * finite in-range values        -> round-to-nearest-even to int16
#   * finite out-of-range positive  -> INT16_MAX (32767)
#   * finite out-of-range negative  -> INT16_MIN (-32768)
#   * +inf / -inf                   -> INT16_MAX / INT16_MIN
#   * NaN                           -> 0
#
# f32 spans far beyond si16, so out-of-range clamping is the dominant path.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256
INT16_MAX = 32767
INT16_MIN = -32768


def f32(x):
    return np.float32(x)


# A 32-value probe. ELEMS = 256 = 8 * 32, giving a clean 8x-repeat schedule.
PROBE = [
    # --- in-range RN edges + typical values ---
    f32(0.0),          # 0
    f32(0.5),          # RN -> 0
    f32(1.5),          # RN -> 2
    f32(2.5),          # RN -> 2
    f32(-0.5),         # RN -> 0
    f32(-1.5),         # RN -> -2
    f32(-2.5),         # RN -> -2
    f32(1.0),          # 1
    f32(-1.0),         # -1
    f32(127.0),        # 127
    f32(-128.0),       # -128

    # --- boundary clamping ---
    f32(32767.0),      # exactly INT16_MAX
    f32(32767.5),      # RN -> 32768 -> clamp 32767
    f32(32768.0),      # clamp 32767
    f32(32769.0),      # clamp 32767
    f32(-32768.0),     # exactly INT16_MIN
    f32(-32768.5),     # RN -> -32769 -> clamp -32768
    f32(-32769.0),     # clamp -32768

    # --- well out of range -> clamp ---
    f32(40000.0),      # clamp 32767
    f32(-40000.0),     # clamp -32768
    f32(100000.0),     # clamp 32767
    f32(-100000.0),    # clamp -32768
    f32(1e6),          # clamp 32767
    f32(-1e6),         # clamp -32768
    f32(3.4e38),       # near f32 max -> clamp 32767
    f32(-3.4e38),      # clamp -32768
    f32(float("inf")), # clamp 32767
    f32(float("-inf")),# clamp -32768

    # --- NaN saturates to 0 under V300 SAT ---
    f32(float("nan")),

    # --- extra RN edges ---
    f32(0.25),         # RN -> 0
    f32(0.75),         # RN -> 1
    f32(-0.25),        # RN -> 0
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def rn_to_int(x):
    return int(np.rint(x))


def saturating_fptosi(v):
    if np.isnan(v):
        return 0
    if np.isposinf(v):
        return INT16_MAX
    if np.isneginf(v):
        return INT16_MIN
    return max(INT16_MIN, min(INT16_MAX, rn_to_int(v)))


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float32)
    src = np.tile(probe, ELEMS // len(PROBE))[:ELEMS].astype(np.float32)
    golden = np.array([saturating_fptosi(v) for v in src], dtype=np.int16)

    dst = np.full(ELEMS, -21846, dtype=np.int16)  # 0xAAAA sentinel

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
