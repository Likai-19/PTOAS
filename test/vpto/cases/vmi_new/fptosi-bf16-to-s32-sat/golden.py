#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# bf16 -> si32 (FpToSi) with saturate = "SAT" golden.
#
# V300 SAT policy for FpToSi (rounding = "R", i.e. RN):
#   * finite in-range values          -> round-to-nearest-even to int32
#   * finite out-of-range (positive)  -> INT32_MAX (0x7FFFFFFF)
#   * finite out-of-range (negative)  -> INT32_MIN (0x80000000)
#   * +inf                            -> INT32_MAX
#   * -inf                            -> INT32_MIN
#   * NaN                             -> 0
#
# bfloat16 has the same exponent range as f32 but only 7 mantissa bits, so the
# golden is computed on the EXACT bf16 value (after round-to-nearest-even
# f32->bf16) -- the same bits the kernel consumes -- not on the original probe.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256
INT32_MAX = np.int32(0x7FFFFFFF)
INT32_MIN = np.int32(-0x80000000)


def f32(x):
    return np.float32(x)


def to_bf16_bits(x):
    """f32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    r = ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    # Force any NaN to a canonical bf16 NaN (rounding must not turn it into inf).
    r = np.where(np.isnan(x), np.uint16(0x7FC0), r)
    return r


def bf16_bits_to_f32(bits):
    u = bits.astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


# A 32-value probe. ELEMS = 256 = 8 * 32, giving a clean 8x-repeat schedule.
PROBE = [
    # --- in-range: RN round-half-to-even edges + typical values ---
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
    f32(32767.0),      # 32767
    f32(-32768.0),     # -32768
    f32(1e6),          # bf16-representable ~1e6
    f32(-1e6),
    f32(999999.0),
    f32(1234567.5),    # rounds to nearest bf16, golden uses that exact value
    f32(-1234567.5),

    # --- out-of-range positive: clamp to INT32_MAX ---
    f32(2.147484e9),   # slightly above INT32_MAX
    f32(3.0e9),
    f32(1.0e10),
    f32(3.4e38),       # near f32/bf16 max
    f32(float("inf")), # +inf

    # --- out-of-range negative: clamp to INT32_MIN ---
    f32(-2.147484e9),
    f32(-3.0e9),
    f32(-1.0e10),
    f32(-3.4e38),
    f32(float("-inf")),

    # --- NaN saturates to 0 under V300 SAT ---
    f32(float("nan")),

    # --- extra RN edges ---
    f32(0.25),         # RN -> 0
    f32(0.75),         # RN -> 1
    f32(-0.25),        # RN -> 0
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def rn_to_int(x):
    # np.rint uses banker's rounding, matching RN on integer boundaries.
    return int(np.rint(x))


def saturating_fptosi(v):
    if np.isnan(v):
        return 0
    if np.isposinf(v):
        return int(INT32_MAX)
    if np.isneginf(v):
        return int(INT32_MIN)
    r = rn_to_int(v)
    if r >= int(INT32_MAX):
        return int(INT32_MAX)
    if r <= int(INT32_MIN):
        return int(INT32_MIN)
    return r


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float32)
    probe_bf16 = to_bf16_bits(probe)  # uint16 bf16 patterns

    src = np.tile(probe_bf16, ELEMS // len(PROBE))[:ELEMS].astype(np.uint16)
    src_f32 = bf16_bits_to_f32(src)
    golden = np.array([saturating_fptosi(v) for v in src_f32], dtype=np.int32)

    # Sentinel so a completely-missed store shows up as garbage, not coincidence.
    dst = np.full(ELEMS, -559038737, dtype=np.int32)  # 0xDEADBEEF

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
