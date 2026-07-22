#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# bf16 -> f16 (F2F) with saturate = "SAT" golden.
#
# bf16 and f16 share the same exponent bias; f16 has MORE mantissa bits
# (10 vs 7), so every bf16 value whose magnitude is <= 65504 (f16 max) is
# exactly representable in f16 and the conversion is lossless -- no rounding,
# no saturation. The probe deliberately uses only such values, so the golden
# is the exact f16 bit pattern of each bf16 value.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256


def f32(x):
    return np.float32(x)


def to_bf16_bits(x):
    """f32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    r = ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    r = np.where(np.isnan(x), np.uint16(0x7FC0), r)
    return r


def bf16_bits_to_f32(bits):
    u = bits.astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


# 32-value probe, all |v| <= 65504 (exactly representable in f16).
PROBE = [
    f32(0.0),          # 0
    f32(0.5),          # 0.5
    f32(1.5),          # 1.5
    f32(2.5),          # 2.5
    f32(-0.5),         # -0.5
    f32(-1.5),         # -1.5
    f32(-2.5),         # -2.5
    f32(1.0),          # 1
    f32(-1.0),         # -1
    f32(127.0),        # 127
    f32(-128.0),       # -128
    f32(32767.0),      # 32767
    f32(-32768.0),     # -32768
    f32(60000.0),      # 60000
    f32(-60000.0),     # -60000
    f32(65504.0),      # f16 max
    f32(-65504.0),     # f16 min
    f32(0.25),         # 0.25
    f32(0.75),         # 0.75
    f32(-0.25),        # -0.25
    f32(-0.75),        # -0.75
    f32(1234.5),       # 1234.5
    f32(-1234.5),      # -1234.5
    f32(2.0),          # 2
    f32(-2.0),         # -2
    f32(100.5),        # 100.5
    f32(-100.5),       # -100.5
    f32(16384.0),      # 16384
    f32(-16384.0),     # -16384
    f32(3.0),          # 3
    f32(-3.0),         # -3
    f32(5.5),          # 5.5
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float32)
    probe_bf16 = to_bf16_bits(probe)

    src = np.tile(probe_bf16, ELEMS // len(PROBE))[:ELEMS].astype(np.uint16)
    src_f32 = bf16_bits_to_f32(src)
    # bf16 RNE can round a probe value just above f16 max (65504 -> 65536).
    # Hardware SAT clamps such finite out-of-range values to f16 max/min
    # (65504 / -65504); np.float16 alone would overflow to +-inf, so clamp first.
    F16_MAX = 65504.0
    src_f32_clamped = np.clip(src_f32, -F16_MAX, F16_MAX)
    golden_f16 = src_f32_clamped.astype(np.float16)
    golden = golden_f16.view(np.uint16).astype(np.uint16)

    # Sentinel so a completely-missed store shows up as garbage, not coincidence.
    dst = np.full(ELEMS, 0xAAAA, dtype=np.uint16)

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
