#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# f16 -> u8 (FpToUi) with saturate = "SAT" golden.
#
# V300 SAT policy: negatives → 0, overflow → 255, NaN → 0,
# +inf → 255, -inf → 0.  RN (round-half-to-even) before clamp.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
U8_MAX = 255
U8_MIN = 0


def f16(x):
    return np.float16(x)


# 32-entry probe, tiled 8× → 256 lanes.
PROBE = [
    # --- zeros / smallest RN ties ---
    f16(0.0),
    f16(0.25),         # RN → 0
    f16(0.5),          # RN → 0 (banker's)
    f16(0.75),         # RN → 1
    f16(1.0),
    f16(1.5),          # RN → 2
    f16(2.5),          # RN → 2

    # --- typical in-range ---
    f16(3.0),
    f16(42.0),
    f16(127.0),
    f16(200.0),

    # --- upper boundary + RN ties near 255 ---
    f16(254.0),
    f16(254.5),        # RN → 254 (banker's)
    f16(255.0),
    f16(255.5),        # RN → 256 → SAT → 255

    # --- overflow high, must clamp to 255 ---
    f16(256.0),
    f16(1000.0),
    f16(65504.0),      # f16 max finite

    # --- negatives (all → 0 under V300 SAT) ---
    f16(-0.25),
    f16(-1.0),
    f16(-42.0),
    f16(-128.0),
    f16(-255.0),
    f16(-1000.0),
    f16(-65504.0),     # f16 min finite

    # --- IEEE specials ---
    f16(float("inf")),   # → 255
    f16(float("-inf")),  # → 0
    f16(float("nan")),   # → 0

    # --- pad to 32 ---
    f16(64.0),
    f16(128.0),
    f16(192.0),
    f16(32.0),
]
assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def sat_f16_to_u8(v: np.float16) -> int:
    """V300 SAT policy for f16 → u8 (RN, clamp to [0, 255])."""
    if np.isnan(v):
        return 0
    if np.isposinf(v):
        return U8_MAX
    if np.isneginf(v):
        return U8_MIN
    # Round-half-to-even, then clamp.
    r = int(np.rint(np.float32(v)))
    return max(U8_MIN, min(U8_MAX, r))


def nosat_f16_to_u8(v: np.float16) -> int:
    """NOSAT reference: low-8 unsigned truncation of RN-rounded int."""
    if np.isnan(v):
        return 0
    if np.isinf(v):
        # NOSAT on infinities: low-8 truncation of the saturated int value.
        # +inf → INT32_MAX → 0x7FFFFFFF → low-8 = 0xFF = 255
        # -inf → INT32_MIN → 0x80000000 → low-8 = 0x00 = 0
        return 255 if v > 0 else 0
    r = int(np.rint(np.float32(v)))
    return r & 0xFF


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float16)
    src = np.tile(probe, ELEMS // len(probe))[:ELEMS].astype(np.float16)

    golden_sat = np.array([sat_f16_to_u8(v) for v in src], dtype=np.uint8)
    nosat_ref = np.array([nosat_f16_to_u8(v) for v in src], dtype=np.uint8)

    # Guardrail: SAT golden MUST differ from NOSAT reference (otherwise the
    # lowering could silently produce NOSAT behavior).
    if np.array_equal(golden_sat, nosat_ref):
        raise SystemExit(
            "[FATAL] SAT golden equals NOSAT reference — guardrail failed. "
            "The probe set does not exercise SAT clamping. "
            "Add negative or overflow values to the probe."
        )

    # Sentinel-fill pre-kernel destination so a missed store shows up as
    # garbage instead of coincidentally-matching zeros.
    pre_dst = np.full(ELEMS, 0xCD, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    pre_dst.tofile(output_dir / "v2.bin")
    golden_sat.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
