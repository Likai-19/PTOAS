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


SLOT_ELEMS = 256
OUTPUT_ELEMS = 5 * SLOT_ELEMS
ACTIVE_LANES = {
    64: (0, 63),
    128: (0, 63, 64, 127),
    256: (0, 63, 64, 127, 128, 255),
}
OUTPUT_SENTINEL = np.uint32(0xA5A5A5A5)


def signed_pattern(size: int, salt: int) -> np.ndarray:
    lanes = np.arange(size, dtype=np.int64)
    magnitude = ((lanes + 1) * (104729 + salt * 8191)) % 2_000_000_000 + 1
    signs = np.where(((lanes + salt) & 1) == 0, 1, -1)
    return (magnitude * signs).astype(np.int32)


def unsigned_pattern(size: int, salt: int) -> np.ndarray:
    lanes = np.arange(size, dtype=np.uint64)
    values = (np.uint64(0x80000001 + salt) + lanes * np.uint64(0x0010003D))
    return (values & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def apply_oracle(
    golden_low: np.ndarray,
    golden_high: np.ndarray,
    slot: int,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    signed: bool,
) -> None:
    lanes = lhs.size
    active = np.zeros(lanes, dtype=np.bool_)
    active[list(ACTIVE_LANES[lanes])] = True

    if signed:
        product = lhs.astype(np.int64) * rhs.astype(np.int64)
        low = (product & np.int64(0xFFFFFFFF)).astype(np.uint32)
        high = ((product >> np.int64(32)) & np.int64(0xFFFFFFFF)).astype(np.uint32)
    else:
        product = lhs.astype(np.uint64) * rhs.astype(np.uint64)
        low = (product & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        high = ((product >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    begin = slot * SLOT_ELEMS
    end = begin + lanes
    golden_low[begin:end] = np.where(active, low, np.uint32(0))
    golden_high[begin:end] = np.where(active, high, np.uint32(0))


def generate(output_dir: Path) -> None:
    lhs32 = np.empty(3 * SLOT_ELEMS, dtype=np.uint32)
    rhs32 = np.empty(3 * SLOT_ELEMS, dtype=np.uint32)

    lhs64 = signed_pattern(64, 1)
    rhs64 = signed_pattern(64, 2)
    lhs64[0], rhs64[0] = np.int32(-1), np.int32(2)
    lhs64[63], rhs64[63] = np.int32(-0x80000000), np.int32(-1)
    lhs32[0:64] = lhs64.view(np.uint32)
    rhs32[0:64] = rhs64.view(np.uint32)
    lhs32[64:SLOT_ELEMS] = unsigned_pattern(SLOT_ELEMS - 64, 3)
    rhs32[64:SLOT_ELEMS] = unsigned_pattern(SLOT_ELEMS - 64, 4)

    lhs128 = signed_pattern(128, 5)
    rhs128 = signed_pattern(128, 6)
    lhs128[0], rhs128[0] = np.int32(-0x80000000), np.int32(2)
    lhs128[63], rhs128[63] = np.int32(0x7FFFFFFF), np.int32(0x7FFFFFFF)
    lhs128[64], rhs128[64] = np.int32(-123456789), np.int32(17)
    lhs128[127], rhs128[127] = np.int32(-2000000000), np.int32(-3)
    lhs32[SLOT_ELEMS:SLOT_ELEMS + 128] = lhs128.view(np.uint32)
    rhs32[SLOT_ELEMS:SLOT_ELEMS + 128] = rhs128.view(np.uint32)
    lhs32[SLOT_ELEMS + 128:2 * SLOT_ELEMS] = unsigned_pattern(128, 7)
    rhs32[SLOT_ELEMS + 128:2 * SLOT_ELEMS] = unsigned_pattern(128, 8)

    lhs256 = unsigned_pattern(256, 9)
    rhs256 = unsigned_pattern(256, 10)
    lhs256[0], rhs256[0] = np.uint32(0xFFFFFFFF), np.uint32(0xFFFFFFFF)
    lhs256[63], rhs256[63] = np.uint32(0x80000000), np.uint32(2)
    lhs256[64], rhs256[64] = np.uint32(0xFFFFFFFE), np.uint32(3)
    lhs256[127], rhs256[127] = np.uint32(0xDEADBEEF), np.uint32(0x10203040)
    lhs256[128], rhs256[128] = np.uint32(0x80000001), np.uint32(0x7FFFFFFF)
    lhs256[255], rhs256[255] = np.uint32(0xFEDCBA98), np.uint32(0x89ABCDEF)
    lhs32[2 * SLOT_ELEMS:3 * SLOT_ELEMS] = lhs256
    rhs32[2 * SLOT_ELEMS:3 * SLOT_ELEMS] = rhs256

    lhs16 = (((np.arange(128, dtype=np.int32) * 197 + 11) % 32767) + 1).astype(np.int16)
    rhs16 = -((((np.arange(128, dtype=np.int32) * 251 + 17) % 32767) + 1).astype(np.int16))
    lhs16[0], rhs16[0] = np.int16(-32768), np.int16(-1)
    lhs16[63], rhs16[63] = np.int16(32767), np.int16(32767)
    lhs16[64], rhs16[64] = np.int16(-12345), np.int16(2345)
    lhs16[127], rhs16[127] = np.int16(-30000), np.int16(-2)

    lhs8 = ((np.arange(256, dtype=np.uint16) * 37 + 1) % 255 + 1).astype(np.uint8)
    rhs8 = ((np.arange(256, dtype=np.uint16) * 53 + 3) % 255 + 1).astype(np.uint8)
    lhs8[[0, 63, 64, 127, 128, 255]] = np.array([255, 128, 254, 213, 129, 251], dtype=np.uint8)
    rhs8[[0, 63, 64, 127, 128, 255]] = np.array([255, 2, 3, 97, 127, 239], dtype=np.uint8)

    golden_low = np.full(OUTPUT_ELEMS, OUTPUT_SENTINEL, dtype=np.uint32)
    golden_high = np.full(OUTPUT_ELEMS, OUTPUT_SENTINEL, dtype=np.uint32)
    apply_oracle(golden_low, golden_high, 0, lhs64, rhs64, signed=True)
    apply_oracle(golden_low, golden_high, 1, lhs128, rhs128, signed=True)
    apply_oracle(golden_low, golden_high, 2, lhs256, rhs256, signed=False)
    apply_oracle(
        golden_low, golden_high, 3, lhs16.astype(np.int32), rhs16.astype(np.int32), signed=True
    )
    apply_oracle(
        golden_low, golden_high, 4, lhs8.astype(np.uint32), rhs8.astype(np.uint32), signed=False
    )

    # Keep the documented signed/unsigned boundary examples explicit in the
    # oracle itself so a generator regression cannot silently remove them.
    assert (golden_low[0], golden_high[0]) == (0xFFFFFFFE, 0xFFFFFFFF)
    assert (golden_low[63], golden_high[63]) == (0x80000000, 0x00000000)
    assert (golden_low[256], golden_high[256]) == (0x00000000, 0xFFFFFFFF)
    assert (golden_low[319], golden_high[319]) == (0x00000001, 0x3FFFFFFF)
    assert (golden_low[512], golden_high[512]) == (0x00000001, 0xFFFFFFFE)
    assert (golden_low[575], golden_high[575]) == (0x00000000, 0x00000001)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs32.tofile(output_dir / "lhs32.bin")
    rhs32.tofile(output_dir / "rhs32.bin")
    lhs16.tofile(output_dir / "lhs16.bin")
    rhs16.tofile(output_dir / "rhs16.bin")
    lhs8.tofile(output_dir / "lhs8.bin")
    rhs8.tofile(output_dir / "rhs8.bin")
    np.full(OUTPUT_ELEMS, OUTPUT_SENTINEL, dtype=np.uint32).tofile(output_dir / "low.bin")
    np.full(OUTPUT_ELEMS, OUTPUT_SENTINEL, dtype=np.uint32).tofile(output_dir / "high.bin")
    golden_low.tofile(output_dir / "golden_low.bin")
    golden_high.tofile(output_dir / "golden_high.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
