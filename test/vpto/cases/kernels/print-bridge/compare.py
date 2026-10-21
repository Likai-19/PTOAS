#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Compare: golden copy (golden_v2.bin) matches kernel output (v2.bin).
# The kernel is an identity passthrough, so output must equal input.

import os
import re
import sys

import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: golden {golden.shape} vs output {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(abs_diff))
        print(
            f"[ERROR] Mismatch: max diff={float(abs_diff[idx])} at idx={idx} "
            f"(golden={float(golden[idx])}, out={float(output[idx])}, dtype={dtype_np})"
        )
        return False
    return True


def check_print_output() -> bool:
    """The simulator run log must carry the DebugTunnel-decoded print text:
    the scalar diagnostics (offset/n), the tprint header and shape lines,
    and both valid tile regions rendered with the %6.2f value format."""
    log_path = "validation.log"
    if not os.path.exists(log_path):
        print(f"[ERROR] run log missing: {log_path}")
        return False
    with open(log_path, "r", errors="replace") as f:
        log = f.read()
    expected_fragments = [
        "offset = +001.500",
        "n = 64",
        "=== [TPRINT Tile] Data Type: float32, Layout: ND, TileType: Vec ===",
        "  Shape: [8, 8], Valid Shape: [2, 3]",
        "  Shape: [4, 8], Valid Shape: [2, 3]",
    ]
    missing = [frag for frag in expected_fragments if frag not in log]
    if missing:
        for frag in missing:
            print(f"[ERROR] print output missing: {frag!r}")
        return False
    values = np.fromfile("golden_v2.bin", dtype=np.float32)
    expected_tiles = [values.reshape(8, 8)[:2, :3].reshape(-1),
                      values[32:].reshape(4, 8)[:2, :3].reshape(-1)]
    shapes = ["  Shape: [8, 8], Valid Shape: [2, 3]",
              "  Shape: [4, 8], Valid Shape: [2, 3]"]
    for shape, expected in zip(shapes, expected_tiles):
        section = log.split(shape, 1)[1].split("=== [TPRINT Tile]", 1)[0]
        actual = re.findall(r"(?<![\w.])[-+]?\d+\.\d{2}(?!\d)", section)
        rendered = [f"{float(value):.2f}" for value in expected]
        if actual != rendered:
            print(f"[ERROR] tprint values differ for {shape}: {actual}")
            return False
    return True


def main():
    ok = True
    ok = compare_bin("golden_v2.bin", "v2.bin", np.float32, 1e-6) and ok
    ok = check_print_output() and ok
    if not ok:
        print("[ERROR] compare failed")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
