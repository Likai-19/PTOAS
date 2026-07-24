# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use this
# file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
# "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

N = 256
SENTINEL = 0

def generate(output_dir: Path) -> None:
    """Golden for ui16 round-trip via UNPK_B16 → PK_B16.

    PK_B16 writes 1 byte per mask bit into each ptr<ui16> slot (low byte),
    high byte remains sentinel (0).  Effective result: dst[i] = src[i] & 0xFF.
    """
    rng = np.random.default_rng(42)
    src = rng.integers(0, 2**16 - 1, size=N, dtype=np.uint16)

    golden = (src & 0xFF).astype(np.uint16)
    dst = np.zeros(N, dtype=np.uint16)
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
