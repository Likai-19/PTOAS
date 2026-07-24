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

N = 256
SENTINEL_U32 = np.uint32(0xDEADBEEF)


def generate(output_dir: Path) -> None:
    """Generate ui16 input and golden ui32 output via zero-extension (UNPK_B16)."""
    rng = np.random.default_rng(42)
    # ui16 values — low 16 bits of each ui32 carrier slot after unpack
    src = rng.integers(0, 2**16 - 1, size=N, dtype=np.uint16)

    # Golden: each ui16 zero-extended to ui32 (UNPK_B16 places source in low 16 bits)
    golden = src.astype(np.uint32)

    # Output buffer initialised with sentinel
    dst = np.full(N, SENTINEL_U32, dtype=np.uint32)

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
