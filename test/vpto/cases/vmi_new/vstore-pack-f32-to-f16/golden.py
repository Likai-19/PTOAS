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
    """Generate f32 input and golden f16 output via PK_B32 bit-level truncation.

    PK_B32 takes the low 16 bits of each f32 lane — no rounding, no range
    clamping.  Equivalent to: f32 bits → extract low 16 → reinterpret as f16.
    """
    rng = np.random.default_rng(42)
    src = rng.uniform(-8.0, 8.0, size=N).astype(np.float32)

    # PK_B32: low 16 bits of each u32 lane → reinterpret as f16
    src_u32 = src.view(np.uint32)
    lo16 = (src_u32 & 0xFFFF).astype(np.uint16)
    golden = lo16.view(np.float16)
    dst = np.zeros(N, dtype=np.float16)
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
