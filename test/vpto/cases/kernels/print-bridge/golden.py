#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# The kernel performs an identity passthrough (input -> UB -> output).
# Golden output = input data.

import argparse
from pathlib import Path

import numpy as np

ELEM_COUNT = 64
SEED = 42


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    input_data = rng.uniform(-10.0, 10.0, size=(ELEM_COUNT,)).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_data.tofile(output_dir / "v1.bin")
    # For an identity passthrough kernel, the golden output equals the input.
    input_data.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
