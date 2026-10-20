#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_testcase.py")
SPEC = importlib.util.spec_from_file_location("generate_testcase", SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ptoas-cce-assert-") as temp_dir:
        root = Path(temp_dir)
        source = root / "assert_emitc" / "assert_emitc.cpp"
        source.parent.mkdir()
        source.write_text(
            'extern "C" __global__ AICORE void assert_probe(__gm__ int* out) {\n'
            '  bool condition = out[0] != 0;\n'
            '  DEBUG_CHECK(condition, "check failed");\n'
            '}\n',
            encoding="utf-8",
        )
        output = root / "generated"
        GENERATOR.generate_testcase(source, output, "assert_probe", "npu", "Ascend910")
        cmake = (output / source.parent.name / "assert_probe" / "CMakeLists.txt").read_text(encoding="utf-8")
        assert "--cce-enable-print" in cmake
        assert "-DPTOAS_ENABLE_CCE_PRINT=1" in cmake


if __name__ == "__main__":
    main()
