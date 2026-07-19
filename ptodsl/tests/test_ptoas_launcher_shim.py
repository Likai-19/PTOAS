#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
import importlib.util
import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_SOURCE = REPO_ROOT / "ptodsl" / "ptoas" / "_launcher.py"


def _load_source_launcher():
    spec = importlib.util.spec_from_file_location("test_ptoas_launcher_shim_source", LAUNCHER_SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(REPO_ROOT / "ptodsl"))
        spec.loader.exec_module(module)
    finally:
        sys.path = saved_sys_path
    return module


class LauncherShimTests(unittest.TestCase):
    def test_main_forwards_directly_to_wheel_bootstrap(self):
        launcher = _load_source_launcher()

        with mock.patch.object(launcher, "_wheel_bootstrap_main") as wheel_main:
            launcher.main()

        wheel_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
