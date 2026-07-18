# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Wheel-installed `ptoas` console entry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

from ptoas import _runtime_entry


def main() -> NoReturn:
    package_root = Path(__file__).resolve().parent
    wrapper = _runtime_entry.resolve_wrapper_path()
    layout = _runtime_entry.resolve_wheel_layout(package_root, wrapper)
    raise SystemExit(_runtime_entry.launch(layout, sys.argv[1:]))


if __name__ == "__main__":
    main()
