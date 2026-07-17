#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Executable Python wrapper for the `ptoas` entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_python_roots() -> list[Path]:
    script_path = Path(__file__).resolve()
    roots: list[Path] = []

    env_root = os.environ.get("PTOAS_PYTHON_ROOT")
    if env_root:
        roots.append(Path(env_root))

    if len(script_path.parents) >= 3:
        roots.append(script_path.parents[2] / "python")
    if len(script_path.parents) >= 2:
        roots.append(script_path.parents[1])

    env_install = os.environ.get("PTO_INSTALL_DIR")
    if env_install:
        roots.append(Path(env_install))

    roots.append(script_path.parent)
    return roots


def _bootstrap_python_path() -> None:
    for root in _candidate_python_roots():
        launcher = root / "ptoas" / "_launcher.py"
        if launcher.is_file():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return
    raise SystemExit("unable to locate the ptoas Python package root")


def main() -> None:
    _bootstrap_python_path()
    from ptoas._launcher import main as launcher_main

    launcher_main()


if __name__ == "__main__":
    main()
