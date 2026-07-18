#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build/install-tree wrapper for the `ptoas` entrypoint.

This script is configured into the build tree and installed as `bin/ptoas`.
It bootstraps the Python package root for tree layouts, then delegates to the
shared in-process runtime entry contract in `ptoas._runtime_entry`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_BUILD_PYTHON_ROOT = Path("@CMAKE_BINARY_DIR@") / "python"
_DEFAULT_INSTALL_ROOT = Path("@CMAKE_INSTALL_PREFIX@")


def _candidate_python_roots() -> list[Path]:
    script_path = Path(__file__).resolve()
    roots: list[Path] = []

    env_root = os.environ.get("PTOAS_PYTHON_ROOT")
    if env_root:
        roots.append(Path(env_root))

    roots.append(_DEFAULT_BUILD_PYTHON_ROOT)

    if len(script_path.parents) >= 2:
        roots.append(script_path.parents[1])

    env_install = os.environ.get("PTO_INSTALL_DIR")
    if env_install:
        roots.append(Path(env_install))

    roots.append(_DEFAULT_INSTALL_ROOT)
    return roots


def _bootstrap_python_path() -> None:
    for root in _candidate_python_roots():
        runtime_entry = root / "ptoas" / "_runtime_entry.py"
        if runtime_entry.is_file():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return
    raise SystemExit("unable to locate the ptoas Python package root for the build/install-tree wrapper")


def main() -> None:
    _bootstrap_python_path()
    from ptoas import _runtime_entry

    wrapper = Path(__file__).resolve()
    layout = _runtime_entry.resolve_tree_layout(
        wrapper,
        build_python_root=_DEFAULT_BUILD_PYTHON_ROOT,
        install_root=_DEFAULT_INSTALL_ROOT,
    )
    raise SystemExit(_runtime_entry.launch(layout, sys.argv[1:]))


if __name__ == "__main__":
    main()
