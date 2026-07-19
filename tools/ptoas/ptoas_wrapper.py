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

import shutil
import sys
from pathlib import Path


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    found = shutil.which(wrapper.name or "ptoas")
    if found:
        return Path(found).resolve()

    raise SystemExit(f"unable to locate the installed ptoas wrapper: {candidate}")


def _build_tree_root(wrapper: Path) -> Path | None:
    if len(wrapper.parents) < 3:
        return None
    return wrapper.parents[2]


def _install_tree_root(wrapper: Path) -> Path | None:
    if len(wrapper.parents) < 2:
        return None
    return wrapper.parents[1]


def _is_build_tree_wrapper(wrapper: Path) -> bool:
    return len(wrapper.parents) >= 2 and wrapper.parents[1].name == "tools"


def _require_python_root(python_root: Path, *, context: str) -> Path:
    helper = python_root / "ptoas" / "_runtime_entry.py"
    if helper.is_file():
        return python_root.resolve()
    raise SystemExit(
        f"unable to locate the {context} ptoas Python package root: "
        f"expected {python_root}/ptoas/_runtime_entry.py"
    )


def _require_runtime_root(runtime_root: Path) -> Path:
    shared_module = runtime_root / "lib" / "ptoas.so"
    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    if shared_module.is_file() and shared_module.stat().st_size > 0 and tileops_dir.is_dir():
        return runtime_root.resolve()
    raise SystemExit(
        "unable to locate the ptoas runtime tree: expected "
        f"{runtime_root}/lib/ptoas.so and {runtime_root}/share/ptoas/TileOps"
    )


def _resolve_build_tree_layout(wrapper: Path):
    from ptoas import _runtime_entry

    build_root = _build_tree_root(wrapper)
    if build_root is None:
        raise SystemExit("unable to locate the build-tree layout for ptoas")

    python_root = _require_python_root(build_root / "python", context="build-tree")
    runtime_root = _require_runtime_root(build_root / "runtime-staging")

    return _runtime_entry.PTOASRuntimeLayout(
        wrapper=wrapper,
        runtime_root=runtime_root,
        python_root=python_root,
        shared_module=(runtime_root / "lib" / "ptoas.so").resolve(),
        tileops_dir=(runtime_root / "share" / "ptoas" / "TileOps").resolve(),
        isolated_env=False,
    )


def _resolve_install_tree_layout(wrapper: Path):
    from ptoas import _runtime_entry

    install_root = _install_tree_root(wrapper)
    if install_root is None:
        raise SystemExit("unable to locate the install-tree layout for ptoas")

    python_root = _require_python_root(install_root, context="install-tree")
    runtime_root = _require_runtime_root(install_root)

    return _runtime_entry.PTOASRuntimeLayout(
        wrapper=wrapper,
        runtime_root=runtime_root,
        python_root=python_root,
        shared_module=(runtime_root / "lib" / "ptoas.so").resolve(),
        tileops_dir=(runtime_root / "share" / "ptoas" / "TileOps").resolve(),
        isolated_env=False,
    )


def _bootstrap_python_path(wrapper: Path) -> None:
    if _is_build_tree_wrapper(wrapper):
        build_root = _build_tree_root(wrapper)
        if build_root is not None:
            root = build_root / "python"
            _require_python_root(root, context="build-tree")
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return
        raise SystemExit("unable to locate the build-tree layout for ptoas")

    install_root = _install_tree_root(wrapper)
    if install_root is not None:
        _require_python_root(install_root, context="install-tree")
        root_text = str(install_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        return
    raise SystemExit("unable to locate the ptoas Python package root for the build/install-tree wrapper")


def main() -> None:
    wrapper = _resolve_wrapper_path()
    _bootstrap_python_path(wrapper)
    from ptoas import _runtime_entry

    if _is_build_tree_wrapper(wrapper):
        layout = _resolve_build_tree_layout(wrapper)
    else:
        layout = _resolve_install_tree_layout(wrapper)
    raise SystemExit(_runtime_entry.launch(layout, sys.argv[1:]))


if __name__ == "__main__":
    main()
