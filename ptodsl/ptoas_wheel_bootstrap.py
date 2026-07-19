# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shadow-safe wheel bootstrap for the `ptoas` console entry.

The console script must not import `ptoas._launcher` directly because caller
environments may prepend unrelated `ptoas` packages via `PYTHONPATH`, editable
checkouts, or build-tree staging directories. This bootstrap has a unique
top-level module name, resolves the active wrapper's sibling site-packages
payload, then loads the wheel-owned `ptoas._runtime_entry` helper by file path.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import NoReturn


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    found = shutil.which(wrapper.name or "ptoas")
    if found:
        return Path(found).resolve()

    raise SystemExit(f"unable to locate the installed ptoas wrapper: {candidate}")


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = path.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _candidate_wheel_python_roots(wrapper: Path, module_file: str | None = None) -> list[Path]:
    candidates: list[Path] = []

    if module_file:
        candidates.append(Path(module_file).resolve().parent)

    for scheme_name in ("purelib", "platlib"):
        scheme_path = sysconfig.get_path(scheme_name)
        if scheme_path:
            candidates.append(Path(scheme_path))

    if len(wrapper.parents) >= 2:
        prefix = wrapper.parents[1]
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates.extend(
            [
                prefix / "lib" / py_version / "site-packages",
                prefix / "lib" / py_version / "dist-packages",
                prefix / "lib64" / py_version / "site-packages",
                prefix / "Lib" / "site-packages",
            ]
        )

    return _unique_paths(candidates)


def _resolve_wheel_python_root(wrapper: Path, module_file: str | None = None) -> tuple[Path, Path]:
    searched: list[str] = []
    for python_root in _candidate_wheel_python_roots(wrapper, module_file):
        package_root = python_root / "ptoas"
        runtime_entry = package_root / "_runtime_entry.py"
        runtime_root = package_root / "_runtime"
        searched.append(str(package_root))
        if runtime_entry.is_file() and runtime_root.is_dir():
            return python_root, package_root

    rendered = ", ".join(searched) if searched else "<none>"
    raise SystemExit(
        "unable to locate the wheel-installed ptoas package next to the active wrapper; "
        "expected <site-packages>/ptoas/_runtime_entry.py and <site-packages>/ptoas/_runtime. "
        f"Searched: {rendered}. This usually means an external PYTHONPATH is shadowing the installed wheel."
    )


def _resolve_wheel_layout(runtime_entry: ModuleType, package_root: Path, wrapper: Path):
    package_root = package_root.resolve()
    wrapper = wrapper.resolve()
    runtime_root = package_root / "_runtime"
    if not runtime_root.is_dir():
        raise SystemExit(
            "wheel-installed ptoas is missing the packaged runtime root: "
            "expected ptoas/_runtime"
        )

    shared_module = runtime_root / "lib" / "ptoas.so"
    if not shared_module.is_file() or shared_module.stat().st_size <= 0:
        raise SystemExit(
            "wheel-installed ptoas is missing the packaged shared module: "
            "expected ptoas/_runtime/lib/ptoas.so"
        )

    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    if not tileops_dir.is_dir():
        raise SystemExit(
            "wheel-installed ptoas is missing the packaged TileOps directory: "
            "expected ptoas/_runtime/share/ptoas/TileOps"
        )

    return runtime_entry.PTOASRuntimeLayout(
        wrapper=wrapper,
        runtime_root=runtime_root.resolve(),
        python_root=package_root.parent.resolve(),
        shared_module=shared_module.resolve(),
        tileops_dir=tileops_dir.resolve(),
        isolated_env=True,
    )


def _load_runtime_entry(python_root: Path, package_root: Path) -> ModuleType:
    python_root_text = str(python_root)
    if python_root_text in sys.path:
        sys.path.remove(python_root_text)
    sys.path.insert(0, python_root_text)

    runtime_entry_path = package_root / "_runtime_entry.py"
    module_name = "_ptoas_wheel_runtime_entry"
    spec = importlib.util.spec_from_file_location(module_name, runtime_entry_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load ptoas runtime entry helper from {runtime_entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_stage2_env(python_root: Path, package_root: Path) -> dict[str, str]:
    runtime_lib_dir = package_root / "_runtime" / "lib"
    env = os.environ.copy()
    env["PTOAS_WHEEL_STAGE2"] = "1"
    env["PYTHONPATH"] = str(python_root)
    env["LD_LIBRARY_PATH"] = str(runtime_lib_dir)
    env["DYLD_LIBRARY_PATH"] = str(runtime_lib_dir)
    return env


def _maybe_reexec_isolated_process(wrapper: Path, python_root: Path, package_root: Path) -> None:
    if os.environ.get("PTOAS_WHEEL_STAGE2") == "1":
        return
    env = _prepare_stage2_env(python_root, package_root)
    os.execve(str(wrapper), [str(wrapper), *sys.argv[1:]], env)


def main() -> NoReturn:
    wrapper = _resolve_wrapper_path()
    python_root, package_root = _resolve_wheel_python_root(wrapper, __file__)
    _maybe_reexec_isolated_process(wrapper, python_root, package_root)
    runtime_entry = _load_runtime_entry(python_root, package_root)
    layout = _resolve_wheel_layout(runtime_entry, package_root, wrapper)
    raise SystemExit(runtime_entry.launch(layout, sys.argv[1:]))


if __name__ == "__main__":
    main()
