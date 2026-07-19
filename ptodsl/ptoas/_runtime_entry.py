# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared runtime entry helpers for `ptoas` Python launchers.

This module owns the common in-process ctypes launch contract. Thin entry
wrappers such as the wheel console entry and the build/install-tree wrapper
should resolve their own filesystem layouts, then delegate here for
environment setup and `ptoas_entrypoint` execution.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Sequence


@dataclass(frozen=True)
class PTOASRuntimeLayout:
    wrapper: Path
    runtime_root: Path
    python_root: Path
    shared_module: Path
    tileops_dir: Path
    isolated_env: bool


def _prepend_env_path(env: MutableMapping[str, str], name: str, value: Path) -> None:
    if not value.exists():
        return
    current = env.get(name, "")
    rendered = str(value)
    parts = [part for part in current.split(os.pathsep) if part]
    if rendered in parts:
        parts.remove(rendered)
    parts.insert(0, rendered)
    env[name] = os.pathsep.join(parts)


def _has_cli_option(argv: Sequence[str], option: str) -> bool:
    option_with_value = f"{option}="
    for arg in argv:
        if arg == option or arg.startswith(option_with_value):
            return True
    return False


def _iter_runtime_library_files(runtime_lib_dir: Path) -> list[Path]:
    if not runtime_lib_dir.is_dir():
        return []

    libraries: dict[Path, None] = {}
    for pattern in ("*.so", "*.so.*", "*.dylib", "*.dylib.*"):
        for candidate in sorted(runtime_lib_dir.glob(pattern)):
            if candidate.is_file():
                libraries[candidate.resolve()] = None
    return list(libraries)


def _preload_runtime_libraries(shared_module: Path, runtime_lib_dir: Path) -> None:
    runtime_lib_dir = runtime_lib_dir.resolve()
    if not runtime_lib_dir.is_dir():
        return

    shared_module = shared_module.resolve()
    pending = [
        path for path in _iter_runtime_library_files(runtime_lib_dir)
        if path != shared_module and path.name != shared_module.name
    ]
    if not pending:
        return

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    last_error: OSError | None = None
    while pending:
        next_pending: list[Path] = []
        made_progress = False
        for candidate in pending:
            try:
                ctypes.CDLL(str(candidate), mode=mode)
                made_progress = True
            except OSError as exc:
                last_error = exc
                next_pending.append(candidate)
        if not next_pending:
            return
        if not made_progress:
            if last_error is not None:
                raise last_error
            raise OSError(f"unable to preload runtime libraries from {runtime_lib_dir}")
        pending = next_pending


def _load_shared_entrypoint(shared_module: Path, runtime_lib_dir: Path):
    _preload_runtime_libraries(shared_module, runtime_lib_dir)
    library = ctypes.CDLL(str(shared_module), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    entrypoint = library.ptoas_entrypoint
    entrypoint.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    entrypoint.restype = ctypes.c_int
    return entrypoint


def launch(layout: PTOASRuntimeLayout, user_args: Sequence[str]) -> int:
    env = os.environ
    env["PTOAS_HOME"] = str(layout.runtime_root)
    env["PTOAS_BIN"] = str(layout.wrapper)
    env["PTOAS_TILEOPS_DIR"] = str(layout.tileops_dir)

    _prepend_env_path(env, "PATH", layout.wrapper.parent)
    runtime_lib_dir = layout.runtime_root / "lib"
    if layout.isolated_env:
        env["PYTHONPATH"] = str(layout.python_root)
        env["LD_LIBRARY_PATH"] = str(runtime_lib_dir)
        env["DYLD_LIBRARY_PATH"] = str(runtime_lib_dir)
    else:
        _prepend_env_path(env, "PYTHONPATH", layout.python_root)
        _prepend_env_path(env, "LD_LIBRARY_PATH", runtime_lib_dir)
        _prepend_env_path(env, "DYLD_LIBRARY_PATH", runtime_lib_dir)

    argv = [str(layout.wrapper)]
    if not _has_cli_option(user_args, "--tilelang-path"):
        argv.extend(["--tilelang-path", str(layout.tileops_dir)])
    if not _has_cli_option(user_args, "--tilelang-pkg-path"):
        argv.extend(["--tilelang-pkg-path", str(layout.python_root)])
    if not _has_cli_option(user_args, "--ptodsl-pkg-path"):
        argv.extend(["--ptodsl-pkg-path", str(layout.python_root)])
    argv.extend(user_args)

    entrypoint = _load_shared_entrypoint(layout.shared_module, runtime_lib_dir)
    argv_bytes = [os.fsencode(arg) for arg in argv]
    c_argv = (ctypes.c_char_p * len(argv_bytes))(*argv_bytes)
    return int(entrypoint(len(argv_bytes), c_argv))
