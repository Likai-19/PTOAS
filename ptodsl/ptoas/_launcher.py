# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Wheel-installed `ptoas` console entry."""

from __future__ import annotations

import ctypes
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import NoReturn


def _prepend_env_path(env: dict[str, str], name: str, value: Path) -> None:
    if not value.exists():
        return
    current = env.get(name, "")
    rendered = str(value)
    parts = [part for part in current.split(os.pathsep) if part]
    if rendered in parts:
        parts.remove(rendered)
    parts.insert(0, rendered)
    env[name] = os.pathsep.join(parts)


def _has_cli_option(argv: list[str], option: str) -> bool:
    option_with_value = f"{option}="
    for arg in argv:
        if arg == option or arg.startswith(option_with_value):
            return True
    return False


def _resolve_wrapper_path() -> Path:
    argv0 = Path(sys.argv[0])
    if argv0.exists():
        return argv0.resolve()

    found = shutil.which(argv0.name or "ptoas")
    if found:
        return Path(found).resolve()

    raise SystemExit(f"unable to locate the installed ptoas wrapper: {sys.argv[0]}")


def _resolve_shared_module_path(package_root: Path, runtime_root: Path, wrapper: Path) -> Path:
    candidates = [
        package_root.parent / "pto" / "ptoas.so",
        wrapper.parent / "ptoas.so",
        runtime_root / "lib" / "ptoas.so",
        runtime_root / "pto" / "ptoas.so",
    ]

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    raise SystemExit(
        "wheel/runtime is missing the packaged shared module: expected pto.ptoas "
        "or a local ptoas.so next to the wrapper/install tree"
    )


def _resolve_runtime_root(package_root: Path) -> Path:
    runtime_root = package_root / "_runtime"
    if runtime_root.exists():
        return runtime_root
    env_install_dir = os.environ.get("PTO_INSTALL_DIR")
    if env_install_dir:
        return Path(env_install_dir)
    return package_root.parent.parent / "install"


def _iter_runtime_library_deps(shared_module: Path, runtime_lib_dir: Path) -> list[Path]:
    if not runtime_lib_dir.is_dir():
        return []

    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    rendered = str(runtime_lib_dir)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [rendered] + [part for part in current.split(os.pathsep) if part]
    )

    output = subprocess.check_output(
        ["ldd", str(shared_module)],
        text=True,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    deps: list[Path] = []
    for line in output.splitlines():
        line = line.strip()
        if "=>" in line:
            candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        else:
            candidate = line.split(" ", 1)[0]
        if not candidate.startswith("/"):
            continue
        dep_path = Path(candidate)
        try:
            if runtime_lib_dir.resolve() not in dep_path.resolve().parents:
                continue
        except FileNotFoundError:
            continue
        deps.append(dep_path)
    return deps


def _preload_runtime_libraries(shared_module: Path, runtime_lib_dir: Path) -> None:
    runtime_lib_dir = runtime_lib_dir.resolve()
    if not runtime_lib_dir.is_dir():
        return

    loaded: set[Path] = set()
    visiting: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in loaded or resolved in visiting:
            return
        visiting.add(resolved)
        try:
            for dep in _iter_runtime_library_deps(resolved, runtime_lib_dir):
                visit(dep)
            ctypes.CDLL(str(resolved), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            loaded.add(resolved)
        finally:
            visiting.discard(resolved)

    for dep in _iter_runtime_library_deps(shared_module, runtime_lib_dir):
        visit(dep)


def _load_shared_entrypoint(shared_module: Path, runtime_lib_dir: Path):
    _preload_runtime_libraries(shared_module, runtime_lib_dir)
    library = ctypes.CDLL(str(shared_module), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    entrypoint = library.ptoas_entrypoint
    entrypoint.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    entrypoint.restype = ctypes.c_int
    return entrypoint


def main() -> NoReturn:
    package_root = Path(__file__).resolve().parent
    runtime_root = _resolve_runtime_root(package_root)
    wrapper = _resolve_wrapper_path()
    shared_module = _resolve_shared_module_path(package_root, runtime_root, wrapper)

    python_root = package_root.parent if runtime_root.name == "_runtime" else runtime_root
    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    env = os.environ.copy()
    env["PTOAS_HOME"] = str(runtime_root)
    env["PTOAS_BIN"] = str(wrapper)
    env["PTOAS_TILEOPS_DIR"] = str(tileops_dir)

    _prepend_env_path(env, "PATH", wrapper.parent)
    _prepend_env_path(env, "PYTHONPATH", python_root)
    _prepend_env_path(env, "LD_LIBRARY_PATH", runtime_root / "lib")
    _prepend_env_path(env, "DYLD_LIBRARY_PATH", runtime_root / "lib")
    os.environ.update(env)

    argv = [str(wrapper)]
    user_args = sys.argv[1:]
    if not _has_cli_option(user_args, "--tilelang-path"):
        argv.extend(["--tilelang-path", str(tileops_dir)])
    if not _has_cli_option(user_args, "--tilelang-pkg-path"):
        argv.extend(["--tilelang-pkg-path", str(python_root)])
    argv.extend(user_args)

    entrypoint = _load_shared_entrypoint(shared_module, runtime_root / "lib")
    argv_bytes = [os.fsencode(arg) for arg in argv]
    c_argv = (ctypes.c_char_p * len(argv_bytes))(*argv_bytes)
    raise SystemExit(entrypoint(len(argv_bytes), c_argv))


if __name__ == "__main__":
    main()
