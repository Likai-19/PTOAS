# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Artifact cache layout for JIT-compiled native libraries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _default_cache_root() -> Path | None:
    """Return the persistent artifact cache root, or None when disabled.

    The on-disk cache is off by default.  The specialization key embeds
    process addresses (a pybind11 method address in _DType.__repr__ and
    id(callback)), so it never hits across sessions; the directory grows
    without bound; and artifacts are published without locking, a temporary
    name, or a fallback, so concurrent writers can share a directory and fail.
    Set PTODSL_CACHE_DIR to opt back in; PTODSL_NO_CACHE wins over it.
    """
    if os.environ.get("PTODSL_NO_CACHE"):
        return None
    override = os.environ.get("PTODSL_CACHE_DIR")
    if override:
        return Path(override)
    return None


def cache_enabled() -> bool:
    """Whether a persistent on-disk artifact cache is in use."""
    return _default_cache_root() is not None


def _specialization_digest(specialization_key) -> str:
    payload = repr(specialization_key).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeBuildArtifacts:
    """Paths to one compiled native kernel specialization."""

    cache_dir: Path
    mlir_path: Path
    kernel_object: Path
    launch_cpp: Path
    shared_library: Path
    manifest_path: Path


def artifact_paths(
    py_name: str,
    ir_function_name: str,
    specialization_key,
    cache_root: Path | None = None,
) -> NativeBuildArtifacts:
    """Return stable artifact paths for one compiled specialization.

    ``cache_root`` carries the directory to build in when the persistent cache
    is disabled; callers pass the ``tempfile.TemporaryDirectory`` they manage so
    its standard cleanup owns the artifact lifetime.
    """
    digest = _specialization_digest(specialization_key)
    safe_name = py_name.replace("/", "_")
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    if root is None:
        # A direct caller with the cache disabled still receives a stable path
        # under the standard temporary directory; build_native_library always
        # passes its own TemporaryDirectory so those artifacts are removed.
        root = Path(tempfile.gettempdir())
    cache_dir = root / f"{safe_name}_{digest}"
    lib_name = f"lib{ir_function_name}.so"
    return NativeBuildArtifacts(
        cache_dir=cache_dir,
        mlir_path=cache_dir / "kernel.mlir",
        kernel_object=cache_dir / "kernel.o",
        launch_cpp=cache_dir / "launch.cpp",
        shared_library=cache_dir / lib_name,
        manifest_path=cache_dir / "manifest.json",
    )


def write_manifest(
    artifacts: NativeBuildArtifacts,
    *,
    ir_function_name: str,
    launch_symbol: str,
    mlir_digest: str,
    launch_cpp_digest: str,
    compile_config_digest: str,
    link_config_digest: str,
) -> None:
    artifacts.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ir_function_name": ir_function_name,
        "launch_symbol": launch_symbol,
        "shared_library": str(artifacts.shared_library),
        "mlir_digest": mlir_digest,
        "launch_cpp_digest": launch_cpp_digest,
        "compile_config_digest": compile_config_digest,
        "link_config_digest": link_config_digest,
    }
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_manifest(artifacts: NativeBuildArtifacts) -> dict:
    return json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))


def is_native_build_current(
    artifacts: NativeBuildArtifacts,
    *,
    mlir_text: str,
    launch_cpp_text: str,
    compile_config_text: str,
    link_config_text: str,
) -> bool:
    required = (
        artifacts.mlir_path,
        artifacts.kernel_object,
        artifacts.launch_cpp,
        artifacts.shared_library,
        artifacts.manifest_path,
    )
    if not all(path.is_file() for path in required):
        return False

    try:
        manifest = read_manifest(artifacts)
    except Exception:
        return False

    return (
        manifest.get("mlir_digest") == _content_digest(mlir_text)
        and manifest.get("launch_cpp_digest") == _content_digest(launch_cpp_text)
        and manifest.get("compile_config_digest") == _content_digest(compile_config_text)
        and manifest.get("link_config_digest") == _content_digest(link_config_text)
    )


__all__ = [
    "NativeBuildArtifacts",
    "artifact_paths",
    "cache_enabled",
    "_content_digest",
    "is_native_build_current",
    "read_manifest",
    "write_manifest",
]
