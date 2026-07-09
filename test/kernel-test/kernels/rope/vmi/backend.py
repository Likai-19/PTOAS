# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VMI backend for the rope kernel."""

import os
from pathlib import Path

from ptodsl import pto

from kernel_test.backends import RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from ..runtime import RopeLaunchArgs, prepare_launch_args

_VMI_ROOT = Path(__file__).resolve().parent
_PTO_FILE_F16 = _VMI_ROOT / "rope_f16.vmi.pto"
_PTO_FILE_BF16 = _VMI_ROOT / "rope_bf16.vmi.pto"
_PTO_FILE_F32 = _VMI_ROOT / "rope_f32.vmi.pto"


@pto.jit(
    name="rope_vmi_f16",
    target="a5",
    kernel_kind="vector",
    source=str(_PTO_FILE_F16),
)
def rope_vmi_f16(
    x_ptr: pto.ptr(pto.f16, "gm"),
    cos_ptr: pto.ptr(pto.f16, "gm"),
    sin_ptr: pto.ptr(pto.f16, "gm"),
    y_ptr: pto.ptr(pto.f16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


@pto.jit(
    name="rope_vmi_bf16",
    target="a5",
    kernel_kind="vector",
    source=str(_PTO_FILE_BF16),
)
def rope_vmi_bf16(
    x_ptr: pto.ptr(pto.bf16, "gm"),
    cos_ptr: pto.ptr(pto.f16, "gm"),
    sin_ptr: pto.ptr(pto.f16, "gm"),
    y_ptr: pto.ptr(pto.bf16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


@pto.jit(
    name="rope_vmi_f32",
    target="a5",
    kernel_kind="vector",
    source=str(_PTO_FILE_F32),
)
def rope_vmi_f32(
    x_ptr: pto.ptr(pto.f32, "gm"),
    cos_ptr: pto.ptr(pto.f32, "gm"),
    sin_ptr: pto.ptr(pto.f32, "gm"),
    y_ptr: pto.ptr(pto.f32, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


_COMPILED: dict[str, object] = {}


def _kernel_for_dtype(dtype: str):
    if dtype == "f16":
        return "f16", rope_vmi_f16
    if dtype == "bf16":
        return "bf16", rope_vmi_bf16
    if dtype == "f32":
        return "f32", rope_vmi_f32
    raise ValueError(f"unsupported vmi dtype: {dtype}")


def _prepare(dtype: str) -> object:
    key, kernel = _kernel_for_dtype(dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = kernel.compile()
        _COMPILED[key] = compiled
    return compiled


def _launch(launch_args: RopeLaunchArgs):
    compiled = _prepare(launch_args.dtype)
    compiled[1, stream_ptr()](
        launch_args.x.data_ptr(),
        launch_args.cos.data_ptr(),
        launch_args.sin.data_ptr(),
        launch_args.y.data_ptr(),
        launch_args.s_count,
        launch_args.n_count,
        launch_args.mode_value,
    )
    sync()
    return launch_args.y


def rope_f16(launch_args: RopeLaunchArgs):
    """Launch the local rope f16 VMI kernel."""

    return _launch(launch_args)


def rope_bf16(launch_args: RopeLaunchArgs):
    """Launch the local rope bf16 VMI kernel."""

    return _launch(launch_args)


def rope_f32(launch_args: RopeLaunchArgs):
    """Launch the local rope f32 VMI kernel."""

    return _launch(launch_args)


class RopeVmiBackend:
    """Source-backed PTODSL VMI backend for rope."""

    name = "vmi"
    _launchers = {
        "f16": rope_f16,
        "bf16": rope_bf16,
        "f32": rope_f32,
    }

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        del purpose
        supported = case["dtype"] in {"f16", "bf16", "f32"} and case["mode"] in {"half", "interleave"}
        if supported:
            return True, None
        return False, "backend=vmi not wired for this case"

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        ensure_runtime("rope")
        launch_args = prepare_launch_args(case, cycle=purpose == "cycle")
        return self._launchers[launch_args.dtype](launch_args)

    def cache_tag(self) -> str:
        return (
            f"vmi:"
            f"{_PTO_FILE_F16}:{os.path.getmtime(_PTO_FILE_F16):.0f}:"
            f"{_PTO_FILE_BF16}:{os.path.getmtime(_PTO_FILE_BF16):.0f}:"
            f"{_PTO_FILE_F32}:{os.path.getmtime(_PTO_FILE_F32):.0f}"
        )
