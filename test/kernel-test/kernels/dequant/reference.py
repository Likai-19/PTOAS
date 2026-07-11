# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference case metadata for the dequant kernel."""

from __future__ import annotations

import numpy as np
import torch

DTYPES = ("f32", "bf16", "f16")
SRC_FORMATS = ("e4m3", "e5m2")
SCALE_FORMAT = "f32"
DEFAULT_LOOP_NUM2VF = 1
TOLERANCE = {
    "f32": 1e-6,
    "bf16": 4e-3,
    "f16": 1e-3,
}

_SEED = 42
_FP8_HALF_LANES = 256
_CHUNK_LANES = 64
_SCALE_LANES_PER_LOOP = 128
_ELEMS_PER_LOOP = 512


def _case_id(src_fmt: str, dst_fmt: str) -> str:
    return f"{src_fmt}_{dst_fmt}"


def _expand_scale(scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, _SCALE_LANES_PER_LOOP)
    scale_lo = np.tile(scale[:, :_CHUNK_LANES], (1, _FP8_HALF_LANES // _CHUNK_LANES))
    scale_hi = np.tile(scale[:, _CHUNK_LANES:], (1, _FP8_HALF_LANES // _CHUNK_LANES))
    return np.concatenate((scale_lo, scale_hi), axis=1).reshape(-1)


def _torch_src_dtype(src_fmt: str) -> torch.dtype:
    if src_fmt == "e4m3":
        return torch.float8_e4m3fn
    if src_fmt == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unknown src_fmt: {src_fmt}")


def _quantize_src_to_bits(src_fmt: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values_t = torch.from_numpy(np.asarray(values, dtype=np.float32))
    quantized_t = values_t.to(_torch_src_dtype(src_fmt))
    x_bits = quantized_t.view(torch.uint8).numpy().copy()
    x_decoded = quantized_t.float().numpy().astype(np.float32, copy=True)
    return x_bits, x_decoded


def _cast_output(dst_fmt: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if dst_fmt == "f32":
        return values.astype(np.float32, copy=True)
    if dst_fmt == "f16":
        return values.astype(np.float16).astype(np.float32)
    if dst_fmt == "bf16":
        return torch.from_numpy(values).to(torch.bfloat16).float().numpy().astype(np.float32, copy=True)
    raise ValueError(f"unknown dst_fmt: {dst_fmt}")


def generate_case(
    src_fmt: str,
    dst_fmt: str,
    *,
    loop_num2vf: int = DEFAULT_LOOP_NUM2VF,
) -> dict[str, object]:
    """Generate one lightweight CPU reference case."""

    if src_fmt not in SRC_FORMATS:
        raise ValueError(f"unknown src_fmt: {src_fmt}")
    if dst_fmt not in DTYPES:
        raise ValueError(f"unknown dst_fmt: {dst_fmt}")

    rng = np.random.default_rng(_SEED)
    x = rng.normal(0.0, 0.5, size=loop_num2vf * _ELEMS_PER_LOOP).astype(np.float32)
    scale = rng.uniform(
        0.25,
        1.75,
        size=loop_num2vf * _SCALE_LANES_PER_LOOP,
    ).astype(np.float32)
    x_bits, x_quantized = _quantize_src_to_bits(src_fmt, x)
    scale_expanded = _expand_scale(scale)
    y_f32 = x_quantized * scale_expanded
    y_expected = _cast_output(dst_fmt, y_f32)

    return {
        "case_id": _case_id(src_fmt, dst_fmt),
        "src_fmt": src_fmt,
        "scale_fmt": SCALE_FORMAT,
        "dst_fmt": dst_fmt,
        "loop_num2vf": loop_num2vf,
        "x_bits": x_bits,
        "x_f32": x_quantized,
        "scale": scale,
        "scale_expanded": scale_expanded,
        "y_expected": y_expected,
        "tolerance": TOLERANCE[dst_fmt],
        "default_alias": src_fmt == "e4m3" and dst_fmt == "f32",
    }


def generate_all(*, loop_num2vf: int = DEFAULT_LOOP_NUM2VF) -> dict[str, dict[str, object]]:
    """Generate the default dequant case matrix."""

    return {
        _case_id(src_fmt, dst_fmt): generate_case(
            src_fmt,
            dst_fmt,
            loop_num2vf=loop_num2vf,
        )
        for src_fmt in SRC_FORMATS
        for dst_fmt in DTYPES
    }
