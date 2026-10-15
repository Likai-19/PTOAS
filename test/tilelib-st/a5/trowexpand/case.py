#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib ST coverage for the A5 row-expand binary family.

The seven row-expand binary TileOps (``trowexpandadd/sub/mul/div/max/min/expdif``)
share the TileLib template ``lib/TileOps/a5/_expand_binary.py``'s
``_emit_row_expand_body``.  Their broadcast form follows the expanded operand's
layout, mirroring ``pto-isa/docs/isa/TROWEXPANDADD.md`` (``B = 32 / sizeof(dtype)``)::

    Mode 2 (src1 RowMajor, B columns):  dst[r, c] = src0[r, c] OP src1[r, c % B]
    Mode 1 (src1 ColMajor, 1 column):   dst[r, c] = src0[r, c] OP src1[r, 0]

Coverage in this suite:

* all seven interfaces on an f32 8x32 dst with an 8x8 RowMajor ``src1`` (B = 8),
  i.e. ``src1[r, c % 8]``;
* a Mode 1 (ColMajor single-column) control per interface, so the scalar-splat
  form stays covered alongside the block form;
* cross-vector-chunk RowMajor (dst cols > one 256-byte vector), short/tail RowMajor
  (dst cols not a multiple of the vector width) and the 16-bit / 8-bit block widths.

This suite is PTODSL auto mode, so it compiles through
``ptoas --pto-backend=vpto --enable-tile-op-expand`` -- the path that lowers the
row-expand template.  See ``test/tilelib-st/README.md``.
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# ── per-dtype facts ───────────────────────────────────────────────────────────

# 32-byte block width, i.e. pto-isa Mode 2's ``32 / sizeof(dtype)`` column count.
BLOCK_ELEMS = {"f32": 8, "i32": 8, "f16": 16, "i16": 16, "i8": 32}

# A5 vector register = 256 bytes, so this many elements per column chunk.
VREG_ELEMS = {"f32": 64, "i32": 64, "f16": 128, "i16": 128, "i8": 256}

_DTYPE = {
    "f32": (pto.f32, np.float32),
    "f16": (pto.f16, np.float16),
    "i8": (pto.i8, np.int8),
}


def _exp_dif(lhs, rhs):
    """``trowexpandexpdif`` semantics: ``exp(src0 - src1)``."""
    return np.exp(lhs - rhs)


# op name -> (PTODSL tile op, NumPy reference, floating-point only?)
_OPS = {
    "add": (pto.tile.rowexpandadd, np.add, False),
    "sub": (pto.tile.rowexpandsub, np.subtract, False),
    "mul": (pto.tile.rowexpandmul, np.multiply, False),
    "div": (pto.tile.rowexpanddiv, np.divide, True),
    "max": (pto.tile.rowexpandmax, np.maximum, False),
    "min": (pto.tile.rowexpandmin, np.minimum, False),
    "expdif": (pto.tile.rowexpandexpdif, _exp_dif, True),
}

INTERFACES = tuple(_OPS)


# ── kernel factory ────────────────────────────────────────────────────────────

def _make_kernel(case_name, op_name, dtype_key, rows, cols, mode, expanded):
    """Build one ``@pto.jit`` kernel for one (op, dtype, shape, mode, direction).

    ``mode`` is ``"rm"`` for ISA Mode 2 (RowMajor expanded operand with ``B`` valid
    columns, the 32-byte block repeats) or ``"cm"`` for ISA Mode 1 (ColMajor expanded
    operand with a single valid column, one scalar per row).

    ``expanded`` says which operand is the row-expanded one (``"src0"`` or ``"src1"``);
    the other matches ``dst``.
    """
    pto_dtype, _np_dtype = _DTYPE[dtype_key]
    block = BLOCK_ELEMS[dtype_key] if mode == "rm" else 1
    op_fn = _OPS[op_name][0]
    blayout = "RowMajor" if mode == "rm" else "ColMajor"
    # A single-column logical tile is authored ColMajor (see pto alloc_tile docs).
    expanded_strides = [block, 1] if mode == "rm" else [1, 1]
    src0_is_full = expanded == "src1"
    src0_shape = [rows, cols] if src0_is_full else [rows, block]
    src1_shape = [rows, block] if src0_is_full else [rows, cols]
    src0_strides = [cols, 1] if src0_is_full else expanded_strides
    src1_strides = expanded_strides if src0_is_full else [cols, 1]

    @pto.jit(name=case_name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=src0_shape, strides=src0_strides)
        src1_view = pto.make_tensor_view(src1_ptr, shape=src1_shape, strides=src1_strides)
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        if src0_is_full:
            src0 = pto.alloc_tile(
                shape=src0_shape, dtype=pto_dtype, valid_shape=src0_shape
            )
            src1 = pto.alloc_tile(
                shape=src1_shape, dtype=pto_dtype, valid_shape=src1_shape, blayout=blayout
            )
        else:
            src0 = pto.alloc_tile(
                shape=src0_shape, dtype=pto_dtype, valid_shape=src0_shape, blayout=blayout
            )
            src1 = pto.alloc_tile(
                shape=src1_shape, dtype=pto_dtype, valid_shape=src1_shape
            )
        dst = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype, valid_shape=[rows, cols])

        pto.tile.load(src0_view, src0)
        pto.tile.load(src1_view, src1)
        op_fn(src0, src1, dst)
        pto.tile.store(dst, dst_view)

    return _kernel


# ── inputs / golden ───────────────────────────────────────────────────────────

def _make_inputs(case_name, op_name, dtype_key, rows, cols, mode, expanded):
    """Deterministic per-case inputs, keeping every op in a well-conditioned range."""
    _pto_dtype, np_dtype = _DTYPE[dtype_key]
    block = BLOCK_ELEMS[dtype_key] if mode == "rm" else 1
    rng = np.random.default_rng(zlib.crc32(case_name.encode("utf-8")) & 0xFFFFFFFF)

    def _draw(shape, kind):
        if np.issubdtype(np_dtype, np.integer):
            # Small positive values: add/sub/mul must not overflow the narrow dtypes.
            return rng.integers(1, 5, size=shape).astype(np_dtype)
        if op_name == "div":
            # The divisor stays bounded away from zero whichever operand it is.
            low, high = (0.5, 2.0) if kind == "expanded" else (1.0, 4.0)
            return rng.uniform(low, high, size=shape).astype(np_dtype)
        if op_name == "expdif":
            # Keep exp() in range.
            return rng.uniform(-1.0, 1.0, size=shape).astype(np_dtype)
        return rng.uniform(0.25, 2.0, size=shape).astype(np_dtype)

    full_shape, expanded_shape = (rows, cols), (rows, block)
    if expanded == "src0":
        return [_draw(expanded_shape, "expanded"), _draw(full_shape, "full")]
    return [_draw(full_shape, "full"), _draw(expanded_shape, "expanded")]


def _make_expected(op_name, dtype_key, mode, expanded, src0, src1):
    """Golden for both ISA modes and both expansion directions."""
    _pto_dtype, np_dtype = _DTYPE[dtype_key]
    full, expanded_operand = (src0, src1) if expanded == "src1" else (src1, src0)
    if mode == "rm":
        block = BLOCK_ELEMS[dtype_key]
        broadcast = expanded_operand[:, np.arange(full.shape[1]) % block]
    else:
        broadcast = np.broadcast_to(expanded_operand[:, :1], full.shape)
    op_fn = _OPS[op_name][1]
    # Keep the authored operand order: dst = src0 OP src1.
    if expanded == "src1":
        return op_fn(full, broadcast).astype(np_dtype)
    return op_fn(broadcast, full).astype(np_dtype)


def _tolerances(op_name, dtype_key):
    if np.issubdtype(_DTYPE[dtype_key][1], np.integer):
        return 0.0, 0.0
    if dtype_key == "f16":
        return 1e-2, 1e-2
    if op_name == "expdif":
        return 1e-4, 1e-4
    if op_name == "div":
        return 1e-5, 1e-5
    return 1e-6, 1e-6


# ── case matrix ───────────────────────────────────────────────────────────────

# (op, dtype, rows, cols, mode, qualifier[, expanded])
# Primary shape: f32 8x32 dst with an 8x8 expanded operand (B = 8).
_PRIMARY_SHAPE = (8, 32)

CASE_SPECS = []

# 1) All seven interfaces in Mode 2 (block repeat) ...
for _op in INTERFACES:
    CASE_SPECS.append((_op, "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "rm", ""))
# 2) ... each with a Mode 1 (ColMajor single column) control.
for _op in INTERFACES:
    CASE_SPECS.append((_op, "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "cm", ""))

# 3) Extra shape and dtype scenarios.
CASE_SPECS += [
    # dst cols spanning more than one 256-byte vector: 128 = 2 x 64 f32 lanes, so the
    # second column chunk must re-read the same 32-byte block without period drift.
    ("add", "f32", 8, 128, "rm", "multichunk"),
    # dst cols not a multiple of the vector width: 96 = 64 + 32, so the tail chunk is
    # partially masked while the block period still holds.
    ("add", "f32", 8, 96, "rm", "tail"),
    # Fewer than one vector of dst cols: 40 < 64, a single masked chunk.
    ("add", "f32", 8, 40, "rm", "tail_short"),
    # 16-bit block width (B = 16, 128 lanes per vector).
    ("add", "f16", 8, 64, "rm", "w16"),
    # 8-bit block width (B = 32, 256 lanes per vector).
    ("add", "i8", 8, 128, "rm", "w8"),
]

# 4) src0-expanded regression: src1 matches dst and src0 is the row-expanded operand.
#    The non-commutative ops pin the authored operand order (dst = src0 OP src1).
for _op in ("sub", "div", "expdif"):
    CASE_SPECS.append((_op, "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "rm", "", "src0"))
# ... plus the Mode 1 (single-column scalar) form of the same direction.
CASE_SPECS.append(("sub", "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "cm", "", "src0"))

# Entries without an explicit direction expand src1.
CASE_SPECS = [spec if len(spec) == 7 else (*spec, "src1") for spec in CASE_SPECS]


def _case_name(op, dtype_key, rows, cols, mode, qualifier, expanded="src1"):
    name = f"trowexpand{op}_{dtype_key}_{rows}x{cols}_{mode}"
    if expanded == "src0":
        name += "_src0"
    return f"{name}_{qualifier}" if qualifier else name


def _build_cases():
    cases = []
    for op, dtype_key, rows, cols, mode, qualifier, expanded in CASE_SPECS:
        name = _case_name(op, dtype_key, rows, cols, mode, qualifier, expanded)
        kernel = _make_kernel(name, op, dtype_key, rows, cols, mode, expanded)
        rtol, atol = _tolerances(op, dtype_key)

        def _inputs(_name=name, _op=op, _dt=dtype_key, _r=rows, _c=cols, _mode=mode,
                    _expanded=expanded):
            return _make_inputs(_name, _op, _dt, _r, _c, _mode, _expanded)

        def _expected(src0, src1, _op=op, _dt=dtype_key, _mode=mode, _expanded=expanded):
            return _make_expected(_op, _dt, _mode, _expanded, src0, src1)

        cases.append(
            golden_output_case(
                name,
                kernel,
                inputs=_inputs,
                expected=_expected,
                rtol=rtol,
                atol=atol,
            )
        )
    return cases


CASES = _build_cases()


auto_main(globals())
