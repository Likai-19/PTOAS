#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/cube-matmul/mad_acc."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// Copyright (c) 2026 Huawei Technologies Co., Ltd.\n// This program is free software, you can redistribute '
    'it and/or modify it under the terms and conditions of\n// CANN Open Software License Agreement Version 2.0 ('
    'the "License").\n// Please refer to the License for details. You may not use this file except in compliance '
    'with the License.\n// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER '
    'EXPRESS OR IMPLIED,\n// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PAR'
    'TICULAR PURPOSE.\n// See LICENSE in the root of the software repository for the full text of the License.\n'
    '\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<cube>} {\n  func.func @mad_a'
    'cc_kernel(%a_gm: !pto.ptr<f16, gm>,\n                            %b_gm: !pto.ptr<f16, gm>,\n                '
    '            %a_acc_gm: !pto.ptr<f16, gm>,\n                            %b_acc_gm: !pto.ptr<f16, gm>,\n      '
    '                      %c_gm: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %c0_i64 = arith.constant 0 : '
    'i64\n    %c1_i64 = arith.constant 1 : i64\n    %c16_i64 = arith.constant 16 : i64\n    %c32_i64 = arith.cons'
    'tant 32 : i64\n    %c512_i64 = arith.constant 512 : i64\n    %c1024_i64 = arith.constant 1024 : i64\n    %c1'
    '536_i64 = arith.constant 1536 : i64\n    %false = arith.constant false\n\n    // L1 is split into four 16x16'
    ' f16 tiles: the first pair computes the\n    // initial C = A * B, and the second pair is accumulated with m'
    'ad_acc.\n    %l1_a = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l1>\n    %l1_b = pto.castptr %c512_i64 : i64'
    ' -> !pto.ptr<f16, l1>\n    %l1_a_acc = pto.castptr %c1024_i64 : i64 -> !pto.ptr<f16, l1>\n    %l1_b_acc = pt'
    'o.castptr %c1536_i64 : i64 -> !pto.ptr<f16, l1>\n    %l0a = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l0a>'
    '\n    %l0b = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l0b>\n    %l0c = pto.castptr %c0_i64 : i64 -> !pto.p'
    'tr<f32, l0c>\n\n    // Step 1: load all four source matrices from GM to L1. nd2nz converts\n    // normal ro'
    'w-major GM tiles into the cube-friendly NZ layout.\n    pto.mte_gm_l1_frac %a_gm, %l1_a, nd2nz,\n        sha'
    'pe(%c16_i64, %c16_i64),\n        src_layout(%c32_i64),\n        dst_group(%c1_i64, %c1_i64, %c16_i64, %c0_i6'
    '4),\n        ctrl(%c0_i64, %false)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n        shape i64,'
    ' i64, src_layout(i64),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n    pto.mte_gm_l1_frac %b_gm, %l'
    '1_b, nd2nz,\n        shape(%c16_i64, %c16_i64),\n        src_layout(%c32_i64),\n        dst_group(%c1_i64, %'
    'c1_i64, %c16_i64, %c0_i64),\n        ctrl(%c0_i64, %false)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd'
    '2nz,\n        shape i64, i64, src_layout(i64),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n    pto.'
    'mte_gm_l1_frac %a_acc_gm, %l1_a_acc, nd2nz,\n        shape(%c16_i64, %c16_i64),\n        src_layout(%c32_i64'
    '),\n        dst_group(%c1_i64, %c1_i64, %c16_i64, %c0_i64),\n        ctrl(%c0_i64, %false)\n      : !pto.ptr'
    '<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n        shape i64, i64, src_layout(i64),\n        dst_group i64, i64, '
    'i64, i64, ctrl i64, i1\n    pto.mte_gm_l1_frac %b_acc_gm, %l1_b_acc, nd2nz,\n        shape(%c16_i64, %c16_i6'
    '4),\n        src_layout(%c32_i64),\n        dst_group(%c1_i64, %c1_i64, %c16_i64, %c0_i64),\n        ctrl(%c'
    '0_i64, %false)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n        shape i64, i64, src_layout(i64'
    '),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_'
    'ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    // Step 2: feed the first A/B pair in'
    'to L0A/L0B. B is transposed because\n    // the MAD instruction consumes the right-hand operand from L0B in '
    'that form.\n    pto.mte_l1_l0a %l1_a, %l0a, %c16_i64, %c16_i64, %c0_i64, %c0_i64\n      : !pto.ptr<f16, l1>,'
    ' !pto.ptr<f16, l0a>, i64, i64, i64, i64\n    pto.mte_l1_l0b %l1_b, %l0b, %c16_i64, %c16_i64, %c0_i64, %c0_i6'
    '4 {transpose = true}\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["P'
    'IPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    // Step 3: com'
    'pute the base product in L0C:\n    //   L0C = f32(A * B)\n    // check_only leaves the accumulation buffer a'
    'vailable for the next MAD.\n    pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64 unit_flag(check_only)'
    ' disable_gemv n_dir\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64\n\n  '
    '  pto.set_flag["PIPE_M", "PIPE_MTE1", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", "PIPE_MTE1", "EVENT_ID1"]\n '
    '   // Step 4: load the second A/B pair into L0A/L0B after the first MAD has\n    // produced L0C, so the nex'
    't instruction can accumulate into the same L0C.\n    pto.mte_l1_l0a %l1_a_acc, %l0a, %c16_i64, %c16_i64, %c0'
    '_i64, %c0_i64\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64\n    pto.mte_l1_l0b %l1_b_a'
    'cc, %l0b, %c16_i64, %c16_i64, %c0_i64, %c0_i64 {transpose = true}\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, '
    'l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID1"]\n    pto.wait_flag["PIPE_MTE1'
    '", "PIPE_M", "EVENT_ID1"]\n    // Step 5: accumulate the second product:\n    //   L0C += f32(A_acc * B_acc)'
    '\n    // check_and_set marks the accumulated result ready for FIX/writeback.\n    pto.mad_acc %l0a, %l0b, %l'
    '0c, %c16_i64, %c16_i64, %c16_i64 unit_flag(check_and_set) disable_gemv n_dir\n      : !pto.ptr<f16, l0a>, !p'
    'to.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n'
    '    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n\n    // Step 6: convert L0C back from NZ to normal la'
    'yout and store the final\n    // f32 matrix C to GM.\n    pto.mte_l0c_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c'
    '16_i64, %c16_i64,\n      %c0_i64, %c0_i64,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64,'
    ' i64, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
)

def _golden_scope():
    _files = {}

    output_dir = None

    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.


    import argparse
    from pathlib import Path
    import numpy as np

    M = 16
    N = 16
    K = 16


    def _generate(seed: int) -> dict:
        row = np.arange(M, dtype=np.float32).reshape(M, 1)
        col = np.arange(K, dtype=np.float32).reshape(1, K)
        a = (((row * 3 + col * 5) % 17) - 8).astype(np.float16) / np.float16(4.0)
        k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
        n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
        b = (((k_idx * 7 - n_idx * 2) % 19) - 9).astype(np.float16) / np.float16(5.0)
        a_acc = (((row * 2 - col * 7) % 13) - 6).astype(np.float16) / np.float16(3.0)
        b_acc = (((k_idx * 11 + n_idx * 3) % 17) - 8).astype(np.float16) / np.float16(4.0)
        c = np.zeros((M, N), dtype=np.float32)
        golden_c = a.astype(np.float32) @ b.astype(np.float32)
        golden_c += a_acc.astype(np.float32) @ b_acc.astype(np.float32)

        _files["v1.bin"] = a.reshape(-1)
        _files["v2.bin"] = b.reshape(-1)
        _files["v3.bin"] = c.reshape(-1)
        _files["v4.bin"] = a_acc.reshape(-1)
        _files["v5.bin"] = b_acc.reshape(-1)
        _files["golden_v3.bin"] = golden_c.reshape(-1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        args = parser.parse_args()
        generate(args.output_dir)


    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.


    import os
    import sys

    import numpy as np


    def compare_bin(golden_path: str, output_path: str) -> bool:
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
          return False
        golden = np.fromfile(golden_path, dtype=np.float32)
        output = np.fromfile(output_path, dtype=np.float32)
        if golden.shape != output.shape:
            print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
            return False
        if np.allclose(golden, output, atol=1e-2, rtol=1e-2):
            return True
        diff = np.where(np.abs(golden - output) > (1e-2 + 1e-2 * np.abs(golden)))[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] first mismatch at idx={idx}: golden={float(golden[idx])}, out={float(output[idx])}")
        return False


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_bin("golden_v3.bin", "v3.bin")
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = None
COMPARE_PAIRS = [
    (
        'golden_v3.bin',
        'v3.bin'
    )
]

NUMPY_DTYPE = {
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'si8': 'int8',
    'si16': 'int16',
    'si32': 'int32',
    'si64': 'int64',
    'ui8': 'uint8',
    'ui16': 'uint16',
    'ui32': 'uint32',
    'ui64': 'uint64',
    'f16': 'float16',
    'f32': 'float32',
    'f64': 'float64',
    'bf16': 'uint16',
    'hif8': 'uint8',
    'f8E4M3FN': 'uint8',
    'f8E5M2': 'uint8',
    'f8e4m3': 'uint8',
    'f8e5m2': 'uint8',
    'f8e8m0': 'uint8',
    'f8E8M0FNU': 'uint8',
    'f4E2M1x2': 'uint8',
    'f4E1M2x2': 'uint8',
    'hif8x2': 'uint16',
    'f8e4m3x2': 'uint16',
    'f8e5m2x2': 'uint16',
    'f8e4m3x4': 'uint32',
    'f8e4m3x8': 'uint64',
    'f8e5m2x4': 'uint32',
    'f8e5m2x8': 'uint64',
    'f16x2': 'uint32',
    'bf16x2': 'uint32',
    'i8x2': 'uint16',
    'i16x2': 'uint32',
    'i32x2': 'uint64',
    'f32x2': 'uint64'
}


def _as_buffer_array(value, numpy_dtype):
    """Match a golden array to the raw bytes a kernel element type expects.

    A golden may build its array with ml_dtypes types (bfloat16) while the
    kernel argument is a plain numpy type (uint16); torch.from_numpy rejects
    the former, so reinterpret the bytes when the element widths agree.
    """
    array = np.asarray(value)
    want = np.dtype(numpy_dtype)
    if array.dtype == want:
        return array
    # The golden bytes are the device buffer; reinterpret them whenever the
    # byte count divides evenly so a narrow golden (for example uint8 init or
    # an f16 source) still fills the element type the kernel argument expects
    # instead of shrinking or doubling the buffer through astype.
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.nbytes % want.itemsize == 0:
        return flat.view(want)
    return array.astype(want)

BUFFERS = [
    (
        'v1',
        'f16',
        'a_gm'
    ),
    (
        'v2',
        'f16',
        'b_gm'
    ),
    (
        'v4',
        'f16',
        'a_acc_gm'
    ),
    (
        'v5',
        'f16',
        'b_acc_gm'
    ),
    (
        'v3',
        'f32',
        'c_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='mad_acc_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    a_gm: pto.ptr(pto.f16, "gm"),
    b_gm: pto.ptr(pto.f16, "gm"),
    a_acc_gm: pto.ptr(pto.f16, "gm"),
    b_acc_gm: pto.ptr(pto.f16, "gm"),
    c_gm: pto.ptr(pto.f32, "gm"),
):
    pass


class _TrackedArray(np.ndarray):
    """An array that records the positions written through __setitem__."""

    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        obj._write_mask = np.zeros(obj.shape, dtype=bool)
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self._write_mask = getattr(source, "_write_mask", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        mask = getattr(self, "_write_mask", None)
        if mask is None or mask.shape != self.shape:
            mask = np.zeros(self.shape, dtype=bool)
            self._write_mask = mask
        mask[key] = True


def _materialize():
    """Run the case's own golden generator in memory; no .bin files.

    The golden's output buffers are created through the numpy constructors and
    then filled in place, so wrapping those constructors records exactly which
    positions the golden defines; the rest are don't care for the compare.
    """
    trackers = {name: getattr(np, name) for name in ("zeros", "full", "empty", "ones")}

    building = [False]

    def _track(name):
        original = trackers[name]

        def wrapped(*args, **kwargs):
            if building[0]:
                return original(*args, **kwargs)
            building[0] = True
            try:
                tracked = _TrackedArray(original(*args, **kwargs))
                if name != "empty":
                    # zeros/full/ones initialize every element, so the whole
                    # buffer is defined and compared; only np.empty leaves
                    # elements undefined, where the __setitem__ trace defines
                    # the valid mask.
                    tracked._write_mask = np.ones(tracked.shape, dtype=bool)
                return tracked
            finally:
                building[0] = False

        return wrapped

    for name in trackers:
        setattr(np, name, _track(name))
    try:
        namespace = _golden_scope()
        if "_generate" in namespace:
            seed = namespace[SEED_EXPR] if SEED_EXPR else 0
            namespace["_generate"](seed)
        else:
            sys.argv = ["golden.py"]
            namespace["main"]()
    finally:
        for name, original in trackers.items():
            setattr(np, name, original)
    produced = namespace["_files"]
    masks = {}
    for key, value in produced.items():
        if not key.startswith("golden_"):
            continue
        mask = getattr(value, "_write_mask", None)
        if mask is not None and np.asarray(mask).size and bool(np.asarray(mask).any()):
            masks[key] = np.asarray(mask).reshape(-1).copy()
        else:
            # a computed output defines every element; there is no partial
            # in-place write to record, so the whole file is compared.
            masks[key] = np.ones(np.asarray(value).size, dtype=bool)
    if masks:
        produced["@write_mask"] = masks
    return produced


def _apply_write_masks(files, produced):
    """Restrict each golden/output comparison to the golden's written set."""
    masks = produced.get("@write_mask") if hasattr(produced, "get") else None
    if not masks:
        return
    for golden_key, mask in masks.items():
        if not golden_key.startswith("golden_"):
            continue
        output_key = golden_key[len("golden_"):]
        if golden_key not in files or output_key not in files:
            continue
        golden = np.asarray(files[golden_key])
        output = np.asarray(files[output_key]).reshape(-1)
        flat = golden.reshape(-1)
        written = np.asarray(mask, dtype=bool).reshape(-1)
        size = min(flat.size, output.size, written.size)
        if size == 0 or not written[:size].any():
            continue
        adjusted = flat.copy()
        donotcare = ~written[:size]
        if output.dtype != flat.dtype and output.dtype.itemsize == flat.dtype.itemsize:
            # Same byte width but a different element type (for example the
            # kernel typed the buffer f16 while the golden built uint16): copy
            # the raw bytes so the donot-care fill stays bit-exact instead of
            # rounding through a numeric cast such as f16(-1.0) -> 0xffff.
            output = np.ascontiguousarray(output).view(flat.dtype)
        adjusted[:size][donotcare] = output[:size][donotcare]
        files[golden_key] = adjusted.reshape(golden.shape)


def _run_compare(files):
    """Run the case's own compare.py against in-memory arrays."""
    with tempfile.TemporaryDirectory(
            prefix="legacy-helpers-", dir=os.environ.get("TMPDIR")) as _helpers:
        _run_compare_in(files, Path(_helpers))


def _run_compare_in(files, helper_dir):
    for name, text in HELPERS.items():
        (helper_dir / name).write_text(text)

    real_fromfile, real_exists, real_argv = np.fromfile, os.path.exists, sys.argv
    real_exists_path = Path.exists
    real_environ, real_path = dict(os.environ), list(sys.path)

    def _fromfile(path, dtype=None, count=-1, offset=0, *args, **kwargs):
        name = os.path.basename(str(path))
        if name not in files:
            raise FileNotFoundError(name)
        # The legacy compare read the raw .bin bytes; reinterpret the stored
        # array's bytes instead of casting, so a deep-merged buffer written
        # through one element type still reads back as the type compare asks
        # for. count/offset mirror np.fromfile's element prefix semantics.
        array = np.asarray(files[name])
        data = array.tobytes()
        if offset:
            data = data[int(offset):]
        if dtype is None:
            dtype = array.dtype
        itemsize = np.dtype(dtype).itemsize
        available = len(data) // itemsize
        if count is None or int(count) < 0 or int(count) > available:
            count = available
        return np.frombuffer(data, dtype=dtype, count=count)

    np.fromfile = _fromfile
    os.path.exists = lambda p: os.path.basename(str(p)) in files or real_exists(p)
    Path.exists = lambda self: os.path.basename(str(self)) in files or real_exists_path(self)
    sys.argv = ["compare.py"]
    sys.path.insert(0, str(helper_dir))
    os.environ["COMPARE_STRICT"] = "1"
    try:
        namespace = _compare_scope()
        if "main" in namespace:
            namespace["main"]()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise AssertionError("compare.py exited with %r" % (exc.code,))
    finally:
        np.fromfile, os.path.exists, Path.exists, sys.argv = (
            real_fromfile, real_exists, real_exists_path, real_argv)
        os.environ.clear()
        os.environ.update(real_environ)
        sys.path[:] = real_path


def make_case():
    import torch
    produced = _materialize()
    output_to_golden = {out: gold for gold, out in COMPARE_PAIRS}
    arrays = []
    for name, ty, _slot in BUFFERS:
        if name.startswith("@same:"):
            arrays.append(np.asarray(produced["%s.bin" % name[len("@same:"):]]))
            continue
        if name.startswith("@scratch:"):
            arrays.append(np.empty_like(np.asarray(produced["%s.bin" % name[len("@scratch:"):]]),
                                        dtype=np.dtype(NUMPY_DTYPE[ty])))
            continue
        key = "%s.bin" % name
        if key in produced:
            arrays.append(_as_buffer_array(produced[key], NUMPY_DTYPE[ty]))
            continue
        golden_key = output_to_golden.get(key)
        if golden_key is None or golden_key not in produced:
            raise RuntimeError("golden did not produce %s" % key)
        arrays.append(np.zeros_like(np.asarray(produced[golden_key]),
                                    dtype=np.dtype(NUMPY_DTYPE[ty])))
    values = list(SCALARS)
    for _source, _ty in zip(SCALAR_SOURCES, SCALAR_DTYPES):
        _array = np.asarray(produced["%s.bin" % _source]).reshape(-1)
        values.append(float(_array[0]) if _ty == "f32" else int(_array[0]))
    if GRID_FROM_VALUES is not None:
        CASES[0]["grid"] = GRID_FROM_VALUES(values)
    return arrays, produced, values


def check_case(device_inputs, produced):
    if COMPARE_PAIRS:
        files = {gold: np.asarray(produced[gold]) for gold, _out in COMPARE_PAIRS if gold in produced}
    else:
        # no explicit pairs: the case compares whole files, so expose every
        # array the golden built (its .tofile names became dictionary keys).
        files = {name: np.asarray(value) for name, value in produced.items()
                 if not name.startswith("@")}
    if not files:
        raise AssertionError("golden.py produced no golden output")
    for _source in SCALAR_SOURCES:
        _key = "%s.bin" % _source
        if _key in produced:
            files[_key] = np.asarray(produced[_key])
    for (name, _ty, _slot), tensor in zip(BUFFERS, device_inputs):
        if not name.startswith("@"):
            files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_cube_matmul_mad_acc',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
