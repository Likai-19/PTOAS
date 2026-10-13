#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/predicate-load-store/psts-pk-plds-us.

Fallback style for a case whose golden cannot run in memory and whose C++
wrapper binds one host buffer to several kernel arguments: the case's own
golden/compare scripts run in a work directory and the aliased device tensors
are launched through ARG_TO_BUFFER.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @pst'
    's_pk_plds_us_deep_merged_kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<ui8, gm>, %arg2: !pto.ptr<ui8, gm>'
    ', %arg3: !pto.ptr<f32, gm>, %arg4: !pto.ptr<ui8, gm>, %arg5: !pto.ptr<ui8, gm>) attributes {pto.kernel} {\n '
    '   %__deep_merge_guard = arith.constant false\n    // inactive merged from psts_pk_plds_us_prefix_boundary_k'
    'ernel_2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m0 = arith.constant 0 : index\n    %c1_m0 = arith.cons'
    'tant 1 : index\n    %c16_m0 = arith.constant 16 : index\n    %c173_m0 = arith.constant 173 : i32\n    %c0_i6'
    '4_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c32_i64_m0 = arith.constant 32 '
    ': i64\n    %c8192_i64_m0 = arith.constant 8192 : i64\n    %c10240_i64_m0 = arith.constant 10240 : i64\n    %'
    'false_m0 = arith.constant false\n\n    %ub_mid_m0 = pto.castptr %c8192_i64_m0 : i64 -> !pto.ptr<ui8, ub>\n  '
    '  %ub_out_m0 = pto.castptr %c10240_i64_m0 : i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg1, %ub_mid_m0, %'
    'c0_i64_m0, %c32_i64_m0\n      nburst(%c32_i64_m0, %c32_i64_m0, %c32_i64_m0)\n      : !pto.ptr<ui8, gm>, !pto'
    '.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait'
    '_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m0:1 = scf.for %iv_m0 = %c0_m0 to '
    '%c1_m0 step %c1_m0 iter_args(%remaining_m0 = %c173_m0) -> (i32) {\n        %byte_offset_m0 = arith.addi %iv_'
    'm0, %c16_m0 : index\n        %src_m0, %next_m0 = pto.plt_b8 %remaining_m0 : i32 -> !pto.mask<b8>, i32\n     '
    '   pto.psts %src_m0, %ub_mid_m0[%byte_offset_m0], "PK" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n        pt'
    'o.mem_bar "VST_VLD"\n        %loaded_m0 = pto.plds %ub_mid_m0[%byte_offset_m0], "US" : !pto.ptr<ui8, ub>, in'
    'dex -> !pto.mask<b8>\n        pto.psts %loaded_m0, %ub_out_m0[%c0_m0], "NORM" : !pto.mask<b8>, !pto.ptr<ui8,'
    ' ub>, index\n        scf.yield %next_m0 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EV'
    'ENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.set_loop1_stride_ubtoout %c32_i64_'
    'm0, %c32_i64_m0 : i64, i64\n    pto.set_loop2_stride_ubtoout %c32_i64_m0, %c32_i64_m0 : i64, i64\n    pto.mt'
    'e_ub_gm %ub_out_m0, %arg2, %c32_i64_m0\n      nburst(%c32_i64_m0, %c32_i64_m0, %c32_i64_m0)\n      : !pto.pt'
    'r<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // acti'
    've merged from psts_pk_plds_us_kernel_2d\n\n    %c0_m1 = arith.constant 0 : index\n    %c1_m1 = arith.consta'
    'nt 1 : index\n    %c16_m1 = arith.constant 16 : index\n    %c171_m1 = arith.constant 171 : i32\n    %c0_i64_'
    'm1 = arith.constant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.constant 32 : '
    'i64\n    %c8192_i64_m1 = arith.constant 8192 : i64\n    %c10240_i64_m1 = arith.constant 10240 : i64\n    %fa'
    'lse_m1 = arith.constant false\n\n    %ub_mid_m1 = pto.castptr %c8192_i64_m1 : i64 -> !pto.ptr<ui8, ub>\n    '
    '%ub_out_m1 = pto.castptr %c10240_i64_m1 : i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg4, %ub_mid_m1, %c0'
    '_i64_m1, %c32_i64_m1\n      nburst(%c32_i64_m1, %c32_i64_m1, %c32_i64_m1)\n      : !pto.ptr<ui8, gm>, !pto.p'
    'tr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_f'
    'lag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m1:1 = scf.for %iv_m1 = %c0_m1 to %c'
    '1_m1 step %c1_m1 iter_args(%remaining_m1 = %c171_m1) -> (i32) {\n        %byte_offset_m1 = arith.addi %iv_m1'
    ', %c16_m1 : index\n        %src_m1, %next_m1 = pto.plt_b8 %remaining_m1 : i32 -> !pto.mask<b8>, i32\n       '
    ' pto.psts %src_m1, %ub_mid_m1[%byte_offset_m1], "PK" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n        pto.'
    'mem_bar "VST_VLD"\n        %loaded_m1 = pto.plds %ub_mid_m1[%byte_offset_m1], "US" : !pto.ptr<ui8, ub>, inde'
    'x -> !pto.mask<b8>\n        pto.psts %loaded_m1, %ub_out_m1[%c0_m1], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, u'
    'b>, index\n        scf.yield %next_m1 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.set_loop1_stride_ubtoout %c32_i64_m1'
    ', %c32_i64_m1 : i64, i64\n    pto.set_loop2_stride_ubtoout %c32_i64_m1, %c32_i64_m1 : i64, i64\n    pto.mte_'
    'ub_gm %ub_out_m1, %arg5, %c32_i64_m1\n      nburst(%c32_i64_m1, %c32_i64_m1, %c32_i64_m1)\n      : !pto.ptr<'
    'ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    return\n  }\n}\n'
)

def _golden_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/predicate-load-store/psts-pk-plds-us
    # family: predicate-load-store
    # target_ops: pto.plds, pto.psts
    # scenarios: predicate-load-store-composition, dynamic-offset, load-store-pair-preservation,
    # representative-logical-elements

    import argparse
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from _predicate_load_store_case import pk_us_compose, prefix_bits, write_case


    SEED = 19
    ACTIVE_BITS = 171


    def generate(output_dir: Path, seed: int) -> None:
        del seed
        write_case(output_dir, pk_us_compose(prefix_bits(ACTIVE_BITS)))


    def main() -> None:
        parser = argparse.ArgumentParser(description="Generate inputs/golden for psts-pk-plds-us.")
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()
        generate(args.output_dir, args.seed)


    if __name__ == "__main__":
        main()
    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/predicate-load-store/psts-pk-plds-us
    # family: predicate-load-store
    # target_ops: pto.plds, pto.psts
    # scenarios: predicate-load-store-composition, dynamic-offset, load-store-pair-preservation,
    # representative-logical-elements

    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from _predicate_load_store_case import compare_norm_store


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_norm_store("golden_v3.bin", "v3.bin")
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {
    '_predicate_load_store_case.py': (
        '#!/usr/bin/env python3\n# Copyright (c) 2026 Huawei Technologies Co., Ltd.\n# This program is free softwar'
        'e, you can redistribute it and/or modify it under the terms and conditions of\n# CANN Open Software Licens'
        'e Agreement Version 2.0 (the "License").\n# Please refer to the License for details. You may not use this '
        'file except in compliance with the License.\n# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARR'
        'ANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTA'
        'BILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n# See LICENSE in the root of the software repository for the'
        ' full text of the License.\n\n\nfrom pathlib import Path\n\nimport numpy as np\n\n\nROWS = 32\nCOLS = 32\n'
        'OUTPUT_BYTES = ROWS * COLS\nPREDICATE_BITS = 256\n# For the current A5 predicate load/store surface used b'
        'y these composition\n# cases, the user-visible packed NORM footprint is 16 bytes. Bytes beyond that\n# ran'
        'ge are not part of the checked result footprint.\nNORM_STORAGE_BYTES = 16\n\n\ndef prefix_bits(active_bits'
        ': int) -> np.ndarray:\n    bits = np.zeros((PREDICATE_BITS,), dtype=np.uint8)\n    bits[:active_bits] = 1'
        '\n    return bits\n\n\ndef pk_us_compose(bits: np.ndarray) -> np.ndarray:\n    packed = bits[::2]\n    ret'
        'urn np.repeat(packed, 2).astype(np.uint8, copy=False)\n\n\ndef norm_ds_compose(bits: np.ndarray) -> np.nda'
        'rray:\n    source = np.concatenate(\n        [bits.astype(np.uint8, copy=False), np.zeros_like(bits, dtype'
        '=np.uint8)]\n    )\n    return source[::2][:PREDICATE_BITS].astype(np.uint8, copy=False)\n\n\ndef norm_sto'
        're_bytes(bits: np.ndarray) -> np.ndarray:\n    packed = np.packbits(bits.astype(np.uint8, copy=False), bit'
        'order="little")\n    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)\n    out[:NORM_STORAGE_BYTES] = packe'
        'd[:NORM_STORAGE_BYTES]\n    return out\n\n\ndef write_default_inputs(output_dir: Path) -> None:\n    np.ze'
        'ros((ROWS * COLS,), dtype=np.float32).tofile(output_dir / "v1.bin")\n    np.zeros((ROWS * COLS,), dtype=np'
        '.float32).tofile(output_dir / "v2.bin")\n    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir /'
        ' "v3.bin")\n\n\ndef write_case(output_dir: Path, bits: np.ndarray) -> None:\n    output_dir.mkdir(parents='
        'True, exist_ok=True)\n    write_default_inputs(output_dir)\n    norm_store_bytes(bits).tofile(output_dir /'
        ' "golden_v3.bin")\n\n\ndef compare_norm_store(golden_path: str, output_path: str) -> bool:\n    golden = n'
        'p.fromfile(golden_path, dtype=np.uint8)\n    output = np.fromfile(output_path, dtype=np.uint8)\n    if gol'
        'den.size < NORM_STORAGE_BYTES or output.size < NORM_STORAGE_BYTES:\n        return False\n    if not np.ar'
        'ray_equal(golden[:NORM_STORAGE_BYTES], output[:NORM_STORAGE_BYTES]):\n        diff = np.nonzero(golden[:NO'
        'RM_STORAGE_BYTES] != output[:NORM_STORAGE_BYTES])[0]\n        idx = int(diff[0]) if diff.size else 0\n    '
        '    print(\n            f"[ERROR] Mismatch (predicate load/store composition): idx={idx} "\n            f"'
        'golden={int(golden[idx])} out={int(output[idx])}"\n        )\n        return False\n    return True\n'
    )
}

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


def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='psts_pk_plds_us_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.ui8, "gm"),
    arg2: pto.ptr(pto.ui8, "gm"),
    arg3: pto.ptr(pto.f32, "gm"),
    arg4: pto.ptr(pto.ui8, "gm"),
    arg5: pto.ptr(pto.ui8, "gm"),
):
    pass


def _run_legacy_script(scope, cwd):
    """Run one inlined legacy script in-process under the legacy context."""
    for name, text in HELPERS.items():
        (Path(cwd) / name).write_text(text)
    real_argv, real_path = list(sys.argv), list(sys.path)
    real_environ, real_cwd = dict(os.environ), os.getcwd()
    sys.argv = ["_legacy_script.py"]
    sys.path.insert(0, str(cwd))
    os.environ["COMPARE_STRICT"] = "1"
    os.chdir(str(cwd))
    try:
        namespace = scope()
        if "_generate" in namespace:
            namespace["_generate"](0)
        elif "main" in namespace:
            namespace["main"]()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError("legacy script exited with %r" % (exc.code,))
    finally:
        os.chdir(real_cwd)
        sys.argv = real_argv
        sys.path[:] = real_path
        os.environ.clear()
        os.environ.update(real_environ)


def make_case():
    import torch
    work_dir = tempfile.TemporaryDirectory(
        prefix="legacy-case-", dir=os.environ.get("TMPDIR"))
    work = Path(work_dir.name)
    _run_legacy_script(_golden_scope, work)
    base, per_arg = {}, []
    for index, ty in zip(ARG_TO_BUFFER, ARG_DTYPES):
        name, buf_ty = BUFFER_FILES[index]
        if index not in base:
            array = np.fromfile(work / (name + ".bin"), dtype=np.dtype(NUMPY_DTYPE[buf_ty]))
            base[index] = torch.from_numpy(np.ascontiguousarray(array)).to("npu:0")
        tensor = base[index]
        want = _dtype_map()[NUMPY_DTYPE[ty]]
        if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
            tensor = tensor.view(want)
        per_arg.append(tensor)
    return per_arg, work_dir, list(SCALARS)


def check_case(device_inputs, work_dir):
    work = Path(work_dir.name)
    try:
        seen = {}
        for index, tensor in zip(ARG_TO_BUFFER, device_inputs):
            seen.setdefault(index, tensor)
        for index, tensor in seen.items():
            name, _ty = BUFFER_FILES[index]
            (Path(work) / (name + ".bin")).write_bytes(tensor.cpu().numpy().tobytes())
        _run_legacy_script(_compare_scope, work)
    finally:
        work_dir.cleanup()


BUFFER_FILES = [
    (
        'v1',
        'f32'
    ),
    (
        'v2',
        'ui8'
    ),
    (
        'v3',
        'ui8'
    )
]
ARG_TO_BUFFER = [
    0,
    0,
    0,
    0,
    1,
    2
]
ARG_DTYPES = [
    'f32',
    'ui8',
    'ui8',
    'f32',
    'ui8',
    'ui8'
]
SCALARS = []

GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_predicate_load_store_psts_pk_plds_us',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
