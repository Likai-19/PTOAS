#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/predicate-load-store/psti-norm-pldi-ds.

Fallback style: the case's own golden.py/compare.py run in a work directory,
exactly as the legacy runner did.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// -----------------------------------------------------------------------------\n// case: micro-op/predicat'
    'e-load-store/psti-norm-pldi-ds\n// family: predicate-load-store\n// target_ops: pto.pldi, pto.psti\n// scena'
    'rios: predicate-load-store-composition, immediate-offset, load-store-pair-preservation, representative-logic'
    'al-elements\n// -----------------------------------------------------------------------------\nmodule attrib'
    'utes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @psti_norm_pldi_ds_'
    'kernel_2d(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<ui8, gm>,\n                                         %arg'
    '2: !pto.ptr<ui8, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constan'
    't 1 : index\n    %c32 = arith.constant 32 : index\n    %c143 = arith.constant 143 : i32\n    %c0_i64 = arith'
    '.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c8192_'
    'i64 = arith.constant 8192 : i64\n    %c10240_i64 = arith.constant 10240 : i64\n    %false = arith.constant f'
    'alse\n\n    %ub_mid = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_out = pto.castptr %c10240_i'
    '64 : i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg1, %ub_mid, %c0_i64, %c32_i64\n      nburst(%c32_i64, %'
    'c32_i64, %c32_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_fla'
    'g["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecs'
    'cope {\n      %_:1 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%remaining = %c143) -> (i32) {\n        %sr'
    'c, %next = pto.plt_b8 %remaining : i32 -> !pto.mask<b8>, i32\n        pto.psti %src, %ub_mid[%c32], "NORM" :'
    ' !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n        pto.mem_bar "VST_VLD"\n        %loaded = pto.pldi %ub_mid['
    '%c32], "DS" : !pto.ptr<ui8, ub>, index -> !pto.mask<b8>\n        pto.psts %loaded, %ub_out[%c0], "NORM" : !p'
    'to.mask<b8>, !pto.ptr<ui8, ub>, index\n        scf.yield %next : i32\n      }\n    }\n\n    pto.set_flag["PI'
    'PE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.set_loop1_s'
    'tride_ubtoout %c32_i64, %c32_i64 : i64, i64\n    pto.set_loop2_stride_ubtoout %c32_i64, %c32_i64 : i64, i64'
    '\n    pto.mte_ub_gm %ub_out, %arg2, %c32_i64\n      nburst(%c32_i64, %c32_i64, %c32_i64)\n      : !pto.ptr<u'
    'i8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard'
    ' = arith.constant false\n\n    // inactive cross-case merge from micro-op/predicate-load-store/psts-norm-pld'
    's-ds\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg10_1 = arith.constant 0 : index\n          '
    '%c1_cmg10_1 = arith.constant 1 : index\n          %c32_cmg10_1 = arith.constant 32 : index\n          %c64_c'
    'mg10_1 = arith.constant 64 : index\n          %c175_cmg10_1 = arith.constant 175 : i32\n          %c0_i32_cm'
    'g10_1 = arith.constant 0 : i32\n          %c0_i64_cmg10_1 = arith.constant 0 : i64\n          %c1_i64_cmg10_'
    '1 = arith.constant 1 : i64\n          %c32_i64_cmg10_1 = arith.constant 32 : i64\n          %c8192_i64_cmg10'
    '_1 = arith.constant 8192 : i64\n          %c10240_i64_cmg10_1 = arith.constant 10240 : i64\n          %false'
    '_cmg10_1 = arith.constant false\n      \n          %ub_mid_cmg10_1 = pto.castptr %c8192_i64_cmg10_1 : i64 ->'
    ' !pto.ptr<ui8, ub>\n          %ub_out_cmg10_1 = pto.castptr %c10240_i64_cmg10_1 : i64 -> !pto.ptr<ui8, ub>\n'
    '          pto.mte_gm_ub %arg1, %ub_mid_cmg10_1, %c0_i64_cmg10_1, %c32_i64_cmg10_1\n            nburst(%c32_i'
    '64_cmg10_1, %c32_i64_cmg10_1, %c32_i64_cmg10_1)\n            : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i6'
    '4, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_fla'
    'g["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__cmg10_1:1 = scf.for '
    '%iv_cmg10_1 = %c0_cmg10_1 to %c1_cmg10_1 step %c1_cmg10_1 iter_args(%remaining_cmg10_1 = %c175_cmg10_1) -> ('
    'i32) {\n              %byte_offset_cmg10_1 = arith.addi %iv_cmg10_1, %c32_cmg10_1 : index\n              %sr'
    'c_cmg10_1, %next_cmg10_1 = pto.plt_b8 %remaining_cmg10_1 : i32 -> !pto.mask<b8>, i32\n              %zero_cm'
    'g10_1, %_unused_cmg10_1 = pto.plt_b8 %c0_i32_cmg10_1 : i32 -> !pto.mask<b8>, i32\n              pto.psts %sr'
    'c_cmg10_1, %ub_mid_cmg10_1[%byte_offset_cmg10_1], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n        '
    '      pto.psts %zero_cmg10_1, %ub_mid_cmg10_1[%c64_cmg10_1], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, inde'
    'x\n              pto.mem_bar "VST_VLD"\n              %loaded_cmg10_1 = pto.plds %ub_mid_cmg10_1[%byte_offse'
    't_cmg10_1], "DS" : !pto.ptr<ui8, ub>, index -> !pto.mask<b8>\n              pto.psts %loaded_cmg10_1, %ub_ou'
    't_cmg10_1[%c0_cmg10_1], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              scf.yield %next_cmg1'
    '0_1 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n '
    '         pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.set_loop1_stride_ubtoout %c32_i64_'
    'cmg10_1, %c32_i64_cmg10_1 : i64, i64\n          pto.set_loop2_stride_ubtoout %c32_i64_cmg10_1, %c32_i64_cmg1'
    '0_1 : i64, i64\n          pto.mte_ub_gm %ub_out_cmg10_1, %arg2, %c32_i64_cmg10_1\n            nburst(%c32_i6'
    '4_cmg10_1, %c32_i64_cmg10_1, %c32_i64_cmg10_1)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64'
    ', i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n    return\n  }\n}\n'
)

def _golden_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/predicate-load-store/psti-norm-pldi-ds
    # family: predicate-load-store
    # target_ops: pto.pldi, pto.psti
    # scenarios: predicate-load-store-composition, immediate-offset, load-store-pair-preservation,
    # representative-logical-elements

    import argparse
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from _predicate_load_store_case import norm_ds_compose, prefix_bits, write_case


    SEED = 19
    ACTIVE_BITS = 143


    def generate(output_dir: Path, seed: int, src_elem_bytes: int) -> None:
        del seed
        del src_elem_bytes
        write_case(output_dir, norm_ds_compose(prefix_bits(ACTIVE_BITS)))


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate numpy-based inputs/golden for psti-norm-pldi-ds."
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("."),
            help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
        )
        parser.add_argument("--seed", type=int, default=SEED, help="Numpy random seed.")
        parser.add_argument(
            "--src-elem-bytes",
            type=int,
            default=4,
            help="Unused compatibility option kept for the shared runner surface.",
        )
        args = parser.parse_args()
        generate(args.output_dir, args.seed, args.src_elem_bytes)


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

    # case: micro-op/predicate-load-store/psti-norm-pldi-ds
    # family: predicate-load-store
    # target_ops: pto.pldi, pto.psti
    # scenarios: predicate-load-store-composition, immediate-offset, load-store-pair-preservation,
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
        'f32',
        'arg0'
    ),
    (
        'v2',
        'ui8',
        'arg1'
    ),
    (
        'v3',
        'ui8',
        'arg2'
    )
]
SCALARS = []


@pto.jit(
    name='psti_norm_pldi_ds_kernel_2d',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.ui8, "gm"),
    arg2: pto.ptr(pto.ui8, "gm"),
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
    work_dir = tempfile.TemporaryDirectory(
        prefix="legacy-case-", dir=os.environ.get("TMPDIR"))
    work = Path(work_dir.name)
    _run_legacy_script(_golden_scope, work)
    arrays = [np.fromfile(work / (name + ".bin"), dtype=np.dtype(NUMPY_DTYPE[ty]))
              for name, ty, _slot in BUFFERS]
    return [*arrays, *[np.asarray(v) for v in SCALARS]], work_dir


def check_case(device_inputs, work_dir):
    work = Path(work_dir.name)
    try:
        for (name, ty, _slot), tensor in zip(BUFFERS, device_inputs):
            tensor.cpu().numpy().astype(np.dtype(NUMPY_DTYPE[ty])).tofile(
                Path(work) / (name + ".bin"))
        _run_legacy_script(_compare_scope, work)
    finally:
        work_dir.cleanup()


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_predicate_load_store_psti_norm_pldi_ds',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
