#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/cv-mixed/fixpipe-acc-store-sat-f16-cv."""

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
    '\n// -----------------------------------------------------------------------------\n// case: micro-op/cv-mix'
    'ed/fixpipe-acc-store-sat-f16-cv\n// family: cv-mixed\n// target_ops: pto.mte_gm_l1_frac, pto.mte_l1_l0a, pto'
    '.mte_l1_l0b, pto.mad,\n//   pto.mte_gm_l1, pto.mte_l1_fb, pto.mte_l0c_ub, pto.mte_l0c_gm, pto.mte_l0c_l1,\n/'
    '/   pto.mte_l1_ub, pto.mte_ub_gm, pto.get_ctrl, pto.sbitset1, pto.set_ctrl,\n//   pto.set_intra_block, pto.w'
    'ait_intra_block\n// scenarios: acc-store-ub-gm-l1, sat-ab, qf322f16, ctrl48-restore, ub-to-gm, l1-to-ub\n// '
    '-----------------------------------------------------------------------------\n\nmodule attributes {pto.targ'
    'et_arch = "a5"} {\n  func.func @fixpipe_acc_store_sat_f16_cv_kernel(%src_gm: !pto.ptr<f16, gm>,\n           '
    '                                      %id_gm: !pto.ptr<f16, gm>,\n                                          '
    '       %fp_gm: !pto.ptr<ui32, gm>,\n                                                 %out_ub_sat_gm: !pto.pt'
    'r<f16, gm>,\n                                                 %out_ub_nosat_gm: !pto.ptr<f16, gm>,\n        '
    '                                         %out_gm_sat_gm: !pto.ptr<f16, gm>,\n                               '
    '                  %out_gm_nosat_gm: !pto.ptr<f16, gm>,\n                                                 %ou'
    't_l1_sat_gm: !pto.ptr<f16, gm>,\n                                                 %out_l1_nosat_gm: !pto.ptr'
    '<f16, gm>)\n      attributes {pto.kernel} {\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.const'
    'ant 1 : i64\n    %c8_i64 = arith.constant 8 : i64\n    %c40_i64 = arith.constant 40 : i64\n    %c48_i64 = ar'
    'ith.constant 48 : i64\n    %c50_i64 = arith.constant 50 : i64\n    %c64_i64 = arith.constant 64 : i64\n    %'
    'c100_i64 = arith.constant 100 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c512_i64 = arith.constan'
    't 512 : i64\n    %c2560_i64 = arith.constant 2560 : i64\n    %c49152_i64 = arith.constant 49152 : i64\n    %'
    'c65536_i64 = arith.constant 65536 : i64\n    %c131072_i64 = arith.constant 131072 : i64\n    %c196608_i64 = '
    'arith.constant 196608 : i64\n    %c262144_i64 = arith.constant 262144 : i64\n    %c32768_i64 = arith.constan'
    't 32768 : i64\n    %c98304_i64 = arith.constant 98304 : i64\n    %false = arith.constant false\n\n    %mat_s'
    'rc = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l1>\n    %mat_id = pto.castptr %c65536_i64 : i64 -> !pto.ptr'
    '<f16, l1>\n    %l1_fp_raw = pto.castptr %c131072_i64 : i64 -> !pto.ptr<ui32, l1>\n    %l1_fp = pto.castptr %'
    'c131072_i64 : i64 -> !pto.ptr<ui64, l1>\n    %l1_sat = pto.castptr %c196608_i64 : i64 -> !pto.ptr<f16, l1>\n'
    '    %l1_nosat = pto.castptr %c262144_i64 : i64 -> !pto.ptr<f16, l1>\n    %fb_fp = pto.castptr %c0_i64 : i64 '
    '-> !pto.ptr<f32, fb>\n    %ub_sat = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, ub>\n    %ub_nosat = pto.cast'
    'ptr %c32768_i64 : i64 -> !pto.ptr<f16, ub>\n    %ub_l1_sat = pto.castptr %c65536_i64 : i64 -> !pto.ptr<f16, '
    'ub>\n    %ub_l1_nosat = pto.castptr %c98304_i64 : i64 -> !pto.ptr<f16, ub>\n    %l0a = pto.castptr %c0_i64 :'
    ' i64 -> !pto.ptr<f16, l0a>\n    %l0b = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l0b>\n    %l0c = pto.castp'
    'tr %c0_i64 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.section.cube {\n      pto.mte_gm_l1_frac %id_gm, %mat_src,'
    ' nd2nz,\n          shape(%c40_i64, %c50_i64),\n          src_layout(%c100_i64),\n          dst_group(%c1_i64'
    ', %c1_i64, %c48_i64, %c0_i64),\n          ctrl(%c0_i64, %false)\n        : !pto.ptr<f16, gm>, !pto.ptr<f16, '
    'l1>, nd2nz,\n          shape i64, i64, src_layout(i64),\n          dst_group i64, i64, i64, i64, ctrl i64, i'
    '1\n      pto.mte_gm_l1_frac %src_gm, %mat_id, nd2nz,\n          shape(%c50_i64, %c64_i64),\n          src_la'
    'yout(%c128_i64),\n          dst_group(%c1_i64, %c1_i64, %c64_i64, %c0_i64),\n          ctrl(%c0_i64, %false)'
    '\n        : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n          shape i64, i64, src_layout(i64),\n      '
    '    dst_group i64, i64, i64, i64, ctrl i64, i1\n      pto.mte_gm_l1 %fp_gm, %l1_fp_raw, %c512_i64\n        n'
    'burst(%c1_i64, %c0_i64, %c0_i64)\n        : !pto.ptr<ui32, gm>, !pto.ptr<ui32, l1>, i64, i64, i64, i64\n\n  '
    '    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n      pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVEN'
    'T_ID0"]\n      pto.mte_l1_l0a %mat_src, %l0a, %c40_i64, %c50_i64, %c0_i64, %c0_i64\n        : !pto.ptr<f16, '
    'l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64\n      pto.mte_l1_l0b %mat_id, %l0b, %c50_i64, %c64_i64, %c0_i64'
    ', %c0_i64 {transpose = true}\n        : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n\n      p'
    'to.set_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n      pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n  '
    '    pto.mad %l0a, %l0b, %l0c, %c40_i64, %c64_i64, %c50_i64\n        : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>'
    ', !pto.ptr<f32, l0c>, i64, i64, i64\n\n      pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n      pto.wait'
    '_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n      pto.set_flag["PIPE_MTE2", "PIPE_FIX", "EVENT_ID0"]\n      pt'
    'o.wait_flag["PIPE_MTE2", "PIPE_FIX", "EVENT_ID0"]\n      pto.mte_l1_fb %l1_fp, %fb_fp, %c8_i64\n        nbur'
    'st(%c1_i64, %c0_i64, %c0_i64)\n        : !pto.ptr<ui64, l1>, !pto.ptr<f32, fb>, i64, i64, i64, i64\n\n      '
    '%ctrl = pto.get_ctrl : i64\n      %ctrl_preserve_inf = pto.sbitset1 %ctrl, %c48_i64 : i64, i64 -> i64\n     '
    ' pto.set_ctrl %ctrl_preserve_inf : i64\n\n      pto.mte_l0c_ub %l0c, %ub_sat, %c40_i64, %c64_i64, %c48_i64, '
    '%c64_i64, dst_mode(%c0_i64),\n        pre_quant(%fb_fp, mode = qf322f16_pre_vec),\n        nz2nd,\n        l'
    'oop3(%c1_i64, %c49152_i64, %c2560_i64),\n        sat\n        : !pto.ptr<f32, l0c>, !pto.ptr<f16, ub>, i64, '
    'i64, i64, i64, i64,\n          !pto.ptr<f32, fb>, i64, i64, i64\n      pto.mte_l0c_ub %l0c, %ub_nosat, %c40_'
    'i64, %c64_i64, %c48_i64, %c64_i64, dst_mode(%c0_i64),\n        pre_quant(%fb_fp, mode = qf322f16_pre_vec),\n'
    '        nz2nd,\n        loop3(%c1_i64, %c49152_i64, %c2560_i64),\n        nosat\n        : !pto.ptr<f32, l0c'
    '>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,\n          !pto.ptr<f32, fb>, i64, i64, i64\n\n      pto.mte_'
    'l0c_gm %l0c, %out_gm_sat_gm, %c40_i64, %c64_i64, %c48_i64, %c64_i64, %c0_i64, %c0_i64,\n        pre_quant(%f'
    'b_fp, mode = qf322f16_pre_vec),\n        nz2nd,\n        loop3(%c1_i64, %c49152_i64, %c2560_i64),\n        s'
    'at\n        : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64,\n          !pto.ptr<f32, '
    'fb>, i64, i64, i64\n      pto.mte_l0c_gm %l0c, %out_gm_nosat_gm, %c40_i64, %c64_i64, %c48_i64, %c64_i64, %c0'
    '_i64, %c0_i64,\n        pre_quant(%fb_fp, mode = qf322f16_pre_vec),\n        nz2nd,\n        loop3(%c1_i64, '
    '%c49152_i64, %c2560_i64),\n        nosat\n        : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i6'
    '4, i64, i64,\n          !pto.ptr<f32, fb>, i64, i64, i64\n\n      pto.mte_l0c_l1 %l0c, %l1_sat, %c40_i64, %c'
    '64_i64, %c48_i64, %c64_i64,\n        pre_quant(%fb_fp, mode = qf322f16_pre_vec),\n        nz2nd,\n        lo'
    'op3(%c1_i64, %c49152_i64, %c2560_i64),\n        sat\n        : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64, i'
    '64, i64, i64,\n          !pto.ptr<f32, fb>, i64, i64, i64\n      pto.mte_l0c_l1 %l0c, %l1_nosat, %c40_i64, %'
    'c64_i64, %c48_i64, %c64_i64,\n        pre_quant(%fb_fp, mode = qf322f16_pre_vec),\n        nz2nd,\n        l'
    'oop3(%c1_i64, %c49152_i64, %c2560_i64),\n        nosat\n        : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64'
    ', i64, i64, i64,\n          !pto.ptr<f32, fb>, i64, i64, i64\n\n      pto.set_flag["PIPE_FIX", "PIPE_MTE1", '
    '"EVENT_ID0"]\n      pto.wait_flag["PIPE_FIX", "PIPE_MTE1", "EVENT_ID0"]\n      pto.mte_l1_ub %l1_sat, %ub_l1'
    '_sat, %c128_i64\n        nburst(%c40_i64, %c0_i64, %c0_i64)\n        : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>,'
    ' i64, i64, i64, i64\n      pto.mte_l1_ub %l1_nosat, %ub_l1_nosat, %c128_i64\n        nburst(%c40_i64, %c0_i6'
    '4, %c0_i64)\n        : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>, i64, i64, i64, i64\n\n      pto.set_flag["PIPE_'
    'MTE1", "PIPE_FIX", "EVENT_ID0"]\n      pto.wait_flag["PIPE_MTE1", "PIPE_FIX", "EVENT_ID0"]\n      pto.set_in'
    'tra_block <PIPE_FIX>, 1\n      pto.set_intra_block <PIPE_FIX>, 17\n    }\n\n    pto.section.vector {\n      '
    '%subblock = pto.get_subblock_idx\n      %is_subblock0 = arith.cmpi eq, %subblock, %c0_i64 : i64\n\n      scf'
    '.if %is_subblock0 {\n        pto.wait_intra_block <PIPE_MTE3>, 1\n        pto.mte_ub_gm %ub_sat, %out_ub_sat'
    '_gm, %c128_i64\n          nburst(%c40_i64, %c128_i64, %c128_i64)\n          : !pto.ptr<f16, ub>, !pto.ptr<f1'
    '6, gm>, i64, i64, i64, i64\n        pto.mte_ub_gm %ub_nosat, %out_ub_nosat_gm, %c128_i64\n          nburst(%'
    'c40_i64, %c128_i64, %c128_i64)\n          : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64\n      '
    '  pto.mte_ub_gm %ub_l1_sat, %out_l1_sat_gm, %c128_i64\n          nburst(%c40_i64, %c128_i64, %c128_i64)\n   '
    '       : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64\n        pto.mte_ub_gm %ub_l1_nosat, %out_'
    'l1_nosat_gm, %c128_i64\n          nburst(%c40_i64, %c128_i64, %c128_i64)\n          : !pto.ptr<f16, ub>, !pt'
    'o.ptr<f16, gm>, i64, i64, i64, i64\n      }\n      pto.barrier #pto.pipe<PIPE_ALL>\n    }\n    return\n  }\n'
    '}\n'
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
    import struct
    from pathlib import Path

    import numpy as np


    M = 40
    N = 64
    K = 50
    FP_QUANT_ELEMS = 64
    FP_TRANSPORT_ELEMS = FP_QUANT_ELEMS * 2
    K_ACTIVE = 6
    CASE_VALUES = np.array(
        [
            np.float16(12000.0),
            np.float16(-12000.0),
            np.float16(np.inf),
            np.float16(-np.inf),
            np.float16(np.nan),
            np.float16(20.0),
        ],
        dtype=np.float16,
    )


    def encode_scale(scale: float) -> np.uint64:
        return np.uint64(struct.unpack("!I", struct.pack("!f", scale))[0])


    def _generate(seed: int) -> dict:

        lhs = np.zeros((M, K), dtype=np.float16)
        rhs = np.zeros((K, N), dtype=np.float16)
        lhs[:, :K_ACTIVE] = np.float16(1.0)
        for col in range(N):
            rhs[:K_ACTIVE, col] = CASE_VALUES[col % len(CASE_VALUES)]
        fp = np.full(FP_QUANT_ELEMS, encode_scale(1.0), dtype=np.uint64)

        matmul = lhs.astype(np.float32) @ rhs.astype(np.float32)
        sat_golden = np.nan_to_num(
            matmul,
            nan=np.float32(0.0),
            posinf=np.finfo(np.float16).max,
            neginf=np.finfo(np.float16).min,
        )
        sat_golden = np.clip(sat_golden, np.finfo(np.float16).min,
                             np.finfo(np.float16).max).astype(np.float16)
        with np.errstate(over="ignore", invalid="ignore"):
            nosat_golden = matmul.astype(np.float16)
        if np.array_equal(sat_golden.view(np.uint16), nosat_golden.view(np.uint16)):
            raise AssertionError("sat and nosat golden outputs must differ")

        zero = np.zeros((M, N), dtype=np.float16)
        _files["v1.bin"] = lhs.reshape(-1)
        _files["v2.bin"] = rhs.reshape(-1)
        _files["v3.bin"] = fp.view(np.uint32).reshape(FP_TRANSPORT_ELEMS)
        _files["v4.bin"] = zero.reshape(-1)
        _files["v5.bin"] = zero.reshape(-1)
        _files["v6.bin"] = zero.reshape(-1)
        _files["v7.bin"] = zero.reshape(-1)
        _files["v8.bin"] = zero.reshape(-1)
        _files["v9.bin"] = zero.reshape(-1)

        _files["golden_v4.bin"] = sat_golden.reshape(-1)
        _files["golden_v5.bin"] = nosat_golden.reshape(-1)
        _files["golden_v6.bin"] = sat_golden.reshape(-1)
        _files["golden_v7.bin"] = nosat_golden.reshape(-1)
        _files["golden_v8.bin"] = sat_golden.reshape(-1)
        _files["golden_v9.bin"] = nosat_golden.reshape(-1)


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
            print(f"[ERROR] missing file: {golden_path} or {output_path}")
            return False
        golden = np.fromfile(golden_path, dtype=np.float16)
        output = np.fromfile(output_path, dtype=np.float16)
        if golden.shape != output.shape:
            print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
            return False
        equal = golden.view(np.uint16) == output.view(np.uint16)
        equal |= np.isnan(golden) & np.isnan(output)
        if bool(np.all(equal)):
            return True
        diff = np.where(~equal)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] first mismatch at idx={idx}: "
            f"golden={float(golden[idx])}, out={float(output[idx])}"
        )
        return False


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = True
        for index in range(4, 10):
            ok = compare_bin(f"golden_v{index}.bin", f"v{index}.bin") and ok
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
COMPARE_PAIRS = []

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
        'v2',
        'f16',
        'src_gm'
    ),
    (
        'v1',
        'f16',
        'id_gm'
    ),
    (
        'v3',
        'ui32',
        'fp_gm'
    ),
    (
        'v4',
        'f16',
        'out_ub_sat_gm'
    ),
    (
        'v5',
        'f16',
        'out_ub_nosat_gm'
    ),
    (
        'v6',
        'f16',
        'out_gm_sat_gm'
    ),
    (
        'v7',
        'f16',
        'out_gm_nosat_gm'
    ),
    (
        'v8',
        'f16',
        'out_l1_sat_gm'
    ),
    (
        'v9',
        'f16',
        'out_l1_nosat_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='fixpipe_acc_store_sat_f16_cv_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.f16, "gm"),
    id_gm: pto.ptr(pto.f16, "gm"),
    fp_gm: pto.ptr(pto.ui32, "gm"),
    out_ub_sat_gm: pto.ptr(pto.f16, "gm"),
    out_ub_nosat_gm: pto.ptr(pto.f16, "gm"),
    out_gm_sat_gm: pto.ptr(pto.f16, "gm"),
    out_gm_nosat_gm: pto.ptr(pto.f16, "gm"),
    out_l1_sat_gm: pto.ptr(pto.f16, "gm"),
    out_l1_nosat_gm: pto.ptr(pto.f16, "gm"),
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
    name='legacy_micro_op_cv_mixed_fixpipe_acc_store_sat_f16_cv',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
