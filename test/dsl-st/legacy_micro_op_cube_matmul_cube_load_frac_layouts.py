#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/cube-matmul/cube-load-frac-layouts."""

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
    '\n// -----------------------------------------------------------------------------\n// case: micro-op/cube-m'
    'atmul/cube-load-frac-layouts\n// family: micro-op/cube-matmul\n// target_ops: pto.mte_gm_l1_frac, pto.copy_g'
    'm_to_cbuf, pto.mte_l1_l0a,\n//   pto.mte_l1_l0b, pto.mad, pto.mte_l0c_gm\n// scenarios: nd2nz, dn2nz, nchw->'
    'nc1hwc0, nchw->fractalz4d,\n//   ncdhw->ndc1hwc0, ncdhw->fractalz3d\n// ------------------------------------'
    '-----------------------------------------\n\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #p'
    'to.kernel_kind<cube>} {\n    func.func @cube_load_frac_layouts_deep_merged_kernel(%arg0: !pto.ptr<f16, gm>, '
    '%arg1: !pto.ptr<f16, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: !pto.ptr<f16, gm>, %arg4: !pto.ptr<f16, gm>, %arg'
    '5: !pto.ptr<f32, gm>, %arg6: !pto.ptr<f16, gm>, %arg7: !pto.ptr<f16, gm>, %arg8: !pto.ptr<f32, gm>, %arg9: !'
    'pto.ptr<f16, gm>, %arg10: !pto.ptr<f16, gm>, %arg11: !pto.ptr<f32, gm>, %arg12: !pto.ptr<f16, gm>, %arg13: !'
    'pto.ptr<f16, gm>, %arg14: !pto.ptr<f32, gm>, %arg15: !pto.ptr<f16, gm>, %arg16: !pto.ptr<f16, gm>, %arg17: !'
    'pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %__deep_merge_guard = arith.constant false\n    // inactive'
    ' merged from cube_load_frac_dn2nz_kernel\n    scf.if %__deep_merge_guard {\n\n    %c0_i64_m0 = arith.constan'
    't 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c2_i64_m0 = arith.constant 2 : i64\n    %c8_i64_m0 '
    '= arith.constant 8 : i64\n    %c16_i64_m0 = arith.constant 16 : i64\n    %c512_i64_m0 = arith.constant 512 :'
    ' i64\n    %false_m0 = arith.constant false\n\n    %mat_id_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<f16,'
    ' l1>\n    %mat_src_m0 = pto.castptr %c512_i64_m0 : i64 -> !pto.ptr<f16, l1>\n    %l0a_m0 = pto.castptr %c0_i'
    '64_m0 : i64 -> !pto.ptr<f16, l0a>\n    %l0b_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<f16, l0b>\n    %l0'
    'c_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.copy_gm_to_cbuf %arg0, %mat_src_m0, %c1'
    '6_i64_m0, %c16_i64_m0, %c0_i64_m0, %c0_i64_m0\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, '
    'i64\n    pto.mte_gm_l1_frac %arg1, %mat_id_m0, dn2nz,\n        shape(%c8_i64_m0, %c16_i64_m0),\n        src_'
    'layout(%c2_i64_m0),\n        dst_group(%c1_i64_m0, %c1_i64_m0, %c8_i64_m0, %c0_i64_m0),\n        ctrl(%c0_i6'
    '4_m0, %false_m0)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, dn2nz,\n        shape i64, i64, src_layout(i'
    '64),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mte_l1_l0a %mat_src_m0, %l0a_m0'
    ', %c16_i64_m0, %c16_i64_m0, %c0_i64_m0, %c0_i64_m0\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64,'
    ' i64, i64\n    pto.mte_l1_l0b %mat_id_m0, %l0b_m0, %c16_i64_m0, %c16_i64_m0, %c0_i64_m0, %c0_i64_m0 {transpo'
    'se = true}\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1",'
    ' "PIPE_M", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.mad %l0a_m0, %l0b_m0'
    ', %l0c_m0, %c16_i64_m0, %c16_i64_m0, %c16_i64_m0\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f'
    '32, l0c>, i64, i64, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", '
    '"PIPE_FIX", "EVENT_ID1"]\n\n    pto.mte_l0c_gm %l0c_m0, %arg2, %c16_i64_m0, %c16_i64_m0, %c16_i64_m0, %c16_i'
    '64_m0, %c0_i64_m0, %c0_i64_m0,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i'
    '64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from cube_load_frac_nchw_'
    'nc1hwc0_kernel\n    scf.if %__deep_merge_guard {\n\n    %c0_i64_m1 = arith.constant 0 : i64\n    %c1_i64_m1 '
    '= arith.constant 1 : i64\n    %c10_i64_m1 = arith.constant 10 : i64\n    %c16_i64_m1 = arith.constant 16 : i'
    '64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c512_i64_m1 = arith.constant 512 : i64\n    %false_m1 = '
    'arith.constant false\n\n    %mat_id_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<f16, l1>\n    %mat_src_m1 '
    '= pto.castptr %c512_i64_m1 : i64 -> !pto.ptr<f16, l1>\n    %l0a_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.pt'
    'r<f16, l0a>\n    %l0b_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<f16, l0b>\n    %l0c_m1 = pto.castptr %c0'
    '_i64_m1 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.copy_gm_to_cbuf %arg3, %mat_src_m1, %c16_i64_m1, %c16_i64_m1,'
    ' %c0_i64_m1, %c0_i64_m1\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64\n    pto.mte_gm_l1'
    '_frac %arg4, %mat_id_m1, dn2nz,\n        shape(%c16_i64_m1, %c10_i64_m1),\n        src_layout(%c32_i64_m1),'
    '\n        dst_group(%c1_i64_m1, %c1_i64_m1, %c16_i64_m1, %c16_i64_m1),\n        ctrl(%c0_i64_m1, %false_m1)'
    '\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, dn2nz,\n        shape i64, i64, src_layout(i64),\n        ds'
    't_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n    pto'
    '.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mte_l1_l0a %mat_src_m1, %l0a_m1, %c16_i64_m1, %'
    'c16_i64_m1, %c0_i64_m1, %c0_i64_m1\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64\n    p'
    'to.mte_l1_l0b %mat_id_m1, %l0b_m1, %c16_i64_m1, %c10_i64_m1, %c0_i64_m1, %c0_i64_m1 {transpose = true}\n    '
    '  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.mad %l0a_m1, %l0b_m1, %l0c_m1, %c16_'
    'i64_m1, %c16_i64_m1, %c16_i64_m1\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i'
    '64, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVE'
    'NT_ID1"]\n\n    pto.mte_l0c_gm %l0c_m1, %arg5, %c16_i64_m1, %c16_i64_m1, %c16_i64_m1, %c16_i64_m1, %c0_i64_m'
    '1, %c0_i64_m1,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64\n  '
    '  pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from cube_load_frac_nchw_fz4d_kernel\n   '
    ' scf.if %__deep_merge_guard {\n\n    %c0_i64_m2 = arith.constant 0 : i64\n    %c1_i64_m2 = arith.constant 1 '
    ': i64\n    %c2_i64_m2 = arith.constant 2 : i64\n    %c5_i64_m2 = arith.constant 5 : i64\n    %c16_i64_m2 = a'
    'rith.constant 16 : i64\n    %c32_i64_m2 = arith.constant 32 : i64\n    %c80_i64_m2 = arith.constant 80 : i64'
    '\n    %c160_i64_m2 = arith.constant 160 : i64\n    %c512_i64_m2 = arith.constant 512 : i64\n    %false_m2 = '
    'arith.constant false\n\n    %mat_id_m2 = pto.castptr %c0_i64_m2 : i64 -> !pto.ptr<f16, l1>\n    %mat_src_m2 '
    '= pto.castptr %c512_i64_m2 : i64 -> !pto.ptr<f16, l1>\n    %l0a_m2 = pto.castptr %c0_i64_m2 : i64 -> !pto.pt'
    'r<f16, l0a>\n    %l0b_m2 = pto.castptr %c0_i64_m2 : i64 -> !pto.ptr<f16, l0b>\n    %l0c_m2 = pto.castptr %c0'
    '_i64_m2 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.copy_gm_to_cbuf %arg6, %mat_src_m2, %c16_i64_m2, %c16_i64_m2,'
    ' %c0_i64_m2, %c0_i64_m2\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64\n    pto.mte_gm_l1'
    '_frac %arg7, %mat_id_m2, dn2nz,\n        shape(%c16_i64_m2, %c5_i64_m2),\n        src_layout(%c32_i64_m2, %c'
    '160_i64_m2),\n        dst_group(%c1_i64_m2, %c16_i64_m2, %c80_i64_m2, %c1_i64_m2),\n        ctrl(%c0_i64_m2,'
    ' %false_m2)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, dn2nz,\n        shape i64, i64, src_layout(i64, i'
    '64),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mte_l1_l0a %mat_src_m2, %l0a_m2'
    ', %c16_i64_m2, %c16_i64_m2, %c0_i64_m2, %c0_i64_m2\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64,'
    ' i64, i64\n    pto.mte_l1_l0b %mat_id_m2, %l0b_m2, %c16_i64_m2, %c16_i64_m2, %c0_i64_m2, %c0_i64_m2 {transpo'
    'se = true}\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1",'
    ' "PIPE_M", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.mad %l0a_m2, %l0b_m2'
    ', %l0c_m2, %c16_i64_m2, %c16_i64_m2, %c16_i64_m2\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f'
    '32, l0c>, i64, i64, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", '
    '"PIPE_FIX", "EVENT_ID1"]\n\n    pto.mte_l0c_gm %l0c_m2, %arg8, %c16_i64_m2, %c16_i64_m2, %c16_i64_m2, %c16_i'
    '64_m2, %c0_i64_m2, %c0_i64_m2,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i'
    '64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from cube_load_frac_ncdhw'
    '_ndc1hwc0_kernel\n    scf.if %__deep_merge_guard {\n\n    %c0_i64_m3 = arith.constant 0 : i64\n    %c1_i64_m'
    '3 = arith.constant 1 : i64\n    %c7_i64_m3 = arith.constant 7 : i64\n    %c16_i64_m3 = arith.constant 16 : i'
    '64\n    %c32_i64_m3 = arith.constant 32 : i64\n    %c512_i64_m3 = arith.constant 512 : i64\n    %false_m3 = '
    'arith.constant false\n\n    %mat_id_m3 = pto.castptr %c0_i64_m3 : i64 -> !pto.ptr<f16, l1>\n    %mat_src_m3 '
    '= pto.castptr %c512_i64_m3 : i64 -> !pto.ptr<f16, l1>\n    %l0a_m3 = pto.castptr %c0_i64_m3 : i64 -> !pto.pt'
    'r<f16, l0a>\n    %l0b_m3 = pto.castptr %c0_i64_m3 : i64 -> !pto.ptr<f16, l0b>\n    %l0c_m3 = pto.castptr %c0'
    '_i64_m3 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.copy_gm_to_cbuf %arg9, %mat_src_m3, %c16_i64_m3, %c16_i64_m3,'
    ' %c0_i64_m3, %c0_i64_m3\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64\n    pto.mte_gm_l1'
    '_frac %arg10, %mat_id_m3, dn2nz,\n        shape(%c16_i64_m3, %c7_i64_m3),\n        src_layout(%c32_i64_m3),'
    '\n        dst_group(%c1_i64_m3, %c1_i64_m3, %c16_i64_m3, %c16_i64_m3),\n        ctrl(%c0_i64_m3, %false_m3)'
    '\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, dn2nz,\n        shape i64, i64, src_layout(i64),\n        ds'
    't_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n    pto'
    '.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mte_l1_l0a %mat_src_m3, %l0a_m3, %c16_i64_m3, %'
    'c16_i64_m3, %c0_i64_m3, %c0_i64_m3\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64\n    p'
    'to.mte_l1_l0b %mat_id_m3, %l0b_m3, %c16_i64_m3, %c7_i64_m3, %c0_i64_m3, %c0_i64_m3 {transpose = true}\n     '
    ' : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVENT'
    '_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.mad %l0a_m3, %l0b_m3, %l0c_m3, %c16_i'
    '64_m3, %c16_i64_m3, %c16_i64_m3\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i6'
    '4, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVEN'
    'T_ID1"]\n\n    pto.mte_l0c_gm %l0c_m3, %arg11, %c16_i64_m3, %c16_i64_m3, %c16_i64_m3, %c16_i64_m3, %c0_i64_m'
    '3, %c0_i64_m3,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64\n  '
    '  pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from cube_load_frac_ncdhw_fz3d_kernel\n  '
    '  scf.if %__deep_merge_guard {\n\n    %c0_i64_m4 = arith.constant 0 : i64\n    %c1_i64_m4 = arith.constant 1'
    ' : i64\n    %c2_i64_m4 = arith.constant 2 : i64\n    %c3_i64_m4 = arith.constant 3 : i64\n    %c16_i64_m4 = '
    'arith.constant 16 : i64\n    %c32_i64_m4 = arith.constant 32 : i64\n    %c512_i64_m4 = arith.constant 512 : '
    'i64\n    %c68720525568_i64_m4 = arith.constant 68720525568 : i64\n    %false_m4 = arith.constant false\n\n  '
    '  %mat_id_m4 = pto.castptr %c0_i64_m4 : i64 -> !pto.ptr<f16, l1>\n    %mat_src_m4 = pto.castptr %c512_i64_m4'
    ' : i64 -> !pto.ptr<f16, l1>\n    %l0a_m4 = pto.castptr %c0_i64_m4 : i64 -> !pto.ptr<f16, l0a>\n    %l0b_m4 ='
    ' pto.castptr %c0_i64_m4 : i64 -> !pto.ptr<f16, l0b>\n    %l0c_m4 = pto.castptr %c0_i64_m4 : i64 -> !pto.ptr<'
    'f32, l0c>\n\n    pto.copy_gm_to_cbuf %arg12, %mat_src_m4, %c16_i64_m4, %c16_i64_m4, %c0_i64_m4, %c0_i64_m4\n'
    '      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64\n    pto.mte_gm_l1_frac %arg13, %mat_id_m4,'
    ' dn2nz,\n        shape(%c1_i64_m4, %c16_i64_m4),\n        src_layout(%c2_i64_m4, %c32_i64_m4),\n        dst_'
    'group(%c3_i64_m4, %c16_i64_m4, %c16_i64_m4, %c1_i64_m4),\n        ctrl(%c0_i64_m4, %false_m4)\n      : !pto.'
    'ptr<f16, gm>, !pto.ptr<f16, l1>, dn2nz,\n        shape i64, i64, src_layout(i64, i64),\n        dst_group i6'
    '4, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n    pto.wait_flag'
    '["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mte_l1_l0a %mat_src_m4, %l0a_m4, %c16_i64_m4, %c16_i64_m4'
    ', %c0_i64_m4, %c0_i64_m4\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64\n    pto.mte_l1_'
    'l0b %mat_id_m4, %l0b_m4, %c16_i64_m4, %c16_i64_m4, %c0_i64_m4, %c0_i64_m4 {transpose = true}\n      : !pto.p'
    'tr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n '
    '   pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.mad %l0a_m4, %l0b_m4, %l0c_m4, %c16_i64_m4, %c'
    '16_i64_m4, %c16_i64_m4\n      : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64\n'
    '\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]'
    '\n\n    pto.mte_l0c_gm %l0c_m4, %arg14, %c16_i64_m4, %c16_i64_m4, %c16_i64_m4, %c16_i64_m4, %c0_i64_m4, %c0_'
    'i64_m4,\n      nz2nd\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64\n    pto.b'
    'arrier #pto.pipe<PIPE_ALL>\n\n    }\n    // active merged from cube_load_frac_nd2nz_kernel\n\n    %c0_i64_m5'
    ' = arith.constant 0 : i64\n    %c1_i64_m5 = arith.constant 1 : i64\n    %c32_i64_m5 = arith.constant 32 : i6'
    '4\n    %c40_i64_m5 = arith.constant 40 : i64\n    %c48_i64_m5 = arith.constant 48 : i64\n    %c50_i64_m5 = a'
    'rith.constant 50 : i64\n    %c60_i64_m5 = arith.constant 60 : i64\n    %c64_i64_m5 = arith.constant 64 : i64'
    '\n    %c100_i64_m5 = arith.constant 100 : i64\n    %c120_i64_m5 = arith.constant 120 : i64\n    %c2400_i64_m'
    '5 = arith.constant 2400 : i64\n    %c46080_i64_m5 = arith.constant 46080 : i64\n    %c65536_i64_m5 = arith.c'
    'onstant 65536 : i64\n    %false_m5 = arith.constant false\n\n    %mat_src_m5 = pto.castptr %c0_i64_m5 : i64 '
    '-> !pto.ptr<f16, l1>\n    %mat_id_m5 = pto.castptr %c65536_i64_m5 : i64 -> !pto.ptr<f16, l1>\n    %l0a_m5 = '
    'pto.castptr %c0_i64_m5 : i64 -> !pto.ptr<f16, l0a>\n    %l0b_m5 = pto.castptr %c0_i64_m5 : i64 -> !pto.ptr<f'
    '16, l0b>\n    %l0c_m5 = pto.castptr %c0_i64_m5 : i64 -> !pto.ptr<f32, l0c>\n\n    pto.mte_gm_l1_frac %arg16,'
    ' %mat_src_m5, nd2nz,\n        shape(%c40_i64_m5, %c50_i64_m5),\n        src_layout(%c100_i64_m5),\n        d'
    'st_group(%c1_i64_m5, %c1_i64_m5, %c48_i64_m5, %c0_i64_m5),\n        ctrl(%c0_i64_m5, %false_m5)\n      : !pt'
    'o.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n        shape i64, i64, src_layout(i64),\n        dst_group i64, '
    'i64, i64, i64, ctrl i64, i1\n    pto.mte_gm_l1_frac %arg15, %mat_id_m5, nd2nz,\n        shape(%c50_i64_m5, %'
    'c60_i64_m5),\n        src_layout(%c120_i64_m5),\n        dst_group(%c1_i64_m5, %c1_i64_m5, %c64_i64_m5, %c0_'
    'i64_m5),\n        ctrl(%c0_i64_m5, %false_m5)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz,\n       '
    ' shape i64, i64, src_layout(i64),\n        dst_group i64, i64, i64, i64, ctrl i64, i1\n\n    pto.set_flag["P'
    'IPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_MTE1", "EVENT_ID0"]\n\n    pto.mt'
    'e_l1_l0a %mat_src_m5, %l0a_m5, %c40_i64_m5, %c50_i64_m5, %c0_i64_m5, %c0_i64_m5\n      : !pto.ptr<f16, l1>, '
    '!pto.ptr<f16, l0a>, i64, i64, i64, i64\n    pto.mte_l1_l0b %mat_id_m5, %l0b_m5, %c50_i64_m5, %c60_i64_m5, %c'
    '0_i64_m5, %c0_i64_m5 {transpose = true}\n      : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64\n'
    '    pto.set_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE1", "PIPE_M", "EVENT_ID0"]'
    '\n    pto.mad %l0a_m5, %l0b_m5, %l0c_m5, %c40_i64_m5, %c60_i64_m5, %c50_i64_m5\n      : !pto.ptr<f16, l0a>, '
    '!pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64\n\n    pto.set_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]'
    '\n    pto.wait_flag["PIPE_M", "PIPE_FIX", "EVENT_ID1"]\n\n    pto.mte_l0c_gm %l0c_m5, %arg17, %c40_i64_m5, %'
    'c60_i64_m5, %c48_i64_m5, %c60_i64_m5, %c0_i64_m5, %c0_i64_m5,\n      nz2nd,\n      loop3(%c1_i64_m5, %c46080'
    '_i64_m5, %c2400_i64_m5)\n      : !pto.ptr<f32, l0c>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i'
    '64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    return\n  }\n}\n'
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


    def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
        n, c, h, w = nchw_tensor.shape
        c1 = (c + c0 - 1) // c0
        padded = np.pad(nchw_tensor, ((0, 0), (0, c1 * c0 - c), (0, 0), (0, 0)))
        return np.transpose(padded.reshape(n, c1, c0, h, w), (0, 1, 3, 4, 2))


    def nchw_to_c1hw_n16_16_c0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
        n, c, h, w = nchw_tensor.shape
        n_pad = ((n + 15) // 16) * 16
        c_pad = ((c + c0 - 1) // c0) * c0
        c1 = c_pad // c0
        padded = np.pad(nchw_tensor, ((0, n_pad - n), (0, c_pad - c), (0, 0), (0, 0)))
        nc1c0hw = padded.reshape(n_pad, c1, c0, h, w)
        n16 = nc1c0hw.reshape(n_pad // 16, 16, c1, c0, h, w)
        return np.transpose(n16, (2, 4, 5, 0, 1, 3)).reshape(c1 * h * w, n_pad // 16, 16, c0)


    def ncdhw_to_ndc1hwc0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
        n, c, d, h, w = ncdhw_tensor.shape
        c1 = (c + c0 - 1) // c0
        padded = np.pad(ncdhw_tensor, ((0, 0), (0, c1 * c0 - c), (0, 0), (0, 0), (0, 0)))
        nc1c0dhw = padded.reshape(n, c1, c0, d, h, w)
        return np.transpose(nc1c0dhw, (0, 3, 1, 4, 5, 2))


    def ncdhw_to_c1dhw_n16_16_c0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
        n, c, d, h, w = ncdhw_tensor.shape
        n_pad = ((n + 15) // 16) * 16
        c_pad = ((c + c0 - 1) // c0) * c0
        c1 = c_pad // c0
        padded = np.pad(ncdhw_tensor, ((0, n_pad - n), (0, c_pad - c), (0, 0), (0, 0), (0, 0)))
        nc1c0dhw = padded.reshape(n_pad, c1, c0, d, h, w)
        n16 = nc1c0dhw.reshape(n_pad // 16, 16, c1, c0, d, h, w)
        return np.transpose(n16, (2, 4, 5, 6, 0, 1, 3)).reshape(c1 * d * h * w, n_pad // 16, 16, c0)


    def write(path: Path, array: np.ndarray) -> None:
        _files[path] = array.reshape(-1)


    def _generate(seed: int) -> dict:

        lhs_nd2nz_case1 = (np.arange(40 * 50, dtype=np.float16).reshape(40, 50) * np.float16(0.5) +
                           np.float16(17)).astype(np.float16)
        nd2nz_case1 = (np.arange(50 * 60, dtype=np.float16).reshape(50, 60) * np.float16(0.25) +
                       np.float16(3)).astype(np.float16)
        lhs = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 2001).astype(np.float16)
        dn2nz = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 301).astype(np.float16)
        nchw = (np.arange(1 * 10 * 1 * 16, dtype=np.float16).reshape(1, 10, 1, 16) + 601).astype(np.float16)
        nchw_fz4d = (np.arange(5 * 16, dtype=np.float16).reshape(5, 16, 1, 1) + 901).astype(np.float16)
        ncdhw = (np.arange(1 * 7 * 1 * 1 * 16, dtype=np.float16).reshape(1, 7, 1, 1, 16) + 1201).astype(np.float16)
        ncdhw_fz3d = (np.arange(3 * 16, dtype=np.float16).reshape(3, 16, 1, 1, 1) + 1501).astype(np.float16)

        lhs_nd2nz_case1_f32 = lhs_nd2nz_case1.astype(np.float32)
        lhs_f32 = lhs.astype(np.float32)
        golden_nd2nz = lhs_nd2nz_case1_f32 @ nd2nz_case1.astype(np.float32)
        golden_dn2nz = lhs_f32 @ dn2nz.astype(np.float32)
        golden_nchw_nc1hwc0 = lhs_f32 @ nchw_to_nc1hwc0(nchw).reshape(16, 16).astype(np.float32)
        golden_nchw_fz4d = lhs_f32 @ nchw_to_c1hw_n16_16_c0(nchw_fz4d).reshape(16, 16).astype(np.float32)
        golden_ncdhw_ndc1hwc0 = lhs_f32 @ ncdhw_to_ndc1hwc0(ncdhw).reshape(16, 16).astype(np.float32)
        golden_ncdhw_fz3d = lhs_f32 @ ncdhw_to_c1dhw_n16_16_c0(ncdhw_fz3d).reshape(16, 16).astype(np.float32)

        zeros_nd2nz = np.zeros((40, 60), dtype=np.float32)
        zeros = np.zeros((16, 16), dtype=np.float32)

        write("lhs_nd2nz_case1.bin", lhs_nd2nz_case1)
        write("src_nd2nz_case1.bin", nd2nz_case1)
        write("identity.bin", lhs)
        write("src_dn2nz.bin", dn2nz)
        write("src_nchw_nc1hwc0.bin", nchw)
        write("src_nchw_fz4d.bin", nchw_fz4d)
        write("src_ncdhw_ndc1hwc0.bin", ncdhw)
        write("src_ncdhw_fz3d.bin", ncdhw_fz3d)

        write("out_nd2nz_case1.bin", zeros_nd2nz)
        write("out_dn2nz.bin", zeros)
        write("out_nchw_nc1hwc0.bin", zeros)
        write("out_nchw_fz4d.bin", zeros)
        write("out_ncdhw_ndc1hwc0.bin", zeros)
        write("out_ncdhw_fz3d.bin", zeros)

        write("golden_nd2nz_case1.bin", golden_nd2nz)
        write("golden_dn2nz.bin", golden_dn2nz)
        write("golden_nchw_nc1hwc0.bin", golden_nchw_nc1hwc0)
        write("golden_nchw_fz4d.bin", golden_nchw_fz4d)
        write("golden_ncdhw_ndc1hwc0.bin", golden_ncdhw_ndc1hwc0)
        write("golden_ncdhw_fz3d.bin", golden_ncdhw_fz3d)


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


    CASES = (
        ("golden_nd2nz_case1.bin", "out_nd2nz_case1.bin"),
    )


    def compare_one(golden_path: str, output_path: str) -> bool:
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            print(f"[ERROR] missing file: {golden_path} or {output_path}")
            return False
        golden = np.fromfile(golden_path, dtype=np.float32)
        output = np.fromfile(output_path, dtype=np.float32)
        if golden.shape != output.shape:
            print(f"[ERROR] shape mismatch for {output_path}: {golden.shape} vs {output.shape}")
            return False
        if np.allclose(golden, output, atol=1e-3, rtol=1e-3):
            return True
        diff = np.nonzero(~np.isclose(golden, output, atol=1e-3, rtol=1e-3))[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] {output_path} mismatch at idx={idx}: golden={golden[idx]}, out={output[idx]}")
        return False


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = all(compare_one(golden_path, output_path) for golden_path, output_path in CASES)
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
        'golden_nd2nz_case1.bin',
        'out_nd2nz_case1.bin'
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

# launch.cpp binds some host buffers to several kernel arguments; ARG_TO_BUFFER
# says which distinct buffer each kernel argument receives, so the module can
# pass aliased device tensors. BUFFER_FILES lists those distinct buffers.
BUFFER_FILES = [
    (
        'src_nd2nz_case1',
        'f16'
    ),
    (
        'lhs_nd2nz_case1',
        'f16'
    ),
    (
        'out_nd2nz_case1',
        'f32'
    )
]
ARG_TO_BUFFER = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    2
]
ARG_DTYPES = [
    'f16',
    'f16',
    'f32',
    'f16',
    'f16',
    'f32',
    'f16',
    'f16',
    'f32',
    'f16',
    'f16',
    'f32',
    'f16',
    'f16',
    'f32',
    'f16',
    'f16',
    'f32'
]
SCALARS = []

def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='cube_load_frac_layouts_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f16, "gm"),
    arg1: pto.ptr(pto.f16, "gm"),
    arg2: pto.ptr(pto.f32, "gm"),
    arg3: pto.ptr(pto.f16, "gm"),
    arg4: pto.ptr(pto.f16, "gm"),
    arg5: pto.ptr(pto.f32, "gm"),
    arg6: pto.ptr(pto.f16, "gm"),
    arg7: pto.ptr(pto.f16, "gm"),
    arg8: pto.ptr(pto.f32, "gm"),
    arg9: pto.ptr(pto.f16, "gm"),
    arg10: pto.ptr(pto.f16, "gm"),
    arg11: pto.ptr(pto.f32, "gm"),
    arg12: pto.ptr(pto.f16, "gm"),
    arg13: pto.ptr(pto.f16, "gm"),
    arg14: pto.ptr(pto.f32, "gm"),
    arg15: pto.ptr(pto.f16, "gm"),
    arg16: pto.ptr(pto.f16, "gm"),
    arg17: pto.ptr(pto.f32, "gm"),
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
    base, per_arg = {}, []
    for index, ty in zip(ARG_TO_BUFFER, ARG_DTYPES):
        name, _buf_ty = BUFFER_FILES[index]
        key = "%s.bin" % name
        if index not in base:
            if key in produced:
                # One host buffer can back several kernel arguments of
                # different widths, so keep the golden bytes and let each
                # argument reinterpret them below instead of narrowing the
                # shared buffer to the first argument type.
                array = np.ascontiguousarray(np.asarray(produced[key]))
            else:
                golden_key = output_to_golden.get(key)
                if golden_key is None or golden_key not in produced:
                    raise RuntimeError("golden did not produce %s" % key)
                array = np.ascontiguousarray(
                    np.zeros_like(np.asarray(produced[golden_key]), dtype=np.dtype(NUMPY_DTYPE[ty])))
            base[index] = torch.from_numpy(array).to("npu:0")
        tensor = base[index]
        want = _dtype_map()[NUMPY_DTYPE[ty]]
        if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
            tensor = tensor.view(want)
        per_arg.append(tensor)
    return per_arg, produced, list(SCALARS)


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
    seen = {}
    for index, tensor in zip(ARG_TO_BUFFER, device_inputs):
        seen.setdefault(index, tensor)
    for index, tensor in seen.items():
        name, _ty = BUFFER_FILES[index]
        files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_cube_matmul_cube_load_frac_layouts',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
