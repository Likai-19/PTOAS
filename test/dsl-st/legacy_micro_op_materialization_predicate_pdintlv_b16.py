#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/materialization-predicate/pdintlv_b16."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @pdi'
    'ntlv_b16_deep_merged_kernel(%arg0: !pto.ptr<ui32, gm>, %arg1: !pto.ptr<ui32, gm>) attributes {pto.kernel} {'
    '\n    %__deep_merge_guard = arith.constant false\n    // inactive merged from pdintlv_b16_nontrivial_kernel_'
    '2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m0 = arith.constant 0 : index\n    %c1_m0 = arith.constant 1'
    ' : index\n    %c32_m0 = arith.constant 32 : index\n    %c16_m0 = arith.constant 16 : index\n    %c0_i64_m0 ='
    ' arith.constant 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c64_i64_m0 = arith.constant 64 : i64'
    '\n\n    %ub_out_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<ui8, ub>\n    %gm_out_m0 = pto.castptr %arg0 :'
    ' !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n\n    pto.vecscope {\n      scf.for %iter_m0 = %c0_m0 to %c1_m0 st'
    'ep %c1_m0 {\n        %lhs_m0 = pto.pset_b16 "PAT_VL8" : !pto.mask<b16>\n        %rhs_m0 = pto.pset_b16 "PAT_'
    'M4" : !pto.mask<b16>\n        %low_m0, %high_m0 = pto.pdintlv_b16 %lhs_m0, %rhs_m0 : !pto.mask<b16>, !pto.ma'
    'sk<b16> -> !pto.mask<b16>, !pto.mask<b16>\n        pto.psts %low_m0, %ub_out_m0[%c0_m0], "NORM" : !pto.mask<'
    'b16>, !pto.ptr<ui8, ub>, index\n        pto.psts %high_m0, %ub_out_m0[%c32_m0], "NORM" : !pto.mask<b16>, !pt'
    'o.ptr<ui8, ub>, index\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_'
    'flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m0, %gm_out_m0, %c64_i64_m0\n      nburs'
    't(%c1_i64_m0, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n '
    '   pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // active merged from pdintlv_b16_kernel_2d\n\n    %c0_m1 ='
    ' arith.constant 0 : index\n    %c1_m1 = arith.constant 1 : index\n    %c32_m1 = arith.constant 32 : index\n '
    '   %c16_m1 = arith.constant 16 : index\n    %c0_i64_m1 = arith.constant 0 : i64\n    %c1_i64_m1 = arith.cons'
    'tant 1 : i64\n    %c64_i64_m1 = arith.constant 64 : i64\n\n    %ub_out_m1 = pto.castptr %c0_i64_m1 : i64 -> '
    '!pto.ptr<ui8, ub>\n    %gm_out_m1 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n\n    pto.v'
    'ecscope {\n      scf.for %iter_m1 = %c0_m1 to %c1_m1 step %c1_m1 {\n        %lhs_m1 = pto.pset_b16 "PAT_ALL"'
    ' : !pto.mask<b16>\n        %rhs_m1 = pto.pset_b16 "PAT_ALLF" : !pto.mask<b16>\n        %low_m1, %high_m1 = p'
    'to.pdintlv_b16 %lhs_m1, %rhs_m1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>\n        '
    'pto.psts %low_m1, %ub_out_m1[%c0_m1], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n        pto.psts %h'
    'igh_m1, %ub_out_m1[%c32_m1], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n      }\n    }\n\n    pto.se'
    't_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.m'
    'te_ub_gm %ub_out_m1, %gm_out_m1, %c64_i64_m1\n      nburst(%c1_i64_m1, %c64_i64_m1, %c64_i64_m1)\n      : !p'
    'to.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_mer'
    'ge_guard = arith.constant false\n\n    // inactive cross-case merge from micro-op/materialization-predicate/'
    'pdintlv_b32\n    scf.if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg8_1 = arith.constant'
    ' false\n          // inactive merged from pdintlv_b32_nontrivial_kernel_2d\n          scf.if %__deep_merge_g'
    'uard_cmg8_1 {\n      \n          %c0_m0_cmg8_1 = arith.constant 0 : index\n          %c1_m0_cmg8_1 = arith.c'
    'onstant 1 : index\n          %c32_m0_cmg8_1 = arith.constant 32 : index\n          %c16_m0_cmg8_1 = arith.co'
    'nstant 16 : index\n          %c0_i64_m0_cmg8_1 = arith.constant 0 : i64\n          %c1_i64_m0_cmg8_1 = arith'
    '.constant 1 : i64\n          %c64_i64_m0_cmg8_1 = arith.constant 64 : i64\n      \n          %ub_out_m0_cmg8'
    '_1 = pto.castptr %c0_i64_m0_cmg8_1 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m0_cmg8_1 = pto.castptr %ar'
    'g0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m0'
    '_cmg8_1 = %c0_m0_cmg8_1 to %c1_m0_cmg8_1 step %c1_m0_cmg8_1 {\n              %lhs_m0_cmg8_1 = pto.pset_b32 "'
    'PAT_VL8" : !pto.mask<b32>\n              %rhs_m0_cmg8_1 = pto.pset_b32 "PAT_M4" : !pto.mask<b32>\n          '
    '    %low_m0_cmg8_1, %high_m0_cmg8_1 = pto.pdintlv_b32 %lhs_m0_cmg8_1, %rhs_m0_cmg8_1 : !pto.mask<b32>, !pto.'
    'mask<b32> -> !pto.mask<b32>, !pto.mask<b32>\n              pto.psts %low_m0_cmg8_1, %ub_out_m0_cmg8_1[%c0_m0'
    '_cmg8_1], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m0_cmg8_1, %ub_out'
    '_m0_cmg8_1[%c32_m0_cmg8_1], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n            }\n          }\n '
    '     \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_M'
    'TE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_1, %gm_out_m0_cmg8_1, %c64_i64_m0_cmg8_1\n      '
    '      nburst(%c1_i64_m0_cmg8_1, %c64_i64_m0_cmg8_1, %c64_i64_m0_cmg8_1)\n            : !pto.ptr<ui8, ub>, !p'
    'to.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }\n       '
    '   // active merged from pdintlv_b32_kernel_2d\n      \n          %c0_m1_cmg8_1 = arith.constant 0 : index\n'
    '          %c1_m1_cmg8_1 = arith.constant 1 : index\n          %c32_m1_cmg8_1 = arith.constant 32 : index\n  '
    '        %c16_m1_cmg8_1 = arith.constant 16 : index\n          %c0_i64_m1_cmg8_1 = arith.constant 0 : i64\n  '
    '        %c1_i64_m1_cmg8_1 = arith.constant 1 : i64\n          %c64_i64_m1_cmg8_1 = arith.constant 64 : i64\n'
    '      \n          %ub_out_m1_cmg8_1 = pto.castptr %c0_i64_m1_cmg8_1 : i64 -> !pto.ptr<ui8, ub>\n          %g'
    'm_out_m1_cmg8_1 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscop'
    'e {\n            scf.for %iter_m1_cmg8_1 = %c0_m1_cmg8_1 to %c1_m1_cmg8_1 step %c1_m1_cmg8_1 {\n            '
    '  %lhs_m1_cmg8_1 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>\n              %rhs_m1_cmg8_1 = pto.pset_b32 "PAT'
    '_ALLF" : !pto.mask<b32>\n              %low_m1_cmg8_1, %high_m1_cmg8_1 = pto.pdintlv_b32 %lhs_m1_cmg8_1, %rh'
    's_m1_cmg8_1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>\n              pto.psts %low_'
    'm1_cmg8_1, %ub_out_m1_cmg8_1[%c0_m1_cmg8_1], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n            '
    '  pto.psts %high_m1_cmg8_1, %ub_out_m1_cmg8_1[%c32_m1_cmg8_1], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, i'
    'ndex\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n      '
    '    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_1, %gm_out_m1'
    '_cmg8_1, %c64_i64_m1_cmg8_1\n            nburst(%c1_i64_m1_cmg8_1, %c64_i64_m1_cmg8_1, %c64_i64_m1_cmg8_1)\n'
    '            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE'
    '_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/materialization-predicate/pdintlv_b8\n    scf'
    '.if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg8_2 = arith.constant false\n          //'
    ' inactive merged from pdintlv_b8_nontrivial_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_2 {\n      '
    '\n          %c0_m0_cmg8_2 = arith.constant 0 : index\n          %c1_m0_cmg8_2 = arith.constant 1 : index\n  '
    '        %c32_m0_cmg8_2 = arith.constant 32 : index\n          %c16_m0_cmg8_2 = arith.constant 16 : index\n  '
    '        %c0_i64_m0_cmg8_2 = arith.constant 0 : i64\n          %c1_i64_m0_cmg8_2 = arith.constant 1 : i64\n  '
    '        %c64_i64_m0_cmg8_2 = arith.constant 64 : i64\n      \n          %ub_out_m0_cmg8_2 = pto.castptr %c0_'
    'i64_m0_cmg8_2 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m0_cmg8_2 = pto.castptr %arg0 : !pto.ptr<ui32, g'
    'm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m0_cmg8_2 = %c0_m0_cmg8'
    '_2 to %c1_m0_cmg8_2 step %c1_m0_cmg8_2 {\n              %lhs_m0_cmg8_2 = pto.pset_b8 "PAT_VL8" : !pto.mask<b'
    '8>\n              %rhs_m0_cmg8_2 = pto.pset_b8 "PAT_M4" : !pto.mask<b8>\n              %low_m0_cmg8_2, %high'
    '_m0_cmg8_2 = pto.pdintlv_b8 %lhs_m0_cmg8_2, %rhs_m0_cmg8_2 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, '
    '!pto.mask<b8>\n              pto.psts %low_m0_cmg8_2, %ub_out_m0_cmg8_2[%c0_m0_cmg8_2], "NORM" : !pto.mask<b'
    '8>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m0_cmg8_2, %ub_out_m0_cmg8_2[%c32_m0_cmg8_2], "NO'
    'RM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag["P'
    'IPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pt'
    'o.mte_ub_gm %ub_out_m0_cmg8_2, %gm_out_m0_cmg8_2, %c64_i64_m0_cmg8_2\n            nburst(%c1_i64_m0_cmg8_2, '
    '%c64_i64_m0_cmg8_2, %c64_i64_m0_cmg8_2)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, '
    'i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }\n          // active merged from pdintlv'
    '_b8_kernel_2d\n      \n          %c0_m1_cmg8_2 = arith.constant 0 : index\n          %c1_m1_cmg8_2 = arith.c'
    'onstant 1 : index\n          %c32_m1_cmg8_2 = arith.constant 32 : index\n          %c16_m1_cmg8_2 = arith.co'
    'nstant 16 : index\n          %c0_i64_m1_cmg8_2 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_2 = arith'
    '.constant 1 : i64\n          %c64_i64_m1_cmg8_2 = arith.constant 64 : i64\n      \n          %ub_out_m1_cmg8'
    '_2 = pto.castptr %c0_i64_m1_cmg8_2 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_2 = pto.castptr %ar'
    'g1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m1'
    '_cmg8_2 = %c0_m1_cmg8_2 to %c1_m1_cmg8_2 step %c1_m1_cmg8_2 {\n              %lhs_m1_cmg8_2 = pto.pset_b8 "P'
    'AT_ALL" : !pto.mask<b8>\n              %rhs_m1_cmg8_2 = pto.pset_b8 "PAT_ALLF" : !pto.mask<b8>\n            '
    '  %low_m1_cmg8_2, %high_m1_cmg8_2 = pto.pdintlv_b8 %lhs_m1_cmg8_2, %rhs_m1_cmg8_2 : !pto.mask<b8>, !pto.mask'
    '<b8> -> !pto.mask<b8>, !pto.mask<b8>\n              pto.psts %low_m1_cmg8_2, %ub_out_m1_cmg8_2[%c0_m1_cmg8_2'
    '], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m1_cmg8_2, %ub_out_m1_cmg8'
    '_2[%c32_m1_cmg8_2], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      \n  '
    '        pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EV'
    'ENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_2, %gm_out_m1_cmg8_2, %c64_i64_m1_cmg8_2\n            nbu'
    'rst(%c1_i64_m1_cmg8_2, %c64_i64_m1_cmg8_2, %c64_i64_m1_cmg8_2)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui'
    '8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case m'
    'erge from micro-op/materialization-predicate/pge-tail-mask\n    scf.if %__case_merge_guard {\n      \n      '
    '    %__deep_merge_guard_cmg8_3 = arith.constant false\n          // inactive merged from pge_tail_mask_bound'
    'ary_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_3 {\n      \n          %c0_m0_cmg8_3 = arith.consta'
    'nt 0 : index\n          %c1_m0_cmg8_3 = arith.constant 1 : index\n          %c32_m0_cmg8_3 = arith.constant '
    '32 : index\n          %c64_m0_cmg8_3 = arith.constant 64 : index\n          %c0_i64_m0_cmg8_3 = arith.consta'
    'nt 0 : i64\n          %c1_i64_m0_cmg8_3 = arith.constant 1 : i64\n          %c96_i64_m0_cmg8_3 = arith.const'
    'ant 96 : i64\n      \n          %ub_out_m0_cmg8_3 = pto.castptr %c0_i64_m0_cmg8_3 : i64 -> !pto.ptr<ui8, ub>'
    '\n          %gm_out_m0_cmg8_3 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n       '
    '   pto.vecscope {\n            scf.for %iter_m0_cmg8_3 = %c0_m0_cmg8_3 to %c1_m0_cmg8_3 step %c1_m0_cmg8_3 {'
    '\n              %m0_m0_cmg8_3 = pto.pge_b8 "PAT_VL1" : !pto.mask<b8>\n              %m1_m0_cmg8_3 = pto.pge_'
    'b16 "PAT_VL1" : !pto.mask<b16>\n              %m2_m0_cmg8_3 = pto.pge_b32 "PAT_VL1" : !pto.mask<b32>\n      '
    '        pto.psts %m0_m0_cmg8_3, %ub_out_m0_cmg8_3[%c0_m0_cmg8_3], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>,'
    ' index\n              pto.psts %m1_m0_cmg8_3, %ub_out_m0_cmg8_3[%c32_m0_cmg8_3], "NORM" : !pto.mask<b16>, !p'
    'to.ptr<ui8, ub>, index\n              pto.psts %m2_m0_cmg8_3, %ub_out_m0_cmg8_3[%c64_m0_cmg8_3], "NORM" : !p'
    'to.mask<b32>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag["PIPE_V",'
    ' "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_u'
    'b_gm %ub_out_m0_cmg8_3, %gm_out_m0_cmg8_3, %c96_i64_m0_cmg8_3\n            nburst(%c1_i64_m0_cmg8_3, %c96_i6'
    '4_m0_cmg8_3, %c96_i64_m0_cmg8_3)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n  '
    '        pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }\n          // active merged from pge_tail_mask_'
    'kernel_2d\n      \n          %c0_m1_cmg8_3 = arith.constant 0 : index\n          %c1_m1_cmg8_3 = arith.const'
    'ant 1 : index\n          %c32_m1_cmg8_3 = arith.constant 32 : index\n          %c64_m1_cmg8_3 = arith.consta'
    'nt 64 : index\n          %c0_i64_m1_cmg8_3 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_3 = arith.con'
    'stant 1 : i64\n          %c96_i64_m1_cmg8_3 = arith.constant 96 : i64\n      \n          %ub_out_m1_cmg8_3 ='
    ' pto.castptr %c0_i64_m1_cmg8_3 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_3 = pto.castptr %arg1 :'
    ' !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m1_cmg'
    '8_3 = %c0_m1_cmg8_3 to %c1_m1_cmg8_3 step %c1_m1_cmg8_3 {\n              %m0_m1_cmg8_3 = pto.pge_b8 "PAT_VL8'
    '" : !pto.mask<b8>\n              %m1_m1_cmg8_3 = pto.pge_b16 "PAT_VL8" : !pto.mask<b16>\n              %m2_m'
    '1_cmg8_3 = pto.pge_b32 "PAT_VL8" : !pto.mask<b32>\n              pto.psts %m0_m1_cmg8_3, %ub_out_m1_cmg8_3[%'
    'c0_m1_cmg8_3], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %m1_m1_cmg8_3, %ub_o'
    'ut_m1_cmg8_3[%c32_m1_cmg8_3], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n              pto.psts %m2_'
    'm1_cmg8_3, %ub_out_m1_cmg8_3[%c64_m1_cmg8_3], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n           '
    ' }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag'
    '["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_3, %gm_out_m1_cmg8_3, %c96_i64'
    '_m1_cmg8_3\n            nburst(%c1_i64_m1_cmg8_3, %c96_i64_m1_cmg8_3, %c96_i64_m1_cmg8_3)\n            : !pt'
    'o.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n '
    '   // inactive cross-case merge from micro-op/materialization-predicate/pintlv_b16\n    scf.if %__case_merge'
    '_guard {\n      \n          %__deep_merge_guard_cmg8_4 = arith.constant false\n          // inactive merged '
    'from pintlv_b16_nontrivial_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_4 {\n      \n          %c0_m'
    '0_cmg8_4 = arith.constant 0 : index\n          %c1_m0_cmg8_4 = arith.constant 1 : index\n          %c32_m0_c'
    'mg8_4 = arith.constant 32 : index\n          %c16_m0_cmg8_4 = arith.constant 16 : index\n          %c0_i64_m'
    '0_cmg8_4 = arith.constant 0 : i64\n          %c1_i64_m0_cmg8_4 = arith.constant 1 : i64\n          %c64_i64_'
    'm0_cmg8_4 = arith.constant 64 : i64\n      \n          %ub_out_m0_cmg8_4 = pto.castptr %c0_i64_m0_cmg8_4 : i'
    '64 -> !pto.ptr<ui8, ub>\n          %gm_out_m0_cmg8_4 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui'
    '8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m0_cmg8_4 = %c0_m0_cmg8_4 to %c1_m0_cmg8'
    '_4 step %c1_m0_cmg8_4 {\n              %lhs_m0_cmg8_4 = pto.pset_b16 "PAT_VL8" : !pto.mask<b16>\n           '
    '   %rhs_m0_cmg8_4 = pto.pset_b16 "PAT_M4" : !pto.mask<b16>\n              %low_m0_cmg8_4, %high_m0_cmg8_4 = '
    'pto.pintlv_b16 %lhs_m0_cmg8_4, %rhs_m0_cmg8_4 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<'
    'b16>\n              pto.psts %low_m0_cmg8_4, %ub_out_m0_cmg8_4[%c0_m0_cmg8_4], "NORM" : !pto.mask<b16>, !pto'
    '.ptr<ui8, ub>, index\n              pto.psts %high_m0_cmg8_4, %ub_out_m0_cmg8_4[%c32_m0_cmg8_4], "NORM" : !p'
    'to.mask<b16>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag["PIPE_V",'
    ' "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_u'
    'b_gm %ub_out_m0_cmg8_4, %gm_out_m0_cmg8_4, %c64_i64_m0_cmg8_4\n            nburst(%c1_i64_m0_cmg8_4, %c64_i6'
    '4_m0_cmg8_4, %c64_i64_m0_cmg8_4)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n  '
    '        pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }\n          // active merged from pintlv_b16_ker'
    'nel_2d\n      \n          %c0_m1_cmg8_4 = arith.constant 0 : index\n          %c1_m1_cmg8_4 = arith.constant'
    ' 1 : index\n          %c32_m1_cmg8_4 = arith.constant 32 : index\n          %c16_m1_cmg8_4 = arith.constant '
    '16 : index\n          %c0_i64_m1_cmg8_4 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_4 = arith.consta'
    'nt 1 : i64\n          %c64_i64_m1_cmg8_4 = arith.constant 64 : i64\n      \n          %ub_out_m1_cmg8_4 = pt'
    'o.castptr %c0_i64_m1_cmg8_4 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_4 = pto.castptr %arg1 : !p'
    'to.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m1_cmg8_4'
    ' = %c0_m1_cmg8_4 to %c1_m1_cmg8_4 step %c1_m1_cmg8_4 {\n              %lhs_m1_cmg8_4 = pto.pset_b16 "PAT_ALL'
    '" : !pto.mask<b16>\n              %rhs_m1_cmg8_4 = pto.pset_b16 "PAT_ALLF" : !pto.mask<b16>\n              %'
    'low_m1_cmg8_4, %high_m1_cmg8_4 = pto.pintlv_b16 %lhs_m1_cmg8_4, %rhs_m1_cmg8_4 : !pto.mask<b16>, !pto.mask<b'
    '16> -> !pto.mask<b16>, !pto.mask<b16>\n              pto.psts %low_m1_cmg8_4, %ub_out_m1_cmg8_4[%c0_m1_cmg8_'
    '4], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m1_cmg8_4, %ub_out_m1_cm'
    'g8_4[%c32_m1_cmg8_4], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      '
    '\n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3",'
    ' "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_4, %gm_out_m1_cmg8_4, %c64_i64_m1_cmg8_4\n           '
    ' nburst(%c1_i64_m1_cmg8_4, %c64_i64_m1_cmg8_4, %c64_i64_m1_cmg8_4)\n            : !pto.ptr<ui8, ub>, !pto.pt'
    'r<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-ca'
    'se merge from micro-op/materialization-predicate/pintlv_b32\n    scf.if %__case_merge_guard {\n      \n     '
    '     %__deep_merge_guard_cmg8_5 = arith.constant false\n          // inactive merged from pintlv_b32_nontriv'
    'ial_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_5 {\n      \n          %c0_m0_cmg8_5 = arith.consta'
    'nt 0 : index\n          %c1_m0_cmg8_5 = arith.constant 1 : index\n          %c32_m0_cmg8_5 = arith.constant '
    '32 : index\n          %c16_m0_cmg8_5 = arith.constant 16 : index\n          %c0_i64_m0_cmg8_5 = arith.consta'
    'nt 0 : i64\n          %c1_i64_m0_cmg8_5 = arith.constant 1 : i64\n          %c64_i64_m0_cmg8_5 = arith.const'
    'ant 64 : i64\n      \n          %ub_out_m0_cmg8_5 = pto.castptr %c0_i64_m0_cmg8_5 : i64 -> !pto.ptr<ui8, ub>'
    '\n          %gm_out_m0_cmg8_5 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n       '
    '   pto.vecscope {\n            scf.for %iter_m0_cmg8_5 = %c0_m0_cmg8_5 to %c1_m0_cmg8_5 step %c1_m0_cmg8_5 {'
    '\n              %lhs_m0_cmg8_5 = pto.pset_b32 "PAT_VL8" : !pto.mask<b32>\n              %rhs_m0_cmg8_5 = pto'
    '.pset_b32 "PAT_M4" : !pto.mask<b32>\n              %low_m0_cmg8_5, %high_m0_cmg8_5 = pto.pintlv_b32 %lhs_m0_'
    'cmg8_5, %rhs_m0_cmg8_5 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>\n              pto'
    '.psts %low_m0_cmg8_5, %ub_out_m0_cmg8_5[%c0_m0_cmg8_5], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n '
    '             pto.psts %high_m0_cmg8_5, %ub_out_m0_cmg8_5[%c32_m0_cmg8_5], "NORM" : !pto.mask<b32>, !pto.ptr<'
    'ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID'
    '0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_5,'
    ' %gm_out_m0_cmg8_5, %c64_i64_m0_cmg8_5\n            nburst(%c1_i64_m0_cmg8_5, %c64_i64_m0_cmg8_5, %c64_i64_m'
    '0_cmg8_5)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pt'
    'o.pipe<PIPE_ALL>\n      \n          }\n          // active merged from pintlv_b32_kernel_2d\n      \n       '
    '   %c0_m1_cmg8_5 = arith.constant 0 : index\n          %c1_m1_cmg8_5 = arith.constant 1 : index\n          %'
    'c32_m1_cmg8_5 = arith.constant 32 : index\n          %c16_m1_cmg8_5 = arith.constant 16 : index\n          %'
    'c0_i64_m1_cmg8_5 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_5 = arith.constant 1 : i64\n          %'
    'c64_i64_m1_cmg8_5 = arith.constant 64 : i64\n      \n          %ub_out_m1_cmg8_5 = pto.castptr %c0_i64_m1_cm'
    'g8_5 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_5 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pt'
    'o.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m1_cmg8_5 = %c0_m1_cmg8_5 to %c1'
    '_m1_cmg8_5 step %c1_m1_cmg8_5 {\n              %lhs_m1_cmg8_5 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>\n   '
    '           %rhs_m1_cmg8_5 = pto.pset_b32 "PAT_ALLF" : !pto.mask<b32>\n              %low_m1_cmg8_5, %high_m1'
    '_cmg8_5 = pto.pintlv_b32 %lhs_m1_cmg8_5, %rhs_m1_cmg8_5 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, '
    '!pto.mask<b32>\n              pto.psts %low_m1_cmg8_5, %ub_out_m1_cmg8_5[%c0_m1_cmg8_5], "NORM" : !pto.mask<'
    'b32>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m1_cmg8_5, %ub_out_m1_cmg8_5[%c32_m1_cmg8_5], "'
    'NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag'
    '["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n         '
    ' pto.mte_ub_gm %ub_out_m1_cmg8_5, %gm_out_m1_cmg8_5, %c64_i64_m1_cmg8_5\n            nburst(%c1_i64_m1_cmg8_'
    '5, %c64_i64_m1_cmg8_5, %c64_i64_m1_cmg8_5)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i6'
    '4, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/m'
    'aterialization-predicate/pintlv_b8\n    scf.if %__case_merge_guard {\n      \n          %__deep_merge_guard_'
    'cmg8_6 = arith.constant false\n          // inactive merged from pintlv_b8_nontrivial_kernel_2d\n          s'
    'cf.if %__deep_merge_guard_cmg8_6 {\n      \n          %c0_m0_cmg8_6 = arith.constant 0 : index\n          %c'
    '1_m0_cmg8_6 = arith.constant 1 : index\n          %c32_m0_cmg8_6 = arith.constant 32 : index\n          %c16'
    '_m0_cmg8_6 = arith.constant 16 : index\n          %c0_i64_m0_cmg8_6 = arith.constant 0 : i64\n          %c1_'
    'i64_m0_cmg8_6 = arith.constant 1 : i64\n          %c64_i64_m0_cmg8_6 = arith.constant 64 : i64\n      \n    '
    '      %ub_out_m0_cmg8_6 = pto.castptr %c0_i64_m0_cmg8_6 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m0_cmg'
    '8_6 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n       '
    '     scf.for %iter_m0_cmg8_6 = %c0_m0_cmg8_6 to %c1_m0_cmg8_6 step %c1_m0_cmg8_6 {\n              %lhs_m0_cm'
    'g8_6 = pto.pset_b8 "PAT_VL8" : !pto.mask<b8>\n              %rhs_m0_cmg8_6 = pto.pset_b8 "PAT_M4" : !pto.mas'
    'k<b8>\n              %low_m0_cmg8_6, %high_m0_cmg8_6 = pto.pintlv_b8 %lhs_m0_cmg8_6, %rhs_m0_cmg8_6 : !pto.m'
    'ask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>\n              pto.psts %low_m0_cmg8_6, %ub_out_m0_cm'
    'g8_6[%c0_m0_cmg8_6], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %high_m0_cmg8_'
    '6, %ub_out_m0_cmg8_6[%c32_m0_cmg8_6], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n            }\n     '
    '     }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V"'
    ', "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_6, %gm_out_m0_cmg8_6, %c64_i64_m0_cmg8_'
    '6\n            nburst(%c1_i64_m0_cmg8_6, %c64_i64_m0_cmg8_6, %c64_i64_m0_cmg8_6)\n            : !pto.ptr<ui8'
    ', ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }'
    '\n          // active merged from pintlv_b8_kernel_2d\n      \n          %c0_m1_cmg8_6 = arith.constant 0 : '
    'index\n          %c1_m1_cmg8_6 = arith.constant 1 : index\n          %c32_m1_cmg8_6 = arith.constant 32 : in'
    'dex\n          %c16_m1_cmg8_6 = arith.constant 16 : index\n          %c0_i64_m1_cmg8_6 = arith.constant 0 : '
    'i64\n          %c1_i64_m1_cmg8_6 = arith.constant 1 : i64\n          %c64_i64_m1_cmg8_6 = arith.constant 64 '
    ': i64\n      \n          %ub_out_m1_cmg8_6 = pto.castptr %c0_i64_m1_cmg8_6 : i64 -> !pto.ptr<ui8, ub>\n     '
    '     %gm_out_m1_cmg8_6 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.'
    'vecscope {\n            scf.for %iter_m1_cmg8_6 = %c0_m1_cmg8_6 to %c1_m1_cmg8_6 step %c1_m1_cmg8_6 {\n     '
    '         %lhs_m1_cmg8_6 = pto.pset_b8 "PAT_ALL" : !pto.mask<b8>\n              %rhs_m1_cmg8_6 = pto.pset_b8 '
    '"PAT_ALLF" : !pto.mask<b8>\n              %low_m1_cmg8_6, %high_m1_cmg8_6 = pto.pintlv_b8 %lhs_m1_cmg8_6, %r'
    'hs_m1_cmg8_6 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>\n              pto.psts %low_m1_'
    'cmg8_6, %ub_out_m1_cmg8_6[%c0_m1_cmg8_6], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pt'
    'o.psts %high_m1_cmg8_6, %ub_out_m1_cmg8_6[%c32_m1_cmg8_6], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index'
    '\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          '
    'pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_6, %gm_out_m1_cmg'
    '8_6, %c64_i64_m1_cmg8_6\n            nburst(%c1_i64_m1_cmg8_6, %c64_i64_m1_cmg8_6, %c64_i64_m1_cmg8_6)\n    '
    '        : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL'
    '>\n    }\n\n    // inactive cross-case merge from micro-op/materialization-predicate/plt-tail-mask\n    scf.'
    'if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg8_7 = arith.constant false\n          // '
    'inactive merged from plt_tail_mask_boundary_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_7 {\n      '
    '\n          %c0_m0_cmg8_7 = arith.constant 0 : index\n          %c1_m0_cmg8_7 = arith.constant 1 : index\n  '
    '        %c32_m0_cmg8_7 = arith.constant 32 : index\n          %c64_m0_cmg8_7 = arith.constant 64 : index\n  '
    '        %c0_i64_m0_cmg8_7 = arith.constant 0 : i64\n          %c1_i64_m0_cmg8_7 = arith.constant 1 : i64\n  '
    '        %c96_i64_m0_cmg8_7 = arith.constant 96 : i64\n          %c1_i32_m0_cmg8_7 = arith.constant 1 : i32\n'
    '      \n          %ub_out_m0_cmg8_7 = pto.castptr %c0_i64_m0_cmg8_7 : i64 -> !pto.ptr<ui8, ub>\n          %g'
    'm_out_m0_cmg8_7 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscop'
    'e {\n            scf.for %iter_m0_cmg8_7 = %c0_m0_cmg8_7 to %c1_m0_cmg8_7 step %c1_m0_cmg8_7 {\n            '
    '  %m0_m0_cmg8_7, %s0_m0_cmg8_7 = pto.plt_b8 %c1_i32_m0_cmg8_7 : i32 -> !pto.mask<b8>, i32\n              %m1'
    '_m0_cmg8_7, %s1_m0_cmg8_7 = pto.plt_b16 %c1_i32_m0_cmg8_7 : i32 -> !pto.mask<b16>, i32\n              %m2_m0'
    '_cmg8_7, %s2_m0_cmg8_7 = pto.plt_b32 %c1_i32_m0_cmg8_7 : i32 -> !pto.mask<b32>, i32\n              pto.psts '
    '%m0_m0_cmg8_7, %ub_out_m0_cmg8_7[%c0_m0_cmg8_7], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n         '
    '     pto.psts %m1_m0_cmg8_7, %ub_out_m0_cmg8_7[%c32_m0_cmg8_7], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, '
    'index\n              pto.psts %m2_m0_cmg8_7, %ub_out_m0_cmg8_7[%c64_m0_cmg8_7], "NORM" : !pto.mask<b32>, !pt'
    'o.ptr<ui8, ub>, index\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EV'
    'ENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_c'
    'mg8_7, %gm_out_m0_cmg8_7, %c96_i64_m0_cmg8_7\n            nburst(%c1_i64_m0_cmg8_7, %c96_i64_m0_cmg8_7, %c96'
    '_i64_m0_cmg8_7)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barri'
    'er #pto.pipe<PIPE_ALL>\n      \n          }\n          // active merged from plt_tail_mask_kernel_2d\n      '
    '\n          %c0_m1_cmg8_7 = arith.constant 0 : index\n          %c1_m1_cmg8_7 = arith.constant 1 : index\n  '
    '        %c32_m1_cmg8_7 = arith.constant 32 : index\n          %c64_m1_cmg8_7 = arith.constant 64 : index\n  '
    '        %c0_i64_m1_cmg8_7 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_7 = arith.constant 1 : i64\n  '
    '        %c96_i64_m1_cmg8_7 = arith.constant 96 : i64\n          %c13_m1_cmg8_7 = arith.constant 13 : i32\n  '
    '        %c7_m1_cmg8_7 = arith.constant 7 : i32\n          %c3_m1_cmg8_7 = arith.constant 3 : i32\n      \n  '
    '        %ub_out_m1_cmg8_7 = pto.castptr %c0_i64_m1_cmg8_7 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_c'
    'mg8_7 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n     '
    '       scf.for %iter_m1_cmg8_7 = %c0_m1_cmg8_7 to %c1_m1_cmg8_7 step %c1_m1_cmg8_7 {\n              %m0_m1_c'
    'mg8_7, %s0_m1_cmg8_7 = pto.plt_b8 %c13_m1_cmg8_7 : i32 -> !pto.mask<b8>, i32\n              %m1_m1_cmg8_7, %'
    's1_m1_cmg8_7 = pto.plt_b16 %c7_m1_cmg8_7 : i32 -> !pto.mask<b16>, i32\n              %m2_m1_cmg8_7, %s2_m1_c'
    'mg8_7 = pto.plt_b32 %c3_m1_cmg8_7 : i32 -> !pto.mask<b32>, i32\n              pto.psts %m0_m1_cmg8_7, %ub_ou'
    't_m1_cmg8_7[%c0_m1_cmg8_7], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %m1_m1_'
    'cmg8_7, %ub_out_m1_cmg8_7[%c32_m1_cmg8_7], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n              '
    'pto.psts %m2_m1_cmg8_7, %ub_out_m1_cmg8_7[%c64_m1_cmg8_7], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index'
    '\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          '
    'pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_7, %gm_out_m1_cmg'
    '8_7, %c96_i64_m1_cmg8_7\n            nburst(%c1_i64_m1_cmg8_7, %c96_i64_m1_cmg8_7, %c96_i64_m1_cmg8_7)\n    '
    '        : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL'
    '>\n    }\n\n    // inactive cross-case merge from micro-op/materialization-predicate/ppack-punpack\n    scf.'
    'if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg8_8 = arith.constant false\n          // '
    'inactive merged from ppack_punpack_nontrivial_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_8 {\n    '
    '  \n          %c0_m0_cmg8_8 = arith.constant 0 : index\n          %c1_m0_cmg8_8 = arith.constant 1 : index\n'
    '          %c32_m0_cmg8_8 = arith.constant 32 : index\n          %c0_i64_m0_cmg8_8 = arith.constant 0 : i64\n'
    '          %c1_i64_m0_cmg8_8 = arith.constant 1 : i64\n          %c64_i64_m0_cmg8_8 = arith.constant 64 : i64'
    '\n      \n          %ub_out_m0_cmg8_8 = pto.castptr %c0_i64_m0_cmg8_8 : i64 -> !pto.ptr<ui8, ub>\n          '
    '%gm_out_m0_cmg8_8 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecsc'
    'ope {\n            scf.for %iter_m0_cmg8_8 = %c0_m0_cmg8_8 to %c1_m0_cmg8_8 step %c1_m0_cmg8_8 {\n          '
    '    %src_m0_cmg8_8 = pto.pset_b32 "PAT_M4" : !pto.mask<b32>\n              %packed_m0_cmg8_8 = pto.ppack %sr'
    'c_m0_cmg8_8, "LOWER" : !pto.mask<b32> -> !pto.mask<b32>\n              %roundtrip_m0_cmg8_8 = pto.punpack %p'
    'acked_m0_cmg8_8, "LOWER" : !pto.mask<b32> -> !pto.mask<b32>\n              pto.psts %packed_m0_cmg8_8, %ub_o'
    'ut_m0_cmg8_8[%c0_m0_cmg8_8], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n              pto.psts %roun'
    'dtrip_m0_cmg8_8, %ub_out_m0_cmg8_8[%c32_m0_cmg8_8], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n     '
    '       }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wai'
    't_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_8, %gm_out_m0_cmg8_8, %c'
    '64_i64_m0_cmg8_8\n            nburst(%c1_i64_m0_cmg8_8, %c64_i64_m0_cmg8_8, %c64_i64_m0_cmg8_8)\n           '
    ' : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    '
    '  \n          }\n          // active merged from ppack_punpack_kernel_2d\n      \n          %c0_m1_cmg8_8 = '
    'arith.constant 0 : index\n          %c1_m1_cmg8_8 = arith.constant 1 : index\n          %c32_m1_cmg8_8 = ari'
    'th.constant 32 : index\n          %c0_i64_m1_cmg8_8 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_8 = '
    'arith.constant 1 : i64\n          %c64_i64_m1_cmg8_8 = arith.constant 64 : i64\n      \n          %ub_out_m1'
    '_cmg8_8 = pto.castptr %c0_i64_m1_cmg8_8 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_8 = pto.castpt'
    'r %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %it'
    'er_m1_cmg8_8 = %c0_m1_cmg8_8 to %c1_m1_cmg8_8 step %c1_m1_cmg8_8 {\n              %src_m1_cmg8_8 = pto.pset_'
    'b32 "PAT_ALL" : !pto.mask<b32>\n              %packed_m1_cmg8_8 = pto.ppack %src_m1_cmg8_8, "LOWER" : !pto.m'
    'ask<b32> -> !pto.mask<b32>\n              %roundtrip_m1_cmg8_8 = pto.punpack %packed_m1_cmg8_8, "LOWER" : !p'
    'to.mask<b32> -> !pto.mask<b32>\n              pto.psts %packed_m1_cmg8_8, %ub_out_m1_cmg8_8[%c0_m1_cmg8_8], '
    '"NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n              pto.psts %roundtrip_m1_cmg8_8, %ub_out_m1_c'
    'mg8_8[%c32_m1_cmg8_8], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n            }\n          }\n      '
    '\n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3",'
    ' "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_8, %gm_out_m1_cmg8_8, %c64_i64_m1_cmg8_8\n           '
    ' nburst(%c1_i64_m1_cmg8_8, %c64_i64_m1_cmg8_8, %c64_i64_m1_cmg8_8)\n            : !pto.ptr<ui8, ub>, !pto.pt'
    'r<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-ca'
    'se merge from micro-op/materialization-predicate/psel\n    scf.if %__case_merge_guard {\n      \n          %'
    '__deep_merge_guard_cmg8_9 = arith.constant false\n          // inactive merged from psel_tail_kernel_2d\n   '
    '       scf.if %__deep_merge_guard_cmg8_9 {\n      \n          %c0_m0_cmg8_9 = arith.constant 0 : index\n    '
    '      %c1_m0_cmg8_9 = arith.constant 1 : index\n          %c32_m0_cmg8_9 = arith.constant 32 : index\n      '
    '    %c64_m0_cmg8_9 = arith.constant 64 : index\n          %c0_i64_m0_cmg8_9 = arith.constant 0 : i64\n      '
    '    %c1_i64_m0_cmg8_9 = arith.constant 1 : i64\n          %c32_i64_m0_cmg8_9 = arith.constant 32 : i64\n    '
    '      %c64_i64_m0_cmg8_9 = arith.constant 64 : i64\n          %c13_m0_cmg8_9 = arith.constant 13 : i32\n    '
    '  \n          %ub_out_m0_cmg8_9 = pto.castptr %c0_i64_m0_cmg8_9 : i64 -> !pto.ptr<ui8, ub>\n          %gm_ou'
    't_m0_cmg8_9 = pto.castptr %arg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {'
    '\n            scf.for %iter_m0_cmg8_9 = %c0_m0_cmg8_9 to %c1_m0_cmg8_9 step %c1_m0_cmg8_9 {\n              %'
    'src0_m0_cmg8_9 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>\n              %sel_m0_cmg8_9, %next_m0_cmg8_9 = pt'
    'o.plt_b32 %c13_m0_cmg8_9 : i32 -> !pto.mask<b32>, i32\n              %out_m0_cmg8_9 = pto.psel %src0_m0_cmg8'
    '_9, %sel_m0_cmg8_9, %sel_m0_cmg8_9 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>\n     '
    '         pto.psts %out_m0_cmg8_9, %ub_out_m0_cmg8_9[%c0_m0_cmg8_9], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, u'
    'b>, index\n              %out_next_m0_cmg8_9 = pto.psel %sel_m0_cmg8_9, %src0_m0_cmg8_9, %sel_m0_cmg8_9 : !p'
    'to.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>\n              pto.psts %out_next_m0_cmg8_9, '
    '%ub_out_m0_cmg8_9[%c32_m0_cmg8_9], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n            }\n       '
    '   }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", '
    '"PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_9, %gm_out_m0_cmg8_9, %c64_i64_m0_cmg8_9'
    '\n            nburst(%c1_i64_m0_cmg8_9, %c64_i64_m0_cmg8_9, %c64_i64_m0_cmg8_9)\n            : !pto.ptr<ui8,'
    ' ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }'
    '\n          // active merged from psel_kernel_2d\n      \n          %c0_m1_cmg8_9 = arith.constant 0 : index'
    '\n          %c1_m1_cmg8_9 = arith.constant 1 : index\n          %c32_m1_cmg8_9 = arith.constant 32 : index\n'
    '          %c0_i64_m1_cmg8_9 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_9 = arith.constant 1 : i64\n'
    '          %c32_i64_m1_cmg8_9 = arith.constant 32 : i64\n          %c13_m1_cmg8_9 = arith.constant 13 : i32\n'
    '      \n          %ub_out_m1_cmg8_9 = pto.castptr %c0_i64_m1_cmg8_9 : i64 -> !pto.ptr<ui8, ub>\n          %g'
    'm_out_m1_cmg8_9 = pto.castptr %arg1 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscop'
    'e {\n            scf.for %iter_m1_cmg8_9 = %c0_m1_cmg8_9 to %c1_m1_cmg8_9 step %c1_m1_cmg8_9 {\n            '
    '  %src0_m1_cmg8_9 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>\n              %sel_m1_cmg8_9, %next_m1_cmg8_9 ='
    ' pto.plt_b32 %c13_m1_cmg8_9 : i32 -> !pto.mask<b32>, i32\n              %out_m1_cmg8_9 = pto.psel %src0_m1_c'
    'mg8_9, %sel_m1_cmg8_9, %sel_m1_cmg8_9 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>\n  '
    '            pto.psts %out_m1_cmg8_9, %ub_out_m1_cmg8_9[%c0_m1_cmg8_9], "NORM" : !pto.mask<b32>, !pto.ptr<ui8'
    ', ub>, index\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]'
    '\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_9, %g'
    'm_out_m1_cmg8_9, %c32_i64_m1_cmg8_9\n            nburst(%c1_i64_m1_cmg8_9, %c32_i64_m1_cmg8_9, %c32_i64_m1_c'
    'mg8_9)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.p'
    'ipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/materialization-predicate/pset-patter'
    'n\n    scf.if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg8_10 = arith.constant false\n '
    '         // inactive merged from pset_pattern_fragment_kernel_2d\n          scf.if %__deep_merge_guard_cmg8_'
    '10 {\n      \n          %c0_m0_cmg8_10 = arith.constant 0 : index\n          %c1_m0_cmg8_10 = arith.constant'
    ' 1 : index\n          %c32_m0_cmg8_10 = arith.constant 32 : index\n          %c64_m0_cmg8_10 = arith.constan'
    't 64 : index\n          %c0_i64_m0_cmg8_10 = arith.constant 0 : i64\n          %c1_i64_m0_cmg8_10 = arith.co'
    'nstant 1 : i64\n          %c96_i64_m0_cmg8_10 = arith.constant 96 : i64\n      \n          %ub_out_m0_cmg8_1'
    '0 = pto.castptr %c0_i64_m0_cmg8_10 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m0_cmg8_10 = pto.castptr %a'
    'rg0 : !pto.ptr<ui32, gm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m'
    '0_cmg8_10 = %c0_m0_cmg8_10 to %c1_m0_cmg8_10 step %c1_m0_cmg8_10 {\n              %m0_m0_cmg8_10 = pto.pset_'
    'b8 "PAT_M3" : !pto.mask<b8>\n              %m1_m0_cmg8_10 = pto.pset_b16 "PAT_H" : !pto.mask<b16>\n         '
    '     %m2_m0_cmg8_10 = pto.pset_b32 "PAT_Q" : !pto.mask<b32>\n              pto.psts %m0_m0_cmg8_10, %ub_out_'
    'm0_cmg8_10[%c0_m0_cmg8_10], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %m1_m0_'
    'cmg8_10, %ub_out_m0_cmg8_10[%c32_m0_cmg8_10], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n           '
    '   pto.psts %m2_m0_cmg8_10, %ub_out_m0_cmg8_10[%c64_m0_cmg8_10], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>,'
    ' index\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    '
    '      pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg8_10, %gm_out'
    '_m0_cmg8_10, %c96_i64_m0_cmg8_10\n            nburst(%c1_i64_m0_cmg8_10, %c96_i64_m0_cmg8_10, %c96_i64_m0_cm'
    'g8_10)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.p'
    'ipe<PIPE_ALL>\n      \n          }\n          // active merged from pset_pattern_kernel_2d\n      \n        '
    '  %c0_m1_cmg8_10 = arith.constant 0 : index\n          %c1_m1_cmg8_10 = arith.constant 1 : index\n          '
    '%c32_m1_cmg8_10 = arith.constant 32 : index\n          %c64_m1_cmg8_10 = arith.constant 64 : index\n        '
    '  %c0_i64_m1_cmg8_10 = arith.constant 0 : i64\n          %c1_i64_m1_cmg8_10 = arith.constant 1 : i64\n      '
    '    %c96_i64_m1_cmg8_10 = arith.constant 96 : i64\n      \n          %ub_out_m1_cmg8_10 = pto.castptr %c0_i6'
    '4_m1_cmg8_10 : i64 -> !pto.ptr<ui8, ub>\n          %gm_out_m1_cmg8_10 = pto.castptr %arg1 : !pto.ptr<ui32, g'
    'm> -> !pto.ptr<ui8, gm>\n      \n          pto.vecscope {\n            scf.for %iter_m1_cmg8_10 = %c0_m1_cmg'
    '8_10 to %c1_m1_cmg8_10 step %c1_m1_cmg8_10 {\n              %m0_m1_cmg8_10 = pto.pset_b8 "PAT_ALL" : !pto.ma'
    'sk<b8>\n              %m1_m1_cmg8_10 = pto.pset_b16 "PAT_VL8" : !pto.mask<b16>\n              %m2_m1_cmg8_10'
    ' = pto.pset_b32 "PAT_VL16" : !pto.mask<b32>\n              pto.psts %m0_m1_cmg8_10, %ub_out_m1_cmg8_10[%c0_m'
    '1_cmg8_10], "NORM" : !pto.mask<b8>, !pto.ptr<ui8, ub>, index\n              pto.psts %m1_m1_cmg8_10, %ub_out'
    '_m1_cmg8_10[%c32_m1_cmg8_10], "NORM" : !pto.mask<b16>, !pto.ptr<ui8, ub>, index\n              pto.psts %m2_'
    'm1_cmg8_10, %ub_out_m1_cmg8_10[%c64_m1_cmg8_10], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n        '
    '    }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_f'
    'lag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg8_10, %gm_out_m1_cmg8_10, %c9'
    '6_i64_m1_cmg8_10\n            nburst(%c1_i64_m1_cmg8_10, %c96_i64_m1_cmg8_10, %c96_i64_m1_cmg8_10)\n        '
    '    : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n '
    '   }\n\n    return\n  }\n}\n'
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

    # case: micro-op/materialization-predicate/pdintlv_b16
    # family: materialization-predicate
    # target_ops: pto.pdintlv_b16
    # scenarios: predicate-transform, lane-order
    # coding=utf-8

    import argparse
    from pathlib import Path

    import numpy as np


    SEED = 19
    OUTPUT_WORDS = 32
    GOLDEN_PREFIX_WORDS = np.array([1431655765, 1431655765, 1431655765, 1431655765, 0, 0, 0, 0, 1431655765, 1431655765,
        1431655765, 1431655765, 0, 0, 0, 0], dtype=np.uint32)


    def _generate(seed: int) -> dict:
        del seed
        output_init = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
        golden = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
        golden[: GOLDEN_PREFIX_WORDS.size] = GOLDEN_PREFIX_WORDS

        _files["v1.bin"] = output_init
        _files["golden_v1.bin"] = golden


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate packed predicate golden for VPTO micro-op validation."
        )
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()
        generate(args.output_dir, args.seed)


    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/materialization-predicate/pdintlv_b16
    # family: materialization-predicate
    # target_ops: pto.pdintlv_b16
    # scenarios: predicate-transform, lane-order
    # coding=utf-8

    import os
    import sys

    import numpy as np


    EXPECTED_WORDS = 32
    PREFIX_WORDS = 16


    def compare_words(golden_path, output_path):
        if not os.path.exists(output_path) or not os.path.exists(golden_path):
            print(f"[ERROR] Missing file: golden={golden_path} out={output_path}")
            return False
        golden = np.fromfile(golden_path, dtype=np.uint32)
        output = np.fromfile(output_path, dtype=np.uint32)
        if golden.size != EXPECTED_WORDS or output.size != EXPECTED_WORDS:
            print(
                f"[ERROR] Unexpected word count: golden={golden.size} "
                f"out={output.size} expected={EXPECTED_WORDS}"
            )
            return False
        golden = golden[:PREFIX_WORDS]
        output = output[:PREFIX_WORDS]
        if not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0]) if diff.size else 0
            print(
                f"[ERROR] Mismatch (packed predicate words): idx={idx} "
                f"golden={int(golden[idx])} out={int(output[idx])}"
            )
            return False
        return True


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_words("golden_v1.bin", "v1.bin")
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v1.bin',
        'v1.bin'
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
        'v1',
        'ui32'
    )
]
ARG_TO_BUFFER = [
    0,
    0
]
ARG_DTYPES = [
    'ui32',
    'ui32'
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
    name='pdintlv_b16_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.ui32, "gm"),
    arg1: pto.ptr(pto.ui32, "gm"),
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
    name='legacy_micro_op_materialization_predicate_pdintlv_b16',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
