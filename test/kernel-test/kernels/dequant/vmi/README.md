<!--
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
-->

# dequant VMI backend

This directory keeps the PTODSL VMI rewrite for the `dequant` kernel test.

## Scope

- source kernel: `test/kernel-test/kernels/dequant/anti_mx_quant_tail_axis.h`
- current VMI rewrite: `anti_mx_quant_tail_axis_vmi.py`
- currently covered path: FP8 input + FP32 scale buffer + `f32` / `bf16` / `f16` output
- the FP4 + BF16-scale path in the source kernel is not modeled here yet

## Artifact layout

Running the `backend=vmi` correctness path, or `kernel-test/run.py --emit-mlir`,
emits artifacts under:

```text
test/kernel-test/kernels/dequant/generated/
```

Per-case outputs live in a case directory, and the backend also refreshes a few
root-level aliases for quick manual inspection:

- `vmi.pto`: latest lowered source VMI IR
- `mi.pto`: latest lowered MI IR

This matches the current manual-debug habit: first open the stable root alias,
and only jump into the per-case directory when a specific specialization needs
to be checked.

## Current VMI modeling

The working FP8 path is intentionally written in surface VMI terms instead of
trying to mirror every source-side physical shuffle:

- `x` is modeled as two dense 256-lane FP8 `vload`s
- each half is widened by `pto.vmi.vcvt(..., pto.f32)`
- `ComputeScale` writes a compact converted scale buffer: one scale value per
  32 input elements
- `ComputeData` consumes that compact buffer with `vload(dist_mode="brc",
  group=8)`, matching the source `DIST_E2B_B32` scale-load contract
- dequant itself is just `vmul`
- output uses `vstore`, with an extra `vcvt` only for `bf16` / `f16`

This version is functionally correct and avoids the earlier invalid formulation
that mixed deinterleaved VMI data with a surface `vintlv`.

## Static efficiency comparison

The notes below compare the generated MI with the original handwritten ASC/CCE
for the FP8 path. This is a static code-shape comparison, not a profiler result.

### Overall conclusion

The current lowered MI is likely slower than the original ASC implementation.

- `f32` output: probably somewhat slower
- `bf16` / `f16` output: likely more noticeably slower

The main reason is not the math itself, but that the generated MI still needs
to materialize more explicit layout transforms around load/broadcast/store.

### Where the original ASC is tighter

The original source uses specialized distribution modes that already match the
intended data choreography fairly well:

- `DIST_DINTLV_B8` for the FP8 payload load
- `DIST_E2B_B32` for scale expansion
- direct store-side layout/stype handling for `f32`
- relatively compact cast-pack-store handling for `bf16` / `f16`

So a lot of the "how to arrange lanes" work is implicit in the source
instruction choice itself.

### What the current lowered MI does

For the `f32` path, the generated MI roughly expands into:

- 2 dense `vlds` for FP8 input
- 8 `vcvt` ops (`P0`..`P3` for each 256-lane half)
- compact scale conversion stores
- 2 direct `E2B_B32` scale loads for group-broadcast scale consumption
- 8 `vmul`
- multiple `vintlv` plus `pintlv_b32`
- 8 `vsts`

Compared with the source ASC, the extra cost is mainly on store-side re-layout
before `vsts`.

For the `bf16` / `f16` path, the gap is usually larger because lowering still
has to do more explicit rearrangement before the final packed 16-bit store.

## Interpretation

The current result should be treated as:

- semantically correct VMI surface modeling
- good for validating VMI expression and lowering correctness
- not yet the most efficient MI shape PTOAS could theoretically emit

The biggest optimization opportunities are likely:

- reduce redundant store-side layout materialization for `bf16` / `f16`
- improve lowering so the final MI better preserves the compactness of the
  source distribution-based implementation
