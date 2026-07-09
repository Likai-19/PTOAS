// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Upstream: cce/tile_kernels_port/rope/csrc/inc/rope_cce_shim.h
#ifndef ROPE_CCE_SHIM_H
#define ROPE_CCE_SHIM_H

#include <cstdint>

/*===========================================================================
 *
 * ROPE CCE SHIM — Typed wrappers around raw CCE intrinsics
 * ========================================================
 *
 * This header provides a small set of strongly-typed template wrappers
 * around the raw CCE SIMD intrinsics.  The wrappers use `__simd_callee__`
 * + `always_inline` to keep the compiler happy while enabling bisheng's
 * SIMD-inlining / fusion pass.
 *
 * Naming convention:
 *
 *   vlds_<mode>_<type>           — load into register
 *   vsts_<mode>_<type>           — store from register
 *   vcvt_<from>_to_<to>_<suffix> — type conversion
 *   vmul/vadd/vsub_<type>        — element-wise arithmetic
 *   vdintlv_x2 / vintlv_x2       — in-register deinterleave / interleave
 *   vbr_<type>                   — scalar broadcast
 *   make_mask[_b16](cnt)         — construct partial predicate mask
 *
 * Each wrapper maps 1:1 to a PTO IR primitive.  See the inline comments
 * above each wrapper for the specific PTO equivalent, hardware semantics,
 * and typical use cases within the RoPE kernel.
 *
 * Register width summary on Ascend 950DT (dav-c310-vec):
 *
 *   vector register         = 256 bytes = 2048 bits
 *   vector_f16 / vector_bf16 / vector_u16 = 128 elements
 *   vector_f32 / vector_u32 / vector_s32  = 64 elements
 *
 *===========================================================================*/

#ifndef __CPU_SIM
#define AICORE [aicore]
#else
#define AICORE
#endif

#ifdef __CPU_SIM
#define ROPE_CCE_INTERNAL inline
#define ROPE_SIMD_FN inline
#else
// `__simd_callee__` marks the function as a candidate for bisheng's
// SIMD-inlining pass — the compiler will then merge adjacent simd calls
// into a single wide op when possible.  `always_inline` ensures no call
// overhead even at -O0.
#define ROPE_CCE_INTERNAL AICORE inline __attribute__((always_inline))
#define ROPE_SIMD_FN __attribute__((always_inline)) __simd_callee__ inline
#endif

namespace rope_cce {

// Vector lane counts (elements per 256-byte register).
constexpr uint16_t VL_F32 = 64;   // fp32 / i32
constexpr uint16_t VL_F16 = 128;  // fp16 / bf16 / i16
// CCE 32-byte hardware block size (used for aligning strided UB offsets).
constexpr uint16_t BLOCK_BYTE_32 = 32;

// Integer ceiling helpers used throughout the kernel.
ROPE_CCE_INTERNAL constexpr int32_t cdiv(int32_t a, int32_t b) { return (a + b - 1) / b; }
ROPE_CCE_INTERNAL constexpr int32_t calign(int32_t a, int32_t b) { return (a + b - 1) / b * b; }

// Predicate register alias — used as the `mask` parameter of every
// predicated intrinsic (vadd, vmul, vcvt, vsts, ...).
using MaskReg = vector_bool;

namespace simd_inlined {

// Load 64 b16 values from UB into a 128-lane vector register using
// UNPK_B16 mode. The hardware places the 64 valid elements at EVEN
// halfword positions [0, 2, 4, ..., 126] within the register; the odd
// positions [1, 3, 5, ..., 127] are filled with zero/padding.
//
// This matches vlds_unpk_b16_bf16 below and pairs exclusively with
// vcvt_*_to_fp32_*even (PART_EVEN). Using vcvt_odd after UNPK_B16
// would extract the zero/padding lanes and produce garbage.
template <typename F16Dst, typename U16Src>
ROPE_SIMD_FN void vlds_unpk_b16(F16Dst &dst, U16Src src, int32_t off)
{
    vlds((vector_f16 &)dst, (__ubuf__ half *)src, off, UNPK_B16);
}

// Load 128 b16 values densely (NORM mode). All 128 halfword positions
// [0, 1, 2, ..., 127] in the register are filled with valid data.
// This is the load used by ComputeF16, which stays entirely in fp16
// arithmetic and does not widen to fp32 (no vcvt call needed).
template <typename F16Dst, typename U16Src>
ROPE_SIMD_FN void vlds_norm_b16(F16Dst &dst, U16Src src, int32_t off)
{
    vlds((vector_f16 &)dst, (__ubuf__ half *)src, off, NORM);
}

// Load 64 bf16 values from UB into a 128-lane vector register using
// UNPK_B16 mode. Same even-lane-only placement as vlds_unpk_b16 above.
template <typename BF16Dst, typename U16Src>
ROPE_SIMD_FN void vlds_unpk_b16_bf16(BF16Dst &dst, U16Src src, int32_t off)
{
    vlds((vector_bf16 &)dst, (__ubuf__ bfloat16_t *)src, off, UNPK_B16);
}

// Extract fp32 values from ODD halfword positions [1, 3, 5, ..., 127].
// UNUSED by this kernel: RoPE loads b16 data via UNPK_B16, which places
// valid elements at EVEN positions only, so PART_EVEN covers all 64
// elements. PART_ODD would extract zero/padding from the odd lanes.
//
// Kept in the shim for completeness. A kernel that fills all 128 b16
// lanes (e.g. via DINTLV_B16 x2 + vintlv x2 double-width loads, as in
// mx_quant) needs BOTH vcvt_even and vcvt_odd to widen the full register.
template <typename F32Dst, typename F16Src>
ROPE_SIMD_FN void vcvt_fp16_to_fp32_odd(F32Dst &dst, F16Src src, MaskReg mask)
{
    vcvt((vector_f32 &)dst, (vector_f16 &)src, mask, PART_ODD, MODE_ZEROING);
}

// Extract fp32 values from EVEN halfword positions [0, 2, 4, ..., 126].
// This is the widen half of the UNPK_B16 → PART_EVEN pair: after
// vlds_unpk_b16 places 64 valid b16 elements at even positions,
// vcvt_even recovers all 64 into a fp32 vector register.
template <typename F32Dst, typename F16Src>
ROPE_SIMD_FN void vcvt_fp16_to_fp32_even(F32Dst &dst, F16Src src, MaskReg mask)
{
    vcvt((vector_f32 &)dst, (vector_f16 &)src, mask, PART_EVEN, MODE_ZEROING);
}

// Extract fp32 values from EVEN halfword positions of a bf16 register.
// Pairs with vlds_unpk_b16_bf16 (UNPK_B16 load). Same even-only
// semantics as vcvt_fp16_to_fp32_even; no _odd counterpart needed.
template <typename F32Dst, typename BF16Src>
ROPE_SIMD_FN void vcvt_bf16_to_fp32_even(F32Dst &dst, BF16Src src, MaskReg mask)
{
    vcvt((vector_f32 &)dst, (vector_bf16 &)src, mask, PART_EVEN, MODE_ZEROING);
}

// --- Narrowing conversions: fp32 → fp16 / fp32 → bf16 ---
//
// Both use ROUND_R (round-to-nearest-even) and RS_DISABLE (no
// saturation).  PART_EVEN places the narrowed 16-bit result at EVEN
// halfword positions [0, 2, ..., 126] so that a later PK_B32 store
// can pack them densely into UB.
//
// After this call the output register has valid bf16/fp16 bits only at
// the even halfword positions; the odd lanes are zero.
//
// Note: `vcvt_fp32_to_fp16_narrow` is NOT USED in this kernel.
// ComputeF16 stays entirely in fp16 (no widening/narrowing at all) and
// ComputeF32 stays entirely in fp32.  It is kept in the shim for
// completeness (useful when porting a fp32-accum → fp16-output kernel).
template <typename F16Dst, typename F32Src>
ROPE_SIMD_FN void vcvt_fp32_to_fp16_narrow(F16Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f16 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
}

template <typename BF16Dst, typename F32Src>
ROPE_SIMD_FN void vcvt_f32_to_bf16_narrow(BF16Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_bf16 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
}

// --- Arithmetic: element-wise vmul / vadd / vsub in fp16 or fp32 ---
//
// PTO IR: pto.vmul, pto.vadd, pto.vsub
// Hardware: 1-cycle throughput per instruction (fully pipelined).
//
// `mask` controls which lanes participate in the compute.
// `MODE_ZEROING` writes 0 to inactive lanes (vs. `MODE_MERGE` which
// keeps the prior register bits).  For RoPE we always use `MODE_ZEROING`.
template <typename F32Dst, typename F32SrcA, typename F32SrcB>
ROPE_SIMD_FN void vmul_f32(F32Dst &dst, F32SrcA a, F32SrcB b, MaskReg mask)
{
    vmul((vector_f32 &)dst, (vector_f32 &)a, (vector_f32 &)b, mask, MODE_ZEROING);
}

template <typename F16Dst, typename F16SrcA, typename F16SrcB>
ROPE_SIMD_FN void vmul_f16(F16Dst &dst, F16SrcA a, F16SrcB b, MaskReg mask)
{
    vmul((vector_f16 &)dst, (vector_f16 &)a, (vector_f16 &)b, mask, MODE_ZEROING);
}

template <typename F32Dst, typename F32SrcA, typename F32SrcB>
ROPE_SIMD_FN void vadd_f32(F32Dst &dst, F32SrcA a, F32SrcB b, MaskReg mask)
{
    vadd((vector_f32 &)dst, (vector_f32 &)a, (vector_f32 &)b, mask, MODE_ZEROING);
}

template <typename F16Dst, typename F16SrcA, typename F16SrcB>
ROPE_SIMD_FN void vadd_f16(F16Dst &dst, F16SrcA a, F16SrcB b, MaskReg mask)
{
    vadd((vector_f16 &)dst, (vector_f16 &)a, (vector_f16 &)b, mask, MODE_ZEROING);
}

template <typename F32Dst, typename F32SrcA, typename F32SrcB>
ROPE_SIMD_FN void vsub_f32(F32Dst &dst, F32SrcA a, F32SrcB b, MaskReg mask)
{
    vsub((vector_f32 &)dst, (vector_f32 &)a, (vector_f32 &)b, mask, MODE_ZEROING);
}

template <typename F16Dst, typename F16SrcA, typename F16SrcB>
ROPE_SIMD_FN void vsub_f16(F16Dst &dst, F16SrcA a, F16SrcB b, MaskReg mask)
{
    vsub((vector_f16 &)dst, (vector_f16 &)a, (vector_f16 &)b, mask, MODE_ZEROING);
}

// --- In-register interleave / deinterleave ---
//
// PTO IR: pto.vintlv, pto.vdintlv
//
// vdintlv_x2(dst_even, dst_odd, s0, s1):
//   Splits elements with EVEN and ODD indices from two source registers
//   into two output registers:
//     dst_even[i] = src[2i]       (even indices: 0, 2, 4, ...)
//     dst_odd[i]  = src[2i + 1]   (odd  indices: 1, 3, 5, ...)
//   Used by RoPE INTERLEAVE mode to separate the "real" from "imaginary"
//   part of each (x[2k], x[2k+1]) complex pair.
//
// vintlv_x2(dst_low, dst_high, s0, s1):
//   Merges two streams element-by-element:
//     dst_low[i]  = s0[i/2] if i even else s1[i/2]     (interleaved first half)
//     dst_high[i] = ...                                  (second half, for cnt > lane width)
//   In RoPE this is used to re-interleave (neg_odd, even) back into:
//     [-x1, x0, -x3, x2, ...]  — the "rotated partner" vector.
//
// The "_x2" suffix in these wrappers reflects that the underlying CCE
// `vintlv` / `vdintlv` intrinsics always produce CONSUMER two outputs.
template <typename Dst, typename Src>
ROPE_SIMD_FN void vdintlv_x2(Dst &d0, Dst &d1, Src s0, Src s1)
{
    vdintlv(d0, d1, s0, s1);
}

template <typename Dst, typename Src>
ROPE_SIMD_FN void vintlv_x2(Dst &d0, Dst &d1, Src s0, Src s1)
{
    vintlv(d0, d1, s0, s1);
}

// --- Broadcast: scalar → every lane of a vector register ---
//
// PTO IR: pto.vbr
//
// `vbr_f32(dst, val)` sets each of the 64 fp32 lanes to `val`.
// `vbr_f16(dst, val)` sets each of the 128 fp16 lanes to `val`.
//
// Used to materialise the `-1.0` constant for the odd-element negation
// in INTERLEAVE mode.  A single broadcast is much more efficient than
// loading a pre-filled constant tensor from UB.
template <typename F32Dst>
ROPE_SIMD_FN void vbr_f32(F32Dst &dst, float val)
{
    vbr((vector_f32 &)dst, val);
}

template <typename F16Dst>
ROPE_SIMD_FN void vbr_f16(F16Dst &dst, half val)
{
    vbr((vector_f16 &)dst, val);
}

// --- Mask construction: plt_b32 / plt_b16 ---
//
// PTO IR: pto.plt_b32, pto.plt_b16
//
// `plt_b32(cnt, POST_UPDATE)` sets the first `cnt` bits of a b32 mask
// (1 bit per fp32 lane, 64 lanes total), leaving the rest zero.
// `plt_b16(cnt, POST_UPDATE)` is the b16 variant (1 bit per fp16 lane,
// 128 lanes total).
//
// `POST_UPDATE` means that the predicate counter advances AFTER the
// current use — this is the standard choice for a one-shot partial
// mask (the counter doesn't carry state across calls in our code).
//
// Usage: `cnt = min(remaining_elements, VL)` for the last partial block
// of a D tile — otherwise full-block loads use cnt = VL directly.
ROPE_CCE_INTERNAL MaskReg make_mask(uint32_t cnt)
{
    return plt_b32(cnt, POST_UPDATE);
}

ROPE_CCE_INTERNAL MaskReg make_mask_b16(uint32_t cnt)
{
    return plt_b16(cnt, POST_UPDATE);
}

// --- Store operations: PK_B32 vs NORM_B16 vs NORM_B32 ---
//
// PTO IR: pto.vsts with the appropriate {dist = "..."}.
//
// Store modes:
//   * PK_B32:    PACK_B32 store. Used after `vcvt_*_to_*_even`
//                conversions that placed 16-bit results at EVEN
//                halfword positions [0, 2, ..., 126] of the register.
//                PK_B32 extracts those even-halfword slots and writes
//                them densely (N consecutive halfwords) to UB.
//                The mask controls how many halfwords are written.
//   * NORM_B16:  Dense store of N fp16 elements from lanes [0 .. N).
//                Used by ComputeF16 where each store writes the full
//                128-lane packed fp16 block.
//   * NORM_B32:  Dense store of N fp32 elements from lanes [0 .. N).
//                Used by ComputeF32.
//
// `vsts_pk_b32` (fp16 variant): NOT USED in this kernel.
//   ComputeF16 stores via `vsts_norm_b16` directly, and ComputeBf16
//   stores via `vsts_pk_b32_bf16`. Kept for completeness (useful when
//   porting a fp32-accum → fp16-output kernel).
template <typename F16Src, typename U16Dst>
ROPE_SIMD_FN void vsts_pk_b32(F16Src src, U16Dst dst, int32_t off, MaskReg mask)
{
    vsts((vector_f16 &)src, (__ubuf__ half *)dst, off, PK_B32, mask);
}

// `vsts_pk_b32_bf16`: USED by ComputeBf16 (narrowing fp32 → bf16 store).
template <typename BF16Src, typename U16Dst>
ROPE_SIMD_FN void vsts_pk_b32_bf16(BF16Src src, U16Dst dst, int32_t off, MaskReg mask)
{
    vsts((vector_bf16 &)src, (__ubuf__ bfloat16_t *)dst, off, PK_B32, mask);
}

template <typename F16Src, typename U16Dst>
ROPE_SIMD_FN void vsts_norm_b16(F16Src src, U16Dst dst, int32_t off, MaskReg mask)
{
    vsts((vector_f16 &)src, (__ubuf__ half *)dst, off, NORM_B16, mask);
}

template <typename F32Src, typename U32Dst>
ROPE_SIMD_FN void vsts_norm_b32(F32Src src, U32Dst dst, int32_t off, MaskReg mask)
{
    vsts((vector_f32 &)src, (__ubuf__ float *)dst, off, NORM_B32, mask);
}

// --- fp32 load: vlds_norm_b32 ---
//
// Load 64 fp32 elements densely from UB into a 64-lane fp32 vector
// register. Used by ComputeF32 (NORM load matches the NORM store it
// round-trips through).
//
// PTO IR: pto.vlds {dist = "DIST_NORM_B32"}
template <typename F32Dst, typename U32Src>
ROPE_SIMD_FN void vlds_norm_b32(F32Dst &dst, U32Src src, int32_t off)
{
    vlds((vector_f32 &)dst, (__ubuf__ float *)src, off, NORM);
}

}

}

#endif
