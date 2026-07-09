// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Upstream: cce/tile_kernels_port/rope/csrc/inc/rope_cce_compute.h
#ifndef ROPE_CCE_COMPUTE_H
#define ROPE_CCE_COMPUTE_H

#include "rope_cce_shim.h"

#if defined(__DAV_VEC__)

namespace rope_cce {

/*===========================================================================
 *
 * ROPES — ROTARY POSITIONAL EMBEDDING FOR NPU CCE
 * ================================================
 *
 * This file contains the **in-register compute kernels** for RoPE
 * (Rotary Positional Embedding), executed on a single AIV (vector core).
 *
 * **Mathematical model** (per complex pair `[x0, x1]`):
 *
 *     y0 = x0 * cos(θ) − x1 * sin(θ)
 *     y1 = x0 * sin(θ) + x1 * cos(θ)
 *
 * where θ = position × frequency, and `(cos, sin)` are precomputed tables
 * stored in the same UB as the input `x`.
 *
 * **Data layout on UB** (AB layout, `[B, S, N, D]`):
 *
 *   For a given `(s, n, d)` the RoPE kernel operates on two halves of `D`:
 *
 *     x_0 = x[s, n, d ∈ [0, D/2)]      — "low"  half
 *     x_1 = x[s, n, d ∈ [D/2, D)]      — "high" half
 *
 *   cos and sin are similarly stored as two halves, sharing the same `s` stride.
 *
 * **Two rotation modes**:
 *
 *   Half (mode 0, NeoX-style):
 *     Pairs are `(x[d], x[d+D/2])` for each `d ∈ [0, D/2)`.
 *     Load two independent half-registers, apply the rotation, store back.
 *     Used by Llama, Qwen, and most transformer variants.
 *
 *   Interleave (mode 1, GPT-J style):
 *     Pairs are `(x[2k], x[2k+1])` for each even index `2k`.
 *     Uses `vdintlv_x2` / `vintlv_x2` to form the partner pairs within
 *     one register.
 *     Used by GPT-J and PaLM.
 *
 * **Intrinsic call pattern** (CCE intrinsics, no AscendC API):
 *
 *   - `vlds`       — load from UB into vector register
 *   - `vcvt`       — type conversion (b16↔f32, fp16↔fp32)
 *   - `vmul/vadd/vsub` — element-wise floating-point arithmetic
 *   - `vdintlv/vintlv` — deinterleave/interleave two registers
 *   - `vsts`       — store from register to UB
 *   - `plt_b16/b32` — construct mask for `cnt` active lanes
 *
 * **Three compute variants**:
 *
 *   - `ComputeF16`   — fp16 throughout (fastest, lowest precision)
 *   - `ComputeBf16`  — bf16 input → fp32 inner math → bf16 output  (mid precision, same speed)
 *   - `ComputeF32`   — fp32 throughout (slowest, bit-exact reference)
 *
 * All three variants follow the same outer loop structure:
 *
 *     for each sequence slice `s`:
 *         for each half-d-block `rep`:
 *             load cos/sin half-registers  (shared across heads)
 *             for each head `n`:
 *                 load x half-registers  (per head)
 *                 compute y0 = cos * x0 - sin * x1
 *                 compute y1 = sin * x0 + cos * x1
 *                 store y0, y1
 *
 * The cos/sin loads are hoisted outside the `n` loop because cos/sin are
 * shared across all heads (`N` dimension).
 *
 *===========================================================================*/
/*===========================================================================
 * ComputeF16 — fp16 RoPE computation (HALF and INTERLEAVE modes)
 *
 * INPUTS:  x, cos, sin in UB, all fp16 (`__ubuf__ half *`)
 * OUTPUTS: y in UB, fp16
 * PRECISION: full fp16 native — each vmul/vadd/vsub is an IEEE fp16 op.
 *
 * Data flow per (s, n):
 *   1. Load cos_0, cos_1, sin_0, sin_1, x_0, x_1 from UB (vlds NORM).
 *   2. y0 = cos_0 * x0 - sin_0 * x1       ;  vmul, vmul, vsub
 *   3. y1 = cos_1 * x1 + sin_1 * x0       ;  vmul, vmul, vadd
 *   4. Store y0, y1 to UB (vsts NORM_B16).
 *
 * Vector register width: 256 B = 128 fp16 elements (VL_F16 = 128).
 * Mask granularity: b16 (one bit per fp16 element).
 *
 * PTO equivalents (for cross-reference):
 *   - vlds_norm_b16      →  pto.vlds {dist = "DIST_NORM_B16"}
 *   - vmul_f16, vadd_f16 →  pto.vmul / pto.vadd
 *   - vsub_f16           →  pto.vsub
 *   - vsts_norm_b16      →  pto.vsts {dist = "DIST_NORM_B16"}
 *
 * Register pressure: 10 vector_f16 registers (all 256 B each).
 *===========================================================================*/
ROPE_CCE_INTERNAL void ComputeF16(
    __ubuf__ uint16_t *x_ub16,
    __ubuf__ uint16_t *cos_ub16,
    __ubuf__ uint16_t *sin_ub16,
    __ubuf__ uint16_t *y_ub16,
    int32_t sCount, int32_t nCount,
    int32_t dLen, int32_t dAlign,
    int32_t xSStep, int32_t xNStep,
    int32_t csSStep,
    int32_t ySStep, int32_t yNStep,
    int32_t mode)
{
    __VEC_SCOPE__
    {
        // Expected UB effect of this vector scope (local tensor view):
        //   x_ub16   -> x:   fp16[sCount, nCount, dLen] with strides xSStep/xNStep
        //   cos_ub16 -> cos: fp16[sCount, dLen]          with stride csSStep
        //   sin_ub16 -> sin: fp16[sCount, dLen]          with stride csSStep
        //   y_ub16   -> y:   fp16[sCount, nCount, dLen] with strides ySStep/yNStep
        //   Typical values from vf_sim tests:
        //     correctness default: sCount=15, nCount=32, dLen=dAlign=64
        //     xNStep=yNStep=csSStep=64, xSStep=ySStep=nCount*64=2048
        //     wall-time configs: (s,n)=(1,2),(15,4),(15,8),(15,16),(15,32)
        //     so xSStep/ySStep range from 128 to 2048 elements; compile-time caps
        //     are sCount<=15 and nCount<=32 in the standalone VF harness.
        //
        //   Pseudocode after this scope completes:
        //     y[...] is overwritten; x/cos/sin are read-only.
        //
        //     if mode == 0:  # HALF / NeoX layout
        //       half = dLen // 2
        //       for s in range(sCount):
        //         for n in range(nCount):
        //           for d in range(half):
        //             y[s,n,d]      = fp16(x[s,n,d]      * cos[s,d]
        //                                  - x[s,n,d+half] * sin[s,d])
        //             y[s,n,d+half] = fp16(x[s,n,d+half] * cos[s,d+half]
        //                                  + x[s,n,d]      * sin[s,d+half])
        //
        //     else:  # INTERLEAVE / GPT-J layout
        //       rot = [-x1, x0, -x3, x2, ...] per (s,n) row
        //       y[s,n,:] = fp16(x[s,n,:] * cos[s,:] + rot * sin[s,:])
        __ubuf__ half *xH = (__ubuf__ half *)x_ub16;
        __ubuf__ half *cosH = (__ubuf__ half *)cos_ub16;
        __ubuf__ half *sinH = (__ubuf__ half *)sin_ub16;
        __ubuf__ half *yH = (__ubuf__ half *)y_ub16;

        int32_t halfD = dLen / 2;
        int32_t halfDAl = calign(halfD, (int32_t)BLOCK_BYTE_32 / 2);

        vector_f16 hx1, hx2, hc1, hc2, hs1, hs2, ht1, ht2, hout1, hout2;

        // ---- HALF mode (mode == 0, NeoX-style) ----
        //
        // Loop structure:
        //   s ∈ [0, sCount):           outer sequence slice
        //   rep ∈ [0, repeatTimes):    inner half-D tile (128 fp16 per iter)
        //     [cos/sin loads hoisted here — shared across all n]
        //     n ∈ [0, nCount):         per-head loop
        //
        // Register roles (each 256 B = 128 fp16):
        //   hc1 / hc2: cos values for (low half, high half) of D
        //   hs1 / hs2: sin values for (low half, high half) of D
        //   hx1 / hx2: x input for (low half, high half) of D  (per head)
        //   ht1 / ht2: temp intermediates
        //   hout1 / hout2: y output for (low half, high half)
        //
        // repeatTimesF16 = ceil(halfD / 128) — how many 128-element chunks
        // are needed to cover half of D.
        if (mode == 0) {
            int32_t repeatTimesF16 = cdiv(halfD, (int32_t)VL_F16);
            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t rep = 0; rep < (uint16_t)repeatTimesF16; rep++) {
                    uint16_t elemOff = (uint16_t)((uint32_t)rep * VL_F16);
                    uint32_t cnt = (uint32_t)(halfD - (int32_t)elemOff);
                    if (cnt > (uint32_t)VL_F16) cnt = (uint32_t)VL_F16;
                    // b16 granularity mask — one bit per fp16 element, `cnt` active.
                    MaskReg mask16 = simd_inlined::make_mask_b16(cnt);

                    // ---- Hoisted cos/sin loads (shared across heads) ----
                    // Load the 4 half-width cos/sin halves:
                    //   hc1 = cos[s, d=elemOff .. elemOff+128]        (low half)
                    //   hc2 = cos[s, d=elemOff+halfDAl ..             ]  (high half)
                    //   hs1 = sin[s, d=elemOff .. elemOff+128]        (low half)
                    //   hs2 = sin[s, d=elemOff+halfDAl ..             ]  (high half)
                    simd_inlined::vlds_norm_b16(hc1, cosH + csOff + elemOff, 0);
                    simd_inlined::vlds_norm_b16(hc2, cosH + csOff + elemOff + halfDAl, 0);
                    simd_inlined::vlds_norm_b16(hs1, sinH + csOff + elemOff, 0);
                    simd_inlined::vlds_norm_b16(hs2, sinH + csOff + elemOff + halfDAl, 0);

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x loads ----
                        // Load x halves for this head/position:
                        //   hx1 = x[s, n, d=elemOff .. elemOff+128]       (low half)
                        //   hx2 = x[s, n, d=elemOff+halfDAl ..       ]    (high half)
                        simd_inlined::vlds_norm_b16(hx1, xH + xOff + elemOff, 0);
                        simd_inlined::vlds_norm_b16(hx2, xH + xOff + elemOff + halfDAl, 0);

                        // ---- Compute y0 = cos_0 * x0 - sin_0 * x1 ----
                        simd_inlined::vmul_f16(ht1, hc1, hx1, mask16);   // ht1 = cos_0 * x0
                        simd_inlined::vmul_f16(ht2, hs1, hx2, mask16);   // ht2 = sin_0 * x1
                        simd_inlined::vsub_f16(hout1, ht1, ht2, mask16); // y0  = cos_0*x0 - sin_0*x1

                        // ---- Compute y1 = cos_1 * x1 + sin_1 * x0 ----
                        simd_inlined::vmul_f16(ht1, hc2, hx2, mask16);   // ht1 = cos_1 * x1
                        simd_inlined::vmul_f16(ht2, hs2, hx1, mask16);   // ht2 = sin_1 * x0
                        simd_inlined::vadd_f16(hout2, ht1, ht2, mask16); // y1  = cos_1*x1 + sin_1*x0

                        // ---- Store result (dense 128-fp16 → UB) ----
                        simd_inlined::vsts_norm_b16(hout1, yH + yOff + elemOff, 0, mask16);
                        simd_inlined::vsts_norm_b16(hout2, yH + yOff + elemOff + halfDAl, 0, mask16);
                    }
                }
            }
        } else {
            // ---- INTERLEAVE mode (mode == 1, GPT-J-style) ----
            //
            // Pairs are adjacent elements: (x[2k], x[2k+1]).
            // Strategy:
            //   1. Load 64 contiguous fp16 (x[0..63]).
            //   2. `vdintlv_x2(even, odd, x, x)` splits into:
            //        even = x[0], x[2], ..., x[62]   (32 elements)
            //        odd  = x[1], x[3], ..., x[63]   (32 elements)
            //   3. Negate the odd part to form the "imaginary" partner:
            //        neg_odd = -x[1], -x[3], ..., -x[63]
            //   4. `vintlv_x2(low, high, neg_odd, even)` rebuilds:
            //        new = [-x1, x0, -x3, x2, ..., -x63, x62]
            //      This is exactly the "rotated partner" vector needed for
            //      y = x * cos + rotated(x) * sin.
            //
            // blockSize is set to 64 (VL_F32) because each interleave step
            // consumes 64 fp16 elements (32 even + 32 odd). 256 B register
            // = 4 such blocks per iteration.
            //
            // Register roles (each 256 B = 128 fp16):
            //   xr     : loaded x block
            //   cosr   : loaded cos block  (64 fp16, interleaved pattern)
            //   sinr   : loaded sin block  (same)
            //   heven  : even-indexed elements after vdintlv
            //   hodd   : odd-indexed elements after vdintlv
            //   hnegodd: -odd
            //   hxnew / hxnew_hi: rotated partner vector (low / high halves)
            //   hta / htb: arithmetic intermediates
            //   negOne : broadcast scalar -1.0f (all 128 lanes)
            vector_f16 xr, cosr, sinr, heven, hodd, hnegodd, hxnew, hxnew_hi, hta, htb;
            vector_f16 negOne;
            simd_inlined::vbr_f16(negOne, (half)(-1.0f));

            int32_t blockSize = (int32_t)VL_F32;        // 64 fp16 per interleave block
            int32_t dBlocks = cdiv(dLen, blockSize);

            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t blk = 0; blk < (uint16_t)dBlocks; blk++) {
                    int32_t off = (int32_t)blk * blockSize;
                    int32_t remaining = dLen - off;
                    uint32_t cnt = (remaining > blockSize) ? (uint32_t)blockSize : (uint32_t)remaining;
                    // pairCnt = number of (even, odd) pairs in this block = cnt/2 (rounded up).
                    uint32_t pairCnt = (cnt + 1U) / 2U;

                    MaskReg mask = simd_inlined::make_mask_b16(cnt);      // full block
                    MaskReg maskPair = simd_inlined::make_mask_b16(pairCnt); // pair-only

                    // ---- Hoisted cos/sin loads (shared across heads) ----
                    simd_inlined::vlds_norm_b16(cosr, cosH + csOff + off, 0);
                    simd_inlined::vlds_norm_b16(sinr, sinH + csOff + off, 0);

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x load ----
                        // xr = x[s, n, d=off .. off+blockSize]  (64 contiguous fp16)
                        simd_inlined::vlds_norm_b16(xr, xH + xOff + off, 0);

                        // ---- Form the rotated partner vector ----
                        // Step 1: split xr into even/odd half-index streams:
                        //   heven = x[0], x[2], ..., x[62]
                        //   hodd  = x[1], x[3], ..., x[63]
                        simd_inlined::vdintlv_x2(heven, hodd, xr, xr);
                        // Step 2: negate the odd elements to match the rotation
                        // identity [-x1, -x3, ..., -x63].
                        simd_inlined::vmul_f16(hnegodd, hodd, negOne, maskPair);
                        // Step 3: re-interleave (-odd, even) to form the partner:
                        //   hxnew = [-x1, x0, -x3, x2, ..., -x63, x62]
                        //   hxnew_hi = high half of the interleaved result (for cnt > 64)
                        simd_inlined::vintlv_x2(hxnew, hxnew_hi, hnegodd, heven);

                        // ---- y = x * cos + rotated(x) * sin ----
                        simd_inlined::vmul_f16(hta, xr, cosr, mask);     // hta = x * cos
                        simd_inlined::vmul_f16(htb, hxnew, sinr, mask);  // htb = rotated(x) * sin
                        simd_inlined::vadd_f16(htb, hta, htb, mask);     // y   = hta + htb
                        simd_inlined::vsts_norm_b16(htb, yH + yOff + off, 0, mask);
                    }
                }
            }
        }
    }
}

/*
 * ComputeBf16 — bf16 RoPE computation (HALF and INTERLEAVE modes)
 * x/y are bf16 in GM, cos/sin are fp16 in GM.
 * Internal compute is done in fp32: load bf16→fp32, load fp16→fp32,
 * fp32 math, fp32→bf16 store.
 *
 * Load/conversion convention:
 *   All b16 loads in this function use UNPK_B16 mode, which places 64
 *   valid elements at the EVEN halfword positions [0, 2, ..., 126] of
 *   the 128-lane register; the odd positions contain zero/padding.
 *   This is why only vcvt_*_to_fp32_even (PART_EVEN) is used to widen
 *   to fp32 — it recovers all 64 valid elements. vcvt_*_to_fp32_odd
 *   would extract the padding lanes and produce garbage. (The _odd
 *   wrapper exists in the shim for other kernels that fill all 128
 *   lanes via denser loads, e.g. mx_quant with DINTLV_B16 x2.)
 *
 * HALF mode (mode==0): loads bf16 x and fp16 cos/sin via UNPK_B16,
 *   widens both to fp32 via PART_EVEN, does fp32 mul/add/sub, narrows
 *   back to bf16 via PART_EVEN + PK_B32.
 *
 * INTERLEAVE mode (mode==1): same load pattern, plus fp32-domain
 *   vdintlv/vintlv for the even/odd pair shuffle. Avoids the
 *   incorrect fp16 bit-reinterpretation approach.
 */
ROPE_CCE_INTERNAL void ComputeBf16(
    __ubuf__ uint16_t *x_ub16,
    __ubuf__ uint16_t *cos_ub16,
    __ubuf__ uint16_t *sin_ub16,
    __ubuf__ uint16_t *y_ub16,
    int32_t sCount, int32_t nCount,
    int32_t dLen, int32_t dAlign,
    int32_t xSStep, int32_t xNStep,
    int32_t csSStep,
    int32_t ySStep, int32_t yNStep,
    int32_t mode)
{
    __VEC_SCOPE__
    {
        // Expected UB effect of this vector scope (local tensor view):
        //   x_ub16   -> x:   bf16[sCount, nCount, dLen] with strides xSStep/xNStep
        //   cos_ub16 -> cos: fp16[sCount, dLen]          with stride csSStep
        //   sin_ub16 -> sin: fp16[sCount, dLen]          with stride csSStep
        //   y_ub16   -> y:   bf16[sCount, nCount, dLen] with strides ySStep/yNStep
        //   Typical values from vf_sim tests:
        //     correctness default: sCount=15, nCount=32, dLen=dAlign=64
        //     xNStep=yNStep=csSStep=64, xSStep=ySStep=nCount*64=2048
        //     wall-time configs: (s,n)=(1,2),(15,4),(15,8),(15,16),(15,32)
        //     so xSStep/ySStep range from 128 to 2048 elements; compile-time caps
        //     are sCount<=15 and nCount<=32 in the standalone VF harness.
        //
        //   Pseudocode after this scope completes:
        //     y[...] is overwritten; x/cos/sin are read-only.
        //     All arithmetic below is fp32; stores narrow back to bf16.
        //
        //     if mode == 0:  # HALF / NeoX layout
        //       half = dLen // 2
        //       for s in range(sCount):
        //         for n in range(nCount):
        //           xf = x[s,n,:].astype(float32)
        //           cf = cos[s,:].astype(float32)
        //           sf = sin[s,:].astype(float32)
        //           for d in range(half):
        //             y[s,n,d]      = bf16(xf[d]      * cf[d]
        //                                  - xf[d+half] * sf[d])
        //             y[s,n,d+half] = bf16(xf[d+half] * cf[d+half]
        //                                  + xf[d]      * sf[d+half])
        //
        //     else:  # INTERLEAVE / GPT-J layout
        //       xf = x[s,n,:].astype(float32)
        //       rot = [-xf[1], xf[0], -xf[3], xf[2], ...]
        //       y[s,n,:] = bf16(xf * cos[s,:].astype(float32)
        //                        + rot * sin[s,:].astype(float32))
        // ---- Typed UB pointer setup ----
        // x/y are bf16 in UB; cos/sin are fp16 in UB.
        // Casts only set the address-space / element-type view.
        __ubuf__ bfloat16_t *xB = (__ubuf__ bfloat16_t *)x_ub16;
        __ubuf__ bfloat16_t *yB = (__ubuf__ bfloat16_t *)y_ub16;
        __ubuf__ half *cosH = (__ubuf__ half *)cos_ub16;
        __ubuf__ half *sinH = (__ubuf__ half *)sin_ub16;

        // halfD = D/2 (the width of each rotatable half)
        // halfDAl = halfD aligned up to (BLOCK_BYTE_32/2) = 16 elements,
        //   ensuring strided offsets land on 32-byte boundaries.
        // repeatTimes = ceil(halfD / VL_F32) — each iteration processes
        //   VL_F32 = 64 fp32 elements (= one full vector register).
        int32_t halfD = dLen / 2;
        int32_t halfDAl = calign(halfD, (int32_t)BLOCK_BYTE_32 / 2);
        int32_t repeatTimes = cdiv(halfD, (int32_t)VL_F32);

        // ---- Register allocation ----
        // fc, fc2, fs, fs2: cos/sin widened to fp32 (low and high halves)
        // fx0, fx1: x halves widened to fp32
        // ft, ft2: arithmetic intermediates
        // htmp0, htmp1: scratch 128-fp16 / 128-bf16 slots (used as the
        //   landing zone for UNPK_B16 loads and for PK_B32 stores)
        vector_f32 fc, fc2, fs, fs2, fx0, fx1, ft, ft2;
        vector_f16 htmp0, htmp1;

        if (mode == 0) {
            // ==== ComputeBf16 HALF mode ====
            // Same outer loop structure as ComputeF16 HALF, but with
            // widening load + narrowing store:
            //   vlds(UNPK_B16)         → b16 at even register positions
            //   vcvt_*_to_fp32_even    → fp32 (64 lanes, matching active mask)
            //   vmul / vadd / vsub     → fp32 arithmetic
            //   vcvt_f32_to_bf16_narrow → bf16 at even register positions
            //   vsts(PK_B32)           → dense bf16 written back to UB
            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t rep = 0; rep < (uint16_t)repeatTimes; rep++) {
                    uint32_t elemOff = (uint32_t)rep * (uint32_t)VL_F32;
                    int32_t elemOffH = (int32_t)elemOff;
                    uint32_t cnt = (uint32_t)(halfD - elemOffH);
                    if (cnt > (uint32_t)VL_F32) cnt = (uint32_t)VL_F32;
                    // 64-lane (b32) mask — `cnt` active fp32 lanes.
                    MaskReg mask32 = simd_inlined::make_mask(cnt);

                    // ---- Hoisted cos/sin load-and-widen (shared across heads) ----
                    // For both halves (low & high) of D:
                    //   load 64 fp16 via UNPK_B16 (landing at even positions)
                    //   widen to 64 fp32 via PART_EVEN.
                    simd_inlined::vlds_unpk_b16(htmp0, cosH + csOff + elemOffH, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fc, htmp0, mask32);   // fc  = cos_low  (fp32)
                    simd_inlined::vlds_unpk_b16(htmp0, sinH + csOff + elemOffH, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fs, htmp0, mask32);   // fs  = sin_low  (fp32)

                    simd_inlined::vlds_unpk_b16(htmp0, cosH + csOff + elemOffH + halfDAl, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fc2, htmp0, mask32);  // fc2 = cos_high (fp32)
                    simd_inlined::vlds_unpk_b16(htmp0, sinH + csOff + elemOffH + halfDAl, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fs2, htmp0, mask32);  // fs2 = sin_high (fp32)

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x load-and-widen ----
                        // Load two bf16 halves via UNPK_B16, widen each to fp32.
                        simd_inlined::vlds_unpk_b16_bf16(htmp0, xB + xOff + elemOffH, 0);
                        simd_inlined::vcvt_bf16_to_fp32_even(fx0, htmp0, mask32);  // fx0 = x_low  (fp32)
                        simd_inlined::vlds_unpk_b16_bf16(htmp1, xB + xOff + elemOffH + halfDAl, 0);
                        simd_inlined::vcvt_bf16_to_fp32_even(fx1, htmp1, mask32);  // fx1 = x_high (fp32)

                        // ---- Compute y0 = cos_low * x_low - sin_low * x_high ----
                        simd_inlined::vmul_f32(ft,  fc, fx0, mask32);   // ft  = cos_low * x_low
                        simd_inlined::vmul_f32(ft2, fs, fx1, mask32);   // ft2 = sin_low * x_high
                        simd_inlined::vsub_f32(ft,  ft, ft2, mask32);   // y0  = ft - ft2

                        // ---- Narrow y0 to bf16 and store (low half of y) ----
                        simd_inlined::vcvt_f32_to_bf16_narrow(htmp0, ft, mask32); // fp32 → bf16 at even lanes
                        simd_inlined::vsts_pk_b32_bf16(htmp0, yB + yOff + elemOffH, 0, mask32);  // dense bf16 → UB

                        // ---- Compute y1 = cos_high * x_high + sin_high * x_low ----
                        simd_inlined::vmul_f32(ft,  fc2, fx1, mask32);  // ft  = cos_high * x_high
                        simd_inlined::vmul_f32(ft2, fs2, fx0, mask32);  // ft2 = sin_high * x_low
                        simd_inlined::vadd_f32(ft,  ft, ft2, mask32);   // y1  = ft + ft2

                        // ---- Narrow y1 to bf16 and store (high half of y) ----
                        simd_inlined::vcvt_f32_to_bf16_narrow(htmp0, ft, mask32);
                        simd_inlined::vsts_pk_b32_bf16(htmp0, yB + yOff + elemOffH + halfDAl, 0, mask32);
                    }
                }
            }
        } else {
            // ==== ComputeBf16 INTERLEAVE mode ====
            // Same strategy as ComputeF16 INTERLEAVE (vdintlv/vintlv),
            // but performed in fp32 register space after widening the
            // bf16 x via UNPK_B16 + PART_EVEN.
            //
            // negOne is a broadcast fp32 -1.0 to all 64 lanes — used
            // to negate the odd-indexed x elements with a single vmul.
            vector_f32 fx_even, fx_odd, negOne;
            simd_inlined::vbr_f32(negOne, -1.0f);

            // blockSize = 64 fp32 lanes; each block covers 64 bf16 input elements.
            int32_t blockSize = (int32_t)VL_F32;
            int32_t dBlocks = cdiv(dLen, blockSize);

            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t blk = 0; blk < (uint16_t)dBlocks; blk++) {
                    int32_t off = (int32_t)blk * blockSize;
                    int32_t remaining = dLen - off;
                    uint32_t cnt = (remaining > blockSize) ? (uint32_t)blockSize : (uint32_t)remaining;
                    // pairCnt = ceil(cnt/2) — number of even/odd pairs in this block.
                    uint32_t pairCnt = (cnt + 1U) / 2U;

                    MaskReg mask32   = simd_inlined::make_mask(cnt);      // full block (b32)
                    MaskReg maskHalf = simd_inlined::make_mask(pairCnt);  // pairs only

                    // ---- Hoisted cos/sin load-and-widen ----
                    simd_inlined::vlds_unpk_b16(htmp0, cosH + csOff + off, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fc, htmp0, mask32);   // fc = cos (fp32)
                    simd_inlined::vlds_unpk_b16(htmp0, sinH + csOff + off, 0);
                    simd_inlined::vcvt_fp16_to_fp32_even(fs, htmp0, mask32);   // fs = sin (fp32)

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x load-and-widen ----
                        simd_inlined::vlds_unpk_b16_bf16(htmp0, xB + xOff + off, 0);
                        simd_inlined::vcvt_bf16_to_fp32_even(fx0, htmp0, mask32);  // fx0 = x (fp32)

                        // ---- Form the rotated partner in fp32 space ----
                        // Step 1: split into even/odd streams:
                        //   fx_even = x[0], x[2], x[4], ...
                        //   fx_odd  = x[1], x[3], x[5], ...
                        simd_inlined::vdintlv_x2(fx_even, fx_odd, fx0, fx0);
                        // Step 2: negate the odd elements to -x[1], -x[3], ...
                        simd_inlined::vmul_f32(fx_odd, fx_odd, negOne, maskHalf);
                        // Step 3: re-interleave to form the rotated partner:
                        //   ft = [-x1, x0, -x3, x2, ...]   (low half)
                        //   ft2 = high half of interleaved result
                        simd_inlined::vintlv_x2(ft, ft2, fx_odd, fx_even);

                        // ---- y = x * cos + rotated(x) * sin ----
                        // ft2 = x * cos  (using the ORIGINAL widened `fx0`, not the
                        //   deinterleaved-even copy — both hold the same values but
                        //   `fx0` avoids aliasing concerns with the vintlv output).
                        // ft  = rotated(x) * sin
                        // then add to get y.
                        simd_inlined::vmul_f32(ft2, fx0, fc, mask32);   // ft2 = x * cos
                        simd_inlined::vmul_f32(ft,  ft,  fs, mask32);   // ft  = rotated(x) * sin
                        simd_inlined::vadd_f32(ft,  ft2, ft, mask32);   // y   = x*cos + rotated*sin

                        // ---- Narrow and store ----
                        simd_inlined::vcvt_f32_to_bf16_narrow(htmp0, ft, mask32);
                        simd_inlined::vsts_pk_b32_bf16(htmp0, yB + yOff + off, 0, mask32);
                    }
                }
            }
        }
    }
}

/*===========================================================================
 * ComputeF32 — fp32 RoPE computation (HALF and INTERLEAVE modes)
 *
 * INPUTS:  x, cos, sin in UB, all fp32 (`__ubuf__ float *`)
 * OUTPUTS: y in UB, fp32
 * PRECISION: native fp32 — bit-exact vs fp64 PyTorch reference.
 *
 * This variant exists primarily for validation (max_diff = 0.0 vs fp64
 * reference).  The cost is 2× higher memory traffic (4 B/element) and
 * therefore 2× lower bandwidth than fp16/bf16 for the same shape.
 *
 * Data flow per (s, n):
 *   Same as ComputeF16, but:
 *     - Loads via `vlds_norm_b32` (dense 64-fp32 per register).
 *     - All arithmetic in fp32 registers.
 *     - Stores via `vsts_norm_b32` (dense 64-fp32 back to UB).
 *
 * Vector register width: 256 B = 64 fp32 elements (VL_F32 = 64).
 * Mask granularity: b32 (one bit per fp32 element).
 *===========================================================================*/
ROPE_CCE_INTERNAL void ComputeF32(
    __ubuf__ uint16_t *x_ub16,
    __ubuf__ uint16_t *cos_ub16,
    __ubuf__ uint16_t *sin_ub16,
    __ubuf__ uint16_t *y_ub16,
    int32_t sCount, int32_t nCount,
    int32_t dLen, int32_t dAlign,
    int32_t xSStep, int32_t xNStep,
    int32_t csSStep,
    int32_t ySStep, int32_t yNStep,
    int32_t mode)
{
    __VEC_SCOPE__
    {
        // Expected UB effect of this vector scope (local tensor view):
        //   x_ub16   -> x:   fp32[sCount, nCount, dLen] with strides xSStep/xNStep
        //   cos_ub16 -> cos: fp32[sCount, dLen]          with stride csSStep
        //   sin_ub16 -> sin: fp32[sCount, dLen]          with stride csSStep
        //   y_ub16   -> y:   fp32[sCount, nCount, dLen] with strides ySStep/yNStep
        //   Typical values from vf_sim tests:
        //     correctness default: sCount=15, nCount=32, dLen=dAlign=64
        //     xNStep=yNStep=csSStep=64, xSStep=ySStep=nCount*64=2048
        //     wall-time configs: (s,n)=(1,2),(15,4),(15,8),(15,16),(15,32)
        //     so xSStep/ySStep range from 128 to 2048 elements; compile-time caps
        //     are sCount<=15 and nCount<=32 in the standalone VF harness.
        //
        //   Pseudocode after this scope completes:
        //     y[...] is overwritten; x/cos/sin are read-only.
        //
        //     if mode == 0:  # HALF / NeoX layout
        //       half = dLen // 2
        //       for s in range(sCount):
        //         for n in range(nCount):
        //           for d in range(half):
        //             y[s,n,d]      = x[s,n,d]      * cos[s,d]
        //                             - x[s,n,d+half] * sin[s,d]
        //             y[s,n,d+half] = x[s,n,d+half] * cos[s,d+half]
        //                             + x[s,n,d]      * sin[s,d+half]
        //
        //     else:  # INTERLEAVE / GPT-J layout
        //       xe, xo = x[s,n,0::2], x[s,n,1::2]
        //       ce, co = cos[s,0::2], cos[s,1::2]
        //       se, so = sin[s,0::2], sin[s,1::2]
        //       y[s,n,0::2] = xe * ce - xo * se
        //       y[s,n,1::2] = xo * co + xe * so
        __ubuf__ float *xF   = (__ubuf__ float *)x_ub16;
        __ubuf__ float *cosF = (__ubuf__ float *)cos_ub16;
        __ubuf__ float *sinF = (__ubuf__ float *)sin_ub16;
        __ubuf__ float *yF   = (__ubuf__ float *)y_ub16;

        // halfDAl aligned to (BLOCK_BYTE_32 / 4) = 8 fp32 elements = 32 bytes.
        int32_t halfD = dLen / 2;
        int32_t halfDAl = calign(halfD, (int32_t)BLOCK_BYTE_32 / 4);
        int32_t repeatTimes = cdiv(halfD, (int32_t)VL_F32);

        // Register roles (each 256 B = 64 fp32):
        //   fc0/fc1, fs0/fs1: cos/sin halves from both halves of D.
        //   fx0/fx1: x halves per head.
        //   ft0/ft1: arithmetic intermediates.
        //   fy0/fy1: y output halves.
        vector_f32 fx0, fx1, fc0, fc1, fs0, fs1, ft0, ft1, fy0, fy1;

        // ---- HALF mode (mode == 0) ----
        // Identical dataflow to ComputeF16 HALF modulo element precision.
        if (mode == 0) {
            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t rep = 0; rep < (uint16_t)repeatTimes; rep++) {
                    uint32_t elemOff = (uint32_t)rep * (uint32_t)VL_F32;
                    uint32_t cnt = (uint32_t)(halfD - (int32_t)elemOff);
                    if (cnt > (uint32_t)VL_F32) cnt = (uint32_t)VL_F32;
                    MaskReg mask32 = simd_inlined::make_mask(cnt);

                    // ---- Hoisted cos/sin loads (shared across heads) ----
                    // fc0 = cos_low,  fc1 = cos_high
                    // fs0 = sin_low,  fs1 = sin_high
                    simd_inlined::vlds_norm_b32(fc0, cosF + csOff + elemOff, 0);
                    simd_inlined::vlds_norm_b32(fc1, cosF + csOff + elemOff + halfDAl, 0);
                    simd_inlined::vlds_norm_b32(fs0, sinF + csOff + elemOff, 0);
                    simd_inlined::vlds_norm_b32(fs1, sinF + csOff + elemOff + halfDAl, 0);

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x loads ----
                        simd_inlined::vlds_norm_b32(fx0, xF + xOff + elemOff, 0);           // x_low
                        simd_inlined::vlds_norm_b32(fx1, xF + xOff + elemOff + halfDAl, 0); // x_high

                        // y0 = cos_low * x_low - sin_low * x_high
                        simd_inlined::vmul_f32(ft0, fc0, fx0, mask32);
                        simd_inlined::vmul_f32(ft1, fs0, fx1, mask32);
                        simd_inlined::vsub_f32(fy0, ft0, ft1, mask32);
                        // y1 = cos_high * x_high + sin_high * x_low
                        simd_inlined::vmul_f32(ft0, fc1, fx1, mask32);
                        simd_inlined::vmul_f32(ft1, fs1, fx0, mask32);
                        simd_inlined::vadd_f32(fy1, ft0, ft1, mask32);

                        // y → UB
                        simd_inlined::vsts_norm_b32(fy0, yF + yOff + elemOff, 0, mask32);
                        simd_inlined::vsts_norm_b32(fy1, yF + yOff + elemOff + halfDAl, 0, mask32);
                    }
                }
            }
        } else {
            // ---- INTERLEAVE mode (mode == 1) ----
            // Same pattern as ComputeF16/ComputeBf16 INTERLEAVE (vdintlv/vintlv
            // to form the rotated partner), in fp32 register space.
            // Note: since fp32 registers have 64 lanes, blockSize = 64, so each
            // block corresponds to 64 fp32 elements, i.e. 32 complex pairs.
            vector_f32 heven, hodd, hnegOdd, hxnew, hxnew_hi, fta, ftb;
            vector_f32 negOne;
            simd_inlined::vbr_f32(negOne, -1.0f);

            int32_t blockSize = (int32_t)VL_F32;      // 64 fp32 per block
            int32_t dBlocks = cdiv(dLen, blockSize);

            for (uint16_t s = 0; s < (uint16_t)sCount; s++) {
                int32_t csOff = s * csSStep;
                for (uint16_t blk = 0; blk < (uint16_t)dBlocks; blk++) {
                    int32_t off = (int32_t)blk * blockSize;
                    int32_t remaining = dLen - off;
                    uint32_t cnt = (remaining > blockSize) ? (uint32_t)blockSize : (uint32_t)remaining;

                    MaskReg mask = simd_inlined::make_mask(cnt);
                    MaskReg maskHalf = simd_inlined::make_mask((cnt + 1U) / 2U);

                    // ---- Hoisted cos/sin loads ----
                    simd_inlined::vlds_norm_b32(fc0, cosF + csOff + off, 0);
                    simd_inlined::vlds_norm_b32(fs0, sinF + csOff + off, 0);

                    for (uint16_t n = 0; n < (uint16_t)nCount; n++) {
                        int32_t xOff = s * xSStep + n * xNStep;
                        int32_t yOff = s * ySStep + n * yNStep;

                        // ---- Per-head x load ----
                        simd_inlined::vlds_norm_b32(fx0, xF + xOff + off, 0);

                        // ---- Form the rotated partner: [-x1, x0, -x3, x2, ...] ----
                        simd_inlined::vdintlv_x2(heven, hodd, fx0, fx0);      // split even/odd
                        simd_inlined::vmul_f32(hnegOdd, hodd, negOne, maskHalf); // negate odd
                        simd_inlined::vintlv_x2(hxnew, hxnew_hi, hnegOdd, heven); // rebuild partner

                        // y = x * cos + rotated(x) * sin
                        simd_inlined::vmul_f32(fta, fx0, fc0, mask);          // x * cos
                        simd_inlined::vmul_f32(ftb, hxnew, fs0, mask);        // rotated(x) * sin
                        simd_inlined::vadd_f32(fy0, fta, ftb, mask);          // y
                        simd_inlined::vsts_norm_b32(fy0, yF + yOff + off, 0, mask);
                    }
                }
            }
        }
    }
}

}

#endif

#endif
