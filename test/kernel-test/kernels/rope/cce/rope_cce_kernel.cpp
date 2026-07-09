// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Standalone single-core rope CCE kernel.
 * Wraps ComputeF16/Bf16/F32 (__VEC_SCOPE__) with trivial GM<->UB I/O.
 */
#include "rope_cce_compute.h"
#include "rope_cce_gm_io.h"

using namespace rope_cce;

namespace {

ROPE_CCE_INTERNAL void ReadTileParams(
    __gm__ int32_t *params_g, int32_t &sCount, int32_t &nCount)
{
    sCount = params_g[0];
    nCount = params_g[1];
}

template <int32_t kMode, int32_t kDtypeMode>
ROPE_CCE_INTERNAL void RunRopeTile(
    __gm__ uint16_t *x_g,
    __gm__ uint16_t *cos_g,
    __gm__ uint16_t *sin_g,
    __gm__ uint16_t *y_g,
    int32_t sCount,
    int32_t nCount)
{
    int32_t es = (kDtypeMode == 2) ? kEsF32 : kEsF16;
    int32_t cosSinElems = sCount * kMaxDAlign;
    int32_t xyElems = sCount * nCount * kMaxDAlign;
    int32_t cosSinBytes = cosSinElems * es;
    int32_t xyBytes = xyElems * es;

    __ubuf__ uint16_t *ub_base = (__ubuf__ uint16_t *)0x00000;
    __ubuf__ uint16_t *cos_ub = ub_base;
    __ubuf__ uint16_t *sin_ub = (__ubuf__ uint16_t *)((uintptr_t)cos_ub + cosSinBytes);
    __ubuf__ uint16_t *x_ub = (__ubuf__ uint16_t *)((uintptr_t)sin_ub + cosSinBytes);
    __ubuf__ uint16_t *y_ub = (__ubuf__ uint16_t *)((uintptr_t)x_ub + xyBytes);

    GmLoadContig(cos_ub, cos_g, (uint32_t)cosSinElems, es);
    GmLoadContig(sin_ub, sin_g, (uint32_t)cosSinElems, es);
    GmLoadContig(x_ub, x_g, (uint32_t)xyElems, es);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (kDtypeMode == 0) {
        ComputeF16(
            x_ub, cos_ub, sin_ub, y_ub,
            sCount, nCount,
            kMaxD, kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMode);
    } else if constexpr (kDtypeMode == 1) {
        ComputeBf16(
            x_ub, cos_ub, sin_ub, y_ub,
            sCount, nCount,
            kMaxD, kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMode);
    } else {
        ComputeF32(
            x_ub, cos_ub, sin_ub, y_ub,
            sCount, nCount,
            kMaxD, kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMaxDAlign,
            nCount * kMaxDAlign, kMaxDAlign,
            kMode);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    GmStoreContig(y_g, y_ub, (uint32_t)xyElems, es);
}

template <int32_t kMode, int32_t kDtypeMode>
__global__ AICORE void rope_cce_sim_kernel(
    __gm__ uint16_t *x_g,
    __gm__ uint16_t *cos_g,
    __gm__ uint16_t *sin_g,
    __gm__ uint16_t *y_g,
    __gm__ int32_t *params_g)
{
#if defined(__DAV_VEC__)
    int32_t sCount = 0;
    int32_t nCount = 0;
    ReadTileParams(params_g, sCount, nCount);
    RunRopeTile<kMode, kDtypeMode>(x_g, cos_g, sin_g, y_g, sCount, nCount);
#endif
}

template <int32_t kMode, int32_t kDtypeMode>
__global__ AICORE void rope_cce_cycle_kernel(
    __gm__ uint16_t *x_g,
    __gm__ uint16_t *cos_g,
    __gm__ uint16_t *sin_g,
    __gm__ uint16_t *y_g,
    __gm__ int32_t *params_g)
{
#if defined(__DAV_VEC__)
    int32_t sCount = 0;
    int32_t nCount = 0;
    ReadTileParams(params_g, sCount, nCount);
    RunRopeTile<kMode, kDtypeMode>(x_g, cos_g, sin_g, y_g, sCount, nCount);
#endif
}

} // namespace

#define ROPE_CCE_LAUNCH_SIM(mode, dtype_mode, suffix) \
    void call_rope_cce_sim_##suffix( \
        void *stream, void *x, void *cos, void *sin, void *y, void *params) \
    { \
        rope_cce_sim_kernel<mode, dtype_mode><<<1, nullptr, stream>>>( \
            (__gm__ uint16_t *)x, \
            (__gm__ uint16_t *)cos, \
            (__gm__ uint16_t *)sin, \
            (__gm__ uint16_t *)y, \
            (__gm__ int32_t *)params); \
    }

#define ROPE_CCE_LAUNCH_CYCLE(mode, dtype_mode, suffix) \
    void call_rope_cce_cycle_##suffix( \
        void *stream, void *x, void *cos, void *sin, void *y, void *params) \
    { \
        rope_cce_cycle_kernel<mode, dtype_mode><<<1, nullptr, stream>>>( \
            (__gm__ uint16_t *)x, \
            (__gm__ uint16_t *)cos, \
            (__gm__ uint16_t *)sin, \
            (__gm__ uint16_t *)y, \
            (__gm__ int32_t *)params); \
    }

extern "C" {

ROPE_CCE_LAUNCH_SIM(0, 0, half_f16)
ROPE_CCE_LAUNCH_SIM(1, 0, interleave_f16)
ROPE_CCE_LAUNCH_SIM(0, 1, half_bf16)
ROPE_CCE_LAUNCH_SIM(1, 1, interleave_bf16)
ROPE_CCE_LAUNCH_SIM(0, 2, half_f32)
ROPE_CCE_LAUNCH_SIM(1, 2, interleave_f32)

ROPE_CCE_LAUNCH_CYCLE(0, 0, half_f16)
ROPE_CCE_LAUNCH_CYCLE(1, 0, interleave_f16)
ROPE_CCE_LAUNCH_CYCLE(0, 1, half_bf16)
ROPE_CCE_LAUNCH_CYCLE(1, 1, interleave_bf16)
ROPE_CCE_LAUNCH_CYCLE(0, 2, half_f32)
ROPE_CCE_LAUNCH_CYCLE(1, 2, interleave_f32)

}
