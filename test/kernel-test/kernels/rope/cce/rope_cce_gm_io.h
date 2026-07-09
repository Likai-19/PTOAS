// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef ROPE_CCE_GM_IO_H
#define ROPE_CCE_GM_IO_H

#include "rope_cce_shim.h"

namespace rope_cce {

// UB caps sized for the production AB tile (ubFactorBS=15, ubFactorN=32, D=64).
// Runtime sCount/nCount passed to the kernel must stay within these limits.
constexpr int32_t kMaxS = 15;
constexpr int32_t kMaxN = 32;
constexpr int32_t kMaxD = 64;
constexpr int32_t kMaxDAlign = 64;

constexpr int32_t kEsF16 = 2;
constexpr int32_t kEsF32 = 4;

ROPE_CCE_INTERNAL void GmLoadContig(
    __ubuf__ uint16_t *ub,
    __gm__ uint16_t *gm,
    uint32_t elemCount,
    int32_t es)
{
    uint32_t bytes = elemCount * (uint32_t)es;
    copy_gm_to_ubuf_align_v2(
        ub, gm,
        0, 1, bytes,
        0, 0, false, 0,
        bytes, bytes);
}

ROPE_CCE_INTERNAL void GmStoreContig(
    __gm__ uint16_t *gm,
    __ubuf__ uint16_t *ub,
    uint32_t elemCount,
    int32_t es)
{
    uint32_t bytes = elemCount * (uint32_t)es;
    copy_ubuf_to_gm_align_v2(
        gm, ub,
        0, 1, bytes,
        0, bytes, bytes);
}

ROPE_CCE_INTERNAL void GmLoadF16Contig(
    __ubuf__ uint16_t *ub,
    __gm__ uint16_t *gm,
    uint32_t elemCount)
{
    GmLoadContig(ub, gm, elemCount, kEsF16);
}

ROPE_CCE_INTERNAL void GmStoreF16Contig(
    __gm__ uint16_t *gm,
    __ubuf__ uint16_t *ub,
    uint32_t elemCount)
{
    GmStoreContig(gm, ub, elemCount, kEsF16);
}

} // namespace rope_cce

#endif
