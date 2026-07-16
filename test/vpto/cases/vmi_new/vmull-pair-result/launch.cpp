// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void vmi_vmull_pair_result_kernel(
    __gm__ int32_t *lhs32, __gm__ int32_t *rhs32,
    __gm__ int16_t *lhs16, __gm__ int16_t *rhs16,
    __gm__ uint8_t *lhs8, __gm__ uint8_t *rhs8,
    __gm__ int32_t *low, __gm__ int32_t *high);

void LaunchVmiVmullPairResult(
    int32_t *lhs32, int32_t *rhs32, int16_t *lhs16, int16_t *rhs16,
    uint8_t *lhs8, uint8_t *rhs8, uint32_t *low, uint32_t *high,
    void *stream) {
  vmi_vmull_pair_result_kernel<<<1, nullptr, stream>>>(
      (__gm__ int32_t *)lhs32, (__gm__ int32_t *)rhs32,
      (__gm__ int16_t *)lhs16, (__gm__ int16_t *)rhs16,
      (__gm__ uint8_t *)lhs8, (__gm__ uint8_t *)rhs8,
      (__gm__ int32_t *)low, (__gm__ int32_t *)high);
}
