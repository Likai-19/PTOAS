// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/tdump-vec
// family: kernels
// target_ops: pto.mte_gm_ub, pto.tdump
// scenarios: tdump-vec-f32-4x16, header-plus-row-major-payload
// -----------------------------------------------------------------------------
#include "test_common.h"
#include "acl/acl.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace PtoTestCommon;

void LaunchTdump_vec_kernel(float *v1, float *v2, int32_t assertCondition, void *stream);

int main() {
    size_t elemCountIn = 8 * 32;
    size_t fileSizeIn = elemCountIn * sizeof(float);
    // 64-byte header + 4x16 f32 payload.
    size_t fileSizeOut = 64 + 4 * 16 * sizeof(float);
    const int32_t assertCondition = std::getenv("PTOAS_TEST_ASSERT_FAIL") == nullptr ? 1 : 0;
    float *v1Host = nullptr;
    uint8_t *v2Host = nullptr;
    float *v1Device = nullptr;
    float *v2Device = nullptr;
    aclrtStream stream = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;

    const auto cleanup = [&]() {
        if (v1Device != nullptr) {
            aclrtFree(v1Device);
        }
        if (v2Device != nullptr) {
            aclrtFree(v2Device);
        }
        if (v1Host != nullptr) {
            aclrtFreeHost(v1Host);
        }
        if (v2Host != nullptr) {
            aclrtFreeHost(v2Host);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (aclInited) {
            aclFinalize();
        }
    };

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0') {                                      \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                   \
            }                                                                                   \
            cleanup();                                                                           \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        char *end = nullptr;
        const long parsed = std::strtol(envDevice, &end, 10);
        if (end != envDevice && *end == '\0' && parsed >= 0 &&
            parsed <= std::numeric_limits<int>::max()) {
            deviceId = static_cast<int>(parsed);
        }
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSizeIn));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSizeOut));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSizeIn, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(v2Host, fileSizeOut, uint8_t{0xA5});
    ACL_CHECK(aclrtMemcpy(v2Device, fileSizeOut, v2Host, fileSizeOut, ACL_MEMCPY_HOST_TO_DEVICE));
    ReadFile("./v1.bin", fileSizeIn, v1Host, fileSizeIn);
    ACL_CHECK(aclrtMemcpy(v1Device, fileSizeIn, v1Host, fileSizeIn, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchTdump_vec_kernel(v1Device, v2Device, assertCondition, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v2Host, fileSizeOut, v2Device, fileSizeOut, ACL_MEMCPY_DEVICE_TO_HOST));
    if (assertCondition == 0) {
        const bool outputUnchanged = std::all_of(v2Host, v2Host + fileSizeOut,
            [](uint8_t value) { return value == uint8_t{0xA5}; });
        if (!outputUnchanged) {
            std::fprintf(stderr, "[ERROR] false assertion did not stop post-assert dump writes\n");
            rc = 1;
        } else {
            std::printf("[INFO] false assertion left the entire dump buffer unchanged\n");
        }
    } else {
        WriteFile("./v2.bin", v2Host, fileSizeOut);
    }

    cleanup();
    return rc;
}
