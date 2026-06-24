// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchVmi_simdvf_per_token_cast_to_fp8_kernel(float *src, float *scale,
                                                   uint8_t *out8,
                                                   void *stream);

int main() {
  constexpr size_t kElems = 256;
  size_t srcBytes = kElems * sizeof(float);
  size_t scaleBytes = kElems * sizeof(float);
  size_t out8Bytes = kElems * sizeof(uint8_t);
  float *srcHost = nullptr;
  float *scaleHost = nullptr;
  uint8_t *out8Host = nullptr;
  float *srcDevice = nullptr;
  float *scaleDevice = nullptr;
  uint8_t *out8Device = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), srcBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&out8Host), out8Bytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&scaleDevice, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&out8Device, out8Bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", scaleBytes, scaleHost, scaleBytes);
  ReadFile("./v3.bin", out8Bytes, out8Host, out8Bytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(scaleDevice, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(out8Device, out8Bytes, out8Host, out8Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_simdvf_per_token_cast_to_fp8_kernel(srcDevice, scaleDevice,
                                                out8Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(scaleHost, scaleBytes, scaleDevice, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(out8Host, out8Bytes, out8Device, out8Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", scaleHost, scaleBytes);
  WriteFile("./v3.bin", out8Host, out8Bytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(scaleDevice);
  aclrtFree(out8Device);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(scaleHost);
  aclrtFreeHost(out8Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
