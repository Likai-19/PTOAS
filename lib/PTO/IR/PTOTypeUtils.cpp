// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTOTypeUtils.h"

#include "PTO/IR/PTO.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kBitsPerByte = 8;
} // namespace

bool mlir::pto::isPTOFloat8Type(Type t) {
  return t.isFloat8E4M3() || t.isFloat8E4M3FN() || t.isFloat8E4M3FNUZ() ||
         t.isFloat8E4M3B11FNUZ() || t.isFloat8E5M2() || t.isFloat8E5M2FNUZ();
}

bool mlir::pto::isPTOHiFloat8Type(Type t) { return isa<HiF8Type>(t); }

bool mlir::pto::isPTOF8E8M0Type(Type t) { return isa<F8E8M0Type>(t); }

bool mlir::pto::isPTOHiFloat8x2Type(Type t) { return isa<HiF8x2Type>(t); }

bool mlir::pto::isPTOFloat4PackedType(Type t) {
  return isa<F4E1M2x2Type, F4E2M1x2Type>(t);
}

bool mlir::pto::isPTOLowPrecisionType(Type t) {
  return isPTOFloat8Type(t) || isPTOHiFloat8Type(t) || isPTOF8E8M0Type(t) ||
         isPTOHiFloat8x2Type(t) || isPTOFloat4PackedType(t);
}

unsigned mlir::pto::getPTOStorageElemBitWidth(Type t) {
  if (isPTOHiFloat8x2Type(t))
    return 16;
  if (isPTOLowPrecisionType(t))
    return kBitsPerByte;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth();
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  return 0;
}

unsigned mlir::pto::getPTOStorageElemByteSize(Type t) {
  unsigned bitWidth = getPTOStorageElemBitWidth(t);
  return bitWidth == 0 ? 0 : bitWidth / kBitsPerByte;
}
