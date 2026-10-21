// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir::pto {

enum class BridgeFamily : uint8_t { Pipe, Cube, Print };
enum class BridgeCoreKind : uint32_t;
enum class BridgeValueKind : uint8_t {
  Pointer,
  F32,
  I32,
  I64,
  U64,
  ObjectHandle,
};
// Address space of a Pointer value. Private is the per-function object storage
// produced by bridge.object.create; Global is a GM pointer such as a format
// string. Non-pointer values use None.
enum class BridgePointerSpace : uint8_t { None, Private, Global };
enum class BridgeRendererKind : uint8_t { Pipe, CubeDirect, Print };

struct BridgeValueType {
  BridgeValueKind kind;
  BridgePointerSpace space = BridgePointerSpace::None;
};
enum class BridgeEntry : uint32_t;
using BridgeEntryId = BridgeEntry;

struct BridgeFunctionDesc {
  BridgeEntryId id;
  BridgeFamily family;
  BridgeCoreKind core;
  BridgeRendererKind renderer;
  llvm::StringLiteral opName;
  llvm::StringLiteral implementation;
  llvm::StringLiteral symbolBase;
  llvm::ArrayRef<BridgeValueType> arguments;
  llvm::ArrayRef<BridgeValueType> results;
  bool createsObject;
  // Object storage contract: an upper bound on sizeof/alignof the rendered
  // C++ object. Storage is a fixed entry-block alloca; the wrapper
  // static_asserts the real type fits. Zero for entries that create nothing.
  uint64_t storageSize;
  uint64_t storageAlign;
};

llvm::ArrayRef<BridgeFunctionDesc> getBridgeFunctionRegistry();
const BridgeFunctionDesc *findBridgeFunction(BridgeEntryId id);
const BridgeFunctionDesc *findBridgeFunctionByOp(llvm::StringRef opName);
llvm::StringRef stringifyBridgeEntryId(BridgeEntryId id);
llvm::StringRef stringifyBridgeFamily(BridgeFamily family);

} // namespace mlir::pto

#endif
