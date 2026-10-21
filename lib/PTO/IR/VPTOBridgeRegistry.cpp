// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/VPTOBridgeRegistry.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir::pto;
namespace {
constexpr BridgeValueType kStorage{BridgeValueKind::Pointer, BridgePointerSpace::Private};
constexpr BridgeValueType kGlobalPtr{BridgeValueKind::Pointer, BridgePointerSpace::Global};
constexpr BridgeValueType kHandle{BridgeValueKind::ObjectHandle};
constexpr BridgeValueType kF32{BridgeValueKind::F32};
constexpr BridgeValueType kI32{BridgeValueKind::I32};
constexpr BridgeValueType kI64{BridgeValueKind::I64};
constexpr BridgeValueType kU64{BridgeValueKind::U64};
constexpr BridgeValueType kPipeInitArgs[] = {kStorage, kI32};
constexpr BridgeValueType kPipePushArgs[] = {kHandle, kU64};
constexpr BridgeValueType kPipeObjectArgs[] = {kHandle};
constexpr BridgeValueType kPrintPointerArgs[] = {kGlobalPtr};
constexpr BridgeValueType kU64Results[] = {kU64};
constexpr BridgeValueType kPrintF32Args[] = {kGlobalPtr, kF32};
constexpr BridgeValueType kPrintI64Args[] = {kGlobalPtr, kI64};
constexpr BridgeValueType kPrintU64Args[] = {kGlobalPtr, kU64};
constexpr BridgeValueType kCubeTMatmulArgs[] = {kU64, kU64, kU64};
constexpr BridgeValueType kCubeTMatmulAccArgs[] = {kU64, kU64, kU64, kU64};
constexpr llvm::ArrayRef<BridgeValueType> kNoValues;
// Conservative bound for pto::TPipe; WrapperGen static_asserts it per instance.
constexpr uint64_t kPipeStorageSize = 256;
constexpr uint64_t kPipeStorageAlign = 32;
constexpr BridgeFunctionDesc kRegistry[] = {
    {BridgeEntryId::PipeInit, BridgeFamily::Pipe, BridgeCoreKind::Any, BridgeRendererKind::Pipe,
     "pto.initialize_l2l_pipe", "pipe.v1", "pto_vpto_pipe_init", kPipeInitArgs, kNoValues, true,
     kPipeStorageSize, kPipeStorageAlign},
    {BridgeEntryId::PipePush, BridgeFamily::Pipe, BridgeCoreKind::Any, BridgeRendererKind::Pipe,
     "pto.tpush", "pipe.v1", "pto_vpto_pipe_push", kPipePushArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::PipePop, BridgeFamily::Pipe, BridgeCoreKind::Any, BridgeRendererKind::Pipe,
     "pto.tpop", "pipe.v1", "pto_vpto_pipe_pop", kPipeObjectArgs, kU64Results, false, 0, 0},
    {BridgeEntryId::PipeFree, BridgeFamily::Pipe, BridgeCoreKind::Any, BridgeRendererKind::Pipe,
     "pto.tfree", "pipe.v1", "pto_vpto_pipe_free", kPipeObjectArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::CubeTMatmul, BridgeFamily::Cube, BridgeCoreKind::Cube, BridgeRendererKind::CubeDirect,
     "pto.tmatmul", "cube.pto_isa_v1", "pto_vpto_tmatmul", kCubeTMatmulArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::CubeTMatmulAcc, BridgeFamily::Cube, BridgeCoreKind::Cube, BridgeRendererKind::CubeDirect,
     "pto.tmatmul.acc", "cube.pto_isa_v1", "pto_vpto_tmatmul_acc", kCubeTMatmulAccArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintF32, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "pto.print", "print.cce_v1", "pto_vpto_print_f32", kPrintF32Args, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintI64, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "", "print.cce_v1", "pto_vpto_print_i64", kPrintI64Args, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintU64, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "", "print.cce_v1", "pto_vpto_print_u64", kPrintU64Args, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintString, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "", "print.cce_v1", "pto_vpto_print_str", kPrintPointerArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintInit, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "", "print.cce_v1", "pto_vpto_print_init", kPrintPointerArgs, kNoValues, false, 0, 0},
    {BridgeEntryId::PrintFinish, BridgeFamily::Print, BridgeCoreKind::Any, BridgeRendererKind::Print,
     "", "print.cce_v1", "pto_vpto_print_finish", kPrintPointerArgs, kNoValues, false, 0, 0},
};
} // namespace

// The registry is the single owner of entry identity: a duplicate id, symbol
// base or operation name would make routing and instance symbols ambiguous.
// Checked here rather than at every use.
namespace {
// llvm::StringRef comparison is not constexpr here, so compare the literals
// directly to keep the uniqueness check at compile time.
constexpr bool sameText(llvm::StringLiteral lhs, llvm::StringLiteral rhs) {
  const bool differentLength = lhs.size() != rhs.size();
  if (differentLength) {
    return false;
  }
  for (size_t index = 0; index < lhs.size(); ++index) {
    if (lhs.data()[index] != rhs.data()[index]) {
      return false;
    }
  }
  return true;
}

constexpr bool hasUniqueEntries() {
  for (size_t i = 0; i < std::size(kRegistry); ++i) {
    for (size_t j = i + 1; j < std::size(kRegistry); ++j) {
      const bool sameId = kRegistry[i].id == kRegistry[j].id;
      const bool sameSymbol =
          sameText(kRegistry[i].symbolBase, kRegistry[j].symbolBase);
      const bool sameOp = kRegistry[i].opName.size() != 0 &&
                          sameText(kRegistry[i].opName, kRegistry[j].opName);
      if (sameId || sameSymbol || sameOp) {
        return false;
      }
    }
  }
  return true;
}
} // namespace

static_assert(hasUniqueEntries(),
              "bridge registry entries must have unique ids, symbols and op "
              "names: routing and instance symbols derive from them");

llvm::ArrayRef<BridgeFunctionDesc> mlir::pto::getBridgeFunctionRegistry() { return kRegistry; }
const BridgeFunctionDesc *mlir::pto::findBridgeFunction(BridgeEntryId id) {
  auto it = llvm::find_if(kRegistry, [id](const auto &entry) { return entry.id == id; });
  return it == std::end(kRegistry) ? nullptr : &*it;
}
const BridgeFunctionDesc *mlir::pto::findBridgeFunctionByOp(llvm::StringRef opName) {
  if (opName.empty()) {
    return nullptr;
  }
  auto it = llvm::find_if(kRegistry, [opName](const auto &entry) { return entry.opName == opName; });
  return it == std::end(kRegistry) ? nullptr : &*it;
}
llvm::StringRef mlir::pto::stringifyBridgeEntryId(BridgeEntryId id) {
  switch (id) {
  case BridgeEntryId::PipeInit: return "pipe.init";
  case BridgeEntryId::PipePush: return "pipe.push";
  case BridgeEntryId::PipePop: return "pipe.pop";
  case BridgeEntryId::PipeFree: return "pipe.free";
  case BridgeEntryId::CubeTMatmul: return "cube.tmatmul";
  case BridgeEntryId::CubeTMatmulAcc: return "cube.tmatmul.acc";
  case BridgeEntryId::PrintF32: return "print.f32";
  case BridgeEntryId::PrintI64: return "print.i64";
  case BridgeEntryId::PrintU64: return "print.u64";
  case BridgeEntryId::PrintString: return "print.string";
  case BridgeEntryId::PrintInit: return "print.init";
  case BridgeEntryId::PrintFinish: return "print.finish";
  }
  return "<unknown>";
}
llvm::StringRef mlir::pto::stringifyBridgeFamily(BridgeFamily family) {
  switch (family) {
  case BridgeFamily::Pipe: return "pipe";
  case BridgeFamily::Cube: return "cube";
  case BridgeFamily::Print: return "print";
  }
  return "<unknown>";
}
