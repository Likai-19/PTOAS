// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIPreAssignmentCombine.cpp - Pre-assignment VMI combines ---------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "VMIInternal.h"
#include "VMIIndexUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREASSIGNMENTCOMBINE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static LogicalResult canonicalizeContiguousGroupLoads(ModuleOp module) {
  SmallVector<VMIGroupLoadOp> loads;
  module.walk([&](VMIGroupLoadOp load) {
    auto resultType = dyn_cast<VMIVRegType>(load.getResult().getType());
    if (!resultType) {
      return;
    }

    int64_t numGroups = load.getNumGroupsAttr().getInt();
    int64_t laneCount = resultType.getElementCount();
    if (numGroups <= 0 || laneCount % numGroups != 0) {
      return;
    }

    std::optional<int64_t> rowStride =
        getConstantIndexValue(load.getRowStride());
    if (rowStride && *rowStride == laneCount / numGroups) {
      loads.push_back(load);
    }
  });

  OpBuilder builder(module.getContext());
  for (VMIGroupLoadOp load : loads) {
    builder.setInsertionPoint(load);
    auto replacement = builder.create<VMILoadOp>(
        load.getLoc(), load.getResult().getType(), load.getSource(),
        load.getOffset());
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }
  return success();
}

static LogicalResult fuseGroupSlotBroadcastLoads(ModuleOp module) {
  SmallVector<VMIGroupBroadcastOp> broadcasts;
  module.walk([&](VMIGroupBroadcastOp broadcast) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load || !load.getResult().hasOneUse()) {
      return;
    }
    if (load.getNumGroupsAttr().getInt() !=
        broadcast.getNumGroupsAttr().getInt()) {
      return;
    }

    if (!isa<VMIVRegType>(broadcast.getResult().getType())) {
      return;
    }
    broadcasts.push_back(broadcast);
  });

  OpBuilder builder(module.getContext());
  for (VMIGroupBroadcastOp broadcast : broadcasts) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load) {
      continue;
    }

    builder.setInsertionPoint(broadcast);
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast.getLoc(), broadcast.getResult().getType(), load.getSource(),
        load.getOffset(), load.getSourceGroupStride(),
        broadcast.getNumGroupsAttr());
    broadcast.getResult().replaceAllUsesWith(fused.getResult());
    broadcast.erase();
    if (load->use_empty()) {
      load.erase();
    }
  }
  return success();
}

static bool canMoveLoadPastInterveningOps(Operation *load,
                                          Operation *broadcast,
                                          Value loadSource,
                                          AliasAnalysis &aliasAnalysis) {
  bool isInSameBlock = load->getBlock() == broadcast->getBlock();
  if (!isInSameBlock) {
    return false;
  }

  for (Operation *op = load->getNextNode(); op && op != broadcast;
       op = op->getNextNode()) {
    if (isPure(op)) {
      continue;
    }

    auto effects = getEffectsRecursively(op);
    if (!effects) {
      return false;
    }
    for (const auto &effect : *effects) {
      bool usesUnsupportedResource =
          effect.getResource() != SideEffects::DefaultResource::get();
      bool affectsFullRegion = effect.getEffectOnFullRegion();
      if (usesUnsupportedResource || affectsFullRegion) {
        return false;
      }

      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        continue;
      }
      if (!isa<MemoryEffects::Write>(effect.getEffect())) {
        return false;
      }

      Value writtenSource = effect.getValue();
      if (!writtenSource ||
          !aliasAnalysis.alias(loadSource, writtenSource).isNo()) {
        return false;
      }
    }
  }
  return true;
}

static LogicalResult fuseVMISingleLaneLoadBroadcasts(
    ModuleOp module, AliasAnalysis &aliasAnalysis) {
  SmallVector<VMIBroadcastOp> broadcasts;
  module.walk([&](VMIBroadcastOp broadcast) {
    auto load = broadcast.getValue().getDefiningOp<VMILoadOp>();
    if (!load) {
      return;
    }

    bool loadHasSingleUse = load.getResult().hasOneUse();
    bool canReorderLoad = canMoveLoadPastInterveningOps(
        load, broadcast, load.getSource(), aliasAnalysis);
    if (!loadHasSingleUse || !canReorderLoad) {
      return;
    }

    auto loadType = dyn_cast<VMIVRegType>(load.getResult().getType());
    bool isSingleLaneLoad = loadType && loadType.getElementCount() == 1;
    bool hasVectorResult =
        isa<VMIVRegType>(broadcast.getResult().getType());
    if (!isSingleLaneLoad || !hasVectorResult) {
      return;
    }
    broadcasts.push_back(broadcast);
  });

  OpBuilder builder(module.getContext());
  for (VMIBroadcastOp broadcast : broadcasts) {
    auto load = broadcast.getValue().getDefiningOp<VMILoadOp>();
    if (!load) {
      continue;
    }

    builder.setInsertionPoint(broadcast);
    Value zeroStride = builder.create<arith::ConstantOp>(
        broadcast.getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast.getLoc(), broadcast.getResult().getType(), load.getSource(),
        load.getOffset(), zeroStride, builder.getI64IntegerAttr(1));
    broadcast.getResult().replaceAllUsesWith(fused.getResult());
    broadcast.erase();
    if (load->use_empty()) {
      load.erase();
    }
  }
  return success();
}

static PTOLoadOp getFusablePTOScalarUBLoad(VMIBroadcastOp broadcast,
                                           AliasAnalysis &aliasAnalysis) {
  auto load = broadcast.getValue().getDefiningOp<PTOLoadOp>();
  if (!load) {
    return {};
  }

  bool hasSingleUse = load.getValue().hasOneUse();
  bool canReorderLoad = canMoveLoadPastInterveningOps(
      load, broadcast, load.getPtr(), aliasAnalysis);
  if (!hasSingleUse || !canReorderLoad) {
    return {};
  }

  auto resultType = dyn_cast<VMIVRegType>(broadcast.getResult().getType());
  if (!resultType) {
    return {};
  }

  Type sourceType = load.getPtr().getType();
  Type elementType = resultType.getElementType();
  bool sourceIsUB = isUBBackedMemoryType(sourceType);
  bool sourceElementMatches =
      getMemoryElementType(sourceType) == load.getValue().getType();
  bool resultElementMatches = load.getValue().getType() == elementType;
  if (!sourceIsUB || !sourceElementMatches || !resultElementMatches) {
    return {};
  }

  unsigned bitWidth = getVMIElementBitWidth(elementType);
  if (bitWidth != kValue8 && bitWidth != kValue16 &&
      bitWidth != kValue32) {
    return {};
  }
  return load;
}

static LogicalResult fusePTOScalarUBLoadBroadcasts(
    ModuleOp module, AliasAnalysis &aliasAnalysis) {
  SmallVector<VMIBroadcastOp> broadcasts;
  module.walk([&](VMIBroadcastOp broadcast) {
    if (getFusablePTOScalarUBLoad(broadcast, aliasAnalysis)) {
      broadcasts.push_back(broadcast);
    }
  });

  OpBuilder builder(module.getContext());
  for (VMIBroadcastOp broadcast : broadcasts) {
    auto load = getFusablePTOScalarUBLoad(broadcast, aliasAnalysis);
    if (!load) {
      continue;
    }

    builder.setInsertionPoint(broadcast);
    Value zeroStride = builder.create<arith::ConstantOp>(
        broadcast.getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast.getLoc(), broadcast.getResult().getType(), load.getPtr(),
        load.getOffset(), zeroStride, builder.getI64IntegerAttr(1));
    broadcast.getResult().replaceAllUsesWith(fused.getResult());
    broadcast.erase();
    if (load->use_empty()) {
      load.erase();
    }
  }
  return success();
}

struct VMIPreAssignmentCombinePass
    : pto::impl::VMIPreAssignmentCombineBase<VMIPreAssignmentCombinePass> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    if (failed(canonicalizeContiguousGroupLoads(getOperation())) ||
        failed(fuseGroupSlotBroadcastLoads(getOperation())) ||
        failed(fuseVMISingleLaneLoadBroadcasts(getOperation(), aliasAnalysis)) ||
        failed(fusePTOScalarUBLoadBroadcasts(getOperation(), aliasAnalysis))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPreAssignmentCombinePass() {
  return std::make_unique<VMIPreAssignmentCombinePass>();
}
