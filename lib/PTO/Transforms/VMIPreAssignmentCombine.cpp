// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMIPreAssignmentCombine.cpp - Pre-assignment VMI combines ---------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIPREASSIGNMENTCOMBINE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static LogicalResult fuseGroupSlotBroadcastLoads(ModuleOp module) {
  SmallVector<VMIGroupBroadcastOp> broadcasts;
  module.walk([&](VMIGroupBroadcastOp broadcast) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load || !load.getResult().hasOneUse())
      return;
    if (load.getNumGroupsAttr().getInt() !=
        broadcast.getNumGroupsAttr().getInt())
      return;

    if (!isa<VMIVRegType>(broadcast.getResult().getType()))
      return;
    broadcasts.push_back(broadcast);
  });

  OpBuilder builder(module.getContext());
  for (VMIGroupBroadcastOp broadcast : broadcasts) {
    auto load = broadcast.getSource().getDefiningOp<VMIGroupSlotLoadOp>();
    if (!load)
      continue;

    builder.setInsertionPoint(broadcast);
    auto fused = builder.create<VMIGroupBroadcastLoadOp>(
        broadcast.getLoc(), broadcast.getResult().getType(), load.getSource(),
        load.getOffset(), load.getSourceGroupStride(),
        broadcast.getNumGroupsAttr());
    broadcast.getResult().replaceAllUsesWith(fused.getResult());
    broadcast.erase();
    if (load->use_empty())
      load.erase();
  }
  return success();
}

struct VMIPreAssignmentCombinePass
    : pto::impl::VMIPreAssignmentCombineBase<VMIPreAssignmentCombinePass> {
  void runOnOperation() override {
    if (failed(fuseGroupSlotBroadcastLoads(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIPreAssignmentCombinePass() {
  return std::make_unique<VMIPreAssignmentCombinePass>();
}
