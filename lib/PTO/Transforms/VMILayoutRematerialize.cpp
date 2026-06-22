// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutRematerialize.cpp - Rematerialize VMI producers ----------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTREMATERIALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasConcreteLayout(VMIVRegType type) {
  return type && static_cast<bool>(type.getLayoutAttr());
}

static bool hasConcreteLayout(VMIMaskType type) {
  return type && static_cast<bool>(type.getLayoutAttr());
}

static std::optional<Value> rematerializeDataProducer(Value value,
                                                      VMIVRegType resultType,
                                                      Location loc,
                                                      OpBuilder &builder) {
  if (!hasConcreteLayout(resultType))
    return std::nullopt;

  if (auto constant = value.getDefiningOp<VMIConstantOp>()) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (denseAttr && denseAttr.isSplat())
      return builder
          .create<VMIConstantOp>(loc, resultType, constant.getValue())
          .getResult();
  }

  if (auto broadcast = value.getDefiningOp<VMIBroadcastOp>())
    return builder.create<VMIBroadcastOp>(loc, resultType,
                                          broadcast.getValue())
        .getResult();

  if (auto iota = value.getDefiningOp<VMIIotaOp>())
    return builder
        .create<VMIIotaOp>(loc, resultType, iota.getBase(),
                           iota.getOrderAttr())
        .getResult();

  return std::nullopt;
}

static std::optional<Value> rematerializeMaskProducer(Value value,
                                                      VMIMaskType resultType,
                                                      Location loc,
                                                      OpBuilder &builder) {
  if (!hasConcreteLayout(resultType))
    return std::nullopt;

  if (auto createMask = value.getDefiningOp<VMICreateMaskOp>())
    return builder
        .create<VMICreateMaskOp>(loc, resultType, createMask.getActiveLanes())
        .getResult();

  if (auto createGroupMask = value.getDefiningOp<VMICreateGroupMaskOp>())
    return builder
        .create<VMICreateGroupMaskOp>(
            loc, resultType, createGroupMask.getActiveElemsPerGroup(),
            createGroupMask.getNumGroupsAttr(), createGroupMask.getGroupSizeAttr())
        .getResult();

  if (auto constantMask = value.getDefiningOp<VMIConstantMaskOp>())
    return builder
        .create<VMIConstantMaskOp>(loc, resultType,
                                   constantMask.getValueAttr())
        .getResult();

  return std::nullopt;
}

static bool tryReplaceDataEnsure(VMIEnsureLayoutOp ensure) {
  auto resultType = dyn_cast<VMIVRegType>(ensure.getResult().getType());
  if (!resultType)
    return false;

  OpBuilder builder(ensure);
  auto result = rematerializeDataProducer(ensure.getSource(), resultType,
                                          ensure->getLoc(), builder);
  if (!result)
    return false;

  ensure.getResult().replaceAllUsesWith(*result);
  ensure.erase();
  return true;
}

template <typename EnsureOp>
static bool tryReplaceMaskEnsure(EnsureOp ensure) {
  auto resultType = dyn_cast<VMIMaskType>(ensure.getResult().getType());
  if (!resultType)
    return false;

  OpBuilder builder(ensure);
  auto result = rematerializeMaskProducer(ensure.getSource(), resultType,
                                          ensure->getLoc(), builder);
  if (!result)
    return false;

  ensure.getResult().replaceAllUsesWith(*result);
  ensure.erase();
  return true;
}

struct VMILayoutRematerializePass
    : public mlir::pto::impl::VMILayoutRematerializeBase<
          VMILayoutRematerializePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutRematerializePass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> helpers;
    module.walk([&](Operation *op) {
      if (isa<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp,
              VMIEnsureMaskGranularityOp>(op))
        helpers.push_back(op);
    });

    for (Operation *op : helpers) {
      if (op->getBlock() == nullptr)
        continue;

      if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
        tryReplaceDataEnsure(ensure);
        continue;
      }

      if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
        tryReplaceMaskEnsure(ensure);
        continue;
      }

      if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op))
        tryReplaceMaskEnsure(ensure);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutRematerializePass() {
  return std::make_unique<VMILayoutRematerializePass>();
}
