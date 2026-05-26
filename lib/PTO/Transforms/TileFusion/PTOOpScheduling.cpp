// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_OPSCHEDULING
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

enum class SchedulingBarrierKind {
  Movable,
  LocalBoundary,
  HardBoundary,
};

struct GroupMember {
  Operation *op = nullptr;
  int64_t order = 0;
  unsigned originalIndex = 0;
};

struct ScheduledGroup {
  int64_t groupId = 0;
  unsigned firstOriginalIndex = 0;
  SmallVector<GroupMember, 8> members;
};

static std::optional<int64_t> getRequiredI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static bool hasIncompleteFusionMetadata(Operation *op) {
  const bool hasGroupId = op->hasAttr(kFusionGroupIdAttr);
  const bool hasOrder = op->hasAttr(kFusionOrderAttr);
  return hasGroupId != hasOrder;
}

static bool sharesAnyValue(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  for (Value value : lhs)
    if (llvm::is_contained(rhs, value))
      return true;
  return false;
}

static SchedulingBarrierKind classifySchedulingBarrier(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>() || !op->getRegions().empty())
    return SchedulingBarrierKind::HardBoundary;
  if (isa<CallOpInterface>(op))
    return SchedulingBarrierKind::HardBoundary;

  FailureOr<pto::FusionOpSemantics> semanticsOr = pto::getFusionOpSemantics(op);
  if (succeeded(semanticsOr)) {
    switch (semanticsOr->kind) {
    case pto::FusionOpKind::Compute:
      return SchedulingBarrierKind::Movable;
    case pto::FusionOpKind::LocalBoundary:
      return SchedulingBarrierKind::LocalBoundary;
    case pto::FusionOpKind::HardBoundary:
      return SchedulingBarrierKind::HardBoundary;
    }
  }
  if (!isMemoryEffectFree(op))
    return SchedulingBarrierKind::HardBoundary;
  return SchedulingBarrierKind::Movable;
}

static bool hasTileDependency(Operation *opA, Operation *opB) {
  FailureOr<pto::FusionOpSemantics> aSemOr = pto::getFusionOpSemantics(opA);
  FailureOr<pto::FusionOpSemantics> bSemOr = pto::getFusionOpSemantics(opB);
  if (failed(aSemOr) || failed(bSemOr))
    return true;

  const pto::FusionOpSemantics &a = *aSemOr;
  const pto::FusionOpSemantics &b = *bSemOr;

  return sharesAnyValue(a.tileOutputs, b.tileInputs) ||
         sharesAnyValue(b.tileOutputs, a.tileInputs) ||
         sharesAnyValue(a.tileOutputs, b.tileOutputs);
}

static bool crossesOperandDefinition(Operation *movingOp, Operation *candidate) {
  for (Value operand : movingOp->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (defOp == candidate)
      return true;
  }
  return false;
}

static bool canMoveEarlierAcross(Operation *movingOp, Operation *candidate) {
  if (crossesOperandDefinition(movingOp, candidate))
    return false;

  switch (classifySchedulingBarrier(candidate)) {
  case SchedulingBarrierKind::Movable:
  case SchedulingBarrierKind::LocalBoundary:
    return !hasTileDependency(movingOp, candidate);
  case SchedulingBarrierKind::HardBoundary:
    return false;
  }
  return false;
}

static bool canMoveLaterAcross(Operation *movingOp, Operation *candidate) {
  for (Value operand : candidate->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (defOp == movingOp)
      return false;
  }

  switch (classifySchedulingBarrier(candidate)) {
  case SchedulingBarrierKind::Movable:
  case SchedulingBarrierKind::LocalBoundary:
    return !hasTileDependency(movingOp, candidate);
  case SchedulingBarrierKind::HardBoundary:
    return false;
  }
  return false;
}

static bool canMoveAfter(Operation *movingOp, Operation *anchorOp) {
  if (!movingOp || !anchorOp || movingOp == anchorOp)
    return false;
  if (movingOp->getBlock() != anchorOp->getBlock())
    return false;

  Operation *cursor = anchorOp->getNextNode();
  while (cursor && cursor != movingOp) {
    if (!canMoveEarlierAcross(movingOp, cursor))
      return false;
    cursor = cursor->getNextNode();
  }
  return cursor == movingOp;
}

static LogicalResult
collectScheduledGroups(Block &block, SmallVectorImpl<ScheduledGroup> &groups) {
  DenseMap<int64_t, unsigned> groupIndexById;

  unsigned originalIndex = 0;
  for (Operation &op : block) {
    if (hasIncompleteFusionMetadata(&op)) {
      op.emitError("expected pto.fusion.group_id and pto.fusion.order to "
                   "either both exist or both be absent");
      return failure();
    }

    std::optional<int64_t> groupId =
        getRequiredI64Attr(&op, kFusionGroupIdAttr);
    if (!groupId) {
      ++originalIndex;
      continue;
    }

    std::optional<int64_t> order = getRequiredI64Attr(&op, kFusionOrderAttr);
    if (!order) {
      op.emitError("missing required pto.fusion.order attribute");
      return failure();
    }

    auto [it, inserted] = groupIndexById.try_emplace(*groupId, groups.size());
    if (inserted) {
      ScheduledGroup group;
      group.groupId = *groupId;
      group.firstOriginalIndex = originalIndex;
      groups.push_back(std::move(group));
    }

    ScheduledGroup &group = groups[it->second];
    group.members.push_back(GroupMember{&op, *order, originalIndex});
    ++originalIndex;
  }

  llvm::sort(groups, [](const ScheduledGroup &lhs, const ScheduledGroup &rhs) {
    if (lhs.firstOriginalIndex != rhs.firstOriginalIndex)
      return lhs.firstOriginalIndex < rhs.firstOriginalIndex;
    return lhs.groupId < rhs.groupId;
  });

  for (ScheduledGroup &group : groups) {
    llvm::sort(group.members, [](const GroupMember &lhs, const GroupMember &rhs) {
      if (lhs.order != rhs.order)
        return lhs.order < rhs.order;
      return lhs.originalIndex < rhs.originalIndex;
    });

    std::optional<int64_t> previousOrder;
    for (const GroupMember &member : group.members) {
      if (classifySchedulingBarrier(member.op) !=
          SchedulingBarrierKind::Movable) {
        member.op->emitError("fusion scheduling metadata must only annotate "
                             "movable compute ops");
        return failure();
      }
      if (previousOrder && *previousOrder == member.order) {
        member.op->emitError("duplicate pto.fusion.order within one fusion "
                             "group");
        return failure();
      }
      previousOrder = member.order;
    }
  }

  return success();
}

static void scheduleGroup(ScheduledGroup &group) {
  if (group.members.size() < 2)
    return;

  Operation *placement = group.members.front().op;
  for (GroupMember &member : llvm::drop_begin(group.members)) {
    Operation *op = member.op;
    while (op != placement && op != placement->getNextNode()) {
      if (canMoveAfter(op, placement)) {
        op->moveAfter(placement);
        break;
      }

      Operation *blockingOp = placement->getNextNode();
      if (!blockingOp || blockingOp == op ||
          !canMoveLaterAcross(placement, blockingOp))
        break;

      placement->moveAfter(blockingOp);
    }
    placement = op;
  }
}

static LogicalResult scheduleRegion(Region &region) {
  for (Block &block : region.getBlocks()) {
    SmallVector<ScheduledGroup, 8> groups;
    if (failed(collectScheduledGroups(block, groups)))
      return failure();
    for (ScheduledGroup &group : groups)
      scheduleGroup(group);

    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        if (failed(scheduleRegion(nestedRegion)))
          return failure();
  }
  return success();
}

struct OpSchedulingPass
    : public pto::impl::OpSchedulingBase<OpSchedulingPass> {
  using pto::impl::OpSchedulingBase<OpSchedulingPass>::OpSchedulingBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    if (failed(scheduleRegion(func.getRegion())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createOpSchedulingPass() {
  return std::make_unique<OpSchedulingPass>();
}
