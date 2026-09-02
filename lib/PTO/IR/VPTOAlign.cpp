// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAlign.cpp - VPTO align chain helpers ----------------------------===//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Support/CodeConstants.h"


#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                  StringRef roleDescription) {
  if (!isa<AlignType>(type)) {
    return op->emitOpError() << roleDescription << " must be !pto.align";
  }
  return success();
}

static bool isStoreAlignProducer(Operation *op) {
  return isa<InitAlignOp, PstuOp, VstusOp, VsturOp>(op);
}

static bool isStoreAlignSink(Operation *op) {
  return isa<VstasOp, VstarOp>(op);
}

static bool isLoadAlignProducer(Operation *op) {
  return isa<VldasOp, VldusOp>(op);
}

static scf::IfOp getEnclosingBranchIf(Operation *op) {
  for (Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(cursor);
    if (!ifOp) {
      continue;
    }
    Region *parentRegion = op->getParentRegion();
    if (parentRegion == &ifOp.getThenRegion() || parentRegion == &ifOp.getElseRegion()) {
      return ifOp;
    }
  }
  return nullptr;
}

static bool isValueOwnedByRegion(Value value, Region *region) {
  Region *valueRegion = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    valueRegion = blockArg.getParentRegion();
  } else if (Operation *def = value.getDefiningOp()) {
    valueRegion = def->getParentRegion();
  }

  while (valueRegion) {
    if (valueRegion == region) {
      return true;
    }
    Operation *parentOp = valueRegion->getParentOp();
    valueRegion = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return false;
}

static FailureOr<Value> resolveStoreAlignRoot(Value value, Operation *user);
static FailureOr<Value> resolveLoadAlignRoot(Value value, Operation *user);
static FailureOr<Value> resolveLoadAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited);
static FailureOr<Value> resolveAlignForResult(Value current, scf::ForOp forOp);

static FailureOr<Value> resolveStoreAlignBlockArg(Value current) {
  auto blockArg = cast<BlockArgument>(current);
  auto *owner = blockArg.getOwner();
  auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp());
  if (!forOp) {
    return failure();
  }
  unsigned argNumber = blockArg.getArgNumber();
  unsigned ivCount = forOp.getNumInductionVars();
  if (argNumber < ivCount) {
    return failure();
  }
  unsigned iterIdx = argNumber - ivCount;
  if (iterIdx >= forOp.getInitArgs().size()) {
    return failure();
  }
  return forOp.getInitArgs()[iterIdx];
}


static std::optional<Value> getStoreAlignStateOpIn(Operation *def) {
  if (auto stateOp = dyn_cast<PstuOp>(def)) {
    return stateOp.getAlignIn();
  }
  if (auto stateOp = dyn_cast<VstusOp>(def)) {
    return stateOp.getAlignIn();
  }
  if (auto stateOp = dyn_cast<VsturOp>(def)) {
    return stateOp.getAlignIn();
  }
  return std::nullopt;
}

static FailureOr<Value> resolveStoreAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited);



template <typename ResolveRootFn>
static FailureOr<Value> resolveAlignIfResultImpl(
    Value current, scf::IfOp ifOp,
    llvm::SmallPtrSet<void *, mlir::pto::kValue8> &visited,
    ResolveRootFn resolveRoot) {
  auto result = dyn_cast<OpResult>(current);
  if (!result || !ifOp.elseBlock()) {
    return failure();
  }
  unsigned resultIdx = result.getResultNumber();
  auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  if (!thenYield || !elseYield || resultIdx >= thenYield.getNumOperands() ||
      resultIdx >= elseYield.getNumOperands()) {
    return failure();
  }
  FailureOr<Value> thenRoot =
      resolveRoot(thenYield.getOperand(resultIdx), visited);
  FailureOr<Value> elseRoot =
      resolveRoot(elseYield.getOperand(resultIdx), visited);
  if (failed(thenRoot) || failed(elseRoot) || *thenRoot != *elseRoot) {
    return failure();
  }
  return *thenRoot;
}

static FailureOr<Value> resolveLoadAlignIfResult(
    Value current, scf::IfOp ifOp,
    llvm::SmallPtrSet<void *, mlir::pto::kValue8> &visited) {
  return resolveAlignIfResultImpl(
      current, ifOp, visited,
      [](Value v, llvm::SmallPtrSet<void *, mlir::pto::kValue8> &vis) {
        return resolveLoadAlignRootImpl(v, vis);
      });
}

static FailureOr<Value> resolveLoadAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited) {
  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second) { return failure(); }
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto next = resolveStoreAlignBlockArg(current);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      continue;
    }
    Operation *def = current.getDefiningOp();
    if (!def) {
      return failure();
    }
    if (isa<VldasOp>(def)) {
      return current;
    }
    if (auto stateOp = dyn_cast<VldusOp>(def)) {
      current = stateOp.getAlign();
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      auto next = resolveAlignForResult(current, forOp);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      return resolveLoadAlignIfResult(current, ifOp, visited);
    }
    return failure();
  }
}


static FailureOr<Value> resolveAlignForResult(Value current, scf::ForOp forOp) {
  auto result = dyn_cast<OpResult>(current);
  if (!result) {
    return failure();
  }
  unsigned resultIdx = result.getResultNumber();
  if (resultIdx >= forOp.getYieldedValues().size()) {
    return failure();
  }
  return forOp.getYieldedValues()[resultIdx];
}

static FailureOr<Value> resolveStoreAlignIfResult(
    Value current, scf::IfOp ifOp,
    llvm::SmallPtrSet<void *, mlir::pto::kValue8> &visited) {
  return resolveAlignIfResultImpl(
      current, ifOp, visited,
      [](Value v, llvm::SmallPtrSet<void *, mlir::pto::kValue8> &vis) {
        return resolveStoreAlignRootImpl(v, vis);
      });
}

static FailureOr<Value> resolveStoreAlignRootImpl(
    Value current, llvm::SmallPtrSet<void *, mlir::pto::kValue8> visited) {
  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second) {
      return failure();
    }
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto next = resolveStoreAlignBlockArg(current);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      continue;
    }
    Operation *def = current.getDefiningOp();
    if (!def) {
      return failure();
    }
    if (isa<InitAlignOp>(def)) {
      return current;
    }
    if (auto nextState = getStoreAlignStateOpIn(def)) {
      current = *nextState;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      auto next = resolveAlignForResult(current, forOp);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      return resolveStoreAlignIfResult(current, ifOp, visited);
    }
    return failure();
  }
}


static FailureOr<Value> resolveStoreAlignRoot(Value value, Operation *user) {
  (void)user;
  return resolveStoreAlignRootImpl(value, {});
}

static LogicalResult verifyAlignLoopThreading(Value align, Operation *user,
                                               StringRef roleDescription) {
  Operation *cursor = user;
  while (auto forOp = cursor->getParentOfType<scf::ForOp>()) {
    Region *body = &forOp.getRegion();
    if (isValueOwnedByRegion(align, body)) {
      return success();
    }
    if (!isValueOwnedByRegion(align, body)) {
      return user->emitOpError()
             << roleDescription
             << " must be threaded through scf.for iter_args when used inside a "
                "loop";
    }
    cursor = forOp;
  }
  return success();
}

static FailureOr<Value> resolveSingleAlignIfResult(scf::IfOp ifOp) {
  SmallVector<unsigned> alignResultIndices;
  for (auto [index, type] : llvm::enumerate(ifOp.getResultTypes())) {
    if (isa<AlignType>(type)) {
      alignResultIndices.push_back(index);
    }
  }
  if (alignResultIndices.size() != 1) {
    return failure();
  }
  return ifOp.getResult(alignResultIndices.front());
}


static LogicalResult appendStoreAlignForInitUse(scf::ForOp forOp,
                                                unsigned operandNumber,
                                                Operation *user,
                                                SmallVectorImpl<Value> &nextValues) {
  unsigned firstInitArg = forOp.getNumControlOperands();
  if (operandNumber < firstInitArg) {
    return user->emitOpError()
           << "found unexpected scf.for control operand use for !pto.align";
  }
  unsigned iterIdx = operandNumber - firstInitArg;
  if (iterIdx >= forOp.getRegionIterArgs().size()) {
    return user->emitOpError()
           << "found invalid scf.for iter_args use for !pto.align";
  }
  nextValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
  return success();
}

static LogicalResult appendStoreAlignYieldUse(scf::YieldOp yieldOp,
                                              unsigned operandNumber,
                                              Operation *user,
                                              SmallVectorImpl<Value> &nextValues) {
  auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  if (!forOp) {
    return user->emitOpError()
           << "found !pto.align yielded from non-scf.for loop";
  }
  if (operandNumber >= forOp.getNumResults()) {
    return user->emitOpError()
           << "found invalid scf.yield result mapping for !pto.align";
  }
  nextValues.push_back(forOp.getResult(operandNumber));
  return success();
}

static scf::IfOp getCommonStoreAlignBranchIf(
    ArrayRef<Operation *> branchUsers) {
  scf::IfOp commonIf;
  for (Operation *branchUser : branchUsers) {
    scf::IfOp enclosingIf = getEnclosingBranchIf(branchUser);
    if (!enclosingIf) {
      return nullptr;
    }
    if (!commonIf) {
      commonIf = enclosingIf;
    } else if (commonIf != enclosingIf) {
      return nullptr;
    }
  }
  return commonIf;
}


static LogicalResult collectStoreAlignLinearUses(
    Value current, Operation *user, SmallVectorImpl<Value> &nextValues,
    SmallVectorImpl<Operation *> &terminalUsers,
    SmallVectorImpl<Operation *> &branchUsers) {
  for (OpOperand &use : current.getUses()) {
    Operation *owner = use.getOwner();
    if (isStoreAlignSink(owner)) {
      terminalUsers.push_back(owner);
      branchUsers.push_back(owner);
      continue;
    }
    if (auto stateOp = dyn_cast<PstuOp>(owner)) {
      nextValues.push_back(stateOp.getAlignOut());
      branchUsers.push_back(owner);
      continue;
    }
    if (auto stateOp = dyn_cast<VstusOp>(owner)) {
      nextValues.push_back(stateOp.getAlignOut());
      branchUsers.push_back(owner);
      continue;
    }
    if (auto stateOp = dyn_cast<VsturOp>(owner)) {
      nextValues.push_back(stateOp.getAlignOut());
      branchUsers.push_back(owner);
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      if (failed(appendStoreAlignForInitUse(forOp, use.getOperandNumber(),
                                            user, nextValues))) {
        return failure();
      }
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
      if (failed(appendStoreAlignYieldUse(yieldOp, use.getOperandNumber(),
                                          user, nextValues))) {
        return failure();
      }
      continue;
    }
    return user->emitOpError()
           << "found unsupported !pto.align consumer " << owner->getName();
  }
  return success();
}


static LogicalResult tryMergeAlignBranches(
    unsigned branchCount, ArrayRef<Operation *> branchUsers, Operation *user,
    Value &mergedValue, bool &didMerge) {
  didMerge = false;
  if (branchCount <= 1) {
    return success();
  }
  scf::IfOp commonIf = getCommonStoreAlignBranchIf(branchUsers);
  if (commonIf) {
    FailureOr<Value> merged = resolveSingleAlignIfResult(commonIf);
    if (succeeded(merged)) {
      mergedValue = *merged;
      didMerge = true;
      return success();
    }
  }
  return user->emitOpError()
         << "!pto.align value must form a single linear store-state chain";
}

template <typename CollectFn>
static LogicalResult verifyAlignLinearUsesImpl(Value value, Operation *user,
                                               CollectFn collect) {
  llvm::SmallPtrSet<void *, mlir::pto::kValue16> visited;
  Value current = value;
  while (visited.insert(current.getAsOpaquePointer()).second) {
    SmallVector<Value> nextValues;
    SmallVector<Operation *> branchUsers;
    size_t terminalCount = 0;
    if (failed(collect(current, user, nextValues, branchUsers, terminalCount))) {
      return failure();
    }
    Value mergedValue;
    bool didMerge = false;
    if (failed(tryMergeAlignBranches(
            nextValues.size() + terminalCount, branchUsers, user, mergedValue,
            didMerge))) {
      return failure();
    }
    if (didMerge) {
      current = mergedValue;
      continue;
    }
    if (nextValues.empty()) {
      return success();
    }
    current = nextValues.front();
  }
  return success();
}

static LogicalResult verifyStoreAlignLinearUses(Value value, Operation *user) {
  return verifyAlignLinearUsesImpl(
      value, user,
      [](Value current, Operation *user, SmallVectorImpl<Value> &nextValues,
         SmallVectorImpl<Operation *> &branchUsers, size_t &terminalCount) {
        SmallVector<Operation *> terminalUsers;
        if (failed(collectStoreAlignLinearUses(current, user, nextValues,
                                               terminalUsers, branchUsers))) {
          return failure();
        }
        terminalCount = terminalUsers.size();
        return success();
      });
}



LogicalResult verifyStoreAlignChain(Value align, Operation *user,
                                           StringRef roleDescription) {
  if (disableVPTOAlignChainVerification) {
    return success();
  }

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription))) {
    return failure();
  }

  if (failed(verifyAlignLoopThreading(align, user, roleDescription))) {
    return failure();
  }

  FailureOr<Value> root = resolveStoreAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.init_align or a prior store-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isStoreAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op, got "
           << def->getName();
  }

  return verifyStoreAlignLinearUses(*root, user);
}

static FailureOr<Value> resolveLoadAlignRoot(Value value, Operation *user) {
  (void)user;
  return resolveLoadAlignRootImpl(value, {});
}




static LogicalResult collectLoadAlignLinearUses(
    Value current, Operation *user, SmallVectorImpl<Value> &nextValues,
    SmallVectorImpl<Operation *> &branchUsers) {
  for (OpOperand &use : current.getUses()) {
    Operation *owner = use.getOwner();
    if (auto stateOp = dyn_cast<VldusOp>(owner)) {
      nextValues.push_back(stateOp.getUpdatedAlign());
      branchUsers.push_back(owner);
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      if (failed(appendStoreAlignForInitUse(forOp, use.getOperandNumber(),
                                            user, nextValues))) {
        return failure();
      }
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
      if (failed(appendStoreAlignYieldUse(yieldOp, use.getOperandNumber(),
                                          user, nextValues))) {
        return failure();
      }
      continue;
    }
    return user->emitOpError()
           << "found unsupported !pto.align consumer " << owner->getName();
  }
  return success();
}

static LogicalResult verifyLoadAlignLinearUses(Value value, Operation *user) {
  return verifyAlignLinearUsesImpl(
      value, user,
      [](Value current, Operation *user, SmallVectorImpl<Value> &nextValues,
         SmallVectorImpl<Operation *> &branchUsers, size_t &terminalCount) {
        terminalCount = 0;
        return collectLoadAlignLinearUses(current, user, nextValues,
                                          branchUsers);
      });
}



LogicalResult verifyLoadAlignChain(Value align, Operation *user,
                                          StringRef roleDescription) {
  if (disableVPTOAlignChainVerification) {
    return success();
  }

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription))) {
    return failure();
  }

  if (failed(verifyAlignLoopThreading(align, user, roleDescription))) {
    return failure();
  }

  FailureOr<Value> root = resolveLoadAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.vldas or a prior load-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isLoadAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op, got "
           << def->getName();
  }

  return verifyLoadAlignLinearUses(*root, user);
}

